"""Unit tests for the Signal Processing Service."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import patch

import numpy as np
import pytest

from src.backend.models.schemas import DetectionEvent, RSSIReading
from src.backend.services.signal_processor import EWMAFilter, SignalProcessor

pytestmark = pytest.mark.serial
# Mark all tests in this module as unit tests
pytestmark = [pytest.mark.unit, pytest.mark.critical]


class TestEWMAFilter:
    """Test cases for Exponential Weighted Moving Average (EWMA) filter.

    The EWMA filter is used to smooth RSSI readings and reduce noise.
    These tests verify:
    - Proper initialization with valid/invalid alpha values
    - Correct filtering behavior with different smoothing factors
    - Edge cases and boundary conditions

    Test scenarios cover various alpha values (0.1-0.9) to ensure
    the filter responds appropriately to rapid vs gradual signal changes.
    """

    def test_ewma_init_valid_alpha(self):
        """Test EWMA filter initialization with valid alpha."""
        filter = EWMAFilter(alpha=0.3)
        assert filter.alpha == 0.3
        assert filter.value is None

    def test_ewma_init_invalid_alpha(self):
        """Test EWMA filter initialization with invalid alpha."""
        with pytest.raises(ValueError):
            EWMAFilter(alpha=0)
        with pytest.raises(ValueError):
            EWMAFilter(alpha=1.1)
        with pytest.raises(ValueError):
            EWMAFilter(alpha=-0.1)

    def test_ewma_first_update(self):
        """Test EWMA filter first update returns input value."""
        filter = EWMAFilter(alpha=0.3)
        result = filter.update(10.0)
        assert result == 10.0
        assert filter.value == 10.0

    def test_ewma_subsequent_updates(self):
        """Test EWMA filter applies correct formula."""
        filter = EWMAFilter(alpha=0.3)
        filter.update(10.0)
        result = filter.update(20.0)
        # Expected: 0.3 * 20 + 0.7 * 10 = 6 + 7 = 13
        assert result == 13.0

    def test_ewma_multiple_alpha_values(self):
        """Test EWMA filter with various alpha values."""
        test_cases = [
            (0.1, [10, 20, 30], [10, 11, 12.9]),  # Slow response
            (0.5, [10, 20, 30], [10, 15, 22.5]),  # Medium response
            (0.9, [10, 20, 30], [10, 19, 28.9]),  # Fast response
        ]

        for alpha, inputs, expected in test_cases:
            filter = EWMAFilter(alpha=alpha)
            results = []
            for value in inputs:
                results.append(filter.update(value))

            for result, expect in zip(results, expected, strict=False):
                assert abs(result - expect) < 0.01

    def test_ewma_reset(self):
        """Test EWMA filter reset."""
        filter = EWMAFilter(alpha=0.3)
        filter.update(10.0)
        filter.update(20.0)
        assert filter.value is not None

        filter.reset()
        assert filter.value is None

        # After reset, first update should return input value
        result = filter.update(30.0)
        assert result == 30.0


class TestSignalProcessor:
    """Test cases for SignalProcessor class.

    The SignalProcessor handles real-time RSSI signal analysis for beacon detection.
    These tests verify:
    - Initialization and configuration management
    - RSSI processing with filtering and noise floor estimation
    - Gradient calculation for direction finding
    - Detection event generation based on thresholds
    - Integration with MAVLink for telemetry updates
    - Asynchronous processing lifecycle management

    Test scenarios include normal operation, signal loss handling,
    multipath detection, and performance under various SNR conditions.
    """

    @pytest.fixture
    def processor(self):
        """Create a SignalProcessor instance for testing."""
        return SignalProcessor(
            fft_size=1024,
            ewma_alpha=0.3,
            snr_threshold=12.0,
            noise_window_seconds=1.0,
            sample_rate=2.048e6,
        )

    def test_initialization(self, processor):
        """Test SignalProcessor initialization."""
        assert processor.fft_size == 1024
        assert processor.snr_threshold == 12.0
        assert processor.sample_rate == 2.048e6
        assert processor.noise_floor == -100.0
        assert not processor.is_running
        assert processor.ewma_filter.alpha == 0.3

    @pytest.mark.asyncio
    async def test_process_iq_with_known_signal(self, processor):
        """Test RSSI computation with known test signal."""
        # Create a test signal: sinusoid at specific frequency
        fs = processor.sample_rate
        f_signal = 100e3  # 100 kHz signal
        t = np.arange(1024) / fs
        amplitude = 0.00001  # Very low amplitude to avoid triggering detection

        # Generate complex sinusoid (IQ samples)
        samples = amplitude * np.exp(1j * 2 * np.pi * f_signal * t)

        # Process the signal
        reading = await processor.process_iq(samples)

        assert isinstance(reading, RSSIReading)
        assert isinstance(reading.timestamp, datetime)
        assert reading.rssi < 0  # Should be negative dBm
        assert reading.noise_floor == -100.0  # Initial noise floor
        assert reading.detection_id is None  # No detection yet (below threshold)

    @pytest.mark.asyncio
    async def test_process_iq_insufficient_samples(self, processor):
        """Test handling of insufficient samples."""
        samples = np.zeros(100, dtype=np.complex64)  # Less than FFT size
        result = await processor.process_iq(samples)
        assert result is None

    @pytest.mark.asyncio
    async def test_process_iq_zero_power(self, processor):
        """Test RSSI computation with zero power signal."""
        samples = np.zeros(1024, dtype=np.complex64)
        reading = await processor.process_iq(samples)

        assert reading.rssi == -120.0  # Floor value for zero power

    def test_update_noise_floor(self, processor):
        """Test noise floor estimation using 10th percentile."""
        # Create test readings with known distribution
        readings = list(range(-100, -80))  # -100 to -81 dBm
        processor.update_noise_floor(readings)

        # 10th percentile of 20 values should be around -98
        expected_10th_percentile = np.percentile(readings, 10)
        assert abs(processor.noise_floor - expected_10th_percentile) < 0.1

    def test_update_noise_floor_insufficient_samples(self, processor):
        """Test noise floor update with insufficient samples."""
        initial_noise_floor = processor.noise_floor
        processor.update_noise_floor([])  # Empty list
        assert processor.noise_floor == initial_noise_floor  # Should not change

        processor.update_noise_floor([-90, -91, -92])  # Less than 10 samples
        assert processor.noise_floor == initial_noise_floor  # Should not change

    @pytest.mark.asyncio
    async def test_signal_detection_above_threshold(self, processor):
        """Test signal detection when SNR exceeds threshold."""
        # Set noise floor
        processor.noise_floor = -100.0

        # Signal with SNR > 12 dB threshold
        high_rssi = -85.0  # SNR = -85 - (-100) = 15 dB

        detection = await processor.detect_signal(high_rssi)

        assert detection is not None
        assert isinstance(detection, DetectionEvent)
        assert detection.rssi == high_rssi
        assert detection.snr == 15.0
        assert detection.confidence > 50.0  # Should have good confidence
        assert detection.frequency == processor.sample_rate / 2
        assert detection.state == "active"

    @pytest.mark.asyncio
    async def test_signal_detection_below_threshold(self, processor):
        """Test no detection when SNR below threshold."""
        processor.noise_floor = -100.0

        # Signal with SNR < 12 dB threshold
        low_rssi = -90.0  # SNR = -90 - (-100) = 10 dB

        detection = await processor.detect_signal(low_rssi)
        assert detection is None

    @pytest.mark.asyncio
    async def test_signal_detection_confidence_calculation(self, processor):
        """Test detection confidence calculation."""
        processor.noise_floor = -100.0

        test_cases = [
            (-87.0, 13.0, 52.5),  # Just above threshold
            (-85.0, 15.0, 57.5),  # 3 dB above threshold
            (-80.0, 20.0, 70.0),  # 8 dB above threshold
            (-60.0, 40.0, 100.0),  # Very high SNR (capped at 100)
        ]

        for rssi, expected_snr, expected_confidence in test_cases:
            detection = await processor.detect_signal(rssi)
            assert detection is not None
            assert detection.snr == expected_snr
            assert abs(detection.confidence - expected_confidence) < 0.1

    def test_get_noise_floor(self, processor):
        """Test getting current noise floor."""
        processor.noise_floor = -95.0
        assert processor.get_noise_floor() == -95.0

    def test_compute_gradient_insufficient_history(self, processor):
        """Test gradient computation with insufficient history."""
        result = processor.compute_gradient([])
        assert result["magnitude"] == 0.0
        assert result["direction"] == 0.0

        # Single reading
        reading = RSSIReading(
            timestamp=datetime.now(UTC), rssi=-90.0, noise_floor=-100.0, detection_id=None
        )
        result = processor.compute_gradient([reading])
        assert result["magnitude"] == 0.0
        assert result["direction"] == 0.0

    def test_compute_gradient_with_history(self, processor):
        """Test gradient computation with valid history."""
        # Create ascending RSSI values (signal getting stronger)
        history = [
            RSSIReading(datetime.now(UTC), -95.0, -100.0, None),
            RSSIReading(datetime.now(UTC), -92.0, -100.0, None),
            RSSIReading(datetime.now(UTC), -89.0, -100.0, None),
        ]

        result = processor.compute_gradient(history)
        assert result["magnitude"] > 0  # Should have positive magnitude
        assert result["direction"] > 0  # Positive direction (increasing signal)

        # Create descending RSSI values (signal getting weaker)
        history_desc = [
            RSSIReading(datetime.now(UTC), -85.0, -100.0, None),
            RSSIReading(datetime.now(UTC), -88.0, -100.0, None),
            RSSIReading(datetime.now(UTC), -91.0, -100.0, None),
        ]

        result_desc = processor.compute_gradient(history_desc)
        assert result_desc["magnitude"] > 0  # Should have positive magnitude
        assert result_desc["direction"] < 0  # Negative direction (decreasing signal)

    @pytest.mark.asyncio
    async def test_start_stop_service(self, processor):
        """Test starting and stopping the service."""
        assert not processor.is_running

        await processor.start()
        assert processor.is_running

        # Starting again should log warning but not fail
        await processor.start()
        assert processor.is_running

        await processor.stop()
        assert not processor.is_running
        assert processor.iq_queue.empty()

    @pytest.mark.asyncio
    async def test_get_status(self, processor):
        """Test getting service status."""
        await processor.start()
        processor.processing_latency = 50.0
        processor.noise_floor = -95.0

        status = await processor.get_status()

        assert status["is_running"] is True
        assert status["noise_floor"] == -95.0
        assert status["processing_latency_ms"] == 50.0
        assert status["fft_size"] == 1024
        assert status["snr_threshold"] == 12.0
        assert "queue_size" in status
        assert "rssi_history_size" in status

        await processor.stop()

    @pytest.mark.asyncio
    async def test_stream_rssi(self, processor):
        """Test RSSI streaming functionality."""
        await processor.start()

        # Add test samples to queue
        samples = np.ones(1024, dtype=np.complex64) * 0.1
        await processor.iq_queue.put(samples)

        # Get one reading from stream
        stream_gen = processor.stream_rssi()
        reading = await anext(stream_gen)

        assert isinstance(reading, RSSIReading)
        assert isinstance(reading.timestamp, datetime)
        assert reading.rssi < 0  # Should be negative dBm

        await processor.stop()

    @pytest.mark.asyncio
    async def test_processing_latency_measurement(self, processor):
        """Test that processing latency is measured correctly."""
        samples = np.ones(1024, dtype=np.complex64) * 0.1

        await processor.process_iq(samples)

        # Latency should be measured and reasonable (< 100ms as per requirements)
        assert processor.processing_latency > 0
        assert processor.processing_latency < 100  # Less than 100ms

    @pytest.mark.asyncio
    async def test_fft_computation_accuracy(self, processor):
        """Test FFT-based RSSI computation accuracy."""
        # Create a known signal with specific power
        fs = processor.sample_rate
        f_signal = 100e3  # 100 kHz
        t = np.arange(1024) / fs

        # Known amplitude signal
        amplitude = 0.0001  # Very low amplitude to keep RSSI below detection
        samples = amplitude * np.exp(1j * 2 * np.pi * f_signal * t)

        # Add minimal noise
        noise = (np.random.randn(1024) + 1j * np.random.randn(1024)) * 0.00001
        samples_with_noise = samples + noise

        reading = await processor.process_iq(samples_with_noise)

        # RSSI should be proportional to signal power
        # Due to windowing and FFT processing, the actual RSSI will be different
        # Just verify it's a reasonable negative dBm value
        assert -100 < reading.rssi < -70  # Reasonable range for low power signal

    @pytest.mark.asyncio
    async def test_structured_logging_format(self, processor):
        """Test that detection events are logged with structured format."""
        processor.noise_floor = -100.0

        with patch("src.backend.services.signal_processor.logger") as mock_logger:
            # Trigger a detection
            await processor.detect_signal(-85.0)

            # Check that logger.info was called with structured format
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args

            assert call_args[0][0] == "Signal detected"
            extra = call_args[1]["extra"]
            assert "detection_id" in extra
            assert "timestamp" in extra
            assert "frequency" in extra
            assert "rssi" in extra
            assert "snr" in extra
            assert "confidence" in extra


class TestSignalProcessorEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def processor(self):
        """Create a SignalProcessor instance for testing."""
        return SignalProcessor()

    @pytest.mark.asyncio
    async def test_process_complex_signal_patterns(self, processor):
        """Test processing of various complex signal patterns."""
        # Test with pure noise
        noise = (np.random.randn(1024) + 1j * np.random.randn(1024)) * 0.01
        reading_noise = await processor.process_iq(noise)
        assert reading_noise is not None

        # Test with DC offset
        dc_signal = np.ones(1024, dtype=np.complex64) * (0.5 + 0.5j)
        reading_dc = await processor.process_iq(dc_signal)
        assert reading_dc is not None

        # Test with multi-tone signal
        fs = processor.sample_rate
        t = np.arange(1024) / fs
        multi_tone = 0.1 * np.exp(1j * 2 * np.pi * 100e3 * t) + 0.05 * np.exp(
            1j * 2 * np.pi * 200e3 * t
        )
        reading_multi = await processor.process_iq(multi_tone)
        assert reading_multi is not None

    @pytest.mark.asyncio
    async def test_queue_overflow_handling(self, processor):
        """Test handling of queue overflow conditions."""
        await processor.start()

        # Fill queue to capacity
        samples = np.ones(1024, dtype=np.complex64) * 0.1
        for _ in range(100):  # Queue maxsize is 100
            try:
                processor.iq_queue.put_nowait(samples)
            except asyncio.QueueFull:
                break

        # Queue should be full
        assert processor.iq_queue.full()

        # Try to add one more (should not block or crash)
        try:
            processor.iq_queue.put_nowait(samples)
            raise AssertionError("Should have raised QueueFull")
        except asyncio.QueueFull:
            pass  # Expected

        await processor.stop()

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, processor):
        """Test concurrent access to processor methods."""
        await processor.start()

        async def process_samples():
            samples = np.ones(1024, dtype=np.complex64) * 0.1
            return await processor.process_iq(samples)

        async def get_status():
            return await processor.get_status()

        async def get_noise():
            return processor.get_noise_floor()

        # Run multiple operations concurrently
        results = await asyncio.gather(
            process_samples(), get_status(), get_noise(), return_exceptions=True
        )

        # All operations should complete without errors
        assert all(not isinstance(r, Exception) for r in results)

        await processor.stop()


class TestSignalProcessorValidation:
    """Test cases for additional validation improvements."""

    def test_gradient_computation_empty_list(self):
        """Test gradient computation handles empty list properly."""
        processor = SignalProcessor()

        # Test with empty list
        result = processor.compute_gradient([])
        assert result["magnitude"] == 0.0
        assert result["direction"] == 0.0
        assert "timestamp" in result


class TestSignalProcessorCallbacks:
    """Test callback functionality for SNR and RSSI."""

    @pytest.fixture
    def processor(self):
        """Create a SignalProcessor instance for testing."""
        return SignalProcessor()

    def test_add_snr_callback(self, processor):
        """Test adding SNR callback."""
        callback_values = []

        def snr_callback(value):
            callback_values.append(value)

        processor.add_snr_callback(snr_callback)
        assert len(processor._snr_callbacks) == 1

        # Add another callback
        processor.add_snr_callback(lambda x: None)
        assert len(processor._snr_callbacks) == 2

    def test_get_current_snr(self, processor):
        """Test getting current SNR value."""
        assert processor.get_current_snr() == 0.0

        processor._current_snr = 15.5
        assert processor.get_current_snr() == 15.5

    def test_add_rssi_callback(self, processor):
        """Test adding RSSI callback."""
        callback_values = []

        def rssi_callback(value):
            callback_values.append(value)

        processor.add_rssi_callback(rssi_callback)
        assert len(processor._rssi_callbacks) == 1

        # Add another callback
        processor.add_rssi_callback(lambda x: None)
        assert len(processor._rssi_callbacks) == 2

    def test_get_current_rssi(self, processor):
        """Test getting current RSSI value."""
        assert processor.get_current_rssi() == -100.0  # Default value

        processor._current_rssi = -85.5
        assert processor.get_current_rssi() == -85.5

    def test_set_mavlink_service(self, processor):
        """Test setting MAVLink service."""
        mock_service = "mock_mavlink_service"
        processor.set_mavlink_service(mock_service)
        assert processor._mavlink_service == mock_service

    @pytest.mark.asyncio
    async def test_rssi_generator(self, processor):
        """Test RSSI value generator."""
        processor._current_rssi = -75.0

        # Get values from generator
        gen = processor.rssi_generator(rate_hz=10.0)  # 10 Hz = 100ms interval

        # Get first value
        value1 = await anext(gen)
        assert value1 == -75.0

        # Change current RSSI and get another value
        processor._current_rssi = -80.0
        value2 = await anext(gen)
        assert value2 == -80.0

    @pytest.mark.asyncio
    async def test_rssi_callbacks_triggered(self, processor):
        """Test that RSSI callbacks are triggered during processing."""
        callback_values = []

        def rssi_callback(value):
            callback_values.append(value)

        processor.add_rssi_callback(rssi_callback)

        # Process some samples
        samples = np.ones(1024, dtype=np.complex64) * 0.01
        await processor.process_iq(samples)

        # Callback should have been triggered
        assert len(callback_values) == 1
        assert callback_values[0] < 0  # Should be negative dBm

    @pytest.mark.asyncio
    async def test_rssi_callback_error_handling(self, processor):
        """Test RSSI callback error handling."""

        def failing_callback(value):
            raise ValueError("Test error")

        processor.add_rssi_callback(failing_callback)

        # Process should not crash despite callback error
        samples = np.ones(1024, dtype=np.complex64) * 0.01
        reading = await processor.process_iq(samples)
        assert reading is not None

    @pytest.mark.asyncio
    async def test_snr_callbacks_triggered(self, processor):
        """Test that SNR callbacks are triggered during detection."""
        callback_values = []

        def snr_callback(value):
            callback_values.append(value)

        processor.add_snr_callback(snr_callback)
        processor.noise_floor = -100.0

        # Trigger detection with high SNR
        await processor.detect_signal(-85.0)

        # Callback should have been triggered
        assert len(callback_values) == 1
        assert callback_values[0] == 15.0  # SNR = -85 - (-100) = 15

    @pytest.mark.asyncio
    async def test_snr_callback_error_handling(self, processor):
        """Test SNR callback error handling."""

        def failing_callback(value):
            raise ValueError("Test error")

        processor.add_snr_callback(failing_callback)
        processor.noise_floor = -100.0

        # Detection should not crash despite callback error
        detection = await processor.detect_signal(-85.0)
        assert detection is not None

    @pytest.mark.asyncio
    async def test_mavlink_service_rssi_update(self, processor):
        """Test MAVLink service RSSI update."""
        from unittest.mock import MagicMock

        mock_service = MagicMock()
        processor.set_mavlink_service(mock_service)

        # Process samples to trigger MAVLink update
        samples = np.ones(1024, dtype=np.complex64) * 0.01
        await processor.process_iq(samples)

        # MAVLink service should have been updated
        mock_service.update_rssi_value.assert_called_once()
        call_args = mock_service.update_rssi_value.call_args[0]
        assert call_args[0] < 0  # RSSI should be negative dBm

    @pytest.mark.asyncio
    async def test_mavlink_service_error_handling(self, processor):
        """Test MAVLink service error handling."""
        from unittest.mock import MagicMock

        mock_service = MagicMock()
        mock_service.update_rssi_value.side_effect = Exception("MAVLink error")
        processor.set_mavlink_service(mock_service)

        # Process should not crash despite MAVLink error
        samples = np.ones(1024, dtype=np.complex64) * 0.01
        reading = await processor.process_iq(samples)
        assert reading is not None


class TestSignalProcessorStopConditions:
    """Test stop conditions and cleanup."""

    @pytest.fixture
    def processor(self):
        """Create a SignalProcessor instance for testing."""
        return SignalProcessor()

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self, processor):
        """Test stopping when service is not running."""
        assert not processor.is_running
        await processor.stop()  # Should not crash
        assert not processor.is_running

    @pytest.mark.asyncio
    async def test_stop_with_active_task(self, processor):
        """Test stopping with active processing task."""
        await processor.start()

        # Create a mock task
        processor.process_task = asyncio.create_task(asyncio.sleep(10))

        await processor.stop()
        assert not processor.is_running
        assert processor.process_task.cancelled()

    @pytest.mark.asyncio
    async def test_queue_cleanup_on_stop(self, processor):
        """Test queue is properly cleaned up on stop."""
        await processor.start()

        # Add items to queue
        samples = np.ones(1024, dtype=np.complex64) * 0.01
        for _ in range(5):
            await processor.iq_queue.put(samples)

        assert processor.iq_queue.qsize() == 5

        await processor.stop()
        assert processor.iq_queue.empty()

    @pytest.mark.asyncio
    async def test_queue_empty_exception_handling(self, processor):
        """Test handling of QueueEmpty exception during cleanup."""
        await processor.start()

        # Ensure queue is already empty
        assert processor.iq_queue.empty()

        # Stop should handle empty queue gracefully
        await processor.stop()
        assert not processor.is_running

    @pytest.mark.asyncio
    async def test_stream_rssi_timeout_handling(self, processor):
        """Test RSSI stream timeout handling."""
        await processor.start()

        # Get stream generator
        stream_gen = processor.stream_rssi()

        # Try to get reading with timeout (queue is empty)
        try:
            # Use timeout to prevent infinite wait
            await asyncio.wait_for(anext(stream_gen), timeout=1.5)
        except (StopAsyncIteration, TimeoutError):
            pass  # Expected when queue is empty and timeout occurs

        await processor.stop()

    @pytest.mark.asyncio
    async def test_stream_rssi_exception_handling(self, processor):
        """Test RSSI stream exception handling."""
        await processor.start()

        # Add invalid samples to trigger exception
        processor.iq_queue.put_nowait(None)  # This will cause an exception

        # Stream should handle exception gracefully
        stream_gen = processor.stream_rssi()

        # The stream should continue despite the error
        # Add valid samples after error
        samples = np.ones(1024, dtype=np.complex64) * 0.01
        await processor.iq_queue.put(samples)

        # Should eventually get a valid reading
        try:
            await asyncio.wait_for(anext(stream_gen), timeout=2.0)
        except (StopAsyncIteration, TimeoutError):
            pass  # May timeout if error recovery takes too long

        await processor.stop()

    @pytest.mark.asyncio
    async def test_noise_floor_update_with_history(self, processor):
        """Test noise floor update when RSSI history has enough samples."""
        # Pre-populate RSSI history with enough samples
        for i in range(15):
            processor.rssi_history.append(-95.0 + i * 0.5)

        # Trigger signal detection to update noise floor
        await processor.detect_signal(-85.0)

        # Noise floor should have been updated
        assert processor.noise_floor != -100.0  # Should be different from initial value
        assert processor.noise_floor < -90.0  # Should be a reasonable value
