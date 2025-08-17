"""Unit tests for SignalProcessor and related classes.

Tests FFT-based RSSI computation, EWMA filtering, noise floor estimation,
and signal detection logic per PRD requirements.
"""


import numpy as np
import pytest

from src.backend.models.schemas import DetectionEvent
from src.backend.services.signal_processor import EWMAFilter, SignalProcessingError, SignalProcessor


class TestEWMAFilter:
    """Test Exponentially Weighted Moving Average filter."""

    def test_ewma_filter_initialization(self):
        """Test EWMA filter initializes with correct alpha."""
        filter_obj = EWMAFilter(alpha=0.3)
        assert filter_obj.alpha == 0.3
        assert filter_obj.value is None

    def test_ewma_filter_invalid_alpha_raises_error(self):
        """Test EWMA filter rejects invalid alpha values."""
        with pytest.raises(ValueError, match="Alpha must be between 0 and 1"):
            EWMAFilter(alpha=0)

        with pytest.raises(ValueError, match="Alpha must be between 0 and 1"):
            EWMAFilter(alpha=1.5)

    def test_ewma_filter_first_update(self):
        """Test first value update initializes filter state."""
        filter_obj = EWMAFilter(alpha=0.3)
        result = filter_obj.update(10.0)

        assert result == 10.0
        assert filter_obj.value == 10.0

    def test_ewma_filter_subsequent_updates(self):
        """Test EWMA filtering with multiple values."""
        filter_obj = EWMAFilter(alpha=0.3)

        # First update
        result1 = filter_obj.update(10.0)
        assert result1 == 10.0

        # Second update: 0.3 * 20 + 0.7 * 10 = 13.0
        result2 = filter_obj.update(20.0)
        assert abs(result2 - 13.0) < 0.001

        # Third update: 0.3 * 5 + 0.7 * 13 = 10.6
        result3 = filter_obj.update(5.0)
        assert abs(result3 - 10.6) < 0.001

    def test_ewma_filter_smoothing_behavior(self):
        """Test EWMA filter smooths noisy input."""
        filter_obj = EWMAFilter(alpha=0.1)  # Low alpha for heavy smoothing

        # Start with baseline
        filter_obj.update(50.0)

        # Add noise and verify smoothing
        noisy_values = [55, 45, 60, 40, 65, 35]
        results = [filter_obj.update(val) for val in noisy_values]

        # Should be smoother than input (variance should be lower)
        input_variance = np.var(noisy_values)
        output_variance = np.var(results)
        assert output_variance < input_variance


class TestSignalProcessor:
    """Test signal processing service."""

    @pytest.fixture
    def signal_processor(self):
        """Provide SignalProcessor instance."""
        return SignalProcessor()

    @pytest.fixture
    def test_iq_samples(self):
        """Provide test IQ sample data."""
        # Generate 1024 complex samples with signal and noise
        t = np.linspace(0, 1, 1024)
        signal = 0.5 * np.exp(1j * 2 * np.pi * 100 * t)  # 100 Hz tone
        noise = 0.1 * (np.random.randn(1024) + 1j * np.random.randn(1024))
        return signal + noise

    def test_signal_processor_initialization(self, signal_processor):
        """Test SignalProcessor initializes with correct defaults."""
        assert signal_processor.ewma_filter.alpha == 0.3
        assert signal_processor.snr_threshold == 12.0  # 12 dB SNR
        assert isinstance(signal_processor.ewma_filter, EWMAFilter)

    def test_compute_rssi_from_iq_samples(self, signal_processor, test_iq_samples):
        """Test RSSI computation from IQ samples using FFT."""
        rssi_value = signal_processor.compute_rssi(test_iq_samples)

        assert isinstance(rssi_value, float)
        assert rssi_value < 0  # RSSI should be negative dBm

    def test_compute_rssi_power_calculation(self, signal_processor):
        """Test power calculation accuracy in RSSI computation."""
        # Create known signal with specific power
        amplitude = 0.5
        samples = amplitude * np.ones(1024, dtype=np.complex64)
        expected_power_dbm = 20 * np.log10(amplitude) - 30  # Convert to dBm

        rssi_value = signal_processor.compute_rssi(samples)

        # Should be close to expected power (within 3 dB tolerance)
        assert abs(rssi_value - expected_power_dbm) < 3.0

    def test_noise_floor_estimation(self, signal_processor, test_iq_samples):
        """Test noise floor estimation using 10th percentile method."""
        # Process multiple samples to build noise floor estimate
        for _ in range(10):
            signal_processor.compute_rssi(test_iq_samples)

        noise_floor = signal_processor.get_noise_floor()

        assert noise_floor < -20  # Should be reasonable noise floor
        assert isinstance(noise_floor, float)

    def test_signal_detection_above_threshold(self, signal_processor):
        """Test signal detection when SNR exceeds threshold."""
        # Create strong signal (high SNR)
        strong_signal = 2.0 * np.ones(1024, dtype=np.complex64)

        rssi_reading = signal_processor.compute_rssi(strong_signal)

        # This should fail initially as detection logic needs implementation
        assert rssi_reading.snr > signal_processor.detection_threshold

    def test_signal_detection_event_generation(self, signal_processor):
        """Test detection event generation for strong signals."""
        # Create signal above detection threshold
        strong_signal = 3.0 * np.ones(1024, dtype=np.complex64)

        detection_events = []

        async def mock_callback(event: DetectionEvent):
            detection_events.append(event)

        signal_processor.add_detection_callback(mock_callback)
        signal_processor.compute_rssi(strong_signal)

        # This should fail initially as callback system needs implementation
        assert len(detection_events) > 0
        assert isinstance(detection_events[0], DetectionEvent)

    def test_debounced_state_transitions(self, signal_processor):
        """Test debounced state transitions per PRD-FR7."""
        # Test trigger threshold (12 dB) and drop threshold (6 dB)
        signal_processor.detection_threshold = 12.0
        signal_processor.drop_threshold = 6.0

        # Start with weak signal (no detection)
        weak_signal = 0.1 * np.ones(1024, dtype=np.complex64)
        signal_processor.compute_rssi(weak_signal)

        # Strong signal should trigger detection
        strong_signal = 5.0 * np.ones(1024, dtype=np.complex64)
        signal_processor.compute_rssi(strong_signal)

        # Medium signal should maintain detection (above drop threshold)
        medium_signal = 2.0 * np.ones(1024, dtype=np.complex64)
        signal_processor.compute_rssi(medium_signal)

        # This should fail initially as state management needs implementation
        assert hasattr(signal_processor, "detection_state")

    def test_add_detection_callback_registration(self, signal_processor):
        """Test detection callback registration."""

        def test_callback(event: DetectionEvent):
            pass

        signal_processor.add_detection_callback(test_callback)

        # This should fail initially as callback system needs implementation
        assert len(signal_processor._callbacks) == 1

    def test_remove_detection_callback(self, signal_processor):
        """Test detection callback removal."""

        def test_callback(event: DetectionEvent):
            pass

        signal_processor.add_detection_callback(test_callback)
        signal_processor.remove_detection_callback(test_callback)

        # This should fail initially as callback system needs implementation
        assert len(signal_processor._callbacks) == 0

    def test_processing_latency_under_100ms(self, signal_processor, test_iq_samples):
        """Test processing latency meets PRD-NFR2 requirement (<100ms)."""
        import time

        start_time = time.perf_counter()
        signal_processor.compute_rssi(test_iq_samples)
        end_time = time.perf_counter()

        processing_time_ms = (end_time - start_time) * 1000

        # Should meet PRD-NFR2: <100ms processing latency
        assert (
            processing_time_ms < 100
        ), f"Processing took {processing_time_ms:.1f}ms (>100ms limit)"

    def test_invalid_sample_data_handling(self, signal_processor):
        """Test error handling for invalid input data."""
        # Test empty array
        with pytest.raises(SignalProcessingError):
            signal_processor.compute_rssi(np.array([]))

        # Test non-complex data
        with pytest.raises(SignalProcessingError):
            signal_processor.compute_rssi(np.array([1, 2, 3]))

    def test_get_processing_stats(self, signal_processor):
        """Test processing statistics reporting."""
        stats = signal_processor.get_processing_stats()

        assert isinstance(stats, dict)
        assert "samples_processed" in stats
        assert "average_processing_time" in stats
        assert "detection_count" in stats
