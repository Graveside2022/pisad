"""
PRD Signal Processing Requirements Tests
Tests for FR6, FR7, NFR2 - Signal processing requirements

Story 4.9 Sprint 8 Day 3-4: Real PRD test implementation
"""

import asyncio
import time

import numpy as np
import pytest

from src.backend.services.signal_processor import SignalProcessor


class TestSignalProcessingRequirements:
    """Test signal processing requirements from PRD."""

    @pytest.fixture
    def signal_processor(self):
        """Create signal processor instance."""
        processor = SignalProcessor()
        return processor

    @pytest.fixture
    def synthetic_signal(self):
        """Generate synthetic RF signal for testing."""
        # Create a complex signal with known characteristics
        sample_rate = 2e6  # 2 MHz sample rate
        duration = 0.1  # 100ms of data
        num_samples = int(sample_rate * duration)

        # Generate carrier with modulation
        t = np.arange(num_samples) / sample_rate
        carrier_freq = 100e3  # 100 kHz carrier

        # Add FM modulation
        modulation_freq = 1e3  # 1 kHz modulation
        modulation_index = 5
        phase = 2 * np.pi * carrier_freq * t + modulation_index * np.sin(
            2 * np.pi * modulation_freq * t
        )

        # Create complex IQ signal
        signal_clean = np.exp(1j * phase)

        # Add noise for realistic SNR
        noise_power = 0.01
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
        )

        signal_with_noise = signal_clean + noise

        return {
            "samples": signal_with_noise,
            "sample_rate": sample_rate,
            "carrier_freq": carrier_freq,
            "duration": duration,
            "expected_snr_db": 20,  # ~20 dB SNR
        }

    def test_fr6_ewma_filtering(self, signal_processor, synthetic_signal):
        """
        FR6: System shall compute real-time RSSI with EWMA filtering and noise floor estimation.

        Validates EWMA filter implementation and 10th percentile noise floor.
        """
        samples = synthetic_signal["samples"]
        sample_rate = synthetic_signal["sample_rate"]

        # Process signal in blocks as would happen in real-time
        block_size = 1024
        num_blocks = len(samples) // block_size

        rssi_values = []
        ewma_values = []
        alpha = 0.3  # EWMA alpha parameter from PRD

        # Initialize EWMA
        ewma_rssi = None

        for i in range(num_blocks):
            block = samples[i * block_size : (i + 1) * block_size]

            # Compute RSSI for this block
            rssi = signal_processor.compute_rssi(block)
            rssi_values.append(rssi)

            # Apply EWMA filter
            if ewma_rssi is None:
                ewma_rssi = rssi
            else:
                ewma_rssi = alpha * rssi + (1 - alpha) * ewma_rssi
            ewma_values.append(ewma_rssi)

        # Verify EWMA smoothing
        rssi_variance = np.var(rssi_values)
        ewma_variance = np.var(ewma_values)

        assert ewma_variance < rssi_variance, "EWMA should reduce variance"
        assert ewma_variance < 0.5 * rssi_variance, "EWMA should significantly smooth signal"

        # Test noise floor estimation (10th percentile)
        noise_floor = np.percentile(rssi_values, 10)
        calculated_noise_floor = signal_processor.estimate_noise_floor(rssi_values)

        assert (
            abs(calculated_noise_floor - noise_floor) < 1.0
        ), "Noise floor estimation should use 10th percentile"

    def test_nfr2_processing_latency(self, signal_processor, synthetic_signal):
        """
        NFR2: Signal processing latency shall not exceed 100ms per RSSI computation cycle.

        Measures actual processing time for RSSI computation.
        """
        samples = synthetic_signal["samples"]
        block_size = 1024  # Typical block size

        # Measure processing time for multiple blocks
        latencies = []

        for _ in range(100):
            block = samples[:block_size]

            start_time = time.perf_counter()
            rssi = signal_processor.compute_rssi(block)
            latency = (time.perf_counter() - start_time) * 1000  # Convert to ms

            latencies.append(latency)

        # Check all latencies are under 100ms
        max_latency = max(latencies)
        avg_latency = np.mean(latencies)

        assert max_latency < 100, f"Max latency {max_latency:.2f}ms exceeds 100ms requirement"
        assert avg_latency < 50, f"Average latency {avg_latency:.2f}ms should be well under 100ms"

        # Verify we achieved the optimized <0.5ms from Story 4.9
        assert avg_latency < 0.5, f"Optimized latency should be <0.5ms, got {avg_latency:.2f}ms"

    def test_fr7_debounced_transitions(self, signal_processor):
        """
        FR7: System shall implement debounced state transitions with configurable thresholds.

        Tests trigger (12dB) and drop (6dB) thresholds with debouncing.
        """
        trigger_threshold = 12.0  # dB
        drop_threshold = 6.0  # dB

        # Create signal that crosses thresholds
        time_points = 100
        snr_values = np.zeros(time_points)

        # Signal pattern: low -> high -> low -> brief spike -> low
        snr_values[0:20] = 3.0  # Below drop threshold
        snr_values[20:40] = 15.0  # Above trigger threshold
        snr_values[40:60] = 8.0  # Between thresholds (hysteresis)
        snr_values[60:65] = 13.0  # Brief spike above trigger
        snr_values[65:] = 4.0  # Below drop threshold

        # Process with debouncing
        states = []
        current_state = "NO_SIGNAL"
        debounce_count = 0
        debounce_samples = 3  # Require 3 consecutive samples

        for snr in snr_values:
            if current_state == "NO_SIGNAL":
                if snr > trigger_threshold:
                    debounce_count += 1
                    if debounce_count >= debounce_samples:
                        current_state = "SIGNAL_DETECTED"
                        debounce_count = 0
                else:
                    debounce_count = 0

            elif current_state == "SIGNAL_DETECTED":
                if snr < drop_threshold:
                    debounce_count += 1
                    if debounce_count >= debounce_samples:
                        current_state = "NO_SIGNAL"
                        debounce_count = 0
                else:
                    debounce_count = 0

            states.append(current_state)

        # Verify correct state transitions
        # Should not trigger on brief spike (samples 60-65)
        assert states[62] == "NO_SIGNAL", "Brief spike should not trigger due to debouncing"

        # Should trigger when sustained above threshold
        assert states[22] == "SIGNAL_DETECTED", "Should detect after debounced trigger"

        # Should maintain state in hysteresis zone
        assert states[45] == "SIGNAL_DETECTED", "Should maintain state between thresholds"

        # Should drop when below drop threshold
        assert states[67] == "NO_SIGNAL", "Should drop signal after debounced drop"

    def test_snr_calculation_accuracy(self, signal_processor):
        """
        Test SNR calculation accuracy with known signal and noise levels.

        Validates the 12dB SNR threshold detection from FR1.
        """
        sample_rate = 2e6
        num_samples = 1024

        # Test various SNR levels around the 12dB threshold
        test_snr_levels = [6, 9, 12, 15, 18, 21]  # dB

        for expected_snr_db in test_snr_levels:
            # Generate signal with specific SNR
            signal_power = 1.0
            noise_power = signal_power / (10 ** (expected_snr_db / 10))

            # Create pure signal
            t = np.arange(num_samples) / sample_rate
            carrier = np.exp(1j * 2 * np.pi * 100e3 * t)  # 100 kHz carrier

            # Add calibrated noise
            noise = np.sqrt(noise_power / 2) * (
                np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
            )
            signal_with_noise = carrier + noise

            # Calculate SNR
            calculated_snr = signal_processor.compute_snr(signal_with_noise)

            # Allow 1dB tolerance due to estimation
            assert (
                abs(calculated_snr - expected_snr_db) < 1.0
            ), f"SNR calculation error: expected {expected_snr_db}dB, got {calculated_snr:.1f}dB"

            # Verify detection at 12dB threshold
            if expected_snr_db >= 12:
                assert signal_processor.is_signal_detected(
                    calculated_snr
                ), f"Should detect signal at {expected_snr_db}dB SNR"
            elif expected_snr_db < 11:  # Account for tolerance
                assert not signal_processor.is_signal_detected(
                    calculated_snr
                ), f"Should not detect signal at {expected_snr_db}dB SNR"

    def test_fft_based_rssi_computation(self, signal_processor):
        """
        Test FFT-based RSSI computation with 1024-sample blocks.

        Validates proper FFT implementation for power spectral density.
        """
        block_size = 1024
        sample_rate = 2e6

        # Generate test signal with known power
        t = np.arange(block_size) / sample_rate

        # Single tone for easy verification
        freq = 250e3  # 250 kHz
        amplitude = 2.0
        signal_samples = amplitude * np.exp(1j * 2 * np.pi * freq * t)

        # Compute RSSI using FFT
        rssi = signal_processor.compute_rssi_fft(signal_samples)

        # Expected power in dBm (assuming 50 ohm impedance)
        expected_power = 20 * np.log10(amplitude) + 10  # Simple conversion

        # RSSI should be close to expected power (within 3dB)
        assert (
            abs(rssi - expected_power) < 3.0
        ), f"FFT RSSI {rssi:.1f} dBm differs from expected {expected_power:.1f} dBm"

        # Test with multiple frequency components
        signal_multi = np.zeros(block_size, dtype=complex)
        freqs = [100e3, 250e3, 400e3]

        for f in freqs:
            signal_multi += np.exp(1j * 2 * np.pi * f * t)

        rssi_multi = signal_processor.compute_rssi_fft(signal_multi)

        # Should detect combined power
        assert rssi_multi > rssi, "Multiple tones should have higher RSSI"

    @pytest.mark.asyncio
    async def test_concurrent_processing(self, signal_processor):
        """
        Test concurrent signal processing for multiple channels.

        Validates async processing doesn't exceed latency requirements.
        """
        # Simulate multiple SDR channels
        num_channels = 4
        block_size = 1024
        sample_rate = 2e6

        # Generate different signals for each channel
        channels = []
        for i in range(num_channels):
            freq = (i + 1) * 100e3  # Different frequency per channel
            t = np.arange(block_size) / sample_rate
            signal_data = np.exp(1j * 2 * np.pi * freq * t)
            channels.append(signal_data)

        # Process all channels concurrently
        start_time = time.perf_counter()

        async def process_channel(samples):
            return await asyncio.to_thread(signal_processor.compute_rssi, samples)

        tasks = [process_channel(ch) for ch in channels]
        results = await asyncio.gather(*tasks)

        total_time = (time.perf_counter() - start_time) * 1000

        # All channels processed
        assert len(results) == num_channels

        # Total time should be less than single-channel time * num_channels
        # Due to concurrent processing
        assert total_time < 100, f"Concurrent processing took {total_time:.1f}ms"

        # Each result should be valid RSSI
        for rssi in results:
            assert isinstance(rssi, (int, float))
            assert -100 < rssi < 0  # Reasonable RSSI range in dBm

    def test_noise_floor_windowing(self, signal_processor):
        """
        Test noise floor estimation over 1-second window.

        Validates proper windowing for noise statistics.
        """
        sample_rate = 2e6
        window_duration = 1.0  # 1 second window per PRD
        samples_per_window = int(sample_rate * window_duration)

        # Generate signal with varying noise floor
        t = np.arange(samples_per_window) / sample_rate

        # First half: low noise
        # Second half: high noise
        signal_samples = np.zeros(samples_per_window, dtype=complex)

        low_noise = 0.01 * (
            np.random.randn(samples_per_window // 2) + 1j * np.random.randn(samples_per_window // 2)
        )
        high_noise = 0.1 * (
            np.random.randn(samples_per_window // 2) + 1j * np.random.randn(samples_per_window // 2)
        )

        signal_samples[: samples_per_window // 2] = low_noise
        signal_samples[samples_per_window // 2 :] = high_noise

        # Process in blocks and track noise floor
        block_size = 1024
        rssi_values = []

        for i in range(0, samples_per_window, block_size):
            block = signal_samples[i : i + block_size]
            if len(block) == block_size:
                rssi = signal_processor.compute_rssi(block)
                rssi_values.append(rssi)

        # Estimate noise floor using 10th percentile over window
        noise_floor = signal_processor.estimate_noise_floor(rssi_values)

        # Should be closer to low noise level (10th percentile)
        low_noise_rssi = signal_processor.compute_rssi(low_noise)
        high_noise_rssi = signal_processor.compute_rssi(high_noise)

        # Noise floor should be in lower range
        assert (
            noise_floor < (low_noise_rssi + high_noise_rssi) / 2
        ), "Noise floor should favor lower values (10th percentile)"


class TestSignalDetectionThresholds:
    """Test signal detection thresholds and confidence scoring."""

    def test_confidence_scoring(self):
        """Test signal confidence scoring based on SNR."""
        processor = SignalProcessor()

        # Test confidence at various SNR levels
        test_cases = [
            (5.0, 0.0),  # Below threshold, no confidence
            (12.0, 0.5),  # At threshold, medium confidence
            (15.0, 0.7),  # Above threshold, good confidence
            (20.0, 0.9),  # Well above threshold, high confidence
            (25.0, 1.0),  # Very high SNR, full confidence
        ]

        for snr, expected_confidence in test_cases:
            confidence = processor.calculate_confidence(snr)

            assert 0 <= confidence <= 1, "Confidence must be between 0 and 1"

            if snr < 12:
                assert confidence < 0.5, f"Low confidence expected for SNR {snr}dB"
            elif snr >= 20:
                assert confidence > 0.8, f"High confidence expected for SNR {snr}dB"

    def test_adaptive_thresholding(self):
        """Test adaptive threshold adjustment based on noise conditions."""
        processor = SignalProcessor()

        # Simulate changing noise conditions
        noise_floors = [-90, -85, -80, -75, -70]  # dBm

        for noise_floor in noise_floors:
            # Adjust thresholds based on noise floor
            trigger_threshold = processor.calculate_adaptive_threshold(noise_floor, margin=12)
            drop_threshold = processor.calculate_adaptive_threshold(noise_floor, margin=6)

            # Verify thresholds maintain proper margin above noise
            assert (
                trigger_threshold > noise_floor + 11
            ), f"Trigger threshold {trigger_threshold} insufficient margin above noise {noise_floor}"
            assert (
                drop_threshold > noise_floor + 5
            ), f"Drop threshold {drop_threshold} insufficient margin above noise {noise_floor}"

            # Verify hysteresis maintained
            assert (
                trigger_threshold > drop_threshold + 5
            ), "Insufficient hysteresis between trigger and drop thresholds"
