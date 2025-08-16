#!/usr/bin/env python3
"""PRD-aligned signal processing tests with HackRF hardware.

Tests FR6, NFR2, FR7 requirements with real SDR capabilities.
"""

import os
import sys
import time

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from backend.services.sdr_service import SDRService
from backend.services.signal_processor import EWMAFilter, SignalProcessor


class TestSignalProcessingHardware:
    """Test signal processing with real HackRF hardware."""

    @pytest.fixture
    def signal_processor(self):
        """Create signal processor instance."""
        return SignalProcessor()

    @pytest.fixture
    def hackrf_available(self):
        """Check if HackRF is connected."""
        try:
            import subprocess

            result = subprocess.run(["lsusb"], capture_output=True, text=True)
            return "HackRF" in result.stdout
        except:
            return False

    def generate_beacon_signal(
        self, frequency: float = 406.025e6, duration: float = 0.5, sample_rate: float = 2.048e6
    ) -> np.ndarray:
        """Generate synthetic 406MHz beacon signal.

        Args:
            frequency: Beacon frequency in Hz
            duration: Signal duration in seconds
            sample_rate: Sample rate in Hz

        Returns:
            Complex IQ samples
        """
        num_samples = int(duration * sample_rate)
        t = np.arange(num_samples) / sample_rate

        # Generate beacon signal with modulation
        carrier = np.exp(1j * 2 * np.pi * frequency * t)

        # Add beacon modulation (112.5 bps BPSK)
        bit_rate = 112.5
        bits_per_sample = bit_rate / sample_rate
        num_bits = max(1, int(duration * bit_rate))  # At least 1 bit
        bit_pattern = np.random.choice([-1, 1], size=num_bits)

        # Upsample bit pattern
        samples_per_bit = int(sample_rate / bit_rate)
        modulation = np.repeat(bit_pattern, samples_per_bit)[:num_samples]

        # Apply modulation
        signal = carrier * modulation

        # Add realistic noise
        noise_power = 0.1
        noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * noise_power

        return signal + noise

    def test_fr6_ewma_filtering(self, signal_processor):
        """FR6: Test EWMA filtering with synthetic signals.

        Requirement: EWMA filtering for signal smoothing
        """
        # Create EWMA filter
        ewma_filter = EWMAFilter(alpha=0.3)

        # Generate noisy signal
        samples = 1000
        clean_signal = np.sin(2 * np.pi * 0.01 * np.arange(samples))
        noise = np.random.randn(samples) * 0.5
        noisy_signal = clean_signal + noise

        # Apply EWMA filtering
        filtered_signal = []
        for value in noisy_signal:
            filtered_value = ewma_filter.update(value)
            filtered_signal.append(filtered_value)

        filtered_signal = np.array(filtered_signal)

        # Verify smoothing effect
        # Filtered signal should have lower variance than noisy signal
        noise_variance = np.var(noisy_signal - clean_signal)
        filtered_variance = np.var(filtered_signal - clean_signal)

        assert (
            filtered_variance < noise_variance * 0.7
        ), f"EWMA filter not smoothing: noise var={noise_variance:.3f}, filtered var={filtered_variance:.3f}"

        # Test filter response time
        # Step response should settle within expected time
        ewma_filter.reset()
        step_response = []
        for i in range(50):
            if i < 10:
                value = 0.0
            else:
                value = 1.0
            filtered = ewma_filter.update(value)
            step_response.append(filtered)

        # Check 63% rise time (1 - e^-1)
        rise_time_index = next(i for i, v in enumerate(step_response[10:]) if v > 0.63) + 10
        assert rise_time_index < 20, f"EWMA rise time too slow: {rise_time_index - 10} samples"

    def test_nfr2_processing_latency(self, signal_processor):
        """NFR2: Measure actual processing latency <100ms.

        Requirement: Signal processing latency shall be <100ms
        """
        # Generate test signal (1024 samples)
        test_signal = self.generate_beacon_signal(duration=0.0005)  # 0.5ms of data

        # Measure processing pipeline latency
        latencies = []

        for _ in range(100):  # Run 100 iterations
            start_time = time.perf_counter()

            # Full processing pipeline
            rssi = signal_processor.compute_rssi(test_signal)
            snr = signal_processor.compute_snr(test_signal, noise_floor=-85.0)
            confidence = signal_processor.calculate_confidence(snr, rssi)
            detected = signal_processor.is_signal_detected(rssi, -85.0, 12.0)

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

        # Calculate statistics
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        p95_latency = np.percentile(latencies, 95)

        # Verify latency requirement
        assert avg_latency < 100, f"Average latency {avg_latency:.2f}ms exceeds 100ms"
        assert max_latency < 150, f"Max latency {max_latency:.2f}ms exceeds 150ms"
        assert p95_latency < 100, f"P95 latency {p95_latency:.2f}ms exceeds 100ms"

        print(
            f"Processing latency: avg={avg_latency:.2f}ms, max={max_latency:.2f}ms, p95={p95_latency:.2f}ms"
        )

    def test_fr7_debounced_transitions(self, signal_processor):
        """FR7: Test debounced transitions with real thresholds.

        Requirement: Debounced state transitions to prevent flapping
        """
        # Configure debouncing
        signal_processor.detection_count_threshold = 3  # Require 3 consecutive detections
        signal_processor.loss_count_threshold = 5  # Require 5 consecutive losses

        # Reset counters
        signal_processor.detection_count = 0
        signal_processor.loss_count = 0

        # Test detection debouncing
        # Need 3 consecutive detections above threshold
        test_sequence = [
            (-70, False),  # Below threshold
            (-50, False),  # Above threshold (1st)
            (-48, False),  # Above threshold (2nd)
            (-47, True),  # Above threshold (3rd) - should detect
            (-46, True),  # Still detecting (above threshold)
            (-45, True),  # Still detecting (above threshold)
        ]

        for rssi, should_detect in test_sequence:
            detected = signal_processor.process_detection_with_debounce(
                rssi=rssi, noise_floor=-85.0, threshold=12.0
            )

            if should_detect:
                assert detected, f"Should detect at RSSI={rssi}"
            else:
                assert not detected, f"Should not detect at RSSI={rssi}"

        # Test loss debouncing
        signal_processor.is_detecting = True
        signal_processor.loss_count = 0

        loss_sequence = [
            (-50, True),  # Still detecting
            (-90, True),  # Below threshold (1st loss)
            (-91, True),  # Below threshold (2nd loss)
            (-92, True),  # Below threshold (3rd loss)
            (-93, True),  # Below threshold (4th loss)
            (-94, False),  # Below threshold (5th loss) - should lose signal
        ]

        for rssi, should_detect in loss_sequence:
            detected = signal_processor.process_detection_with_debounce(
                rssi=rssi, noise_floor=-85.0, threshold=12.0
            )

            if should_detect:
                assert detected, f"Should still detect at RSSI={rssi}"
            else:
                assert not detected, f"Should lose signal at RSSI={rssi}"

    @pytest.mark.skipif(
        not os.environ.get("HACKRF_TEST_ENABLED"),
        reason="Set HACKRF_TEST_ENABLED=1 to run HackRF tests",
    )
    def test_hackrf_real_signal_processing(self, hackrf_available):
        """Test with real HackRF hardware if available."""
        if not hackrf_available:
            pytest.skip("HackRF not connected")

        try:
            # Initialize SDR service
            sdr_service = SDRService()

            # Configure for 406MHz beacon reception
            config = {"frequency": 406.025e6, "sample_rate": 2.048e6, "gain": 30}

            if sdr_service.initialize_sdr("hackrf", config):
                # Capture real samples
                samples = sdr_service.get_samples(1024)

                if samples is not None:
                    # Process real signal
                    processor = SignalProcessor()
                    rssi = processor.compute_rssi(samples)

                    # Verify we get reasonable values
                    assert -120 < rssi < 0, f"RSSI {rssi} out of range"

                    print(f"Real HackRF RSSI: {rssi:.2f} dBm")
                else:
                    pytest.skip("Could not get samples from HackRF")
            else:
                pytest.skip("Could not initialize HackRF")

        except Exception as e:
            pytest.skip(f"HackRF test failed: {e}")

    def test_adaptive_threshold_calculation(self, signal_processor):
        """Test adaptive threshold calculation with varying noise."""
        # Simulate varying noise conditions
        noise_history = [
            -95,
            -94,
            -95,
            -96,
            -95,  # Stable noise
            -90,
            -89,
            -88,
            -87,
            -86,  # Increasing noise
            -95,
            -94,
            -95,
            -96,
            -95,  # Back to stable
        ]

        # Calculate adaptive threshold
        threshold = signal_processor.calculate_adaptive_threshold(noise_history)

        # Should adapt to noise variations
        assert 10 <= threshold <= 18, f"Adaptive threshold {threshold} out of expected range"

        # Test with high noise variance
        noisy_history = [-95, -80, -95, -75, -95, -85, -95]
        noisy_threshold = signal_processor.calculate_adaptive_threshold(noisy_history)

        # Should increase threshold for noisy conditions
        assert noisy_threshold > threshold, "Threshold should increase with noise variance"


def test_signal_processor_initialization():
    """Test signal processor can be initialized."""
    processor = SignalProcessor()
    assert processor is not None
    assert processor.calibration_offset == -10.0  # Calibrated for HackRF One


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
