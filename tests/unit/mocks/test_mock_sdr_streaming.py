"""Mock SDR Streaming Tests.

Tests for SDR IQ sample generation and streaming
without requiring real hardware.
"""

import threading
import time

import numpy as np
import pytest

from backend.hal.mock_hackrf import MockHackRF

pytestmark = pytest.mark.serial


@pytest.mark.mock_hardware
@pytest.mark.sdr
class TestMockSDRStreaming:
    """Test SDR streaming with mock hardware."""

    @pytest.fixture
    def mock_sdr(self) -> MockHackRF:
        """Create a mock SDR device."""
        device = MockHackRF()
        device.open()
        device.set_freq(3_200_000_000)
        device.set_sample_rate(20_000_000)
        return device

    def test_start_stop_streaming(self, mock_sdr: MockHackRF) -> None:
        """Test starting and stopping IQ streaming."""
        samples_received = []

        def callback(samples: bytes) -> None:
            samples_received.append(len(samples))

        # Start streaming
        assert mock_sdr.start_rx(callback) == 0
        assert mock_sdr.is_streaming is True

        # Wait for samples
        time.sleep(0.1)

        # Stop streaming
        assert mock_sdr.stop() == 0
        assert mock_sdr.is_streaming is False

        # Should have received samples
        assert len(samples_received) > 0

    def test_iq_sample_generation(self, mock_sdr: MockHackRF) -> None:
        """Test IQ sample data generation."""
        samples_list = []

        def callback(samples: bytes) -> None:
            samples_list.append(samples)

        # Configure and start
        mock_sdr.set_freq(1_000_000_000)  # 1 GHz
        mock_sdr.start_rx(callback)

        # Collect samples
        time.sleep(0.2)
        mock_sdr.stop()

        # Verify samples
        assert len(samples_list) > 0

        # Check sample format (interleaved float32 I/Q)
        for samples in samples_list:
            assert len(samples) > 0
            assert len(samples) % 8 == 0  # Each complex sample is 8 bytes

            # Convert to numpy array
            iq_array = np.frombuffer(samples, dtype=np.float32)
            assert len(iq_array) % 2 == 0  # Must have even number (I,Q pairs)

            # Extract I and Q
            i_samples = iq_array[0::2]
            q_samples = iq_array[1::2]

            # Verify reasonable values
            assert np.all(np.abs(i_samples) < 2.0)  # Should be normalized
            assert np.all(np.abs(q_samples) < 2.0)

    def test_sample_rate_consistency(self, mock_sdr: MockHackRF) -> None:
        """Test that samples are generated at configured rate."""
        sample_times = []
        sample_counts = []

        def callback(samples: bytes) -> None:
            sample_times.append(time.time())
            # Each complex sample is 8 bytes (2 * float32)
            sample_counts.append(len(samples) // 8)

        # Set known sample rate
        mock_sdr.set_sample_rate(2_000_000)  # 2 Msps
        mock_sdr.start_rx(callback)

        # Collect for 1 second
        time.sleep(1.0)
        mock_sdr.stop()

        # Calculate total samples
        total_samples = sum(sample_counts)

        # Should be close to configured rate (allow 20% tolerance for mock)
        expected = 2_000_000
        assert expected * 0.8 <= total_samples <= expected * 1.2

    def test_streaming_with_frequency_change(self, mock_sdr: MockHackRF) -> None:
        """Test changing frequency during streaming."""
        frequencies_seen = []

        def callback(samples: bytes) -> None:
            # Mock embeds frequency info
            frequencies_seen.append(mock_sdr.frequency)

        mock_sdr.start_rx(callback)

        # Change frequency while streaming
        for freq in [1_000_000_000, 2_000_000_000, 3_000_000_000]:
            mock_sdr.set_freq(freq)
            time.sleep(0.1)

        mock_sdr.stop()

        # Should have seen multiple frequencies
        unique_freqs = set(frequencies_seen)
        assert len(unique_freqs) >= 2

    def test_streaming_buffer_management(self, mock_sdr: MockHackRF) -> None:
        """Test buffer management during streaming."""
        buffer_sizes = []

        def callback(samples: bytes) -> None:
            buffer_sizes.append(len(samples))

        mock_sdr.start_rx(callback)
        time.sleep(0.5)
        mock_sdr.stop()

        # Check buffer sizes are consistent
        assert len(buffer_sizes) > 0
        assert all(size > 0 for size in buffer_sizes)

        # Buffers should be reasonably sized (not too small or large)
        assert all(1024 <= size <= 131072 for size in buffer_sizes)

    def test_streaming_thread_safety(self, mock_sdr: MockHackRF) -> None:
        """Test thread safety of streaming operations."""
        results = {"errors": 0, "samples": 0}
        lock = threading.Lock()

        def callback(samples: bytes) -> None:
            with lock:
                results["samples"] += 1

        # Start streaming
        mock_sdr.start_rx(callback)

        # Perform concurrent operations
        threads = []

        def change_freq() -> None:
            try:
                for _ in range(10):
                    mock_sdr.set_freq(np.random.randint(1e9, 6e9))
                    time.sleep(0.01)
            except Exception:
                with lock:
                    results["errors"] += 1

        def change_gain() -> None:
            try:
                for _ in range(10):
                    mock_sdr.set_lna_gain(np.random.choice([0, 8, 16, 24, 32, 40]))
                    time.sleep(0.01)
            except Exception:
                with lock:
                    results["errors"] += 1

        # Start threads
        for _ in range(3):
            t1 = threading.Thread(target=change_freq)
            t2 = threading.Thread(target=change_gain)
            threads.extend([t1, t2])
            t1.start()
            t2.start()

        # Wait for threads
        for t in threads:
            t.join()

        mock_sdr.stop()

        # Should have no errors and received samples
        assert results["errors"] == 0
        assert results["samples"] > 0


@pytest.mark.mock_hardware
@pytest.mark.sdr
class TestMockSDRSignalGeneration:
    """Test mock signal generation capabilities."""

    @pytest.fixture
    def mock_sdr(self) -> MockHackRF:
        """Create a mock SDR device."""
        device = MockHackRF()
        device.open()
        return device

    def test_signal_with_noise(self, mock_sdr: MockHackRF) -> None:
        """Test that generated signals include noise."""
        samples_list = []

        def callback(samples: bytes) -> None:
            samples_list.append(samples)

        mock_sdr.start_rx(callback)
        time.sleep(0.1)
        mock_sdr.stop()

        # Analyze signal
        all_samples = b"".join(samples_list)
        iq_array = np.frombuffer(all_samples, dtype=np.float32)

        # Convert to complex
        complex_samples = iq_array[0::2] + 1j * iq_array[1::2]

        # Calculate SNR estimate
        signal_power = np.mean(np.abs(complex_samples) ** 2)
        noise_estimate = np.std(np.abs(complex_samples))

        # Should have both signal and noise
        assert signal_power > 0
        assert noise_estimate > 0

    def test_signal_spectrum(self, mock_sdr: MockHackRF) -> None:
        """Test frequency content of generated signal."""
        samples_list = []

        def callback(samples: bytes) -> None:
            samples_list.append(samples)

        mock_sdr.set_sample_rate(2_048_000)  # 2.048 MHz
        mock_sdr.start_rx(callback)
        time.sleep(0.2)
        mock_sdr.stop()

        # Get samples
        all_samples = b"".join(samples_list)
        iq_array = np.frombuffer(all_samples, dtype=np.float32)
        complex_samples = iq_array[0::2] + 1j * iq_array[1::2]

        # Compute FFT
        fft_size = min(1024, len(complex_samples))
        if len(complex_samples) >= fft_size:
            spectrum = np.fft.fft(complex_samples[:fft_size])
            power_spectrum = np.abs(spectrum) ** 2

            # Should have energy across spectrum
            assert np.max(power_spectrum) > np.mean(power_spectrum)

    def test_gain_effect_on_signal(self, mock_sdr: MockHackRF) -> None:
        """Test that gain affects signal amplitude."""
        results = {}

        def measure_amplitude(gain: int) -> float:
            samples_list = []

            def callback(samples: bytes) -> None:
                samples_list.append(samples)

            mock_sdr.set_lna_gain(gain)
            mock_sdr.set_vga_gain(20)  # Fixed VGA
            mock_sdr.start_rx(callback)
            time.sleep(0.1)
            mock_sdr.stop()

            # Measure amplitude
            all_samples = b"".join(samples_list)
            iq_array = np.frombuffer(all_samples, dtype=np.float32)
            return np.mean(np.abs(iq_array))

        # Measure at different gains
        low_gain_amplitude = measure_amplitude(0)
        high_gain_amplitude = measure_amplitude(40)

        # Higher gain should produce higher amplitude
        assert high_gain_amplitude > low_gain_amplitude


@pytest.mark.mock_hardware
@pytest.mark.sdr
@pytest.mark.integration
class TestMockSDRErrorHandling:
    """Test error handling in mock SDR."""

    @pytest.fixture
    def mock_sdr(self) -> MockHackRF:
        """Create a mock SDR device."""
        device = MockHackRF()
        device.open()
        return device

    def test_streaming_without_open(self) -> None:
        """Test streaming fails without opening device."""
        device = MockHackRF()

        def callback(samples: bytes) -> None:
            pass

        # Should fail
        assert device.start_rx(callback) == -1
        assert device.is_streaming is False

    def test_double_start(self, mock_sdr: MockHackRF) -> None:
        """Test that double start is handled."""

        def callback(samples: bytes) -> None:
            pass

        # First start should succeed
        assert mock_sdr.start_rx(callback) == 0

        # Second start should fail or be ignored
        result = mock_sdr.start_rx(callback)
        assert result == -1 or mock_sdr.is_streaming is True

    def test_stop_without_start(self, mock_sdr: MockHackRF) -> None:
        """Test stopping when not streaming."""
        # Should not crash
        result = mock_sdr.stop()
        assert result == 0 or result == -1
        assert mock_sdr.is_streaming is False

    def test_callback_exception_handling(self, mock_sdr: MockHackRF) -> None:
        """Test that callback exceptions don't crash streaming."""
        exception_count = [0]
        sample_count = [0]

        def bad_callback(samples: bytes) -> None:
            sample_count[0] += 1
            if sample_count[0] % 3 == 0:
                exception_count[0] += 1
                raise ValueError("Test exception")

        # Start with problematic callback
        mock_sdr.start_rx(bad_callback)
        time.sleep(0.2)
        mock_sdr.stop()

        # Should have continued despite exceptions
        assert sample_count[0] > exception_count[0]
        assert exception_count[0] > 0
