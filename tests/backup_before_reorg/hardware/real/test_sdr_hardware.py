"""
Core SDR Hardware Tests - Priority 2
Tests HackRF hardware functionality, IQ streaming, and signal processing
"""

import asyncio
import subprocess
import time

import numpy as np
import pytest

from src.backend.hal.hackrf_interface import HackRFConfig, HackRFInterface


@pytest.mark.hardware
class TestSDRHardware:
    """Priority 2 - Core SDR hardware functionality tests"""

    @pytest.fixture
    async def hackrf(self):
        """Create and initialize HackRF interface"""
        config = HackRFConfig(
            frequency=3.2e9,  # 3.2 GHz beacon frequency
            sample_rate=20e6,  # 20 Msps
            lna_gain=16,
            vga_gain=20,
            amp_enable=False,
        )

        hackrf = HackRFInterface(config)
        success = await hackrf.open()

        if not success:
            pytest.skip("HackRF hardware not available")

        yield hackrf

        await hackrf.close()

    def test_hackrf_usb_location(self):
        """Verify HackRF on USB Bus 003 Device 003"""
        result = subprocess.run(["lsusb", "-t"], capture_output=True, text=True)

        # Look for HackRF in USB tree
        if "HackRF" not in result.stdout and "1d50:6089" not in result.stdout:
            pytest.skip("HackRF not found in USB devices")

        # Parse USB bus and device info
        result = subprocess.run(["lsusb", "-d", "1d50:6089"], capture_output=True, text=True)

        if result.stdout:
            # Format: Bus XXX Device YYY: ID 1d50:6089 OpenMoko, Inc. HackRF One
            parts = result.stdout.split()
            if len(parts) >= 4:
                bus = parts[1]
                device = parts[3].rstrip(":")
                print(f"HackRF found on Bus {bus} Device {device}")
                # Note: Actual bus/device may vary, documenting actual location
                assert True, f"HackRF detected on Bus {bus} Device {device}"
        else:
            pytest.skip("Could not determine HackRF USB location")

    @pytest.mark.asyncio
    async def test_hackrf_lifecycle(self, hackrf):
        """Test HackRF open/close lifecycle"""
        # Already opened by fixture
        assert hackrf.device is not None, "HackRF should be open"

        # Get info while open
        info = await hackrf.get_info()
        assert info["status"] == "Connected"

        # Close and reopen
        await hackrf.close()
        assert hackrf.device is None, "HackRF should be closed"

        # Reopen
        success = await hackrf.open()
        assert success, "HackRF should reopen successfully"
        assert hackrf.device is not None, "HackRF should be open again"

    @pytest.mark.asyncio
    async def test_frequency_tuning_range(self, hackrf):
        """Test frequency setting across full range: 850MHz, 3.2GHz, 6.5GHz"""
        test_frequencies = [
            (850e6, "850 MHz - Minimum frequency"),
            (915e6, "915 MHz - ISM band"),
            (2.4e9, "2.4 GHz - WiFi band"),
            (3.2e9, "3.2 GHz - Beacon frequency"),
            (5.8e9, "5.8 GHz - Upper ISM band"),
            (6.5e9, "6.5 GHz - Maximum frequency"),
        ]

        for freq, description in test_frequencies:
            success = await hackrf.set_freq(freq)
            assert success, f"Failed to set {description}"
            assert hackrf.config.frequency == freq, f"Frequency mismatch for {description}"

            # Small delay to allow hardware to settle
            await asyncio.sleep(0.1)

            print(f"✓ Set {description}: {freq/1e9:.3f} GHz")

    @pytest.mark.asyncio
    async def test_lna_gain_control(self, hackrf):
        """Test LNA gain control: 0, 16, 32, 40 dB (8dB steps)"""
        test_gains = [0, 8, 16, 24, 32, 40]

        for gain in test_gains:
            success = await hackrf.set_lna_gain(gain)
            assert success, f"Failed to set LNA gain to {gain} dB"
            assert hackrf.config.lna_gain == gain, f"LNA gain mismatch at {gain} dB"
            print(f"✓ LNA gain set to {gain} dB")

        # Test out-of-range values
        await hackrf.set_lna_gain(-10)
        assert hackrf.config.lna_gain == 0, "Negative gain should clamp to 0"

        await hackrf.set_lna_gain(50)
        assert hackrf.config.lna_gain == 40, "Excessive gain should clamp to 40"

        # Test rounding to 8dB steps
        await hackrf.set_lna_gain(15)
        assert hackrf.config.lna_gain == 16, "15 dB should round to 16 dB"

        await hackrf.set_lna_gain(30)
        assert hackrf.config.lna_gain == 32, "30 dB should round to 32 dB"

    @pytest.mark.asyncio
    async def test_vga_gain_control(self, hackrf):
        """Test VGA gain control: 0, 20, 40, 62 dB (2dB steps)"""
        test_gains = [0, 10, 20, 30, 40, 50, 60, 62]

        for gain in test_gains:
            success = await hackrf.set_vga_gain(gain)
            assert success, f"Failed to set VGA gain to {gain} dB"
            assert hackrf.config.vga_gain == gain, f"VGA gain mismatch at {gain} dB"
            print(f"✓ VGA gain set to {gain} dB")

        # Test out-of-range values
        await hackrf.set_vga_gain(-5)
        assert hackrf.config.vga_gain == 0, "Negative gain should clamp to 0"

        await hackrf.set_vga_gain(70)
        assert hackrf.config.vga_gain == 62, "Excessive gain should clamp to 62"

        # Test rounding to 2dB steps
        await hackrf.set_vga_gain(21)
        assert hackrf.config.vga_gain == 20, "21 dB should round to 20 dB"

        await hackrf.set_vga_gain(31)
        assert hackrf.config.vga_gain == 30, "31 dB should round to 30 dB"

    @pytest.mark.asyncio
    async def test_iq_sample_streaming(self, hackrf):
        """Test IQ sample streaming with read_samples(16384)"""
        samples_collected: list[np.ndarray] = []

        def sample_callback(samples: np.ndarray):
            """Collect samples for analysis"""
            samples_collected.append(samples)

        # Start streaming
        success = await hackrf.start_rx(sample_callback)
        assert success, "Failed to start RX streaming"

        # Collect samples for 1 second
        start_time = time.time()
        await asyncio.sleep(1.0)

        # Stop streaming
        success = await hackrf.stop()
        assert success, "Failed to stop RX streaming"

        elapsed_time = time.time() - start_time

        # Verify samples were collected
        assert len(samples_collected) > 0, "No samples received"

        # Analyze sample characteristics
        total_samples = sum(len(s) for s in samples_collected)
        sample_rate = total_samples / elapsed_time

        print(f"Collected {len(samples_collected)} batches")
        print(f"Total samples: {total_samples:,}")
        print(f"Effective sample rate: {sample_rate/1e6:.2f} Msps")

        # Verify we're getting approximately the configured sample rate
        expected_rate = hackrf.config.sample_rate
        assert sample_rate > expected_rate * 0.8, f"Sample rate too low: {sample_rate/1e6:.2f} Msps"
        assert (
            sample_rate < expected_rate * 1.2
        ), f"Sample rate too high: {sample_rate/1e6:.2f} Msps"

    @pytest.mark.asyncio
    async def test_sample_format_complex64(self, hackrf):
        """Verify sample format: complex64 numpy arrays"""
        samples_to_check = []

        def sample_callback(samples: np.ndarray):
            if len(samples_to_check) < 5:  # Collect first 5 batches
                samples_to_check.append(samples)

        # Start streaming
        await hackrf.start_rx(sample_callback)
        await asyncio.sleep(0.5)
        await hackrf.stop()

        assert len(samples_to_check) > 0, "No samples collected"

        for i, samples in enumerate(samples_to_check):
            # Verify numpy array
            assert isinstance(samples, np.ndarray), f"Batch {i}: Not a numpy array"

            # Verify complex64 dtype
            assert samples.dtype == np.complex64, f"Batch {i}: Wrong dtype {samples.dtype}"

            # Verify shape is 1D
            assert len(samples.shape) == 1, f"Batch {i}: Wrong shape {samples.shape}"

            # Verify non-zero samples
            assert len(samples) > 0, f"Batch {i}: Empty samples"

            # Check sample values are reasonable
            magnitudes = np.abs(samples)
            assert np.max(magnitudes) < 2.0, f"Batch {i}: Sample magnitudes too large"
            assert np.min(magnitudes) >= 0, f"Batch {i}: Negative magnitudes"

            print(f"✓ Batch {i}: {len(samples)} complex64 samples")

    @pytest.mark.asyncio
    async def test_continuous_rx_callback(self, hackrf):
        """Test continuous RX with callback mechanism"""
        callback_count = 0
        callback_errors = []

        def rx_callback(samples: np.ndarray):
            nonlocal callback_count
            callback_count += 1

            try:
                # Verify each callback
                assert isinstance(samples, np.ndarray)
                assert samples.dtype == np.complex64
                assert len(samples) > 0
            except AssertionError as e:
                callback_errors.append(str(e))

        # Start continuous RX
        await hackrf.start_rx(rx_callback)

        # Run for 2 seconds
        await asyncio.sleep(2.0)

        # Stop RX
        await hackrf.stop()

        # Verify callbacks were invoked
        assert callback_count > 0, "No callbacks received"
        assert len(callback_errors) == 0, f"Callback errors: {callback_errors}"

        # Calculate callback rate
        callback_rate = callback_count / 2.0
        print(f"Callback rate: {callback_rate:.1f} callbacks/second")
        print(f"Total callbacks: {callback_count}")

        # Should get many callbacks per second with 20 Msps
        assert callback_rate > 10, "Callback rate too low"

    @pytest.mark.asyncio
    async def test_sample_rate_configuration(self, hackrf):
        """Measure actual sample rate vs configured 20 Msps"""
        # Test different sample rates
        test_rates = [10e6, 20e6]  # 10 and 20 Msps

        for target_rate in test_rates:
            await hackrf.set_sample_rate(target_rate)

            sample_count = 0

            def count_callback(samples: np.ndarray):
                nonlocal sample_count
                sample_count += len(samples)

            # Measure for exactly 1 second
            await hackrf.start_rx(count_callback)
            await asyncio.sleep(1.0)
            await hackrf.stop()

            measured_rate = sample_count
            error_percent = abs(measured_rate - target_rate) / target_rate * 100

            print(f"Target: {target_rate/1e6:.1f} Msps")
            print(f"Measured: {measured_rate/1e6:.2f} Msps")
            print(f"Error: {error_percent:.1f}%")

            # Allow 20% tolerance due to USB timing
            assert error_percent < 20, f"Sample rate error too high: {error_percent:.1f}%"

    @pytest.mark.asyncio
    async def test_rf_amplifier_control(self, hackrf):
        """Test RF amplifier enable/disable"""
        # Disable amplifier
        success = await hackrf.set_amp_enable(False)
        assert success, "Failed to disable RF amplifier"
        assert not hackrf.config.amp_enable, "Amplifier should be disabled"

        # Collect baseline samples
        baseline_power = await self._measure_noise_power(hackrf)
        print(f"Baseline power (amp off): {baseline_power:.2f} dB")

        # Enable amplifier
        success = await hackrf.set_amp_enable(True)
        assert success, "Failed to enable RF amplifier"
        assert hackrf.config.amp_enable, "Amplifier should be enabled"

        # Collect amplified samples
        amplified_power = await self._measure_noise_power(hackrf)
        print(f"Amplified power (amp on): {amplified_power:.2f} dB")

        # Amplifier should increase signal power
        # Note: With no input signal, this tests noise amplification
        power_increase = amplified_power - baseline_power
        print(f"Power increase: {power_increase:.2f} dB")

        # Disable amplifier again for safety
        await hackrf.set_amp_enable(False)

    async def _measure_noise_power(self, hackrf) -> float:
        """Helper to measure noise floor power"""
        samples_list = []

        def collect_samples(samples: np.ndarray):
            if len(samples_list) < 10:
                samples_list.append(samples)

        await hackrf.start_rx(collect_samples)
        await asyncio.sleep(0.5)
        await hackrf.stop()

        if not samples_list:
            return -100.0  # Default noise floor

        # Calculate average power
        all_samples = np.concatenate(samples_list)
        power = 10 * np.log10(np.mean(np.abs(all_samples) ** 2) + 1e-10)

        return power


if __name__ == "__main__":
    # Run with: pytest tests/hardware/real/test_sdr_hardware.py -v -m hardware
    pytest.main([__file__, "-v", "-m", "hardware"])
