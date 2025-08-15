"""
Test HackRF API Verification - Priority 0
Verifies the HackRF interface is using the correct API
"""

import asyncio
import os
import subprocess

import numpy as np
import pytest

from src.backend.hal.hackrf_interface import HackRFConfig, HackRFInterface, auto_detect_hackrf


@pytest.mark.hardware
class TestHackRFAPIVerification:
    """Priority 0 - MUST verify HackRF API is correct"""

    def test_hackrf_hardware_detection(self):
        """Verify HackRF hardware is detected via lsusb"""
        result = subprocess.run(["lsusb"], capture_output=True, text=True)

        # Check for HackRF in USB devices
        hackrf_found = "HackRF" in result.stdout or "1d50:6089" in result.stdout

        if not hackrf_found:
            pytest.skip("HackRF hardware not connected")

        assert hackrf_found, "HackRF should be detected in USB devices"

    def test_libhackrf_installed(self):
        """Verify libhackrf.so is installed"""
        # Check common library paths
        lib_paths = [
            "/usr/local/lib/libhackrf.so",
            "/usr/lib/libhackrf.so",
            "/usr/lib/x86_64-linux-gnu/libhackrf.so",
            "/usr/lib/aarch64-linux-gnu/libhackrf.so",  # For Raspberry Pi
        ]

        lib_found = any(os.path.exists(path) for path in lib_paths)

        if not lib_found:
            pytest.skip("libhackrf.so not found - install with: sudo apt-get install libhackrf-dev")

        assert lib_found, "libhackrf.so should be installed"

    @pytest.mark.asyncio
    async def test_auto_detect_hackrf(self):
        """Test auto_detect_hackrf returns valid interface"""
        hackrf = await auto_detect_hackrf()

        if hackrf is None:
            pytest.skip("HackRF auto-detection failed - hardware may not be connected")

        try:
            assert hackrf is not None, "auto_detect_hackrf should return HackRFInterface"
            assert isinstance(hackrf, HackRFInterface)

            # Check that device is open
            assert hackrf.device is not None, "HackRF device should be open"

            # Get device info
            info = await hackrf.get_info()
            assert info["status"] == "Connected", "HackRF should be connected"
            assert "frequency" in info
            assert "sample_rate" in info

        finally:
            if hackrf:
                await hackrf.close()

    @pytest.mark.asyncio
    async def test_hackrf_basic_operations(self):
        """Test basic HackRF operations: open, set_freq, set_gains, close"""
        config = HackRFConfig(
            frequency=3.2e9,  # 3.2 GHz
            sample_rate=20e6,  # 20 Msps
            lna_gain=16,
            vga_gain=20,
        )

        hackrf = HackRFInterface(config)

        # Test open
        success = await hackrf.open()
        if not success:
            pytest.skip("Failed to open HackRF - hardware may not be connected")

        try:
            assert success, "HackRF should open successfully"

            # Test frequency setting
            assert await hackrf.set_freq(2.4e9), "Should set frequency to 2.4 GHz"
            assert hackrf.config.frequency == 2.4e9

            # Test gain settings
            assert await hackrf.set_lna_gain(24), "Should set LNA gain"
            assert hackrf.config.lna_gain == 24  # Rounded to 8dB step

            assert await hackrf.set_vga_gain(30), "Should set VGA gain"
            assert hackrf.config.vga_gain == 30  # Rounded to 2dB step

            # Test amplifier control
            assert await hackrf.set_amp_enable(True), "Should enable amplifier"
            assert hackrf.config.amp_enable == True

            assert await hackrf.set_amp_enable(False), "Should disable amplifier"
            assert hackrf.config.amp_enable == False

            # Test sample rate
            assert await hackrf.set_sample_rate(10e6), "Should set sample rate to 10 Msps"
            assert hackrf.config.sample_rate == 10e6

        finally:
            await hackrf.close()

    @pytest.mark.asyncio
    async def test_hackrf_frequency_range(self):
        """Test HackRF frequency range: 850 MHz to 6.5 GHz"""
        hackrf = HackRFInterface()

        success = await hackrf.open()
        if not success:
            pytest.skip("Failed to open HackRF")

        try:
            # Test minimum frequency
            assert await hackrf.set_freq(850e6), "Should set 850 MHz"
            assert hackrf.config.frequency == 850e6

            # Test typical beacon frequency
            assert await hackrf.set_freq(3.2e9), "Should set 3.2 GHz"
            assert hackrf.config.frequency == 3.2e9

            # Test maximum frequency
            assert await hackrf.set_freq(6.5e9), "Should set 6.5 GHz"
            assert hackrf.config.frequency == 6.5e9

        finally:
            await hackrf.close()

    @pytest.mark.asyncio
    async def test_hackrf_gain_steps(self):
        """Test HackRF gain control steps"""
        hackrf = HackRFInterface()

        success = await hackrf.open()
        if not success:
            pytest.skip("Failed to open HackRF")

        try:
            # Test LNA gain steps (0-40 dB in 8dB steps)
            lna_steps = [0, 8, 16, 24, 32, 40]
            for gain in lna_steps:
                assert await hackrf.set_lna_gain(gain), f"Should set LNA gain to {gain} dB"
                assert hackrf.config.lna_gain == gain

            # Test VGA gain steps (0-62 dB in 2dB steps)
            vga_steps = [0, 10, 20, 30, 40, 50, 60, 62]
            for gain in vga_steps:
                assert await hackrf.set_vga_gain(gain), f"Should set VGA gain to {gain} dB"
                assert hackrf.config.vga_gain == gain

            # Test gain rounding
            await hackrf.set_lna_gain(15)  # Should round to 16
            assert hackrf.config.lna_gain == 16

            await hackrf.set_vga_gain(31)  # Should round to 30
            assert hackrf.config.vga_gain == 30

        finally:
            await hackrf.close()

    @pytest.mark.asyncio
    async def test_hackrf_sample_streaming(self):
        """Test HackRF IQ sample streaming"""
        hackrf = HackRFInterface()

        success = await hackrf.open()
        if not success:
            pytest.skip("Failed to open HackRF")

        try:
            samples_received = []

            def sample_callback(samples: np.ndarray):
                """Callback to receive samples"""
                samples_received.append(samples)

            # Start RX
            assert await hackrf.start_rx(sample_callback), "Should start RX"

            # Wait for samples
            await asyncio.sleep(0.5)

            # Stop RX
            assert await hackrf.stop(), "Should stop RX"

            # Verify samples were received
            assert len(samples_received) > 0, "Should receive samples"

            # Verify sample format
            for samples in samples_received[:5]:  # Check first 5 batches
                assert isinstance(samples, np.ndarray), "Samples should be numpy array"
                assert samples.dtype == np.complex64, "Samples should be complex64"
                assert len(samples) > 0, "Should have non-zero samples"

        finally:
            await hackrf.close()

    @pytest.mark.asyncio
    async def test_hackrf_api_differences_documented(self):
        """Document API differences between pyhackrf expectations and reality"""
        # This test documents the discovered API differences
        api_differences = {
            "module_name": "hackrf (not pyhackrf)",
            "class_name": "HackRF",
            "auto_open": "Device opens automatically on HackRF() creation",
            "frequency_method": "set_freq(int) - takes Hz as integer",
            "sample_rate_method": "set_sample_rate(int) - takes Hz as integer",
            "gain_methods": "set_lna_gain(int), set_vga_gain(int)",
            "amp_control": "enable_amp(), disable_amp() - no boolean parameter",
            "read_method": "read_samples(count) - returns numpy array",
            "serial_method": "get_serial_no() - returns string",
            "close_method": "close() - no return value",
        }

        # Document in test output
        print("\n=== HackRF API Documentation ===")
        for key, value in api_differences.items():
            print(f"{key}: {value}")

        # This test always passes - it's for documentation
        assert True, "API differences documented"


if __name__ == "__main__":
    # Run with: pytest tests/hardware/real/test_hackrf_api_verification.py -v -m hardware
    pytest.main([__file__, "-v", "-m", "hardware"])
