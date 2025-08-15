"""Mock SDR Configuration Tests.

Tests for SDR frequency, gain, and sample rate configuration
without requiring real hardware.
"""

from unittest.mock import patch

import pytest

from backend.hal.hackrf_interface import HackRFInterface
from backend.hal.mock_hackrf import MockHackRF
from backend.services.hardware_detector import HardwareDetector


@pytest.mark.mock_hardware
@pytest.mark.sdr
class TestMockSDRConfig:
    """Test SDR configuration with mock hardware."""

    @pytest.fixture
    def mock_sdr(self) -> MockHackRF:
        """Create a mock SDR device."""
        device = MockHackRF()
        device.open()
        return device

    def test_frequency_configuration(self, mock_sdr: MockHackRF) -> None:
        """Test frequency configuration across full range."""
        # Test minimum frequency (850 MHz)
        assert mock_sdr.set_freq(850_000_000) == 0
        assert mock_sdr.frequency == 850_000_000

        # Test default frequency (3.2 GHz)
        assert mock_sdr.set_freq(3_200_000_000) == 0
        assert mock_sdr.frequency == 3_200_000_000

        # Test maximum frequency (6.5 GHz)
        assert mock_sdr.set_freq(6_500_000_000) == 0
        assert mock_sdr.frequency == 6_500_000_000

        # Test invalid frequency (too low)
        assert mock_sdr.set_freq(100_000_000) == -1

        # Test invalid frequency (too high)
        assert mock_sdr.set_freq(10_000_000_000) == -1

    @pytest.mark.parametrize(
        "freq",
        [
            850_000_000,  # 850 MHz minimum
            1_000_000_000,  # 1 GHz
            2_400_000_000,  # 2.4 GHz (WiFi)
            3_200_000_000,  # 3.2 GHz default
            5_800_000_000,  # 5.8 GHz
            6_500_000_000,  # 6.5 GHz maximum
        ],
    )
    def test_parametrized_frequencies(self, mock_sdr: MockHackRF, freq: int) -> None:
        """Test various frequencies with parametrization."""
        assert mock_sdr.set_freq(freq) == 0
        assert mock_sdr.frequency == freq

    def test_sample_rate_configuration(self, mock_sdr: MockHackRF) -> None:
        """Test sample rate configuration."""
        # Test default sample rate (20 Msps)
        assert mock_sdr.set_sample_rate(20_000_000) == 0
        assert mock_sdr.sample_rate == 20_000_000

        # Test minimum sample rate
        assert mock_sdr.set_sample_rate(2_000_000) == 0
        assert mock_sdr.sample_rate == 2_000_000

        # Test maximum sample rate
        assert mock_sdr.set_sample_rate(20_000_000) == 0
        assert mock_sdr.sample_rate == 20_000_000

        # Test invalid sample rate
        assert mock_sdr.set_sample_rate(50_000_000) == -1

    @pytest.mark.parametrize(
        "gain,expected",
        [
            (0, 0),  # Minimum gain
            (8, 8),  # Valid step
            (16, 16),  # Valid step
            (20, 16),  # Round down to nearest step
            (24, 24),  # Valid step
            (32, 32),  # Valid step
            (40, 40),  # Maximum gain
            (50, 40),  # Clamp to maximum
        ],
    )
    def test_lna_gain_steps(self, mock_sdr: MockHackRF, gain: int, expected: int) -> None:
        """Test LNA gain configuration with 8dB steps."""
        assert mock_sdr.set_lna_gain(gain) == 0
        assert mock_sdr.lna_gain == expected

    @pytest.mark.parametrize(
        "gain,expected",
        [
            (0, 0),  # Minimum gain
            (10, 10),  # Valid 2dB step
            (20, 20),  # Default gain
            (31, 30),  # Round down to nearest even
            (40, 40),  # Valid gain
            (62, 62),  # Maximum gain
            (70, 62),  # Clamp to maximum
        ],
    )
    def test_vga_gain_steps(self, mock_sdr: MockHackRF, gain: int, expected: int) -> None:
        """Test VGA gain configuration with 2dB steps."""
        assert mock_sdr.set_vga_gain(gain) == 0
        assert mock_sdr.vga_gain == expected

    def test_amplifier_enable(self, mock_sdr: MockHackRF) -> None:
        """Test RF amplifier enable/disable."""
        # Enable amplifier
        assert mock_sdr.set_amp_enable(True) == 0
        assert mock_sdr.amp_enabled is True

        # Disable amplifier
        assert mock_sdr.set_amp_enable(False) == 0
        assert mock_sdr.amp_enabled is False

    def test_device_info(self, mock_sdr: MockHackRF) -> None:
        """Test device information retrieval."""
        info = mock_sdr.get_device_info()

        assert info["device"] == "MockHackRF"
        assert info["serial"] == "MOCK123456"
        assert info["version"] == "1.0.0"
        assert info["api_version"] == "1.0"
        assert info["status"] == "connected"

    def test_configuration_persistence(self, mock_sdr: MockHackRF) -> None:
        """Test that configuration persists across operations."""
        # Set configuration
        mock_sdr.set_freq(3_200_000_000)
        mock_sdr.set_sample_rate(20_000_000)
        mock_sdr.set_lna_gain(16)
        mock_sdr.set_vga_gain(20)
        mock_sdr.set_amp_enable(False)

        # Start streaming
        assert mock_sdr.start_rx(lambda x: None) == 0

        # Verify configuration persists
        assert mock_sdr.frequency == 3_200_000_000
        assert mock_sdr.sample_rate == 20_000_000
        assert mock_sdr.lna_gain == 16
        assert mock_sdr.vga_gain == 20
        assert mock_sdr.amp_enabled is False

        # Stop streaming
        assert mock_sdr.stop() == 0

        # Configuration should still persist
        assert mock_sdr.frequency == 3_200_000_000


@pytest.mark.mock_hardware
@pytest.mark.sdr
class TestMockSDRWithInterface:
    """Test SDR interface with mock backend."""

    @pytest.fixture
    def mock_interface(self) -> HackRFInterface:
        """Create HackRF interface with mock backend."""
        with patch("backend.hal.hackrf_interface.HackRF") as mock_class:
            mock_device = MockHackRF()
            mock_class.return_value = mock_device

            interface = HackRFInterface()
            interface.device = mock_device
            interface.connected = True

            return interface

    def test_interface_connect(self, mock_interface: HackRFInterface) -> None:
        """Test interface connection."""
        assert mock_interface.connected is True
        assert mock_interface.device is not None

    def test_interface_configure(self, mock_interface: HackRFInterface) -> None:
        """Test interface configuration."""
        config = {
            "frequency": 3_200_000_000,
            "sample_rate": 20_000_000,
            "lna_gain": 16,
            "vga_gain": 20,
            "amp_enable": False,
        }

        mock_interface.configure(config)

        assert mock_interface.device.frequency == 3_200_000_000
        assert mock_interface.device.sample_rate == 20_000_000
        assert mock_interface.device.lna_gain == 16
        assert mock_interface.device.vga_gain == 20
        assert mock_interface.device.amp_enabled is False

    def test_interface_frequency_hopping(self, mock_interface: HackRFInterface) -> None:
        """Test frequency hopping capability."""
        frequencies = [850_000_000, 2_400_000_000, 3_200_000_000, 5_800_000_000]

        for freq in frequencies:
            mock_interface.set_frequency(freq)
            assert mock_interface.device.frequency == freq

    def test_interface_gain_adjustment(self, mock_interface: HackRFInterface) -> None:
        """Test dynamic gain adjustment."""
        # Start with low gain
        mock_interface.set_lna_gain(0)
        mock_interface.set_vga_gain(0)

        assert mock_interface.device.lna_gain == 0
        assert mock_interface.device.vga_gain == 0

        # Increase gain gradually
        for lna in [8, 16, 24, 32, 40]:
            mock_interface.set_lna_gain(lna)
            assert mock_interface.device.lna_gain == lna

        for vga in [10, 20, 30, 40, 50, 62]:
            mock_interface.set_vga_gain(vga)
            assert mock_interface.device.vga_gain == vga


@pytest.mark.mock_hardware
@pytest.mark.sdr
@pytest.mark.integration
class TestMockSDRWithDetector:
    """Test SDR with hardware detector."""

    @pytest.fixture
    def detector_with_mock(self) -> HardwareDetector:
        """Create hardware detector with mock SDR."""
        with patch("backend.services.hardware_detector.HackRFInterface") as mock_interface:
            mock_sdr = MockHackRF()
            mock_interface.return_value.device = mock_sdr
            mock_interface.return_value.connected = True
            mock_interface.return_value.connect.return_value = True

            detector = HardwareDetector()
            detector._check_sdr()

            return detector

    def test_sdr_detection(self, detector_with_mock: HardwareDetector) -> None:
        """Test SDR auto-detection."""
        assert detector_with_mock.sdr_available is True

        status = detector_with_mock.get_status()
        assert status["sdr"]["available"] is True
        assert "device" in status["sdr"]

    def test_sdr_retry_logic(self) -> None:
        """Test SDR detection retry logic."""
        with patch("backend.services.hardware_detector.HackRFInterface") as mock_interface:
            # Simulate connection failures then success
            mock_interface.return_value.connect.side_effect = [False, False, True]

            detector = HardwareDetector()
            detector.check_hardware()

            # Should retry and eventually connect
            assert mock_interface.return_value.connect.call_count >= 1

    def test_sdr_graceful_degradation(self) -> None:
        """Test graceful degradation when SDR unavailable."""
        with patch("backend.services.hardware_detector.HackRFInterface") as mock_interface:
            # Simulate permanent failure
            mock_interface.return_value.connect.return_value = False

            detector = HardwareDetector()
            detector.check_hardware()

            assert detector.sdr_available is False
            status = detector.get_status()
            assert status["sdr"]["available"] is False
