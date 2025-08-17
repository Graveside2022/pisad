"""Unit tests for SDRService.

Tests the SDR hardware interface layer including device enumeration,
configuration, streaming, and health monitoring functionality.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.backend.models.schemas import SDRConfig, SDRStatus
from src.backend.services.sdr_service import (
    SOAPY_AVAILABLE,
    SDRNotFoundError,
    SDRService,
)


class TestSDRService:
    """Test SDR hardware interface service."""

    @pytest.fixture
    def sdr_config(self):
        """Provide test SDR configuration."""
        return SDRConfig(
            frequency=433920000,  # 433.92 MHz
            sampleRate=2000000,  # 2 Msps
            gain=30,
            bandwidth=2000000,
        )

    @pytest.fixture
    def sdr_service(self):
        """Provide SDRService instance."""
        return SDRService()

    @pytest.fixture
    def mock_sdr_hardware(self):
        """Provide comprehensive SDR hardware mocking."""

        def _setup_mocks(sdr_service):
            """Set up all required mocks for SDR hardware."""
            # Mock device enumeration
            enumerate_patcher = patch.object(sdr_service, "enumerate_devices")
            mock_enumerate = enumerate_patcher.start()
            mock_enumerate.return_value = [
                {"driver": "hackrf", "label": "HackRF One #0", "serial": "000000000457c3dc27255c5f"}
            ]

            # Mock SoapySDR Device
            device_patcher = patch("SoapySDR.Device")
            mock_device_class = device_patcher.start()
            mock_device = MagicMock()

            # Mock frequency range for validation
            mock_range = MagicMock()
            mock_range.minimum.return_value = 24e6  # 24 MHz
            mock_range.maximum.return_value = 1.75e9  # 1.75 GHz (HackRF range)
            mock_device.getFrequencyRange.return_value = [mock_range]

            # Mock sample rates
            mock_device.listSampleRates.return_value = [1e6, 2e6, 4e6, 8e6, 10e6, 20e6]

            # Mock gain ranges
            mock_device.getGainRange.return_value = [0, 62]  # HackRF gain range

            mock_device_class.return_value = mock_device

            return {
                "enumerate_patcher": enumerate_patcher,
                "device_patcher": device_patcher,
                "mock_device": mock_device,
                "mock_enumerate": mock_enumerate,
            }

        return _setup_mocks

    def test_sdr_service_initialization(self, sdr_service):
        """Test SDRService initializes with correct defaults."""
        assert sdr_service.device is None
        assert isinstance(sdr_service.config, SDRConfig)
        assert sdr_service._stream_active is False
        assert sdr_service._health_check_task is None

    @pytest.mark.skipif(not SOAPY_AVAILABLE, reason="SoapySDR not available")
    def test_enumerate_devices(self):
        """Test device enumeration returns expected format."""
        with patch("SoapySDR.Device.enumerate") as mock_enumerate:
            mock_enumerate.return_value = [
                {"driver": "hackrf", "serial": "000000000001"},
                {"driver": "usrp", "type": "b200"},
            ]

            devices = SDRService.enumerate_devices()

            assert len(devices) == 2
            assert devices[0]["driver"] == "hackrf"
            assert devices[1]["driver"] == "usrp"
            mock_enumerate.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_with_config(self, sdr_service, sdr_config):
        """Test service initialization with configuration."""
        # Mock device enumeration to return available device
        with patch.object(sdr_service, "enumerate_devices") as mock_enumerate:
            mock_enumerate.return_value = [
                {"driver": "hackrf", "label": "HackRF One #0", "serial": "000000000457c3dc27255c5f"}
            ]

            with patch("SoapySDR.Device") as mock_device_class:
                mock_device = MagicMock()

                # Mock frequency range for validation
                mock_range = MagicMock()
                mock_range.minimum.return_value = 24e6  # 24 MHz
                mock_range.maximum.return_value = 1.75e9  # 1.75 GHz (HackRF range)
                mock_device.getFrequencyRange.return_value = [mock_range]

                # Mock sample rates
                mock_device.listSampleRates.return_value = [1e6, 2e6, 4e6, 8e6, 10e6, 20e6]

                # Mock gain ranges
                mock_device.getGainRange.return_value = [0, 62]  # HackRF gain range

                mock_device_class.return_value = mock_device

                await sdr_service.initialize(sdr_config)

                assert sdr_service.config == sdr_config
                assert sdr_service.device is not None
                mock_device_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_without_soapy_raises_error(self, sdr_service, sdr_config):
        """Test initialization fails gracefully when SoapySDR unavailable."""
        with patch.object(sdr_service, "__class__") as mock_class:
            mock_class.SOAPY_AVAILABLE = False

            with pytest.raises(SDRNotFoundError, match="SoapySDR not available"):
                await sdr_service.initialize(sdr_config)

    @pytest.mark.asyncio
    async def test_stream_iq_generates_complex_samples(self, sdr_service, sdr_config):
        """Test IQ streaming produces complex64 samples."""
        # Mock device enumeration
        with patch.object(sdr_service, "enumerate_devices") as mock_enumerate:
            mock_enumerate.return_value = [
                {"driver": "hackrf", "label": "HackRF One #0", "serial": "000000000457c3dc27255c5f"}
            ]

            with patch("SoapySDR.Device") as mock_device_class:
                mock_device = MagicMock()

                # Complete hardware mocking
                mock_range = MagicMock()
                mock_range.minimum.return_value = 24e6
                mock_range.maximum.return_value = 1.75e9
                mock_device.getFrequencyRange.return_value = [mock_range]
                mock_device.listSampleRates.return_value = [1e6, 2e6, 4e6, 8e6, 10e6, 20e6]
                mock_device.getGainRange.return_value = [0, 62]

                # Mock readStream to return sample data
                mock_samples = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex64)
                mock_device.readStream.return_value = (1024, 0)  # (samples_read, flags)

                mock_device_class.return_value = mock_device

                await sdr_service.initialize(sdr_config)

            # This should fail initially as streaming logic needs implementation
            async for samples in sdr_service.stream_iq():
                assert samples.dtype == np.complex64
                assert len(samples) > 0
                break  # Only test first batch

    @pytest.mark.asyncio
    async def test_set_frequency_updates_device(self, sdr_service, sdr_config):
        """Test frequency setting updates device configuration."""
        with patch("SoapySDR.Device") as mock_device_class:
            mock_device = MagicMock()
            mock_device_class.return_value = mock_device

            await sdr_service.initialize(sdr_config)

            new_frequency = 868000000  # 868 MHz
            sdr_service.set_frequency(new_frequency)

            assert sdr_service.config.frequency == new_frequency
            mock_device.setFrequency.assert_called_with(0, new_frequency)

    def test_get_status_returns_current_state(self, sdr_service, sdr_config):
        """Test status reporting includes device state."""
        with patch("SoapySDR.Device") as mock_device_class:
            mock_device = MagicMock()
            mock_device_class.return_value = mock_device
            mock_device.getFrequency.return_value = 433920000
            mock_device.getGain.return_value = 30

            sdr_service.config = sdr_config
            sdr_service.device = mock_device

            status = sdr_service.get_status()

            assert isinstance(status, SDRStatus)
            assert status.status == "CONNECTED"
            # Note: frequency check removed as SDRStatus doesn't have frequency field

    @pytest.mark.asyncio
    async def test_health_monitor_detects_disconnection(self, sdr_service, sdr_config):
        """Test health monitoring detects device disconnection."""
        with patch("SoapySDR.Device") as mock_device_class:
            mock_device = MagicMock()
            mock_device_class.return_value = mock_device

            # Simulate device disconnection
            mock_device.getFrequency.side_effect = RuntimeError("Device disconnected")

            await sdr_service.initialize(sdr_config)

            # Health monitor should detect and handle disconnection
            status = sdr_service.get_status()
            assert status.status == "ERROR"

    @pytest.mark.asyncio
    async def test_calibrate_returns_metrics(self, sdr_service, sdr_config):
        """Test calibration process returns performance metrics."""
        with patch("SoapySDR.Device") as mock_device_class:
            mock_device = MagicMock()
            mock_device_class.return_value = mock_device

            await sdr_service.initialize(sdr_config)

            metrics = await sdr_service.calibrate()

            assert isinstance(metrics, dict)
            assert "noise_floor" in metrics
            assert "frequency_accuracy" in metrics

    @pytest.mark.asyncio
    async def test_shutdown_cleanup(self, sdr_service, sdr_config):
        """Test proper cleanup during shutdown."""
        with patch("SoapySDR.Device") as mock_device_class:
            mock_device = MagicMock()
            mock_device_class.return_value = mock_device

            await sdr_service.initialize(sdr_config)
            await sdr_service.shutdown()

            assert sdr_service.device is None
            assert sdr_service._stream_active is False
