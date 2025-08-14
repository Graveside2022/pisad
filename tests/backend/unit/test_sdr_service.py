"""Unit tests for SDR service module."""

import asyncio
import contextlib
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock SoapySDR before importing the service
sys.modules["SoapySDR"] = MagicMock()

from src.backend.services.sdr_service import (  # noqa: E402
    SDRConfigError,
    SDRNotFoundError,
    SDRService,
    SDRStreamError,
)
from src.backend.models.schemas import SDRConfig, SDRStatus


@pytest.fixture
def mock_soapy():
    """Create a mock SoapySDR module."""
    with patch("src.backend.services.sdr_service.SoapySDR") as mock:
        # Mock constants
        mock.SOAPY_SDR_RX = 0
        mock.SOAPY_SDR_CF32 = "CF32"
        mock.SOAPY_SDR_HAS_TIME = 1
        mock.SOAPY_SDR_TIMEOUT = -1
        mock.errToStr = lambda x: f"Error {x}"

        yield mock


@pytest.fixture
def mock_device(mock_soapy):
    """Create a mock SDR device."""
    device = MagicMock()

    # Mock frequency range
    freq_range = MagicMock()
    freq_range.minimum.return_value = 1e6
    freq_range.maximum.return_value = 6e9
    device.getFrequencyRange.return_value = [freq_range]

    # Mock sample rates
    device.listSampleRates.return_value = [1e6, 2e6, 5e6, 10e6]

    # Mock gain capabilities
    device.hasGainMode.return_value = True
    device.listGains.return_value = ["LNA", "VGA", "AMP"]
    gain_range = MagicMock()
    gain_range.minimum.return_value = 0
    gain_range.maximum.return_value = 40
    device.getGainRange.return_value = gain_range

    # Mock antennas and sensors
    device.listAntennas.return_value = ["RX"]
    device.listSensors.return_value = ["temperature"]
    device.readSensor.return_value = "25.5"

    # Mock hardware info
    device.getHardwareKey.return_value = "hackrf"

    # Mock stream
    stream = MagicMock()
    device.setupStream.return_value = stream
    device.activateStream.return_value = None
    device.deactivateStream.return_value = None
    device.closeStream.return_value = None

    return device


@pytest.fixture
def sdr_config():
    """Create a default SDR configuration."""
    return SDRConfig(
        frequency=2.437e9, sampleRate=2e6, gain="AUTO", bandwidth=2e6, buffer_size=1024
    )


class TestSDRService:
    """Test SDR service functionality."""

    @pytest.mark.asyncio
    async def test_enumerate_devices(self, mock_soapy):
        """Test device enumeration."""
        mock_soapy.Device.enumerate.return_value = [
            {"driver": "hackrf", "label": "HackRF One"},
            {"driver": "uhd", "label": "USRP B205mini"},
        ]

        devices = SDRService.enumerate_devices()

        assert len(devices) == 2
        assert devices[0]["driver"] == "hackrf"
        assert devices[1]["driver"] == "uhd"
        mock_soapy.Device.enumerate.assert_called_once()

    @pytest.mark.asyncio
    async def test_enumerate_devices_empty(self, mock_soapy):
        """Test device enumeration with no devices."""
        mock_soapy.Device.enumerate.return_value = []

        devices = SDRService.enumerate_devices()

        assert len(devices) == 0

    @pytest.mark.asyncio
    async def test_enumerate_devices_error(self, mock_soapy):
        """Test device enumeration with error."""
        mock_soapy.Device.enumerate.side_effect = Exception("USB error")

        devices = SDRService.enumerate_devices()

        assert len(devices) == 0

    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_soapy, mock_device, sdr_config):
        """Test successful SDR initialization."""
        mock_soapy.Device.enumerate.return_value = [{"driver": "hackrf", "label": "HackRF One"}]
        mock_soapy.Device.return_value = mock_device

        service = SDRService()
        await service.initialize(sdr_config)

        assert service.status.status == "CONNECTED"
        assert service.status.device_name == "HackRF One"
        assert service.status.driver == "hackrf"
        assert service.device == mock_device

        # Verify configuration calls
        mock_device.setFrequency.assert_called_with(0, 0, 2.437e9)
        mock_device.setSampleRate.assert_called_with(0, 0, 2e6)
        mock_device.setBandwidth.assert_called_with(0, 0, 2e6)
        mock_device.setGainMode.assert_called_with(0, 0, True)

        await service.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_no_devices(self, mock_soapy):
        """Test initialization with no devices found."""
        mock_soapy.Device.enumerate.return_value = []

        service = SDRService()

        with pytest.raises(SDRNotFoundError):
            await service.initialize()

    @pytest.mark.asyncio
    async def test_initialize_invalid_frequency(self, mock_soapy, mock_device, sdr_config):
        """Test initialization with invalid frequency."""
        mock_soapy.Device.enumerate.return_value = [{"driver": "hackrf", "label": "HackRF One"}]
        mock_soapy.Device.return_value = mock_device

        # Set frequency outside range
        sdr_config.frequency = 10e9  # 10 GHz - outside range

        service = SDRService()

        with pytest.raises(SDRConfigError) as exc_info:
            await service.initialize(sdr_config)

        assert "out of range" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_manual_gain_setting(self, mock_soapy, mock_device):
        """Test manual gain configuration."""
        mock_soapy.Device.enumerate.return_value = [{"driver": "hackrf", "label": "HackRF One"}]
        mock_soapy.Device.return_value = mock_device

        config = SDRConfig(gain=20.0)  # Manual gain

        service = SDRService()
        await service.initialize(config)

        mock_device.setGain.assert_called_with(0, 0, 20.0)

        await service.shutdown()

    @pytest.mark.asyncio
    async def test_stream_iq_success(self, mock_soapy, mock_device, sdr_config):
        """Test successful IQ streaming."""
        mock_soapy.Device.enumerate.return_value = [{"driver": "hackrf", "label": "HackRF One"}]
        mock_soapy.Device.return_value = mock_device

        # Mock stream reads
        samples = np.random.randn(1024) + 1j * np.random.randn(1024)
        samples = samples.astype(np.complex64)

        read_results = [
            (1024, 0, 0),  # First read success
            (1024, 0, 0),  # Second read success
            (-1, 0, 0),  # Timeout
        ]
        mock_device.readStream.side_effect = read_results

        service = SDRService()
        await service.initialize(sdr_config)

        # Collect streamed samples
        received = []
        count = 0
        async for chunk in service.stream_iq():
            received.append(chunk)
            count += 1
            if count >= 2:
                service._stream_active = False

        assert len(received) == 2
        assert service.status.stream_active is False

        await service.shutdown()

    @pytest.mark.asyncio
    async def test_stream_iq_overflow(self, mock_soapy, mock_device, sdr_config):
        """Test IQ streaming with buffer overflow."""
        mock_soapy.Device.enumerate.return_value = [{"driver": "hackrf", "label": "HackRF One"}]
        mock_soapy.Device.return_value = mock_device

        # Mock stream read with overflow flag
        mock_device.readStream.return_value = (1024, mock_soapy.SOAPY_SDR_HAS_TIME, 0)

        service = SDRService()
        await service.initialize(sdr_config)

        count = 0
        async for _chunk in service.stream_iq():
            count += 1
            if count >= 1:
                service._stream_active = False

        assert service.status.buffer_overflows == 1

        await service.shutdown()

    @pytest.mark.asyncio
    async def test_stream_iq_error(self, mock_soapy, mock_device, sdr_config):
        """Test IQ streaming with error."""
        mock_soapy.Device.enumerate.return_value = [{"driver": "hackrf", "label": "HackRF One"}]
        mock_soapy.Device.return_value = mock_device

        # Mock stream read error
        mock_device.readStream.return_value = (-5, 0, 0)  # Error code

        service = SDRService()
        await service.initialize(sdr_config)

        with pytest.raises(SDRStreamError):
            async for _chunk in service.stream_iq():
                pass

        await service.shutdown()

    @pytest.mark.asyncio
    async def test_set_frequency(self, mock_soapy, mock_device, sdr_config):
        """Test frequency setting."""
        mock_soapy.Device.enumerate.return_value = [{"driver": "hackrf", "label": "HackRF One"}]
        mock_soapy.Device.return_value = mock_device

        service = SDRService()
        await service.initialize(sdr_config)

        # Change frequency
        new_freq = 2.45e9
        service.set_frequency(new_freq)

        assert service.config.frequency == new_freq
        mock_device.setFrequency.assert_called_with(0, 0, new_freq)

        await service.shutdown()

    @pytest.mark.asyncio
    async def test_get_status(self, mock_soapy, mock_device, sdr_config):
        """Test status retrieval."""
        mock_soapy.Device.enumerate.return_value = [{"driver": "hackrf", "label": "HackRF One"}]
        mock_soapy.Device.return_value = mock_device

        service = SDRService()
        await service.initialize(sdr_config)

        status = service.get_status()

        assert isinstance(status, SDRStatus)
        assert status.status == "CONNECTED"
        assert status.device_name == "HackRF One"
        assert status.driver == "hackrf"
        assert status.temperature == 25.5

        await service.shutdown()

    @pytest.mark.asyncio
    async def test_health_monitor(self, mock_soapy, mock_device, sdr_config):
        """Test health monitoring functionality."""
        mock_soapy.Device.enumerate.return_value = [{"driver": "hackrf", "label": "HackRF One"}]
        mock_soapy.Device.return_value = mock_device

        service = SDRService()
        await service.initialize(sdr_config)

        # Wait for health check
        await asyncio.sleep(0.1)

        # Simulate device disconnection
        mock_device.getHardwareKey.side_effect = Exception("Device disconnected")

        # Wait for health check to detect disconnection
        await asyncio.sleep(5.1)

        # Status should change to disconnected
        assert service.status.status == "DISCONNECTED"

        await service.shutdown()

    @pytest.mark.asyncio
    async def test_reconnection_logic(self, mock_soapy, mock_device, sdr_config):
        """Test automatic reconnection."""
        mock_soapy.Device.enumerate.return_value = []  # No devices initially

        service = SDRService()

        # First initialization should fail
        with pytest.raises(SDRNotFoundError):
            await service.initialize(sdr_config)

        assert service.status.status == "ERROR"

        # Now make device available
        mock_soapy.Device.enumerate.return_value = [{"driver": "hackrf", "label": "HackRF One"}]
        mock_soapy.Device.return_value = mock_device

        # Wait for reconnection attempt
        await asyncio.sleep(1.5)

        # Should reconnect
        assert service.status.status == "CONNECTED"

        await service.shutdown()

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_soapy, mock_device, sdr_config):
        """Test async context manager functionality."""
        mock_soapy.Device.enumerate.return_value = [{"driver": "hackrf", "label": "HackRF One"}]
        mock_soapy.Device.return_value = mock_device

        async with SDRService() as service:
            assert service.status.status == "CONNECTED"

        # Should be shutdown after context exit
        assert service.status.status == "DISCONNECTED"

    @pytest.mark.asyncio
    async def test_device_selection_with_args(self, mock_soapy, mock_device):
        """Test device selection with specific arguments."""
        mock_soapy.Device.enumerate.return_value = [
            {"driver": "hackrf", "label": "HackRF One", "serial": "12345"},
            {"driver": "uhd", "label": "USRP B205mini", "serial": "67890"},
        ]
        mock_soapy.Device.return_value = mock_device

        config = SDRConfig(device_args="driver=uhd")

        service = SDRService()
        device_dict = service._select_device(config.device_args)

        assert device_dict["driver"] == "uhd"
        assert device_dict["label"] == "USRP B205mini"

    @pytest.mark.asyncio
    async def test_unsupported_sample_rate(self, mock_soapy, mock_device):
        """Test handling of unsupported sample rate."""
        mock_soapy.Device.enumerate.return_value = [{"driver": "hackrf", "label": "HackRF One"}]
        mock_soapy.Device.return_value = mock_device

        # Request unsupported rate
        config = SDRConfig(sampleRate=3e6)  # 3 Msps not in supported list

        service = SDRService()
        await service.initialize(config)

        # Should use closest supported rate (2 Msps)
        assert service.config.sampleRate == 2e6
        mock_device.setSampleRate.assert_called_with(0, 0, 2e6)

        await service.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_cleanup(self, mock_soapy, mock_device, sdr_config):
        """Test proper cleanup on shutdown."""
        mock_soapy.Device.enumerate.return_value = [{"driver": "hackrf", "label": "HackRF One"}]
        mock_soapy.Device.return_value = mock_device

        # Mock readStream to return timeout
        mock_device.readStream.return_value = (-1, 0, 0)  # Timeout

        service = SDRService()
        await service.initialize(sdr_config)

        # Start streaming
        stream_task = asyncio.create_task(service.stream_iq().__anext__())
        await asyncio.sleep(0.1)

        # Shutdown
        await service.shutdown()

        # Verify cleanup
        assert service.status.status == "DISCONNECTED"
        assert service._stream_active is False
        assert service.device is None

        # Tasks should be cancelled
        assert service._health_check_task is None or service._health_check_task.cancelled()
        assert service._reconnect_task is None or service._reconnect_task.cancelled()

        stream_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
            await stream_task
