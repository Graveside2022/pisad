"""
Comprehensive HAL failure scenario tests for safety-critical coverage.

Tests hardware abstraction layer failure modes, error recovery,
circuit breaker patterns, and graceful degradation per PRD requirements.
"""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.backend.core.exceptions import MAVLinkError, SDRError
from src.backend.hal.hackrf_interface import HackRFConfig, HackRFInterface, auto_detect_hackrf
from src.backend.hal.mavlink_interface import MAVLinkConfig, MAVLinkInterface
from src.backend.hal.mock_hackrf import MockHackRF
from src.backend.services.safety_manager import SafetyManager


class TestHackRFFailureScenarios:
    """Test HackRF hardware failure scenarios for safety validation."""

    @pytest.fixture
    def hackrf_config(self):
        """Provide HackRF test configuration."""
        return HackRFConfig(
            frequency=3.2e9, sample_rate=20e6, lna_gain=16, vga_gain=20, amp_enable=False
        )

    @pytest.fixture
    def hackrf_interface(self, hackrf_config):
        """Provide HackRF interface with mock device."""
        interface = HackRFInterface(hackrf_config)
        # Force using mock for testing
        interface.device = MockHackRF()
        return interface

    @pytest.mark.asyncio
    async def test_hackrf_device_disconnect_during_streaming(self, hackrf_interface):
        """Test HackRF device disconnect during active streaming (FR13 failure mode)."""
        # Start streaming
        callback = Mock()
        assert await hackrf_interface.start_rx(callback) is True

        # Simulate device disconnect by clearing device reference (realistic failure)
        hackrf_interface.device = None

        # Should detect disconnect and stop gracefully
        assert await hackrf_interface.stop() is False  # Expect False when device None

        # Device should be marked as disconnected
        info = await hackrf_interface.get_info()
        assert info["status"] == "Not connected"

    @pytest.mark.asyncio
    async def test_hackrf_frequency_set_failure(self, hackrf_interface):
        """Test HackRF frequency setting failure (invalid frequency range)."""
        # Test frequency outside valid range (850 MHz - 6.5 GHz per PRD-FR1)
        invalid_freq = 100e6  # 100 MHz - below minimum

        # The mock returns -1 for invalid frequency, interface should handle gracefully
        result = await hackrf_interface.set_freq(invalid_freq)

        # Should return False for invalid frequency per HackRF interface spec
        assert result is False

        # Original frequency should be preserved
        assert hackrf_interface.config.frequency == 3.2e9

    @pytest.mark.asyncio
    async def test_hackrf_sample_rate_failure(self, hackrf_interface):
        """Test HackRF sample rate setting failure."""
        # Test invalid sample rate (outside 2-20 Msps range)
        invalid_rate = 100e6  # 100 Msps - above maximum

        result = await hackrf_interface.set_sample_rate(invalid_rate)

        # Should fail gracefully with validation
        assert result is False

        # Original sample rate should be preserved
        assert hackrf_interface.config.sample_rate == 20e6

    @pytest.mark.asyncio
    async def test_hackrf_gain_setting_boundary_conditions(self, hackrf_interface):
        """Test HackRF gain setting boundary conditions and failures."""
        # Test LNA gain beyond valid range (0-40 dB)
        result = await hackrf_interface.set_lna_gain(50)  # Above maximum

        # Should clamp to valid range
        assert result is True
        assert hackrf_interface.config.lna_gain <= 40

        # Test VGA gain beyond valid range (0-62 dB)
        result = await hackrf_interface.set_vga_gain(70)  # Above maximum

        # Should clamp to valid range
        assert result is True
        assert hackrf_interface.config.vga_gain <= 62

    @pytest.mark.asyncio
    async def test_hackrf_streaming_callback_failure(self, hackrf_interface):
        """Test HackRF streaming with callback that throws exceptions."""
        # Ensure device is in connected state for this test
        hackrf_interface.device.connected = True

        def failing_callback(samples):
            raise ValueError("Callback processing error")

        # Start streaming with failing callback
        result = await hackrf_interface.start_rx(failing_callback)
        assert result is True

        # Allow some time for potential callback errors
        await asyncio.sleep(0.1)

        # Should continue streaming despite callback errors
        assert hackrf_interface._rx_active is True

        # Stop streaming
        await hackrf_interface.stop()

    @pytest.mark.asyncio
    async def test_hackrf_device_busy_error(self, hackrf_interface):
        """Test HackRF device busy error (already streaming)."""
        callback = Mock()

        # Start first streaming session
        assert await hackrf_interface.start_rx(callback) is True

        # Try to start second streaming session
        result = await hackrf_interface.start_rx(callback)

        # Should fail because device is already streaming
        assert result is False

        # Stop streaming
        await hackrf_interface.stop()

    @pytest.mark.asyncio
    async def test_hackrf_close_without_open(self, hackrf_interface):
        """Test HackRF close operation without device being opened."""
        # Ensure device is not opened
        hackrf_interface.device = None

        # Should handle close gracefully
        await hackrf_interface.close()

        # No exceptions should be raised

    @pytest.mark.asyncio
    async def test_hackrf_configuration_after_device_loss(self, hackrf_interface):
        """Test HackRF configuration operations after device is lost."""
        # Simulate device loss
        hackrf_interface.device = None

        # All configuration operations should fail gracefully
        assert await hackrf_interface.set_freq(3.2e9) is False
        assert await hackrf_interface.set_sample_rate(20e6) is False
        assert await hackrf_interface.set_lna_gain(16) is False
        assert await hackrf_interface.set_vga_gain(20) is False
        assert await hackrf_interface.set_amp_enable(True) is False

    @pytest.mark.asyncio
    async def test_hackrf_mock_streaming_interrupt(self, hackrf_interface):
        """Test HackRF mock streaming interruption scenarios."""
        callback = Mock()

        # Start streaming
        assert await hackrf_interface.start_rx(callback) is True

        # Simulate streaming thread interruption
        hackrf_interface._running = False

        # Wait for stream to stop
        await asyncio.sleep(0.1)

        # Should detect interruption and stop gracefully
        assert hackrf_interface._rx_active is False


class TestMAVLinkFailureScenarios:
    """Test MAVLink hardware failure scenarios for safety validation."""

    @pytest.fixture
    def mavlink_config(self):
        """Provide MAVLink test configuration."""
        return MAVLinkConfig(
            device="/dev/ttyACM0", baudrate=115200, source_system=255, source_component=190
        )

    @pytest.fixture
    def mavlink_interface(self, mavlink_config):
        """Provide MAVLink interface for testing."""
        return MAVLinkInterface(mavlink_config)

    @pytest.mark.asyncio
    async def test_mavlink_connection_timeout(self, mavlink_interface):
        """Test MAVLink connection timeout (no heartbeat received)."""
        # Mock mavutil to simulate no device
        with patch("src.backend.hal.mavlink_interface.mavutil") as mock_mavutil:
            mock_connection = Mock()
            mock_connection.wait_heartbeat.return_value = None  # No heartbeat
            mock_mavutil.mavlink_connection.return_value = mock_connection

            result = await mavlink_interface.connect()

            # Should fail to connect
            assert result is False
            # Connection might be set but marked as failed due to no heartbeat
            assert not mavlink_interface._running

    @pytest.mark.asyncio
    async def test_mavlink_device_permission_error(self, mavlink_interface):
        """Test MAVLink device permission error (access denied)."""
        with patch("src.backend.hal.mavlink_interface.mavutil") as mock_mavutil:
            mock_mavutil.mavlink_connection.side_effect = PermissionError("Permission denied")

            result = await mavlink_interface.connect()

            # Should fail gracefully
            assert result is False

    @pytest.mark.asyncio
    async def test_mavlink_velocity_command_without_connection(self, mavlink_interface):
        """Test MAVLink velocity command without active connection (FR15 safety)."""
        # Ensure no connection
        mavlink_interface.connection = None

        result = await mavlink_interface.send_velocity_ned(1.0, 0.0, 0.0, 0.1)

        # Should fail safely
        assert result is False

    @pytest.mark.asyncio
    async def test_mavlink_velocity_command_transmission_error(self, mavlink_interface):
        """Test MAVLink velocity command transmission error."""
        # Mock connection with transmission error
        mock_connection = Mock()
        mock_connection.mav.set_position_target_local_ned_send.side_effect = MAVLinkError(
            "Transmission failed"
        )
        mavlink_interface.connection = mock_connection

        result = await mavlink_interface.send_velocity_ned(1.0, 0.0, 0.0, 0.1)

        # Should handle error gracefully
        assert result is False

    @pytest.mark.asyncio
    async def test_mavlink_telemetry_read_failure(self, mavlink_interface):
        """Test MAVLink telemetry read failure scenarios."""
        # Mock connection without telemetry
        mock_connection = Mock()
        mock_connection.recv_match.return_value = None
        mavlink_interface.connection = mock_connection

        # All telemetry functions should return safe defaults
        position = await mavlink_interface.get_position()
        assert position == (0, 0, 0)

        battery = await mavlink_interface.get_battery()
        assert battery == {"voltage": 0, "current": 0, "remaining": 0}

        mode = await mavlink_interface.get_flight_mode()
        assert mode == "UNKNOWN"

        channels = await mavlink_interface.get_rc_channels()
        assert channels == {}

        gps = await mavlink_interface.get_gps_status()
        assert gps == {"fix_type": 0, "satellites": 0, "hdop": 99.99}

    @pytest.mark.asyncio
    async def test_mavlink_mode_change_failure(self, mavlink_interface):
        """Test MAVLink mode change failure scenarios."""
        mock_connection = Mock()
        mock_connection.mode_mapping.return_value = None  # No mode mapping available
        mavlink_interface.connection = mock_connection

        result = await mavlink_interface.set_mode("GUIDED")

        # Should fail gracefully
        assert result is False

    @pytest.mark.asyncio
    async def test_mavlink_arm_disarm_command_ack_timeout(self, mavlink_interface):
        """Test MAVLink arm/disarm command ACK timeout."""
        mock_connection = Mock()
        mock_connection.recv_match.return_value = None  # No ACK received
        mavlink_interface.connection = mock_connection

        # Test arm command timeout
        result = await mavlink_interface.arm()
        assert result is False

        # Test disarm command timeout
        result = await mavlink_interface.disarm()
        assert result is False

    @pytest.mark.asyncio
    async def test_mavlink_connection_loss_during_operation(self, mavlink_interface):
        """Test MAVLink connection loss during operation (FR10 trigger)."""
        # Simulate initial connection
        mock_connection = Mock()
        mavlink_interface.connection = mock_connection
        mavlink_interface._running = True

        # Simulate connection loss
        mavlink_interface.connection = None

        # Operations should fail safely
        result = await mavlink_interface.send_velocity_ned(1.0, 0.0, 0.0)
        assert result is False

        result = await mavlink_interface.send_statustext("Test message")
        assert result is False

    @pytest.mark.asyncio
    async def test_mavlink_close_connection_error(self, mavlink_interface):
        """Test MAVLink close with connection error."""
        mock_connection = Mock()
        mock_connection.close.side_effect = ConnectionError("Close failed")
        mavlink_interface.connection = mock_connection
        mavlink_interface._running = True

        # Should handle close error gracefully
        await mavlink_interface.close()

        # Connection should be cleared despite error
        assert mavlink_interface.connection is None
        assert mavlink_interface._running is False


class TestHALIntegratedFailureScenarios:
    """Test integrated HAL failure scenarios with safety systems."""

    @pytest.fixture
    def safety_manager(self):
        """Provide SafetyManager for integrated testing."""
        return SafetyManager()

    @pytest.mark.asyncio
    async def test_dual_hal_failure_safety_response(self, safety_manager):
        """Test safety system response to simultaneous HAL failures."""
        # Mock both MAVLink and HackRF failures
        mock_mavlink = Mock()
        mock_mavlink.emergency_stop.side_effect = MAVLinkError("MAVLink failed")
        mock_hackrf = Mock()
        mock_hackrf.close.side_effect = SDRError("HackRF failed")

        safety_manager.mavlink = mock_mavlink

        # Emergency stop should still execute with fallback
        result = safety_manager.trigger_emergency_stop()

        # Should complete with error handling and execute fallback
        assert isinstance(result, dict)
        assert "response_time_ms" in result
        
        # Should report that fallback was attempted
        assert "fallback_attempted" in result
        assert result["fallback_attempted"] is True
        assert "error" in result  # Original error should be logged
        
        # Fallback should succeed even if primary method failed
        assert result["success"] is True  # Fallback executed successfully

    @pytest.mark.asyncio
    async def test_hal_timeout_during_emergency_stop(self, safety_manager):
        """Test emergency stop timing with HAL timeout (FR16 - <500ms requirement)."""

        # Mock slow MAVLink response
        def slow_emergency_stop():
            time.sleep(0.1)  # 100ms delay
            return True

        mock_mavlink = Mock()
        mock_mavlink.emergency_stop = slow_emergency_stop
        safety_manager.mavlink = mock_mavlink

        start_time = time.perf_counter()
        result = safety_manager.trigger_emergency_stop()
        end_time = time.perf_counter()

        actual_time_ms = (end_time - start_time) * 1000

        # Should complete within safety requirements
        assert actual_time_ms < 500  # FR16 requirement
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_hal_device_enumeration_failure(self):
        """Test HAL device enumeration failure scenarios."""
        # Test HackRF enumeration failure
        with patch("src.backend.hal.hackrf_interface.hackrf") as mock_hackrf_module:
            mock_hackrf_module.HackRF.side_effect = SDRError("No devices found")

            from src.backend.hal.hackrf_interface import auto_detect_hackrf

            result = await auto_detect_hackrf()

            assert result is None

    @pytest.mark.asyncio
    async def test_hal_configuration_corruption_recovery(self):
        """Test HAL configuration corruption and recovery."""
        config = HackRFConfig()
        interface = HackRFInterface(config)

        # Interface should handle invalid configuration gracefully
        mock_device = MockHackRF()
        interface.device = mock_device

        # Test with invalid frequency (outside valid range)
        result = await interface.set_freq(-1)  # Invalid frequency
        assert result is False

        # Test with invalid sample rate (outside valid range)
        result = await interface.set_sample_rate(0)  # Invalid sample rate
        assert result is False

        # Valid configuration should still work
        result = await interface.set_freq(3.2e9)  # Valid frequency
        assert result is True

        result = await interface.set_sample_rate(20e6)  # Valid sample rate
        assert result is True

    @pytest.mark.asyncio
    async def test_hal_memory_exhaustion_scenario(self):
        """Test HAL behavior under memory exhaustion."""
        interface = HackRFInterface()
        interface.device = MockHackRF()

        def memory_exhaustion_callback(samples):
            # Simulate memory allocation failure
            raise MemoryError("Out of memory")

        # Should handle memory errors gracefully
        result = await interface.start_rx(memory_exhaustion_callback)
        assert result is True

        # Allow brief processing time
        await asyncio.sleep(0.05)

        # Should continue operating despite callback failures
        await interface.stop()

    @pytest.mark.asyncio
    async def test_hal_concurrent_access_safety(self):
        """Test HAL thread safety under concurrent access."""
        interface = HackRFInterface()
        interface.device = MockHackRF()

        # Multiple concurrent configuration attempts
        tasks = []
        for i in range(5):
            tasks.append(interface.set_freq(3.2e9 + i * 1e6))
            tasks.append(interface.set_lna_gain(16 + i))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Should handle concurrent access without crashes
        for result in results:
            assert not isinstance(result, Exception)

    @pytest.mark.asyncio
    async def test_hal_resource_cleanup_on_failure(self):
        """Test proper resource cleanup when HAL operations fail."""
        interface = HackRFInterface()

        # Simulate device initialization failure
        with patch.object(interface, "device", None):
            await interface.close()

        # Should handle cleanup gracefully
        assert interface.device is None

    def test_hal_mock_failure_injection_framework(self):
        """Test HAL mock framework supports failure injection."""
        mock_device = MockHackRF()

        # Test device can simulate various failure states
        assert hasattr(mock_device, "connected")
        assert hasattr(mock_device, "is_streaming")

        # Test failure state simulation
        mock_device.connected = False
        assert mock_device.connected is False

        mock_device.is_streaming = False
        assert mock_device.is_streaming is False

    @pytest.mark.asyncio
    async def test_hal_circuit_breaker_pattern(self):
        """Test circuit breaker pattern for HAL failure recovery."""
        interface = HackRFInterface()
        failure_count = 0

        # Simulate repeated failures
        class FailingDevice:
            def set_freq(self, freq):
                nonlocal failure_count
                failure_count += 1
                if failure_count < 3:
                    raise SDRError(f"Failure {failure_count}")
                return 0  # Success on third attempt

        interface.device = FailingDevice()

        # Should implement retry logic (circuit breaker pattern)
        # First attempts should fail
        result1 = await interface.set_freq(3.2e9)
        assert result1 is False

        result2 = await interface.set_freq(3.2e9)
        assert result2 is False

        # Third attempt should succeed
        result3 = await interface.set_freq(3.2e9)
        assert result3 is True


class TestHackRFInterfaceAdditionalCoverage:
    """Additional tests to improve HackRF interface coverage to 85%+."""

    @pytest.fixture
    def hackrf_config(self):
        """Provide HackRF test configuration."""
        return HackRFConfig(
            frequency=3.2e9, sample_rate=20e6, lna_gain=16, vga_gain=20, amp_enable=True
        )

    @pytest.mark.asyncio
    async def test_hackrf_open_device_creation_success(self):
        """Test successful HackRF device opening and configuration (open method coverage)."""
        interface = HackRFInterface()

        # Create a mock device that simulates successful opening
        mock_device = MockHackRF()

        # Patch hackrf module to return our mock
        with patch("src.backend.hal.hackrf_interface.hackrf") as mock_hackrf_module:
            mock_hackrf_module.HackRF.return_value = mock_device
            mock_device.get_serial_no = Mock(return_value="MOCK123456")

            result = await interface.open()

            # Should succeed
            assert result is True
            assert interface.device is not None

    @pytest.mark.asyncio
    async def test_hackrf_open_device_already_opened(self):
        """Test HackRF opening when device is already opened."""
        interface = HackRFInterface()

        mock_device = MockHackRF()
        mock_device.device_opened = True  # Simulate already opened

        with patch("src.backend.hal.hackrf_interface.hackrf") as mock_hackrf_module:
            mock_hackrf_module.HackRF.return_value = mock_device
            mock_device.get_serial_no = Mock(return_value="MOCK123456")

            result = await interface.open()

            # Should succeed without trying to open again
            assert result is True

    @pytest.mark.asyncio
    async def test_hackrf_open_serial_number_exception(self):
        """Test HackRF opening when serial number retrieval fails."""
        interface = HackRFInterface()

        mock_device = MockHackRF()

        with patch("src.backend.hal.hackrf_interface.hackrf") as mock_hackrf_module:
            mock_hackrf_module.HackRF.return_value = mock_device
            mock_device.get_serial_no = Mock(side_effect=Exception("Serial unavailable"))

            result = await interface.open()

            # Should still succeed
            assert result is True

    @pytest.mark.asyncio
    async def test_hackrf_open_with_amp_enable_config(self):
        """Test HackRF opening with amp enable configuration."""
        config = HackRFConfig(amp_enable=True)
        interface = HackRFInterface(config)

        mock_device = MockHackRF()

        with patch("src.backend.hal.hackrf_interface.hackrf") as mock_hackrf_module:
            mock_hackrf_module.HackRF.return_value = mock_device
            mock_device.get_serial_no = Mock(return_value="MOCK123456")

            result = await interface.open()

            # Should succeed and enable amp
            assert result is True
            assert interface.config.amp_enable is True

    @pytest.mark.asyncio
    async def test_hackrf_open_failure_error_code(self):
        """Test HackRF opening failure with error code."""
        interface = HackRFInterface()

        mock_device = MockHackRF()
        mock_device.device_opened = False

        with patch("src.backend.hal.hackrf_interface.hackrf") as mock_hackrf_module:
            mock_hackrf_module.HackRF.return_value = mock_device
            mock_device.open = Mock(return_value=-1)  # Error code

            result = await interface.open()

            # Should fail
            assert result is False

    @pytest.mark.asyncio
    async def test_hackrf_frequency_set_success_path(self):
        """Test successful frequency setting (coverage for success path)."""
        interface = HackRFInterface()
        mock_device = MockHackRF()
        interface.device = mock_device

        # Test valid frequency
        result = await interface.set_freq(3.2e9)

        assert result is True
        assert interface.config.frequency == 3.2e9

    @pytest.mark.asyncio
    async def test_hackrf_sample_rate_success_path(self):
        """Test successful sample rate setting (coverage for success path)."""
        interface = HackRFInterface()
        mock_device = MockHackRF()
        interface.device = mock_device

        # Test valid sample rate
        result = await interface.set_sample_rate(20e6)

        assert result is True
        assert interface.config.sample_rate == 20e6

    @pytest.mark.asyncio
    async def test_hackrf_gains_success_path(self):
        """Test successful gain setting (coverage for success paths)."""
        interface = HackRFInterface()
        mock_device = MockHackRF()
        interface.device = mock_device

        # Test LNA gain (should round to 8dB steps)
        result = await interface.set_lna_gain(18)  # Should round to 16
        assert result is True
        assert interface.config.lna_gain == 16

        # Test VGA gain (should round to 2dB steps)
        result = await interface.set_vga_gain(21)  # Should round to 20
        assert result is True
        assert interface.config.vga_gain == 20

    @pytest.mark.asyncio
    async def test_hackrf_amp_enable_success_path(self):
        """Test successful amp enable/disable (coverage for success path)."""
        interface = HackRFInterface()
        mock_device = MockHackRF()
        interface.device = mock_device

        # Test amp enable
        result = await interface.set_amp_enable(True)
        assert result is True
        assert interface.config.amp_enable is True

        # Test amp disable
        result = await interface.set_amp_enable(False)
        assert result is True
        assert interface.config.amp_enable is False

    @pytest.mark.asyncio
    async def test_hackrf_get_info_with_mock_device(self):
        """Test get_info with mock device (coverage for mock device path)."""
        interface = HackRFInterface()
        mock_device = MockHackRF()
        mock_device.connected = True
        mock_device.get_serial_no = Mock(return_value="MOCK123456")
        interface.device = mock_device

        info = await interface.get_info()

        assert info["status"] == "Connected"
        assert info["serial_number"] == "MOCK123456"
        assert "frequency" in info
        assert "sample_rate" in info

    @pytest.mark.asyncio
    async def test_hackrf_get_info_serial_exception(self):
        """Test get_info when serial number retrieval fails."""
        interface = HackRFInterface()
        mock_device = MockHackRF()
        mock_device.connected = True
        mock_device.get_serial_no = Mock(side_effect=Exception("Serial error"))
        interface.device = mock_device

        info = await interface.get_info()

        assert info["status"] == "Connected"
        assert info["serial_number"] == "Unknown"

    @pytest.mark.asyncio
    async def test_hackrf_close_with_sdr_error(self):
        """Test close operation with SDR error."""
        interface = HackRFInterface()
        mock_device = MockHackRF()
        mock_device.close = Mock(side_effect=SDRError("Close error"))
        interface.device = mock_device

        # Should handle error gracefully
        await interface.close()

        # Device should still be cleared
        assert interface.device is None

    @pytest.mark.asyncio
    async def test_auto_detect_hackrf_not_available(self):
        """Test auto detection when hackrf module not available."""
        # Mock HACKRF_AVAILABLE as False
        with patch("src.backend.hal.hackrf_interface.HACKRF_AVAILABLE", False):
            result = await auto_detect_hackrf()

            assert result is None

    @pytest.mark.asyncio
    async def test_auto_detect_hackrf_open_failure(self):
        """Test auto detection when device open fails."""
        with patch("src.backend.hal.hackrf_interface.HACKRF_AVAILABLE", True):
            with patch("src.backend.hal.hackrf_interface.HackRFInterface") as mock_interface_class:
                mock_interface = Mock()
                mock_interface.open = AsyncMock(return_value=False)
                mock_interface_class.return_value = mock_interface

                result = await auto_detect_hackrf()

                assert result is None

    @pytest.mark.asyncio
    async def test_auto_detect_hackrf_success(self):
        """Test successful auto detection."""
        with patch("src.backend.hal.hackrf_interface.HACKRF_AVAILABLE", True):
            with patch("src.backend.hal.hackrf_interface.HackRFInterface") as mock_interface_class:
                mock_interface = Mock()
                mock_interface.open = AsyncMock(return_value=True)
                mock_interface.get_info = AsyncMock(return_value={"status": "Connected"})
                mock_interface_class.return_value = mock_interface

                result = await auto_detect_hackrf()

                assert result is not None
                assert result == mock_interface
