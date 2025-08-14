"""
Additional unit tests to boost Phase 1 code coverage to 90%.
Focuses on uncovered paths in SDR, MAVLink, and State Machine services.
"""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.backend.models.schemas import SDRConfig
from src.backend.services.mavlink_service import (
    MAVLinkService,
)
from src.backend.services.sdr_service import (
    SDRConfigError,
    SDRService,
)
from src.backend.services.state_machine import SearchSubstate, StateMachine, SystemState


class TestSDRServiceCoverageBoost:
    """Additional tests for SDR service uncovered paths."""

    @pytest.mark.asyncio
    async def test_sdr_update_config_with_device(self):
        """Test updating configuration with active device."""
        with patch("src.backend.services.sdr_service.SOAPY_AVAILABLE", True):
            with patch("src.backend.services.sdr_service.SoapySDR") as mock_soapy:
                # Setup mock device
                mock_device = MagicMock()
                mock_device.getFrequencyRange.return_value = [
                    MagicMock(minimum=lambda: 100e6, maximum=lambda: 6000e6)
                ]
                mock_device.listSampleRates.return_value = [2e6, 4e6, 8e6]
                mock_soapy.Device.return_value = mock_device
                mock_soapy.Device.enumerate.return_value = [{"driver": "hackrf"}]
                mock_soapy.SOAPY_SDR_RX = 0

                # Initialize service
                sdr_service = SDRService()
                await sdr_service.initialize()

                # Update config
                new_config = SDRConfig(frequency=500e6, sampleRate=4e6)
                await sdr_service.update_config(new_config)

                assert sdr_service.config.frequency == 500e6
                assert sdr_service.config.sampleRate == 4e6

    @pytest.mark.asyncio
    async def test_sdr_update_config_rollback_on_error(self):
        """Test configuration rollback on update failure."""
        with patch("src.backend.services.sdr_service.SOAPY_AVAILABLE", True):
            with patch("src.backend.services.sdr_service.SoapySDR") as mock_soapy:
                # Setup mock device that will fail
                mock_device = MagicMock()
                mock_device.getFrequencyRange.side_effect = Exception("Config error")
                mock_soapy.Device.return_value = mock_device
                mock_soapy.Device.enumerate.return_value = [{"driver": "hackrf"}]

                # Initialize service
                sdr_service = SDRService()
                await sdr_service.initialize()
                old_config = sdr_service.config

                # Try to update config (should fail and rollback)
                new_config = SDRConfig(frequency=500e6)
                with pytest.raises(Exception):
                    await sdr_service.update_config(new_config)

                # Config should be rolled back
                assert sdr_service.config == old_config

    @pytest.mark.asyncio
    async def test_sdr_set_frequency_no_device(self):
        """Test set_frequency without initialized device."""
        sdr_service = SDRService()

        with pytest.raises(SDRConfigError, match="Device not initialized"):
            sdr_service.set_frequency(433e6)

    @pytest.mark.asyncio
    async def test_sdr_get_status_with_temperature_sensor(self):
        """Test status retrieval with temperature sensor."""
        with patch("src.backend.services.sdr_service.SOAPY_AVAILABLE", True):
            with patch("src.backend.services.sdr_service.SoapySDR") as mock_soapy:
                # Setup mock device with temperature sensor
                mock_device = MagicMock()
                mock_device.listSensors.return_value = ["temperature", "other"]
                mock_device.readSensor.return_value = "55.5"
                mock_soapy.Device.return_value = mock_device
                mock_soapy.Device.enumerate.return_value = [{"driver": "hackrf"}]
                mock_soapy.SOAPY_SDR_RX = 0

                # Initialize and get status
                sdr_service = SDRService()
                await sdr_service.initialize()
                status = sdr_service.get_status()

                assert status.temperature == 55.5

    @pytest.mark.asyncio
    async def test_sdr_stream_timeout_handling(self):
        """Test IQ stream timeout handling."""
        with patch("src.backend.services.sdr_service.SOAPY_AVAILABLE", True):
            with patch("src.backend.services.sdr_service.SoapySDR") as mock_soapy:
                # Setup mock device
                mock_device = MagicMock()
                mock_device.readStream.return_value = (mock_soapy.SOAPY_SDR_TIMEOUT, 0, 0)
                mock_soapy.Device.return_value = mock_device
                mock_soapy.Device.enumerate.return_value = [{"driver": "hackrf"}]
                mock_soapy.SOAPY_SDR_RX = 0
                mock_soapy.SOAPY_SDR_CF32 = 1
                mock_soapy.SOAPY_SDR_TIMEOUT = -1

                # Initialize service
                sdr_service = SDRService()
                await sdr_service.initialize()

                # Stream with timeout
                samples_received = 0
                async for samples in sdr_service.stream_iq():
                    samples_received += 1
                    if samples_received > 2:
                        sdr_service._stream_active = False
                        break

                # Should handle timeout gracefully
                assert samples_received > 0


class TestMAVLinkServiceCoverageBoost:
    """Additional tests for MAVLink service uncovered paths."""

    @pytest.mark.asyncio
    async def test_mavlink_process_sys_status_no_battery(self):
        """Test processing SYS_STATUS with no battery info."""
        mavlink_service = MAVLinkService()

        # Create mock message with -1 values (no battery)
        mock_msg = MagicMock()
        mock_msg.get_type.return_value = "SYS_STATUS"
        mock_msg.voltage_battery = -1
        mock_msg.current_battery = -1
        mock_msg.battery_remaining = -1

        # Process message
        mavlink_service._process_sys_status(mock_msg)

        # Battery values should remain at defaults
        assert mavlink_service.telemetry["battery"]["voltage"] == 0.0
        assert mavlink_service.telemetry["battery"]["current"] == 0.0
        assert mavlink_service.telemetry["battery"]["percentage"] == 0.0

    @pytest.mark.asyncio
    async def test_mavlink_send_health_status_no_temperature(self):
        """Test health status without temperature file."""
        mavlink_service = MAVLinkService()
        mavlink_service.connection = MagicMock()

        with patch("builtins.open", side_effect=FileNotFoundError):
            with patch("psutil.cpu_percent", return_value=50.0):
                with patch("psutil.virtual_memory") as mock_mem:
                    mock_mem.return_value.percent = 60.0

                    await mavlink_service._send_health_status()

                    # Should send status without temperature
                    mavlink_service.connection.mav.statustext_send.assert_called()

    @pytest.mark.asyncio
    async def test_mavlink_upload_mission_timeout(self):
        """Test mission upload with timeout."""
        mavlink_service = MAVLinkService()
        mavlink_service.connection = MagicMock()
        mavlink_service.connection.recv_match.return_value = None  # Timeout

        waypoints = [{"lat": 47.397, "lon": 8.502, "alt": 50.0}]

        result = await mavlink_service.upload_mission(waypoints)
        assert result is False

    @pytest.mark.asyncio
    async def test_mavlink_start_mission_no_auto_mode(self):
        """Test starting mission when AUTO mode not available."""
        mavlink_service = MAVLinkService()
        mavlink_service.connection = MagicMock()
        mavlink_service.connection.mode_mapping.return_value = {}  # No AUTO mode

        result = await mavlink_service.start_mission()
        assert result is False

    @pytest.mark.asyncio
    async def test_mavlink_stop_mission_fallback_to_guided(self):
        """Test mission stop with fallback to GUIDED mode."""
        mavlink_service = MAVLinkService()
        mavlink_service.connection = MagicMock()
        mavlink_service.connection.mode_mapping.return_value = {
            "GUIDED": 4  # Only GUIDED available
        }

        result = await mavlink_service.stop_mission()
        assert result is True

    @pytest.mark.asyncio
    async def test_mavlink_get_mission_progress_no_connection(self):
        """Test getting mission progress without connection."""
        mavlink_service = MAVLinkService()
        mavlink_service.connection = None

        current, total = mavlink_service.get_mission_progress()
        assert current == 0
        assert total == 0

    @pytest.mark.asyncio
    async def test_mavlink_telemetry_sender_error_handling(self):
        """Test telemetry sender error recovery."""
        mavlink_service = MAVLinkService()
        mavlink_service._running = True
        mavlink_service.connection = MagicMock()

        # Mock send_named_value_float to fail once then succeed
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Send failed")
            return None

        mavlink_service.send_named_value_float = MagicMock(side_effect=side_effect)

        # Run telemetry sender briefly
        task = asyncio.create_task(mavlink_service.telemetry_sender())
        await asyncio.sleep(0.15)
        mavlink_service._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should have attempted to send despite error
        assert call_count >= 1


class TestStateMachineCoverageBoost:
    """Additional tests for State Machine uncovered paths."""

    @pytest.mark.asyncio
    async def test_state_machine_telemetry_update(self):
        """Test telemetry metrics update via MAVLink."""
        state_machine = StateMachine(enable_persistence=False)

        # Mock MAVLink service
        mock_mavlink = MagicMock()
        mock_mavlink.send_telemetry = MagicMock()
        state_machine.set_mavlink_service(mock_mavlink)

        # Send telemetry update
        await state_machine.send_telemetry_update()

        # Should send metrics via MAVLink
        assert mock_mavlink.send_telemetry.called

    @pytest.mark.asyncio
    async def test_state_machine_search_pattern_management(self):
        """Test search pattern status and management."""
        state_machine = StateMachine(enable_persistence=False)

        # Set search pattern
        state_machine._search_substate = SearchSubstate.EXECUTING
        state_machine._active_pattern = MagicMock()
        state_machine._active_pattern.type = "spiral"
        state_machine._active_pattern.waypoints = [1, 2, 3, 4, 5]
        state_machine._current_waypoint_index = 2

        # Get search pattern status
        status = state_machine.get_search_pattern_status()

        assert status["active"] is True
        assert status["pattern_type"] == "spiral"
        assert status["waypoint_index"] == 2
        assert status["total_waypoints"] == 5
        assert status["progress_percentage"] == 40.0

    @pytest.mark.asyncio
    async def test_state_machine_timeout_handler(self):
        """Test state timeout handler execution."""
        state_machine = StateMachine(enable_persistence=False)
        state_machine._state_timeouts[SystemState.DETECTING] = 0.1  # Short timeout

        # Transition to DETECTING (has timeout)
        await state_machine.transition_to(SystemState.DETECTING)

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Should have transitioned back to IDLE or SEARCHING
        assert state_machine.get_current_state() in [SystemState.IDLE, SystemState.SEARCHING]

    @pytest.mark.asyncio
    async def test_state_machine_force_transition(self):
        """Test force transition bypassing validation."""
        state_machine = StateMachine(enable_persistence=False)

        # Force transition from IDLE directly to HOLDING (normally invalid)
        result = await state_machine.force_transition(
            SystemState.HOLDING, reason="Emergency override"
        )

        assert result is True
        assert state_machine.get_current_state() == SystemState.HOLDING

    @pytest.mark.asyncio
    async def test_state_machine_signal_lost_handling(self):
        """Test signal lost during HOMING state."""
        state_machine = StateMachine(enable_persistence=False)

        # Set up in HOMING state
        await state_machine.force_transition(SystemState.HOMING)

        # Handle signal lost
        await state_machine.handle_signal_lost()

        # Should transition back to SEARCHING
        assert state_machine.get_current_state() == SystemState.SEARCHING

    @pytest.mark.asyncio
    async def test_state_machine_mission_complete_handling(self):
        """Test mission complete transitions to IDLE."""
        state_machine = StateMachine(enable_persistence=False)

        # Set up in HOLDING state
        await state_machine.force_transition(SystemState.HOLDING)

        # Handle mission complete
        await state_machine.handle_mission_complete()

        # Should transition to IDLE
        assert state_machine.get_current_state() == SystemState.IDLE

    @pytest.mark.asyncio
    async def test_state_machine_pause_search_pattern(self):
        """Test pausing search pattern execution."""
        state_machine = StateMachine(enable_persistence=False)

        # Set up search pattern
        state_machine._search_substate = SearchSubstate.EXECUTING
        state_machine._active_pattern = MagicMock()

        # Pause search
        await state_machine.pause_search_pattern()

        assert state_machine._search_substate == SearchSubstate.PAUSED
        assert state_machine._pattern_paused_at is not None

    @pytest.mark.asyncio
    async def test_state_machine_resume_search_pattern(self):
        """Test resuming paused search pattern."""
        state_machine = StateMachine(enable_persistence=False)

        # Set up paused search
        state_machine._search_substate = SearchSubstate.PAUSED
        state_machine._pattern_paused_at = datetime.now(UTC)
        state_machine._active_pattern = MagicMock()

        # Resume search
        await state_machine.resume_search_pattern()

        assert state_machine._search_substate == SearchSubstate.EXECUTING
        assert state_machine._pattern_paused_at is None


class TestHealthEndpointsCoverageBoost:
    """Additional tests for health check endpoints."""

    @pytest.mark.asyncio
    async def test_health_endpoint_high_cpu_degradation(self):
        """Test health degradation with high CPU usage."""
        from src.backend.api.routes.health import health_check

        with patch("src.backend.api.routes.health.get_service_manager") as mock_manager:
            mock_service_manager = AsyncMock()
            mock_service_manager.get_service_health.return_value = {
                "status": "healthy",
                "initialized": True,
                "startup_time": 5.0,
                "services": {},
            }
            mock_manager.return_value = mock_service_manager

            with patch("psutil.cpu_percent", return_value=95.0):  # High CPU
                with patch("psutil.virtual_memory") as mock_mem:
                    mock_mem.return_value.percent = 50.0
                    with patch("psutil.disk_usage") as mock_disk:
                        mock_disk.return_value.percent = 40.0

                        result = await health_check()

                        # Should be degraded due to high CPU
                        assert result["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_health_endpoint_high_temperature_degradation(self):
        """Test health degradation with high temperature."""
        from src.backend.api.routes.health import health_check

        with patch("src.backend.api.routes.health.get_service_manager") as mock_manager:
            mock_service_manager = AsyncMock()
            mock_service_manager.get_service_health.return_value = {
                "status": "healthy",
                "initialized": True,
                "startup_time": 5.0,
                "services": {},
            }
            mock_manager.return_value = mock_service_manager

            with patch("psutil.cpu_percent", return_value=50.0):
                with patch("psutil.virtual_memory") as mock_mem:
                    mock_mem.return_value.percent = 50.0
                    with patch("psutil.disk_usage") as mock_disk:
                        mock_disk.return_value.percent = 40.0
                        with patch("builtins.open", mock_open(read_data="85000")):  # 85°C

                            result = await health_check()

                            # Should be degraded due to high temperature
                            assert result["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_signal_processor_health_unhealthy(self):
        """Test signal processor health when not processing."""
        from src.backend.api.routes.health import signal_processor_health_check

        mock_processor = MagicMock()
        mock_processor.is_processing = False
        mock_processor.current_rssi = -80.0
        mock_processor.noise_floor = -90.0
        mock_processor.signal_detected = False
        mock_processor.detection_confidence = 0.0
        mock_processor.detection_threshold = 12.0
        mock_processor.noise_floor_percentile = 10
        mock_processor.debounce_samples = 5
        mock_processor.get_metrics.return_value = {"processing_errors": 0}

        result = await signal_processor_health_check(signal_processor=mock_processor)

        assert result["health"] == "unhealthy"
        assert result["is_processing"] is False


def mock_open(read_data=""):
    """Helper to create mock file open."""
    from unittest.mock import mock_open as base_mock_open

    return base_mock_open(read_data=read_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
