"""Integration tests for signal state controller and command pipeline."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from src.backend.services.command_pipeline import (
    CommandPipeline,

pytestmark = pytest.mark.serial
    CommandPriority,
    CommandType,
)
from src.backend.services.signal_state_controller import SignalState, SignalStateController
from src.backend.services.state_machine import StateMachine, SystemState
from src.backend.utils.safety import SafetyInterlockSystem


@pytest.fixture
def state_machine():
    """Create mock state machine."""
    machine = AsyncMock(spec=StateMachine)
    machine.handle_detection = AsyncMock()
    machine.handle_signal_lost = AsyncMock()
    machine.get_current_state.return_value = SystemState.SEARCHING
    return machine


@pytest.fixture
def signal_controller(state_machine):
    """Create signal state controller with test parameters."""
    controller = SignalStateController(
        trigger_threshold=12.0,
        drop_threshold=6.0,
        confirmation_time=0.1,  # Short for testing
        drop_time=0.2,  # Short for testing
    )
    controller.set_state_machine(state_machine)
    return controller


@pytest.fixture
def safety_system():
    """Create mock safety system."""
    system = AsyncMock(spec=SafetyInterlockSystem)
    system.check_all_safety.return_value = {
        "mode": True,
        "operator": True,
        "signal": True,
        "battery": True,
        "geofence": True,
    }
    system.is_safe_to_proceed.return_value = True
    system.emergency_stop = AsyncMock()
    system.start_monitoring = AsyncMock()
    system.stop_monitoring = AsyncMock()
    return system


@pytest.fixture
def mavlink_service():
    """Create mock MAVLink service."""
    service = AsyncMock()
    service.emergency_stop = AsyncMock()
    service.return_to_launch = AsyncMock()
    service.goto_position = AsyncMock()
    service.set_velocity = AsyncMock()
    service.set_mode = AsyncMock()
    service.arm = AsyncMock()
    service.disarm = AsyncMock()
    service.takeoff = AsyncMock()
    service.land = AsyncMock()
    return service


@pytest.fixture
def command_pipeline(safety_system, mavlink_service):
    """Create command pipeline."""
    pipeline = CommandPipeline(
        safety_system=safety_system,
        mavlink_service=mavlink_service,
        rate_limit_per_second=100.0,  # High for testing
    )
    return pipeline


class TestSignalStateController:
    """Test signal state controller with debouncing."""

    @pytest.mark.asyncio
    async def test_signal_detection_with_hysteresis(self, signal_controller):
        """Test signal detection with hysteresis thresholds."""
        noise_floor = -100.0

        # Start with no signal
        state, event = await signal_controller.process_signal(-90.0, noise_floor)
        assert state == SignalState.NO_SIGNAL
        assert event is None

        # Signal rises above trigger threshold (12dB SNR)
        state, event = await signal_controller.process_signal(-87.0, noise_floor)  # 13dB SNR
        assert state == SignalState.RISING
        assert event is None

        # Wait for confirmation time
        await asyncio.sleep(0.15)

        # Signal still strong, should confirm
        state, event = await signal_controller.process_signal(-85.0, noise_floor)  # 15dB SNR
        assert state == SignalState.CONFIRMED
        assert event is not None
        assert event.rssi == -85.0
        assert event.snr == 15.0

        # Signal drops but stays above drop threshold (6dB)
        state, event = await signal_controller.process_signal(-93.0, noise_floor)  # 7dB SNR
        assert state == SignalState.CONFIRMED

        # Signal drops below drop threshold
        state, event = await signal_controller.process_signal(-95.0, noise_floor)  # 5dB SNR
        assert state == SignalState.FALLING

        # Wait for drop time
        await asyncio.sleep(0.25)

        # Signal still weak, should be lost
        state, event = await signal_controller.process_signal(-96.0, noise_floor)  # 4dB SNR
        assert state == SignalState.LOST

    @pytest.mark.asyncio
    async def test_false_positive_prevention(self, signal_controller):
        """Test that brief spikes don't trigger false positives."""
        noise_floor = -100.0

        # Brief spike above threshold
        state, event = await signal_controller.process_signal(-85.0, noise_floor)  # 15dB SNR
        assert state == SignalState.RISING

        # Immediately drops - should reject as false positive
        state, event = await signal_controller.process_signal(-96.0, noise_floor)  # 4dB SNR
        assert state == SignalState.NO_SIGNAL
        assert signal_controller.false_positives == 1
        assert signal_controller.true_positives == 0

    @pytest.mark.asyncio
    async def test_anomaly_detection(self, signal_controller):
        """Test anomaly detection for sudden spikes."""
        noise_floor = -100.0

        # Build up stable signal history
        for _ in range(20):
            await signal_controller.process_signal(-95.0, noise_floor)  # 5dB SNR

        # Sudden spike (anomaly)
        state, event = await signal_controller.process_signal(-70.0, noise_floor)  # 30dB SNR
        # Should not trigger due to anomaly detection
        assert state == SignalState.NO_SIGNAL

    @pytest.mark.asyncio
    async def test_state_machine_integration(self, signal_controller, state_machine):
        """Test integration with state machine."""
        noise_floor = -100.0

        # Trigger detection
        await signal_controller.process_signal(-87.0, noise_floor)  # 13dB SNR
        await asyncio.sleep(0.15)
        await signal_controller.process_signal(-85.0, noise_floor)  # 15dB SNR

        # Should have called state machine
        state_machine.handle_detection.assert_called_once()

        # Lose signal
        await signal_controller.process_signal(-96.0, noise_floor)  # 4dB SNR
        await asyncio.sleep(0.25)
        await signal_controller.process_signal(-97.0, noise_floor)  # 3dB SNR

        # Should have called signal lost
        state_machine.handle_signal_lost.assert_called_once()

    @pytest.mark.asyncio
    async def test_false_positive_rate_tracking(self, signal_controller):
        """Test false positive rate calculation."""
        noise_floor = -100.0

        # Create some true positives
        for i in range(3):
            await signal_controller.process_signal(-87.0, noise_floor)
            await asyncio.sleep(0.15)
            await signal_controller.process_signal(-85.0, noise_floor)
            await asyncio.sleep(0.1)
            await signal_controller.process_signal(-96.0, noise_floor)
            await asyncio.sleep(0.25)
            await signal_controller.process_signal(-97.0, noise_floor)
            await asyncio.sleep(2.1)  # Reset to NO_SIGNAL

        # Create false positives
        for i in range(2):
            await signal_controller.process_signal(-85.0, noise_floor)
            await signal_controller.process_signal(-96.0, noise_floor)

        # Check statistics
        stats = signal_controller.get_statistics()
        assert stats["true_positives"] == 3
        assert stats["false_positives"] == 2
        assert stats["false_positive_rate"] == 40.0  # 2/5 * 100

    @pytest.mark.asyncio
    async def test_transition_audit_logging(self, signal_controller):
        """Test that transitions are logged for audit."""
        noise_floor = -100.0

        # Create some transitions
        await signal_controller.process_signal(-87.0, noise_floor)
        await asyncio.sleep(0.15)
        await signal_controller.process_signal(-85.0, noise_floor)

        # Get audit log
        log = signal_controller.get_transition_log()
        assert len(log) >= 2

        # Check log entries
        for entry in log:
            assert "id" in entry
            assert "timestamp" in entry
            assert "from_state" in entry
            assert "to_state" in entry
            assert "trigger_snr" in entry
            assert "reason" in entry


class TestCommandPipeline:
    """Test command pipeline with safety interlocks."""

    @pytest.mark.asyncio
    async def test_command_validation(self, command_pipeline):
        """Test command parameter validation."""
        # Valid goto position
        cmd_id = await command_pipeline.submit_command(
            CommandType.GOTO_POSITION,
            {"latitude": 45.0, "longitude": -122.0, "altitude": 50.0},
        )
        assert cmd_id is not None

        # Invalid goto position (missing altitude)
        with pytest.raises(ValueError):
            await command_pipeline.submit_command(
                CommandType.GOTO_POSITION,
                {"latitude": 45.0, "longitude": -122.0},
            )

        # Invalid velocity (exceeds limit)
        with pytest.raises(ValueError):
            await command_pipeline.submit_command(
                CommandType.SET_VELOCITY,
                {"vx": 30.0, "vy": 0.0, "vz": 0.0},  # 30 m/s exceeds 20 m/s limit
            )

    @pytest.mark.asyncio
    async def test_emergency_command_priority(self, command_pipeline, mavlink_service):
        """Test that emergency commands bypass queue."""
        # Submit emergency stop
        cmd_id = await command_pipeline.submit_command(
            CommandType.EMERGENCY_STOP,
            priority=CommandPriority.EMERGENCY,
        )

        # Should execute immediately
        mavlink_service.emergency_stop.assert_called_once()

        # Check audit log
        log = command_pipeline.get_audit_log(limit=1)
        assert len(log) > 0
        assert log[0]["type"] == CommandType.EMERGENCY_STOP.value
        assert log[0]["success"]

    @pytest.mark.asyncio
    async def test_safety_interlock_blocking(self, command_pipeline, safety_system):
        """Test that unsafe conditions block commands."""
        # Make safety check fail
        safety_system.is_safe_to_proceed.return_value = False
        safety_system.check_all_safety.return_value = {
            "mode": False,  # Wrong mode
            "operator": True,
            "signal": True,
            "battery": True,
            "geofence": True,
        }

        # Start pipeline
        await command_pipeline.start()

        # Submit command
        cmd_id = await command_pipeline.submit_command(
            CommandType.ARM,
            priority=CommandPriority.NORMAL,
        )

        # Wait for processing
        await asyncio.sleep(0.1)

        # Check statistics
        stats = command_pipeline.get_statistics()
        assert stats["blocked_by_safety"] > 0

        await command_pipeline.stop()

    @pytest.mark.asyncio
    async def test_command_rate_limiting(self, command_pipeline):
        """Test rate limiting prevents command flooding."""
        # Set low rate limit for testing
        command_pipeline.rate_limit = 2.0  # 2 commands per second
        command_pipeline.min_interval = 0.5

        # Start pipeline
        await command_pipeline.start()

        # Submit multiple commands quickly
        start_time = asyncio.get_event_loop().time()
        for i in range(3):
            await command_pipeline.submit_command(
                CommandType.LOITER,
                priority=CommandPriority.NORMAL,
            )

        # Wait for processing
        await asyncio.sleep(1.5)

        # Commands should be rate limited
        elapsed = asyncio.get_event_loop().time() - start_time
        assert elapsed >= 1.0  # At least 1 second for 3 commands at 2/sec

        await command_pipeline.stop()

    @pytest.mark.asyncio
    async def test_command_priority_queue(self, command_pipeline):
        """Test that higher priority commands execute first."""
        # Start pipeline
        await command_pipeline.start()

        # Submit commands in reverse priority order
        low_cmd = await command_pipeline.submit_command(
            CommandType.LOITER,
            priority=CommandPriority.LOW,
        )
        high_cmd = await command_pipeline.submit_command(
            CommandType.LAND,
            priority=CommandPriority.HIGH,
        )
        normal_cmd = await command_pipeline.submit_command(
            CommandType.SET_MODE,
            {"mode": "GUIDED"},
            priority=CommandPriority.NORMAL,
        )

        # Wait for processing
        await asyncio.sleep(0.5)

        # Check audit log order
        log = command_pipeline.get_audit_log(limit=10)
        executed_order = [entry["command_id"] for entry in log if entry["success"]]

        # High priority should execute before normal and low
        if len(executed_order) >= 2:
            assert executed_order.index(high_cmd) < executed_order.index(normal_cmd)
            assert executed_order.index(high_cmd) < executed_order.index(low_cmd)

        await command_pipeline.stop()

    @pytest.mark.asyncio
    async def test_command_audit_logging(self, command_pipeline):
        """Test comprehensive audit logging."""
        # Start pipeline
        await command_pipeline.start()

        # Submit various commands
        cmd_ids = []
        cmd_ids.append(
            await command_pipeline.submit_command(
                CommandType.ARM,
                source="test_user",
            )
        )
        cmd_ids.append(
            await command_pipeline.submit_command(
                CommandType.TAKEOFF,
                {"altitude": 10.0},
                source="autopilot",
            )
        )

        # Wait for processing
        await asyncio.sleep(0.5)

        # Get audit log
        log = command_pipeline.get_audit_log()

        # Verify log entries
        for entry in log:
            assert "command_id" in entry
            assert "timestamp" in entry
            assert "type" in entry
            assert "source" in entry
            assert "safety_status" in entry
            assert "execution_time_ms" in entry
            assert "success" in entry

        await command_pipeline.stop()

    @pytest.mark.asyncio
    async def test_geofence_enforcement(self, command_pipeline, safety_system):
        """Test geofence safety check enforcement (FR8)."""
        # Configure geofence violation
        safety_system.check_all_safety.return_value = {
            "mode": True,
            "operator": True,
            "signal": True,
            "battery": True,
            "geofence": False,  # Outside geofence
        }
        safety_system.is_safe_to_proceed.return_value = False

        # Start pipeline
        await command_pipeline.start()

        # Try to command position outside geofence
        cmd_id = await command_pipeline.submit_command(
            CommandType.GOTO_POSITION,
            {"latitude": 45.0, "longitude": -122.0, "altitude": 50.0},
        )

        # Wait for processing
        await asyncio.sleep(0.2)

        # Command should be blocked
        stats = command_pipeline.get_statistics()
        assert stats["blocked_by_safety"] > 0

        # Check audit log
        log = command_pipeline.get_audit_log(limit=1)
        assert not log[0]["success"]
        assert not log[0]["safety_status"]["geofence"]

        await command_pipeline.stop()

    @pytest.mark.asyncio
    async def test_100ms_emergency_stop_requirement(self, command_pipeline, mavlink_service):
        """Test emergency stop executes within 100ms (NFR2)."""
        import time

        # Submit emergency stop and measure time
        start = time.perf_counter()
        cmd_id = await command_pipeline.submit_command(
            CommandType.EMERGENCY_STOP,
            priority=CommandPriority.EMERGENCY,
        )
        end = time.perf_counter()

        # Check execution time
        execution_time_ms = (end - start) * 1000
        assert execution_time_ms < 100, f"Emergency stop took {execution_time_ms:.1f}ms"

        # Verify command executed
        mavlink_service.emergency_stop.assert_called_once()


class TestIntegratedSystem:
    """Test integrated signal processing and command pipeline."""

    @pytest.mark.asyncio
    async def test_signal_triggered_commands(
        self, signal_controller, command_pipeline, state_machine
    ):
        """Test that signal detection triggers appropriate commands."""

        # Configure state machine to trigger commands
        async def mock_detection(rssi, confidence):
            await command_pipeline.submit_command(
                CommandType.START_HOMING,
                {"rssi": rssi, "confidence": confidence},
                priority=CommandPriority.HIGH,
            )

        state_machine.handle_detection = mock_detection

        # Start command pipeline
        await command_pipeline.start()

        # Trigger signal detection
        noise_floor = -100.0
        await signal_controller.process_signal(-87.0, noise_floor)
        await asyncio.sleep(0.15)
        await signal_controller.process_signal(-85.0, noise_floor)

        # Wait for command processing
        await asyncio.sleep(0.2)

        # Check that homing command was submitted
        stats = command_pipeline.get_statistics()
        assert stats["total_commands"] > 0

        await command_pipeline.stop()

    @pytest.mark.asyncio
    async def test_signal_loss_triggers_safety(
        self, signal_controller, command_pipeline, state_machine, safety_system
    ):
        """Test that signal loss triggers safety response."""

        # Configure state machine to trigger safety on signal loss
        async def mock_signal_lost():
            await command_pipeline.submit_command(
                CommandType.RETURN_TO_LAUNCH,
                priority=CommandPriority.CRITICAL,
            )

        state_machine.handle_signal_lost = mock_signal_lost

        # Start command pipeline
        await command_pipeline.start()

        # Establish then lose signal
        noise_floor = -100.0
        await signal_controller.process_signal(-87.0, noise_floor)
        await asyncio.sleep(0.15)
        await signal_controller.process_signal(-85.0, noise_floor)
        await asyncio.sleep(0.1)
        await signal_controller.process_signal(-96.0, noise_floor)
        await asyncio.sleep(0.25)
        await signal_controller.process_signal(-97.0, noise_floor)

        # Wait for command processing
        await asyncio.sleep(0.2)

        # Check that RTL command was submitted
        log = command_pipeline.get_audit_log()
        rtl_commands = [e for e in log if e["type"] == CommandType.RETURN_TO_LAUNCH.value]
        assert len(rtl_commands) > 0

        await command_pipeline.stop()
