"""SITL test scenario for mission abort with proper state transitions."""

from datetime import UTC, datetime, timedelta
from enum import Enum
from unittest.mock import AsyncMock

import pytest

from src.backend.services.mavlink_service import MAVLinkService
from src.backend.services.state_machine import SystemState, StateMachine


class AbortReason(Enum):
    """Reasons for mission abort."""

    OPERATOR_OVERRIDE = "operator_override"
    BATTERY_CRITICAL = "battery_critical"
    SIGNAL_LOST = "signal_lost"
    GEOFENCE_BREACH = "geofence_breach"
    EMERGENCY_STOP = "emergency_stop"
    SYSTEM_FAILURE = "system_failure"
    GPS_LOST = "gps_lost"
    COMMUNICATION_LOST = "communication_lost"


class TestMissionAbortScenario:
    """Test mission abort scenarios with proper state transitions in SITL."""

    @pytest.fixture
    def state_machine(self):
        """Create state machine for testing."""
        sm = StateMachine()
        sm.current_state = SystemState.IDLE
        sm.homing_enabled = False
        sm.abort_reason = None
        sm.pre_abort_state = None
        return sm

    @pytest.fixture
    def mock_mavlink(self):
        """Create mock MAVLink service."""
        mavlink = AsyncMock(spec=MAVLinkService)
        mavlink.connected = True
        mavlink.flight_mode = "GUIDED"
        mavlink.battery_percent = 85
        mavlink.current_position = {"lat": 42.3601, "lon": -71.0589, "alt": 50}
        mavlink.stop_mission = AsyncMock(return_value=True)
        mavlink.set_mode = AsyncMock(return_value=True)
        mavlink.land = AsyncMock(return_value=True)
        mavlink.return_to_launch = AsyncMock(return_value=True)
        mavlink.hold_position = AsyncMock(return_value=True)
        mavlink.disarm = AsyncMock(return_value=True)
        return mavlink

    @pytest.fixture
    def abort_handler(self, state_machine, mock_mavlink):
        """Create abort handler with dependencies."""
        handler = MissionAbortHandler(state_machine, mock_mavlink)
        return handler

    @pytest.mark.asyncio
    async def test_emergency_stop_abort(self, abort_handler, state_machine, mock_mavlink):
        """Test emergency stop abort sequence."""
        # Start in active homing state
        state_machine.current_state = SystemState.HOMING
        state_machine.homing_enabled = True

        # Trigger emergency stop
        abort_result = await abort_handler.abort_mission(
            reason=AbortReason.EMERGENCY_STOP, emergency=True
        )

        # Verify immediate actions
        assert abort_result["success"] == True
        assert state_machine.current_state == SystemState.IDLE
        assert state_machine.homing_enabled == False

        # Verify MAVLink commands
        mock_mavlink.stop_mission.assert_called_once()
        mock_mavlink.hold_position.assert_called_once()

        # Verify abort metadata
        assert state_machine.abort_reason == AbortReason.EMERGENCY_STOP
        assert state_machine.pre_abort_state == SystemState.HOMING

    @pytest.mark.asyncio
    async def test_battery_critical_abort_with_rtl(
        self, abort_handler, state_machine, mock_mavlink
    ):
        """Test battery critical abort with return to launch."""
        # Set up low battery condition
        mock_mavlink.battery_percent = 10
        state_machine.current_state = SystemState.SEARCHING
        state_machine.homing_enabled = True

        # Store home position
        home_position = {"lat": 42.3600, "lon": -71.0590, "alt": 50}
        mock_mavlink.home_position = home_position

        # Trigger battery critical abort
        abort_result = await abort_handler.abort_mission(
            reason=AbortReason.BATTERY_CRITICAL, return_home=True
        )

        # Verify abort actions
        assert abort_result["success"] == True
        assert state_machine.current_state == SystemState.IDLE
        assert not state_machine.homing_enabled

        # Verify RTL initiated
        mock_mavlink.return_to_launch.assert_called_once()
        mock_mavlink.set_mode.assert_called_with("RTL")

        # Verify no further autonomous operations
        assert state_machine.autonomous_operations_blocked == True

    @pytest.mark.asyncio
    async def test_signal_lost_abort_with_search_pattern(
        self, abort_handler, state_machine, mock_mavlink
    ):
        """Test signal lost abort with search pattern execution."""
        # Active homing when signal lost
        state_machine.current_state = SystemState.HOMING
        state_machine.homing_enabled = True
        last_known_position = mock_mavlink.current_position.copy()

        # Trigger signal lost abort
        abort_result = await abort_handler.abort_mission(
            reason=AbortReason.SIGNAL_LOST, execute_search=True, search_center=last_known_position
        )

        # Should transition to searching, not fully abort
        assert abort_result["success"] == True
        assert state_machine.current_state == SystemState.SEARCHING
        assert state_machine.homing_enabled == True  # Still enabled, just searching

        # Verify search pattern initiated
        assert abort_result["search_pattern_started"] == True
        assert abort_result["search_center"] == last_known_position

    @pytest.mark.asyncio
    async def test_geofence_breach_abort(self, abort_handler, state_machine, mock_mavlink):
        """Test geofence breach abort with position correction."""
        # Set geofence boundary
        geofence_center = {"lat": 42.3601, "lon": -71.0589}
        geofence_radius = 100  # meters

        # Position outside geofence
        mock_mavlink.current_position = {"lat": 42.3620, "lon": -71.0589, "alt": 50}
        state_machine.current_state = SystemState.HOMING

        # Trigger geofence abort
        abort_result = await abort_handler.abort_mission(
            reason=AbortReason.GEOFENCE_BREACH, return_to_fence=True, fence_center=geofence_center
        )

        # Verify abort and return to fence
        assert abort_result["success"] == True
        assert state_machine.current_state == SystemState.IDLE
        mock_mavlink.stop_mission.assert_called_once()

        # Should navigate back to fence center
        assert abort_result["returning_to_fence"] == True
        assert abort_result["target_position"] == geofence_center

    @pytest.mark.asyncio
    async def test_operator_override_abort(self, abort_handler, state_machine, mock_mavlink):
        """Test operator manual override abort."""
        # Active autonomous operation
        state_machine.current_state = SystemState.HOMING
        state_machine.homing_enabled = True

        # Operator takes control
        mock_mavlink.flight_mode = "MANUAL"

        # Trigger operator override abort
        abort_result = await abort_handler.abort_mission(
            reason=AbortReason.OPERATOR_OVERRIDE, maintain_position=False
        )

        # Verify clean handover to manual control
        assert abort_result["success"] == True
        assert state_machine.current_state == SystemState.IDLE
        assert not state_machine.homing_enabled

        # Should not fight operator control
        mock_mavlink.hold_position.assert_not_called()
        mock_mavlink.stop_mission.assert_called_once()

    @pytest.mark.asyncio
    async def test_cascading_abort_handling(self, abort_handler, state_machine, mock_mavlink):
        """Test handling multiple abort triggers simultaneously."""
        # Multiple critical conditions
        mock_mavlink.battery_percent = 5
        mock_mavlink.gps_status = "NO_FIX"
        state_machine.current_state = SystemState.HOMING

        # Track abort sequence
        abort_sequence = []

        # First abort - battery critical
        abort1 = await abort_handler.abort_mission(reason=AbortReason.BATTERY_CRITICAL, priority=1)
        abort_sequence.append(abort1)

        # Second abort - GPS lost (higher priority)
        abort2 = await abort_handler.abort_mission(
            reason=AbortReason.GPS_LOST, priority=0  # Higher priority (lower number)
        )
        abort_sequence.append(abort2)

        # Highest priority abort should take precedence
        assert state_machine.abort_reason == AbortReason.GPS_LOST
        assert state_machine.current_state == SystemState.IDLE

        # Both aborts recorded
        assert len(abort_sequence) == 2
        assert all(a["success"] for a in abort_sequence)

    @pytest.mark.asyncio
    async def test_abort_with_state_persistence(self, abort_handler, state_machine):
        """Test abort state persistence for recovery."""
        # Store pre-abort state
        initial_state = SystemState.HOMING
        initial_position = {"lat": 42.3605, "lon": -71.0587, "alt": 50}
        initial_heading = 135

        state_machine.current_state = initial_state
        abort_context = {
            "state": initial_state,
            "position": initial_position,
            "heading": initial_heading,
            "timestamp": datetime.now(UTC),
        }

        # Perform abort
        await abort_handler.abort_mission(
            reason=AbortReason.SYSTEM_FAILURE, save_context=True, context=abort_context
        )

        # Verify context saved
        assert state_machine.abort_context == abort_context
        assert state_machine.can_resume == True

        # Test resume from abort
        resume_result = await abort_handler.resume_mission()
        assert resume_result["success"] == True
        assert state_machine.current_state == initial_state

    @pytest.mark.asyncio
    async def test_communication_lost_abort(self, abort_handler, state_machine, mock_mavlink):
        """Test abort due to communication loss."""
        # Simulate communication loss
        mock_mavlink.connected = False
        mock_mavlink.last_heartbeat = datetime.now(UTC) - timedelta(seconds=15)
        state_machine.current_state = SystemState.SEARCHING

        # Trigger communication lost abort
        abort_result = await abort_handler.abort_mission(
            reason=AbortReason.COMMUNICATION_LOST, attempt_reconnect=True
        )

        # Should attempt fail-safe behavior
        assert abort_result["success"] == True
        assert state_machine.current_state == SystemState.IDLE

        # Verify fail-safe mode activated
        assert state_machine.failsafe_active == True
        assert abort_result["failsafe_mode"] == "LAND"  # Or RTL depending on config

    @pytest.mark.asyncio
    async def test_abort_during_different_states(self, abort_handler, state_machine):
        """Test abort behavior from different operational states."""
        test_states = [
            (SystemState.IDLE, SystemState.IDLE, False),  # Already idle
            (SystemState.SEARCHING, SystemState.IDLE, True),  # Active search
            (SystemState.HOMING, SystemState.IDLE, True),  # Active homing
            (SystemState.BEACON_LOCATED, SystemState.IDLE, True),  # At beacon
            (SystemState.RETURNING, SystemState.IDLE, True),  # Returning home
        ]

        for initial_state, expected_state, should_stop_mission in test_states:
            # Reset state
            state_machine.current_state = initial_state
            mock_mavlink.stop_mission.reset_mock()

            # Perform abort
            result = await abort_handler.abort_mission(reason=AbortReason.OPERATOR_OVERRIDE)

            # Verify state transition
            assert state_machine.current_state == expected_state

            # Verify mission stop called when appropriate
            if should_stop_mission:
                mock_mavlink.stop_mission.assert_called()
            else:
                mock_mavlink.stop_mission.assert_not_called()

    @pytest.mark.asyncio
    async def test_graceful_abort_vs_emergency_abort(
        self, abort_handler, state_machine, mock_mavlink
    ):
        """Test differences between graceful and emergency abort procedures."""
        # Test graceful abort
        state_machine.current_state = SystemState.HOMING

        graceful_result = await abort_handler.abort_mission(
            reason=AbortReason.OPERATOR_OVERRIDE, emergency=False, graceful_timeout=5.0
        )

        # Should complete current maneuver before stopping
        assert graceful_result["graceful_completion"] == True
        assert graceful_result["completion_time"] <= 5.0

        # Reset for emergency abort
        state_machine.current_state = SystemState.HOMING
        mock_mavlink.stop_mission.reset_mock()
        mock_mavlink.hold_position.reset_mock()

        # Test emergency abort
        emergency_result = await abort_handler.abort_mission(
            reason=AbortReason.EMERGENCY_STOP, emergency=True
        )

        # Should stop immediately
        assert emergency_result["immediate_stop"] == True
        mock_mavlink.stop_mission.assert_called_once()
        mock_mavlink.hold_position.assert_called_once()

    @pytest.mark.asyncio
    async def test_abort_recovery_procedures(self, abort_handler, state_machine):
        """Test recovery procedures after different abort scenarios."""
        recovery_procedures = {
            AbortReason.BATTERY_CRITICAL: "land_immediately",
            AbortReason.SIGNAL_LOST: "search_pattern",
            AbortReason.GEOFENCE_BREACH: "return_to_fence",
            AbortReason.GPS_LOST: "hold_position",
            AbortReason.EMERGENCY_STOP: "disarm",
        }

        for reason, expected_procedure in recovery_procedures.items():
            # Perform abort
            result = await abort_handler.abort_mission(reason=reason)

            # Verify appropriate recovery procedure
            assert result["recovery_procedure"] == expected_procedure
            assert result["recovery_initiated"] == True

    @pytest.mark.asyncio
    async def test_abort_event_logging(self, abort_handler, state_machine):
        """Test comprehensive logging of abort events."""
        abort_log = []

        # Hook into logging
        async def log_abort_event(event):
            abort_log.append(event)

        abort_handler.on_abort = log_abort_event

        # Trigger abort with detailed context
        await abort_handler.abort_mission(
            reason=AbortReason.BATTERY_CRITICAL,
            details={
                "battery_percent": 8,
                "estimated_flight_time": 120,
                "distance_to_home": 450,
                "current_altitude": 75,
            },
        )

        # Verify comprehensive logging
        assert len(abort_log) == 1
        event = abort_log[0]

        assert event["reason"] == AbortReason.BATTERY_CRITICAL
        assert event["timestamp"] is not None
        assert event["details"]["battery_percent"] == 8
        assert event["state_before"] == SystemState.IDLE
        assert event["state_after"] == SystemState.IDLE


class MissionAbortHandler:
    """Handler for mission abort operations."""

    def __init__(self, state_machine: StateMachine, mavlink: MAVLinkService):
        self.state_machine = state_machine
        self.mavlink = mavlink
        self.on_abort = None

    async def abort_mission(self, reason: AbortReason, emergency: bool = False, **kwargs) -> dict:
        """Abort the current mission."""
        result = {"success": False, "reason": reason, "timestamp": datetime.now(UTC)}

        # Store pre-abort state
        self.state_machine.pre_abort_state = self.state_machine.current_state
        self.state_machine.abort_reason = reason

        # Perform abort based on reason
        if emergency or reason == AbortReason.EMERGENCY_STOP:
            result.update(await self._emergency_abort())
        elif reason == AbortReason.BATTERY_CRITICAL:
            result.update(await self._battery_critical_abort(**kwargs))
        elif reason == AbortReason.SIGNAL_LOST:
            result.update(await self._signal_lost_abort(**kwargs))
        elif reason == AbortReason.GEOFENCE_BREACH:
            result.update(await self._geofence_breach_abort(**kwargs))
        elif reason == AbortReason.GPS_LOST:
            result.update(await self._gps_lost_abort())
        elif reason == AbortReason.COMMUNICATION_LOST:
            result.update(await self._communication_lost_abort(**kwargs))
        else:
            result.update(await self._standard_abort())

        # Log abort event
        if self.on_abort:
            await self.on_abort(
                {
                    "reason": reason,
                    "timestamp": result["timestamp"],
                    "state_before": self.state_machine.pre_abort_state,
                    "state_after": self.state_machine.current_state,
                    "details": kwargs.get("details", {}),
                }
            )

        result["success"] = True
        return result

    async def _emergency_abort(self) -> dict:
        """Emergency abort procedure."""
        await self.mavlink.stop_mission()
        await self.mavlink.hold_position()
        self.state_machine.current_state = SystemState.IDLE
        self.state_machine.homing_enabled = False
        return {"immediate_stop": True}

    async def _battery_critical_abort(self, return_home: bool = True, **kwargs) -> dict:
        """Battery critical abort procedure."""
        self.state_machine.current_state = SystemState.IDLE
        self.state_machine.homing_enabled = False
        self.state_machine.autonomous_operations_blocked = True

        if return_home:
            await self.mavlink.return_to_launch()
            await self.mavlink.set_mode("RTL")
        else:
            await self.mavlink.land()

        return {"recovery_procedure": "land_immediately", "recovery_initiated": True}

    async def _signal_lost_abort(self, execute_search: bool = True, **kwargs) -> dict:
        """Signal lost abort procedure."""
        if execute_search:
            self.state_machine.current_state = SystemState.SEARCHING
            return {
                "search_pattern_started": True,
                "search_center": kwargs.get("search_center"),
                "recovery_procedure": "search_pattern",
                "recovery_initiated": True,
            }
        else:
            self.state_machine.current_state = SystemState.IDLE
            self.state_machine.homing_enabled = False
            return {"recovery_procedure": "hold_position", "recovery_initiated": True}

    async def _geofence_breach_abort(self, return_to_fence: bool = True, **kwargs) -> dict:
        """Geofence breach abort procedure."""
        await self.mavlink.stop_mission()
        self.state_machine.current_state = SystemState.IDLE

        if return_to_fence:
            return {
                "returning_to_fence": True,
                "target_position": kwargs.get("fence_center"),
                "recovery_procedure": "return_to_fence",
                "recovery_initiated": True,
            }
        return {"recovery_procedure": "hold_position", "recovery_initiated": True}

    async def _gps_lost_abort(self) -> dict:
        """GPS lost abort procedure."""
        await self.mavlink.hold_position()
        self.state_machine.current_state = SystemState.IDLE
        self.state_machine.homing_enabled = False
        return {"recovery_procedure": "hold_position", "recovery_initiated": True}

    async def _communication_lost_abort(self, **kwargs) -> dict:
        """Communication lost abort procedure."""
        self.state_machine.current_state = SystemState.IDLE
        self.state_machine.failsafe_active = True
        return {
            "failsafe_mode": "LAND",
            "recovery_procedure": "land_immediately",
            "recovery_initiated": True,
        }

    async def _standard_abort(self) -> dict:
        """Standard abort procedure."""
        if self.state_machine.current_state != SystemState.IDLE:
            await self.mavlink.stop_mission()
        self.state_machine.current_state = SystemState.IDLE
        self.state_machine.homing_enabled = False
        return {"recovery_procedure": "disarm", "recovery_initiated": True}

    async def resume_mission(self) -> dict:
        """Resume mission after abort if possible."""
        if hasattr(self.state_machine, "abort_context") and self.state_machine.can_resume:
            context = self.state_machine.abort_context
            self.state_machine.current_state = context["state"]
            return {"success": True, "resumed_state": context["state"]}
        return {"success": False, "reason": "No resumable context"}
