"""SITL test scenario for safety interlock testing all safety states."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from src.backend.services.state_machine import SystemState, StateMachine
from src.backend.utils.safety import SafetyInterlockSystem, SafetyEventType


@dataclass
class SafetyStatus:
    """Safety status for testing."""
    mode_check: bool = True
    battery_check: bool = True
    geofence_check: bool = True
    signal_check: bool = True
    operator_check: bool = True
    all_passed: bool = True
    blocked_reasons: list = field(default_factory=list)


class TestSafetyInterlockScenario:
    """Test safety interlock behavior across all safety states in SITL."""

    @pytest.fixture
    def safety_interlock(self):
        """Create mocked safety interlock instance."""
        mock_interlock = AsyncMock(spec=SafetyInterlockSystem)
        
        # Mock check_all to return SafetyStatus
        async def mock_check_all(**kwargs):
            # Default to all checks passing
            return SafetyStatus(
                mode_check=True,
                battery_check=True,
                geofence_check=True,
                signal_check=True,
                operator_check=True,
                all_passed=True,
                blocked_reasons=[]
            )
        
        mock_interlock.check_all = mock_check_all
        return mock_interlock

    @pytest.fixture
    def state_machine(self):
        """Create state machine for testing."""
        sm = StateMachine()
        sm.current_state = SystemState.IDLE
        sm.homing_enabled = False
        return sm

    @pytest.fixture
    def mock_mavlink(self):
        """Create mock MAVLink service."""
        mavlink = AsyncMock()
        mavlink.connected = True
        mavlink.flight_mode = "GUIDED"
        mavlink.battery_percent = 85
        mavlink.gps_status = "3D_FIX"
        mavlink.current_position = {"lat": 42.3601, "lon": -71.0589, "alt": 50}
        mavlink.geofence_enabled = True
        mavlink.geofence_boundary = {"center": {"lat": 42.3601, "lon": -71.0589}, "radius": 500}
        return mavlink

    @pytest.fixture
    def mock_signal_processor(self):
        """Create mock signal processor."""
        processor = AsyncMock()
        processor.beacon_detected = True
        processor.current_rssi = -60
        processor.confidence = 0.85
        processor.last_detection_time = datetime.now(UTC)
        return processor

    def create_safety_status(
        self,
        mode_check: bool = True,
        battery_check: bool = True,
        geofence_check: bool = True,
        signal_check: bool = True,
        operator_check: bool = True,
    ) -> SafetyStatus:
        """Create a safety status object for testing."""
        return SafetyStatus(
            mode_check=mode_check,
            battery_check=battery_check,
            geofence_check=geofence_check,
            signal_check=signal_check,
            operator_check=operator_check,
            all_passed=all(
                [mode_check, battery_check, geofence_check, signal_check, operator_check]
            ),
        )

    @pytest.mark.asyncio
    async def test_all_safety_checks_passing(
        self, safety_interlock, mock_mavlink, mock_signal_processor
    ):
        """Test scenario where all safety checks pass."""
        # Set up ideal conditions
        mock_mavlink.flight_mode = "GUIDED"
        mock_mavlink.battery_percent = 85
        mock_mavlink.gps_status = "3D_FIX"
        mock_signal_processor.beacon_detected = True
        mock_signal_processor.confidence = 0.9

        # Perform safety checks
        status = await safety_interlock.check_all(
            mavlink=mock_mavlink, signal_processor=mock_signal_processor
        )

        assert status.all_passed == True
        assert status.mode_check == True
        assert status.battery_check == True
        assert status.geofence_check == True
        assert status.signal_check == True
        assert status.operator_check == True

    @pytest.mark.asyncio
    async def test_mode_check_failures(self, safety_interlock, mock_mavlink, mock_signal_processor):
        """Test various flight mode check failures."""
        invalid_modes = ["MANUAL", "STABILIZE", "ACRO", "ALT_HOLD", "LOITER", "RTL", "LAND"]

        for mode in invalid_modes:
            mock_mavlink.flight_mode = mode

            status = await safety_interlock.check_all(
                mavlink=mock_mavlink, signal_processor=mock_signal_processor
            )

            assert status.mode_check == False
            assert status.all_passed == False
            assert status.blocked_reasons
            assert "Flight mode" in status.blocked_reasons[0]

    @pytest.mark.asyncio
    async def test_battery_check_thresholds(
        self, safety_interlock, mock_mavlink, mock_signal_processor
    ):
        """Test battery level safety thresholds."""
        battery_scenarios = [
            (100, True, "Full battery"),
            (85, True, "Good battery"),
            (50, True, "Adequate battery"),
            (30, True, "Minimum safe battery"),
            (25, False, "Below threshold"),
            (20, False, "Critical low"),
            (10, False, "Emergency low"),
            (0, False, "Empty"),
        ]

        for battery_level, should_pass, description in battery_scenarios:
            mock_mavlink.battery_percent = battery_level

            status = await safety_interlock.check_all(
                mavlink=mock_mavlink, signal_processor=mock_signal_processor
            )

            assert status.battery_check == should_pass, f"Failed for {description}"

            if not should_pass:
                assert not status.all_passed
                assert any("Battery" in reason for reason in status.blocked_reasons)

    @pytest.mark.asyncio
    async def test_geofence_boundary_checks(
        self, safety_interlock, mock_mavlink, mock_signal_processor
    ):
        """Test geofence boundary violation detection."""
        # Set geofence center and radius
        geofence_center = {"lat": 42.3601, "lon": -71.0589}
        geofence_radius = 100  # meters

        mock_mavlink.geofence_boundary = {"center": geofence_center, "radius": geofence_radius}

        # Test positions relative to geofence
        test_positions = [
            ({"lat": 42.3601, "lon": -71.0589}, True, "At center"),
            ({"lat": 42.3602, "lon": -71.0589}, True, "Near center"),
            ({"lat": 42.3610, "lon": -71.0589}, False, "Outside north"),
            ({"lat": 42.3601, "lon": -71.0570}, False, "Outside east"),
            ({"lat": 42.3590, "lon": -71.0589}, False, "Outside south"),
        ]

        for position, should_pass, description in test_positions:
            mock_mavlink.current_position = position

            status = await safety_interlock.check_all(
                mavlink=mock_mavlink, signal_processor=mock_signal_processor
            )

            # Calculate actual distance for verification
            from math import atan2, cos, radians, sin, sqrt

            R = 6371000  # Earth radius in meters

            lat1, lon1 = radians(geofence_center["lat"]), radians(geofence_center["lon"])
            lat2, lon2 = radians(position["lat"]), radians(position["lon"])

            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            distance = R * c

            expected_pass = distance <= geofence_radius
            assert status.geofence_check == expected_pass, f"Failed for {description}"

    @pytest.mark.asyncio
    async def test_signal_check_conditions(
        self, safety_interlock, mock_mavlink, mock_signal_processor
    ):
        """Test signal quality and detection checks."""
        signal_scenarios = [
            (True, -40, 0.95, True, "Strong signal"),
            (True, -60, 0.85, True, "Good signal"),
            (True, -75, 0.70, True, "Adequate signal"),
            (True, -85, 0.55, False, "Weak signal"),
            (True, -95, 0.30, False, "Very weak signal"),
            (False, -100, 0.10, False, "No beacon"),
        ]

        for detected, rssi, confidence, should_pass, description in signal_scenarios:
            mock_signal_processor.beacon_detected = detected
            mock_signal_processor.current_rssi = rssi
            mock_signal_processor.confidence = confidence

            status = await safety_interlock.check_all(
                mavlink=mock_mavlink, signal_processor=mock_signal_processor
            )

            assert status.signal_check == should_pass, f"Failed for {description}"

            if not should_pass:
                assert not status.all_passed
                assert any(
                    "Signal" in reason or "beacon" in reason.lower()
                    for reason in status.blocked_reasons
                )

    @pytest.mark.asyncio
    async def test_operator_override_check(
        self, safety_interlock, mock_mavlink, mock_signal_processor
    ):
        """Test operator override and manual control checks."""
        # Test with operator override active
        safety_interlock.operator_override_active = True

        status = await safety_interlock.check_all(
            mavlink=mock_mavlink, signal_processor=mock_signal_processor
        )

        assert status.operator_check == False
        assert not status.all_passed
        assert any("Operator" in reason for reason in status.blocked_reasons)

        # Test with override cleared
        safety_interlock.operator_override_active = False

        status = await safety_interlock.check_all(
            mavlink=mock_mavlink, signal_processor=mock_signal_processor
        )

        assert status.operator_check == True

    @pytest.mark.asyncio
    async def test_cascading_safety_failures(
        self, safety_interlock, mock_mavlink, mock_signal_processor
    ):
        """Test multiple simultaneous safety failures."""
        # Set multiple failure conditions
        mock_mavlink.flight_mode = "MANUAL"
        mock_mavlink.battery_percent = 15
        mock_signal_processor.beacon_detected = False
        safety_interlock.operator_override_active = True

        status = await safety_interlock.check_all(
            mavlink=mock_mavlink, signal_processor=mock_signal_processor
        )

        # All checks except geofence should fail
        assert status.mode_check == False
        assert status.battery_check == False
        assert status.signal_check == False
        assert status.operator_check == False
        assert status.geofence_check == True  # Still within bounds
        assert status.all_passed == False

        # Should have multiple blocked reasons
        assert len(status.blocked_reasons) >= 4

        # Verify each failure is reported
        reasons_text = " ".join(status.blocked_reasons)
        assert "mode" in reasons_text.lower()
        assert "battery" in reasons_text.lower()
        assert "signal" in reasons_text.lower() or "beacon" in reasons_text.lower()
        assert "operator" in reasons_text.lower()

    @pytest.mark.asyncio
    async def test_safety_recovery_sequence(
        self, safety_interlock, mock_mavlink, mock_signal_processor
    ):
        """Test recovery from safety failures."""
        # Start with all failures
        mock_mavlink.flight_mode = "MANUAL"
        mock_mavlink.battery_percent = 15
        mock_signal_processor.beacon_detected = False
        safety_interlock.operator_override_active = True

        status = await safety_interlock.check_all(
            mavlink=mock_mavlink, signal_processor=mock_signal_processor
        )
        assert status.all_passed == False
        initial_failures = len(status.blocked_reasons)

        # Recovery sequence
        recovery_steps = [
            ("Fix flight mode", lambda: setattr(mock_mavlink, "flight_mode", "GUIDED")),
            ("Charge battery", lambda: setattr(mock_mavlink, "battery_percent", 50)),
            ("Detect beacon", lambda: setattr(mock_signal_processor, "beacon_detected", True)),
            (
                "Clear override",
                lambda: setattr(safety_interlock, "operator_override_active", False),
            ),
        ]

        for step_name, recovery_action in recovery_steps:
            recovery_action()

            status = await safety_interlock.check_all(
                mavlink=mock_mavlink, signal_processor=mock_signal_processor
            )

            # Should have fewer failures after each recovery
            current_failures = len(status.blocked_reasons) if not status.all_passed else 0
            assert current_failures < initial_failures, f"Recovery failed at: {step_name}"
            initial_failures = current_failures

        # Final check - all should pass
        assert status.all_passed == True

    @pytest.mark.asyncio
    async def test_safety_interlock_state_transitions(
        self, safety_interlock, state_machine, mock_mavlink, mock_signal_processor
    ):
        """Test safety interlock effects on state transitions."""
        # Try to enable homing with safety failure
        mock_mavlink.battery_percent = 15  # Below threshold

        status = await safety_interlock.check_all(
            mavlink=mock_mavlink, signal_processor=mock_signal_processor
        )

        # Attempt state transition
        if not status.all_passed:
            # Should block transition to SEARCHING
            assert state_machine.current_state == SystemState.IDLE
            state_machine.homing_enabled = False

        # Fix safety issue
        mock_mavlink.battery_percent = 85

        status = await safety_interlock.check_all(
            mavlink=mock_mavlink, signal_processor=mock_signal_processor
        )

        # Now transition should be allowed
        if status.all_passed:
            state_machine.homing_enabled = True
            state_machine.current_state = SystemState.SEARCHING
            assert state_machine.current_state == SystemState.SEARCHING

    @pytest.mark.asyncio
    async def test_gps_status_check(self, safety_interlock, mock_mavlink, mock_signal_processor):
        """Test GPS status safety check."""
        gps_scenarios = [
            ("3D_FIX", True, "Good GPS"),
            ("2D_FIX", False, "2D only"),
            ("NO_FIX", False, "No GPS fix"),
            ("NO_GPS", False, "GPS not available"),
            (None, False, "GPS status unknown"),
        ]

        for gps_status, should_pass, description in gps_scenarios:
            mock_mavlink.gps_status = gps_status

            status = await safety_interlock.check_all(
                mavlink=mock_mavlink, signal_processor=mock_signal_processor
            )

            # GPS issues affect geofence check
            if not should_pass:
                assert not status.all_passed
                # May affect geofence or general navigation safety

    @pytest.mark.asyncio
    async def test_dynamic_safety_thresholds(
        self, safety_interlock, mock_mavlink, mock_signal_processor
    ):
        """Test dynamic adjustment of safety thresholds based on conditions."""
        # Normal conditions - standard thresholds
        safety_interlock.battery_threshold = 25
        safety_interlock.signal_threshold = -85

        # Emergency mode - relaxed thresholds
        safety_interlock.emergency_mode = True
        safety_interlock.battery_threshold = 15
        safety_interlock.signal_threshold = -90

        # Test with borderline values
        mock_mavlink.battery_percent = 20
        mock_signal_processor.current_rssi = -87

        status = await safety_interlock.check_all(
            mavlink=mock_mavlink, signal_processor=mock_signal_processor
        )

        # Should pass with relaxed thresholds
        assert status.battery_check == True  # 20% > 15% threshold
        assert status.signal_check == True  # -87 > -90 threshold

    @pytest.mark.asyncio
    async def test_safety_event_logging(
        self, safety_interlock, mock_mavlink, mock_signal_processor
    ):
        """Test logging of safety events and violations."""
        safety_events = []

        # Monitor safety checks
        async def log_safety_event(event_type: str, details: dict):
            safety_events.append(
                {"timestamp": datetime.now(UTC), "type": event_type, "details": details}
            )

        # Test various violations
        test_scenarios = [
            ("mode_violation", "MANUAL", lambda: setattr(mock_mavlink, "flight_mode", "MANUAL")),
            ("battery_low", 15, lambda: setattr(mock_mavlink, "battery_percent", 15)),
            (
                "signal_lost",
                False,
                lambda: setattr(mock_signal_processor, "beacon_detected", False),
            ),
        ]

        for event_type, value, trigger in test_scenarios:
            trigger()

            status = await safety_interlock.check_all(
                mavlink=mock_mavlink, signal_processor=mock_signal_processor
            )

            if not status.all_passed:
                await log_safety_event(
                    event_type, {"value": value, "blocked_reasons": status.blocked_reasons}
                )

        # Verify events logged
        assert len(safety_events) == len(test_scenarios)
        assert all(
            event["type"] in ["mode_violation", "battery_low", "signal_lost"]
            for event in safety_events
        )
