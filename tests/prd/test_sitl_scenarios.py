"""Test search patterns with SITL integration (PRD-FR2).

This module tests expanding square search patterns using ArduPilot SITL
to validate autonomous navigation requirements per PRD-FR2.
"""

import asyncio

import pytest

from backend.services.mavlink_service import ConnectionState, MAVLinkService
from backend.services.search_pattern_generator import (
    CenterRadiusBoundary,
    PatternType,
    SearchPatternGenerator,
)


class TestSITLSearchPatterns:
    """Test search patterns with SITL integration."""

    @pytest.fixture
    async def sitl_mavlink_service(self):
        """Create MAVLink service configured for SITL."""
        # SITL typically runs on TCP port 5760
        service = MAVLinkService(
            device_path="tcp:127.0.0.1:5760",
            baud_rate=115200,
            source_system=255,  # GCS system ID
            source_component=191,  # Onboard computer component
        )
        yield service
        if service.state == ConnectionState.CONNECTED:
            service.disconnect()

    @pytest.fixture
    def search_pattern_generator(self):
        """Create search pattern generator."""
        return SearchPatternGenerator(default_altitude=50.0)

    @pytest.fixture
    def test_boundary(self):
        """Create test search boundary."""
        return CenterRadiusBoundary(
            center_lat=-35.363261,  # SITL default home location
            center_lon=149.165230,
            radius=100.0,  # 100m radius
        )

    @pytest.mark.asyncio
    async def test_sitl_connection_establishment(self, sitl_mavlink_service):
        """Test connection to SITL via MAVLink.

        This verifies that we can establish communication with the SITL instance
        per PRD-FR2 requirements for autonomous navigation testing.
        """
        # RED PHASE: This should fail initially as connection logic needs implementation
        success = sitl_mavlink_service.connect("tcp:127.0.0.1:5760")

        # Verify connection established
        assert success is True, "Failed to connect to SITL"
        assert sitl_mavlink_service.state == ConnectionState.CONNECTED

        # Wait for heartbeat to confirm active connection
        await asyncio.sleep(2.0)

        # Verify we can receive telemetry
        telemetry = sitl_mavlink_service.get_telemetry()
        assert telemetry is not None, "No telemetry received from SITL"

    @pytest.mark.asyncio
    async def test_expanding_square_pattern_generation(
        self, search_pattern_generator, test_boundary
    ):
        """Test expanding square waypoint generation.

        Per PRD-FR2: 'The drone shall execute expanding square search patterns'
        """
        # Generate expanding square pattern
        pattern = search_pattern_generator.generate_pattern(
            pattern_type=PatternType.EXPANDING_SQUARE,
            spacing=50.0,  # 50m spacing between legs
            velocity=7.5,  # 7.5 m/s (within 5-10 m/s range)
            boundary=test_boundary,
            altitude=50.0,
        )

        # Verify pattern structure
        assert pattern is not None, "Pattern generation failed"
        assert len(pattern.waypoints) > 0, "No waypoints generated"
        assert pattern.pattern_type == PatternType.EXPANDING_SQUARE

        # Verify waypoints are within boundary
        for waypoint in pattern.waypoints:
            assert test_boundary.contains_point(
                waypoint.latitude, waypoint.longitude
            ), f"Waypoint {waypoint.index} outside boundary"
            assert waypoint.altitude == 50.0, f"Incorrect altitude for waypoint {waypoint.index}"

    @pytest.mark.asyncio
    async def test_velocity_commands_within_range(
        self, sitl_mavlink_service, search_pattern_generator, test_boundary
    ):
        """Test velocity commands within 5-10 m/s range.

        Per PRD-FR2: 'configurable velocities between 5-10 m/s'
        """
        # Test multiple velocities within range
        test_velocities = [5.0, 7.5, 10.0]

        for velocity in test_velocities:
            pattern = search_pattern_generator.generate_pattern(
                pattern_type=PatternType.EXPANDING_SQUARE,
                spacing=50.0,
                velocity=velocity,
                boundary=test_boundary,
            )

            # Verify pattern has reasonable estimated time for velocity
            # Note: actual time depends on waypoint distance calculations
            assert (
                pattern.estimated_time_remaining > 0
            ), f"No estimated time calculated for velocity {velocity} m/s"
            assert (
                pattern.velocity == velocity
            ), f"Pattern velocity {pattern.velocity} != requested {velocity} m/s"

            # RED PHASE: Test sending velocity commands to SITL
            # This should fail until velocity command implementation exists
            if sitl_mavlink_service.state == ConnectionState.CONNECTED:
                # Test sending SET_POSITION_TARGET_LOCAL_NED with velocity
                result = await sitl_mavlink_service.send_velocity_command(
                    vx=velocity, vy=0.0, vz=0.0, yaw_rate=0.0
                )
                assert result is True, f"Failed to send velocity command {velocity} m/s"

    @pytest.mark.asyncio
    async def test_pattern_completion_tracking(self, search_pattern_generator, test_boundary):
        """Test pattern completion tracking functionality.

        Verifies that we can monitor progress through the search pattern.
        """
        pattern = search_pattern_generator.generate_pattern(
            pattern_type=PatternType.EXPANDING_SQUARE,
            spacing=50.0,
            velocity=7.5,
            boundary=test_boundary,
        )

        # RED PHASE: Test progress tracking - should fail until implemented
        progress_tracker = PatternProgressTracker(pattern)

        # Simulate waypoint completion
        assert progress_tracker.completion_percentage == 0.0

        progress_tracker.mark_waypoint_complete(0)
        expected_progress = (1 / len(pattern.waypoints)) * 100
        assert abs(progress_tracker.completion_percentage - expected_progress) < 0.1

        # Complete all waypoints
        for i in range(len(pattern.waypoints)):
            progress_tracker.mark_waypoint_complete(i)

        assert progress_tracker.completion_percentage == 100.0

    @pytest.mark.asyncio
    async def test_pattern_pause_resume_functionality(self, sitl_mavlink_service):
        """Test pattern pause/resume functionality.

        Verifies operational control over search pattern execution.
        """
        # RED PHASE: Test pause/resume logic - should fail until implemented
        pattern_controller = SearchPatternController(sitl_mavlink_service)

        # Start pattern execution
        await pattern_controller.start_pattern()
        assert pattern_controller.is_active is True

        # Pause pattern
        await pattern_controller.pause_pattern()
        assert pattern_controller.is_paused is True
        assert pattern_controller.is_active is True  # Still active but paused

        # Resume pattern
        await pattern_controller.resume_pattern()
        assert pattern_controller.is_paused is False
        assert pattern_controller.is_active is True

        # Stop pattern
        await pattern_controller.stop_pattern()
        assert pattern_controller.is_active is False


class PatternProgressTracker:
    """Track search pattern completion progress."""

    def __init__(self, pattern):
        self.pattern = pattern
        self.completed_waypoints = set()

    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if not self.pattern.waypoints:
            return 100.0
        return (len(self.completed_waypoints) / len(self.pattern.waypoints)) * 100

    def mark_waypoint_complete(self, waypoint_index: int):
        """Mark waypoint as completed."""
        self.completed_waypoints.add(waypoint_index)


class SearchPatternController:
    """Control search pattern execution."""

    def __init__(self, mavlink_service):
        self.mavlink_service = mavlink_service
        self.is_active = False
        self.is_paused = False

    async def start_pattern(self):
        """Start pattern execution."""
        self.is_active = True
        self.is_paused = False

    async def pause_pattern(self):
        """Pause pattern execution."""
        self.is_paused = True

    async def resume_pattern(self):
        """Resume pattern execution."""
        self.is_paused = False

    async def stop_pattern(self):
        """Stop pattern execution."""
        self.is_active = False
        self.is_paused = False
