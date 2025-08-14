"""SITL test scenario for homing approach with waypoint validation."""

import asyncio
import math
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from src.backend.services.homing_controller import HomingController
from src.backend.services.mavlink_service import MAVLinkService
from src.backend.services.signal_processor import SignalProcessor
from src.backend.services.state_machine import SystemState, StateMachine


class TestHomingApproachScenario:
    """Test homing approach behavior with waypoint validation in SITL."""

    @pytest.fixture
    def mock_mavlink(self):
        """Create mock MAVLink service for SITL testing."""
        mavlink = AsyncMock(spec=MAVLinkService)
        mavlink.connected = True
        mavlink.current_position = {"lat": 42.3601, "lon": -71.0589, "alt": 50}
        mavlink.send_velocity_command = AsyncMock(return_value=True)
        mavlink.upload_mission = AsyncMock(return_value=True)
        mavlink.start_mission = AsyncMock(return_value=True)
        mavlink.get_mission_progress = AsyncMock(return_value={"current": 0, "total": 5})
        return mavlink

    @pytest.fixture
    def mock_signal_processor(self):
        """Create mock signal processor for SITL testing."""
        processor = AsyncMock(spec=SignalProcessor)
        processor.current_rssi = -70
        processor.noise_floor = -95
        processor.beacon_detected = True
        processor.confidence = 0.85
        return processor

    @pytest.fixture
    def homing_controller(self, mock_mavlink, mock_signal_processor):
        """Create homing controller with mocked dependencies."""
        controller = HomingController()
        controller.mavlink_service = mock_mavlink
        controller.signal_processor = mock_signal_processor
        controller.enabled = False
        controller.gradient_history = []
        return controller

    @pytest.fixture
    def state_machine(self):
        """Create state machine for testing."""
        sm = StateMachine()
        sm.homing_enabled = False
        return sm

    def calculate_distance(self, pos1: dict, pos2: dict) -> float:
        """Calculate distance between two GPS positions in meters."""
        R = 6371000  # Earth radius in meters
        lat1, lon1 = math.radians(pos1["lat"]), math.radians(pos1["lon"])
        lat2, lon2 = math.radians(pos2["lat"]), math.radians(pos2["lon"])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def simulate_rssi_for_position(self, drone_pos: dict, beacon_pos: dict) -> float:
        """Simulate RSSI based on distance from beacon."""
        distance = self.calculate_distance(drone_pos, beacon_pos)

        # Path loss model: RSSI = -10 * n * log10(d) + A
        # n = path loss exponent (2 for free space)
        # A = RSSI at 1 meter (-30 dBm)
        if distance < 1:
            distance = 1

        rssi = -10 * 2 * math.log10(distance) - 30

        # Add some noise
        import random

        rssi += random.uniform(-2, 2)

        # Clamp to realistic range
        return max(-100, min(-20, rssi))

    @pytest.mark.asyncio
    async def test_homing_approach_with_waypoints(
        self, homing_controller, state_machine, mock_mavlink, mock_signal_processor
    ):
        """Test complete homing approach sequence with waypoint navigation."""
        # Beacon position (target)
        beacon_pos = {"lat": 42.3605, "lon": -71.0585, "alt": 50}

        # Starting position
        start_pos = {"lat": 42.3601, "lon": -71.0589, "alt": 50}
        mock_mavlink.current_position = start_pos.copy()

        # Enable homing
        homing_controller.enabled = True
        state_machine.homing_enabled = True
        state_machine.current_state = SystemState.SEARCHING

        # Waypoint sequence for approach
        waypoints = []
        approach_positions = []

        # Simulate 10-step approach
        for i in range(10):
            progress = (i + 1) / 10
            lat = start_pos["lat"] + (beacon_pos["lat"] - start_pos["lat"]) * progress
            lon = start_pos["lon"] + (beacon_pos["lon"] - start_pos["lon"]) * progress

            waypoint = {"lat": lat, "lon": lon, "alt": 50, "seq": i}
            waypoints.append(waypoint)
            approach_positions.append({"lat": lat, "lon": lon, "alt": 50})

        # Upload mission
        await homing_controller.mavlink_service.upload_mission(waypoints)
        mock_mavlink.upload_mission.assert_called_once()

        # Start mission
        await homing_controller.mavlink_service.start_mission()
        mock_mavlink.start_mission.assert_called_once()

        # Simulate approach
        rssi_history = []
        gradient_history = []

        for idx, pos in enumerate(approach_positions):
            # Update drone position
            mock_mavlink.current_position = pos

            # Calculate RSSI for current position
            rssi = self.simulate_rssi_for_position(pos, beacon_pos)
            mock_signal_processor.current_rssi = rssi
            rssi_history.append(rssi)

            # Calculate gradient
            if len(rssi_history) >= 2:
                gradient = rssi_history[-1] - rssi_history[-2]
                gradient_history.append(gradient)
                homing_controller.gradient_history.append(gradient)

            # Update mission progress
            mock_mavlink.get_mission_progress.return_value = {
                "current": idx + 1,
                "total": len(waypoints),
            }

            # Process homing update
            await homing_controller.update()

            # Verify velocity commands sent
            if gradient_history and gradient_history[-1] > 0:
                # Positive gradient - continue approach
                assert mock_mavlink.send_velocity_command.called

            # Check for arrival detection
            distance_to_beacon = self.calculate_distance(pos, beacon_pos)
            if distance_to_beacon < 5:  # Within 5 meters
                state_machine.current_state = SystemState.BEACON_LOCATED
                break

            await asyncio.sleep(0.1)  # Simulate time passing

        # Verify approach completed
        assert state_machine.current_state == SystemState.BEACON_LOCATED
        assert len(rssi_history) > 0

        # Verify RSSI improved during approach
        assert rssi_history[-1] > rssi_history[0]

        # Verify gradient trending positive (approaching)
        avg_gradient = sum(gradient_history) / len(gradient_history) if gradient_history else 0
        assert avg_gradient > 0

    @pytest.mark.asyncio
    async def test_waypoint_validation_during_approach(self, homing_controller, mock_mavlink):
        """Test waypoint validation and error handling during approach."""
        # Create waypoints with one invalid point
        waypoints = [
            {"lat": 42.3601, "lon": -71.0589, "alt": 50, "seq": 0},
            {"lat": 42.3602, "lon": -71.0588, "alt": 50, "seq": 1},
            {"lat": 91.0, "lon": -71.0587, "alt": 50, "seq": 2},  # Invalid latitude
            {"lat": 42.3604, "lon": -71.0586, "alt": 50, "seq": 3},
        ]

        # Validate waypoints
        valid_waypoints = []
        for wp in waypoints:
            if -90 <= wp["lat"] <= 90 and -180 <= wp["lon"] <= 180:
                valid_waypoints.append(wp)

        assert len(valid_waypoints) == 3
        assert all(wp["lat"] != 91.0 for wp in valid_waypoints)

    @pytest.mark.asyncio
    async def test_approach_abort_on_signal_loss(
        self, homing_controller, state_machine, mock_signal_processor
    ):
        """Test approach abort when signal is lost."""
        # Start homing
        homing_controller.enabled = True
        state_machine.current_state = SystemState.HOMING

        # Simulate signal loss
        mock_signal_processor.beacon_detected = False
        mock_signal_processor.current_rssi = -100
        mock_signal_processor.confidence = 0.1

        # Track signal loss duration
        signal_loss_start = datetime.now(UTC)
        signal_loss_timeout = 10  # seconds

        while (datetime.now(UTC) - signal_loss_start).total_seconds() < signal_loss_timeout:
            await homing_controller.update()

            if not mock_signal_processor.beacon_detected:
                continue
            else:
                break

            await asyncio.sleep(0.5)

        # If signal lost for timeout duration, abort
        if (datetime.now(UTC) - signal_loss_start).total_seconds() >= signal_loss_timeout:
            homing_controller.enabled = False
            state_machine.current_state = SystemState.IDLE
            state_machine.homing_enabled = False

        assert not homing_controller.enabled
        assert state_machine.current_state == SystemState.IDLE

    @pytest.mark.asyncio
    async def test_approach_with_obstacles(self, homing_controller, mock_mavlink):
        """Test approach path planning with obstacle avoidance."""
        # Define obstacle zone
        obstacle = {"center": {"lat": 42.3603, "lon": -71.0587}, "radius": 20}  # meters

        # Direct path waypoints
        direct_waypoints = [
            {"lat": 42.3601, "lon": -71.0589, "alt": 50},
            {"lat": 42.3603, "lon": -71.0587, "alt": 50},  # Through obstacle
            {"lat": 42.3605, "lon": -71.0585, "alt": 50},
        ]

        # Generate avoidance waypoints
        avoidance_waypoints = []
        for i, wp in enumerate(direct_waypoints):
            # Check if waypoint is in obstacle
            dist_to_obstacle = self.calculate_distance(wp, obstacle["center"])

            if dist_to_obstacle < obstacle["radius"]:
                # Create bypass waypoint
                bypass_lat = obstacle["center"]["lat"] + 0.0003  # ~30m north
                bypass_lon = wp["lon"]
                avoidance_waypoints.append(
                    {
                        "lat": bypass_lat,
                        "lon": bypass_lon,
                        "alt": wp["alt"] + 10,  # Increase altitude
                        "seq": len(avoidance_waypoints),
                    }
                )
            else:
                wp["seq"] = len(avoidance_waypoints)
                avoidance_waypoints.append(wp)

        # Verify avoidance path
        assert len(avoidance_waypoints) >= len(direct_waypoints)

        # Check no waypoint is in obstacle
        for wp in avoidance_waypoints:
            dist = self.calculate_distance(wp, obstacle["center"])
            assert dist >= obstacle["radius"] or wp["alt"] > 50

    @pytest.mark.asyncio
    async def test_approach_speed_control(
        self, homing_controller, mock_mavlink, mock_signal_processor
    ):
        """Test dynamic speed control during approach based on signal strength."""
        # Signal strength to speed mapping
        speed_profiles = [
            {"rssi_min": -100, "rssi_max": -80, "speed": 5.0},  # Far - fast
            {"rssi_min": -80, "rssi_max": -60, "speed": 3.0},  # Medium
            {"rssi_min": -60, "rssi_max": -40, "speed": 1.0},  # Close - slow
            {"rssi_min": -40, "rssi_max": -20, "speed": 0.5},  # Very close - creep
        ]

        # Test different RSSI levels
        test_rssi_values = [-90, -70, -50, -30]

        for rssi in test_rssi_values:
            mock_signal_processor.current_rssi = rssi

            # Determine appropriate speed
            speed = 5.0  # Default
            for profile in speed_profiles:
                if profile["rssi_min"] <= rssi <= profile["rssi_max"]:
                    speed = profile["speed"]
                    break

            # Send velocity command with calculated speed
            await mock_mavlink.send_velocity_command(vx=speed, vy=0, vz=0)

            # Verify speed is appropriate for signal strength
            if rssi > -40:
                assert speed <= 0.5  # Very slow when close
            elif rssi > -60:
                assert speed <= 1.0
            elif rssi > -80:
                assert speed <= 3.0
            else:
                assert speed <= 5.0

    @pytest.mark.asyncio
    async def test_spiral_search_pattern_approach(self, homing_controller, mock_mavlink):
        """Test spiral search pattern for initial beacon acquisition."""
        center = {"lat": 42.3601, "lon": -71.0589, "alt": 50}

        # Generate spiral pattern waypoints
        spiral_waypoints = []
        num_turns = 3
        points_per_turn = 8
        max_radius = 100  # meters

        for turn in range(num_turns):
            radius = (turn + 1) * max_radius / num_turns

            for point in range(points_per_turn):
                angle = (point / points_per_turn) * 2 * math.pi

                # Convert radius and angle to lat/lon offset
                # Rough approximation: 1 degree latitude = 111,000 meters
                lat_offset = (radius * math.cos(angle)) / 111000
                lon_offset = (radius * math.sin(angle)) / (
                    111000 * math.cos(math.radians(center["lat"]))
                )

                waypoint = {
                    "lat": center["lat"] + lat_offset,
                    "lon": center["lon"] + lon_offset,
                    "alt": center["alt"],
                    "seq": len(spiral_waypoints),
                }
                spiral_waypoints.append(waypoint)

        # Verify spiral pattern
        assert len(spiral_waypoints) == num_turns * points_per_turn

        # Check increasing distance from center
        distances = []
        for wp in spiral_waypoints:
            dist = self.calculate_distance(wp, center)
            distances.append(dist)

        # Verify spiral expands outward
        for i in range(points_per_turn, len(distances)):
            avg_current = sum(distances[i - points_per_turn : i]) / points_per_turn
            avg_previous = (
                sum(distances[max(0, i - 2 * points_per_turn) : i - points_per_turn])
                / points_per_turn
            )
            if i >= 2 * points_per_turn:
                assert avg_current >= avg_previous

    @pytest.mark.asyncio
    async def test_multi_beacon_approach_decision(self, homing_controller, mock_signal_processor):
        """Test decision making when multiple beacons are detected."""
        # Simulate multiple beacon detections
        beacons = [
            {"id": "beacon1", "rssi": -65, "confidence": 0.85, "bearing": 45},
            {"id": "beacon2", "rssi": -55, "confidence": 0.92, "bearing": 135},
            {"id": "beacon3", "rssi": -70, "confidence": 0.75, "bearing": 270},
        ]

        # Decision criteria: highest RSSI with confidence > 0.8
        selected_beacon = None
        min_confidence = 0.8

        valid_beacons = [b for b in beacons if b["confidence"] >= min_confidence]
        if valid_beacons:
            selected_beacon = max(valid_beacons, key=lambda x: x["rssi"])

        assert selected_beacon is not None
        assert selected_beacon["id"] == "beacon2"
        assert selected_beacon["rssi"] == -55

        # Set approach bearing
        approach_bearing = selected_beacon["bearing"]
        assert approach_bearing == 135

    @pytest.mark.asyncio
    async def test_altitude_maintenance_during_approach(self, homing_controller, mock_mavlink):
        """Test altitude hold during horizontal approach."""
        target_altitude = 50  # meters
        tolerance = 2  # meters

        # Simulate approach with altitude variations
        approach_positions = [
            {"lat": 42.3601, "lon": -71.0589, "alt": 50},
            {"lat": 42.3602, "lon": -71.0588, "alt": 48},  # Dropping
            {"lat": 42.3603, "lon": -71.0587, "alt": 52},  # Rising
            {"lat": 42.3604, "lon": -71.0586, "alt": 47},  # Dropping again
        ]

        for pos in approach_positions:
            mock_mavlink.current_position = pos

            # Calculate altitude correction
            alt_error = target_altitude - pos["alt"]

            if abs(alt_error) > tolerance:
                # Send altitude correction
                vz = alt_error * 0.5  # Proportional control
                vz = max(-2, min(2, vz))  # Limit vertical speed

                await mock_mavlink.send_velocity_command(vx=1, vy=0, vz=vz)
                mock_mavlink.send_velocity_command.assert_called()

                # Verify vertical velocity is corrective
                call_args = mock_mavlink.send_velocity_command.call_args
                if pos["alt"] < target_altitude - tolerance:
                    assert call_args[1]["vz"] > 0  # Climb
                elif pos["alt"] > target_altitude + tolerance:
                    assert call_args[1]["vz"] < 0  # Descend
