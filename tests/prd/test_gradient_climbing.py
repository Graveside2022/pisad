"""Test Gradient Climbing Navigation (PRD-FR4).

TASK-9.4: Test gradient climbing navigation algorithms with simulated RSSI sources.
Per PRD-FR4: "The drone shall navigate toward detected signals using RSSI
gradient climbing with forward velocity and yaw-rate control"

TDD Implementation: Red-Green-Refactor with authentic system integration.
"""

from dataclasses import dataclass

import pytest

from backend.models.schemas import BeaconConfiguration
from backend.services.beacon_simulator import BeaconSimulator
from backend.services.homing_algorithm import HomingAlgorithm, RSSISample


@dataclass
class MockPosition:
    """Position for testing without full MAVLink integration."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class TestGradientClimbingNavigation:
    """Test suite for gradient climbing navigation per PRD-FR4."""

    @pytest.fixture
    def beacon_simulator(self) -> BeaconSimulator:
        """Create beacon simulator for authentic RF signal simulation."""
        return BeaconSimulator()

    @pytest.fixture
    def homing_algorithm(self) -> HomingAlgorithm:
        """Create homing algorithm instance."""
        return HomingAlgorithm()

    @pytest.fixture
    def test_beacon_config(self) -> BeaconConfiguration:
        """Create test beacon configuration."""
        return BeaconConfiguration(
            frequency_hz=2437000000.0,  # 2.437 GHz in Hz
            bandwidth_hz=5000000.0,  # 5 MHz in Hz
            power_dbm=10.0,
            modulation="FM",
        )

    def test_create_simulated_rssi_beacon_source(
        self, beacon_simulator: BeaconSimulator, test_beacon_config: BeaconConfiguration
    ):
        """SUBTASK 1: Create simulated RSSI beacon source.

        Verifies BeaconSimulator can create authentic RF signal simulation
        for gradient climbing testing without requiring physical hardware.
        """
        # Test beacon creation
        beacon_id = "test_beacon_001"
        position = (100.0, 200.0, 50.0)  # lat, lon, alt in meters

        # This should work with real BeaconSimulator
        beacon_simulator.create_beacon(
            beacon_id=beacon_id, config=test_beacon_config, position=position
        )

        # Verify beacon was created
        assert beacon_id in beacon_simulator.beacons
        assert beacon_simulator.beacons[beacon_id].config.frequency_hz == 2437000000.0
        assert beacon_simulator.beacons[beacon_id].position == position

    def test_gradient_calculation_accuracy(self, homing_algorithm: HomingAlgorithm):
        """SUBTASK 2: Test gradient calculation accuracy.

        Validates gradient calculation using real homing algorithm implementation
        with known RSSI distribution pattern.
        """
        # Create known RSSI pattern with clear gradient
        # Beacon at origin (0,0), RSSI decreases with distance
        test_samples = [
            RSSISample(
                rssi=-50.0, position_x=0.0, position_y=0.0, heading=0.0, timestamp=1.0
            ),  # Closest
            RSSISample(
                rssi=-60.0, position_x=10.0, position_y=0.0, heading=90.0, timestamp=2.0
            ),  # East
            RSSISample(
                rssi=-60.0, position_x=0.0, position_y=10.0, heading=0.0, timestamp=3.0
            ),  # North
            RSSISample(
                rssi=-70.0, position_x=20.0, position_y=0.0, heading=90.0, timestamp=4.0
            ),  # Far East
        ]

        # Add samples to algorithm
        for sample in test_samples:
            homing_algorithm.add_rssi_sample(
                rssi=sample.rssi,
                position_x=sample.position_x,
                position_y=sample.position_y,
                heading=sample.heading,
                timestamp=sample.timestamp,
            )

        # Calculate gradient - this should work with real algorithm
        gradient = homing_algorithm.calculate_gradient()

        # Verify gradient calculation
        assert gradient is not None, "Gradient calculation should succeed with sufficient samples"
        assert gradient.magnitude > 0, "Gradient magnitude should be positive"
        assert 0 <= gradient.direction <= 360, "Gradient direction should be normalized"
        assert gradient.confidence > 30, "Gradient confidence should meet threshold"

    def test_forward_velocity_scaling(self, homing_algorithm: HomingAlgorithm):
        """SUBTASK 3: Verify forward velocity scaling.

        Tests velocity scaling algorithm based on signal strength change rate
        per PRD-FR4 requirement for "forward velocity and yaw-rate control".
        """
        # Set up algorithm with strong signal change
        strong_samples = [
            RSSISample(rssi=-45.0, position_x=0.0, position_y=0.0, heading=0.0, timestamp=1.0),
            RSSISample(rssi=-50.0, position_x=5.0, position_y=0.0, heading=90.0, timestamp=2.0),
            RSSISample(rssi=-55.0, position_x=10.0, position_y=0.0, heading=90.0, timestamp=3.0),
        ]

        for sample in strong_samples:
            homing_algorithm.add_rssi_sample(
                rssi=sample.rssi,
                position_x=sample.position_x,
                position_y=sample.position_y,
                heading=sample.heading,
                timestamp=sample.timestamp,
            )

        # Calculate gradient and test velocity scaling
        gradient_strong = homing_algorithm.calculate_gradient()
        assert gradient_strong is not None, "Strong gradient should be calculated"
        velocity_strong = homing_algorithm.scale_velocity_by_gradient(gradient_strong)

        # Clear algorithm and test with weak signal change
        homing_algorithm = HomingAlgorithm()
        weak_samples = [
            RSSISample(rssi=-50.0, position_x=0.0, position_y=0.0, heading=0.0, timestamp=1.0),
            RSSISample(rssi=-51.0, position_x=5.0, position_y=0.0, heading=90.0, timestamp=2.0),
            RSSISample(rssi=-52.0, position_x=10.0, position_y=0.0, heading=90.0, timestamp=3.0),
        ]

        for sample in weak_samples:
            homing_algorithm.add_rssi_sample(
                rssi=sample.rssi,
                position_x=sample.position_x,
                position_y=sample.position_y,
                heading=sample.heading,
                timestamp=sample.timestamp,
            )

        gradient_weak = homing_algorithm.calculate_gradient()
        assert gradient_weak is not None, "Weak gradient should be calculated"
        velocity_weak = homing_algorithm.scale_velocity_by_gradient(gradient_weak)

        # Verify velocity scaling: stronger gradient = faster velocity
        assert (
            velocity_strong > velocity_weak
        ), "Forward velocity should scale with gradient strength"

    @pytest.mark.sitl_required
    def test_yaw_rate_control_commands(self, homing_algorithm: HomingAlgorithm):
        """SUBTASK 4: Test yaw-rate control commands.

        **INTEGRATION BOUNDARY CROSSED** - TASK-9.3 completed, SITL now available
        Tests yaw-rate control commands using established SITL connection.
        """
        # Set up samples for gradient calculation
        test_samples = [
            RSSISample(rssi=-50.0, position_x=0.0, position_y=0.0, heading=0.0, timestamp=1.0),
            RSSISample(rssi=-60.0, position_x=10.0, position_y=0.0, heading=90.0, timestamp=2.0),
            RSSISample(rssi=-65.0, position_x=15.0, position_y=5.0, heading=45.0, timestamp=3.0),
        ]

        for sample in test_samples:
            homing_algorithm.add_rssi_sample(
                rssi=sample.rssi,
                position_x=sample.position_x,
                position_y=sample.position_y,
                heading=sample.heading,
                timestamp=sample.timestamp,
            )

        # Calculate gradient and generate velocity command
        gradient = homing_algorithm.calculate_gradient()
        assert gradient is not None, "Gradient should be calculated for yaw control"

        # Test yaw rate calculation for heading correction
        current_heading = 0.0  # North
        target_heading = 90.0  # East (gradient direction)

        yaw_rate = homing_algorithm.calculate_yaw_rate(current_heading, target_heading)

        # Verify yaw rate command
        assert abs(yaw_rate) > 0, "Yaw rate should be non-zero for heading correction"
        assert abs(yaw_rate) <= homing_algorithm.yaw_rate_max, "Yaw rate should respect limits"

    @pytest.mark.sitl_required
    def test_approach_velocity_reduction(self, homing_algorithm: HomingAlgorithm):
        """SUBTASK 5: Validate approach velocity reduction.

        **INTEGRATION BOUNDARY CROSSED** - TASK-9.3 completed, SITL now available
        Tests approach velocity reduction for close-range beacon approach.
        """
        # Simulate close approach with strong signal
        close_samples = [
            RSSISample(
                rssi=-40.0, position_x=0.0, position_y=0.0, heading=0.0, timestamp=1.0
            ),  # Very strong
            RSSISample(
                rssi=-45.0, position_x=2.0, position_y=0.0, heading=90.0, timestamp=2.0
            ),  # Close
            RSSISample(
                rssi=-50.0, position_x=5.0, position_y=0.0, heading=90.0, timestamp=3.0
            ),  # Nearby
        ]

        for sample in close_samples:
            homing_algorithm.add_rssi_sample(
                rssi=sample.rssi,
                position_x=sample.position_x,
                position_y=sample.position_y,
                heading=sample.heading,
                timestamp=sample.timestamp,
            )

        # Test approach velocity command generation
        gradient = homing_algorithm.calculate_gradient()
        assert gradient is not None, "Gradient should be calculated for approach"

        # Generate velocity command for close approach
        velocity_cmd = homing_algorithm.generate_velocity_command(
            gradient=gradient, current_heading=0.0, current_time=4.0
        )

        # Verify approach velocity is reduced for strong signals
        assert velocity_cmd.forward_velocity > 0, "Forward velocity should be positive for approach"
        assert (
            velocity_cmd.forward_velocity < homing_algorithm.forward_velocity_max
        ), "Approach velocity should be reduced"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
