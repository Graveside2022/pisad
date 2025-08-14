#!/usr/bin/env python3
"""Field test script for approach accuracy validation.

Executes approach tests from specified distances and measures
final position accuracy relative to beacon location.
"""

import asyncio
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.backend.models.schemas import BeaconConfiguration
from src.backend.services.beacon_simulator import BeaconSimulator
from src.backend.services.field_test_service import FieldTestService
from src.backend.services.homing_controller import HomingController
from src.backend.services.mavlink_service import MAVLinkService
from src.backend.services.signal_processor import SignalProcessor
from src.backend.services.state_machine import StateMachine
from src.backend.utils.logging import get_logger
from src.backend.utils.safety import SafetyInterlockSystem
from src.backend.utils.test_logger import TestLogger

logger = get_logger(__name__)


class ApproachAccuracyTest:
    """Approach accuracy field test executor."""

    def __init__(self):
        """Initialize test executor."""
        self.test_logger = TestLogger("data/field_test_results.db")
        self.state_machine = StateMachine()
        self.mavlink = MAVLinkService()
        self.signal_processor = SignalProcessor()
        self.safety_manager = SafetyInterlockSystem()
        self.homing_controller = HomingController(
            mavlink_service=self.mavlink,
            signal_processor=self.signal_processor,
        )

        self.field_test_service = FieldTestService(
            test_logger=self.test_logger,
            state_machine=self.state_machine,
            mavlink_service=self.mavlink,
            signal_processor=self.signal_processor,
            safety_manager=self.safety_manager,
        )

        self.beacon_simulator = BeaconSimulator()
        self.test_results = []
        self.waypoints = []
        self.beacon_position = None

    async def setup(self):
        """Initialize services and connections."""
        logger.info("Setting up approach accuracy test")

        # Initialize services
        await self.mavlink.connect()
        await self.signal_processor.initialize()
        await self.state_machine.initialize()
        await self.homing_controller.initialize()

        # Validate preflight
        checklist = await self.field_test_service.validate_preflight_checklist()
        if not all(checklist.values()):
            failed = [k for k, v in checklist.items() if not v]
            raise RuntimeError(f"Preflight checks failed: {failed}")

        logger.info("Setup complete")

    async def configure_test_waypoints(
        self,
        beacon_position: tuple[float, float, float],
        start_distance_m: float,
        approach_directions: list[float],
    ) -> list[dict[str, float]]:
        """Configure waypoints for consistent approach patterns.

        Args:
            beacon_position: Beacon GPS position (lat, lon, alt)
            start_distance_m: Starting distance from beacon
            approach_directions: List of approach headings in degrees

        Returns:
            List of starting waypoints
        """
        waypoints = []

        for heading_deg in approach_directions:
            # Calculate starting position
            # Simplified calculation - in real implementation use proper GPS math
            heading_rad = math.radians(heading_deg)

            # Approximate meters to degrees (very rough)
            meters_per_degree = 111000  # At equator

            lat_offset = (start_distance_m * math.cos(heading_rad)) / meters_per_degree
            lon_offset = (start_distance_m * math.sin(heading_rad)) / meters_per_degree

            waypoint = {
                "latitude": beacon_position[0] - lat_offset,
                "longitude": beacon_position[1] - lon_offset,
                "altitude": beacon_position[2] + 50,  # 50m above beacon
                "heading": heading_deg,
            }
            waypoints.append(waypoint)

        logger.info(f"Configured {len(waypoints)} approach waypoints")
        return waypoints

    async def execute_approach_test(
        self,
        start_waypoint: dict[str, float],
        beacon_config: BeaconConfiguration,
        target_radius_m: float = 50.0,
    ) -> dict[str, Any]:
        """Execute single approach test.

        Args:
            start_waypoint: Starting position and heading
            beacon_config: Beacon configuration
            target_radius_m: Target approach radius in meters

        Returns:
            Approach test results
        """
        results = {
            "start_position": start_waypoint,
            "target_radius_m": target_radius_m,
            "approach_time_s": 0.0,
            "final_position": None,
            "final_distance_m": 0.0,
            "approach_error_m": 0.0,
            "max_rssi_dbm": -120.0,
            "trajectory": [],
            "success": False,
        }

        try:
            # Navigate to starting position
            logger.info(f"Moving to start position: {start_waypoint}")
            await self.mavlink.goto_position(
                start_waypoint["latitude"],
                start_waypoint["longitude"],
                start_waypoint["altitude"],
            )

            # Wait for arrival at start position
            await self.wait_for_position(start_waypoint, tolerance_m=10)

            # Start beacon
            beacon = self.beacon_simulator.create_beacon(
                beacon_id="approach_test_beacon",
                config=beacon_config,
                position=self.beacon_position,
            )
            await self.beacon_simulator.start_beacon(beacon.beacon_id)

            # Record start time
            start_time = datetime.now(UTC)

            # Transition to HOMING state
            logger.info("Starting homing approach")
            await self.state_machine.request_transition("DETECTING")
            await asyncio.sleep(2)  # Allow detection
            await self.state_machine.request_transition("HOMING")

            # Monitor approach
            max_rssi = -120.0
            trajectory = []

            while self.state_machine.current_state == "HOMING":
                # Get current position
                telemetry = await self.mavlink.get_telemetry()
                if telemetry:
                    current_pos = {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "latitude": telemetry.get("latitude", 0.0),
                        "longitude": telemetry.get("longitude", 0.0),
                        "altitude": telemetry.get("altitude", 0.0),
                        "rssi": self.signal_processor.current_rssi,
                    }
                    trajectory.append(current_pos)

                    # Track max RSSI
                    max_rssi = max(max_rssi, self.signal_processor.current_rssi)

                    # Check if within target radius
                    distance = self.calculate_distance(
                        (
                            current_pos["latitude"],
                            current_pos["longitude"],
                            current_pos["altitude"],
                        ),
                        self.beacon_position,
                    )

                    if distance <= target_radius_m:
                        logger.info(f"Reached target radius: {distance:.1f}m")
                        results["success"] = True
                        break

                await asyncio.sleep(0.5)

            # Record approach time
            approach_time = (datetime.now(UTC) - start_time).total_seconds()
            results["approach_time_s"] = approach_time

            # Get final position
            final_telemetry = await self.mavlink.get_telemetry()
            if final_telemetry:
                results["final_position"] = {
                    "latitude": final_telemetry.get("latitude", 0.0),
                    "longitude": final_telemetry.get("longitude", 0.0),
                    "altitude": final_telemetry.get("altitude", 0.0),
                }

                # Calculate final distance and error
                final_distance = self.calculate_distance(
                    (
                        results["final_position"]["latitude"],
                        results["final_position"]["longitude"],
                        results["final_position"]["altitude"],
                    ),
                    self.beacon_position,
                )
                results["final_distance_m"] = final_distance
                results["approach_error_m"] = max(0, final_distance - target_radius_m)

            results["max_rssi_dbm"] = max_rssi
            results["trajectory"] = trajectory

            # Stop beacon
            await self.beacon_simulator.stop_beacon(beacon.beacon_id)

            # Return to IDLE
            await self.state_machine.request_transition("IDLE")

            logger.info(
                f"Approach complete: time={approach_time:.1f}s, "
                f"final_distance={results['final_distance_m']:.1f}m, "
                f"error={results['approach_error_m']:.1f}m"
            )

        except Exception as e:
            logger.error(f"Approach test failed: {e}")
            results["error"] = str(e)

        return results

    async def wait_for_position(
        self,
        target_position: dict[str, float],
        tolerance_m: float = 10.0,
        timeout_s: float = 120.0,
    ):
        """Wait for drone to reach target position.

        Args:
            target_position: Target GPS position
            tolerance_m: Position tolerance in meters
            timeout_s: Maximum wait time
        """
        start_time = datetime.now(UTC)

        while (datetime.now(UTC) - start_time).total_seconds() < timeout_s:
            telemetry = await self.mavlink.get_telemetry()
            if telemetry:
                current_pos = (
                    telemetry.get("latitude", 0.0),
                    telemetry.get("longitude", 0.0),
                    telemetry.get("altitude", 0.0),
                )
                target_pos = (
                    target_position["latitude"],
                    target_position["longitude"],
                    target_position["altitude"],
                )

                distance = self.calculate_distance(current_pos, target_pos)
                if distance <= tolerance_m:
                    logger.info(f"Reached position within {distance:.1f}m")
                    return

            await asyncio.sleep(1)

        raise TimeoutError(f"Failed to reach position within {timeout_s}s")

    def calculate_distance(
        self,
        pos1: tuple[float, float, float],
        pos2: tuple[float, float, float],
    ) -> float:
        """Calculate distance between two GPS positions.

        Args:
            pos1: First position (lat, lon, alt)
            pos2: Second position (lat, lon, alt)

        Returns:
            Distance in meters
        """
        # Simplified calculation - use haversine formula in production
        lat1, lon1, alt1 = pos1
        lat2, lon2, alt2 = pos2

        # Approximate calculation
        meters_per_degree = 111000
        dlat = (lat2 - lat1) * meters_per_degree
        dlon = (lon2 - lon1) * meters_per_degree * math.cos(math.radians(lat1))
        dalt = alt2 - alt1

        return math.sqrt(dlat**2 + dlon**2 + dalt**2)

    async def run_approach_tests(
        self,
        repetitions: int = 5,
        start_distance_m: float = 500,
        target_radius_m: float = 50,
    ) -> list[dict[str, Any]]:
        """Run multiple approach accuracy tests.

        Args:
            repetitions: Number of test repetitions
            start_distance_m: Starting distance from beacon
            target_radius_m: Target approach radius

        Returns:
            List of test results
        """
        results = []

        # Set beacon position (would be actual GPS in field)
        self.beacon_position = (37.7749, -122.4194, 100.0)  # Example: San Francisco

        # Configure beacon
        beacon_config = BeaconConfiguration(
            frequency_hz=433_000_000,
            power_dbm=15.0,
            modulation="LoRa",
            spreading_factor=7,
            bandwidth_hz=125_000,
            coding_rate=5,
            pulse_rate_hz=1.0,
            pulse_duration_ms=100,
        )

        # Configure approach waypoints from different directions
        approach_headings = [0, 90, 180, 270, 45]  # N, E, S, W, NE
        waypoints = await self.configure_test_waypoints(
            self.beacon_position,
            start_distance_m,
            approach_headings[:repetitions],
        )

        # Execute tests
        for i, waypoint in enumerate(waypoints):
            logger.info(f"Approach test {i+1}/{repetitions}")

            result = await self.execute_approach_test(
                start_waypoint=waypoint,
                beacon_config=beacon_config,
                target_radius_m=target_radius_m,
            )

            results.append(result)

            # Save intermediate results
            await self.save_results(results)

            # Pause between tests
            await asyncio.sleep(10)

        # Calculate statistics
        await self.calculate_statistics(results)

        return results

    async def calculate_statistics(self, results: list[dict[str, Any]]):
        """Calculate approach accuracy statistics.

        Args:
            results: List of approach test results
        """
        errors = [r["approach_error_m"] for r in results if "approach_error_m" in r]
        times = [r["approach_time_s"] for r in results if "approach_time_s" in r]
        distances = [r["final_distance_m"] for r in results if "final_distance_m" in r]

        stats = {
            "total_tests": len(results),
            "successful_approaches": sum(1 for r in results if r.get("success", False)),
            "success_rate": sum(1 for r in results if r.get("success", False)) / len(results),
            "mean_error_m": np.mean(errors) if errors else 0,
            "std_error_m": np.std(errors) if errors else 0,
            "max_error_m": max(errors) if errors else 0,
            "min_error_m": min(errors) if errors else 0,
            "mean_time_s": np.mean(times) if times else 0,
            "mean_final_distance_m": np.mean(distances) if distances else 0,
        }

        # Validate 50m radius achievement
        within_50m = sum(1 for d in distances if d <= 50)
        stats["50m_achievement_rate"] = within_50m / len(distances) if distances else 0

        logger.info(
            f"Approach Statistics:\n"
            f"  Success rate: {stats['success_rate']*100:.1f}%\n"
            f"  Mean error: {stats['mean_error_m']:.1f}m ± {stats['std_error_m']:.1f}m\n"
            f"  Mean time: {stats['mean_time_s']:.1f}s\n"
            f"  50m achievement: {stats['50m_achievement_rate']*100:.1f}%"
        )

        # Save statistics
        stats_file = Path("data/field_tests/approach_accuracy/statistics.json")
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

    async def save_results(self, results: list[dict[str, Any]]):
        """Save test results to file.

        Args:
            results: List of test results
        """
        results_dir = Path("data/field_tests/approach_accuracy")
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"approach_accuracy_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump(
                {
                    "test_type": "approach_accuracy",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "results": results,
                },
                f,
                indent=2,
                default=str,
            )

        logger.info(f"Results saved to {results_file}")

    async def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up")

        self.beacon_simulator.clear_all_beacons()
        await self.mavlink.disconnect()
        await self.signal_processor.shutdown()
        await self.homing_controller.shutdown()

        logger.info("Cleanup complete")


@click.command()
@click.option("--repetitions", "-r", default=5, help="Number of approach tests")
@click.option("--start-distance", "-d", default=500, help="Starting distance in meters")
@click.option("--target-radius", "-t", default=50, help="Target approach radius in meters")
def main(repetitions: int, start_distance: float, target_radius: float):
    """Execute approach accuracy field tests."""
    logger.info(
        f"Starting approach accuracy tests:\n"
        f"  Repetitions: {repetitions}\n"
        f"  Start distance: {start_distance}m\n"
        f"  Target radius: {target_radius}m"
    )

    test_executor = ApproachAccuracyTest()

    async def run_tests():
        """Async test execution."""
        try:
            await test_executor.setup()

            results = await test_executor.run_approach_tests(
                repetitions=repetitions,
                start_distance_m=start_distance,
                target_radius_m=target_radius,
            )

            logger.info(f"Completed {len(results)} approach tests")

        finally:
            await test_executor.cleanup()

    asyncio.run(run_tests())


if __name__ == "__main__":
    main()
