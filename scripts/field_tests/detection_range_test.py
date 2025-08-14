#!/usr/bin/env python3
"""Field test script for detection range validation.

Executes open field detection tests at various distances and power levels,
recording RSSI values and environmental conditions.
"""

import asyncio
import json
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
from src.backend.services.mavlink_service import MAVLinkService
from src.backend.services.signal_processor import SignalProcessor
from src.backend.services.state_machine import StateMachine
from src.backend.utils.logging import get_logger
from src.backend.utils.safety import SafetyInterlockSystem
from src.backend.utils.test_logger import TestLogger

logger = get_logger(__name__)


class DetectionRangeTest:
    """Detection range field test executor."""

    def __init__(self):
        """Initialize test executor."""
        self.test_logger = TestLogger("data/field_test_results.db")
        self.state_machine = StateMachine()
        self.mavlink = MAVLinkService()
        self.signal_processor = SignalProcessor()
        self.safety_manager = SafetyInterlockSystem()

        self.field_test_service = FieldTestService(
            test_logger=self.test_logger,
            state_machine=self.state_machine,
            mavlink_service=self.mavlink,
            signal_processor=self.signal_processor,
            safety_manager=self.safety_manager,
        )

        self.beacon_simulator = BeaconSimulator()
        self.test_results = []
        self.environmental_data = {}

    async def setup(self):
        """Initialize services and connections."""
        logger.info("Setting up detection range test")

        # Initialize MAVLink connection
        await self.mavlink.connect()

        # Initialize signal processor
        await self.signal_processor.initialize()

        # Initialize state machine
        await self.state_machine.initialize()

        # Validate preflight checklist
        checklist = await self.field_test_service.validate_preflight_checklist()
        if not all(checklist.values()):
            failed = [k for k, v in checklist.items() if not v]
            raise RuntimeError(f"Preflight checks failed: {failed}")

        logger.info("Setup complete, all systems ready")

    async def record_environmental_conditions(self) -> dict[str, Any]:
        """Record current environmental conditions."""
        conditions = {
            "timestamp": datetime.now(UTC).isoformat(),
            "temperature_c": 20.0,  # Would read from sensor
            "humidity_percent": 50.0,  # Would read from sensor
            "wind_speed_mps": 2.5,  # Would read from sensor
            "wind_direction_deg": 180.0,  # Would read from sensor
            "pressure_hpa": 1013.25,  # Would read from sensor
            "visibility_m": 10000,  # Would read from sensor
            "precipitation": "none",
            "cloud_cover": "partly_cloudy",
            "solar_radiation_wm2": 500.0,  # Would read from sensor
            "rf_noise_floor_dbm": -110.0,  # Measured baseline
        }

        # Get GPS location for test site
        telemetry = await self.mavlink.get_telemetry()
        if telemetry:
            conditions["test_location"] = {
                "latitude": telemetry.get("latitude", 0.0),
                "longitude": telemetry.get("longitude", 0.0),
                "altitude_m": telemetry.get("altitude", 0.0),
            }

        logger.info(f"Environmental conditions: {conditions}")
        return conditions

    async def execute_detection_test(
        self,
        distance_m: float,
        beacon_power_dbm: float,
        repetitions: int = 5,
    ) -> dict[str, Any]:
        """Execute detection test at specific distance and power level.

        Args:
            distance_m: Test distance in meters
            beacon_power_dbm: Beacon transmission power in dBm
            repetitions: Number of test repetitions

        Returns:
            Test results dictionary
        """
        logger.info(f"Testing detection at {distance_m}m with {beacon_power_dbm}dBm")

        results = {
            "distance_m": distance_m,
            "beacon_power_dbm": beacon_power_dbm,
            "detections": [],
            "rssi_values": [],
            "snr_values": [],
            "detection_rate": 0.0,
            "avg_rssi_dbm": 0.0,
            "avg_snr_db": 0.0,
            "std_rssi_dbm": 0.0,
            "std_snr_db": 0.0,
        }

        # Configure beacon
        beacon_config = BeaconConfiguration(
            frequency_hz=433_000_000,
            power_dbm=beacon_power_dbm,
            modulation="LoRa",
            spreading_factor=7,
            bandwidth_hz=125_000,
            coding_rate=5,
            pulse_rate_hz=1.0,
            pulse_duration_ms=100,
        )

        # Create simulated beacon at distance
        beacon = self.beacon_simulator.create_beacon(
            beacon_id=f"test_beacon_{distance_m}m_{beacon_power_dbm}dbm",
            config=beacon_config,
            position=(distance_m, 0, 0),  # Simple linear distance
        )

        # Start beacon transmission
        await self.beacon_simulator.start_beacon(beacon.beacon_id)

        # Execute repetitions
        for i in range(repetitions):
            logger.info(f"Detection test iteration {i+1}/{repetitions}")

            # Reset signal processor
            self.signal_processor.reset_detection()

            # Wait for detection (timeout 30s)
            detection_start = datetime.now(UTC)
            detected = False
            rssi = -120.0
            snr = 0.0

            timeout = 30.0
            while (datetime.now(UTC) - detection_start).total_seconds() < timeout:
                # Check for detection
                if self.signal_processor.beacon_detected:
                    detected = True
                    rssi = self.signal_processor.current_rssi
                    snr = self.signal_processor.current_snr
                    detection_time = (datetime.now(UTC) - detection_start).total_seconds()
                    logger.info(
                        f"Beacon detected in {detection_time:.2f}s: RSSI={rssi:.1f}dBm, SNR={snr:.1f}dB"
                    )
                    break
                await asyncio.sleep(0.1)

            results["detections"].append(detected)
            if detected:
                results["rssi_values"].append(rssi)
                results["snr_values"].append(snr)

            # Brief pause between iterations
            await asyncio.sleep(2)

        # Stop beacon
        await self.beacon_simulator.stop_beacon(beacon.beacon_id)

        # Calculate statistics with proper edge case handling
        detection_count = sum(results["detections"])
        results["detection_rate"] = detection_count / repetitions if repetitions > 0 else 0.0

        if results["rssi_values"]:
            results["avg_rssi_dbm"] = float(np.mean(results["rssi_values"]))
            results["std_rssi_dbm"] = float(np.std(results["rssi_values"]))
            results["min_rssi_dbm"] = float(np.min(results["rssi_values"]))
            results["max_rssi_dbm"] = float(np.max(results["rssi_values"]))
        else:
            results["avg_rssi_dbm"] = -120.0
            results["std_rssi_dbm"] = 0.0
            results["min_rssi_dbm"] = -120.0
            results["max_rssi_dbm"] = -120.0

        if results["snr_values"]:
            results["avg_snr_db"] = float(np.mean(results["snr_values"]))
            results["std_snr_db"] = float(np.std(results["snr_values"]))
        else:
            results["avg_snr_db"] = 0.0
            results["std_snr_db"] = 0.0

        logger.info(
            f"Detection rate: {results['detection_rate']*100:.1f}%, "
            f"Avg RSSI: {results['avg_rssi_dbm']:.1f}dBm, "
            f"Avg SNR: {results['avg_snr_db']:.1f}dB"
        )

        return results

    async def run_test_matrix(
        self,
        distances: list[float],
        power_levels: list[float],
        repetitions: int = 5,
    ) -> list[dict[str, Any]]:
        """Run complete test matrix of distances and power levels.

        Args:
            distances: List of test distances in meters
            power_levels: List of beacon power levels in dBm
            repetitions: Number of repetitions per test point

        Returns:
            List of all test results
        """
        all_results = []

        # Record initial environmental conditions
        self.environmental_data = await self.record_environmental_conditions()

        total_tests = len(distances) * len(power_levels)
        test_num = 0

        for power_dbm in power_levels:
            for distance_m in distances:
                test_num += 1
                logger.info(f"Test {test_num}/{total_tests}: {distance_m}m @ {power_dbm}dBm")

                # Execute detection test
                result = await self.execute_detection_test(
                    distance_m=distance_m,
                    beacon_power_dbm=power_dbm,
                    repetitions=repetitions,
                )

                # Add environmental data
                result["environmental_conditions"] = self.environmental_data

                all_results.append(result)

                # Save intermediate results
                await self.save_results(all_results)

                # Pause between test points
                await asyncio.sleep(5)

        return all_results

    async def save_results(self, results: list[dict[str, Any]]):
        """Save test results to file.

        Args:
            results: List of test results
        """
        # Create results directory
        results_dir = Path("data/field_tests/detection_range")
        results_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"detection_range_{timestamp}.json"

        # Save results
        with open(results_file, "w") as f:
            json.dump(
                {
                    "test_type": "detection_range",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "environmental_conditions": self.environmental_data,
                    "results": results,
                },
                f,
                indent=2,
                default=str,
            )

        logger.info(f"Results saved to {results_file}")

    async def generate_performance_curves(self, results: list[dict[str, Any]]):
        """Generate detection range vs power level performance curves.

        Args:
            results: List of test results
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available, skipping performance curves")
            return

        # Organize data by power level
        power_curves = {}
        for result in results:
            power = result["beacon_power_dbm"]
            if power not in power_curves:
                power_curves[power] = {"distances": [], "detection_rates": [], "rssi": []}

            power_curves[power]["distances"].append(result["distance_m"])
            power_curves[power]["detection_rates"].append(result["detection_rate"])
            power_curves[power]["rssi"].append(result["avg_rssi_dbm"])

        # Create detection rate plot
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        for power, data in power_curves.items():
            plt.plot(data["distances"], data["detection_rates"], marker="o", label=f"{power} dBm")

        plt.xlabel("Distance (m)")
        plt.ylabel("Detection Rate")
        plt.title("Detection Rate vs Distance")
        plt.grid(True)
        plt.legend()
        plt.ylim(0, 1.1)

        # Create RSSI plot
        plt.subplot(2, 1, 2)
        for power, data in power_curves.items():
            plt.plot(data["distances"], data["rssi"], marker="s", label=f"{power} dBm")

        plt.xlabel("Distance (m)")
        plt.ylabel("Average RSSI (dBm)")
        plt.title("RSSI vs Distance")
        plt.grid(True)
        plt.legend()

        # Save plot
        plots_dir = Path("data/field_tests/detection_range/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        plot_file = plots_dir / f"performance_curves_{timestamp}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Performance curves saved to {plot_file}")

    async def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up test resources")

        # Clear beacons
        self.beacon_simulator.clear_all_beacons()

        # Disconnect services
        await self.mavlink.disconnect()
        await self.signal_processor.shutdown()

        logger.info("Cleanup complete")


@click.command()
@click.option(
    "--distances",
    "-d",
    default="100,250,500,750",
    help="Comma-separated test distances in meters",
)
@click.option(
    "--power-levels",
    "-p",
    default="5,10,15,20",
    help="Comma-separated beacon power levels in dBm",
)
@click.option(
    "--repetitions",
    "-r",
    default=5,
    type=int,
    help="Number of repetitions per test point",
)
@click.option(
    "--generate-plots",
    is_flag=True,
    help="Generate performance curve plots",
)
def main(distances: str, power_levels: str, repetitions: int, generate_plots: bool):
    """Execute detection range field tests."""
    # Parse distances and power levels
    test_distances = [float(d) for d in distances.split(",")]
    test_power_levels = [float(p) for p in power_levels.split(",")]

    logger.info(
        f"Starting detection range tests:\n"
        f"  Distances: {test_distances} m\n"
        f"  Power levels: {test_power_levels} dBm\n"
        f"  Repetitions: {repetitions}"
    )

    # Create test executor
    test_executor = DetectionRangeTest()

    async def run_tests():
        """Async test execution."""
        try:
            # Setup
            await test_executor.setup()

            # Run test matrix
            results = await test_executor.run_test_matrix(
                distances=test_distances,
                power_levels=test_power_levels,
                repetitions=repetitions,
            )

            # Generate plots if requested
            if generate_plots:
                await test_executor.generate_performance_curves(results)

            # Summary
            logger.info(f"Completed {len(results)} test points")

            # Find maximum detection range
            max_range = 0
            for result in results:
                if result["detection_rate"] >= 0.8:  # 80% threshold
                    max_range = max(max_range, result["distance_m"])

            logger.info(f"Maximum reliable detection range: {max_range} m")

        finally:
            await test_executor.cleanup()

    # Run async tests
    asyncio.run(run_tests())


if __name__ == "__main__":
    main()
