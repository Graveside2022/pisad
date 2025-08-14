#!/usr/bin/env python3
"""Field test script for state transition performance validation.

Measures transition latencies between system states during field operations.
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
from src.backend.services.mavlink_service import MAVLinkService
from src.backend.services.signal_processor import SignalProcessor
from src.backend.services.state_machine import StateMachine, SystemState
from src.backend.utils.logging import get_logger

logger = get_logger(__name__)


class StateTransitionTest:
    """State transition performance test executor."""

    def __init__(self):
        """Initialize test executor."""
        self.state_machine = StateMachine()
        self.mavlink = MAVLinkService()
        self.signal_processor = SignalProcessor()
        self.beacon_simulator = BeaconSimulator()

        self.transition_times = []
        self.test_results = []

    async def setup(self):
        """Initialize services."""
        logger.info("Setting up state transition test")

        await self.mavlink.connect()
        await self.signal_processor.initialize()
        await self.state_machine.initialize()

        # Register state change callback
        self.state_machine.add_transition_callback(self.on_state_change)

        logger.info("Setup complete")

    def on_state_change(self, old_state: SystemState, new_state: SystemState):
        """Callback for state transitions.

        Args:
            old_state: Previous state
            new_state: New state
        """
        transition = {
            "timestamp": datetime.now(UTC),
            "from_state": old_state,
            "to_state": new_state,
        }
        self.transition_times.append(transition)
        logger.info(f"State transition: {old_state} -> {new_state}")

    async def measure_transition(
        self,
        from_state: str,
        to_state: str,
        trigger_action: callable | None = None,
    ) -> float:
        """Measure single state transition time.

        Args:
            from_state: Starting state
            to_state: Target state
            trigger_action: Optional action to trigger transition

        Returns:
            Transition time in milliseconds
        """
        # Ensure we're in the starting state
        if self.state_machine.current_state != from_state:
            await self.state_machine.request_transition(from_state)
            await asyncio.sleep(0.5)

        # Clear transition log
        self.transition_times.clear()

        # Record start time
        start_time = datetime.now(UTC)

        # Trigger transition
        if trigger_action:
            await trigger_action()
        else:
            await self.state_machine.request_transition(to_state)

        # Wait for transition to complete
        max_wait = 10.0  # seconds
        elapsed = 0.0
        while self.state_machine.current_state != to_state and elapsed < max_wait:
            await asyncio.sleep(0.01)  # 10ms resolution
            elapsed = (datetime.now(UTC) - start_time).total_seconds()

        if self.state_machine.current_state != to_state:
            logger.error(f"Transition to {to_state} failed (timeout)")
            return -1

        # Calculate transition time
        transition_time_ms = elapsed * 1000
        logger.info(f"Transition {from_state} -> {to_state}: {transition_time_ms:.1f}ms")

        return transition_time_ms

    async def test_searching_to_detecting(self, repetitions: int = 10) -> dict[str, Any]:
        """Test SEARCHING to DETECTING transition.

        Args:
            repetitions: Number of test repetitions

        Returns:
            Test results
        """
        logger.info("Testing SEARCHING -> DETECTING transition")

        results = {
            "transition": "SEARCHING_TO_DETECTING",
            "measurements": [],
            "mean_ms": 0,
            "std_ms": 0,
            "max_ms": 0,
            "min_ms": 0,
        }

        # Create beacon for detection
        beacon_config = BeaconConfiguration(
            frequency_hz=433_000_000,
            power_dbm=15.0,
        )
        beacon = self.beacon_simulator.create_beacon(
            "test_beacon",
            beacon_config,
            (100, 0, 0),
        )

        for i in range(repetitions):
            logger.info(f"Iteration {i+1}/{repetitions}")

            # Start in SEARCHING
            await self.state_machine.request_transition("SEARCHING")
            await asyncio.sleep(1)

            # Trigger detection by starting beacon
            async def trigger_detection():
                await self.beacon_simulator.start_beacon(beacon.beacon_id)
                # Simulate detection
                self.signal_processor.current_rssi = -75.0
                self.signal_processor.beacon_detected = True

            # Measure transition
            time_ms = await self.measure_transition(
                "SEARCHING",
                "DETECTING",
                trigger_detection,
            )

            if time_ms > 0:
                results["measurements"].append(time_ms)

            # Stop beacon
            await self.beacon_simulator.stop_beacon(beacon.beacon_id)
            self.signal_processor.beacon_detected = False

            await asyncio.sleep(2)

        # Calculate statistics
        if results["measurements"]:
            results["mean_ms"] = np.mean(results["measurements"])
            results["std_ms"] = np.std(results["measurements"])
            results["max_ms"] = max(results["measurements"])
            results["min_ms"] = min(results["measurements"])

        return results

    async def test_detecting_to_homing(self, repetitions: int = 10) -> dict[str, Any]:
        """Test DETECTING to HOMING transition.

        Args:
            repetitions: Number of test repetitions

        Returns:
            Test results
        """
        logger.info("Testing DETECTING -> HOMING transition")

        results = {
            "transition": "DETECTING_TO_HOMING",
            "measurements": [],
            "mean_ms": 0,
            "std_ms": 0,
            "max_ms": 0,
            "min_ms": 0,
        }

        for i in range(repetitions):
            logger.info(f"Iteration {i+1}/{repetitions}")

            # Start in DETECTING
            await self.state_machine.request_transition("DETECTING")
            # Ensure detection conditions
            self.signal_processor.beacon_detected = True
            self.signal_processor.current_rssi = -75.0
            await asyncio.sleep(1)

            # Measure transition
            time_ms = await self.measure_transition("DETECTING", "HOMING")

            if time_ms > 0:
                results["measurements"].append(time_ms)

            await asyncio.sleep(2)

        # Calculate statistics
        if results["measurements"]:
            results["mean_ms"] = np.mean(results["measurements"])
            results["std_ms"] = np.std(results["measurements"])
            results["max_ms"] = max(results["measurements"])
            results["min_ms"] = min(results["measurements"])

        return results

    async def test_all_transitions(self, repetitions: int = 10) -> list[dict[str, Any]]:
        """Test all critical state transitions.

        Args:
            repetitions: Number of repetitions per transition

        Returns:
            List of all test results
        """
        all_results = []

        # Test SEARCHING -> DETECTING
        result = await self.test_searching_to_detecting(repetitions)
        all_results.append(result)

        # Test DETECTING -> HOMING
        result = await self.test_detecting_to_homing(repetitions)
        all_results.append(result)

        # Test HOMING -> SUCCESS
        result = await self.test_transition("HOMING", "SUCCESS", repetitions)
        all_results.append(result)

        # Test any -> ERROR
        result = await self.test_transition("HOMING", "ERROR", repetitions)
        all_results.append(result)

        # Test ERROR -> IDLE (recovery)
        result = await self.test_transition("ERROR", "IDLE", repetitions)
        all_results.append(result)

        return all_results

    async def test_transition(
        self,
        from_state: str,
        to_state: str,
        repetitions: int,
    ) -> dict[str, Any]:
        """Generic transition test.

        Args:
            from_state: Starting state
            to_state: Target state
            repetitions: Number of repetitions

        Returns:
            Test results
        """
        logger.info(f"Testing {from_state} -> {to_state} transition")

        results = {
            "transition": f"{from_state}_TO_{to_state}",
            "measurements": [],
            "mean_ms": 0,
            "std_ms": 0,
            "max_ms": 0,
            "min_ms": 0,
        }

        for i in range(repetitions):
            logger.info(f"Iteration {i+1}/{repetitions}")

            time_ms = await self.measure_transition(from_state, to_state)

            if time_ms > 0:
                results["measurements"].append(time_ms)

            await asyncio.sleep(1)

        # Calculate statistics
        if results["measurements"]:
            results["mean_ms"] = np.mean(results["measurements"])
            results["std_ms"] = np.std(results["measurements"])
            results["max_ms"] = max(results["measurements"])
            results["min_ms"] = min(results["measurements"])

        return results

    async def validate_transition_latency(
        self,
        results: list[dict[str, Any]],
        max_latency_ms: float = 2000,
    ) -> dict[str, bool]:
        """Validate transition latencies against requirement.

        Args:
            results: List of test results
            max_latency_ms: Maximum allowed latency

        Returns:
            Validation results
        """
        validation = {}

        for result in results:
            transition = result["transition"]
            mean_latency = result["mean_ms"]

            passed = mean_latency > 0 and mean_latency < max_latency_ms
            validation[transition] = passed

            if passed:
                logger.info(f"✓ {transition}: {mean_latency:.1f}ms < {max_latency_ms}ms")
            else:
                logger.error(f"✗ {transition}: {mean_latency:.1f}ms >= {max_latency_ms}ms")

        return validation

    async def save_results(self, results: list[dict[str, Any]]):
        """Save test results.

        Args:
            results: List of test results
        """
        results_dir = Path("data/field_tests/state_transitions")
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"state_transitions_{timestamp}.json"

        # Calculate total transition latency
        total_latency = sum(r["mean_ms"] for r in results if r["mean_ms"] > 0)

        with open(results_file, "w") as f:
            json.dump(
                {
                    "test_type": "state_transitions",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "total_latency_ms": total_latency,
                    "results": results,
                },
                f,
                indent=2,
                default=str,
            )

        logger.info(f"Results saved to {results_file}")

    async def cleanup(self):
        """Clean up resources."""
        self.beacon_simulator.clear_all_beacons()
        await self.mavlink.disconnect()
        await self.signal_processor.shutdown()


@click.command()
@click.option("--repetitions", "-r", default=10, help="Repetitions per transition")
@click.option("--max-latency", "-m", default=2000, help="Max allowed latency in ms")
def main(repetitions: int, max_latency: float):
    """Execute state transition performance tests."""
    logger.info(
        f"Starting state transition tests:\n"
        f"  Repetitions: {repetitions}\n"
        f"  Max latency: {max_latency}ms"
    )

    test_executor = StateTransitionTest()

    async def run_tests():
        """Async test execution."""
        try:
            await test_executor.setup()

            # Run all transition tests
            results = await test_executor.test_all_transitions(repetitions)

            # Validate latencies
            validation = await test_executor.validate_transition_latency(
                results,
                max_latency,
            )

            # Save results
            await test_executor.save_results(results)

            # Summary
            passed = sum(validation.values())
            total = len(validation)
            logger.info(f"Validation: {passed}/{total} transitions passed")

            if all(validation.values()):
                logger.info("✓ All transitions meet latency requirements")
            else:
                failed = [k for k, v in validation.items() if not v]
                logger.error(f"✗ Failed transitions: {failed}")

        finally:
            await test_executor.cleanup()

    asyncio.run(run_tests())


if __name__ == "__main__":
    main()
