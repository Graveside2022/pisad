#!/usr/bin/env python3
"""Field test script for safety system validation.

Tests emergency stop, geofence enforcement, battery failsafe,
signal loss handling, and manual override capabilities.
"""

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.backend.models.schemas import BeaconConfiguration
from src.backend.services.beacon_simulator import BeaconSimulator
from src.backend.services.mavlink_service import MAVLinkService
from src.backend.services.signal_processor import SignalProcessor
from src.backend.services.state_machine import StateMachine
from src.backend.utils.logging import get_logger
from src.backend.utils.safety import SafetyInterlockSystem

logger = get_logger(__name__)


class SafetyValidationTest:
    """Safety system field validation test executor."""

    def __init__(self):
        """Initialize test executor."""
        self.state_machine = StateMachine()
        self.mavlink = MAVLinkService()
        self.signal_processor = SignalProcessor()
        self.safety_manager = SafetyInterlockSystem()
        self.beacon_simulator = BeaconSimulator()

        self.safety_events = []
        self.test_results = []

    async def setup(self):
        """Initialize services."""
        logger.info("Setting up safety validation test")

        await self.mavlink.connect()
        await self.signal_processor.initialize()
        await self.state_machine.initialize()
        self.safety_manager.initialize(self.mavlink)

        # Register safety callbacks
        self.safety_manager.register_callback(self.on_safety_event)

        logger.info("Setup complete")

    def on_safety_event(self, event_type: str, details: dict[str, Any]):
        """Callback for safety events.

        Args:
            event_type: Type of safety event
            details: Event details
        """
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "type": event_type,
            "details": details,
        }
        self.safety_events.append(event)
        logger.info(f"Safety event: {event_type} - {details}")

    async def test_emergency_stop(self) -> dict[str, Any]:
        """Test emergency stop during active homing.

        Returns:
            Test results
        """
        logger.info("Testing emergency stop")

        result = {
            "test": "emergency_stop",
            "initial_state": "",
            "final_state": "",
            "stop_time_ms": 0,
            "velocity_before": 0,
            "velocity_after": 0,
            "success": False,
        }

        try:
            # Create and start beacon
            beacon_config = BeaconConfiguration(
                frequency_hz=433_000_000,
                power_dbm=15.0,
            )
            beacon = self.beacon_simulator.create_beacon(
                "safety_test_beacon",
                beacon_config,
                (100, 0, 0),
            )
            await self.beacon_simulator.start_beacon(beacon.beacon_id)

            # Simulate detection and start homing
            self.signal_processor.beacon_detected = True
            self.signal_processor.current_rssi = -75.0
            await self.state_machine.request_transition("HOMING")
            await asyncio.sleep(2)  # Let homing start

            # Record initial state
            result["initial_state"] = self.state_machine.current_state
            telemetry = await self.mavlink.get_telemetry()
            if telemetry:
                result["velocity_before"] = telemetry.get("groundspeed", 0)

            # Trigger emergency stop
            self.safety_events.clear()
            start_time = datetime.now(UTC)

            await self.safety_manager.emergency_stop("Field test validation")

            # Wait for stop to complete
            await asyncio.sleep(1)

            # Record stop time
            stop_time = (datetime.now(UTC) - start_time).total_seconds() * 1000
            result["stop_time_ms"] = stop_time

            # Check final state
            result["final_state"] = self.state_machine.current_state
            telemetry = await self.mavlink.get_telemetry()
            if telemetry:
                result["velocity_after"] = telemetry.get("groundspeed", 0)

            # Validate success
            result["success"] = (
                result["final_state"] in ["IDLE", "ERROR"]
                and result["velocity_after"] < 0.5  # Essentially stopped
                and stop_time < 3000  # Within 3 seconds
            )

            # Stop beacon
            await self.beacon_simulator.stop_beacon(beacon.beacon_id)

            logger.info(f"Emergency stop: {result['success']} in {stop_time:.0f}ms")

        except Exception as e:
            logger.error(f"Emergency stop test failed: {e}")
            result["error"] = str(e)

        return result

    async def test_geofence_enforcement(self) -> dict[str, Any]:
        """Test geofence enforcement during homing.

        Returns:
            Test results
        """
        logger.info("Testing geofence enforcement")

        result = {
            "test": "geofence_enforcement",
            "geofence_configured": False,
            "breach_detected": False,
            "action_taken": "",
            "recovery_time_ms": 0,
            "success": False,
        }

        try:
            # Configure geofence
            telemetry = await self.mavlink.get_telemetry()
            if telemetry:
                current_pos = (
                    telemetry.get("latitude", 0),
                    telemetry.get("longitude", 0),
                )

                # Set geofence 200m around current position
                self.safety_manager.set_geofence(
                    center_lat=current_pos[0],
                    center_lon=current_pos[1],
                    radius_m=200,
                )
                result["geofence_configured"] = True

            # Simulate approaching geofence boundary
            self.safety_events.clear()

            # Command position outside geofence
            await self.mavlink.goto_position(
                current_pos[0] + 0.003,  # ~300m north
                current_pos[1],
                100,
            )

            # Wait for geofence response
            start_time = datetime.now(UTC)
            max_wait = 10.0

            while (datetime.now(UTC) - start_time).total_seconds() < max_wait:
                # Check for geofence events
                for event in self.safety_events:
                    if event["type"] == "geofence_breach":
                        result["breach_detected"] = True
                        result["action_taken"] = event["details"].get("action", "")
                        break

                if result["breach_detected"]:
                    break

                await asyncio.sleep(0.1)

            # Measure recovery time
            if result["breach_detected"]:
                recovery_start = datetime.now(UTC)

                # Wait for return to safe area
                while (datetime.now(UTC) - recovery_start).total_seconds() < 30:
                    telemetry = await self.mavlink.get_telemetry()
                    # Check if back within geofence
                    # Simplified check
                    if telemetry and self.state_machine.current_state == "IDLE":
                        result["recovery_time_ms"] = (
                            datetime.now(UTC) - recovery_start
                        ).total_seconds() * 1000
                        break
                    await asyncio.sleep(0.5)

            result["success"] = result["breach_detected"] and result["action_taken"] != ""

            logger.info(f"Geofence test: {result['success']}")

        except Exception as e:
            logger.error(f"Geofence test failed: {e}")
            result["error"] = str(e)

        return result

    async def test_battery_failsafe(self) -> dict[str, Any]:
        """Test battery failsafe trigger.

        Returns:
            Test results
        """
        logger.info("Testing battery failsafe")

        result = {
            "test": "battery_failsafe",
            "initial_battery": 0,
            "trigger_level": 20,  # 20% threshold
            "failsafe_triggered": False,
            "action": "",
            "success": False,
        }

        try:
            # Get initial battery level
            telemetry = await self.mavlink.get_telemetry()
            if telemetry:
                result["initial_battery"] = telemetry.get("battery_percent", 100)

            # Simulate low battery
            self.safety_events.clear()

            # Override battery reading (would be actual in field)
            self.safety_manager.battery_percent = 15  # Below threshold

            # Check safety interlocks
            await self.safety_manager.check_battery_level()

            # Wait for failsafe action
            await asyncio.sleep(2)

            # Check for battery failsafe events
            for event in self.safety_events:
                if event["type"] == "battery_failsafe":
                    result["failsafe_triggered"] = True
                    result["action"] = event["details"].get("action", "")
                    break

            result["success"] = result["failsafe_triggered"]

            # Reset battery level
            self.safety_manager.battery_percent = result["initial_battery"]

            logger.info(f"Battery failsafe: {result['success']}")

        except Exception as e:
            logger.error(f"Battery failsafe test failed: {e}")
            result["error"] = str(e)

        return result

    async def test_signal_loss_recovery(self) -> dict[str, Any]:
        """Test signal loss and recovery.

        Returns:
            Test results
        """
        logger.info("Testing signal loss recovery")

        result = {
            "test": "signal_loss_recovery",
            "signal_lost": False,
            "recovery_action": "",
            "recovery_time_ms": 0,
            "final_state": "",
            "success": False,
        }

        try:
            # Start with beacon detected
            beacon_config = BeaconConfiguration(
                frequency_hz=433_000_000,
                power_dbm=15.0,
            )
            beacon = self.beacon_simulator.create_beacon(
                "signal_test_beacon",
                beacon_config,
                (100, 0, 0),
            )
            await self.beacon_simulator.start_beacon(beacon.beacon_id)

            # Start homing
            self.signal_processor.beacon_detected = True
            self.signal_processor.current_rssi = -75.0
            await self.state_machine.request_transition("HOMING")
            await asyncio.sleep(2)

            # Simulate signal loss
            self.safety_events.clear()
            start_time = datetime.now(UTC)

            await self.beacon_simulator.stop_beacon(beacon.beacon_id)
            self.signal_processor.beacon_detected = False
            self.signal_processor.current_rssi = -120.0

            result["signal_lost"] = True

            # Wait for recovery action
            max_wait = 10.0
            while (datetime.now(UTC) - start_time).total_seconds() < max_wait:
                # Check state change
                if self.state_machine.current_state != "HOMING":
                    result["recovery_time_ms"] = (
                        datetime.now(UTC) - start_time
                    ).total_seconds() * 1000
                    result["recovery_action"] = self.state_machine.current_state
                    break
                await asyncio.sleep(0.1)

            result["final_state"] = self.state_machine.current_state
            result["success"] = (
                result["signal_lost"]
                and result["recovery_action"] in ["SEARCHING", "IDLE", "ERROR"]
                and result["recovery_time_ms"] > 0
            )

            logger.info(
                f"Signal loss recovery: {result['success']} in {result['recovery_time_ms']:.0f}ms"
            )

        except Exception as e:
            logger.error(f"Signal loss test failed: {e}")
            result["error"] = str(e)

        return result

    async def test_manual_override(self) -> dict[str, Any]:
        """Test manual override from each system state.

        Returns:
            Test results
        """
        logger.info("Testing manual override")

        result = {
            "test": "manual_override",
            "states_tested": [],
            "override_success": {},
            "average_response_ms": 0,
            "success": False,
        }

        states_to_test = ["SEARCHING", "DETECTING", "HOMING"]
        response_times = []

        try:
            for state in states_to_test:
                logger.info(f"Testing override from {state}")

                # Transition to state
                await self.state_machine.request_transition(state)
                await asyncio.sleep(1)

                # Trigger manual override
                start_time = datetime.now(UTC)
                await self.state_machine.manual_override()

                # Wait for state change
                max_wait = 3.0
                elapsed = 0
                while elapsed < max_wait:
                    if self.state_machine.current_state == "MANUAL":
                        response_time = elapsed * 1000
                        response_times.append(response_time)
                        result["override_success"][state] = True
                        break
                    await asyncio.sleep(0.01)
                    elapsed = (datetime.now(UTC) - start_time).total_seconds()
                else:
                    result["override_success"][state] = False

                result["states_tested"].append(state)

                # Return to IDLE
                await self.state_machine.request_transition("IDLE")
                await asyncio.sleep(1)

            # Calculate average response time
            if response_times:
                result["average_response_ms"] = sum(response_times) / len(response_times)

            # Check overall success
            result["success"] = all(result["override_success"].values())

            logger.info(f"Manual override: {result['success']}")

        except Exception as e:
            logger.error(f"Manual override test failed: {e}")
            result["error"] = str(e)

        return result

    async def validate_all_safety_interlocks(self) -> dict[str, Any]:
        """Validate all safety interlocks per preflight checklist.

        Returns:
            Validation results
        """
        logger.info("Validating all safety interlocks")

        result = {
            "test": "safety_interlocks",
            "interlocks": {},
            "all_passed": False,
        }

        try:
            # Check all safety interlocks
            interlock_status = await self.safety_manager.check_all_safety_interlocks()

            result["interlocks"] = interlock_status
            result["all_passed"] = all(interlock_status.values())

            # Log results
            for check, passed in interlock_status.items():
                if passed:
                    logger.info(f"✓ {check}: PASSED")
                else:
                    logger.error(f"✗ {check}: FAILED")

        except Exception as e:
            logger.error(f"Safety interlock validation failed: {e}")
            result["error"] = str(e)

        return result

    async def run_all_safety_tests(self) -> list[dict[str, Any]]:
        """Run all safety validation tests.

        Returns:
            List of all test results
        """
        all_results = []

        # Test emergency stop
        result = await self.test_emergency_stop()
        all_results.append(result)
        await asyncio.sleep(5)

        # Test geofence
        result = await self.test_geofence_enforcement()
        all_results.append(result)
        await asyncio.sleep(5)

        # Test battery failsafe
        result = await self.test_battery_failsafe()
        all_results.append(result)
        await asyncio.sleep(5)

        # Test signal loss
        result = await self.test_signal_loss_recovery()
        all_results.append(result)
        await asyncio.sleep(5)

        # Test manual override
        result = await self.test_manual_override()
        all_results.append(result)
        await asyncio.sleep(5)

        # Validate all interlocks
        result = await self.validate_all_safety_interlocks()
        all_results.append(result)

        return all_results

    async def save_results(self, results: list[dict[str, Any]]):
        """Save test results.

        Args:
            results: List of test results
        """
        results_dir = Path("data/field_tests/safety_validation")
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"safety_validation_{timestamp}.json"

        # Count successful tests
        successful = sum(1 for r in results if r.get("success", False))
        total = len(results)

        with open(results_file, "w") as f:
            json.dump(
                {
                    "test_type": "safety_validation",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "summary": {
                        "total_tests": total,
                        "successful": successful,
                        "failed": total - successful,
                        "success_rate": successful / total if total > 0 else 0,
                    },
                    "safety_events": self.safety_events,
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
@click.option("--skip-dangerous", is_flag=True, help="Skip potentially dangerous tests")
def main(skip_dangerous: bool):
    """Execute safety system validation tests."""
    logger.info("Starting safety validation tests")

    if skip_dangerous:
        logger.warning("Skipping potentially dangerous tests")

    test_executor = SafetyValidationTest()

    async def run_tests():
        """Async test execution."""
        try:
            await test_executor.setup()

            # Run all safety tests
            results = await test_executor.run_all_safety_tests()

            # Save results
            await test_executor.save_results(results)

            # Summary
            successful = sum(1 for r in results if r.get("success", False))
            total = len(results)

            logger.info(f"Safety validation: {successful}/{total} tests passed")

            if successful == total:
                logger.info("✓ All safety systems validated successfully")
            else:
                failed = [r["test"] for r in results if not r.get("success", False)]
                logger.error(f"✗ Failed tests: {failed}")

        finally:
            await test_executor.cleanup()

    asyncio.run(run_tests())


if __name__ == "__main__":
    main()
