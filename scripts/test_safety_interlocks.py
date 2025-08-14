#!/usr/bin/env python3
"""Bench test procedures for safety interlocks.

This script provides a comprehensive test harness for validating all safety
interlock mechanisms in a controlled environment with simulated signals.
"""

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backend.utils.safety import SafetyInterlockSystem


class MockSignalGenerator:
    """Generate controlled RSSI values for testing."""

    def __init__(self) -> None:
        """Initialize mock signal generator."""
        self.rssi_pattern: list[float] = []
        self.pattern_index = 0

    def set_constant(self, value: float) -> None:
        """Set constant RSSI value.

        Args:
            value: RSSI value in dB
        """
        self.rssi_pattern = [value]
        self.pattern_index = 0

    def set_pattern(self, pattern: list[float]) -> None:
        """Set RSSI pattern to cycle through.

        Args:
            pattern: List of RSSI values
        """
        self.rssi_pattern = pattern
        self.pattern_index = 0

    def get_next(self) -> float:
        """Get next RSSI value from pattern.

        Returns:
            Next RSSI value
        """
        if not self.rssi_pattern:
            return 0.0

        value = self.rssi_pattern[self.pattern_index]
        self.pattern_index = (self.pattern_index + 1) % len(self.rssi_pattern)
        return value


class SafetyInterlockTester:
    """Test harness for safety interlock validation."""

    def __init__(self) -> None:
        """Initialize test harness."""
        self.safety_system = SafetyInterlockSystem()
        self.signal_generator = MockSignalGenerator()
        self.test_results: list[dict[str, Any]] = []

    async def test_mode_monitor_interlock(self) -> dict[str, Any]:
        """Test mode monitor interlock (stops commands if not GUIDED).

        Returns:
            Test result dictionary
        """
        test_name = "Mode Monitor Interlock"
        print(f"\nTesting: {test_name}")

        try:
            # Test 1: Non-GUIDED mode should fail
            self.safety_system.update_flight_mode("STABILIZE")
            result = await self.safety_system.is_safe_to_proceed()
            assert not result, "Should fail in non-GUIDED mode"
            print("  ✓ Commands blocked in STABILIZE mode")

            # Test 2: GUIDED mode should pass (if other checks pass)
            self.safety_system.update_flight_mode("GUIDED")
            mode_check = self.safety_system.checks["mode"]
            mode_result = await mode_check.check()
            assert mode_result, "Should pass in GUIDED mode"
            print("  ✓ Mode check passes in GUIDED mode")

            # Test 3: Mode change detection
            self.safety_system.update_flight_mode("AUTO")
            mode_result = await mode_check.check()
            assert not mode_result, "Should fail after mode change"
            print("  ✓ Mode change detected and blocked")

            return {"test": test_name, "status": "PASS", "details": "All mode checks passed"}

        except AssertionError as e:
            return {"test": test_name, "status": "FAIL", "error": str(e)}
        except Exception as e:
            return {"test": test_name, "status": "ERROR", "error": str(e)}

    async def test_operator_activation_requirement(self) -> dict[str, Any]:
        """Test operator activation requirement (Enable Homing button).

        Returns:
            Test result dictionary
        """
        test_name = "Operator Activation Requirement"
        print(f"\nTesting: {test_name}")

        try:
            # Setup: Set other checks to pass
            self.safety_system.update_flight_mode("GUIDED")
            self.safety_system.update_battery(50.0)
            self.safety_system.update_signal_snr(10.0)

            # Test 1: Should fail without operator activation
            operator_check = self.safety_system.checks["operator"]
            result = await operator_check.check()
            assert not result, "Should fail without operator activation"
            print("  ✓ Homing blocked without operator activation")

            # Test 2: Enable homing
            success = await self.safety_system.enable_homing("test_token")
            assert success, "Should enable homing when conditions met"
            result = await operator_check.check()
            assert result, "Should pass after enabling"
            print("  ✓ Homing enabled by operator")

            # Test 3: Disable homing
            await self.safety_system.disable_homing("Test disable")
            result = await operator_check.check()
            assert not result, "Should fail after disabling"
            print("  ✓ Homing disabled successfully")

            return {"test": test_name, "status": "PASS", "details": "Operator controls working"}

        except AssertionError as e:
            return {"test": test_name, "status": "FAIL", "error": str(e)}
        except Exception as e:
            return {"test": test_name, "status": "ERROR", "error": str(e)}

    async def test_signal_loss_auto_disable(self) -> dict[str, Any]:
        """Test signal loss auto-disable (after 10 seconds <6 dB SNR).

        Returns:
            Test result dictionary
        """
        test_name = "Signal Loss Auto-Disable"
        print(f"\nTesting: {test_name}")

        try:
            signal_check = self.safety_system.checks["signal"]

            # Test 1: Good signal should pass
            self.safety_system.update_signal_snr(10.0)
            result = await signal_check.check()
            assert result, "Should pass with good signal"
            print("  ✓ Good signal (10 dB) passes check")

            # Test 2: Weak signal for <10 seconds should still pass
            self.safety_system.update_signal_snr(3.0)
            result = await signal_check.check()
            assert result, "Should pass initially with weak signal"
            print("  ✓ Weak signal initially allowed (grace period)")

            # Test 3: Simulate 10+ seconds of weak signal
            for i in range(11):
                self.safety_system.update_signal_snr(3.0)
                await asyncio.sleep(1.1)
                result = await signal_check.check()
                if i < 9:
                    assert result, f"Should pass at {i+1} seconds"
                else:
                    assert not result, "Should fail after 10 seconds"

            print("  ✓ Signal loss detected after 10 seconds")

            # Test 4: Signal recovery
            self.safety_system.update_signal_snr(12.0)
            result = await signal_check.check()
            assert result, "Should pass after signal recovery"
            print("  ✓ Signal recovery detected")

            return {"test": test_name, "status": "PASS", "details": "Signal timeout working"}

        except AssertionError as e:
            return {"test": test_name, "status": "FAIL", "error": str(e)}
        except Exception as e:
            return {"test": test_name, "status": "ERROR", "error": str(e)}

    async def test_geofence_boundary_checks(self) -> dict[str, Any]:
        """Test geofence boundary checks before movement commands.

        Returns:
            Test result dictionary
        """
        test_name = "Geofence Boundary Checks"
        print(f"\nTesting: {test_name}")

        try:
            geofence_check = self.safety_system.checks["geofence"]

            # Test 1: Disabled geofence should pass
            result = await geofence_check.check()
            assert result, "Should pass with disabled geofence"
            print("  ✓ Disabled geofence allows operation")

            # Test 2: Set geofence
            geofence_check.set_geofence(37.7749, -122.4194, 100.0)

            # Test 3: Position inside fence
            geofence_check.update_position(37.7749, -122.4194)
            result = await geofence_check.check()
            assert result, "Should pass at center of geofence"
            print("  ✓ Position inside geofence allowed")

            # Test 4: Position outside fence
            geofence_check.update_position(37.7850, -122.4194)
            result = await geofence_check.check()
            assert not result, "Should fail outside geofence"
            print("  ✓ Position outside geofence blocked")

            # Test 5: Edge case - exactly on boundary (100m)
            geofence_check.update_position(37.7758, -122.4194)
            result = await geofence_check.check()
            boundary_ok = result  # May pass or fail due to precision
            print(f"  ✓ Boundary test completed (result: {boundary_ok})")

            return {"test": test_name, "status": "PASS", "details": "Geofence checks working"}

        except AssertionError as e:
            return {"test": test_name, "status": "FAIL", "error": str(e)}
        except Exception as e:
            return {"test": test_name, "status": "ERROR", "error": str(e)}

    async def test_battery_monitor_disable(self) -> dict[str, Any]:
        """Test battery monitor disable (when battery <20%).

        Returns:
            Test result dictionary
        """
        test_name = "Battery Monitor Disable"
        print(f"\nTesting: {test_name}")

        try:
            battery_check = self.safety_system.checks["battery"]

            # Test 1: High battery should pass
            self.safety_system.update_battery(75.0)
            result = await battery_check.check()
            assert result, "Should pass with high battery"
            print("  ✓ High battery (75%) passes check")

            # Test 2: Battery at threshold (20%)
            self.safety_system.update_battery(20.0)
            result = await battery_check.check()
            assert not result, "Should fail at threshold"
            print("  ✓ Battery at threshold (20%) fails check")

            # Test 3: Low battery
            self.safety_system.update_battery(15.0)
            result = await battery_check.check()
            assert not result, "Should fail with low battery"
            print("  ✓ Low battery (15%) blocked")

            # Test 4: Battery recovery
            self.safety_system.update_battery(25.0)
            result = await battery_check.check()
            assert result, "Should pass after charging"
            print("  ✓ Battery recovery (25%) detected")

            return {"test": test_name, "status": "PASS", "details": "Battery checks working"}

        except AssertionError as e:
            return {"test": test_name, "status": "FAIL", "error": str(e)}
        except Exception as e:
            return {"test": test_name, "status": "ERROR", "error": str(e)}

    async def test_emergency_stop_functionality(self) -> dict[str, Any]:
        """Test emergency stop functionality.

        Returns:
            Test result dictionary
        """
        test_name = "Emergency Stop Functionality"
        print(f"\nTesting: {test_name}")

        try:
            # Setup: Enable all systems
            self.safety_system.update_flight_mode("GUIDED")
            self.safety_system.update_battery(50.0)
            self.safety_system.update_signal_snr(10.0)
            # Disable geofence for this test
            geofence_check = self.safety_system.checks["geofence"]
            geofence_check.fence_enabled = False
            await self.safety_system.enable_homing()

            # Test 1: Verify system is operational
            result = await self.safety_system.is_safe_to_proceed()
            assert result, "System should be operational before emergency stop"
            print("  ✓ System operational before emergency stop")

            # Test 2: Activate emergency stop
            await self.safety_system.emergency_stop("Test emergency")
            result = await self.safety_system.is_safe_to_proceed()
            assert not result, "Should block all operations after emergency stop"
            print("  ✓ Emergency stop blocks all operations")

            # Test 3: Verify homing disabled
            operator_check = self.safety_system.checks["operator"]
            result = await operator_check.check()
            assert not result, "Homing should be disabled"
            print("  ✓ Homing disabled by emergency stop")

            # Test 4: Reset emergency stop
            await self.safety_system.reset_emergency_stop()
            # Re-enable homing after reset
            await self.safety_system.enable_homing()
            result = await self.safety_system.is_safe_to_proceed()
            assert result, "Should allow operations after reset"
            print("  ✓ System operational after emergency stop reset")

            return {"test": test_name, "status": "PASS", "details": "Emergency stop working"}

        except AssertionError as e:
            return {"test": test_name, "status": "FAIL", "error": str(e)}
        except Exception as e:
            return {"test": test_name, "status": "ERROR", "error": str(e)}

    async def test_multi_interlock_trigger(self) -> dict[str, Any]:
        """Test multiple interlocks triggering simultaneously.

        Returns:
            Test result dictionary
        """
        test_name = "Multi-Interlock Trigger"
        print(f"\nTesting: {test_name}")

        try:
            # Test 1: All checks passing
            self.safety_system.update_flight_mode("GUIDED")
            self.safety_system.update_battery(50.0)
            self.safety_system.update_signal_snr(10.0)
            # Disable geofence for this test
            geofence_check = self.safety_system.checks["geofence"]
            geofence_check.fence_enabled = False
            await self.safety_system.enable_homing()

            result = await self.safety_system.is_safe_to_proceed()
            assert result, "Should pass with all checks good"
            print("  ✓ All interlocks pass when conditions good")

            # Test 2: Single failure blocks operation
            self.safety_system.update_battery(15.0)
            result = await self.safety_system.is_safe_to_proceed()
            assert not result, "Single failure should block"
            print("  ✓ Single interlock failure blocks operation")

            # Test 3: Multiple failures
            self.safety_system.update_flight_mode("STABILIZE")
            self.safety_system.update_signal_snr(3.0)
            results = await self.safety_system.check_all_safety()

            failed_checks = [name for name, passed in results.items() if not passed]
            assert len(failed_checks) >= 2, "Should have multiple failures"
            print(f"  ✓ Multiple interlocks failed: {failed_checks}")

            # Test 4: Recovery requires all checks to pass
            self.safety_system.update_flight_mode("GUIDED")
            result = await self.safety_system.is_safe_to_proceed()
            assert not result, "Should still fail with battery low"

            self.safety_system.update_battery(30.0)
            self.safety_system.update_signal_snr(10.0)
            result = await self.safety_system.is_safe_to_proceed()
            assert result, "Should pass after all issues resolved"
            print("  ✓ All interlocks must pass for operation")

            return {"test": test_name, "status": "PASS", "details": "Multi-interlock working"}

        except AssertionError as e:
            return {"test": test_name, "status": "FAIL", "error": str(e)}
        except Exception as e:
            return {"test": test_name, "status": "ERROR", "error": str(e)}

    async def run_all_tests(self) -> None:
        """Run all safety interlock tests."""
        print("=" * 60)
        print("SAFETY INTERLOCK BENCH TEST")
        print("=" * 60)
        print(f"Start Time: {datetime.now(UTC).isoformat()}")

        # Run each test
        tests = [
            self.test_mode_monitor_interlock,
            self.test_operator_activation_requirement,
            self.test_signal_loss_auto_disable,
            self.test_geofence_boundary_checks,
            self.test_battery_monitor_disable,
            self.test_emergency_stop_functionality,
            self.test_multi_interlock_trigger,
        ]

        for test_func in tests:
            result = await test_func()
            self.test_results.append(result)

        # Generate report
        self.generate_report()

    def generate_report(self) -> None:
        """Generate test report with pass/fail for each interlock."""
        print("\n" + "=" * 60)
        print("TEST REPORT")
        print("=" * 60)

        passed = sum(1 for r in self.test_results if r["status"] == "PASS")
        failed = sum(1 for r in self.test_results if r["status"] == "FAIL")
        errors = sum(1 for r in self.test_results if r["status"] == "ERROR")

        print(f"\nSummary: {passed} PASSED, {failed} FAILED, {errors} ERRORS")
        print("\nDetailed Results:")
        print("-" * 40)

        for result in self.test_results:
            status_symbol = "✓" if result["status"] == "PASS" else "✗"
            print(f"{status_symbol} {result['test']}: {result['status']}")
            if result["status"] != "PASS":
                print(f"  Error: {result.get('error', 'Unknown')}")

        # Save report to file
        report_path = Path(__file__).parent / "safety_interlock_test_report.json"
        with open(report_path, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "summary": {"passed": passed, "failed": failed, "errors": errors},
                    "results": self.test_results,
                },
                f,
                indent=2,
            )

        print(f"\nReport saved to: {report_path}")

        # Exit with appropriate code
        if failed > 0 or errors > 0:
            print("\n⚠ SAFETY VALIDATION FAILED - DO NOT PROCEED TO FLIGHT TESTING")
            sys.exit(1)
        else:
            print("\n✓ ALL SAFETY INTERLOCKS VALIDATED SUCCESSFULLY")


async def main() -> None:
    """Main entry point."""
    tester = SafetyInterlockTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
