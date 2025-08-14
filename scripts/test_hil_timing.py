#!/usr/bin/env python3
"""Hardware-in-loop timing validation tests.

This script measures actual timing performance of safety responses
using real hardware (Raspberry Pi 5 and flight controller).
"""

import asyncio
import contextlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backend.services.mavlink_service import MAVLinkService
from backend.utils.safety import SafetyInterlockSystem


class HILTimingTester:
    """Hardware-in-loop timing validation."""

    def __init__(self, connection_string: str = "/dev/ttyACM0"):
        """Initialize HIL tester.

        Args:
            connection_string: MAVLink connection string
        """
        self.connection_string = connection_string
        self.mavlink_service = MAVLinkService(connection_string)
        self.safety_system = SafetyInterlockSystem()
        self.test_results: list[dict[str, Any]] = []

    async def setup(self) -> None:
        """Set up hardware connections."""
        print("Setting up hardware connections...")
        await self.mavlink_service.connect()
        await self.safety_system.start_monitoring()
        print("Hardware setup complete")

    async def teardown(self) -> None:
        """Clean up connections."""
        await self.safety_system.stop_monitoring()
        await self.mavlink_service.disconnect()

    async def test_mode_change_detection_latency(self) -> dict[str, Any]:
        """Test mode change detection latency (<100ms requirement).

        Returns:
            Test result with timing data
        """
        test_name = "Mode Change Detection Latency"
        print(f"\nTesting: {test_name}")
        timings = []

        for i in range(10):
            # Start in GUIDED
            await self.mavlink_service.set_mode("GUIDED")
            await asyncio.sleep(0.5)

            # Measure mode change detection
            start_time = time.perf_counter()
            await self.mavlink_service.set_mode("LOITER")
            self.safety_system.update_flight_mode("LOITER")

            # Check if mode change detected
            mode_check = self.safety_system.checks["mode"]
            await mode_check.check()

            detection_time = (time.perf_counter() - start_time) * 1000
            timings.append(detection_time)
            print(f"  Run {i+1}: {detection_time:.2f}ms")

        avg_time = statistics.mean(timings)
        max_time = max(timings)
        min_time = min(timings)
        std_dev = statistics.stdev(timings) if len(timings) > 1 else 0

        result = {
            "test": test_name,
            "requirement_ms": 100,
            "average_ms": avg_time,
            "max_ms": max_time,
            "min_ms": min_time,
            "std_dev_ms": std_dev,
            "samples": len(timings),
            "pass": max_time < 100,
        }

        print(f"  Average: {avg_time:.2f}ms, Max: {max_time:.2f}ms")
        print(f"  Result: {'PASS' if result['pass'] else 'FAIL'}")
        return result

    async def test_emergency_stop_response_time(self) -> dict[str, Any]:
        """Test emergency stop response time (<500ms requirement).

        Returns:
            Test result with timing data
        """
        test_name = "Emergency Stop Response Time"
        print(f"\nTesting: {test_name}")
        timings = []

        for i in range(10):
            # Reset and enable system
            await self.safety_system.reset_emergency_stop()
            self.safety_system.update_flight_mode("GUIDED")
            self.safety_system.update_battery(50.0)
            self.safety_system.update_signal_snr(10.0)
            await self.safety_system.enable_homing()
            await asyncio.sleep(0.5)

            # Measure emergency stop response
            start_time = time.perf_counter()
            await self.safety_system.emergency_stop("Test timing")

            # Verify system stopped
            await self.safety_system.is_safe_to_proceed()

            response_time = (time.perf_counter() - start_time) * 1000
            timings.append(response_time)
            print(f"  Run {i+1}: {response_time:.2f}ms")

        avg_time = statistics.mean(timings)
        max_time = max(timings)
        min_time = min(timings)
        std_dev = statistics.stdev(timings) if len(timings) > 1 else 0

        result = {
            "test": test_name,
            "requirement_ms": 500,
            "average_ms": avg_time,
            "max_ms": max_time,
            "min_ms": min_time,
            "std_dev_ms": std_dev,
            "samples": len(timings),
            "pass": max_time < 500,
        }

        print(f"  Average: {avg_time:.2f}ms, Max: {max_time:.2f}ms")
        print(f"  Result: {'PASS' if result['pass'] else 'FAIL'}")
        return result

    async def test_velocity_command_cessation_timing(self) -> dict[str, Any]:
        """Test velocity command cessation timing.

        Returns:
            Test result with timing data
        """
        test_name = "Velocity Command Cessation"
        print(f"\nTesting: {test_name}")
        timings = []

        for i in range(10):
            # Start sending velocity commands
            await self.mavlink_service.set_mode("GUIDED")

            # Measure time to stop commands
            start_time = time.perf_counter()

            # Trigger safety interlock
            self.safety_system.update_battery(15.0)  # Below threshold
            await self.safety_system.check_all_safety()

            # Time when commands would be blocked
            stop_time = (time.perf_counter() - start_time) * 1000
            timings.append(stop_time)
            print(f"  Run {i+1}: {stop_time:.2f}ms")

            # Reset for next test
            self.safety_system.update_battery(50.0)
            await asyncio.sleep(0.5)

        avg_time = statistics.mean(timings)
        max_time = max(timings)

        result = {
            "test": test_name,
            "requirement_ms": 500,
            "average_ms": avg_time,
            "max_ms": max_time,
            "min_ms": min(timings),
            "std_dev_ms": statistics.stdev(timings) if len(timings) > 1 else 0,
            "samples": len(timings),
            "pass": max_time < 500,
        }

        print(f"  Average: {avg_time:.2f}ms, Max: {max_time:.2f}ms")
        print(f"  Result: {'PASS' if result['pass'] else 'FAIL'}")
        return result

    async def test_safety_interlock_trigger_timing(self) -> dict[str, Any]:
        """Test safety interlock trigger to action timing.

        Returns:
            Test result with timing data
        """
        test_name = "Safety Interlock Trigger Timing"
        print(f"\nTesting: {test_name}")
        timings = []

        interlocks = [
            ("mode", lambda: self.safety_system.update_flight_mode("STABILIZE")),
            ("battery", lambda: self.safety_system.update_battery(15.0)),
            ("signal", lambda: self.safety_system.update_signal_snr(3.0)),
        ]

        for interlock_name, trigger_func in interlocks:
            # Reset system
            self.safety_system.update_flight_mode("GUIDED")
            self.safety_system.update_battery(50.0)
            self.safety_system.update_signal_snr(10.0)
            await asyncio.sleep(0.1)

            # Measure trigger to action
            start_time = time.perf_counter()
            trigger_func()
            await self.safety_system.check_all_safety()
            action_time = (time.perf_counter() - start_time) * 1000

            timings.append(action_time)
            print(f"  {interlock_name}: {action_time:.2f}ms")

        avg_time = statistics.mean(timings)
        max_time = max(timings)

        result = {
            "test": test_name,
            "requirement_ms": 100,
            "average_ms": avg_time,
            "max_ms": max_time,
            "min_ms": min(timings),
            "std_dev_ms": statistics.stdev(timings) if len(timings) > 1 else 0,
            "samples": len(timings),
            "pass": max_time < 100,
        }

        print(f"  Average: {avg_time:.2f}ms, Max: {max_time:.2f}ms")
        print(f"  Result: {'PASS' if result['pass'] else 'FAIL'}")
        return result

    async def profile_cpu_usage_during_safety(self) -> dict[str, Any]:
        """Profile CPU usage during safety responses.

        Returns:
            CPU usage statistics
        """
        test_name = "CPU Usage During Safety Response"
        print(f"\nTesting: {test_name}")

        import psutil

        cpu_samples = []

        # Baseline CPU
        baseline_cpu = psutil.cpu_percent(interval=1)
        print(f"  Baseline CPU: {baseline_cpu:.1f}%")

        # CPU during safety checks
        for _i in range(10):
            # Trigger multiple safety checks
            await self.safety_system.check_all_safety()
            cpu = psutil.cpu_percent(interval=0.1)
            cpu_samples.append(cpu)

        avg_cpu = statistics.mean(cpu_samples)
        max_cpu = max(cpu_samples)

        result = {
            "test": test_name,
            "baseline_cpu_percent": baseline_cpu,
            "average_cpu_percent": avg_cpu,
            "max_cpu_percent": max_cpu,
            "samples": len(cpu_samples),
            "pass": max_cpu < 50,  # Should use less than 50% CPU
        }

        print(f"  Average CPU: {avg_cpu:.1f}%, Max: {max_cpu:.1f}%")
        print(f"  Result: {'PASS' if result['pass'] else 'FAIL'}")
        return result

    async def test_timing_under_load(self) -> dict[str, Any]:
        """Verify timing under various CPU load conditions.

        Returns:
            Test results under load
        """
        test_name = "Timing Under CPU Load"
        print(f"\nTesting: {test_name}")

        results = []

        # Test with no load
        print("  Testing with no additional load...")
        no_load_timing = await self._measure_response_time()
        results.append({"load": "none", "response_ms": no_load_timing})

        # Test with CPU load
        print("  Testing with CPU load...")
        load_task = asyncio.create_task(self._generate_cpu_load())
        await asyncio.sleep(1)

        with_load_timing = await self._measure_response_time()
        results.append({"load": "cpu", "response_ms": with_load_timing})

        load_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await load_task

        # Calculate degradation
        degradation = ((with_load_timing - no_load_timing) / no_load_timing) * 100

        result = {
            "test": test_name,
            "no_load_ms": no_load_timing,
            "with_load_ms": with_load_timing,
            "degradation_percent": degradation,
            "pass": with_load_timing < 500,  # Still meet requirement under load
        }

        print(f"  No load: {no_load_timing:.2f}ms")
        print(f"  With load: {with_load_timing:.2f}ms")
        print(f"  Degradation: {degradation:.1f}%")
        print(f"  Result: {'PASS' if result['pass'] else 'FAIL'}")
        return result

    async def _measure_response_time(self) -> float:
        """Measure a single response time.

        Returns:
            Response time in milliseconds
        """
        start_time = time.perf_counter()
        await self.safety_system.emergency_stop("Test")
        await self.safety_system.reset_emergency_stop()
        return (time.perf_counter() - start_time) * 1000

    async def _generate_cpu_load(self) -> None:
        """Generate CPU load for testing."""
        while True:
            # Busy loop to generate load
            for _ in range(1000000):
                _ = 2**10
            await asyncio.sleep(0.001)

    async def run_all_tests(self) -> None:
        """Run all timing validation tests."""
        print("=" * 60)
        print("HARDWARE-IN-LOOP TIMING VALIDATION")
        print("=" * 60)
        print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            await self.setup()

            # Run each test
            self.test_results.append(await self.test_mode_change_detection_latency())
            self.test_results.append(await self.test_emergency_stop_response_time())
            self.test_results.append(await self.test_velocity_command_cessation_timing())
            self.test_results.append(await self.test_safety_interlock_trigger_timing())
            self.test_results.append(await self.profile_cpu_usage_during_safety())
            self.test_results.append(await self.test_timing_under_load())

        finally:
            await self.teardown()

        self.generate_report()

    def generate_report(self) -> None:
        """Generate timing validation report."""
        print("\n" + "=" * 60)
        print("TIMING VALIDATION REPORT")
        print("=" * 60)

        passed = sum(1 for r in self.test_results if r.get("pass", False))
        failed = len(self.test_results) - passed

        print(f"\nSummary: {passed} PASSED, {failed} FAILED")
        print("\nDetailed Results:")
        print("-" * 40)

        for result in self.test_results:
            status = "✓" if result.get("pass", False) else "✗"
            test_name = result.get("test", "Unknown")
            print(f"{status} {test_name}")

            if "average_ms" in result:
                print(f"  Average: {result['average_ms']:.2f}ms")
            if "max_ms" in result:
                print(f"  Maximum: {result['max_ms']:.2f}ms")
            if "requirement_ms" in result:
                print(f"  Requirement: <{result['requirement_ms']}ms")

        # Save report
        report_path = Path(__file__).parent / "hil_timing_report.json"
        with open(report_path, "w") as f:
            json.dump(
                {
                    "timestamp": time.time(),
                    "summary": {"passed": passed, "failed": failed},
                    "results": self.test_results,
                },
                f,
                indent=2,
            )

        print(f"\nReport saved to: {report_path}")

        if failed > 0:
            print("\n⚠ TIMING REQUIREMENTS NOT MET")
            sys.exit(1)
        else:
            print("\n✓ ALL TIMING REQUIREMENTS MET")


async def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Hardware-in-loop timing validation")
    parser.add_argument(
        "--connection",
        default="/dev/ttyACM0",
        help="MAVLink connection string",
    )

    args = parser.parse_args()

    tester = HILTimingTester(args.connection)
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
