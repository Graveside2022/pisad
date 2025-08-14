#!/usr/bin/env python3
"""Test script for GCS telemetry compatibility testing.

This script validates MAVLink telemetry with QGroundControl and Mission Planner.
"""

import argparse
import json
import logging
import sys
import time
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pymavlink import mavutil  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GCSType(Enum):
    """Supported GCS types."""

    QGROUNDCONTROL = "qgroundcontrol"
    MISSION_PLANNER = "missionplanner"
    BOTH = "both"


@dataclass
class TelemetryStats:
    """Statistics for telemetry validation."""

    messages_received: int = 0
    rssi_messages: int = 0
    statustext_messages: int = 0
    health_messages: int = 0
    state_changes: int = 0
    detection_events: int = 0
    last_rssi_value: float = -100.0
    last_rssi_time: float = 0.0
    rssi_rate_hz: float = 0.0
    message_types: Counter = None

    def __post_init__(self):
        if self.message_types is None:
            self.message_types = Counter()


class GCSTelemetryTester:
    """Test MAVLink telemetry compatibility with GCS."""

    def __init__(
        self,
        connection_string: str,
        gcs_type: GCSType = GCSType.BOTH,
        test_duration: int = 30,
    ):
        """Initialize GCS telemetry tester.

        Args:
            connection_string: MAVLink connection string (e.g., "udp:127.0.0.1:14550")
            gcs_type: Type of GCS to test
            test_duration: Test duration in seconds
        """
        self.connection_string = connection_string
        self.gcs_type = gcs_type
        self.test_duration = test_duration
        self.connection = None
        self.stats = TelemetryStats()
        self.rssi_timestamps = []

    def connect(self) -> bool:
        """Establish MAVLink connection.

        Returns:
            True if connected successfully, False otherwise
        """
        try:
            logger.info(f"Connecting to {self.connection_string}")
            self.connection = mavutil.mavlink_connection(
                self.connection_string, source_system=255, source_component=1
            )

            # Wait for heartbeat
            msg = self.connection.wait_heartbeat(timeout=5)
            if msg:
                logger.info(f"Connected to system {msg.get_srcSystem()}")
                return True
            else:
                logger.error("No heartbeat received")
                return False
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    def monitor_messages(self, duration: int) -> None:
        """Monitor MAVLink messages for specified duration.

        Args:
            duration: Monitoring duration in seconds
        """
        logger.info(f"Monitoring messages for {duration} seconds...")
        start_time = time.time()

        while time.time() - start_time < duration:
            msg = self.connection.recv_match(blocking=False)
            if msg:
                self._process_message(msg)
            time.sleep(0.001)  # Small delay to prevent CPU spinning

    def _process_message(self, msg: Any) -> None:
        """Process received MAVLink message.

        Args:
            msg: MAVLink message
        """
        msg_type = msg.get_type()
        self.stats.messages_received += 1
        self.stats.message_types[msg_type] += 1

        if msg_type == "NAMED_VALUE_FLOAT":
            self._process_named_value_float(msg)
        elif msg_type == "STATUSTEXT":
            self._process_statustext(msg)

    def _process_named_value_float(self, msg: Any) -> None:
        """Process NAMED_VALUE_FLOAT message.

        Args:
            msg: NAMED_VALUE_FLOAT message
        """
        name = msg.name.decode("utf-8").rstrip("\x00")
        value = msg.value

        if name == "PISAD_RSSI":
            self.stats.rssi_messages += 1
            self.stats.last_rssi_value = value
            current_time = time.time()
            self.stats.last_rssi_time = current_time
            self.rssi_timestamps.append(current_time)

            # Calculate rate if we have enough samples
            if len(self.rssi_timestamps) > 10:
                time_diff = self.rssi_timestamps[-1] - self.rssi_timestamps[-11]
                self.stats.rssi_rate_hz = 10.0 / time_diff

            logger.debug(f"RSSI: {value:.2f} dBm")

    def _process_statustext(self, msg: Any) -> None:
        """Process STATUSTEXT message.

        Args:
            msg: STATUSTEXT message
        """
        text = msg.text.decode("utf-8").rstrip("\x00")
        severity = msg.severity

        self.stats.statustext_messages += 1

        if text.startswith("PISAD:"):
            # Parse PISAD messages
            if "State changed to" in text:
                self.stats.state_changes += 1
                logger.info(f"State change: {text}")
            elif "Signal detected" in text:
                self.stats.detection_events += 1
                logger.info(f"Detection: {text}")
            elif "Health" in text:
                self.stats.health_messages += 1
                # Try to parse health JSON
                try:
                    health_json = text.split("Health")[1].strip()
                    health = json.loads(health_json)
                    logger.info(f"Health status: {health}")
                except (IndexError, json.JSONDecodeError):
                    logger.debug(f"Could not parse health: {text}")
            else:
                logger.info(f"PISAD message: {text}")
        else:
            logger.debug(f"STATUSTEXT [{severity}]: {text}")

    def validate_qgroundcontrol(self) -> dict[str, Any]:
        """Validate QGroundControl compatibility.

        Returns:
            Validation results
        """
        logger.info("Validating QGroundControl compatibility...")

        results = {
            "gcs": "QGroundControl",
            "connection": "udp",
            "rssi_display": self.stats.rssi_messages > 0,
            "statustext_display": self.stats.statustext_messages > 0,
            "rssi_rate_hz": self.stats.rssi_rate_hz,
            "issues": [],
        }

        # Check RSSI rate
        if self.stats.rssi_rate_hz < 1.5 or self.stats.rssi_rate_hz > 2.5:
            results["issues"].append(
                f"RSSI rate out of spec: {self.stats.rssi_rate_hz:.2f} Hz (expected 2Hz)"
            )

        # Check message reception
        if self.stats.rssi_messages == 0:
            results["issues"].append("No RSSI messages received")
        if self.stats.statustext_messages == 0:
            results["issues"].append("No STATUSTEXT messages received")

        # QGC-specific checks
        if "NAMED_VALUE_FLOAT" not in self.stats.message_types:
            results["issues"].append("QGC may not display NAMED_VALUE_FLOAT properly")

        results["passed"] = len(results["issues"]) == 0
        return results

    def validate_mission_planner(self) -> dict[str, Any]:
        """Validate Mission Planner compatibility.

        Returns:
            Validation results
        """
        logger.info("Validating Mission Planner compatibility...")

        results = {
            "gcs": "Mission Planner",
            "connection": "tcp",
            "rssi_display": self.stats.rssi_messages > 0,
            "statustext_display": self.stats.statustext_messages > 0,
            "rssi_rate_hz": self.stats.rssi_rate_hz,
            "issues": [],
        }

        # Mission Planner specific checks
        if self.stats.health_messages == 0:
            results["issues"].append("No health status messages received")

        # Check message format
        if self.stats.statustext_messages > 0:
            # Mission Planner may have issues with long STATUSTEXT
            pass  # Add specific checks if needed

        results["passed"] = len(results["issues"]) == 0
        return results

    def print_statistics(self) -> None:
        """Print telemetry statistics."""
        print("\n" + "=" * 60)
        print("TELEMETRY STATISTICS")
        print("=" * 60)
        print(f"Total messages received: {self.stats.messages_received}")
        print(f"RSSI messages: {self.stats.rssi_messages}")
        print(f"STATUSTEXT messages: {self.stats.statustext_messages}")
        print(f"Health messages: {self.stats.health_messages}")
        print(f"State changes: {self.stats.state_changes}")
        print(f"Detection events: {self.stats.detection_events}")
        print(f"Last RSSI value: {self.stats.last_rssi_value:.2f} dBm")
        print(f"RSSI rate: {self.stats.rssi_rate_hz:.2f} Hz")
        print("\nMessage type distribution:")
        for msg_type, count in self.stats.message_types.most_common(10):
            print(f"  {msg_type}: {count}")

    def run_test(self) -> dict[str, Any]:
        """Run GCS compatibility test.

        Returns:
            Test results
        """
        if not self.connect():
            return {"error": "Failed to connect"}

        # Monitor messages
        self.monitor_messages(self.test_duration)

        # Print statistics
        self.print_statistics()

        # Run validation based on GCS type
        results = {"stats": self.stats.__dict__, "validations": []}

        if self.gcs_type in [GCSType.QGROUNDCONTROL, GCSType.BOTH]:
            results["validations"].append(self.validate_qgroundcontrol())

        if self.gcs_type in [GCSType.MISSION_PLANNER, GCSType.BOTH]:
            results["validations"].append(self.validate_mission_planner())

        # Overall pass/fail
        results["overall_passed"] = all(v["passed"] for v in results["validations"])

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test GCS telemetry compatibility")
    parser.add_argument(
        "--connection",
        default="udp:127.0.0.1:14550",
        help="MAVLink connection string (default: udp:127.0.0.1:14550)",
    )
    parser.add_argument(
        "--gcs",
        type=str,
        choices=["qgroundcontrol", "missionplanner", "both"],
        default="both",
        help="GCS type to test (default: both)",
    )
    parser.add_argument(
        "--duration", type=int, default=30, help="Test duration in seconds (default: 30)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Create tester
    tester = GCSTelemetryTester(
        connection_string=args.connection,
        gcs_type=GCSType(args.gcs),
        test_duration=args.duration,
    )

    # Run test
    results = tester.run_test()

    # Print results
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    for validation in results.get("validations", []):
        print(f"\n{validation['gcs']}:")
        print(f"  Passed: {validation['passed']}")
        if validation["issues"]:
            print("  Issues:")
            for issue in validation["issues"]:
                print(f"    - {issue}")
        else:
            print("  No issues found")

    # Exit code based on results
    if results.get("overall_passed", False):
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
