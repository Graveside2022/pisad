#!/usr/bin/env python3
"""
ArduPilot SITL setup and management script for PISAD development.

This script helps set up and manage ArduPilot SITL (Software In The Loop)
simulation for testing MAVLink communication without physical hardware.
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SITLManager:
    """Manages ArduPilot SITL setup and execution."""

    def __init__(
        self,
        vehicle: str = "copter",
        location: str = "lat=-35.363261,lon=149.165230,alt=584,heading=90",
    ):
        """Initialize SITL manager.

        Args:
            vehicle: Vehicle type (copter, plane, rover, sub)
            location: Starting location string
        """
        self.vehicle = vehicle
        self.location = location
        self.sitl_process: subprocess.Popen | None = None
        self.ardupilot_path = Path.home() / "ardupilot"

    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed."""
        logger.info("Checking dependencies...")

        # Check for Python
        try:
            result = subprocess.run(
                ["python3", "--version"], capture_output=True, text=True, check=True
            )
            logger.info(f"Python: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("Python 3 is not installed")
            return False

        # Check for MAVProxy (optional but recommended)
        try:
            result = subprocess.run(
                ["mavproxy.py", "--version"], capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                logger.info("MAVProxy is installed")
            else:
                logger.warning("MAVProxy not found - install with: pip install MAVProxy")
        except FileNotFoundError:
            logger.warning("MAVProxy not found - install with: pip install MAVProxy")

        return True

    def install_ardupilot(self) -> bool:
        """Clone and set up ArduPilot if not already installed."""
        if self.ardupilot_path.exists():
            logger.info(f"ArduPilot already exists at {self.ardupilot_path}")
            return True

        logger.info("Installing ArduPilot...")

        try:
            # Clone ArduPilot repository
            logger.info("Cloning ArduPilot repository...")
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/ArduPilot/ardupilot.git",
                    str(self.ardupilot_path),
                ],
                check=True,
            )

            # Update submodules
            logger.info("Updating submodules...")
            subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive"],
                cwd=self.ardupilot_path,
                check=True,
            )

            # Install prerequisites
            logger.info("Installing prerequisites...")
            prereq_script = (
                self.ardupilot_path / "Tools" / "environment_install" / "install-prereqs-ubuntu.sh"
            )
            if prereq_script.exists():
                subprocess.run(
                    ["bash", str(prereq_script), "-y"], cwd=self.ardupilot_path, check=True
                )
            else:
                logger.warning(
                    "Prerequisites script not found - manual installation may be required"
                )

            logger.info("ArduPilot installation complete")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install ArduPilot: {e}")
            return False

    def build_sitl(self) -> bool:
        """Build SITL for the specified vehicle."""
        if not self.ardupilot_path.exists():
            logger.error("ArduPilot not found - run install first")
            return False

        vehicle_path = self.ardupilot_path / f"Ardupilot{self.vehicle.capitalize()}"
        if not vehicle_path.exists():
            # Try alternative path structure
            vehicle_path = self.ardupilot_path / "ArduCopter"  # Default to copter for now

        if not vehicle_path.exists():
            logger.error(f"Vehicle directory not found: {vehicle_path}")
            return False

        logger.info(f"Building SITL for {self.vehicle}...")

        try:
            # Configure waf
            subprocess.run(
                ["./waf", "configure", "--board", "sitl"], cwd=self.ardupilot_path, check=True
            )

            # Build SITL
            subprocess.run(["./waf", self.vehicle], cwd=self.ardupilot_path, check=True)

            logger.info("SITL build complete")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to build SITL: {e}")
            return False

    def start_sitl(self, wipe_eeprom: bool = True, console: bool = True, map: bool = False) -> bool:
        """Start SITL simulation.

        Args:
            wipe_eeprom: Whether to wipe EEPROM on start
            console: Whether to show console output
            map: Whether to show map (requires MAVProxy)
        """
        if self.sitl_process and self.sitl_process.poll() is None:
            logger.warning("SITL is already running")
            return True

        logger.info(f"Starting SITL {self.vehicle} simulation...")

        # Build command
        cmd = [
            "python3",
            str(self.ardupilot_path / "Tools" / "autotest" / "sim_vehicle.py"),
            "-v",
            self.vehicle.upper(),
            "-L",
            self.location,
            "--out",
            "tcp:127.0.0.1:5760",  # MAVLink output for PISAD
            "--out",
            "tcp:127.0.0.1:14550",  # MAVLink output for GCS
        ]

        if wipe_eeprom:
            cmd.append("-w")

        if console:
            cmd.append("--console")

        if map:
            cmd.append("--map")

        try:
            # Start SITL process
            self.sitl_process = subprocess.Popen(
                cmd,
                cwd=self.ardupilot_path,
                stdout=subprocess.PIPE if not console else None,
                stderr=subprocess.PIPE if not console else None,
            )

            # Wait for SITL to start
            logger.info("Waiting for SITL to initialize...")
            time.sleep(5)

            if self.sitl_process.poll() is not None:
                logger.error("SITL process terminated unexpectedly")
                return False

            logger.info("SITL started successfully")
            logger.info("MAVLink available on:")
            logger.info("  - TCP 127.0.0.1:5760 (for PISAD)")
            logger.info("  - TCP 127.0.0.1:14550 (for GCS/MAVProxy)")
            return True

        except Exception as e:
            logger.error(f"Failed to start SITL: {e}")
            return False

    def stop_sitl(self):
        """Stop SITL simulation."""
        if self.sitl_process and self.sitl_process.poll() is None:
            logger.info("Stopping SITL...")
            self.sitl_process.terminate()

            # Wait for graceful shutdown
            try:
                self.sitl_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("SITL did not stop gracefully, killing process")
                self.sitl_process.kill()
                self.sitl_process.wait()

            logger.info("SITL stopped")
        else:
            logger.info("SITL is not running")

    def test_connection(self) -> bool:
        """Test MAVLink connection to SITL."""
        logger.info("Testing MAVLink connection...")

        try:
            # Try to import pymavlink
            from pymavlink import mavutil

            # Try to connect
            connection = mavutil.mavlink_connection("tcp:127.0.0.1:5760")

            # Wait for heartbeat
            logger.info("Waiting for heartbeat...")
            msg = connection.wait_heartbeat(timeout=10)

            if msg:
                logger.info(f"Connection successful! System ID: {msg.get_srcSystem()}")
                return True
            else:
                logger.error("No heartbeat received")
                return False

        except ImportError:
            logger.error("pymavlink not installed - run: pip install pymavlink")
            return False
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ArduPilot SITL setup for PISAD")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Install command
    subparsers.add_parser("install", help="Install ArduPilot")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build SITL")
    build_parser.add_argument(
        "--vehicle",
        default="copter",
        choices=["copter", "plane", "rover", "sub"],
        help="Vehicle type",
    )

    # Start command
    start_parser = subparsers.add_parser("start", help="Start SITL")
    start_parser.add_argument(
        "--vehicle",
        default="copter",
        choices=["copter", "plane", "rover", "sub"],
        help="Vehicle type",
    )
    start_parser.add_argument("--no-wipe", action="store_true", help="Do not wipe EEPROM on start")
    start_parser.add_argument(
        "--no-console", action="store_true", help="Do not show console output"
    )
    start_parser.add_argument("--map", action="store_true", help="Show map (requires MAVProxy)")

    # Stop command
    subparsers.add_parser("stop", help="Stop SITL")

    # Test command
    subparsers.add_parser("test", help="Test MAVLink connection")

    # Quick start command
    quick_parser = subparsers.add_parser("quick", help="Quick start (install, build, start)")
    quick_parser.add_argument(
        "--vehicle",
        default="copter",
        choices=["copter", "plane", "rover", "sub"],
        help="Vehicle type",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Create manager
    vehicle = getattr(args, "vehicle", "copter")
    manager = SITLManager(vehicle=vehicle)

    # Check dependencies first
    if not manager.check_dependencies():
        return 1

    # Execute command
    if args.command == "install":
        success = manager.install_ardupilot()

    elif args.command == "build":
        success = manager.build_sitl()

    elif args.command == "start":
        success = manager.start_sitl(
            wipe_eeprom=not args.no_wipe, console=not args.no_console, map=args.map
        )

        if success:
            try:
                # Keep running until interrupted
                logger.info("SITL is running. Press Ctrl+C to stop.")
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                manager.stop_sitl()

    elif args.command == "stop":
        manager.stop_sitl()
        success = True

    elif args.command == "test":
        success = manager.test_connection()

    elif args.command == "quick":
        # Quick start - install, build, and start
        logger.info("Running quick start...")

        if not manager.ardupilot_path.exists() and not manager.install_ardupilot():
            return 1

        if not manager.build_sitl():
            return 1

        success = manager.start_sitl()

        if success:
            # Test connection
            time.sleep(2)
            manager.test_connection()

            try:
                # Keep running until interrupted
                logger.info("SITL is running. Press Ctrl+C to stop.")
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                manager.stop_sitl()
    else:
        parser.print_help()
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
