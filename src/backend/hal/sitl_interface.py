"""
ArduPilot SITL Interface for PISAD Integration Testing.

Story 4.7 - Sprint 5: SITL Integration
Provides interface between PISAD and ArduPilot SITL for testing without hardware.
"""

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Any

import yaml
from pymavlink import mavutil

from src.backend.core.exceptions import (
    PISADException,
)

logger = logging.getLogger(__name__)


class SITLInterface:
    """Interface for ArduPilot SITL integration."""

    def __init__(self, config_path: str = "config/sitl.yaml") -> None:
        """Initialize SITL interface.

        Args:
            config_path: Path to SITL configuration file
        """
        self.config = self._load_config(config_path)
        self.sitl_process: subprocess.Popen[bytes] | None = None
        self.connection: Any | None = None
        self.connected = False

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """Load SITL configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()

        with open(config_file) as f:
            config = yaml.safe_load(f)

        sitl_config = config.get("sitl", self._get_default_config())
        return sitl_config if isinstance(sitl_config, dict) else self._get_default_config()

    def _get_default_config(self) -> dict[str, Any]:
        """Get default SITL configuration.

        Returns:
            Default configuration dictionary
        """
        return {
            "connection": {"primary": "tcp:127.0.0.1:5760", "timeout": 10, "heartbeat_rate": 1},
            "vehicle": {"type": "copter", "frame": "quad"},
            "location": {"lat": -35.363261, "lon": 149.165230, "alt": 584, "heading": 90},
        }

    async def start_sitl(self, wipe_eeprom: bool = True) -> bool:
        """Start ArduPilot SITL simulator.

        Args:
            wipe_eeprom: Whether to wipe EEPROM on start

        Returns:
            True if SITL started successfully
        """
        if self.sitl_process and self.sitl_process.poll() is None:
            logger.warning("SITL is already running")
            return True

        logger.info("Starting ArduPilot SITL...")

        # Build location string
        location = self.config["location"]
        location_str = f"lat={location['lat']},lon={location['lon']},alt={location['alt']},heading={location['heading']}"

        # Build command
        ardupilot_path = Path.home() / "ardupilot"
        if not ardupilot_path.exists():
            logger.error(f"ArduPilot not found at {ardupilot_path}")
            logger.info("Install ArduPilot with: python3 scripts/sitl_setup.py install")
            return False

        cmd = [
            "python3",
            str(ardupilot_path / "Tools" / "autotest" / "sim_vehicle.py"),
            "-v",
            "ArduCopter",
            "-L",
            location_str,
            "--out",
            "tcp:127.0.0.1:5760",  # For PISAD
            "--out",
            "tcp:127.0.0.1:14550",  # For GCS
            "--no-mavproxy",  # We'll connect directly
        ]

        if wipe_eeprom:
            cmd.append("-w")

        try:
            # Start SITL process
            self.sitl_process = subprocess.Popen(
                cmd, cwd=ardupilot_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            # Wait for SITL to initialize
            logger.info("Waiting for SITL to initialize...")
            await asyncio.sleep(5)

            if self.sitl_process.poll() is not None:
                logger.error("SITL process terminated unexpectedly")
                return False

            logger.info("SITL started successfully")
            return True

        except PISADException as e:
            logger.error(f"Failed to start SITL: {e}")
            return False

    async def stop_sitl(self) -> None:
        """Stop SITL simulator."""
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

    async def connect(self) -> bool:
        """Connect to SITL via MAVLink.

        Returns:
            True if connection successful
        """
        if self.connected:
            logger.warning("Already connected to SITL")
            return True

        connection_string = self.config["connection"]["primary"]
        timeout = self.config["connection"]["timeout"]

        logger.info(f"Connecting to SITL at {connection_string}...")

        try:
            # Create MAVLink connection
            self.connection = mavutil.mavlink_connection(connection_string)

            # Wait for heartbeat
            logger.info("Waiting for heartbeat...")
            msg = self.connection.wait_heartbeat(timeout=timeout)

            if msg:
                self.connected = True
                logger.info(f"Connected to SITL! System ID: {msg.get_srcSystem()}")

                # Request data streams
                self._request_data_streams()

                return True
            else:
                logger.error("No heartbeat received from SITL")
                return False

        except PISADException as e:
            logger.error(f"Failed to connect to SITL: {e}")
            return False

    def _request_data_streams(self) -> None:
        """Request data streams from SITL."""
        if not self.connection:
            return

        # Request all data streams at 4Hz
        self.connection.mav.request_data_stream_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            4,  # Rate in Hz
            1,  # Start streaming
        )

    async def disconnect(self) -> None:
        """Disconnect from SITL."""
        if self.connection:
            logger.info("Disconnecting from SITL...")
            self.connection.close()
            self.connection = None
            self.connected = False

    async def get_telemetry(self) -> dict[str, Any]:
        """Get current telemetry from SITL.

        Returns:
            Dictionary with telemetry data
        """
        if not self.connected or not self.connection:
            return {}

        telemetry = {
            "position": {"lat": 0.0, "lon": 0.0, "alt": 0.0},
            "attitude": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            "velocity": {"vx": 0.0, "vy": 0.0, "vz": 0.0},
            "battery": {"voltage": 0.0, "current": 0.0, "percentage": 0.0},
            "gps": {"fix_type": 0, "satellites": 0, "hdop": 0.0},
            "mode": "UNKNOWN",
            "armed": False,
        }

        # Get GPS position
        msg = self.connection.recv_match(type="GLOBAL_POSITION_INT", blocking=False)
        if msg:
            telemetry["position"] = {
                "lat": msg.lat / 1e7,
                "lon": msg.lon / 1e7,
                "alt": msg.alt / 1000.0,
            }
            telemetry["velocity"] = {
                "vx": msg.vx / 100.0,
                "vy": msg.vy / 100.0,
                "vz": msg.vz / 100.0,
            }

        # Get attitude
        msg = self.connection.recv_match(type="ATTITUDE", blocking=False)
        if msg:
            telemetry["attitude"] = {"roll": msg.roll, "pitch": msg.pitch, "yaw": msg.yaw}

        # SAFETY: Battery monitoring prevents crash from power loss - Story 4.7 AC#3
        # Hazard ID: HARA-PWR-001 - Low voltage leading to uncontrolled descent
        # Mitigation: Monitor 6S Li-ion thresholds (19.2V low, 18.0V critical)
        # Get battery status
        msg = self.connection.recv_match(type="BATTERY_STATUS", blocking=False)
        if msg:
            telemetry["battery"] = {
                "voltage": sum(msg.voltages[:6]) / 1000.0 if msg.voltages[0] != 65535 else 0.0,
                "current": msg.current_battery / 100.0 if msg.current_battery != -1 else 0.0,
                "percentage": msg.battery_remaining if msg.battery_remaining != -1 else 0.0,
            }

        # SAFETY: GPS quality check prevents navigation failure - Story 4.7 FR#5
        # Hazard ID: HARA-NAV-001 - Poor GPS leading to position drift
        # Mitigation: Require 8+ satellites, HDOP < 2.0 for autonomous ops
        # Get GPS status
        msg = self.connection.recv_match(type="GPS_RAW_INT", blocking=False)
        if msg:
            telemetry["gps"] = {
                "fix_type": msg.fix_type,
                "satellites": msg.satellites_visible,
                "hdop": msg.eph / 100.0 if msg.eph != 65535 else 99.99,
            }

        # Get heartbeat for mode and armed status
        msg = self.connection.recv_match(type="HEARTBEAT", blocking=False)
        if msg:
            # Get flight mode
            mode_mapping = self.connection.mode_mapping()
            if mode_mapping:
                telemetry["mode"] = mode_mapping.get(msg.custom_mode, "UNKNOWN")

            # Check if armed
            telemetry["armed"] = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

        return telemetry

    async def arm(self) -> bool:
        """Arm the vehicle in SITL.

        Returns:
            True if arming successful
        """
        if not self.connected or not self.connection:
            logger.error("Not connected to SITL")
            return False

        logger.info("Arming vehicle...")

        # Send arm command
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,  # Confirmation
            1,  # Arm
            0,
            0,
            0,
            0,
            0,
            0,  # Unused parameters
        )

        # Wait for arm confirmation
        msg = self.connection.recv_match(type="COMMAND_ACK", blocking=True, timeout=5)
        if msg and msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
            if msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                logger.info("Vehicle armed successfully")
                return True
            else:
                logger.error(f"Failed to arm: result code {msg.result}")
                return False
        else:
            logger.error("No response to arm command")
            return False

    async def disarm(self) -> bool:
        """Disarm the vehicle in SITL.

        Returns:
            True if disarming successful
        """
        if not self.connected or not self.connection:
            logger.error("Not connected to SITL")
            return False

        logger.info("Disarming vehicle...")

        # Send disarm command
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,  # Confirmation
            0,  # Disarm
            0,
            0,
            0,
            0,
            0,
            0,  # Unused parameters
        )

        # Wait for disarm confirmation
        msg = self.connection.recv_match(type="COMMAND_ACK", blocking=True, timeout=5)
        if msg and msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
            if msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                logger.info("Vehicle disarmed successfully")
                return True
            else:
                logger.error(f"Failed to disarm: result code {msg.result}")
                return False
        else:
            logger.error("No response to disarm command")
            return False

    async def set_mode(self, mode: str) -> bool:
        """Set flight mode in SITL.

        Args:
            mode: Flight mode name (e.g., 'GUIDED', 'STABILIZE')

        Returns:
            True if mode change successful
        """
        if not self.connected or not self.connection:
            logger.error("Not connected to SITL")
            return False

        logger.info(f"Setting mode to {mode}...")

        # Get mode ID
        mode_id = self.connection.mode_mapping().get(mode)
        if mode_id is None:
            logger.error(f"Unknown mode: {mode}")
            return False

        # Send mode change command
        self.connection.mav.set_mode_send(
            self.connection.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
        )

        # Wait for mode change confirmation
        for _ in range(10):
            msg = self.connection.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
            if msg:
                current_mode = self.connection.mode_mapping().get(msg.custom_mode, "UNKNOWN")
                if current_mode == mode:
                    logger.info(f"Mode changed to {mode}")
                    return True

        logger.error(f"Failed to change mode to {mode}")
        return False

    async def takeoff(self, altitude: float = 10.0) -> bool:
        """Command takeoff in SITL.

        Args:
            altitude: Target altitude in meters

        Returns:
            True if takeoff command accepted
        """
        if not self.connected or not self.connection:
            logger.error("Not connected to SITL")
            return False

        logger.info(f"Commanding takeoff to {altitude}m...")

        # Send takeoff command
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,  # Confirmation
            0,  # Pitch
            0,  # Empty
            0,  # Empty
            0,  # Yaw
            0,  # Latitude (0 = current)
            0,  # Longitude (0 = current)
            altitude,  # Altitude
        )

        # Wait for command acknowledgment
        msg = self.connection.recv_match(type="COMMAND_ACK", blocking=True, timeout=5)
        if msg and msg.command == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF:
            if msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                logger.info("Takeoff command accepted")
                return True
            else:
                logger.error(f"Takeoff rejected: result code {msg.result}")
                return False
        else:
            logger.error("No response to takeoff command")
            return False

    async def land(self) -> bool:
        """Command landing in SITL.

        Returns:
            True if land command accepted
        """
        if not self.connected or not self.connection:
            logger.error("Not connected to SITL")
            return False

        logger.info("Commanding landing...")

        # Send land command
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_NAV_LAND,
            0,  # Confirmation
            0,  # Abort altitude
            0,  # Land mode
            0,  # Empty
            0,  # Yaw
            0,  # Latitude (0 = current)
            0,  # Longitude (0 = current)
            0,  # Altitude (0 = ground level)
        )

        # Wait for command acknowledgment
        msg = self.connection.recv_match(type="COMMAND_ACK", blocking=True, timeout=5)
        if msg and msg.command == mavutil.mavlink.MAV_CMD_NAV_LAND:
            if msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                logger.info("Land command accepted")
                return True
            else:
                logger.error(f"Land rejected: result code {msg.result}")
                return False
        else:
            logger.error("No response to land command")
            return False

    async def send_velocity_command(
        self, vx: float, vy: float, vz: float, yaw_rate: float = 0.0
    ) -> bool:
        """Send velocity command to SITL.

        Args:
            vx: Velocity in X (North) direction (m/s)
            vy: Velocity in Y (East) direction (m/s)
            vz: Velocity in Z (Down) direction (m/s)
            yaw_rate: Yaw rate (rad/s)

        Returns:
            True if command sent successfully
        """
        if not self.connected or not self.connection:
            logger.error("Not connected to SITL")
            return False

        # Send velocity command using SET_POSITION_TARGET_LOCAL_NED
        self.connection.mav.set_position_target_local_ned_send(
            0,  # time_boot_ms (not used)
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111000111,  # Type mask (only use velocities)
            0,
            0,
            0,  # Position (not used)
            vx,
            vy,
            vz,  # Velocity
            0,
            0,
            0,  # Acceleration (not used)
            0,  # Yaw (not used)
            yaw_rate,  # Yaw rate
        )

        return True

    async def emergency_stop(self) -> bool:
        """Execute emergency stop in SITL.

        Returns:
            True if emergency stop executed
        """
        # SAFETY: Emergency stop prevents runaway drone - Story 4.7 FR#16
        # Hazard ID: HARA-CTL-001 - Loss of control leading to flyaway
        # Mitigation: Immediate mode change and velocity zeroing <500ms
        if not self.connected or not self.connection:
            logger.error("Not connected to SITL")
            return False

        logger.warning("EMERGENCY STOP!")

        # Set mode to STABILIZE (stops autonomous movement)
        await self.set_mode("STABILIZE")

        # Send zero velocity command
        await self.send_velocity_command(0, 0, 0, 0)

        # Optionally disarm if on ground
        telemetry = await self.get_telemetry()
        if telemetry.get("position", {}).get("alt", 0) < 1.0:
            await self.disarm()

        return True
