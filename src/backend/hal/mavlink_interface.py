"""
MAVLink Hardware Abstraction Layer for Cube Orange+
Uses pymavlink v2.4.49 (installed via uv)
"""

import asyncio
import logging
from dataclasses import dataclass

from pymavlink import mavutil

logger = logging.getLogger(__name__)


@dataclass
class MAVLinkConfig:
    """MAVLink configuration parameters"""

    device: str = "/dev/ttyACM0"  # Cube Orange+ primary
    baudrate: int = 115200
    source_system: int = 255
    source_component: int = 190
    target_system: int = 1
    target_component: int = 1


class MAVLinkInterface:
    """Hardware interface for Cube Orange+ via MAVLink"""

    def __init__(self, config: MAVLinkConfig):
        self.config = config
        self.connection: mavutil.mavlink_connection | None = None
        self._running = False
        self._heartbeat_task = None

    async def connect(self, device: str | None = None, baud: int | None = None):
        """Auto-discover and connect to Cube Orange"""
        device = device or self.config.device
        baud = baud or self.config.baudrate

        # Try primary and secondary ports
        ports_to_try = [
            "/dev/ttyACM0",  # CubeOrange+ primary (confirmed)
            "/dev/ttyACM1",  # CubeOrange+ secondary (confirmed)
            device,  # User-specified
        ]

        for port in ports_to_try:
            try:
                logger.info(f"Trying to connect to {port} at {baud} baud")

                # Create MAVLink connection
                self.connection = mavutil.mavlink_connection(
                    f"{port}:{baud}",
                    source_system=self.config.source_system,
                    source_component=self.config.source_component,
                )

                # Wait for heartbeat to confirm connection
                logger.info("Waiting for heartbeat...")
                msg = self.connection.wait_heartbeat(timeout=3)

                if msg:
                    logger.info(f"Connected to Cube Orange+ on {port}")
                    logger.info(f"Autopilot type: {msg.autopilot}")
                    logger.info(f"System ID: {msg.get_srcSystem()}")

                    self._running = True

                    # Start heartbeat task
                    self._heartbeat_task = asyncio.create_task(self._send_heartbeat())

                    return True

            except Exception as e:
                logger.debug(f"Failed to connect to {port}: {e}")
                continue

        logger.error("Failed to connect to Cube Orange+ on any port")
        return False

    async def _send_heartbeat(self):
        """Send heartbeat at 1 Hz"""
        while self._running:
            if self.connection:
                self.connection.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0
                )
            await asyncio.sleep(1)

    async def send_velocity_ned(self, vn: float, ve: float, vd: float, yaw_rate: float = 0):
        """Send velocity in NED frame (North-East-Down)
        Units: meters per second (m/s)
        Frame: MAV_FRAME_LOCAL_NED (body frame)
        """
        if not self.connection:
            return False

        try:
            # Type mask (only velocity and yaw rate)
            type_mask = (
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
            )

            # Send SET_POSITION_TARGET_LOCAL_NED
            self.connection.mav.set_position_target_local_ned_send(
                0,  # time_boot_ms (not used)
                self.config.target_system,
                self.config.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                type_mask,
                0,
                0,
                0,  # x, y, z (ignored)
                vn,
                ve,
                vd,  # vx, vy, vz
                0,
                0,
                0,  # ax, ay, az (ignored)
                0,  # yaw (ignored)
                yaw_rate,  # yaw_rate
            )

            return True

        except Exception as e:
            logger.error(f"Failed to send velocity command: {e}")
            return False

    async def get_position(self) -> tuple[float, float, float]:
        """Get GPS position from GLOBAL_POSITION_INT
        Returns: (lat, lon, alt) in degrees and meters
        """
        if not self.connection:
            return (0, 0, 0)

        msg = self.connection.recv_match(type="GLOBAL_POSITION_INT", blocking=False)
        if msg:
            lat = msg.lat / 1e7  # Convert from 1e7 degrees
            lon = msg.lon / 1e7
            alt = msg.alt / 1000  # Convert from mm to m
            return (lat, lon, alt)

        return (0, 0, 0)

    async def get_battery(self) -> dict:
        """Get battery status from SYS_STATUS
        Returns: dict with voltage, current, remaining
        """
        if not self.connection:
            return {"voltage": 0, "current": 0, "remaining": 0}

        msg = self.connection.recv_match(type="SYS_STATUS", blocking=False)
        if msg:
            # 6S Li-ion battery (18-25.2V range)
            voltage = msg.voltage_battery / 1000.0  # mV to V
            current = msg.current_battery / 100.0 if msg.current_battery != -1 else 0  # cA to A
            remaining = msg.battery_remaining if msg.battery_remaining != -1 else 0  # Percentage

            return {"voltage": voltage, "current": current, "remaining": remaining}

        return {"voltage": 0, "current": 0, "remaining": 0}

    async def get_flight_mode(self) -> str:
        """Get current flight mode"""
        if not self.connection:
            return "UNKNOWN"

        msg = self.connection.recv_match(type="HEARTBEAT", blocking=False)
        if msg:
            # Get flight mode from custom_mode
            mode_map = self.connection.mode_mapping()
            if mode_map:
                for mode_name, mode_num in mode_map.items():
                    if mode_num == msg.custom_mode:
                        return mode_name

            # Fallback to base mode
            if msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_GUIDED_ENABLED:
                return "GUIDED"

        return "UNKNOWN"

    async def get_rc_channels(self) -> dict:
        """Get RC channel values for override detection"""
        if not self.connection:
            return {}

        msg = self.connection.recv_match(type="RC_CHANNELS", blocking=False)
        if msg:
            return {
                "chan1": msg.chan1_raw,  # Roll
                "chan2": msg.chan2_raw,  # Pitch
                "chan3": msg.chan3_raw,  # Throttle
                "chan4": msg.chan4_raw,  # Yaw
                "chan5": msg.chan5_raw,  # Mode switch
                "chan6": msg.chan6_raw,  # Aux 1
                "chan7": msg.chan7_raw,  # Aux 2
                "chan8": msg.chan8_raw,  # Aux 3
            }

        return {}

    async def check_rc_override(self, deadband: int = 50) -> bool:
        """Check if RC sticks moved (override detection)
        Deadband: ±50 PWM units from center (1500)
        """
        channels = await self.get_rc_channels()

        if not channels:
            return False

        # Check main control channels (1-4)
        for i in range(1, 5):
            chan_val = channels.get(f"chan{i}", 1500)
            if abs(chan_val - 1500) > deadband:
                logger.warning(f"RC override detected on channel {i}: {chan_val}")
                return True

        return False

    async def get_gps_status(self) -> dict:
        """Get GPS status from GPS_RAW_INT"""
        if not self.connection:
            return {"fix_type": 0, "satellites": 0, "hdop": 99.99}

        msg = self.connection.recv_match(type="GPS_RAW_INT", blocking=False)
        if msg:
            return {
                "fix_type": msg.fix_type,  # 0=No GPS, 1=No Fix, 2=2D, 3=3D, 4=DGPS, 5=RTK Float, 6=RTK Fixed
                "satellites": msg.satellites_visible,
                "hdop": msg.eph / 100.0 if msg.eph != 65535 else 99.99,  # HDOP
            }

        return {"fix_type": 0, "satellites": 0, "hdop": 99.99}

    async def send_statustext(self, text: str, severity: int = 6):
        """Send status text to GCS
        Severity: 0=EMERGENCY, 1=ALERT, 2=CRITICAL, 3=ERROR, 4=WARNING, 5=NOTICE, 6=INFO, 7=DEBUG
        """
        if not self.connection:
            return False

        try:
            # Truncate to 50 chars (MAVLink limit)
            text = text[:50]

            self.connection.mav.statustext_send(severity, text.encode("utf-8"))
            return True

        except Exception as e:
            logger.error(f"Failed to send status text: {e}")
            return False

    async def set_mode(self, mode: str) -> bool:
        """Set flight mode"""
        if not self.connection:
            return False

        try:
            # Get mode number from string
            mode_map = self.connection.mode_mapping()
            if not mode_map or mode not in mode_map:
                logger.error(f"Unknown mode: {mode}")
                return False

            mode_num = mode_map[mode]

            # Send mode change command
            self.connection.mav.command_long_send(
                self.config.target_system,
                self.config.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                0,  # confirmation
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_num,
                0,
                0,
                0,
                0,
                0,
            )

            # Wait for ACK
            ack = self.connection.recv_match(type="COMMAND_ACK", blocking=True, timeout=1)
            if ack and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                logger.info(f"Mode changed to {mode}")
                return True

        except Exception as e:
            logger.error(f"Failed to set mode: {e}")

        return False

    async def arm(self) -> bool:
        """Arm the vehicle"""
        if not self.connection:
            return False

        try:
            self.connection.mav.command_long_send(
                self.config.target_system,
                self.config.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,  # confirmation
                1,  # arm
                0,
                0,
                0,
                0,
                0,
                0,
            )

            # Wait for ACK
            ack = self.connection.recv_match(type="COMMAND_ACK", blocking=True, timeout=1)
            if ack and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                logger.info("Vehicle armed")
                return True

        except Exception as e:
            logger.error(f"Failed to arm: {e}")

        return False

    async def disarm(self) -> bool:
        """Disarm the vehicle"""
        if not self.connection:
            return False

        try:
            self.connection.mav.command_long_send(
                self.config.target_system,
                self.config.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,  # confirmation
                0,  # disarm
                0,
                0,
                0,
                0,
                0,
                0,
            )

            # Wait for ACK
            ack = self.connection.recv_match(type="COMMAND_ACK", blocking=True, timeout=1)
            if ack and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                logger.info("Vehicle disarmed")
                return True

        except Exception as e:
            logger.error(f"Failed to disarm: {e}")

        return False

    async def close(self):
        """Close MAVLink connection"""
        self._running = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()

        if self.connection:
            try:
                self.connection.close()
                logger.info("MAVLink connection closed")
            except (ConnectionError, AttributeError) as e:
                logger.warning(f"Error closing MAVLink: {e}")
            finally:
                self.connection = None

    async def get_info(self) -> dict:
        """Get connection information"""
        if not self.connection:
            return {"status": "disconnected"}

        mode = await self.get_flight_mode()
        battery = await self.get_battery()
        gps = await self.get_gps_status()

        return {
            "status": "connected",
            "device": self.config.device,
            "baudrate": self.config.baudrate,
            "flight_mode": mode,
            "battery": battery,
            "gps": gps,
        }


# Auto-detection function
async def auto_detect_cube_orange() -> MAVLinkInterface | None:
    """Auto-detect and connect to Cube Orange+"""
    try:
        config = MAVLinkConfig()
        mavlink = MAVLinkInterface(config)

        if await mavlink.connect():
            logger.info("Cube Orange+ auto-detected and connected")

            # Send initial status
            await mavlink.send_statustext("PISAD Payload Connected", severity=6)

            return mavlink

    except Exception as e:
        logger.error(f"Cube Orange+ auto-detection failed: {e}")

    return None
