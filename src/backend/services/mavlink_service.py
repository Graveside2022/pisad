"""MAVLink communication service for drone telemetry and control."""

import asyncio
import logging
import time
from collections.abc import Callable
from enum import Enum
from typing import Any

from pymavlink import mavutil

from backend.core.exceptions import CallbackError, MAVLinkError, SafetyInterlockError
from src.backend.utils.doppler_compensation import PlatformVelocity

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """MAVLink logging verbosity levels."""

    ERROR = logging.ERROR
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    TRACE = 5  # Custom level below DEBUG


class ConnectionState(Enum):
    """MAVLink connection states."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"


class MAVLinkService:
    """Service for MAVLink communication with flight controller."""

    def __init__(
        self,
        device_path: str = "/dev/ttyACM0",
        baud_rate: int = 115200,
        source_system: int = 1,
        source_component: int = 191,  # MAV_COMP_ID_ONBOARD_COMPUTER
        target_system: int = 1,
        target_component: int = 1,
        log_level: LogLevel = LogLevel.INFO,
        log_messages: list[str] | None = None,
    ):
        """Initialize MAVLink service.

        Args:
            device_path: Serial device path or TCP connection string
            baud_rate: Serial baud rate (ignored for TCP)
            source_system: System ID for this system
            source_component: Component ID for this component
            target_system: Target system ID
            target_component: Target component ID
            log_level: Logging verbosity level
            log_messages: Specific message types to log (None = all)
        """
        self.device_path = device_path
        self.baud_rate = baud_rate
        self.source_system = source_system
        self.source_component = source_component
        self.target_system = target_system
        self.target_component = target_component
        self.log_level = log_level
        self.log_messages = log_messages if log_messages else []

        # Configure logging level
        logging.addLevelName(LogLevel.TRACE.value, "TRACE")
        logger.setLevel(log_level.value)

        self.connection: mavutil.mavlink_connection | None = None
        self.state = ConnectionState.DISCONNECTED
        # Alias for backwards compatibility with tests
        self.connection_state = self.state
        self.last_heartbeat_received = 0.0
        self.last_heartbeat_sent = 0.0
        self.heartbeat_timeout = 3.0  # seconds

        self._running = False
        self._tasks: list[asyncio.Task[None]] = []
        self._reconnect_delay = 1.0  # Initial reconnect delay
        self._max_reconnect_delay = 30.0

        # Telemetry data storage
        self.telemetry: dict[str, Any] = {
            "position": {"lat": 0.0, "lon": 0.0, "alt": 0.0},
            "velocity": {"vx": 0.0, "vy": 0.0, "vz": 0.0, "ground_speed": 0.0},
            "attitude": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            "battery": {"voltage": 0.0, "current": 0.0, "percentage": 0.0},
            "gps": {"fix_type": 0, "satellites": 0, "hdop": 0.0},
            "flight_mode": "UNKNOWN",
            "armed": False,
        }

        # Velocity command settings
        self._velocity_commands_enabled = False  # DISABLED by default for safety
        self._last_velocity_command_time = 0.0
        self._velocity_command_rate_limit = 0.1  # 10Hz max
        self._max_velocity = 5.0  # m/s max velocity for safety

        # Callbacks for state changes
        self._state_callbacks: list[Callable[[ConnectionState], None]] = []

        # Callbacks for mode changes
        self._mode_callbacks: list[Callable[[str], None]] = []

        # Callbacks for battery updates
        self._battery_callbacks: list[Callable[[float], None]] = []

        # Callbacks for position updates
        self._position_callbacks: list[Callable[[float, float], None]] = []

        # Safety check callback for velocity commands
        self._safety_check_callback: Callable[[], bool] | None = None

        # Telemetry streaming tasks
        self._telemetry_tasks: list[asyncio.Task[None]] = []
        self._rssi_value: float = -100.0  # Default noise floor
        self._current_rssi: float = -100.0  # Current RSSI for compatibility
        self._telemetry_config = {
            "rssi_rate_hz": 2.0,
            "health_interval_seconds": 10,
            "detection_throttle_ms": 500,
            # Legacy format for backwards compatibility
            "rate": 2.0,
            "precision": 1,
        }
        self._last_detection_time = 0.0
        self._last_state_sent = ""

    async def telemetry_sender(self) -> None:
        """Send telemetry messages to GCS at configured rates."""
        rssi_interval = 1.0 / self._telemetry_config["rssi_rate_hz"]
        last_rssi_time = 0.0
        last_health_time = 0.0

        while self._running:
            try:
                if not self.is_connected() or not self.connection:
                    await asyncio.sleep(0.1)
                    continue

                current_time = time.time()

                # Send RSSI at configured rate
                if current_time - last_rssi_time >= rssi_interval:
                    self.send_named_value_float("PISAD_RSSI", self._rssi_value, current_time)
                    last_rssi_time = current_time

                # Send health status every configured interval
                if (
                    current_time - last_health_time
                    >= self._telemetry_config["health_interval_seconds"]
                ):
                    await self._send_health_status()
                    last_health_time = current_time

                await asyncio.sleep(0.05)  # 20Hz loop rate

            except asyncio.CancelledError:
                break
            except (ConnectionError, TimeoutError, AttributeError) as e:
                logger.error(
                    f"Error in telemetry sender: {e}",
                    extra={"connection_state": self.connection_state.value},
                )
                await asyncio.sleep(1.0)

    def send_named_value_float(
        self, name: str, value: float, timestamp: float | None = None, time_ms: int | None = None
    ) -> bool:
        """Send NAMED_VALUE_FLOAT message for continuous telemetry.

        Args:
            name: Parameter name (max 10 chars)
            value: Float value to send
            timestamp: Optional timestamp (uses current time if not provided)

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.is_connected() or not self.connection:
            return False

        try:
            # Truncate name to 10 characters
            name = name[:10]

            # Use current time if not provided
            if timestamp is None and time_ms is None:
                timestamp = time.time()

            # Convert timestamp to milliseconds or use provided time_ms
            if time_ms is not None:
                time_boot_ms = time_ms
            else:
                # timestamp is guaranteed to be float here
                assert timestamp is not None
                time_boot_ms = int((timestamp % 86400) * 1000)

            self.connection.mav.named_value_float_send(time_boot_ms, name.encode("utf-8"), value)

            logger.debug(f"NAMED_VALUE_FLOAT sent: {name}={value:.2f}")
            return True
        except (AttributeError, ConnectionError, ValueError) as e:
            logger.error(
                f"Failed to send NAMED_VALUE_FLOAT: {e}", extra={"name": name, "value": value}
            )
            return False

    def send_state_change(self, state: str) -> bool:
        """Send state change notification via STATUSTEXT.

        Args:
            state: New state name

        Returns:
            True if sent successfully, False otherwise
        """
        if state == self._last_state_sent:
            return True  # Don't resend same state

        message = f"PISAD: State changed to {state}"
        success = self.send_statustext(message, severity=6)  # INFO level

        if success:
            self._last_state_sent = state

        return success

    def send_detection_event(self, rssi: float, confidence: float) -> bool:
        """Send detection event via STATUSTEXT with throttling.

        Args:
            rssi: Signal strength in dBm
            confidence: Detection confidence percentage

        Returns:
            True if sent successfully, False otherwise
        """
        current_time = time.time()
        time_since_last = (current_time - self._last_detection_time) * 1000  # ms

        # Throttle detection messages
        if time_since_last < self._telemetry_config["detection_throttle_ms"]:
            return False

        message = f"PISAD: Signal detected {rssi:.1f}dBm @ {confidence:.0f}%"
        success = self.send_statustext(message, severity=5)  # NOTICE level

        if success:
            self._last_detection_time = current_time

        return success

    async def _send_health_status(self) -> None:
        """Send system health status via STATUSTEXT."""
        try:
            # Import psutil for system monitoring
            import json

            import psutil

            # Get system health metrics
            health = {
                "cpu": round(psutil.cpu_percent(interval=0.1), 1),
                "mem": round(psutil.virtual_memory().percent, 1),
                "sdr": "OK" if hasattr(self, "_sdr_connected") and self._sdr_connected else "ERR",
            }

            # Get CPU temperature on Raspberry Pi
            try:
                with open("/sys/class/thermal/thermal_zone0/temp") as f:
                    temp = int(f.read()) / 1000.0
                    health["temp"] = round(temp, 1)
            except (OSError, ValueError):
                pass  # Not on Pi or can't read temp

            # Format as compact JSON
            health_json = json.dumps(health, separators=(",", ":"))
            message = f"PISAD: Health {health_json}"

            # Send with INFO severity
            self.send_statustext(message[:50], severity=6)

            # Log warnings for high resource usage
            if health["cpu"] > 80:
                logger.warning(f"High CPU usage: {health['cpu']}%")
            if health["mem"] > 85:
                logger.warning(f"High memory usage: {health['mem']}%")
            if health.get("temp", 0) > 80:
                logger.warning(f"High CPU temperature: {health.get('temp')}°C")

        except (AttributeError, KeyError, ConnectionError) as e:
            logger.error(
                f"Failed to send health status: {e}", extra={"error_type": type(e).__name__}
            )

    def update_rssi_value(self, rssi: float) -> None:
        """Update RSSI value for telemetry streaming.

        Args:
            rssi: Current RSSI value in dBm
        """
        self._rssi_value = rssi
        self._current_rssi = rssi  # For backwards compatibility

    def update_telemetry_config(self, config: dict[str, Any]) -> None:
        """Update telemetry configuration.

        Args:
            config: Dictionary with telemetry configuration values
        """
        if "rssi_rate_hz" in config:
            self._telemetry_config["rssi_rate_hz"] = max(0.1, min(10.0, config["rssi_rate_hz"]))
        if "health_interval_seconds" in config:
            self._telemetry_config["health_interval_seconds"] = max(
                1, min(60, config["health_interval_seconds"])
            )
        if "detection_throttle_ms" in config:
            self._telemetry_config["detection_throttle_ms"] = max(
                100, min(5000, config["detection_throttle_ms"])
            )

        # Handle legacy format
        if "rate" in config:
            self._telemetry_config["rssi_rate_hz"] = config["rate"]
            self._telemetry_config["rate"] = config["rate"]
        if "precision" in config:
            self._telemetry_config["precision"] = config["precision"]

        logger.info(f"Telemetry config updated: {self._telemetry_config}")

    def get_telemetry_config(self) -> dict[str, Any]:
        """Get current telemetry configuration."""
        return self._telemetry_config.copy()

    def add_state_callback(self, callback: Callable[[ConnectionState], None]) -> None:
        """Add callback for connection state changes."""
        self._state_callbacks.append(callback)

    def add_mode_callback(self, callback: Callable[[str], None]) -> None:
        """Add callback for flight mode changes."""
        self._mode_callbacks.append(callback)

    def add_battery_callback(self, callback: Callable[[float], None]) -> None:
        """Add callback for battery percentage updates."""
        self._battery_callbacks.append(callback)

    def add_position_callback(self, callback: Callable[[float, float], None]) -> None:
        """Add callback for position updates."""
        self._position_callbacks.append(callback)

    def set_safety_check_callback(self, callback: Callable[[], bool]) -> None:
        """Set callback for safety checks before velocity commands.

        Args:
            callback: Function that returns True if safe to proceed
        """
        self._safety_check_callback = callback

    def _set_state(self, new_state: ConnectionState) -> None:
        """Update connection state and notify callbacks."""
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            self.connection_state = new_state  # Update alias
            logger.info(f"MAVLink connection state changed: {old_state.value} -> {new_state.value}")

            for callback in self._state_callbacks:
                try:
                    callback(new_state)
                except CallbackError as e:
                    logger.error(f"Error in state callback: {e}")

    async def start(self) -> None:
        """Start MAVLink service and connection tasks."""
        if self._running:
            logger.warning("MAVLink service already running")
            return

        self._running = True
        logger.info(f"Starting MAVLink service on {self.device_path}")

        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._connection_manager()),
            asyncio.create_task(self._heartbeat_sender()),
            asyncio.create_task(self._message_receiver()),
            asyncio.create_task(self._connection_monitor()),
            asyncio.create_task(self.telemetry_sender()),
        ]

    async def stop(self) -> None:
        """Stop MAVLink service and cleanup."""
        self._running = False

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        # Close connection
        if self.connection:
            self.connection.close()
            self.connection = None

        self._set_state(ConnectionState.DISCONNECTED)
        logger.info("MAVLink service stopped")

    async def _connection_manager(self) -> None:
        """Manage MAVLink connection with automatic reconnection."""
        while self._running:
            try:
                if self.state == ConnectionState.DISCONNECTED:
                    await self._connect()

                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except MAVLinkError as e:
                logger.error(f"Connection manager error: {e}")
                await asyncio.sleep(1)

    async def _connect(self) -> None:
        """Establish MAVLink connection."""
        self._set_state(ConnectionState.CONNECTING)

        try:
            # Determine connection type
            if self.device_path.startswith("tcp:"):
                # TCP connection for SITL
                logger.info(f"Connecting to SITL at {self.device_path}")
                self.connection = mavutil.mavlink_connection(
                    self.device_path,
                    source_system=self.source_system,
                    source_component=self.source_component,
                )
            else:
                # Serial connection for hardware
                logger.info(
                    f"Connecting to serial device {self.device_path} at {self.baud_rate} baud"
                )
                self.connection = mavutil.mavlink_connection(
                    self.device_path,
                    baud=self.baud_rate,
                    source_system=self.source_system,
                    source_component=self.source_component,
                )

            # Wait for heartbeat to confirm connection
            msg = self.connection.wait_heartbeat(timeout=5)
            if msg:
                self.last_heartbeat_received = time.time()
                self._set_state(ConnectionState.CONNECTED)
                self._reconnect_delay = 1.0  # Reset reconnect delay
                logger.info(f"MAVLink connected to system {msg.get_srcSystem()}")
            else:
                raise MAVLinkError("No heartbeat received")

        except MAVLinkError as e:
            logger.error(f"Failed to connect: {e}")
            self._set_state(ConnectionState.DISCONNECTED)

            if self.connection:
                self.connection.close()
                self.connection = None

            # Exponential backoff for reconnection
            await asyncio.sleep(self._reconnect_delay)
            self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)

    async def _heartbeat_sender(self) -> None:
        """Send heartbeat messages at 1Hz."""
        while self._running:
            try:
                if self.state == ConnectionState.CONNECTED and self.connection:
                    # Send heartbeat
                    self.connection.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                        0,  # base_mode
                        0,  # custom_mode
                        mavutil.mavlink.MAV_STATE_ACTIVE,
                    )
                    self.last_heartbeat_sent = time.time()
                    logger.debug("Heartbeat sent")

                await asyncio.sleep(1.0)  # 1Hz rate

            except asyncio.CancelledError:
                break
            except MAVLinkError as e:
                logger.error(f"Error sending heartbeat: {e}")
                await asyncio.sleep(1.0)

    async def _message_receiver(self) -> None:
        """Receive and process MAVLink messages."""
        while self._running:
            try:
                if self.state == ConnectionState.CONNECTED and self.connection:
                    # Non-blocking message receive
                    msg = self.connection.recv_match(blocking=False)

                    if msg:
                        await self._process_message(msg)

                await asyncio.sleep(0.01)  # Small delay to prevent CPU spinning

            except asyncio.CancelledError:
                break
            except MAVLinkError as e:
                logger.error(f"Error receiving message: {e}")
                await asyncio.sleep(0.1)

    async def _process_message(self, msg: Any) -> None:
        """Process received MAVLink message."""
        msg_type = msg.get_type()

        # Log message at TRACE level if enabled
        if self.log_level == LogLevel.TRACE and (
            not self.log_messages or msg_type in self.log_messages
        ):
            logger.log(LogLevel.TRACE.value, f"Received {msg_type}: {msg.to_dict()}")

        if msg_type == "HEARTBEAT":
            self.last_heartbeat_received = time.time()
            self._process_heartbeat(msg)
        elif msg_type == "GLOBAL_POSITION_INT":
            self._process_global_position(msg)
        elif msg_type == "ATTITUDE":
            self._process_attitude(msg)
        elif msg_type == "SYS_STATUS":
            self._process_sys_status(msg)
        elif msg_type == "GPS_RAW_INT":
            self._process_gps_raw(msg)

    def _process_heartbeat(self, msg: Any) -> None:
        """Process HEARTBEAT message."""
        # Extract flight mode
        mode = msg.custom_mode
        armed = msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED

        new_mode = self._get_flight_mode_name(mode)
        old_mode = self.telemetry["flight_mode"]

        self.telemetry["flight_mode"] = new_mode
        self.telemetry["armed"] = bool(armed)

        # Notify mode callbacks if mode changed
        if new_mode != old_mode:
            for callback in self._mode_callbacks:
                try:
                    callback(new_mode)
                except CallbackError as e:
                    logger.error(f"Error in mode callback: {e}")

    def _process_global_position(self, msg: Any) -> None:
        """Process GLOBAL_POSITION_INT message."""
        lat = msg.lat / 1e7
        lon = msg.lon / 1e7

        self.telemetry["position"]["lat"] = lat
        self.telemetry["position"]["lon"] = lon
        self.telemetry["position"]["alt"] = msg.alt / 1000.0  # mm to meters
        
        # Extract velocity data (convert from cm/s to m/s)
        vx_ms = msg.vx / 100.0 if hasattr(msg, 'vx') else 0.0  # North velocity
        vy_ms = msg.vy / 100.0 if hasattr(msg, 'vy') else 0.0  # East velocity  
        vz_ms = msg.vz / 100.0 if hasattr(msg, 'vz') else 0.0  # Down velocity
        
        # Calculate ground speed
        ground_speed_ms = (vx_ms**2 + vy_ms**2)**0.5
        
        self.telemetry["velocity"]["vx"] = vx_ms
        self.telemetry["velocity"]["vy"] = vy_ms
        self.telemetry["velocity"]["vz"] = vz_ms
        self.telemetry["velocity"]["ground_speed"] = ground_speed_ms

        # Notify position callbacks
        for callback in self._position_callbacks:
            try:
                callback(lat, lon)
            except CallbackError as e:
                logger.error(f"Error in position callback: {e}")

    def _process_attitude(self, msg: Any) -> None:
        """Process ATTITUDE message."""
        import math

        self.telemetry["attitude"]["roll"] = math.degrees(msg.roll)
        self.telemetry["attitude"]["pitch"] = math.degrees(msg.pitch)
        self.telemetry["attitude"]["yaw"] = math.degrees(msg.yaw)

    def _process_sys_status(self, msg: Any) -> None:
        """Process SYS_STATUS message."""
        if msg.voltage_battery != -1:
            self.telemetry["battery"]["voltage"] = msg.voltage_battery / 1000.0  # mV to V
        if msg.current_battery != -1:
            self.telemetry["battery"]["current"] = msg.current_battery / 100.0  # cA to A
        if msg.battery_remaining != -1:
            old_percentage = self.telemetry["battery"]["percentage"]
            new_percentage = msg.battery_remaining
            self.telemetry["battery"]["percentage"] = new_percentage

            # Notify battery callbacks if percentage changed
            if new_percentage != old_percentage:
                for callback in self._battery_callbacks:
                    try:
                        callback(new_percentage)
                    except CallbackError as e:
                        logger.error(f"Error in battery callback: {e}")

    def _process_gps_raw(self, msg: Any) -> None:
        """Process GPS_RAW_INT message."""
        self.telemetry["gps"]["fix_type"] = msg.fix_type
        self.telemetry["gps"]["satellites"] = msg.satellites_visible
        self.telemetry["gps"]["hdop"] = msg.eph / 100.0  # cm to m

    def _get_flight_mode_name(self, mode: int) -> str:
        """Convert flight mode number to name."""
        # ArduCopter flight modes
        modes = {
            0: "STABILIZE",
            1: "ACRO",
            2: "ALT_HOLD",
            3: "AUTO",
            4: "GUIDED",
            5: "LOITER",
            6: "RTL",
            7: "CIRCLE",
            9: "LAND",
            11: "DRIFT",
            13: "SPORT",
            14: "FLIP",
            15: "AUTOTUNE",
            16: "POSHOLD",
            17: "BRAKE",
            18: "THROW",
            19: "AVOID_ADSB",
            20: "GUIDED_NOGPS",
            21: "SMART_RTL",
            22: "FLOWHOLD",
            23: "FOLLOW",
            24: "ZIGZAG",
            25: "SYSTEMID",
            26: "AUTOROTATE",
            27: "AUTO_RTL",
        }
        return modes.get(mode, "UNKNOWN")

    async def _connection_monitor(self) -> None:
        """Monitor connection health and trigger reconnection if needed."""
        while self._running:
            try:
                if self.state == ConnectionState.CONNECTED:
                    # Check heartbeat timeout
                    time_since_heartbeat = time.time() - self.last_heartbeat_received

                    if time_since_heartbeat > self.heartbeat_timeout:
                        logger.warning(f"Heartbeat timeout ({time_since_heartbeat:.1f}s)")
                        self._set_state(ConnectionState.DISCONNECTED)

                        if self.connection:
                            self.connection.close()
                            self.connection = None

                await asyncio.sleep(1.0)

            except asyncio.CancelledError:
                break
            except MAVLinkError as e:
                logger.error(f"Connection monitor error: {e}")
                await asyncio.sleep(1.0)

    def get_telemetry(self) -> dict[str, Any]:
        """Get current telemetry data."""
        return {
            **self.telemetry,
            "connected": self.state == ConnectionState.CONNECTED,
            "connection_state": self.state.value,
        }

    def get_platform_velocity(self) -> PlatformVelocity | None:
        """Get current platform velocity for Doppler compensation.
        
        Returns:
            PlatformVelocity object with velocity components or None if unavailable
        """
        velocity = self.telemetry.get("velocity")
        if not velocity:
            return None
            
        return PlatformVelocity(
            vx_ms=velocity["vx"],
            vy_ms=velocity["vy"], 
            vz_ms=velocity["vz"],
            ground_speed_ms=velocity["ground_speed"]
        )

    def is_connected(self) -> bool:
        """Check if MAVLink is connected."""
        return self.state == ConnectionState.CONNECTED

    def send_statustext(self, text: str, severity: int = 6) -> bool:
        """Send STATUSTEXT message to ground control station.

        Args:
            text: Status text message (max 50 chars)
            severity: MAV_SEVERITY level (0=emergency, 6=info, 7=debug)

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.is_connected() or not self.connection:
            return False

        try:
            # Truncate text to 50 characters
            text = text[:50]

            self.connection.mav.statustext_send(severity, text.encode("utf-8"))

            logger.debug(f"STATUSTEXT sent: {text}")
            return True
        except MAVLinkError as e:
            logger.error(f"Failed to send STATUSTEXT: {e}")
            return False

    def enable_velocity_commands(self, enable: bool = True) -> None:
        """Enable or disable velocity command sending.

        Args:
            enable: True to enable, False to disable (default is disabled for safety)
        """
        self._velocity_commands_enabled = enable
        logger.warning(f"Velocity commands {'ENABLED' if enable else 'DISABLED'}")

    async def send_velocity_command(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        vz: float = 0.0,
        yaw_rate: float = 0.0,
        coordinate_frame: int = mavutil.mavlink.MAV_FRAME_LOCAL_NED,
    ) -> bool:
        """Send velocity command to flight controller.

        Args:
            vx: Velocity in X direction (m/s) - positive is forward
            vy: Velocity in Y direction (m/s) - positive is right
            vz: Velocity in Z direction (m/s) - positive is down
            yaw_rate: Yaw rate (rad/s)
            coordinate_frame: MAVLink coordinate frame

        Returns:
            True if command was sent, False otherwise
        """
        # Safety check - commands must be explicitly enabled
        if not self._velocity_commands_enabled:
            logger.warning("Velocity commands are disabled for safety")
            return False

        # Check connection
        if not self.is_connected() or not self.connection:
            logger.warning("Cannot send velocity command - not connected")
            return False

        # Check external safety callback if provided
        if self._safety_check_callback:
            try:
                if not self._safety_check_callback():
                    logger.warning("Velocity command blocked by safety interlock")
                    return False
            except SafetyInterlockError as e:
                logger.error(f"Error in safety check callback: {e}")
                return False

        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self._last_velocity_command_time
        if time_since_last < self._velocity_command_rate_limit:
            logger.debug(f"Rate limiting velocity command ({time_since_last:.3f}s since last)")
            return False

        # Velocity bounds checking
        vx = max(-self._max_velocity, min(self._max_velocity, vx))
        vy = max(-self._max_velocity, min(self._max_velocity, vy))
        vz = max(-self._max_velocity, min(self._max_velocity, vz))

        try:
            # Build SET_POSITION_TARGET_LOCAL_NED message
            type_mask = (
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
                | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
            )

            self.connection.mav.set_position_target_local_ned_send(
                0,  # time_boot_ms (not used)
                self.target_system,
                self.target_component,
                coordinate_frame,
                type_mask,
                0,
                0,
                0,  # position (ignored)
                vx,
                vy,
                vz,  # velocity
                0,
                0,
                0,  # acceleration (ignored)
                0,  # yaw (ignored)
                yaw_rate,  # yaw_rate
            )

            self._last_velocity_command_time = current_time
            logger.debug(
                f"Velocity command sent: vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}, yaw_rate={yaw_rate:.2f}"
            )
            return True

        except MAVLinkError as e:
            logger.error(f"Failed to send velocity command: {e}")
            return False

    async def send_heartbeat(self) -> bool:
        """Send heartbeat message for performance testing.

        Returns:
            bool: True if heartbeat sent successfully, False otherwise
        """
        if not self.connection:
            return False

        try:
            self.connection.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0,
                0,
                0,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send heartbeat: {e}")
            return False

    def get_gps_status_string(self) -> str:
        """Get GPS status as string."""
        fix_type = self.telemetry["gps"]["fix_type"]
        if fix_type == 0 or fix_type == 1:
            return "NO_FIX"
        elif fix_type == 2:
            return "2D_FIX"
        elif fix_type == 3:
            return "3D_FIX"
        elif fix_type >= 4:
            return "RTK"
        return "UNKNOWN"

    async def upload_mission(self, waypoints: list[dict[str, float]]) -> bool:
        """Upload mission waypoints to flight controller.

        Args:
            waypoints: List of waypoints with lat, lon, alt fields

        Returns:
            True if upload successful, False otherwise
        """
        if not self.connection:
            logger.error("Cannot upload mission: not connected")
            return False

        try:
            # Request mission clear
            self.connection.mav.mission_clear_all_send(self.target_system, self.target_component)

            # Wait for acknowledgment
            ack = self.connection.recv_match(type="MISSION_ACK", blocking=True, timeout=2)
            if not ack:
                logger.error("Mission clear not acknowledged")
                return False

            # Send mission count
            self.connection.mav.mission_count_send(
                self.target_system, self.target_component, len(waypoints)
            )

            # Upload each waypoint
            for seq, wp in enumerate(waypoints):
                # Wait for mission request
                req = self.connection.recv_match(type="MISSION_REQUEST", blocking=True, timeout=2)
                if not req or req.seq != seq:
                    logger.error(f"Mission request mismatch for waypoint {seq}")
                    return False

                # Send waypoint
                self.connection.mav.mission_item_send(
                    self.target_system,
                    self.target_component,
                    seq,
                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                    mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                    0,  # current (0 = not current waypoint)
                    1,  # autocontinue
                    0,  # param1 - hold time
                    0,  # param2 - acceptance radius
                    0,  # param3 - pass radius
                    float("nan"),  # param4 - yaw
                    wp["lat"],
                    wp["lon"],
                    wp.get("alt", 50.0),
                )

            # Wait for final acknowledgment
            final_ack = self.connection.recv_match(type="MISSION_ACK", blocking=True, timeout=2)
            if not final_ack:
                logger.error("Mission upload not acknowledged")
                return False

            if final_ack.type != mavutil.mavlink.MAV_MISSION_ACCEPTED:
                logger.error(f"Mission upload failed with type: {final_ack.type}")
                return False

            logger.info(f"Successfully uploaded {len(waypoints)} waypoints")
            return True

        except MAVLinkError as e:
            logger.error(f"Failed to upload mission: {e}")
            return False

    async def start_mission(self) -> bool:
        """Start the uploaded mission.

        Returns:
            True if mission started, False otherwise
        """
        if not self.connection:
            logger.error("Cannot start mission: not connected")
            return False

        try:
            # Set mode to AUTO
            mode_id = self.connection.mode_mapping().get("AUTO", None)
            if mode_id is None:
                logger.error("AUTO mode not available")
                return False

            self.connection.mav.set_mode_send(
                self.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode_id
            )

            # Arm the vehicle if not armed
            if not self.telemetry["armed"]:
                self.connection.mav.command_long_send(
                    self.target_system,
                    self.target_component,
                    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                    0,
                    1,  # arm
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                )

            # Start mission
            self.connection.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_CMD_MISSION_START,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            )

            logger.info("Mission start command sent")
            return True

        except MAVLinkError as e:
            logger.error(f"Failed to start mission: {e}")
            return False

    async def pause_mission(self) -> bool:
        """Pause the current mission.

        Returns:
            True if mission paused, False otherwise
        """
        if not self.connection:
            logger.error("Cannot pause mission: not connected")
            return False

        try:
            # Send pause command
            self.connection.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_CMD_DO_PAUSE_CONTINUE,
                0,
                0,  # 0 = pause, 1 = continue
                0,
                0,
                0,
                0,
                0,
                0,
            )

            logger.info("Mission pause command sent")
            return True

        except MAVLinkError as e:
            logger.error(f"Failed to pause mission: {e}")
            return False

    async def resume_mission(self) -> bool:
        """Resume the paused mission.

        Returns:
            True if mission resumed, False otherwise
        """
        if not self.connection:
            logger.error("Cannot resume mission: not connected")
            return False

        try:
            # Send continue command
            self.connection.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_CMD_DO_PAUSE_CONTINUE,
                0,
                1,  # 0 = pause, 1 = continue
                0,
                0,
                0,
                0,
                0,
                0,
            )

            logger.info("Mission resume command sent")
            return True

        except MAVLinkError as e:
            logger.error(f"Failed to resume mission: {e}")
            return False

    async def stop_mission(self) -> bool:
        """Stop the current mission and return to LOITER mode.

        Returns:
            True if mission stopped, False otherwise
        """
        if not self.connection:
            logger.error("Cannot stop mission: not connected")
            return False

        try:
            # Set mode to LOITER
            mode_id = self.connection.mode_mapping().get("LOITER", None)
            if mode_id is None:
                # Fallback to GUIDED if LOITER not available
                mode_id = self.connection.mode_mapping().get("GUIDED", None)

            if mode_id is None:
                logger.error("No suitable mode for mission stop")
                return False

            self.connection.mav.set_mode_send(
                self.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode_id
            )

            logger.info("Mission stopped, switched to LOITER/GUIDED mode")
            return True

        except MAVLinkError as e:
            logger.error(f"Failed to stop mission: {e}")
            return False

    def get_mission_progress(self) -> tuple[int, int]:
        """Get current mission progress.

        Returns:
            Tuple of (current_waypoint, total_waypoints)
        """
        if not self.connection:
            return (0, 0)

        try:
            # Request mission current
            self.connection.mav.mission_request_int_send(
                self.target_system,
                self.target_component,
                0,  # request current waypoint
            )

            # Get response
            msg = self.connection.recv_match(type="MISSION_CURRENT", blocking=True, timeout=1)
            if msg:
                current_wp = msg.seq

                # Get total count
                self.connection.mav.mission_request_list_send(
                    self.target_system, self.target_component
                )
                count_msg = self.connection.recv_match(
                    type="MISSION_COUNT", blocking=True, timeout=1
                )

                if count_msg:
                    return (current_wp, count_msg.count)

            return (0, 0)

        except MAVLinkError as e:
            logger.error(f"Failed to get mission progress: {e}")
            return (0, 0)

    def set_log_level(self, level: LogLevel) -> None:
        """Set logging verbosity level.

        Args:
            level: New logging level
        """
        self.log_level = level
        logger.setLevel(level.value)
        logger.info(f"MAVLink logging level set to {level.name}")

    def set_log_filters(self, message_types: list[str] | None = None) -> None:
        """Set message type filters for logging.

        Args:
            message_types: List of message types to log (None = all)
        """
        self.log_messages = message_types if message_types else []
        if self.log_messages:
            logger.info(f"MAVLink logging filtered to: {', '.join(self.log_messages)}")
        else:
            logger.info("MAVLink logging all message types")

    # API Methods for Story 4.5 Implementation

    def connect(self, connection_string: str | None = None) -> bool:
        """Establish MAVLink connection synchronously.

        Args:
            connection_string: Connection string (e.g., "tcp:127.0.0.1:5760" or "/dev/ttyACM0:57600")
                             If None, uses device_path and baud_rate from init

        Returns:
            True if connection successful, False otherwise
        """
        if self.state == ConnectionState.CONNECTED:
            logger.info("Already connected to MAVLink")
            return True

        # Parse connection string if provided
        if connection_string:
            if ":" in connection_string:
                if connection_string.startswith("tcp:"):
                    self.device_path = connection_string
                else:
                    # Serial connection with baud rate
                    parts = connection_string.rsplit(":", 1)
                    self.device_path = parts[0]
                    try:
                        self.baud_rate = int(parts[1])
                    except ValueError:
                        logger.error(f"Invalid baud rate in connection string: {parts[1]}")
                        return False
            else:
                self.device_path = connection_string

        try:
            self._set_state(ConnectionState.CONNECTING)

            # Determine connection type
            if self.device_path.startswith("tcp:"):
                # TCP connection for SITL
                logger.info(f"Connecting to SITL at {self.device_path}")
                self.connection = mavutil.mavlink_connection(
                    self.device_path,
                    source_system=self.source_system,
                    source_component=self.source_component,
                )
            else:
                # Serial connection for hardware
                logger.info(
                    f"Connecting to serial device {self.device_path} at {self.baud_rate} baud"
                )
                self.connection = mavutil.mavlink_connection(
                    self.device_path,
                    baud=self.baud_rate,
                    source_system=self.source_system,
                    source_component=self.source_component,
                )

            # Wait for heartbeat to confirm connection
            msg = self.connection.wait_heartbeat(timeout=5)
            if msg:
                self.last_heartbeat_received = time.time()
                self._set_state(ConnectionState.CONNECTED)
                logger.info(f"MAVLink connected to system {msg.get_srcSystem()}")
                return True
            else:
                raise MAVLinkError("No heartbeat received")

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self._set_state(ConnectionState.DISCONNECTED)
            if self.connection:
                self.connection.close()
                self.connection = None
            return False

    def disconnect(self) -> None:
        """Close MAVLink connection and clean up resources."""
        if self.connection:
            logger.info("Disconnecting MAVLink...")
            try:
                # Send final heartbeat
                if self.state == ConnectionState.CONNECTED:
                    self.connection.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                        0,
                        0,
                        mavutil.mavlink.MAV_STATE_POWEROFF,
                    )

                # Close connection
                self.connection.close()
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
            finally:
                self.connection = None
                self._set_state(ConnectionState.DISCONNECTED)
                logger.info("MAVLink disconnected")

    def send_telemetry(self, telemetry_data: dict[str, Any]) -> None:
        """Send telemetry data via MAVLink.

        Args:
            telemetry_data: Dictionary containing telemetry values to send
        """
        if not self.connection or self.state != ConnectionState.CONNECTED:
            logger.warning("Cannot send telemetry: not connected")
            return

        try:
            # Send different telemetry types based on data keys
            if "rssi" in telemetry_data:
                self.send_named_value_float("RSSI", telemetry_data["rssi"])

            if "snr" in telemetry_data:
                self.send_named_value_float("SNR", telemetry_data["snr"])

            if "confidence" in telemetry_data:
                self.send_named_value_float("CONF", telemetry_data["confidence"])

            if "state" in telemetry_data:
                self.send_state_change(telemetry_data["state"])

            if "detection" in telemetry_data:
                det = telemetry_data["detection"]
                self.send_detection_event(
                    rssi=det.get("rssi", -100.0), confidence=det.get("confidence", 0.0)
                )

            if "status_text" in telemetry_data:
                self.send_statustext(telemetry_data["status_text"])

        except Exception as e:
            logger.error(f"Failed to send telemetry: {e}")

    async def send_detection_telemetry(
        self, rssi: float, snr: float, confidence: float, state: str
    ) -> None:
        """Send detection event telemetry asynchronously.

        Args:
            rssi: Signal strength in dBm
            snr: Signal-to-noise ratio in dB
            confidence: Detection confidence percentage
            state: Current system state
        """
        if not self.connection or self.state != ConnectionState.CONNECTED:
            logger.warning("Cannot send detection telemetry: not connected")
            return

        try:
            # Send multiple telemetry values
            self.send_named_value_float("DET_RSSI", rssi)
            self.send_named_value_float("DET_SNR", snr)
            self.send_named_value_float("DET_CONF", confidence)

            # Send status text with detection info
            text = f"Detection: RSSI={rssi:.1f}dBm SNR={snr:.1f}dB Conf={confidence:.0f}%"
            self.send_statustext(text, severity=6)  # INFO level

            # Log detection event
            logger.info(f"Sent detection telemetry: {text}")

        except Exception as e:
            logger.error(f"Failed to send detection telemetry: {e}")

    def process_gcs_command(self, command: dict[str, Any]) -> dict[str, Any]:
        """Process command from GCS.

        SAFETY: GCS override is critical safety feature for operator control
        HAZARD: HARA-CMD-001 - Failed override causing loss of operator control
        HAZARD: HARA-CMD-002 - Command injection leading to unauthorized control

        Args:
            command: Command dictionary with type and parameters

        Returns:
            Result dictionary with status
        """
        result = {"status": "unknown"}

        try:
            cmd_type = command.get("command")

            if cmd_type == "MANUAL_OVERRIDE":
                # Handle manual override
                self._control_mode = command.get("mode", "MANUAL")
                result["status"] = "accepted"
                logger.info(f"GCS override to {self._control_mode} mode")

            elif cmd_type == "AUTO_HOMING":
                # Check if manual override active
                if self._control_mode == "MANUAL":
                    result["status"] = "rejected"
                    result["reason"] = "Manual override active"
                else:
                    result["status"] = "accepted"

            else:
                result["status"] = "unknown_command"

        except Exception as e:
            logger.error(f"Error processing GCS command: {e}")
            result["status"] = "error"
            result["error"] = str(e)

        return result

    def get_control_mode(self) -> str:
        """Get current control mode.

        Returns:
            Current control mode string
        """
        if not hasattr(self, "_control_mode"):
            self._control_mode = "AUTO"
        return self._control_mode

    async def send_signal_lost_telemetry(self) -> None:
        """Send signal lost event telemetry."""
        if not self.connection or self.state != ConnectionState.CONNECTED:
            return

        try:
            self.send_statustext("Signal lost - returning to search", severity=5)  # NOTICE level
            self.send_named_value_float("SIGNAL", 0.0)  # Signal indicator = 0
        except Exception as e:
            logger.error(f"Failed to send signal lost telemetry: {e}")
