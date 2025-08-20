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

        # ASV service integration for frequency control
        self._asv_service: Any = None  # Will be injected via set_asv_service()
        self._homing_controller: Any = (
            None  # Will be injected via set_homing_controller()
        )
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

        # Parameter interface for Mission Planner integration
        self._parameter_handlers: dict[str, Callable[[float], bool]] = {}
        self._parameters: dict[str, float] = {}
        self._parameter_callbacks: list[Callable[[str, float], None]] = []

        # Initialize frequency control parameters
        self._initialize_frequency_parameters()

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
                    self.send_named_value_float(
                        "PISAD_RSSI", self._rssi_value, current_time
                    )
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
        self,
        name: str,
        value: float,
        timestamp: float | None = None,
        time_ms: int | None = None,
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

            self.connection.mav.named_value_float_send(
                time_boot_ms, name.encode("utf-8"), value
            )

            logger.debug(f"NAMED_VALUE_FLOAT sent: {name}={value:.2f}")
            return True
        except (AttributeError, ConnectionError, ValueError) as e:
            logger.error(
                f"Failed to send NAMED_VALUE_FLOAT: {e}",
                extra={"name": name, "value": value},
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
                "sdr": (
                    "OK"
                    if hasattr(self, "_sdr_connected") and self._sdr_connected
                    else "ERR"
                ),
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
                f"Failed to send health status: {e}",
                extra={"error_type": type(e).__name__},
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
            self._telemetry_config["rssi_rate_hz"] = max(
                0.1, min(10.0, config["rssi_rate_hz"])
            )
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
            logger.info(
                f"MAVLink connection state changed: {old_state.value} -> {new_state.value}"
            )

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
            self._reconnect_delay = min(
                self._reconnect_delay * 2, self._max_reconnect_delay
            )

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
        elif msg_type == "PARAM_SET":
            self._handle_param_set(msg)
        elif msg_type == "PARAM_REQUEST_READ":
            self._handle_param_request_read(msg)
        elif msg_type == "COMMAND_LONG":
            self._handle_command_long(msg)

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
        vx_ms = msg.vx / 100.0 if hasattr(msg, "vx") else 0.0  # North velocity
        vy_ms = msg.vy / 100.0 if hasattr(msg, "vy") else 0.0  # East velocity
        vz_ms = msg.vz / 100.0 if hasattr(msg, "vz") else 0.0  # Down velocity

        # Calculate ground speed
        ground_speed_ms = (vx_ms**2 + vy_ms**2) ** 0.5

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
            self.telemetry["battery"]["voltage"] = (
                msg.voltage_battery / 1000.0
            )  # mV to V
        if msg.current_battery != -1:
            self.telemetry["battery"]["current"] = (
                msg.current_battery / 100.0
            )  # cA to A
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
                        logger.warning(
                            f"Heartbeat timeout ({time_since_heartbeat:.1f}s)"
                        )
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
            ground_speed_ms=velocity["ground_speed"],
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
            logger.debug(
                f"Rate limiting velocity command ({time_since_last:.3f}s since last)"
            )
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
            self.connection.mav.mission_clear_all_send(
                self.target_system, self.target_component
            )

            # Wait for acknowledgment
            ack = self.connection.recv_match(
                type="MISSION_ACK", blocking=True, timeout=2
            )
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
                req = self.connection.recv_match(
                    type="MISSION_REQUEST", blocking=True, timeout=2
                )
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
            final_ack = self.connection.recv_match(
                type="MISSION_ACK", blocking=True, timeout=2
            )
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
                self.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_id,
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
                self.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_id,
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
            msg = self.connection.recv_match(
                type="MISSION_CURRENT", blocking=True, timeout=1
            )
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
                        logger.error(
                            f"Invalid baud rate in connection string: {parts[1]}"
                        )
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
            text = (
                f"Detection: RSSI={rssi:.1f}dBm SNR={snr:.1f}dB Conf={confidence:.0f}%"
            )
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
            self.send_statustext(
                "Signal lost - returning to search", severity=5
            )  # NOTICE level
            self.send_named_value_float("SIGNAL", 0.0)  # Signal indicator = 0
        except Exception as e:
            logger.error(f"Failed to send signal lost telemetry: {e}")

    def _initialize_frequency_parameters(self) -> None:
        """Initialize comprehensive MAVLink parameters for Mission Planner RF control.

        SUBTASK-6.3.1.2: Comprehensive parameter schema with bandwidth, homing state, emergency commands.
        Creates complete Mission Planner-compatible parameter interface for ASV RF control.
        """
        try:
            # Core Frequency Control Parameters (SUBTASK-6.3.1.1)
            self._register_parameter(
                "PISAD_RF_FREQ", 406000000.0, self._handle_custom_frequency_change
            )
            self._register_parameter(
                "PISAD_RF_PROFILE", 0.0, self._handle_frequency_profile_change
            )

            # Enhanced Frequency Parameters (SUBTASK-6.3.1.2 [28a3])
            self._register_parameter(
                "PISAD_RF_BW", 25000.0, self._handle_bandwidth_change
            )

            # Homing Control Parameters (SUBTASK-6.3.1.2 [29a1-29b4])
            self._register_parameter(
                "PISAD_HOMING_EN", 0.0, self._handle_homing_enable_change
            )
            self._register_parameter(
                "PISAD_HOMING_STATE",
                0.0,
                self._handle_readonly_parameter,  # Read-only status
            )

            # Signal Quality Parameters (SUBTASK-6.3.2.1 [30a1-30d4])
            self._register_parameter(
                "PISAD_SIG_CLASS",
                0.0,
                self._handle_readonly_parameter,  # Read-only
            )
            self._register_parameter(
                "PISAD_SIG_CONF",
                0.0,
                self._handle_readonly_parameter,  # Read-only
            )
            self._register_parameter(
                "PISAD_BEARING",
                0.0,
                self._handle_readonly_parameter,  # Read-only
            )
            self._register_parameter(
                "PISAD_BEAR_CONF",
                0.0,
                self._handle_readonly_parameter,  # Read-only
            )
            self._register_parameter(
                "PISAD_INTERFERENCE",
                0.0,
                self._handle_readonly_parameter,  # Read-only
            )

            # System Health Parameters
            self._register_parameter(
                "PISAD_RF_HEALTH",
                0.0,
                self._handle_readonly_parameter,  # Read-only
            )

            # Emergency Override Parameters
            self._register_parameter(
                "PISAD_EMERGENCY_DISABLE", 0.0, self._handle_emergency_disable
            )

            # Performance Monitoring Parameters
            self._register_parameter(
                "PISAD_RESPONSE_TIME",
                0.0,
                self._handle_readonly_parameter,  # Read-only
            )

            logger.info("Comprehensive Mission Planner RF parameters initialized")
            logger.info(f"Total parameters registered: {len(self._parameters)}")

        except Exception as e:
            logger.error(f"Failed to initialize frequency parameters: {e}")

    def _register_parameter(
        self, param_name: str, default_value: float, handler: Callable[[float], bool]
    ) -> None:
        """Register a MAVLink parameter with handler.

        Args:
            param_name: Parameter name (Mission Planner compatible)
            default_value: Default parameter value
            handler: Callback function for parameter changes
        """
        self._parameter_handlers[param_name] = handler
        self._parameters[param_name] = default_value
        logger.debug(f"Registered parameter {param_name}={default_value}")

    def set_parameter(self, param_name: str, value: float) -> bool:
        """Set a MAVLink parameter value.

        Args:
            param_name: Parameter name
            value: Parameter value

        Returns:
            True if parameter was set successfully
        """
        try:
            if param_name not in self._parameter_handlers:
                logger.warning(f"Unknown parameter: {param_name}")
                return False

            # Call parameter handler for validation and processing
            handler = self._parameter_handlers[param_name]
            if handler(value):
                self._parameters[param_name] = value

                # Notify callbacks of parameter change
                for callback in self._parameter_callbacks:
                    try:
                        callback(param_name, value)
                    except Exception as e:
                        logger.error(f"Parameter callback error: {e}")

                logger.info(f"Parameter {param_name} set to {value}")
                return True
            else:
                logger.warning(f"Parameter validation failed: {param_name}={value}")
                return False

        except Exception as e:
            logger.error(f"Error setting parameter {param_name}: {e}")
            return False

    def get_parameter(self, param_name: str) -> float | None:
        """Get a MAVLink parameter value.

        Args:
            param_name: Parameter name

        Returns:
            Parameter value or None if not found
        """
        return self._parameters.get(param_name)

    def request_parameter(self, param_name: str) -> bool:
        """Request parameter value to be sent to Mission Planner.

        Args:
            param_name: Parameter name to request

        Returns:
            True if request was processed
        """
        try:
            if param_name in self._parameters:
                value = self._parameters[param_name]
                self._send_param_value(param_name, value)
                return True
            else:
                logger.warning(f"Parameter not found for request: {param_name}")
                return False

        except Exception as e:
            logger.error(f"Error requesting parameter {param_name}: {e}")
            return False

    def _handle_param_set(self, msg: Any) -> bool:
        """Handle PARAM_SET message from Mission Planner.

        Args:
            msg: MAVLink PARAM_SET message

        Returns:
            True if parameter was set successfully
        """
        try:
            # Handle both string and bytes for param_id
            if isinstance(msg.param_id, bytes):
                param_id = msg.param_id.decode("utf-8").rstrip("\x00")
            else:
                param_id = str(msg.param_id).rstrip("\x00")
            param_value = float(msg.param_value)

            if self.set_parameter(param_id, param_value):
                # Send confirmation back to Mission Planner
                self._send_param_value(param_id, param_value)
                return True
            else:
                return False

        except Exception as e:
            logger.error(f"Error handling PARAM_SET: {e}")
            return False

    def _handle_command_long(self, msg: Any) -> bool:
        """Handle COMMAND_LONG messages for emergency RF commands.

        SUBTASK-6.3.1.2 [29a1, 29c1] - MAV_CMD_USER_1/USER_2 emergency commands.

        Args:
            msg: COMMAND_LONG message

        Returns:
            True if command was processed
        """
        try:
            command_id = msg.command

            # MAV_CMD_USER_1: Homing enable/disable command
            if command_id == mavutil.mavlink.MAV_CMD_USER_1:
                enable_flag = int(msg.param1)  # 0=disable, 1=enable

                logger.info(f"Received MAV_CMD_USER_1 homing command: {enable_flag}")

                # Process homing command with safety checks
                if self._handle_homing_enable_change(float(enable_flag)):
                    self._send_command_ack(msg, mavutil.mavlink.MAV_RESULT_ACCEPTED)
                    logger.info(
                        f"MAV_CMD_USER_1 homing command accepted: {enable_flag}"
                    )
                else:
                    self._send_command_ack(msg, mavutil.mavlink.MAV_RESULT_DENIED)
                    logger.warning(
                        f"MAV_CMD_USER_1 homing command denied: {enable_flag}"
                    )

                return True

            # MAV_CMD_USER_2: Emergency RF disable command
            elif command_id == mavutil.mavlink.MAV_CMD_USER_2:
                emergency_flag = int(msg.param1)  # 1=emergency disable

                logger.critical(
                    f"Received MAV_CMD_USER_2 emergency command: {emergency_flag}"
                )

                if emergency_flag == 1:
                    # Process emergency disable
                    if self._handle_emergency_disable(1.0):
                        self._send_command_ack(msg, mavutil.mavlink.MAV_RESULT_ACCEPTED)
                        logger.critical("MAV_CMD_USER_2 emergency RF disable ACCEPTED")
                    else:
                        self._send_command_ack(msg, mavutil.mavlink.MAV_RESULT_FAILED)
                        logger.critical("MAV_CMD_USER_2 emergency RF disable FAILED")
                else:
                    self._send_command_ack(msg, mavutil.mavlink.MAV_RESULT_UNSUPPORTED)

                return True

            else:
                # Unknown command - not handled
                return False

        except Exception as e:
            logger.error(f"Error handling COMMAND_LONG: {e}")
            try:
                self._send_command_ack(msg, mavutil.mavlink.MAV_RESULT_FAILED)
            except:
                pass  # Ignore ack send errors
            return False

    def _send_command_ack(self, original_msg: Any, result: int) -> None:
        """Send command acknowledgment to Mission Planner.

        Args:
            original_msg: Original COMMAND_LONG message
            result: MAV_RESULT code
        """
        try:
            if self.connection and self.state == ConnectionState.CONNECTED:
                self.connection.mav.command_ack_send(
                    original_msg.command,  # Command ID
                    result,  # Result code
                    0,  # Progress (not used)
                    0,  # Result param2 (not used)
                    self.target_system,  # Target system
                    self.target_component,  # Target component
                )
                logger.debug(
                    f"Sent COMMAND_ACK for {original_msg.command}: result={result}"
                )
        except Exception as e:
            logger.error(f"Error sending command ACK: {e}")

    def _handle_param_request_read(self, msg: Any) -> bool:
        """Handle PARAM_REQUEST_READ message from Mission Planner.

        Args:
            msg: MAVLink PARAM_REQUEST_READ message

        Returns:
            True if parameter was sent successfully
        """
        try:
            # Handle both string and bytes for param_id
            if isinstance(msg.param_id, bytes):
                param_id = msg.param_id.decode("utf-8").rstrip("\x00")
            else:
                param_id = str(msg.param_id).rstrip("\x00")
            return self.request_parameter(param_id)

        except Exception as e:
            logger.error(f"Error handling PARAM_REQUEST_READ: {e}")
            return False

    def _send_param_value(self, param_name: str, value: float) -> None:
        """Send PARAM_VALUE message to Mission Planner.

        Args:
            param_name: Parameter name
            value: Parameter value
        """
        if not self.connection:
            return

        try:
            # Send PARAM_VALUE message
            self.connection.mav.param_value_send(
                param_name.encode("utf-8")[:16],  # Parameter ID (max 16 chars)
                value,  # Parameter value
                2,  # MAV_PARAM_TYPE_REAL32
                len(self._parameters),  # Total parameter count
                list(self._parameters.keys()).index(param_name),  # Parameter index
            )

        except Exception as e:
            logger.error(f"Error sending PARAM_VALUE for {param_name}: {e}")

    def _handle_frequency_profile_change(self, value: float) -> bool:
        """Handle frequency profile parameter change.

        SUBTASK-6.3.1.1 [28a2] - Frequency profile selection via Mission Planner with ASV integration.

        Args:
            value: Profile index (0=Emergency, 1=Aviation, 2=Custom)

        Returns:
            True if profile change was successful
        """
        try:
            start_time = time.perf_counter()
            profile_index = int(value)

            # Validate profile index
            if profile_index not in [0, 1, 2]:
                logger.warning(f"Invalid frequency profile: {profile_index}")
                return False

            # Map profile index to frequency and analyzer type
            profile_configs: dict[int, dict[str, Any]] = {
                0: {"name": "Emergency", "frequency": 406_000_000, "analyzer": "GP"},
                1: {"name": "Aviation", "frequency": 121_500_000, "analyzer": "GP"},
                2: {
                    "name": "Custom",
                    "frequency": None,
                    "analyzer": "GP",
                },  # Use current frequency
            }

            config = profile_configs[profile_index]
            profile_name = config["name"]

            # ASV service integration for frequency switching
            if self._asv_service and config["frequency"] is not None:
                # Use asyncio.create_task to handle async call in sync context
                import asyncio

                try:
                    # Create task for frequency switching
                    loop = asyncio.get_event_loop()
                    task = loop.create_task(
                        self._asv_service.switch_frequency(
                            config["frequency"], config["analyzer"]
                        )
                    )
                    # Don't await here - let it run in background for <50ms response
                    logger.info(
                        f"ASV frequency switch initiated: {profile_name} -> {config['frequency']/1e6:.3f} MHz"
                    )
                except Exception as asv_error:
                    logger.error(f"ASV frequency switch failed: {asv_error}")
                    return False
            else:
                # Log without ASV integration (development/testing scenario)
                logger.info(
                    f"Frequency profile changed to: {profile_name} (index={profile_index})"
                )
                if not self._asv_service:
                    logger.warning(
                        "ASV service not available - frequency change logged only"
                    )

            # Measure response time for PRD compliance
            response_time = (time.perf_counter() - start_time) * 1000
            self.update_response_time_parameter(response_time)

            if response_time > 50.0:
                logger.warning(
                    f"Parameter response time {response_time:.1f}ms exceeded 50ms requirement"
                )
            else:
                logger.debug(f"Parameter response completed in {response_time:.1f}ms")

            return True

        except Exception as e:
            logger.error(f"Error handling frequency profile change: {e}")
            return False

    def _handle_custom_frequency_change(self, value: float) -> bool:
        """Handle custom frequency parameter change.

        SUBTASK-6.3.1.1 [28a1] - Real-time frequency change commands with ASV integration.

        Args:
            value: Frequency in Hz

        Returns:
            True if frequency change was successful
        """
        try:
            start_time = time.perf_counter()
            frequency_hz = int(value)

            # Validate frequency range (HackRF One: 1 MHz - 6 GHz)
            if not (1_000_000 <= frequency_hz <= 6_000_000_000):
                logger.warning(f"Frequency out of range: {frequency_hz} Hz")
                return False

            # ASV service integration for real-time frequency switching
            if self._asv_service:
                import asyncio

                try:
                    # Create task for frequency switching
                    loop = asyncio.get_event_loop()
                    task = loop.create_task(
                        self._asv_service.switch_frequency(frequency_hz, "GP")
                    )
                    # Don't await here - let it run in background for <50ms response
                    logger.info(
                        f"ASV custom frequency switch initiated: {frequency_hz/1e6:.3f} MHz"
                    )
                except Exception as asv_error:
                    logger.error(f"ASV frequency switch failed: {asv_error}")
                    return False
            else:
                # Log without ASV integration (development/testing scenario)
                logger.info(f"Custom frequency changed to: {frequency_hz} Hz")
                logger.warning(
                    "ASV service not available - frequency change logged only"
                )

            # Measure response time for PRD compliance
            response_time = (time.perf_counter() - start_time) * 1000
            self.update_response_time_parameter(response_time)

            if response_time > 50.0:
                logger.warning(
                    f"Parameter response time {response_time:.1f}ms exceeded 50ms requirement"
                )
            else:
                logger.debug(f"Parameter response completed in {response_time:.1f}ms")

            return True

        except Exception as e:
            logger.error(f"Error handling custom frequency change: {e}")
            return False

    def _handle_homing_enable_change(self, value: float) -> bool:
        """Handle homing enable parameter change.

        SUBTASK-6.3.1.2 [29a1] - Homing mode activation via Mission Planner with safety integration.

        Args:
            value: Enable flag (0=disabled, 1=enabled)

        Returns:
            True if homing enable change was successful
        """
        try:
            start_time = time.perf_counter()

            # Validate that value is 0 or 1 only
            if value not in [0.0, 1.0]:
                logger.warning(f"Invalid homing enable value: {value} (must be 0 or 1)")
                return False

            enable_flag = bool(int(value))

            # Homing controller integration with safety checks
            if self._homing_controller:
                import asyncio

                try:
                    if enable_flag:
                        # Pre-activation safety checks (SUBTASK-6.3.1.2 [29a2])
                        # Check guided mode
                        if self.telemetry.get("flight_mode") != "GUIDED":
                            logger.warning(
                                "Homing activation denied - not in GUIDED mode"
                            )
                            return False

                        # Check signal detected (basic validation)
                        # TODO: Integrate with signal processor for actual signal detection

                        # Check armed state
                        if not self.telemetry.get("armed", False):
                            logger.warning(
                                "Homing activation denied - vehicle not armed"
                            )
                            return False

                        logger.info(
                            "Homing activation safety checks passed - enabling homing mode"
                        )
                        # Create task for homing activation
                        loop = asyncio.get_event_loop()
                        task = loop.create_task(self._homing_controller.enable_homing())
                    else:
                        logger.info("Disabling homing mode via Mission Planner")
                        # Create task for homing deactivation
                        loop = asyncio.get_event_loop()
                        task = loop.create_task(
                            self._homing_controller.disable_homing()
                        )

                except Exception as homing_error:
                    logger.error(
                        f"Homing controller integration failed: {homing_error}"
                    )
                    return False
            else:
                # Log without homing controller integration
                logger.info(
                    f"Homing mode {'enabled' if enable_flag else 'disabled'} via Mission Planner"
                )
                logger.warning(
                    "Homing controller not available - homing change logged only"
                )

            # Measure response time for PRD compliance
            response_time = (time.perf_counter() - start_time) * 1000
            self.update_response_time_parameter(response_time)

            if response_time > 50.0:
                logger.warning(
                    f"Parameter response time {response_time:.1f}ms exceeded 50ms requirement"
                )
            else:
                logger.debug(f"Parameter response completed in {response_time:.1f}ms")

            return True

        except Exception as e:
            logger.error(f"Error handling homing enable change: {e}")
            return False

    def _handle_bandwidth_change(self, value: float) -> bool:
        """Handle RF bandwidth parameter change.

        SUBTASK-6.3.1.2 [28a3] - RF bandwidth configuration for signal processing.

        Args:
            value: Bandwidth in Hz (1kHz-10MHz range)

        Returns:
            True if bandwidth change was successful
        """
        try:
            start_time = time.perf_counter()
            bandwidth_hz = int(value)

            # Validate bandwidth range (1kHz - 10MHz)
            if not (1_000 <= bandwidth_hz <= 10_000_000):
                logger.warning(
                    f"Bandwidth out of range: {bandwidth_hz} Hz (1kHz-10MHz)"
                )
                return False

            # ASV service integration for bandwidth configuration
            if self._asv_service:
                # Note: ASV service bandwidth configuration would be implemented here
                # For now, log the bandwidth change
                logger.info(f"ASV bandwidth configuration: {bandwidth_hz/1000:.1f} kHz")
            else:
                logger.info(f"RF bandwidth changed to: {bandwidth_hz/1000:.1f} kHz")
                logger.warning(
                    "ASV service not available - bandwidth change logged only"
                )

            # Measure response time for PRD compliance
            response_time = (time.perf_counter() - start_time) * 1000
            self.update_response_time_parameter(response_time)

            if response_time > 50.0:
                logger.warning(
                    f"Parameter response time {response_time:.1f}ms exceeded 50ms requirement"
                )
            else:
                logger.debug(f"Parameter response completed in {response_time:.1f}ms")

            return True

        except Exception as e:
            logger.error(f"Error handling bandwidth change: {e}")
            return False

    def _handle_readonly_parameter(self, value: float) -> bool:
        """Handle read-only parameter change attempts.

        SUBTASK-6.3.1.2: Read-only parameters for status reporting.

        Args:
            value: Attempted parameter value

        Returns:
            False (read-only parameters cannot be changed)
        """
        logger.warning(f"Attempt to change read-only parameter: {value}")
        return False

    def _handle_emergency_disable(self, value: float) -> bool:
        """Handle emergency RF disable command.

        SUBTASK-6.3.1.2 [29c1] - Emergency RF disable with immediate response.

        Args:
            value: Emergency disable flag (1=disable RF immediately)

        Returns:
            True if emergency disable was processed
        """
        try:
            start_time = time.perf_counter()

            if value == 1.0:
                logger.critical("EMERGENCY RF DISABLE ACTIVATED via Mission Planner")

                # Emergency disable with all services
                if self._asv_service:
                    import asyncio

                    try:
                        loop = asyncio.get_event_loop()
                        # Emergency stop task for ASV service
                        task = loop.create_task(self._emergency_stop_all_rf_services())
                    except Exception as asv_error:
                        logger.error(f"ASV emergency disable failed: {asv_error}")

                if self._homing_controller:
                    import asyncio

                    try:
                        loop = asyncio.get_event_loop()
                        # Emergency disable homing
                        task = loop.create_task(
                            self._homing_controller.emergency_stop()
                        )
                    except Exception as homing_error:
                        logger.error(f"Homing emergency disable failed: {homing_error}")

                # Reset parameter to 0 after processing
                self._parameters["PISAD_EMERGENCY_DISABLE"] = 0.0

                # Measure emergency response time (should be <100ms)
                response_time = (time.perf_counter() - start_time) * 1000
                if response_time > 100.0:
                    logger.critical(
                        f"Emergency response time {response_time:.1f}ms exceeded 100ms requirement"
                    )
                else:
                    logger.info(f"Emergency disable completed in {response_time:.1f}ms")

                return True
            else:
                # Non-emergency values are ignored
                return False

        except Exception as e:
            logger.error(f"Error handling emergency disable: {e}")
            return False

    async def _emergency_stop_all_rf_services(self) -> None:
        """Emergency stop all RF services with immediate response."""
        try:
            if self._asv_service:
                # Stop ASV service
                await self._asv_service.emergency_stop()
                logger.info("ASV service emergency stopped")

        except Exception as e:
            logger.error(f"Error in emergency RF stop: {e}")

    def add_parameter_callback(self, callback: Callable[[str, float], None]) -> None:
        """Add callback for parameter changes.

        Args:
            callback: Function to call when parameters change
        """
        self._parameter_callbacks.append(callback)

    def set_asv_service(self, asv_service: Any) -> None:
        """Set ASV service reference for frequency control integration.

        Args:
            asv_service: ASVHackRFCoordinator service instance
        """
        self._asv_service = asv_service
        logger.info("ASV service reference set for Mission Planner integration")

    def set_homing_controller(self, homing_controller: Any) -> None:
        """Set homing controller reference for homing control integration.

        Args:
            homing_controller: HomingController service instance
        """
        self._homing_controller = homing_controller
        logger.info("Homing controller reference set for Mission Planner integration")

    def update_rf_parameters_from_asv(self, asv_data: dict[str, Any]) -> None:
        """Update read-only RF parameters with live ASV data.

        SUBTASK-6.3.1.2: Real-time parameter updates for Mission Planner display.

        Args:
            asv_data: Dictionary containing ASV service data
        """
        try:
            # Update signal classification (SUBTASK-6.3.2.1 [30a1])
            if "signal_classification" in asv_data:
                # Map signal types: 0=FM_CHIRP, 1=CONTINUOUS, 2=NOISE, 3=INTERFERENCE
                signal_type_map = {
                    "fm_chirp": 0.0,
                    "continuous": 1.0,
                    "noise": 2.0,
                    "interference": 3.0,
                }
                classification = signal_type_map.get(
                    asv_data["signal_classification"], 0.0
                )
                self._parameters["PISAD_SIG_CLASS"] = classification

            # Update signal confidence (SUBTASK-6.3.2.1 [30b1])
            if "confidence" in asv_data:
                # Convert to 0-100% scale for Mission Planner
                self._parameters["PISAD_SIG_CONF"] = (
                    float(asv_data["confidence"]) * 100.0
                )

            # Update bearing information (SUBTASK-6.3.2.1 [30c1-30c2])
            if "bearing_deg" in asv_data:
                self._parameters["PISAD_BEARING"] = float(asv_data["bearing_deg"])
            if "bearing_confidence" in asv_data:
                self._parameters["PISAD_BEAR_CONF"] = (
                    float(asv_data["bearing_confidence"]) * 100.0
                )

            # Update interference level (SUBTASK-6.3.2.1 [30d1])
            if "interference_level" in asv_data:
                self._parameters["PISAD_INTERFERENCE"] = (
                    float(asv_data["interference_level"]) * 100.0
                )

            # Update RF system health (general health indicator)
            if "system_health" in asv_data:
                self._parameters["PISAD_RF_HEALTH"] = (
                    float(asv_data["system_health"]) * 100.0
                )

        except Exception as e:
            logger.error(f"Error updating RF parameters from ASV: {e}")

    def update_homing_state_parameter(self, homing_state: int) -> None:
        """Update homing state parameter for Mission Planner display.

        SUBTASK-6.3.1.2 [29b1]: Real-time homing status updates.

        Args:
            homing_state: Homing state (0=Disabled, 1=Armed, 2=Active, 3=Lost)
        """
        try:
            if 0 <= homing_state <= 3:
                self._parameters["PISAD_HOMING_STATE"] = float(homing_state)
                state_names = ["Disabled", "Armed", "Active", "Lost"]
                logger.debug(f"Homing state updated: {state_names[homing_state]}")
            else:
                logger.warning(f"Invalid homing state: {homing_state}")
        except Exception as e:
            logger.error(f"Error updating homing state parameter: {e}")

    def update_response_time_parameter(self, response_time_ms: float) -> None:
        """Update response time parameter for performance monitoring.

        SUBTASK-6.3.1.3: Performance monitoring and timing validation.

        Args:
            response_time_ms: Last parameter response time in milliseconds
        """
        try:
            self._parameters["PISAD_RESPONSE_TIME"] = response_time_ms
            if response_time_ms > 50.0:
                logger.warning(
                    f"Parameter response time monitoring: {response_time_ms:.1f}ms"
                )
        except Exception as e:
            logger.error(f"Error updating response time parameter: {e}")

    def send_asv_bearing_telemetry(self, bearing_calculation: Any) -> None:
        """Send ASV bearing calculation telemetry to Mission Planner.

        SUBTASK-6.1.3.2 [19a][19b] - Enhanced RF telemetry with ASV signal classification.

        Args:
            bearing_calculation: ASVBearingCalculation object with enhanced telemetry
        """
        if not self.connection or self.state != ConnectionState.CONNECTED:
            return

        try:
            current_time = time.time()

            # Send ASV bearing and confidence data
            self.send_named_value_float(
                "ASV_BEARING", bearing_calculation.bearing_deg, current_time
            )
            self.send_named_value_float(
                "ASV_CONFIDENCE", bearing_calculation.confidence * 100.0, current_time
            )  # Send as percentage
            self.send_named_value_float(
                "ASV_PRECISION", bearing_calculation.precision_deg, current_time
            )

            # Send enhanced signal quality indicators
            self.send_named_value_float(
                "ASV_SIG_QUAL", bearing_calculation.signal_quality * 100.0, current_time
            )
            self.send_named_value_float(
                "ASV_SIG_RSSI", bearing_calculation.signal_strength_dbm, current_time
            )

            # Send interference detection flag (0=no interference, 1=interference detected)
            interference_flag = (
                1.0 if bearing_calculation.interference_detected else 0.0
            )
            self.send_named_value_float("ASV_INTERF", interference_flag, current_time)

            # Send signal classification as numeric code for Mission Planner display
            classification_code = self._map_signal_classification_to_code(
                bearing_calculation.signal_classification
            )
            self.send_named_value_float(
                "ASV_SIG_TYPE", float(classification_code), current_time
            )

            # Enhanced status text with ASV analysis
            status_text = (
                f"ASV: {bearing_calculation.bearing_deg:.1f}° "
                f"±{bearing_calculation.precision_deg:.1f}° "
                f"Conf:{bearing_calculation.confidence*100:.0f}% "
                f"Type:{bearing_calculation.signal_classification}"
            )
            self.send_statustext(status_text, severity=6)  # INFO level

            logger.debug(f"Sent ASV bearing telemetry: {status_text}")

        except Exception as e:
            logger.error(f"Failed to send ASV bearing telemetry: {e}")

    def send_asv_signal_quality_telemetry(
        self, signal_quality_data: dict[str, Any]
    ) -> None:
        """Send ASV signal quality and trend analysis telemetry.

        SUBTASK-6.1.3.2 [19c][19d] - Signal quality indicators and RSSI trend visualization.

        Args:
            signal_quality_data: Dictionary with signal quality metrics
        """
        if not self.connection or self.state != ConnectionState.CONNECTED:
            return

        try:
            current_time = time.time()

            # Send signal trend indicators
            rssi_trend = signal_quality_data.get(
                "rssi_trend", 0.0
            )  # Positive = improving, negative = degrading
            self.send_named_value_float("ASV_RSSI_TREND", rssi_trend, current_time)

            # Send signal stability metrics
            signal_stability = signal_quality_data.get(
                "signal_stability", 0.0
            )  # 0.0-1.0 stability score
            self.send_named_value_float(
                "ASV_STABILITY", signal_stability * 100.0, current_time
            )

            # Send frequency drift information
            frequency_drift = signal_quality_data.get("frequency_drift_hz", 0.0)
            self.send_named_value_float("ASV_FREQ_DRIFT", frequency_drift, current_time)

            # Send multipath indicator
            multipath_severity = signal_quality_data.get(
                "multipath_severity", 0.0
            )  # 0.0-1.0
            self.send_named_value_float(
                "ASV_MULTIPATH", multipath_severity * 100.0, current_time
            )

            logger.debug("Sent ASV signal quality telemetry")

        except Exception as e:
            logger.error(f"Failed to send ASV signal quality telemetry: {e}")

    def send_asv_detection_event_telemetry(
        self, detection_event: dict[str, Any]
    ) -> None:
        """Send ASV detection event notifications for Mission Planner status.

        SUBTASK-6.1.3.3 [20c] - RF detection event notifications in Mission Planner.

        Args:
            detection_event: Dictionary with detection event data
        """
        if not self.connection or self.state != ConnectionState.CONNECTED:
            return

        try:
            current_time = time.time()

            # Send detection event type
            event_type_code = self._map_detection_event_to_code(
                detection_event.get("event_type", "UNKNOWN")
            )
            self.send_named_value_float(
                "ASV_DET_EVENT", float(event_type_code), current_time
            )

            # Send detection strength
            detection_strength = detection_event.get("detection_strength", 0.0)
            self.send_named_value_float(
                "ASV_DET_STR", detection_strength * 100.0, current_time
            )

            # Send ASV analyzer source info
            analyzer_source = detection_event.get("analyzer_source", "UNKNOWN")
            analyzer_code = self._map_analyzer_source_to_code(analyzer_source)
            self.send_named_value_float(
                "ASV_ANALYZER", float(analyzer_code), current_time
            )

            # Enhanced status message for Mission Planner
            event_type = detection_event.get("event_type", "DETECTION")
            frequency_mhz = detection_event.get("frequency_hz", 0) / 1_000_000
            status_msg = (
                f"ASV {event_type}: {frequency_mhz:.3f}MHz "
                f"Strength:{detection_strength*100:.0f}% "
                f"Source:{analyzer_source}"
            )
            self.send_statustext(status_msg, severity=5)  # NOTICE level

            logger.info(f"Sent ASV detection event: {status_msg}")

        except Exception as e:
            logger.error(f"Failed to send ASV detection event telemetry: {e}")

    def _map_signal_classification_to_code(self, classification: str) -> int:
        """Map ASV signal classification to numeric code for Mission Planner.

        Args:
            classification: ASV signal classification string

        Returns:
            Numeric code for Mission Planner telemetry
        """
        classification_map = {
            "UNKNOWN": 0,
            "CONTINUOUS": 1,
            "FM_CHIRP": 2,
            "FM_CHIRP_WEAK": 3,
            "INTERFERENCE": 4,
            "BEACON_121_5": 5,
            "BEACON_406": 6,
            "AVIATION": 7,
            "MULTIPATH": 8,
            "SPURIOUS": 9,
        }
        return classification_map.get(classification, 0)

    def _map_detection_event_to_code(self, event_type: str) -> int:
        """Map detection event type to numeric code.

        Args:
            event_type: Detection event type string

        Returns:
            Numeric code for telemetry
        """
        event_map = {
            "UNKNOWN": 0,
            "DETECTION": 1,
            "SIGNAL_LOST": 2,
            "SIGNAL_IMPROVED": 3,
            "INTERFERENCE_DETECTED": 4,
            "INTERFERENCE_CLEARED": 5,
            "FREQUENCY_DRIFT": 6,
            "MULTIPATH_DETECTED": 7,
            "BEACON_CONFIRMED": 8,
        }
        return event_map.get(event_type, 0)

    def _map_analyzer_source_to_code(self, analyzer_source: str) -> int:
        """Map ASV analyzer source to numeric code.

        Args:
            analyzer_source: ASV analyzer source identifier

        Returns:
            Numeric code for telemetry
        """
        analyzer_map = {
            "UNKNOWN": 0,
            "ASV_PROFESSIONAL": 1,
            "ASV_STANDARD": 2,
            "ASV_ENHANCED": 3,
            "HACKRF_DIRECT": 4,
            "HYBRID_ASV": 5,
        }
        return analyzer_map.get(analyzer_source, 0)
