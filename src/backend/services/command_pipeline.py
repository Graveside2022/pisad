"""Command Pipeline with safety interlocks and validation.

This module implements a safe command execution pipeline with:
- Safety interlock checks before commands
- Command validation and sanitization
- Rate limiting to prevent flooding
- Priority queue for emergency commands
- Audit logging for all commands (FR12)
"""

import asyncio
import contextlib
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, IntEnum
from typing import Any
from uuid import uuid4

from src.backend.services.mavlink_service import MAVLinkService
from src.backend.utils.logging import get_logger
from src.backend.utils.safety import SafetyInterlockSystem

logger = get_logger(__name__)


class CommandPriority(IntEnum):
    """Command priority levels."""

    EMERGENCY = 0  # Highest priority - emergency stop, RTL
    CRITICAL = 1  # Safety-critical commands
    HIGH = 2  # Important operational commands
    NORMAL = 3  # Regular commands
    LOW = 4  # Background tasks


class CommandType(Enum):
    """Types of commands."""

    # Emergency commands
    EMERGENCY_STOP = "emergency_stop"
    RETURN_TO_LAUNCH = "return_to_launch"

    # Navigation commands
    GOTO_POSITION = "goto_position"
    SET_VELOCITY = "set_velocity"
    SET_MODE = "set_mode"
    LOITER = "loiter"

    # Mission commands
    START_MISSION = "start_mission"
    PAUSE_MISSION = "pause_mission"
    RESUME_MISSION = "resume_mission"
    ABORT_MISSION = "abort_mission"

    # System commands
    ARM = "arm"
    DISARM = "disarm"
    TAKEOFF = "takeoff"
    LAND = "land"

    # Homing commands
    START_HOMING = "start_homing"
    STOP_HOMING = "stop_homing"
    UPDATE_BEARING = "update_bearing"


@dataclass
class Command:
    """Represents a command to be executed."""

    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    type: CommandType = CommandType.LOITER
    priority: CommandPriority = CommandPriority.NORMAL
    parameters: dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"  # Who issued the command
    validated: bool = False
    safety_checked: bool = False
    executed: bool = False
    result: Any = None
    error: str | None = None

    def __lt__(self, other: "Command") -> bool:
        """Compare commands for priority queue ordering."""
        # Lower priority value = higher priority
        return self.priority < other.priority


@dataclass
class CommandAuditEntry:
    """Audit log entry for command execution (FR12)."""

    command_id: str
    timestamp: datetime
    command_type: str
    priority: int
    source: str
    safety_status: dict[str, bool]
    execution_time_ms: float
    success: bool
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class CommandPipeline:
    """Safe command execution pipeline with multiple safety layers.

    Implements PRD requirements:
    - FR8: Geofence safety checks
    - FR12: Command audit logging
    - NFR2: 100ms emergency stop response time
    """

    def __init__(
        self,
        safety_system: SafetyInterlockSystem | None = None,
        mavlink_service: MAVLinkService | None = None,
        rate_limit_per_second: float = 10.0,
        max_queue_size: int = 100,
    ):
        """Initialize command pipeline.

        Args:
            safety_system: Safety interlock system for validation
            mavlink_service: MAVLink service for command execution
            rate_limit_per_second: Maximum commands per second
            max_queue_size: Maximum queue size
        """
        self.safety_system = safety_system or SafetyInterlockSystem()
        self.mavlink_service = mavlink_service

        # Command queue with priority
        self.command_queue: asyncio.PriorityQueue[tuple[int, Command]] = asyncio.PriorityQueue(
            maxsize=max_queue_size
        )

        # Rate limiting
        self.rate_limit = rate_limit_per_second
        self.last_command_time = 0.0
        self.min_interval = 1.0 / rate_limit_per_second

        # Audit logging (FR12)
        self.audit_log: deque[CommandAuditEntry] = deque(maxlen=10000)

        # Command validators
        self.validators: dict[CommandType, Callable[[Command], bool]] = {}
        self._register_default_validators()

        # Processing state
        self.is_running = False
        self.process_task: asyncio.Task | None = None

        # Statistics
        self.total_commands = 0
        self.successful_commands = 0
        self.failed_commands = 0
        self.blocked_by_safety = 0

        logger.info(
            f"CommandPipeline initialized: rate_limit={rate_limit_per_second}/s, "
            f"max_queue={max_queue_size}"
        )

    def _register_default_validators(self) -> None:
        """Register default command validators."""
        # Emergency commands - minimal validation
        self.validators[CommandType.EMERGENCY_STOP] = self._validate_emergency_stop
        self.validators[CommandType.RETURN_TO_LAUNCH] = self._validate_rtl

        # Navigation commands
        self.validators[CommandType.GOTO_POSITION] = self._validate_goto_position
        self.validators[CommandType.SET_VELOCITY] = self._validate_set_velocity
        self.validators[CommandType.SET_MODE] = self._validate_set_mode

        # Mission commands
        self.validators[CommandType.START_MISSION] = self._validate_mission_command
        self.validators[CommandType.ABORT_MISSION] = self._validate_mission_command

        # System commands
        self.validators[CommandType.ARM] = self._validate_arm
        self.validators[CommandType.TAKEOFF] = self._validate_takeoff
        self.validators[CommandType.LAND] = self._validate_land

    async def submit_command(
        self,
        command_type: CommandType,
        parameters: dict[str, Any] | None = None,
        priority: CommandPriority = CommandPriority.NORMAL,
        source: str = "api",
    ) -> str:
        """Submit a command for execution.

        Args:
            command_type: Type of command
            parameters: Command parameters
            priority: Command priority
            source: Source of command (for audit)

        Returns:
            Command ID for tracking

        Raises:
            ValueError: If command validation fails
            asyncio.QueueFull: If queue is full
        """
        command = Command(
            type=command_type,
            priority=priority,
            parameters=parameters or {},
            source=source,
        )

        # Validate command
        if not await self._validate_command(command):
            raise ValueError(f"Command validation failed: {command.type.value}")

        command.validated = True

        # Emergency commands bypass queue
        if priority == CommandPriority.EMERGENCY:
            logger.warning(f"Emergency command submitted: {command_type.value}")
            await self._execute_emergency_command(command)
        else:
            # Add to priority queue
            await self.command_queue.put((priority, command))
            logger.info(
                f"Command queued: {command_type.value} (priority={priority}, id={command.id})"
            )

        self.total_commands += 1
        return command.id

    async def _execute_emergency_command(self, command: Command) -> None:
        """Execute emergency command immediately (100ms requirement).

        Args:
            command: Emergency command to execute
        """
        start_time = time.perf_counter()
        minimal_safety = {}  # Initialize for finally block

        try:
            # Emergency commands bypass most safety checks
            minimal_safety = await self._check_minimal_safety()

            if command.type == CommandType.EMERGENCY_STOP:
                # Immediate stop - highest priority
                if self.mavlink_service:
                    await self.mavlink_service.emergency_stop()
                await self.safety_system.emergency_stop("Command pipeline emergency stop")
                command.executed = True
                command.result = "Emergency stop executed"

            elif command.type == CommandType.RETURN_TO_LAUNCH:
                # Return to launch
                if self.mavlink_service:
                    await self.mavlink_service.return_to_launch()
                command.executed = True
                command.result = "RTL initiated"

            self.successful_commands += 1

        except Exception as e:
            command.error = str(e)
            self.failed_commands += 1
            logger.error(f"Emergency command failed: {e}")

        finally:
            # Ensure we meet 100ms requirement
            execution_time = (time.perf_counter() - start_time) * 1000
            if execution_time > 100:
                logger.error(
                    f"Emergency command exceeded 100ms requirement: {execution_time:.1f}ms"
                )

            # Audit log
            await self._log_command_execution(
                command, {"minimal_safety": minimal_safety}, execution_time, command.executed
            )

    async def _validate_command(self, command: Command) -> bool:
        """Validate and sanitize command parameters.

        Args:
            command: Command to validate

        Returns:
            True if valid, False otherwise
        """
        # Check if validator exists
        validator = self.validators.get(command.type)
        if not validator:
            logger.warning(f"No validator for command type: {command.type.value}")
            return True  # Allow by default if no validator

        try:
            return validator(command)
        except Exception as e:
            logger.error(f"Validation error for {command.type.value}: {e}")
            return False

    def _validate_emergency_stop(self, command: Command) -> bool:
        """Validate emergency stop command."""
        # Emergency stop always valid
        return True

    def _validate_rtl(self, command: Command) -> bool:
        """Validate return to launch command."""
        # RTL always valid
        return True

    def _validate_goto_position(self, command: Command) -> bool:
        """Validate goto position command."""
        params = command.parameters

        # Check required parameters
        if not all(k in params for k in ["latitude", "longitude", "altitude"]):
            return False

        # Validate ranges
        lat = params["latitude"]
        lon = params["longitude"]
        alt = params["altitude"]

        if not (-90 <= lat <= 90):
            return False
        if not (-180 <= lon <= 180):
            return False
        if not (0 <= alt <= 500):  # Max 500m altitude
            return False

        return True

    def _validate_set_velocity(self, command: Command) -> bool:
        """Validate set velocity command."""
        params = command.parameters

        # Check required parameters
        if not all(k in params for k in ["vx", "vy", "vz"]):
            return False

        # Validate velocity limits (m/s)
        max_velocity = 20.0  # 20 m/s max

        return all(abs(v) <= max_velocity for v in [params["vx"], params["vy"], params["vz"]])

    def _validate_set_mode(self, command: Command) -> bool:
        """Validate set mode command."""
        params = command.parameters

        if "mode" not in params:
            return False

        # Valid modes
        valid_modes = [
            "MANUAL", "STABILIZE", "GUIDED", "AUTO", "LOITER",
            "RTL", "LAND", "POSHOLD", "BRAKE"
        ]

        return params["mode"] in valid_modes

    def _validate_mission_command(self, command: Command) -> bool:
        """Validate mission commands."""
        # Mission commands typically don't need parameters
        return True

    def _validate_arm(self, command: Command) -> bool:
        """Validate arm command."""
        # Arming requires extra safety checks
        return True

    def _validate_takeoff(self, command: Command) -> bool:
        """Validate takeoff command."""
        params = command.parameters

        if "altitude" not in params:
            return False

        # Validate altitude
        alt = params["altitude"]
        if not (1 <= alt <= 100):  # 1-100m takeoff altitude
            return False

        return True

    def _validate_land(self, command: Command) -> bool:
        """Validate land command."""
        # Landing always valid
        return True

    async def _check_minimal_safety(self) -> dict[str, bool]:
        """Check minimal safety for emergency commands.

        Returns:
            Safety check results
        """
        # Only check critical safety items
        results = {
            "battery": True,  # Assume OK for emergency
            "geofence": True,  # Override for emergency
        }

        # Check battery if possible (handle mock safety system)
        try:
            if hasattr(self.safety_system, "checks") and "battery" in self.safety_system.checks:
                battery_check = self.safety_system.checks["battery"]
                results["battery"] = await battery_check.check()
        except Exception:
            # Mock or simplified safety system - assume OK for emergency
            pass

        return results

    async def _check_full_safety(self, command: Command) -> tuple[bool, dict[str, bool]]:
        """Check all safety interlocks for a command.

        Args:
            command: Command to check

        Returns:
            Tuple of (safe_to_proceed, check_results)
        """
        # Check all safety interlocks
        results = await self.safety_system.check_all_safety()

        # Special handling for certain commands
        if command.type in [CommandType.ARM, CommandType.TAKEOFF]:
            # Extra strict for arming/takeoff
            safe = all(results.values())
        elif command.type in [CommandType.LAND, CommandType.RETURN_TO_LAUNCH]:
            # Allow landing even with some failures
            safe = results.get("battery", True)  # Only battery critical for landing
        else:
            # Normal commands
            safe = await self.safety_system.is_safe_to_proceed()

        return safe, results

    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting to prevent command flooding."""
        current_time = time.time()
        time_since_last = current_time - self.last_command_time

        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            await asyncio.sleep(sleep_time)

        self.last_command_time = time.time()

    async def _execute_command(self, command: Command) -> None:
        """Execute a validated command.

        Args:
            command: Command to execute
        """
        if not self.mavlink_service:
            command.error = "No MAVLink service available"
            return

        try:
            if command.type == CommandType.GOTO_POSITION:
                params = command.parameters
                await self.mavlink_service.goto_position(
                    params["latitude"], params["longitude"], params["altitude"]
                )
                command.result = "Position command sent"

            elif command.type == CommandType.SET_VELOCITY:
                params = command.parameters
                await self.mavlink_service.set_velocity(
                    params["vx"], params["vy"], params["vz"]
                )
                command.result = "Velocity command sent"

            elif command.type == CommandType.SET_MODE:
                await self.mavlink_service.set_mode(command.parameters["mode"])
                command.result = f"Mode set to {command.parameters['mode']}"

            elif command.type == CommandType.ARM:
                await self.mavlink_service.arm()
                command.result = "Armed"

            elif command.type == CommandType.DISARM:
                await self.mavlink_service.disarm()
                command.result = "Disarmed"

            elif command.type == CommandType.TAKEOFF:
                await self.mavlink_service.takeoff(command.parameters["altitude"])
                command.result = f"Takeoff to {command.parameters['altitude']}m"

            elif command.type == CommandType.LAND:
                await self.mavlink_service.land()
                command.result = "Landing"

            elif command.type == CommandType.LOITER:
                await self.mavlink_service.set_mode("LOITER")
                command.result = "Loitering"

            else:
                command.error = f"Unhandled command type: {command.type.value}"
                return

            command.executed = True
            self.successful_commands += 1

        except Exception as e:
            command.error = str(e)
            self.failed_commands += 1
            logger.error(f"Command execution failed: {e}")

    async def _log_command_execution(
        self,
        command: Command,
        safety_status: dict[str, bool],
        execution_time_ms: float,
        success: bool,
    ) -> None:
        """Log command execution for audit trail (FR12).

        Args:
            command: Executed command
            safety_status: Safety check results
            execution_time_ms: Execution time in milliseconds
            success: Whether command succeeded
        """
        entry = CommandAuditEntry(
            command_id=command.id,
            timestamp=command.timestamp,
            command_type=command.type.value,
            priority=command.priority,
            source=command.source,
            safety_status=safety_status,
            execution_time_ms=execution_time_ms,
            success=success,
            error=command.error,
            metadata={
                "parameters": command.parameters,
                "result": command.result,
            },
        )

        self.audit_log.append(entry)

        # Log to file/database if needed
        logger.info(
            f"Command audit: {command.type.value} from {command.source} - "
            f"{'SUCCESS' if success else 'FAILED'} ({execution_time_ms:.1f}ms)"
        )

    async def start(self) -> None:
        """Start command processing."""
        if self.is_running:
            return

        self.is_running = True
        self.process_task = asyncio.create_task(self._process_loop())

        # Start safety monitoring
        await self.safety_system.start_monitoring()

        logger.info("Command pipeline started")

    async def stop(self) -> None:
        """Stop command processing."""
        if not self.is_running:
            return

        self.is_running = False

        if self.process_task:
            self.process_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.process_task

        # Stop safety monitoring
        await self.safety_system.stop_monitoring()

        logger.info("Command pipeline stopped")

    async def _process_loop(self) -> None:
        """Main command processing loop."""
        while self.is_running:
            try:
                # Get next command with timeout
                priority, command = await asyncio.wait_for(
                    self.command_queue.get(), timeout=1.0
                )

                start_time = time.perf_counter()

                # Apply rate limiting
                await self._apply_rate_limit()

                # Check safety interlocks
                safe, safety_status = await self._check_full_safety(command)
                command.safety_checked = True

                if not safe:
                    command.error = "Blocked by safety interlock"
                    self.blocked_by_safety += 1
                    logger.warning(
                        f"Command {command.type.value} blocked by safety: {safety_status}"
                    )
                else:
                    # Execute command
                    await self._execute_command(command)

                # Calculate execution time
                execution_time = (time.perf_counter() - start_time) * 1000

                # Log for audit
                await self._log_command_execution(
                    command, safety_status, execution_time, command.executed
                )

            except TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in command processing loop: {e}")

    def get_statistics(self) -> dict[str, Any]:
        """Get pipeline statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_commands": self.total_commands,
            "successful_commands": self.successful_commands,
            "failed_commands": self.failed_commands,
            "blocked_by_safety": self.blocked_by_safety,
            "queue_size": self.command_queue.qsize(),
            "rate_limit": self.rate_limit,
            "success_rate": (
                (self.successful_commands / self.total_commands * 100)
                if self.total_commands > 0
                else 0
            ),
        }

    def get_audit_log(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent audit log entries.

        Args:
            limit: Maximum entries to return

        Returns:
            List of audit entries
        """
        entries = list(self.audit_log)[-limit:]

        return [
            {
                "command_id": entry.command_id,
                "timestamp": entry.timestamp.isoformat(),
                "type": entry.command_type,
                "priority": entry.priority,
                "source": entry.source,
                "safety_status": entry.safety_status,
                "execution_time_ms": entry.execution_time_ms,
                "success": entry.success,
                "error": entry.error,
                "metadata": entry.metadata,
            }
            for entry in entries
        ]
