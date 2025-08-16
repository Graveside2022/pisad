"""State machine service for managing system states and telemetry integration."""

import asyncio
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from backend.core.exceptions import (
    DatabaseError,
    SafetyInterlockError,
    StateTransitionError,
)
from backend.utils.logging import get_logger

if TYPE_CHECKING:
    from backend.services.search_pattern_generator import SearchPattern

logger = get_logger(__name__)


class SystemState(Enum):
    """System operational states."""

    IDLE = "IDLE"
    SEARCHING = "SEARCHING"
    DETECTING = "DETECTING"
    HOMING = "HOMING"
    HOLDING = "HOLDING"
    EMERGENCY = "EMERGENCY"  # Emergency stop state


class SearchSubstate(Enum):
    """Search pattern execution substates."""

    IDLE = "IDLE"
    EXECUTING = "EXECUTING"
    PAUSED = "PAUSED"


@dataclass
class StateChangeEvent:
    """Event representing a state change."""

    from_state: SystemState
    to_state: SystemState
    timestamp: datetime
    reason: str | None = None


class StateMachine:
    """Manages system state transitions and telemetry events."""

    def __init__(self, db_path: str = "data/pisad.db", enable_persistence: bool = True):
        """Initialize state machine.

        Args:
            db_path: Path to SQLite database file
            enable_persistence: Whether to enable state persistence
        """
        self._current_state = SystemState.IDLE
        self._previous_state = SystemState.IDLE
        self._state_history: list[StateChangeEvent] = []
        self._is_running = False
        self._homing_enabled = False

        # Store persistence settings
        self._db_path = db_path
        self._enable_persistence = enable_persistence
        self._state_db: Any = None

        # MAVLink service reference (set externally)
        self._mavlink_service: Any = None

        # Signal processor reference (set externally)
        self._signal_processor: Any = None

        # State change callbacks
        self._state_callbacks: list[Any] = []

        # Detection event tracking
        self._last_detection_time = 0.0
        self._detection_count = 0

        # Search pattern management
        self._search_substate = SearchSubstate.IDLE
        self._active_pattern: SearchPattern | None = None
        self._current_waypoint_index = 0
        self._pattern_paused_at: datetime | None = None

        # Entry/Exit action hooks
        self._entry_actions: dict[SystemState, list[Callable[[], Coroutine[Any, Any, None]]]] = {
            state: [] for state in SystemState
        }
        self._exit_actions: dict[SystemState, list[Callable[[], Coroutine[Any, Any, None]]]] = {
            state: [] for state in SystemState
        }

        # Add default HOMING entry action
        self._entry_actions[SystemState.HOMING] = [self._on_homing_entry]
        self._exit_actions[SystemState.HOMING] = [self._on_homing_exit]

        # State timeout configurations (seconds)
        self._state_timeouts: dict[SystemState, float] = {
            SystemState.IDLE: 0,  # No timeout for IDLE
            SystemState.SEARCHING: 300,  # 5 minutes max search
            SystemState.DETECTING: 60,  # 1 minute to confirm detection
            SystemState.HOMING: 180,  # 3 minutes max homing
            SystemState.HOLDING: 120,  # 2 minutes max hold
        }
        self._state_entered_time: float = time.time()
        self._timeout_task: asyncio.Task[None] | None = None
        self._telemetry_task: asyncio.Task[None] | None = None

        # Telemetry metrics
        self._state_metrics: dict[str, Any] = {
            "total_transitions": 0,
            "state_durations": {state.value: 0.0 for state in SystemState},
            "transition_frequencies": {},
            "transition_times": [],  # Last 100 transition times in ms
            "state_entry_counts": {state.value: 0 for state in SystemState},
        }
        self._last_metrics_update = time.time()

        # State persistence
        if self._enable_persistence:
            try:
                from backend.models.database import StateHistoryDB

                self._state_db = StateHistoryDB(db_path)
                # Try to restore previous state
                self._restore_state()
            except (ImportError, DatabaseError, OSError) as e:
                logger.error(
                    f"Failed to initialize state persistence: {e}", extra={"db_path": db_path}
                )
                self._state_db = None
                # Continue without persistence - not critical

        # Register default entry/exit actions
        self._register_default_actions()

        logger.info(f"StateMachine initialized in {self._current_state.value} state")

    def set_mavlink_service(self, mavlink_service: Any) -> None:
        """Set MAVLink service reference for telemetry.

        Args:
            mavlink_service: MAVLink service instance
        """
        self._mavlink_service = mavlink_service
        logger.info("MAVLink service connected to state machine")

    def set_signal_processor(self, signal_processor: Any) -> None:
        """Set signal processor reference for RSSI data.

        Args:
            signal_processor: Signal processor instance
        """
        self._signal_processor = signal_processor
        logger.info("Signal processor connected to state machine")

    def get_current_state(self) -> SystemState:
        """Get current system state."""
        return self._current_state

    @property
    def current_state(self) -> SystemState:
        """Property for backward compatibility."""
        return self._current_state

    def get_state_string(self) -> str:
        """Get current state as string."""
        return self._current_state.value

    def add_state_callback(self, callback: Any) -> None:
        """Add callback for state changes.

        Args:
            callback: Function to call on state change
        """
        self._state_callbacks.append(callback)

    def _register_default_actions(self) -> None:
        """Register default entry/exit actions for each state."""
        # IDLE state actions
        self.register_entry_action(SystemState.IDLE, self._on_idle_entry)
        self.register_exit_action(SystemState.IDLE, self._on_idle_exit)

        # SEARCHING state actions
        self.register_entry_action(SystemState.SEARCHING, self._on_searching_entry)
        self.register_exit_action(SystemState.SEARCHING, self._on_searching_exit)

        # DETECTING state actions
        self.register_entry_action(SystemState.DETECTING, self._on_detecting_entry)
        self.register_exit_action(SystemState.DETECTING, self._on_detecting_exit)

        # HOMING state actions
        self.register_entry_action(SystemState.HOMING, self._on_homing_entry)
        self.register_exit_action(SystemState.HOMING, self._on_homing_exit)

        # HOLDING state actions
        self.register_entry_action(SystemState.HOLDING, self._on_holding_entry)
        self.register_exit_action(SystemState.HOLDING, self._on_holding_exit)

    async def _on_idle_entry(self) -> None:
        """Entry action for IDLE state."""
        logger.debug("Entering IDLE state - releasing resources")
        # Release any allocated resources
        self._search_substate = SearchSubstate.IDLE
        self._current_waypoint_index = 0

    async def _on_idle_exit(self) -> None:
        """Exit action for IDLE state."""
        logger.debug("Exiting IDLE state")

    async def _on_searching_entry(self) -> None:
        """Entry action for SEARCHING state."""
        logger.debug("Entering SEARCHING state - initializing search resources")
        # Initialize search resources
        if self._signal_processor:
            try:
                # Enable signal processing for searching
                pass  # Placeholder for actual signal processor initialization
            except (AttributeError, ConnectionError) as e:
                logger.error(
                    f"Failed to initialize signal processor for searching: {e}",
                    extra={"state": "SEARCHING"},
                )
                # Signal processor issues shouldn't prevent state transition

    async def _on_searching_exit(self) -> None:
        """Exit action for SEARCHING state."""
        logger.debug("Exiting SEARCHING state")
        # Pause or stop search pattern if active
        if self._search_substate == SearchSubstate.EXECUTING:
            await self.pause_search_pattern()

    async def _on_detecting_entry(self) -> None:
        """Entry action for DETECTING state."""
        logger.debug("Entering DETECTING state - enhancing signal processing")
        # Enhance signal processing for detection
        if self._signal_processor:
            try:
                # Increase sampling rate or sensitivity
                pass  # Placeholder for actual signal processor configuration
            except (AttributeError, ValueError, ConnectionError) as e:
                logger.error(
                    f"Failed to enhance signal processing: {e}", extra={"state": "DETECTING"}
                )
                # Continue with default signal processing

    async def _on_detecting_exit(self) -> None:
        """Exit action for DETECTING state."""
        logger.debug("Exiting DETECTING state")

    async def _on_homing_entry(self) -> None:
        """Entry action for HOMING state."""
        logger.debug("Entering HOMING state - initializing homing algorithm")
        # Initialize homing algorithm resources
        if self._mavlink_service:
            try:
                # Set flight mode for homing
                pass  # Placeholder for MAVLink configuration
            except (AttributeError, ConnectionError, SafetyInterlockError) as e:
                logger.error(f"Failed to initialize homing mode: {e}", extra={"state": "HOMING"})
                raise StateTransitionError(f"Cannot enter HOMING: {e}") from e

    async def _on_homing_exit(self) -> None:
        """Exit action for HOMING state."""
        logger.debug("Exiting HOMING state - cleaning up homing resources")

    async def _on_holding_entry(self) -> None:
        """Entry action for HOLDING state."""
        logger.debug("Entering HOLDING state - maintaining position")
        # Configure for position holding
        if self._mavlink_service:
            try:
                # Enable position hold mode
                pass  # Placeholder for MAVLink position hold
            except (AttributeError, ConnectionError, ValueError) as e:
                logger.error(f"Failed to enable position hold: {e}", extra={"state": "HOLDING"})
                # Continue with current flight mode

    async def _on_holding_exit(self) -> None:
        """Exit action for HOLDING state."""
        logger.debug("Exiting HOLDING state")

    def register_entry_action(
        self, state: SystemState, action: Callable[[], Coroutine[Any, Any, None]]
    ) -> None:
        """Register an entry action for a state.

        Args:
            state: The state to register the action for
            action: Async function to execute on state entry
        """
        if action not in self._entry_actions[state]:
            self._entry_actions[state].append(action)
            logger.debug(f"Registered entry action for {state.value}")

    def register_exit_action(
        self, state: SystemState, action: Callable[[], Coroutine[Any, Any, None]]
    ) -> None:
        """Register an exit action for a state.

        Args:
            state: The state to register the action for
            action: Async function to execute on state exit
        """
        if action not in self._exit_actions[state]:
            self._exit_actions[state].append(action)
            logger.debug(f"Registered exit action for {state.value}")

    async def _execute_exit_actions(self, state: SystemState) -> float:
        """Execute all exit actions for a state.

        Args:
            state: The state being exited

        Returns:
            Total execution time in milliseconds
        """
        start_time = time.time()
        actions = self._exit_actions.get(state, [])

        for action in actions:
            try:
                await action()
            except (AttributeError, TypeError, ValueError) as e:
                logger.error(
                    f"Error executing exit action for {state.value}: {e}",
                    extra={
                        "action": action.__name__ if hasattr(action, "__name__") else str(action)
                    },
                )
                # Continue with other exit actions

        duration_ms = (time.time() - start_time) * 1000
        if actions:
            logger.debug(
                f"Executed {len(actions)} exit actions for {state.value} in {duration_ms:.2f}ms"
            )
        return duration_ms

    async def _execute_entry_actions(self, state: SystemState) -> float:
        """Execute all entry actions for a state.

        Args:
            state: The state being entered

        Returns:
            Total execution time in milliseconds
        """
        start_time = time.time()
        actions = self._entry_actions.get(state, [])

        for action in actions:
            try:
                await action()
            except StateTransitionError as e:
                logger.error(f"Error executing entry action for {state.value}: {e}")

        duration_ms = (time.time() - start_time) * 1000
        if actions:
            logger.debug(
                f"Executed {len(actions)} entry actions for {state.value} in {duration_ms:.2f}ms"
            )
        return duration_ms

    def _update_telemetry_metrics(
        self, from_state: SystemState, to_state: SystemState, action_duration_ms: float
    ) -> None:
        """Update telemetry metrics for state transition.

        Args:
            from_state: Previous state
            to_state: New state
            action_duration_ms: Time taken for entry/exit actions
        """
        current_time = time.time()

        # Update state duration for the state we're leaving
        if self._last_metrics_update > 0:
            duration = current_time - self._last_metrics_update
            self._state_metrics["state_durations"][from_state.value] += duration

        # Update transition count
        self._state_metrics["total_transitions"] += 1

        # Update state entry count
        self._state_metrics["state_entry_counts"][to_state.value] += 1

        # Track transition frequency
        transition_key = f"{from_state.value}->{to_state.value}"
        if transition_key not in self._state_metrics["transition_frequencies"]:
            self._state_metrics["transition_frequencies"][transition_key] = 0
        self._state_metrics["transition_frequencies"][transition_key] += 1

        # Track transition time
        self._state_metrics["transition_times"].append(action_duration_ms)
        # Keep only last 100 transition times
        if len(self._state_metrics["transition_times"]) > 100:
            self._state_metrics["transition_times"] = self._state_metrics["transition_times"][-100:]

        self._last_metrics_update = current_time

    def get_telemetry_metrics(self) -> dict[str, Any]:
        """Get comprehensive telemetry metrics.

        Returns:
            Dictionary with telemetry metrics
        """
        # Update current state duration
        current_time = time.time()
        if self._last_metrics_update > 0:
            duration = current_time - self._last_metrics_update
            current_metrics = self._state_metrics.copy()
            current_metrics["state_durations"] = self._state_metrics["state_durations"].copy()
            current_metrics["state_durations"][self._current_state.value] += duration
        else:
            current_metrics = self._state_metrics

        # Calculate average transition time
        avg_transition_time = 0.0
        if self._state_metrics["transition_times"]:
            avg_transition_time = sum(self._state_metrics["transition_times"]) / len(
                self._state_metrics["transition_times"]
            )

        # Add computed metrics
        current_metrics["average_transition_time_ms"] = avg_transition_time
        current_metrics["current_state_duration_s"] = self.get_state_duration()
        current_metrics["uptime_seconds"] = (
            current_time
            - self._state_entered_time
            + sum(self._state_metrics["state_durations"].values())
        )

        return current_metrics

    async def send_telemetry_update(self) -> None:
        """Send telemetry update via MAVLink if available."""
        if not self._mavlink_service:
            return

        try:
            metrics = self.get_telemetry_metrics()

            # Send key metrics via MAVLink
            self._mavlink_service.send_telemetry("state_transitions", metrics["total_transitions"])
            self._mavlink_service.send_telemetry(
                "state_duration_ms", int(metrics["current_state_duration_s"] * 1000)
            )
            self._mavlink_service.send_telemetry(
                "avg_transition_ms", int(metrics["average_transition_time_ms"])
            )

            # Send state-specific metrics
            for state in SystemState:
                duration_key = f"state_{state.value.lower()}_duration_s"
                self._mavlink_service.send_telemetry(
                    duration_key, metrics["state_durations"][state.value]
                )
        except DatabaseError as e:
            logger.error(f"Failed to send telemetry update: {e}")

    async def transition_to(self, new_state: SystemState | str, reason: str | None = None) -> bool:
        """Transition to a new state with validation.

        Args:
            new_state: Target state (enum or string)
            reason: Optional reason for transition

        Returns:
            True if transition was successful, False otherwise
        """
        # Convert string to enum if necessary
        if isinstance(new_state, str):
            try:
                new_state = SystemState(new_state)
            except ValueError:
                logger.error(f"Invalid state string: {new_state}")
                return False

        # Validate transition
        if not self._is_valid_transition(self._current_state, new_state):
            logger.warning(
                f"Invalid state transition from {self._current_state.value} to {new_state.value}"
            )
            return False

        old_state = self._current_state

        # Skip if transitioning to the same state
        if old_state == new_state:
            return True

        # Cancel any existing timeout task
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()

        # Execute exit actions for the current state
        exit_duration_ms = await self._execute_exit_actions(old_state)

        # Record state change
        self._previous_state = old_state
        self._current_state = new_state
        self._state_entered_time = time.time()

        # Execute entry actions for the new state
        entry_duration_ms = await self._execute_entry_actions(new_state)

        # Total action duration
        action_duration_ms = exit_duration_ms + entry_duration_ms

        # Start timeout task for new state if configured
        timeout = self._state_timeouts.get(new_state, 0)
        if timeout > 0:
            self._timeout_task = asyncio.create_task(self._handle_state_timeout(new_state, timeout))

        event = StateChangeEvent(
            from_state=old_state, to_state=new_state, timestamp=datetime.now(UTC), reason=reason
        )
        self._state_history.append(event)

        # Update telemetry metrics
        self._update_telemetry_metrics(old_state, new_state, action_duration_ms)

        # Persist state change to database
        if self._state_db:
            try:
                self._state_db.save_state_change(
                    from_state=old_state.value,
                    to_state=new_state.value,
                    timestamp=event.timestamp,
                    reason=reason,
                    action_duration_ms=int(action_duration_ms),
                )
                self._state_db.save_current_state(
                    state=new_state.value,
                    previous_state=old_state.value,
                    homing_enabled=self._homing_enabled,
                    last_detection_time=self._last_detection_time,
                    detection_count=self._detection_count,
                )
            except StateTransitionError as e:
                logger.error(f"Failed to persist state change: {e}")

        logger.info(
            f"State transition: {old_state.value} -> {new_state.value}"
            + (f" (reason: {reason})" if reason else "")
            + f" [actions: {action_duration_ms:.2f}ms]"
        )

        # Send telemetry if MAVLink service is available
        if self._mavlink_service:
            try:
                self._mavlink_service.send_state_change(new_state.value)
            except StateTransitionError as e:
                logger.error(f"Failed to send state change telemetry: {e}")

        # Notify callbacks
        for callback in self._state_callbacks:
            try:
                await callback(old_state, new_state, reason)
            except StateTransitionError as e:
                logger.error(f"Error in state callback: {e}")

        return True

    def _is_valid_transition(self, from_state: SystemState, to_state: SystemState) -> bool:
        """Check if state transition is valid with guard conditions.

        Args:
            from_state: Current state
            to_state: Target state

        Returns:
            True if transition is valid, False otherwise
        """
        # Define valid transitions with guard conditions
        valid_transitions = {
            SystemState.IDLE: [SystemState.SEARCHING, SystemState.EMERGENCY],
            SystemState.SEARCHING: [SystemState.IDLE, SystemState.DETECTING, SystemState.EMERGENCY],
            SystemState.DETECTING: [
                SystemState.SEARCHING,
                SystemState.HOMING,
                SystemState.IDLE,
                SystemState.EMERGENCY,
            ],
            SystemState.HOMING: [
                SystemState.HOLDING,
                SystemState.SEARCHING,
                SystemState.IDLE,
                SystemState.EMERGENCY,
            ],
            SystemState.HOLDING: [
                SystemState.HOMING,
                SystemState.SEARCHING,
                SystemState.IDLE,
                SystemState.EMERGENCY,
            ],
            SystemState.EMERGENCY: [SystemState.IDLE],  # Can only return to IDLE from EMERGENCY
        }

        # Allow transition to same state (no-op)
        if from_state == to_state:
            return True

        # Check if transition is in valid list
        if to_state not in valid_transitions.get(from_state, []):
            return False

        # Apply guard conditions for specific transitions
        return self._check_transition_guards(from_state, to_state)

    async def _handle_state_timeout(self, state: SystemState, timeout: float) -> None:
        """Handle state timeout by transitioning to IDLE.

        Args:
            state: The state that may timeout
            timeout: Timeout duration in seconds
        """
        try:
            await asyncio.sleep(timeout)

            # Check if we're still in the same state
            if self._current_state == state:
                logger.warning(f"State {state.value} timed out after {timeout} seconds")
                await self.transition_to(SystemState.IDLE, f"Timeout after {timeout}s")
        except asyncio.CancelledError:
            # Task was cancelled (state changed before timeout)
            pass

    def set_state_timeout(self, state: SystemState, timeout: float) -> None:
        """Set or update timeout for a specific state.

        Args:
            state: The state to configure
            timeout: Timeout in seconds (0 to disable)
        """
        self._state_timeouts[state] = timeout
        logger.info(f"Set timeout for {state.value} to {timeout} seconds")

    def get_state_duration(self) -> float:
        """Get duration in current state.

        Returns:
            Duration in seconds
        """
        return time.time() - self._state_entered_time

    def _check_transition_guards(self, from_state: SystemState, to_state: SystemState) -> bool:
        """Check guard conditions for state transitions.

        Args:
            from_state: Current state
            to_state: Target state

        Returns:
            True if guard conditions pass, False otherwise
        """
        # IDLE -> SEARCHING: Check if resources are available
        if from_state == SystemState.IDLE and to_state == SystemState.SEARCHING:
            if not self._signal_processor:
                logger.warning("Cannot start searching: Signal processor not available")
                return False
            # Additional checks could include SDR status, sufficient battery, etc.
            return True

        # SEARCHING -> DETECTING: Check if we have a valid detection
        if from_state == SystemState.SEARCHING and to_state == SystemState.DETECTING:
            # This transition typically happens through handle_detection()
            # which provides RSSI and confidence values
            return True

        # DETECTING -> HOMING: Check if homing is enabled and conditions are met
        if from_state == SystemState.DETECTING and to_state == SystemState.HOMING:
            if not self._homing_enabled:
                logger.warning("Cannot transition to HOMING: Homing is disabled")
                return False
            if not self._mavlink_service:
                logger.warning("Cannot transition to HOMING: MAVLink service not available")
                return False
            # Could add additional checks like minimum detection confidence
            return True

        # HOMING -> HOLDING: Check if position hold is supported
        if from_state == SystemState.HOMING and to_state == SystemState.HOLDING:
            if not self._mavlink_service:
                logger.warning("Cannot transition to HOLDING: MAVLink service not available")
                return False
            # Could check if drone supports position hold mode
            return True

        # Emergency transitions to IDLE are always allowed
        if to_state == SystemState.IDLE:
            return True

        # Default: allow if no specific guard conditions
        return True

    async def handle_detection(self, rssi: float, confidence: float) -> None:
        """Handle signal detection event.

        Args:
            rssi: Signal strength in dBm
            confidence: Detection confidence percentage
        """
        import time

        current_time = time.time()
        self._last_detection_time = current_time
        self._detection_count += 1

        # Transition to DETECTING if in SEARCHING state
        if self._current_state == SystemState.SEARCHING:
            await self.transition_to(SystemState.DETECTING, f"Signal detected at {rssi:.1f}dBm")

        # Send detection telemetry if MAVLink service is available
        if self._mavlink_service:
            try:
                self._mavlink_service.send_detection_event(rssi, confidence)
            except Exception as e:
                logger.error(f"Failed to send detection telemetry: {e}")

        # Auto-transition to HOMING if enabled and confidence is high
        if (
            self._current_state == SystemState.DETECTING
            and self._homing_enabled
            and confidence > 80.0
        ):
            await self.transition_to(SystemState.HOMING, "High confidence detection")

    async def handle_signal_lost(self) -> None:
        """Handle loss of signal."""
        # Transition back to SEARCHING if in DETECTING or HOMING
        if self._current_state in [SystemState.DETECTING, SystemState.HOMING]:
            await self.transition_to(SystemState.SEARCHING, "Signal lost")

    def enable_homing(self, enabled: bool = True) -> None:
        """Enable or disable automatic homing.

        Args:
            enabled: True to enable homing, False to disable
        """
        self._homing_enabled = enabled
        logger.info(f"Homing {'enabled' if enabled else 'disabled'}")

    async def emergency_stop(self, reason: str = "Emergency stop") -> None:
        """Perform emergency stop and return to IDLE.

        Args:
            reason: Reason for emergency stop
        """
        logger.critical(f"Emergency stop initiated: {reason}")
        await self.transition_to(SystemState.IDLE, reason)
        self._homing_enabled = False

    def _restore_state(self) -> None:
        """Restore state from database on startup."""
        if not self._state_db:
            return

        try:
            saved_state = self._state_db.restore_state()
            if saved_state:
                # Restore state values
                self._current_state = SystemState(saved_state["state"])
                self._previous_state = SystemState(saved_state["previous_state"])
                self._homing_enabled = saved_state["homing_enabled"]
                self._last_detection_time = saved_state["last_detection_time"]
                self._detection_count = saved_state["detection_count"]

                # Load recent history from database
                history = self._state_db.get_state_history(limit=10)
                for record in reversed(history):
                    self._state_history.append(
                        StateChangeEvent(
                            from_state=SystemState(record["from_state"]),
                            to_state=SystemState(record["to_state"]),
                            timestamp=datetime.fromisoformat(record["timestamp"]),
                            reason=record["reason"],
                        )
                    )

                logger.info(f"Restored state: {self._current_state.value} from database")
            else:
                logger.info("No previous state found in database")
        except StateTransitionError as e:
            logger.error(f"Failed to restore state from database: {e}")

    async def force_transition(
        self, target_state: SystemState, reason: str, operator_id: str | None = None
    ) -> bool:
        """Force a state transition (for manual override).

        This method bypasses normal transition validation but still executes
        entry/exit actions and persists the change.

        Args:
            target_state: The state to transition to
            reason: Reason for the forced transition
            operator_id: Optional operator ID for audit trail

        Returns:
            True if transition successful, False otherwise
        """
        if self._current_state == target_state:
            logger.warning(f"Already in state {target_state.value}")
            return True

        old_state = self._current_state

        logger.warning(
            f"FORCED state transition: {old_state.value} -> {target_state.value} "
            f"by operator: {operator_id or 'unknown'}"
        )

        # Cancel any existing timeout task
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()

        # Execute exit actions for the current state
        exit_duration_ms = await self._execute_exit_actions(old_state)

        # Force the state change
        self._previous_state = old_state
        self._current_state = target_state
        self._state_entered_time = time.time()

        # Execute entry actions for the new state
        entry_duration_ms = await self._execute_entry_actions(target_state)

        # Start timeout task for new state if configured
        timeout = self._state_timeouts.get(target_state, 0)
        if timeout > 0:
            self._timeout_task = asyncio.create_task(
                self._handle_state_timeout(target_state, timeout)
            )

        # Total action duration
        action_duration_ms = exit_duration_ms + entry_duration_ms

        event = StateChangeEvent(
            from_state=old_state,
            to_state=target_state,
            timestamp=datetime.now(UTC),
            reason=f"FORCED: {reason}",
        )
        self._state_history.append(event)

        # Update telemetry metrics
        self._update_telemetry_metrics(old_state, target_state, action_duration_ms)

        # Persist forced state change to database with operator ID
        if self._state_db:
            try:
                self._state_db.save_state_change(
                    from_state=old_state.value,
                    to_state=target_state.value,
                    timestamp=event.timestamp,
                    reason=f"FORCED: {reason}",
                    operator_id=operator_id,
                    action_duration_ms=int(action_duration_ms),
                )
                self._state_db.save_current_state(
                    state=target_state.value,
                    previous_state=old_state.value,
                    homing_enabled=self._homing_enabled,
                    last_detection_time=self._last_detection_time,
                    detection_count=self._detection_count,
                )
            except StateTransitionError as e:
                logger.error(f"Failed to persist forced state change: {e}")

        # Send telemetry if MAVLink service is available
        if self._mavlink_service:
            try:
                self._mavlink_service.send_state_change(target_state.value)
            except StateTransitionError as e:
                logger.error(f"Failed to send forced state change telemetry: {e}")

        # Notify callbacks
        for callback in self._state_callbacks:
            try:
                await callback(old_state, target_state, f"FORCED: {reason}")
            except StateTransitionError as e:
                logger.error(f"Error in state callback: {e}")

        return True

    def get_allowed_transitions(self) -> list[SystemState]:
        """Get list of states that can be transitioned to from current state.

        Returns:
            List of allowed target states
        """
        valid_transitions = {
            SystemState.IDLE: [SystemState.SEARCHING, SystemState.EMERGENCY],
            SystemState.SEARCHING: [SystemState.IDLE, SystemState.DETECTING, SystemState.EMERGENCY],
            SystemState.DETECTING: [
                SystemState.SEARCHING,
                SystemState.HOMING,
                SystemState.IDLE,
                SystemState.EMERGENCY,
            ],
            SystemState.HOMING: [
                SystemState.HOLDING,
                SystemState.SEARCHING,
                SystemState.IDLE,
                SystemState.EMERGENCY,
            ],
            SystemState.HOLDING: [
                SystemState.HOMING,
                SystemState.SEARCHING,
                SystemState.IDLE,
                SystemState.EMERGENCY,
            ],
            SystemState.EMERGENCY: [SystemState.IDLE],  # Can only return to IDLE from EMERGENCY
        }

        return valid_transitions.get(self._current_state, [])

    def save_state(self) -> dict[str, Any]:
        """Save current state for persistence.

        Returns:
            Dict containing state information
        """
        return {
            "current_state": self._current_state.value,
            "previous_state": self._previous_state.value,
            "homing_enabled": self._homing_enabled,
            "detection_count": self._detection_count,
            "search_substate": self._search_substate.value,
            "current_waypoint_index": self._current_waypoint_index,
        }

    def restore_state(self, saved_state: dict[str, Any]) -> None:
        """Restore state from saved data.

        Args:
            saved_state: Dict containing saved state information
        """
        if "current_state" in saved_state:
            self._current_state = SystemState(saved_state["current_state"])
        if "previous_state" in saved_state:
            self._previous_state = SystemState(saved_state["previous_state"])
        if "homing_enabled" in saved_state:
            self._homing_enabled = saved_state["homing_enabled"]
        if "detection_count" in saved_state:
            self._detection_count = saved_state["detection_count"]
        if "search_substate" in saved_state:
            self._search_substate = SearchSubstate(saved_state["search_substate"])
        if "current_waypoint_index" in saved_state:
            self._current_waypoint_index = saved_state["current_waypoint_index"]

        logger.info(f"State restored to {self._current_state.value}")

    async def emergency_stop(self) -> bool:
        """Trigger emergency stop - immediately transition to safe state.

        Returns:
            True if emergency stop was successful
        """
        logger.warning("EMERGENCY STOP triggered!")
        # Disable homing immediately
        self._homing_enabled = False
        # Transition to EMERGENCY state
        result = await self.transition_to(SystemState.EMERGENCY, reason="Emergency Stop")
        # Stop any velocity commands if MAVLink service available
        if self._mavlink_service:
            try:
                await self._mavlink_service.stop_velocity_commands()
            except Exception as e:
                logger.error(f"Error stopping velocity commands: {e}")
        return result

    async def _on_homing_entry(self) -> None:
        """Entry action for HOMING state."""
        logger.info("Entering HOMING state")
        # Enable homing mode
        self._homing_enabled = True

    async def _on_homing_exit(self) -> None:
        """Exit action for HOMING state."""
        logger.info("Exiting HOMING state")
        # Disable homing mode
        self._homing_enabled = False

    def get_state_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent state change history.

        Args:
            limit: Maximum number of events to return

        Returns:
            List of state change events
        """
        history = self._state_history[-limit:] if limit > 0 else self._state_history

        return [
            {
                "from_state": event.from_state.value,
                "to_state": event.to_state.value,
                "timestamp": event.timestamp.isoformat(),
                "reason": event.reason,
            }
            for event in history
        ]

    def get_statistics(self) -> dict[str, Any]:
        """Get state machine statistics.

        Returns:
            Dictionary with statistics
        """
        import time

        return {
            "current_state": self._current_state.value,
            "previous_state": self._previous_state.value,
            "homing_enabled": self._homing_enabled,
            "detection_count": self._detection_count,
            "last_detection_time": self._last_detection_time,
            "time_since_detection": (
                time.time() - self._last_detection_time if self._last_detection_time > 0 else None
            ),
            "state_changes": len(self._state_history),
            "state_duration_seconds": self.get_state_duration(),
            "state_timeout_seconds": self._state_timeouts.get(self._current_state, 0),
        }

    async def start(self) -> None:
        """Start the state machine service."""
        if self._is_running:
            logger.warning("StateMachine already running")
            return

        self._is_running = True

        # Start periodic telemetry task
        self._telemetry_task = asyncio.create_task(self._telemetry_loop())

        logger.info("StateMachine started")

    async def _telemetry_loop(self) -> None:
        """Periodic telemetry update loop."""
        while self._is_running:
            try:
                await asyncio.sleep(10)  # Send telemetry every 10 seconds
                await self.send_telemetry_update()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in telemetry loop: {e}")

    async def stop(self) -> None:
        """Stop the state machine service."""
        self._is_running = False
        await self.transition_to(SystemState.IDLE, "Service stopped")
        logger.info("StateMachine stopped")

    # Search Pattern Management Methods

    def set_search_pattern(self, pattern: "SearchPattern") -> None:
        """Set the active search pattern.

        Args:
            pattern: Search pattern to execute
        """
        self._active_pattern = pattern
        self._current_waypoint_index = 0
        self._search_substate = SearchSubstate.IDLE
        logger.info(f"Search pattern {pattern.id} loaded with {pattern.total_waypoints} waypoints")

    def get_search_pattern(self) -> Optional["SearchPattern"]:
        """Get the active search pattern."""
        return self._active_pattern

    def get_search_substate(self) -> SearchSubstate:
        """Get current search execution substate."""
        return self._search_substate

    async def start_search_pattern(self) -> bool:
        """Start executing the search pattern.

        Returns:
            True if pattern started, False otherwise
        """
        if not self._active_pattern:
            logger.error("No search pattern loaded")
            return False

        if self._current_state != SystemState.SEARCHING:
            success = await self.transition_to(SystemState.SEARCHING, "Starting search pattern")
            if not success:
                return False

        self._search_substate = SearchSubstate.EXECUTING
        self._active_pattern.state = "EXECUTING"
        self._active_pattern.started_at = datetime.now(UTC)

        logger.info(f"Started executing search pattern {self._active_pattern.id}")
        return True

    async def pause_search_pattern(self) -> bool:
        """Pause search pattern execution.

        Returns:
            True if paused, False otherwise
        """
        if self._search_substate != SearchSubstate.EXECUTING:
            logger.warning("Cannot pause - pattern not executing")
            return False

        self._search_substate = SearchSubstate.PAUSED
        self._pattern_paused_at = datetime.now(UTC)

        if self._active_pattern:
            self._active_pattern.state = "PAUSED"
            self._active_pattern.paused_at = self._pattern_paused_at

        logger.info("Search pattern paused")
        return True

    async def resume_search_pattern(self) -> bool:
        """Resume paused search pattern.

        Returns:
            True if resumed, False otherwise
        """
        if self._search_substate != SearchSubstate.PAUSED:
            logger.warning("Cannot resume - pattern not paused")
            return False

        self._search_substate = SearchSubstate.EXECUTING
        self._pattern_paused_at = None

        if self._active_pattern:
            self._active_pattern.state = "EXECUTING"
            self._active_pattern.paused_at = None

        logger.info("Search pattern resumed")
        return True

    async def stop_search_pattern(self) -> bool:
        """Stop search pattern execution and return to IDLE.

        Returns:
            True if stopped, False otherwise
        """
        if not self._active_pattern:
            return False

        self._search_substate = SearchSubstate.IDLE
        self._active_pattern.state = "IDLE"
        self._active_pattern.completed_waypoints = 0
        self._active_pattern.progress_percent = 0.0
        self._current_waypoint_index = 0

        await self.transition_to(SystemState.IDLE, "Search pattern stopped")

        logger.info("Search pattern stopped")
        return True

    # API Methods for Story 4.5 Implementation

    async def initialize(self) -> bool:
        """Initialize state machine and all components.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing StateMachine...")

            # Set initial state
            self._current_state = SystemState.IDLE
            self._previous_state = SystemState.IDLE
            self._is_running = False
            self._homing_enabled = False

            # Clear any existing state
            self._state_history.clear()
            self._detection_count = 0
            self._last_detection_time = 0.0

            # Initialize search pattern state
            self._search_substate = SearchSubstate.IDLE
            self._active_pattern = None
            self._current_waypoint_index = 0
            self._pattern_paused_at = None

            # Register default entry/exit actions
            self._register_default_actions()

            # Initialize database if enabled
            if self._enable_persistence:
                try:
                    from backend.services.state_db import StateDatabase

                    self._state_db = StateDatabase(self._db_path)
                    self._restore_state()
                    logger.info("State persistence initialized")
                except DatabaseError as e:
                    logger.warning(f"State persistence unavailable: {e}")
                    self._state_db = None
            else:
                self._state_db = None

            # Verify MAVLink service if connected
            if self._mavlink_service:
                logger.info("MAVLink service connected")
            else:
                logger.warning("MAVLink service not connected - telemetry disabled")

            # Verify signal processor if connected
            if self._signal_processor:
                logger.info("Signal processor connected")
            else:
                logger.warning("Signal processor not connected - detection disabled")

            logger.info("StateMachine initialization complete")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize StateMachine: {e}")
            return False

    async def shutdown(self) -> None:
        """Gracefully shutdown state machine.

        Saves current state, releases resources, and notifies connected clients.
        """
        logger.info("Shutting down StateMachine...")

        # Stop any running operations
        self._is_running = False

        # Cancel timeout task if running
        if hasattr(self, "_timeout_task") and self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()
            try:
                await self._timeout_task
            except asyncio.CancelledError:
                pass

        # Cancel telemetry task if running
        if (
            hasattr(self, "_telemetry_task")
            and self._telemetry_task
            and not self._telemetry_task.done()
        ):
            self._telemetry_task.cancel()
            try:
                await self._telemetry_task
            except asyncio.CancelledError:
                pass

        # Stop any active search pattern
        if self._active_pattern:
            await self.stop_search_pattern()

        # Transition to IDLE state
        if self._current_state != SystemState.IDLE:
            await self.transition_to(SystemState.IDLE, "System shutdown")

        # Save final state if persistence enabled
        if self._state_db:
            try:
                self._state_db.save_current_state(
                    state=self._current_state.value,
                    previous_state=self._previous_state.value,
                    homing_enabled=self._homing_enabled,
                    last_detection_time=self._last_detection_time,
                    detection_count=self._detection_count,
                )
                logger.info("Final state saved to database")
            except DatabaseError as e:
                logger.error(f"Failed to save final state: {e}")

        # Notify callbacks of shutdown
        for callback in self._state_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(SystemState.IDLE, "shutdown")
                else:
                    callback(SystemState.IDLE, "shutdown")
            except Exception as e:
                logger.error(f"Error notifying callback of shutdown: {e}")

        # Clear references
        self._mavlink_service = None
        self._signal_processor = None
        self._state_callbacks.clear()

        logger.info("StateMachine shutdown complete")

    def get_valid_transitions(self) -> list[SystemState]:
        """Get list of valid states from current state.

        Returns:
            List of SystemState values that are valid transitions
        """
        valid_states = []

        for state in SystemState:
            if self._is_valid_transition(self._current_state, state):
                valid_states.append(state)

        return valid_states

    async def on_signal_detected(self, detection_event: Any) -> None:
        """Handle signal detection event.

        Args:
            detection_event: Detection event containing RSSI, SNR, confidence
        """
        import time

        # Extract detection details
        rssi = detection_event.rssi if hasattr(detection_event, "rssi") else -100.0
        snr = detection_event.snr if hasattr(detection_event, "snr") else 0.0
        confidence = detection_event.confidence if hasattr(detection_event, "confidence") else 0.0

        # Update detection tracking
        self._last_detection_time = time.time()
        self._detection_count += 1

        logger.info(
            f"Signal detected: RSSI={rssi:.1f}dBm, SNR={snr:.1f}dB, Confidence={confidence:.1f}%"
        )

        # State-specific handling
        if self._current_state == SystemState.SEARCHING:
            # Transition to DETECTING state
            await self.transition_to(
                SystemState.DETECTING,
                f"Signal detected at {rssi:.1f}dBm with {confidence:.1f}% confidence",
            )

        elif self._current_state == SystemState.DETECTING:
            # Already detecting, check if we should transition to HOMING
            if self._homing_enabled and confidence > 80.0:
                await self.transition_to(
                    SystemState.HOMING, f"High confidence detection ({confidence:.1f}%)"
                )

        # Send telemetry update if MAVLink connected
        if self._mavlink_service:
            try:
                await self._mavlink_service.send_detection_telemetry(
                    rssi=rssi, snr=snr, confidence=confidence, state=self._current_state.value
                )
            except Exception as e:
                logger.error(f"Failed to send detection telemetry: {e}")

    async def on_signal_lost(self) -> None:
        """Handle signal loss event."""
        logger.warning("Signal lost")

        # Only react if we were actively tracking a signal
        if self._current_state in [SystemState.DETECTING, SystemState.HOMING]:
            # Return to searching
            await self.transition_to(SystemState.SEARCHING, "Signal lost")

            # Reset detection count
            self._detection_count = 0

            # Notify MAVLink if connected
            if self._mavlink_service:
                try:
                    await self._mavlink_service.send_signal_lost_telemetry()
                except Exception as e:
                    logger.error(f"Failed to send signal lost telemetry: {e}")

    async def _check_signal_loss_timeout(self) -> None:
        """Check if signal has been lost for too long.

        SAFETY: Prevents drone from continuing to search indefinitely
        HAZARD: HARA-NAV-001 - Extended signal loss causing battery depletion
        HAZARD: HARA-NAV-002 - Uncontrolled drift during signal loss
        """
        if not hasattr(self, "_signal_lost_time"):
            return

        if self._signal_lost_time and self._homing_enabled:
            elapsed = time.time() - self._signal_lost_time

            if elapsed > 30.0:  # 30 second timeout
                logger.warning(f"Signal lost for {elapsed:.1f}s, disabling homing")
                self._homing_enabled = False
                await self.transition_to(SystemState.IDLE, "Signal loss timeout")

    async def on_mode_change(
        self, old_mode: str | None = None, new_mode: str | None = None
    ) -> None:
        """Handle operation mode change.

        SAFETY: Ensures safe transition between flight modes
        HAZARD: HARA-MODE-001 - Unsafe mode transition causing loss of control
        HAZARD: HARA-MODE-002 - Mode confusion leading to unexpected behavior

        Args:
            old_mode: Previous operation mode (optional)
            new_mode: New operation mode (MANUAL, AUTO, GUIDED, etc.)
        """
        # Handle backward compatibility - if only one arg passed, it's the new_mode
        if new_mode is None and old_mode is not None:
            new_mode = old_mode
            old_mode = None
        logger.info(f"Operation mode changed to: {new_mode}")

        # Handle mode-specific logic
        if new_mode == "MANUAL":
            # Disable automatic features
            self._homing_enabled = False
            if self._current_state == SystemState.HOMING:
                await self.transition_to(SystemState.IDLE, "Manual mode activated")

        elif new_mode == "AUTO":
            # Enable automatic features
            self._homing_enabled = True

        elif new_mode == "GUIDED":
            # Ready for guided operations
            if self._current_state == SystemState.DETECTING:
                # Can transition to homing if signal is strong
                pass

        # Notify callbacks
        for callback in self._state_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self._current_state, f"mode_change:{new_mode}")
                else:
                    callback(self._current_state, f"mode_change:{new_mode}")
            except Exception as e:
                logger.error(f"Error notifying callback of mode change: {e}")

    def update_waypoint_progress(self, waypoint_index: int) -> None:
        """Update waypoint completion progress.

        Args:
            waypoint_index: Index of completed waypoint
        """
        if not self._active_pattern:
            return

        self._current_waypoint_index = waypoint_index
        self._active_pattern.completed_waypoints = waypoint_index

        # Calculate progress percentage
        if self._active_pattern.total_waypoints > 0:
            self._active_pattern.progress_percent = (
                waypoint_index / self._active_pattern.total_waypoints * 100
            )

        # Check if pattern complete
        if waypoint_index >= self._active_pattern.total_waypoints:
            self._search_substate = SearchSubstate.IDLE
            self._active_pattern.state = "COMPLETED"
            self._active_pattern.progress_percent = 100.0
            logger.info(f"Search pattern {self._active_pattern.id} completed")

    def get_next_waypoint(self) -> Any | None:
        """Get the next waypoint to navigate to.

        Returns:
            Next waypoint or None if pattern complete
        """
        if not self._active_pattern or not self._active_pattern.waypoints:
            return None

        if self._current_waypoint_index >= len(self._active_pattern.waypoints):
            return None

        return self._active_pattern.waypoints[self._current_waypoint_index]

    def get_search_pattern_status(self) -> dict[str, Any]:
        """Get search pattern execution status.

        Returns:
            Dictionary with pattern status
        """
        if not self._active_pattern:
            return {"has_pattern": False, "search_substate": self._search_substate.value}

        return {
            "has_pattern": True,
            "pattern_id": self._active_pattern.id,
            "pattern_type": self._active_pattern.pattern_type.value,
            "search_substate": self._search_substate.value,
            "pattern_state": self._active_pattern.state,
            "total_waypoints": self._active_pattern.total_waypoints,
            "completed_waypoints": self._active_pattern.completed_waypoints,
            "current_waypoint": self._current_waypoint_index,
            "progress_percent": self._active_pattern.progress_percent,
            "estimated_time_remaining": self._active_pattern.estimated_time_remaining,
        }
