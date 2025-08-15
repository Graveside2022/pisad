"""Refactored state machine with single responsibility components."""

from datetime import datetime
from typing import Any

from src.backend.core.exceptions import StateTransitionError
from src.backend.services.state import (
    StateChangeEvent,
    StateEventHandler,
    StateHistory,
    StatePersistence,
    StateTransitionManager,
    StateValidator,
    SystemState,
)
from src.backend.utils.logging import get_logger

logger = get_logger(__name__)


class StateMachineRefactored:
    """Orchestrates state management using specialized components.

    This refactored version delegates responsibilities to focused components:
    - StateTransitionManager: Core transition logic
    - StateValidator: Guard conditions and safety
    - StateEventHandler: Entry/exit actions and callbacks
    - StatePersistence: Database operations
    - StateHistory: Statistics and history tracking

    Cyclomatic complexity: <20 (orchestration only)
    """

    def __init__(
        self,
        db_path: str = "data/pisad.db",
        enable_persistence: bool = True,
        max_history: int = 100,
    ):
        """Initialize state machine with all components.

        Args:
            db_path: Path to SQLite database
            enable_persistence: Whether to enable state persistence
            max_history: Maximum history entries to keep in memory
        """
        # Initialize components
        self.transition_manager = StateTransitionManager()
        self.validator = StateValidator()
        self.event_handler = StateEventHandler()
        self.persistence = StatePersistence(db_path, enable_persistence)
        self.history = StateHistory(max_history)

        # External service references
        self._mavlink_service: Any | None = None
        self._signal_processor: Any | None = None

        # Restore previous state if persistence is enabled
        if enable_persistence:
            self._restore_state()

        logger.info("State machine initialized with refactored architecture")

    def _restore_state(self) -> None:
        """Restore previous state from database."""
        last_state = self.persistence.restore_last_state()
        if last_state and last_state != SystemState.IDLE:
            try:
                # Force transition without validation for restoration
                self.transition_manager._current_state = last_state
                logger.info(f"Restored state to {last_state.value}")
            except StateTransitionError as e:
                logger.error(f"Failed to restore state: {e}")
                self.transition_manager.reset()

    async def transition_to(self, new_state: SystemState, reason: str | None = None) -> bool:
        """Execute a state transition with full validation and actions.

        Args:
            new_state: Target state to transition to
            reason: Optional reason for the transition

        Returns:
            True if transition succeeded, False otherwise
        """
        current_state = self.transition_manager.get_current_state()

        # Check if transition is structurally valid
        if not self.transition_manager.can_transition(new_state):
            logger.warning(f"Invalid transition from {current_state.value} to {new_state.value}")
            return False

        # Validate with guards and safety checks
        valid, validation_reason = self.validator.validate_transition(current_state, new_state)
        if not valid:
            logger.warning(f"Transition validation failed: {validation_reason}")
            return False

        try:
            # Execute exit actions for current state
            await self.event_handler.execute_exit_actions(current_state)

            # Perform the transition
            self.transition_manager.transition(new_state)

            # Execute entry actions for new state
            await self.event_handler.execute_entry_actions(new_state)

            # Record the transition
            event = StateChangeEvent(
                from_state=current_state,
                to_state=new_state,
                timestamp=datetime.now(),
                reason=reason or validation_reason,
            )
            self.history.record_transition(event)

            # Persist if enabled
            self.persistence.save_state(
                new_state,
                previous_state=current_state,
                reason=reason,
            )

            # Notify callbacks
            await self.event_handler.notify_callbacks(current_state, new_state)

            # Send telemetry if MAVLink is available
            if self._mavlink_service:
                await self._send_telemetry_update(new_state)

            logger.info(f"State transition completed: {current_state.value} → {new_state.value}")
            return True

        except StateTransitionError as e:
            logger.error(f"State transition failed: {e}")
            # Try to restore previous state
            self.transition_manager._current_state = current_state
            return False

    async def _send_telemetry_update(self, state: SystemState) -> None:
        """Send state update via MAVLink telemetry.

        Args:
            state: Current state to send
        """
        if self._mavlink_service:
            try:
                await self._mavlink_service.send_named_value_float(
                    "STATE", float(list(SystemState).index(state))
                )
            except DatabaseError as e:
                logger.error(f"Failed to send telemetry update: {e}")

    # Public API methods matching original interface

    def get_current_state(self) -> SystemState:
        """Get current system state."""
        return self.transition_manager.get_current_state()

    def get_state_string(self) -> str:
        """Get current state as string."""
        return self.transition_manager.get_current_state().value

    def get_allowed_transitions(self) -> list[SystemState]:
        """Get list of valid transitions from current state."""
        return self.transition_manager.get_allowed_transitions()

    def enable_homing(self, enabled: bool = True) -> None:
        """Enable or disable homing mode."""
        self.validator.enable_homing(enabled)

    def is_homing_enabled(self) -> bool:
        """Check if homing is enabled."""
        return self.validator.is_homing_enabled()

    def get_state_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent state history."""
        return self.history.get_recent_history(limit)

    def get_statistics(self) -> dict[str, Any]:
        """Get state machine statistics."""
        stats = self.history.get_statistics()
        stats["current_state"] = self.get_state_string()
        stats["homing_enabled"] = self.is_homing_enabled()
        return stats

    def get_state_duration(self) -> float:
        """Get duration of current state in seconds."""
        current_state = self.get_current_state()
        duration = self.history.get_current_state_duration(current_state)
        return duration if duration else 0.0

    def add_state_callback(self, callback: Any) -> None:
        """Add callback for state changes."""
        self.event_handler.add_state_callback(callback)

    def register_entry_action(self, state: SystemState, action: Any) -> None:
        """Register entry action for a state."""
        self.event_handler.register_entry_action(state, action)

    def register_exit_action(self, state: SystemState, action: Any) -> None:
        """Register exit action for a state."""
        self.event_handler.register_exit_action(state, action)

    def set_mavlink_service(self, mavlink_service: Any) -> None:
        """Set MAVLink service reference."""
        self._mavlink_service = mavlink_service
        logger.debug("MAVLink service registered")

    def set_signal_processor(self, signal_processor: Any) -> None:
        """Set signal processor reference."""
        self._signal_processor = signal_processor
        logger.debug("Signal processor registered")

    async def reset(self) -> None:
        """Reset state machine to IDLE."""
        await self.transition_to(SystemState.IDLE, reason="Manual reset")

    def get_component_stats(self) -> dict[str, Any]:
        """Get statistics from all components (for debugging).

        Returns:
            Component-level statistics
        """
        return {
            "transition_manager": {
                "current_state": self.get_state_string(),
                "allowed_transitions": [s.value for s in self.get_allowed_transitions()],
            },
            "validator": {
                "homing_enabled": self.is_homing_enabled(),
                "guard_count": self.validator.get_guard_count(),
            },
            "event_handler": self.event_handler.get_action_count(),
            "persistence": {
                "enabled": self.persistence.is_enabled(),
                "history_count": len(self.persistence.get_state_history(100)),
            },
            "history": self.history.get_statistics(),
        }
