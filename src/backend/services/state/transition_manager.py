"""Core state transition management logic."""

from src.backend.core.exceptions import StateTransitionError
from src.backend.services.state.types import SystemState
from src.backend.utils.logging import get_logger

logger = get_logger(__name__)


class StateTransitionManager:
    """Manages state transitions and validation rules.

    This class is responsible ONLY for:
    - Maintaining current state
    - Defining valid transitions
    - Executing state changes

    Cyclomatic complexity: <10
    """

    def __init__(self):
        """Initialize with default IDLE state."""
        self._current_state = SystemState.IDLE
        self._previous_state = SystemState.IDLE
        self._transition_rules = self._init_transition_rules()

    def _init_transition_rules(self) -> dict[SystemState, set[SystemState]]:
        """Define valid state transitions based on PRD requirements.

        Returns:
            Mapping of states to their valid target states
        """
        return {
            SystemState.IDLE: {
                SystemState.SEARCHING,
                SystemState.DETECTING,
            },
            SystemState.SEARCHING: {
                SystemState.IDLE,
                SystemState.DETECTING,
                SystemState.HOLDING,
            },
            SystemState.DETECTING: {
                SystemState.HOMING,
                SystemState.SEARCHING,
                SystemState.IDLE,
            },
            SystemState.HOMING: {
                SystemState.HOLDING,
                SystemState.SEARCHING,
                SystemState.IDLE,
            },
            SystemState.HOLDING: {
                SystemState.SEARCHING,
                SystemState.IDLE,
            },
        }

    def get_current_state(self) -> SystemState:
        """Get the current system state."""
        return self._current_state

    def get_previous_state(self) -> SystemState:
        """Get the previous system state."""
        return self._previous_state

    def can_transition(self, to_state: SystemState) -> bool:
        """Check if transition to target state is valid.

        Args:
            to_state: Target state to transition to

        Returns:
            True if transition is valid, False otherwise
        """
        valid_targets = self._transition_rules.get(self._current_state, set())
        return to_state in valid_targets

    def get_allowed_transitions(self) -> list[SystemState]:
        """Get list of states we can transition to from current state.

        Returns:
            List of valid target states
        """
        return list(self._transition_rules.get(self._current_state, set()))

    def transition(self, to_state: SystemState) -> None:
        """Execute state transition if valid.

        Args:
            to_state: Target state to transition to

        Raises:
            StateTransitionError: If transition is not valid
        """
        if not self.can_transition(to_state):
            valid_states = self.get_allowed_transitions()
            raise StateTransitionError(
                f"Invalid transition from {self._current_state.value} to {to_state.value}. "
                f"Valid transitions: {[s.value for s in valid_states]}"
            )

        logger.info(f"State transition: {self._current_state.value} → {to_state.value}")
        self._previous_state = self._current_state
        self._current_state = to_state

    def reset(self) -> None:
        """Reset state machine to IDLE state."""
        logger.info(f"Resetting state machine from {self._current_state.value} to IDLE")
        self._previous_state = self._current_state
        self._current_state = SystemState.IDLE
