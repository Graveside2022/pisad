"""State transition validation with guard conditions."""

from typing import Any

from src.backend.core.exceptions import PISADException, SafetyInterlockError
from src.backend.services.state.types import GuardCondition, SystemState
from src.backend.utils.logging import get_logger

logger = get_logger(__name__)


class StateValidator:
    """Validates state transitions with guard conditions.

    This class is responsible ONLY for:
    - Registering guard conditions
    - Validating transitions against guards
    - Safety checking before transitions

    Cyclomatic complexity: <15
    """

    def __init__(self, safety_checker: Any | None = None):
        """Initialize validator with optional safety checker.

        Args:
            safety_checker: Optional safety interlock checker
        """
        self._guards: dict[tuple[SystemState, SystemState], list[GuardCondition]] = {}
        self._safety_checker = safety_checker
        self._homing_enabled = False

    def enable_homing(self, enabled: bool = True) -> None:
        """Enable or disable homing mode.

        Args:
            enabled: Whether homing should be enabled
        """
        self._homing_enabled = enabled
        logger.info(f"Homing mode {'enabled' if enabled else 'disabled'}")

    def is_homing_enabled(self) -> bool:
        """Check if homing mode is enabled."""
        return self._homing_enabled

    def register_guard(
        self,
        from_state: SystemState,
        to_state: SystemState,
        guard_fn: GuardCondition,
    ) -> None:
        """Register a guard condition for a specific transition.

        Args:
            from_state: Source state
            to_state: Target state
            guard_fn: Function that returns True if transition is allowed
        """
        key = (from_state, to_state)
        if key not in self._guards:
            self._guards[key] = []
        self._guards[key].append(guard_fn)
        logger.debug(f"Registered guard for {from_state.value} → {to_state.value}")

    def validate_transition(
        self, from_state: SystemState, to_state: SystemState
    ) -> tuple[bool, str]:
        """Validate transition with all guard conditions.

        Args:
            from_state: Source state
            to_state: Target state

        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        # Special check for HOMING state
        if to_state == SystemState.HOMING and not self._homing_enabled:
            return False, "Homing mode is not enabled"

        # Check safety interlock if available
        if self._safety_checker:
            try:
                if not self._safety_checker.is_safe(to_state):
                    return False, "Safety interlock engaged"
            except SafetyInterlockError as e:
                logger.error(f"Safety check failed: {e}")
                return False, f"Safety check error: {e!s}"

        # Check all registered guards for this transition
        guards = self._guards.get((from_state, to_state), [])
        for guard in guards:
            try:
                if not guard():
                    guard_name = getattr(guard, "__name__", "unknown")
                    return False, f"Guard condition failed: {guard_name}"
            except PISADException as e:
                logger.error(f"Guard execution failed: {e}")
                return False, f"Guard error: {e!s}"

        return True, "Valid transition"

    def clear_guards(self) -> None:
        """Clear all registered guard conditions."""
        self._guards.clear()
        logger.debug("Cleared all guard conditions")

    def get_guard_count(self) -> int:
        """Get total number of registered guards.

        Returns:
            Total count of guard conditions
        """
        return sum(len(guards) for guards in self._guards.values())
