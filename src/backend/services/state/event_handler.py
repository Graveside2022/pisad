"""State entry/exit actions and callback management."""

import asyncio

from src.backend.services.state.types import (
    AsyncStateCallback,
    StateCallback,
    SystemState,
)
from src.backend.utils.logging import get_logger

logger = get_logger(__name__)


class StateEventHandler:
    """Handles state entry/exit actions and callbacks.

    This class is responsible ONLY for:
    - Managing entry/exit actions
    - Executing callbacks on state changes
    - Handling both sync and async actions

    Cyclomatic complexity: <20
    """

    def __init__(self):
        """Initialize empty action and callback registries."""
        self._entry_actions: dict[SystemState, list[StateCallback | AsyncStateCallback]] = {}
        self._exit_actions: dict[SystemState, list[StateCallback | AsyncStateCallback]] = {}
        self._state_callbacks: list[StateCallback | AsyncStateCallback] = []

    def register_entry_action(
        self,
        state: SystemState,
        action: StateCallback | AsyncStateCallback,
    ) -> None:
        """Register an action to execute when entering a state.

        Args:
            state: State to register action for
            action: Callable to execute on state entry
        """
        if state not in self._entry_actions:
            self._entry_actions[state] = []
        self._entry_actions[state].append(action)
        logger.debug(f"Registered entry action for {state.value}")

    def register_exit_action(
        self,
        state: SystemState,
        action: StateCallback | AsyncStateCallback,
    ) -> None:
        """Register an action to execute when exiting a state.

        Args:
            state: State to register action for
            action: Callable to execute on state exit
        """
        if state not in self._exit_actions:
            self._exit_actions[state] = []
        self._exit_actions[state].append(action)
        logger.debug(f"Registered exit action for {state.value}")

    def add_state_callback(self, callback: StateCallback | AsyncStateCallback) -> None:
        """Add a callback to be notified of all state changes.

        Args:
            callback: Callable to execute on any state change
        """
        self._state_callbacks.append(callback)
        logger.debug(f"Added state callback: {callback.__name__}")

    def remove_state_callback(self, callback: StateCallback | AsyncStateCallback) -> bool:
        """Remove a state change callback.

        Args:
            callback: Callback to remove

        Returns:
            True if callback was removed, False if not found
        """
        try:
            self._state_callbacks.remove(callback)
            logger.debug(f"Removed state callback: {callback.__name__}")
            return True
        except ValueError:
            return False

    async def execute_entry_actions(self, state: SystemState) -> None:
        """Execute all entry actions for a state.

        Args:
            state: State being entered
        """
        actions = self._entry_actions.get(state, [])
        for action in actions:
            try:
                if asyncio.iscoroutinefunction(action):
                    await action(state)
                else:
                    action(state)
            except Exception as e:
                action_name = getattr(action, "__name__", "unknown")
                logger.error(f"Entry action '{action_name}' failed for {state.value}: {e}")

    async def execute_exit_actions(self, state: SystemState) -> None:
        """Execute all exit actions for a state.

        Args:
            state: State being exited
        """
        actions = self._exit_actions.get(state, [])
        for action in actions:
            try:
                if asyncio.iscoroutinefunction(action):
                    await action(state)
                else:
                    action(state)
            except Exception as e:
                action_name = getattr(action, "__name__", "unknown")
                logger.error(f"Exit action '{action_name}' failed for {state.value}: {e}")

    async def notify_callbacks(self, from_state: SystemState, to_state: SystemState) -> None:
        """Notify all callbacks of a state change.

        Args:
            from_state: Previous state
            to_state: New state
        """
        for callback in self._state_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback({"from": from_state, "to": to_state})
                else:
                    callback({"from": from_state, "to": to_state})
            except Exception as e:
                callback_name = getattr(callback, "__name__", "unknown")
                logger.error(f"State callback '{callback_name}' failed: {e}")

    def clear_actions(self) -> None:
        """Clear all registered actions and callbacks."""
        self._entry_actions.clear()
        self._exit_actions.clear()
        self._state_callbacks.clear()
        logger.debug("Cleared all actions and callbacks")

    def get_action_count(self) -> dict[str, int]:
        """Get count of registered actions.

        Returns:
            Dictionary with counts for each action type
        """
        entry_count = sum(len(actions) for actions in self._entry_actions.values())
        exit_count = sum(len(actions) for actions in self._exit_actions.values())

        return {
            "entry_actions": entry_count,
            "exit_actions": exit_count,
            "callbacks": len(self._state_callbacks),
            "total": entry_count + exit_count + len(self._state_callbacks),
        }
