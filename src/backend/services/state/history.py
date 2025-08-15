"""State change history and statistics tracking."""

from collections import Counter, defaultdict, deque
from datetime import datetime
from typing import Any

from src.backend.services.state.types import StateChangeEvent, SystemState
from src.backend.utils.logging import get_logger

logger = get_logger(__name__)


class StateHistory:
    """Tracks state change history and provides statistics.

    This class is responsible ONLY for:
    - Recording state transitions
    - Calculating state durations
    - Providing statistical analysis

    Cyclomatic complexity: <10
    """

    def __init__(self, max_history: int = 100):
        """Initialize history tracker with maximum size.

        Args:
            max_history: Maximum number of events to keep in memory
        """
        self._history: deque[StateChangeEvent] = deque(maxlen=max_history)
        self._state_durations: dict[SystemState, list[float]] = defaultdict(list)
        self._state_entry_times: dict[SystemState, datetime] = {}
        self._transition_counts: Counter[tuple[SystemState, SystemState]] = Counter()

    def record_transition(self, event: StateChangeEvent) -> None:
        """Record a state transition event.

        Args:
            event: State change event to record
        """
        # Add to history
        self._history.append(event)

        # Update transition counter
        self._transition_counts[(event.from_state, event.to_state)] += 1

        # Calculate duration for the state we're leaving
        if event.from_state in self._state_entry_times:
            duration = (event.timestamp - self._state_entry_times[event.from_state]).total_seconds()
            self._state_durations[event.from_state].append(duration)
            del self._state_entry_times[event.from_state]

        # Record entry time for new state
        self._state_entry_times[event.to_state] = event.timestamp

        logger.debug(f"Recorded transition: {event.from_state.value} → {event.to_state.value}")

    def get_current_state_duration(self, current_state: SystemState) -> float | None:
        """Get duration of current state.

        Args:
            current_state: The current system state

        Returns:
            Duration in seconds or None if state not tracked
        """
        if current_state in self._state_entry_times:
            duration = (datetime.now() - self._state_entry_times[current_state]).total_seconds()
            return duration
        return None

    def get_recent_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Get recent state change history.

        Args:
            limit: Maximum number of events to return (None for all)

        Returns:
            List of state change events as dictionaries
        """
        events = list(self._history)
        if limit:
            events = events[-limit:]

        return [
            {
                "from_state": event.from_state.value,
                "to_state": event.to_state.value,
                "timestamp": event.timestamp.isoformat(),
                "reason": event.reason,
                "metadata": event.metadata,
            }
            for event in events
        ]

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive state machine statistics.

        Returns:
            Dictionary containing various statistics
        """
        # Calculate average durations
        avg_durations = {}
        for state, durations in self._state_durations.items():
            if durations:
                avg_durations[state.value] = {
                    "average": sum(durations) / len(durations),
                    "min": min(durations),
                    "max": max(durations),
                    "total": sum(durations),
                    "count": len(durations),
                }

        # Count state visits
        state_visits = Counter[SystemState]()
        for event in self._history:
            state_visits[event.to_state] += 1

        # Most common transitions
        common_transitions = [
            {
                "from": from_state.value,
                "to": to_state.value,
                "count": count,
            }
            for (from_state, to_state), count in self._transition_counts.most_common(5)
        ]

        return {
            "total_transitions": len(self._history),
            "state_durations": avg_durations,
            "state_visit_counts": {state.value: count for state, count in state_visits.items()},
            "most_common_transitions": common_transitions,
            "history_size": len(self._history),
        }

    def clear(self) -> None:
        """Clear all history and statistics."""
        self._history.clear()
        self._state_durations.clear()
        self._state_entry_times.clear()
        self._transition_counts.clear()
        logger.debug("Cleared state history")

    def get_transition_count(self, from_state: SystemState, to_state: SystemState) -> int:
        """Get count of specific state transitions.

        Args:
            from_state: Source state
            to_state: Target state

        Returns:
            Number of times this transition occurred
        """
        return self._transition_counts.get((from_state, to_state), 0)
