"""Type definitions for state management."""

from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

# Type aliases for clarity
StateCallback = Callable[[Any], None]
AsyncStateCallback = Callable[[Any], Coroutine[Any, Any, None]]
GuardCondition = Callable[[], bool]


class SystemState(Enum):
    """System operational states."""

    IDLE = "IDLE"
    SEARCHING = "SEARCHING"
    DETECTING = "DETECTING"
    HOMING = "HOMING"
    HOLDING = "HOLDING"


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
    metadata: dict[str, Any] | None = None


@dataclass
class TransitionRule:
    """Defines a valid state transition with optional guard."""

    from_state: SystemState
    to_state: SystemState
    guard: GuardCondition | None = None
    description: str | None = None
