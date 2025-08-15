"""
Circuit Breaker Pattern Implementation

SAFETY: Prevents cascade failures from repeatedly failing callbacks
HAZARD: HARA-CB-001 - Callback failures could crash event loop
MITIGATION: Automatic circuit opening after threshold failures

This module implements the Circuit Breaker pattern to protect the system
from cascading failures caused by repeatedly failing callbacks.
"""

import asyncio
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class CircuitState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation, allowing calls
    OPEN = "open"  # Circuit is open, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior"""

    failure_threshold: int = 3  # Failures before opening
    success_threshold: int = 2  # Successes to close from half-open
    timeout: timedelta = timedelta(seconds=30)  # Time before trying half-open
    expected_exceptions: tuple[type[Exception], ...] = field(default_factory=lambda: (Exception,))


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""

    pass


class CallbackCircuitBreaker(Generic[T, R]):
    """
    Circuit breaker for callback functions.

    Monitors callback failures and automatically opens circuit to prevent
    cascade failures. After timeout, enters half-open state to test recovery.

    Example:
        breaker = CallbackCircuitBreaker[float, None](threshold=3)

        async def process_rssi(value: float) -> None:
            # Process RSSI value
            pass

        # Use breaker to protect callback
        await breaker.call_async(process_rssi, rssi_value)
    """

    def __init__(self, config: CircuitBreakerConfig | None = None, name: str = "CircuitBreaker"):
        """
        Initialize circuit breaker.

        Args:
            config: Circuit breaker configuration
            name: Name for logging
        """
        self.config = config or CircuitBreakerConfig()
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: datetime | None = None
        self._lock = asyncio.Lock()

    async def call_async(self, callback: Callable[[T], Coroutine[Any, Any, R]], value: T) -> R:
        """
        Call async callback through circuit breaker.

        Args:
            callback: Async callback function
            value: Value to pass to callback

        Returns:
            Result from callback

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: If callback fails (when circuit closed)
        """
        async with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info(f"{self.name}: Entering HALF_OPEN state")
                else:
                    raise CircuitBreakerError(f"{self.name}: Circuit breaker is OPEN")

            # Attempt the call
            try:
                result = await callback(value)
                await self._on_success()
                return result
            except self.config.expected_exceptions as e:
                await self._on_failure(e)
                raise

    def call_sync(self, callback: Callable[[T], R], value: T) -> R:
        """
        Call synchronous callback through circuit breaker.

        Args:
            callback: Sync callback function
            value: Value to pass to callback

        Returns:
            Result from callback

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: If callback fails (when circuit closed)
        """
        # Check state without lock (sync context)
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info(f"{self.name}: Entering HALF_OPEN state")
            else:
                raise CircuitBreakerError(f"{self.name}: Circuit breaker is OPEN")

        # Attempt the call
        try:
            result = callback(value)
            self._on_success_sync()
            return result
        except self.config.expected_exceptions as e:
            self._on_failure_sync(e)
            raise

    async def _on_success(self) -> None:
        """Handle successful callback execution"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info(f"{self.name}: Circuit CLOSED after recovery")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0  # Reset on success

    def _on_success_sync(self) -> None:
        """Handle successful sync callback execution"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info(f"{self.name}: Circuit CLOSED after recovery")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0  # Reset on success

    async def _on_failure(self, exception: Exception) -> None:
        """Handle callback failure"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        logger.warning(
            f"{self.name}: Callback failed ({self.failure_count}/{self.config.failure_threshold}): {exception}"
        )

        if self.state == CircuitState.HALF_OPEN:
            # Single failure in half-open reopens circuit
            self.state = CircuitState.OPEN
            logger.error(f"{self.name}: Circuit REOPENED after failure in HALF_OPEN")
        elif self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(f"{self.name}: Circuit OPENED after {self.failure_count} failures")

    def _on_failure_sync(self, exception: Exception) -> None:
        """Handle sync callback failure"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        logger.warning(
            f"{self.name}: Callback failed ({self.failure_count}/{self.config.failure_threshold}): {exception}"
        )

        if self.state == CircuitState.HALF_OPEN:
            # Single failure in half-open reopens circuit
            self.state = CircuitState.OPEN
            logger.error(f"{self.name}: Circuit REOPENED after failure in HALF_OPEN")
        elif self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(f"{self.name}: Circuit OPENED after {self.failure_count} failures")

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try half-open state"""
        if self.last_failure_time is None:
            return True
        return datetime.now() - self.last_failure_time > self.config.timeout

    def reset(self) -> None:
        """Manually reset circuit breaker to closed state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        logger.info(f"{self.name}: Circuit manually RESET to CLOSED")

    def get_state(self) -> dict[str, Any]:
        """Get current circuit breaker state for monitoring"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "timeout_seconds": self.config.timeout.total_seconds(),
            },
        }


class MultiCallbackCircuitBreaker:
    """
    Manages multiple circuit breakers for different callbacks.

    Useful when you have multiple callbacks that should be independently protected.
    """

    def __init__(self, default_config: CircuitBreakerConfig | None = None):
        """Initialize multi-callback circuit breaker manager"""
        self.default_config = default_config or CircuitBreakerConfig()
        self.breakers: dict[str, CallbackCircuitBreaker] = {}

    def get_breaker(
        self, callback_name: str, config: CircuitBreakerConfig | None = None
    ) -> CallbackCircuitBreaker:
        """Get or create circuit breaker for callback"""
        if callback_name not in self.breakers:
            breaker_config = config or self.default_config
            self.breakers[callback_name] = CallbackCircuitBreaker(
                config=breaker_config, name=f"CB_{callback_name}"
            )
        return self.breakers[callback_name]

    async def call_async(self, callback_name: str, callback: Callable, value: Any) -> Any:
        """Call callback through its circuit breaker"""
        breaker = self.get_breaker(callback_name)
        return await breaker.call_async(callback, value)

    def call_sync(self, callback_name: str, callback: Callable, value: Any) -> Any:
        """Call sync callback through its circuit breaker"""
        breaker = self.get_breaker(callback_name)
        return breaker.call_sync(callback, value)

    def get_all_states(self) -> dict[str, dict]:
        """Get state of all circuit breakers"""
        return {name: breaker.get_state() for name, breaker in self.breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuit breakers"""
        for breaker in self.breakers.values():
            breaker.reset()
