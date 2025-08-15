"""
Base mock hardware interface for testing.

This module provides the base class for all hardware mocks to prevent
circular imports and ensure consistent testing interfaces.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class MockHardwareConfig:
    """Configuration for mock hardware devices."""

    device_id: str = "mock_device"
    sample_rate: float = 2e6
    frequency: float = 3.2e9
    gain: float = 30.0
    enabled: bool = True
    simulate_errors: bool = False
    error_rate: float = 0.0
    response_delay: float = 0.0


class MockHardwareInterface(ABC):
    """
    Base class for all hardware mock implementations.

    This abstract base class ensures all hardware mocks follow the same
    interface pattern, preventing circular imports and enabling easy
    substitution in tests.
    """

    def __init__(self, config: MockHardwareConfig | None = None):
        """Initialize mock hardware with optional configuration."""
        self.config = config or MockHardwareConfig()
        self._is_connected = False
        self._is_streaming = False
        self._error_count = 0
        self._call_count = {}

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the mock hardware device."""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the mock hardware device."""
        pass

    @abstractmethod
    async def configure(self, **kwargs) -> bool:
        """Configure the mock hardware device."""
        pass

    @abstractmethod
    async def start_streaming(self) -> bool:
        """Start data streaming from the mock device."""
        pass

    @abstractmethod
    async def stop_streaming(self) -> bool:
        """Stop data streaming from the mock device."""
        pass

    @abstractmethod
    async def get_status(self) -> dict[str, Any]:
        """Get current status of the mock device."""
        pass

    def reset_mock(self) -> None:
        """Reset mock state for testing."""
        self._is_connected = False
        self._is_streaming = False
        self._error_count = 0
        self._call_count.clear()

    def get_call_count(self, method_name: str) -> int:
        """Get number of times a method was called."""
        return self._call_count.get(method_name, 0)

    def _track_call(self, method_name: str) -> None:
        """Track method calls for verification."""
        self._call_count[method_name] = self._call_count.get(method_name, 0) + 1

    async def _simulate_delay(self) -> None:
        """Simulate response delay if configured."""
        if self.config.response_delay > 0:
            await asyncio.sleep(self.config.response_delay)

    def _should_fail(self) -> bool:
        """Determine if operation should fail based on error rate."""
        if not self.config.simulate_errors:
            return False
        import random

        return random.random() < self.config.error_rate
