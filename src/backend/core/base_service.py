"""
Base Service Class for PISAD Services

Provides common functionality for service lifecycle management.
"""

import asyncio
from abc import ABC, abstractmethod

from src.backend.utils.logging import get_logger

logger = get_logger(__name__)


class BaseService(ABC):
    """Base class for PISAD services with lifecycle management."""

    def __init__(self, service_name: str = "base_service"):
        """Initialize base service."""
        self.service_name = service_name
        self._is_running = False
        self._startup_time: float | None = None

    @property
    def is_running(self) -> bool:
        """Check if service is currently running."""
        return self._is_running

    @abstractmethod
    async def start_service(self) -> None:
        """Start the service. Must be implemented by subclasses."""
        pass

    @abstractmethod
    async def stop_service(self) -> None:
        """Stop the service. Must be implemented by subclasses."""
        pass

    async def start(self) -> None:
        """Start service with error handling."""
        try:
            if self._is_running:
                logger.warning(f"Service {self.service_name} is already running")
                return

            logger.info(f"Starting service: {self.service_name}")
            await self.start_service()

            self._is_running = True
            self._startup_time = asyncio.get_event_loop().time()

            logger.info(f"Service {self.service_name} started successfully")

        except Exception as e:
            logger.error(f"Failed to start service {self.service_name}: {e}")
            self._is_running = False
            raise

    async def stop(self) -> None:
        """Stop service with error handling."""
        try:
            if not self._is_running:
                logger.warning(f"Service {self.service_name} is not running")
                return

            logger.info(f"Stopping service: {self.service_name}")
            await self.stop_service()

            self._is_running = False
            self._startup_time = None

            logger.info(f"Service {self.service_name} stopped successfully")

        except Exception as e:
            logger.error(f"Error stopping service {self.service_name}: {e}")
            self._is_running = False
            raise

    async def restart(self) -> None:
        """Restart the service."""
        if self._is_running:
            await self.stop()
        await self.start()

    def get_status(self) -> dict:
        """Get service status information."""
        return {
            "service_name": self.service_name,
            "is_running": self._is_running,
            "startup_time": self._startup_time,
            "uptime_seconds": (
                asyncio.get_event_loop().time() - self._startup_time
                if self._startup_time else 0.0
            )
        }

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
        return False  # Don't suppress exceptions
