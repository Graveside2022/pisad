"""
Dual SDR Coordination Service

Intelligent coordination between ground SDR++ and drone PISAD signal processing
with automatic fallback and safety preservation.

PRD References:
- FR1: Enhanced SDR interface with dual coordination
- FR6: Enhanced RSSI computation with data fusion
- NFR2: <100ms latency maintained through coordination
- NFR12: Deterministic timing for coordination decisions
"""

import asyncio
import contextlib
import time
from typing import Any

from src.backend.utils.logging import get_logger

logger = get_logger(__name__)


class DualSDRCoordinator:
    """
    Coordinates signal processing between ground SDR++ and drone PISAD systems.

    Provides intelligent source selection, automatic fallback, and performance
    optimization while maintaining safety authority with the drone PISAD system.
    """

    def __init__(self) -> None:
        """Initialize dual SDR coordinator with default configuration."""
        # Coordination timing requirements per PRD-NFR2 and Epic 5 Story 5.3
        self.coordination_interval = 0.05  # 50ms for <100ms total latency requirement
        self.fallback_timeout = 10.0  # 10 seconds per PRD specifications

        # Service state
        self.is_running = False
        self.active_source = "drone"  # Default to drone for safety
        self.fallback_active = False

        # Dependencies (will be injected)
        self._signal_processor: Any | None = None
        self._tcp_bridge: Any | None = None

        # Performance tracking
        self._last_decision_time = 0.0
        self._coordination_latencies: list[float] = []

        # Coordination loop task
        self._coordination_task: asyncio.Task[None] | None = None

        logger.info(
            "DualSDRCoordinator initialized with coordination_interval=%.3fs, "
            "fallback_timeout=%.1fs",
            self.coordination_interval,
            self.fallback_timeout,
        )

    async def start(self) -> None:
        """Start the coordination service."""
        if self.is_running:
            logger.warning("DualSDRCoordinator already running")
            return

        self.is_running = True
        self._coordination_task = asyncio.create_task(self._coordination_loop())
        logger.info("DualSDRCoordinator started")

    async def stop(self) -> None:
        """Stop the coordination service."""
        if not self.is_running:
            return

        self.is_running = False

        if self._coordination_task:
            self._coordination_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._coordination_task

        logger.info("DualSDRCoordinator stopped")

    async def synchronize_frequency(self, frequency: float) -> None:
        """
        Synchronize frequency between ground SDR++ and drone HackRF.

        Args:
            frequency: Target frequency in Hz
        """
        logger.info("Synchronizing frequency to %.3f GHz", frequency / 1e9)

        # Update drone SDR frequency
        if self._signal_processor and hasattr(self._signal_processor, "set_frequency"):
            self._signal_processor.set_frequency(frequency)

        # Update ground SDR++ frequency via TCP bridge
        if self._tcp_bridge and hasattr(self._tcp_bridge, "send_frequency_control"):
            await self._tcp_bridge.send_frequency_control(frequency)

    async def get_best_rssi(self) -> float:
        """
        Get best available RSSI from data fusion of ground and drone sources.

        Returns:
            Best RSSI value in dBm
        """
        ground_rssi = None
        drone_rssi = None

        # Get ground RSSI if available
        if (
            self._tcp_bridge
            and hasattr(self._tcp_bridge, "get_ground_rssi")
            and getattr(self._tcp_bridge, "is_running", False)
        ):
            try:
                ground_rssi = self._tcp_bridge.get_ground_rssi()
            except Exception as e:
                logger.warning("Failed to get ground RSSI: %s", e)

        # Get drone RSSI
        if self._signal_processor and hasattr(self._signal_processor, "get_current_rssi"):
            try:
                drone_rssi = self._signal_processor.get_current_rssi()
            except Exception as e:
                logger.warning("Failed to get drone RSSI: %s", e)

        # Select best source based on signal strength
        if ground_rssi is not None and drone_rssi is not None:
            if ground_rssi > drone_rssi:  # Higher RSSI is better
                self.active_source = "ground"
                return ground_rssi
            else:
                self.active_source = "drone"
                return drone_rssi
        elif drone_rssi is not None:
            self.active_source = "drone"
            return drone_rssi
        elif ground_rssi is not None:
            self.active_source = "ground"
            return ground_rssi
        else:
            # No signal available - fallback to drone
            self.active_source = "drone"
            self.fallback_active = True
            return -100.0  # Default weak signal

    async def make_coordination_decision(self) -> None:
        """
        Make coordination decision and measure latency for performance requirements.
        """
        start_time = time.perf_counter()

        # Check ground connection status
        ground_available = self._tcp_bridge and getattr(self._tcp_bridge, "is_running", False)

        if not ground_available:
            # Activate fallback to drone-only
            self.active_source = "drone"
            self.fallback_active = True
        else:
            # Reset fallback if ground is available
            self.fallback_active = False

        # Record decision latency
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        self._coordination_latencies.append(latency_ms)

        # Keep only recent latencies for performance monitoring
        if len(self._coordination_latencies) > 100:
            self._coordination_latencies = self._coordination_latencies[-50:]

        self._last_decision_time = time.time()

    async def select_best_source(self) -> str:
        """
        Select best signal source based on comparison logic.

        Returns:
            "ground" or "drone" based on signal quality
        """
        ground_rssi = None
        drone_rssi = None

        # Get RSSI values
        if (
            self._tcp_bridge
            and hasattr(self._tcp_bridge, "get_ground_rssi")
            and getattr(self._tcp_bridge, "is_running", False)
        ):
            with contextlib.suppress(Exception):
                ground_rssi = self._tcp_bridge.get_ground_rssi()

        if self._signal_processor and hasattr(self._signal_processor, "get_current_rssi"):
            with contextlib.suppress(Exception):
                drone_rssi = self._signal_processor.get_current_rssi()

        # Selection logic - prefer drone for safety when equal
        if ground_rssi is not None and drone_rssi is not None:
            if ground_rssi > drone_rssi:
                return "ground"
            else:
                return "drone"  # Prefer drone for safety
        elif drone_rssi is not None:
            return "drone"
        elif ground_rssi is not None:
            return "ground"
        else:
            return "drone"  # Default to drone for safety

    async def get_health_status(self) -> dict[str, Any]:
        """
        Get comprehensive health status for monitoring.

        Returns:
            Health status dictionary with coordination metrics
        """
        # Calculate average coordination latency
        avg_latency = 0.0
        if self._coordination_latencies:
            avg_latency = sum(self._coordination_latencies) / len(self._coordination_latencies)

        return {
            "coordination_active": self.is_running,
            "active_source": self.active_source,
            "ground_connection_status": (
                getattr(self._tcp_bridge, "is_running", False) if self._tcp_bridge else False
            ),
            "drone_signal_quality": (
                self._signal_processor.get_current_rssi()
                if self._signal_processor and hasattr(self._signal_processor, "get_current_rssi")
                else None
            ),
            "coordination_latency_ms": avg_latency,
            "fallback_active": self.fallback_active,
            "last_decision_timestamp": self._last_decision_time,
        }

    async def _coordination_loop(self) -> None:
        """Main coordination loop running at specified interval."""
        logger.info(
            "Starting coordination loop with %.1fms interval", self.coordination_interval * 1000
        )

        while self.is_running:
            try:
                await self.make_coordination_decision()
                await asyncio.sleep(self.coordination_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in coordination loop: %s", e)
                await asyncio.sleep(self.coordination_interval)

        logger.info("Coordination loop stopped")
