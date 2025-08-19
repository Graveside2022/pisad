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

from src.backend.services.sdr_priority_manager import SDRPriorityManager
from src.backend.utils.coordination_optimizer import CoordinationLatencyTracker
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
        self._safety_manager: Any | None = None

        # Priority management
        self._priority_manager: SDRPriorityManager | None = None

        # Performance tracking with enhanced latency measurement
        self._last_decision_time = 0.0
        self._coordination_latencies: list[float] = []
        self._latency_tracker = CoordinationLatencyTracker(
            max_samples=1000,
            alert_threshold_ms=50.0,  # Target coordination latency per Epic 5.3
            warning_threshold_ms=30.0,
        )

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

    def set_dependencies(
        self, signal_processor: Any = None, tcp_bridge: Any = None, safety_manager: Any = None
    ) -> None:
        """
        Set service dependencies and initialize priority manager.

        Args:
            signal_processor: Signal processing service
            tcp_bridge: TCP bridge service for SDR++ communication
            safety_manager: Safety management service
        """
        self._signal_processor = signal_processor
        self._tcp_bridge = tcp_bridge
        self._safety_manager = safety_manager

        # Initialize priority manager with dependencies
        self._priority_manager = SDRPriorityManager(coordinator=self, safety_manager=safety_manager)

        logger.info("Dependencies set and priority manager initialized")

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
        Make coordination decision using priority manager with enhanced latency tracking.
        """
        # Use high-precision latency measurement context manager
        async with self._latency_tracker.measure():
            await self._perform_coordination_decision()

    async def _perform_coordination_decision(self) -> None:
        """Internal coordination decision logic with timing measurement."""
        # Check ground connection status
        ground_available = self._tcp_bridge and getattr(self._tcp_bridge, "is_running", False)

        if not ground_available:
            # Handle communication loss via priority manager
            if self._priority_manager:
                await self._priority_manager.handle_communication_loss()
            # Activate fallback to drone-only
            self.active_source = "drone"
            self.fallback_active = True
        else:
            # Reset fallback if ground is available
            self.fallback_active = False

            # Use priority manager for intelligent source selection
            if self._priority_manager:
                try:
                    decision = await self._priority_manager.evaluate_source_switch()
                    if decision.switch_recommended:
                        self.active_source = decision.selected_source
                        logger.info(
                            f"Source switched to {decision.selected_source}, "
                            f"reason: {decision.reason}, latency: {decision.latency_ms:.1f}ms"
                        )
                except Exception as e:
                    logger.warning(f"Priority manager decision failed: {e}")
                    # Fall back to basic logic
                    self.active_source = "drone"  # Safety default

        # Update timing for legacy tracking
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

    async def resolve_frequency_conflict(
        self, ground_command: dict[str, Any], drone_command: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Resolve frequency command conflicts using priority manager.

        Args:
            ground_command: Frequency command from ground SDR++
            drone_command: Frequency command from drone PISAD

        Returns:
            Conflict resolution result
        """
        if self._priority_manager:
            try:
                resolution = await self._priority_manager.resolve_frequency_conflict(
                    ground_command, drone_command
                )
                return {
                    "selected_command": resolution.selected_command,
                    "conflict_type": resolution.conflict_type,
                    "resolution_time_ms": resolution.resolution_time_ms,
                    "rejected_command": resolution.rejected_command,
                }
            except Exception as e:
                logger.error(f"Conflict resolution failed: {e}")

        # Fallback: always choose drone command for safety
        return {
            "selected_command": drone_command,
            "conflict_type": "safety_fallback",
            "resolution_time_ms": 0.0,
            "rejected_command": ground_command,
        }

    async def trigger_emergency_override(self) -> dict[str, Any]:
        """
        Trigger emergency override through priority manager.

        Returns:
            Emergency override result with timing
        """
        if self._priority_manager:
            try:
                return await self._priority_manager.trigger_emergency_override()
            except Exception as e:
                logger.error(f"Emergency override via priority manager failed: {e}")

        # Direct emergency fallback
        self.active_source = "drone"
        self.fallback_active = True

        return {
            "source_switched_to": "drone",
            "safety_activated": False,
            "response_time_ms": 0.0,
            "emergency_override_active": True,
            "fallback_method": "direct",
        }

    async def get_priority_status(self) -> dict[str, Any]:
        """
        Get comprehensive priority status including coordination metrics.

        Returns:
            Enhanced status with priority management information
        """
        base_status = await self.get_health_status()

        if self._priority_manager:
            try:
                priority_status = await self._priority_manager.get_priority_status()
                base_status.update(
                    {"priority_management": priority_status, "priority_manager_active": True}
                )
            except Exception as e:
                logger.warning(f"Failed to get priority status: {e}")
                base_status["priority_manager_active"] = False
        else:
            base_status["priority_manager_active"] = False

        return base_status

    def get_ground_rssi(self) -> float | None:
        """
        Get current ground RSSI value for priority manager.

        Returns:
            Ground RSSI in dBm or None if unavailable
        """
        if (
            self._tcp_bridge
            and hasattr(self._tcp_bridge, "get_ground_rssi")
            and getattr(self._tcp_bridge, "is_running", False)
        ):
            try:
                return self._tcp_bridge.get_ground_rssi()
            except Exception as e:
                logger.warning(f"Failed to get ground RSSI: {e}")
        return None

    def get_drone_rssi(self) -> float | None:
        """
        Get current drone RSSI value for priority manager.

        Returns:
            Drone RSSI in dBm or None if unavailable
        """
        if self._signal_processor and hasattr(self._signal_processor, "get_current_rssi"):
            try:
                return self._signal_processor.get_current_rssi()
            except Exception as e:
                logger.warning(f"Failed to get drone RSSI: {e}")
        return None

    def get_latency_statistics(self) -> dict[str, Any]:
        """
        Get comprehensive coordination latency statistics.

        Returns:
            Dictionary with latency statistics and performance metrics
        """
        stats = self._latency_tracker.get_statistics()
        alerts = self._latency_tracker.check_alerts()

        return {
            "measurement_count": stats.count,
            "total_measurements": self._latency_tracker.total_measurements,
            "mean_latency_ms": round(stats.mean, 2),
            "min_latency_ms": round(stats.min_latency, 2),
            "max_latency_ms": round(stats.max_latency, 2),
            "p95_latency_ms": round(stats.p95, 2),
            "p99_latency_ms": round(stats.p99, 2),
            "std_deviation_ms": round(stats.std_dev, 2),
            "current_latency_ms": self._latency_tracker.get_current_latency(),
            "meets_requirements": self._latency_tracker.is_meeting_requirements(),
            "active_alerts": len(alerts),
            "alert_details": [
                {
                    "level": alert.level,
                    "threshold_ms": alert.threshold_ms,
                    "measured_ms": alert.measured_latency_ms,
                    "message": alert.message,
                }
                for alert in alerts
            ],
        }

    def reset_latency_tracking(self) -> None:
        """Reset latency measurements for fresh performance analysis."""
        self._latency_tracker.reset()
        logger.info("Coordination latency tracking reset")

    def get_performance_summary(self) -> dict[str, Any]:
        """
        Get comprehensive performance summary including latency and coordination metrics.

        Returns:
            Performance summary with timing, source selection, and health metrics
        """
        latency_stats = self.get_latency_statistics()

        return {
            "coordination_latency": latency_stats,
            "active_source": self.active_source,
            "fallback_active": self.fallback_active,
            "coordination_interval_ms": self.coordination_interval * 1000,
            "last_decision_time": self._last_decision_time,
            "priority_manager_available": self._priority_manager is not None,
            "ground_connection_available": (
                self._tcp_bridge and getattr(self._tcp_bridge, "is_running", False)
            ),
            "performance_status": (
                "optimal"
                if latency_stats["meets_requirements"] and latency_stats["active_alerts"] == 0
                else "degraded"
                if latency_stats["active_alerts"] > 0
                else "baseline"
            ),
        }
