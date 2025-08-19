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
from datetime import datetime
from typing import Any

from src.backend.services.sdr_priority_manager import SDRPriorityManager
from src.backend.services.safety_authority_manager import (
    SafetyAuthorityManager,
    SafetyAuthorityLevel,
    SafetyDecision,
    SafetyDecisionType,
)
from src.backend.utils.coordination_optimizer import CoordinationLatencyTracker
from src.backend.utils.logging import get_logger

logger = get_logger(__name__)


class DualSDRCoordinator:
    """
    Coordinates signal processing between ground SDR++ and drone PISAD systems.

    Provides intelligent source selection, automatic fallback, and performance
    optimization while maintaining safety authority with the drone PISAD system.
    """

    def __init__(
        self, 
        safety_authority: SafetyAuthorityManager | None = None,
        service_manager: Any = None
    ) -> None:
        """
        Initialize dual SDR coordinator with safety authority integration.
        
        SUBTASK-5.5.3.4 [11a] - Inject SafetyManager into constructor with proper lifecycle.
        
        Args:
            safety_authority: SafetyAuthorityManager for coordination safety validation
            service_manager: ServiceManager for dependency resolution
        """
        # SUBTASK-5.5.3.4 [11a] - Safety authority dependency injection
        self._safety_authority = safety_authority
        self._service_manager = service_manager
        
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
        # Note: _safety_authority already set above on line 56, don't override here

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

    async def initialize(self) -> None:
        """
        SUBTASK-5.5.3.4 [11a] - Initialize coordinator with safety validation.
        
        Validates that SafetyAuthorityManager is properly configured for coordination.
        
        Raises:
            ValueError: If SafetyAuthorityManager is not provided or invalid
        """
        if self._safety_authority is None:
            raise ValueError("SafetyAuthorityManager is required for safe coordination operations")
        
        # Validate safety authority is properly initialized
        if not hasattr(self._safety_authority, 'authorities') or not self._safety_authority.authorities:
            raise ValueError("SafetyAuthorityManager is not properly initialized")
        
        # Verify emergency response capability
        if SafetyAuthorityLevel.EMERGENCY_STOP not in self._safety_authority.authorities:
            raise ValueError("SafetyAuthorityManager missing emergency response capability")
        
        logger.info("DualSDRCoordinator initialized with safety authority integration")

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
        self, signal_processor: Any = None, tcp_bridge: Any = None, safety_manager: Any = None,
        safety_authority: SafetyAuthorityManager = None
    ) -> None:
        """
        Set service dependencies and initialize priority manager.

        Args:
            signal_processor: Signal processing service
            tcp_bridge: TCP bridge service for SDR++ communication
            safety_manager: Safety management service
            safety_authority: Safety authority manager for coordination decisions
        """
        self._signal_processor = signal_processor
        self._tcp_bridge = tcp_bridge
        self._safety_manager = safety_manager
        self._safety_authority = safety_authority

        # Initialize priority manager with dependencies
        self._priority_manager = SDRPriorityManager(coordinator=self, safety_manager=safety_manager)

        logger.info("Dependencies set and priority manager initialized")

    async def synchronize_frequency(self, frequency: float) -> None:
        """
        Synchronize frequency between ground SDR++ and drone HackRF.
        Implements graceful degradation in fallback mode per SUBTASK-5.5.2.2[2j].

        Args:
            frequency: Target frequency in Hz
        """
        logger.info("Synchronizing frequency to %.3f GHz", frequency / 1e9)

        # Always update drone SDR frequency (safety fallback)
        if self._signal_processor and hasattr(self._signal_processor, "set_frequency"):
            self._signal_processor.set_frequency(frequency)

        # Update ground SDR++ frequency via TCP bridge only if not in fallback mode
        if (
            not self.fallback_active
            and self._tcp_bridge
            and hasattr(self._tcp_bridge, "send_frequency_control")
        ):
            try:
                await self._tcp_bridge.send_frequency_control(frequency)
                logger.debug("Ground frequency synchronized to %.3f GHz", frequency / 1e9)
            except Exception as e:
                logger.warning("Failed to synchronize ground frequency, entering fallback: %s", e)
                await self._trigger_fallback_mode("frequency_sync_failed")

    async def get_best_rssi(self) -> float:
        """
        Get best available RSSI from data fusion of ground and drone sources.

        Enhanced with [2h1] zero-interruption source switching and buffering
        for seamless fallback transitions.

        Returns:
            Best RSSI value in dBm
        """
        ground_rssi = None
        drone_rssi = None

        # Get drone RSSI (always available for safety fallback)
        if self._signal_processor and hasattr(self._signal_processor, "get_current_rssi"):
            try:
                drone_rssi = self._signal_processor.get_current_rssi()
            except Exception as e:
                logger.warning("Failed to get drone RSSI: %s", e)

        # [2h1] Zero-interruption RSSI source switching:
        # During fallback mode, only use drone source for seamless operation
        if self.fallback_active:
            if drone_rssi is not None:
                # Ensure source remains drone during fallback for consistency
                self.active_source = "drone"
                return drone_rssi
            else:
                # Emergency fallback with buffered default if drone unavailable
                logger.error("Drone RSSI unavailable during fallback mode - using buffered default")
                self.active_source = "drone"
                return -100.0  # Buffered default for continuity

        # Normal operation: Get ground RSSI if available and not in fallback
        if (
            self._tcp_bridge
            and hasattr(self._tcp_bridge, "get_ground_rssi")
            and getattr(self._tcp_bridge, "is_running", False)
        ):
            try:
                ground_rssi = self._tcp_bridge.get_ground_rssi()
            except Exception as e:
                logger.warning("Failed to get ground RSSI: %s", e)

        # [2h1] Seamless source selection with buffering during normal operation
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
            # No signal available - trigger automatic fallback with buffering
            logger.warning("No RSSI sources available - triggering fallback mode")
            self.active_source = "drone"
            self.fallback_active = True
            return -100.0  # Buffered default for continuity

    async def make_coordination_decision(self) -> None:
        """
        Make coordination decision using priority manager with enhanced latency tracking.
        """
        # Use high-precision latency measurement context manager
        async with self._latency_tracker.measure():
            await self._perform_coordination_decision()

    async def _perform_coordination_decision(self) -> None:
        """Internal coordination decision logic with timing measurement and safety integration."""
        # [2u] Safety-aware coordination decision making - check safety status first
        safety_status = await self._check_safety_status()
        if not safety_status["safe"]:
            logger.warning(
                "Coordination decision delayed due to safety concern: %s", safety_status["reason"]
            )
            # [2w] Coordination shutdown on safety system failures - trigger fallback for safety
            await self._trigger_fallback_mode(f"safety_concern: {safety_status['reason']}")
            return

        # Check ground connection status
        ground_available = self._tcp_bridge and getattr(self._tcp_bridge, "is_running", False)

        if not ground_available and not self.fallback_active:
            # Trigger seamless fallback to drone-only mode per SUBTASK-5.5.2.2[2g]
            logger.info("Ground connection lost - triggering coordination fallback")
            await self._trigger_fallback_mode("communication_loss")
        elif ground_available and self.fallback_active:
            # Automatic recovery when ground communication restored per SUBTASK-5.5.2.2[2l]
            logger.info("Ground connection restored - triggering coordination recovery")
            await self._trigger_recovery_mode("communication_restored")
        else:
            # Reset fallback if ground is available (legacy behavior)
            if ground_available:
                self.fallback_active = False

            # Use priority manager for intelligent source selection
            if self._priority_manager:
                try:
                    decision = await self._priority_manager.evaluate_source_switch()
                    if decision.switch_recommended:
                        old_source = self.active_source
                        self.active_source = decision.selected_source
                        # [2x] Safety event logging for coordination operations
                        logger.info(
                            f"Source switched from {old_source} to {decision.selected_source}, "
                            f"reason: {decision.reason}, latency: {decision.latency_ms:.1f}ms, "
                            f"safety_checked: {safety_status['safe']}"
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

    async def select_best_source_with_safety_validation(self) -> dict[str, Any]:
        """
        Select best signal source with safety authority validation.
        
        Implements [9c] - Safety-first decision making in coordination priority choices.
        
        Returns:
            Dictionary with source selection and safety validation results
        """
        # Get basic source selection
        proposed_source = await self.select_best_source()
        
        # If no safety authority, use basic selection
        if not self._safety_authority:
            logger.warning("No safety authority available - using basic source selection")
            return {
                "selected_source": proposed_source,
                "safety_validated": False,
                "safety_reason": "No safety authority manager",
                "response_time_ms": 0,
            }
        
        # Create safety decision for source selection
        safety_decision = SafetyDecision(
            decision_type=SafetyDecisionType.SOURCE_SELECTION,
            requesting_authority=SafetyAuthorityLevel.COMMUNICATION,  # Communication level for source selection
            details={
                "proposed_source": proposed_source,
                "current_source": self.active_source,
                "coordination_component": "DualSDRCoordinator",
                "ground_available": self._tcp_bridge and getattr(self._tcp_bridge, "is_running", False),
                "drone_available": self._signal_processor is not None,
            }
        )
        
        # Validate decision through safety authority
        try:
            approved, reason, approving_authority = await self._safety_authority.validate_safety_decision(safety_decision)
            
            if approved:
                # Safety approved - use proposed source
                selected_source = proposed_source
                logger.info(f"Source selection '{proposed_source}' approved by {approving_authority} ({safety_decision.response_time_ms}ms)")
            else:
                # Safety rejected - fallback to drone for safety
                selected_source = "drone"
                logger.warning(f"Source selection '{proposed_source}' rejected by safety authority: {reason}")
            
            return {
                "selected_source": selected_source,
                "safety_validated": approved,
                "safety_reason": reason,
                "approving_authority": approving_authority,
                "response_time_ms": safety_decision.response_time_ms,
                "safety_compliant": safety_decision.response_time_ms <= 10000,  # <10s requirement
            }
            
        except Exception as e:
            logger.error(f"Safety validation failed: {e} - defaulting to drone")
            return {
                "selected_source": "drone",  # Safe default
                "safety_validated": False,
                "safety_reason": f"Safety validation error: {e}",
                "response_time_ms": 0,
            }

    async def trigger_emergency_safety_override(self, reason: str) -> dict[str, Any]:
        """
        Trigger emergency safety override that bypasses all coordination.
        
        Implements [9c] - Emergency pathways with <500ms response time.
        
        Args:
            reason: Reason for emergency override
            
        Returns:
            Emergency override result with timing
        """
        start_time = asyncio.get_event_loop().time()
        
        logger.critical(f"EMERGENCY SAFETY OVERRIDE TRIGGERED: {reason}")
        
        # Immediately switch to drone-only mode
        old_source = self.active_source
        self.active_source = "drone"
        self.fallback_active = True
        
        # Trigger safety authority emergency override if available
        safety_override_result = {}
        if self._safety_authority:
            try:
                safety_override_result = await self._safety_authority.trigger_emergency_override(reason)
            except Exception as e:
                logger.error(f"Safety authority emergency override failed: {e}")
        
        # Stop coordination loop if running
        if hasattr(self, '_coordination_task') and self._coordination_task:
            self._coordination_task.cancel()
            logger.warning("Coordination task cancelled due to emergency override")
        
        end_time = asyncio.get_event_loop().time()
        response_time_ms = int((end_time - start_time) * 1000)
        
        result = {
            "emergency_override_active": True,
            "source_switched_to": "drone",
            "previous_source": old_source,
            "fallback_active": True,
            "coordination_stopped": True,
            "response_time_ms": response_time_ms,
            "safety_requirement_met": response_time_ms <= 500,  # <500ms emergency requirement
            "trigger_reason": reason,
            "safety_authority_override": safety_override_result,
            "timestamp": asyncio.get_event_loop().time(),
        }
        
        logger.critical(f"Emergency override completed in {response_time_ms}ms - coordination disabled")
        
        return result

    async def get_health_status(self) -> dict[str, Any]:
        """
        Get comprehensive health status for monitoring.

        Enhanced with [2i1] detailed fallback metrics and timing for operational visibility.

        Returns:
            Health status dictionary with coordination metrics and detailed fallback analytics
        """
        # Calculate average coordination latency
        avg_latency = 0.0
        if self._coordination_latencies:
            avg_latency = sum(self._coordination_latencies) / len(self._coordination_latencies)

        # [2i1] Calculate fallback duration and metrics
        current_time = time.time()
        fallback_duration_ms = 0.0
        if self.fallback_active and hasattr(self, "_fallback_start_time"):
            fallback_duration_ms = (current_time - self._fallback_start_time) * 1000

        # [2i2] Get performance analytics for fallback tracking
        health_status = {
            "coordination_active": self.is_running,
            "active_source": self.active_source,
            "ground_connection_status": (
                getattr(self._tcp_bridge, "is_running", False) if self._tcp_bridge else False
            ),
            "drone_signal_quality": (
                self._signal_processor.get_current_rssi()
                if self._signal_processor and hasattr(self._signal_processor, "get_current_rssi")
                else -100.0
            ),
            "coordination_latency_ms": avg_latency,
            "fallback_active": self.fallback_active,
            "last_decision_timestamp": self._last_decision_time,
        }

        # [2i1] Add detailed fallback metrics when fallback is active or has been triggered
        if self.fallback_active or hasattr(self, "_fallback_trigger_count"):
            health_status.update(
                {
                    "fallback_duration_ms": fallback_duration_ms,
                    "fallback_trigger_count": getattr(self, "_fallback_trigger_count", 0),
                    "fallback_start_time": getattr(self, "_fallback_start_time", 0.0),
                    "fallback_reason": getattr(self, "_fallback_reason", "unknown"),
                }
            )

        # [2i5] Add dashboard-compatible metrics
        if hasattr(self, "_fallback_scenarios") and self._fallback_scenarios:
            health_status["last_fallback_scenario"] = self._fallback_scenarios[-1]

        # [2v] Add safety status monitoring to coordination health checks
        try:
            safety_status = await self._check_safety_status()
            health_status.update(
                {
                    "safety_status": safety_status["safe"],
                    "safety_reason": safety_status["reason"],
                    "safety_manager_available": self._safety_manager is not None,
                }
            )
        except Exception as e:
            health_status.update(
                {
                    "safety_status": False,
                    "safety_reason": f"safety_check_error: {e}",
                    "safety_manager_available": False,
                }
            )

        return health_status

    # SUBTASK-5.5.2.2 [2j] Implementation: Graceful degradation of coordination features

    async def get_available_features(self) -> dict[str, Any]:
        """
        [2j1] Get feature availability matrix for current operation mode.

        Returns:
            Dictionary with available and unavailable features based on current mode
        """
        if self.fallback_active:
            # Drone-only mode features (reduced functionality)
            available_features = [
                "drone_source_only",
                "safety_fallback_operation",
                "basic_coordination",
                "emergency_coordination",
            ]

            unavailable_features = [
                "dual_source_coordination",
                "ground_sdr_integration",
                "tcp_bridge_communication",
                "real_time_rssi_comparison",
                "automatic_source_switching",
                "frequency_synchronization",
                "priority_based_selection",
                "performance_optimization",
            ]

            return {
                "mode": "drone_only",
                "available": available_features,
                "unavailable": unavailable_features,
                "degradation_level": "partial",
                "feature_count": {
                    "available": len(available_features),
                    "unavailable": len(unavailable_features),
                    "total": len(available_features) + len(unavailable_features),
                },
            }
        else:
            # Normal dual-source mode features (full functionality)
            available_features = [
                "dual_source_coordination",
                "ground_sdr_integration",
                "automatic_source_switching",
                "frequency_synchronization",
                "tcp_bridge_communication",
                "real_time_rssi_comparison",
                "priority_based_selection",
                "performance_optimization",
            ]

            return {
                "mode": "dual_source",
                "available": available_features,
                "unavailable": [],
                "degradation_level": "none",
                "feature_count": {
                    "available": len(available_features),
                    "unavailable": 0,
                    "total": len(available_features),
                },
            }

    async def get_degradation_status(self) -> dict[str, Any]:
        """
        [2j4] Get comprehensive degradation status for monitoring.

        Returns:
            Degradation status with detailed metrics and classification
        """
        if not self.fallback_active:
            return {
                "degradation_active": False,
                "degradation_level": "none",
                "disabled_features_count": 0,
                "available_features_count": 8,  # Full feature set
                "degradation_reason": None,
                "degradation_start_time": None,
                "performance_impact_level": "none",
            }

        # Get current feature matrix for degradation analysis
        features = await self.get_available_features()

        # Calculate performance impact based on disabled features
        disabled_count = features["feature_count"]["unavailable"]
        if disabled_count >= 6:
            impact_level = "high"
            degradation_level = "significant"
        elif disabled_count >= 3:
            impact_level = "medium"
            degradation_level = "partial"
        else:
            impact_level = "low"
            degradation_level = "minimal"

        return {
            "degradation_active": True,
            "degradation_level": degradation_level,
            "disabled_features_count": features["feature_count"]["unavailable"],
            "available_features_count": features["feature_count"]["available"],
            "degradation_reason": getattr(self, "_fallback_reason", "unknown"),
            "degradation_start_time": getattr(self, "_fallback_start_time", 0.0),
            "performance_impact_level": impact_level,
            "disabled_features": features["unavailable"],
            "available_features": features["available"],
        }

    async def _notify_feature_degradation(
        self, notification_type: str, details: dict[str, Any]
    ) -> None:
        """
        [2j2] Notify operator of feature degradation during fallback.

        Args:
            notification_type: Type of degradation notification
            details: Detailed information about the degradation
        """
        try:
            # Log the feature degradation for audit trails
            logger.warning(
                "Feature degradation notification: %s - %d features disabled, %d available",
                notification_type,
                details.get("disabled_features_count", 0),
                details.get("available_features_count", 0),
            )

            # Integration with existing notification system via TCP bridge
            if self._tcp_bridge and hasattr(self._tcp_bridge, "auto_notify_communication_issue"):
                notification_data = {
                    "degradation_type": notification_type,
                    "timestamp": time.time(),
                    **details,
                }
                await self._tcp_bridge.auto_notify_communication_issue(
                    "feature_degradation", notification_data
                )

            # Also notify safety manager if available
            if self._safety_manager and hasattr(self._safety_manager, "handle_communication_loss"):
                degradation_event = {
                    "event_type": "feature_degradation",
                    "notification_type": notification_type,
                    "details": details,
                    "timestamp": time.time(),
                }

                if asyncio.iscoroutinefunction(self._safety_manager.handle_communication_loss):
                    await self._safety_manager.handle_communication_loss(degradation_event)
                else:
                    self._safety_manager.handle_communication_loss(degradation_event)

        except Exception as e:
            logger.error("Failed to send feature degradation notification: %s", e)

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
                else "degraded" if latency_stats["active_alerts"] > 0 else "baseline"
            ),
        }

    # SUBTASK-5.5.2.2 Implementation: Seamless drone-only operation fallback

    async def _trigger_fallback_mode(self, reason: str) -> None:
        """
        [2g] Trigger seamless fallback to drone-only mode without flight interruption.

        Enhanced with comprehensive state management per SUBTASK-5.5.2.2[2g1-2g5].

        Args:
            reason: Reason for fallback activation
        """
        logger.warning("Triggering fallback to drone-only mode: %s", reason)

        # [2g1] Comprehensive state management - track fallback timing and state
        fallback_start_time = time.time()
        self._fallback_start_time = fallback_start_time
        self._last_fallback_time = fallback_start_time
        self._fallback_reason = reason
        self._fallback_trigger_count = getattr(self, "_fallback_trigger_count", 0) + 1

        # [2g2] Automatic source switching to drone-only mode for safety authority
        previous_source = self.active_source
        self.active_source = "drone"  # Always switch to drone for safety priority

        # [2g3] Fallback state persistence - mark as active for coordination decisions
        self.fallback_active = True
        self._fallback_state_validated = True

        # [2g4] Multiple failure scenario support - store current failure context
        if not hasattr(self, "_fallback_scenarios"):
            self._fallback_scenarios = []
        self._fallback_scenarios.append(
            {
                "reason": reason,
                "timestamp": fallback_start_time,
                "previous_source": previous_source,
                "trigger_count": self._fallback_trigger_count,
            }
        )

        # [2g5] State validation and integrity checks
        await self._validate_fallback_state_integrity()

        # [2h] Ensure seamless transition without flight operation interruption
        # (Coordination loop continues running, no service interruption)

        # [2k] Create operator notification system for fallback activation
        await self._notify_operator_fallback_activation(reason)

        # [2j2] Intelligent feature disabling with user notification
        try:
            features = await self.get_available_features()
            degradation_details = {
                "disabled_features": features["unavailable"],
                "available_features": features["available"],
                "disabled_features_count": features["feature_count"]["unavailable"],
                "available_features_count": features["feature_count"]["available"],
                "degradation_level": features["degradation_level"],
                "fallback_reason": reason,
            }
            await self._notify_feature_degradation("feature_degradation", degradation_details)
        except Exception as e:
            logger.warning("Failed to notify feature degradation during fallback: %s", e)

        # [2i] Add fallback status monitoring and reporting
        logger.info(
            "Fallback mode activated: source=%s->%s, reason=%s, trigger_count=%d, timing=%.1fms",
            previous_source,
            self.active_source,
            reason,
            self._fallback_trigger_count,
            (time.time() - fallback_start_time) * 1000,
        )

    async def _validate_fallback_state_integrity(self) -> bool:
        """
        [2g5] Validate fallback state integrity and consistency.

        Ensures fallback state is properly maintained and all required
        attributes are correctly set for seamless operation.

        Returns:
            True if state integrity is valid, raises exception otherwise
        """
        try:
            # Validate required fallback state attributes exist
            required_attrs = [
                "fallback_active",
                "_fallback_start_time",
                "_last_fallback_time",
                "_fallback_reason",
                "_fallback_trigger_count",
            ]

            for attr in required_attrs:
                if not hasattr(self, attr):
                    raise ValueError(f"Missing required fallback state attribute: {attr}")

            # Validate state consistency
            if not self.fallback_active:
                raise ValueError("fallback_active must be True during fallback validation")

            if self.active_source != "drone":
                raise ValueError(
                    f"active_source must be 'drone' during fallback, got: {self.active_source}"
                )

            # Validate timing consistency
            current_time = time.time()
            if self._fallback_start_time > current_time:
                raise ValueError("Fallback start time cannot be in the future")

            if self._last_fallback_time < self._fallback_start_time:
                raise ValueError("Last fallback time cannot be before start time")

            # Mark state as validated
            self._fallback_state_validated = True

            logger.debug("Fallback state integrity validation passed")
            return True

        except Exception as e:
            logger.error(f"Fallback state integrity validation failed: {e}")
            self._fallback_state_validated = False
            raise

    async def _trigger_recovery_mode(self, reason: str) -> None:
        """
        [2l] Trigger automatic recovery when ground communication restored.

        Args:
            reason: Reason for recovery
        """
        logger.info("Triggering recovery from fallback mode: %s", reason)

        # Reset fallback state
        self.fallback_active = False

        # Select best source based on signal quality
        ground_rssi = self.get_ground_rssi()
        drone_rssi = self.get_drone_rssi()

        if ground_rssi is not None and drone_rssi is not None:
            if ground_rssi > drone_rssi:
                self.active_source = "ground"
            else:
                self.active_source = "drone"
        elif ground_rssi is not None:
            self.active_source = "ground"
        else:
            self.active_source = "drone"

        # Notify safety manager of recovery
        if self._safety_manager and hasattr(self._safety_manager, "handle_communication_restored"):
            try:
                await self._safety_manager.handle_communication_restored()
                logger.info("Safety manager notified of communication recovery")
            except Exception as e:
                logger.error("Failed to notify safety manager of recovery: %s", e)

        logger.info("Recovery mode completed: source=%s, reason=%s", self.active_source, reason)

    async def _notify_operator_fallback_activation(self, reason: str) -> None:
        """
        [2k] Notify operator of fallback activation through notification system.

        Args:
            reason: Reason for fallback activation
        """
        try:
            # Notify through TCP bridge notification system if available
            if self._tcp_bridge and hasattr(self._tcp_bridge, "auto_notify_communication_issue"):
                await self._tcp_bridge.auto_notify_communication_issue(
                    "fallback_activated",
                    {
                        "reason": reason,
                        "active_source": self.active_source,
                        "fallback_active": self.fallback_active,
                        "timestamp": time.time(),
                    },
                )

            # Also notify safety manager if available
            if self._safety_manager and hasattr(self._safety_manager, "handle_communication_loss"):
                safety_event = {
                    "event_type": "coordination_fallback",
                    "reason": reason,
                    "active_source": self.active_source,
                    "timestamp": time.time(),
                }
                # Check if it's async before awaiting
                if asyncio.iscoroutinefunction(self._safety_manager.handle_communication_loss):
                    await self._safety_manager.handle_communication_loss(safety_event)
                else:
                    self._safety_manager.handle_communication_loss(safety_event)

        except Exception as e:
            logger.error("Failed to notify operator of fallback activation: %s", e)

    async def _check_safety_status(self) -> dict[str, Any]:
        """
        [2u][2v] Check safety status for coordination decision making.

        Returns safety status from safety manager for safety-aware coordination.

        Returns:
            Safety status dictionary with safe/unsafe determination and reason
        """
        if not self._safety_manager:
            return {"safe": True, "reason": "no_safety_manager"}

        try:
            # Check if safety manager has safety status methods
            if hasattr(self._safety_manager, "is_safe_to_operate"):
                is_safe = self._safety_manager.is_safe_to_operate()
                if asyncio.iscoroutine(is_safe):
                    is_safe = await is_safe

                if not is_safe:
                    return {"safe": False, "reason": "safety_manager_unsafe"}

            if hasattr(self._safety_manager, "get_safety_status"):
                status = self._safety_manager.get_safety_status()
                if asyncio.iscoroutine(status):
                    status = await status

                if isinstance(status, dict):
                    if status.get("emergency_stopped", False):
                        return {"safe": False, "reason": "emergency_stopped"}
                    if status.get("critical_alerts"):
                        return {
                            "safe": False,
                            "reason": f"critical_alerts: {status['critical_alerts']}",
                        }
                    if not status.get("safe", True):
                        return {"safe": False, "reason": "safety_status_unsafe"}

            # If we get here, safety manager indicates safe operation
            return {"safe": True, "reason": "safety_manager_ok"}

        except Exception as e:
            logger.error("Error checking safety status: %s", e)
            # [2w] On safety system failure, assume unsafe for coordination
            return {"safe": False, "reason": f"safety_check_failed: {e}"}

    def validate_command_before_execution(self, command: str, params: dict[str, Any]) -> dict[str, Any]:
        """
        SUBTASK-5.5.3.2 [9d] - Validate coordination commands before execution.
        
        Integrates SafetyAuthorityManager validation into all coordination commands.
        
        Args:
            command: Command type to validate
            params: Command parameters
            
        Returns:
            Dict containing validation result
        """
        if not self._safety_authority:
            logger.warning("No safety authority available for command validation")
            return {
                "authorized": False,
                "message": "Safety authority not available",
                "validation_time_ms": 0
            }
        
        try:
            # Map command to authority level and details
            authority_level = SafetyAuthorityLevel.COMMUNICATION  # Default level
            command_type = command
            
            if command in ["emergency_stop", "system_shutdown"]:
                authority_level = SafetyAuthorityLevel.EMERGENCY_STOP
                command_type = "emergency_stop"
            elif command in ["source_switch", "priority_change"]:
                authority_level = SafetyAuthorityLevel.COMMUNICATION
                command_type = "source_selection"
            elif command in ["coordination_override", "fallback_trigger"]:
                authority_level = SafetyAuthorityLevel.FLIGHT_MODE
                command_type = "coordination_override"
            
            # Validate with timing requirement
            authorized, message = self._safety_authority.validate_coordination_command_real_time(
                command_type=command_type,
                authority_level=authority_level,
                details=params,
                response_time_limit_ms=50  # Strict 50ms requirement for coordination
            )
            
            # Log the validation for audit trail
            if self._safety_authority:
                self._safety_authority.log_coordination_decision(
                    component="DualSDRCoordinator",
                    decision_type=f"command_validation_{command}",
                    decision_details=params,
                    authority_level=authority_level,
                    outcome="authorized" if authorized else "denied"
                )
            
            return {
                "authorized": authorized,
                "message": message,
                "command": command,
                "authority_level": authority_level.value,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Command validation failed for {command}: {e}")
            return {
                "authorized": False,
                "message": f"Validation error: {str(e)}",
                "command": command,
                "timestamp": datetime.now().isoformat()
            }
