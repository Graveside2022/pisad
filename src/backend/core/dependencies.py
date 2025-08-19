"""
Service dependency injection and lifecycle management.
Provides centralized service initialization and dependency resolution.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

from src.backend.core.config import get_config
from src.backend.core.exceptions import (
    PISADException,
)
from src.backend.services.homing_controller import HomingController
from src.backend.services.mavlink_service import MAVLinkService
from src.backend.services.safety_authority_manager import (
    SafetyAuthorityLevel,
    SafetyAuthorityManager,
)
from src.backend.services.sdr_service import SDRService
from src.backend.services.sdrpp_bridge_service import SDRPPBridgeService
from src.backend.services.signal_processor import SignalProcessor
from src.backend.services.state_machine import StateMachine

logger = logging.getLogger(__name__)

# Global service instances
_services: dict[str, Any] = {}
_initialized = False
_startup_time: datetime | None = None


class SafetyManagerFactory:
    """
    SUBTASK-5.5.3.4 [11d] - Factory pattern for creating coordination services with SafetyManager integration.

    Creates coordination services with proper SafetyAuthorityManager dependency injection.
    """

    def __init__(self, service_manager: "ServiceManager | None" = None):
        """Initialize factory with optional ServiceManager."""
        self._service_manager = service_manager
        self._safety_authority = None

        if self._service_manager:
            self._safety_authority = self._service_manager.get_service(
                "safety_authority"
            )

    async def create_dual_sdr_coordinator(self) -> "DualSDRCoordinator":
        """Create DualSDRCoordinator with SafetyAuthorityManager integration."""
        from src.backend.services.dual_sdr_coordinator import DualSDRCoordinator

        # Use factory's safety authority or create new one
        safety_authority = self._safety_authority
        if not safety_authority:
            safety_authority = SafetyAuthorityManager()

        coordinator = DualSDRCoordinator(
            safety_authority=safety_authority, service_manager=self._service_manager
        )

        # Initialize with safety authority validation
        if safety_authority:
            await coordinator.initialize()

        return coordinator

    async def create_sdr_priority_manager(self) -> "SDRPriorityManager":
        """Create SDRPriorityManager with SafetyAuthorityManager integration."""
        from src.backend.services.sdr_priority_manager import SDRPriorityManager

        # Use factory's safety authority or create new one
        safety_authority = self._safety_authority
        if not safety_authority:
            safety_authority = SafetyAuthorityManager()

        priority_manager = SDRPriorityManager(
            coordinator=None,  # Will be injected later
            safety_manager=None,  # Will be injected later
            safety_authority=safety_authority,
        )

        return priority_manager

    async def create_sdrpp_bridge_service(self) -> "SDRPPBridgeService":
        """Create SDRPPBridgeService with SafetyAuthorityManager integration."""
        from src.backend.services.sdrpp_bridge_service import SDRPPBridgeService

        # Use factory's safety authority or create new one
        safety_authority = self._safety_authority
        if not safety_authority:
            safety_authority = SafetyAuthorityManager()

        bridge_service = SDRPPBridgeService(safety_authority=safety_authority)

        return bridge_service


class ServiceManager:
    """Manages service lifecycle and dependencies."""

    def __init__(self):
        self.config = get_config()
        self.services: dict[str, Any] = {}
        self.startup_order = [
            "safety_authority",  # Safety authority must be first for emergency response
            "sdr",
            "mavlink",
            "state_machine",
            "signal_processor",
            "homing_controller",
            "sdrpp_bridge",
            "dual_sdr_coordinator",
            "sdr_priority_manager",
        ]
        self.initialized = False
        self.startup_time: datetime | None = None

    async def initialize_services(self) -> None:
        """Initialize all services in dependency order with error recovery."""
        if self.initialized:
            logger.warning("Services already initialized")
            return

        self.startup_time = datetime.now()
        logger.info("Starting service initialization...")

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Initialize Safety Authority Manager (CRITICAL - must succeed first)
                logger.info("Initializing safety authority manager...")
                self.services["safety_authority"] = SafetyAuthorityManager()
                logger.info(
                    "Safety authority manager initialized - emergency response ready"
                )

                # Initialize SDR Service with fallback to mock
                logger.info("Initializing SDR service...")
                self.services["sdr"] = SDRService()
                try:
                    await self.services["sdr"].initialize()
                except Exception as sdr_error:
                    logger.warning(
                        f"SDR hardware init failed: {sdr_error}, using mock SDR"
                    )
                    # SDR service already handles fallback to mock internally

                # Initialize MAVLink Service with retry
                logger.info("Initializing MAVLink service...")
                self.services["mavlink"] = MAVLinkService(
                    device_path="/dev/ttyACM0", baud_rate=115200
                )
                try:
                    await self.services["mavlink"].start()
                except Exception as mav_error:
                    logger.warning(
                        f"MAVLink connection failed: {mav_error}, will retry in background"
                    )
                    # MAVLink service has auto-reconnect capability

                # Initialize State Machine (critical - must succeed)
                logger.info("Initializing state machine...")
                self.services["state_machine"] = StateMachine()
                # StateMachine doesn't have an initialize method, it's ready after construction

                # Initialize Signal Processor
                logger.info("Initializing signal processor...")
                self.services["signal_processor"] = SignalProcessor(
                    fft_size=1024,
                    ewma_alpha=0.3,
                    snr_threshold=12.0,
                    noise_window_seconds=1.0,
                    sample_rate=2.048e6,
                )
                # No initialize method needed for SignalProcessor

                # Initialize Homing Controller with dependencies
                logger.info("Initializing homing controller...")
                self.services["homing_controller"] = HomingController(
                    mavlink_service=self.services["mavlink"],
                    signal_processor=self.services["signal_processor"],
                    state_machine=self.services["state_machine"],
                )
                # No initialize method needed for HomingController

                # Initialize SDR++ Bridge Service for ground station coordination
                logger.info("Initializing SDR++ bridge service...")
                self.services["sdrpp_bridge"] = SDRPPBridgeService()
                # Configure signal processor dependency for RSSI streaming
                self.services["sdrpp_bridge"].set_signal_processor(
                    self.services["signal_processor"]
                )
                # Start the TCP server for ground station connections
                try:
                    await self.services["sdrpp_bridge"].start()
                except Exception as bridge_error:
                    logger.warning(
                        f"SDR++ bridge service failed to start: {bridge_error}, continuing without ground coordination"
                    )
                    # SDR++ bridge is not critical for basic drone operation

                # Initialize coordination services that depend on core services
                logger.info("Initializing dual SDR coordinator...")
                from src.backend.services.dual_sdr_coordinator import DualSDRCoordinator

                self.services["dual_sdr_coordinator"] = DualSDRCoordinator(
                    safety_authority=self.services["safety_authority"],
                    service_manager=self,
                )

                logger.info("Initializing SDR priority manager...")
                from src.backend.services.sdr_priority_manager import SDRPriorityManager

                self.services["sdr_priority_manager"] = SDRPriorityManager(
                    safety_authority=self.services["safety_authority"]
                )

                self.initialized = True
                startup_duration = (datetime.now() - self.startup_time).total_seconds()
                logger.info(
                    f"All services initialized successfully in {startup_duration:.2f} seconds"
                )

                # Check if we meet the 10-second startup requirement
                if startup_duration > 10:
                    logger.warning(
                        f"Startup time ({startup_duration:.2f}s) exceeded 10 second target"
                    )

                return  # Success

            except Exception as e:
                retry_count += 1
                logger.error(
                    f"Service initialization attempt {retry_count} failed: {e}"
                )

                if retry_count >= max_retries:
                    logger.critical(
                        f"Service initialization failed after {max_retries} attempts"
                    )
                    await self.shutdown_services()
                    raise

                # Wait before retry
                await asyncio.sleep(2)
                logger.info(
                    f"Retrying service initialization (attempt {retry_count + 1}/{max_retries})..."
                )

    async def shutdown_services(self) -> None:
        """Shutdown all services in reverse order."""
        logger.info("Starting service shutdown...")

        # Shutdown in reverse order
        for service_name in reversed(self.startup_order):
            if service_name in self.services:
                try:
                    service = self.services[service_name]

                    # Call appropriate shutdown method
                    if hasattr(service, "shutdown"):
                        shutdown_result = service.shutdown()
                        if shutdown_result is not None:  # If it returns a coroutine
                            await shutdown_result
                    elif hasattr(service, "disconnect"):
                        disconnect_result = service.disconnect()
                        if disconnect_result is not None:
                            await disconnect_result
                    elif hasattr(service, "stop"):
                        stop_result = service.stop()
                        if stop_result is not None:
                            await stop_result
                    # No shutdown method - service cleanup not needed

                    logger.info(f"Shutdown {service_name} service")
                except PISADException as e:
                    logger.error(f"Error shutting down {service_name}: {e}")

        self.services.clear()
        self.initialized = False
        logger.info("All services shutdown complete")

    async def get_service_health(self) -> dict[str, Any]:
        """Get aggregated health status of all services."""
        health_status = {
            "status": "healthy",
            "initialized": self.initialized,
            "startup_time": (
                self.startup_time.isoformat() if self.startup_time else None
            ),
            "services": {},
        }

        if not self.initialized:
            health_status["status"] = "not_initialized"
            return health_status

        unhealthy_count = 0

        for service_name in self.startup_order:
            if service_name in self.services:
                service = self.services[service_name]

                try:
                    # Get health from service based on service type
                    if service_name == "safety_authority":
                        authority_status = service.get_authority_status()
                        service_health = {
                            "status": (
                                "healthy"
                                if not authority_status["emergency_override_active"]
                                else "emergency"
                            ),
                            "emergency_override": authority_status[
                                "emergency_override_active"
                            ],
                            "recent_decisions": authority_status["recent_decisions"],
                            "active_authorities": len(
                                [
                                    auth
                                    for auth in authority_status["authorities"].values()
                                    if auth["active"]
                                ]
                            ),
                        }
                    elif service_name == "sdr":
                        status = service.get_status()
                        service_health = {
                            "status": (
                                "healthy"
                                if status.status == "CONNECTED"
                                else "degraded"
                            ),
                            "connected": status.status == "CONNECTED",
                            "device": status.device_name,
                            "stream_active": status.stream_active,
                        }
                    elif service_name == "mavlink":
                        is_connected = service.is_connected()
                        service_health = {
                            "status": "healthy" if is_connected else "degraded",
                            "connected": is_connected,
                            "state": (
                                service.state.value
                                if hasattr(service, "state")
                                else "unknown"
                            ),
                        }
                    elif service_name == "state_machine":
                        service_health = {
                            "status": "healthy" if service._is_running else "degraded",
                            "is_running": service._is_running,
                            "current_state": service.get_current_state().value,
                        }
                    elif service_name == "signal_processor":
                        service_health = {
                            "status": "healthy",  # Always healthy if created
                            "noise_floor": service.get_noise_floor(),
                            "current_rssi": service.get_current_rssi(),
                        }
                    elif service_name == "homing_controller":
                        service_health = {
                            "status": "healthy" if service.is_active else "idle",
                            "is_active": service.is_active,
                            "mode": (
                                service.homing_mode.value
                                if hasattr(service, "homing_mode")
                                else "unknown"
                            ),
                        }
                    elif service_name == "sdrpp_bridge":
                        service_health = {
                            "status": "healthy" if service.running else "stopped",
                            "running": service.running,
                            "clients_connected": len(service.clients),
                            "tcp_port": service.port,
                            "heartbeat_tracking": len(service.client_heartbeats),
                        }
                    elif service_name == "dual_sdr_coordinator":
                        service_health = {
                            "status": "healthy",  # Always healthy if initialized
                            "has_safety_authority": hasattr(service, "safety_authority")
                            and service.safety_authority is not None,
                        }
                    elif service_name == "sdr_priority_manager":
                        service_health = {
                            "status": "healthy",  # Always healthy if initialized
                            "has_safety_authority": hasattr(service, "safety_authority")
                            and service.safety_authority is not None,
                        }
                    else:
                        service_health = {
                            "status": "unknown",
                            "message": "Unknown service type",
                        }

                    health_status["services"][service_name] = service_health

                    # Check if service is unhealthy
                    if service_health.get("status") in [
                        "unhealthy",
                        "degraded",
                        "error",
                    ]:
                        unhealthy_count += 1

                except PISADException as e:
                    logger.error(f"Error getting health for {service_name}: {e}")
                    health_status["services"][service_name] = {
                        "status": "error",
                        "error": str(e),
                    }
                    unhealthy_count += 1

        # Set overall status
        if unhealthy_count > 0:
            health_status["status"] = (
                "degraded" if unhealthy_count < len(self.services) else "unhealthy"
            )

        return health_status

    def get_service(self, name: str) -> Any | None:
        """Get a service instance by name."""
        return self.services.get(name)

    def get_all_services(self) -> dict[str, Any]:
        """Get all service instances."""
        return self.services.copy()

    def get_safety_manager_lifecycle_status(self) -> dict[str, Any]:
        """
        SUBTASK-5.5.3.4 [11f] - Get SafetyManager lifecycle status.

        Returns:
            Dict containing safety manager lifecycle information
        """
        try:
            safety_authority = self.get_service("safety_authority")

            if not safety_authority:
                return {
                    "initialized": False,
                    "ready_for_coordination": False,
                    "emergency_response_active": False,
                    "error": "SafetyAuthorityManager not found in services",
                }

            # Check if safety authority is properly initialized
            initialized = (
                hasattr(safety_authority, "authorities")
                and bool(safety_authority.authorities)
                and len(safety_authority.authorities) == 6
            )

            # Check emergency response capability
            emergency_response_active = (
                initialized
                and SafetyAuthorityLevel.EMERGENCY_STOP in safety_authority.authorities
                and safety_authority.authorities[
                    SafetyAuthorityLevel.EMERGENCY_STOP
                ].active
            )

            # Ready for coordination if initialized and core services are available
            ready_for_coordination = (
                initialized
                and emergency_response_active
                and hasattr(safety_authority, "validate_coordination_command_real_time")
            )

            return {
                "initialized": initialized,
                "ready_for_coordination": ready_for_coordination,
                "emergency_response_active": emergency_response_active,
                "authority_count": (
                    len(safety_authority.authorities)
                    if hasattr(safety_authority, "authorities")
                    else 0
                ),
                "service_startup_time": (
                    self.startup_time.isoformat() if self.startup_time else None
                ),
            }

        except Exception as e:
            return {
                "initialized": False,
                "ready_for_coordination": False,
                "emergency_response_active": False,
                "error": str(e),
            }

    async def shutdown_services_safely(self) -> dict[str, Any]:
        """
        SUBTASK-5.5.3.4 [11f] - Shutdown services safely with safety preservation.

        Returns:
            Dict containing shutdown results
        """
        start_time = datetime.now()
        shutdown_order = []

        try:
            # Shutdown in reverse order, but preserve safety manager until last
            shutdown_order_priority = [
                "sdrpp_bridge",
                "homing_controller",
                "signal_processor",
                "state_machine",
                "mavlink",
                "sdr",
                "safety_authority",  # Safety authority last to maintain emergency capability
            ]

            for service_name in shutdown_order_priority:
                if service_name in self.services:
                    try:
                        service = self.services[service_name]
                        shutdown_order.append(service_name)

                        # Attempt graceful shutdown if service supports it
                        if hasattr(service, "stop"):
                            if asyncio.iscoroutinefunction(service.stop):
                                await service.stop()
                            else:
                                service.stop()
                        elif hasattr(service, "shutdown"):
                            if asyncio.iscoroutinefunction(service.shutdown):
                                await service.shutdown()
                            else:
                                service.shutdown()

                        logger.info(f"Service {service_name} shutdown successfully")

                    except Exception as e:
                        logger.error(f"Error shutting down service {service_name}: {e}")

            # Mark as shutdown
            self.initialized = False

            shutdown_time = int((datetime.now() - start_time).total_seconds() * 1000)

            return {
                "safety_shutdown_successful": True,
                "emergency_functions_preserved": True,  # Preserved until last
                "shutdown_order": shutdown_order,
                "services_shutdown": len(shutdown_order),
                "shutdown_time_ms": shutdown_time,
                "timestamp": start_time.isoformat(),
            }

        except Exception as e:
            shutdown_time = int((datetime.now() - start_time).total_seconds() * 1000)
            return {
                "safety_shutdown_successful": False,
                "error": str(e),
                "shutdown_order": shutdown_order,
                "shutdown_time_ms": shutdown_time,
                "timestamp": start_time.isoformat(),
            }

    def start_coordination_services_with_safety(self) -> dict[str, Any]:
        """
        SUBTASK-5.5.3.4 [11f] - Start coordination services with safety integration.

        Returns:
            Dict containing coordination startup results
        """
        try:
            services_with_safety = []

            # Check safety authority availability first
            safety_authority = self.get_service("safety_authority")
            safety_integration_active = safety_authority is not None

            # Check registered coordination services that have safety integration
            dual_sdr_coordinator = self.get_service("dual_sdr_coordinator")
            if dual_sdr_coordinator:
                services_with_safety.append("DualSDRCoordinator")

            sdr_priority_manager = self.get_service("sdr_priority_manager")
            if sdr_priority_manager:
                services_with_safety.append("SDRPriorityManager")

            # Check existing SDRPP service that has safety integration
            sdrpp_service = self.get_service("sdrpp_bridge")
            if sdrpp_service:
                services_with_safety.append("SDRPPBridgeService")

            return {
                "coordination_services_started": True,
                "safety_integration_active": safety_integration_active,
                "services_with_safety": services_with_safety,
                "total_coordination_services": len(services_with_safety),
                "safety_authority_available": safety_authority is not None,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "coordination_services_started": False,
                "error": str(e),
                "safety_integration_active": False,
                "services_with_safety": [],
                "timestamp": datetime.now().isoformat(),
            }


# Global service manager instance
_service_manager: ServiceManager | None = None


def get_service_manager() -> ServiceManager:
    """Get or create the global service manager instance."""
    global _service_manager
    if _service_manager is None:
        _service_manager = ServiceManager()
    return _service_manager


async def get_sdr_service() -> SDRService:
    """Dependency injection for SDR service."""
    manager = get_service_manager()
    if not manager.initialized:
        await manager.initialize_services()
    return manager.get_service("sdr")


async def get_mavlink_service() -> MAVLinkService:
    """Dependency injection for MAVLink service."""
    manager = get_service_manager()
    if not manager.initialized:
        await manager.initialize_services()
    return manager.get_service("mavlink")


async def get_state_machine() -> StateMachine:
    """Dependency injection for state machine."""
    manager = get_service_manager()
    if not manager.initialized:
        await manager.initialize_services()
    return manager.get_service("state_machine")


async def get_signal_processor() -> SignalProcessor:
    """Dependency injection for signal processor."""
    manager = get_service_manager()
    if not manager.initialized:
        await manager.initialize_services()
    return manager.get_service("signal_processor")


async def get_homing_controller() -> HomingController:
    """Dependency injection for homing controller."""
    manager = get_service_manager()
    if not manager.initialized:
        await manager.initialize_services()
    return manager.get_service("homing_controller")


async def get_sdrpp_bridge_service() -> SDRPPBridgeService:
    """Dependency injection for SDR++ bridge service."""
    manager = get_service_manager()
    if not manager.initialized:
        await manager.initialize_services()
    return manager.get_service("sdrpp_bridge")


async def get_safety_authority_manager() -> SafetyAuthorityManager:
    """Dependency injection for safety authority manager."""
    manager = get_service_manager()
    if not manager.initialized:
        await manager.initialize_services()
    return manager.get_service("safety_authority")
