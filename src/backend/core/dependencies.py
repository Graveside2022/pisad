"""
Service dependency injection and lifecycle management.
Provides centralized service initialization and dependency resolution.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from src.backend.core.config import get_config
from src.backend.services.sdr_service import SDRService
from src.backend.services.mavlink_service import MAVLinkService
from src.backend.services.state_machine import StateMachine
from src.backend.services.signal_processor import SignalProcessor
from src.backend.services.homing_controller import HomingController

logger = logging.getLogger(__name__)

# Global service instances
_services: Dict[str, Any] = {}
_initialized = False
_startup_time: Optional[datetime] = None


class ServiceManager:
    """Manages service lifecycle and dependencies."""
    
    def __init__(self):
        self.config = get_config()
        self.services: Dict[str, Any] = {}
        self.startup_order = [
            "sdr",
            "mavlink", 
            "state_machine",
            "signal_processor",
            "homing_controller"
        ]
        self.initialized = False
        self.startup_time: Optional[datetime] = None
        
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
                # Initialize SDR Service with fallback to mock
                logger.info("Initializing SDR service...")
                self.services["sdr"] = SDRService()
                try:
                    await self.services["sdr"].initialize()
                except Exception as sdr_error:
                    logger.warning(f"SDR hardware init failed: {sdr_error}, using mock SDR")
                    # SDR service already handles fallback to mock internally
                
                # Initialize MAVLink Service with retry
                logger.info("Initializing MAVLink service...")
                self.services["mavlink"] = MAVLinkService(
                    device_path="/dev/ttyACM0",
                    baud_rate=115200
                )
                try:
                    await self.services["mavlink"].start()
                except Exception as mav_error:
                    logger.warning(f"MAVLink connection failed: {mav_error}, will retry in background")
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
                    sample_rate=2.048e6
                )
                # No initialize method needed for SignalProcessor
                
                # Initialize Homing Controller with dependencies
                logger.info("Initializing homing controller...")
                self.services["homing_controller"] = HomingController(
                    mavlink_service=self.services["mavlink"],
                    signal_processor=self.services["signal_processor"],
                    state_machine=self.services["state_machine"]
                )
                # No initialize method needed for HomingController
                
                self.initialized = True
                startup_duration = (datetime.now() - self.startup_time).total_seconds()
                logger.info(f"All services initialized successfully in {startup_duration:.2f} seconds")
                
                # Check if we meet the 10-second startup requirement
                if startup_duration > 10:
                    logger.warning(f"Startup time ({startup_duration:.2f}s) exceeded 10 second target")
                
                return  # Success
                
            except Exception as e:
                retry_count += 1
                logger.error(f"Service initialization attempt {retry_count} failed: {e}")
                
                if retry_count >= max_retries:
                    logger.critical(f"Service initialization failed after {max_retries} attempts")
                    await self.shutdown_services()
                    raise
                
                # Wait before retry
                await asyncio.sleep(2)
                logger.info(f"Retrying service initialization (attempt {retry_count + 1}/{max_retries})...")
    
    async def shutdown_services(self) -> None:
        """Shutdown all services in reverse order."""
        logger.info("Starting service shutdown...")
        
        # Shutdown in reverse order
        for service_name in reversed(self.startup_order):
            if service_name in self.services:
                try:
                    service = self.services[service_name]
                    
                    # Call appropriate shutdown method
                    if hasattr(service, 'shutdown'):
                        await service.shutdown()
                    elif hasattr(service, 'disconnect'):
                        await service.disconnect()
                    elif hasattr(service, 'stop'):
                        await service.stop()
                        
                    logger.info(f"Shutdown {service_name} service")
                except Exception as e:
                    logger.error(f"Error shutting down {service_name}: {e}")
        
        self.services.clear()
        self.initialized = False
        logger.info("All services shutdown complete")
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Get aggregated health status of all services."""
        health_status = {
            "status": "healthy",
            "initialized": self.initialized,
            "startup_time": self.startup_time.isoformat() if self.startup_time else None,
            "services": {}
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
                    if service_name == "sdr":
                        status = service.get_status()
                        service_health = {
                            "status": "healthy" if status.status == "CONNECTED" else "degraded",
                            "connected": status.status == "CONNECTED",
                            "device": status.device_name,
                            "stream_active": status.stream_active
                        }
                    elif service_name == "mavlink":
                        is_connected = service.is_connected()
                        service_health = {
                            "status": "healthy" if is_connected else "degraded",
                            "connected": is_connected,
                            "state": service.state.value if hasattr(service, 'state') else "unknown"
                        }
                    elif service_name == "state_machine":
                        service_health = {
                            "status": "healthy" if service._is_running else "degraded",
                            "is_running": service._is_running,
                            "current_state": service.get_current_state().value
                        }
                    elif service_name == "signal_processor":
                        service_health = {
                            "status": "healthy",  # Always healthy if created
                            "noise_floor": service.get_noise_floor(),
                            "current_rssi": service.get_current_rssi()
                        }
                    elif service_name == "homing_controller":
                        service_health = {
                            "status": "healthy" if service.is_active else "idle",
                            "is_active": service.is_active,
                            "mode": service.homing_mode.value if hasattr(service, 'homing_mode') else "unknown"
                        }
                    else:
                        service_health = {
                            "status": "unknown",
                            "message": "Unknown service type"
                        }
                    
                    health_status["services"][service_name] = service_health
                    
                    # Check if service is unhealthy
                    if service_health.get("status") in ["unhealthy", "degraded", "error"]:
                        unhealthy_count += 1
                        
                except Exception as e:
                    logger.error(f"Error getting health for {service_name}: {e}")
                    health_status["services"][service_name] = {
                        "status": "error",
                        "error": str(e)
                    }
                    unhealthy_count += 1
        
        # Set overall status
        if unhealthy_count > 0:
            health_status["status"] = "degraded" if unhealthy_count < len(self.services) else "unhealthy"
        
        return health_status
    
    def get_service(self, name: str) -> Optional[Any]:
        """Get a service instance by name."""
        return self.services.get(name)
    
    def get_all_services(self) -> Dict[str, Any]:
        """Get all service instances."""
        return self.services.copy()


# Global service manager instance
_service_manager: Optional[ServiceManager] = None


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