"""
Health check endpoints for all system services.
"""

import logging
import time
from datetime import UTC, datetime
from typing import Any

import psutil
from fastapi import APIRouter, Depends, HTTPException

from src.backend.services.mavlink_service import MAVLinkService
from src.backend.services.sdr_service import SDRService
from src.backend.services.state_machine import StateMachine
from src.backend.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["health"])

# Service instances (will be injected via dependencies)
sdr_service: SDRService | None = None
mavlink_service: MAVLinkService | None = None
state_machine: StateMachine | None = None


def get_sdr_service() -> SDRService | None:
    """Get SDR service instance."""
    return sdr_service


def get_mavlink_service() -> MAVLinkService | None:
    """Get MAVLink service instance."""
    return mavlink_service


def get_state_machine() -> StateMachine | None:
    """Get state machine instance."""
    return state_machine


@router.get("")
async def health_check() -> dict[str, Any]:
    """
    Overall system health check.
    
    Returns:
        Aggregated health status of all services.
    """
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        
        # Get temperature if available (Raspberry Pi)
        temperature = None
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                temperature = float(f.read()) / 1000.0
        except (FileNotFoundError, PermissionError):
            pass
        
        # Check each service
        services_health = {}
        
        # SDR Service
        if sdr_service:
            sdr_status = sdr_service.get_status()
            services_health["sdr"] = {
                "status": sdr_status.status,
                "connected": sdr_status.status == "CONNECTED",
                "device": sdr_status.device_name,
                "driver": sdr_status.driver,
                "stream_active": sdr_status.stream_active,
                "samples_per_second": sdr_status.samples_per_second,
                "buffer_overflows": sdr_status.buffer_overflows,
                "temperature": sdr_status.temperature,
            }
        else:
            services_health["sdr"] = {"status": "NOT_INITIALIZED"}
        
        # MAVLink Service
        if mavlink_service:
            services_health["mavlink"] = {
                "connected": mavlink_service.is_connected(),
                "state": mavlink_service.state.value,
                "telemetry": mavlink_service.get_telemetry(),
            }
        else:
            services_health["mavlink"] = {"status": "NOT_INITIALIZED"}
        
        # State Machine
        if state_machine:
            services_health["state_machine"] = {
                "current_state": state_machine.get_current_state().value,
                "is_running": state_machine._is_running,
                "statistics": state_machine.get_statistics(),
            }
        else:
            services_health["state_machine"] = {"status": "NOT_INITIALIZED"}
        
        # Determine overall health
        overall_status = "healthy"
        if not all(
            s.get("connected", s.get("status") == "CONNECTED")
            for s in services_health.values()
            if s.get("status") != "NOT_INITIALIZED"
        ):
            overall_status = "degraded"
        
        if cpu_percent > 90 or memory.percent > 90:
            overall_status = "degraded"
        
        if temperature and temperature > 80:
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now(UTC).isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "temperature": temperature,
                "uptime": int(time.time() - psutil.boot_time()),
            },
            "services": services_health,
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sdr")
async def sdr_health_check() -> dict[str, Any]:
    """
    SDR service health check.
    
    Returns:
        Detailed SDR service status.
    """
    if not sdr_service:
        raise HTTPException(status_code=503, detail="SDR service not initialized")
    
    try:
        status = sdr_service.get_status()
        
        # Determine health level
        health = "healthy"
        if status.status == "DISCONNECTED":
            health = "unhealthy"
        elif status.status == "ERROR":
            health = "critical"
        elif status.buffer_overflows > 0:
            health = "degraded"
        
        return {
            "health": health,
            "status": status.status,
            "device_name": status.device_name,
            "driver": status.driver,
            "stream_active": status.stream_active,
            "samples_per_second": status.samples_per_second,
            "buffer_overflows": status.buffer_overflows,
            "temperature": status.temperature,
            "last_error": status.last_error,
            "config": {
                "frequency": sdr_service.config.frequency,
                "sample_rate": sdr_service.config.sampleRate,
                "bandwidth": sdr_service.config.bandwidth,
                "gain": sdr_service.config.gain,
                "buffer_size": sdr_service.config.buffer_size,
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        logger.error(f"SDR health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mavlink")
async def mavlink_health_check() -> dict[str, Any]:
    """
    MAVLink service health check.
    
    Returns:
        Detailed MAVLink service status.
    """
    if not mavlink_service:
        raise HTTPException(status_code=503, detail="MAVLink service not initialized")
    
    try:
        is_connected = mavlink_service.is_connected()
        telemetry = mavlink_service.get_telemetry()
        
        # Determine health level
        health = "healthy" if is_connected else "unhealthy"
        
        # Check for stale heartbeat
        if is_connected:
            time_since_heartbeat = time.time() - mavlink_service.last_heartbeat_received
            if time_since_heartbeat > mavlink_service.heartbeat_timeout:
                health = "degraded"
        
        return {
            "health": health,
            "connected": is_connected,
            "connection_state": mavlink_service.state.value,
            "device_path": mavlink_service.device_path,
            "baud_rate": mavlink_service.baud_rate,
            "last_heartbeat_received": mavlink_service.last_heartbeat_received,
            "last_heartbeat_sent": mavlink_service.last_heartbeat_sent,
            "time_since_heartbeat": (
                time.time() - mavlink_service.last_heartbeat_received
                if mavlink_service.last_heartbeat_received > 0
                else None
            ),
            "telemetry": telemetry,
            "velocity_commands_enabled": mavlink_service._velocity_commands_enabled,
            "telemetry_config": mavlink_service.get_telemetry_config(),
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        logger.error(f"MAVLink health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/state")
async def state_machine_health_check() -> dict[str, Any]:
    """
    State machine health check.
    
    Returns:
        Detailed state machine status.
    """
    if not state_machine:
        raise HTTPException(status_code=503, detail="State machine not initialized")
    
    try:
        current_state = state_machine.get_current_state()
        statistics = state_machine.get_statistics()
        telemetry_metrics = state_machine.get_telemetry_metrics()
        
        # Determine health level
        health = "healthy"
        if not state_machine._is_running:
            health = "unhealthy"
        elif statistics.get("state_duration_seconds", 0) > 300:  # Stuck in state > 5 min
            health = "degraded"
        
        return {
            "health": health,
            "is_running": state_machine._is_running,
            "current_state": current_state.value,
            "previous_state": state_machine._previous_state.value,
            "allowed_transitions": [s.value for s in state_machine.get_allowed_transitions()],
            "statistics": statistics,
            "telemetry_metrics": telemetry_metrics,
            "search_pattern_status": state_machine.get_search_pattern_status(),
            "state_history": state_machine.get_state_history(limit=5),
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        logger.error(f"State machine health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/signal")
async def signal_processor_health_check() -> dict[str, Any]:
    """
    Signal processor health check.
    
    Returns:
        Detailed signal processor status.
    """
    # Import signal processor if available
    try:
        from src.backend.services.signal_processor import signal_processor_instance
        
        if not signal_processor_instance:
            raise HTTPException(status_code=503, detail="Signal processor not initialized")
        
        # Get signal processor metrics
        metrics = signal_processor_instance.get_metrics()
        
        # Determine health level
        health = "healthy"
        if not signal_processor_instance.is_processing:
            health = "unhealthy"
        elif metrics.get("processing_errors", 0) > 10:
            health = "degraded"
        
        return {
            "health": health,
            "is_processing": signal_processor_instance.is_processing,
            "current_rssi": signal_processor_instance.current_rssi,
            "noise_floor": signal_processor_instance.noise_floor,
            "signal_detected": signal_processor_instance.signal_detected,
            "detection_confidence": signal_processor_instance.detection_confidence,
            "metrics": metrics,
            "config": {
                "detection_threshold": signal_processor_instance.detection_threshold,
                "noise_floor_percentile": signal_processor_instance.noise_floor_percentile,
                "debounce_samples": signal_processor_instance.debounce_samples,
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="Signal processor module not available")
    except Exception as e:
        logger.error(f"Signal processor health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def set_service_instances(
    sdr: SDRService | None = None,
    mavlink: MAVLinkService | None = None,
    state: StateMachine | None = None,
) -> None:
    """
    Set service instances for health checks.
    
    Args:
        sdr: SDR service instance
        mavlink: MAVLink service instance
        state: State machine instance
    """
    global sdr_service, mavlink_service, state_machine
    sdr_service = sdr
    mavlink_service = mavlink
    state_machine = state
    logger.info("Health check service instances configured")