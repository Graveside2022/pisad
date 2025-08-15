"""
Health check endpoints for all system services.
"""

import time
from datetime import UTC, datetime
from typing import Any

import psutil
from fastapi import APIRouter, Depends, HTTPException

from src.backend.core.dependencies import (
    get_mavlink_service,
    get_sdr_service,
    get_service_manager,
    get_signal_processor,
    get_state_machine,
)
from src.backend.core.exceptions import (
    MAVLinkError,
    PISADException,
    SDRError,
    SignalProcessingError,
    StateTransitionError,
)
from src.backend.services.mavlink_service import MAVLinkService
from src.backend.services.sdr_service import SDRService
from src.backend.services.signal_processor import SignalProcessor
from src.backend.services.state_machine import StateMachine
from src.backend.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health_check() -> dict[str, Any]:
    """
    Overall system health check.

    Returns:
        Aggregated health status of all services.
    """
    try:
        # Get service manager health
        service_manager = get_service_manager()
        manager_health = await service_manager.get_service_health()

        # Add system metrics
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

        # Check if system resources are stressed
        if cpu_percent > 90 or memory.percent > 90:
            manager_health["status"] = "degraded"

        if temperature and temperature > 80:
            manager_health["status"] = "degraded"

        return {
            "status": manager_health["status"],
            "timestamp": datetime.now(UTC).isoformat(),
            "initialized": manager_health["initialized"],
            "startup_time": manager_health["startup_time"],
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "temperature": temperature,
                "uptime": int(time.time() - psutil.boot_time()),
            },
            "services": manager_health["services"],
        }
    except PISADException as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sdr")
async def sdr_health_check(sdr_service: SDRService = Depends(get_sdr_service)) -> dict[str, Any]:
    """
    SDR service health check.

    Returns:
        Detailed SDR service status.
    """
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
    except SDRError as e:
        logger.error(f"SDR health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mavlink")
async def mavlink_health_check(
    mavlink_service: MAVLinkService = Depends(get_mavlink_service),
) -> dict[str, Any]:
    """
    MAVLink service health check.

    Returns:
        Detailed MAVLink service status.
    """
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
    except MAVLinkError as e:
        logger.error(f"MAVLink health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/state")
async def state_machine_health_check(
    state_machine: StateMachine = Depends(get_state_machine),
) -> dict[str, Any]:
    """
    State machine health check.

    Returns:
        Detailed state machine status.
    """
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
    except StateTransitionError as e:
        logger.error(f"State machine health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/signal")
async def signal_processor_health_check(
    signal_processor: SignalProcessor = Depends(get_signal_processor),
) -> dict[str, Any]:
    """
    Signal processor health check.

    Returns:
        Detailed signal processor status.
    """
    try:
        # Get signal processor metrics
        metrics = signal_processor.get_metrics()

        # Determine health level
        health = "healthy"
        if not signal_processor.is_processing:
            health = "unhealthy"
        elif metrics.get("processing_errors", 0) > 10:
            health = "degraded"

        return {
            "health": health,
            "is_processing": signal_processor.is_processing,
            "current_rssi": signal_processor.current_rssi,
            "noise_floor": signal_processor.noise_floor,
            "signal_detected": signal_processor.signal_detected,
            "detection_confidence": signal_processor.detection_confidence,
            "metrics": metrics,
            "config": {
                "detection_threshold": signal_processor.detection_threshold,
                "noise_floor_percentile": signal_processor.noise_floor_percentile,
                "debounce_samples": signal_processor.debounce_samples,
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except SignalProcessingError as e:
        logger.error(f"Signal processor health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
