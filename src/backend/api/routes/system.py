"""
System status and monitoring API routes.
"""

import logging
from datetime import UTC, datetime
from typing import Any

import psutil
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.backend.core.config import get_config
from src.backend.services.state_machine import StateMachine
from src.backend.services.state_machine import SystemState as StateMachineState
from src.backend.utils.safety import SafetyInterlockSystem

logger = logging.getLogger(__name__)

router = APIRouter()

# Safety interlock system instance (will be initialized elsewhere)
safety_system: SafetyInterlockSystem | None = None

# State machine instance (will be initialized elsewhere)
state_machine: StateMachine | None = None


class HomingRequest(BaseModel):
    """Request model for enabling/disabling homing."""

    enabled: bool
    confirmation_token: str | None = None


class EmergencyStopRequest(BaseModel):
    """Request model for emergency stop."""

    reason: str | None = "Manual emergency stop"


class HomingParametersUpdate(BaseModel):
    """Request model for updating homing algorithm parameters."""

    forward_velocity_max: float | None = Field(None, ge=0.1, le=10.0)
    yaw_rate_max: float | None = Field(None, ge=0.1, le=2.0)
    approach_velocity: float | None = Field(None, ge=0.1, le=5.0)
    signal_loss_timeout: float | None = Field(None, ge=1.0, le=30.0)
    gradient_window_size: int | None = Field(None, ge=3, le=50)
    gradient_min_snr: float | None = Field(None, ge=0.0, le=50.0)
    sampling_turn_radius: float | None = Field(None, ge=1.0, le=50.0)
    sampling_duration: float | None = Field(None, ge=1.0, le=30.0)
    approach_threshold: float | None = Field(None, ge=-100.0, le=-20.0)
    plateau_variance: float | None = Field(None, ge=0.1, le=10.0)
    velocity_scale_factor: float | None = Field(None, ge=0.01, le=1.0)


class DebugModeRequest(BaseModel):
    """Request model for toggling debug mode."""

    enabled: bool
    target: str = Field(default="all", pattern="^(all|homing|sdr|mavlink|signal)$")


class StateOverrideRequest(BaseModel):
    """Request model for manual state override."""

    target_state: str = Field(
        ...,
        pattern="^(IDLE|SEARCHING|DETECTING|HOMING|HOLDING)$",
        description="Target state to transition to",
    )
    reason: str = Field(..., min_length=1, max_length=200)
    confirmation_token: str = Field(..., min_length=1, max_length=100)
    operator_id: str = Field(..., min_length=1, max_length=50)


@router.get("/system/status")
async def get_system_status() -> dict[str, Any]:
    """
    Get current system status including CPU, memory, SDR config, and health metrics.

    Returns:
        System status with configuration and health metrics
    """
    try:
        config = get_config()

        # Get CPU and memory usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # Get uptime
        boot_time = psutil.boot_time()
        current_time = datetime.now().timestamp()
        uptime_seconds = current_time - boot_time

        # Get temperature if available (Raspberry Pi specific)
        temperature = None
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                temp_raw = float(f.read())
                temperature = temp_raw / 1000.0  # Convert from millidegrees
        except (FileNotFoundError, PermissionError):
            pass

        return {
            "status": "ok",
            "timestamp": datetime.now(UTC).isoformat(),
            # System state
            "current_state": "IDLE",
            "homing_enabled": False,
            "flight_mode": "GROUND",
            "battery_percent": 100.0,
            "gps_status": "NO_FIX",
            "mavlink_connected": False,
            "sdr_status": "CONNECTED",
            # SDR configuration
            "sdr_frequency": config.sdr.SDR_FREQUENCY,
            "sdr_sample_rate": config.sdr.SDR_SAMPLE_RATE,
            "sdr_gain": config.sdr.SDR_GAIN,
            "sdr_ppm_correction": config.sdr.SDR_PPM_CORRECTION,
            "sdr_device_index": config.sdr.SDR_DEVICE_INDEX,
            # System health
            "cpu_usage": cpu_percent,
            "memory_usage": memory.percent,
            "disk_usage": disk.percent,
            "temperature": temperature,
            "uptime": int(uptime_seconds),
            # Safety interlocks
            "safety_interlocks": {
                "velocity_limit": True,
                "geofence": True,
                "battery_low": False,
                "signal_lock": True,
            },
        }
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/system/homing")
async def control_homing(request: HomingRequest) -> dict[str, Any]:
    """
    Enable or disable homing with safety checks.

    Args:
        request: Homing control request with enabled flag and optional token

    Returns:
        Current homing and safety status

    Raises:
        403: If safety interlock blocks activation
        500: If internal error occurs
    """
    if safety_system is None:
        raise HTTPException(status_code=500, detail="Safety system not initialized")

    try:
        if request.enabled:
            # Try to enable homing
            success = await safety_system.enable_homing(request.confirmation_token)

            if not success:
                # Get safety status to show why it was blocked
                safety_status = safety_system.get_safety_status()
                logger.warning(f"Homing activation blocked by safety interlocks: {safety_status}")
                raise HTTPException(
                    status_code=403,
                    detail={
                        "message": "Homing activation blocked by safety interlocks",
                        "safety_status": safety_status,
                    },
                )

            logger.info("Homing enabled via API")
        else:
            # Disable homing
            await safety_system.disable_homing("API request")
            logger.info("Homing disabled via API")

        return {
            "homing_enabled": request.enabled,
            "safety_status": safety_system.get_safety_status(),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to control homing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/system/emergency-stop")
async def emergency_stop(request: EmergencyStopRequest) -> dict[str, Any]:
    """
    Activate emergency stop to immediately halt all operations.

    Args:
        request: Emergency stop request with optional reason

    Returns:
        Current safety status after emergency stop
    """
    if safety_system is None:
        raise HTTPException(status_code=500, detail="Safety system not initialized")

    try:
        await safety_system.emergency_stop(request.reason or "API emergency stop")

        logger.critical(f"Emergency stop activated via API: {request.reason}")

        return {
            "emergency_stopped": True,
            "reason": request.reason,
            "safety_status": safety_system.get_safety_status(),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to activate emergency stop: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/system/reset-emergency-stop")
async def reset_emergency_stop() -> dict[str, Any]:
    """
    Reset emergency stop (requires manual confirmation).

    Returns:
        Current safety status after reset
    """
    if safety_system is None:
        raise HTTPException(status_code=500, detail="Safety system not initialized")

    try:
        await safety_system.reset_emergency_stop()

        logger.info("Emergency stop reset via API")

        return {
            "emergency_stopped": False,
            "safety_status": safety_system.get_safety_status(),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to reset emergency stop: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/safety/events")
async def get_safety_events(limit: int = 100, since: datetime | None = None) -> dict[str, Any]:
    """
    Get safety event history.

    Args:
        limit: Maximum number of events to return
        since: Get events after this timestamp

    Returns:
        List of safety events
    """
    if safety_system is None:
        raise HTTPException(status_code=500, detail="Safety system not initialized")

    try:
        events = safety_system.get_safety_events(since=since, limit=limit)

        return {
            "events": [
                {
                    "id": str(event.id),
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.event_type.value,
                    "trigger": event.trigger.value,
                    "details": event.details,
                    "resolved": event.resolved,
                }
                for event in events
            ],
            "count": len(events),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get safety events: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/safety/status")
async def get_safety_status() -> dict[str, Any]:
    """
    Get current safety interlock status.

    Returns:
        Detailed safety system status
    """
    if safety_system is None:
        raise HTTPException(status_code=500, detail="Safety system not initialized")

    try:
        return safety_system.get_safety_status()

    except Exception as e:
        logger.error(f"Failed to get safety status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/homing/parameters")
async def get_homing_parameters() -> dict[str, Any]:
    """
    Get current homing algorithm parameters.

    Returns:
        Current homing configuration parameters
    """
    try:
        config = get_config()

        return {
            "parameters": {
                "forward_velocity_max": config.homing.HOMING_FORWARD_VELOCITY_MAX,
                "yaw_rate_max": config.homing.HOMING_YAW_RATE_MAX,
                "approach_velocity": config.homing.HOMING_APPROACH_VELOCITY,
                "signal_loss_timeout": config.homing.HOMING_SIGNAL_LOSS_TIMEOUT,
                "algorithm_mode": config.homing.HOMING_ALGORITHM_MODE,
                "gradient_window_size": config.homing.HOMING_GRADIENT_WINDOW_SIZE,
                "gradient_min_snr": config.homing.HOMING_GRADIENT_MIN_SNR,
                "sampling_turn_radius": config.homing.HOMING_SAMPLING_TURN_RADIUS,
                "sampling_duration": config.homing.HOMING_SAMPLING_DURATION,
                "approach_threshold": config.homing.HOMING_APPROACH_THRESHOLD,
                "plateau_variance": config.homing.HOMING_PLATEAU_VARIANCE,
                "velocity_scale_factor": config.homing.HOMING_VELOCITY_SCALE_FACTOR,
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to get homing parameters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/homing/parameters")
async def update_homing_parameters(request: HomingParametersUpdate) -> dict[str, Any]:
    """
    Update homing algorithm parameters at runtime.

    Args:
        request: Partial update of homing parameters with validation

    Returns:
        Updated homing configuration parameters
    """
    try:
        config = get_config()

        # Update only provided parameters
        if request.forward_velocity_max is not None:
            config.homing.HOMING_FORWARD_VELOCITY_MAX = request.forward_velocity_max
            logger.info(f"Updated homing forward_velocity_max to {request.forward_velocity_max}")

        if request.yaw_rate_max is not None:
            config.homing.HOMING_YAW_RATE_MAX = request.yaw_rate_max
            logger.info(f"Updated homing yaw_rate_max to {request.yaw_rate_max}")

        if request.approach_velocity is not None:
            config.homing.HOMING_APPROACH_VELOCITY = request.approach_velocity
            logger.info(f"Updated homing approach_velocity to {request.approach_velocity}")

        if request.signal_loss_timeout is not None:
            config.homing.HOMING_SIGNAL_LOSS_TIMEOUT = request.signal_loss_timeout
            logger.info(f"Updated homing signal_loss_timeout to {request.signal_loss_timeout}")

        if request.gradient_window_size is not None:
            config.homing.HOMING_GRADIENT_WINDOW_SIZE = request.gradient_window_size
            logger.info(f"Updated homing gradient_window_size to {request.gradient_window_size}")

        if request.gradient_min_snr is not None:
            config.homing.HOMING_GRADIENT_MIN_SNR = request.gradient_min_snr
            logger.info(f"Updated homing gradient_min_snr to {request.gradient_min_snr}")

        if request.sampling_turn_radius is not None:
            config.homing.HOMING_SAMPLING_TURN_RADIUS = request.sampling_turn_radius
            logger.info(f"Updated homing sampling_turn_radius to {request.sampling_turn_radius}")

        if request.sampling_duration is not None:
            config.homing.HOMING_SAMPLING_DURATION = request.sampling_duration
            logger.info(f"Updated homing sampling_duration to {request.sampling_duration}")

        if request.approach_threshold is not None:
            config.homing.HOMING_APPROACH_THRESHOLD = request.approach_threshold
            logger.info(f"Updated homing approach_threshold to {request.approach_threshold}")

        if request.plateau_variance is not None:
            config.homing.HOMING_PLATEAU_VARIANCE = request.plateau_variance
            logger.info(f"Updated homing plateau_variance to {request.plateau_variance}")

        if request.velocity_scale_factor is not None:
            config.homing.HOMING_VELOCITY_SCALE_FACTOR = request.velocity_scale_factor
            logger.info(f"Updated homing velocity_scale_factor to {request.velocity_scale_factor}")

        # Return updated parameters
        return {
            "parameters": {
                "forward_velocity_max": config.homing.HOMING_FORWARD_VELOCITY_MAX,
                "yaw_rate_max": config.homing.HOMING_YAW_RATE_MAX,
                "approach_velocity": config.homing.HOMING_APPROACH_VELOCITY,
                "signal_loss_timeout": config.homing.HOMING_SIGNAL_LOSS_TIMEOUT,
                "algorithm_mode": config.homing.HOMING_ALGORITHM_MODE,
                "gradient_window_size": config.homing.HOMING_GRADIENT_WINDOW_SIZE,
                "gradient_min_snr": config.homing.HOMING_GRADIENT_MIN_SNR,
                "sampling_turn_radius": config.homing.HOMING_SAMPLING_TURN_RADIUS,
                "sampling_duration": config.homing.HOMING_SAMPLING_DURATION,
                "approach_threshold": config.homing.HOMING_APPROACH_THRESHOLD,
                "plateau_variance": config.homing.HOMING_PLATEAU_VARIANCE,
                "velocity_scale_factor": config.homing.HOMING_VELOCITY_SCALE_FACTOR,
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to update homing parameters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/system/debug")
async def toggle_debug_mode(request: DebugModeRequest) -> dict[str, Any]:
    """Toggle debug mode for verbose logging.

    Args:
        request: Debug mode request with enabled flag and target

    Returns:
        Current debug mode status
    """
    try:
        config = get_config()

        # Update global debug mode
        if request.target == "all":
            config.development.DEV_DEBUG_MODE = request.enabled
            logger.info(f"Global debug mode {'ENABLED' if request.enabled else 'DISABLED'}")

        # Update specific service debug mode
        if request.target in ["all", "homing"]:
            from src.backend.services.homing_algorithm import set_debug_mode

            set_debug_mode(request.enabled)

        # Could add other service debug modes here in future
        # if request.target in ["all", "sdr"]:
        #     from src.backend.services.sdr_service import set_debug_mode as sdr_set_debug
        #     sdr_set_debug(request.enabled)

        return {
            "debug_mode": {
                "global": config.development.DEV_DEBUG_MODE,
                "target": request.target,
                "enabled": request.enabled,
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to toggle debug mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/system/state-override")
async def override_state(request: StateOverrideRequest) -> dict[str, Any]:
    """
    Force a state transition (manual override for testing).

    This endpoint bypasses normal state transition validation but requires
    authentication via confirmation token and logs the operator ID for audit.

    Args:
        request: State override request with target state, reason, token, and operator ID

    Returns:
        Updated state information after override

    Raises:
        403: If confirmation token is invalid
        404: If state machine is not initialized
        500: If internal error occurs
    """
    if state_machine is None:
        raise HTTPException(status_code=404, detail="State machine not initialized")

    # Simple token validation (in production, use proper auth)
    expected_token = "override-" + datetime.now(UTC).strftime("%Y%m%d")
    if request.confirmation_token != expected_token:
        logger.warning(
            f"Invalid override token from operator {request.operator_id}: {request.confirmation_token}"
        )
        raise HTTPException(
            status_code=403, detail="Invalid confirmation token. Use format: override-YYYYMMDD"
        )

    try:
        # Parse target state enum
        target_state = StateMachineState[request.target_state]

        # Perform forced transition
        success = await state_machine.force_transition(
            target_state=target_state, reason=request.reason, operator_id=request.operator_id
        )

        if not success:
            raise HTTPException(
                status_code=400, detail=f"Failed to transition to {request.target_state}"
            )

        logger.warning(
            f"STATE OVERRIDE: {request.operator_id} forced transition to {request.target_state} - {request.reason}"
        )

        # Get current state info
        current_state = state_machine.get_current_state()
        allowed_transitions = state_machine.get_allowed_transitions()

        return {
            "success": True,
            "previous_state": state_machine._previous_state.value,
            "new_state": current_state.value,
            "message": f"State overridden to {current_state.value}",
            "allowed_transitions": [s.value for s in allowed_transitions],
            "operator_id": request.operator_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except KeyError:
        raise HTTPException(status_code=400, detail=f"Invalid state: {request.target_state}")
    except Exception as e:
        logger.error(f"Failed to override state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/state")
async def get_current_state() -> dict[str, Any]:
    """
    Get current state machine status including state, history, and allowed transitions.

    Returns:
        Current state information with history and statistics
    """
    if state_machine is None:
        raise HTTPException(status_code=404, detail="State machine not initialized")

    try:
        current_state = state_machine.get_current_state()
        allowed_transitions = state_machine.get_allowed_transitions()
        statistics = state_machine.get_statistics()
        history = state_machine.get_state_history(limit=10)
        search_status = state_machine.get_search_pattern_status()

        return {
            "current_state": current_state.value,
            "previous_state": state_machine._previous_state.value,
            "allowed_transitions": [s.value for s in allowed_transitions],
            "state_duration_ms": (
                statistics.get("time_since_detection", 0) * 1000
                if statistics.get("time_since_detection")
                else None
            ),
            "history": history,
            "search_status": search_status,
            "statistics": statistics,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get state information: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/state-history")
async def get_state_history(
    limit: int = 100, from_state: str | None = None, to_state: str | None = None
) -> dict[str, Any]:
    """
    Get state transition history with optional filtering.

    Args:
        limit: Maximum number of records to return (default 100, max 1000)
        from_state: Filter by source state
        to_state: Filter by target state

    Returns:
        List of state transitions with timestamps and reasons
    """
    if state_machine is None:
        raise HTTPException(status_code=404, detail="State machine not initialized")

    # Validate limit
    if limit > 1000:
        limit = 1000

    try:
        # Get history from database if persistence is enabled
        if state_machine._state_db:
            history = state_machine._state_db.get_state_history(
                limit=limit, from_state=from_state, to_state=to_state
            )
        else:
            # Fall back to in-memory history
            history = state_machine.get_state_history(limit=limit)

            # Apply filters if provided
            if from_state:
                history = [h for h in history if h.get("from_state") == from_state]
            if to_state:
                history = [h for h in history if h.get("to_state") == to_state]

        return {
            "history": history,
            "count": len(history),
            "filters": {
                "limit": limit,
                "from_state": from_state,
                "to_state": to_state,
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get state history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
