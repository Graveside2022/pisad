"""API routes for state machine management."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.backend.services.state_machine import SystemState
from src.backend.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/state", tags=["state"])


class StateTransitionRequest(BaseModel):
    """Request model for state transitions."""

    target_state: str = Field(..., description="Target state to transition to")
    reason: str | None = Field(None, description="Reason for transition")


class ForceTransitionRequest(BaseModel):
    """Request model for forced state transitions."""

    target_state: str = Field(..., description="Target state to force transition to")
    reason: str = Field(..., description="Reason for forced transition")
    operator_id: str | None = Field(None, description="Operator ID for audit trail")


class StateTimeoutRequest(BaseModel):
    """Request model for setting state timeouts."""

    state: str = Field(..., description="State to configure timeout for")
    timeout_seconds: float = Field(..., ge=0, description="Timeout in seconds (0 to disable)")


class StateResponse(BaseModel):
    """Response model for state information."""

    current_state: str
    previous_state: str
    allowed_transitions: list[str]
    homing_enabled: bool
    state_duration_seconds: float
    state_timeout_seconds: float


class StateHistoryResponse(BaseModel):
    """Response model for state history."""

    history: list[dict[str, Any]]
    total_transitions: int


class StateStatisticsResponse(BaseModel):
    """Response model for state statistics."""

    current_state: str
    previous_state: str
    homing_enabled: bool
    detection_count: int
    last_detection_time: float | None
    time_since_detection: float | None
    state_changes: int
    state_duration_seconds: float
    state_timeout_seconds: float


class TelemetryMetricsResponse(BaseModel):
    """Response model for telemetry metrics."""

    total_transitions: int
    state_durations: dict[str, float]
    transition_frequencies: dict[str, int]
    average_transition_time_ms: float
    current_state_duration_s: float
    uptime_seconds: float
    state_entry_counts: dict[str, int]


def get_state_machine():
    """Get state machine instance from app state."""
    from src.backend.core.app import app

    if not hasattr(app.state, "state_machine"):
        raise HTTPException(status_code=503, detail="State machine not initialized")
    return app.state.state_machine


@router.get("/current", response_model=StateResponse)
async def get_current_state(state_machine=Depends(get_state_machine)):
    """Get current state and available transitions.

    Returns:
        Current state information including allowed transitions
    """
    try:
        current = state_machine.get_current_state()
        allowed = state_machine.get_allowed_transitions()

        return StateResponse(
            current_state=current.value,
            previous_state=state_machine._previous_state.value,
            allowed_transitions=[s.value for s in allowed],
            homing_enabled=state_machine._homing_enabled,
            state_duration_seconds=state_machine.get_state_duration(),
            state_timeout_seconds=state_machine._state_timeouts.get(current, 0),
        )
    except Exception as e:
        logger.error(f"Failed to get current state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transition")
async def transition_state(
    request: StateTransitionRequest, state_machine=Depends(get_state_machine)
):
    """Request a state transition.

    Args:
        request: State transition request with target state and optional reason

    Returns:
        Success status and new state information

    Raises:
        HTTPException: If transition is invalid or fails
    """
    try:
        target_state = SystemState(request.target_state.upper())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid state: {request.target_state}")

    try:
        success = await state_machine.transition_to(target_state, request.reason)

        if not success:
            allowed = state_machine.get_allowed_transitions()
            raise HTTPException(
                status_code=400,
                detail=f"Invalid transition from {state_machine.get_current_state().value} "
                f"to {target_state.value}. Allowed: {[s.value for s in allowed]}",
            )

        return {
            "success": True,
            "current_state": state_machine.get_current_state().value,
            "message": f"Transitioned to {target_state.value}",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"State transition failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/force-transition")
async def force_transition(
    request: ForceTransitionRequest, state_machine=Depends(get_state_machine)
):
    """Force a state transition (override normal validation).

    This endpoint allows manual override of state transitions for testing
    or emergency situations. Use with caution.

    Args:
        request: Force transition request with target state, reason, and operator ID

    Returns:
        Success status and new state information
    """
    try:
        target_state = SystemState(request.target_state.upper())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid state: {request.target_state}")

    try:
        success = await state_machine.force_transition(
            target_state, request.reason, request.operator_id
        )

        return {
            "success": success,
            "current_state": state_machine.get_current_state().value,
            "message": f"Forced transition to {target_state.value}",
            "warning": "This was a forced transition bypassing normal validation",
        }
    except Exception as e:
        logger.error(f"Forced transition failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emergency-stop")
async def emergency_stop(state_machine=Depends(get_state_machine)):
    """Trigger emergency stop and return to IDLE state.

    Immediately transitions the system to IDLE state and disables homing.

    Returns:
        Success status
    """
    try:
        await state_machine.emergency_stop("API emergency stop request")
        return {
            "success": True,
            "current_state": "IDLE",
            "message": "Emergency stop executed - system returned to IDLE",
        }
    except Exception as e:
        logger.error(f"Emergency stop failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/homing/{enabled}")
async def set_homing_enabled(enabled: bool, state_machine=Depends(get_state_machine)):
    """Enable or disable automatic homing.

    Args:
        enabled: True to enable homing, False to disable

    Returns:
        Success status and current homing state
    """
    try:
        state_machine.enable_homing(enabled)
        return {
            "success": True,
            "homing_enabled": enabled,
            "message": f"Homing {'enabled' if enabled else 'disabled'}",
        }
    except Exception as e:
        logger.error(f"Failed to set homing enabled: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/timeout")
async def set_state_timeout(request: StateTimeoutRequest, state_machine=Depends(get_state_machine)):
    """Set or update timeout for a specific state.

    Args:
        request: Timeout configuration request

    Returns:
        Success status
    """
    try:
        target_state = SystemState(request.state.upper())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid state: {request.state}")

    try:
        state_machine.set_state_timeout(target_state, request.timeout_seconds)
        return {
            "success": True,
            "state": target_state.value,
            "timeout_seconds": request.timeout_seconds,
            "message": f"Timeout for {target_state.value} set to {request.timeout_seconds} seconds",
        }
    except Exception as e:
        logger.error(f"Failed to set state timeout: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=StateHistoryResponse)
async def get_state_history(limit: int = 10, state_machine=Depends(get_state_machine)):
    """Get recent state change history.

    Args:
        limit: Maximum number of events to return (default: 10, 0 for all)

    Returns:
        State change history
    """
    try:
        history = state_machine.get_state_history(limit=limit)
        return StateHistoryResponse(
            history=history, total_transitions=len(state_machine._state_history)
        )
    except Exception as e:
        logger.error(f"Failed to get state history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics", response_model=StateStatisticsResponse)
async def get_state_statistics(state_machine=Depends(get_state_machine)):
    """Get comprehensive state machine statistics.

    Returns:
        Current statistics including detection counts and timing information
    """
    try:
        stats = state_machine.get_statistics()
        return StateStatisticsResponse(**stats)
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/telemetry", response_model=TelemetryMetricsResponse)
async def get_telemetry_metrics(state_machine=Depends(get_state_machine)):
    """Get detailed telemetry metrics.

    Returns:
        Comprehensive telemetry metrics including state durations and transition frequencies
    """
    try:
        metrics = state_machine.get_telemetry_metrics()
        return TelemetryMetricsResponse(**metrics)
    except Exception as e:
        logger.error(f"Failed to get telemetry metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/telemetry/send")
async def send_telemetry_update(state_machine=Depends(get_state_machine)):
    """Manually trigger telemetry update via MAVLink.

    Returns:
        Success status
    """
    try:
        await state_machine.send_telemetry_update()
        return {"success": True, "message": "Telemetry update sent"}
    except Exception as e:
        logger.error(f"Failed to send telemetry update: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search-pattern/status")
async def get_search_pattern_status(state_machine=Depends(get_state_machine)):
    """Get current search pattern execution status.

    Returns:
        Search pattern status including progress and waypoint information
    """
    try:
        status = state_machine.get_search_pattern_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get search pattern status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search-pattern/start")
async def start_search_pattern(state_machine=Depends(get_state_machine)):
    """Start executing the loaded search pattern.

    Returns:
        Success status

    Raises:
        HTTPException: If no pattern is loaded or start fails
    """
    try:
        success = await state_machine.start_search_pattern()
        if not success:
            raise HTTPException(
                status_code=400,
                detail="Failed to start search pattern. Ensure pattern is loaded and system is ready.",
            )
        return {"success": True, "message": "Search pattern started"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start search pattern: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search-pattern/pause")
async def pause_search_pattern(state_machine=Depends(get_state_machine)):
    """Pause search pattern execution.

    Returns:
        Success status
    """
    try:
        success = await state_machine.pause_search_pattern()
        if not success:
            raise HTTPException(status_code=400, detail="Cannot pause - pattern not executing")
        return {"success": True, "message": "Search pattern paused"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to pause search pattern: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search-pattern/resume")
async def resume_search_pattern(state_machine=Depends(get_state_machine)):
    """Resume paused search pattern.

    Returns:
        Success status
    """
    try:
        success = await state_machine.resume_search_pattern()
        if not success:
            raise HTTPException(status_code=400, detail="Cannot resume - pattern not paused")
        return {"success": True, "message": "Search pattern resumed"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resume search pattern: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search-pattern/stop")
async def stop_search_pattern(state_machine=Depends(get_state_machine)):
    """Stop search pattern execution and return to IDLE.

    Returns:
        Success status
    """
    try:
        success = await state_machine.stop_search_pattern()
        if not success:
            raise HTTPException(status_code=400, detail="No active search pattern")
        return {"success": True, "message": "Search pattern stopped"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop search pattern: {e}")
        raise HTTPException(status_code=500, detail=str(e))
