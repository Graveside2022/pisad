"""API routes for telemetry configuration and management."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.backend.core.config import get_config
from src.backend.core.exceptions import (
    ConfigurationError,
    PISADException,
    SignalProcessingError,
)
from src.backend.services.mavlink_service import MAVLinkService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/telemetry", tags=["telemetry"])


class TelemetryConfigRequest(BaseModel):
    """Request model for telemetry configuration updates."""

    rssi_rate_hz: float | None = Field(None, ge=0.1, le=10.0, description="RSSI update rate in Hz")
    health_interval_seconds: int | None = Field(
        None, ge=1, le=60, description="Health status interval in seconds"
    )
    detection_throttle_ms: int | None = Field(
        None, ge=100, le=5000, description="Detection event throttle in milliseconds"
    )


class TelemetryConfigResponse(BaseModel):
    """Response model for telemetry configuration."""

    rssi_rate_hz: float
    health_interval_seconds: int
    detection_throttle_ms: int
    statustext_severity: str
    max_bandwidth_kbps: float


class TelemetryStatusResponse(BaseModel):
    """Response model for telemetry status."""

    connected: bool
    current_rssi: float
    messages_sent: int
    bandwidth_usage_kbps: float
    config: TelemetryConfigResponse


# Dependency to get MAVLink service instance
# In production, this would be injected from the app state
_mavlink_service: MAVLinkService | None = None


def get_mavlink_service() -> MAVLinkService:
    """Get MAVLink service instance."""
    global _mavlink_service
    if _mavlink_service is None:
        config = get_config()
        # Initialize MAVLink service with config
        _mavlink_service = MAVLinkService(
            device_path="/dev/ttyACM0",  # This would come from config
            baud_rate=115200,
        )
        # Apply telemetry config
        _mavlink_service.update_telemetry_config(
            {
                "rssi_rate_hz": config.telemetry.TELEMETRY_RSSI_RATE_HZ,
                "health_interval_seconds": config.telemetry.TELEMETRY_HEALTH_INTERVAL_SECONDS,
                "detection_throttle_ms": config.telemetry.TELEMETRY_DETECTION_THROTTLE_MS,
            }
        )
    return _mavlink_service


@router.get("/config", response_model=TelemetryConfigResponse)
async def get_telemetry_config(
    mavlink: MAVLinkService = Depends(get_mavlink_service),
) -> TelemetryConfigResponse:
    """
    Get current telemetry configuration.

    Returns current telemetry rate settings and bandwidth limits.
    """
    try:
        config = get_config()
        current_config = mavlink.get_telemetry_config()

        return TelemetryConfigResponse(
            rssi_rate_hz=current_config["rssi_rate_hz"],
            health_interval_seconds=current_config["health_interval_seconds"],
            detection_throttle_ms=current_config["detection_throttle_ms"],
            statustext_severity=config.telemetry.TELEMETRY_STATUSTEXT_SEVERITY,
            max_bandwidth_kbps=config.telemetry.TELEMETRY_MAX_BANDWIDTH_KBPS,
        )
    except ConfigurationError as e:
        logger.error(f"Failed to get telemetry config: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve telemetry configuration")


@router.put("/config", response_model=TelemetryConfigResponse)
async def update_telemetry_config(
    request: TelemetryConfigRequest, mavlink: MAVLinkService = Depends(get_mavlink_service)
) -> TelemetryConfigResponse:
    """
    Update telemetry configuration.

    Adjusts telemetry message rates to prevent bandwidth saturation.
    Changes take effect immediately.
    """
    try:
        # Build update dict with only provided values
        update_config: dict[str, Any] = {}
        if request.rssi_rate_hz is not None:
            update_config["rssi_rate_hz"] = request.rssi_rate_hz
        if request.health_interval_seconds is not None:
            update_config["health_interval_seconds"] = request.health_interval_seconds
        if request.detection_throttle_ms is not None:
            update_config["detection_throttle_ms"] = request.detection_throttle_ms

        # Apply configuration
        mavlink.update_telemetry_config(update_config)

        # Return updated configuration
        return await get_telemetry_config(mavlink)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ConfigurationError as e:
        logger.error(f"Failed to update telemetry config: {e}")
        raise HTTPException(status_code=500, detail="Failed to update telemetry configuration")


@router.get("/status", response_model=TelemetryStatusResponse)
async def get_telemetry_status(
    mavlink: MAVLinkService = Depends(get_mavlink_service),
) -> TelemetryStatusResponse:
    """
    Get current telemetry status.

    Returns connection status, current values, and message statistics.
    """
    try:
        config = get_config()
        current_config = mavlink.get_telemetry_config()

        # Calculate approximate bandwidth usage
        # NAMED_VALUE_FLOAT: ~20 bytes per message
        # STATUSTEXT: ~60 bytes per message
        rssi_bps = current_config["rssi_rate_hz"] * 20 * 8
        health_bps = (1.0 / current_config["health_interval_seconds"]) * 60 * 8
        detection_bps = 10 * 8  # Estimate for detection events
        total_bandwidth_kbps = (rssi_bps + health_bps + detection_bps) / 1000.0

        return TelemetryStatusResponse(
            connected=mavlink.is_connected(),
            current_rssi=mavlink._rssi_value,
            messages_sent=0,  # Would need to track this in service
            bandwidth_usage_kbps=round(total_bandwidth_kbps, 2),
            config=TelemetryConfigResponse(
                rssi_rate_hz=current_config["rssi_rate_hz"],
                health_interval_seconds=current_config["health_interval_seconds"],
                detection_throttle_ms=current_config["detection_throttle_ms"],
                statustext_severity=config.telemetry.TELEMETRY_STATUSTEXT_SEVERITY,
                max_bandwidth_kbps=config.telemetry.TELEMETRY_MAX_BANDWIDTH_KBPS,
            ),
        )
    except PISADException as e:
        logger.error(f"Failed to get telemetry status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve telemetry status")


class TestRSSIRequest(BaseModel):
    """Request model for test RSSI telemetry."""

    rssi: float = Field(..., ge=-120.0, le=0.0, description="RSSI value in dBm")


@router.post("/test/rssi")
async def test_rssi_telemetry(
    request: TestRSSIRequest, mavlink: MAVLinkService = Depends(get_mavlink_service)
) -> dict[str, Any]:
    """
    Test RSSI telemetry by sending a test value.

    Useful for verifying GCS reception of telemetry data.
    """
    try:
        if not mavlink.is_connected():
            raise HTTPException(status_code=503, detail="MAVLink not connected")

        # Update RSSI value
        mavlink.update_rssi_value(request.rssi)

        # Send immediately
        success = mavlink.send_named_value_float("PISAD_RSSI", request.rssi)

        if success:
            return {"message": f"Test RSSI value {request.rssi} dBm sent successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to send test RSSI value")

    except HTTPException:
        raise
    except SignalProcessingError as e:
        logger.error(f"Failed to send test RSSI: {e}")
        raise HTTPException(status_code=500, detail="Failed to send test telemetry")


class TestStatusRequest(BaseModel):
    """Request model for test status message."""

    message: str = Field(..., max_length=43, description="Message text (max 43 chars)")
    severity: int = Field(
        6, ge=0, le=7, description="Severity level (0=emergency, 6=info, 7=debug)"
    )


@router.post("/test/status")
async def test_status_message(
    request: TestStatusRequest, mavlink: MAVLinkService = Depends(get_mavlink_service)
) -> dict[str, Any]:
    """
    Test STATUSTEXT message sending.

    Sends a test message to verify GCS reception.
    Severity levels: 0=emergency, 6=info, 7=debug
    """
    try:
        if not mavlink.is_connected():
            raise HTTPException(status_code=503, detail="MAVLink not connected")

        # Add PISAD prefix
        full_message = f"PISAD: {request.message}"

        # Send status message
        success = mavlink.send_statustext(full_message, request.severity)

        if success:
            return {"message": f"Test status message sent: '{full_message}'"}
        else:
            raise HTTPException(status_code=500, detail="Failed to send test status message")

    except HTTPException:
        raise
    except PISADException as e:
        logger.error(f"Failed to send test status: {e}")
        raise HTTPException(status_code=500, detail="Failed to send test message")
