"""Search pattern API routes."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.backend.core.exceptions import (
    DatabaseError,
    MAVLinkError,
    PISADException,
    StateTransitionError,
)
from src.backend.services.mavlink_service import MAVLinkService
from src.backend.services.search_pattern_generator import (
    CenterRadiusBoundary,
    CornerBoundary,
    PatternType,
    SearchPattern,
    SearchPatternGenerator,
)
from src.backend.services.state_machine import StateMachine
from src.backend.services.waypoint_exporter import WaypointExporter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/search", tags=["search"])


# Store active pattern in memory (in production, use database)
active_patterns: dict[str, SearchPattern] = {}

# Pattern persistence file
PATTERN_STATE_FILE = Path("/tmp/pisad_search_patterns.json")


def save_pattern_state():
    """Save active patterns to disk for recovery."""
    try:
        # Convert patterns to serializable format
        patterns_data = {}
        for pattern_id, pattern in active_patterns.items():
            patterns_data[pattern_id] = {
                "id": pattern.id,
                "pattern_type": pattern.pattern_type.value,
                "state": pattern.state,
                "completed_waypoints": pattern.completed_waypoints,
                "current_waypoint_index": getattr(pattern, "current_waypoint_index", 0),
                "progress_percent": pattern.progress_percent,
                "created_at": pattern.created_at.isoformat() if pattern.created_at else None,
                "started_at": pattern.started_at.isoformat() if pattern.started_at else None,
                "paused_at": pattern.paused_at.isoformat() if pattern.paused_at else None,
            }

        # Write to file
        with open(PATTERN_STATE_FILE, "w") as f:
            json.dump(patterns_data, f)

        logger.debug(f"Saved {len(patterns_data)} pattern states to disk")
    except StateTransitionError as e:
        logger.error(f"Failed to save pattern state: {e}")


def load_pattern_state():
    """Load patterns from disk on startup."""
    try:
        if not PATTERN_STATE_FILE.exists():
            return

        with open(PATTERN_STATE_FILE) as f:
            patterns_data = json.load(f)

        # Note: This is simplified - in production would need to reconstruct full patterns
        logger.info(f"Loaded {len(patterns_data)} pattern states from disk")

    except StateTransitionError as e:
        logger.error(f"Failed to load pattern state: {e}")


# Load patterns on module import
load_pattern_state()


async def broadcast_pattern_update(pattern: SearchPattern, event_type: str = "pattern_update"):
    """Broadcast pattern updates via WebSocket.

    Args:
        pattern: The search pattern to broadcast
        event_type: Type of event (pattern_update, pattern_created, pattern_status, etc.)
    """
    try:
        from src.backend.api.websocket import broadcast_message

        message = {
            "type": event_type,
            "data": {
                "pattern_id": pattern.id,
                "state": pattern.state,
                "progress_percent": pattern.progress_percent,
                "completed_waypoints": pattern.completed_waypoints,
                "total_waypoints": pattern.total_waypoints,
                "estimated_time_remaining": pattern.estimated_time_remaining,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        }

        await broadcast_message(message)
    except DatabaseError as e:
        logger.error(f"Failed to broadcast pattern update: {e}")


class CenterRadiusBoundaryRequest(BaseModel):
    """Center-radius boundary request model."""

    type: Literal["center_radius"]
    center: dict[str, float] = Field(..., description="Center point with lat/lon")
    radius: float = Field(..., ge=0, description="Radius in meters")


class CornerBoundaryRequest(BaseModel):
    """Corner boundary request model."""

    type: Literal["corners"]
    corners: list[dict[str, float]] = Field(..., min_items=3, description="Corner coordinates")


class SearchPatternRequest(BaseModel):
    """Search pattern generation request."""

    pattern: Literal["expanding_square", "spiral", "lawnmower"]
    spacing: float = Field(..., ge=50, le=100, description="Track spacing in meters")
    velocity: float = Field(..., ge=5, le=10, description="Search velocity in m/s")
    bounds: CenterRadiusBoundaryRequest | CornerBoundaryRequest
    altitude: float | None = Field(50.0, ge=10, le=400, description="Flight altitude in meters")


class SearchPatternResponse(BaseModel):
    """Search pattern generation response."""

    pattern_id: str
    waypoint_count: int
    estimated_duration: float
    total_distance: float


class PatternPreviewResponse(BaseModel):
    """Pattern preview response."""

    waypoints: list[dict[str, float]]
    boundary: dict | None = None
    total_distance: float
    estimated_time: float


class PatternStatusResponse(BaseModel):
    """Pattern status response."""

    pattern_id: str
    state: str
    progress_percent: float
    completed_waypoints: int
    total_waypoints: int
    current_waypoint: int
    estimated_time_remaining: float


class PatternControlRequest(BaseModel):
    """Pattern control request."""

    action: Literal["pause", "resume", "stop"]


class PatternControlResponse(BaseModel):
    """Pattern control response."""

    success: bool
    new_state: str


# Dependency to get pattern generator
def get_pattern_generator() -> SearchPatternGenerator:
    """Get pattern generator instance."""
    return SearchPatternGenerator(default_altitude=50.0)


# Dependency to get state machine
def get_state_machine() -> StateMachine | None:
    """Get state machine instance."""
    # This will be integrated with the actual state machine
    from src.backend.core.app import get_app

    app = get_app()
    if hasattr(app.state, "state_machine"):
        return app.state.state_machine
    return None


# Dependency to get MAVLink service
def get_mavlink_service() -> MAVLinkService | None:
    """Get MAVLink service instance."""
    from src.backend.core.app import get_app

    app = get_app()
    if hasattr(app.state, "mavlink_service"):
        return app.state.mavlink_service
    return None


@router.post("/pattern", response_model=SearchPatternResponse)
async def create_search_pattern(
    request: SearchPatternRequest,
    generator: SearchPatternGenerator = Depends(get_pattern_generator),
) -> SearchPatternResponse:
    """Create a new search pattern.

    Args:
        request: Search pattern parameters
        generator: Pattern generator instance

    Returns:
        Pattern creation response with ID and metadata
    """
    try:
        # Convert request boundary to internal model
        if request.bounds.type == "center_radius":
            boundary = CenterRadiusBoundary(
                center_lat=request.bounds.center["lat"],
                center_lon=request.bounds.center["lon"],
                radius=request.bounds.radius,
            )
        else:
            corners = [(c["lat"], c["lon"]) for c in request.bounds.corners]
            boundary = CornerBoundary(corners=corners)

        # Validate boundary
        generator.validate_boundary(boundary)

        # Generate pattern
        pattern_type = PatternType(request.pattern)
        pattern = generator.generate_pattern(
            pattern_type=pattern_type,
            spacing=request.spacing,
            velocity=request.velocity,
            boundary=boundary,
            altitude=request.altitude,
        )

        # Store pattern
        active_patterns[pattern.id] = pattern

        # Save state to disk
        save_pattern_state()

        # Calculate total distance
        total_distance = generator._calculate_total_distance(pattern.waypoints)

        # Broadcast pattern creation
        await broadcast_pattern_update(pattern, "pattern_created")

        logger.info(f"Created search pattern {pattern.id} with {pattern.total_waypoints} waypoints")

        return SearchPatternResponse(
            pattern_id=pattern.id,
            waypoint_count=pattern.total_waypoints,
            estimated_duration=pattern.estimated_time_remaining,
            total_distance=total_distance,
        )

    except ValueError as e:
        logger.error(f"Invalid pattern parameters: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except PISADException as e:
        logger.error(f"Failed to create search pattern: {e}")
        raise HTTPException(status_code=500, detail="Failed to create search pattern")


@router.get("/pattern/preview", response_model=PatternPreviewResponse)
async def get_pattern_preview(pattern_id: str | None = None) -> PatternPreviewResponse:
    """Get preview of search pattern waypoints.

    Args:
        pattern_id: Optional pattern ID to preview

    Returns:
        Pattern preview with waypoints and metadata
    """
    try:
        # Get the most recent pattern if no ID specified
        if pattern_id:
            if pattern_id not in active_patterns:
                raise HTTPException(status_code=404, detail="Pattern not found")
            pattern = active_patterns[pattern_id]
        else:
            if not active_patterns:
                raise HTTPException(status_code=404, detail="No active patterns")
            # Get most recent pattern
            pattern = list(active_patterns.values())[-1]

        # Convert waypoints to response format
        waypoints = [wp.to_dict() for wp in pattern.waypoints]

        # Create boundary GeoJSON (simplified for now)
        boundary = None
        if isinstance(pattern.boundary, CenterRadiusBoundary):
            boundary = {
                "type": "Circle",
                "center": [pattern.boundary.center_lon, pattern.boundary.center_lat],
                "radius": pattern.boundary.radius,
            }
        elif isinstance(pattern.boundary, CornerBoundary):
            boundary = {
                "type": "Polygon",
                "coordinates": [[[lon, lat] for lat, lon in pattern.boundary.corners]],
            }

        # Calculate total distance
        generator = SearchPatternGenerator()
        total_distance = generator._calculate_total_distance(pattern.waypoints)

        return PatternPreviewResponse(
            waypoints=waypoints,
            boundary=boundary,
            total_distance=total_distance,
            estimated_time=pattern.estimated_time_remaining,
        )

    except HTTPException:
        raise
    except PISADException as e:
        logger.error(f"Failed to get pattern preview: {e}")
        raise HTTPException(status_code=500, detail="Failed to get pattern preview")


@router.get("/pattern/status", response_model=PatternStatusResponse)
async def get_pattern_status(pattern_id: str | None = None) -> PatternStatusResponse:
    """Get current status of search pattern execution.

    Args:
        pattern_id: Optional pattern ID to check status

    Returns:
        Pattern execution status
    """
    try:
        # Get the most recent pattern if no ID specified
        if pattern_id:
            if pattern_id not in active_patterns:
                raise HTTPException(status_code=404, detail="Pattern not found")
            pattern = active_patterns[pattern_id]
        else:
            if not active_patterns:
                raise HTTPException(status_code=404, detail="No active patterns")
            # Get most recent pattern
            pattern = list(active_patterns.values())[-1]

        # Calculate current waypoint (for simulation, would come from MAVLink)
        current_waypoint = pattern.completed_waypoints
        if pattern.state == "EXECUTING" and current_waypoint < pattern.total_waypoints:
            current_waypoint += 1

        return PatternStatusResponse(
            pattern_id=pattern.id,
            state=pattern.state,
            progress_percent=pattern.progress_percent,
            completed_waypoints=pattern.completed_waypoints,
            total_waypoints=pattern.total_waypoints,
            current_waypoint=current_waypoint,
            estimated_time_remaining=pattern.estimated_time_remaining,
        )

    except HTTPException:
        raise
    except PISADException as e:
        logger.error(f"Failed to get pattern status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get pattern status")


@router.post("/pattern/control", response_model=PatternControlResponse)
async def control_pattern(
    request: PatternControlRequest, pattern_id: str | None = None
) -> PatternControlResponse:
    """Control search pattern execution (pause/resume/stop).

    Args:
        request: Control action request
        pattern_id: Optional pattern ID to control

    Returns:
        Control response with new state
    """
    try:
        # Get the most recent pattern if no ID specified
        if pattern_id:
            if pattern_id not in active_patterns:
                raise HTTPException(status_code=404, detail="Pattern not found")
            pattern = active_patterns[pattern_id]
        else:
            if not active_patterns:
                raise HTTPException(status_code=404, detail="No active patterns")
            # Get most recent pattern
            pattern = list(active_patterns.values())[-1]

        # Handle control action
        if request.action == "pause":
            if pattern.state != "EXECUTING":
                raise HTTPException(status_code=400, detail="Pattern is not executing")
            pattern.state = "PAUSED"
            pattern.paused_at = datetime.now(UTC)
            new_state = "PAUSED"

        elif request.action == "resume":
            if pattern.state != "PAUSED":
                raise HTTPException(status_code=400, detail="Pattern is not paused")
            pattern.state = "EXECUTING"
            pattern.paused_at = None
            new_state = "EXECUTING"

        elif request.action == "stop":
            pattern.state = "IDLE"
            pattern.completed_waypoints = 0
            pattern.progress_percent = 0.0
            pattern.started_at = None
            pattern.paused_at = None
            new_state = "IDLE"

        # Broadcast pattern state change
        await broadcast_pattern_update(pattern, f"pattern_{request.action}")

        # Save state to disk
        save_pattern_state()

        logger.info(
            f"Pattern {pattern.id} control action '{request.action}' -> state '{new_state}'"
        )

        return PatternControlResponse(success=True, new_state=new_state)

    except HTTPException:
        raise
    except PISADException as e:
        logger.error(f"Failed to control pattern: {e}")
        raise HTTPException(status_code=500, detail="Failed to control pattern")


@router.post("/pattern/import")
async def import_pattern(
    content: str,
    format: Literal["qgc"] = "qgc",
    generator: SearchPatternGenerator = Depends(get_pattern_generator),
) -> SearchPatternResponse:
    """Import search pattern from waypoint file.

    Args:
        content: Waypoint file content
        format: Import format (currently only QGC WPL)
        generator: Pattern generator instance

    Returns:
        Pattern creation response with ID and metadata
    """
    try:
        exporter = WaypointExporter()

        if format == "qgc":
            waypoints = exporter.import_qgc_wpl(content)
        else:
            raise HTTPException(status_code=400, detail="Unsupported import format")

        # Validate imported waypoints
        exporter.validate_waypoints(waypoints)

        # Create a pattern from imported waypoints
        # This is a simplified version - in production would need more metadata
        import uuid
        from datetime import UTC, datetime

        pattern = SearchPattern(
            id=str(uuid.uuid4()),
            pattern_type=PatternType.LAWNMOWER,  # Default type for imports
            spacing=75.0,  # Default spacing
            velocity=7.0,  # Default velocity
            boundary=None,  # No boundary for imported patterns
            waypoints=waypoints,
            total_waypoints=len(waypoints),
            completed_waypoints=0,
            state="IDLE",
            progress_percent=0.0,
            estimated_time_remaining=len(waypoints) * 10,  # Rough estimate
            created_at=datetime.now(UTC),
            started_at=None,
            paused_at=None,
        )

        # Store pattern
        active_patterns[pattern.id] = pattern

        # Calculate total distance
        total_distance = generator._calculate_total_distance(waypoints)

        logger.info(f"Imported pattern {pattern.id} with {len(waypoints)} waypoints")

        return SearchPatternResponse(
            pattern_id=pattern.id,
            waypoint_count=len(waypoints),
            estimated_duration=pattern.estimated_time_remaining,
            total_distance=total_distance,
        )

    except ValueError as e:
        logger.error(f"Invalid import format: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except PISADException as e:
        logger.error(f"Failed to import pattern: {e}")
        raise HTTPException(status_code=500, detail="Failed to import pattern")


@router.post("/pattern/upload")
async def upload_pattern_to_mavlink(
    pattern_id: str | None = None,
    mavlink_service: MAVLinkService | None = Depends(get_mavlink_service),
    state_machine: StateMachine | None = Depends(get_state_machine),
) -> dict:
    """Upload search pattern to flight controller via MAVLink.

    Args:
        pattern_id: Optional pattern ID to upload
        mavlink_service: MAVLink service instance
        state_machine: State machine instance

    Returns:
        Upload status response
    """
    try:
        if not mavlink_service:
            raise HTTPException(status_code=503, detail="MAVLink service not available")

        # Get the pattern
        if pattern_id:
            if pattern_id not in active_patterns:
                raise HTTPException(status_code=404, detail="Pattern not found")
            pattern = active_patterns[pattern_id]
        else:
            if not active_patterns:
                raise HTTPException(status_code=404, detail="No active patterns")
            pattern = list(active_patterns.values())[-1]

        # Convert waypoints to MAVLink format
        mavlink_waypoints = [
            {"lat": wp.latitude, "lon": wp.longitude, "alt": wp.altitude}
            for wp in pattern.waypoints
        ]

        # Upload to flight controller
        success = await mavlink_service.upload_mission(mavlink_waypoints)

        if not success:
            raise HTTPException(
                status_code=500, detail="Failed to upload mission to flight controller"
            )

        # Update state machine with pattern
        if state_machine:
            state_machine.set_search_pattern(pattern)

        logger.info(
            f"Uploaded pattern {pattern.id} with {len(mavlink_waypoints)} waypoints to MAVLink"
        )

        return {
            "success": True,
            "pattern_id": pattern.id,
            "waypoint_count": len(mavlink_waypoints),
            "message": "Pattern uploaded to flight controller",
        }

    except HTTPException:
        raise
    except MAVLinkError as e:
        logger.error(f"Failed to upload pattern to MAVLink: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload pattern")


@router.post("/pattern/mavlink/start")
async def start_mavlink_mission(
    mavlink_service: MAVLinkService | None = Depends(get_mavlink_service),
    state_machine: StateMachine | None = Depends(get_state_machine),
) -> dict:
    """Start the uploaded mission on the flight controller.

    Args:
        mavlink_service: MAVLink service instance
        state_machine: State machine instance

    Returns:
        Start status response
    """
    try:
        if not mavlink_service:
            raise HTTPException(status_code=503, detail="MAVLink service not available")

        # Start mission
        success = await mavlink_service.start_mission()

        if not success:
            raise HTTPException(status_code=500, detail="Failed to start mission")

        # Update state machine
        if state_machine:
            await state_machine.start_search_pattern()

        return {"success": True, "message": "Mission started on flight controller"}

    except HTTPException:
        raise
    except MAVLinkError as e:
        logger.error(f"Failed to start MAVLink mission: {e}")
        raise HTTPException(status_code=500, detail="Failed to start mission")


@router.post("/pattern/mavlink/pause")
async def pause_mavlink_mission(
    mavlink_service: MAVLinkService | None = Depends(get_mavlink_service),
    state_machine: StateMachine | None = Depends(get_state_machine),
) -> dict:
    """Pause the current mission on the flight controller.

    Args:
        mavlink_service: MAVLink service instance
        state_machine: State machine instance

    Returns:
        Pause status response
    """
    try:
        if not mavlink_service:
            raise HTTPException(status_code=503, detail="MAVLink service not available")

        # Pause mission
        success = await mavlink_service.pause_mission()

        if not success:
            raise HTTPException(status_code=500, detail="Failed to pause mission")

        # Update state machine
        if state_machine:
            await state_machine.pause_search_pattern()

        return {"success": True, "message": "Mission paused on flight controller"}

    except HTTPException:
        raise
    except MAVLinkError as e:
        logger.error(f"Failed to pause MAVLink mission: {e}")
        raise HTTPException(status_code=500, detail="Failed to pause mission")


@router.post("/pattern/mavlink/resume")
async def resume_mavlink_mission(
    mavlink_service: MAVLinkService | None = Depends(get_mavlink_service),
    state_machine: StateMachine | None = Depends(get_state_machine),
) -> dict:
    """Resume the paused mission on the flight controller.

    Args:
        mavlink_service: MAVLink service instance
        state_machine: State machine instance

    Returns:
        Resume status response
    """
    try:
        if not mavlink_service:
            raise HTTPException(status_code=503, detail="MAVLink service not available")

        # Resume mission
        success = await mavlink_service.resume_mission()

        if not success:
            raise HTTPException(status_code=500, detail="Failed to resume mission")

        # Update state machine
        if state_machine:
            await state_machine.resume_search_pattern()

        return {"success": True, "message": "Mission resumed on flight controller"}

    except HTTPException:
        raise
    except MAVLinkError as e:
        logger.error(f"Failed to resume MAVLink mission: {e}")
        raise HTTPException(status_code=500, detail="Failed to resume mission")


@router.post("/pattern/mavlink/stop")
async def stop_mavlink_mission(
    mavlink_service: MAVLinkService | None = Depends(get_mavlink_service),
    state_machine: StateMachine | None = Depends(get_state_machine),
) -> dict:
    """Stop the current mission on the flight controller.

    Args:
        mavlink_service: MAVLink service instance
        state_machine: State machine instance

    Returns:
        Stop status response
    """
    try:
        if not mavlink_service:
            raise HTTPException(status_code=503, detail="MAVLink service not available")

        # Stop mission
        success = await mavlink_service.stop_mission()

        if not success:
            raise HTTPException(status_code=500, detail="Failed to stop mission")

        # Update state machine
        if state_machine:
            await state_machine.stop_search_pattern()

        return {"success": True, "message": "Mission stopped on flight controller"}

    except HTTPException:
        raise
    except MAVLinkError as e:
        logger.error(f"Failed to stop MAVLink mission: {e}")
        raise HTTPException(status_code=500, detail="Failed to stop mission")


@router.get("/pattern/mavlink/progress")
async def get_mavlink_mission_progress(
    mavlink_service: MAVLinkService | None = Depends(get_mavlink_service),
) -> dict:
    """Get current mission progress from flight controller.

    Args:
        mavlink_service: MAVLink service instance

    Returns:
        Mission progress response
    """
    try:
        if not mavlink_service:
            raise HTTPException(status_code=503, detail="MAVLink service not available")

        # Get progress
        current_wp, total_wp = mavlink_service.get_mission_progress()

        progress_percent = 0.0
        if total_wp > 0:
            progress_percent = (current_wp / total_wp) * 100

        return {
            "current_waypoint": current_wp,
            "total_waypoints": total_wp,
            "progress_percent": progress_percent,
        }

    except MAVLinkError as e:
        logger.error(f"Failed to get MAVLink mission progress: {e}")
        raise HTTPException(status_code=500, detail="Failed to get mission progress")


@router.get("/pattern/export")
async def export_pattern(
    pattern_id: str | None = None, format: Literal["qgc", "mission_planner", "kml"] = "qgc"
) -> str:
    """Export search pattern as waypoint file.

    Args:
        pattern_id: Optional pattern ID to export
        format: Export format (QGC WPL, Mission Planner, or KML)

    Returns:
        Waypoint file content as string
    """
    try:
        # Get the most recent pattern if no ID specified
        if pattern_id:
            if pattern_id not in active_patterns:
                raise HTTPException(status_code=404, detail="Pattern not found")
            pattern = active_patterns[pattern_id]
        else:
            if not active_patterns:
                raise HTTPException(status_code=404, detail="No active patterns")
            # Get most recent pattern
            pattern = list(active_patterns.values())[-1]

        # Use WaypointExporter for format conversion
        exporter = WaypointExporter()

        if format == "qgc":
            # Get home position from first waypoint or use default
            if pattern.waypoints:
                home_lat = pattern.waypoints[0].latitude
                home_lon = pattern.waypoints[0].longitude
            else:
                home_lat = 0.0
                home_lon = 0.0

            return exporter.export_qgc_wpl(pattern.waypoints, home_lat, home_lon)

        elif format == "mission_planner":
            return exporter.export_mission_planner(pattern.waypoints)

        elif format == "kml":
            pattern_name = f"Search Pattern {pattern.pattern_type.value}"
            return exporter.export_kml(pattern.waypoints, pattern_name)

        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")

    except HTTPException:
        raise
    except PISADException as e:
        logger.error(f"Failed to export pattern: {e}")
        raise HTTPException(status_code=500, detail="Failed to export pattern")
