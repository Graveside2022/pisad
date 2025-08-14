"""Analytics API routes for performance data and reporting."""

import csv
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field

from src.backend.services.mission_replay_service import (
    MissionReplayService,
    PlaybackSpeed,
)
from src.backend.services.performance_analytics import PerformanceAnalytics
from src.backend.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

# Service instances
replay_service = MissionReplayService()
analytics_service = PerformanceAnalytics()


class ExportRequest(BaseModel):
    """Data export request model."""

    mission_id: UUID
    format: str = Field(pattern="^(csv|json)$")
    data_type: str = Field(pattern="^(telemetry|detections|metrics|all)$")
    start_time: datetime | None = None
    end_time: datetime | None = None
    include_sensitive: bool = False


class ReplayControlRequest(BaseModel):
    """Replay control request model."""

    action: str = Field(pattern="^(play|pause|stop|seek)$")
    position: int | None = None
    speed: float | None = None


class MetricsResponse(BaseModel):
    """Performance metrics response model."""

    mission_id: UUID
    detection_metrics: dict[str, Any]
    approach_metrics: dict[str, Any]
    search_metrics: dict[str, Any]
    false_positive_analysis: dict[str, Any]
    environmental_correlation: dict[str, Any]
    baseline_comparison: dict[str, Any]
    overall_score: float
    recommendations: list[str]


@router.get("/metrics", response_model=MetricsResponse)
async def get_performance_metrics(
    mission_id: UUID,
    include_recommendations: bool = True,
) -> MetricsResponse:
    """
    Retrieve performance metrics for a mission.

    Args:
        mission_id: Mission identifier
        include_recommendations: Include improvement recommendations

    Returns:
        Performance metrics data
    """
    try:
        # Load mission data from files
        data_dir = Path("data/missions") / str(mission_id)
        telemetry_file = data_dir / "telemetry.csv"
        detections_file = data_dir / "detections.json"

        if not telemetry_file.exists():
            raise HTTPException(status_code=404, detail="Mission data not found")

        # Load telemetry data
        telemetry_data = []
        with open(telemetry_file) as f:
            reader = csv.DictReader(f)
            telemetry_data = list(reader)

        # Load detection events if available
        detection_events = []
        if detections_file.exists():
            with open(detections_file) as f:
                detection_events = json.load(f)

        # Generate performance report
        report = analytics_service.generate_performance_report(
            mission_id=mission_id,
            telemetry_data=telemetry_data,
            detection_events=detection_events,
            search_area_km2=2.0,  # Default search area
        )

        if not include_recommendations:
            report.recommendations = []

        return MetricsResponse(**report.model_dump())

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Mission data not found")
    except Exception as e:
        logger.error(f"Error retrieving metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/replay/{mission_id}")
async def get_replay_data(mission_id: UUID) -> dict[str, Any]:
    """
    Get replay data for a mission.

    Args:
        mission_id: Mission identifier

    Returns:
        Replay status and timeline data
    """
    try:
        # Load mission data if not already loaded
        if replay_service.mission_id != mission_id:
            data_dir = Path("data/missions") / str(mission_id)
            telemetry_file = data_dir / "telemetry.csv"
            detections_file = data_dir / "detections.json"
            state_file = data_dir / "states.json"

            if not telemetry_file.exists():
                raise HTTPException(status_code=404, detail="Mission data not found")

            success = await replay_service.load_mission_data(
                mission_id=mission_id,
                telemetry_file=telemetry_file,
                detections_file=detections_file if detections_file.exists() else None,
                state_file=state_file if state_file.exists() else None,
            )

            if not success:
                raise HTTPException(status_code=500, detail="Failed to load mission data")

        # Get current status
        status = replay_service.get_status()
        timeline_range = replay_service.get_timeline_range()

        return {
            "status": status,
            "timeline_range": {
                "start": timeline_range[0].isoformat() if timeline_range else None,
                "end": timeline_range[1].isoformat() if timeline_range else None,
            },
        }

    except Exception as e:
        logger.error(f"Error getting replay data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/replay/{mission_id}/control")
async def control_replay(mission_id: UUID, request: ReplayControlRequest) -> dict[str, Any]:
    """
    Control mission replay playback.

    Args:
        mission_id: Mission identifier
        request: Control action request

    Returns:
        Updated replay status
    """
    try:
        # Ensure correct mission is loaded
        if replay_service.mission_id != mission_id:
            raise HTTPException(status_code=400, detail="Mission not loaded for replay")

        # Execute control action
        if request.action == "play":
            await replay_service.play()
        elif request.action == "pause":
            await replay_service.pause()
        elif request.action == "stop":
            await replay_service.stop()
        elif request.action == "seek" and request.position is not None:
            await replay_service.seek(request.position)

        # Set speed if provided
        if request.speed is not None:
            try:
                speed = PlaybackSpeed(request.speed)
                await replay_service.set_speed(speed)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid playback speed")

        return replay_service.get_status()

    except Exception as e:
        logger.error(f"Error controlling replay: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export")
async def export_data(request: ExportRequest) -> Response:
    """
    Export mission data in various formats.

    Args:
        request: Export request parameters

    Returns:
        Exported data file
    """
    try:
        data_dir = Path("data/missions") / str(request.mission_id)
        if not data_dir.exists():
            raise HTTPException(status_code=404, detail="Mission data not found")

        # Collect requested data
        export_data = {}

        if request.data_type in ["telemetry", "all"]:
            telemetry_file = data_dir / "telemetry.csv"
            if telemetry_file.exists():
                with open(telemetry_file) as f:
                    reader = csv.DictReader(f)
                    telemetry = list(reader)
                    export_data["telemetry"] = _filter_by_time_range(
                        telemetry, request.start_time, request.end_time
                    )

        if request.data_type in ["detections", "all"]:
            detections_file = data_dir / "detections.json"
            if detections_file.exists():
                with open(detections_file) as f:
                    detections = json.load(f)
                    export_data["detections"] = _filter_by_time_range(
                        detections, request.start_time, request.end_time
                    )

        if request.data_type in ["metrics", "all"]:
            # Get performance metrics
            try:
                metrics_response = await get_performance_metrics(
                    request.mission_id, include_recommendations=True
                )
                export_data["metrics"] = metrics_response.model_dump()
            except Exception:
                export_data["metrics"] = {}

        # Sanitize data if requested
        if not request.include_sensitive:
            export_data = _sanitize_data(export_data)

        # Format response based on requested format
        if request.format == "json":
            content = json.dumps(export_data, indent=2, default=str)
            media_type = "application/json"
            filename = f"mission_{request.mission_id}_export.json"
        else:  # CSV
            # For CSV, we'll export telemetry as primary data
            if export_data.get("telemetry"):
                output = io.StringIO()
                fieldnames = export_data["telemetry"][0].keys()
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(export_data["telemetry"])
                content = output.getvalue()
            else:
                content = ""
            media_type = "text/csv"
            filename = f"mission_{request.mission_id}_telemetry.csv"

        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reports/{mission_id}")
async def get_mission_report(
    mission_id: UUID, format: str = Query("json", pattern="^(json|pdf)$")
) -> Response:
    """
    Get generated report for a mission.

    Args:
        mission_id: Mission identifier
        format: Report format (json or pdf)

    Returns:
        Mission report
    """
    try:
        # For now, return JSON report
        # PDF generation will be implemented with report_generator service
        metrics = await get_performance_metrics(mission_id)

        report_data = {
            "mission_id": str(mission_id),
            "generated_at": datetime.now().isoformat(),
            "metrics": metrics.model_dump(),
        }

        if format == "json":
            content = json.dumps(report_data, indent=2, default=str)
            media_type = "application/json"
        else:
            # PDF generation placeholder
            raise HTTPException(status_code=501, detail="PDF report generation not yet implemented")

        return Response(content=content, media_type=media_type)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations")
async def get_system_recommendations() -> dict[str, Any]:
    """
    Get system improvement recommendations based on all missions.

    Returns:
        System-wide recommendations
    """
    try:
        # Aggregate recommendations from recent missions
        data_dir = Path("data/missions")
        all_recommendations = []
        mission_count = 0

        if data_dir.exists():
            for mission_dir in data_dir.iterdir():
                if mission_dir.is_dir():
                    try:
                        mission_id = UUID(mission_dir.name)
                        metrics = await get_performance_metrics(mission_id)
                        all_recommendations.extend(metrics.recommendations)
                        mission_count += 1
                    except Exception:
                        continue

        # Deduplicate and prioritize recommendations
        unique_recommendations = list(set(all_recommendations))
        recommendation_counts = {
            rec: all_recommendations.count(rec) for rec in unique_recommendations
        }
        sorted_recommendations = sorted(
            unique_recommendations, key=lambda x: recommendation_counts[x], reverse=True
        )

        return {
            "missions_analyzed": mission_count,
            "top_recommendations": sorted_recommendations[:10],
            "recommendation_frequency": recommendation_counts,
        }

    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _filter_by_time_range(
    data: list[dict[str, Any]],
    start_time: datetime | None,
    end_time: datetime | None,
) -> list[dict[str, Any]]:
    """Filter data by time range."""
    if not start_time and not end_time:
        return data

    filtered = []
    for item in data:
        timestamp_str = item.get("timestamp")
        if not timestamp_str:
            continue

        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            if start_time and timestamp < start_time:
                continue
            if end_time and timestamp > end_time:
                continue
            filtered.append(item)
        except Exception:
            continue

    return filtered


def _sanitize_data(data: dict[str, Any]) -> dict[str, Any]:
    """Remove sensitive information from export data."""
    sensitive_fields = [
        "operator_id",
        "operator_name",
        "api_key",
        "password",
        "secret",
        "token",
    ]

    def sanitize_dict(d: dict[str, Any]) -> dict[str, Any]:
        sanitized = {}
        for key, value in d.items():
            if any(field in key.lower() for field in sensitive_fields):
                continue
            if isinstance(value, dict):
                sanitized[key] = sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    sanitize_dict(item) if isinstance(item, dict) else item for item in value
                ]
            else:
                sanitized[key] = value
        return sanitized

    return sanitize_dict(data)
