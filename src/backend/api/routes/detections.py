"""
Signal detection API routes.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter()

# Mock detection data store (in production, this would come from a database)
mock_detections: list[dict[str, Any]] = []


def generate_mock_detections() -> None:
    """Generate mock detection data for testing."""
    if not mock_detections:
        base_time = datetime.now(UTC)
        for i in range(10):
            detection = {
                "id": str(uuid4()),
                "timestamp": (base_time - timedelta(minutes=i * 2)).isoformat(),
                "frequency": 433920000 + (i * 1000),  # Vary frequency slightly
                "rssi": -65 - (i * 2),  # Vary RSSI
                "snr": 15 - (i * 0.5),  # Vary SNR
                "confidence": max(30, 95 - (i * 5)),  # Vary confidence
                "location": (
                    {
                        "latitude": 37.7749 + (i * 0.001),
                        "longitude": -122.4194 - (i * 0.001),
                        "altitude": 100 + (i * 10),
                    }
                    if i % 2 == 0
                    else None
                ),  # Only some have location
                "state": "DETECTING" if i < 3 else "SEARCHING",
            }
            mock_detections.append(detection)


@router.get("/detections")
async def get_detections(
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    frequency_min: float | None = Query(default=None, description="Minimum frequency in Hz"),
    frequency_max: float | None = Query(default=None, description="Maximum frequency in Hz"),
    confidence_min: float | None = Query(default=None, ge=0, le=100),
) -> dict[str, Any]:
    """
    Get recent signal detections with optional filtering.

    Args:
        limit: Maximum number of detections to return (1-100)
        offset: Number of detections to skip for pagination
        frequency_min: Minimum frequency filter in Hz
        frequency_max: Maximum frequency filter in Hz
        confidence_min: Minimum confidence filter (0-100)

    Returns:
        List of signal detections matching the criteria
    """
    try:
        # Generate mock data if needed
        if not mock_detections:
            generate_mock_detections()

        # Apply filters
        filtered_detections = mock_detections.copy()

        if frequency_min is not None:
            filtered_detections = [
                d for d in filtered_detections if d["frequency"] >= frequency_min
            ]

        if frequency_max is not None:
            filtered_detections = [
                d for d in filtered_detections if d["frequency"] <= frequency_max
            ]

        if confidence_min is not None:
            filtered_detections = [
                d for d in filtered_detections if d["confidence"] >= confidence_min
            ]

        # Apply pagination
        total_count = len(filtered_detections)
        paginated_detections = filtered_detections[offset : offset + limit]

        return {
            "detections": paginated_detections,
            "count": len(paginated_detections),
            "total": total_count,
            "offset": offset,
            "limit": limit,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get detections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/detections/{detection_id}")
async def get_detection(detection_id: str) -> dict[str, Any]:
    """
    Get a specific detection by ID.

    Args:
        detection_id: UUID of the detection

    Returns:
        Detection details

    Raises:
        404: Detection not found
    """
    try:
        # Generate mock data if needed
        if not mock_detections:
            generate_mock_detections()

        # Find detection
        for detection in mock_detections:
            if detection["id"] == detection_id:
                return detection

        raise HTTPException(status_code=404, detail="Detection not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get detection {detection_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
