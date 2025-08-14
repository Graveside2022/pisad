"""Testing API routes for PiSAD system.

Provides endpoints for retrieving test results and managing test data.
"""

import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.models.schemas import BeaconConfiguration
from backend.services.field_test_service import FieldTestConfig, FieldTestService
from backend.utils.test_logger import TestLogger, TestType

router = APIRouter(prefix="/api/testing", tags=["testing"])

# Initialize test logger with default database
test_logger = TestLogger("data/test_results.db")

# Field test service will be initialized in main app
field_test_service: FieldTestService | None = None


class FieldTestStartRequest(BaseModel):
    """Request model for starting field test."""

    test_name: str
    test_type: str
    beacon_config: dict[str, Any]
    environmental_conditions: dict[str, Any]


class FieldTestStartResponse(BaseModel):
    """Response model for field test start."""

    test_id: str
    status: str
    start_time: str
    checklist_status: str


def get_field_test_service() -> FieldTestService:
    """Get field test service instance."""
    if not field_test_service:
        raise HTTPException(status_code=500, detail="Field test service not initialized")
    return field_test_service


@router.get("/results")
async def get_test_results(
    limit: int = Query(10, ge=1, le=100), test_type: str | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Get recent test results.

    Args:
        limit: Maximum number of test runs to return (1-100)
        test_type: Optional filter by test type

    Returns:
        List of test run summaries with results
    """
    try:
        # Parse test type if provided
        type_filter = None
        if test_type:
            try:
                type_filter = TestType(test_type.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid test type: {test_type}")

        # Get recent test runs
        runs = test_logger.get_recent_runs(limit=limit, test_type=type_filter)

        # Format response
        test_runs = []
        for run in runs:
            test_runs.append(
                {
                    "id": run["run_id"],
                    "timestamp": run["timestamp"],
                    "test_type": run["test_type"],
                    "passed": run["passed"],
                    "failed": run["failed"],
                    "duration_ms": run["total_duration_ms"],
                    "configuration": run.get("system_config", {}),
                    "results": [],  # Summary only, detailed results via /results/{id}
                }
            )

        return {"test_runs": test_runs}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve test results: {e!s}")


@router.get("/results/{run_id}")
async def get_test_run_details(run_id: str) -> dict[str, Any]:
    """Get detailed results for a specific test run.

    Args:
        run_id: Test run UUID

    Returns:
        Detailed test run data including all individual test results
    """
    try:
        test_run = test_logger.get_test_run(run_id)

        if not test_run:
            raise HTTPException(status_code=404, detail=f"Test run {run_id} not found")

        return test_run

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve test run: {e!s}")


@router.get("/statistics")
async def get_test_statistics() -> dict[str, Any]:
    """Get overall test statistics.

    Returns:
        Statistics including pass rates, average durations, etc.
    """
    try:
        stats = test_logger.get_statistics()
        return stats

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve statistics: {e!s}")


@router.post("/field-test/start", response_model=FieldTestStartResponse)
async def start_field_test(
    request: FieldTestStartRequest, service: FieldTestService = Depends(get_field_test_service)
):
    """Start a new field test.

    Args:
        request: Field test configuration
        service: Field test service instance

    Returns:
        Test ID and initial status

    Raises:
        HTTPException: If preflight checks fail or test cannot be started
    """
    try:
        # Validate preflight checklist first
        checklist = await service.validate_preflight_checklist()
        if not all(checklist.values()):
            failed_checks = [k for k, v in checklist.items() if not v]
            raise HTTPException(
                status_code=400,
                detail=f"Preflight checks failed: {', '.join(failed_checks)}",
            )

        # Create field test configuration
        beacon_config = BeaconConfiguration(**request.beacon_config)
        config = FieldTestConfig(
            test_name=request.test_name,
            test_type=request.test_type,  # type: ignore
            beacon_config=beacon_config,
            environmental_conditions=request.environmental_conditions,
        )

        # Start the test
        status = await service.start_field_test(config)

        return FieldTestStartResponse(
            test_id=status.test_id,
            status=status.status,
            start_time=status.start_time.isoformat(),
            checklist_status="passed",
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start field test: {e!s}")


@router.get("/field-test/{test_id}/status")
async def get_field_test_status(
    test_id: str, service: FieldTestService = Depends(get_field_test_service)
):
    """Get current status of a field test.

    Args:
        test_id: Test identifier
        service: Field test service instance

    Returns:
        Current test status and progress

    Raises:
        HTTPException: If test not found
    """
    try:
        status = await service.get_test_status(test_id)
        if not status:
            raise HTTPException(status_code=404, detail=f"Test {test_id} not found")

        return {
            "test_id": status.test_id,
            "phase": status.phase,
            "status": status.status,
            "start_time": status.start_time.isoformat(),
            "current_iteration": status.current_iteration,
            "total_iterations": status.total_iterations,
            "current_distance_m": status.current_distance_m,
            "current_rssi_dbm": status.current_rssi_dbm,
            "beacon_detected": status.beacon_detected,
            "error_message": status.error_message,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get test status: {e!s}")


@router.get("/field-test/{test_id}/metrics")
async def get_field_test_metrics(
    test_id: str, service: FieldTestService = Depends(get_field_test_service)
):
    """Get metrics for a field test.

    Args:
        test_id: Test identifier
        service: Field test service instance

    Returns:
        Test metrics and telemetry file path

    Raises:
        HTTPException: If test not found or metrics unavailable
    """
    try:
        metrics = await service.get_test_metrics(test_id)
        if not metrics:
            raise HTTPException(status_code=404, detail=f"Metrics for test {test_id} not found")

        # Get test status to determine if still running
        status = await service.get_test_status(test_id)
        test_status = status.status if status else "completed"

        return {
            "test_id": test_id,
            "metrics": {
                "beacon_power_dbm": metrics.beacon_power_dbm,
                "detection_range_m": metrics.detection_range_m,
                "approach_accuracy_m": metrics.approach_accuracy_m,
                "time_to_locate_s": metrics.time_to_locate_s,
                "transition_latency_ms": metrics.transition_latency_ms,
                "max_rssi_dbm": metrics.max_rssi_dbm,
                "min_rssi_dbm": metrics.min_rssi_dbm,
                "avg_rssi_dbm": metrics.avg_rssi_dbm,
                "signal_loss_count": metrics.signal_loss_count,
                "environmental_conditions": metrics.environmental_conditions,
                "safety_events": metrics.safety_events,
                "success": metrics.success,
            },
            "telemetry_file": f"data/field_tests/{test_id}_telemetry.log",
            "status": test_status,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get test metrics: {e!s}")


@router.post("/field-test/{test_id}/export")
async def export_field_test_data(
    test_id: str,
    format: str = Query("csv", regex="^(csv|json)$"),
    service: FieldTestService = Depends(get_field_test_service),
):
    """Export field test data in specified format.

    Args:
        test_id: Test identifier
        format: Export format (csv or json)
        service: Field test service instance

    Returns:
        Path to exported file

    Raises:
        HTTPException: If test not found or export fails
    """
    try:
        export_path = await service.export_test_data(test_id, format)
        if not export_path:
            raise HTTPException(status_code=404, detail=f"Unable to export test {test_id}")

        return {"export_path": str(export_path), "format": format}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export test data: {e!s}")


@router.get("/field-test/preflight")
async def get_preflight_checklist(service: FieldTestService = Depends(get_field_test_service)):
    """Get current preflight checklist status.

    Args:
        service: Field test service instance

    Returns:
        Dictionary of checklist items and their status
    """
    try:
        checklist = await service.validate_preflight_checklist()
        all_passed = all(checklist.values())

        return {
            "checklist": checklist,
            "all_passed": all_passed,
            "ready_for_test": all_passed,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get preflight status: {e!s}")


@router.post("/field-test/validate-beacon")
async def validate_beacon_signal(
    beacon_config: BeaconConfiguration,
    service: FieldTestService = Depends(get_field_test_service),
):
    """Validate beacon signal output using SDR spectrum analyzer.

    Args:
        beacon_config: Beacon configuration to validate
        service: Field test service instance

    Returns:
        Validation results including frequency and power measurements

    Raises:
        HTTPException: If validation fails
    """
    try:
        validation_results = await service.validate_beacon_signal(beacon_config)

        return {
            "validation_passed": validation_results["validation_passed"],
            "frequency_match": validation_results["frequency_match"],
            "power_level_match": validation_results["power_level_match"],
            "modulation_match": validation_results["modulation_match"],
            "measurements": {
                "measured_frequency_hz": validation_results["measured_frequency_hz"],
                "measured_power_dbm": validation_results["measured_power_dbm"],
                "frequency_error_hz": validation_results["frequency_error_hz"],
                "power_error_dbm": validation_results["power_error_dbm"],
            },
            "spectrum_data": validation_results.get("spectrum_data", []),
            "error": validation_results.get("error"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to validate beacon signal: {e!s}")
