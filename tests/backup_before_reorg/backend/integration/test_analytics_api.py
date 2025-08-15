"""Integration tests for analytics API endpoints."""

import csv
import io
import json
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.backend.api.routes.analytics import router
from src.backend.core.app import create_app


@pytest.fixture
def app():
    """Create test app."""
    app = create_app()
    # The router already has prefix="/api/analytics" defined
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def setup_mission_data(tmp_path, monkeypatch):
    """Setup test mission data."""
    mission_id = uuid4()
    data_dir = tmp_path / "data" / "missions" / str(mission_id)
    data_dir.mkdir(parents=True)

    # Create telemetry data
    base_time = datetime.now()
    telemetry_data = []
    for i in range(50):
        telemetry_data.append(
            {
                "timestamp": (base_time + timedelta(seconds=i)).isoformat(),
                "latitude": 47.6062 + i * 0.0001,
                "longitude": -122.3321 + i * 0.0001,
                "altitude": 100.0 + i * 0.5,
                "groundspeed": 5.0,
                "airspeed": 5.5,
                "climb_rate": 0.5,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "heading": 90.0,
                "battery_percent": 90.0 - i * 0.5,
                "battery_voltage": 12.6,
                "battery_current": 5.0,
                "rssi_dbm": -80.0 + i * 0.3,
                "snr_db": 10.0 + i * 0.1,
                "beacon_detected": i > 25,
                "system_state": "APPROACHING" if i > 25 else "SEARCHING",
                "armed": True,
                "mode": "AUTO",
                "cpu_percent": 45.0,
                "memory_percent": 30.0,
                "temperature_c": 55.0,
            }
        )

    # Write telemetry CSV
    telemetry_file = data_dir / "telemetry.csv"
    with open(telemetry_file, "w", newline="") as f:
        fieldnames = telemetry_data[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(telemetry_data)

    # Create detection events
    detections = []
    for i in [10, 20, 30, 40]:
        detections.append(
            {
                "id": str(uuid4()),
                "timestamp": (base_time + timedelta(seconds=i)).isoformat(),
                "frequency": 121500000,
                "rssi": -70.0 + i * 0.5,
                "snr": 12.0 + i * 0.1,
                "confidence": 80.0 + i * 0.2,
                "location": {
                    "lat": 47.6062 + i * 0.0001,
                    "lon": -122.3321 + i * 0.0001,
                },
                "state": "SEARCHING" if i < 25 else "APPROACHING",
            }
        )

    detections_file = data_dir / "detections.json"
    with open(detections_file, "w") as f:
        json.dump(detections, f)

    # Create state transitions
    states = [
        {
            "timestamp": base_time.isoformat(),
            "from_state": "IDLE",
            "to_state": "SEARCHING",
            "trigger": "start",
        },
        {
            "timestamp": (base_time + timedelta(seconds=25)).isoformat(),
            "from_state": "SEARCHING",
            "to_state": "APPROACHING",
            "trigger": "beacon_detected",
        },
        {
            "timestamp": (base_time + timedelta(seconds=45)).isoformat(),
            "from_state": "APPROACHING",
            "to_state": "HOVERING",
            "trigger": "target_reached",
        },
    ]

    states_file = data_dir / "states.json"
    with open(states_file, "w") as f:
        json.dump(states, f)

    # Patch Path constructor to return temp paths for data/missions
    monkeypatch.chdir(tmp_path)

    return mission_id, data_dir


def test_get_performance_metrics(app, tmp_path, monkeypatch):
    """Test retrieving performance metrics."""
    # Create test data in a location the API expects
    mission_id = uuid4()

    # Mock os.getcwd to return our temp directory
    import os

    monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))

    # Create the data structure that the API expects
    data_dir = tmp_path / "data" / "missions" / str(mission_id)
    data_dir.mkdir(parents=True)

    # Create minimal test files
    telemetry_file = data_dir / "telemetry.csv"
    telemetry_file.write_text(
        "timestamp,latitude,longitude\n2024-01-01T00:00:00,47.6062,-122.3321\n"
    )

    detections_file = data_dir / "detections.json"
    detections_file.write_text('[{"id": "1", "timestamp": "2024-01-01T00:00:00"}]')

    # Mock the Path constructor to use our tmp directory for data/missions
    from pathlib import Path as OrigPath

    import src.backend.api.routes.analytics as analytics_module

    class MockPath(OrigPath):
        def __new__(cls, *args, **kwargs):
            if len(args) == 1 and args[0] == "data/missions":
                return OrigPath.__new__(cls, tmp_path / "data" / "missions")
            return OrigPath.__new__(cls, *args, **kwargs)

    monkeypatch.setattr(analytics_module, "Path", MockPath)

    # Create client after mocking
    with TestClient(app) as client:
        # Debug: check available routes
        routes = []
        for route in app.routes:
            if hasattr(route, "path"):
                routes.append(route.path)
        print(f"Available routes: {[r for r in routes if 'analytics' in r]}")

        response = client.get(f"/api/analytics/metrics?mission_id={mission_id}")

    print(f"Response status: {response.status_code}")
    if response.status_code != 200:
        print(f"Response body: {response.text}")
        import traceback

        traceback.print_exc()
    assert response.status_code == 200
    data = response.json()
    assert "mission_id" in data
    assert str(data["mission_id"]) == str(mission_id)
    assert "detection_metrics" in data
    assert "approach_metrics" in data
    assert "search_metrics" in data
    assert "overall_score" in data
    assert 0 <= data["overall_score"] <= 100


def test_get_performance_metrics_with_recommendations(app, setup_mission_data, monkeypatch):
    """Test retrieving metrics with recommendations."""
    mission_id, _ = setup_mission_data

    response = client.get(
        f"/api/analytics/metrics?mission_id={mission_id}&include_recommendations=true"
    )

    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)


def test_get_performance_metrics_without_recommendations(client, setup_mission_data):
    """Test retrieving metrics without recommendations."""
    mission_id, _ = setup_mission_data

    response = client.get(
        f"/api/analytics/metrics?mission_id={mission_id}&include_recommendations=false"
    )

    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert data["recommendations"] == []


def test_get_performance_metrics_not_found(client):
    """Test retrieving metrics for non-existent mission."""
    fake_id = uuid4()

    response = client.get(f"/api/analytics/metrics?mission_id={fake_id}")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_replay_data(client, setup_mission_data):
    """Test getting replay data."""
    mission_id, _ = setup_mission_data

    response = client.get(f"/api/analytics/replay/{mission_id}")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"]["mission_id"] == str(mission_id)
    assert "state" in data["status"]
    assert "position" in data["status"]
    assert "total" in data["status"]
    assert "timeline_range" in data
    assert "start" in data["timeline_range"]
    assert "end" in data["timeline_range"]


def test_get_replay_data_not_found(client):
    """Test getting replay data for non-existent mission."""
    fake_id = uuid4()

    response = client.get(f"/api/analytics/replay/{fake_id}")

    assert response.status_code == 404


def test_control_replay_play(client, setup_mission_data):
    """Test replay play control."""
    mission_id, _ = setup_mission_data

    # Load replay data first
    client.get(f"/api/analytics/replay/{mission_id}")

    # Send play command
    response = client.post(
        f"/api/analytics/replay/{mission_id}/control",
        json={"action": "play"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "state" in data
    assert "position" in data


def test_control_replay_pause(client, setup_mission_data):
    """Test replay pause control."""
    mission_id, _ = setup_mission_data

    client.get(f"/api/analytics/replay/{mission_id}")

    response = client.post(
        f"/api/analytics/replay/{mission_id}/control",
        json={"action": "pause"},
    )

    assert response.status_code == 200


def test_control_replay_seek(client, setup_mission_data):
    """Test replay seek control."""
    mission_id, _ = setup_mission_data

    client.get(f"/api/analytics/replay/{mission_id}")

    response = client.post(
        f"/api/analytics/replay/{mission_id}/control",
        json={"action": "seek", "position": 10},
    )

    assert response.status_code == 200
    data = response.json()
    assert "position" in data


def test_control_replay_speed(client, setup_mission_data):
    """Test replay speed control."""
    mission_id, _ = setup_mission_data

    client.get(f"/api/analytics/replay/{mission_id}")

    response = client.post(
        f"/api/analytics/replay/{mission_id}/control",
        json={"action": "play", "speed": 2.0},
    )

    assert response.status_code == 200
    data = response.json()
    assert "speed" in data


def test_export_data_json_all(client, setup_mission_data):
    """Test exporting all data in JSON format."""
    mission_id, _ = setup_mission_data

    response = client.post(
        "/api/analytics/export",
        json={
            "mission_id": str(mission_id),
            "format": "json",
            "data_type": "all",
            "include_sensitive": False,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"

    data = json.loads(response.content)
    assert "telemetry" in data
    assert "detections" in data
    assert len(data["telemetry"]) == 50
    assert len(data["detections"]) == 4


def test_export_data_csv_telemetry(client, setup_mission_data):
    """Test exporting telemetry in CSV format."""
    mission_id, _ = setup_mission_data

    response = client.post(
        "/api/analytics/export",
        json={
            "mission_id": str(mission_id),
            "format": "csv",
            "data_type": "telemetry",
            "include_sensitive": False,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv"

    # Parse CSV
    csv_content = response.content.decode("utf-8")
    reader = csv.DictReader(io.StringIO(csv_content))
    rows = list(reader)
    assert len(rows) == 50


def test_export_data_with_time_filter(client, setup_mission_data):
    """Test exporting data with time range filter."""
    mission_id, _ = setup_mission_data

    base_time = datetime.now()
    start_time = base_time + timedelta(seconds=10)
    end_time = base_time + timedelta(seconds=30)

    response = client.post(
        "/api/analytics/export",
        json={
            "mission_id": str(mission_id),
            "format": "json",
            "data_type": "telemetry",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "include_sensitive": False,
        },
    )

    assert response.status_code == 200
    data = json.loads(response.content)
    assert "telemetry" in data
    # Should have filtered data (approximately 20 seconds worth)
    assert 15 <= len(data["telemetry"]) <= 25


def test_get_mission_report_json(client, setup_mission_data):
    """Test getting mission report in JSON format."""
    mission_id, _ = setup_mission_data

    response = client.get(f"/api/analytics/reports/{mission_id}?format=json")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"

    data = json.loads(response.content)
    assert "mission_id" in data
    assert "generated_at" in data
    assert "metrics" in data


def test_get_mission_report_pdf_not_implemented(client, setup_mission_data):
    """Test that PDF report generation returns not implemented."""
    mission_id, _ = setup_mission_data

    response = client.get(f"/api/analytics/reports/{mission_id}?format=pdf")

    assert response.status_code == 501
    assert "not yet implemented" in response.json()["detail"].lower()


def test_get_system_recommendations_single_mission(client, setup_mission_data):
    """Test getting system recommendations with single mission."""
    mission_id, _ = setup_mission_data

    response = client.get("/api/analytics/recommendations")

    assert response.status_code == 200
    data = response.json()
    assert "missions_analyzed" in data
    assert data["missions_analyzed"] == 1
    assert "top_recommendations" in data
    assert isinstance(data["top_recommendations"], list)
    assert "recommendation_frequency" in data


def test_get_system_recommendations_multiple_missions(client, tmp_path, monkeypatch):
    """Test getting system recommendations with multiple missions."""
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    # Create multiple mission directories
    missions_dir = tmp_path / "data" / "missions"
    missions_dir.mkdir(parents=True)

    for i in range(3):
        mission_id = uuid4()
        mission_dir = missions_dir / str(mission_id)
        mission_dir.mkdir()

        # Create minimal telemetry file
        telemetry_file = mission_dir / "telemetry.csv"
        with open(telemetry_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "latitude",
                    "longitude",
                    "altitude",
                    "rssi_dbm",
                    "beacon_detected",
                    "system_state",
                ],
            )
            writer.writeheader()
            for j in range(10):
                writer.writerow(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "latitude": 47.6062 + j * 0.0001,
                        "longitude": -122.3321,
                        "altitude": 100,
                        "rssi_dbm": -70,
                        "beacon_detected": False,
                        "system_state": "SEARCHING",
                    }
                )

    response = client.get("/api/analytics/recommendations")

    assert response.status_code == 200
    data = response.json()
    assert data["missions_analyzed"] == 3


def test_get_system_recommendations_no_missions(client, tmp_path, monkeypatch):
    """Test getting recommendations with no missions."""
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    response = client.get("/api/analytics/recommendations")

    assert response.status_code == 200
    data = response.json()
    assert data["missions_analyzed"] == 0
    assert data["top_recommendations"] == []


def test_export_invalid_format(client, setup_mission_data):
    """Test export with invalid format."""
    mission_id, _ = setup_mission_data

    response = client.post(
        "/api/analytics/export",
        json={
            "mission_id": str(mission_id),
            "format": "xml",  # Invalid
            "data_type": "all",
            "include_sensitive": False,
        },
    )

    assert response.status_code == 422


def test_export_invalid_data_type(client, setup_mission_data):
    """Test export with invalid data type."""
    mission_id, _ = setup_mission_data

    response = client.post(
        "/api/analytics/export",
        json={
            "mission_id": str(mission_id),
            "format": "json",
            "data_type": "invalid",  # Invalid
            "include_sensitive": False,
        },
    )

    assert response.status_code == 422


def test_control_replay_invalid_action(client, setup_mission_data):
    """Test replay control with invalid action."""
    mission_id, _ = setup_mission_data

    client.get(f"/api/analytics/replay/{mission_id}")

    response = client.post(
        f"/api/analytics/replay/{mission_id}/control",
        json={"action": "invalid"},  # Invalid action
    )

    assert response.status_code == 422


def test_control_replay_mission_not_loaded(client):
    """Test replay control when mission not loaded."""
    fake_id = uuid4()

    response = client.post(
        f"/api/analytics/replay/{fake_id}/control",
        json={"action": "play"},
    )

    assert response.status_code == 400
    assert "not loaded" in response.json()["detail"].lower()
