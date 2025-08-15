"""Integration tests for analytics data export functionality."""

import csv
import io
import json
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.backend.core.app import create_app

pytestmark = pytest.mark.serial


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def mock_data_path(sample_mission_data, monkeypatch):
    """Mock the data path to use test directory."""
    mission_id, mission_dir = sample_mission_data

    # Mock Path to return our test directory
    original_path = Path

    def mock_path(path_str):
        if path_str == "data/missions":
            return mission_dir.parent
        return original_path(path_str)

    monkeypatch.setattr("src.backend.api.routes.analytics.Path", mock_path)
    return mission_id, mission_dir


@pytest.fixture
def sample_mission_data(tmp_path):
    """Create sample mission data files."""
    mission_id = uuid4()
    mission_dir = tmp_path / "data" / "missions" / str(mission_id)
    mission_dir.mkdir(parents=True)

    # Create telemetry CSV
    telemetry_file = mission_dir / "telemetry.csv"
    base_time = datetime.now()
    telemetry_data = []
    for i in range(20):
        telemetry_data.append(
            {
                "timestamp": (base_time + timedelta(seconds=i)).isoformat(),
                "latitude": 47.6062 + i * 0.0001,
                "longitude": -122.3321 + i * 0.0001,
                "altitude": 100.0 + i,
                "groundspeed": 5.0,
                "airspeed": 5.5,
                "climb_rate": 0.5,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "heading": 90.0,
                "battery_percent": 90.0 - i,
                "battery_voltage": 12.6,
                "battery_current": 5.0,
                "rssi_dbm": -70.0 + i * 0.5,
                "snr_db": 10.0,
                "beacon_detected": i > 10,
                "system_state": "APPROACHING" if i > 10 else "SEARCHING",
                "armed": True,
                "mode": "AUTO",
                "cpu_percent": 45.0,
                "memory_percent": 30.0,
                "temperature_c": 55.0,
            }
        )

    with open(telemetry_file, "w", newline="") as f:
        fieldnames = telemetry_data[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(telemetry_data)

    # Create detections JSON
    detections_file = mission_dir / "detections.json"
    detections = []
    for i in [5, 10, 15]:
        detections.append(
            {
                "timestamp": (base_time + timedelta(seconds=i)).isoformat(),
                "frequency": 121500000,
                "rssi": -65.0 + i,
                "snr": 12.0,
                "confidence": 85.0,
                "location": {
                    "lat": 47.6062 + i * 0.0001,
                    "lon": -122.3321 + i * 0.0001,
                },
                "state": "SEARCHING",
            }
        )

    with open(detections_file, "w") as f:
        json.dump(detections, f)

    # Create states JSON
    states_file = mission_dir / "states.json"
    states = [
        {
            "timestamp": base_time.isoformat(),
            "from_state": "IDLE",
            "to_state": "SEARCHING",
            "trigger": "start",
        },
        {
            "timestamp": (base_time + timedelta(seconds=10)).isoformat(),
            "from_state": "SEARCHING",
            "to_state": "APPROACHING",
            "trigger": "beacon_detected",
        },
    ]

    with open(states_file, "w") as f:
        json.dump(states, f)

    # Create metrics JSON with proper structure
    metrics_file = mission_dir / "metrics.json"
    metrics = {
        "detection_metrics": {
            "total_detections": 3,
            "avg_rssi": -70.0,
            "avg_snr": 12.0,
            "detection_rate": 0.15,
        },
        "approach_metrics": {
            "total_approaches": 1,
            "successful_approaches": 0,
            "avg_approach_time": 120.0,
            "avg_final_distance": 50.0,
        },
        "search_metrics": {
            "total_search_time": 600.0,
            "area_covered": 1000.0,
            "avg_speed": 5.0,
            "pattern_efficiency": 0.85,
        },
    }

    with open(metrics_file, "w") as f:
        json.dump(metrics, f)

    return mission_id, mission_dir


def test_export_json_all_data(client, mock_data_path):
    """Test exporting all data in JSON format."""
    mission_id, mission_dir = mock_data_path

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
    assert len(data["telemetry"]) == 20
    assert len(data["detections"]) == 3


def test_export_csv_telemetry(client, mock_data_path):
    """Test exporting telemetry data in CSV format."""
    mission_id, mission_dir = mock_data_path

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
    assert response.headers["content-type"].startswith("text/csv")

    # Parse CSV response
    csv_content = response.content.decode("utf-8")
    reader = csv.DictReader(io.StringIO(csv_content))
    rows = list(reader)
    assert len(rows) == 20
    assert "latitude" in rows[0]
    assert "longitude" in rows[0]
    assert "rssi_dbm" in rows[0]


def test_export_with_time_filter(client, mock_data_path):
    """Test exporting data with time range filter."""
    mission_id, mission_dir = mock_data_path

    base_time = datetime.now()
    start_time = base_time + timedelta(seconds=5)
    end_time = base_time + timedelta(seconds=15)

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
    # Should have filtered data (approximately 10 seconds worth)
    assert 8 <= len(data["telemetry"]) <= 12


def test_export_detections_only(client, mock_data_path):
    """Test exporting only detection events."""
    mission_id, mission_dir = mock_data_path

    response = client.post(
        "/api/analytics/export",
        json={
            "mission_id": str(mission_id),
            "format": "json",
            "data_type": "detections",
            "include_sensitive": False,
        },
    )

    assert response.status_code == 200
    data = json.loads(response.content)
    assert "detections" in data
    assert "telemetry" not in data
    assert len(data["detections"]) == 3


def test_export_metrics_only(client, mock_data_path):
    """Test exporting only performance metrics."""
    mission_id, mission_dir = mock_data_path

    response = client.post(
        "/api/analytics/export",
        json={
            "mission_id": str(mission_id),
            "format": "json",
            "data_type": "metrics",
            "include_sensitive": False,
        },
    )

    assert response.status_code == 200
    data = json.loads(response.content)
    assert "metrics" in data
    assert "detection_metrics" in data["metrics"]
    assert "approach_metrics" in data["metrics"]
    assert "search_metrics" in data["metrics"]


def test_export_nonexistent_mission(client, monkeypatch):
    """Test exporting data for non-existent mission."""
    # Create a temporary path for the mock
    temp_path = Path("/tmp/test_missions")

    # Mock Path to return our test directory
    original_path = Path

    def mock_path(path_str):
        if path_str == "data/missions":
            return temp_path
        return original_path(path_str)

    monkeypatch.setattr("src.backend.api.routes.analytics.Path", mock_path)

    fake_mission_id = uuid4()

    response = client.post(
        "/api/analytics/export",
        json={
            "mission_id": str(fake_mission_id),
            "format": "json",
            "data_type": "all",
            "include_sensitive": False,
        },
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_export_data_sanitization(client, mock_data_path):
    """Test that sensitive data is removed when include_sensitive=False."""
    mission_id, mission_dir = mock_data_path

    # Add sensitive data to telemetry
    telemetry_file = mission_dir / "telemetry.csv"
    with open(telemetry_file) as f:
        reader = csv.DictReader(f)
        data = list(reader)

    # Add sensitive field
    for row in data:
        row["operator_id"] = "sensitive_id_123"
        row["api_key"] = "secret_key_456"

    # Rewrite file with sensitive data
    with open(telemetry_file, "w", newline="") as f:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    response = client.post(
        "/api/analytics/export",
        json={
            "mission_id": str(mission_id),
            "format": "json",
            "data_type": "telemetry",
            "include_sensitive": False,
        },
    )

    assert response.status_code == 200
    export_data = json.loads(response.content)

    # Check that sensitive fields are removed
    for item in export_data["telemetry"]:
        assert "operator_id" not in item
        assert "api_key" not in item
        # Non-sensitive fields should still be present
        assert "latitude" in item
        assert "longitude" in item


def test_export_invalid_format(client, mock_data_path):
    """Test export with invalid format parameter."""
    mission_id, mission_dir = mock_data_path

    response = client.post(
        "/api/analytics/export",
        json={
            "mission_id": str(mission_id),
            "format": "xml",  # Invalid format
            "data_type": "all",
            "include_sensitive": False,
        },
    )

    assert response.status_code == 422  # Validation error


def test_export_invalid_data_type(client, mock_data_path):
    """Test export with invalid data_type parameter."""
    mission_id, mission_dir = mock_data_path

    response = client.post(
        "/api/analytics/export",
        json={
            "mission_id": str(mission_id),
            "format": "json",
            "data_type": "invalid",  # Invalid data type
            "include_sensitive": False,
        },
    )

    assert response.status_code == 422  # Validation error


def test_export_with_sensitive_data(client, mock_data_path):
    """Test exporting with sensitive data included."""
    mission_id, mission_dir = mock_data_path

    # Add sensitive data
    telemetry_file = mission_dir / "telemetry.csv"
    with open(telemetry_file) as f:
        reader = csv.DictReader(f)
        data = list(reader)

    for row in data:
        row["operator_id"] = "sensitive_id_123"

    with open(telemetry_file, "w", newline="") as f:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    response = client.post(
        "/api/analytics/export",
        json={
            "mission_id": str(mission_id),
            "format": "json",
            "data_type": "telemetry",
            "include_sensitive": True,  # Include sensitive data
        },
    )

    assert response.status_code == 200
    export_data = json.loads(response.content)

    # Check that sensitive fields are included
    assert "operator_id" in export_data["telemetry"][0]
    assert export_data["telemetry"][0]["operator_id"] == "sensitive_id_123"
