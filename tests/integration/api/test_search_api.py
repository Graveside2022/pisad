"""Integration tests for search pattern API endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.backend.core.app import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def center_boundary_request():
    """Sample center-radius boundary request."""
    return {
        "pattern": "expanding_square",
        "spacing": 75.0,
        "velocity": 7.0,
        "bounds": {
            "type": "center_radius",
            "center": {"lat": 37.7749, "lon": -122.4194},
            "radius": 500,
        },
        "altitude": 50.0,
    }


@pytest.fixture
def corner_boundary_request():
    """Sample corner boundary request."""
    return {
        "pattern": "lawnmower",
        "spacing": 100.0,
        "velocity": 10.0,
        "bounds": {
            "type": "corners",
            "corners": [
                {"lat": 37.770, "lon": -122.425},
                {"lat": 37.780, "lon": -122.425},
                {"lat": 37.780, "lon": -122.415},
                {"lat": 37.770, "lon": -122.415},
            ],
        },
        "altitude": 75.0,
    }


class TestSearchPatternAPI:
    """Test search pattern API endpoints."""

    def test_create_pattern_with_center_boundary(self, client, center_boundary_request):
        """Test creating pattern with center-radius boundary."""
        response = client.post("/api/search/pattern", json=center_boundary_request)

        assert response.status_code == 200
        data = response.json()
        assert "pattern_id" in data
        assert data["waypoint_count"] > 0
        assert data["estimated_duration"] > 0
        assert data["total_distance"] > 0

    def test_create_pattern_with_corner_boundary(self, client, corner_boundary_request):
        """Test creating pattern with corner boundary."""
        response = client.post("/api/search/pattern", json=corner_boundary_request)

        assert response.status_code == 200
        data = response.json()
        assert "pattern_id" in data
        assert data["waypoint_count"] > 0

    def test_create_pattern_invalid_spacing(self, client, center_boundary_request):
        """Test creating pattern with invalid spacing."""
        center_boundary_request["spacing"] = 150.0  # Too large
        response = client.post("/api/search/pattern", json=center_boundary_request)

        assert response.status_code == 422  # Validation error

    def test_create_pattern_invalid_velocity(self, client, center_boundary_request):
        """Test creating pattern with invalid velocity."""
        center_boundary_request["velocity"] = 15.0  # Too fast
        response = client.post("/api/search/pattern", json=center_boundary_request)

        assert response.status_code == 422  # Validation error

    def test_get_pattern_preview(self, client, center_boundary_request):
        """Test getting pattern preview."""
        # Create pattern first
        create_response = client.post("/api/search/pattern", json=center_boundary_request)
        assert create_response.status_code == 200
        pattern_id = create_response.json()["pattern_id"]

        # Get preview
        response = client.get(f"/api/search/pattern/preview?pattern_id={pattern_id}")
        assert response.status_code == 200

        data = response.json()
        assert "waypoints" in data
        assert len(data["waypoints"]) > 0
        assert "total_distance" in data
        assert "estimated_time" in data

        # Check waypoint structure
        waypoint = data["waypoints"][0]
        assert "index" in waypoint
        assert "lat" in waypoint
        assert "lon" in waypoint
        assert "alt" in waypoint

    def test_get_pattern_preview_no_pattern(self, client):
        """Test getting preview when no pattern exists."""
        response = client.get("/api/search/pattern/preview")
        assert response.status_code == 404

    def test_get_pattern_status(self, client, center_boundary_request):
        """Test getting pattern status."""
        # Create pattern first
        create_response = client.post("/api/search/pattern", json=center_boundary_request)
        assert create_response.status_code == 200
        pattern_id = create_response.json()["pattern_id"]

        # Get status
        response = client.get(f"/api/search/pattern/status?pattern_id={pattern_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["pattern_id"] == pattern_id
        assert data["state"] == "IDLE"
        assert data["progress_percent"] == 0.0
        assert data["completed_waypoints"] == 0
        assert data["total_waypoints"] > 0

    def test_control_pattern_pause(self, client, center_boundary_request):
        """Test pausing pattern execution."""
        # Create pattern first
        create_response = client.post("/api/search/pattern", json=center_boundary_request)
        assert create_response.status_code == 200
        pattern_id = create_response.json()["pattern_id"]

        # Can't pause a pattern that's not executing
        response = client.post(
            f"/api/search/pattern/control?pattern_id={pattern_id}", json={"action": "pause"}
        )
        assert response.status_code == 400

    def test_control_pattern_stop(self, client, center_boundary_request):
        """Test stopping pattern execution."""
        # Create pattern first
        create_response = client.post("/api/search/pattern", json=center_boundary_request)
        assert create_response.status_code == 200
        pattern_id = create_response.json()["pattern_id"]

        # Stop pattern
        response = client.post(
            f"/api/search/pattern/control?pattern_id={pattern_id}", json={"action": "stop"}
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["new_state"] == "IDLE"

    def test_export_pattern_qgc_format(self, client, center_boundary_request):
        """Test exporting pattern in QGC format."""
        # Create pattern first
        create_response = client.post("/api/search/pattern", json=center_boundary_request)
        assert create_response.status_code == 200
        pattern_id = create_response.json()["pattern_id"]

        # Export as QGC
        response = client.get(f"/api/search/pattern/export?pattern_id={pattern_id}&format=qgc")
        assert response.status_code == 200

        content = response.text
        assert content.startswith("QGC WPL 110")
        lines = content.split("\n")
        assert len(lines) > 1  # Header + waypoints

    def test_export_pattern_mission_planner_format(self, client, corner_boundary_request):
        """Test exporting pattern in Mission Planner format."""
        # Create pattern first
        create_response = client.post("/api/search/pattern", json=corner_boundary_request)
        assert create_response.status_code == 200
        pattern_id = create_response.json()["pattern_id"]

        # Export as Mission Planner
        response = client.get(
            f"/api/search/pattern/export?pattern_id={pattern_id}&format=mission_planner"
        )
        assert response.status_code == 200

        content = response.text
        lines = content.split("\n")
        assert len(lines) > 0

        # Check format
        first_line = lines[0]
        parts = first_line.split(",")
        assert len(parts) == 4  # WP_NUMBER,LAT,LON,ALT

    def test_all_pattern_types(self, client):
        """Test creating all pattern types."""
        patterns = ["expanding_square", "spiral", "lawnmower"]

        for pattern_type in patterns:
            request = {
                "pattern": pattern_type,
                "spacing": 75.0,
                "velocity": 7.0,
                "bounds": {
                    "type": "center_radius",
                    "center": {"lat": 37.7749, "lon": -122.4194},
                    "radius": 300,
                },
            }

            response = client.post("/api/search/pattern", json=request)
            assert response.status_code == 200, f"Failed to create {pattern_type} pattern"

            data = response.json()
            assert data["waypoint_count"] > 0, f"{pattern_type} has no waypoints"
