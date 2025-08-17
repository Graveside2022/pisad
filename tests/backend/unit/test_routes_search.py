"""
Test suite for search pattern API routes.

Validates search pattern generation, persistence, and control functionality
per PRD-FR2: expanding square search patterns with configurable velocities.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.backend.core.app import create_app


@pytest.fixture
def client():
    """Create test client for search routes."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def mock_services():
    """Mock external service dependencies."""
    with patch("src.backend.core.dependencies.get_service_manager") as mock_manager:
        # Mock service manager
        mock_service_manager = Mock()
        mock_service_manager.initialize_services = AsyncMock()
        mock_service_manager.shutdown_services = AsyncMock()
        mock_manager.return_value = mock_service_manager

        # Mock individual services
        mock_search_generator = Mock()
        mock_state_machine = Mock()
        mock_mavlink_service = Mock()

        mock_service_manager.search_pattern_generator = mock_search_generator
        mock_service_manager.state_machine = mock_state_machine
        mock_service_manager.mavlink_service = mock_mavlink_service

        yield {
            "search_generator": mock_search_generator,
            "state_machine": mock_state_machine,
            "mavlink_service": mock_mavlink_service,
            "service_manager": mock_service_manager,
        }


class TestSearchPatternCreation:
    """Test search pattern creation endpoints per PRD-FR2."""

    def test_create_expanding_square_pattern_success(self, client, mock_services):
        """Test successful creation of expanding square search pattern per PRD-FR2."""
        # TDD RED PHASE: Test real API endpoint with actual request format

        pattern_request = {
            "pattern": "expanding_square",
            "spacing": 100,  # 50-100m per PRD requirements
            "velocity": 8.0,  # 5-10 m/s per PRD-FR2
            "bounds": {
                "type": "center_radius",
                "center": {"lat": 40.7128, "lon": -74.0060},
                "radius": 1000,
            },
            "altitude": 50.0,
        }

        response = client.post("/api/search/pattern", json=pattern_request)

        # Test should fail initially in RED phase - endpoint needs to exist
        assert response.status_code == 200  # Should be 201 when fully implemented
        data = response.json()
        assert "pattern_id" in data
        assert "waypoint_count" in data
        assert "estimated_duration" in data
        assert "total_distance" in data

    def test_create_pattern_invalid_velocity_range(self, client, mock_services):
        """Test pattern creation fails with velocity outside 5-10 m/s range per PRD-FR2."""
        # TDD RED PHASE: Test velocity validation per PRD requirements

        pattern_request = {
            "pattern": "expanding_square",
            "spacing": 100,
            "velocity": 15.0,  # Exceeds PRD maximum of 10 m/s
            "bounds": {
                "type": "center_radius",
                "center": {"lat": 40.7128, "lon": -74.0060},
                "radius": 1000,
            },
            "altitude": 50.0,
        }

        response = client.post("/api/search/pattern", json=pattern_request)

        # Should fail validation
        assert response.status_code == 422
        data = response.json()
        assert "velocity" in str(data["detail"])


class TestSearchPatternControl:
    """Test search pattern control endpoints per PRD requirements."""

    def test_get_pattern_status_success(self, client, mock_services):
        """Test getting pattern status for existing pattern."""
        # First create a pattern to get its ID
        pattern_request = {
            "pattern": "expanding_square",
            "spacing": 75,
            "velocity": 6.0,
            "bounds": {
                "type": "center_radius",
                "center": {"lat": 40.7589, "lon": -73.9851},
                "radius": 500,
            },
            "altitude": 100.0,
        }

        create_response = client.post("/api/search/pattern", json=pattern_request)
        assert create_response.status_code == 200
        pattern_id = create_response.json()["pattern_id"]

        # Test getting status
        response = client.get(f"/api/search/pattern/status?pattern_id={pattern_id}")

        assert response.status_code == 200
        data = response.json()
        assert "pattern_id" in data
        assert "state" in data
        assert "progress_percent" in data

    def test_pattern_control_start_stop(self, client, mock_services):
        """Test pattern start/stop control per operator requirements."""
        # Create pattern first
        pattern_request = {
            "pattern": "lawnmower",
            "spacing": 50,  # Minimum spacing
            "velocity": 5.0,  # Minimum velocity per PRD-FR2
            "bounds": {
                "type": "center_radius",
                "center": {"lat": 35.6762, "lon": 139.6503},
                "radius": 800,
            },
            "altitude": 75.0,
        }

        create_response = client.post("/api/search/pattern", json=pattern_request)
        pattern_id = create_response.json()["pattern_id"]

        # Test start command
        start_request = {"action": "start"}
        response = client.post(
            f"/api/search/pattern/control?pattern_id={pattern_id}", json=start_request
        )

        assert response.status_code == 200
        data = response.json()
        assert "success" in data

        # Test stop command
        stop_request = {"action": "stop"}
        response = client.post(
            f"/api/search/pattern/control?pattern_id={pattern_id}", json=stop_request
        )

        assert response.status_code == 200

    def test_pattern_preview_functionality(self, client, mock_services):
        """Test pattern preview for mission planning."""
        # Create a spiral pattern for testing
        pattern_request = {
            "pattern": "spiral",
            "spacing": 100,
            "velocity": 10.0,  # Maximum velocity per PRD-FR2
            "bounds": {
                "type": "center_radius",
                "center": {"lat": 51.5074, "lon": -0.1278},
                "radius": 1200,
            },
            "altitude": 50.0,
        }

        create_response = client.post("/api/search/pattern", json=pattern_request)
        pattern_id = create_response.json()["pattern_id"]

        # Test preview endpoint
        response = client.get(f"/api/search/pattern/preview?pattern_id={pattern_id}")

        assert response.status_code == 200
        data = response.json()
        assert "waypoints" in data
        assert "metadata" in data


class TestSearchPatternValidation:
    """Test search pattern input validation per PRD specifications."""

    def test_spacing_validation_bounds(self, client, mock_services):
        """Test spacing validation within 50-100m range."""
        # Test below minimum
        pattern_request = {
            "pattern": "expanding_square",
            "spacing": 25,  # Below 50m minimum
            "velocity": 7.0,
            "bounds": {
                "type": "center_radius",
                "center": {"lat": 40.7128, "lon": -74.0060},
                "radius": 500,
            },
            "altitude": 50.0,
        }

        response = client.post("/api/search/pattern", json=pattern_request)
        assert response.status_code == 422
        assert "spacing" in str(response.json()["detail"])

        # Test above maximum
        pattern_request["spacing"] = 150  # Above 100m maximum
        response = client.post("/api/search/pattern", json=pattern_request)
        assert response.status_code == 422

    def test_altitude_validation_bounds(self, client, mock_services):
        """Test altitude validation within operational limits."""
        pattern_request = {
            "pattern": "expanding_square",
            "spacing": 75,
            "velocity": 8.0,
            "bounds": {
                "type": "center_radius",
                "center": {"lat": 40.7128, "lon": -74.0060},
                "radius": 500,
            },
            "altitude": 500.0,  # Above 400m maximum
        }

        response = client.post("/api/search/pattern", json=pattern_request)
        assert response.status_code == 422
        assert "altitude" in str(response.json()["detail"])

    def test_boundary_validation_corner_format(self, client, mock_services):
        """Test corner boundary validation for polygon search areas."""
        pattern_request = {
            "pattern": "lawnmower",
            "spacing": 60,
            "velocity": 6.5,
            "bounds": {
                "type": "corners",
                "corners": [
                    {"lat": 40.7128, "lon": -74.0060},
                    {"lat": 40.7200, "lon": -74.0060},
                    {"lat": 40.7200, "lon": -73.9950},
                    {"lat": 40.7128, "lon": -73.9950},
                ],
            },
            "altitude": 60.0,
        }

        response = client.post("/api/search/pattern", json=pattern_request)
        # Should succeed with valid 4-corner polygon
        assert response.status_code == 200

    def test_invalid_pattern_type(self, client, mock_services):
        """Test validation of unsupported pattern types."""
        pattern_request = {
            "pattern": "invalid_pattern",  # Not in allowed enum
            "spacing": 75,
            "velocity": 7.0,
            "bounds": {
                "type": "center_radius",
                "center": {"lat": 40.7128, "lon": -74.0060},
                "radius": 500,
            },
            "altitude": 50.0,
        }

        response = client.post("/api/search/pattern", json=pattern_request)
        assert response.status_code == 422
        assert "pattern" in str(response.json()["detail"])
