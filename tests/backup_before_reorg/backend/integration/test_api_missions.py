"""Integration tests for mission API endpoints."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.backend.core.app import create_app


@pytest.fixture
def app():
    """Create test app."""
    return create_app()


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_mission_data():
    """Mock mission data."""
    return {
        "id": "mission-123",
        "name": "Test Mission",
        "status": "ACTIVE",
        "created_at": datetime.now(UTC).isoformat(),
        "started_at": datetime.now(UTC).isoformat(),
        "search_pattern": {
            "type": "SPIRAL",
            "center_lat": 37.4419,
            "center_lon": -122.1430,
            "radius": 100,
            "spacing": 10,
        },
        "detections": [],
        "telemetry": {
            "total_distance": 500.0,
            "total_time": 300.0,
            "max_altitude": 120.0,
            "average_speed": 5.0,
        },
    }


@pytest.fixture
def mock_database():
    """Mock database service."""
    mock = MagicMock()
    mock.create_mission = AsyncMock(return_value="mission-123")
    mock.get_mission = AsyncMock(return_value={})
    mock.list_missions = AsyncMock(return_value=[])
    mock.update_mission = AsyncMock(return_value=True)
    mock.delete_mission = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_state_machine():
    """Mock state machine."""
    mock = MagicMock()
    mock.transition_to = AsyncMock(return_value=True)
    mock.get_current_state = MagicMock(return_value="IDLE")
    return mock


class TestMissionManagementEndpoints:
    """Test mission management endpoints."""

    def test_create_mission(self, client, mock_database, mock_mission_data):
        """Test creating a new mission."""
        with patch("src.backend.api.routes.missions.database", mock_database):
            response = client.post(
                "/api/missions",
                json={
                    "name": "Test Mission",
                    "search_pattern": {
                        "type": "SPIRAL",
                        "center_lat": 37.4419,
                        "center_lon": -122.1430,
                        "radius": 100,
                        "spacing": 10,
                    },
                    "config_overrides": {"sdr": {"gain": 45}},
                },
            )

            assert response.status_code == 201
            data = response.json()
            assert "mission_id" in data
            assert data["mission_id"] == "mission-123"

    def test_get_mission(self, client, mock_database, mock_mission_data):
        """Test getting mission details."""
        mock_database.get_mission = AsyncMock(return_value=mock_mission_data)

        with patch("src.backend.api.routes.missions.database", mock_database):
            response = client.get("/api/missions/mission-123")

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "mission-123"
            assert data["name"] == "Test Mission"
            assert data["status"] == "ACTIVE"

    def test_get_mission_not_found(self, client, mock_database):
        """Test getting non-existent mission."""
        mock_database.get_mission = AsyncMock(return_value=None)

        with patch("src.backend.api.routes.missions.database", mock_database):
            response = client.get("/api/missions/invalid-id")

            assert response.status_code == 404
            data = response.json()
            assert "detail" in data

    def test_list_missions(self, client, mock_database, mock_mission_data):
        """Test listing all missions."""
        mock_database.list_missions = AsyncMock(return_value=[mock_mission_data])

        with patch("src.backend.api.routes.missions.database", mock_database):
            response = client.get("/api/missions")

            assert response.status_code == 200
            data = response.json()
            assert "missions" in data
            assert len(data["missions"]) == 1
            assert data["missions"][0]["id"] == "mission-123"

    def test_update_mission(self, client, mock_database):
        """Test updating mission."""
        with patch("src.backend.api.routes.missions.database", mock_database):
            response = client.put(
                "/api/missions/mission-123",
                json={"name": "Updated Mission", "notes": "Updated notes"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    def test_delete_mission(self, client, mock_database):
        """Test deleting mission."""
        with patch("src.backend.api.routes.missions.database", mock_database):
            response = client.delete("/api/missions/mission-123")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True


class TestMissionControlEndpoints:
    """Test mission control endpoints."""

    def test_start_mission(self, client, mock_database, mock_state_machine, mock_mission_data):
        """Test starting a mission."""
        mock_database.get_mission = AsyncMock(return_value=mock_mission_data)

        with patch("src.backend.api.routes.missions.database", mock_database):
            with patch("src.backend.api.routes.missions.state_machine", mock_state_machine):
                response = client.post("/api/missions/mission-123/start")

                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True

                mock_state_machine.transition_to.assert_called()

    def test_pause_mission(self, client, mock_database, mock_state_machine):
        """Test pausing a mission."""
        with patch("src.backend.api.routes.missions.database", mock_database):
            with patch("src.backend.api.routes.missions.state_machine", mock_state_machine):
                response = client.post("/api/missions/mission-123/pause")

                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True

    def test_resume_mission(self, client, mock_database, mock_state_machine):
        """Test resuming a mission."""
        with patch("src.backend.api.routes.missions.database", mock_database):
            with patch("src.backend.api.routes.missions.state_machine", mock_state_machine):
                response = client.post("/api/missions/mission-123/resume")

                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True

    def test_stop_mission(self, client, mock_database, mock_state_machine):
        """Test stopping a mission."""
        with patch("src.backend.api.routes.missions.database", mock_database):
            with patch("src.backend.api.routes.missions.state_machine", mock_state_machine):
                response = client.post("/api/missions/mission-123/stop")

                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True

                mock_state_machine.transition_to.assert_called()


class TestMissionDataEndpoints:
    """Test mission data endpoints."""

    def test_get_mission_telemetry(self, client, mock_database):
        """Test getting mission telemetry."""
        mock_database.get_mission_telemetry = AsyncMock(
            return_value=[
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "lat": 37.4419,
                    "lon": -122.1430,
                    "alt": 100.0,
                    "rssi": -75.0,
                    "snr": 15.0,
                }
            ]
        )

        with patch("src.backend.api.routes.missions.database", mock_database):
            response = client.get("/api/missions/mission-123/telemetry")

            assert response.status_code == 200
            data = response.json()
            assert "telemetry" in data
            assert len(data["telemetry"]) == 1

    def test_get_mission_detections(self, client, mock_database):
        """Test getting mission detections."""
        mock_database.get_mission_detections = AsyncMock(
            return_value=[
                {
                    "id": "detection-1",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "lat": 37.4419,
                    "lon": -122.1430,
                    "rssi": -70.0,
                    "snr": 20.0,
                    "confidence": 85.0,
                }
            ]
        )

        with patch("src.backend.api.routes.missions.database", mock_database):
            response = client.get("/api/missions/mission-123/detections")

            assert response.status_code == 200
            data = response.json()
            assert "detections" in data
            assert len(data["detections"]) == 1

    def test_export_mission_data(self, client, mock_database, mock_mission_data):
        """Test exporting mission data."""
        mock_database.get_mission = AsyncMock(return_value=mock_mission_data)
        mock_database.get_mission_telemetry = AsyncMock(return_value=[])
        mock_database.get_mission_detections = AsyncMock(return_value=[])

        with patch("src.backend.api.routes.missions.database", mock_database):
            response = client.get("/api/missions/mission-123/export")

            assert response.status_code == 200
            assert response.headers["content-type"] == "application/json"

            data = response.json()
            assert "mission" in data
            assert "telemetry" in data
            assert "detections" in data
