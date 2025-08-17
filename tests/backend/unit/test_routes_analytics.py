"""
Test suite for analytics API routes.

Validates performance analytics, RSSI processing, mission replay,
and data export functionality per PRD requirements.
"""

import json
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.backend.core.app import create_app


@pytest.fixture
def client():
    """Create test client for analytics routes."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def mock_services():
    """Mock analytics service dependencies."""
    with patch("src.backend.core.dependencies.get_service_manager") as mock_manager:
        # Mock service manager
        mock_service_manager = Mock()
        mock_service_manager.initialize_services = AsyncMock()
        mock_service_manager.shutdown_services = AsyncMock()
        mock_manager.return_value = mock_service_manager

        # Mock analytics and replay services
        mock_analytics = Mock()
        mock_replay = Mock()

        # Configure service manager
        mock_service_manager.performance_analytics = mock_analytics
        mock_service_manager.mission_replay_service = mock_replay

        yield {
            "analytics": mock_analytics,
            "replay": mock_replay,
            "service_manager": mock_service_manager,
        }


@pytest.fixture
def sample_mission_id():
    """Sample mission ID for testing."""
    return uuid4()


class TestPerformanceMetrics:
    """Test performance metrics endpoints per PRD-NFR requirements."""

    def test_get_performance_metrics_success(self, client, mock_services, sample_mission_id):
        """Test successful retrieval of performance metrics per PRD-NFR2."""
        # TDD RED PHASE: Test real API endpoint for performance metrics

        # Mock performance data that aligns with PRD requirements
        mock_metrics = {
            "rssi_processing_latency": 45.2,  # Should be <100ms per PRD-NFR2
            "mavlink_packet_loss": 0.3,  # Should be <1% per PRD-NFR1
            "signal_detection_rate": 95.8,
            "homing_success_rate": 87.5,
            "flight_duration": 1850.0,  # seconds
            "coverage_area": 25600.0,  # square meters
            "false_positive_rate": 2.1,  # Should be <5% per PRD-NFR7
        }

        mock_services["analytics"].get_mission_metrics.return_value = mock_metrics

        response = client.get(f"/api/analytics/metrics?mission_id={sample_mission_id}")

        # Test assertions - should pass in GREEN phase
        assert response.status_code == 200
        data = response.json()
        assert "rssi_processing_latency" in data
        assert "mavlink_packet_loss" in data
        assert "signal_detection_rate" in data

        # Validate PRD compliance
        assert data["rssi_processing_latency"] < 100.0  # PRD-NFR2
        assert data["mavlink_packet_loss"] < 1.0  # PRD-NFR1
        assert data["false_positive_rate"] < 5.0  # PRD-NFR7

    def test_get_metrics_invalid_mission_id(self, client, mock_services):
        """Test metrics retrieval with invalid mission ID."""
        invalid_id = "invalid-uuid"

        response = client.get(f"/api/analytics/metrics?mission_id={invalid_id}")

        # Should fail validation
        assert response.status_code == 422
        assert "mission_id" in str(response.json()["detail"])

    def test_get_metrics_nonexistent_mission(self, client, mock_services):
        """Test metrics retrieval for non-existent mission."""
        nonexistent_id = uuid4()

        # Mock service to raise exception for non-existent mission
        mock_services["analytics"].get_mission_metrics.side_effect = Exception("Mission not found")

        response = client.get(f"/api/analytics/metrics?mission_id={nonexistent_id}")

        # Should handle gracefully
        assert response.status_code == 500


class TestMissionReplay:
    """Test mission replay functionality for post-flight analysis."""

    def test_get_replay_data_success(self, client, mock_services, sample_mission_id):
        """Test successful retrieval of mission replay data."""
        # Mock replay data structure
        mock_replay_data = {
            "mission_id": str(sample_mission_id),
            "duration": 1200.0,
            "waypoints": [
                {"timestamp": "2025-08-17T15:00:00Z", "lat": 40.7128, "lon": -74.0060, "alt": 50.0},
                {"timestamp": "2025-08-17T15:01:00Z", "lat": 40.7130, "lon": -74.0058, "alt": 50.0},
            ],
            "detections": [
                {"timestamp": "2025-08-17T15:05:30Z", "rssi": -65.2, "frequency": 3200000000}
            ],
            "state_changes": [
                {
                    "timestamp": "2025-08-17T15:00:00Z",
                    "from_state": "IDLE",
                    "to_state": "SEARCHING",
                },
                {
                    "timestamp": "2025-08-17T15:05:30Z",
                    "from_state": "SEARCHING",
                    "to_state": "HOMING",
                },
            ],
        }

        mock_services["replay"].get_mission_data.return_value = mock_replay_data

        response = client.get(f"/api/analytics/replay/{sample_mission_id}")

        assert response.status_code == 200
        data = response.json()
        assert "mission_id" in data
        assert "duration" in data
        assert "waypoints" in data
        assert "detections" in data
        assert "state_changes" in data

    def test_replay_control_play_pause(self, client, mock_services, sample_mission_id):
        """Test replay control operations - play and pause."""
        # Test play command
        play_request = {"action": "play", "speed": 1.0}

        mock_services["replay"].control_playback.return_value = {
            "status": "playing",
            "position": 0,
            "speed": 1.0,
        }

        response = client.post(
            f"/api/analytics/replay/{sample_mission_id}/control", json=play_request
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "playing"

        # Test pause command
        pause_request = {"action": "pause"}

        mock_services["replay"].control_playback.return_value = {
            "status": "paused",
            "position": 150,
            "speed": 0.0,
        }

        response = client.post(
            f"/api/analytics/replay/{sample_mission_id}/control", json=pause_request
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "paused"

    def test_replay_control_invalid_action(self, client, mock_services, sample_mission_id):
        """Test replay control with invalid action."""
        invalid_request = {"action": "invalid_action"}

        response = client.post(
            f"/api/analytics/replay/{sample_mission_id}/control", json=invalid_request
        )

        # Should fail validation
        assert response.status_code == 422
        assert "action" in str(response.json()["detail"])


class TestDataExport:
    """Test data export functionality for analysis and reporting."""

    def test_export_telemetry_csv_success(self, client, mock_services, sample_mission_id):
        """Test successful CSV export of telemetry data."""
        export_request = {
            "mission_id": str(sample_mission_id),
            "format": "csv",
            "data_type": "telemetry",
            "include_sensitive": False,
        }

        # Mock CSV data
        mock_csv_data = (
            "timestamp,lat,lon,alt,rssi\n2025-08-17T15:00:00Z,40.7128,-74.0060,50.0,-65.2\n"
        )
        mock_services["analytics"].export_mission_data.return_value = mock_csv_data

        response = client.post("/api/analytics/export", json=export_request)

        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]
        assert "timestamp,lat,lon,alt,rssi" in response.text

    def test_export_detections_json_success(self, client, mock_services, sample_mission_id):
        """Test successful JSON export of detection data per PRD-FR1."""
        export_request = {
            "mission_id": str(sample_mission_id),
            "format": "json",
            "data_type": "detections",
            "start_time": "2025-08-17T15:00:00Z",
            "end_time": "2025-08-17T15:30:00Z",
        }

        # Mock detection data aligned with PRD-FR1 (beacon detection)
        mock_detection_data = {
            "detections": [
                {
                    "timestamp": "2025-08-17T15:05:30Z",
                    "frequency": 3200000000,  # 3.2 GHz default per PRD-FR1
                    "rssi": -65.2,
                    "snr": 15.8,  # >12 dB threshold per PRD-FR1
                    "confidence": 0.95,
                    "location": {"lat": 40.7128, "lon": -74.0060},
                }
            ],
            "summary": {"total_detections": 1, "detection_rate": 95.8, "average_snr": 15.8},
        }

        mock_services["analytics"].export_mission_data.return_value = json.dumps(
            mock_detection_data
        )

        response = client.post("/api/analytics/export", json=export_request)

        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]
        data = json.loads(response.text)
        assert "detections" in data
        assert "summary" in data

    def test_export_invalid_format(self, client, mock_services, sample_mission_id):
        """Test export with invalid format."""
        export_request = {
            "mission_id": str(sample_mission_id),
            "format": "xml",  # Invalid format
            "data_type": "telemetry",
        }

        response = client.post("/api/analytics/export", json=export_request)

        # Should fail validation
        assert response.status_code == 422
        assert "format" in str(response.json()["detail"])

    def test_export_invalid_data_type(self, client, mock_services, sample_mission_id):
        """Test export with invalid data type."""
        export_request = {
            "mission_id": str(sample_mission_id),
            "format": "json",
            "data_type": "invalid_type",  # Invalid data type
        }

        response = client.post("/api/analytics/export", json=export_request)

        # Should fail validation
        assert response.status_code == 422
        assert "data_type" in str(response.json()["detail"])


class TestSystemRecommendations:
    """Test system recommendations for optimization."""

    def test_get_system_recommendations_success(self, client, mock_services):
        """Test successful retrieval of system optimization recommendations."""
        mock_recommendations = {
            "performance": [
                {
                    "category": "rssi_processing",
                    "priority": "high",
                    "description": "RSSI processing latency averaging 75ms, consider optimization",
                    "target": "< 50ms for improved responsiveness",
                }
            ],
            "safety": [
                {
                    "category": "geofencing",
                    "priority": "medium",
                    "description": "Consider expanding geofence buffer by 20%",
                    "target": "Improved safety margin",
                }
            ],
            "operational": [
                {
                    "category": "search_patterns",
                    "priority": "low",
                    "description": "Reduce search spacing to 75m for better coverage",
                    "target": "Increased detection probability",
                }
            ],
        }

        mock_services["analytics"].get_recommendations.return_value = mock_recommendations

        response = client.get("/api/analytics/recommendations")

        assert response.status_code == 200
        data = response.json()
        assert "performance" in data
        assert "safety" in data
        assert "operational" in data

    def test_get_mission_report_json(self, client, mock_services, sample_mission_id):
        """Test mission report generation in JSON format."""
        mock_report = {
            "mission_summary": {
                "mission_id": str(sample_mission_id),
                "start_time": "2025-08-17T15:00:00Z",
                "end_time": "2025-08-17T15:30:00Z",
                "duration": 1800.0,
                "status": "completed",
            },
            "performance_metrics": {
                "detection_count": 3,
                "false_positives": 0,
                "coverage_area": 12800.0,
                "average_speed": 7.5,
            },
            "compliance": {
                "prd_nfr1": True,  # MAVLink <1% packet loss
                "prd_nfr2": True,  # RSSI <100ms latency
                "prd_nfr7": True,  # <5% false positive rate
            },
        }

        mock_services["analytics"].generate_mission_report.return_value = mock_report

        response = client.get(f"/api/analytics/reports/{sample_mission_id}?format=json")

        assert response.status_code == 200
        data = response.json()
        assert "mission_summary" in data
        assert "performance_metrics" in data
        assert "compliance" in data
