"""Integration tests for Configuration API routes."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Mock the websocket broadcast function before importing the app
with patch("src.backend.api.routes.config.broadcast_message") as mock_broadcast:
    mock_broadcast.return_value = None
    from src.backend.core.app import app

client = TestClient(app)


@pytest.fixture
def sample_profile_data():
    """Sample profile data for testing."""
    return {
        "name": "test_profile",
        "description": "Test configuration profile",
        "sdrConfig": {
            "frequency": 2437000000.0,
            "sampleRate": 2000000.0,
            "gain": 40,
            "bandwidth": 2000000.0,
        },
        "signalConfig": {
            "fftSize": 1024,
            "ewmaAlpha": 0.1,
            "triggerThreshold": -60.0,
            "dropThreshold": -70.0,
        },
        "homingConfig": {
            "forwardVelocityMax": 5.0,
            "yawRateMax": 1.0,
            "approachVelocity": 2.0,
            "signalLossTimeout": 5.0,
        },
        "isDefault": False,
    }


class TestConfigurationRoutes:
    """Test suite for configuration API routes."""

    def test_list_profiles(self):
        """Test GET /config/profiles endpoint."""
        response = client.get("/config/profiles")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        # Should include preset profiles
        profile_names = [p["name"] for p in data]
        assert "wifi_beacon" in profile_names
        assert "lora_beacon" in profile_names
        assert "custom" in profile_names

    def test_create_profile(self, sample_profile_data):
        """Test POST /config/profiles endpoint."""
        response = client.post("/config/profiles", json=sample_profile_data)
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == sample_profile_data["name"]
        assert data["description"] == sample_profile_data["description"]
        assert "id" in data
        assert "createdAt" in data
        assert "updatedAt" in data

        # Verify profile was created
        response = client.get("/config/profiles")
        profiles = response.json()
        profile_names = [p["name"] for p in profiles]
        assert sample_profile_data["name"] in profile_names

    def test_create_profile_invalid_data(self, sample_profile_data):
        """Test creating a profile with invalid data."""
        # Invalid frequency
        invalid_data = sample_profile_data.copy()
        invalid_data["sdrConfig"]["frequency"] = 0.5e6  # Below 1 MHz

        response = client.post("/config/profiles", json=invalid_data)
        assert response.status_code == 400
        assert "Invalid profile configuration" in response.json()["detail"]

    def test_update_profile(self, sample_profile_data):
        """Test PUT /config/profiles/{id} endpoint."""
        # First create a profile
        create_response = client.post("/config/profiles", json=sample_profile_data)
        assert create_response.status_code == 201
        profile_id = create_response.json()["id"]

        # Update the profile
        updated_data = sample_profile_data.copy()
        updated_data["description"] = "Updated description"
        updated_data["sdrConfig"]["gain"] = 50

        response = client.put(f"/config/profiles/{profile_id}", json=updated_data)
        assert response.status_code == 200
        data = response.json()
        assert data["description"] == "Updated description"
        assert data["sdrConfig"]["gain"] == 50

    def test_update_nonexistent_profile(self, sample_profile_data):
        """Test updating a non-existent profile."""
        response = client.put("/config/profiles/nonexistent-id", json=sample_profile_data)
        assert response.status_code == 404
        assert "Profile not found" in response.json()["detail"]

    @patch("src.backend.api.routes.config.broadcast_message")
    def test_activate_profile(self, mock_broadcast, sample_profile_data):
        """Test POST /config/profiles/{id}/activate endpoint."""
        # Create a profile first
        create_response = client.post("/config/profiles", json=sample_profile_data)
        assert create_response.status_code == 201
        profile_id = create_response.json()["id"]

        # Activate the profile
        response = client.post(f"/config/profiles/{profile_id}/activate")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "activated successfully" in data["message"]

        # Verify broadcast was called
        mock_broadcast.assert_called()

    def test_activate_nonexistent_profile(self):
        """Test activating a non-existent profile."""
        response = client.post("/config/profiles/nonexistent-id/activate")
        assert response.status_code == 404
        assert "Profile not found" in response.json()["detail"]

    def test_delete_profile(self, sample_profile_data):
        """Test DELETE /config/profiles/{id} endpoint."""
        # Create a profile first
        create_response = client.post("/config/profiles", json=sample_profile_data)
        assert create_response.status_code == 201
        profile_id = create_response.json()["id"]

        # Delete the profile
        response = client.delete(f"/config/profiles/{profile_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

        # Verify profile was deleted
        response = client.get("/config/profiles")
        profiles = response.json()
        profile_ids = [p["id"] for p in profiles]
        assert profile_id not in profile_ids

    def test_delete_nonexistent_profile(self):
        """Test deleting a non-existent profile."""
        response = client.delete("/config/profiles/nonexistent-id")
        assert response.status_code == 404
        assert "Profile not found" in response.json()["detail"]

    def test_profile_validation(self):
        """Test profile validation during creation."""
        test_cases = [
            {
                "description": "Invalid EWMA alpha",
                "data": {"signalConfig": {"ewmaAlpha": 2.0}},  # > 1
                "expected_error": "EWMA alpha must be between 0 and 1",
            },
            {
                "description": "Invalid thresholds",
                "data": {
                    "signalConfig": {"triggerThreshold": -60.0, "dropThreshold": -50.0}  # > trigger
                },
                "expected_error": "Drop threshold must be less than trigger threshold",
            },
            {
                "description": "Invalid forward velocity",
                "data": {"homingConfig": {"forwardVelocityMax": -1.0}},  # negative
                "expected_error": "Forward velocity max must be positive",
            },
        ]

        base_profile = {
            "name": "validation_test",
            "description": "Test validation",
            "sdrConfig": {
                "frequency": 2437000000.0,
                "sampleRate": 2000000.0,
                "gain": 40,
                "bandwidth": 2000000.0,
            },
            "signalConfig": {
                "fftSize": 1024,
                "ewmaAlpha": 0.1,
                "triggerThreshold": -60.0,
                "dropThreshold": -70.0,
            },
            "homingConfig": {
                "forwardVelocityMax": 5.0,
                "yawRateMax": 1.0,
                "approachVelocity": 2.0,
                "signalLossTimeout": 5.0,
            },
        }

        for test_case in test_cases:
            # Merge test data with base profile
            test_profile = base_profile.copy()
            for key, value in test_case["data"].items():
                if key in test_profile:
                    test_profile[key].update(value)

            response = client.post("/config/profiles", json=test_profile)
            assert response.status_code == 400, f"Failed for: {test_case['description']}"
            assert test_case["expected_error"] in response.json()["detail"]
