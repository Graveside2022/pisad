"""
Test Suite for ASV Frequency Profile API Endpoints

SUBTASK-6.1.2.2 [15d-1] - Test for ASV frequency profile REST API implementation

This test module validates the ASV frequency profile API endpoints including:
- Profile listing and retrieval functionality
- Real-time frequency switching with <50ms response time
- Profile validation against hardware capabilities
- Integration with ASVConfigurationManager and ASVHackRFCoordinator
- Error handling for invalid frequencies and safety constraints
"""

import asyncio
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from src.backend.core.app import app
from src.backend.services.asv_integration.asv_configuration_manager import (
    ASVConfigurationManager, 
    ASVFrequencyProfile
)
from src.backend.services.asv_integration.asv_hackrf_coordinator import ASVHackRFCoordinator
from src.backend.services.asv_integration.exceptions import ASVFrequencyError, ASVHardwareError

# Test client for API testing
client = TestClient(app)


class TestASVFrequencyProfileAPI:
    """Test suite for ASV frequency profile REST API endpoints."""

    @pytest.fixture
    def mock_config_manager(self):
        """Mock ASVConfigurationManager for testing."""
        manager = MagicMock(spec=ASVConfigurationManager)
        
        # Mock default frequency profiles
        manager.get_all_frequency_profiles.return_value = {
            "emergency_beacon_406": ASVFrequencyProfile(
                name="emergency_beacon_406",
                description="Emergency Beacon Detection at 406 MHz",
                center_frequency_hz=406_000_000,
                bandwidth_hz=50_000,
                analyzer_type="GP",
                ref_power_dbm=-120.0,
                priority=1
            ),
            "aviation_emergency": ASVFrequencyProfile(
                name="aviation_emergency", 
                description="Aviation Emergency Frequency",
                center_frequency_hz=121_500_000,
                bandwidth_hz=25_000,
                analyzer_type="GP",
                ref_power_dbm=-110.0,
                priority=2
            ),
            "custom_frequency": ASVFrequencyProfile(
                name="custom_frequency",
                description="Custom User-Defined Frequency",
                center_frequency_hz=2_400_000_000,
                bandwidth_hz=200_000,
                analyzer_type="GP", 
                ref_power_dbm=-100.0,
                priority=3
            )
        }
        
        return manager

    @pytest.fixture  
    def mock_coordinator(self):
        """Mock ASVHackRFCoordinator for testing."""
        coordinator = MagicMock(spec=ASVHackRFCoordinator)
        coordinator.switch_frequency = AsyncMock(return_value=True)
        coordinator.get_current_frequency = AsyncMock(return_value=406_000_000)
        coordinator.is_running = True
        return coordinator

    def test_get_frequency_profiles_endpoint_exists(self):
        """Test that GET /api/asv/frequency-profiles endpoint exists."""
        # This test will initially fail until we implement the endpoint
        response = client.get("/api/asv/frequency-profiles")
        
        # Should not return 404 once implemented
        assert response.status_code != 404, "ASV frequency profiles endpoint should exist"

    def test_get_frequency_profiles_returns_all_profiles(self, mock_config_manager):
        """Test that frequency profiles endpoint returns all available profiles."""
        with patch('src.backend.services.asv_integration.asv_configuration_manager.ASVConfigurationManager') as mock_asv_config:
            mock_asv_config.return_value = mock_config_manager
            
            response = client.get("/api/asv/frequency-profiles")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "profiles" in data
            assert len(data["profiles"]) == 3
            
            # Verify emergency beacon profile
            emergency_profile = next(p for p in data["profiles"] if p["name"] == "emergency_beacon_406")
            assert emergency_profile["center_frequency_hz"] == 406_000_000
            assert emergency_profile["description"] == "Emergency Beacon Detection at 406 MHz"
            assert emergency_profile["analyzer_type"] == "GP"

    def test_switch_frequency_profile_endpoint_exists(self):
        """Test that POST /api/asv/switch-frequency endpoint exists."""
        # This test will initially fail until we implement the endpoint
        response = client.post("/api/asv/switch-frequency", json={"profile_name": "emergency_beacon_406"})
        
        # Should not return 404 once implemented
        assert response.status_code != 404, "ASV frequency switching endpoint should exist"

    @pytest.mark.asyncio
    async def test_switch_frequency_profile_success(self, mock_config_manager, mock_coordinator):
        """Test successful frequency profile switching."""
        with patch('src.backend.services.asv_integration.asv_configuration_manager.ASVConfigurationManager') as mock_asv_config:
            mock_asv_config.return_value = mock_config_manager
            
            response = client.post("/api/asv/switch-frequency", json={
                "profile_name": "aviation_emergency"
            })
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["profile_name"] == "aviation_emergency"
            assert data["frequency_hz"] == 121_500_000
            assert "switch_time_ms" in data
            assert data["switch_time_ms"] < 50  # Must meet <50ms requirement

    def test_switch_frequency_profile_invalid_profile(self):
        """Test frequency switching with invalid profile name."""
        response = client.post("/api/asv/switch-frequency", json={
            "profile_name": "nonexistent_profile"
        })
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "Profile not found" in data["error"]

    def test_switch_frequency_profile_hardware_error(self, mock_config_manager, mock_coordinator):
        """Test frequency switching with hardware error."""
        with patch('src.backend.api.routes.asv_integration.asv_config_manager', mock_config_manager), \
             patch('src.backend.api.routes.asv_integration.asv_coordinator', mock_coordinator):
            
            # Mock hardware failure
            mock_coordinator.switch_frequency.side_effect = ASVHardwareError("HackRF communication failed")
            
            response = client.post("/api/asv/switch-frequency", json={
                "profile_name": "emergency_beacon_406"
            })
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert "HackRF communication failed" in data["error"]

    def test_frequency_validation_out_of_range(self):
        """Test frequency validation for out-of-range frequencies."""
        # Test frequency below HackRF range (< 1 MHz)
        response = client.post("/api/asv/switch-frequency", json={
            "profile_name": "invalid_low_frequency"
        })
        
        # Should fail validation
        assert response.status_code in [400, 422]

    def test_current_frequency_profile_endpoint(self, mock_coordinator):
        """Test GET /api/asv/current-frequency endpoint."""
        with patch('src.backend.api.routes.asv_integration.asv_coordinator', mock_coordinator):
            mock_coordinator.get_current_frequency.return_value = 406_000_000
            
            response = client.get("/api/asv/current-frequency")
            
            assert response.status_code == 200
            data = response.json()
            assert data["frequency_hz"] == 406_000_000
            assert "profile_name" in data

    def test_create_custom_frequency_profile(self, mock_config_manager):
        """Test creating custom frequency profile."""
        with patch('src.backend.api.routes.asv_integration.asv_config_manager', mock_config_manager):
            custom_profile_data = {
                "name": "test_custom",
                "description": "Test Custom Frequency Profile",
                "center_frequency_hz": 915_000_000,  # 915 MHz ISM band
                "bandwidth_hz": 100_000,
                "analyzer_type": "GP"
            }
            
            response = client.post("/api/asv/frequency-profiles", json=custom_profile_data)
            
            assert response.status_code == 201
            data = response.json()
            assert data["success"] is True
            assert data["profile_name"] == "test_custom"

    def test_performance_requirement_frequency_switching(self, mock_config_manager, mock_coordinator):
        """Test that frequency switching meets <50ms performance requirement."""
        with patch('src.backend.api.routes.asv_integration.asv_config_manager', mock_config_manager), \
             patch('src.backend.api.routes.asv_integration.asv_coordinator', mock_coordinator):
            
            # Simulate fast switching
            import time
            start_time = time.time()
            mock_coordinator.switch_frequency.return_value = True
            
            response = client.post("/api/asv/switch-frequency", json={
                "profile_name": "emergency_beacon_406"
            })
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify performance requirement
            assert data["switch_time_ms"] < 50, f"Frequency switching took {data['switch_time_ms']}ms, must be <50ms"