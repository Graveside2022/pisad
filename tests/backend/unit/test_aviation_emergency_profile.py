"""
Test Suite for Aviation Emergency Profile (TASK-6.3.1 [28c2])

Tests for Aviation Emergency profile (121.5MHz, 50kHz bandwidth, aviation radio characteristics):
- Frequency specification (121.5MHz aviation emergency frequency)
- Bandwidth specification (50kHz for aviation radio)
- Aviation radio signal characteristics configuration
- Profile integration with Mission Planner parameter system
"""

from pathlib import Path

import pytest
import yaml

from src.backend.services.asv_integration.asv_configuration_manager import ASVConfigurationManager
from src.backend.services.mavlink_service import MAVLinkService


class TestAviationEmergencyProfile:
    """Test suite for Aviation Emergency profile implementation."""

    @pytest.fixture
    def config_manager(self):
        """Create ASV configuration manager for testing."""
        return ASVConfigurationManager()

    @pytest.fixture
    def mavlink_service(self):
        """Create MAVLink service for testing."""
        service = MAVLinkService()
        service._parameters = {}
        return service

    def test_aviation_emergency_profile_frequency_specification(self, config_manager):
        """Test Aviation Emergency profile frequency specification (121.5MHz) [28c2]."""
        # Get Aviation Emergency profile configuration
        profile = config_manager.get_frequency_profile("aviation_emergency")

        assert profile is not None
        assert profile.center_frequency_hz == 121_500_000  # 121.5 MHz aviation emergency
        assert profile.name == "aviation_emergency"

    def test_aviation_emergency_profile_bandwidth(self, config_manager):
        """Test Aviation Emergency profile bandwidth specification (50kHz) [28c2]."""
        profile = config_manager.get_frequency_profile("aviation_emergency")

        assert profile is not None
        assert profile.bandwidth_hz == 50_000  # 50 kHz as per task requirements for aviation radio

    def test_aviation_emergency_profile_aviation_characteristics(self, config_manager):
        """Test Aviation Emergency profile aviation radio characteristics [28c2]."""
        profile = config_manager.get_frequency_profile("aviation_emergency")

        assert profile is not None

        # Aviation radio signal characteristics
        assert profile.analyzer_type == "GP"  # General Purpose for aviation emergency detection
        assert profile.priority == 1  # Emergency priority
        assert profile.ref_power_dbm <= -110.0  # Sensitive detection for aviation emergency signals
        assert profile.processing_timeout_ms <= 60  # Reasonable response for aviation emergency
        assert profile.calibration_enabled == True  # Ensure accuracy for aviation emergency signals

    def test_aviation_emergency_profile_mission_planner_integration(self, mavlink_service):
        """Test Aviation Emergency profile integration with Mission Planner parameter system [28c2]."""
        # Test applying Aviation Emergency profile parameters
        parameter_set = {
            "PISAD_RF_FREQ": 121_500_000,  # Aviation Emergency frequency
            "PISAD_RF_PROFILE": 1,  # Aviation Emergency profile (1 = Aviation)
            "PISAD_RF_BW": 50_000,  # Aviation Emergency bandwidth
        }

        result = mavlink_service.validate_parameter_set(parameter_set)
        assert result["valid"] == True
        assert "validation_results" in result

        # Test that frequency validation passes for Aviation Emergency
        freq_result = mavlink_service.validate_frequency_parameter("PISAD_RF_FREQ", 121_500_000)
        assert freq_result["valid"] == True

    def test_aviation_emergency_profile_frequency_validation(self, mavlink_service):
        """Test Aviation Emergency profile frequency validation [28c2]."""
        # Test valid Aviation Emergency frequency (121.5 MHz)
        result = mavlink_service.validate_frequency_parameter("PISAD_RF_FREQ", 121_500_000)
        assert result["valid"] == True

        # Should be authorized despite any conflicts (emergency authorization)
        if result.get("conflicts"):
            assert result.get("authorized") == True
            assert "emergency" in result.get("authorization_reason", "").lower()

    def test_aviation_emergency_profile_yaml_configuration(self):
        """Test Aviation Emergency profile YAML configuration correctness [28c2]."""
        config_path = Path(
            "/home/pisad/projects/pisad/config/asv_integration/frequency_profiles.yaml"
        )

        with open(config_path, "r") as f:
            profiles = yaml.safe_load(f)

        assert "aviation_emergency" in profiles
        aviation_profile = profiles["aviation_emergency"]

        # Verify aviation-specific configuration
        assert aviation_profile["center_frequency_hz"] == 121_500_000
        assert (
            aviation_profile["bandwidth_hz"] == 50_000
        )  # Task requirement: 50kHz for aviation radio
        assert aviation_profile["description"] == "Aviation Emergency Frequency"
        assert aviation_profile["analyzer_type"] == "GP"
        assert aviation_profile["priority"] == 1  # Emergency priority

    def test_aviation_emergency_profile_analyzer_integration(self):
        """Test Aviation Emergency profile integration with emergency analyzer [28c2]."""
        analyzer_config_path = Path(
            "/home/pisad/projects/pisad/config/asv_integration/analyzer_profiles.yaml"
        )

        with open(analyzer_config_path, "r") as f:
            analyzers = yaml.safe_load(f)

        assert "emergency_analyzer" in analyzers
        emergency_analyzer = analyzers["emergency_analyzer"]

        # Verify aviation_emergency is in frequency profiles
        assert "aviation_emergency" in emergency_analyzer["frequency_profiles"]

        # Verify emergency analyzer characteristics suitable for aviation
        assert emergency_analyzer["analyzer_type"] == "GP"
        assert emergency_analyzer["enabled"] == True

        # Aviation emergency detection requires sensitive threshold
        processing_config = emergency_analyzer["processing_config"]
        assert processing_config["detection_threshold_db"] <= -110

    def test_aviation_emergency_profile_aviation_frequency_band(self, config_manager):
        """Test Aviation Emergency profile is within aviation frequency band [28c2]."""
        profile = config_manager.get_frequency_profile("aviation_emergency")

        assert profile is not None

        # 121.5 MHz is the international aviation emergency frequency
        # It's within the VHF aviation band (118-137 MHz)
        center_freq = profile.center_frequency_hz
        bandwidth = profile.bandwidth_hz

        # Calculate frequency range
        lower_freq = center_freq - (bandwidth / 2)
        upper_freq = center_freq + (bandwidth / 2)

        # Verify within aviation VHF band
        assert lower_freq >= 118_000_000  # >= 118.0 MHz (aviation VHF start)
        assert upper_freq <= 137_000_000  # <= 137.0 MHz (aviation VHF end)
        assert center_freq == 121_500_000  # Exactly 121.5 MHz emergency frequency

    def test_aviation_emergency_profile_response_timing(self, config_manager):
        """Test Aviation Emergency profile response timing requirements [28c2]."""
        import time

        start_time = time.perf_counter()
        profile = config_manager.get_frequency_profile("aviation_emergency")
        end_time = time.perf_counter()

        response_time_ms = (end_time - start_time) * 1000

        assert profile is not None
        assert response_time_ms < 50.0  # <50ms requirement
        assert profile.processing_timeout_ms <= 60  # Reasonable aviation emergency processing
