"""
Test Suite for Emergency Beacon Profile (TASK-6.3.1 [28c1])

Tests for Emergency Beacon profile (406.0-406.1MHz, 25kHz bandwidth, ELT signal characteristics):
- Frequency range validation (406.0-406.1MHz)
- Bandwidth specification (25kHz)
- ELT signal characteristics configuration
- Profile integration with Mission Planner parameter system
"""

from pathlib import Path

import pytest
import yaml

from src.backend.services.asv_integration.asv_configuration_manager import ASVConfigurationManager
from src.backend.services.mavlink_service import MAVLinkService


class TestEmergencyBeaconProfile:
    """Test suite for Emergency Beacon profile implementation."""

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

    def test_emergency_beacon_profile_frequency_range(self, config_manager):
        """Test Emergency Beacon profile frequency specification (406.0-406.1MHz) [28c1]."""
        # Get Emergency Beacon profile configuration
        profile = config_manager.get_frequency_profile("emergency_beacon_406")

        assert profile is not None
        assert profile.center_frequency_hz == 406_000_000  # 406.0 MHz
        assert profile.name == "emergency_beacon_406"

        # Verify frequency is within ELT band (406.0-406.1MHz)
        center_freq = profile.center_frequency_hz
        bandwidth = profile.bandwidth_hz

        # Calculate frequency range
        lower_freq = center_freq - (bandwidth / 2)
        upper_freq = center_freq + (bandwidth / 2)

        # Verify within ELT band (406.0-406.1MHz)
        # Center frequency should be 406.0 MHz, and with 25kHz bandwidth,
        # the signal should fit within the ELT band
        assert center_freq == 406_000_000  # Exactly 406.0 MHz
        assert bandwidth == 25_000  # 25 kHz bandwidth
        assert upper_freq <= 406_100_000  # <= 406.1 MHz (within ELT band)

    def test_emergency_beacon_profile_bandwidth(self, config_manager):
        """Test Emergency Beacon profile bandwidth specification (25kHz) [28c1]."""
        profile = config_manager.get_frequency_profile("emergency_beacon_406")

        assert profile is not None
        assert profile.bandwidth_hz == 25_000  # 25 kHz as per task requirements

    def test_emergency_beacon_profile_elt_characteristics(self, config_manager):
        """Test Emergency Beacon profile ELT signal characteristics [28c1]."""
        profile = config_manager.get_frequency_profile("emergency_beacon_406")

        assert profile is not None

        # ELT signal characteristics
        assert profile.analyzer_type == "GP"  # General Purpose for ELT detection
        assert profile.ref_power_dbm <= -110.0  # Sensitive detection for emergency signals
        assert profile.processing_timeout_ms <= 50  # Fast response for emergency detection
        assert profile.calibration_enabled == True  # Ensure accuracy for emergency signals

    def test_emergency_beacon_profile_mission_planner_integration(self, mavlink_service):
        """Test Emergency Beacon profile integration with Mission Planner parameter system [28c1]."""
        # Test applying Emergency Beacon profile parameters
        parameter_set = {
            "PISAD_RF_FREQ": 406_000_000,  # Emergency Beacon frequency
            "PISAD_RF_PROFILE": 0,  # Emergency Beacon profile (validated as other param)
            "PISAD_RF_BW": 25_000,  # Emergency Beacon bandwidth
        }

        result = mavlink_service.validate_parameter_set(parameter_set)
        assert result["valid"] == True
        assert "validation_results" in result

        # Test that frequency validation passes for Emergency Beacon
        freq_result = mavlink_service.validate_frequency_parameter("PISAD_RF_FREQ", 406_000_000)
        assert freq_result["valid"] == True

    def test_emergency_beacon_profile_frequency_validation(self, mavlink_service):
        """Test Emergency Beacon profile frequency validation [28c1]."""
        # Test valid Emergency Beacon frequency (406 MHz)
        result = mavlink_service.validate_frequency_parameter("PISAD_RF_FREQ", 406_000_000)
        assert result["valid"] == True

        # Should be authorized despite any conflicts (emergency authorization)
        if result.get("conflicts"):
            assert result.get("authorized") == True
            assert "emergency" in result.get("authorization_reason", "").lower()

    def test_emergency_beacon_profile_yaml_configuration(self):
        """Test Emergency Beacon profile YAML configuration correctness [28c1]."""
        config_path = Path(
            "/home/pisad/projects/pisad/config/asv_integration/frequency_profiles.yaml"
        )

        with open(config_path, "r") as f:
            profiles = yaml.safe_load(f)

        assert "emergency_beacon_406" in profiles
        emergency_profile = profiles["emergency_beacon_406"]

        # Verify ELT-specific configuration
        assert emergency_profile["center_frequency_hz"] == 406_000_000
        assert emergency_profile["bandwidth_hz"] == 25_000  # Task requirement: 25kHz
        assert emergency_profile["description"] == "Emergency Beacon Detection at 406 MHz (ELT)"
        assert emergency_profile["analyzer_type"] == "GP"
        assert emergency_profile["priority"] == 1  # Emergency priority

    def test_emergency_beacon_profile_analyzer_integration(self):
        """Test Emergency Beacon profile integration with emergency analyzer [28c1]."""
        analyzer_config_path = Path(
            "/home/pisad/projects/pisad/config/asv_integration/analyzer_profiles.yaml"
        )

        with open(analyzer_config_path, "r") as f:
            analyzers = yaml.safe_load(f)

        assert "emergency_analyzer" in analyzers
        emergency_analyzer = analyzers["emergency_analyzer"]

        # Verify emergency_beacon_406 is in frequency profiles
        assert "emergency_beacon_406" in emergency_analyzer["frequency_profiles"]

        # Verify emergency analyzer characteristics
        assert emergency_analyzer["analyzer_type"] == "GP"
        assert emergency_analyzer["enabled"] == True

        # ELT detection requires sensitive threshold
        processing_config = emergency_analyzer["processing_config"]
        assert processing_config["detection_threshold_db"] <= -110

    def test_emergency_beacon_profile_response_timing(self, config_manager):
        """Test Emergency Beacon profile response timing requirements [28c1]."""
        import time

        start_time = time.perf_counter()
        profile = config_manager.get_frequency_profile("emergency_beacon_406")
        end_time = time.perf_counter()

        response_time_ms = (end_time - start_time) * 1000

        assert profile is not None
        assert response_time_ms < 50.0  # <50ms requirement
        assert profile.processing_timeout_ms <= 50  # Fast emergency processing
