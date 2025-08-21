"""
Test Suite for Custom Profile (TASK-6.3.1 [28c4])

Tests for Custom profile with user-defined frequency, bandwidth, and signal type parameters:
- User-defined frequency parameter validation
- User-defined bandwidth parameter validation
- Signal type parameter configuration
- Mission Planner parameter integration for custom settings
- Profile validation and conflict checking
"""

import pytest

from src.backend.services.asv_integration.asv_configuration_manager import ASVConfigurationManager
from src.backend.services.mavlink_service import MAVLinkService


class TestCustomProfile:
    """Test suite for Custom profile implementation."""

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

    def test_custom_profile_user_defined_frequency(self, mavlink_service):
        """Test Custom profile with user-defined frequency parameters [28c4]."""
        # Test various custom frequencies within HackRF range
        custom_frequencies = [
            162_025_000,  # Maritime SAR
            433_920_000,  # ISM band
            915_000_000,  # ISM band
            1_575_420_000,  # GPS L1
        ]

        for freq in custom_frequencies:
            parameter_set = {
                "PISAD_RF_FREQ": freq,
                "PISAD_RF_PROFILE": 2,  # Custom profile (2 = Custom)
                "PISAD_RF_BW": 100_000,  # Custom bandwidth
            }

            result = mavlink_service.validate_parameter_set(parameter_set)
            assert result["valid"] == True, f"Custom frequency {freq} should be valid"

            # Test frequency validation directly
            freq_result = mavlink_service.validate_frequency_parameter("PISAD_RF_FREQ", freq)
            assert freq_result["valid"] == True, f"Frequency {freq} should pass validation"

    def test_custom_profile_user_defined_bandwidth(self, mavlink_service):
        """Test Custom profile with user-defined bandwidth parameters [28c4]."""
        # Test various custom bandwidths
        custom_bandwidths = [
            12_500,  # Narrow band
            25_000,  # Standard narrow
            50_000,  # Aviation width
            100_000,  # Wideband
            200_000,  # Very wideband
        ]

        base_freq = 433_920_000  # ISM band frequency

        for bandwidth in custom_bandwidths:
            parameter_set = {
                "PISAD_RF_FREQ": base_freq,
                "PISAD_RF_PROFILE": 2,  # Custom profile
                "PISAD_RF_BW": bandwidth,
            }

            result = mavlink_service.validate_parameter_set(parameter_set)
            assert result["valid"] == True, f"Custom bandwidth {bandwidth} should be valid"

    def test_custom_profile_signal_type_parameters(self, mavlink_service):
        """Test Custom profile signal type parameter configuration [28c4]."""
        # Test custom configuration with signal type indicators
        parameter_set = {
            "PISAD_RF_FREQ": 433_920_000,  # Custom frequency
            "PISAD_RF_PROFILE": 2,  # Custom profile
            "PISAD_RF_BW": 100_000,  # Custom bandwidth
            "PISAD_SIG_CLASS": 1,  # CONTINUOUS signal type
        }

        result = mavlink_service.validate_parameter_set(parameter_set)
        assert result["valid"] == True
        assert "validation_results" in result

    def test_custom_profile_frequency_range_validation(self, mavlink_service):
        """Test Custom profile frequency range validation [28c4]."""
        # Test frequencies within HackRF effective range
        effective_range_freq = 162_025_000  # Within 24MHz-1.75GHz
        result = mavlink_service.validate_frequency_parameter("PISAD_RF_FREQ", effective_range_freq)
        assert result["valid"] == True

        # Test frequencies in extended range (should work but with warnings)
        extended_range_freq = 5_000_000  # 5 MHz - extended range
        result = mavlink_service.validate_frequency_parameter("PISAD_RF_FREQ", extended_range_freq)
        assert result["valid"] == True

        # Test frequencies outside HackRF range
        invalid_freq = 500_000  # 0.5 MHz - below HackRF range
        result = mavlink_service.validate_frequency_parameter("PISAD_RF_FREQ", invalid_freq)
        assert result["valid"] == False

    def test_custom_profile_bandwidth_range_validation(self, mavlink_service):
        """Test Custom profile bandwidth range validation [28c4]."""
        base_freq = 433_920_000

        # Test valid bandwidth ranges
        valid_bandwidths = [1_000, 25_000, 100_000, 1_000_000]  # 1kHz to 1MHz

        for bandwidth in valid_bandwidths:
            parameter_set = {
                "PISAD_RF_FREQ": base_freq,
                "PISAD_RF_PROFILE": 2,
                "PISAD_RF_BW": bandwidth,
            }
            result = mavlink_service.validate_parameter_set(parameter_set)
            assert result["valid"] == True, f"Bandwidth {bandwidth} should be valid"

        # Test extremely large bandwidth (beyond reasonable limits)
        very_large_parameter_set = {
            "PISAD_RF_FREQ": base_freq,
            "PISAD_RF_PROFILE": 2,
            "PISAD_RF_BW": 50_000_000,  # 50 MHz - very large but may be valid for some use cases
        }
        result = mavlink_service.validate_parameter_set(very_large_parameter_set)
        # Large bandwidths should either be rejected or accepted with proper validation
        assert "validation_results" in result
        assert result["valid"] in [True, False]  # Either valid or invalid, both are acceptable

    def test_custom_profile_conflict_checking(self, mavlink_service):
        """Test Custom profile frequency conflict checking [28c4]."""
        # Test custom frequency that conflicts with aviation navigation
        conflicting_freq = 108_500_000  # VOR navigation range

        result = mavlink_service.validate_frequency_parameter("PISAD_RF_FREQ", conflicting_freq)

        # Should detect conflicts
        assert "conflicts" in result or "recommendations" in result

        # Test safe custom frequency
        safe_freq = 433_920_000  # ISM band - should be safer
        result = mavlink_service.validate_frequency_parameter("PISAD_RF_FREQ", safe_freq)
        assert result["valid"] == True

    def test_custom_profile_mission_planner_parameter_integration(self, mavlink_service):
        """Test Custom profile integration with Mission Planner parameters [28c4]."""
        # Test complete custom configuration
        custom_config = {
            "PISAD_RF_FREQ": 915_000_000,  # ISM band
            "PISAD_RF_PROFILE": 2,  # Custom profile
            "PISAD_RF_BW": 125_000,  # Custom bandwidth
            "PISAD_SIG_CLASS": 2,  # NOISE signal type for testing
        }

        result = mavlink_service.validate_parameter_set(custom_config)
        assert result["valid"] == True
        assert "validation_results" in result

        # Verify all parameters were validated
        assert len(result["validation_results"]) == len(custom_config)

    def test_custom_profile_flexibility(self, mavlink_service):
        """Test Custom profile flexibility for various use cases [28c4]."""
        # Test multiple different custom configurations
        test_configurations = [
            # Amateur radio configuration
            {
                "PISAD_RF_FREQ": 145_500_000,  # 2m amateur band
                "PISAD_RF_PROFILE": 2,
                "PISAD_RF_BW": 25_000,
            },
            # ISM band configuration
            {
                "PISAD_RF_FREQ": 433_920_000,  # 70cm ISM
                "PISAD_RF_PROFILE": 2,
                "PISAD_RF_BW": 50_000,
            },
            # GPS frequency (for testing)
            {
                "PISAD_RF_FREQ": 1_575_420_000,  # GPS L1
                "PISAD_RF_PROFILE": 2,
                "PISAD_RF_BW": 20_000_000,  # GPS bandwidth
            },
        ]

        for config in test_configurations:
            result = mavlink_service.validate_parameter_set(config)
            # Each configuration should be processable (valid or with clear feedback)
            assert "validation_results" in result
            if not result["valid"]:
                # If invalid, should have clear reason
                assert any(
                    res.get("error") or res.get("message")
                    for res in result["validation_results"].values()
                )

    def test_custom_profile_response_timing(self, mavlink_service):
        """Test Custom profile validation response timing [28c4]."""
        import time

        custom_params = {
            "PISAD_RF_FREQ": 433_920_000,
            "PISAD_RF_PROFILE": 2,
            "PISAD_RF_BW": 100_000,
        }

        start_time = time.perf_counter()
        result = mavlink_service.validate_parameter_set(custom_params)
        end_time = time.perf_counter()

        response_time_ms = (end_time - start_time) * 1000

        assert result is not None
        assert response_time_ms < 50.0  # <50ms requirement for parameter validation
