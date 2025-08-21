"""
Test Suite for Parameter Validation Pipeline (TASK-6.3.1 [28b2])

Tests for parameter validation pipeline with frequency range checking and conflict resolution:
- Parameter validation with frequency range checking
- Conflict resolution integration
- MAVLink parameter validation pipeline
- Error handling and response validation
"""

import pytest

from src.backend.services.mavlink_service import MAVLinkService
from src.backend.services.parameter_storage import ParameterStorage


class TestParameterValidationPipeline:
    """Test suite for parameter validation pipeline."""

    @pytest.fixture
    def mavlink_service(self):
        """Create MAVLink service for testing."""
        service = MAVLinkService()
        service._parameters = {}
        return service

    @pytest.fixture
    def parameter_storage(self):
        """Create parameter storage for testing."""
        return ParameterStorage()

    def test_frequency_parameter_validation_valid_range(self, mavlink_service):
        """Test frequency parameter validation with valid HackRF range."""
        # Test valid frequencies
        valid_frequencies = [
            162_025_000,  # Maritime SAR
            406_000_000,  # Emergency beacon
            121_500_000,  # Aviation emergency
            100_000_000,  # Valid HackRF range
        ]

        for freq in valid_frequencies:
            result = mavlink_service.validate_frequency_parameter("PISAD_RF_FREQ", freq)
            assert result["valid"] == True
            assert "error" not in result

    def test_frequency_parameter_validation_invalid_range(self, mavlink_service):
        """Test frequency parameter validation with invalid range."""
        # Test invalid frequencies
        invalid_frequencies = [
            500_000,  # Below HackRF minimum
            7_000_000_000,  # Above HackRF maximum
            0,  # Zero frequency
            -1_000_000,  # Negative frequency
        ]

        for freq in invalid_frequencies:
            result = mavlink_service.validate_frequency_parameter("PISAD_RF_FREQ", freq)
            assert result["valid"] == False
            assert "error" in result
            assert "frequency range" in result["error"].lower()

    def test_frequency_conflict_resolution_integration(self, mavlink_service):
        """Test frequency conflict resolution integration."""
        # Test frequency with known conflicts
        fm_broadcast_freq = 101_500_000  # Known to conflict with FM broadcast

        result = mavlink_service.validate_frequency_parameter("PISAD_RF_FREQ", fm_broadcast_freq)

        # Should detect conflicts but provide recommendations
        assert "conflicts" in result
        assert result["conflicts"] == True
        assert "recommendations" in result
        assert len(result["recommendations"]) > 0

    def test_frequency_conflict_resolution_emergency_authorization(self, mavlink_service):
        """Test emergency frequency authorization in conflict resolution."""
        # Test maritime SAR frequency (authorized despite conflicts)
        maritime_freq = 162_025_000

        result = mavlink_service.validate_frequency_parameter("PISAD_RF_FREQ", maritime_freq)

        # Should have conflicts but be authorized for emergency use
        assert result["valid"] == True
        assert result.get("conflicts") == True
        assert result.get("authorized") == True
        assert "emergency" in result.get("authorization_reason", "").lower()

    def test_parameter_validation_pipeline_integration(self, mavlink_service):
        """Test complete parameter validation pipeline."""
        # Test full validation pipeline
        test_params = {
            "PISAD_RF_FREQ": 162_025_000,
            "PISAD_RF_PROFILE": 2,  # SAR profile
            "PISAD_RF_BW": 25_000,
        }

        result = mavlink_service.validate_parameter_set(test_params)

        assert result["valid"] == True
        assert "validation_results" in result
        assert len(result["validation_results"]) == len(test_params)

    def test_parameter_validation_response_timing(self, mavlink_service):
        """Test parameter validation response timing (<50ms requirement)."""
        import time

        start_time = time.perf_counter()
        result = mavlink_service.validate_frequency_parameter("PISAD_RF_FREQ", 406_000_000)
        end_time = time.perf_counter()

        response_time_ms = (end_time - start_time) * 1000

        assert result["valid"] == True
        assert response_time_ms < 50.0  # <50ms requirement
        assert "response_time_ms" in result
        assert result["response_time_ms"] < 50.0

    def test_parameter_validation_error_handling(self, mavlink_service):
        """Test parameter validation error handling."""
        # Test invalid parameter names
        result = mavlink_service.validate_frequency_parameter("INVALID_PARAM", 162_025_000)
        assert result["valid"] == False
        assert "unknown parameter" in result["error"].lower()

        # Test invalid data types
        result = mavlink_service.validate_frequency_parameter("PISAD_RF_FREQ", "invalid")
        assert result["valid"] == False
        assert "invalid type" in result["error"].lower()

    def test_parameter_validation_conflict_recommendations(self, mavlink_service):
        """Test parameter validation provides conflict recommendations."""
        # Test frequency with conflicts
        conflict_freq = 108_500_000  # VOR navigation range

        result = mavlink_service.validate_frequency_parameter("PISAD_RF_FREQ", conflict_freq)

        assert "recommendations" in result
        recommendations = result["recommendations"]
        assert len(recommendations) > 0

        # Should suggest alternative frequencies
        alternative_found = False
        for rec in recommendations:
            if "alternative" in rec.lower() or "consider" in rec.lower():
                alternative_found = True
                break
        assert alternative_found == True

    def test_parameter_validation_persistence_integration(self, parameter_storage):
        """Test parameter validation integration with persistence."""
        # Test validation before persistence
        valid_params = {
            "PISAD_RF_FREQ": 162_025_000,
            "PISAD_RF_PROFILE": 2,
        }

        # Should validate before storing
        result = parameter_storage.store_parameters_validated(valid_params)
        assert result["success"] == True
        assert result["validation_passed"] == True

        # Test invalid parameters are rejected
        invalid_params = {
            "PISAD_RF_FREQ": 500_000,  # Invalid range
        }

        result = parameter_storage.store_parameters_validated(invalid_params)
        assert result["success"] == False
        assert result["validation_passed"] == False
        assert "validation_errors" in result
