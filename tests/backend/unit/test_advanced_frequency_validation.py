"""
Test Suite for Advanced Frequency Validation Pipeline (TASK-6.3.1 [28d1-28d4])

Tests for missing advanced frequency validation features:
- HackRF range validation (28d1)
- Frequency conflict detection (28d2)
- RF regulation compliance (28d3)
- ASV-recommended frequency optimization (28d4)
"""

from src.backend.services.asv_integration.asv_configuration_manager import ASVConfigurationManager
from src.backend.services.rf_regulation_validator import RFRegulationValidator
from src.backend.utils.frequency_conflict_detector import FrequencyConflictDetector


class TestAdvancedFrequencyValidation:
    """Test suite for advanced frequency validation pipeline."""

    def test_hackrf_range_validation_effective_range(self):
        """Test HackRF effective range validation (24MHz-1.75GHz) [28d1]."""
        validator = RFRegulationValidator()

        # Test frequencies within HackRF effective range
        result_162 = validator.validate_hackrf_range(162_025_000)
        assert result_162["valid"] == True
        assert result_162["range"] == "effective"
        assert result_162["performance"] == "optimal"

        result_406 = validator.validate_hackrf_range(406_000_000)
        assert result_406["valid"] == True
        assert result_406["range"] == "effective"
        assert result_406["performance"] == "optimal"

        result_121 = validator.validate_hackrf_range(121_500_000)
        assert result_121["valid"] == True
        assert result_121["range"] == "effective"
        assert result_121["performance"] == "optimal"

        # Test frequencies outside effective range but within absolute range
        result_low = validator.validate_hackrf_range(5_000_000)  # 5 MHz
        assert result_low["valid"] == True
        assert result_low["range"] == "extended"
        assert "reduced performance" in result_low["warning"]

        result_high = validator.validate_hackrf_range(3_000_000_000)  # 3 GHz
        assert result_high["valid"] == True
        assert result_high["range"] == "extended"
        assert "reduced performance" in result_high["warning"]

        # Test frequencies completely outside HackRF range
        result_invalid = validator.validate_hackrf_range(500_000)  # 500 kHz
        assert result_invalid["valid"] == False
        assert "outside HackRF range" in result_invalid["error"]

    def test_frequency_conflict_detection(self):
        """Test frequency conflict detection with existing radio systems [28d2]."""
        detector = FrequencyConflictDetector()

        # Test conflict with aviation navigation (VOR range)
        vor_conflict = detector.detect_conflicts(108_500_000, 25_000)
        assert vor_conflict["conflicts"] == True
        assert "VOR navigation" in vor_conflict["conflicting_services"]

        # Test conflict with FM broadcast
        fm_conflict = detector.detect_conflicts(101_500_000, 200_000)
        assert fm_conflict["conflicts"] == True
        assert "FM broadcast" in fm_conflict["conflicting_services"]

        # Test maritime SAR frequency (has authorized conflicts within maritime VHF)
        maritime_sar = detector.detect_conflicts(162_025_000, 25_000)
        assert maritime_sar["conflicts"] == True  # Expected conflicts with maritime VHF
        assert "maritime_vhf" in maritime_sar["conflicting_services"]
        assert maritime_sar["severity"] == "high"  # High protection level

    def test_rf_regulation_compliance_us(self):
        """Test RF regulation compliance for US frequency allocations [28d3]."""
        validator = RFRegulationValidator()

        # Test maritime emergency frequency (should be compliant)
        maritime_result = validator.validate_us_compliance(162_025_000, "maritime_emergency")
        assert maritime_result["compliant"] == True
        assert maritime_result["allocation"] == "maritime_mobile"

        # Test aviation emergency (should be compliant)
        aviation_result = validator.validate_us_compliance(121_500_000, "aviation_emergency")
        assert aviation_result["compliant"] == True
        assert aviation_result["allocation"] == "aeronautical_mobile"

        # Test restricted frequency
        restricted_result = validator.validate_us_compliance(155_000_000, "custom")
        assert restricted_result["compliant"] == False
        assert "License required" in restricted_result["recommendation"]

    def test_asv_frequency_optimization(self):
        """Test ASV-recommended frequency optimization [28d4]."""
        config_manager = ASVConfigurationManager()

        # Test optimization for emergency beacon detection
        optimization = config_manager.get_frequency_optimization("emergency_beacon_406")
        assert optimization["recommended_frequency"] == 406_000_000
        assert optimization["optimal_bandwidth"] == 50_000
        assert optimization["confidence"] > 0.8

        # Test optimization for maritime SAR
        maritime_opt = config_manager.get_frequency_optimization("maritime_sar_162")
        assert maritime_opt["recommended_frequency"] == 162_025_000
        assert maritime_opt["optimal_bandwidth"] == 25_000

        # Test optimization suggests better frequency for custom profile
        custom_opt = config_manager.get_frequency_optimization_for_frequency(2_400_000_000)
        assert "alternative_frequencies" in custom_opt
        assert len(custom_opt["alternative_frequencies"]) > 0

    def test_validation_pipeline_integration(self):
        """Test complete validation pipeline integration."""
        config_manager = ASVConfigurationManager()

        # Test complete validation for maritime frequency
        validation_result = config_manager.validate_frequency_complete(162_025_000, "maritime_sar")

        assert validation_result["hackrf_validation"]["valid"] == True
        assert (
            validation_result["conflict_detection"]["conflicts"] == True
        )  # Expected maritime VHF conflicts
        assert validation_result["regulation_compliance"]["compliant"] == True
        assert validation_result["optimization"]["confidence"] > 0.7
        # Should still be approved despite conflicts due to maritime emergency authorization
        assert validation_result["overall_status"] in ["approved", "conditional"]

    def test_validation_pipeline_with_conflicts(self):
        """Test validation pipeline properly detects and reports conflicts."""
        config_manager = ASVConfigurationManager()

        # Test frequency with known conflicts (FM broadcast range)
        validation_result = config_manager.validate_frequency_complete(101_500_000, "custom")

        assert validation_result["conflict_detection"]["conflicts"] == True
        assert validation_result["regulation_compliance"]["compliant"] == False
        assert validation_result["overall_status"] == "rejected"
        assert len(validation_result["warnings"]) > 0
