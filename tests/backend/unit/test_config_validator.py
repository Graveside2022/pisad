"""
Tests for configuration validation system.
"""

import tempfile
from pathlib import Path

import pytest

from src.backend.core.config_validator import ConfigValidator, validate_startup_config


class TestConfigValidator:
    """Test configuration validation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ConfigValidator()

    def test_valid_configuration_passes(self):
        """Test that valid configuration passes validation."""
        valid_config = {
            "APP_NAME": "PISAD",
            "APP_VERSION": "1.0.0",
            "APP_ENV": "development",
            "APP_HOST": "0.0.0.0",
            "APP_PORT": 8080,
            "SDR_FREQUENCY": 3200000000,
            "SDR_SAMPLE_RATE": 2048000,
            "SDR_GAIN": 30,
            "LOG_LEVEL": "INFO",
            "LOG_FILE_PATH": "logs/pisad.log",
        }

        is_valid, errors = self.validator.validate_config_dict(valid_config)
        assert is_valid
        assert len(errors) == 0

    def test_missing_required_field_fails(self):
        """Test that missing required fields cause validation failure."""
        invalid_config = {
            "APP_NAME": "PISAD",
            # Missing APP_VERSION (required)
            "APP_ENV": "development",
            "APP_HOST": "0.0.0.0",
            "APP_PORT": 8080,
        }

        is_valid, errors = self.validator.validate_config_dict(invalid_config)
        assert not is_valid
        assert any("APP_VERSION" in error for error in errors)

    def test_invalid_data_type_fails(self):
        """Test that invalid data types cause validation failure."""
        invalid_config = {
            "APP_NAME": "PISAD",
            "APP_VERSION": "1.0.0",
            "APP_ENV": "development",
            "APP_HOST": "0.0.0.0",
            "APP_PORT": "invalid_port",  # Should be integer
            "SDR_FREQUENCY": 3200000000,
            "SDR_SAMPLE_RATE": 2048000,
            "SDR_GAIN": 30,
            "LOG_LEVEL": "INFO",
            "LOG_FILE_PATH": "logs/pisad.log",
        }

        is_valid, errors = self.validator.validate_config_dict(invalid_config)
        assert not is_valid
        assert any("APP_PORT" in error for error in errors)

    def test_out_of_range_values_fail(self):
        """Test that out-of-range values cause validation failure."""
        invalid_config = {
            "APP_NAME": "PISAD",
            "APP_VERSION": "1.0.0",
            "APP_ENV": "development",
            "APP_HOST": "0.0.0.0",
            "APP_PORT": 8080,
            "SDR_FREQUENCY": 100000000,  # Below minimum (850MHz)
            "SDR_SAMPLE_RATE": 2048000,
            "SDR_GAIN": 30,
            "LOG_LEVEL": "INFO",
            "LOG_FILE_PATH": "logs/pisad.log",
        }

        is_valid, errors = self.validator.validate_config_dict(invalid_config)
        assert not is_valid
        assert any("SDR_FREQUENCY" in error for error in errors)

    def test_packet_loss_threshold_ordering(self):
        """Test that packet loss thresholds must be in ascending order."""
        invalid_config = {
            "APP_NAME": "PISAD",
            "APP_VERSION": "1.0.0",
            "APP_ENV": "development",
            "APP_HOST": "0.0.0.0",
            "APP_PORT": 8080,
            "SDR_FREQUENCY": 3200000000,
            "SDR_SAMPLE_RATE": 2048000,
            "SDR_GAIN": 30,
            "LOG_LEVEL": "INFO",
            "LOG_FILE_PATH": "logs/pisad.log",
            "NETWORK_PACKET_LOSS_LOW_THRESHOLD": 0.10,  # Higher than medium
            "NETWORK_PACKET_LOSS_MEDIUM_THRESHOLD": 0.05,
            "NETWORK_PACKET_LOSS_HIGH_THRESHOLD": 0.15,
            "NETWORK_PACKET_LOSS_CRITICAL_THRESHOLD": 0.20,
        }

        is_valid, errors = self.validator.validate_parameter_ranges(invalid_config)
        assert not is_valid
        assert any("ascending order" in error for error in errors)

    def test_velocity_constraints(self):
        """Test that velocity constraints are properly validated."""
        invalid_config = {
            "APP_NAME": "PISAD",
            "APP_VERSION": "1.0.0",
            "APP_ENV": "development",
            "APP_HOST": "0.0.0.0",
            "APP_PORT": 8080,
            "SDR_FREQUENCY": 3200000000,
            "SDR_SAMPLE_RATE": 2048000,
            "SDR_GAIN": 30,
            "LOG_LEVEL": "INFO",
            "LOG_FILE_PATH": "logs/pisad.log",
            "HOMING_APPROACH_VELOCITY": 10.0,  # Higher than max
            "HOMING_FORWARD_VELOCITY_MAX": 5.0,
        }

        is_valid, errors = self.validator.validate_parameter_ranges(invalid_config)
        assert not is_valid
        assert any("HOMING_APPROACH_VELOCITY" in error for error in errors)

    def test_yaml_file_validation_success(self):
        """Test successful YAML file validation."""
        valid_yaml_content = """
APP_NAME: "PISAD"
APP_VERSION: "1.0.0"
APP_ENV: "development"
APP_HOST: "0.0.0.0"
APP_PORT: 8080
SDR_FREQUENCY: 3200000000
SDR_SAMPLE_RATE: 2048000
SDR_GAIN: 30
LOG_LEVEL: "INFO"
LOG_FILE_PATH: "logs/pisad.log"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(valid_yaml_content)
            f.flush()

            is_valid, errors = self.validator.validate_yaml_file(Path(f.name))
            assert is_valid
            assert len(errors) == 0

            # Cleanup
            Path(f.name).unlink()

    def test_yaml_syntax_error_detection(self):
        """Test detection of YAML syntax errors."""
        invalid_yaml_content = """
APP_NAME: "PISAD"
APP_VERSION: "1.0.0"
APP_ENV: "development"
INVALID_YAML: [unclosed bracket
APP_PORT: 8080
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(invalid_yaml_content)
            f.flush()

            is_valid, errors = self.validator.validate_yaml_file(Path(f.name))
            assert not is_valid
            assert any("YAML syntax error" in error for error in errors)

            # Cleanup
            Path(f.name).unlink()

    def test_file_not_found_error(self):
        """Test handling of missing configuration files."""
        nonexistent_path = Path("/nonexistent/config.yaml")

        is_valid, errors = self.validator.validate_yaml_file(nonexistent_path)
        assert not is_valid
        assert any("not found" in error for error in errors)

    def test_startup_validation_success(self):
        """Test successful startup configuration validation."""
        valid_yaml_content = """
APP_NAME: "PISAD"
APP_VERSION: "1.0.0"
APP_ENV: "development"
APP_HOST: "0.0.0.0"
APP_PORT: 8080
SDR_FREQUENCY: 3200000000
SDR_SAMPLE_RATE: 2048000
SDR_GAIN: 30
LOG_LEVEL: "INFO"
LOG_FILE_PATH: "logs/pisad.log"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(valid_yaml_content)
            f.flush()

            # Should not raise exception
            validate_startup_config(Path(f.name))

            # Cleanup
            Path(f.name).unlink()

    def test_startup_validation_failure(self):
        """Test startup validation failure with invalid configuration."""
        invalid_yaml_content = """
APP_NAME: "PISAD"
# Missing required fields
APP_PORT: "invalid_port"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(invalid_yaml_content)
            f.flush()

            with pytest.raises(ValueError, match="Configuration validation failed"):
                validate_startup_config(Path(f.name))

            # Cleanup
            Path(f.name).unlink()
