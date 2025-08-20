"""
Configuration validation for PISAD application.
Provides JSON schema validation for YAML configuration files.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import jsonschema
import yaml

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates configuration files against JSON schemas."""

    def __init__(self) -> None:
        """Initialize the configuration validator."""
        self.schemas = self._load_schemas()

    def _load_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Load JSON schema definitions for configuration validation."""

        # Main configuration schema
        main_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                # Application settings
                "APP_NAME": {"type": "string", "minLength": 1},
                "APP_VERSION": {"type": "string", "pattern": r"^\d+\.\d+\.\d+$"},
                "APP_ENV": {"type": "string", "enum": ["development", "production", "testing"]},
                "APP_HOST": {"type": "string", "format": "ipv4"},
                "APP_PORT": {"type": "integer", "minimum": 1, "maximum": 65535},
                # SDR Configuration
                "SDR_FREQUENCY": {"type": "integer", "minimum": 850000000, "maximum": 6500000000},
                "SDR_SAMPLE_RATE": {"type": "integer", "minimum": 1000000, "maximum": 50000000},
                "SDR_GAIN": {"type": "integer", "minimum": 0, "maximum": 100},
                "SDR_PPM_CORRECTION": {"type": "integer", "minimum": -100, "maximum": 100},
                "SDR_DEVICE_INDEX": {"type": "integer", "minimum": 0, "maximum": 10},
                "SDR_BUFFER_SIZE": {"type": "integer", "minimum": 1024, "maximum": 1048576},
                # Signal Processing
                "SIGNAL_RSSI_THRESHOLD": {"type": "number", "minimum": -120, "maximum": 0},
                "SIGNAL_AVERAGING_WINDOW": {"type": "integer", "minimum": 1, "maximum": 100},
                "SIGNAL_MIN_DURATION_MS": {"type": "integer", "minimum": 1, "maximum": 10000},
                "SIGNAL_MAX_GAP_MS": {"type": "integer", "minimum": 1, "maximum": 1000},
                # Logging Configuration
                "LOG_LEVEL": {
                    "type": "string",
                    "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                },
                "LOG_FORMAT": {"type": "string", "minLength": 10},
                "LOG_FILE_PATH": {"type": "string", "minLength": 1},
                "LOG_FILE_MAX_BYTES": {
                    "type": "integer",
                    "minimum": 1048576,
                    "maximum": 1073741824,
                },
                "LOG_FILE_BACKUP_COUNT": {"type": "integer", "minimum": 1, "maximum": 50},
                "LOG_ENABLE_CONSOLE": {"type": "boolean"},
                "LOG_ENABLE_FILE": {"type": "boolean"},
                "LOG_ENABLE_JOURNAL": {"type": "boolean"},
                # Safety Configuration
                "SAFETY_VELOCITY_MAX_MPS": {"type": "number", "minimum": 0.1, "maximum": 15.0},
                "SAFETY_INTERLOCK_ENABLED": {"type": "boolean"},
                "SAFETY_EMERGENCY_STOP_GPIO": {"type": "integer", "minimum": 1, "maximum": 40},
                # Network Configuration
                "NETWORK_PACKET_LOSS_LOW_THRESHOLD": {
                    "type": "number",
                    "minimum": 0.001,
                    "maximum": 0.5,
                },
                "NETWORK_PACKET_LOSS_MEDIUM_THRESHOLD": {
                    "type": "number",
                    "minimum": 0.001,
                    "maximum": 0.5,
                },
                "NETWORK_PACKET_LOSS_HIGH_THRESHOLD": {
                    "type": "number",
                    "minimum": 0.001,
                    "maximum": 0.5,
                },
                "NETWORK_PACKET_LOSS_CRITICAL_THRESHOLD": {
                    "type": "number",
                    "minimum": 0.001,
                    "maximum": 0.5,
                },
                "NETWORK_LATENCY_THRESHOLD_MS": {
                    "type": "number",
                    "minimum": 1.0,
                    "maximum": 10000.0,
                },
                # Homing Algorithm Configuration
                "HOMING_FORWARD_VELOCITY_MAX": {"type": "number", "minimum": 0.1, "maximum": 15.0},
                "HOMING_YAW_RATE_MAX": {"type": "number", "minimum": 0.1, "maximum": 2.0},
                "HOMING_APPROACH_VELOCITY": {"type": "number", "minimum": 0.1, "maximum": 5.0},
                "HOMING_SIGNAL_LOSS_TIMEOUT": {"type": "number", "minimum": 1.0, "maximum": 60.0},
                "HOMING_ALGORITHM_MODE": {"type": "string", "enum": ["SIMPLE", "GRADIENT"]},
                "HOMING_GRADIENT_WINDOW_SIZE": {"type": "integer", "minimum": 3, "maximum": 100},
                "HOMING_GRADIENT_MIN_SNR": {"type": "number", "minimum": 1.0, "maximum": 50.0},
            },
            "required": [
                "APP_NAME",
                "APP_VERSION",
                "APP_ENV",
                "APP_HOST",
                "APP_PORT",
                "SDR_FREQUENCY",
                "SDR_SAMPLE_RATE",
                "SDR_GAIN",
                "LOG_LEVEL",
                "LOG_FILE_PATH",
            ],
            "additionalProperties": True,  # Allow additional config keys
        }

        return {"main": main_schema}

    def validate_yaml_file(self, file_path: Path) -> tuple[bool, list[str]]:
        """
        Validate a YAML configuration file against its schema.

        Args:
            file_path: Path to the YAML file to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        try:
            # Check if file exists
            if not file_path.exists():
                errors.append(f"Configuration file not found: {file_path}")
                return False, errors

            # Load and parse YAML
            with open(file_path, "r") as f:
                try:
                    config_data = yaml.safe_load(f)
                except yaml.YAMLError as e:
                    # Handle cases where problem_mark might not be available
                    line_info = "unknown"
                    if hasattr(e, "problem_mark") and e.problem_mark:
                        line_info = str(e.problem_mark.line + 1)
                    errors.append(f"YAML syntax error at line {line_info}: {e}")
                    return False, errors

            # Validate against schema
            schema = self.schemas["main"]
            try:
                jsonschema.validate(config_data, schema)
                logger.info(f"Configuration file validation passed: {file_path}")
                return True, []

            except jsonschema.ValidationError as e:
                error_path = (
                    " -> ".join(str(p) for p in e.absolute_path) if e.absolute_path else "root"
                )
                errors.append(f"Validation error at {error_path}: {e.message}")
                return False, errors

            except jsonschema.SchemaError as e:
                errors.append(f"Schema error: {e.message}")
                return False, errors

        except Exception as e:
            errors.append(f"Unexpected error validating configuration: {str(e)}")
            return False, errors

    def validate_config_dict(self, config_data: Dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validate configuration data dictionary against schema.

        Args:
            config_data: Configuration dictionary to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        try:
            schema = self.schemas["main"]
            jsonschema.validate(config_data, schema)
            return True, []

        except jsonschema.ValidationError as e:
            error_path = " -> ".join(str(p) for p in e.absolute_path) if e.absolute_path else "root"
            errors.append(f"Validation error at {error_path}: {e.message}")
            return False, errors

        except jsonschema.SchemaError as e:
            errors.append(f"Schema error: {e.message}")
            return False, errors

    def validate_parameter_ranges(self, config_data: Dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Perform additional parameter range validation beyond schema.

        Args:
            config_data: Configuration dictionary to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Validate packet loss threshold ordering
        if all(
            key in config_data
            for key in [
                "NETWORK_PACKET_LOSS_LOW_THRESHOLD",
                "NETWORK_PACKET_LOSS_MEDIUM_THRESHOLD",
                "NETWORK_PACKET_LOSS_HIGH_THRESHOLD",
                "NETWORK_PACKET_LOSS_CRITICAL_THRESHOLD",
            ]
        ):
            thresholds = [
                config_data["NETWORK_PACKET_LOSS_LOW_THRESHOLD"],
                config_data["NETWORK_PACKET_LOSS_MEDIUM_THRESHOLD"],
                config_data["NETWORK_PACKET_LOSS_HIGH_THRESHOLD"],
                config_data["NETWORK_PACKET_LOSS_CRITICAL_THRESHOLD"],
            ]

            if not (thresholds[0] < thresholds[1] < thresholds[2] < thresholds[3]):
                errors.append(
                    "Packet loss thresholds must be in ascending order: "
                    f"low({thresholds[0]}) < medium({thresholds[1]}) < "
                    f"high({thresholds[2]}) < critical({thresholds[3]})"
                )

        # Validate velocity constraints
        if (
            "HOMING_APPROACH_VELOCITY" in config_data
            and "HOMING_FORWARD_VELOCITY_MAX" in config_data
        ):
            if config_data["HOMING_APPROACH_VELOCITY"] > config_data["HOMING_FORWARD_VELOCITY_MAX"]:
                errors.append(
                    "HOMING_APPROACH_VELOCITY must not exceed HOMING_FORWARD_VELOCITY_MAX"
                )

        # Validate safety velocity limits
        if (
            "SAFETY_VELOCITY_MAX_MPS" in config_data
            and "HOMING_FORWARD_VELOCITY_MAX" in config_data
        ):
            if config_data["HOMING_FORWARD_VELOCITY_MAX"] > config_data["SAFETY_VELOCITY_MAX_MPS"]:
                errors.append("HOMING_FORWARD_VELOCITY_MAX must not exceed SAFETY_VELOCITY_MAX_MPS")

        return len(errors) == 0, errors


def validate_startup_config(config_path: Path | None = None) -> None:
    """
    Validate configuration on application startup.

    Args:
        config_path: Optional path to configuration file

    Raises:
        ValueError: If configuration validation fails
    """
    if config_path is None:
        # Default configuration path
        project_root = Path(__file__).parent.parent.parent.parent
        config_path = project_root / "config" / "default.yaml"

    validator = ConfigValidator()
    is_valid, errors = validator.validate_yaml_file(config_path)

    if not is_valid:
        error_msg = f"Configuration validation failed for {config_path}:\n" + "\n".join(
            f"  - {error}" for error in errors
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("Configuration validation completed successfully")
