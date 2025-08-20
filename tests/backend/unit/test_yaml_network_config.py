"""
Test suite for YAML configuration parsing with NETWORK_ prefixed settings.

This test suite validates that the YAML configuration parser correctly
handles all NETWORK_ prefixed settings including the enhanced fields.

Test Requirements:
- [8e6b] YAML parser supports enhanced NETWORK_ settings with proper type conversion
- [8e6g] Configuration validation with bounds checking
- [8e6h] Comprehensive configuration loading testing

PRD References:
- PRD-AC5.6.6: Network bandwidth optimization with operator control
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.backend.core.config import ConfigLoader


class TestYAMLNetworkConfigParsing:
    """Test YAML configuration parsing for enhanced NetworkConfig fields."""

    def test_yaml_network_config_parsing(self):
        """Test that YAML parser correctly loads enhanced NETWORK_ settings."""
        # Create test YAML configuration
        test_config = {
            "NETWORK_PACKET_LOSS_LOW_THRESHOLD": 0.005,
            "NETWORK_PACKET_LOSS_MEDIUM_THRESHOLD": 0.02,
            "NETWORK_PACKET_LOSS_HIGH_THRESHOLD": 0.08,
            "NETWORK_PACKET_LOSS_CRITICAL_THRESHOLD": 0.15,
            "NETWORK_RUNTIME_ADJUSTMENT_ENABLED": False,
            "NETWORK_OPERATOR_OVERRIDE_ENABLED": True,
            "NETWORK_MONITORING_INTERVAL_MS": 500,
            "NETWORK_ADAPTIVE_RATE_ENABLED": False,
            "NETWORK_CONGESTION_DETECTOR_ENABLED": True,
            "NETWORK_BASELINE_LATENCY_MS": 10.0,
            "NETWORK_LATENCY_THRESHOLD_MS": 75.0,
        }

        # Create temporary YAML file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(test_config, f)
            temp_yaml_path = f.name

        try:
            # Load configuration from YAML
            loader = ConfigLoader(temp_yaml_path)
            config = loader.load()

            # Verify all NETWORK_ settings were parsed correctly
            assert config.network.NETWORK_PACKET_LOSS_LOW_THRESHOLD == 0.005
            assert config.network.NETWORK_PACKET_LOSS_MEDIUM_THRESHOLD == 0.02
            assert config.network.NETWORK_PACKET_LOSS_HIGH_THRESHOLD == 0.08
            assert config.network.NETWORK_PACKET_LOSS_CRITICAL_THRESHOLD == 0.15

            # Test enhanced fields
            assert config.network.NETWORK_RUNTIME_ADJUSTMENT_ENABLED is False
            assert config.network.NETWORK_OPERATOR_OVERRIDE_ENABLED is True
            assert config.network.NETWORK_MONITORING_INTERVAL_MS == 500
            assert config.network.NETWORK_ADAPTIVE_RATE_ENABLED is False

            # Test existing fields
            assert config.network.NETWORK_CONGESTION_DETECTOR_ENABLED is True
            assert config.network.NETWORK_BASELINE_LATENCY_MS == 10.0
            assert config.network.NETWORK_LATENCY_THRESHOLD_MS == 75.0

        finally:
            # Clean up temporary file
            Path(temp_yaml_path).unlink()

    def test_yaml_type_conversion(self):
        """Test proper type conversion for different NETWORK_ field types."""
        test_config = {
            "NETWORK_PACKET_LOSS_LOW_THRESHOLD": "0.001",  # String -> float
            "NETWORK_MONITORING_INTERVAL_MS": "2000",  # String -> int
            "NETWORK_RUNTIME_ADJUSTMENT_ENABLED": "true",  # String -> bool
            "NETWORK_OPERATOR_OVERRIDE_ENABLED": "false",  # String -> bool
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(test_config, f)
            temp_yaml_path = f.name

        try:
            loader = ConfigLoader(temp_yaml_path)
            config = loader.load()

            # Verify type conversions
            assert isinstance(config.network.NETWORK_PACKET_LOSS_LOW_THRESHOLD, float)
            assert config.network.NETWORK_PACKET_LOSS_LOW_THRESHOLD == 0.001

            assert isinstance(config.network.NETWORK_MONITORING_INTERVAL_MS, int)
            assert config.network.NETWORK_MONITORING_INTERVAL_MS == 2000

            assert isinstance(config.network.NETWORK_RUNTIME_ADJUSTMENT_ENABLED, bool)
            assert config.network.NETWORK_RUNTIME_ADJUSTMENT_ENABLED is True

            assert isinstance(config.network.NETWORK_OPERATOR_OVERRIDE_ENABLED, bool)
            assert config.network.NETWORK_OPERATOR_OVERRIDE_ENABLED is False

        finally:
            Path(temp_yaml_path).unlink()

    def test_yaml_validation_during_parsing(self):
        """Test that YAML parsing triggers validation of packet loss thresholds."""
        # Test invalid threshold values
        invalid_config = {
            "NETWORK_PACKET_LOSS_LOW_THRESHOLD": 0.6,  # Above maximum
            "NETWORK_PACKET_LOSS_MEDIUM_THRESHOLD": 0.05,
            "NETWORK_PACKET_LOSS_HIGH_THRESHOLD": 0.10,
            "NETWORK_PACKET_LOSS_CRITICAL_THRESHOLD": 0.20,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(invalid_config, f)
            temp_yaml_path = f.name

        try:
            loader = ConfigLoader(temp_yaml_path)
            with pytest.raises(
                ValueError, match="Packet loss threshold must be between 0.001 and 0.5"
            ):
                loader.load()
        finally:
            Path(temp_yaml_path).unlink()

    def test_yaml_threshold_ordering_validation(self):
        """Test that YAML parsing validates threshold ordering."""
        # Test invalid ordering
        invalid_config = {
            "NETWORK_PACKET_LOSS_LOW_THRESHOLD": 0.10,  # Higher than medium
            "NETWORK_PACKET_LOSS_MEDIUM_THRESHOLD": 0.05,
            "NETWORK_PACKET_LOSS_HIGH_THRESHOLD": 0.15,
            "NETWORK_PACKET_LOSS_CRITICAL_THRESHOLD": 0.20,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(invalid_config, f)
            temp_yaml_path = f.name

        try:
            loader = ConfigLoader(temp_yaml_path)
            with pytest.raises(
                ValueError, match="Packet loss thresholds must be in ascending order"
            ):
                loader.load()
        finally:
            Path(temp_yaml_path).unlink()

    def test_environment_variable_override(self):
        """Test that environment variables can override YAML network settings."""
        import os

        # Set environment variables
        env_vars = {
            "PISAD_NETWORK_PACKET_LOSS_LOW_THRESHOLD": "0.002",
            "PISAD_NETWORK_RUNTIME_ADJUSTMENT_ENABLED": "false",
            "PISAD_NETWORK_MONITORING_INTERVAL_MS": "3000",
        }

        # Store original values
        original_env = {}
        for key, value in env_vars.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            # Create YAML with different values
            yaml_config = {
                "NETWORK_PACKET_LOSS_LOW_THRESHOLD": 0.01,
                "NETWORK_RUNTIME_ADJUSTMENT_ENABLED": True,
                "NETWORK_MONITORING_INTERVAL_MS": 1000,
            }

            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                yaml.dump(yaml_config, f)
                temp_yaml_path = f.name

            try:
                loader = ConfigLoader(temp_yaml_path)
                config = loader.load()

                # Environment variables should override YAML values
                assert config.network.NETWORK_PACKET_LOSS_LOW_THRESHOLD == 0.002
                assert config.network.NETWORK_RUNTIME_ADJUSTMENT_ENABLED is False
                assert config.network.NETWORK_MONITORING_INTERVAL_MS == 3000

            finally:
                Path(temp_yaml_path).unlink()

        finally:
            # Restore original environment
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
