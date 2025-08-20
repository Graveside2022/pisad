"""
Test suite for default.yaml loading with enhanced network configuration.

This test validates that the default.yaml template loads correctly
with all network configuration settings and proper validation.

Test Requirements:
- [8e6e] YAML configuration template loads with network threshold defaults
- [8e6g] Configuration validation works with template values
- [8e6h] Comprehensive configuration loading testing

PRD References:
- PRD-AC5.6.6: Network bandwidth optimization with operator control
"""

from pathlib import Path

import pytest

from src.backend.core.config import ConfigLoader


class TestDefaultYAMLLoading:
    """Test default.yaml template loading with network configuration."""

    def test_default_yaml_loads_successfully(self):
        """Test that default.yaml loads without errors."""
        # Use the actual default.yaml file
        project_root = Path(__file__).parent.parent.parent.parent
        default_yaml_path = project_root / "config" / "default.yaml"

        assert default_yaml_path.exists(), f"default.yaml not found at {default_yaml_path}"

        # Load configuration using default path (should use our default.yaml)
        loader = ConfigLoader(default_yaml_path)
        config = loader.load()

        # Verify configuration loaded successfully
        assert config is not None
        assert config.network is not None

    def test_default_yaml_network_settings(self):
        """Test that default.yaml contains all required network settings."""
        project_root = Path(__file__).parent.parent.parent.parent
        default_yaml_path = project_root / "config" / "default.yaml"

        loader = ConfigLoader(default_yaml_path)
        config = loader.load()

        # Verify all network threshold settings are present with expected defaults
        assert config.network.NETWORK_PACKET_LOSS_LOW_THRESHOLD == 0.01  # 1%
        assert config.network.NETWORK_PACKET_LOSS_MEDIUM_THRESHOLD == 0.05  # 5%
        assert config.network.NETWORK_PACKET_LOSS_HIGH_THRESHOLD == 0.10  # 10%
        assert config.network.NETWORK_PACKET_LOSS_CRITICAL_THRESHOLD == 0.20  # 20%

        # Verify enhanced network settings
        assert config.network.NETWORK_RUNTIME_ADJUSTMENT_ENABLED is True
        assert config.network.NETWORK_OPERATOR_OVERRIDE_ENABLED is True
        assert config.network.NETWORK_MONITORING_INTERVAL_MS == 1000
        assert config.network.NETWORK_ADAPTIVE_RATE_ENABLED is True

        # Verify existing network settings
        assert config.network.NETWORK_CONGESTION_DETECTOR_ENABLED is True
        assert config.network.NETWORK_BASELINE_LATENCY_MS == 0.0
        assert config.network.NETWORK_LATENCY_THRESHOLD_MS == 100.0

    def test_default_yaml_validation_passes(self):
        """Test that default.yaml values pass validation."""
        project_root = Path(__file__).parent.parent.parent.parent
        default_yaml_path = project_root / "config" / "default.yaml"

        # This should load without raising validation errors
        loader = ConfigLoader(default_yaml_path)
        config = loader.load()

        # Verify validation passes by checking methods work
        assert config.network.get_threshold_by_severity("low") == 0.01
        assert config.network.get_threshold_by_severity("medium") == 0.05
        assert config.network.get_threshold_by_severity("high") == 0.10
        assert config.network.get_threshold_by_severity("critical") == 0.20

        # Test that to_dict method works
        network_dict = config.network.to_dict()
        assert isinstance(network_dict, dict)
        assert "NETWORK_PACKET_LOSS_LOW_THRESHOLD" in network_dict
        assert "NETWORK_RUNTIME_ADJUSTMENT_ENABLED" in network_dict

    def test_default_yaml_threshold_ordering(self):
        """Test that default.yaml thresholds are properly ordered."""
        project_root = Path(__file__).parent.parent.parent.parent
        default_yaml_path = project_root / "config" / "default.yaml"

        loader = ConfigLoader(default_yaml_path)
        config = loader.load()

        # Verify thresholds are in ascending order
        thresholds = [
            config.network.NETWORK_PACKET_LOSS_LOW_THRESHOLD,
            config.network.NETWORK_PACKET_LOSS_MEDIUM_THRESHOLD,
            config.network.NETWORK_PACKET_LOSS_HIGH_THRESHOLD,
            config.network.NETWORK_PACKET_LOSS_CRITICAL_THRESHOLD,
        ]

        assert thresholds == sorted(thresholds), f"Thresholds not in order: {thresholds}"
        assert thresholds[0] < thresholds[1] < thresholds[2] < thresholds[3]

    def test_default_yaml_all_sections_load(self):
        """Test that all configuration sections load from default.yaml."""
        project_root = Path(__file__).parent.parent.parent.parent
        default_yaml_path = project_root / "config" / "default.yaml"

        loader = ConfigLoader(default_yaml_path)
        config = loader.load()

        # Verify all major configuration sections are present
        assert config.app is not None
        assert config.sdr is not None
        assert config.signal is not None
        assert config.database is not None
        assert config.logging is not None
        assert config.websocket is not None
        assert config.safety is not None
        assert config.performance is not None
        assert config.api is not None
        assert config.monitoring is not None
        assert config.homing is not None
        assert config.network is not None  # Our enhanced section
        assert config.development is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
