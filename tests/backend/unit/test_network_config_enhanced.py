"""
Test suite for NetworkConfig class enhancements - packet loss threshold configuration.

This test suite validates the enhanced NetworkConfig class with additional
packet loss threshold fields, runtime adjustment capabilities, and bounds validation.

Test Requirements:
- [8e6a] NetworkConfig enhanced with additional threshold fields
- [8e6g] Configuration validation with bounds checking (0.001-0.5 range)
- [8e6h] Comprehensive configuration loading and validation testing

PRD References:
- PRD-NFR1: MAVLink communication <1% packet loss
- PRD-AC5.6.6: Network bandwidth optimization with operator control
"""

import pytest

from src.backend.core.config import NetworkConfig


class TestNetworkConfigEnhancements:
    """Test NetworkConfig class enhancements for packet loss configuration."""

    def test_network_config_default_values(self):
        """Test NetworkConfig initializes with proper default values."""
        config = NetworkConfig()

        # Existing thresholds should remain unchanged
        assert config.NETWORK_PACKET_LOSS_LOW_THRESHOLD == 0.01  # 1%
        assert config.NETWORK_PACKET_LOSS_MEDIUM_THRESHOLD == 0.05  # 5%
        assert config.NETWORK_PACKET_LOSS_HIGH_THRESHOLD == 0.10  # 10%
        assert config.NETWORK_PACKET_LOSS_CRITICAL_THRESHOLD == 0.20  # 20%

        # Enhanced fields for runtime adjustment
        assert hasattr(config, "NETWORK_RUNTIME_ADJUSTMENT_ENABLED")
        assert config.NETWORK_RUNTIME_ADJUSTMENT_ENABLED is True

        assert hasattr(config, "NETWORK_OPERATOR_OVERRIDE_ENABLED")
        assert config.NETWORK_OPERATOR_OVERRIDE_ENABLED is True

        # Performance monitoring thresholds
        assert hasattr(config, "NETWORK_MONITORING_INTERVAL_MS")
        assert config.NETWORK_MONITORING_INTERVAL_MS == 1000  # 1 second

        # Adaptive transmission rate thresholds
        assert hasattr(config, "NETWORK_ADAPTIVE_RATE_ENABLED")
        assert config.NETWORK_ADAPTIVE_RATE_ENABLED is True

    def test_packet_loss_threshold_validation_bounds(self):
        """Test packet loss threshold validation enforces 0.001-0.5 range."""
        # Test valid values at boundaries
        config = NetworkConfig(
            NETWORK_PACKET_LOSS_LOW_THRESHOLD=0.001, NETWORK_PACKET_LOSS_CRITICAL_THRESHOLD=0.5
        )
        # Should not raise exception

        # Test invalid values below minimum
        with pytest.raises(ValueError, match="Packet loss threshold must be between 0.001 and 0.5"):
            NetworkConfig(NETWORK_PACKET_LOSS_LOW_THRESHOLD=0.0005)

        # Test invalid values above maximum
        with pytest.raises(ValueError, match="Packet loss threshold must be between 0.001 and 0.5"):
            NetworkConfig(NETWORK_PACKET_LOSS_CRITICAL_THRESHOLD=0.6)

    def test_threshold_ordering_validation(self):
        """Test that thresholds are properly ordered (low < medium < high < critical)."""
        # Valid ordering should pass
        config = NetworkConfig(
            NETWORK_PACKET_LOSS_LOW_THRESHOLD=0.01,
            NETWORK_PACKET_LOSS_MEDIUM_THRESHOLD=0.05,
            NETWORK_PACKET_LOSS_HIGH_THRESHOLD=0.10,
            NETWORK_PACKET_LOSS_CRITICAL_THRESHOLD=0.20,
        )
        # Should not raise exception

        # Invalid ordering should fail
        with pytest.raises(ValueError, match="Packet loss thresholds must be in ascending order"):
            NetworkConfig(
                NETWORK_PACKET_LOSS_LOW_THRESHOLD=0.10,
                NETWORK_PACKET_LOSS_MEDIUM_THRESHOLD=0.05,
                NETWORK_PACKET_LOSS_HIGH_THRESHOLD=0.01,
                NETWORK_PACKET_LOSS_CRITICAL_THRESHOLD=0.20,
            )

    def test_get_threshold_by_severity(self):
        """Test method to get threshold by severity level."""
        config = NetworkConfig()

        assert config.get_threshold_by_severity("low") == config.NETWORK_PACKET_LOSS_LOW_THRESHOLD
        assert (
            config.get_threshold_by_severity("medium")
            == config.NETWORK_PACKET_LOSS_MEDIUM_THRESHOLD
        )
        assert config.get_threshold_by_severity("high") == config.NETWORK_PACKET_LOSS_HIGH_THRESHOLD
        assert (
            config.get_threshold_by_severity("critical")
            == config.NETWORK_PACKET_LOSS_CRITICAL_THRESHOLD
        )

        with pytest.raises(ValueError, match="Unknown severity level"):
            config.get_threshold_by_severity("invalid")

    def test_update_threshold_method(self):
        """Test method to update individual thresholds with validation."""
        config = NetworkConfig()

        # Valid update
        config.update_threshold("medium", 0.03)
        assert config.NETWORK_PACKET_LOSS_MEDIUM_THRESHOLD == 0.03

        # Invalid update should raise exception
        with pytest.raises(ValueError, match="Packet loss threshold must be between 0.001 and 0.5"):
            config.update_threshold("high", 0.6)

        # Update that breaks ordering should raise exception
        with pytest.raises(ValueError, match="would break threshold ordering"):
            config.update_threshold("low", 0.15)  # Higher than medium threshold


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
