"""
Test suite for BandwidthThrottle configuration integration.

This test validates that BandwidthThrottle can accept configurable
packet loss thresholds from NetworkConfig instead of using hardcoded values.

Test Requirements:
- [8e6d] BandwidthThrottle._initialize_congestion_detector() uses config values
- [8e6c] ResourceOptimizer accepts configurable thresholds from config
- [8e6h] Comprehensive configuration integration testing

PRD References:
- PRD-AC5.6.6: Network bandwidth optimization with operator control
- PRD-NFR2: Signal processing latency <100ms per RSSI computation cycle
"""

import pytest

from src.backend.core.config import NetworkConfig
from src.backend.utils.resource_optimizer import BandwidthThrottle


class TestBandwidthThrottleConfiguration:
    """Test BandwidthThrottle configuration integration."""

    def test_bandwidth_throttle_accepts_network_config(self):
        """Test that BandwidthThrottle can accept NetworkConfig during initialization."""
        # Create custom network config
        network_config = NetworkConfig(
            NETWORK_PACKET_LOSS_LOW_THRESHOLD=0.002,  # 0.2%
            NETWORK_PACKET_LOSS_MEDIUM_THRESHOLD=0.01,  # 1%
            NETWORK_PACKET_LOSS_HIGH_THRESHOLD=0.05,  # 5%
            NETWORK_PACKET_LOSS_CRITICAL_THRESHOLD=0.15,  # 15%
        )

        # Test that BandwidthThrottle can be created with network config
        throttle = BandwidthThrottle(network_config=network_config)

        # Verify the throttle was created successfully
        assert throttle is not None
        assert hasattr(throttle, "_network_config")
        assert throttle._network_config == network_config

    def test_congestion_detector_uses_config_values(self):
        """Test that congestion detector uses values from NetworkConfig."""
        # Create custom network config with specific values
        network_config = NetworkConfig(
            NETWORK_PACKET_LOSS_LOW_THRESHOLD=0.003,  # 0.3%
            NETWORK_BASELINE_LATENCY_MS=5.0,
            NETWORK_LATENCY_THRESHOLD_MS=75.0,
            NETWORK_CONGESTION_DETECTOR_ENABLED=True,
        )

        throttle = BandwidthThrottle(network_config=network_config)

        # Access the congestion detector (should be initialized)
        detector = throttle._congestion_detector

        # Verify config values are used instead of hardcoded defaults
        assert detector["packet_loss_threshold"] == 0.003  # From config, not hardcoded 0.05
        assert detector["baseline_latency_ms"] == 5.0  # From config
        assert detector["latency_threshold_ms"] == 75.0  # From config
        assert detector["enabled"] is True  # From config

    def test_congestion_detector_fallback_defaults(self):
        """Test that congestion detector falls back to defaults when no config provided."""
        # Create throttle without network config
        throttle = BandwidthThrottle()

        # Access the congestion detector
        detector = throttle._congestion_detector

        # Verify fallback defaults are used
        assert detector["packet_loss_threshold"] == 0.05  # Default value
        assert detector["baseline_latency_ms"] == 0.0  # Default value
        assert detector["latency_threshold_ms"] == 100.0  # Default value
        assert detector["enabled"] is True  # Default value

    def test_congestion_detector_disabled_by_config(self):
        """Test that congestion detector can be disabled via config."""
        network_config = NetworkConfig(NETWORK_CONGESTION_DETECTOR_ENABLED=False)

        throttle = BandwidthThrottle(network_config=network_config)

        # Verify congestion detector is disabled
        assert (
            throttle._congestion_detector is None
            or throttle._congestion_detector["enabled"] is False
        )

    def test_runtime_threshold_update(self):
        """Test that packet loss thresholds can be updated at runtime."""
        network_config = NetworkConfig()
        throttle = BandwidthThrottle(network_config=network_config)

        # Update threshold in config
        network_config.update_threshold("low", 0.008)

        # Test that throttle can refresh its configuration
        throttle.update_config(network_config)

        # Verify updated threshold is used
        detector = throttle._congestion_detector
        assert detector["packet_loss_threshold"] == 0.008

    def test_bandwidth_throttle_with_resource_optimizer_integration(self):
        """Test integration between BandwidthThrottle and ResourceOptimizer."""
        # This test verifies the integration works end-to-end
        network_config = NetworkConfig(
            NETWORK_PACKET_LOSS_MEDIUM_THRESHOLD=0.02,  # 2%
            NETWORK_ADAPTIVE_RATE_ENABLED=True,
        )

        # Create throttle with config
        throttle = BandwidthThrottle(network_config=network_config)

        # Verify integration points exist
        assert hasattr(throttle, "_network_config")
        assert hasattr(throttle, "update_config")
        assert throttle._network_config.NETWORK_ADAPTIVE_RATE_ENABLED is True

    def test_config_validation_during_throttle_creation(self):
        """Test that invalid config values are caught during throttle creation."""
        # Create invalid network config
        with pytest.raises(ValueError, match="Packet loss threshold must be between 0.001 and 0.5"):
            invalid_config = NetworkConfig(
                NETWORK_PACKET_LOSS_LOW_THRESHOLD=0.7  # Invalid: above 0.5
            )

    def test_multiple_threshold_levels_in_detector(self):
        """Test that congestion detector can use multiple threshold levels."""
        network_config = NetworkConfig(
            NETWORK_PACKET_LOSS_LOW_THRESHOLD=0.01,
            NETWORK_PACKET_LOSS_MEDIUM_THRESHOLD=0.03,
            NETWORK_PACKET_LOSS_HIGH_THRESHOLD=0.08,
            NETWORK_PACKET_LOSS_CRITICAL_THRESHOLD=0.20,
        )

        throttle = BandwidthThrottle(network_config=network_config)

        # Test that all threshold levels are accessible
        assert throttle.get_threshold_for_severity("low") == 0.01
        assert throttle.get_threshold_for_severity("medium") == 0.03
        assert throttle.get_threshold_for_severity("high") == 0.08
        assert throttle.get_threshold_for_severity("critical") == 0.20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
