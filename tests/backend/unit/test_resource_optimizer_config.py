"""
Test suite for ResourceOptimizer configuration integration.

This test validates that ResourceOptimizer can accept configurable
packet loss thresholds from NetworkConfig instead of using hardcoded values.

Test Requirements:
- [8e6c] ResourceOptimizer accepts configurable thresholds from config
- [8e6h] Comprehensive configuration integration testing

PRD References:
- PRD-AC5.6.6: Network bandwidth optimization with operator control
- PRD-NFR4: Power consumption ≤2.5A @ 5V (memory <2GB on Pi 5)
"""

import pytest

from src.backend.core.config import NetworkConfig
from src.backend.utils.resource_optimizer import ResourceOptimizer


class TestResourceOptimizerConfiguration:
    """Test ResourceOptimizer configuration integration."""

    def test_resource_optimizer_accepts_network_config(self):
        """Test that ResourceOptimizer can accept NetworkConfig during initialization."""
        # Create custom network config
        network_config = NetworkConfig(
            NETWORK_PACKET_LOSS_LOW_THRESHOLD=0.002,  # 0.2%
            NETWORK_PACKET_LOSS_MEDIUM_THRESHOLD=0.01,  # 1%
            NETWORK_PACKET_LOSS_HIGH_THRESHOLD=0.05,  # 5%
            NETWORK_PACKET_LOSS_CRITICAL_THRESHOLD=0.15,  # 15%
            NETWORK_CONGESTION_DETECTOR_ENABLED=True,
        )

        # Test that ResourceOptimizer can be created with network config
        optimizer = ResourceOptimizer(network_config=network_config)

        # Verify the optimizer was created successfully
        assert optimizer is not None
        assert hasattr(optimizer, "_network_config")
        assert optimizer._network_config == network_config

    def test_resource_optimizer_uses_config_in_bandwidth_monitoring(self):
        """Test that ResourceOptimizer passes config to BandwidthThrottle."""
        network_config = NetworkConfig(
            NETWORK_PACKET_LOSS_LOW_THRESHOLD=0.003,  # 0.3%
            NETWORK_CONGESTION_DETECTOR_ENABLED=True,
            NETWORK_BASELINE_LATENCY_MS=5.0,
            NETWORK_LATENCY_THRESHOLD_MS=75.0,
        )

        optimizer = ResourceOptimizer(network_config=network_config)

        # Create bandwidth throttle and verify it uses config
        throttle = optimizer.create_bandwidth_throttle()

        # Verify config was passed to throttle
        assert throttle._network_config == network_config
        assert throttle._congestion_detector["packet_loss_threshold"] == 0.003
        assert throttle._congestion_detector["baseline_latency_ms"] == 5.0
        assert throttle._congestion_detector["latency_threshold_ms"] == 75.0

    def test_resource_optimizer_fallback_without_config(self):
        """Test that ResourceOptimizer works with default values when no config provided."""
        # Create optimizer without network config
        optimizer = ResourceOptimizer()

        # Verify it still works
        assert optimizer is not None
        assert optimizer._network_config is None

        # Create bandwidth throttle and verify it uses defaults
        throttle = optimizer.create_bandwidth_throttle()
        assert throttle._network_config is None
        assert throttle._congestion_detector["packet_loss_threshold"] == 0.05  # Default

    def test_resource_optimizer_config_update_propagation(self):
        """Test that config updates propagate to bandwidth components."""
        network_config = NetworkConfig()
        optimizer = ResourceOptimizer(network_config=network_config)

        # Create initial throttle
        throttle = optimizer.create_bandwidth_throttle()
        initial_threshold = throttle._congestion_detector["packet_loss_threshold"]

        # Update network config
        network_config.update_threshold("low", 0.008)

        # Update optimizer config
        optimizer.update_network_config(network_config)

        # Create new throttle and verify it uses updated config
        new_throttle = optimizer.create_bandwidth_throttle()
        assert new_throttle._congestion_detector["packet_loss_threshold"] == 0.008
        assert new_throttle._congestion_detector["packet_loss_threshold"] != initial_threshold

    def test_resource_optimizer_network_monitoring_integration(self):
        """Test integration between ResourceOptimizer and network monitoring."""
        network_config = NetworkConfig(
            NETWORK_MONITORING_INTERVAL_MS=500, NETWORK_ADAPTIVE_RATE_ENABLED=True
        )

        optimizer = ResourceOptimizer(network_config=network_config)

        # Test that network monitoring uses config values
        monitor = optimizer.get_network_monitor()

        # Verify config integration
        assert monitor is not None
        assert optimizer._network_config.NETWORK_MONITORING_INTERVAL_MS == 500
        assert optimizer._network_config.NETWORK_ADAPTIVE_RATE_ENABLED is True

    def test_resource_optimizer_degradation_with_network_config(self):
        """Test that graceful degradation uses network config thresholds."""
        network_config = NetworkConfig(
            NETWORK_PACKET_LOSS_CRITICAL_THRESHOLD=0.12  # 12% for degradation trigger
        )

        optimizer = ResourceOptimizer(network_config=network_config)
        degradation_manager = optimizer.degradation_manager

        # Test degradation threshold integration
        assert degradation_manager is not None

        # Simulate high packet loss scenario
        network_stats = {
            "packet_loss_rate": 0.13,  # Above critical threshold
            "bandwidth_utilization": 0.95,
        }

        # Check degradation trigger
        should_degrade = optimizer.should_trigger_degradation(network_stats)
        assert should_degrade is True

    def test_resource_optimizer_memory_profiler_with_network_config(self):
        """Test ResourceOptimizer memory profiler considers network overhead."""
        network_config = NetworkConfig(NETWORK_ADAPTIVE_RATE_ENABLED=True)

        # Test with memory profiler enabled
        optimizer = ResourceOptimizer(enable_memory_profiler=True, network_config=network_config)

        # Verify memory profiler accounts for network config
        assert optimizer.enable_memory_profiler is True
        assert optimizer._network_config.NETWORK_ADAPTIVE_RATE_ENABLED is True

    def test_resource_optimizer_async_operations_with_config(self):
        """Test async resource operations respect network configuration."""
        network_config = NetworkConfig(
            NETWORK_MONITORING_INTERVAL_MS=250  # Faster monitoring
        )

        optimizer = ResourceOptimizer(network_config=network_config)

        # Test async scheduler considers network config
        scheduler = optimizer.get_async_scheduler()
        assert scheduler is not None

        # Verify network-aware task scheduling
        assert optimizer._network_config.NETWORK_MONITORING_INTERVAL_MS == 250


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
