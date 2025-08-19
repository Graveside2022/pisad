"""Unit tests for Network Configuration functionality.

Tests YAML-based packet loss threshold configuration per TASK-5.6.2.3 [8e6].
Validates NetworkConfig class, YAML parsing, and runtime updates.
"""

from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from src.backend.core.config import ConfigLoader, NetworkConfig


class TestNetworkConfig:
    """Test network configuration management."""

    def test_network_config_dataclass_creation(self):
        """Test NetworkConfig dataclass can be created with defaults."""
        # TDD Red Phase - This should fail because NetworkConfig doesn't exist yet
        config = NetworkConfig()

        # Verify default packet loss thresholds
        assert hasattr(config, "NETWORK_PACKET_LOSS_LOW_THRESHOLD")
        assert hasattr(config, "NETWORK_PACKET_LOSS_MEDIUM_THRESHOLD")
        assert hasattr(config, "NETWORK_PACKET_LOSS_HIGH_THRESHOLD")
        assert hasattr(config, "NETWORK_PACKET_LOSS_CRITICAL_THRESHOLD")

        # Verify default values match existing hard-coded values
        assert config.NETWORK_PACKET_LOSS_LOW_THRESHOLD == 0.01  # 1%
        assert config.NETWORK_PACKET_LOSS_MEDIUM_THRESHOLD == 0.05  # 5%
        assert config.NETWORK_PACKET_LOSS_HIGH_THRESHOLD == 0.10  # 10%
        assert config.NETWORK_PACKET_LOSS_CRITICAL_THRESHOLD == 0.20  # 20%

    def test_network_config_validation_bounds(self):
        """Test NetworkConfig validates packet loss thresholds within bounds."""
        # TDD Red Phase - Should fail because validation doesn't exist yet
        with pytest.raises(ValueError, match="Packet loss threshold must be between 0.001 and 0.5"):
            NetworkConfig(NETWORK_PACKET_LOSS_LOW_THRESHOLD=0.0)  # Too low

        with pytest.raises(ValueError, match="Packet loss threshold must be between 0.001 and 0.5"):
            NetworkConfig(NETWORK_PACKET_LOSS_HIGH_THRESHOLD=0.6)  # Too high

    def test_config_loader_includes_network_section(self):
        """Test ConfigLoader includes NetworkConfig in main configuration."""
        # TDD Red Phase - Should fail because NetworkConfig not integrated yet
        loader = ConfigLoader()
        assert hasattr(loader.config, "network")
        assert isinstance(loader.config.network, NetworkConfig)

    def test_yaml_parsing_network_section(self):
        """Test YAML parser handles NETWORK_ prefixed configuration."""
        # TDD Red Phase - Should fail because NETWORK_ parsing not implemented
        yaml_content = """NETWORK_PACKET_LOSS_LOW_THRESHOLD: 0.02
NETWORK_PACKET_LOSS_MEDIUM_THRESHOLD: 0.08
NETWORK_PACKET_LOSS_HIGH_THRESHOLD: 0.15
NETWORK_PACKET_LOSS_CRITICAL_THRESHOLD: 0.25"""

        # Mock the Path.exists method to return True for the config file
        with (
            patch.object(Path, "exists", return_value=True),
            patch("builtins.open", mock_open(read_data=yaml_content)),
        ):
            loader = ConfigLoader(config_path=Path("test_config.yaml"))
            loader.load()  # Actually load the configuration

        assert loader.config.network.NETWORK_PACKET_LOSS_LOW_THRESHOLD == 0.02
        assert loader.config.network.NETWORK_PACKET_LOSS_MEDIUM_THRESHOLD == 0.08
        assert loader.config.network.NETWORK_PACKET_LOSS_HIGH_THRESHOLD == 0.15
        assert loader.config.network.NETWORK_PACKET_LOSS_CRITICAL_THRESHOLD == 0.25
