"""Debug test to understand the YAML loading issue."""

from pathlib import Path
from unittest.mock import mock_open, patch

import yaml

from src.backend.core.config import ConfigLoader


def test_yaml_loading_debug():
    """Debug test to see what's happening with YAML loading."""
    yaml_content = """NETWORK_PACKET_LOSS_LOW_THRESHOLD: 0.02
NETWORK_PACKET_LOSS_MEDIUM_THRESHOLD: 0.08"""

    # First test: can we parse the YAML correctly?
    parsed = yaml.safe_load(yaml_content)
    print(f"Parsed YAML: {parsed}")

    # Second test: does the ConfigLoader constructor call the YAML loading?
    with (
        patch.object(Path, "exists", return_value=True),
        patch("builtins.open", mock_open(read_data=yaml_content)) as mock_file,
        patch("yaml.safe_load", return_value=parsed) as mock_yaml,
    ):
        loader = ConfigLoader(config_path=Path("test_config.yaml"))

        print(f"Mock file called: {mock_file.called}")
        print(f"Mock yaml called: {mock_yaml.called}")
        print(f"Network config values: {loader.config.network}")


if __name__ == "__main__":
    test_yaml_loading_debug()
