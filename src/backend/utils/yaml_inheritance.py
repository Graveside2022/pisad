"""
YAML Configuration Inheritance System
Allows YAML files to extend from base configurations to reduce duplication.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class YAMLInheritanceLoader:
    """
    Loads YAML configuration files with inheritance support.
    Files can specify 'extends: base_file.yaml' to inherit from another file.
    """

    def __init__(self, base_dir: Path | None = None):
        """
        Initialize the inheritance loader.

        Args:
            base_dir: Base directory for resolving relative paths
        """
        self.base_dir = base_dir or Path.cwd()
        self._loading_stack: list[Path] = []  # Prevent circular inheritance

    def load(self, config_path: Path | str) -> dict[str, Any]:
        """
        Load a YAML configuration file with inheritance support.

        Args:
            config_path: Path to the configuration file

        Returns:
            Merged configuration dictionary

        Raises:
            ValueError: If circular inheritance is detected
            FileNotFoundError: If config file not found
        """
        config_path = Path(config_path)

        # Make path absolute if relative
        if not config_path.is_absolute():
            config_path = self.base_dir / config_path

        # Check for circular inheritance
        if config_path in self._loading_stack:
            cycle = " -> ".join(str(p) for p in self._loading_stack)
            raise ValueError(f"Circular inheritance detected: {cycle} -> {config_path}")

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Add to loading stack
        self._loading_stack.append(config_path)

        try:
            # Load the current file
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}

            # Check for inheritance
            if "extends" in config:
                extends_path = config.pop("extends")

                # Resolve relative to current file's directory
                if not Path(extends_path).is_absolute():
                    extends_path = config_path.parent / extends_path

                # Load parent configuration
                parent_config = self.load(extends_path)

                # Deep merge parent with current (current overrides parent)
                config = self._deep_merge(parent_config, config)

                logger.debug(f"Loaded {config_path} extending {extends_path}")
            else:
                logger.debug(f"Loaded {config_path} (no inheritance)")

            return config

        finally:
            # Remove from loading stack
            self._loading_stack.pop()

    def _deep_merge(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """
        Deep merge two dictionaries, with override taking precedence.

        Args:
            base: Base dictionary
            override: Override dictionary

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                result[key] = self._deep_merge(result[key], value)
            else:
                # Override the value
                result[key] = value

        return result


def load_config_with_inheritance(
    config_path: Path | str, base_dir: Path | None = None
) -> dict[str, Any]:
    """
    Convenience function to load a YAML config file with inheritance.

    Args:
        config_path: Path to the configuration file
        base_dir: Base directory for resolving relative paths

    Returns:
        Loaded configuration dictionary
    """
    loader = YAMLInheritanceLoader(base_dir)
    return loader.load(config_path)


def validate_config_structure(config: dict[str, Any], required_keys: list[str]) -> bool:
    """
    Validate that a configuration contains all required keys.

    Args:
        config: Configuration dictionary to validate
        required_keys: List of required top-level keys

    Returns:
        True if all required keys are present
    """
    missing_keys = [key for key in required_keys if key not in config]

    if missing_keys:
        logger.error(f"Missing required configuration keys: {missing_keys}")
        return False

    return True


def get_config_value(config: dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get a value from nested configuration using dot notation.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to the value (e.g., 'sdr.frequency')
        default: Default value if key not found

    Returns:
        Configuration value or default
    """
    keys = key_path.split(".")
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value
