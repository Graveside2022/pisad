"""Configuration Profile Service for managing SDR and system configurations."""

import logging
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from src.backend.models.schemas import (
    ConfigProfile,
    HomingConfig,
    SDRConfig,
    SignalConfig,
)

logger = logging.getLogger(__name__)


class ConfigService:
    """Service for managing configuration profiles."""

    def __init__(self, profiles_dir: str = "/home/pisad/projects/pisad/config/profiles"):
        """Initialize the configuration service.

        Args:
            profiles_dir: Directory path for storing profile YAML files
        """
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def load_profile(self, profile_name: str) -> ConfigProfile | None:
        """Load a configuration profile from YAML file.

        Args:
            profile_name: Name of the profile file (without .yaml extension)

        Returns:
            ConfigProfile object if found, None otherwise
        """
        profile_path = self.profiles_dir / f"{profile_name}.yaml"

        if not profile_path.exists():
            return None

        try:
            with open(profile_path) as f:
                data = yaml.safe_load(f)

            # Add metadata if not present
            if "id" not in data:
                data["id"] = str(uuid.uuid4())
            if "createdAt" not in data:
                data["createdAt"] = datetime.now(UTC)
            else:
                data["createdAt"] = datetime.fromisoformat(data["createdAt"])
            if "updatedAt" not in data:
                data["updatedAt"] = datetime.now(UTC)
            else:
                data["updatedAt"] = datetime.fromisoformat(data["updatedAt"])

            # Convert nested dictionaries to dataclass instances
            if data.get("sdrConfig"):
                data["sdrConfig"] = SDRConfig(**data["sdrConfig"])
            if data.get("signalConfig"):
                data["signalConfig"] = SignalConfig(**data["signalConfig"])
            if data.get("homingConfig"):
                data["homingConfig"] = HomingConfig(**data["homingConfig"])

            return ConfigProfile(**data)

        except (yaml.YAMLError, ValueError) as e:
            logger.error(f"Error loading profile {profile_name}: {e}")
            return None

    def save_profile(self, profile: ConfigProfile) -> bool:
        """Save a configuration profile to YAML file.

        Args:
            profile: ConfigProfile object to save

        Returns:
            True if saved successfully, False otherwise
        """
        profile_path = self.profiles_dir / f"{profile.name}.yaml"

        try:
            # Update the updatedAt timestamp
            profile.updatedAt = datetime.now(UTC)

            # Convert to dict for saving
            profile_dict = {
                "id": profile.id,
                "name": profile.name,
                "description": profile.description,
                "isDefault": profile.isDefault,
                "createdAt": profile.createdAt.isoformat() if profile.createdAt else None,
                "updatedAt": profile.updatedAt.isoformat() if profile.updatedAt else None,
            }

            # Convert nested dataclasses to dicts
            if profile.sdrConfig:
                profile_dict["sdrConfig"] = asdict(profile.sdrConfig)
            if profile.signalConfig:
                profile_dict["signalConfig"] = asdict(profile.signalConfig)
            if profile.homingConfig:
                profile_dict["homingConfig"] = asdict(profile.homingConfig)

            with open(profile_path, "w") as f:
                yaml.safe_dump(profile_dict, f, default_flow_style=False, sort_keys=False)

            return True

        except (OSError, yaml.YAMLError) as e:
            logger.error(f"Error saving profile {profile.name}: {e}")
            return False

    def list_profiles(self) -> list[str]:
        """List all available profile names.

        Returns:
            List of profile names (without .yaml extension)
        """
        profiles = []

        for profile_file in self.profiles_dir.glob("*.yaml"):
            profiles.append(profile_file.stem)

        return sorted(profiles)

    def delete_profile(self, profile_name: str) -> bool:
        """Delete a configuration profile.

        Args:
            profile_name: Name of the profile to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        profile_path = self.profiles_dir / f"{profile_name}.yaml"

        if not profile_path.exists():
            return False

        try:
            profile_path.unlink()
            return True
        except OSError as e:
            logger.error(f"Error deleting profile {profile_name}: {e}")
            return False

    def get_default_profile(self) -> ConfigProfile | None:
        """Get the default configuration profile.

        Returns:
            Default ConfigProfile if found, None otherwise
        """
        # First check for a profile marked as default
        for profile_name in self.list_profiles():
            profile = self.load_profile(profile_name)
            if profile and profile.isDefault:
                return profile

        # If no default found, try to load 'default' profile
        default_profile = self.load_profile("default")
        if default_profile:
            return default_profile

        # If still no default, try to load 'custom' profile as fallback
        return self.load_profile("custom")

    def set_default_profile(self, profile_name: str) -> bool:
        """Set a profile as the default.

        Args:
            profile_name: Name of the profile to set as default

        Returns:
            True if set successfully, False otherwise
        """
        # First, unset any existing defaults
        for name in self.list_profiles():
            profile = self.load_profile(name)
            if profile and profile.isDefault:
                profile.isDefault = False
                self.save_profile(profile)

        # Set the new default
        profile = self.load_profile(profile_name)
        if profile:
            profile.isDefault = True
            return self.save_profile(profile)

        return False

    def validate_profile(self, profile: ConfigProfile) -> dict[str, Any]:
        """Validate a configuration profile's parameters.

        Args:
            profile: ConfigProfile to validate

        Returns:
            Dictionary with 'valid' boolean and 'errors' list
        """
        errors = []

        # Validate SDR configuration
        if profile.sdrConfig:
            if not (1e6 <= profile.sdrConfig.frequency <= 6e9):
                errors.append("Frequency must be between 1 MHz and 6 GHz")
            if not (0.25e6 <= profile.sdrConfig.sampleRate <= 20e6):
                errors.append("Sample rate must be between 0.25 Msps and 20 Msps")
            if isinstance(profile.sdrConfig.gain, int | float) and not (
                -10 <= profile.sdrConfig.gain <= 70
            ):
                errors.append("Gain must be between -10 dB and 70 dB")

        # Validate signal configuration
        if profile.signalConfig:
            if not (0 < profile.signalConfig.ewmaAlpha <= 1):
                errors.append("EWMA alpha must be between 0 and 1")
            if profile.signalConfig.dropThreshold >= profile.signalConfig.triggerThreshold:
                errors.append("Drop threshold must be less than trigger threshold")

        # Validate homing configuration
        if profile.homingConfig:
            if profile.homingConfig.forwardVelocityMax <= 0:
                errors.append("Forward velocity max must be positive")
            if profile.homingConfig.yawRateMax <= 0:
                errors.append("Yaw rate max must be positive")
            if profile.homingConfig.signalLossTimeout <= 0:
                errors.append("Signal loss timeout must be positive")

        return {"valid": len(errors) == 0, "errors": errors}
