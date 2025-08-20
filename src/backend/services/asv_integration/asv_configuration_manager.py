"""Configuration bridge for ASV-to-PISAD settings integration.

SUBTASK-6.1.1.3-d: Design configuration bridge for ASV-to-PISAD settings

This module provides configuration management and translation between
ASV .NET configuration system and PISAD's YAML-based configuration.
"""

import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from src.backend.services.asv_integration.exceptions import (
    ASVConfigurationError,
    ASVFrequencyError,
)

logger = logging.getLogger(__name__)


@dataclass
class ASVFrequencyProfile:
    """Frequency profile configuration for ASV analyzers."""

    name: str
    description: str
    center_frequency_hz: int
    bandwidth_hz: int
    analyzer_type: str
    ref_power_dbm: float = -50.0
    calibration_enabled: bool = True
    processing_timeout_ms: int = 100
    priority: int = 1  # 1=highest, 10=lowest


@dataclass
class ASVAnalyzerProfile:
    """Complete analyzer profile with all configuration settings."""

    analyzer_id: str
    analyzer_type: str  # GP, VOR, LLZ
    enabled: bool
    frequency_profiles: list[ASVFrequencyProfile]
    hardware_config: dict[str, Any]
    processing_config: dict[str, Any]
    calibration_config: dict[str, Any]


class ASVConfigurationManager:
    """Manages configuration bridge between ASV and PISAD systems."""

    def __init__(self, config_dir: str = "config/asv_integration"):
        """Initialize configuration manager.

        Args:
            config_dir: Directory for ASV integration configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self._analyzer_profiles: dict[str, ASVAnalyzerProfile] = {}
        self._frequency_profiles: dict[str, ASVFrequencyProfile] = {}
        self._global_config: dict[str, Any] = {}

        # Load configuration on initialization
        self._load_configuration()

    def _load_configuration(self) -> None:
        """Load ASV configuration from YAML files."""
        try:
            # Load global ASV configuration
            global_config_path = self.config_dir / "asv_global.yaml"
            if global_config_path.exists():
                with open(global_config_path) as f:
                    self._global_config = yaml.safe_load(f) or {}
                logger.info(
                    f"Loaded global ASV configuration from {global_config_path}"
                )
            else:
                self._global_config = self._get_default_global_config()
                self._save_global_config()
                logger.info("Created default global ASV configuration")

            # Load frequency profiles
            freq_profiles_path = self.config_dir / "frequency_profiles.yaml"
            if freq_profiles_path.exists():
                with open(freq_profiles_path) as f:
                    profiles_data = yaml.safe_load(f) or {}

                for name, profile_data in profiles_data.items():
                    try:
                        profile = ASVFrequencyProfile(**profile_data)
                        self._frequency_profiles[name] = profile
                    except TypeError as e:
                        logger.warning(f"Invalid frequency profile {name}: {e}")

                logger.info(
                    f"Loaded {len(self._frequency_profiles)} frequency profiles"
                )
            else:
                self._frequency_profiles = self._get_default_frequency_profiles()
                self._save_frequency_profiles()
                logger.info("Created default frequency profiles")

            # Load analyzer profiles
            analyzer_profiles_path = self.config_dir / "analyzer_profiles.yaml"
            if analyzer_profiles_path.exists():
                with open(analyzer_profiles_path) as f:
                    analyzers_data = yaml.safe_load(f) or {}

                for analyzer_id, analyzer_data in analyzers_data.items():
                    try:
                        # Convert frequency profile references to objects
                        freq_profiles = []
                        for freq_name in analyzer_data.get("frequency_profiles", []):
                            if freq_name in self._frequency_profiles:
                                freq_profiles.append(
                                    self._frequency_profiles[freq_name]
                                )

                        analyzer_data["frequency_profiles"] = freq_profiles
                        profile = ASVAnalyzerProfile(**analyzer_data)
                        self._analyzer_profiles[analyzer_id] = profile
                    except TypeError as e:
                        logger.warning(f"Invalid analyzer profile {analyzer_id}: {e}")

                logger.info(f"Loaded {len(self._analyzer_profiles)} analyzer profiles")
            else:
                self._analyzer_profiles = self._get_default_analyzer_profiles()
                self._save_analyzer_profiles()
                logger.info("Created default analyzer profiles")

        except Exception as e:
            logger.error(f"Failed to load ASV configuration: {e}")
            raise ASVConfigurationError(f"Configuration loading failed: {e}")

    def _get_default_global_config(self) -> dict[str, Any]:
        """Get default global ASV configuration."""
        return {
            "asv_integration": {
                "enabled": True,
                "max_concurrent_analyzers": 5,
                "processing_thread_pool_size": 4,
                "signal_buffer_size_mb": 64,
                "performance_monitoring": True,
                "error_retry_attempts": 3,
                "error_retry_delay_ms": 500,
            },
            "dotnet_runtime": {
                "dotnet_root": "~/.dotnet",
                "use_coreclr": True,
                "runtime_config_path": None,
                "assembly_load_timeout_ms": 10000,
            },
            "hardware_interface": {
                "preserve_existing_hackrf": True,
                "preserve_existing_mavlink": True,
                "sdr_sample_rate": 2048000,  # 2.048 MHz
                "sdr_gain": 14,  # dB
            },
            "safety_integration": {
                "respect_safety_interlocks": True,
                "emergency_stop_propagation": True,
                "safety_authority_compliance": True,
            },
        }

    def _get_default_frequency_profiles(self) -> dict[str, ASVFrequencyProfile]:
        """Get default frequency profiles for SAR operations."""
        profiles = {}

        # Emergency beacon profile (406 MHz)
        profiles["emergency_beacon_406"] = ASVFrequencyProfile(
            name="emergency_beacon_406",
            description="Emergency Beacon Detection at 406 MHz",
            center_frequency_hz=406_000_000,
            bandwidth_hz=50_000,  # 50 kHz
            analyzer_type="GP",
            ref_power_dbm=-120.0,
            calibration_enabled=True,
            processing_timeout_ms=50,  # Fast processing for emergencies
            priority=1,
        )

        # VOR navigation profile
        profiles["vor_aviation"] = ASVFrequencyProfile(
            name="vor_aviation",
            description="VOR Aviation Navigation Signals",
            center_frequency_hz=112_500_000,  # Mid VOR range
            bandwidth_hz=200_000,  # 200 kHz
            analyzer_type="VOR",
            ref_power_dbm=-100.0,
            calibration_enabled=True,
            processing_timeout_ms=100,
            priority=2,
        )

        # ILS Localizer profile
        profiles["ils_localizer"] = ASVFrequencyProfile(
            name="ils_localizer",
            description="ILS Localizer Landing System",
            center_frequency_hz=109_500_000,  # Mid localizer range
            bandwidth_hz=150_000,  # 150 kHz
            analyzer_type="LLZ",
            ref_power_dbm=-90.0,
            calibration_enabled=True,
            processing_timeout_ms=75,
            priority=3,
        )

        # Additional emergency frequencies
        profiles["aviation_emergency"] = ASVFrequencyProfile(
            name="aviation_emergency",
            description="Aviation Emergency Frequency",
            center_frequency_hz=121_500_000,  # 121.5 MHz
            bandwidth_hz=25_000,  # 25 kHz
            analyzer_type="GP",
            ref_power_dbm=-110.0,
            calibration_enabled=True,
            processing_timeout_ms=60,
            priority=1,
        )

        return profiles

    def _get_default_analyzer_profiles(self) -> dict[str, ASVAnalyzerProfile]:
        """Get default analyzer profiles."""
        profiles = {}

        # Emergency beacon analyzer
        profiles["emergency_analyzer"] = ASVAnalyzerProfile(
            analyzer_id="emergency_analyzer",
            analyzer_type="GP",
            enabled=True,
            frequency_profiles=[
                self._frequency_profiles["emergency_beacon_406"],
                self._frequency_profiles["aviation_emergency"],
            ],
            hardware_config={
                "sdr_device": "hackrf",
                "antenna_gain_db": 0,
                "lna_enabled": True,
            },
            processing_config={
                "fft_size": 1024,
                "overlap_factor": 0.5,
                "window_function": "hann",
                "detection_threshold_db": -110,
            },
            calibration_config={
                "auto_calibration": True,
                "calibration_interval_s": 300,
                "reference_source": "internal",
            },
        )

        # Aviation navigation analyzer
        profiles["aviation_analyzer"] = ASVAnalyzerProfile(
            analyzer_id="aviation_analyzer",
            analyzer_type="VOR",
            enabled=True,
            frequency_profiles=[
                self._frequency_profiles["vor_aviation"],
            ],
            hardware_config={
                "sdr_device": "hackrf",
                "antenna_gain_db": 2,
                "lna_enabled": True,
            },
            processing_config={
                "fft_size": 2048,
                "overlap_factor": 0.75,
                "window_function": "blackman",
                "detection_threshold_db": -95,
            },
            calibration_config={
                "auto_calibration": True,
                "calibration_interval_s": 600,
                "reference_source": "internal",
            },
        )

        # Landing system analyzer
        profiles["landing_analyzer"] = ASVAnalyzerProfile(
            analyzer_id="landing_analyzer",
            analyzer_type="LLZ",
            enabled=True,
            frequency_profiles=[
                self._frequency_profiles["ils_localizer"],
            ],
            hardware_config={
                "sdr_device": "hackrf",
                "antenna_gain_db": 1,
                "lna_enabled": True,
            },
            processing_config={
                "fft_size": 1024,
                "overlap_factor": 0.5,
                "window_function": "hann",
                "detection_threshold_db": -85,
            },
            calibration_config={
                "auto_calibration": True,
                "calibration_interval_s": 450,
                "reference_source": "internal",
            },
        )

        return profiles

    def _save_global_config(self) -> None:
        """Save global configuration to file."""
        try:
            config_path = self.config_dir / "asv_global.yaml"
            with open(config_path, "w") as f:
                yaml.safe_dump(
                    self._global_config, f, default_flow_style=False, indent=2
                )
            logger.debug(f"Saved global config to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save global config: {e}")

    def _save_frequency_profiles(self) -> None:
        """Save frequency profiles to file."""
        try:
            profiles_data = {
                name: asdict(profile)
                for name, profile in self._frequency_profiles.items()
            }
            config_path = self.config_dir / "frequency_profiles.yaml"
            with open(config_path, "w") as f:
                yaml.safe_dump(profiles_data, f, default_flow_style=False, indent=2)
            logger.debug(f"Saved frequency profiles to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save frequency profiles: {e}")

    def _save_analyzer_profiles(self) -> None:
        """Save analyzer profiles to file."""
        try:
            profiles_data = {}
            for analyzer_id, profile in self._analyzer_profiles.items():
                profile_dict = asdict(profile)
                # Convert frequency profile objects to names for serialization
                profile_dict["frequency_profiles"] = [
                    fp.name for fp in profile.frequency_profiles
                ]
                profiles_data[analyzer_id] = profile_dict

            config_path = self.config_dir / "analyzer_profiles.yaml"
            with open(config_path, "w") as f:
                yaml.safe_dump(profiles_data, f, default_flow_style=False, indent=2)
            logger.debug(f"Saved analyzer profiles to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save analyzer profiles: {e}")

    def get_global_config(self, key: str | None = None) -> dict[str, Any] | Any:
        """Get global configuration value(s).

        Args:
            key: Specific config key (dot notation supported)

        Returns:
            Configuration value or full config if key is None
        """
        if key is None:
            return self._global_config.copy()

        # Support dot notation for nested keys
        keys = key.split(".")
        value = self._global_config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            raise ASVConfigurationError(f"Configuration key not found: {key}")

    def set_global_config(self, key: str, value: Any, save: bool = True) -> None:
        """Set global configuration value.

        Args:
            key: Configuration key (dot notation supported)
            value: Value to set
            save: Whether to save to file immediately
        """
        keys = key.split(".")
        config = self._global_config

        # Navigate to parent of target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set the value
        config[keys[-1]] = value

        if save:
            self._save_global_config()

        logger.info(f"Updated global config: {key} = {value}")

    def get_frequency_profile(self, name: str) -> ASVFrequencyProfile:
        """Get frequency profile by name.

        Args:
            name: Profile name

        Returns:
            Frequency profile
        """
        if name not in self._frequency_profiles:
            raise ASVConfigurationError(f"Frequency profile not found: {name}")

        return self._frequency_profiles[name]

    def get_all_frequency_profiles(self) -> dict[str, ASVFrequencyProfile]:
        """Get all frequency profiles."""
        return self._frequency_profiles.copy()

    def create_frequency_profile(
        self, profile: ASVFrequencyProfile, save: bool = True
    ) -> None:
        """Create or update frequency profile.

        Args:
            profile: Frequency profile to create/update
            save: Whether to save to file immediately
        """
        # Validate frequency range
        self._validate_frequency(profile.center_frequency_hz)

        self._frequency_profiles[profile.name] = profile

        if save:
            self._save_frequency_profiles()

        logger.info(f"Created frequency profile: {profile.name}")

    def get_analyzer_profile(self, analyzer_id: str) -> ASVAnalyzerProfile:
        """Get analyzer profile by ID.

        Args:
            analyzer_id: Analyzer identifier

        Returns:
            Analyzer profile
        """
        if analyzer_id not in self._analyzer_profiles:
            raise ASVConfigurationError(f"Analyzer profile not found: {analyzer_id}")

        return self._analyzer_profiles[analyzer_id]

    def get_all_analyzer_profiles(self) -> dict[str, ASVAnalyzerProfile]:
        """Get all analyzer profiles."""
        return self._analyzer_profiles.copy()

    def create_analyzer_profile(
        self, profile: ASVAnalyzerProfile, save: bool = True
    ) -> None:
        """Create or update analyzer profile.

        Args:
            profile: Analyzer profile to create/update
            save: Whether to save to file immediately
        """
        # Validate analyzer configuration
        if profile.analyzer_type not in ["GP", "VOR", "LLZ"]:
            raise ASVConfigurationError(
                f"Invalid analyzer type: {profile.analyzer_type}"
            )

        self._analyzer_profiles[profile.analyzer_id] = profile

        if save:
            self._save_analyzer_profiles()

        logger.info(f"Created analyzer profile: {profile.analyzer_id}")

    def _validate_frequency(self, frequency_hz: int) -> None:
        """Validate frequency is within acceptable range.

        Args:
            frequency_hz: Frequency to validate

        Raises:
            ASVFrequencyError: If frequency is out of range
        """
        # HackRF frequency range: 1 MHz to 6 GHz
        min_freq = 1_000_000  # 1 MHz
        max_freq = 6_000_000_000  # 6 GHz

        if not (min_freq <= frequency_hz <= max_freq):
            raise ASVFrequencyError(
                f"Frequency {frequency_hz:,} Hz out of range",
                frequency_hz,
                (min_freq, max_freq),
            )

    def get_pisad_compatible_config(self, analyzer_id: str) -> dict[str, Any]:
        """Get analyzer configuration in PISAD-compatible format.

        Args:
            analyzer_id: Analyzer to get configuration for

        Returns:
            PISAD-compatible configuration dictionary
        """
        if analyzer_id not in self._analyzer_profiles:
            raise ASVConfigurationError(f"Analyzer profile not found: {analyzer_id}")

        profile = self._analyzer_profiles[analyzer_id]

        # Convert ASV profile to PISAD format
        pisad_config = {
            "sdr_service": {
                "enabled": profile.enabled,
                "device_type": "hackrf",  # Always HackRF for ASV integration
                "sample_rate": self.get_global_config(
                    "hardware_interface.sdr_sample_rate"
                ),
                "gain": profile.hardware_config.get("antenna_gain_db", 14),
            },
            "signal_processing": {
                "fft_size": profile.processing_config.get("fft_size", 1024),
                "window_function": profile.processing_config.get(
                    "window_function", "hann"
                ),
                "overlap_factor": profile.processing_config.get("overlap_factor", 0.5),
            },
            "frequency_profiles": [],
        }

        # Add frequency profiles
        for freq_profile in profile.frequency_profiles:
            pisad_freq_config = {
                "name": freq_profile.name,
                "frequency_hz": freq_profile.center_frequency_hz,
                "bandwidth_hz": freq_profile.bandwidth_hz,
                "detection_threshold_dbm": freq_profile.ref_power_dbm,
                "processing_timeout_ms": freq_profile.processing_timeout_ms,
            }
            pisad_config["frequency_profiles"].append(pisad_freq_config)

        return pisad_config

    def reload_configuration(self) -> None:
        """Reload all configuration from files."""
        logger.info("Reloading ASV configuration")
        self._load_configuration()

    def get_configuration_summary(self) -> dict[str, Any]:
        """Get summary of current configuration.

        Returns:
            Configuration summary for monitoring
        """
        return {
            "global_config": {
                "asv_enabled": self.get_global_config("asv_integration.enabled"),
                "max_analyzers": self.get_global_config(
                    "asv_integration.max_concurrent_analyzers"
                ),
                "safety_compliance": self.get_global_config(
                    "safety_integration.safety_authority_compliance"
                ),
            },
            "frequency_profiles": {
                "count": len(self._frequency_profiles),
                "profiles": list(self._frequency_profiles.keys()),
            },
            "analyzer_profiles": {
                "count": len(self._analyzer_profiles),
                "enabled": len(
                    [p for p in self._analyzer_profiles.values() if p.enabled]
                ),
                "analyzers": list(self._analyzer_profiles.keys()),
            },
            "config_files": {
                "config_dir": str(self.config_dir),
                "global_config": str(self.config_dir / "asv_global.yaml"),
                "frequency_profiles": str(self.config_dir / "frequency_profiles.yaml"),
                "analyzer_profiles": str(self.config_dir / "analyzer_profiles.yaml"),
            },
        }

    def get_dotnet_runtime_path(self) -> str | None:
        """Get .NET runtime path for pythonnet integration.

        Returns:
            Path to .NET runtime or None for system default
        """
        try:
            dotnet_config = self.get_global_config("dotnet_runtime")
            dotnet_root = dotnet_config.get("dotnet_root", "~/.dotnet")

            # Expand user path
            dotnet_root = os.path.expanduser(dotnet_root)

            # Check if custom runtime config path is specified
            runtime_config = dotnet_config.get("runtime_config_path")
            if runtime_config:
                runtime_config = os.path.expanduser(runtime_config)
                if os.path.exists(runtime_config):
                    logger.debug(f"Using custom .NET runtime config: {runtime_config}")
                    return runtime_config

            # Default to system .NET installation if exists
            if os.path.exists(dotnet_root):
                logger.debug(f"Using .NET runtime at: {dotnet_root}")
                return dotnet_root

            # Fall back to system default
            logger.debug("Using system default .NET runtime")
            return None

        except Exception as e:
            logger.warning(f"Failed to get .NET runtime path: {e}")
            return None
