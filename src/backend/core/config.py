"""
Configuration management for PISAD application.
Loads configuration from YAML files and environment variables.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.backend.core.exceptions import (
    ConfigurationError,
    SDRError,
)
from src.backend.services.config_service import ConfigService

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """Application configuration."""

    APP_NAME: str = "PISAD"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8080


@dataclass
class SDRConfig:
    """SDR configuration."""

    SDR_FREQUENCY: int = 433920000
    SDR_SAMPLE_RATE: int = 2048000
    SDR_GAIN: int = 30
    SDR_PPM_CORRECTION: int = 0
    SDR_DEVICE_INDEX: int = 0
    SDR_BUFFER_SIZE: int = 16384


@dataclass
class SignalConfig:
    """Signal processing configuration."""

    SIGNAL_RSSI_THRESHOLD: float = -70.0
    SIGNAL_AVERAGING_WINDOW: int = 10
    SIGNAL_MIN_DURATION_MS: int = 100
    SIGNAL_MAX_GAP_MS: int = 50


@dataclass
class InterferometryConfig:
    """Interferometry configuration."""

    INTERFEROMETRY_ENABLED: bool = False
    INTERFEROMETRY_BASELINE_MM: int = 200
    INTERFEROMETRY_CALIBRATION_FILE: str = "config/calibration.yaml"


@dataclass
class DatabaseConfig:
    """Database configuration."""

    DB_PATH: str = "data/pisad.db"
    DB_CONNECTION_POOL_SIZE: int = 5
    DB_ENABLE_WAL: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""

    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE_PATH: str = "logs/pisad.log"
    LOG_FILE_MAX_BYTES: int = 10485760
    LOG_FILE_BACKUP_COUNT: int = 5
    LOG_ENABLE_CONSOLE: bool = True
    LOG_ENABLE_FILE: bool = True
    LOG_ENABLE_JOURNAL: bool = True


@dataclass
class WebSocketConfig:
    """WebSocket configuration."""

    WS_RSSI_UPDATE_INTERVAL_MS: int = 100
    WS_HEARTBEAT_INTERVAL_S: int = 30
    WS_MAX_CONNECTIONS: int = 10


@dataclass
class SafetyConfig:
    """Safety configuration."""

    SAFETY_VELOCITY_MAX_MPS: float = 2.0
    SAFETY_INTERLOCK_ENABLED: bool = False
    SAFETY_EMERGENCY_STOP_GPIO: int = 23


@dataclass
class PerformanceConfig:
    """Performance configuration."""

    PERF_WORKER_THREADS: int = 4
    PERF_ENABLE_PROFILING: bool = False
    PERF_CACHE_SIZE_MB: int = 100


@dataclass
class APIConfig:
    """API configuration."""

    API_RATE_LIMIT_ENABLED: bool = False
    API_RATE_LIMIT_REQUESTS: int = 100
    API_CORS_ENABLED: bool = True
    API_CORS_ORIGINS: list[str] = field(default_factory=lambda: ["*"])
    API_KEY_ENABLED: bool = False
    API_KEY: str = ""


@dataclass
class MonitoringConfig:
    """System monitoring configuration."""

    MONITORING_ENABLED: bool = False
    MONITORING_PORT: int = 9090
    MONITORING_INTERVAL_S: int = 60


@dataclass
class TelemetryConfig:
    """MAVLink telemetry configuration."""

    TELEMETRY_RSSI_RATE_HZ: float = 2.0
    TELEMETRY_HEALTH_INTERVAL_SECONDS: int = 10
    TELEMETRY_DETECTION_THROTTLE_MS: int = 500
    TELEMETRY_STATUSTEXT_SEVERITY: str = "INFO"
    TELEMETRY_MAX_BANDWIDTH_KBPS: float = 10.0


@dataclass
class HomingConfig:
    """Homing algorithm configuration."""

    HOMING_FORWARD_VELOCITY_MAX: float = 5.0
    HOMING_YAW_RATE_MAX: float = 0.5
    HOMING_APPROACH_VELOCITY: float = 1.0
    HOMING_SIGNAL_LOSS_TIMEOUT: float = 5.0
    HOMING_ALGORITHM_MODE: str = "GRADIENT"
    HOMING_GRADIENT_WINDOW_SIZE: int = 10
    HOMING_GRADIENT_MIN_SNR: float = 10.0
    HOMING_SAMPLING_TURN_RADIUS: float = 10.0
    HOMING_SAMPLING_DURATION: float = 5.0
    HOMING_APPROACH_THRESHOLD: float = -50.0
    HOMING_PLATEAU_VARIANCE: float = 2.0
    HOMING_VELOCITY_SCALE_FACTOR: float = 0.1


@dataclass
class HardwareConfig:
    """Hardware configuration for SDR and MAVLink."""

    # SDR hardware settings
    HARDWARE_SDR_DEVICE: str = "hackrf"
    HARDWARE_SDR_FREQUENCY: int = 3200000000
    HARDWARE_SDR_SAMPLE_RATE: int = 20000000
    HARDWARE_SDR_LNA_GAIN: int = 16
    HARDWARE_SDR_VGA_GAIN: int = 20
    HARDWARE_SDR_AMP_ENABLE: bool = False
    HARDWARE_SDR_ANTENNA: str = "log_periodic"

    # MAVLink hardware settings
    HARDWARE_MAVLINK_CONNECTION: str = "serial:///dev/ttyACM0:115200"
    HARDWARE_MAVLINK_FALLBACK: str = "serial:///dev/ttyACM1:115200"
    HARDWARE_MAVLINK_HEARTBEAT_RATE: int = 1
    HARDWARE_MAVLINK_TIMEOUT: int = 10
    HARDWARE_MAVLINK_SYSTEM_ID: int = 255
    HARDWARE_MAVLINK_COMPONENT_ID: int = 190

    # Beacon settings
    HARDWARE_BEACON_MODE: str = "software"
    HARDWARE_BEACON_FREQUENCY_MIN: int = 850000000
    HARDWARE_BEACON_FREQUENCY_MAX: int = 6500000000
    HARDWARE_BEACON_PULSE_WIDTH: float = 0.001
    HARDWARE_BEACON_PULSE_PERIOD: float = 0.1

    # Performance settings
    HARDWARE_PERFORMANCE_RSSI_UPDATE_RATE: int = 1
    HARDWARE_PERFORMANCE_TELEMETRY_RATE: int = 4
    HARDWARE_PERFORMANCE_LOG_LEVEL: str = "INFO"
    HARDWARE_PERFORMANCE_MAX_MEMORY_MB: int = 512
    HARDWARE_PERFORMANCE_CPU_TARGET: int = 30


@dataclass
class NetworkConfig:
    """Network performance and packet loss threshold configuration."""

    NETWORK_PACKET_LOSS_LOW_THRESHOLD: float = 0.01  # 1%
    NETWORK_PACKET_LOSS_MEDIUM_THRESHOLD: float = 0.05  # 5%
    NETWORK_PACKET_LOSS_HIGH_THRESHOLD: float = 0.10  # 10%
    NETWORK_PACKET_LOSS_CRITICAL_THRESHOLD: float = 0.20  # 20%
    NETWORK_CONGESTION_DETECTOR_ENABLED: bool = True
    NETWORK_BASELINE_LATENCY_MS: float = 0.0
    NETWORK_LATENCY_THRESHOLD_MS: float = 100.0

    # Enhanced fields for runtime adjustment capability
    NETWORK_RUNTIME_ADJUSTMENT_ENABLED: bool = True
    NETWORK_OPERATOR_OVERRIDE_ENABLED: bool = True
    NETWORK_MONITORING_INTERVAL_MS: int = 1000  # 1 second
    NETWORK_ADAPTIVE_RATE_ENABLED: bool = True

    def __post_init__(self):
        """Validate packet loss thresholds are within acceptable bounds."""
        thresholds = [
            self.NETWORK_PACKET_LOSS_LOW_THRESHOLD,
            self.NETWORK_PACKET_LOSS_MEDIUM_THRESHOLD,
            self.NETWORK_PACKET_LOSS_HIGH_THRESHOLD,
            self.NETWORK_PACKET_LOSS_CRITICAL_THRESHOLD,
        ]

        # Validate bounds (0.001-0.5 range)
        for threshold in thresholds:
            if not (0.001 <= threshold <= 0.5):
                raise ValueError(
                    f"Packet loss threshold must be between 0.001 and 0.5, got {threshold}"
                )

        # Validate threshold ordering (low < medium < high < critical)
        if not (thresholds[0] < thresholds[1] < thresholds[2] < thresholds[3]):
            raise ValueError(
                "Packet loss thresholds must be in ascending order: "
                f"low({thresholds[0]}) < medium({thresholds[1]}) < "
                f"high({thresholds[2]}) < critical({thresholds[3]})"
            )

    def get_threshold_by_severity(self, severity: str) -> float:
        """Get packet loss threshold by severity level."""
        severity_map = {
            "low": self.NETWORK_PACKET_LOSS_LOW_THRESHOLD,
            "medium": self.NETWORK_PACKET_LOSS_MEDIUM_THRESHOLD,
            "high": self.NETWORK_PACKET_LOSS_HIGH_THRESHOLD,
            "critical": self.NETWORK_PACKET_LOSS_CRITICAL_THRESHOLD,
        }

        if severity not in severity_map:
            raise ValueError(f"Unknown severity level: {severity}")

        return severity_map[severity]

    def update_threshold(self, severity: str, value: float) -> None:
        """Update individual threshold with validation."""
        # Validate bounds
        if not (0.001 <= value <= 0.5):
            raise ValueError(
                f"Packet loss threshold must be between 0.001 and 0.5, got {value}"
            )

        # Get current thresholds for ordering validation
        current_thresholds = {
            "low": self.NETWORK_PACKET_LOSS_LOW_THRESHOLD,
            "medium": self.NETWORK_PACKET_LOSS_MEDIUM_THRESHOLD,
            "high": self.NETWORK_PACKET_LOSS_HIGH_THRESHOLD,
            "critical": self.NETWORK_PACKET_LOSS_CRITICAL_THRESHOLD,
        }

        # Update the threshold
        current_thresholds[severity] = value

        # Validate ordering
        thresholds = list(current_thresholds.values())
        if not (thresholds[0] < thresholds[1] < thresholds[2] < thresholds[3]):
            raise ValueError(
                f"Updating {severity} threshold to {value} would break threshold ordering"
            )

        # Apply the update
        if severity == "low":
            self.NETWORK_PACKET_LOSS_LOW_THRESHOLD = value
        elif severity == "medium":
            self.NETWORK_PACKET_LOSS_MEDIUM_THRESHOLD = value
        elif severity == "high":
            self.NETWORK_PACKET_LOSS_HIGH_THRESHOLD = value
        elif severity == "critical":
            self.NETWORK_PACKET_LOSS_CRITICAL_THRESHOLD = value
        else:
            raise ValueError(f"Unknown severity level: {severity}")

    def to_dict(self) -> dict[str, Any]:
        """Convert NetworkConfig to dictionary for API responses."""
        return {
            "NETWORK_PACKET_LOSS_LOW_THRESHOLD": self.NETWORK_PACKET_LOSS_LOW_THRESHOLD,
            "NETWORK_PACKET_LOSS_MEDIUM_THRESHOLD": self.NETWORK_PACKET_LOSS_MEDIUM_THRESHOLD,
            "NETWORK_PACKET_LOSS_HIGH_THRESHOLD": self.NETWORK_PACKET_LOSS_HIGH_THRESHOLD,
            "NETWORK_PACKET_LOSS_CRITICAL_THRESHOLD": self.NETWORK_PACKET_LOSS_CRITICAL_THRESHOLD,
            "NETWORK_CONGESTION_DETECTOR_ENABLED": self.NETWORK_CONGESTION_DETECTOR_ENABLED,
            "NETWORK_BASELINE_LATENCY_MS": self.NETWORK_BASELINE_LATENCY_MS,
            "NETWORK_LATENCY_THRESHOLD_MS": self.NETWORK_LATENCY_THRESHOLD_MS,
            "NETWORK_RUNTIME_ADJUSTMENT_ENABLED": self.NETWORK_RUNTIME_ADJUSTMENT_ENABLED,
            "NETWORK_OPERATOR_OVERRIDE_ENABLED": self.NETWORK_OPERATOR_OVERRIDE_ENABLED,
            "NETWORK_MONITORING_INTERVAL_MS": self.NETWORK_MONITORING_INTERVAL_MS,
            "NETWORK_ADAPTIVE_RATE_ENABLED": self.NETWORK_ADAPTIVE_RATE_ENABLED,
        }


@dataclass
class DevelopmentConfig:
    """Development settings."""

    DEV_HOT_RELOAD: bool = True
    DEV_DEBUG_MODE: bool = False
    DEV_MOCK_SDR: bool = False


@dataclass
class Config:
    """Main configuration container."""

    app: AppConfig = field(default_factory=AppConfig)
    sdr: SDRConfig = field(default_factory=SDRConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    interferometry: InterferometryConfig = field(default_factory=InterferometryConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    api: APIConfig = field(default_factory=APIConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    homing: HomingConfig = field(default_factory=HomingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    development: DevelopmentConfig = field(default_factory=DevelopmentConfig)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "app": self.app.__dict__,
            "sdr": self.sdr.__dict__,
            "signal": self.signal.__dict__,
            "interferometry": self.interferometry.__dict__,
            "database": self.database.__dict__,
            "logging": self.logging.__dict__,
            "websocket": self.websocket.__dict__,
            "safety": self.safety.__dict__,
            "performance": self.performance.__dict__,
            "api": self.api.__dict__,
            "monitoring": self.monitoring.__dict__,
            "telemetry": self.telemetry.__dict__,
            "homing": self.homing.__dict__,
            "hardware": self.hardware.__dict__,
            "network": self.network.__dict__,
            "development": self.development.__dict__,
        }


class ConfigLoader:
    """Configuration loader that handles YAML files and environment variables."""

    def __init__(self, config_path: str | Path | None = None):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to configuration file. Defaults to profile-based selection.
        """
        if config_path is None:
            # Get project root (3 levels up from this file)
            project_root = Path(__file__).parent.parent.parent.parent

            # Select configuration profile based on environment
            profile = os.getenv("PISAD_CONFIG_PROFILE", "default")
            if profile in ["development", "dev"]:
                config_file = "development.yaml"
            elif profile in ["production", "prod"]:
                config_file = "production.yaml"
            else:
                config_file = "default.yaml"

            self.config_path = project_root / "config" / config_file
            logger.info(f"Selected configuration profile: {profile} -> {config_file}")
        else:
            self.config_path = Path(config_path)
        self.config = Config()

    def load(self) -> Config:
        """
        Load configuration from file and environment variables.

        Environment variables override file configuration.

        Returns:
            Loaded configuration object
        """
        # Load configuration with inheritance support
        config_data = self._load_with_inheritance()

        if config_data:
            try:
                # Validate merged configuration data
                from src.backend.core.config_validator import ConfigValidator

                validator = ConfigValidator()
                is_valid, errors = validator.validate_config_dict(config_data)
                if not is_valid:
                    error_msg = "Configuration validation failed:\n" + "\n".join(
                        f"  - {error}" for error in errors
                    )
                    raise ValueError(error_msg)

                # Apply validated configuration
                self._apply_yaml_config(config_data)
                logger.info(
                    f"Loaded and validated configuration from {self.config_path}"
                )
            except (ConfigurationError, ValueError) as e:
                logger.error(
                    f"Failed to load configuration from {self.config_path}: {e}"
                )
                raise ConfigurationError(str(e))
        else:
            logger.warning(
                f"Configuration file {self.config_path} not found, using defaults"
            )

        # Load hardware configuration (with fallback to mock)
        self._load_hardware_config()

        # Override with environment variables
        self._apply_env_overrides()

        # Load default configuration profile if available
        self._load_default_profile()

        # Validate configuration after all loading is complete
        self._validate_config()

        return self.config

    def _load_with_inheritance(self) -> dict[str, Any] | None:
        """
        Load configuration with inheritance from base configuration.

        Returns:
            Merged configuration dictionary or None if file not found
        """
        if not self.config_path.exists():
            return None

        # Load the specific configuration file
        with open(self.config_path) as f:
            config_data = yaml.safe_load(f) or {}

        # Check if this is a profile-specific config (not default.yaml)
        if self.config_path.name != "default.yaml":
            # Load base configuration for inheritance
            base_config_path = self.config_path.parent / "default.yaml"
            if base_config_path.exists():
                try:
                    with open(base_config_path) as f:
                        base_config = yaml.safe_load(f) or {}

                    # Merge base config with profile-specific config
                    # Profile-specific settings override base settings
                    merged_config = {**base_config, **config_data}
                    logger.info(f"Inherited base configuration from {base_config_path}")
                    return merged_config

                except Exception as e:
                    logger.warning(f"Failed to load base configuration: {e}")
                    # Fall back to profile-only configuration
                    return config_data

        return config_data

    def _load_hardware_config(self) -> None:
        """Load hardware configuration with fallback to mock configuration."""
        project_root = Path(__file__).parent.parent.parent.parent
        hardware_config_path = project_root / "config" / "hardware.yaml"
        mock_config_path = project_root / "config" / "hardware-mock.yaml"

        # Try loading hardware.yaml first
        config_loaded = False
        if hardware_config_path.exists():
            try:
                # Check if we can detect hardware
                from src.backend.services.hardware_detector import HardwareDetector

                detector = HardwareDetector()

                # Simple check - don't wait for full detection
                import asyncio

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    # Quick detection with timeout
                    status = loop.run_until_complete(
                        asyncio.wait_for(detector.detect_all(), timeout=2.0)
                    )

                    if status.sdr_connected or status.flight_controller_connected:
                        # Hardware detected, use production config
                        with open(hardware_config_path) as f:
                            hw_config = yaml.safe_load(f)
                            self._apply_hardware_yaml(hw_config)
                            logger.info("Loaded production hardware configuration")
                            config_loaded = True
                except (TimeoutError, Exception):
                    logger.info("Hardware not detected, will use mock configuration")
                finally:
                    loop.close()

            except ImportError:
                logger.warning(
                    "Hardware detector not available, using mock configuration"
                )
            except SDRError as e:
                logger.warning(f"Failed to load hardware config: {e}")

        # Fall back to mock configuration if hardware config not loaded
        if not config_loaded and mock_config_path.exists():
            try:
                with open(mock_config_path) as f:
                    mock_config = yaml.safe_load(f)
                    self._apply_hardware_yaml(mock_config)
                    logger.info("Loaded mock hardware configuration for testing")
            except SDRError as e:
                logger.error(f"Failed to load mock hardware config: {e}")

        # Override with USE_MOCK_HARDWARE environment variable
        if (
            os.environ.get("USE_MOCK_HARDWARE", "").lower() in ("true", "1", "yes")
            and mock_config_path.exists()
        ):
            try:
                with open(mock_config_path) as f:
                    mock_config = yaml.safe_load(f)
                    self._apply_hardware_yaml(mock_config)
                    logger.info("Forced mock hardware configuration via environment")
            except ConfigurationError as e:
                logger.error(f"Failed to load mock config: {e}")

    def _apply_hardware_yaml(self, hw_config: dict[str, Any]) -> None:
        """Apply hardware-specific YAML configuration."""
        if not hw_config:
            return

        # SDR configuration
        if "sdr" in hw_config:
            sdr = hw_config["sdr"]
            self.config.hardware.HARDWARE_SDR_DEVICE = sdr.get("device", "hackrf")
            self.config.hardware.HARDWARE_SDR_FREQUENCY = sdr.get(
                "frequency", 3200000000
            )
            self.config.hardware.HARDWARE_SDR_SAMPLE_RATE = sdr.get(
                "sample_rate", 20000000
            )
            self.config.hardware.HARDWARE_SDR_LNA_GAIN = sdr.get("lna_gain", 16)
            self.config.hardware.HARDWARE_SDR_VGA_GAIN = sdr.get("vga_gain", 20)
            self.config.hardware.HARDWARE_SDR_AMP_ENABLE = sdr.get("amp_enable", False)
            self.config.hardware.HARDWARE_SDR_ANTENNA = sdr.get(
                "antenna", "log_periodic"
            )

        # MAVLink configuration
        if "mavlink" in hw_config:
            mav = hw_config["mavlink"]
            self.config.hardware.HARDWARE_MAVLINK_CONNECTION = mav.get(
                "connection", "serial:///dev/ttyACM0:115200"
            )
            self.config.hardware.HARDWARE_MAVLINK_FALLBACK = mav.get(
                "fallback", "serial:///dev/ttyACM1:115200"
            )
            self.config.hardware.HARDWARE_MAVLINK_HEARTBEAT_RATE = mav.get(
                "heartbeat_rate", 1
            )
            self.config.hardware.HARDWARE_MAVLINK_TIMEOUT = mav.get("timeout", 10)
            self.config.hardware.HARDWARE_MAVLINK_SYSTEM_ID = mav.get("system_id", 255)
            self.config.hardware.HARDWARE_MAVLINK_COMPONENT_ID = mav.get(
                "component_id", 190
            )

        # Beacon configuration
        if "beacon" in hw_config:
            beacon = hw_config["beacon"]
            self.config.hardware.HARDWARE_BEACON_MODE = beacon.get("mode", "software")
            self.config.hardware.HARDWARE_BEACON_FREQUENCY_MIN = beacon.get(
                "frequency_min", 850000000
            )
            self.config.hardware.HARDWARE_BEACON_FREQUENCY_MAX = beacon.get(
                "frequency_max", 6500000000
            )
            self.config.hardware.HARDWARE_BEACON_PULSE_WIDTH = beacon.get(
                "pulse_width", 0.001
            )
            self.config.hardware.HARDWARE_BEACON_PULSE_PERIOD = beacon.get(
                "pulse_period", 0.1
            )

        # Performance configuration
        if "performance" in hw_config:
            perf = hw_config["performance"]
            self.config.hardware.HARDWARE_PERFORMANCE_RSSI_UPDATE_RATE = perf.get(
                "rssi_update_rate", 1
            )
            self.config.hardware.HARDWARE_PERFORMANCE_TELEMETRY_RATE = perf.get(
                "telemetry_rate", 4
            )
            self.config.hardware.HARDWARE_PERFORMANCE_LOG_LEVEL = perf.get(
                "log_level", "INFO"
            )
            self.config.hardware.HARDWARE_PERFORMANCE_MAX_MEMORY_MB = perf.get(
                "max_memory_mb", 512
            )
            self.config.hardware.HARDWARE_PERFORMANCE_CPU_TARGET = perf.get(
                "cpu_target", 30
            )

    def _load_default_profile(self) -> None:
        """Load the default configuration profile and apply its settings."""
        try:
            config_service = ConfigService()
            default_profile = config_service.get_default_profile()

            if default_profile:
                logger.info(f"Loading default profile: {default_profile.name}")

                # Apply SDR configuration
                if default_profile.sdrConfig:
                    self.config.sdr.SDR_FREQUENCY = int(
                        default_profile.sdrConfig.frequency
                    )
                    self.config.sdr.SDR_SAMPLE_RATE = int(
                        default_profile.sdrConfig.sampleRate
                    )
                    if isinstance(default_profile.sdrConfig.gain, int | float):
                        self.config.sdr.SDR_GAIN = int(default_profile.sdrConfig.gain)

                # Apply signal configuration
                if default_profile.signalConfig:
                    self.config.signal.SIGNAL_RSSI_THRESHOLD = (
                        default_profile.signalConfig.triggerThreshold
                    )

                # Apply safety/homing configuration
                if default_profile.homingConfig:
                    self.config.safety.SAFETY_VELOCITY_MAX_MPS = (
                        default_profile.homingConfig.forwardVelocityMax
                    )

                logger.info(
                    f"Applied settings from default profile: {default_profile.name}"
                )

        except ConfigurationError as e:
            logger.warning(f"Could not load default profile: {e}")

    def _apply_yaml_config(self, yaml_config: dict[str, Any]) -> None:
        """Apply configuration from YAML dictionary with proper type conversion."""
        if not yaml_config:
            return

        # Map flat YAML keys to nested config structure with type conversion
        for key, value in yaml_config.items():
            # Determine which config section this belongs to and apply with type conversion
            if key.startswith("APP_"):
                self._set_config_value(self.config.app, key, str(value))
            elif key.startswith("SDR_"):
                self._set_config_value(self.config.sdr, key, str(value))
            elif key.startswith("SIGNAL_"):
                self._set_config_value(self.config.signal, key, str(value))
            elif key.startswith("INTERFEROMETRY_"):
                self._set_config_value(self.config.interferometry, key, str(value))
            elif key.startswith("DB_"):
                self._set_config_value(self.config.database, key, str(value))
            elif key.startswith("LOG_"):
                self._set_config_value(self.config.logging, key, str(value))
            elif key.startswith("WS_"):
                self._set_config_value(self.config.websocket, key, str(value))
            elif key.startswith("SAFETY_"):
                self._set_config_value(self.config.safety, key, str(value))
            elif key.startswith("PERF_"):
                self._set_config_value(self.config.performance, key, str(value))
            elif key.startswith("API_"):
                self._set_config_value(self.config.api, key, str(value))
            elif key.startswith("MONITORING_"):
                self._set_config_value(self.config.monitoring, key, str(value))
            elif key.startswith("TELEMETRY_"):
                self._set_config_value(self.config.telemetry, key, str(value))
            elif key.startswith("HOMING_"):
                self._set_config_value(self.config.homing, key, str(value))
            elif key.startswith("HARDWARE_"):
                self._set_config_value(self.config.hardware, key, str(value))
            elif key.startswith("NETWORK_"):
                self._set_config_value(self.config.network, key, str(value))
            elif key.startswith("DEV_"):
                self._set_config_value(self.config.development, key, str(value))

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        # Get all environment variables starting with PISAD_
        env_prefix = "PISAD_"

        for env_key, env_value in os.environ.items():
            if not env_key.startswith(env_prefix):
                continue

            # Remove prefix to get config key
            config_key = env_key[len(env_prefix) :]

            # Determine which config section this belongs to
            if config_key.startswith("APP_"):
                self._set_config_value(self.config.app, config_key, env_value)
            elif config_key.startswith("SDR_"):
                self._set_config_value(self.config.sdr, config_key, env_value)
            elif config_key.startswith("SIGNAL_"):
                self._set_config_value(self.config.signal, config_key, env_value)
            elif config_key.startswith("INTERFEROMETRY_"):
                self._set_config_value(
                    self.config.interferometry, config_key, env_value
                )
            elif config_key.startswith("DB_"):
                self._set_config_value(self.config.database, config_key, env_value)
            elif config_key.startswith("LOG_"):
                self._set_config_value(self.config.logging, config_key, env_value)
            elif config_key.startswith("WS_"):
                self._set_config_value(self.config.websocket, config_key, env_value)
            elif config_key.startswith("SAFETY_"):
                self._set_config_value(self.config.safety, config_key, env_value)
            elif config_key.startswith("PERF_"):
                self._set_config_value(self.config.performance, config_key, env_value)
            elif config_key.startswith("API_"):
                self._set_config_value(self.config.api, config_key, env_value)
            elif config_key.startswith("MONITORING_"):
                self._set_config_value(self.config.monitoring, config_key, env_value)
            elif config_key.startswith("TELEMETRY_"):
                self._set_config_value(self.config.telemetry, config_key, env_value)
            elif config_key.startswith("HOMING_"):
                self._set_config_value(self.config.homing, config_key, env_value)
            elif config_key.startswith("HARDWARE_"):
                self._set_config_value(self.config.hardware, config_key, env_value)
            elif config_key.startswith("NETWORK_"):
                self._set_config_value(self.config.network, config_key, env_value)
            elif config_key.startswith("DEV_"):
                self._set_config_value(self.config.development, config_key, env_value)

    def _set_config_value(self, config_section: Any, key: str, value: str) -> None:
        """
        Set configuration value with appropriate type conversion.

        Args:
            config_section: Configuration section object
            key: Configuration key
            value: String value from environment
        """
        if not hasattr(config_section, key):
            logger.warning(f"Unknown configuration key: {key}")
            return

        # Get the current value to determine type
        current_value = getattr(config_section, key)

        # Convert based on type
        converted_value: Any
        if isinstance(current_value, bool):
            # Convert string to boolean
            converted_value = value.lower() in ("true", "1", "yes", "on")
        elif isinstance(current_value, int):
            try:
                converted_value = int(value)
            except ValueError:
                logger.error(f"Invalid integer value for {key}: {value}")
                return
        elif isinstance(current_value, float):
            try:
                converted_value = float(value)
            except ValueError:
                logger.error(f"Invalid float value for {key}: {value}")
                return
        elif isinstance(current_value, list):
            # Parse comma-separated list
            converted_value = [v.strip() for v in value.split(",")]
        else:
            # Keep as string
            converted_value = value

        setattr(config_section, key, converted_value)
        logger.debug(f"Set {key} = {converted_value} from environment")

    def _validate_config(self) -> None:
        """Validate configuration after all loading is complete."""
        # Validate network configuration thresholds
        thresholds = [
            self.config.network.NETWORK_PACKET_LOSS_LOW_THRESHOLD,
            self.config.network.NETWORK_PACKET_LOSS_MEDIUM_THRESHOLD,
            self.config.network.NETWORK_PACKET_LOSS_HIGH_THRESHOLD,
            self.config.network.NETWORK_PACKET_LOSS_CRITICAL_THRESHOLD,
        ]

        # Validate bounds (0.001-0.5 range)
        for threshold in thresholds:
            if not (0.001 <= threshold <= 0.5):
                raise ValueError(
                    f"Packet loss threshold must be between 0.001 and 0.5, got {threshold}"
                )

        # Validate threshold ordering (low < medium < high < critical)
        if not (thresholds[0] < thresholds[1] < thresholds[2] < thresholds[3]):
            raise ValueError(
                "Packet loss thresholds must be in ascending order: "
                f"low({thresholds[0]}) < medium({thresholds[1]}) < "
                f"high({thresholds[2]}) < critical({thresholds[3]})"
            )


# Global configuration instance
_config: Config | None = None


def get_config(config_path: str | Path | None = None) -> Config:
    """
    Get configuration instance (singleton pattern).

    Args:
        config_path: Optional path to configuration file

    Returns:
        Configuration object
    """
    global _config

    if _config is None:
        loader = ConfigLoader(config_path)
        _config = loader.load()

    return _config


def reload_config(config_path: str | Path | None = None) -> Config:
    """
    Reload configuration from file and environment.

    Args:
        config_path: Optional path to configuration file

    Returns:
        Reloaded configuration object
    """
    global _config

    loader = ConfigLoader(config_path)
    _config = loader.load()

    return _config
