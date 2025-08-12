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

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """Application configuration."""
    APP_NAME: str = "PISAD"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000


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
            "development": self.development.__dict__,
        }


class ConfigLoader:
    """Configuration loader that handles YAML files and environment variables."""

    def __init__(self, config_path: str | Path | None = None):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to configuration file. Defaults to config/default.yaml
        """
        if config_path is None:
            # Get project root (3 levels up from this file)
            project_root = Path(__file__).parent.parent.parent.parent
            self.config_path = project_root / "config" / "default.yaml"
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
        # Load from YAML file if it exists
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    yaml_config = yaml.safe_load(f)
                    self._apply_yaml_config(yaml_config)
                    logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Failed to load configuration from {self.config_path}: {e}")
                raise
        else:
            logger.warning(f"Configuration file {self.config_path} not found, using defaults")

        # Override with environment variables
        self._apply_env_overrides()

        return self.config

    def _apply_yaml_config(self, yaml_config: dict[str, Any]) -> None:
        """Apply configuration from YAML dictionary."""
        if not yaml_config:
            return

        # Map flat YAML keys to nested config structure
        for key, value in yaml_config.items():
            # Determine which config section this belongs to
            if key.startswith("APP_"):
                setattr(self.config.app, key, value)
            elif key.startswith("SDR_"):
                setattr(self.config.sdr, key, value)
            elif key.startswith("SIGNAL_"):
                setattr(self.config.signal, key, value)
            elif key.startswith("INTERFEROMETRY_"):
                setattr(self.config.interferometry, key, value)
            elif key.startswith("DB_"):
                setattr(self.config.database, key, value)
            elif key.startswith("LOG_"):
                setattr(self.config.logging, key, value)
            elif key.startswith("WS_"):
                setattr(self.config.websocket, key, value)
            elif key.startswith("SAFETY_"):
                setattr(self.config.safety, key, value)
            elif key.startswith("PERF_"):
                setattr(self.config.performance, key, value)
            elif key.startswith("API_"):
                setattr(self.config.api, key, value)
            elif key.startswith("MONITORING_"):
                setattr(self.config.monitoring, key, value)
            elif key.startswith("DEV_"):
                setattr(self.config.development, key, value)

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        # Get all environment variables starting with PISAD_
        env_prefix = "PISAD_"

        for env_key, env_value in os.environ.items():
            if not env_key.startswith(env_prefix):
                continue

            # Remove prefix to get config key
            config_key = env_key[len(env_prefix):]

            # Determine which config section this belongs to
            if config_key.startswith("APP_"):
                self._set_config_value(self.config.app, config_key, env_value)
            elif config_key.startswith("SDR_"):
                self._set_config_value(self.config.sdr, config_key, env_value)
            elif config_key.startswith("SIGNAL_"):
                self._set_config_value(self.config.signal, config_key, env_value)
            elif config_key.startswith("INTERFEROMETRY_"):
                self._set_config_value(self.config.interferometry, config_key, env_value)
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
            converted_value = value.lower() in ('true', '1', 'yes', 'on')
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
            converted_value = [v.strip() for v in value.split(',')]
        else:
            # Keep as string
            converted_value = value

        setattr(config_section, key, converted_value)
        logger.debug(f"Set {key} = {converted_value} from environment")


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
