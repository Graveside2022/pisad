"""Parameter persistence system for Mission Planner RF control.

TASK-6.3.1 Parameter Persistence Implementation
Provides persistent storage for Mission Planner RF parameters across system restarts
with backup/restore functionality and parameter migration support.
"""

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class ParameterEntry:
    """Individual parameter entry with metadata."""

    value: float
    last_modified: float = field(default_factory=time.time)
    source: str = "default"  # "default", "mission_planner", "asv_service", "operator"
    validation_status: str = "valid"  # "valid", "invalid", "pending"
    backup_count: int = 0


@dataclass
class ParameterStorageConfig:
    """Configuration for parameter storage system."""

    storage_directory: Path = Path("/home/pisad/projects/pisad/config/parameters")
    backup_directory: Path = Path(
        "/home/pisad/projects/pisad/config/parameters/backups"
    )
    max_backups: int = 10
    auto_backup_interval_hours: int = 24
    validation_timeout_seconds: float = 30.0
    migration_enabled: bool = True


class ParameterStorage:
    """
    Parameter persistence system for Mission Planner RF control.

    Provides:
    - File-based parameter storage with JSON serialization
    - Automatic backup and restore functionality
    - Parameter validation on startup with correction
    - Version migration system for compatibility updates
    - Thread-safe parameter operations
    """

    def __init__(self, config: ParameterStorageConfig | None = None):
        """Initialize parameter storage system.

        Args:
            config: Storage configuration (uses defaults if None)
        """
        self._config = config or ParameterStorageConfig()
        self._parameters: dict[str, ParameterEntry] = {}
        self._storage_file = self._config.storage_directory / "pisad_parameters.json"
        self._metadata_file = self._config.storage_directory / "parameter_metadata.json"
        self._lock = asyncio.Lock() if hasattr(asyncio, "Lock") else None

        # Version tracking for migrations
        self._current_version = "1.0.0"
        self._compatible_versions = ["1.0.0"]

        # Performance metrics
        self._load_time_ms = 0.0
        self._save_count = 0
        self._backup_count = 0
        self._last_auto_backup = 0.0

        self._ensure_directories()
        logger.info("Parameter storage system initialized")

    def _ensure_directories(self) -> None:
        """Ensure storage and backup directories exist."""
        try:
            self._config.storage_directory.mkdir(parents=True, exist_ok=True)
            self._config.backup_directory.mkdir(parents=True, exist_ok=True)
            logger.debug("Parameter storage directories verified")
        except OSError as e:
            logger.error(f"Failed to create parameter directories: {e}")
            raise

    def load_parameters(self) -> dict[str, float]:
        """Load parameters from persistent storage with validation.

        Returns:
            Dictionary of parameter names to values

        Raises:
            ParameterStorageError: If loading fails critically
        """
        start_time = time.perf_counter()

        try:
            if not self._storage_file.exists():
                logger.info("No existing parameter file found, using defaults")
                return self._get_default_parameters()

            # Load parameters from file
            with open(self._storage_file, "r") as f:
                data = json.load(f)

            # Validate file format and version
            if not self._validate_storage_format(data):
                logger.warning("Invalid parameter file format, using defaults")
                return self._get_default_parameters()

            # Load parameter entries
            self._parameters = {}
            for param_name, param_data in data.get("parameters", {}).items():
                try:
                    entry = ParameterEntry(
                        value=float(param_data["value"]),
                        last_modified=param_data.get("last_modified", time.time()),
                        source=param_data.get("source", "file"),
                        validation_status=param_data.get("validation_status", "valid"),
                        backup_count=param_data.get("backup_count", 0),
                    )
                    self._parameters[param_name] = entry
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(f"Skipping invalid parameter {param_name}: {e}")

            # Validate loaded parameters
            validated_params = self._validate_parameters()

            self._load_time_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                f"Loaded {len(validated_params)} parameters in {self._load_time_ms:.1f}ms"
            )

            return validated_params

        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load parameters: {e}")
            return self._get_default_parameters()

    def save_parameters(
        self, parameters: dict[str, float], source: str = "mission_planner"
    ) -> bool:
        """Save parameters to persistent storage.

        Args:
            parameters: Dictionary of parameter names to values
            source: Source of parameter changes

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Update parameter entries
            current_time = time.time()
            for param_name, value in parameters.items():
                if param_name in self._parameters:
                    # Update existing parameter
                    self._parameters[param_name].value = value
                    self._parameters[param_name].last_modified = current_time
                    self._parameters[param_name].source = source
                else:
                    # Create new parameter entry
                    self._parameters[param_name] = ParameterEntry(
                        value=value, last_modified=current_time, source=source
                    )

            # Create backup if needed
            if self._should_create_backup():
                self._create_backup()

            # Prepare data for storage
            storage_data = {
                "version": self._current_version,
                "last_updated": current_time,
                "parameters": {},
                "metadata": {
                    "save_count": self._save_count + 1,
                    "backup_count": self._backup_count,
                    "source": source,
                },
            }

            # Serialize parameter entries
            for param_name, entry in self._parameters.items():
                storage_data["parameters"][param_name] = {
                    "value": entry.value,
                    "last_modified": entry.last_modified,
                    "source": entry.source,
                    "validation_status": entry.validation_status,
                    "backup_count": entry.backup_count,
                }

            # Write to file atomically
            temp_file = self._storage_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(storage_data, f, indent=2)

            # Atomic rename
            temp_file.replace(self._storage_file)

            self._save_count += 1
            logger.debug(f"Saved {len(parameters)} parameters from {source}")

            return True

        except (OSError, json.JSONEncodeError) as e:
            logger.error(f"Failed to save parameters: {e}")
            return False

    def get_parameter(self, param_name: str) -> float | None:
        """Get individual parameter value.

        Args:
            param_name: Parameter name

        Returns:
            Parameter value or None if not found
        """
        entry = self._parameters.get(param_name)
        return entry.value if entry else None

    def set_parameter(self, param_name: str, value: float, source: str = "api") -> bool:
        """Set individual parameter value.

        Args:
            param_name: Parameter name
            value: Parameter value
            source: Source of change

        Returns:
            True if set successfully
        """
        try:
            current_time = time.time()

            if param_name in self._parameters:
                self._parameters[param_name].value = value
                self._parameters[param_name].last_modified = current_time
                self._parameters[param_name].source = source
            else:
                self._parameters[param_name] = ParameterEntry(
                    value=value, last_modified=current_time, source=source
                )

            # Save individual parameter change
            return self.save_parameters({param_name: value}, source)

        except Exception as e:
            logger.error(f"Failed to set parameter {param_name}: {e}")
            return False

    def create_backup(self, backup_name: str | None = None) -> str | None:
        """Create manual backup of current parameters.

        Args:
            backup_name: Optional backup name (timestamp used if None)

        Returns:
            Backup file path or None if failed
        """
        try:
            if backup_name is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                backup_name = f"pisad_parameters_backup_{timestamp}.json"

            backup_path = self._config.backup_directory / backup_name

            if self._storage_file.exists():
                shutil.copy2(self._storage_file, backup_path)
                self._backup_count += 1
                logger.info(f"Created parameter backup: {backup_path}")
                return str(backup_path)
            else:
                logger.warning("No parameter file to backup")
                return None

        except OSError as e:
            logger.error(f"Failed to create backup: {e}")
            return None

    def restore_backup(self, backup_path: str | Path) -> bool:
        """Restore parameters from backup file.

        Args:
            backup_path: Path to backup file

        Returns:
            True if restored successfully
        """
        try:
            backup_file = Path(backup_path)

            if not backup_file.exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False

            # Validate backup file
            with open(backup_file, "r") as f:
                backup_data = json.load(f)

            if not self._validate_storage_format(backup_data):
                logger.error(f"Invalid backup file format: {backup_path}")
                return False

            # Create backup of current state before restore
            current_backup = self.create_backup("pre_restore_backup")
            if current_backup:
                logger.info(f"Created pre-restore backup: {current_backup}")

            # Copy backup to storage location
            shutil.copy2(backup_file, self._storage_file)

            # Reload parameters
            self.load_parameters()

            logger.info(f"Restored parameters from backup: {backup_path}")
            return True

        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Failed to restore backup: {e}")
            return False

    def migrate_parameters(self, source_version: str, target_version: str) -> bool:
        """Migrate parameters between versions.

        Args:
            source_version: Source version string
            target_version: Target version string

        Returns:
            True if migration successful
        """
        if not self._config.migration_enabled:
            logger.info("Parameter migration disabled")
            return True

        try:
            # Create pre-migration backup
            backup_name = f"pre_migration_{source_version}_to_{target_version}.json"
            self.create_backup(backup_name)

            # Apply version-specific migrations
            if source_version == "1.0.0" and target_version == "1.1.0":
                # Example migration: Add new parameters with defaults
                self._migrate_v1_0_to_v1_1()

            logger.info(
                f"Migrated parameters from {source_version} to {target_version}"
            )
            return True

        except Exception as e:
            logger.error(f"Parameter migration failed: {e}")
            return False

    def _get_default_parameters(self) -> dict[str, float]:
        """Get default parameter values for Mission Planner RF control."""
        return {
            # Core Frequency Control Parameters
            "PISAD_RF_FREQ": 406000000.0,  # 406MHz emergency beacon
            "PISAD_RF_PROFILE": 0.0,  # Emergency profile
            "PISAD_RF_BW": 25000.0,  # 25kHz bandwidth
            # Homing Control Parameters
            "PISAD_HOMING_EN": 0.0,  # Disabled by default
            "PISAD_HOMING_STATE": 0.0,  # Disabled state
            # Signal Quality Parameters (read-only)
            "PISAD_SIG_CLASS": 0.0,  # FM_CHIRP
            "PISAD_SIG_CONF": 0.0,  # No signal initially
            "PISAD_BEARING": 0.0,  # No bearing initially
            "PISAD_BEAR_CONF": 0.0,  # No bearing confidence
            "PISAD_INTERFERENCE": 0.0,  # No interference initially
            # Enhanced Signal Telemetry Parameters (TASK-6.3.2)
            "PISAD_SIG_CHAR": 0.0,  # Unknown characteristics
            "PISAD_DETECT_EVENT": 0.0,  # No detection event
            "PISAD_CONF_TREND": 0.0,  # No confidence trend
            "PISAD_CONF_STATUS": 0.0,  # Below threshold
            "PISAD_BEAR_PREC": 0.0,  # No bearing precision
            "PISAD_BEAR_RATE": 0.0,  # No bearing rate
            "PISAD_INTERF_TYPE": 0.0,  # No interference
            "PISAD_INTERF_BEAR": 0.0,  # No interference bearing
            "PISAD_REJECT_STATUS": 0.0,  # No rejection active
            # System Health Parameters
            "PISAD_RF_HEALTH": 0.0,  # Unknown health
            "PISAD_EMERGENCY_DISABLE": 0.0,  # Not disabled
            "PISAD_RESPONSE_TIME": 0.0,  # No response time yet
        }

    def _validate_storage_format(self, data: dict[str, Any]) -> bool:
        """Validate parameter storage file format."""
        required_fields = ["version", "parameters"]

        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required field in parameter file: {field}")
                return False

        # Check version compatibility
        file_version = data.get("version", "unknown")
        if file_version not in self._compatible_versions:
            logger.warning(f"Parameter file version {file_version} may be incompatible")
            # Continue anyway for now, but could trigger migration

        return True

    def _validate_parameters(self) -> dict[str, float]:
        """Validate loaded parameters against expected ranges."""
        validated = {}

        # Define parameter validation rules
        validation_rules = {
            "PISAD_RF_FREQ": (1000000.0, 6000000000.0),  # 1MHz - 6GHz
            "PISAD_RF_PROFILE": (0.0, 2.0),  # 0=Emergency, 1=Aviation, 2=Custom
            "PISAD_RF_BW": (1000.0, 10000000.0),  # 1kHz - 10MHz
            "PISAD_HOMING_EN": (0.0, 1.0),  # Boolean
            "PISAD_HOMING_STATE": (0.0, 3.0),  # 0=Disabled, 1=Armed, 2=Active, 3=Lost
            "PISAD_SIG_CLASS": (0.0, 3.0),  # Signal classification
            "PISAD_SIG_CONF": (0.0, 100.0),  # Confidence percentage
            "PISAD_BEARING": (0.0, 360.0),  # Bearing degrees
            "PISAD_BEAR_CONF": (0.0, 100.0),  # Bearing confidence
            "PISAD_INTERFERENCE": (0.0, 100.0),  # Interference level
            "PISAD_RF_HEALTH": (0.0, 100.0),  # Health percentage
            "PISAD_EMERGENCY_DISABLE": (0.0, 1.0),  # Boolean
            "PISAD_RESPONSE_TIME": (0.0, 1000.0),  # Response time ms
            # Enhanced Signal Telemetry Parameters (TASK-6.3.2)
            "PISAD_SIG_CHAR": (0.0, 10.0),  # Signal characteristics
            "PISAD_DETECT_EVENT": (0.0, 100.0),  # Detection event encoding
            "PISAD_CONF_TREND": (-10.0, 10.0),  # Confidence trend
            "PISAD_CONF_STATUS": (0.0, 2.0),  # Threshold status
            "PISAD_BEAR_PREC": (0.0, 180.0),  # Bearing precision degrees
            "PISAD_BEAR_RATE": (-180.0, 180.0),  # Bearing rate deg/s
            "PISAD_INTERF_TYPE": (0.0, 4.0),  # Interference type
            "PISAD_INTERF_BEAR": (0.0, 360.0),  # Interference bearing
            "PISAD_REJECT_STATUS": (0.0, 100.0),  # Rejection effectiveness
        }

        for param_name, entry in self._parameters.items():
            if param_name in validation_rules:
                min_val, max_val = validation_rules[param_name]
                if min_val <= entry.value <= max_val:
                    validated[param_name] = entry.value
                    entry.validation_status = "valid"
                else:
                    logger.warning(
                        f"Parameter {param_name}={entry.value} out of range [{min_val}, {max_val}], using default"
                    )
                    # Use default value for out-of-range parameters
                    defaults = self._get_default_parameters()
                    validated[param_name] = defaults.get(param_name, 0.0)
                    entry.value = validated[param_name]
                    entry.validation_status = "corrected"
            else:
                # Unknown parameter, keep as-is but log warning
                logger.warning(
                    f"Unknown parameter {param_name}, keeping value {entry.value}"
                )
                validated[param_name] = entry.value
                entry.validation_status = "unknown"

        return validated

    def _should_create_backup(self) -> bool:
        """Check if automatic backup should be created."""
        if not self._config.auto_backup_interval_hours:
            return False

        current_time = time.time()
        time_since_backup = current_time - self._last_auto_backup
        interval_seconds = self._config.auto_backup_interval_hours * 3600

        return time_since_backup >= interval_seconds

    def _create_backup(self) -> None:
        """Create automatic backup and cleanup old backups."""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_name = f"auto_backup_{timestamp}.json"
            backup_path = self.create_backup(backup_name)

            if backup_path:
                self._last_auto_backup = time.time()
                self._cleanup_old_backups()

        except Exception as e:
            logger.error(f"Failed to create automatic backup: {e}")

    def _cleanup_old_backups(self) -> None:
        """Remove old backup files exceeding max_backups limit."""
        try:
            backup_files = list(self._config.backup_directory.glob("*.json"))

            if len(backup_files) > self._config.max_backups:
                # Sort by modification time (oldest first)
                backup_files.sort(key=lambda f: f.stat().st_mtime)

                # Remove excess backups
                files_to_remove = backup_files[: -self._config.max_backups]
                for file_path in files_to_remove:
                    file_path.unlink()
                    logger.debug(f"Removed old backup: {file_path}")

                logger.info(f"Cleaned up {len(files_to_remove)} old backup files")

        except Exception as e:
            logger.error(f"Failed to cleanup old backups: {e}")

    def _migrate_v1_0_to_v1_1(self) -> None:
        """Example migration from version 1.0 to 1.1."""
        # Add new parameters introduced in v1.1
        new_parameters = {
            "PISAD_NEW_FEATURE": 0.0,
        }

        for param_name, default_value in new_parameters.items():
            if param_name not in self._parameters:
                self._parameters[param_name] = ParameterEntry(
                    value=default_value, source="migration", validation_status="valid"
                )
                logger.info(f"Added new parameter during migration: {param_name}")

    def get_storage_statistics(self) -> dict[str, Any]:
        """Get parameter storage statistics."""
        return {
            "total_parameters": len(self._parameters),
            "load_time_ms": self._load_time_ms,
            "save_count": self._save_count,
            "backup_count": self._backup_count,
            "storage_file_size": (
                self._storage_file.stat().st_size if self._storage_file.exists() else 0
            ),
            "last_auto_backup": self._last_auto_backup,
            "version": self._current_version,
            "validation_summary": self._get_validation_summary(),
        }

    def _get_validation_summary(self) -> dict[str, int]:
        """Get summary of parameter validation statuses."""
        summary = {"valid": 0, "invalid": 0, "corrected": 0, "unknown": 0}

        for entry in self._parameters.values():
            status = entry.validation_status
            if status in summary:
                summary[status] += 1
            else:
                summary["unknown"] += 1

        return summary

    def store_parameter(
        self, param_name: str, value: float, source: str = "validation"
    ) -> bool:
        """
        Store individual parameter value.

        TASK-6.3.1 [28b2] - Individual parameter storage for validation pipeline

        Args:
            param_name: Parameter name
            value: Parameter value
            source: Source of parameter change

        Returns:
            True if stored successfully
        """
        return self.set_parameter(param_name, value, source)

    def store_parameters_validated(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store parameters with validation pipeline integration.

        TASK-6.3.1 [28b2] - Parameter validation before persistence

        Args:
            parameters: Dictionary of parameters to validate and store

        Returns:
            Dict containing validation and storage results
        """
        try:
            # Import here to avoid circular imports
            from .mavlink_service import MAVLinkService

            # Create temporary service instance for validation
            mavlink_service = MAVLinkService()

            # Validate parameter set
            validation_result = mavlink_service.validate_parameter_set(parameters)

            if not validation_result["valid"]:
                return {
                    "success": False,
                    "validation_passed": False,
                    "validation_errors": validation_result["validation_results"],
                    "message": "Parameter validation failed",
                }

            # If validation passes, store the parameters
            stored_count = 0
            for param_name, value in parameters.items():
                if self.store_parameter(param_name, value):
                    stored_count += 1

            return {
                "success": True,
                "validation_passed": True,
                "stored_count": stored_count,
                "total_count": len(parameters),
                "validation_results": validation_result["validation_results"],
            }

        except Exception as e:
            logger.error(f"Error in validated parameter storage: {e}")
            return {
                "success": False,
                "validation_passed": False,
                "error": str(e),
                "message": "Storage operation failed",
            }


# Import asyncio at module level for lock creation
import asyncio
