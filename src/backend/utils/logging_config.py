"""
Advanced logging configuration for PISAD application.
Enhanced rotation, disk monitoring, and structured logging.
"""

import logging
import logging.handlers
import os
import shutil
import sys
import time
import uuid
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Context variable for correlation ID
correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


class AdvancedRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Enhanced rotating file handler with time-based rotation and disk monitoring."""

    def __init__(
        self,
        filename: str,
        mode: str = "a",
        maxBytes: int = 10485760,  # 10 MB
        backupCount: int = 7,  # 7 days worth
        encoding: str | None = None,
        delay: bool = False,
        enable_daily_rotation: bool = True,
        min_free_space_mb: int = 100,  # Minimum free space in MB
    ):
        """
        Initialize advanced rotating file handler.

        Args:
            filename: Log file path
            mode: File mode
            maxBytes: Maximum bytes per file (default 10MB)
            backupCount: Number of backup files to keep
            encoding: File encoding
            delay: Delay file opening
            enable_daily_rotation: Enable daily rotation
            min_free_space_mb: Minimum free disk space in MB
        """
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)
        self.enable_daily_rotation = enable_daily_rotation
        self.min_free_space_mb = min_free_space_mb
        self.last_rotation_date = datetime.now().date()

        # Ensure log directory exists
        log_path = Path(filename)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    def should_rollover(self, record: logging.LogRecord) -> bool:
        """
        Determine if rollover should occur.

        Args:
            record: Log record to be written

        Returns:
            True if rollover should occur
        """
        # Check size-based rollover (from parent class)
        size_rollover = False
        if self.maxBytes > 0:
            # Ensure stream is open
            if self.stream is None:
                self.stream = self._open()

            msg = "%s\n" % self.format(record)
            self.stream.seek(0, 2)  # Go to end of file
            if self.stream.tell() + len(msg) >= self.maxBytes:
                size_rollover = True

        # Check daily rollover
        daily_rollover = False
        if self.enable_daily_rotation:
            current_date = datetime.now().date()
            daily_rollover = current_date > self.last_rotation_date

        # Check disk space
        disk_space_low = self._is_disk_space_low()

        return size_rollover or daily_rollover or disk_space_low

    def doRollover(self) -> None:
        """Perform rollover with enhanced cleanup and disk monitoring."""
        # Check if we need emergency cleanup due to low disk space
        if self._is_disk_space_low():
            self._emergency_cleanup()

        # Update rotation date
        self.last_rotation_date = datetime.now().date()

        # Add timestamp to rotated file
        if self.stream:
            self.stream.close()
            self.stream = None

        # Create timestamped backup filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = self.baseFilename
        backup_filename = f"{base_filename}.{timestamp}"

        # Rotate current file
        if os.path.exists(self.baseFilename):
            os.rename(self.baseFilename, backup_filename)

        # Clean up old backups
        self._cleanup_old_backups()

        # Open new file
        if not self.delay:
            self.stream = self._open()

    def _is_disk_space_low(self) -> bool:
        """Check if disk space is below minimum threshold."""
        try:
            log_path = Path(self.baseFilename)
            stat = shutil.disk_usage(log_path.parent)
            free_space_mb = stat.free / (1024 * 1024)
            return free_space_mb < self.min_free_space_mb
        except Exception:
            return False

    def _emergency_cleanup(self) -> None:
        """Perform emergency cleanup when disk space is low."""
        log_dir = Path(self.baseFilename).parent
        log_files = []

        # Find all log backup files
        base_name = Path(self.baseFilename).stem
        for file_path in log_dir.glob(f"{base_name}.*"):
            if file_path.is_file() and file_path.name != Path(self.baseFilename).name:
                log_files.append(file_path)

        # Sort by modification time (oldest first)
        log_files.sort(key=lambda f: f.stat().st_mtime)

        # Remove oldest files until we have enough space or reach minimum count
        removed_count = 0
        for file_path in log_files:
            if removed_count >= len(log_files) - 1:  # Keep at least 1 backup
                break

            try:
                file_path.unlink()
                removed_count += 1
                logging.warning(f"Emergency cleanup: removed {file_path}")

                # Check if we have enough space now
                if not self._is_disk_space_low():
                    break
            except Exception as e:
                logging.error(f"Failed to remove log file {file_path}: {e}")

    def _cleanup_old_backups(self) -> None:
        """Clean up old backup files beyond retention period."""
        if self.backupCount <= 0:
            return

        log_dir = Path(self.baseFilename).parent
        base_name = Path(self.baseFilename).stem

        # Find all backup files
        backup_files = []
        for file_path in log_dir.glob(f"{base_name}.*"):
            if file_path.is_file() and file_path.name != Path(self.baseFilename).name:
                backup_files.append(file_path)

        # Sort by modification time (newest first)
        backup_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        # Remove files beyond backup count
        for file_path in backup_files[self.backupCount :]:
            try:
                file_path.unlink()
                logging.debug(f"Cleaned up old log file: {file_path}")
            except Exception as e:
                logging.error(f"Failed to clean up log file {file_path}: {e}")


class CorrelationIdFilter(logging.Filter):
    """Add correlation ID to log records for request tracking."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to the log record."""
        record.correlation_id = correlation_id.get() or "no-correlation-id"
        return True


class ModuleLevelFilter(logging.Filter):
    """Filter logs based on module-specific log levels."""

    def __init__(self, module_levels: Dict[str, str] | None = None):
        """
        Initialize module level filter.

        Args:
            module_levels: Dictionary mapping module names to log levels
        """
        super().__init__()
        self.module_levels = module_levels or {}

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter record based on module-specific log level."""
        # Get module name from logger name
        module_name = record.name.split(".")[0] if "." in record.name else record.name

        # Check if we have a specific level for this module
        if module_name in self.module_levels:
            module_level = getattr(logging, self.module_levels[module_name].upper())
            return bool(record.levelno >= module_level)

        return True


class PerformanceLoggingFormatter(logging.Formatter):
    """Enhanced formatter with performance timing and correlation IDs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with enhanced information."""
        # Add correlation ID to the message if present
        if (
            hasattr(record, "correlation_id")
            and record.correlation_id != "no-correlation-id"
        ):
            record.msg = f"[{record.correlation_id}] {record.getMessage()}"
            record.args = ()  # Clear args to prevent re-formatting

        # Add performance timing if available
        if hasattr(record, "duration_ms"):
            record.msg = f"{record.getMessage()} (took {record.duration_ms:.2f}ms)"
            record.args = ()

        return super().format(record)


def setup_advanced_logging(
    log_level: str = "INFO",
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_file_path: str | None = None,
    log_file_max_bytes: int = 10485760,  # 10 MB
    log_file_backup_count: int = 7,  # 7 days
    enable_console: bool = True,
    enable_file: bool = True,
    enable_journal: bool = True,
    enable_daily_rotation: bool = True,
    min_free_space_mb: int = 100,
    module_levels: Dict[str, str] | None = None,
) -> None:
    """
    Set up advanced logging configuration with enhanced rotation and monitoring.

    Args:
        log_level: Global logging level
        log_format: Log message format string
        log_file_path: Path to log file
        log_file_max_bytes: Maximum size of log file before rotation
        log_file_backup_count: Number of backup files to keep (days)
        enable_console: Enable console logging
        enable_file: Enable file logging
        enable_journal: Enable systemd journal logging
        enable_daily_rotation: Enable daily log rotation
        min_free_space_mb: Minimum free disk space in MB
        module_levels: Dictionary of module-specific log levels
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Create enhanced formatter
    formatter = PerformanceLoggingFormatter(log_format)

    # Add correlation ID filter
    correlation_filter = CorrelationIdFilter()

    # Add module level filter if specified
    module_filter = ModuleLevelFilter(module_levels) if module_levels else None

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(correlation_filter)
        if module_filter:
            console_handler.addFilter(module_filter)
        root_logger.addHandler(console_handler)

    # Enhanced file handler with rotation
    if enable_file and log_file_path:
        file_handler = AdvancedRotatingFileHandler(
            filename=log_file_path,
            maxBytes=log_file_max_bytes,
            backupCount=log_file_backup_count,
            enable_daily_rotation=enable_daily_rotation,
            min_free_space_mb=min_free_space_mb,
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(correlation_filter)
        if module_filter:
            file_handler.addFilter(module_filter)
        root_logger.addHandler(file_handler)

    # Systemd journal handler
    if enable_journal:
        try:
            from systemd.journal import JournalHandler

            journal_handler = JournalHandler(SYSLOG_IDENTIFIER="rf-homing")
            journal_handler.setFormatter(formatter)
            journal_handler.addFilter(correlation_filter)
            if module_filter:
                journal_handler.addFilter(module_filter)
            root_logger.addHandler(journal_handler)
        except ImportError:
            # systemd-python not installed or not on Linux
            if sys.platform.startswith("linux"):
                logging.warning(
                    "systemd-python not installed, journal logging disabled"
                )

    logging.info(f"Advanced logging configured with level: {log_level}")
    if enable_daily_rotation:
        logging.info("Daily log rotation enabled")
    if min_free_space_mb:
        logging.info(f"Disk space monitoring enabled (minimum: {min_free_space_mb}MB)")


def set_correlation_id(request_id: str | None = None) -> str:
    """
    Set correlation ID for the current context.

    Args:
        request_id: Optional correlation ID. If not provided, generates a new UUID.

    Returns:
        The correlation ID that was set
    """
    if request_id is None:
        request_id = str(uuid.uuid4())

    correlation_id.set(request_id)
    return request_id


def get_correlation_id() -> str | None:
    """
    Get the current correlation ID.

    Returns:
        Current correlation ID or None
    """
    return correlation_id.get()


def clear_correlation_id() -> None:
    """Clear the current correlation ID."""
    correlation_id.set(None)


class PerformanceTimer:
    """Context manager for performance timing with automatic logging."""

    def __init__(
        self, operation: str, logger: logging.Logger, log_level: int = logging.INFO
    ):
        """
        Initialize performance timer.

        Args:
            operation: Name of the operation being timed
            logger: Logger instance to use
            log_level: Log level for timing messages
        """
        self.operation = operation
        self.logger = logger
        self.log_level = log_level
        self.start_time: float | None = None
        self.duration: float | None = None

    def __enter__(self) -> "PerformanceTimer":
        """Start timing."""
        self.start_time = time.perf_counter()
        self.logger.log(self.log_level, f"Starting operation: {self.operation}")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """End timing and log result."""
        if self.start_time is not None:
            self.duration = time.perf_counter() - self.start_time
            duration_ms = self.duration * 1000

            # Create log record with performance timing
            record = self.logger.makeRecord(
                self.logger.name,
                self.log_level,
                __file__,
                0,
                f"Completed operation: {self.operation}",
                (),
                None,
            )
            record.duration_ms = duration_ms
            self.logger.handle(record)


def log_performance(operation: str, logger: logging.Logger) -> PerformanceTimer:
    """
    Create a performance timer context manager.

    Args:
        operation: Name of the operation to time
        logger: Logger instance to use

    Returns:
        PerformanceTimer context manager
    """
    return PerformanceTimer(operation, logger)
