"""
Logging utilities for PISAD application.
Provides structured logging with rotation and systemd journal integration.
"""

import logging
import logging.handlers
import sys
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any

# Context variable for correlation ID
correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


class CorrelationIdFilter(logging.Filter):
    """Add correlation ID to log records for request tracking."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to the log record."""
        record.correlation_id = correlation_id.get() or "no-correlation-id"
        return True


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging with correlation IDs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with correlation ID."""
        # Add correlation ID to the message if present
        if hasattr(record, "correlation_id") and record.correlation_id != "no-correlation-id":
            original_msg = record.getMessage()
            record.msg = f"[{record.correlation_id}] {original_msg}"
            record.args = ()  # Clear args to prevent re-formatting

        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_file_path: str | None = None,
    log_file_max_bytes: int = 10485760,  # 10 MB
    log_file_backup_count: int = 5,
    enable_console: bool = True,
    enable_file: bool = True,
    enable_journal: bool = True,
) -> None:
    """
    Set up logging configuration with multiple handlers.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log message format string
        log_file_path: Path to log file
        log_file_max_bytes: Maximum size of log file before rotation
        log_file_backup_count: Number of backup files to keep
        enable_console: Enable console logging
        enable_file: Enable file logging
        enable_journal: Enable systemd journal logging
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = StructuredFormatter(log_format)

    # Add correlation ID filter
    correlation_filter = CorrelationIdFilter()

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(correlation_filter)
        root_logger.addHandler(console_handler)

    # File handler with rotation
    if enable_file and log_file_path:
        # Create log directory if it doesn't exist
        log_path = Path(log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path, maxBytes=log_file_max_bytes, backupCount=log_file_backup_count
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(correlation_filter)
        root_logger.addHandler(file_handler)

    # Systemd journal handler
    if enable_journal:
        try:
            from systemd.journal import JournalHandler

            journal_handler = JournalHandler(SYSLOG_IDENTIFIER="rf-homing")
            journal_handler.setFormatter(formatter)
            journal_handler.addFilter(correlation_filter)
            root_logger.addHandler(journal_handler)
        except ImportError:
            # systemd-python not installed or not on Linux
            if sys.platform.startswith("linux"):
                logging.warning("systemd-python not installed, journal logging disabled")

    logging.info(f"Logging configured with level: {log_level}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


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


def log_with_context(logger: logging.Logger, level: int, message: str, **context: Any) -> None:
    """
    Log a message with additional context fields.

    Args:
        logger: Logger instance
        level: Log level
        message: Log message
        **context: Additional context fields to include
    """
    if context:
        context_str = " ".join([f"{k}={v}" for k, v in context.items()])
        full_message = f"{message} | {context_str}"
    else:
        full_message = message

    logger.log(level, full_message)


class LogContext:
    """Context manager for managing correlation IDs."""

    def __init__(self, request_id: str | None = None):
        """
        Initialize log context.

        Args:
            request_id: Optional correlation ID
        """
        self.request_id = request_id
        self.token = None

    def __enter__(self) -> str:
        """Enter context and set correlation ID."""
        self.request_id = set_correlation_id(self.request_id)
        return self.request_id

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and clear correlation ID."""
        clear_correlation_id()


# Convenience functions for structured logging
def log_debug(logger: logging.Logger, message: str, **context: Any) -> None:
    """Log debug message with context."""
    log_with_context(logger, logging.DEBUG, message, **context)


def log_info(logger: logging.Logger, message: str, **context: Any) -> None:
    """Log info message with context."""
    log_with_context(logger, logging.INFO, message, **context)


def log_warning(logger: logging.Logger, message: str, **context: Any) -> None:
    """Log warning message with context."""
    log_with_context(logger, logging.WARNING, message, **context)


def log_error(logger: logging.Logger, message: str, **context: Any) -> None:
    """Log error message with context."""
    log_with_context(logger, logging.ERROR, message, **context)


def log_critical(logger: logging.Logger, message: str, **context: Any) -> None:
    """Log critical message with context."""
    log_with_context(logger, logging.CRITICAL, message, **context)


# Performance logging utilities
class PerformanceLogger:
    """Utility for logging performance metrics."""

    def __init__(self, logger: logging.Logger):
        """
        Initialize performance logger.

        Args:
            logger: Logger instance to use
        """
        self.logger = logger
        self.timers: dict[str, float] = {}

    def start_timer(self, operation: str) -> None:
        """
        Start a timer for an operation.

        Args:
            operation: Name of the operation
        """
        import time

        self.timers[operation] = time.perf_counter()

    def end_timer(self, operation: str) -> float:
        """
        End a timer and log the duration.

        Args:
            operation: Name of the operation

        Returns:
            Duration in seconds
        """
        import time

        if operation not in self.timers:
            self.logger.warning(f"Timer for operation '{operation}' was not started")
            return 0.0

        duration = time.perf_counter() - self.timers[operation]
        del self.timers[operation]

        log_info(
            self.logger,
            "Operation completed",
            operation=operation,
            duration_ms=f"{duration * 1000:.2f}",
        )

        return duration

    def log_metric(self, metric_name: str, value: Any, unit: str | None = None) -> None:
        """
        Log a performance metric.

        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Optional unit of measurement
        """
        context = {"metric": metric_name, "value": value}
        if unit:
            context["unit"] = unit

        log_info(self.logger, "Performance metric", **context)
