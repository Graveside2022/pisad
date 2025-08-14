"""
Unit tests for logging utilities.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from backend.utils.logging import (
    CorrelationIdFilter,
    LogContext,
    PerformanceLogger,
    StructuredFormatter,
    clear_correlation_id,
    get_correlation_id,
    get_logger,
    log_critical,
    log_debug,
    log_error,
    log_info,
    log_warning,
    log_with_context,
    set_correlation_id,
    setup_logging,
)


class TestLoggingSetup:
    """Test logging setup functionality."""

    def teardown_method(self):
        """Clean up after each test."""
        # Clear all handlers from root logger
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        # Reset correlation ID
        clear_correlation_id()

    def test_setup_console_logging(self):
        """Test setting up console logging."""
        setup_logging(
            log_level="INFO", enable_console=True, enable_file=False, enable_journal=False
        )

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) == 1
        assert isinstance(root_logger.handlers[0], logging.StreamHandler)
        assert root_logger.level == logging.INFO

    def test_setup_file_logging(self):
        """Test setting up file logging with rotation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")

            setup_logging(
                log_level="DEBUG",
                log_file_path=log_file,
                log_file_max_bytes=1024,
                log_file_backup_count=3,
                enable_console=False,
                enable_file=True,
                enable_journal=False,
            )

            root_logger = logging.getLogger()
            assert len(root_logger.handlers) == 1
            assert isinstance(root_logger.handlers[0], logging.handlers.RotatingFileHandler)
            assert root_logger.handlers[0].maxBytes == 1024
            assert root_logger.handlers[0].backupCount == 3
            assert root_logger.level == logging.DEBUG

            # Test that log file is created
            test_logger = get_logger("test")
            test_logger.info("Test message")
            assert os.path.exists(log_file)

    def test_setup_multiple_handlers(self):
        """Test setting up multiple logging handlers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")

            setup_logging(
                log_file_path=log_file,
                enable_console=True,
                enable_file=True,
                enable_journal=False,  # Disable journal to avoid import issues
            )

            root_logger = logging.getLogger()
            assert len(root_logger.handlers) == 2

            # Check handler types
            handler_types = [type(h) for h in root_logger.handlers]
            assert logging.StreamHandler in handler_types
            assert logging.handlers.RotatingFileHandler in handler_types

    def test_setup_journal_logging(self):
        """Test setting up systemd journal logging."""
        # Create a mock JournalHandler
        mock_journal_handler_class = MagicMock()
        mock_handler = MagicMock()
        mock_handler.level = logging.INFO  # Set proper level attribute
        mock_journal_handler_class.return_value = mock_handler

        # Create mock systemd module
        mock_systemd = MagicMock()
        mock_systemd.journal.JournalHandler = mock_journal_handler_class

        # Patch the import to make it available
        with patch.dict(
            "sys.modules", {"systemd": mock_systemd, "systemd.journal": mock_systemd.journal}
        ):
            setup_logging(enable_console=False, enable_file=False, enable_journal=True)

        mock_journal_handler_class.assert_called_once_with(SYSLOG_IDENTIFIER="rf-homing")
        mock_handler.setFormatter.assert_called_once()
        mock_handler.addFilter.assert_called_once()

    def test_log_level_setting(self):
        """Test different log level settings."""
        test_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level_str in test_levels:
            setup_logging(
                log_level=level_str, enable_console=True, enable_file=False, enable_journal=False
            )
            root_logger = logging.getLogger()
            expected_level = getattr(logging, level_str)
            assert root_logger.level == expected_level
            root_logger.handlers.clear()


class TestCorrelationId:
    """Test correlation ID functionality."""

    def setup_method(self):
        """Set up before each test."""
        clear_correlation_id()

    def teardown_method(self):
        """Clean up after each test."""
        clear_correlation_id()

    def test_set_and_get_correlation_id(self):
        """Test setting and getting correlation ID."""
        assert get_correlation_id() is None

        correlation_id = set_correlation_id("test-id-123")
        assert correlation_id == "test-id-123"
        assert get_correlation_id() == "test-id-123"

    def test_auto_generate_correlation_id(self):
        """Test automatic correlation ID generation."""
        correlation_id = set_correlation_id()
        assert correlation_id is not None
        assert len(correlation_id) == 36  # UUID format
        assert get_correlation_id() == correlation_id

    def test_clear_correlation_id(self):
        """Test clearing correlation ID."""
        set_correlation_id("test-id")
        assert get_correlation_id() == "test-id"

        clear_correlation_id()
        assert get_correlation_id() is None

    def test_correlation_id_filter(self):
        """Test CorrelationIdFilter adds correlation ID to records."""
        set_correlation_id("test-correlation-123")

        filter = CorrelationIdFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = filter.filter(record)
        assert result
        assert record.correlation_id == "test-correlation-123"

    def test_correlation_id_filter_no_id(self):
        """Test CorrelationIdFilter when no correlation ID is set."""
        clear_correlation_id()

        filter = CorrelationIdFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = filter.filter(record)
        assert result
        assert record.correlation_id == "no-correlation-id"

    def test_log_context_manager(self):
        """Test LogContext context manager."""
        assert get_correlation_id() is None

        with LogContext("context-id-456") as correlation_id:
            assert correlation_id == "context-id-456"
            assert get_correlation_id() == "context-id-456"

        assert get_correlation_id() is None

    def test_log_context_auto_generate(self):
        """Test LogContext with auto-generated ID."""
        with LogContext() as correlation_id:
            assert correlation_id is not None
            assert len(correlation_id) == 36
            assert get_correlation_id() == correlation_id

        assert get_correlation_id() is None


class TestStructuredLogging:
    """Test structured logging functionality."""

    def setup_method(self):
        """Set up before each test."""
        clear_correlation_id()

    def teardown_method(self):
        """Clean up after each test."""
        clear_correlation_id()

    def test_log_with_context(self, caplog):
        """Test logging with additional context."""
        logger = get_logger("test")
        caplog.set_level(logging.INFO)

        log_with_context(
            logger, logging.INFO, "Test message", user_id="123", action="login", ip="192.168.1.1"
        )

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert "Test message" in record.message
        assert "user_id=123" in record.message
        assert "action=login" in record.message
        assert "ip=192.168.1.1" in record.message

    def test_convenience_log_functions(self, caplog):
        """Test convenience logging functions."""
        logger = get_logger("test")
        caplog.set_level(logging.DEBUG)

        log_debug(logger, "Debug message", key1="value1")
        log_info(logger, "Info message", key2="value2")
        log_warning(logger, "Warning message", key3="value3")
        log_error(logger, "Error message", key4="value4")
        log_critical(logger, "Critical message", key5="value5")

        assert len(caplog.records) == 5

        levels = [r.levelno for r in caplog.records]
        expected_levels = [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ]
        assert levels == expected_levels

        # Check context is included
        assert "key1=value1" in caplog.records[0].message
        assert "key2=value2" in caplog.records[1].message
        assert "key3=value3" in caplog.records[2].message
        assert "key4=value4" in caplog.records[3].message
        assert "key5=value5" in caplog.records[4].message

    def test_structured_formatter(self):
        """Test StructuredFormatter with correlation ID."""
        set_correlation_id("format-test-id")

        formatter = StructuredFormatter("%(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.correlation_id = "format-test-id"

        formatted = formatter.format(record)
        assert "[format-test-id] Test message" in formatted


class TestPerformanceLogger:
    """Test performance logging utilities."""

    def test_timer_operations(self, caplog):
        """Test timer start and end operations."""
        logger = get_logger("test")
        caplog.set_level(logging.INFO)

        perf_logger = PerformanceLogger(logger)

        perf_logger.start_timer("test_operation")
        import time

        time.sleep(0.01)  # Small delay
        duration = perf_logger.end_timer("test_operation")

        assert duration > 0
        assert len(caplog.records) == 1
        assert "Operation completed" in caplog.records[0].message
        assert "operation=test_operation" in caplog.records[0].message
        assert "duration_ms=" in caplog.records[0].message

    def test_timer_not_started(self, caplog):
        """Test ending a timer that wasn't started."""
        logger = get_logger("test")
        caplog.set_level(logging.WARNING)

        perf_logger = PerformanceLogger(logger)
        duration = perf_logger.end_timer("not_started")

        assert duration == 0.0
        assert len(caplog.records) == 1
        assert "Timer for operation 'not_started' was not started" in caplog.records[0].message

    def test_log_metric(self, caplog):
        """Test logging performance metrics."""
        logger = get_logger("test")
        caplog.set_level(logging.INFO)

        perf_logger = PerformanceLogger(logger)

        perf_logger.log_metric("cpu_usage", 45.2, "percent")
        perf_logger.log_metric("memory_mb", 512)

        assert len(caplog.records) == 2

        # Check first metric
        assert "Performance metric" in caplog.records[0].message
        assert "metric=cpu_usage" in caplog.records[0].message
        assert "value=45.2" in caplog.records[0].message
        assert "unit=percent" in caplog.records[0].message

        # Check second metric (no unit)
        assert "metric=memory_mb" in caplog.records[1].message
        assert "value=512" in caplog.records[1].message
        assert "unit=" not in caplog.records[1].message


class TestGetLogger:
    """Test get_logger functionality."""

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a Logger instance."""
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_get_logger_same_instance(self):
        """Test that get_logger returns the same instance for same name."""
        logger1 = get_logger("test.module")
        logger2 = get_logger("test.module")
        assert logger1 is logger2


class TestEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Set up before each test."""
        clear_correlation_id()

    def teardown_method(self):
        """Clean up after each test."""
        clear_correlation_id()

    def test_setup_logging_creates_directory(self):
        """Test that setup_logging creates log directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "subdir", "logs", "test.log")

            setup_logging(
                log_file_path=log_file, enable_console=False, enable_file=True, enable_journal=False
            )

            # Directory should be created
            assert os.path.exists(os.path.dirname(log_file))

            # Test logging works
            logger = get_logger("test")
            logger.info("Test message")
            assert os.path.exists(log_file)

    def test_empty_context_logging(self, caplog):
        """Test logging with empty context."""
        logger = get_logger("test")
        caplog.set_level(logging.INFO)

        log_with_context(logger, logging.INFO, "Test message")

        assert len(caplog.records) == 1
        assert caplog.records[0].message == "Test message"
        assert "|" not in caplog.records[0].message  # No context separator
