"""
Tests for advanced logging configuration system.
"""

import logging
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from src.backend.utils.logging_config import (
    AdvancedRotatingFileHandler,
    CorrelationIdFilter,
    ModuleLevelFilter,
    PerformanceLoggingFormatter,
    PerformanceTimer,
    clear_correlation_id,
    get_correlation_id,
    log_performance,
    set_correlation_id,
    setup_advanced_logging,
)


class TestAdvancedRotatingFileHandler:
    """Test advanced rotating file handler functionality."""

    def test_size_based_rotation(self):
        """Test rotation based on file size."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"

            # Create handler with small max size
            handler = AdvancedRotatingFileHandler(
                str(log_file),
                maxBytes=1024,  # 1KB
                backupCount=3,
            )

            # Create large log record
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="x" * 2000,  # 2KB message
                args=(),
                exc_info=None,
            )

            # Should trigger rollover
            assert handler.should_rollover(record)

    def test_daily_rotation(self):
        """Test daily rotation functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"

            handler = AdvancedRotatingFileHandler(str(log_file), enable_daily_rotation=True)

            # Simulate previous day
            from datetime import date, timedelta

            handler.last_rotation_date = date.today() - timedelta(days=1)

            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="test message",
                args=(),
                exc_info=None,
            )

            # Should trigger daily rollover
            assert handler.should_rollover(record)

    def test_disk_space_monitoring(self):
        """Test disk space monitoring functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"

            handler = AdvancedRotatingFileHandler(
                str(log_file),
                min_free_space_mb=999999,  # Impossibly high threshold
            )

            # Should detect low disk space
            assert handler._is_disk_space_low()

    def test_emergency_cleanup(self):
        """Test emergency cleanup when disk space is low."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"

            # Create some backup files
            (Path(temp_dir) / "test.log.20250101_120000").touch()
            (Path(temp_dir) / "test.log.20250102_120000").touch()
            (Path(temp_dir) / "test.log.20250103_120000").touch()

            handler = AdvancedRotatingFileHandler(
                str(log_file),
                min_free_space_mb=999999,  # Force emergency cleanup
            )

            # Trigger emergency cleanup
            handler._emergency_cleanup()

            # Should have removed some files but kept at least 1
            remaining_files = list(Path(temp_dir).glob("test.log.*"))
            assert len(remaining_files) >= 1

    def test_backup_cleanup(self):
        """Test cleanup of old backup files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"

            # Create more backup files than backup count
            for i in range(10):
                backup_file = Path(temp_dir) / f"test.log.backup{i}"
                backup_file.touch()
                # Set different modification times
                os.utime(backup_file, (time.time() - i * 3600, time.time() - i * 3600))

            handler = AdvancedRotatingFileHandler(str(log_file), backupCount=3)

            handler._cleanup_old_backups()

            # Should keep only backupCount files
            backup_files = list(Path(temp_dir).glob("test.log.backup*"))
            assert len(backup_files) <= 3


class TestCorrelationIdFilter:
    """Test correlation ID filter functionality."""

    def test_adds_correlation_id(self):
        """Test that filter adds correlation ID to log records."""
        filter_obj = CorrelationIdFilter()

        # Set correlation ID
        test_id = "test-correlation-123"
        set_correlation_id(test_id)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )

        # Filter should add correlation ID
        filter_obj.filter(record)
        assert hasattr(record, "correlation_id")
        assert record.correlation_id == test_id

        # Cleanup
        clear_correlation_id()

    def test_default_correlation_id(self):
        """Test default correlation ID when none is set."""
        filter_obj = CorrelationIdFilter()

        # Ensure no correlation ID is set
        clear_correlation_id()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )

        filter_obj.filter(record)
        assert record.correlation_id == "no-correlation-id"


class TestModuleLevelFilter:
    """Test module-level filtering functionality."""

    def test_module_specific_filtering(self):
        """Test filtering based on module-specific log levels."""
        module_levels = {"test_module": "ERROR", "other_module": "DEBUG"}

        filter_obj = ModuleLevelFilter(module_levels)

        # Create records for different modules
        info_record = logging.LogRecord(
            name="test_module.submodule",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="info message",
            args=(),
            exc_info=None,
        )

        error_record = logging.LogRecord(
            name="test_module.submodule",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="error message",
            args=(),
            exc_info=None,
        )

        # INFO should be filtered out (below ERROR level)
        assert not filter_obj.filter(info_record)

        # ERROR should pass through
        assert filter_obj.filter(error_record)

    def test_default_filtering(self):
        """Test default behavior for modules without specific levels."""
        filter_obj = ModuleLevelFilter({"specific_module": "ERROR"})

        record = logging.LogRecord(
            name="unknown_module",
            level=logging.DEBUG,
            pathname="",
            lineno=0,
            msg="debug message",
            args=(),
            exc_info=None,
        )

        # Should pass through (no specific level configured)
        assert filter_obj.filter(record)


class TestPerformanceLoggingFormatter:
    """Test performance logging formatter."""

    def test_correlation_id_formatting(self):
        """Test correlation ID is added to log messages."""
        formatter = PerformanceLoggingFormatter()

        # Set correlation ID
        test_id = "perf-test-123"
        set_correlation_id(test_id)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )

        # Add correlation ID to record (normally done by filter)
        record.correlation_id = test_id

        formatted = formatter.format(record)
        assert f"[{test_id}]" in formatted

        # Cleanup
        clear_correlation_id()

    def test_performance_timing_formatting(self):
        """Test performance timing is added to log messages."""
        formatter = PerformanceLoggingFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="operation completed",
            args=(),
            exc_info=None,
        )

        # Add performance timing
        record.duration_ms = 123.45

        formatted = formatter.format(record)
        assert "123.45ms" in formatted


class TestPerformanceTimer:
    """Test performance timer functionality."""

    def test_performance_timing(self):
        """Test that performance timer measures duration."""
        logger = logging.getLogger("test")

        with (
            patch.object(logger, "log") as mock_log,
            patch.object(logger, "makeRecord") as mock_make_record,
            patch.object(logger, "handle") as mock_handle,
        ):
            # Mock record creation
            mock_record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="test",
                args=(),
                exc_info=None,
            )
            mock_make_record.return_value = mock_record

            with PerformanceTimer("test_operation", logger):
                time.sleep(0.01)  # Small delay

            # Should have logged start and completion
            assert mock_log.call_count == 1  # Start message
            assert mock_handle.call_count == 1  # Completion message

            # Check that duration was recorded
            args, kwargs = mock_handle.call_args
            record = args[0]
            assert hasattr(record, "duration_ms")
            assert record.duration_ms > 0

    def test_log_performance_context_manager(self):
        """Test log_performance convenience function."""
        logger = logging.getLogger("test")

        timer = log_performance("test_op", logger)
        assert isinstance(timer, PerformanceTimer)
        assert timer.operation == "test_op"
        assert timer.logger == logger


class TestCorrelationIdManagement:
    """Test correlation ID management functions."""

    def test_set_and_get_correlation_id(self):
        """Test setting and getting correlation IDs."""
        test_id = "test-123"

        # Set and verify
        returned_id = set_correlation_id(test_id)
        assert returned_id == test_id
        assert get_correlation_id() == test_id

        # Clear and verify
        clear_correlation_id()
        assert get_correlation_id() is None

    def test_auto_generate_correlation_id(self):
        """Test auto-generation of correlation IDs."""
        # Don't provide an ID
        generated_id = set_correlation_id()

        # Should be a valid UUID string
        assert len(generated_id) == 36  # UUID format
        assert "-" in generated_id
        assert get_correlation_id() == generated_id

        # Cleanup
        clear_correlation_id()


class TestAdvancedLoggingSetup:
    """Test advanced logging setup function."""

    def test_setup_with_file_logging(self):
        """Test setup with file logging enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"

            # Clear any existing handlers
            root_logger = logging.getLogger()
            root_logger.handlers.clear()

            setup_advanced_logging(
                log_level="DEBUG",
                log_file_path=str(log_file),
                enable_console=False,
                enable_journal=False,
                module_levels={"test_module": "ERROR"},
            )

            # Should have created file handler
            assert len(root_logger.handlers) == 1
            handler = root_logger.handlers[0]
            assert isinstance(handler, AdvancedRotatingFileHandler)

    def test_setup_with_module_levels(self):
        """Test setup with module-specific log levels."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"

            root_logger = logging.getLogger()
            root_logger.handlers.clear()

            module_levels = {"test_module": "ERROR", "other_module": "DEBUG"}

            setup_advanced_logging(
                log_file_path=str(log_file),
                enable_console=False,
                enable_journal=False,
                module_levels=module_levels,
            )

            # Should have applied module level filter
            handler = root_logger.handlers[0]

            # Check that module filter was added
            module_filters = [f for f in handler.filters if isinstance(f, ModuleLevelFilter)]
            assert len(module_filters) == 1
            assert module_filters[0].module_levels == module_levels
