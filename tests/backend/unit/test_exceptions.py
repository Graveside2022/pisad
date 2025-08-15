"""Test custom exception hierarchy and error handling."""

import sqlite3
from unittest.mock import Mock

import pytest

from src.backend.core.exceptions import (
    CallbackError,
    ConfigurationError,
    DatabaseError,
    HardwareError,
    MAVLinkError,
    PISADException,
    SafetyInterlockError,
    SDRError,
    SignalProcessingError,
    StateTransitionError,
)


class TestExceptionHierarchy:
    """Test exception inheritance and properties."""

    def test_base_exception(self):
        """Test PISADException is base for all custom exceptions."""
        exc = PISADException("test error")
        assert isinstance(exc, Exception)
        assert str(exc) == "test error"

    def test_signal_processing_error(self):
        """Test SignalProcessingError inheritance."""
        exc = SignalProcessingError("FFT failed")
        assert isinstance(exc, PISADException)
        assert str(exc) == "FFT failed"

    def test_mavlink_error(self):
        """Test MAVLinkError for communication issues."""
        exc = MAVLinkError("No heartbeat received")
        assert isinstance(exc, PISADException)
        assert str(exc) == "No heartbeat received"

    def test_state_transition_error(self):
        """Test StateTransitionError for invalid transitions."""
        exc = StateTransitionError("Cannot transition from IDLE to HOLDING")
        assert isinstance(exc, PISADException)

    def test_hardware_error_hierarchy(self):
        """Test HardwareError and SDRError relationship."""
        hw_exc = HardwareError("Device not found")
        sdr_exc = SDRError("HackRF not connected")

        assert isinstance(hw_exc, PISADException)
        assert isinstance(sdr_exc, HardwareError)
        assert isinstance(sdr_exc, PISADException)

    def test_configuration_error(self):
        """Test ConfigurationError for config issues."""
        exc = ConfigurationError("Invalid YAML format")
        assert isinstance(exc, PISADException)

    def test_safety_interlock_error(self):
        """Test SafetyInterlockError for safety violations."""
        exc = SafetyInterlockError("Emergency stop triggered")
        assert isinstance(exc, PISADException)

    def test_callback_error(self):
        """Test CallbackError for callback failures."""
        exc = CallbackError("State callback failed")
        assert isinstance(exc, PISADException)

    def test_database_error(self):
        """Test DatabaseError for database operations."""
        exc = DatabaseError("Failed to initialize database")
        assert isinstance(exc, PISADException)


class TestExceptionUsage:
    """Test exception usage in actual code patterns."""

    def test_database_error_from_sqlite(self):
        """Test database error wrapping sqlite errors."""
        try:
            # Simulate sqlite error
            raise sqlite3.Error("table not found")
        except sqlite3.Error as e:
            with pytest.raises(DatabaseError) as exc_info:
                raise DatabaseError(f"Database operation failed: {e}") from e

            assert "Database operation failed" in str(exc_info.value)
            assert exc_info.value.__cause__ is not None

    def test_mavlink_error_handling(self):
        """Test MAVLink error in connection scenario."""
        mock_connection = Mock()
        mock_connection.wait_heartbeat.return_value = None

        with pytest.raises(MAVLinkError) as exc_info:
            if not mock_connection.wait_heartbeat(timeout=5):
                raise MAVLinkError("No heartbeat received")

        assert "No heartbeat received" in str(exc_info.value)

    def test_callback_error_handling(self):
        """Test callback error handling pattern."""

        def failing_callback(state):
            raise ValueError("Invalid state data")

        callbacks = [failing_callback]
        errors = []

        for callback in callbacks:
            try:
                callback("SEARCHING")
            except Exception as e:
                errors.append(CallbackError(f"Callback {callback.__name__} failed: {e}"))

        assert len(errors) == 1
        assert isinstance(errors[0], CallbackError)
        assert "failing_callback" in str(errors[0])

    def test_safety_interlock_error(self):
        """Test safety interlock violation."""

        def check_safety():
            snr = 5.0
            threshold = 10.0
            if snr < threshold:
                raise SafetyInterlockError(f"SNR {snr} below threshold {threshold}")
            return True

        with pytest.raises(SafetyInterlockError) as exc_info:
            check_safety()

        assert "SNR 5.0 below threshold 10.0" in str(exc_info.value)

    def test_state_transition_error(self):
        """Test invalid state transition."""
        current_state = "IDLE"
        target_state = "HOLDING"
        valid_transitions = {"IDLE": ["SEARCHING", "DETECTING"]}

        with pytest.raises(StateTransitionError) as exc_info:
            if target_state not in valid_transitions.get(current_state, []):
                raise StateTransitionError(
                    f"Cannot transition from {current_state} to {target_state}"
                )

        assert "Cannot transition from IDLE to HOLDING" in str(exc_info.value)

    def test_sdr_error_handling(self):
        """Test SDR hardware error."""

        def initialize_sdr():
            device_found = False
            if not device_found:
                raise SDRError("No HackRF device found")
            return True

        with pytest.raises(SDRError) as exc_info:
            initialize_sdr()

        assert isinstance(exc_info.value, HardwareError)
        assert "No HackRF device found" in str(exc_info.value)

    def test_exception_chaining(self):
        """Test proper exception chaining with 'from'."""
        try:
            # Original error
            raise ValueError("Invalid value")
        except ValueError as e:
            with pytest.raises(ConfigurationError) as exc_info:
                raise ConfigurationError("Configuration failed") from e

            assert exc_info.value.__cause__ is not None
            assert isinstance(exc_info.value.__cause__, ValueError)
            assert "Invalid value" in str(exc_info.value.__cause__)


class TestExceptionMetrics:
    """Test to validate refactoring metrics."""

    def test_exception_imports(self):
        """Verify exception module imports work correctly."""
        from src.backend.core import exceptions

        assert hasattr(exceptions, "PISADException")
        assert hasattr(exceptions, "SignalProcessingError")
        assert hasattr(exceptions, "MAVLinkError")
        assert hasattr(exceptions, "StateTransitionError")
        assert hasattr(exceptions, "HardwareError")
        assert hasattr(exceptions, "SDRError")
        assert hasattr(exceptions, "ConfigurationError")
        assert hasattr(exceptions, "SafetyInterlockError")
        assert hasattr(exceptions, "CallbackError")
        assert hasattr(exceptions, "DatabaseError")

    def test_exception_hierarchy_depth(self):
        """Test exception hierarchy is properly structured."""
        # Base level
        assert PISADException.__bases__ == (Exception,)

        # Second level - all should inherit from PISADException
        second_level = [
            SignalProcessingError,
            MAVLinkError,
            StateTransitionError,
            HardwareError,
            ConfigurationError,
            SafetyInterlockError,
            CallbackError,
            DatabaseError,
        ]

        for exc_class in second_level:
            assert PISADException in exc_class.__mro__

        # Third level - SDRError inherits from HardwareError
        assert HardwareError in SDRError.__mro__
        assert PISADException in SDRError.__mro__
