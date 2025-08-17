"""Unit tests for core exceptions.

Tests exception hierarchy and error handling.
"""

from src.backend.core.exceptions import (
    CallbackError,
    MAVLinkError,
    PISADException,
    SafetyInterlockError,
    SDRError,
    SignalProcessingError,
)


class TestCoreExceptions:
    """Test core exception classes."""

    def test_pisad_exception_base(self):
        """Test base PISAD exception."""
        msg = "Test error message"
        exc = PISADException(msg)

        assert str(exc) == msg
        assert isinstance(exc, Exception)

    def test_sdr_error_inheritance(self):
        """Test SDR error inherits from base."""
        exc = SDRError("SDR test error")

        assert isinstance(exc, PISADException)
        assert isinstance(exc, Exception)

    def test_mavlink_error_inheritance(self):
        """Test MAVLink error inherits from base."""
        exc = MAVLinkError("MAVLink test error")

        assert isinstance(exc, PISADException)
        assert isinstance(exc, Exception)

    def test_signal_processing_error_inheritance(self):
        """Test signal processing error inherits from base."""
        exc = SignalProcessingError("Signal processing test error")

        assert isinstance(exc, PISADException)
        assert isinstance(exc, Exception)

    def test_safety_interlock_error_inheritance(self):
        """Test safety interlock error inherits from base."""
        exc = SafetyInterlockError("Safety interlock test error")

        assert isinstance(exc, PISADException)
        assert isinstance(exc, Exception)

    def test_callback_error_inheritance(self):
        """Test callback error inherits from base."""
        exc = CallbackError("Callback test error")

        assert isinstance(exc, PISADException)
        assert isinstance(exc, Exception)

    def test_exception_with_details(self):
        """Test exceptions can carry additional details."""
        details = {"error_code": 123, "context": "test"}
        exc = SDRError("Error with details", details)

        assert str(exc) == "Error with details"
        # Exception should have details if supported
        if hasattr(exc, "details"):
            assert exc.details == details

    def test_exception_chaining(self):
        """Test exception chaining works correctly."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise SDRError("SDR error") from e
        except SDRError as sdr_exc:
            assert str(sdr_exc) == "SDR error"
            assert isinstance(sdr_exc.__cause__, ValueError)

    def test_all_exceptions_are_importable(self):
        """Test all exception classes can be imported."""
        # This test ensures all exceptions are properly defined
        exceptions = [
            PISADException,
            SDRError,
            MAVLinkError,
            SignalProcessingError,
            SafetyInterlockError,
            CallbackError,
        ]

        for exc_class in exceptions:
            assert issubclass(exc_class, Exception)
            # Test instantiation
            instance = exc_class("test message")
            assert isinstance(instance, Exception)
