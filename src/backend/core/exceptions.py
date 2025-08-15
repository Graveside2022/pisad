"""
Custom exception classes for the PISAD system.

Story 4.9: Created to replace generic Exception handling with specific exceptions.
"""


class PISADException(Exception):
    """Base exception for all PISAD custom exceptions."""

    pass


class SignalProcessingError(PISADException):
    """Exception raised for signal processing errors."""

    pass


class MAVLinkError(PISADException):
    """Exception raised for MAVLink communication errors."""

    pass


class StateTransitionError(PISADException):
    """Exception raised for invalid state transitions."""

    pass


class HardwareError(PISADException):
    """Exception raised for hardware interface errors."""

    pass


class SDRError(HardwareError):
    """Exception raised for SDR hardware errors."""

    pass


class ConfigurationError(PISADException):
    """Exception raised for configuration errors."""

    pass


class SafetyInterlockError(PISADException):
    """Exception raised when safety interlocks prevent an operation."""

    pass


class CallbackError(PISADException):
    """Exception raised when a callback function fails."""

    pass


class DatabaseError(PISADException):
    """Exception raised for database operations."""

    pass
