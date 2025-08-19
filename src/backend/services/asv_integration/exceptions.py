"""Custom exceptions for ASV .NET integration.

SUBTASK-6.1.1.3-c: Implement error handling and exception translation layer

This module provides comprehensive exception handling and translation
between .NET and Python exception systems.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ASVInteropError(Exception):
    """Base exception for ASV .NET interop errors."""

    def __init__(
        self,
        message: str,
        dotnet_exception: Exception | None = None,
        error_code: str = "ASV_GENERAL",
        context: dict[str, Any] | None = None,
    ):
        """Initialize ASV interop error.

        Args:
            message: Error description
            dotnet_exception: Original .NET exception if available
            error_code: Specific error classification code
            context: Additional context information
        """
        super().__init__(message)
        self.dotnet_exception = dotnet_exception
        self.error_code = error_code
        self.context = context or {}

        # Log the exception for debugging
        logger.error(f"ASV Error [{error_code}]: {message}")
        if dotnet_exception:
            logger.error(f"  .NET Exception: {dotnet_exception}")
        if context:
            logger.error(f"  Context: {context}")

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "error_code": self.error_code,
            "context": self.context,
            "dotnet_exception": (
                str(self.dotnet_exception) if self.dotnet_exception else None
            ),
        }


class ASVAssemblyLoadError(ASVInteropError):
    """Raised when ASV .NET assembly cannot be loaded."""

    def __init__(
        self,
        message: str,
        dotnet_exception: Exception | None = None,
        assembly_path: str | None = None,
    ):
        """Initialize assembly load error.

        Args:
            message: Error description
            dotnet_exception: Original .NET exception
            assembly_path: Path to assembly that failed to load
        """
        context = {"assembly_path": assembly_path} if assembly_path else {}
        super().__init__(message, dotnet_exception, "ASV_ASSEMBLY_LOAD", context)


class ASVHardwareError(ASVInteropError):
    """Exception for ASV hardware interface errors."""

    def __init__(
        self,
        message: str,
        hardware_type: str = "unknown",
        dotnet_exception: Exception | None = None,
        context: dict[str, Any] | None = None,
    ):
        context = context or {}
        context["hardware_type"] = hardware_type
        super().__init__(message, dotnet_exception, "ASV_HARDWARE_ERROR", context)


class ASVAnalyzerError(ASVInteropError):
    """Raised when ASV analyzer operations fail."""

    def __init__(
        self,
        message: str,
        dotnet_exception: Exception | None = None,
        analyzer_type: str | None = None,
        analyzer_id: str | None = None,
    ):
        """Initialize analyzer error.

        Args:
            message: Error description
            dotnet_exception: Original .NET exception
            analyzer_type: Type of analyzer (GP, VOR, LLZ)
            analyzer_id: Specific analyzer instance ID
        """
        context = {}
        if analyzer_type:
            context["analyzer_type"] = analyzer_type
        if analyzer_id:
            context["analyzer_id"] = analyzer_id
        super().__init__(message, dotnet_exception, "ASV_ANALYZER", context)


class ASVConfigurationError(ASVInteropError):
    """Raised when ASV configuration is invalid."""

    def __init__(
        self, message: str, config_key: str | None = None, config_value: Any = None
    ):
        """Initialize configuration error.

        Args:
            message: Error description
            config_key: Configuration key that caused the error
            config_value: Invalid configuration value
        """
        context = {}
        if config_key:
            context["config_key"] = config_key
        if config_value is not None:
            context["config_value"] = str(config_value)
        super().__init__(message, None, "ASV_CONFIGURATION", context)


class ASVRuntimeError(ASVInteropError):
    """Raised when .NET runtime operations fail."""

    def __init__(
        self,
        message: str,
        dotnet_exception: Exception | None = None,
        runtime_state: str | None = None,
    ):
        """Initialize runtime error.

        Args:
            message: Error description
            dotnet_exception: Original .NET exception
            runtime_state: Current runtime state information
        """
        context = {"runtime_state": runtime_state} if runtime_state else {}
        super().__init__(message, dotnet_exception, "ASV_RUNTIME", context)


class ASVFrequencyError(ASVAnalyzerError):
    """Raised when frequency configuration is invalid."""

    def __init__(
        self,
        message: str,
        frequency_hz: int | None = None,
        valid_range: tuple[int, int] | None = None,
    ):
        """Initialize frequency error.

        Args:
            message: Error description
            frequency_hz: Invalid frequency value
            valid_range: Valid frequency range as (min_hz, max_hz)
        """
        context = {}
        if frequency_hz is not None:
            context["frequency_hz"] = frequency_hz
        if valid_range:
            context["valid_range_hz"] = {"min": valid_range[0], "max": valid_range[1]}
        super().__init__(message, None, "ASV_FREQUENCY", context)


class ASVCalibrationError(ASVAnalyzerError):
    """Raised when calibration operations fail."""

    def __init__(
        self,
        message: str,
        dotnet_exception: Exception | None = None,
        calibration_type: str | None = None,
    ):
        """Initialize calibration error.

        Args:
            message: Error description
            dotnet_exception: Original .NET exception
            calibration_type: Type of calibration that failed
        """
        context = {"calibration_type": calibration_type} if calibration_type else {}
        super().__init__(message, dotnet_exception, "ASV_CALIBRATION", context)


class ASVSignalProcessingError(ASVAnalyzerError):
    """Raised when signal processing operations fail."""

    def __init__(
        self,
        message: str,
        dotnet_exception: Exception | None = None,
        processing_stage: str | None = None,
        data_size: int | None = None,
    ):
        """Initialize signal processing error.

        Args:
            message: Error description
            dotnet_exception: Original .NET exception
            processing_stage: Stage of processing that failed
            data_size: Size of data being processed
        """
        context = {}
        if processing_stage:
            context["processing_stage"] = processing_stage
        if data_size is not None:
            context["data_size_bytes"] = data_size
        super().__init__(message, dotnet_exception, "ASV_SIGNAL_PROCESSING", context)


def translate_dotnet_exception(
    dotnet_exception: Exception, context: dict[str, Any] | None = None
) -> ASVInteropError:
    """Translate .NET exception to appropriate Python ASV exception.

    Args:
        dotnet_exception: Original .NET exception
        context: Additional context for error translation

    Returns:
        Appropriate ASVInteropError subclass
    """
    exception_type = type(dotnet_exception).__name__
    message = str(dotnet_exception)
    context = context or {}

    # Map common .NET exceptions to ASV exceptions
    exception_mappings = {
        "FileNotFoundException": ASVAssemblyLoadError,
        "FileLoadException": ASVAssemblyLoadError,
        "BadImageFormatException": ASVAssemblyLoadError,
        "ArgumentException": ASVConfigurationError,
        "ArgumentOutOfRangeException": ASVFrequencyError,
        "InvalidOperationException": ASVRuntimeError,
        "NotSupportedException": ASVRuntimeError,
        "OutOfMemoryException": ASVRuntimeError,
        "TimeoutException": ASVSignalProcessingError,
    }

    # Determine appropriate exception type
    if exception_type in exception_mappings:
        exception_class = exception_mappings[exception_type]
    else:
        # Default to generic analyzer error for unknown .NET exceptions
        exception_class = ASVAnalyzerError

    # Create translated exception with enhanced message
    translated_message = f".NET {exception_type}: {message}"

    if exception_class == ASVAssemblyLoadError:
        return exception_class(
            translated_message, dotnet_exception, context.get("assembly_path")
        )
    elif exception_class == ASVFrequencyError:
        return exception_class(
            translated_message, context.get("frequency_hz"), context.get("valid_range")
        )
    elif exception_class == ASVConfigurationError:
        return exception_class(
            translated_message, context.get("config_key"), context.get("config_value")
        )
    elif exception_class == ASVRuntimeError:
        return exception_class(
            translated_message, dotnet_exception, context.get("runtime_state")
        )
    elif exception_class == ASVSignalProcessingError:
        return exception_class(
            translated_message,
            dotnet_exception,
            context.get("processing_stage"),
            context.get("data_size"),
        )
    else:
        return exception_class(
            translated_message,
            dotnet_exception,
            context.get("analyzer_type"),
            context.get("analyzer_id"),
        )


class ASVExceptionHandler:
    """Centralized exception handler for ASV operations."""

    def __init__(self):
        """Initialize exception handler."""
        self._error_counts = {}
        self._last_errors = {}

    def handle_exception(
        self,
        operation: str,
        exception: Exception,
        context: dict[str, Any] | None = None,
    ) -> ASVInteropError:
        """Handle and translate exception with context.

        Args:
            operation: Name of operation that failed
            exception: Original exception
            context: Additional context

        Returns:
            Translated ASV exception
        """
        # Count error occurrences
        error_key = f"{operation}:{type(exception).__name__}"
        self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1

        # Determine if this is a .NET exception that needs translation
        if hasattr(exception, "__module__") and "System" in str(exception.__module__):
            # This is likely a .NET exception
            translated = translate_dotnet_exception(exception, context)
        elif isinstance(exception, ASVInteropError):
            # Already an ASV exception
            translated = exception
        else:
            # Generic Python exception - wrap as ASV error
            translated = ASVInteropError(
                f"Operation '{operation}' failed: {exception}",
                exception,
                "ASV_PYTHON_ERROR",
                context,
            )

        # Store for analysis
        self._last_errors[operation] = {
            "timestamp": (
                logger._getCurrentTime()
                if hasattr(logger, "_getCurrentTime")
                else "unknown"
            ),
            "exception": translated,
            "count": self._error_counts[error_key],
        }

        return translated

    def get_error_statistics(self) -> dict[str, Any]:
        """Get error statistics for monitoring."""
        return {
            "error_counts": self._error_counts.copy(),
            "total_errors": sum(self._error_counts.values()),
            "unique_error_types": len(self._error_counts),
            "recent_errors": list(self._last_errors.keys())[-10:],  # Last 10
        }

    def clear_statistics(self) -> None:
        """Clear error statistics."""
        self._error_counts.clear()
        self._last_errors.clear()


# Global exception handler instance
exception_handler = ASVExceptionHandler()
