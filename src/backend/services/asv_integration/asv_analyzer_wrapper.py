"""Python wrapper classes for ASV analyzer interfaces.

SUBTASK-6.1.1.3-a: Design Python wrapper classes for ASV analyzer interfaces

This module provides Python wrapper classes for ASV .NET analyzer interfaces,
enabling seamless integration with PISAD's existing service architecture.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from src.backend.services.asv_integration.exceptions import (
    ASVAnalyzerError,
)

logger = logging.getLogger(__name__)


@dataclass
class ASVAnalyzerConfig:
    """Configuration for ASV analyzer instances."""

    frequency_hz: int
    ref_power_dbm: float
    analyzer_type: str
    calibration_enabled: bool = True
    signal_overflow_threshold: float = 0.8
    processing_timeout_ms: int = 100


@dataclass
class ASVSignalData:
    """Standardized signal data from ASV analyzers."""

    timestamp_ns: int
    frequency_hz: int
    signal_strength_dbm: float
    signal_quality: float  # 0.0-1.0
    analyzer_type: str
    overflow_indicator: float
    raw_data: dict[str, Any] | None = None


class ASVAnalyzerBase(ABC):
    """Base class for Python wrappers around ASV .NET analyzers."""

    def __init__(self, config: ASVAnalyzerConfig, dotnet_instance: Any = None):
        """Initialize ASV analyzer wrapper.

        Args:
            config: Analyzer configuration
            dotnet_instance: Actual .NET analyzer instance
        """
        self.config = config
        self._dotnet_instance = dotnet_instance
        self._is_initialized = False
        self._last_signal_data: ASVSignalData | None = None

    @property
    def is_initialized(self) -> bool:
        """Check if analyzer is initialized."""
        return self._is_initialized

    @property
    def analyzer_type(self) -> str:
        """Get analyzer type identifier."""
        return self.config.analyzer_type

    @property
    def frequency_hz(self) -> int:
        """Get current operating frequency."""
        return self.config.frequency_hz

    async def initialize(self) -> bool:
        """Initialize the analyzer with configuration."""
        try:
            if self._dotnet_instance:
                # This would call actual .NET Init method
                # await self._dotnet_instance.Init(
                #     self.config.frequency_hz,
                #     self.config.ref_power_dbm,
                #     calibration_provider,
                #     cancellation_token
                # )
                pass

            self._is_initialized = True
            logger.info(
                f"Initialized {self.analyzer_type} analyzer at {self.config.frequency_hz:,} Hz"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize {self.analyzer_type} analyzer: {e}")
            raise ASVAnalyzerError(
                f"Initialization failed for {self.analyzer_type}: {e}", e
            )

    async def shutdown(self) -> None:
        """Shutdown the analyzer and clean up resources."""
        try:
            if self._dotnet_instance:
                # This would call .NET Dispose method
                # self._dotnet_instance.Dispose()
                pass

            self._is_initialized = False
            logger.info(f"Shutdown {self.analyzer_type} analyzer")

        except Exception as e:
            logger.error(f"Error during {self.analyzer_type} analyzer shutdown: {e}")
            raise ASVAnalyzerError(f"Shutdown failed for {self.analyzer_type}: {e}", e)

    async def emergency_shutdown(self) -> None:
        """Emergency shutdown with enhanced speed for <500ms response time.

        SUBTASK-6.1.2.4 [17b-3]: Enhanced emergency shutdown mode for faster termination.

        This method provides faster shutdown compared to normal shutdown by
        bypassing graceful cleanup steps and focusing on immediate resource release.
        """
        try:
            # Skip graceful cleanup - immediate termination for emergency scenarios
            if self._dotnet_instance:
                # Immediate disposal without cleanup delays
                # self._dotnet_instance.Dispose()
                pass

            # Immediately mark as uninitialized
            self._is_initialized = False

            # Clear telemetry service to prevent broadcasting during emergency
            if hasattr(self, "_telemetry_service"):
                self._telemetry_service = None

            logger.warning(
                f"Emergency shutdown completed for {self.analyzer_type} analyzer"
            )

        except Exception as e:
            # Log error but don't raise - emergency shutdown must always succeed
            logger.error(f"Error during {self.analyzer_type} emergency shutdown: {e}")
            # Ensure state is cleared even on error
            self._is_initialized = False
            if hasattr(self, "_telemetry_service"):
                self._telemetry_service = None

    @abstractmethod
    async def process_signal(self, iq_data: bytes) -> ASVSignalData:
        """Process IQ data and return signal analysis results."""
        pass

    @abstractmethod
    def get_signal_overflow_indicator(self) -> float:
        """Get current signal overflow indicator value."""
        pass

    async def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status for the analyzer.

        SUBTASK-6.1.2.4 [17d-1]: Enhanced health status methods

        Returns:
            Dictionary containing comprehensive health status information
        """
        try:
            current_time = datetime.now(UTC)

            # Get basic operational status
            operational = self._is_initialized and self._dotnet_instance is not None

            # Get performance metrics
            performance_metrics = await self._get_performance_metrics()

            # Detect error conditions
            error_conditions = await self._detect_error_conditions()

            # Check for performance degradation
            performance_degraded = await self._detect_performance_degradation()

            health_status = {
                "operational": operational,
                "signal_processing_healthy": operational and not performance_degraded,
                "performance_metrics": performance_metrics,
                "error_conditions": error_conditions,
                "performance_degraded": performance_degraded,
                "last_health_check_timestamp": current_time.isoformat(),
                "analyzer_type": self.analyzer_type,
                "frequency_hz": self.frequency_hz,
                "initialization_time": getattr(self, "_initialization_time", None),
            }

            return health_status

        except Exception as e:
            logger.error(f"Health status check failed for {self.analyzer_type}: {e}")
            return {
                "operational": False,
                "signal_processing_healthy": False,
                "performance_metrics": {},
                "error_conditions": {"health_check_error": str(e)},
                "performance_degraded": True,
                "last_health_check_timestamp": datetime.now(UTC).isoformat(),
                "analyzer_type": self.analyzer_type,
                "error": str(e),
            }

    async def _get_performance_metrics(self) -> dict[str, float]:
        """Get current performance metrics.

        SUBTASK-6.1.2.4 [17d-3]: Performance monitoring metrics
        """
        try:
            # Get basic performance data
            signal_processing_latency = getattr(self, "_last_processing_time_ms", 0.0)

            # Calculate metrics based on recent signal processing
            last_signal_data = getattr(self, "_last_signal_data", None)

            metrics = {
                "signal_processing_latency_ms": signal_processing_latency,
                "throughput_samples_per_second": 1000.0
                / max(signal_processing_latency, 1.0),
                "error_rate_percentage": getattr(self, "_error_rate", 0.0),
                "memory_usage_bytes": getattr(self, "_memory_usage", 0),
                "cpu_usage_percentage": getattr(self, "_cpu_usage", 0.0),
                "signal_quality_average": (
                    last_signal_data.signal_quality if last_signal_data else 0.0
                ),
                "overflow_indicator": self.get_signal_overflow_indicator(),
            }

            return metrics

        except Exception as e:
            logger.error(f"Performance metrics collection failed: {e}")
            return {
                "signal_processing_latency_ms": 999.0,  # Indicate error
                "throughput_samples_per_second": 0.0,
                "error_rate_percentage": 100.0,
                "memory_usage_bytes": 0,
                "cpu_usage_percentage": 0.0,
                "signal_quality_average": 0.0,
                "overflow_indicator": 1.0,  # Max overflow indicates error
            }

    async def _detect_error_conditions(self) -> dict[str, Any]:
        """Detect and classify current error conditions.

        SUBTASK-6.1.2.4 [17d-2]: Error condition detection and classification
        """
        error_conditions: dict[str, list[str]] = {
            "initialization_errors": [],
            "signal_processing_errors": [],
            "hardware_communication_errors": [],
            "performance_degradation_errors": [],
            "timeout_errors": [],
        }

        try:
            # Check initialization errors
            if not self._is_initialized:
                error_conditions["initialization_errors"].append(
                    "Analyzer not properly initialized"
                )

            if self._dotnet_instance is None:
                error_conditions["hardware_communication_errors"].append(
                    ".NET analyzer instance not available"
                )

            # Check performance degradation
            performance_metrics = await self._get_performance_metrics()

            if (
                performance_metrics["signal_processing_latency_ms"]
                > self.config.processing_timeout_ms
            ):
                error_conditions["performance_degradation_errors"].append(
                    f"Processing latency {performance_metrics['signal_processing_latency_ms']:.1f}ms exceeds timeout {self.config.processing_timeout_ms}ms"
                )

            if performance_metrics["error_rate_percentage"] > 10.0:
                error_conditions["signal_processing_errors"].append(
                    f"High error rate: {performance_metrics['error_rate_percentage']:.1f}%"
                )

            if (
                performance_metrics["overflow_indicator"]
                > self.config.signal_overflow_threshold
            ):
                error_conditions["signal_processing_errors"].append(
                    f"Signal overflow detected: {performance_metrics['overflow_indicator']:.2f} > {self.config.signal_overflow_threshold}"
                )

            return error_conditions

        except Exception as e:
            logger.error(f"Error condition detection failed: {e}")
            error_conditions["health_check_errors"] = [f"Error detection failed: {e}"]
            return error_conditions

    async def _detect_performance_degradation(self) -> bool:
        """Detect if performance has degraded below acceptable thresholds.

        SUBTASK-6.1.2.4 [17d-3]: Performance degradation detection
        """
        try:
            performance_metrics = await self._get_performance_metrics()

            # Define performance degradation thresholds
            degradation_thresholds = getattr(
                self,
                "_performance_thresholds",
                {
                    "max_latency_ms": self.config.processing_timeout_ms
                    * 0.8,  # 80% of timeout
                    "min_throughput_samples_per_sec": 100.0,
                    "max_error_rate_percent": 5.0,
                    "max_overflow_indicator": self.config.signal_overflow_threshold
                    * 0.9,
                },
            )

            # Check each threshold
            if (
                performance_metrics["signal_processing_latency_ms"]
                > degradation_thresholds["max_latency_ms"]
            ):
                return True

            if (
                performance_metrics["throughput_samples_per_second"]
                < degradation_thresholds["min_throughput_samples_per_sec"]
            ):
                return True

            if (
                performance_metrics["error_rate_percentage"]
                > degradation_thresholds["max_error_rate_percent"]
            ):
                return True

            return (
                performance_metrics["overflow_indicator"]
                > degradation_thresholds["max_overflow_indicator"]
            )

        except Exception as e:
            logger.error(f"Performance degradation detection failed: {e}")
            return True  # Assume degraded if check fails

    async def detect_performance_degradation(self) -> bool:
        """Public method to detect performance degradation.

        SUBTASK-6.1.2.4 [17d-3]: Public degradation detection interface
        """
        return await self._detect_performance_degradation()

    async def set_performance_thresholds(self, thresholds: dict[str, float]) -> None:
        """Set custom performance thresholds for degradation detection.

        SUBTASK-6.1.2.4 [17d-3]: Configurable performance thresholds
        """
        self._performance_thresholds = thresholds.copy()
        logger.info(
            f"Updated performance thresholds for {self.analyzer_type}: {thresholds}"
        )

    async def configure_health_alerts(self, alert_config: dict[str, Any]) -> None:
        """Configure health alert thresholds and notification channels.

        SUBTASK-6.1.2.4 [17d-7]: Health alerts and threshold notifications
        """
        self._health_alert_config = alert_config.copy()
        logger.info(
            f"Configured health alerts for {self.analyzer_type}: {alert_config}"
        )

    async def check_health_thresholds(self) -> bool:
        """Check if health thresholds are exceeded and trigger alerts.

        SUBTASK-6.1.2.4 [17d-7]: Health threshold checking
        """
        if not hasattr(self, "_health_alert_config"):
            return False

        try:
            performance_metrics = await self._get_performance_metrics()
            alert_config = self._health_alert_config

            alert_triggered = False

            # Check latency threshold
            if "latency_threshold_ms" in alert_config:
                if (
                    performance_metrics["signal_processing_latency_ms"]
                    > alert_config["latency_threshold_ms"]
                ):
                    alert_triggered = True
                    logger.warning(
                        f"Health alert: {self.analyzer_type} latency {performance_metrics['signal_processing_latency_ms']:.1f}ms exceeds threshold {alert_config['latency_threshold_ms']}ms"
                    )

            # Check error rate threshold
            if "error_rate_threshold_percent" in alert_config:
                if (
                    performance_metrics["error_rate_percentage"]
                    > alert_config["error_rate_threshold_percent"]
                ):
                    alert_triggered = True
                    logger.warning(
                        f"Health alert: {self.analyzer_type} error rate {performance_metrics['error_rate_percentage']:.1f}% exceeds threshold {alert_config['error_rate_threshold_percent']}%"
                    )

            return alert_triggered

        except Exception as e:
            logger.error(f"Health threshold check failed: {e}")
            return True  # Assume alert condition if check fails

    async def register_telemetry_service(self, telemetry_service: Any) -> None:
        """Register telemetry service for health broadcasting.

        SUBTASK-6.1.2.4 [17d-5]: Telemetry integration for health broadcasting

        Args:
            telemetry_service: Telemetry service instance for broadcasting health data
        """
        try:
            self._telemetry_service = telemetry_service
            logger.info(
                f"Registered telemetry service for {self.analyzer_type} analyzer health broadcasting"
            )

        except Exception as e:
            logger.error(f"Failed to register telemetry service: {e}")

    async def broadcast_health_status(self) -> None:
        """Broadcast current health status via telemetry service.

        SUBTASK-6.1.2.4 [17d-5]: Real-time health broadcasting
        """
        try:
            if (
                not hasattr(self, "_telemetry_service")
                or self._telemetry_service is None
            ):
                logger.debug(
                    f"No telemetry service registered for {self.analyzer_type} analyzer"
                )
                return

            # Get current health status
            health_status = await self.get_health_status()

            # Format for telemetry broadcasting
            telemetry_data = {
                "message_type": "asv_analyzer_health",
                "analyzer_id": f"{self.analyzer_type.lower()}_analyzer",
                "analyzer_type": self.analyzer_type,
                "timestamp": datetime.now(UTC).isoformat(),
                "health_status": health_status,
            }

            # Broadcast via telemetry service
            if hasattr(self._telemetry_service, "broadcast_data"):
                await self._telemetry_service.broadcast_data(telemetry_data)
                logger.debug(
                    f"Broadcast health status for {self.analyzer_type} analyzer via telemetry"
                )
            else:
                logger.warning(
                    "Telemetry service does not support broadcast_data method"
                )

        except Exception as e:
            logger.error(f"Failed to broadcast health status: {e}")

    async def get_health_dashboard_format(self) -> dict[str, Any]:
        """Get health status in format compatible with existing PISAD health monitoring dashboard.

        SUBTASK-6.1.2.4 [17d-8]: Integration with existing health monitoring

        Returns:
            Health data formatted for existing dashboard integration
        """
        try:
            health_status = await self.get_health_status()
            performance_metrics = health_status.get("performance_metrics", {})

            # Map to existing dashboard format
            dashboard_format = {
                "service_name": f"asv_{self.analyzer_type.lower()}_analyzer",
                "status": self._map_health_to_dashboard_status(health_status),
                "metrics": {
                    "response_time_ms": performance_metrics.get(
                        "signal_processing_latency_ms", 0
                    ),
                    "success_rate_percent": 100.0
                    - performance_metrics.get("error_rate_percentage", 0.0),
                    "resource_usage_percent": performance_metrics.get(
                        "cpu_usage_percentage", 0.0
                    ),
                    "throughput_samples_per_sec": performance_metrics.get(
                        "throughput_samples_per_second", 0.0
                    ),
                    "memory_usage_mb": performance_metrics.get("memory_usage_bytes", 0)
                    / (1024 * 1024),
                },
                "alerts": self._extract_dashboard_alerts(health_status),
                "last_check_timestamp": health_status.get(
                    "last_health_check_timestamp"
                ),
                "analyzer_specific": {
                    "frequency_hz": self.frequency_hz,
                    "analyzer_type": self.analyzer_type,
                    "initialization_status": self._is_initialized,
                },
            }

            return dashboard_format

        except Exception as e:
            logger.error(f"Failed to format health status for dashboard: {e}")
            return {
                "service_name": f"asv_{self.analyzer_type.lower()}_analyzer",
                "status": "critical",
                "metrics": {},
                "alerts": [f"Health formatting error: {e}"],
                "last_check_timestamp": datetime.now(UTC).isoformat(),
                "error": str(e),
            }

    def _map_health_to_dashboard_status(self, health_status: dict[str, Any]) -> str:
        """Map comprehensive health status to simple dashboard status.

        Args:
            health_status: Comprehensive health status dictionary

        Returns:
            Dashboard status: "healthy", "degraded", "critical", or "offline"
        """
        if not health_status.get("operational", False):
            return "offline"

        if health_status.get("performance_degraded", False):
            error_conditions = health_status.get("error_conditions", {})

            # Check for critical errors
            critical_errors = error_conditions.get(
                "initialization_errors", []
            ) + error_conditions.get("hardware_communication_errors", [])

            if critical_errors:
                return "critical"
            else:
                return "degraded"

        if health_status.get("signal_processing_healthy", False):
            return "healthy"
        else:
            return "degraded"

    def _extract_dashboard_alerts(self, health_status: dict[str, Any]) -> list[str]:
        """Extract alert messages for dashboard display.

        Args:
            health_status: Comprehensive health status dictionary

        Returns:
            List of alert message strings
        """
        alerts = []

        try:
            error_conditions = health_status.get("error_conditions", {})

            # Collect alerts from all error categories
            for category, errors in error_conditions.items():
                if isinstance(errors, list) and errors:
                    for error in errors:
                        alerts.append(f"{category.replace('_', ' ').title()}: {error}")

            # Add performance degradation alert if applicable
            if health_status.get("performance_degraded", False):
                performance_metrics = health_status.get("performance_metrics", {})
                latency = performance_metrics.get("signal_processing_latency_ms", 0)
                if latency > 100:
                    alerts.append(f"High processing latency: {latency:.1f}ms")

            return alerts

        except Exception as e:
            logger.error(f"Failed to extract dashboard alerts: {e}")
            return [f"Alert extraction error: {e}"]


class ASVGpAnalyzer(ASVAnalyzerBase):
    """Python wrapper for ASV GP (GPS/GNSS) analyzer.

    Handles emergency beacon detection at 406 MHz and GNSS signals.
    """

    def __init__(self, config: ASVAnalyzerConfig, dotnet_instance: Any = None):
        """Initialize GP analyzer wrapper."""
        super().__init__(config, dotnet_instance)
        self._beacon_detection_threshold = -120.0  # dBm

    async def process_signal(self, iq_data: bytes) -> ASVSignalData:
        """Process IQ data for GPS/emergency beacon detection."""
        if not self._is_initialized:
            raise ASVAnalyzerError("GP analyzer not initialized")

        try:
            start_time = time.perf_counter_ns()

            # Simulate GP analyzer processing
            # In actual implementation, this would call .NET analyzer methods
            signal_strength = -110.0 + (len(iq_data) % 20)  # Simulated signal strength
            signal_quality = max(
                0.0, min(1.0, (signal_strength + 120) / 20)
            )  # Quality based on strength

            # Check for emergency beacon characteristics
            is_emergency_beacon = signal_strength > self._beacon_detection_threshold

            signal_data = ASVSignalData(
                timestamp_ns=start_time,
                frequency_hz=self.config.frequency_hz,
                signal_strength_dbm=signal_strength,
                signal_quality=signal_quality,
                analyzer_type="GP",
                overflow_indicator=self.get_signal_overflow_indicator(),
                raw_data={
                    "beacon_detected": is_emergency_beacon,
                    "processing_time_ns": time.perf_counter_ns() - start_time,
                },
            )

            self._last_signal_data = signal_data
            return signal_data

        except Exception as e:
            raise ASVAnalyzerError(f"GP analyzer processing failed: {e}", e)

    def get_signal_overflow_indicator(self) -> float:
        """Get GP analyzer signal overflow indicator."""
        # This would query the actual .NET analyzer instance
        # return float(self._dotnet_instance.SignalOverflowIndicator.Value)
        return 0.1  # Simulated low overflow


class ASVVorAnalyzer(ASVAnalyzerBase):
    """Python wrapper for ASV VOR (VHF Omnidirectional Range) analyzer.

    Handles VOR navigation signal analysis for aviation applications.
    """

    def __init__(self, config: ASVAnalyzerConfig, dotnet_instance: Any = None):
        """Initialize VOR analyzer wrapper."""
        super().__init__(config, dotnet_instance)
        self._vor_detection_threshold = -100.0  # dBm

    async def process_signal(self, iq_data: bytes) -> ASVSignalData:
        """Process IQ data for VOR navigation signal analysis."""
        if not self._is_initialized:
            raise ASVAnalyzerError("VOR analyzer not initialized")

        try:
            start_time = time.perf_counter_ns()

            # Simulate VOR analyzer processing
            # In actual implementation, this would call .NET VOR analyzer methods
            signal_strength = -95.0 + (len(iq_data) % 15)  # Simulated signal strength
            signal_quality = max(
                0.0, min(1.0, (signal_strength + 105) / 15)
            )  # Quality based on strength

            # Calculate bearing information (VOR-specific)
            radial_degrees = (time.time_ns() // 1000000) % 360  # Simulated radial

            signal_data = ASVSignalData(
                timestamp_ns=start_time,
                frequency_hz=self.config.frequency_hz,
                signal_strength_dbm=signal_strength,
                signal_quality=signal_quality,
                analyzer_type="VOR",
                overflow_indicator=self.get_signal_overflow_indicator(),
                raw_data={
                    "vor_detected": signal_strength > self._vor_detection_threshold,
                    "radial_degrees": radial_degrees,
                    "station_identifier": "TEST",  # Would be decoded from signal
                    "processing_time_ns": time.perf_counter_ns() - start_time,
                },
            )

            self._last_signal_data = signal_data
            return signal_data

        except Exception as e:
            raise ASVAnalyzerError(f"VOR analyzer processing failed: {e}", e)

    def get_signal_overflow_indicator(self) -> float:
        """Get VOR analyzer signal overflow indicator."""
        return 0.05  # Simulated very low overflow


class ASVLlzAnalyzer(ASVAnalyzerBase):
    """Python wrapper for ASV LLZ (Localizer) analyzer.

    Handles ILS localizer signal analysis for aviation landing systems.
    """

    def __init__(self, config: ASVAnalyzerConfig, dotnet_instance: Any = None):
        """Initialize LLZ analyzer wrapper."""
        super().__init__(config, dotnet_instance)
        self._llz_detection_threshold = -90.0  # dBm

    async def process_signal(self, iq_data: bytes) -> ASVSignalData:
        """Process IQ data for localizer signal analysis."""
        if not self._is_initialized:
            raise ASVAnalyzerError("LLZ analyzer not initialized")

        try:
            start_time = time.perf_counter_ns()

            # Simulate LLZ analyzer processing
            # In actual implementation, this would call .NET LLZ analyzer methods
            signal_strength = -85.0 + (len(iq_data) % 10)  # Simulated signal strength
            signal_quality = max(
                0.0, min(1.0, (signal_strength + 95) / 10)
            )  # Quality based on strength

            # Calculate course deviation indicator (LLZ-specific)
            course_deviation = (
                (time.time_ns() // 1000000) % 200 - 100
            ) / 100.0  # -1.0 to 1.0

            signal_data = ASVSignalData(
                timestamp_ns=start_time,
                frequency_hz=self.config.frequency_hz,
                signal_strength_dbm=signal_strength,
                signal_quality=signal_quality,
                analyzer_type="LLZ",
                overflow_indicator=self.get_signal_overflow_indicator(),
                raw_data={
                    "llz_detected": signal_strength > self._llz_detection_threshold,
                    "course_deviation": course_deviation,
                    "runway_heading": 180,  # Would be decoded from signal
                    "processing_time_ns": time.perf_counter_ns() - start_time,
                },
            )

            self._last_signal_data = signal_data
            return signal_data

        except Exception as e:
            raise ASVAnalyzerError(f"LLZ analyzer processing failed: {e}", e)

    def get_signal_overflow_indicator(self) -> float:
        """Get LLZ analyzer signal overflow indicator."""
        return 0.02  # Simulated minimal overflow


# Analyzer type registry for factory creation
ANALYZER_TYPES = {
    "GP": ASVGpAnalyzer,
    "VOR": ASVVorAnalyzer,
    "LLZ": ASVLlzAnalyzer,
}


def create_analyzer(
    analyzer_type: str, config: ASVAnalyzerConfig, dotnet_instance: Any = None
) -> ASVAnalyzerBase:
    """Factory function to create analyzer instances.

    Args:
        analyzer_type: Type of analyzer to create ("GP", "VOR", "LLZ")
        config: Analyzer configuration
        dotnet_instance: Optional .NET analyzer instance

    Returns:
        Initialized analyzer wrapper instance

    Raises:
        ASVAnalyzerError: If analyzer type is not supported
    """
    if analyzer_type not in ANALYZER_TYPES:
        raise ASVAnalyzerError(f"Unsupported analyzer type: {analyzer_type}")

    analyzer_class = ANALYZER_TYPES[analyzer_type]
    return analyzer_class(config, dotnet_instance)
