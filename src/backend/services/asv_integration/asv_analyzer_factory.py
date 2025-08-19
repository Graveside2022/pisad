"""Service factory for multi-analyzer instantiation.

SUBTASK-6.1.1.3-b: Create service factory for multi-analyzer instantiation

This module provides a factory service for creating and managing multiple
ASV analyzer instances with concurrent processing capabilities.
"""

import logging
from typing import Any

from src.backend.services.asv_integration.asv_analyzer_wrapper import (
    ASVAnalyzerBase,
    ASVAnalyzerConfig,
    ASVSignalData,
    create_analyzer,
)
from src.backend.services.asv_integration.asv_interop_service import ASVInteropService
from src.backend.services.asv_integration.exceptions import (
    ASVAnalyzerError,
    ASVInteropError,
)

logger = logging.getLogger(__name__)


class ASVAnalyzerFactory:
    """Factory for creating and managing single active ASV analyzer instances.

    Refactored for SUBTASK-6.1.2.1 [14b] to support enhanced single-frequency processing
    with rapid analyzer switching when frequency selection changes.
    """

    def __init__(self, interop_service: ASVInteropService):
        """Initialize analyzer factory for single-frequency processing.

        Args:
            interop_service: ASV interop service for .NET assembly access
        """
        self.interop_service = interop_service
        self._analyzers: dict[str, ASVAnalyzerBase] = {}
        self._analyzer_configs: dict[str, ASVAnalyzerConfig] = {}
        self._is_running = False

        # Single-frequency enhancement attributes
        self._active_analyzer_id: str | None = None
        self._current_frequency_hz: int | None = None
        self._max_active_analyzers: int = 1  # Enforce single analyzer limit

    @property
    def is_running(self) -> bool:
        """Check if factory is running."""
        return self._is_running

    @property
    def active_analyzers(self) -> list[str]:
        """Get list of active analyzer IDs."""
        return list(self._analyzers.keys())

    @property
    def active_analyzer_id(self) -> str | None:
        """Get the currently active analyzer ID (single-frequency mode)."""
        return self._active_analyzer_id

    @property
    def current_frequency_hz(self) -> int | None:
        """Get the current operating frequency."""
        return self._current_frequency_hz

    @property
    def max_active_analyzers(self) -> int:
        """Get maximum allowed active analyzers (always 1 for single-frequency)."""
        return self._max_active_analyzers

    async def start(self) -> None:
        """Start the analyzer factory."""
        if self._is_running:
            return

        try:
            # Ensure interop service is running
            if not self.interop_service.is_running:
                await self.interop_service.start()

            self._is_running = True
            logger.info("ASV analyzer factory started")

        except Exception as e:
            logger.error(f"Failed to start analyzer factory: {e}")
            raise ASVInteropError(f"Factory startup failed: {e}", e)

    async def stop(self) -> None:
        """Stop the analyzer factory and all analyzers."""
        if not self._is_running:
            return

        try:
            # Shutdown all analyzers
            for analyzer_id in list(self._analyzers.keys()):
                await self.remove_analyzer(analyzer_id)

            self._is_running = False
            logger.info("ASV analyzer factory stopped")

        except Exception as e:
            logger.error(f"Error stopping analyzer factory: {e}")
            raise ASVInteropError(f"Factory shutdown failed: {e}", e)

    def create_analyzer_config(
        self,
        analyzer_type: str,
        frequency_hz: int,
        ref_power_dbm: float = -50.0,
        **kwargs: Any,
    ) -> ASVAnalyzerConfig:
        """Create analyzer configuration.

        Args:
            analyzer_type: Type of analyzer ("GP", "VOR", "LLZ")
            frequency_hz: Operating frequency in Hz
            ref_power_dbm: Reference power in dBm
            **kwargs: Additional configuration options

        Returns:
            Configured ASVAnalyzerConfig instance
        """
        return ASVAnalyzerConfig(
            analyzer_type=analyzer_type,
            frequency_hz=frequency_hz,
            ref_power_dbm=ref_power_dbm,
            calibration_enabled=kwargs.get("calibration_enabled", True),
            signal_overflow_threshold=kwargs.get("signal_overflow_threshold", 0.8),
            processing_timeout_ms=kwargs.get("processing_timeout_ms", 100),
        )

    async def create_analyzer(self, analyzer_id: str, config: ASVAnalyzerConfig) -> str:
        """Create and initialize a single analyzer instance (single-frequency mode).

        Args:
            analyzer_id: Unique identifier for the analyzer
            config: Analyzer configuration

        Returns:
            The analyzer ID if successful

        Raises:
            ASVAnalyzerError: If analyzer creation fails or multiple analyzers attempted
        """
        if analyzer_id in self._analyzers:
            raise ASVAnalyzerError(f"Analyzer {analyzer_id} already exists")

        # Enforce single analyzer limit for single-frequency processing
        if len(self._analyzers) >= self._max_active_analyzers and self._active_analyzer_id:
            logger.info(
                f"Single-frequency mode: Removing existing analyzer {self._active_analyzer_id} "
                f"to create new analyzer {analyzer_id}"
            )
            await self.remove_analyzer(self._active_analyzer_id)

        try:
            # Get .NET analyzer instance if available
            dotnet_instance = None
            if self.interop_service.is_running:
                analyzer_types = self.interop_service.get_analyzer_types()
                dotnet_type = f"Asv.Drones.Sdr.Core.IAnalyzer{config.analyzer_type}"

                if dotnet_type in analyzer_types:
                    dotnet_instance = self.interop_service.create_analyzer_instance(dotnet_type)

            # Create Python wrapper
            analyzer = create_analyzer(config.analyzer_type, config, dotnet_instance)

            # Initialize analyzer
            await analyzer.initialize()

            # Store analyzer and config (single analyzer only)
            self._analyzers[analyzer_id] = analyzer
            self._analyzer_configs[analyzer_id] = config

            # Update single-frequency state tracking
            self._active_analyzer_id = analyzer_id
            self._current_frequency_hz = config.frequency_hz

            logger.info(
                f"Created single-frequency analyzer {analyzer_id} ({config.analyzer_type}) "
                f"at {config.frequency_hz:,} Hz"
            )
            return analyzer_id

        except Exception as e:
            logger.error(f"Failed to create analyzer {analyzer_id}: {e}")
            raise ASVAnalyzerError(f"Analyzer creation failed for {analyzer_id}: {e}", e)

    async def remove_analyzer(self, analyzer_id: str) -> None:
        """Remove and shutdown an analyzer instance (single-frequency mode).

        Args:
            analyzer_id: Analyzer to remove

        Raises:
            ASVAnalyzerError: If analyzer doesn't exist
        """
        if analyzer_id not in self._analyzers:
            raise ASVAnalyzerError(f"Analyzer {analyzer_id} not found")

        try:
            analyzer = self._analyzers[analyzer_id]
            await analyzer.shutdown()

            del self._analyzers[analyzer_id]
            del self._analyzer_configs[analyzer_id]

            # Clear single-frequency state tracking
            if self._active_analyzer_id == analyzer_id:
                self._active_analyzer_id = None
                self._current_frequency_hz = None

            logger.info(f"Removed single-frequency analyzer {analyzer_id}")

        except Exception as e:
            logger.error(f"Error removing analyzer {analyzer_id}: {e}")
            raise ASVAnalyzerError(f"Analyzer removal failed for {analyzer_id}: {e}", e)

    async def switch_to_frequency(
        self, analyzer_id: str, config: ASVAnalyzerConfig, force_recreate: bool = False
    ) -> str:
        """Switch to analyzer for specific frequency (single-frequency enhancement).

        This method enables rapid frequency switching by creating a new analyzer
        for the target frequency and removing any existing analyzer.

        Args:
            analyzer_id: Unique identifier for the new analyzer
            config: Analyzer configuration including target frequency
            force_recreate: Force recreation even if frequency hasn't changed

        Returns:
            The analyzer ID of the active analyzer

        Raises:
            ASVAnalyzerError: If frequency switching fails
        """
        try:
            # Check if frequency change is needed
            if (
                not force_recreate
                and self._current_frequency_hz == config.frequency_hz
                and self._active_analyzer_id == analyzer_id
                and analyzer_id in self._analyzers
            ):
                logger.debug(f"Frequency switch not needed - already at {config.frequency_hz:,} Hz")
                return self._active_analyzer_id

            # Create new analyzer for target frequency
            # This will automatically remove existing analyzer due to single-frequency limit
            new_analyzer_id = await self.create_analyzer(analyzer_id, config)

            logger.info(
                f"Switched to frequency {config.frequency_hz:,} Hz with analyzer {new_analyzer_id}"
            )
            return new_analyzer_id

        except Exception as e:
            logger.error(f"Failed to switch to frequency {config.frequency_hz:,} Hz: {e}")
            raise ASVAnalyzerError(
                f"Frequency switching failed for {config.frequency_hz:,} Hz: {e}", e
            )

    def get_current_analyzer(self) -> ASVAnalyzerBase | None:
        """Get the currently active analyzer instance.

        Returns:
            Current analyzer instance or None if no analyzer is active
        """
        if self._active_analyzer_id and self._active_analyzer_id in self._analyzers:
            return self._analyzers[self._active_analyzer_id]
        return None

    async def process_signal_concurrent(
        self, iq_data: bytes, analyzer_ids: list[str] | None = None
    ) -> dict[str, ASVSignalData]:
        """Process signal data with single active analyzer (single-frequency mode).

        Refactored for SUBTASK-6.1.2.1 [14b] to process through single analyzer only,
        maintaining compatibility with concurrent interface for existing callers.

        Args:
            iq_data: IQ signal data to process
            analyzer_ids: Specific analyzers to use (None = active analyzer)

        Returns:
            Dictionary mapping analyzer_id -> signal results (single entry max)
        """
        if not self._is_running:
            raise ASVInteropError("Factory not running")

        # Single-frequency mode: use active analyzer or specific analyzer
        target_analyzers = []

        if analyzer_ids:
            # Use specific analyzer if requested and exists
            for analyzer_id in analyzer_ids:
                if analyzer_id in self._analyzers:
                    target_analyzers.append(analyzer_id)
                    break  # Only process first available analyzer
        else:
            # Use active analyzer if available
            if self._active_analyzer_id and self._active_analyzer_id in self._analyzers:
                target_analyzers.append(self._active_analyzer_id)

        if not target_analyzers:
            logger.warning("No active analyzer available for signal processing")
            return {}

        try:
            # Process signal with single analyzer (not concurrent)
            analyzer_id = target_analyzers[0]  # Only one analyzer in single-frequency mode
            analyzer = self._analyzers[analyzer_id]

            result = await analyzer.process_signal(iq_data)

            signal_results = {analyzer_id: result}

            logger.debug(
                f"Processed signal with single-frequency analyzer {analyzer_id} "
                f"at {self._current_frequency_hz:,} Hz"
            )
            return signal_results

        except Exception as e:
            logger.error(f"Single-frequency signal processing failed: {e}")
            raise ASVAnalyzerError(f"Signal processing failed: {e}", e)

    def get_analyzer_status(self) -> dict[str, dict[str, Any]]:
        """Get status information for single active analyzer.

        Returns:
            Dictionary with analyzer status information (single entry for active analyzer)
        """
        status = {}

        for analyzer_id, analyzer in self._analyzers.items():
            config = self._analyzer_configs[analyzer_id]
            is_active = analyzer_id == self._active_analyzer_id

            status[analyzer_id] = {
                "type": analyzer.analyzer_type,
                "frequency_hz": analyzer.frequency_hz,
                "initialized": analyzer.is_initialized,
                "overflow_indicator": analyzer.get_signal_overflow_indicator(),
                "is_active": is_active,  # Single-frequency enhancement: track active analyzer
                "single_frequency_mode": True,  # Indicate single-frequency operation
                "config": {
                    "ref_power_dbm": config.ref_power_dbm,
                    "calibration_enabled": config.calibration_enabled,
                    "processing_timeout_ms": config.processing_timeout_ms,
                },
            }

        return status

    async def create_standard_analyzer_set(self) -> dict[str, str]:
        """Create default analyzer for single-frequency SAR operations.

        In single-frequency mode, creates only the emergency beacon analyzer as default.
        Other analyzers can be switched to via switch_to_frequency() method.

        Returns:
            Dictionary mapping purpose -> analyzer_id (single entry)
        """
        # Single-frequency mode: Start with highest priority analyzer (emergency beacon)
        default_config = self.create_analyzer_config(
            analyzer_type="GP",
            frequency_hz=406_000_000,  # 406 MHz emergency beacons - highest priority
            ref_power_dbm=-120.0,
            calibration_enabled=True,
            processing_timeout_ms=50,  # Fast processing for emergency signals
        )

        created_analyzers = {}

        try:
            analyzer_id = await self.create_analyzer("emergency_beacon", default_config)
            created_analyzers["emergency_beacon"] = analyzer_id
            logger.info(
                f"Created default single-frequency analyzer: emergency_beacon -> {analyzer_id}"
            )
        except Exception as e:
            logger.error(f"Failed to create default single-frequency analyzer: {e}")
            # Don't continue with other analyzers in single-frequency mode

        logger.info(
            f"Single-frequency mode: Created {len(created_analyzers)} default analyzer "
            f"at {default_config.frequency_hz:,} Hz"
        )

        # Store available frequency profiles for later switching
        self._available_frequency_profiles = {
            "emergency_beacon": {
                "analyzer_type": "GP",
                "frequency_hz": 406_000_000,
                "ref_power_dbm": -120.0,
                "priority": 1,
            },
            "aviation_emergency": {
                "analyzer_type": "GP",
                "frequency_hz": 121_500_000,
                "ref_power_dbm": -110.0,
                "priority": 1,
            },
            "aviation_vor": {
                "analyzer_type": "VOR",
                "frequency_hz": 112_500_000,
                "ref_power_dbm": -100.0,
                "priority": 2,
            },
            "aviation_ils": {
                "analyzer_type": "LLZ",
                "frequency_hz": 109_500_000,
                "ref_power_dbm": -90.0,
                "priority": 2,
            },
        }

        return created_analyzers

    def get_available_frequency_profiles(self) -> dict[str, dict[str, Any]]:
        """Get available frequency profiles for operator selection.

        Returns:
            Dictionary of available frequency profiles for single-frequency switching
        """
        if not hasattr(self, "_available_frequency_profiles"):
            return {}
        return self._available_frequency_profiles.copy()

    async def shutdown(self) -> None:
        """Shutdown the analyzer factory and all managed analyzers.

        This is an alias for the stop() method to maintain compatibility
        with the ASVHackRFCoordinator interface.
        """
        await self.stop()


class ASVMultiAnalyzerCoordinator:
    """Coordinator for managing concurrent multi-analyzer operations."""

    def __init__(self, analyzer_factory: ASVAnalyzerFactory, config_manager: Any | None = None):
        """Initialize multi-analyzer coordinator.

        Args:
            analyzer_factory: Analyzer factory instance
            config_manager: Configuration manager (optional)
        """
        self.factory = analyzer_factory
        self.config_manager = config_manager
        self._processing_stats = {
            "total_processed": 0,
            "successful_processing": 0,
            "failed_processing": 0,
            "average_latency_ms": 0.0,
        }
        self._is_initialized = False

    async def initialize(self) -> None:
        """Initialize the multi-analyzer coordinator."""
        self._is_initialized = True
        logger.info("ASVMultiAnalyzerCoordinator initialized")

    async def shutdown(self) -> None:
        """Shutdown the multi-analyzer coordinator."""
        self._is_initialized = False
        logger.info("ASVMultiAnalyzerCoordinator shutdown")

    async def process_signal_with_fusion(self, iq_data: bytes) -> dict[str, Any]:
        """Process signal with multiple analyzers and fuse results.

        Args:
            iq_data: IQ signal data

        Returns:
            Fused signal analysis results
        """
        import time

        start_time = time.perf_counter()

        try:
            # Process with all active analyzers
            results = await self.factory.process_signal_concurrent(iq_data)

            # Simple signal fusion algorithm
            if not results:
                return {"status": "no_analyzers", "signals": []}

            # Find strongest signal across all analyzers
            strongest_signal = None
            strongest_strength = float("-inf")

            signals_detected = []

            for analyzer_id, signal_data in results.items():
                signals_detected.append(
                    {
                        "analyzer": analyzer_id,
                        "type": signal_data.analyzer_type,
                        "frequency_hz": signal_data.frequency_hz,
                        "strength_dbm": signal_data.signal_strength_dbm,
                        "quality": signal_data.signal_quality,
                        "timestamp_ns": signal_data.timestamp_ns,
                        "raw_data": signal_data.raw_data,
                    }
                )

                if signal_data.signal_strength_dbm > strongest_strength:
                    strongest_strength = signal_data.signal_strength_dbm
                    strongest_signal = signal_data

            processing_time = (time.perf_counter() - start_time) * 1000

            # Update statistics
            self._processing_stats["total_processed"] += 1
            self._processing_stats["successful_processing"] += 1

            # Update average latency with exponential moving average
            alpha = 0.1  # Smoothing factor
            if self._processing_stats["average_latency_ms"] == 0:
                self._processing_stats["average_latency_ms"] = processing_time
            else:
                self._processing_stats["average_latency_ms"] = (
                    alpha * processing_time
                    + (1 - alpha) * self._processing_stats["average_latency_ms"]
                )

            return {
                "status": "success",
                "processing_time_ms": processing_time,
                "analyzers_used": len(results),
                "strongest_signal": (
                    {
                        "analyzer": (strongest_signal.analyzer_type if strongest_signal else None),
                        "strength_dbm": (strongest_strength if strongest_signal else None),
                        "frequency_hz": (
                            strongest_signal.frequency_hz if strongest_signal else None
                        ),
                    }
                    if strongest_signal
                    else None
                ),
                "signals": signals_detected,
                "statistics": self._processing_stats.copy(),
            }

        except Exception as e:
            processing_time = (time.perf_counter() - start_time) * 1000
            self._processing_stats["total_processed"] += 1
            self._processing_stats["failed_processing"] += 1

            logger.error(f"Multi-analyzer processing failed after {processing_time:.2f}ms: {e}")

            return {
                "status": "error",
                "error": str(e),
                "processing_time_ms": processing_time,
                "statistics": self._processing_stats.copy(),
            }

    def get_processing_statistics(self) -> dict[str, Any]:
        """Get processing performance statistics."""
        stats = self._processing_stats.copy()
        if stats["total_processed"] > 0:
            stats["success_rate"] = stats["successful_processing"] / stats["total_processed"]
            stats["failure_rate"] = stats["failed_processing"] / stats["total_processed"]
        else:
            stats["success_rate"] = 0.0
            stats["failure_rate"] = 0.0

        return stats
