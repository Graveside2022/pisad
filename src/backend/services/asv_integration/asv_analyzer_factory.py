"""Service factory for multi-analyzer instantiation.

SUBTASK-6.1.1.3-b: Create service factory for multi-analyzer instantiation

This module provides a factory service for creating and managing multiple
ASV analyzer instances with concurrent processing capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set

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
    """Factory for creating and managing multiple ASV analyzer instances."""
    
    def __init__(self, interop_service: ASVInteropService):
        """Initialize analyzer factory.
        
        Args:
            interop_service: ASV interop service for .NET assembly access
        """
        self.interop_service = interop_service
        self._analyzers: Dict[str, ASVAnalyzerBase] = {}
        self._analyzer_configs: Dict[str, ASVAnalyzerConfig] = {}
        self._is_running = False
        
    @property
    def is_running(self) -> bool:
        """Check if factory is running."""
        return self._is_running
        
    @property
    def active_analyzers(self) -> List[str]:
        """Get list of active analyzer IDs."""
        return list(self._analyzers.keys())
        
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
    
    def create_analyzer_config(self, analyzer_type: str, frequency_hz: int, 
                              ref_power_dbm: float = -50.0, **kwargs) -> ASVAnalyzerConfig:
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
            calibration_enabled=kwargs.get('calibration_enabled', True),
            signal_overflow_threshold=kwargs.get('signal_overflow_threshold', 0.8),
            processing_timeout_ms=kwargs.get('processing_timeout_ms', 100),
        )
    
    async def create_analyzer(self, analyzer_id: str, config: ASVAnalyzerConfig) -> str:
        """Create and initialize a new analyzer instance.
        
        Args:
            analyzer_id: Unique identifier for the analyzer
            config: Analyzer configuration
            
        Returns:
            The analyzer ID if successful
            
        Raises:
            ASVAnalyzerError: If analyzer creation fails
        """
        if analyzer_id in self._analyzers:
            raise ASVAnalyzerError(f"Analyzer {analyzer_id} already exists")
            
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
            
            # Store analyzer and config
            self._analyzers[analyzer_id] = analyzer
            self._analyzer_configs[analyzer_id] = config
            
            logger.info(f"Created analyzer {analyzer_id} ({config.analyzer_type}) at {config.frequency_hz:,} Hz")
            return analyzer_id
            
        except Exception as e:
            logger.error(f"Failed to create analyzer {analyzer_id}: {e}")
            raise ASVAnalyzerError(f"Analyzer creation failed for {analyzer_id}: {e}", e)
    
    async def remove_analyzer(self, analyzer_id: str) -> None:
        """Remove and shutdown an analyzer instance.
        
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
            
            logger.info(f"Removed analyzer {analyzer_id}")
            
        except Exception as e:
            logger.error(f"Error removing analyzer {analyzer_id}: {e}")
            raise ASVAnalyzerError(f"Analyzer removal failed for {analyzer_id}: {e}", e)
    
    async def process_signal_concurrent(self, iq_data: bytes, 
                                       analyzer_ids: Optional[List[str]] = None) -> Dict[str, ASVSignalData]:
        """Process signal data concurrently across multiple analyzers.
        
        Args:
            iq_data: IQ signal data to process
            analyzer_ids: Specific analyzers to use (None = all active)
            
        Returns:
            Dictionary mapping analyzer_id -> signal results
        """
        if not self._is_running:
            raise ASVInteropError("Factory not running")
            
        target_analyzers = analyzer_ids or list(self._analyzers.keys())
        
        if not target_analyzers:
            logger.warning("No analyzers available for signal processing")
            return {}
        
        # Validate all requested analyzers exist
        missing_analyzers = set(target_analyzers) - set(self._analyzers.keys())
        if missing_analyzers:
            raise ASVAnalyzerError(f"Analyzers not found: {missing_analyzers}")
        
        try:
            # Process signals concurrently
            tasks = {
                analyzer_id: self._analyzers[analyzer_id].process_signal(iq_data)
                for analyzer_id in target_analyzers
            }
            
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            
            # Collect successful results and log errors
            signal_results = {}
            for analyzer_id, result in zip(target_analyzers, results):
                if isinstance(result, Exception):
                    logger.error(f"Analyzer {analyzer_id} processing failed: {result}")
                    # Continue with other analyzers
                else:
                    signal_results[analyzer_id] = result
            
            logger.debug(f"Processed signal with {len(signal_results)}/{len(target_analyzers)} analyzers")
            return signal_results
            
        except Exception as e:
            logger.error(f"Concurrent signal processing failed: {e}")
            raise ASVAnalyzerError(f"Signal processing failed: {e}", e)
    
    def get_analyzer_status(self) -> Dict[str, dict]:
        """Get status information for all analyzers.
        
        Returns:
            Dictionary with analyzer status information
        """
        status = {}
        
        for analyzer_id, analyzer in self._analyzers.items():
            config = self._analyzer_configs[analyzer_id]
            status[analyzer_id] = {
                "type": analyzer.analyzer_type,
                "frequency_hz": analyzer.frequency_hz,
                "initialized": analyzer.is_initialized,
                "overflow_indicator": analyzer.get_signal_overflow_indicator(),
                "config": {
                    "ref_power_dbm": config.ref_power_dbm,
                    "calibration_enabled": config.calibration_enabled,
                    "processing_timeout_ms": config.processing_timeout_ms,
                }
            }
        
        return status
    
    async def create_standard_analyzer_set(self) -> Dict[str, str]:
        """Create a standard set of analyzers for SAR operations.
        
        Returns:
            Dictionary mapping purpose -> analyzer_id
        """
        standard_configs = {
            "emergency_beacon": self.create_analyzer_config(
                analyzer_type="GP",
                frequency_hz=406_000_000,  # 406 MHz emergency beacons
                ref_power_dbm=-120.0,
                calibration_enabled=True,
                processing_timeout_ms=50,  # Fast processing for emergency signals
            ),
            "aviation_vor": self.create_analyzer_config(
                analyzer_type="VOR", 
                frequency_hz=112_500_000,  # VOR frequency range
                ref_power_dbm=-100.0,
                calibration_enabled=True,
                processing_timeout_ms=100,
            ),
            "aviation_ils": self.create_analyzer_config(
                analyzer_type="LLZ",
                frequency_hz=109_500_000,  # ILS localizer frequency
                ref_power_dbm=-90.0,
                calibration_enabled=True,
                processing_timeout_ms=75,
            ),
        }
        
        created_analyzers = {}
        
        for purpose, config in standard_configs.items():
            try:
                analyzer_id = await self.create_analyzer(purpose, config)
                created_analyzers[purpose] = analyzer_id
                logger.info(f"Created standard analyzer: {purpose} -> {analyzer_id}")
            except Exception as e:
                logger.warning(f"Failed to create standard analyzer {purpose}: {e}")
                # Continue with other analyzers
        
        logger.info(f"Created {len(created_analyzers)} standard analyzers for SAR operations")
        return created_analyzers


class ASVMultiAnalyzerCoordinator:
    """Coordinator for managing concurrent multi-analyzer operations."""
    
    def __init__(self, factory: ASVAnalyzerFactory):
        """Initialize multi-analyzer coordinator.
        
        Args:
            factory: Analyzer factory instance
        """
        self.factory = factory
        self._processing_stats = {
            "total_processed": 0,
            "successful_processing": 0,
            "failed_processing": 0,
            "average_latency_ms": 0.0,
        }
    
    async def process_signal_with_fusion(self, iq_data: bytes) -> dict:
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
            strongest_strength = float('-inf')
            
            signals_detected = []
            
            for analyzer_id, signal_data in results.items():
                signals_detected.append({
                    "analyzer": analyzer_id,
                    "type": signal_data.analyzer_type,
                    "frequency_hz": signal_data.frequency_hz,
                    "strength_dbm": signal_data.signal_strength_dbm,
                    "quality": signal_data.signal_quality,
                    "timestamp_ns": signal_data.timestamp_ns,
                    "raw_data": signal_data.raw_data,
                })
                
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
                    alpha * processing_time + 
                    (1 - alpha) * self._processing_stats["average_latency_ms"]
                )
            
            return {
                "status": "success",
                "processing_time_ms": processing_time,
                "analyzers_used": len(results),
                "strongest_signal": {
                    "analyzer": strongest_signal.analyzer_type if strongest_signal else None,
                    "strength_dbm": strongest_strength if strongest_signal else None,
                    "frequency_hz": strongest_signal.frequency_hz if strongest_signal else None,
                } if strongest_signal else None,
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
    
    def get_processing_statistics(self) -> dict:
        """Get processing performance statistics."""
        stats = self._processing_stats.copy()
        if stats["total_processed"] > 0:
            stats["success_rate"] = stats["successful_processing"] / stats["total_processed"]
            stats["failure_rate"] = stats["failed_processing"] / stats["total_processed"]
        else:
            stats["success_rate"] = 0.0
            stats["failure_rate"] = 0.0
        
        return stats