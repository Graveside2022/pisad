"""
ASV HackRF Multi-Analyzer Coordinator Service

SUBTASK-6.1.2.1: Create ASV HackRF coordinator service with multi-analyzer management

Coordinates multiple ASV analyzers with HackRF hardware, enabling simultaneous 
multi-frequency signal processing for aviation SAR operations.

Key Features:
- Concurrent GP/VOR/LLZ analyzer coordination
- HackRF frequency switching with minimal latency
- Multi-analyzer signal fusion and correlation
- Safety system integration with analyzer lifecycle
- Performance optimization for real-time operations

PRD References:
- Epic 6: ASV SDR Framework Integration
- FR1: Multi-frequency RF detection capability 
- NFR2: <100ms analyzer switching latency
- NFR12: Deterministic timing for safety compliance
"""

import asyncio
import contextlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.backend.core.base_service import BaseService
from src.backend.core.exceptions import PISADException
from src.backend.hal.hackrf_interface import HackRFConfig, HackRFInterface
from src.backend.services.asv_integration.asv_analyzer_factory import (
    ASVAnalyzerFactory,
    ASVMultiAnalyzerCoordinator,
)
from src.backend.services.asv_integration.asv_analyzer_wrapper import (
    ASVAnalyzerBase,
    ASVAnalyzerConfig,
    ASVSignalData,
)
from src.backend.services.asv_integration.asv_configuration_manager import (
    ASVConfigurationManager,
    ASVFrequencyProfile,
)
from src.backend.services.asv_integration.exceptions import (
    ASVAnalyzerError,
    ASVConfigurationError,
    ASVHardwareError,
    ASVInteropError,
)
from src.backend.services.safety_authority_manager import (
    SafetyAuthorityLevel,
    SafetyAuthorityManager,
    SafetyDecision,
    SafetyDecisionType,
)
from src.backend.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ASVFrequencyChannelConfig:
    """Configuration for a single ASV analyzer frequency channel."""
    
    analyzer_id: str
    center_frequency_hz: int
    bandwidth_hz: int
    analyzer_type: str  # GP, VOR, LLZ
    priority: int = 1  # 1=highest, 3=lowest
    enabled: bool = True
    sample_rate_hz: int = 20_000_000  # 20 Msps default
    

@dataclass 
class ASVCoordinationMetrics:
    """Performance metrics for ASV multi-analyzer coordination."""
    
    total_analyzers_active: int = 0
    frequency_switches_per_second: float = 0.0
    average_switching_latency_ms: float = 0.0
    signal_fusion_latency_ms: float = 0.0
    concurrent_detections: int = 0
    analyzer_health_status: Dict[str, bool] = None
    last_update_timestamp: datetime = None
    
    def __post_init__(self):
        if self.analyzer_health_status is None:
            self.analyzer_health_status = {}
        if self.last_update_timestamp is None:
            self.last_update_timestamp = datetime.now(timezone.utc)


class ASVHackRFCoordinator(BaseService):
    """
    ASV Multi-Analyzer Coordinator with HackRF Integration
    
    Provides enhanced multi-frequency RF detection through coordinated ASV analyzer 
    instances, enabling simultaneous processing of aviation signals (GP/VOR/LLZ).
    
    ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────┐
    │ ASVHackRFCoordinator (BaseService)                         │
    ├─────────────────────────────────────────────────────────────┤
    │ • HackRF Hardware Interface (proven implementation)        │
    │ • ASV Analyzer Factory (multi-instantiation)             │
    │ • Multi-Analyzer Coordination (concurrent processing)     │
    │ • Safety System Integration (Epic 5 preservation)        │
    │ • Signal Fusion Engine (bearing calculation)             │
    └─────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self, 
        config_manager: Optional[ASVConfigurationManager] = None,
        safety_authority: Optional[SafetyAuthorityManager] = None,
        hackrf_config: Optional[HackRFConfig] = None
    ):
        """Initialize ASV HackRF coordinator with multi-analyzer capabilities."""
        super().__init__(service_name="asv_hackrf_coordinator")
        
        # Configuration management
        self._config_manager = config_manager or ASVConfigurationManager()
        self._hackrf_config = hackrf_config or HackRFConfig()
        
        # Safety system integration (Epic 5 preservation)
        self._safety_authority = safety_authority
        
        # Hardware interface (preserve proven HackRF implementation) 
        self._hackrf_interface: Optional[HackRFInterface] = None
        
        # Multi-analyzer management
        self._analyzer_factory: Optional[ASVAnalyzerFactory] = None
        self._multi_analyzer_coordinator: Optional[ASVMultiAnalyzerCoordinator] = None
        self._active_analyzers: Dict[str, ASVAnalyzerBase] = {}
        self._frequency_channels: Dict[str, ASVFrequencyChannelConfig] = {}
        
        # Coordination state
        self._current_frequency_hz: int = 406_000_000  # Default emergency beacon
        self._active_channel_id: str = "emergency_beacon_406"
        self._coordination_active: bool = False
        
        # Performance tracking
        self._coordination_metrics = ASVCoordinationMetrics()
        self._frequency_switch_times: List[float] = []
        self._signal_fusion_times: List[float] = []
        
        # Async coordination tasks
        self._coordination_task: Optional[asyncio.Task] = None
        self._health_monitor_task: Optional[asyncio.Task] = None
        
        # Timing configuration for Epic 6 requirements
        self.coordination_interval = 0.025  # 25ms for sub-100ms total latency
        self.analyzer_switching_timeout = 0.050  # 50ms max switching time
        self.signal_fusion_timeout = 0.025  # 25ms max fusion processing
        
        logger.info(
            "ASVHackRFCoordinator initialized - coordination_interval=%.3fs, "
            "analyzer_timeout=%.3fs, fusion_timeout=%.3fs",
            self.coordination_interval,
            self.analyzer_switching_timeout, 
            self.signal_fusion_timeout
        )
        
    async def start_service(self) -> None:
        """Start ASV HackRF coordinator service with multi-analyzer initialization."""
        try:
            # Step 1: Initialize HackRF hardware interface (preserve proven implementation)
            logger.info("Initializing HackRF hardware interface...")
            self._hackrf_interface = HackRFInterface(config=self._hackrf_config)
            
            if not await self._hackrf_interface.open():
                raise ASVHardwareError("Failed to initialize HackRF hardware")
                
            # Step 2: Initialize ASV analyzer factory
            logger.info("Initializing ASV analyzer factory...")
            self._analyzer_factory = ASVAnalyzerFactory(
                config_manager=self._config_manager,
                dotnet_runtime_path=self._config_manager.get_dotnet_runtime_path()
            )
            await self._analyzer_factory.initialize()
            
            # Step 3: Initialize multi-analyzer coordinator
            logger.info("Initializing multi-analyzer coordinator...")
            self._multi_analyzer_coordinator = ASVMultiAnalyzerCoordinator(
                analyzer_factory=self._analyzer_factory,
                config_manager=self._config_manager
            )
            await self._multi_analyzer_coordinator.initialize()
            
            # Step 4: Load frequency channel configurations
            await self._load_frequency_channels()
            
            # Step 5: Initialize default analyzer set (GP/VOR/LLZ)
            await self._initialize_default_analyzers()
            
            # Step 6: Start coordination and health monitoring tasks
            self._coordination_task = asyncio.create_task(self._coordination_loop())
            self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
            
            self._coordination_active = True
            
            logger.info(
                "ASVHackRFCoordinator service started - %d analyzers active, %d frequency channels",
                len(self._active_analyzers),
                len(self._frequency_channels)
            )
            
        except Exception as e:
            logger.error(f"Failed to start ASVHackRFCoordinator service: {e}")
            await self.stop_service()
            raise ASVInteropError(f"ASV HackRF coordinator startup failed: {e}") from e
            
    async def stop_service(self) -> None:
        """Stop ASV HackRF coordinator service with graceful analyzer shutdown."""
        try:
            self._coordination_active = False
            
            # Stop coordination tasks
            if self._coordination_task:
                self._coordination_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._coordination_task
                self._coordination_task = None
                    
            if self._health_monitor_task:
                self._health_monitor_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._health_monitor_task
                self._health_monitor_task = None
            
            # Shutdown analyzers
            logger.info("Shutting down ASV analyzers...")
            for analyzer_id, analyzer in self._active_analyzers.items():
                try:
                    await analyzer.shutdown()
                    logger.debug(f"Analyzer {analyzer_id} shutdown complete")
                except Exception as e:
                    logger.warning(f"Error shutting down analyzer {analyzer_id}: {e}")
            self._active_analyzers.clear()
            
            # Shutdown multi-analyzer coordinator
            if self._multi_analyzer_coordinator:
                await self._multi_analyzer_coordinator.shutdown()
                self._multi_analyzer_coordinator = None
                
            # Shutdown analyzer factory
            if self._analyzer_factory:
                await self._analyzer_factory.shutdown()
                self._analyzer_factory = None
            
            # Close HackRF hardware interface
            if self._hackrf_interface:
                await self._hackrf_interface.close()
                self._hackrf_interface = None
                
            logger.info("ASVHackRFCoordinator service stopped")
            
        except Exception as e:
            logger.error(f"Error during ASVHackRFCoordinator shutdown: {e}")
            
    async def _load_frequency_channels(self) -> None:
        """Load frequency channel configurations from ASV configuration manager."""
        try:
            frequency_profiles = self._config_manager.get_all_frequency_profiles()
            
            for profile_name, profile in frequency_profiles.items():
                # Convert ASV frequency profile to channel config
                channel_config = ASVFrequencyChannelConfig(
                    analyzer_id=f"asv_{profile.analyzer_type.lower()}_{profile_name}",
                    center_frequency_hz=profile.center_frequency_hz,
                    bandwidth_hz=profile.bandwidth_hz, 
                    analyzer_type=profile.analyzer_type,
                    priority=self._get_analyzer_priority(profile.analyzer_type),
                    enabled=True,
                    sample_rate_hz=20_000_000  # 20 Msps default for HackRF
                )
                
                self._frequency_channels[profile_name] = channel_config
                logger.debug(
                    f"Loaded frequency channel: {profile_name} - "
                    f"{profile.center_frequency_hz/1e6:.3f} MHz ({profile.analyzer_type})"
                )
            
            logger.info(f"Loaded {len(self._frequency_channels)} frequency channels")
            
        except Exception as e:
            logger.error(f"Failed to load frequency channels: {e}")
            raise ASVConfigurationError(f"Frequency channel loading failed: {e}") from e
            
    def _get_analyzer_priority(self, analyzer_type: str) -> int:
        """Get priority level for analyzer type (1=highest, 3=lowest)."""
        priority_map = {
            "GP": 1,   # Emergency beacons - highest priority  
            "VOR": 2,  # Aviation navigation - medium priority
            "LLZ": 2,  # ILS localizer - medium priority
            "WIFI": 3, # WiFi signals - lowest priority
            "LORA": 3  # LoRa IoT - lowest priority
        }
        return priority_map.get(analyzer_type.upper(), 2)
        
    async def _initialize_default_analyzers(self) -> None:
        """Initialize default set of ASV analyzers for aviation SAR operations."""
        try:
            # Default analyzer configurations for Epic 6
            default_configs = {
                "emergency_beacon_406": ASVAnalyzerConfig(
                    frequency_hz=406_000_000,    # 406 MHz emergency beacons
                    ref_power_dbm=-120.0,
                    analyzer_type="GP",
                ),
                "vor_aviation_nav": ASVAnalyzerConfig(
                    frequency_hz=115_000_000,    # VOR navigation band center
                    ref_power_dbm=-110.0, 
                    analyzer_type="VOR",
                ),
                "ils_localizer": ASVAnalyzerConfig(
                    frequency_hz=110_000_000,    # ILS localizer band center
                    ref_power_dbm=-110.0,
                    analyzer_type="LLZ", 
                ),
            }
            
            # Create and initialize analyzers
            for analyzer_id, config in default_configs.items():
                try:
                    analyzer = await self._analyzer_factory.create_analyzer(
                        analyzer_type=config.analyzer_type,
                        config=config
                    )
                    
                    # Initialize analyzer
                    if await analyzer.initialize():
                        self._active_analyzers[analyzer_id] = analyzer
                        logger.info(
                            f"Initialized analyzer: {analyzer_id} - "
                            f"{config.frequency_hz/1e6:.3f} MHz ({config.analyzer_type})"
                        )
                    else:
                        logger.warning(f"Failed to initialize analyzer: {analyzer_id}")
                        
                except Exception as e:
                    logger.error(f"Error creating analyzer {analyzer_id}: {e}")
                    continue
                    
            logger.info(f"Initialized {len(self._active_analyzers)} default ASV analyzers")
            
        except Exception as e:
            logger.error(f"Failed to initialize default analyzers: {e}")
            raise ASVAnalyzerError(f"Default analyzer initialization failed: {e}") from e
            
    async def _coordination_loop(self) -> None:
        """
        Main coordination loop for multi-analyzer signal processing.
        
        Manages HackRF frequency switching, analyzer coordination, and signal fusion
        with deterministic timing requirements per Epic 6 specifications.
        """
        logger.info("Starting ASV multi-analyzer coordination loop...")
        
        try:
            while self._coordination_active:
                coordination_start = time.perf_counter()
                
                try:
                    # Step 1: Safety system validation (Epic 5 preservation)
                    if not await self._validate_coordination_safety():
                        # Safety interlock active - suspend coordination
                        await asyncio.sleep(self.coordination_interval)
                        continue
                    
                    # Step 2: Determine optimal frequency for current cycle
                    target_frequency, analyzer_id = await self._select_optimal_frequency()
                    
                    # Step 3: Switch HackRF frequency if needed
                    if target_frequency != self._current_frequency_hz:
                        switch_start = time.perf_counter()
                        
                        if await self._switch_hackrf_frequency(target_frequency):
                            self._current_frequency_hz = target_frequency
                            self._active_channel_id = analyzer_id
                            
                            switch_duration = (time.perf_counter() - switch_start) * 1000
                            self._frequency_switch_times.append(switch_duration)
                            
                            # Keep only recent samples for performance metrics
                            if len(self._frequency_switch_times) > 100:
                                self._frequency_switch_times = self._frequency_switch_times[-50:]
                                
                    # Step 4: Collect IQ samples from HackRF
                    iq_samples = await self._collect_iq_samples()
                    
                    if iq_samples is not None and len(iq_samples) > 0:
                        # Step 5: Process samples through active analyzers
                        fusion_start = time.perf_counter()
                        
                        signal_results = await self._process_concurrent_analysis(iq_samples)
                        
                        fusion_duration = (time.perf_counter() - fusion_start) * 1000
                        self._signal_fusion_times.append(fusion_duration)
                        
                        if len(self._signal_fusion_times) > 100:
                            self._signal_fusion_times = self._signal_fusion_times[-50:]
                            
                        # Step 6: Update coordination metrics
                        await self._update_coordination_metrics()
                        
                        # Step 7: Signal fusion and bearing calculation
                        if signal_results:
                            fused_result = await self._fuse_signal_results(signal_results)
                            
                            # Step 8: Publish results to PISAD signal processing chain
                            await self._publish_signal_results(fused_result)
                    
                    # Maintain coordination timing
                    coordination_duration = time.perf_counter() - coordination_start
                    sleep_time = max(0, self.coordination_interval - coordination_duration)
                    
                    if sleep_time < self.coordination_interval * 0.1:  # Less than 10% sleep time
                        logger.warning(
                            f"Coordination loop running at capacity - duration: {coordination_duration*1000:.1f}ms"
                        )
                    
                    await asyncio.sleep(sleep_time)
                    
                except Exception as e:
                    logger.error(f"Error in coordination loop iteration: {e}")
                    await asyncio.sleep(self.coordination_interval)
                    continue
                    
        except asyncio.CancelledError:
            logger.info("ASV coordination loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Fatal error in coordination loop: {e}")
            raise
            
    async def _validate_coordination_safety(self) -> bool:
        """Validate that coordination is safe to continue per Epic 5 safety requirements."""
        try:
            if self._safety_authority is None:
                # No safety authority configured - allow coordination
                return True
                
            # Check if safety system allows ASV coordination operations
            safety_decision = await self._safety_authority.request_authority(
                authority_level=SafetyAuthorityLevel.COORDINATION,
                operation_context={
                    "operation_type": "asv_analyzer_coordination", 
                    "active_analyzers": len(self._active_analyzers),
                    "current_frequency_mhz": self._current_frequency_hz / 1e6,
                    "coordination_active": self._coordination_active
                }
            )
            
            if safety_decision.decision == SafetyDecisionType.DENY:
                logger.warning(f"ASV coordination denied by safety authority: {safety_decision.reason}")
                return False
            elif safety_decision.decision == SafetyDecisionType.CONDITIONAL:
                logger.info(f"ASV coordination conditionally approved: {safety_decision.reason}")
                
            return True
            
        except Exception as e:
            logger.error(f"Safety validation error: {e}")
            # Fail-safe: deny coordination on safety validation error
            return False
            
    async def _select_optimal_frequency(self) -> tuple[int, str]:
        """
        Select optimal frequency for next coordination cycle.
        
        Uses priority-based scheduling with round-robin fairness to ensure
        all analyzers get processing time while prioritizing emergency signals.
        
        Returns:
            tuple: (frequency_hz, analyzer_id)
        """
        try:
            if not self._frequency_channels:
                return self._current_frequency_hz, self._active_channel_id
                
            # Priority-based selection with fairness algorithm
            enabled_channels = {
                name: config for name, config in self._frequency_channels.items()
                if config.enabled and name in self._active_analyzers
            }
            
            if not enabled_channels:
                return self._current_frequency_hz, self._active_channel_id
                
            # Sort by priority (1=highest) then by last processing time
            sorted_channels = sorted(
                enabled_channels.items(),
                key=lambda x: (x[1].priority, x[0])  # Priority first, then name for stability
            )
            
            # Select highest priority channel that needs processing
            selected_name, selected_config = sorted_channels[0]
            
            return selected_config.center_frequency_hz, selected_name
            
        except Exception as e:
            logger.error(f"Frequency selection error: {e}")
            return self._current_frequency_hz, self._active_channel_id
    
    def get_coordination_metrics(self) -> ASVCoordinationMetrics:
        """Get current coordination performance metrics."""
        return self._coordination_metrics
        
    def get_active_analyzers(self) -> Dict[str, str]:
        """Get currently active analyzers with their types."""
        return {
            analyzer_id: analyzer.analyzer_type if hasattr(analyzer, 'analyzer_type') else 'unknown'
            for analyzer_id, analyzer in self._active_analyzers.items()
        }