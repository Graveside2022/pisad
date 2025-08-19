"""
Test ASV HackRF Multi-Analyzer Coordinator

SUBTASK-6.1.2.1: Tests for ASV HackRF coordinator service with multi-analyzer management
"""

import asyncio
import pytest
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

from src.backend.services.asv_integration.asv_analyzer_wrapper import ASVAnalyzerConfig
from src.backend.services.asv_integration.asv_configuration_manager import ASVConfigurationManager
from src.backend.services.asv_integration.asv_hackrf_coordinator import (
    ASVCoordinationMetrics,
    ASVFrequencyChannelConfig,
    ASVHackRFCoordinator,
)
from src.backend.services.safety_authority_manager import SafetyAuthorityManager
from src.backend.hal.hackrf_interface import HackRFConfig


@pytest.fixture
def temp_config_dir():
    """Create temporary directory for configuration."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_config_manager(temp_config_dir):
    """Create mock ASV configuration manager."""
    return ASVConfigurationManager(temp_config_dir)


@pytest.fixture
def mock_safety_authority():
    """Create mock safety authority manager."""
    mock_safety = AsyncMock(spec=SafetyAuthorityManager)
    
    # Mock safety approval
    from src.backend.services.safety_authority_manager import SafetyDecision, SafetyDecisionType
    mock_decision = SafetyDecision(
        decision=SafetyDecisionType.APPROVE,
        reason="Test approval",
        authority_level=None,
        conditions=[],
        timeout_seconds=None
    )
    mock_safety.request_authority.return_value = mock_decision
    
    return mock_safety


@pytest.fixture
def hackrf_config():
    """Create HackRF configuration for testing."""
    return HackRFConfig(
        frequency=406_000_000,  # 406 MHz emergency beacon
        sample_rate=20_000_000,  # 20 Msps
        lna_gain=16,
        vga_gain=20,
        amp_enable=False
    )


class TestASVHackRFCoordinator:
    """Test ASV HackRF multi-analyzer coordinator functionality."""
    
    @pytest.mark.asyncio
    async def test_coordinator_initialization(
        self,
        mock_config_manager,
        mock_safety_authority,
        hackrf_config
    ):
        """Test ASV HackRF coordinator initialization."""
        
        # Create coordinator
        coordinator = ASVHackRFCoordinator(
            config_manager=mock_config_manager,
            safety_authority=mock_safety_authority,
            hackrf_config=hackrf_config
        )
        
        # Test initialization attributes
        assert coordinator.service_name == "asv_hackrf_coordinator"
        assert coordinator._config_manager is mock_config_manager
        assert coordinator._safety_authority is mock_safety_authority
        assert coordinator._hackrf_config is hackrf_config
        
        # Test timing configuration
        assert coordinator.coordination_interval == 0.025  # 25ms
        assert coordinator.analyzer_switching_timeout == 0.050  # 50ms
        assert coordinator.signal_fusion_timeout == 0.025  # 25ms
        
        # Test initial state
        assert not coordinator._coordination_active
        assert len(coordinator._active_analyzers) == 0
        assert len(coordinator._frequency_channels) == 0
        assert coordinator._current_frequency_hz == 406_000_000
        
        print("✅ ASV HackRF coordinator initialization successful")
        
    def test_frequency_channel_config_creation(self):
        """Test ASV frequency channel configuration creation."""
        
        channel_config = ASVFrequencyChannelConfig(
            analyzer_id="test_gp_analyzer",
            center_frequency_hz=406_000_000,
            bandwidth_hz=20_000,
            analyzer_type="GP",
            priority=1,
            enabled=True,
            sample_rate_hz=20_000_000
        )
        
        assert channel_config.analyzer_id == "test_gp_analyzer"
        assert channel_config.center_frequency_hz == 406_000_000
        assert channel_config.bandwidth_hz == 20_000
        assert channel_config.analyzer_type == "GP"
        assert channel_config.priority == 1
        assert channel_config.enabled is True
        
        print("✅ Frequency channel configuration created successfully")
        
    def test_coordination_metrics_creation(self):
        """Test ASV coordination metrics structure."""
        
        metrics = ASVCoordinationMetrics(
            total_analyzers_active=3,
            frequency_switches_per_second=5.2,
            average_switching_latency_ms=12.5,
            signal_fusion_latency_ms=8.3,
            concurrent_detections=2,
            analyzer_health_status={"gp": True, "vor": True, "llz": False}
        )
        
        assert metrics.total_analyzers_active == 3
        assert metrics.frequency_switches_per_second == 5.2
        assert metrics.average_switching_latency_ms == 12.5
        assert metrics.signal_fusion_latency_ms == 8.3
        assert metrics.concurrent_detections == 2
        assert metrics.analyzer_health_status["gp"] is True
        assert metrics.analyzer_health_status["llz"] is False
        assert metrics.last_update_timestamp is not None
        
        print("✅ Coordination metrics structure validated")
        
    def test_analyzer_priority_mapping(self):
        """Test analyzer priority mapping logic."""
        
        coordinator = ASVHackRFCoordinator()
        
        # Test priority mapping
        assert coordinator._get_analyzer_priority("GP") == 1    # Emergency - highest
        assert coordinator._get_analyzer_priority("VOR") == 2   # Aviation - medium
        assert coordinator._get_analyzer_priority("LLZ") == 2   # Aviation - medium  
        assert coordinator._get_analyzer_priority("WIFI") == 3  # WiFi - lowest
        assert coordinator._get_analyzer_priority("LORA") == 3  # LoRa - lowest
        assert coordinator._get_analyzer_priority("UNKNOWN") == 2  # Default
        
        print("✅ Analyzer priority mapping validated")
        
    @pytest.mark.asyncio
    async def test_frequency_selection_algorithm(self, mock_config_manager):
        """Test optimal frequency selection algorithm."""
        
        coordinator = ASVHackRFCoordinator(config_manager=mock_config_manager)
        
        # Setup test frequency channels
        coordinator._frequency_channels = {
            "emergency_gp": ASVFrequencyChannelConfig(
                analyzer_id="emergency_gp",
                center_frequency_hz=406_000_000,
                bandwidth_hz=20_000,
                analyzer_type="GP",
                priority=1,  # Highest priority
                enabled=True
            ),
            "vor_nav": ASVFrequencyChannelConfig(
                analyzer_id="vor_nav", 
                center_frequency_hz=115_000_000,
                bandwidth_hz=20_000,
                analyzer_type="VOR",
                priority=2,  # Medium priority
                enabled=True
            )
        }
        
        # Mock analyzers
        coordinator._active_analyzers = {
            "emergency_gp": MagicMock(),
            "vor_nav": MagicMock()
        }
        
        # Test frequency selection
        freq, analyzer_id = await coordinator._select_optimal_frequency()
        
        # Should select highest priority (GP emergency beacon)
        assert freq == 406_000_000
        assert analyzer_id == "emergency_gp"
        
        print("✅ Frequency selection algorithm working correctly")
        
    @pytest.mark.asyncio  
    async def test_safety_validation_integration(self, mock_safety_authority):
        """Test safety system integration with coordination."""
        
        coordinator = ASVHackRFCoordinator(safety_authority=mock_safety_authority)
        coordinator._active_analyzers = {"test": MagicMock()}
        coordinator._current_frequency_hz = 406_000_000
        coordinator._coordination_active = True
        
        # Test safety validation
        result = await coordinator._validate_coordination_safety()
        
        # Should approve coordination
        assert result is True
        
        # Verify safety authority was called correctly
        mock_safety_authority.request_authority.assert_called_once()
        call_args = mock_safety_authority.request_authority.call_args
        
        assert "operation_type" in call_args.kwargs["operation_context"]
        assert call_args.kwargs["operation_context"]["operation_type"] == "asv_analyzer_coordination"
        assert call_args.kwargs["operation_context"]["active_analyzers"] == 1
        assert call_args.kwargs["operation_context"]["current_frequency_mhz"] == 406.0
        
        print("✅ Safety validation integration working correctly")
        
    @pytest.mark.asyncio
    async def test_coordination_without_safety_authority(self):
        """Test coordination when no safety authority is configured."""
        
        coordinator = ASVHackRFCoordinator(safety_authority=None)
        
        # Should allow coordination without safety authority
        result = await coordinator._validate_coordination_safety()
        assert result is True
        
        print("✅ Coordination works without safety authority")
        
    @pytest.mark.asyncio
    async def test_analyzer_management_api(self, mock_config_manager):
        """Test analyzer add/remove API methods."""
        
        with patch('src.backend.services.asv_integration.asv_hackrf_coordinator.ASVAnalyzerFactory') as MockFactory:
            # Setup mock factory
            mock_factory = AsyncMock()
            mock_analyzer = AsyncMock()
            mock_analyzer.initialize.return_value = True
            mock_analyzer.analyzer_type = "GP"
            mock_factory.create_analyzer.return_value = mock_analyzer
            MockFactory.return_value = mock_factory
            
            coordinator = ASVHackRFCoordinator(config_manager=mock_config_manager)
            coordinator._analyzer_factory = mock_factory
            
            # Test analyzer addition
            config = ASVAnalyzerConfig(
                frequency_hz=406_000_000,
                ref_power_dbm=-120.0,
                analyzer_type="GP"
            )
            
            result = await coordinator.add_analyzer("test_analyzer", config)
            
            assert result is True
            assert "test_analyzer" in coordinator._active_analyzers
            assert "test_analyzer" in coordinator._frequency_channels
            
            # Verify frequency channel configuration
            channel_config = coordinator._frequency_channels["test_analyzer"]
            assert channel_config.center_frequency_hz == 406_000_000
            assert channel_config.analyzer_type == "GP"
            assert channel_config.priority == 1  # GP should be highest priority
            
            print("✅ Analyzer management API working correctly")
            
    def test_active_analyzers_query(self, mock_config_manager):
        """Test active analyzers query functionality."""
        
        coordinator = ASVHackRFCoordinator(config_manager=mock_config_manager)
        
        # Setup mock analyzers  
        mock_gp = MagicMock()
        mock_gp.analyzer_type = "GP"
        mock_vor = MagicMock()
        mock_vor.analyzer_type = "VOR"
        
        coordinator._active_analyzers = {
            "emergency_beacon": mock_gp,
            "vor_nav": mock_vor
        }
        
        # Test active analyzers query
        active = coordinator.get_active_analyzers()
        
        assert len(active) == 2
        assert active["emergency_beacon"] == "GP"
        assert active["vor_nav"] == "VOR"
        
        print("✅ Active analyzers query working correctly")
        
    def test_coordination_metrics_access(self):
        """Test coordination metrics access."""
        
        coordinator = ASVHackRFCoordinator()
        
        # Get initial metrics
        metrics = coordinator.get_coordination_metrics()
        
        assert isinstance(metrics, ASVCoordinationMetrics)
        assert metrics.total_analyzers_active == 0
        assert metrics.frequency_switches_per_second == 0.0
        assert metrics.last_update_timestamp is not None
        
        print("✅ Coordination metrics access working correctly")
        
    @pytest.mark.asyncio
    async def test_frequency_priority_setting(self, mock_config_manager):
        """Test frequency priority setting functionality."""
        
        coordinator = ASVHackRFCoordinator(config_manager=mock_config_manager)
        
        # Setup test frequency channel
        coordinator._frequency_channels = {
            "test_analyzer": ASVFrequencyChannelConfig(
                analyzer_id="test_analyzer",
                center_frequency_hz=406_000_000,
                bandwidth_hz=20_000,
                analyzer_type="GP",
                priority=1,
                enabled=True
            )
        }
        
        # Test priority setting
        result = await coordinator.set_frequency_priority("test_analyzer", 3)
        
        assert result is True
        assert coordinator._frequency_channels["test_analyzer"].priority == 3
        
        # Test invalid analyzer ID
        result = await coordinator.set_frequency_priority("nonexistent", 2)
        assert result is False
        
        # Test priority bounds (should clamp to 1-3 range)
        await coordinator.set_frequency_priority("test_analyzer", 0)
        assert coordinator._frequency_channels["test_analyzer"].priority == 1
        
        await coordinator.set_frequency_priority("test_analyzer", 5)
        assert coordinator._frequency_channels["test_analyzer"].priority == 3
        
        print("✅ Frequency priority setting working correctly")


if __name__ == "__main__":
    # Run basic tests
    asyncio.run(test_coordinator_basic_functionality())
    
    
async def test_coordinator_basic_functionality():
    """Basic functionality test that can be run directly."""
    
    print("🧪 Running basic ASV HackRF Coordinator tests...")
    
    # Test 1: Basic initialization
    coordinator = ASVHackRFCoordinator()
    assert coordinator.service_name == "asv_hackrf_coordinator"
    print("✅ Basic initialization test passed")
    
    # Test 2: Priority mapping
    assert coordinator._get_analyzer_priority("GP") == 1
    assert coordinator._get_analyzer_priority("VOR") == 2
    assert coordinator._get_analyzer_priority("WIFI") == 3
    print("✅ Priority mapping test passed")
    
    # Test 3: Metrics structure
    metrics = coordinator.get_coordination_metrics()
    assert isinstance(metrics, ASVCoordinationMetrics)
    assert metrics.total_analyzers_active == 0
    print("✅ Metrics structure test passed")
    
    # Test 4: Safety validation without authority
    result = await coordinator._validate_coordination_safety()
    assert result is True
    print("✅ Safety validation test passed")
    
    print("🎉 All basic tests passed!")
    print(f"📊 Total active analyzers: {len(coordinator._active_analyzers)}")
    print(f"📊 Coordination interval: {coordinator.coordination_interval*1000:.1f}ms")
    print(f"📊 Current frequency: {coordinator._current_frequency_hz/1e6:.3f} MHz")