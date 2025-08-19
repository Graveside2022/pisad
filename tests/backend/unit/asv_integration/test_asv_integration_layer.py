"""Comprehensive tests for ASV integration layer architecture.

SUBTASK-6.1.1.3: Create ASV integration layer architecture

This test suite validates the complete ASV integration layer including
wrapper classes, factory, error handling, and configuration management.
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.backend.services.asv_integration.asv_analyzer_wrapper import (
    ASVAnalyzerConfig,
    ASVGpAnalyzer,
    ASVVorAnalyzer,
    ASVLlzAnalyzer,
    ASVSignalData,
    create_analyzer,
)
from src.backend.services.asv_integration.asv_analyzer_factory import (
    ASVAnalyzerFactory,
    ASVMultiAnalyzerCoordinator,
)
from src.backend.services.asv_integration.asv_configuration_manager import (
    ASVConfigurationManager,
    ASVFrequencyProfile,
    ASVAnalyzerProfile,
)
from src.backend.services.asv_integration.asv_interop_service import ASVInteropService
from src.backend.services.asv_integration.exceptions import (
    ASVAnalyzerError,
    ASVConfigurationError,
    ASVFrequencyError,
    translate_dotnet_exception,
    exception_handler,
)


class TestASVAnalyzerWrappers:
    """Test ASV analyzer wrapper classes."""

    def setup_method(self):
        """Set up test environment."""
        self.config = ASVAnalyzerConfig(
            frequency_hz=406_000_000,
            ref_power_dbm=-120.0,
            analyzer_type="GP",
            calibration_enabled=True,
            processing_timeout_ms=100,
        )

    def test_asv_analyzer_config_creation(self):
        """Test analyzer configuration creation."""
        assert self.config.frequency_hz == 406_000_000
        assert self.config.ref_power_dbm == -120.0
        assert self.config.analyzer_type == "GP"
        assert self.config.calibration_enabled is True
        print("✅ ASV analyzer configuration created successfully")

    def test_gp_analyzer_creation(self):
        """Test GP analyzer wrapper creation."""
        analyzer = ASVGpAnalyzer(self.config)

        assert analyzer.analyzer_type == "GP"
        assert analyzer.frequency_hz == 406_000_000
        assert not analyzer.is_initialized
        print("✅ GP analyzer wrapper created")

    def test_vor_analyzer_creation(self):
        """Test VOR analyzer wrapper creation."""
        vor_config = ASVAnalyzerConfig(
            frequency_hz=112_500_000,
            ref_power_dbm=-100.0,
            analyzer_type="VOR",
        )
        analyzer = ASVVorAnalyzer(vor_config)

        assert analyzer.analyzer_type == "VOR"
        assert analyzer.frequency_hz == 112_500_000
        print("✅ VOR analyzer wrapper created")

    def test_llz_analyzer_creation(self):
        """Test LLZ analyzer wrapper creation."""
        llz_config = ASVAnalyzerConfig(
            frequency_hz=109_500_000,
            ref_power_dbm=-90.0,
            analyzer_type="LLZ",
        )
        analyzer = ASVLlzAnalyzer(llz_config)

        assert analyzer.analyzer_type == "LLZ"
        assert analyzer.frequency_hz == 109_500_000
        print("✅ LLZ analyzer wrapper created")

    @pytest.mark.asyncio
    async def test_analyzer_initialization(self):
        """Test analyzer initialization and shutdown."""
        analyzer = ASVGpAnalyzer(self.config)

        # Test initialization
        success = await analyzer.initialize()
        assert success is True
        assert analyzer.is_initialized

        # Test shutdown
        await analyzer.shutdown()
        assert not analyzer.is_initialized
        print("✅ Analyzer initialization and shutdown working")

    @pytest.mark.asyncio
    async def test_signal_processing(self):
        """Test signal processing with different analyzers."""
        test_iq_data = b"test_iq_data_12345"

        # Test GP analyzer
        gp_analyzer = ASVGpAnalyzer(self.config)
        await gp_analyzer.initialize()

        signal_data = await gp_analyzer.process_signal(test_iq_data)
        assert isinstance(signal_data, ASVSignalData)
        assert signal_data.analyzer_type == "GP"
        assert signal_data.frequency_hz == 406_000_000
        assert "beacon_detected" in signal_data.raw_data
        print(f"✅ GP analyzer processed signal: {signal_data.signal_strength_dbm} dBm")

        await gp_analyzer.shutdown()

    def test_analyzer_factory_function(self):
        """Test analyzer factory function."""
        # Test valid analyzer types
        gp_analyzer = create_analyzer("GP", self.config)
        assert isinstance(gp_analyzer, ASVGpAnalyzer)

        vor_config = ASVAnalyzerConfig(
            frequency_hz=112_500_000,
            ref_power_dbm=-100.0,
            analyzer_type="VOR",
        )
        vor_analyzer = create_analyzer("VOR", vor_config)
        assert isinstance(vor_analyzer, ASVVorAnalyzer)

        # Test invalid analyzer type
        with pytest.raises(ASVAnalyzerError):
            create_analyzer("INVALID", self.config)

        print("✅ Analyzer factory function working correctly")


class TestASVAnalyzerFactory:
    """Test ASV analyzer factory and multi-analyzer coordination."""

    def setup_method(self):
        """Set up test environment."""
        self.interop_service = Mock(spec=ASVInteropService)
        self.interop_service.is_running = True
        self.interop_service.get_analyzer_types.return_value = [
            "Asv.Drones.Sdr.Core.IAnalyzerGp",
            "Asv.Drones.Sdr.Core.IAnalyzerVor",
            "Asv.Drones.Sdr.Core.IAnalyzerLlz",
        ]
        self.interop_service.create_analyzer_instance.return_value = "mock_dotnet_instance"

        self.factory = ASVAnalyzerFactory(self.interop_service)

    @pytest.mark.asyncio
    async def test_factory_lifecycle(self):
        """Test analyzer factory startup and shutdown."""
        assert not self.factory.is_running

        await self.factory.start()
        assert self.factory.is_running

        await self.factory.stop()
        assert not self.factory.is_running
        print("✅ Analyzer factory lifecycle working")

    @pytest.mark.asyncio
    async def test_analyzer_creation_and_removal(self):
        """Test creating and removing analyzers."""
        await self.factory.start()

        # Create analyzer configuration
        config = self.factory.create_analyzer_config(
            analyzer_type="GP",
            frequency_hz=406_000_000,
            ref_power_dbm=-120.0,
        )

        # Create analyzer
        analyzer_id = await self.factory.create_analyzer("test_gp", config)
        assert analyzer_id == "test_gp"
        assert "test_gp" in self.factory.active_analyzers

        # Check status
        status = self.factory.get_analyzer_status()
        assert "test_gp" in status
        assert status["test_gp"]["type"] == "GP"

        # Remove analyzer
        await self.factory.remove_analyzer("test_gp")
        assert "test_gp" not in self.factory.active_analyzers

        await self.factory.stop()
        print("✅ Analyzer creation and removal working")

    @pytest.mark.asyncio
    async def test_concurrent_signal_processing(self):
        """Test concurrent processing across multiple analyzers."""
        await self.factory.start()

        # Create multiple analyzers
        configs = {
            "gp_analyzer": self.factory.create_analyzer_config("GP", 406_000_000),
            "vor_analyzer": self.factory.create_analyzer_config("VOR", 112_500_000),
            "llz_analyzer": self.factory.create_analyzer_config("LLZ", 109_500_000),
        }

        for analyzer_id, config in configs.items():
            await self.factory.create_analyzer(analyzer_id, config)

        # Process signal concurrently
        test_iq_data = b"concurrent_test_data"
        results = await self.factory.process_signal_concurrent(test_iq_data)

        assert len(results) == 3
        assert all(isinstance(data, ASVSignalData) for data in results.values())

        # Check each analyzer processed the signal
        assert "gp_analyzer" in results
        assert "vor_analyzer" in results
        assert "llz_analyzer" in results

        await self.factory.stop()
        print(f"✅ Concurrent processing: {len(results)} analyzers processed signal")

    @pytest.mark.asyncio
    async def test_standard_analyzer_set_creation(self):
        """Test creating standard SAR analyzer set."""
        await self.factory.start()

        created_analyzers = await self.factory.create_standard_analyzer_set()

        assert len(created_analyzers) >= 2  # At least emergency and aviation
        assert "emergency_beacon" in created_analyzers or "aviation_vor" in created_analyzers

        # Verify analyzers are active
        for analyzer_id in created_analyzers.values():
            assert analyzer_id in self.factory.active_analyzers

        await self.factory.stop()
        print(f"✅ Created standard analyzer set: {list(created_analyzers.keys())}")


class TestASVMultiAnalyzerCoordinator:
    """Test multi-analyzer coordination and signal fusion."""

    def setup_method(self):
        """Set up test environment."""
        self.interop_service = Mock(spec=ASVInteropService)
        self.interop_service.is_running = True
        self.interop_service.get_analyzer_types.return_value = []
        self.factory = ASVAnalyzerFactory(self.interop_service)
        self.coordinator = ASVMultiAnalyzerCoordinator(self.factory)

    @pytest.mark.asyncio
    async def test_signal_fusion_processing(self):
        """Test signal processing with fusion algorithm."""
        await self.factory.start()

        # Create test analyzers
        gp_config = self.factory.create_analyzer_config("GP", 406_000_000)
        await self.factory.create_analyzer("test_gp", gp_config)

        # Process signal with fusion
        test_iq_data = b"fusion_test_data"
        result = await self.coordinator.process_signal_with_fusion(test_iq_data)

        assert result["status"] == "success"
        assert result["analyzers_used"] >= 1
        assert "signals" in result
        assert "strongest_signal" in result
        assert "processing_time_ms" in result
        assert result["processing_time_ms"] < 1000  # Should be fast

        await self.factory.stop()
        print(f"✅ Signal fusion processing: {result['processing_time_ms']:.2f}ms")

    def test_processing_statistics(self):
        """Test processing statistics tracking."""
        stats = self.coordinator.get_processing_statistics()

        assert "total_processed" in stats
        assert "successful_processing" in stats
        assert "failed_processing" in stats
        assert "average_latency_ms" in stats
        assert "success_rate" in stats
        assert "failure_rate" in stats
        print("✅ Processing statistics tracking working")


class TestASVConfigurationManager:
    """Test ASV configuration management and PISAD integration."""

    def setup_method(self):
        """Set up test environment with temporary config directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ASVConfigurationManager(self.temp_dir)

    def test_default_configuration_creation(self):
        """Test default configuration creation."""
        global_config = self.config_manager.get_global_config()

        assert "asv_integration" in global_config
        assert "dotnet_runtime" in global_config
        assert "hardware_interface" in global_config
        assert "safety_integration" in global_config

        # Check specific values
        assert global_config["asv_integration"]["enabled"] is True
        assert global_config["safety_integration"]["respect_safety_interlocks"] is True
        print("✅ Default configuration created and loaded")

    def test_frequency_profile_management(self):
        """Test frequency profile creation and management."""
        # Get default profiles
        profiles = self.config_manager.get_all_frequency_profiles()
        assert len(profiles) > 0
        assert "emergency_beacon_406" in profiles

        # Test specific profile
        emergency_profile = self.config_manager.get_frequency_profile("emergency_beacon_406")
        assert emergency_profile.center_frequency_hz == 406_000_000
        assert emergency_profile.analyzer_type == "GP"

        # Create new profile
        new_profile = ASVFrequencyProfile(
            name="test_profile",
            description="Test Profile",
            center_frequency_hz=123_456_789,
            bandwidth_hz=100_000,
            analyzer_type="GP",
        )

        self.config_manager.create_frequency_profile(new_profile)
        retrieved_profile = self.config_manager.get_frequency_profile("test_profile")
        assert retrieved_profile.center_frequency_hz == 123_456_789
        print("✅ Frequency profile management working")

    def test_analyzer_profile_management(self):
        """Test analyzer profile creation and management."""
        # Get default profiles
        profiles = self.config_manager.get_all_analyzer_profiles()
        assert len(profiles) > 0

        # Test specific profile
        emergency_analyzer = self.config_manager.get_analyzer_profile("emergency_analyzer")
        assert emergency_analyzer.analyzer_type == "GP"
        assert emergency_analyzer.enabled is True
        assert len(emergency_analyzer.frequency_profiles) > 0
        print("✅ Analyzer profile management working")

    def test_pisad_configuration_compatibility(self):
        """Test PISAD-compatible configuration generation."""
        pisad_config = self.config_manager.get_pisad_compatible_config("emergency_analyzer")

        assert "sdr_service" in pisad_config
        assert "signal_processing" in pisad_config
        assert "frequency_profiles" in pisad_config

        # Check SDR service config
        sdr_config = pisad_config["sdr_service"]
        assert sdr_config["device_type"] == "hackrf"
        assert "sample_rate" in sdr_config
        assert "gain" in sdr_config

        # Check frequency profiles
        freq_profiles = pisad_config["frequency_profiles"]
        assert len(freq_profiles) > 0
        assert "frequency_hz" in freq_profiles[0]
        assert "detection_threshold_dbm" in freq_profiles[0]

        print("✅ PISAD-compatible configuration generation working")

    def test_frequency_validation(self):
        """Test frequency range validation."""
        # Valid frequency should pass
        valid_profile = ASVFrequencyProfile(
            name="valid_test",
            description="Valid Test",
            center_frequency_hz=100_000_000,  # 100 MHz
            bandwidth_hz=1000,
            analyzer_type="GP",
        )

        # Should not raise exception
        self.config_manager.create_frequency_profile(valid_profile)

        # Invalid frequency should fail
        with pytest.raises(ASVFrequencyError):
            invalid_profile = ASVFrequencyProfile(
                name="invalid_test",
                description="Invalid Test",
                center_frequency_hz=100_000,  # 100 kHz (too low)
                bandwidth_hz=1000,
                analyzer_type="GP",
            )
            self.config_manager.create_frequency_profile(invalid_profile)

        print("✅ Frequency validation working correctly")

    def test_configuration_persistence(self):
        """Test configuration saving and loading."""
        # Modify global config
        self.config_manager.set_global_config("asv_integration.max_concurrent_analyzers", 10)

        # Create new manager instance to test loading
        new_manager = ASVConfigurationManager(self.temp_dir)

        # Check value was persisted
        max_analyzers = new_manager.get_global_config("asv_integration.max_concurrent_analyzers")
        assert max_analyzers == 10
        print("✅ Configuration persistence working")


class TestASVExceptionHandling:
    """Test ASV exception handling and translation."""

    def test_exception_translation(self):
        """Test .NET exception translation to ASV exceptions."""
        # Mock .NET exceptions
        mock_file_not_found = Exception("File not found")
        mock_file_not_found.__class__.__name__ = "FileNotFoundException"

        translated = translate_dotnet_exception(mock_file_not_found)
        assert isinstance(translated, ASVAssemblyLoadError)
        assert "FileNotFoundException" in str(translated)

        print("✅ Exception translation working")

    def test_exception_handler_statistics(self):
        """Test exception handler statistics tracking."""
        handler = exception_handler
        handler.clear_statistics()

        # Handle some exceptions
        test_exception = Exception("Test error")
        handled = handler.handle_exception("test_operation", test_exception)

        assert isinstance(handled, ASVInteropError)

        # Check statistics
        stats = handler.get_error_statistics()
        assert stats["total_errors"] == 1
        assert stats["unique_error_types"] == 1
        assert "test_operation" in stats["recent_errors"]

        print("✅ Exception handler statistics working")

    def test_exception_context_preservation(self):
        """Test exception context preservation."""
        context = {"analyzer_id": "test_analyzer", "frequency_hz": 123456789}

        error = ASVAnalyzerError("Test error", analyzer_id="test_analyzer")
        assert error.context["analyzer_id"] == "test_analyzer"

        error_dict = error.to_dict()
        assert "context" in error_dict
        assert error_dict["context"]["analyzer_id"] == "test_analyzer"

        print("✅ Exception context preservation working")


@pytest.mark.integration
class TestASVIntegrationLayerComplete:
    """Integration tests for complete ASV integration layer."""

    @pytest.mark.asyncio
    async def test_complete_integration_workflow(self):
        """Test complete ASV integration workflow."""
        # Set up components
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = ASVConfigurationManager(temp_dir)

            interop_service = Mock(spec=ASVInteropService)
            interop_service.is_running = True
            interop_service.get_analyzer_types.return_value = []

            factory = ASVAnalyzerFactory(interop_service)
            coordinator = ASVMultiAnalyzerCoordinator(factory)

            # Start system
            await factory.start()

            # Create analyzer from configuration
            emergency_profile = config_manager.get_analyzer_profile("emergency_analyzer")

            config = factory.create_analyzer_config(
                analyzer_type=emergency_profile.analyzer_type,
                frequency_hz=emergency_profile.frequency_profiles[0].center_frequency_hz,
                ref_power_dbm=emergency_profile.frequency_profiles[0].ref_power_dbm,
            )

            analyzer_id = await factory.create_analyzer("integration_test", config)

            # Process signals
            test_data = b"integration_test_signal_data"
            fusion_result = await coordinator.process_signal_with_fusion(test_data)

            assert fusion_result["status"] == "success"
            assert fusion_result["analyzers_used"] >= 1

            # Get PISAD-compatible config
            pisad_config = config_manager.get_pisad_compatible_config("emergency_analyzer")
            assert "sdr_service" in pisad_config

            # Clean up
            await factory.stop()

            print("✅ Complete ASV integration workflow successful")
            print(f"   - Analyzer created: {analyzer_id}")
            print(f"   - Signal processed in: {fusion_result['processing_time_ms']:.2f}ms")
            print(f"   - PISAD config generated: {len(pisad_config)} sections")
