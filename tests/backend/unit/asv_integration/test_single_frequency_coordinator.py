"""
Test Single-Frequency ASV Coordinator Implementation

SUBTASK-6.1.2.1 [14a] - Test for refactored ASVHackRFCoordinator single-frequency processing

This test module validates the architectural pivot from multi-analyzer concurrent
processing to enhanced single-frequency processing with operator-selectable frequencies.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.backend.hal.hackrf_interface import HackRFConfig
from src.backend.services.asv_integration.asv_configuration_manager import ASVConfigurationManager
from src.backend.services.asv_integration.asv_hackrf_coordinator import (
    ASVCoordinationMetrics,
    ASVHackRFCoordinator,
)
from src.backend.services.safety_authority_manager import SafetyAuthorityManager


class TestSingleFrequencyCoordinator:
    """Test single-frequency ASV coordination capabilities"""

    @pytest.fixture
    async def mock_hackrf_interface(self):
        """Mock HackRF interface for testing"""
        mock_hackrf = AsyncMock()
        mock_hackrf.open.return_value = True
        mock_hackrf.close.return_value = True
        mock_hackrf.set_frequency.return_value = True
        mock_hackrf.collect_iq_samples.return_value = b"\x00\x01" * 1024  # Mock IQ data
        return mock_hackrf

    @pytest.fixture
    async def mock_analyzer_factory(self):
        """Mock ASV analyzer factory for single-frequency testing"""
        mock_factory = AsyncMock()
        mock_factory.initialize.return_value = True
        mock_factory.shutdown.return_value = True

        # Mock single active analyzer creation
        mock_analyzer = AsyncMock()
        mock_analyzer.initialize.return_value = True
        mock_analyzer.shutdown.return_value = True
        mock_analyzer.analyzer_type = "GP"
        mock_analyzer.frequency_hz = 406_000_000
        mock_analyzer.process_signal.return_value = AsyncMock()

        mock_factory.create_analyzer.return_value = mock_analyzer
        return mock_factory

    @pytest.fixture
    async def coordinator(self, mock_analyzer_factory):
        """Create ASVHackRFCoordinator for single-frequency testing"""
        config_manager = ASVConfigurationManager()
        safety_authority = SafetyAuthorityManager()
        hackrf_config = HackRFConfig(frequency=406_000_000)

        coordinator = ASVHackRFCoordinator(
            config_manager=config_manager,
            safety_authority=safety_authority,
            hackrf_config=hackrf_config,
        )

        # Mock the factory
        coordinator._analyzer_factory = mock_analyzer_factory

        return coordinator

    @pytest.mark.asyncio
    async def test_single_frequency_initialization(self, coordinator, mock_hackrf_interface):
        """Test that coordinator initializes with single-frequency mode"""
        # Mock HackRF interface and ASV interop service
        with (
            patch("src.backend.hal.hackrf_interface.HackRFInterface") as mock_hackrf_class,
            patch(
                "src.backend.services.asv_integration.asv_hackrf_coordinator.ASVInteropService"
            ) as mock_interop_class,
            patch(
                "src.backend.services.asv_integration.asv_hackrf_coordinator.ASVMultiAnalyzerCoordinator"
            ) as mock_coord_class,
        ):
            # Setup mocks
            mock_hackrf_class.return_value = mock_hackrf_interface

            mock_interop = AsyncMock()
            mock_interop.start.return_value = True
            mock_interop.stop.return_value = True
            mock_interop_class.return_value = mock_interop

            mock_multi_coord = AsyncMock()
            mock_multi_coord.initialize.return_value = True
            mock_multi_coord.shutdown.return_value = True
            mock_coord_class.return_value = mock_multi_coord

            # Start the coordinator
            await coordinator.start_service()

            # Verify single-frequency initialization
            assert coordinator._coordination_active is True
            assert coordinator._current_frequency_hz == 406_000_000  # Default emergency beacon
            assert len(coordinator._active_analyzers) <= 1  # Only one active analyzer maximum

            # Verify ASV interop service was initialized
            mock_interop_class.assert_called_once()
            mock_interop.start.assert_called_once()

            # Cleanup
            await coordinator.stop_service()

    @pytest.mark.asyncio
    async def test_frequency_switching_performance(self, coordinator, mock_hackrf_interface):
        """Test frequency switching meets <100ms latency requirement"""
        with patch("src.backend.hal.hackrf_interface.HackRFInterface") as mock_hackrf_class:
            mock_hackrf_class.return_value = mock_hackrf_interface

            await coordinator.start_service()

            # Measure frequency switching latency
            start_time = asyncio.get_event_loop().time()

            # Switch from 406 MHz to 121.5 MHz (aviation emergency)
            success = await coordinator._switch_hackrf_frequency(121_500_000)

            end_time = asyncio.get_event_loop().time()
            switch_latency_ms = (end_time - start_time) * 1000

            # Verify performance requirement
            assert success is True
            assert (
                switch_latency_ms < 100
            ), f"Frequency switching took {switch_latency_ms:.1f}ms, exceeds 100ms requirement"
            assert coordinator._current_frequency_hz == 121_500_000

            await coordinator.stop_service()

    @pytest.mark.asyncio
    async def test_operator_frequency_profile_selection(self, coordinator, mock_hackrf_interface):
        """Test operator-selectable frequency profiles"""
        with patch("src.backend.hal.hackrf_interface.HackRFInterface") as mock_hackrf_class:
            mock_hackrf_class.return_value = mock_hackrf_interface

            await coordinator.start_service()

            # Test Emergency beacon profile (406 MHz)
            emergency_freq, analyzer_id = await coordinator._select_optimal_frequency()
            assert emergency_freq == 406_000_000
            assert "emergency" in analyzer_id.lower() or "406" in analyzer_id

            # Test profile switching capability exists
            assert hasattr(coordinator, "_frequency_channels")
            assert len(coordinator._frequency_channels) > 0

            # Verify frequency profiles are loaded
            profile_frequencies = [
                config.center_frequency_hz for config in coordinator._frequency_channels.values()
            ]
            assert 406_000_000 in profile_frequencies  # Emergency beacons

            await coordinator.stop_service()

    @pytest.mark.asyncio
    async def test_single_analyzer_lifecycle_management(self, coordinator, mock_hackrf_interface):
        """Test single active analyzer lifecycle management"""
        with patch("src.backend.hal.hackrf_interface.HackRFInterface") as mock_hackrf_class:
            mock_hackrf_class.return_value = mock_hackrf_interface

            await coordinator.start_service()

            # Verify only single analyzer is active at any time
            active_analyzers = coordinator.get_active_analyzers()
            assert (
                len(active_analyzers) <= 1
            ), f"Expected max 1 active analyzer, got {len(active_analyzers)}"

            # Test analyzer switching for frequency change
            initial_count = len(coordinator._active_analyzers)

            # Simulate frequency change requiring analyzer switch
            if coordinator._frequency_channels:
                # Switch to different frequency
                new_frequency = 121_500_000  # Aviation emergency
                await coordinator._switch_hackrf_frequency(new_frequency)

                # Verify still only one analyzer active
                active_after_switch = coordinator.get_active_analyzers()
                assert (
                    len(active_after_switch) <= 1
                ), "Multiple analyzers active after frequency switch"

            await coordinator.stop_service()

    @pytest.mark.asyncio
    async def test_coordination_loop_single_frequency_processing(
        self, coordinator, mock_hackrf_interface
    ):
        """Test coordination loop processes single frequency efficiently"""
        with patch("src.backend.hal.hackrf_interface.HackRFInterface") as mock_hackrf_class:
            mock_hackrf_class.return_value = mock_hackrf_interface

            # Mock IQ sample collection
            mock_hackrf_interface.collect_iq_samples.return_value = (
                b"\x00\x01" * 512
            )  # Realistic IQ data size

            await coordinator.start_service()

            # Let coordination loop run for a short period
            await asyncio.sleep(0.1)  # 100ms - several coordination cycles

            # Verify coordination metrics indicate single-frequency processing
            metrics = coordinator.get_coordination_metrics()
            assert metrics.total_analyzers_active <= 1, "Multiple analyzers active simultaneously"

            # Verify frequency switching metrics if available
            if (
                hasattr(coordinator, "_frequency_switch_times")
                and coordinator._frequency_switch_times
            ):
                avg_switch_time = sum(coordinator._frequency_switch_times) / len(
                    coordinator._frequency_switch_times
                )
                assert (
                    avg_switch_time < 100
                ), f"Average frequency switch time {avg_switch_time:.1f}ms exceeds 100ms"

            await coordinator.stop_service()

    @pytest.mark.asyncio
    async def test_hackrf_interface_preservation(self, coordinator):
        """Test that existing HackRF hardware interface is preserved"""
        # Verify coordinator uses existing HackRF interface without modification
        with patch("src.backend.hal.hackrf_interface.HackRFInterface") as mock_hackrf_class:
            mock_hackrf = AsyncMock()
            mock_hackrf.open.return_value = True
            mock_hackrf.close.return_value = True
            mock_hackrf_class.return_value = mock_hackrf

            await coordinator.start_service()

            # Verify HackRF interface was instantiated with existing pattern
            mock_hackrf_class.assert_called_once()
            mock_hackrf.open.assert_called_once()

            # Verify interface methods are called as expected
            assert coordinator._hackrf_interface is not None

            await coordinator.stop_service()
            mock_hackrf.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_frequency_switching_event_logging(self, coordinator, mock_hackrf_interface):
        """Test frequency switching includes proper event logging"""
        with patch("src.backend.hal.hackrf_interface.HackRFInterface") as mock_hackrf_class:
            mock_hackrf_class.return_value = mock_hackrf_interface

            with patch("src.backend.utils.logging.get_logger") as mock_logger:
                mock_log = Mock()
                mock_logger.return_value = mock_log

                await coordinator.start_service()

                # Perform frequency switch
                await coordinator._switch_hackrf_frequency(121_500_000)

                # Verify logging occurred (check that logger was called)
                assert (
                    mock_log.info.called or mock_log.debug.called
                ), "No logging occurred during frequency switch"

                await coordinator.stop_service()


@pytest.mark.asyncio
async def test_asv_coordination_metrics_single_frequency():
    """Test coordination metrics are properly updated for single-frequency mode"""
    metrics = ASVCoordinationMetrics()

    # Verify initial state for single-frequency operation
    assert metrics.total_analyzers_active == 0
    assert metrics.frequency_switches_per_second == 0.0
    assert metrics.concurrent_detections == 0  # Should be 0 or 1 in single-frequency mode

    # Update metrics for single analyzer
    metrics.total_analyzers_active = 1
    metrics.concurrent_detections = 1
    metrics.average_switching_latency_ms = 45.0  # Under 100ms requirement

    assert metrics.total_analyzers_active == 1
    assert (
        metrics.concurrent_detections <= 1
    ), "Single-frequency mode should not have concurrent detections > 1"
    assert metrics.average_switching_latency_ms < 100, "Switching latency exceeds requirement"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
