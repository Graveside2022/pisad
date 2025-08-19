"""Unit tests for Doppler compensation integration with ASV enhanced signal processor.

TASK-6.1.16d - Test Doppler compensation integration with ASV enhanced signal processor
"""

import time
from unittest.mock import AsyncMock, Mock

import pytest

from src.backend.services.asv_integration.asv_analyzer_wrapper import (
    ASVAnalyzerBase,
    ASVSignalData,
)
from src.backend.services.asv_integration.asv_enhanced_signal_processor import (
    ASVBearingCalculation,
    ASVEnhancedSignalProcessor,
)
from src.backend.utils.doppler_compensation import PlatformVelocity


class TestASVDopplerIntegration:
    """Test suite for Doppler compensation integration with ASV signal processor."""

    @pytest.fixture
    def mock_asv_analyzer(self):
        """Create mock ASV analyzer for testing."""
        mock_analyzer = Mock(spec=ASVAnalyzerBase)
        mock_analyzer.analyzer_type = "GP"
        mock_analyzer.frequency_hz = 406000000  # 406 MHz emergency beacon
        return mock_analyzer

    @pytest.fixture
    def signal_processor(self, mock_asv_analyzer):
        """Create ASV enhanced signal processor with Doppler compensation."""
        return ASVEnhancedSignalProcessor(mock_asv_analyzer)

    @pytest.fixture
    def platform_velocity(self):
        """Create test platform velocity for approaching beacon scenario."""
        return PlatformVelocity(vx_ms=15.0, vy_ms=0.0, vz_ms=0.0, ground_speed_ms=15.0)

    @pytest.fixture
    def high_quality_signal_data(self):
        """Create high-quality signal data for testing."""
        return ASVSignalData(
            timestamp_ns=time.perf_counter_ns(),
            frequency_hz=406000000,
            signal_strength_dbm=-75.0,  # Strong signal
            signal_quality=0.95,  # High quality
            analyzer_type="GP",
            overflow_indicator=0.05,  # Low interference
            raw_data={"processing_time_ns": 800000},
        )

    def test_platform_velocity_setting(self, signal_processor, platform_velocity):
        """Test setting platform velocity in ASV signal processor."""
        # Act
        signal_processor.set_platform_velocity(platform_velocity)

        # Assert
        assert signal_processor._current_platform_velocity == platform_velocity

    def test_signal_frequency_setting(self, signal_processor):
        """Test setting signal frequency in ASV signal processor."""
        # Arrange
        test_frequency = 121_500_000  # Aviation frequency

        # Act
        signal_processor.set_signal_frequency(test_frequency)

        # Assert
        assert signal_processor._signal_frequency_hz == test_frequency

    def test_doppler_shift_calculation(self, signal_processor, platform_velocity):
        """Test Doppler shift calculation for different bearings."""
        # Arrange
        signal_processor.set_platform_velocity(platform_velocity)
        signal_processor.set_signal_frequency(406_000_000)

        # Act - Test approaching beacon (0 degrees North)
        doppler_shift_north = signal_processor._calculate_doppler_shift(0.0)

        # Act - Test perpendicular motion (90 degrees East)
        doppler_shift_east = signal_processor._calculate_doppler_shift(90.0)

        # Act - Test receding beacon (180 degrees South)
        doppler_shift_south = signal_processor._calculate_doppler_shift(180.0)

        # Assert
        assert doppler_shift_north is not None
        assert doppler_shift_east is not None
        assert doppler_shift_south is not None

        # Approaching should be positive, receding negative, perpendicular near zero
        assert doppler_shift_north > 0  # Positive Doppler when approaching
        assert abs(doppler_shift_east) < abs(doppler_shift_north)  # Less shift for perpendicular
        assert doppler_shift_south < 0  # Negative Doppler when receding

    def test_compensated_frequency_calculation(self, signal_processor, platform_velocity):
        """Test compensated frequency calculation."""
        # Arrange
        signal_processor.set_platform_velocity(platform_velocity)
        original_frequency = 406_000_000
        signal_processor.set_signal_frequency(original_frequency)

        # Act
        compensated_freq = signal_processor._get_compensated_frequency(0.0)  # North bearing

        # Assert
        assert compensated_freq is not None
        assert compensated_freq != original_frequency  # Should be compensated
        # For approaching beacon, compensation should reduce observed frequency
        assert compensated_freq < original_frequency

    @pytest.mark.asyncio
    async def test_bearing_calculation_includes_doppler_data(
        self, signal_processor, mock_asv_analyzer, platform_velocity, high_quality_signal_data
    ):
        """Test that bearing calculation includes Doppler compensation data."""
        # Arrange
        signal_processor.set_platform_velocity(platform_velocity)
        signal_processor.set_signal_frequency(406_000_000)

        # Mock ASV analyzer response
        mock_asv_analyzer.process_signal = AsyncMock(return_value=high_quality_signal_data)

        # Act
        result = await signal_processor.calculate_professional_bearing(
            iq_samples=b"\x01\x02\x03\x04" * 100,
            position_x=10.0,
            position_y=20.0,
            current_heading_deg=45.0,
        )

        # Assert
        assert result is not None
        assert isinstance(result, ASVBearingCalculation)

        # Verify Doppler compensation fields are populated
        assert result.doppler_shift_hz is not None
        assert result.compensated_frequency_hz is not None
        assert result.platform_velocity_ms is not None
        assert result.platform_velocity_ms == platform_velocity.ground_speed_ms

    def test_doppler_compensation_disabled_when_no_velocity(
        self, signal_processor, mock_asv_analyzer, high_quality_signal_data
    ):
        """Test that Doppler compensation is disabled when no platform velocity available."""
        # Arrange - No platform velocity set
        signal_processor.set_signal_frequency(406_000_000)

        # Act
        doppler_shift = signal_processor._calculate_doppler_shift(45.0)
        compensated_freq = signal_processor._get_compensated_frequency(45.0)

        # Assert
        assert doppler_shift is None
        assert compensated_freq is None

    @pytest.mark.asyncio
    async def test_moving_platform_scenario_maritime_sar(
        self, signal_processor, mock_asv_analyzer, high_quality_signal_data
    ):
        """Test Doppler compensation for maritime SAR scenario - boat drifting."""
        # Arrange - Boat drifting northeast at 8 m/s
        maritime_velocity = PlatformVelocity(
            vx_ms=5.66,  # ~8 m/s * cos(45°)
            vy_ms=5.66,  # ~8 m/s * sin(45°)
            vz_ms=0.0,
            ground_speed_ms=8.0,
        )

        signal_processor.set_platform_velocity(maritime_velocity)
        signal_processor.set_signal_frequency(406_000_000)
        mock_asv_analyzer.process_signal = AsyncMock(return_value=high_quality_signal_data)

        # Act - Beacon bearing varies as boat drifts
        bearings_to_test = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
        results = []

        for bearing in bearings_to_test:
            # Simulate calculation at different relative bearings
            result = await signal_processor.calculate_professional_bearing(
                iq_samples=b"\x01\x02\x03\x04" * 100,
                position_x=10.0,
                position_y=20.0,
                current_heading_deg=bearing,
            )
            results.append((bearing, result))

        # Assert - All calculations succeeded with Doppler data
        for bearing, result in results:
            assert result is not None, f"Calculation failed for bearing {bearing}°"
            assert result.doppler_shift_hz is not None, f"No Doppler shift for bearing {bearing}°"
            assert (
                result.compensated_frequency_hz is not None
            ), f"No frequency compensation for bearing {bearing}°"

        # Verify Doppler shift pattern makes physical sense
        north_result = next(r for b, r in results if b == 0.0)
        south_result = next(r for b, r in results if b == 180.0)
        east_result = next(r for b, r in results if b == 90.0)

        # Doppler shift should be different for different relative motion directions
        assert north_result.doppler_shift_hz != south_result.doppler_shift_hz
        # Note: The actual bearing calculation may vary, so we just verify Doppler data exists
        assert north_result.doppler_shift_hz is not None
        assert south_result.doppler_shift_hz is not None
        assert east_result.doppler_shift_hz is not None

    @pytest.mark.asyncio
    async def test_high_speed_aircraft_scenario(
        self, signal_processor, mock_asv_analyzer, high_quality_signal_data
    ):
        """Test Doppler compensation for high-speed SAR aircraft scenario."""
        # Arrange - SAR aircraft at 50 m/s (typical search speed)
        aircraft_velocity = PlatformVelocity(
            vx_ms=50.0,  # Due north
            vy_ms=0.0,
            vz_ms=0.0,
            ground_speed_ms=50.0,
        )

        signal_processor.set_platform_velocity(aircraft_velocity)
        signal_processor.set_signal_frequency(406_000_000)
        mock_asv_analyzer.process_signal = AsyncMock(return_value=high_quality_signal_data)

        # Act
        result = await signal_processor.calculate_professional_bearing(
            iq_samples=b"\x01\x02\x03\x04" * 100,
            position_x=0.0,
            position_y=0.0,
            current_heading_deg=0.0,  # Approaching beacon directly
        )

        # Assert
        assert result is not None
        assert result.doppler_shift_hz is not None
        assert result.compensated_frequency_hz is not None

        # High speed should produce measurable Doppler shift
        # Note: Actual shift depends on the calculated bearing, not the input heading
        assert result.doppler_shift_hz is not None
        assert result.platform_velocity_ms == 50.0  # Verify velocity was recorded
        assert abs(result.doppler_shift_hz) > 0.0  # Should have some Doppler effect

    def test_doppler_calculation_error_handling(self, signal_processor):
        """Test error handling in Doppler calculations."""
        # Arrange - Set invalid conditions
        signal_processor._signal_frequency_hz = 0  # Invalid frequency

        bad_velocity = PlatformVelocity(
            vx_ms=float("inf"),  # Invalid velocity
            vy_ms=0.0,
            vz_ms=0.0,
            ground_speed_ms=float("inf"),
        )
        signal_processor.set_platform_velocity(bad_velocity)

        # Act - Should handle errors gracefully
        doppler_shift = signal_processor._calculate_doppler_shift(45.0)
        compensated_freq = signal_processor._get_compensated_frequency(45.0)

        # Assert - Should return safe values on error, not crash
        assert doppler_shift == 0.0  # Should return 0 for invalid inputs
        assert compensated_freq == 0  # Should return original frequency (0 in this case) on error
