"""Unit tests for beacon signal validation."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.models.schemas import BeaconConfiguration
from backend.services.field_test_service import FieldTestService


@pytest.fixture
def field_test_service():
    """Create field test service with mocked dependencies."""
    test_logger = MagicMock()
    state_machine = MagicMock()
    mavlink_service = MagicMock()
    signal_processor = MagicMock()
    safety_manager = MagicMock()

    service = FieldTestService(
        test_logger=test_logger,
        state_machine=state_machine,
        mavlink_service=mavlink_service,
        signal_processor=signal_processor,
        safety_manager=safety_manager,
    )
    return service


@pytest.mark.asyncio
@patch('backend.services.sdr_service.SDRService', None)  # Force ImportError path
async def test_validate_beacon_signal_simplified(field_test_service):
    """Test beacon signal validation with simplified fallback."""
    beacon_config = BeaconConfiguration(
        frequency_hz=433_000_000,
        power_dbm=10.0,
    )

    # Test the actual validation without SDR (will use simplified fallback)
    results = await field_test_service.validate_beacon_signal(beacon_config)

    # Verify results structure
    assert "validation_passed" in results
    assert "frequency_match" in results
    assert "power_level_match" in results
    assert "modulation_match" in results

    # Simplified method should pass validation
    assert results["validation_passed"] is True
    assert results["frequency_match"] is True
    assert results["power_level_match"] is True
    assert results["modulation_match"] is True


@pytest.mark.asyncio
@patch('backend.services.sdr_service.SDRService', None)  # Force ImportError path
async def test_validate_beacon_signal_result_structure(field_test_service):
    """Test that beacon validation returns correct result structure."""
    beacon_config = BeaconConfiguration(
        frequency_hz=433_000_000,
        power_dbm=10.0,
        modulation="LoRa",
        spreading_factor=7,
        bandwidth_hz=125_000,
        coding_rate=5,
        pulse_rate_hz=1.0,
        pulse_duration_ms=100,
    )

    # Execute validation (will use simplified path)
    results = await field_test_service.validate_beacon_signal(beacon_config)

    # Check all expected fields are present
    expected_fields = [
        "validation_passed",
        "frequency_match",
        "power_level_match",
        "modulation_match",
        "measured_frequency_hz",
        "measured_power_dbm",
        "frequency_error_hz",
        "power_error_dbm",
        "spectrum_data",
    ]

    for field in expected_fields:
        assert field in results, f"Missing field: {field}"

    # Check field types
    assert isinstance(results["validation_passed"], bool)
    assert isinstance(results["frequency_match"], bool)
    assert isinstance(results["power_level_match"], bool)
    assert isinstance(results["modulation_match"], bool)
    assert isinstance(results["measured_frequency_hz"], float)
    assert isinstance(results["measured_power_dbm"], float)
    assert isinstance(results["frequency_error_hz"], float)
    assert isinstance(results["power_error_dbm"], float)
    assert isinstance(results["spectrum_data"], list)


@pytest.mark.asyncio
@patch('backend.services.sdr_service.SDRService', None)  # Force ImportError path
async def test_validate_beacon_config_variations(field_test_service):
    """Test validation with different beacon configurations."""
    # Test with minimal config
    minimal_config = BeaconConfiguration(
        frequency_hz=433_000_000,
        power_dbm=5.0,
    )
    results = await field_test_service.validate_beacon_signal(minimal_config)
    assert results["validation_passed"] is True

    # Test with high power config
    high_power_config = BeaconConfiguration(
        frequency_hz=433_000_000,
        power_dbm=20.0,
    )
    results = await field_test_service.validate_beacon_signal(high_power_config)
    assert results["validation_passed"] is True

    # Test with different frequency
    diff_freq_config = BeaconConfiguration(
        frequency_hz=915_000_000,
        power_dbm=10.0,
    )
    results = await field_test_service.validate_beacon_signal(diff_freq_config)
    assert results["validation_passed"] is True


@pytest.mark.asyncio
@patch('backend.services.sdr_service.SDRService', None)  # Force ImportError path
async def test_validate_beacon_with_full_config(field_test_service):
    """Test validation with complete beacon configuration."""
    full_config = BeaconConfiguration(
        frequency_hz=433_000_000,
        power_dbm=15.0,
        modulation="FSK",
        spreading_factor=12,
        bandwidth_hz=250_000,
        coding_rate=8,
        pulse_rate_hz=2.0,
        pulse_duration_ms=50,
    )

    results = await field_test_service.validate_beacon_signal(full_config)

    # All fields should be present
    assert results["validation_passed"] is True
    assert results["frequency_match"] is True
    assert results["power_level_match"] is True
    assert results["modulation_match"] is True

    # Check that measurements are present (even if using simplified method)
    assert "measured_frequency_hz" in results
    assert "measured_power_dbm" in results
