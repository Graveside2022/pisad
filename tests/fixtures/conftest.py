"""
Shared pytest fixtures using factory-boy.

This module provides reusable fixtures that eliminate test setup
duplication and prevent circular imports.
"""

import asyncio
from collections.abc import AsyncGenerator

import numpy as np
import pytest

from tests.fixtures.factories import (
    ConfigProfileFactory,
    MockHardwareConfigFactory,
    MockMAVLinkFactory,
    MockSDRFactory,
    RSSIReadingFactory,
    SignalDetectionFactory,
    SystemStateFactory,
    create_test_signal_data,
    create_test_telemetry_sequence,
)
from tests.hardware.mock.mavlink_mock import MockMAVLinkInterface
from tests.hardware.mock.sdr_mock import MockSDRInterface

# ============================================================================
# Hardware Mock Fixtures
# ============================================================================


@pytest.fixture
def mock_sdr_config():
    """Provide a standard SDR configuration for testing."""
    return MockHardwareConfigFactory(
        device_id="test_hackrf", sample_rate=20e6, frequency=3.2e9, gain=30.0
    )


@pytest.fixture
async def mock_sdr(mock_sdr_config) -> AsyncGenerator[MockSDRInterface, None]:
    """Provide a connected mock SDR interface."""
    sdr = MockSDRFactory(config=mock_sdr_config)
    await sdr.connect()
    yield sdr
    await sdr.disconnect()


@pytest.fixture
async def mock_sdr_streaming(mock_sdr) -> AsyncGenerator[MockSDRInterface, None]:
    """Provide a mock SDR that is already streaming."""
    await mock_sdr.start_streaming()
    yield mock_sdr
    await mock_sdr.stop_streaming()


@pytest.fixture
def mock_mavlink_config():
    """Provide a standard MAVLink configuration for testing."""
    return MockHardwareConfigFactory(device_id="test_mavlink")


@pytest.fixture
async def mock_mavlink(mock_mavlink_config) -> AsyncGenerator[MockMAVLinkInterface, None]:
    """Provide a connected mock MAVLink interface."""
    mavlink = MockMAVLinkFactory(config=mock_mavlink_config)
    await mavlink.connect()
    yield mavlink
    await mavlink.disconnect()


@pytest.fixture
async def mock_mavlink_armed(mock_mavlink) -> AsyncGenerator[MockMAVLinkInterface, None]:
    """Provide a mock MAVLink interface with armed vehicle."""
    await mock_mavlink.arm()
    await mock_mavlink.set_mode("GUIDED")
    yield mock_mavlink
    await mock_mavlink.disarm()


# ============================================================================
# Data Fixtures
# ============================================================================


@pytest.fixture
def signal_detection_strong():
    """Provide a strong signal detection."""
    return SignalDetectionFactory.create(strong_signal=True)


@pytest.fixture
def signal_detection_weak():
    """Provide a weak signal detection."""
    return SignalDetectionFactory.create(weak_signal=True)


@pytest.fixture
def signal_detections_batch():
    """Provide a batch of signal detections."""
    return [SignalDetectionFactory.create(strong_signal=True) for _ in range(3)] + [
        SignalDetectionFactory.create(weak_signal=True) for _ in range(7)
    ]


@pytest.fixture
def config_profile_default():
    """Provide a default configuration profile."""
    return ConfigProfileFactory.create(default=True)


@pytest.fixture
def config_profile_wifi():
    """Provide a WiFi beacon configuration profile."""
    return ConfigProfileFactory.create(wifi_beacon=True)


@pytest.fixture
def config_profile_lora():
    """Provide a LoRa beacon configuration profile."""
    return ConfigProfileFactory.create(lora_beacon=True)


@pytest.fixture
def rssi_readings_sequence():
    """Provide a sequence of RSSI readings."""
    return [RSSIReadingFactory.create() for _ in range(100)]


@pytest.fixture
def system_state_ready():
    """Provide a system state ready for homing."""
    return SystemStateFactory.create(ready_for_homing=True)


@pytest.fixture
def system_state_homing():
    """Provide a system state with active homing."""
    return SystemStateFactory.create(homing_active=True)


@pytest.fixture
def system_state_emergency():
    """Provide a system state in emergency condition."""
    return SystemStateFactory.create(emergency=True)


# ============================================================================
# Test Data Generators
# ============================================================================


@pytest.fixture
def test_signal_data():
    """Provide realistic test signal data."""
    return create_test_signal_data(duration_seconds=60, sample_rate=10)


@pytest.fixture
def test_telemetry_sequence():
    """Provide a sequence of telemetry updates."""
    return create_test_telemetry_sequence(duration_seconds=60)


@pytest.fixture
def mock_iq_samples():
    """Provide mock IQ samples for signal processing tests."""
    # Generate complex samples with signal and noise
    num_samples = 1024
    t = np.arange(num_samples) / 2e6
    signal = 0.5 * np.exp(1j * 2 * np.pi * 100e3 * t)
    noise = 0.1 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    return (signal + noise).astype(np.complex64)


# ============================================================================
# Test Environment Fixtures
# ============================================================================


@pytest.fixture
def event_loop():
    """Provide an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def mock_hardware_suite(mock_sdr, mock_mavlink):
    """Provide a complete suite of mock hardware interfaces."""
    return {"sdr": mock_sdr, "mavlink": mock_mavlink}


@pytest.fixture
def fixture_dependency_graph():
    """
    Document the fixture dependency graph.

    This fixture returns a dictionary describing the relationships
    between fixtures to help developers understand the test infrastructure.
    """
    return {
        "hardware_mocks": {
            "mock_sdr": ["mock_sdr_config"],
            "mock_sdr_streaming": ["mock_sdr"],
            "mock_mavlink": ["mock_mavlink_config"],
            "mock_mavlink_armed": ["mock_mavlink"],
            "mock_hardware_suite": ["mock_sdr", "mock_mavlink"],
        },
        "data_fixtures": {
            "signal_detections_batch": ["SignalDetectionFactory"],
            "rssi_readings_sequence": ["RSSIReadingFactory"],
        },
        "independent": [
            "signal_detection_strong",
            "signal_detection_weak",
            "config_profile_default",
            "config_profile_wifi",
            "config_profile_lora",
            "system_state_ready",
            "system_state_homing",
            "system_state_emergency",
            "test_signal_data",
            "test_telemetry_sequence",
            "mock_iq_samples",
        ],
    }


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "e2e: mark test as an end-to-end test")
    config.addinivalue_line("markers", "sitl: mark test as a SITL test")
    config.addinivalue_line(
        "markers", "serial: mark test as requiring serial execution (no parallel)"
    )
    config.addinivalue_line("markers", "slow: mark test as slow running")
