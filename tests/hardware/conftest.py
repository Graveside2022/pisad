"""Hardware Testing Configuration and Fixtures.

This module provides pytest configuration for hardware testing with support for:
- Real hardware tests (requires physical devices)
- Mock hardware tests (for CI/CD)
- Auto-detection of hardware availability
- Test skipping based on environment
"""

import os
import sys
from pathlib import Path
from typing import Generator, Optional, Any
from unittest.mock import MagicMock, Mock, patch

import pytest
from pytest import FixtureRequest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from backend.hal.hackrf_interface import HackRFInterface
from backend.hal.mavlink_interface import MAVLinkInterface
from backend.hal.mock_hackrf import MockHackRF
from backend.services.hardware_detector import HardwareDetector
from backend.services.performance_monitor import PerformanceMonitor


# Test markers
def pytest_configure(config: Any) -> None:
    """Register custom markers for hardware testing."""
    config.addinivalue_line(
        "markers", "hardware: mark test as requiring real hardware (skip in CI/CD)"
    )
    config.addinivalue_line(
        "markers", "mock_hardware: mark test as using mock hardware (run in CI/CD)"
    )
    config.addinivalue_line(
        "markers", "sdr: mark test as requiring SDR hardware"
    )
    config.addinivalue_line(
        "markers", "mavlink: mark test as requiring MAVLink/flight controller"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )


# Environment detection
def is_ci_environment() -> bool:
    """Check if running in CI/CD environment."""
    return any([
        os.environ.get("CI"),
        os.environ.get("GITHUB_ACTIONS"),
        os.environ.get("JENKINS_HOME"),
        os.environ.get("GITLAB_CI"),
    ])


def is_hardware_available() -> bool:
    """Check if real hardware is available for testing."""
    # Skip hardware tests in CI/CD
    if is_ci_environment():
        return False
    
    # Check for explicit hardware test flag
    return os.environ.get("HARDWARE_TESTS", "").lower() in ("true", "1", "yes")


# Auto-skip configuration
def pytest_collection_modifyitems(config: Any, items: list) -> None:
    """Automatically skip hardware tests when hardware is not available."""
    skip_hardware = pytest.mark.skip(reason="Hardware tests disabled (set HARDWARE_TESTS=true to enable)")
    skip_mock = pytest.mark.skip(reason="Mock tests disabled in hardware mode")
    
    hardware_available = is_hardware_available()
    
    for item in items:
        # Skip real hardware tests if no hardware
        if "hardware" in item.keywords and not hardware_available:
            item.add_marker(skip_hardware)
        
        # Skip mock tests if running with real hardware
        if "mock_hardware" in item.keywords and hardware_available:
            item.add_marker(skip_mock)


# Hardware detection fixtures
@pytest.fixture
def hardware_available() -> bool:
    """Fixture to check if hardware is available."""
    return is_hardware_available()


@pytest.fixture
def skip_without_hardware(hardware_available: bool) -> None:
    """Skip test if hardware is not available."""
    if not hardware_available:
        pytest.skip("Hardware not available")


@pytest.fixture
def skip_with_hardware(hardware_available: bool) -> None:
    """Skip test if hardware is available (for mock-only tests)."""
    if hardware_available:
        pytest.skip("Test runs only with mock hardware")


# Mock hardware fixtures
@pytest.fixture
def mock_hackrf() -> Generator[MockHackRF, None, None]:
    """Provide a mock HackRF device for testing."""
    device = MockHackRF()
    device.open()
    yield device
    device.close()


@pytest.fixture
def mock_mavlink() -> Generator[Mock, None, None]:
    """Provide a mock MAVLink connection for testing."""
    mock_conn = Mock()
    mock_conn.recv_match = Mock(return_value=None)
    mock_conn.mav = Mock()
    mock_conn.target_system = 1
    mock_conn.target_component = 1
    
    # Mock telemetry data
    mock_conn.telemetry = {
        "position": {"lat": 37.7749, "lon": -122.4194, "alt": 100.0},
        "attitude": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        "battery": {"voltage": 22.2, "current": 10.0, "percentage": 75.0},
        "gps": {"fix_type": 3, "satellites": 12, "hdop": 0.9},
        "flight_mode": "GUIDED",
        "armed": False,
    }
    
    with patch("backend.hal.mavlink_interface.mavutil.mavlink_connection") as mock_mavutil:
        mock_mavutil.return_value = mock_conn
        yield mock_conn


@pytest.fixture
def mock_hardware_detector() -> Generator[HardwareDetector, None, None]:
    """Provide a mock hardware detector."""
    detector = HardwareDetector()
    
    # Mock detection results
    with patch.object(detector, '_check_sdr') as mock_sdr:
        with patch.object(detector, '_check_mavlink') as mock_mav:
            mock_sdr.return_value = True
            mock_mav.return_value = True
            detector.sdr_available = True
            detector.mavlink_available = True
            detector.hardware_status = {
                "sdr": {
                    "available": True,
                    "device": "MockHackRF",
                    "serial": "MOCK123",
                },
                "mavlink": {
                    "available": True,
                    "port": "/dev/ttyMOCK",
                    "baudrate": 115200,
                },
            }
            yield detector


@pytest.fixture
def mock_performance_monitor() -> Generator[PerformanceMonitor, None, None]:
    """Provide a mock performance monitor."""
    monitor = PerformanceMonitor()
    
    # Mock metrics
    monitor.metrics = {
        "system": {
            "cpu_percent": 25.0,
            "ram_percent": 45.0,
            "disk_percent": 60.0,
            "temperature": 42.0,
        },
        "sdr": {
            "sample_rate": 2048000,
            "drops": 0,
            "buffer_usage": 50.0,
        },
        "mavlink": {
            "latency_ms": 10.0,
            "packet_loss": 0.0,
            "messages_per_sec": 50.0,
        },
    }
    
    yield monitor


# Real hardware fixtures (only available when hardware present)
@pytest.fixture
def real_hackrf(skip_without_hardware: None) -> Generator[Optional[HackRFInterface], None, None]:
    """Provide a real HackRF device for testing (requires hardware)."""
    try:
        device = HackRFInterface()
        device.connect()
        yield device
        device.disconnect()
    except Exception as e:
        pytest.skip(f"HackRF not available: {e}")
        yield None


@pytest.fixture
def real_mavlink(skip_without_hardware: None) -> Generator[Optional[MAVLinkInterface], None, None]:
    """Provide a real MAVLink connection for testing (requires hardware)."""
    try:
        device = MAVLinkInterface()
        device.connect()
        yield device
        device.disconnect()
    except Exception as e:
        pytest.skip(f"MAVLink not available: {e}")
        yield None


# Integration test fixtures
@pytest.fixture
def test_config() -> dict:
    """Provide test configuration."""
    return {
        "sdr": {
            "frequency": 3200000000,  # 3.2 GHz
            "sample_rate": 20000000,  # 20 Msps
            "lna_gain": 16,
            "vga_gain": 20,
            "amp_enable": False,
        },
        "mavlink": {
            "port": "/dev/ttyACM0",
            "baudrate": 115200,
            "system_id": 255,
            "component_id": 190,
        },
        "performance": {
            "cpu_target": 30,
            "ram_limit_mb": 512,
            "sample_buffer_size": 65536,
        },
    }


@pytest.fixture
def performance_baseline() -> dict:
    """Provide performance baseline for comparison."""
    return {
        "latency": {
            "mode_change_ms": 100,
            "emergency_stop_ms": 500,
            "command_response_ms": 50,
        },
        "throughput": {
            "sdr_samples_per_sec": 20000000,
            "mavlink_messages_per_sec": 50,
        },
        "resource_usage": {
            "cpu_percent_max": 30,
            "ram_mb_max": 512,
        },
    }


# Test data fixtures
@pytest.fixture
def sample_iq_data() -> bytes:
    """Provide sample IQ data for testing."""
    import numpy as np
    
    # Generate complex sinusoid
    samples = 1024
    freq = 0.1  # Normalized frequency
    t = np.arange(samples)
    signal = np.exp(2j * np.pi * freq * t)
    
    # Add noise
    noise = (np.random.randn(samples) + 1j * np.random.randn(samples)) * 0.1
    signal += noise
    
    # Convert to interleaved float32
    iq = np.zeros(samples * 2, dtype=np.float32)
    iq[0::2] = signal.real
    iq[1::2] = signal.imag
    
    return iq.tobytes()


@pytest.fixture
def sample_telemetry() -> dict:
    """Provide sample telemetry data for testing."""
    return {
        "position": {
            "lat": 37.7749,
            "lon": -122.4194,
            "alt": 100.0,
            "relative_alt": 50.0,
        },
        "attitude": {
            "roll": 0.05,
            "pitch": -0.02,
            "yaw": 1.57,
        },
        "velocity": {
            "vx": 1.0,
            "vy": 0.5,
            "vz": -0.1,
        },
        "battery": {
            "voltage": 22.2,
            "current": 15.0,
            "percentage": 75.0,
        },
        "gps": {
            "fix_type": 3,
            "satellites": 12,
            "hdop": 0.9,
            "vdop": 1.1,
        },
        "flight_mode": "GUIDED",
        "armed": True,
        "timestamp": 1234567890.0,
    }


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_test_files(request: FixtureRequest) -> Generator[None, None, None]:
    """Clean up any test files created during testing."""
    yield
    
    # Clean up test files
    test_dirs = [
        Path("test_logs"),
        Path("test_data"),
        Path("test_configs"),
    ]
    
    for test_dir in test_dirs:
        if test_dir.exists():
            import shutil
            shutil.rmtree(test_dir)


# Coverage helpers
@pytest.fixture
def coverage_reporter() -> Generator[Any, None, None]:
    """Provide coverage reporting helper."""
    import coverage
    
    cov = coverage.Coverage()
    cov.start()
    
    yield cov
    
    cov.stop()
    cov.save()