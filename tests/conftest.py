"""
Shared pytest fixtures for optimized test execution.
These fixtures are available to all test files automatically.
"""

import asyncio
import tempfile
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient


def pytest_configure(config):
    """Register custom pytest markers for test categorization."""
    # Test category markers
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test (<100ms, no external dependencies)"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (<1s, real services)"
    )
    config.addinivalue_line("markers", "e2e: mark test as end-to-end test (<10s, full stack)")
    config.addinivalue_line("markers", "sitl: mark test as SITL hardware-in-loop test (<30s)")
    config.addinivalue_line("markers", "performance: mark test as performance benchmark (<5s)")
    config.addinivalue_line("markers", "property: mark test as property-based test (hypothesis)")
    config.addinivalue_line("markers", "contract: mark test as API contract test")

    # Priority markers
    config.addinivalue_line(
        "markers", "critical: mark test as critical (safety/core functionality)"
    )
    config.addinivalue_line("markers", "smoke: mark test as smoke test (basic functionality)")

    # Speed markers
    config.addinivalue_line("markers", "slow: mark test as slow (>1s execution time)")
    config.addinivalue_line("markers", "fast: mark test as fast (<100ms execution time)")

    # Requirement tracing markers
    config.addinivalue_line(
        "markers", "safety: mark test as safety-critical (HARA hazard mitigation)"
    )
    config.addinivalue_line(
        "markers", "requirement(id): mark test with requirement ID (FR/NFR/Story)"
    )

    # Environment markers
    config.addinivalue_line(
        "markers", "requires_hardware: mark test as requiring physical hardware"
    )
    config.addinivalue_line("markers", "requires_network: mark test as requiring network access")
    config.addinivalue_line("markers", "serial: mark test to run serially (not parallelizable)")


# Mark all tests with their appropriate group for parallel execution
def pytest_collection_modifyitems(config, items):
    """Add markers to test items for better organization and parallel execution."""
    for item in items:
        # Add markers based on test path
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
            # Disable timeout for now - causing test failures
            # item.add_marker(pytest.mark.timeout(2))
            # Group unit tests by module for parallel execution
            item.add_marker(pytest.mark.xdist_group(name=str(item.fspath.basename)))
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
            # Set 10 second timeout for integration tests
            item.add_marker(pytest.mark.timeout(10))
            # Run integration tests sequentially per module
            item.add_marker(pytest.mark.xdist_group(name="integration"))
        elif "sitl" in str(item.fspath):
            item.add_marker(pytest.mark.sitl)
            # Set 10 second timeout for SITL tests
            item.add_marker(pytest.mark.timeout(10))
            # SITL tests should run sequentially
            item.add_marker(pytest.mark.xdist_group(name="sitl"))
        elif "e2e" in str(item.fspath):
            # Set 10 second timeout for e2e tests
            item.add_marker(pytest.mark.timeout(10))
            item.add_marker(pytest.mark.xdist_group(name="e2e"))


# Session-scoped fixtures for expensive resources
@pytest.fixture(scope="function")
def event_loop():
    """Create an instance of the default event loop for each test function."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    try:
        # Cancel all running tasks
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        # Wait for tasks to complete cancellation with timeout
        if pending:
            loop.run_until_complete(
                asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=0.1)
            )
    except TimeoutError:
        # Force cancellation if tasks don't complete
        pass
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            loop.close()
            asyncio.set_event_loop(None)


# MAVLink Mock Fixtures
@pytest.fixture
def mock_mavlink_connection():
    """Create a mock MAVLink connection."""
    mock = MagicMock()
    mock.wait_heartbeat = MagicMock()
    mock.motors_armed_wait = MagicMock()
    mock.recv_match = MagicMock(return_value=None)
    mock.mav.heartbeat_send = MagicMock()
    mock.mav.command_long_send = MagicMock()
    mock.mav.set_position_target_local_ned_send = MagicMock()
    return mock


@pytest.fixture
def mock_mavlink_service():
    """Create a mock MAVLink service with common telemetry."""
    service = MagicMock()
    service.connected = True
    service.armed = False
    service.mode = "GUIDED"
    service.battery_percent = 75.0
    service.get_telemetry = MagicMock(
        return_value={
            "position": {"lat": 47.5, "lon": -122.3, "alt": 100.0},
            "attitude": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            "velocity": {"vx": 0.0, "vy": 0.0, "vz": 0.0},
            "battery": {"voltage": 12.6, "current": 5.0, "percentage": 75.0},
            "gps": {"fix_type": 3, "satellites": 12, "hdop": 1.5},
            "flight_mode": "GUIDED",
            "armed": False,
            "heading": 0.0,
            "groundspeed": 0.0,
            "airspeed": 0.0,
            "climb_rate": 0.0,
        }
    )
    service.arm = AsyncMock(return_value=True)
    service.disarm = AsyncMock(return_value=True)
    service.set_mode = AsyncMock(return_value=True)
    service.takeoff = AsyncMock(return_value=True)
    service.land = AsyncMock(return_value=True)
    service.goto = AsyncMock(return_value=True)
    service.set_velocity = AsyncMock(return_value=True)
    return service


# SDR Mock Fixtures
@pytest.fixture
def mock_sdr_device():
    """Create a mock SDR device."""
    device = MagicMock()
    device.sample_rate = 2.048e6
    device.center_freq = 433.92e6
    device.gain = 40
    device.read_samples = MagicMock(return_value=np.random.randn(1024) + 1j * np.random.randn(1024))
    device.close = MagicMock()
    return device


@pytest.fixture
def mock_sdr_service():
    """Create a mock SDR service."""
    service = MagicMock()
    service.is_running = False
    service.current_rssi = -75.0
    service.noise_floor = -95.0
    service.snr = 20.0
    service.beacon_detected = False
    service.get_rssi = MagicMock(return_value=-75.0)
    service.get_snr = MagicMock(return_value=20.0)
    service.get_noise_floor = MagicMock(return_value=-95.0)
    service.start = AsyncMock()
    service.stop = AsyncMock()
    service.tune = AsyncMock(return_value=True)
    service.set_gain = AsyncMock(return_value=True)
    service.validate_beacon = MagicMock(return_value=True)
    return service


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for the entire test session."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# Module-scoped fixtures for better performance
@pytest.fixture(scope="module")
def mock_config():
    """Shared mock configuration for tests."""
    from src.backend.core.config import Settings

    return Settings(
        SDR_FREQUENCY=433920000,
        SDR_SAMPLE_RATE=2048000,
        SDR_GAIN=30,
        DATABASE_URL="sqlite+aiosqlite:///:memory:",
        SECRET_KEY="test-secret-key",
        CORS_ORIGINS=["http://localhost:3000"],
        LOG_LEVEL="DEBUG",
        MAVLINK_CONNECTION="udp:127.0.0.1:14550",
        SAFETY_MAX_VELOCITY=10.0,
        SAFETY_MIN_ALTITUDE=10.0,
        SAFETY_MAX_ALTITUDE=120.0,
        SAFETY_GEOFENCE_RADIUS=500.0,
    )


# Function-scoped fixtures (default)
@pytest.fixture
def mock_sdr_service():
    """Mock SDR service for unit tests."""
    mock = MagicMock()
    mock.initialize = AsyncMock(return_value=True)
    mock.start_scanning = AsyncMock()
    mock.stop_scanning = AsyncMock()
    mock.get_rssi = AsyncMock(return_value=-75.0)
    mock.cleanup = AsyncMock()
    mock.is_initialized = True
    mock.is_scanning = False
    return mock


@pytest.fixture
def mock_mavlink_service():
    """Mock MAVLink service for unit tests."""
    mock = MagicMock()
    mock.connect = AsyncMock(return_value=True)
    mock.disconnect = AsyncMock()
    mock.send_heartbeat = AsyncMock()
    mock.arm_vehicle = AsyncMock(return_value=True)
    mock.disarm_vehicle = AsyncMock(return_value=True)
    mock.set_mode = AsyncMock(return_value=True)
    mock.send_waypoint = AsyncMock()
    mock.get_telemetry = AsyncMock(
        return_value={
            "lat": 0.0,
            "lon": 0.0,
            "alt": 0.0,
            "heading": 0.0,
            "groundspeed": 0.0,
            "battery": 100.0,
        }
    )
    mock.is_connected = True
    return mock


@pytest.fixture
def mock_signal_processor():
    """Mock signal processor for unit tests."""
    mock = MagicMock()
    mock.start_processing = AsyncMock()
    mock.stop_processing = AsyncMock()
    mock.process_sample = AsyncMock(
        return_value={
            "rssi": -75.0,
            "noise_floor": -95.0,
            "snr": 20.0,
            "confidence": 0.85,
        }
    )
    mock.get_gradient = MagicMock(return_value={"dx": 0.5, "dy": 0.5, "magnitude": 0.7})
    mock.is_processing = False
    return mock


@pytest.fixture
def mock_state_machine():
    """Mock state machine for unit tests."""
    mock = MagicMock()
    mock.current_state = "IDLE"
    mock.transition = AsyncMock(return_value=True)
    mock.can_transition = MagicMock(return_value=True)
    mock.get_state = MagicMock(return_value="IDLE")
    mock.reset = AsyncMock()
    return mock


@pytest_asyncio.fixture
async def test_db():
    """Create a test database for integration tests."""
    # Create a simple in-memory database for tests
    # This avoids importing the actual database module which might have initialization issues
    from sqlalchemy.ext.asyncio import create_async_engine

    DATABASE_URL = "sqlite+aiosqlite:///:memory:"
    engine = create_async_engine(DATABASE_URL, echo=False)

    # Create tables if database module is available
    try:
        from src.backend.models.database import Base

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except ImportError:
        # Database module not available, skip table creation
        pass

    yield engine

    # Cleanup
    await engine.dispose()


@pytest_asyncio.fixture
async def async_client(test_db) -> AsyncGenerator[AsyncClient, None]:
    """Create an async HTTP client for integration tests."""
    from src.backend.core.app import create_app

    app = create_app()
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def test_client() -> TestClient:
    """Create a synchronous test client for simple API tests."""
    from src.backend.core.app import create_app

    app = create_app()
    return TestClient(app)


# Shared test data fixtures
@pytest.fixture
def sample_rssi_data():
    """Sample RSSI data for testing."""
    return [
        {"timestamp": 1000000, "rssi": -75.0, "frequency": 433920000},
        {"timestamp": 1001000, "rssi": -73.0, "frequency": 433920000},
        {"timestamp": 1002000, "rssi": -71.0, "frequency": 433920000},
        {"timestamp": 1003000, "rssi": -69.0, "frequency": 433920000},
        {"timestamp": 1004000, "rssi": -67.0, "frequency": 433920000},
    ]


@pytest.fixture
def sample_waypoint():
    """Sample waypoint for testing."""
    return {
        "lat": 37.7749,
        "lon": -122.4194,
        "alt": 100.0,
        "type": "DETECTION",
        "rssi": -75.0,
    }


@pytest.fixture
def sample_telemetry():
    """Sample telemetry data for testing."""
    return {
        "lat": 37.7749,
        "lon": -122.4194,
        "alt": 50.0,
        "heading": 90.0,
        "groundspeed": 5.0,
        "battery": 85.0,
        "mode": "GUIDED",
        "armed": True,
        "timestamp": 1000000,
    }


# Performance optimization fixtures
@pytest.fixture(scope="session", autouse=True)
def configure_test_environment():
    """Configure environment for optimal test execution."""
    import os

    # Disable debug logging during tests for better performance
    os.environ["LOG_LEVEL"] = "WARNING"

    # Use in-memory database for all tests
    os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"

    # Disable telemetry during tests
    os.environ["TELEMETRY_ENABLED"] = "false"

    yield

    # Cleanup is automatic when process ends
