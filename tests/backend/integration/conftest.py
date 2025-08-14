"""
Integration test specific fixtures and configuration.
"""

import asyncio
import os
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture(scope="module")
def integration_config():
    """Override configuration for integration tests."""
    # Create a mock config object that matches the structure used in the app
    config = MagicMock()
    
    # App config
    config.app.APP_NAME = "PISAD Test"
    config.app.APP_VERSION = "1.0.0"
    config.app.ENVIRONMENT = "test"
    
    # API config
    config.api.API_CORS_ENABLED = True
    config.api.API_CORS_ORIGINS = ["http://testserver"]
    config.api.API_KEY = "test-api-key"
    
    # Database config
    config.database.DATABASE_URL = "sqlite+aiosqlite:///:memory:"
    
    # Logging config
    config.logging.LOG_LEVEL = "INFO"
    
    # Safety config
    config.safety.SAFETY_MAX_VELOCITY = 15.0
    config.safety.SAFETY_MIN_ALTITUDE = 5.0
    config.safety.SAFETY_MAX_ALTITUDE = 150.0
    config.safety.SAFETY_GEOFENCE_RADIUS = 1000.0
    
    # SDR config
    config.sdr.SDR_ENABLED = False
    config.sdr.SDR_FREQUENCY = 433920000
    config.sdr.SDR_SAMPLE_RATE = 2048000
    config.sdr.SDR_GAIN = 30
    
    return config


@pytest_asyncio.fixture(scope="function")
async def test_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session with isolated transactions."""
    # Create a new in-memory database for each test
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        future=True,
    )

    # Create tables if database module is available
    try:
        from src.backend.models.database import Base
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except ImportError:
        pass  # Database module not available

    # Create session factory
    async_session_maker = sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Provide session
    async with async_session_maker() as session:
        yield session
        await session.rollback()

    # Cleanup
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def integration_app(integration_config):
    """Create application instance for integration tests."""
    with patch('src.backend.core.config.get_config', return_value=integration_config):
        from src.backend.core.app import create_app
        
        app = create_app()
        yield app


@pytest_asyncio.fixture(scope="function")
async def integration_client(integration_app) -> AsyncGenerator[AsyncClient, None]:
    """Create HTTP client for integration tests."""
    async with AsyncClient(
        app=integration_app,
        base_url="http://testserver",
        timeout=10.0,
    ) as client:
        yield client


@pytest.fixture(scope="function")
def mock_websocket_connection():
    """Mock WebSocket connection for integration tests."""
    mock = MagicMock()
    mock.accept = AsyncMock()
    mock.send_text = AsyncMock()
    mock.send_bytes = AsyncMock()
    mock.send_json = AsyncMock()
    mock.receive_text = AsyncMock(return_value='{"type": "ping"}')
    mock.receive_json = AsyncMock(return_value={"type": "ping"})
    mock.close = AsyncMock()
    return mock


@pytest.fixture(scope="module")
def integration_test_data():
    """Shared test data for integration tests."""
    return {
        "valid_config": {
            "sdr": {
                "frequency": 433920000,
                "sample_rate": 2048000,
                "gain": 30,
            },
            "homing": {
                "approach_speed": 5.0,
                "detection_threshold": -80.0,
                "search_radius": 100.0,
            },
            "safety": {
                "max_velocity": 10.0,
                "min_altitude": 10.0,
                "max_altitude": 100.0,
                "geofence_radius": 500.0,
            },
        },
        "valid_mission": {
            "name": "Test Mission",
            "search_area": {
                "center_lat": 37.7749,
                "center_lon": -122.4194,
                "radius": 500.0,
            },
            "beacon_frequency": 433920000,
            "search_pattern": "spiral",
        },
        "valid_telemetry": {
            "lat": 37.7749,
            "lon": -122.4194,
            "alt": 50.0,
            "heading": 90.0,
            "groundspeed": 5.0,
            "battery": 85.0,
            "mode": "GUIDED",
            "armed": True,
        },
    }


# Performance optimization for integration tests
@pytest.fixture(scope="session", autouse=True)
def optimize_integration_tests():
    """Optimize environment for integration test execution."""
    # Use in-memory SQLite for all integration tests
    os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"

    # Disable external services
    os.environ["SDR_ENABLED"] = "false"
    os.environ["MAVLINK_ENABLED"] = "false"
    os.environ["TELEMETRY_ENABLED"] = "false"

    # Reduce timeouts for faster test execution
    os.environ["API_TIMEOUT"] = "5"
    os.environ["WEBSOCKET_TIMEOUT"] = "5"

    yield

    # Cleanup is automatic