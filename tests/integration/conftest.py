"""
Integration test specific configuration and fixtures.
Integration tests can use real services but should complete in <1s.
"""

import asyncio
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
async def integration_db():
    """Create a real test database for integration testing."""
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker

    # Use a real SQLite file for integration tests
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    DATABASE_URL = f"sqlite+aiosqlite:///{db_path}"
    engine = create_async_engine(DATABASE_URL, echo=False)

    # Create tables
    try:
        from src.backend.models.database import Base

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except ImportError:
        pass

    # Create session factory
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    yield async_session

    # Cleanup
    await engine.dispose()
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
async def integration_app():
    """Create FastAPI app with real services for integration testing."""
    from src.backend.core.app import create_app
    from src.backend.core.config import get_settings

    # Override settings for integration tests
    settings = get_settings()
    settings.DATABASE_URL = "sqlite+aiosqlite:///:memory:"
    settings.TESTING = True

    app = create_app()
    return app


@pytest.fixture
def integration_timeout():
    """Enforce 1 second timeout for integration tests."""
    return 1.0


@pytest.fixture
async def mock_mavlink_sitl():
    """Mock SITL connection for integration testing."""

    class MockSITL:
        def __init__(self):
            self.connected = False
            self.mode = "GUIDED"
            self.armed = False

        async def connect(self):
            await asyncio.sleep(0.01)  # Simulate connection time
            self.connected = True
            return True

        async def arm(self):
            if self.connected and self.mode == "GUIDED":
                self.armed = True
                return True
            return False

        async def disconnect(self):
            self.connected = False
            self.armed = False

    return MockSITL()


@pytest.fixture
def integration_benchmark(benchmark_timer):
    """Benchmark fixture for integration tests with 1s timeout."""

    def _benchmark(func, *args, **kwargs):
        with benchmark_timer() as timer:
            result = func(*args, **kwargs)
        assert timer.elapsed < 1.0, f"Integration test took {timer.elapsed:.3f}s (limit: 1s)"
        return result

    return _benchmark
