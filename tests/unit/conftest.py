"""
Unit test specific configuration and fixtures.
Unit tests should be fast (<100ms) and have no external dependencies.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def fast_test_settings(monkeypatch):
    """Auto-apply settings for fast unit test execution."""
    # Disable any network calls
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("LOG_LEVEL", "ERROR")

    # Mock slow imports
    import sys

    sys.modules["cv2"] = MagicMock()
    sys.modules["mavlink"] = MagicMock()


@pytest.fixture
def mock_signal_data():
    """Generate mock IQ signal data for testing."""
    samples = 1024
    # Generate complex IQ samples with some signal
    noise = np.random.randn(samples) + 1j * np.random.randn(samples)
    signal = np.exp(1j * 2 * np.pi * 0.1 * np.arange(samples))
    return (noise * 0.1 + signal * 0.5).astype(np.complex64)


@pytest.fixture
def mock_state_transitions():
    """Valid state transitions for testing."""
    return {
        "IDLE": ["SEARCHING", "IDLE"],
        "SEARCHING": ["IDLE", "DETECTING", "SEARCHING"],
        "DETECTING": ["SEARCHING", "HOMING", "IDLE"],
        "HOMING": ["DETECTING", "HOLDING", "SEARCHING", "IDLE"],
        "HOLDING": ["HOMING", "SEARCHING", "IDLE"],
    }


@pytest.fixture
def unit_benchmark(benchmark_timer):
    """Benchmark fixture for unit tests with 100ms timeout."""

    def _benchmark(func, *args, **kwargs):
        with benchmark_timer() as timer:
            result = func(*args, **kwargs)
        assert timer.elapsed < 0.1, f"Unit test took {timer.elapsed:.3f}s (limit: 100ms)"
        return result

    return _benchmark
