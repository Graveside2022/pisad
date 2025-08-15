"""
Test Circuit Breaker Implementation

BACKWARDS ANALYSIS:
- User Action: System processes signals with multiple registered callbacks
- Expected Result: System continues operating even if some callbacks fail
- Failure Impact: Without circuit breaker, cascading failures could crash system

REQUIREMENT TRACE:
- User Story: 4.9 - Task 8 (Implement Callback Circuit Breaker)

TEST VALUE: Prevents complete system failure from misbehaving callbacks
"""

import asyncio
import pytest
import numpy as np
from datetime import timedelta
from unittest.mock import Mock, MagicMock

from src.backend.utils.circuit_breaker import (
    CallbackCircuitBreaker,

pytestmark = pytest.mark.serial
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
    MultiCallbackCircuitBreaker
)


class TestCallbackCircuitBreaker:
    """Test basic circuit breaker functionality."""

    def test_circuit_starts_closed(self):
        """Circuit should start in closed state allowing calls."""
        breaker = CallbackCircuitBreaker()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_successful_call_resets_failure_count(self):
        """Successful calls should reset failure count."""
        breaker = CallbackCircuitBreaker()
        breaker.failure_count = 2  # Simulate previous failures

        # Successful callback
        callback = Mock(return_value="success")
        result = breaker.call_sync(callback, "test_value")

        assert result == "success"
        assert breaker.failure_count == 0
        assert breaker.state == CircuitState.CLOSED

    def test_circuit_opens_after_threshold_failures(self):
        """Circuit should open after reaching failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CallbackCircuitBreaker(config=config)

        # Failing callback
        failing_callback = Mock(side_effect=Exception("Test failure"))

        # First 3 failures should be allowed but raise exception
        for i in range(3):
            with pytest.raises(Exception, match="Test failure"):
                breaker.call_sync(failing_callback, f"value_{i}")

        # Circuit should now be open
        assert breaker.state == CircuitState.OPEN
        assert breaker.failure_count == 3

        # Next call should be rejected immediately
        with pytest.raises(CircuitBreakerError, match="Circuit breaker is OPEN"):
            breaker.call_sync(failing_callback, "blocked_value")

        # Callback should not have been called
        assert failing_callback.call_count == 3  # Only the first 3 calls

    def test_circuit_half_open_after_timeout(self):
        """Circuit should enter half-open state after timeout."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            timeout=timedelta(milliseconds=100)
        )
        breaker = CallbackCircuitBreaker(config=config)

        # Open the circuit
        failing_callback = Mock(side_effect=Exception("Fail"))
        with pytest.raises(Exception):
            breaker.call_sync(failing_callback, "value")

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        import time
        time.sleep(0.15)  # 150ms > 100ms timeout

        # Next call should transition to half-open and attempt
        success_callback = Mock(return_value="success")
        result = breaker.call_sync(success_callback, "test")

        assert result == "success"
        # After one success in half-open, still need more to close
        assert breaker.state == CircuitState.HALF_OPEN

    def test_circuit_closes_after_success_threshold(self):
        """Circuit should close after success threshold in half-open."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=2,
            timeout=timedelta(milliseconds=100)
        )
        breaker = CallbackCircuitBreaker(config=config)

        # Open the circuit
        with pytest.raises(Exception):
            breaker.call_sync(Mock(side_effect=Exception()), "value")

        # Wait for timeout and enter half-open
        import time
        time.sleep(0.15)

        success_callback = Mock(return_value="ok")

        # First success in half-open
        breaker.call_sync(success_callback, "value1")
        assert breaker.state == CircuitState.HALF_OPEN

        # Second success should close circuit
        breaker.call_sync(success_callback, "value2")
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_circuit_reopens_on_failure_in_half_open(self):
        """Single failure in half-open should reopen circuit."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            timeout=timedelta(milliseconds=100)
        )
        breaker = CallbackCircuitBreaker(config=config)

        # Open the circuit
        with pytest.raises(Exception):
            breaker.call_sync(Mock(side_effect=Exception()), "value")

        # Wait and enter half-open
        import time
        time.sleep(0.15)

        # Failure in half-open should reopen
        with pytest.raises(Exception):
            breaker.call_sync(Mock(side_effect=Exception("Fail again")), "value")

        assert breaker.state == CircuitState.OPEN

    def test_manual_reset(self):
        """Manual reset should close circuit regardless of state."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CallbackCircuitBreaker(config=config)

        # Open the circuit
        with pytest.raises(Exception):
            breaker.call_sync(Mock(side_effect=Exception()), "value")

        assert breaker.state == CircuitState.OPEN

        # Manual reset
        breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
        assert breaker.last_failure_time is None

    @pytest.mark.asyncio
    async def test_async_callback_support(self):
        """Circuit breaker should work with async callbacks."""
        breaker = CallbackCircuitBreaker()

        async def async_callback(value):
            await asyncio.sleep(0.01)
            return f"processed_{value}"

        result = await breaker.call_async(async_callback, "test")
        assert result == "processed_test"

    @pytest.mark.asyncio
    async def test_async_circuit_opens_on_failures(self):
        """Async circuit should open after threshold failures."""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CallbackCircuitBreaker(config=config)

        async def failing_callback(value):
            await asyncio.sleep(0.01)
            raise ValueError(f"Failed on {value}")

        # Trigger failures
        for i in range(2):
            with pytest.raises(ValueError):
                await breaker.call_async(failing_callback, i)

        assert breaker.state == CircuitState.OPEN

        # Should reject without calling
        with pytest.raises(CircuitBreakerError):
            await breaker.call_async(failing_callback, "blocked")


class TestMultiCallbackCircuitBreaker:
    """Test circuit breaker manager for multiple callbacks."""

    def test_independent_breakers_per_callback(self):
        """Each callback should have independent circuit breaker."""
        manager = MultiCallbackCircuitBreaker()

        # Register two callbacks, one failing and one succeeding
        failing_cb = Mock(side_effect=Exception("Fail"))
        success_cb = Mock(return_value="success")

        # Fail callback1 three times to open its circuit
        for i in range(3):
            with pytest.raises(Exception):
                manager.call_sync("callback1", failing_cb, i)

        # callback1 circuit should be open
        with pytest.raises(CircuitBreakerError):
            manager.call_sync("callback1", failing_cb, "blocked")

        # callback2 should still work
        result = manager.call_sync("callback2", success_cb, "value")
        assert result == "success"

    def test_get_all_states(self):
        """Should return state of all circuit breakers."""
        manager = MultiCallbackCircuitBreaker()

        # Create some breakers
        manager.call_sync("cb1", Mock(return_value="ok"), "v1")
        manager.call_sync("cb2", Mock(return_value="ok"), "v2")

        states = manager.get_all_states()

        assert "cb1" in states
        assert "cb2" in states
        assert states["cb1"]["state"] == "closed"
        assert states["cb2"]["state"] == "closed"

    def test_reset_all(self):
        """Reset all should reset every circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=1)
        manager = MultiCallbackCircuitBreaker(default_config=config)

        # Open multiple circuits
        for name in ["cb1", "cb2", "cb3"]:
            with pytest.raises(Exception):
                manager.call_sync(name, Mock(side_effect=Exception()), "v")

        # All should be open
        states = manager.get_all_states()
        for name in ["cb1", "cb2", "cb3"]:
            assert states[name]["state"] == "open"

        # Reset all
        manager.reset_all()

        # All should be closed
        states = manager.get_all_states()
        for name in ["cb1", "cb2", "cb3"]:
            assert states[name]["state"] == "closed"

    @pytest.mark.asyncio
    async def test_async_callback_manager(self):
        """Manager should handle async callbacks."""
        manager = MultiCallbackCircuitBreaker()

        async def async_cb(value):
            await asyncio.sleep(0.01)
            return f"async_{value}"

        result = await manager.call_async("async_cb", async_cb, "test")
        assert result == "async_test"


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration with signal processor."""

    @pytest.mark.asyncio
    async def test_signal_processor_with_circuit_breaker(self):
        """Verify signal processor uses circuit breaker for callbacks."""
        from src.backend.services.signal_processor import SignalProcessor

        processor = SignalProcessor()

        # Track callback invocations
        fail_invocations = []
        success_invocations = []

        # Add a failing callback
        def failing_callback(rssi):
            fail_invocations.append(rssi)
            raise ValueError("Callback failure")

        # Add a successful callback
        def success_callback(rssi):
            success_invocations.append(rssi)

        processor.add_rssi_callback(failing_callback)
        processor.add_rssi_callback(success_callback)

        # Process some samples
        samples = np.random.randn(1024) + 1j * np.random.randn(1024)

        # Process multiple times
        for i in range(6):
            await processor.process_iq(samples)
            await asyncio.sleep(0.01)

        # Success callback should have received all 6 values
        assert len(success_invocations) == 6

        # Failing callback should have been called only 3 times before circuit opened
        # Circuit breaker config has failure_threshold=3
        assert len(fail_invocations) == 3

        # Check circuit breaker state
        states = processor.get_circuit_breaker_states()

        # Failing callback circuit should be open after 3 failures
        failing_cb_state = None
        for name, state in states.items():
            if "failing_callback" in name:
                failing_cb_state = state
                break

        assert failing_cb_state is not None
        assert failing_cb_state["state"] == "open"
        assert failing_cb_state["failure_count"] == 3

    def test_circuit_breaker_metrics(self):
        """Circuit breaker should provide useful metrics."""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CallbackCircuitBreaker(config=config, name="TestBreaker")

        # Get initial state
        state = breaker.get_state()
        assert state["name"] == "TestBreaker"
        assert state["state"] == "closed"
        assert state["failure_count"] == 0
        assert state["config"]["failure_threshold"] == 2

        # Trigger a failure
        with pytest.raises(Exception):
            breaker.call_sync(Mock(side_effect=Exception()), "v")

        state = breaker.get_state()
        assert state["failure_count"] == 1
        assert state["last_failure"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
