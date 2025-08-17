"""
Test Circuit Breaker Pattern Implementation
Tests fault injection and recovery patterns for callback failures.
"""

import asyncio
import pytest
from unittest.mock import Mock

from src.backend.utils.circuit_breaker import (
    CircuitState,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CallbackCircuitBreaker,
    MultiCallbackCircuitBreaker,
)


class TestCircuitBreakerConfig:
    """Test circuit breaker configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 3
        assert config.success_threshold == 2
        assert config.timeout.total_seconds() == 30
        assert config.expected_exceptions == (Exception,)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=3,
            expected_exceptions=(ValueError, TypeError)
        )
        assert config.failure_threshold == 5
        assert config.success_threshold == 3
        assert config.expected_exceptions == (ValueError, TypeError)


class TestCallbackCircuitBreaker:
    """Test single callback circuit breaker."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker with test configuration."""
        config = CircuitBreakerConfig(failure_threshold=2, success_threshold=1)
        return CallbackCircuitBreaker(config=config, name="TestBreaker")

    def test_initial_state(self, circuit_breaker):
        """Test circuit breaker starts in closed state."""
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.success_count == 0

    @pytest.mark.asyncio
    async def test_async_success(self, circuit_breaker):
        """Test successful async callback execution."""
        async def success_callback(value):
            return value * 2

        result = await circuit_breaker.call_async(success_callback, 5)
        assert result == 10
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0

    def test_sync_success(self, circuit_breaker):
        """Test successful sync callback execution."""
        def success_callback(value):
            return value * 2

        result = circuit_breaker.call_sync(success_callback, 5)
        assert result == 10
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_async_failure_opens_circuit(self, circuit_breaker):
        """Test async failures open circuit after threshold."""
        async def failing_callback(value):
            raise ValueError("Test failure")

        # First failure
        with pytest.raises(ValueError):
            await circuit_breaker.call_async(failing_callback, 1)
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 1

        # Second failure opens circuit
        with pytest.raises(ValueError):
            await circuit_breaker.call_async(failing_callback, 1)
        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.failure_count == 2

    def test_sync_failure_opens_circuit(self, circuit_breaker):
        """Test sync failures open circuit after threshold."""
        def failing_callback(value):
            raise ValueError("Test failure")

        # First failure
        with pytest.raises(ValueError):
            circuit_breaker.call_sync(failing_callback, 1)
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 1

        # Second failure opens circuit
        with pytest.raises(ValueError):
            circuit_breaker.call_sync(failing_callback, 1)
        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.failure_count == 2

    @pytest.mark.asyncio
    async def test_open_circuit_rejects_calls(self, circuit_breaker):
        """Test open circuit rejects calls with CircuitBreakerError."""
        from datetime import datetime
        # Force circuit open with recent failure
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.failure_count = 2
        circuit_breaker.last_failure_time = datetime.now()  # Recent failure prevents half-open

        async def success_callback(value):
            return value

        with pytest.raises(CircuitBreakerError):
            await circuit_breaker.call_async(success_callback, 1)

    def test_sync_open_circuit_rejects_calls(self, circuit_breaker):
        """Test sync open circuit rejects calls."""
        from datetime import datetime
        # Force circuit open with recent failure
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.failure_count = 2
        circuit_breaker.last_failure_time = datetime.now()  # Recent failure prevents half-open

        def success_callback(value):
            return value

        with pytest.raises(CircuitBreakerError):
            circuit_breaker.call_sync(success_callback, 1)

    def test_get_state(self, circuit_breaker):
        """Test circuit breaker state monitoring."""
        state = circuit_breaker.get_state()
        assert state["name"] == "TestBreaker"
        assert state["state"] == "closed"
        assert state["failure_count"] == 0
        assert state["success_count"] == 0
        assert state["last_failure"] is None
        assert "config" in state

    def test_reset(self, circuit_breaker):
        """Test manual circuit breaker reset."""
        # Force failure state
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.failure_count = 5
        circuit_breaker.success_count = 2

        circuit_breaker.reset()
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.success_count == 0
        assert circuit_breaker.last_failure_time is None


class TestMultiCallbackCircuitBreaker:
    """Test multi-callback circuit breaker manager."""

    @pytest.fixture
    def multi_breaker(self):
        """Create multi-callback circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=2)
        return MultiCallbackCircuitBreaker(default_config=config)

    def test_get_breaker_creates_new(self, multi_breaker):
        """Test getting breaker creates new instance."""
        breaker = multi_breaker.get_breaker("test_callback")
        assert breaker.name == "CB_test_callback"
        assert "test_callback" in multi_breaker.breakers

    def test_get_breaker_returns_existing(self, multi_breaker):
        """Test getting same breaker returns existing instance."""
        breaker1 = multi_breaker.get_breaker("test_callback")
        breaker2 = multi_breaker.get_breaker("test_callback")
        assert breaker1 is breaker2

    @pytest.mark.asyncio
    async def test_async_call_through_manager(self, multi_breaker):
        """Test async call through multi-breaker manager."""
        async def test_callback(value):
            return value * 3

        result = await multi_breaker.call_async("test", test_callback, 4)
        assert result == 12

    def test_sync_call_through_manager(self, multi_breaker):
        """Test sync call through multi-breaker manager."""
        def test_callback(value):
            return value * 3

        result = multi_breaker.call_sync("test", test_callback, 4)
        assert result == 12

    def test_get_all_states(self, multi_breaker):
        """Test getting all circuit breaker states."""
        # Create some breakers
        multi_breaker.get_breaker("callback1")
        multi_breaker.get_breaker("callback2")

        states = multi_breaker.get_all_states()
        assert len(states) == 2
        assert "callback1" in states
        assert "callback2" in states

    def test_reset_all(self, multi_breaker):
        """Test resetting all circuit breakers."""
        # Create and break some circuits
        breaker1 = multi_breaker.get_breaker("callback1")
        breaker2 = multi_breaker.get_breaker("callback2")
        
        # Force open state
        breaker1.state = CircuitState.OPEN
        breaker2.state = CircuitState.OPEN

        multi_breaker.reset_all()
        assert breaker1.state == CircuitState.CLOSED
        assert breaker2.state == CircuitState.CLOSED


class TestCircuitBreakerHalfOpenState:
    """Test half-open state behavior and recovery."""

    @pytest.fixture
    def quick_timeout_breaker(self):
        """Create breaker with short timeout for testing."""
        from datetime import timedelta
        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=1,
            timeout=timedelta(milliseconds=100)
        )
        return CallbackCircuitBreaker(config=config, name="QuickTimeout")

    @pytest.mark.asyncio
    async def test_half_open_transition(self, quick_timeout_breaker):
        """Test transition from open to half-open state."""
        async def failing_callback(value):
            raise ValueError("Failure")
        
        async def success_callback(value):
            return value

        # Open the circuit
        with pytest.raises(ValueError):
            await quick_timeout_breaker.call_async(failing_callback, 1)
        with pytest.raises(ValueError):
            await quick_timeout_breaker.call_async(failing_callback, 1)
        
        assert quick_timeout_breaker.state == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Next call should enter half-open
        result = await quick_timeout_breaker.call_async(success_callback, 42)
        assert result == 42
        assert quick_timeout_breaker.state == CircuitState.CLOSED  # Success closes circuit

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens(self, quick_timeout_breaker):
        """Test failure in half-open state reopens circuit."""
        async def failing_callback(value):
            raise ValueError("Failure")

        # Open the circuit
        with pytest.raises(ValueError):
            await quick_timeout_breaker.call_async(failing_callback, 1)
        with pytest.raises(ValueError):
            await quick_timeout_breaker.call_async(failing_callback, 1)
        
        assert quick_timeout_breaker.state == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Failure in half-open should reopen
        with pytest.raises(ValueError):
            await quick_timeout_breaker.call_async(failing_callback, 1)
        
        assert quick_timeout_breaker.state == CircuitState.OPEN


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration scenarios."""

    @pytest.mark.asyncio
    async def test_rssi_callback_protection(self):
        """Test circuit breaker protecting RSSI callback failures."""
        config = CircuitBreakerConfig(failure_threshold=3, success_threshold=2)
        breaker = CallbackCircuitBreaker(config=config, name="RSSI_Processor")
        
        # Simulate RSSI processing callback with intermittent failures
        call_count = 0
        async def rssi_callback(rssi_value):
            nonlocal call_count
            call_count += 1
            if call_count in [2, 3, 4]:  # Fail on calls 2, 3, 4
                raise RuntimeError("RSSI processing failed")
            return rssi_value * 1.1  # Some processing

        # First call succeeds
        result = await breaker.call_async(rssi_callback, 10.0)
        assert result == 11.0
        assert breaker.state == CircuitState.CLOSED

        # Calls 2, 3, 4 fail - should open circuit
        for i in range(3):
            with pytest.raises(RuntimeError):
                await breaker.call_async(rssi_callback, 10.0)
        
        assert breaker.state == CircuitState.OPEN
        assert breaker.failure_count == 3

        # Circuit should reject next call
        with pytest.raises(CircuitBreakerError):
            await breaker.call_async(rssi_callback, 10.0)

    @pytest.mark.asyncio
    async def test_multiple_service_callbacks(self):
        """Test multiple services with independent circuit breakers."""
        multi_breaker = MultiCallbackCircuitBreaker()
        
        # Different callback types
        async def rssi_callback(value):
            if value < 0:
                raise ValueError("Invalid RSSI")
            return value
        
        async def mavlink_callback(command):
            if command == "FAIL":
                raise RuntimeError("MAVLink error")
            return f"ACK_{command}"
        
        # Both should work initially
        rssi_result = await multi_breaker.call_async("rssi", rssi_callback, 15.0)
        mavlink_result = await multi_breaker.call_async("mavlink", mavlink_callback, "HOME")
        
        assert rssi_result == 15.0
        assert mavlink_result == "ACK_HOME"
        
        # Break RSSI but not MAVLink
        for _ in range(3):
            with pytest.raises(ValueError):
                await multi_breaker.call_async("rssi", rssi_callback, -1.0)
        
        # RSSI should be open, MAVLink should still work
        rssi_breaker = multi_breaker.get_breaker("rssi")
        mavlink_breaker = multi_breaker.get_breaker("mavlink")
        
        assert rssi_breaker.state == CircuitState.OPEN
        assert mavlink_breaker.state == CircuitState.CLOSED
        
        # MAVLink still works
        result = await multi_breaker.call_async("mavlink", mavlink_callback, "RTL")
        assert result == "ACK_RTL"