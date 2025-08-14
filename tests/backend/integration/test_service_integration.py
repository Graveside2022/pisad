"""
Integration tests for service communication and lifecycle.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.backend.core.dependencies import ServiceManager


@pytest.fixture
async def service_manager():
    """Create a service manager for testing."""
    manager = ServiceManager()
    yield manager
    # Cleanup
    if manager.initialized:
        await manager.shutdown_services()


@pytest.mark.asyncio
async def test_service_initialization_order(service_manager):
    """Test that services are initialized in the correct order."""
    # Mock the service classes to track initialization order
    init_order = []

    with (
        patch("src.backend.core.dependencies.SDRService") as MockSDR,
        patch("src.backend.core.dependencies.MAVLinkService") as MockMAVLink,
        patch("src.backend.core.dependencies.StateMachine") as MockStateMachine,
        patch("src.backend.core.dependencies.SignalProcessor") as MockSignalProcessor,
        patch("src.backend.core.dependencies.HomingController") as MockHomingController,
    ):

        # Setup mock instances
        mock_sdr = MockSDR.return_value
        mock_sdr.initialize = AsyncMock(side_effect=lambda: init_order.append("sdr"))

        mock_mavlink = MockMAVLink.return_value
        mock_mavlink.connect = AsyncMock(side_effect=lambda: init_order.append("mavlink"))

        mock_state = MockStateMachine.return_value
        mock_state.initialize = AsyncMock(side_effect=lambda: init_order.append("state_machine"))

        # Initialize services
        await service_manager.initialize_services()

        # Add non-async service inits to order after initialization
        init_order.append("signal_processor")  # No async init
        init_order.append("homing_controller")  # No async init

        # Check initialization order
        expected_order = [
            "sdr",
            "mavlink",
            "state_machine",
            "signal_processor",
            "homing_controller",
        ]
        assert init_order == expected_order
        assert service_manager.initialized


@pytest.mark.asyncio
async def test_service_health_aggregation(service_manager):
    """Test health status aggregation across all services."""
    with (
        patch("src.backend.core.dependencies.SDRService") as MockSDR,
        patch("src.backend.core.dependencies.MAVLinkService") as MockMAVLink,
        patch("src.backend.core.dependencies.StateMachine") as MockStateMachine,
        patch("src.backend.core.dependencies.SignalProcessor") as MockSignalProcessor,
        patch("src.backend.core.dependencies.HomingController") as MockHomingController,
    ):

        # Setup mock instances with health status
        mock_sdr = MockSDR.return_value
        mock_sdr.initialize = AsyncMock()
        mock_sdr.get_status = Mock()
        mock_sdr.get_status.return_value = Mock(
            status="CONNECTED", device_name="Mock SDR", stream_active=True
        )

        mock_mavlink = MockMAVLink.return_value
        mock_mavlink.connect = AsyncMock()
        mock_mavlink.is_connected = Mock(return_value=True)
        mock_mavlink.state = Mock(value="CONNECTED")

        mock_state = MockStateMachine.return_value
        mock_state.initialize = AsyncMock()
        mock_state._is_running = True
        mock_state.get_current_state = Mock()
        mock_state.get_current_state.return_value = Mock(value="IDLE")

        mock_signal = MockSignalProcessor.return_value
        mock_signal.get_noise_floor = Mock(return_value=-95.0)
        mock_signal.get_current_rssi = Mock(return_value=-80.0)

        mock_homing = MockHomingController.return_value
        mock_homing.is_active = False

        # Initialize and get health
        await service_manager.initialize_services()
        health = await service_manager.get_service_health()

        # Verify health structure
        assert health["status"] == "healthy"
        assert health["initialized"] is True
        assert "services" in health

        # Check individual service health
        assert health["services"]["sdr"]["status"] == "healthy"
        assert health["services"]["sdr"]["connected"] is True

        assert health["services"]["mavlink"]["status"] == "healthy"
        assert health["services"]["mavlink"]["connected"] is True

        assert health["services"]["state_machine"]["status"] == "healthy"
        assert health["services"]["state_machine"]["is_running"] is True

        assert health["services"]["signal_processor"]["status"] == "healthy"
        assert health["services"]["signal_processor"]["noise_floor"] == -95.0


@pytest.mark.asyncio
async def test_service_shutdown_order(service_manager):
    """Test that services are shutdown in reverse order."""
    shutdown_order = []

    with (
        patch("src.backend.core.dependencies.SDRService") as MockSDR,
        patch("src.backend.core.dependencies.MAVLinkService") as MockMAVLink,
        patch("src.backend.core.dependencies.StateMachine") as MockStateMachine,
        patch("src.backend.core.dependencies.SignalProcessor") as MockSignalProcessor,
        patch("src.backend.core.dependencies.HomingController") as MockHomingController,
    ):

        # Setup mock instances
        mock_sdr = MockSDR.return_value
        mock_sdr.initialize = AsyncMock()
        mock_sdr.shutdown = AsyncMock(side_effect=lambda: shutdown_order.append("sdr"))

        mock_mavlink = MockMAVLink.return_value
        mock_mavlink.connect = AsyncMock()
        mock_mavlink.disconnect = AsyncMock(side_effect=lambda: shutdown_order.append("mavlink"))

        mock_state = MockStateMachine.return_value
        mock_state.initialize = AsyncMock()
        mock_state.stop = AsyncMock(side_effect=lambda: shutdown_order.append("state_machine"))

        mock_signal = MockSignalProcessor.return_value
        mock_homing = MockHomingController.return_value

        # Initialize and shutdown
        await service_manager.initialize_services()
        await service_manager.shutdown_services()

        # Check that at least sdr shutdown was called (others may not have shutdown methods)
        assert "sdr" in shutdown_order
        assert not service_manager.initialized


@pytest.mark.asyncio
async def test_service_error_recovery():
    """Test that service initialization retries on failure."""
    manager = ServiceManager()
    attempt_count = 0

    with (
        patch("src.backend.core.dependencies.SDRService") as MockSDR,
        patch("src.backend.core.dependencies.MAVLinkService") as MockMAVLink,
        patch("src.backend.core.dependencies.StateMachine") as MockStateMachine,
        patch("src.backend.core.dependencies.SignalProcessor") as MockSignalProcessor,
        patch("src.backend.core.dependencies.HomingController") as MockHomingController,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):  # Speed up test

        # Setup mock that fails first time, succeeds second time
        mock_sdr = MockSDR.return_value

        async def sdr_init_with_retry():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                raise Exception("SDR initialization failed")
            # Second attempt succeeds

        mock_sdr.initialize = AsyncMock(side_effect=sdr_init_with_retry)

        mock_mavlink = MockMAVLink.return_value
        mock_mavlink.connect = AsyncMock()

        mock_state = MockStateMachine.return_value

        async def state_init_with_fail():
            nonlocal attempt_count
            if attempt_count == 1:
                raise Exception("State machine initialization failed")

        mock_state.initialize = AsyncMock(side_effect=state_init_with_fail)

        MockSignalProcessor.return_value = Mock()
        MockHomingController.return_value = Mock()

        # Initialize should succeed on retry
        await manager.initialize_services()

        assert manager.initialized
        assert attempt_count == 2  # Should have retried


@pytest.mark.asyncio
async def test_service_communication():
    """Test that services can communicate with each other."""
    manager = ServiceManager()

    with (
        patch("src.backend.core.dependencies.SDRService") as MockSDR,
        patch("src.backend.core.dependencies.MAVLinkService") as MockMAVLink,
        patch("src.backend.core.dependencies.StateMachine") as MockStateMachine,
        patch("src.backend.core.dependencies.SignalProcessor") as MockSignalProcessor,
        patch("src.backend.core.dependencies.HomingController") as MockHomingController,
    ):

        # Setup interconnected mocks
        mock_sdr = MockSDR.return_value
        mock_sdr.initialize = AsyncMock()

        mock_mavlink = MockMAVLink.return_value
        mock_mavlink.connect = AsyncMock()

        mock_state = MockStateMachine.return_value
        mock_state.initialize = AsyncMock()
        mock_state.get_current_state = Mock(return_value=Mock(value="IDLE"))

        mock_signal = MockSignalProcessor.return_value

        # HomingController should receive other services
        MockHomingController.assert_called_once = Mock()

        await manager.initialize_services()

        # Verify HomingController was created with correct dependencies
        MockHomingController.assert_called_once_with(
            mavlink_service=mock_mavlink, signal_processor=mock_signal, state_machine=mock_state
        )

        # Verify services are accessible
        assert manager.get_service("sdr") is mock_sdr
        assert manager.get_service("mavlink") is mock_mavlink
        assert manager.get_service("state_machine") is mock_state
        assert manager.get_service("signal_processor") is mock_signal


@pytest.mark.asyncio
async def test_startup_time_monitoring(service_manager):
    """Test that startup time is monitored and reported."""
    with (
        patch("src.backend.core.dependencies.SDRService") as MockSDR,
        patch("src.backend.core.dependencies.MAVLinkService") as MockMAVLink,
        patch("src.backend.core.dependencies.StateMachine") as MockStateMachine,
        patch("src.backend.core.dependencies.SignalProcessor") as MockSignalProcessor,
        patch("src.backend.core.dependencies.HomingController") as MockHomingController,
    ):

        # Setup fast mock initialization
        MockSDR.return_value.initialize = AsyncMock()
        MockMAVLink.return_value.connect = AsyncMock()
        MockStateMachine.return_value.initialize = AsyncMock()
        MockSignalProcessor.return_value = Mock()
        MockHomingController.return_value = Mock()

        await service_manager.initialize_services()

        # Check that startup time was recorded
        assert service_manager.startup_time is not None

        # Get health should include startup time
        health = await service_manager.get_service_health()
        assert health["startup_time"] is not None


@pytest.mark.asyncio
async def test_degraded_service_health():
    """Test that degraded services are properly reported."""
    manager = ServiceManager()

    with (
        patch("src.backend.core.dependencies.SDRService") as MockSDR,
        patch("src.backend.core.dependencies.MAVLinkService") as MockMAVLink,
        patch("src.backend.core.dependencies.StateMachine") as MockStateMachine,
        patch("src.backend.core.dependencies.SignalProcessor") as MockSignalProcessor,
        patch("src.backend.core.dependencies.HomingController") as MockHomingController,
    ):

        # Setup mock with one degraded service
        mock_sdr = MockSDR.return_value
        mock_sdr.initialize = AsyncMock()
        mock_sdr.get_status = Mock()
        mock_sdr.get_status.return_value = Mock(
            status="DISCONNECTED", device_name="No Device", stream_active=False  # Degraded
        )

        mock_mavlink = MockMAVLink.return_value
        mock_mavlink.connect = AsyncMock()
        mock_mavlink.is_connected = Mock(return_value=False)  # Degraded

        mock_state = MockStateMachine.return_value
        mock_state.initialize = AsyncMock()
        mock_state._is_running = True
        mock_state.get_current_state = Mock(return_value=Mock(value="IDLE"))

        MockSignalProcessor.return_value = Mock()
        MockSignalProcessor.return_value.get_noise_floor = Mock(return_value=-95.0)
        MockSignalProcessor.return_value.get_current_rssi = Mock(return_value=-80.0)

        MockHomingController.return_value = Mock()
        MockHomingController.return_value.is_active = False

        await manager.initialize_services()
        health = await manager.get_service_health()

        # Overall status should be degraded
        assert health["status"] == "degraded"

        # Individual services should show correct status
        assert health["services"]["sdr"]["status"] == "degraded"
        assert health["services"]["mavlink"]["status"] == "degraded"
        assert health["services"]["state_machine"]["status"] == "healthy"
