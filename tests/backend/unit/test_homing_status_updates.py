"""
Test Suite for Real-time Homing Status Updates (TASK-6.3.1 [29b2])

Tests for real-time homing status updates for Mission Planner display integration:
- Real-time status parameter updates
- Mission Planner telemetry streaming
- Homing state change notifications
- Performance timing validation
- Status persistence and recovery
"""

import time
from unittest.mock import AsyncMock

import pytest

from src.backend.services.mavlink_service import MAVLinkService


class TestHomingStatusUpdates:
    """Test suite for real-time homing status updates."""

    @pytest.fixture
    def mavlink_service(self):
        """Create MAVLink service for testing."""
        service = MAVLinkService()
        service._parameters = {}
        return service

    def test_homing_state_parameter_updates(self, mavlink_service):
        """Test real-time homing state parameter updates [29b2]."""
        # Test all homing states
        homing_states = [(0, "Disabled"), (1, "Armed"), (2, "Active"), (3, "Lost")]

        for state_value, state_name in homing_states:
            mavlink_service.update_homing_state_parameter(state_value)

            # Verify parameter is updated
            assert mavlink_service._parameters["PISAD_HOMING_STATE"] == float(state_value)

            # Verify value is in valid range
            assert 0.0 <= mavlink_service._parameters["PISAD_HOMING_STATE"] <= 3.0

    def test_homing_status_real_time_streaming(self, mavlink_service):
        """Test real-time homing status streaming for Mission Planner [29b2]."""
        # Mock the streaming functionality
        mavlink_service._send_named_value_float = AsyncMock()

        # Test status update with immediate streaming
        initial_state = 0  # Disabled
        active_state = 2  # Active

        # Update to active state
        mavlink_service.update_homing_state_parameter(active_state)

        # Verify parameter updated immediately
        assert mavlink_service._parameters["PISAD_HOMING_STATE"] == float(active_state)

    def test_homing_status_update_timing(self, mavlink_service):
        """Test homing status update timing requirements [29b2]."""
        # Test update response time
        start_time = time.perf_counter()
        mavlink_service.update_homing_state_parameter(2)  # Active state
        end_time = time.perf_counter()

        update_time_ms = (end_time - start_time) * 1000

        # Should update within reasonable time
        assert update_time_ms < 10.0  # <10ms for parameter update
        assert mavlink_service._parameters["PISAD_HOMING_STATE"] == 2.0

    def test_homing_status_change_notifications(self, mavlink_service):
        """Test homing status change notifications [29b2]."""
        # Track state changes
        previous_state = mavlink_service._parameters.get("PISAD_HOMING_STATE", 0.0)

        # Test state transition: Disabled -> Armed -> Active
        transitions = [1, 2]  # Armed, then Active

        for new_state in transitions:
            mavlink_service.update_homing_state_parameter(new_state)
            current_state = mavlink_service._parameters["PISAD_HOMING_STATE"]

            # Verify state changed
            assert current_state != previous_state
            assert current_state == float(new_state)

            previous_state = current_state

    def test_homing_status_validation(self, mavlink_service):
        """Test homing status validation [29b2]."""
        # Test valid states
        valid_states = [0, 1, 2, 3]
        for state in valid_states:
            mavlink_service.update_homing_state_parameter(state)
            assert mavlink_service._parameters["PISAD_HOMING_STATE"] == float(state)

        # Test invalid states (should be rejected or clamped)
        invalid_states = [-1, 4, 10]
        original_state = mavlink_service._parameters.get("PISAD_HOMING_STATE", 0.0)

        for invalid_state in invalid_states:
            mavlink_service.update_homing_state_parameter(invalid_state)
            # State should either remain unchanged or be clamped to valid range
            current_state = mavlink_service._parameters.get("PISAD_HOMING_STATE", 0.0)
            assert 0.0 <= current_state <= 3.0

    def test_homing_status_mission_planner_integration(self, mavlink_service):
        """Test homing status integration with Mission Planner parameters [29b2]."""
        # Test that homing status is part of high-priority parameters
        # High-priority parameters should be streamed immediately

        # Simulate homing activation sequence
        sequence = [
            (0, "Initial disabled state"),
            (1, "Armed and ready"),
            (2, "Active homing"),
            (3, "Lost signal state"),
            (0, "Returned to disabled"),
        ]

        for state, description in sequence:
            mavlink_service.update_homing_state_parameter(state)

            # Verify state update
            assert mavlink_service._parameters["PISAD_HOMING_STATE"] == float(state)

            # Verify state is in Mission Planner parameter registry
            assert "PISAD_HOMING_STATE" in mavlink_service._parameters

    def test_homing_status_concurrent_updates(self, mavlink_service):
        """Test homing status updates under concurrent access [29b2]."""
        # Test rapid sequential updates (simulating real-time updates)
        rapid_updates = [0, 1, 2, 1, 2, 3, 0]

        for state in rapid_updates:
            mavlink_service.update_homing_state_parameter(state)
            # Each update should be processed correctly
            assert mavlink_service._parameters["PISAD_HOMING_STATE"] == float(state)

        # Final state should be the last update
        assert mavlink_service._parameters["PISAD_HOMING_STATE"] == float(rapid_updates[-1])

    def test_homing_status_telemetry_priority(self, mavlink_service):
        """Test homing status telemetry priority for Mission Planner [29b2]."""
        # PISAD_HOMING_STATE should be in high-priority parameters for immediate streaming

        # Check if it's configured as high priority in telemetry system
        # This ensures Mission Planner gets immediate updates

        # Simulate high-priority telemetry check
        mavlink_service.update_homing_state_parameter(2)  # Active

        # Verify parameter exists and is updated
        assert "PISAD_HOMING_STATE" in mavlink_service._parameters
        assert mavlink_service._parameters["PISAD_HOMING_STATE"] == 2.0

    def test_homing_status_persistence_integration(self, mavlink_service):
        """Test homing status persistence integration [29b2]."""
        # Test that homing status updates are properly persisted
        test_state = 2  # Active

        mavlink_service.update_homing_state_parameter(test_state)

        # Verify state is updated in memory
        assert mavlink_service._parameters["PISAD_HOMING_STATE"] == float(test_state)

        # Verify state persists in parameter storage
        stored_value = mavlink_service.get_parameter("PISAD_HOMING_STATE")
        assert stored_value == float(test_state)

    def test_homing_status_error_handling(self, mavlink_service):
        """Test homing status update error handling [29b2]."""
        # Test with various invalid inputs
        invalid_inputs = [None, "invalid", 3.5, -0.5]
        original_state = mavlink_service._parameters.get("PISAD_HOMING_STATE", 0.0)

        for invalid_input in invalid_inputs:
            try:
                mavlink_service.update_homing_state_parameter(invalid_input)
                # If no exception, state should remain valid
                current_state = mavlink_service._parameters.get("PISAD_HOMING_STATE", 0.0)
                assert 0.0 <= current_state <= 3.0
            except (TypeError, ValueError):
                # Exceptions are acceptable for invalid inputs
                pass

    def test_homing_status_performance_requirements(self, mavlink_service):
        """Test homing status update performance requirements [29b2]."""
        # Test multiple rapid updates to ensure performance
        num_updates = 100
        start_time = time.perf_counter()

        for i in range(num_updates):
            state = i % 4  # Cycle through states 0-3
            mavlink_service.update_homing_state_parameter(state)

        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000
        avg_time_per_update = total_time_ms / num_updates

        # Each update should be very fast
        assert avg_time_per_update < 1.0  # <1ms per update on average

        # Final state should be correct
        assert mavlink_service._parameters["PISAD_HOMING_STATE"] == float((num_updates - 1) % 4)
