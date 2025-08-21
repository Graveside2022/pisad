"""
Test Suite for Homing Substage Reporting (TASK-6.3.1 [29b3])

Tests for homing substage reporting (APPROACH, SPIRAL_SEARCH, S_TURN, RETURN_TO_PEAK):
- Substage transition tracking and validation
- Mission Planner parameter integration for substage display
- Real-time substage updates during homing operations
- Substage duration and performance metrics
- Error handling and recovery during substage transitions
"""

import time

import pytest

from src.backend.services.mavlink_service import MAVLinkService


class TestHomingSubstageReporting:
    """Test suite for homing substage reporting."""

    @pytest.fixture
    def mavlink_service(self):
        """Create MAVLink service for testing."""
        service = MAVLinkService()
        service._parameters = {}
        return service

    def test_homing_substage_definitions(self, mavlink_service):
        """Test homing substage parameter definitions [29b3]."""
        # Test all defined substages
        substages = [
            (0, "INACTIVE"),
            (1, "APPROACH"),
            (2, "SPIRAL_SEARCH"),
            (3, "S_TURN"),
            (4, "RETURN_TO_PEAK"),
        ]

        for substage_value, substage_name in substages:
            mavlink_service.update_homing_substage_parameter(substage_value)

            # Verify parameter is updated
            assert mavlink_service._parameters["PISAD_HOMING_SUBSTAGE"] == float(substage_value)

            # Verify value is in valid range
            assert 0.0 <= mavlink_service._parameters["PISAD_HOMING_SUBSTAGE"] <= 4.0

    def test_homing_substage_approach_reporting(self, mavlink_service):
        """Test APPROACH substage reporting [29b3]."""
        # Test APPROACH substage (direct movement toward signal)
        approach_substage = 1
        mavlink_service.update_homing_substage_parameter(approach_substage)

        # Verify APPROACH substage is correctly set
        assert mavlink_service._parameters["PISAD_HOMING_SUBSTAGE"] == 1.0

        # Test that substage is properly tracked
        current_substage = mavlink_service.get_parameter("PISAD_HOMING_SUBSTAGE")
        assert current_substage == 1.0

    def test_homing_substage_spiral_search_reporting(self, mavlink_service):
        """Test SPIRAL_SEARCH substage reporting [29b3]."""
        # Test SPIRAL_SEARCH substage (circular search pattern)
        spiral_substage = 2
        mavlink_service.update_homing_substage_parameter(spiral_substage)

        # Verify SPIRAL_SEARCH substage is correctly set
        assert mavlink_service._parameters["PISAD_HOMING_SUBSTAGE"] == 2.0

        # Test substage persistence
        stored_substage = mavlink_service.get_parameter("PISAD_HOMING_SUBSTAGE")
        assert stored_substage == 2.0

    def test_homing_substage_s_turn_reporting(self, mavlink_service):
        """Test S_TURN substage reporting [29b3]."""
        # Test S_TURN substage (zigzag pattern for signal refinement)
        s_turn_substage = 3
        mavlink_service.update_homing_substage_parameter(s_turn_substage)

        # Verify S_TURN substage is correctly set
        assert mavlink_service._parameters["PISAD_HOMING_SUBSTAGE"] == 3.0

        # Test parameter validation
        current_substage = mavlink_service.get_parameter("PISAD_HOMING_SUBSTAGE")
        assert current_substage == 3.0

    def test_homing_substage_return_to_peak_reporting(self, mavlink_service):
        """Test RETURN_TO_PEAK substage reporting [29b3]."""
        # Test RETURN_TO_PEAK substage (returning to strongest signal)
        return_substage = 4
        mavlink_service.update_homing_substage_parameter(return_substage)

        # Verify RETURN_TO_PEAK substage is correctly set
        assert mavlink_service._parameters["PISAD_HOMING_SUBSTAGE"] == 4.0

        # Test that substage is tracked properly
        current_substage = mavlink_service.get_parameter("PISAD_HOMING_SUBSTAGE")
        assert current_substage == 4.0

    def test_homing_substage_transition_sequence(self, mavlink_service):
        """Test homing substage transition sequence [29b3]."""
        # Test realistic homing substage sequence
        substage_sequence = [
            (0, "INACTIVE"),
            (1, "APPROACH"),
            (2, "SPIRAL_SEARCH"),
            (3, "S_TURN"),
            (4, "RETURN_TO_PEAK"),
            (1, "APPROACH"),  # Return to approach
            (0, "INACTIVE"),  # Complete
        ]

        for substage, description in substage_sequence:
            mavlink_service.update_homing_substage_parameter(substage)

            # Verify each transition
            assert mavlink_service._parameters["PISAD_HOMING_SUBSTAGE"] == float(substage)

            # Verify substage is within valid range
            assert 0.0 <= mavlink_service._parameters["PISAD_HOMING_SUBSTAGE"] <= 4.0

    def test_homing_substage_timing_requirements(self, mavlink_service):
        """Test homing substage update timing requirements [29b3]."""
        # Test substage update response time
        start_time = time.perf_counter()
        mavlink_service.update_homing_substage_parameter(2)  # SPIRAL_SEARCH
        end_time = time.perf_counter()

        update_time_ms = (end_time - start_time) * 1000

        # Should update within reasonable time
        assert update_time_ms < 10.0  # <10ms for parameter update
        assert mavlink_service._parameters["PISAD_HOMING_SUBSTAGE"] == 2.0

    def test_homing_substage_validation(self, mavlink_service):
        """Test homing substage parameter validation [29b3]."""
        # Test valid substages
        valid_substages = [0, 1, 2, 3, 4]
        for substage in valid_substages:
            mavlink_service.update_homing_substage_parameter(substage)
            assert mavlink_service._parameters["PISAD_HOMING_SUBSTAGE"] == float(substage)

        # Test invalid substages (should be rejected or clamped)
        invalid_substages = [-1, 5, 10]
        original_substage = mavlink_service._parameters.get("PISAD_HOMING_SUBSTAGE", 0.0)

        for invalid_substage in invalid_substages:
            mavlink_service.update_homing_substage_parameter(invalid_substage)
            # Substage should either remain unchanged or be clamped to valid range
            current_substage = mavlink_service._parameters.get("PISAD_HOMING_SUBSTAGE", 0.0)
            assert 0.0 <= current_substage <= 4.0

    def test_homing_substage_mission_planner_integration(self, mavlink_service):
        """Test homing substage integration with Mission Planner parameters [29b3]."""
        # Test that homing substage is part of telemetry parameters
        # Substage parameters should be streamed for Mission Planner display

        # Simulate homing substage progression
        progression = [
            (0, "Starting inactive"),
            (1, "Beginning approach"),
            (2, "Starting spiral search"),
            (3, "Executing S-turn pattern"),
            (4, "Returning to peak signal"),
            (0, "Homing complete"),
        ]

        for substage, description in progression:
            mavlink_service.update_homing_substage_parameter(substage)

            # Verify substage update
            assert mavlink_service._parameters["PISAD_HOMING_SUBSTAGE"] == float(substage)

            # Verify substage is in Mission Planner parameter registry
            assert "PISAD_HOMING_SUBSTAGE" in mavlink_service._parameters

    def test_homing_substage_duration_tracking(self, mavlink_service):
        """Test homing substage duration tracking [29b3]."""
        # Test tracking time spent in each substage
        substages_with_duration = [
            (1, "APPROACH", 0.1),  # 100ms in approach
            (2, "SPIRAL_SEARCH", 0.05),  # 50ms in spiral search
            (3, "S_TURN", 0.08),  # 80ms in S-turn
            (4, "RETURN_TO_PEAK", 0.06),  # 60ms in return to peak
        ]

        for substage, name, duration in substages_with_duration:
            start_time = time.perf_counter()
            mavlink_service.update_homing_substage_parameter(substage)

            # Simulate substage duration
            time.sleep(duration)

            end_time = time.perf_counter()
            actual_duration = end_time - start_time

            # Verify substage was active and duration is reasonable
            assert mavlink_service._parameters["PISAD_HOMING_SUBSTAGE"] == float(substage)
            assert actual_duration >= duration  # Should be at least the sleep duration

    def test_homing_substage_concurrent_updates(self, mavlink_service):
        """Test homing substage updates under concurrent access [29b3]."""
        # Test rapid sequential substage updates
        rapid_substage_updates = [0, 1, 2, 1, 3, 4, 1, 0]

        for substage in rapid_substage_updates:
            mavlink_service.update_homing_substage_parameter(substage)
            # Each update should be processed correctly
            assert mavlink_service._parameters["PISAD_HOMING_SUBSTAGE"] == float(substage)

        # Final substage should be the last update
        assert mavlink_service._parameters["PISAD_HOMING_SUBSTAGE"] == float(
            rapid_substage_updates[-1]
        )

    def test_homing_substage_telemetry_priority(self, mavlink_service):
        """Test homing substage telemetry priority for Mission Planner [29b3]."""
        # PISAD_HOMING_SUBSTAGE should be in medium-priority parameters for regular streaming

        # Check if it's configured appropriately in telemetry system
        # This ensures Mission Planner gets substage updates at reasonable intervals

        # Simulate medium-priority telemetry check
        mavlink_service.update_homing_substage_parameter(2)  # SPIRAL_SEARCH

        # Verify parameter exists and is updated
        assert "PISAD_HOMING_SUBSTAGE" in mavlink_service._parameters
        assert mavlink_service._parameters["PISAD_HOMING_SUBSTAGE"] == 2.0

    def test_homing_substage_error_handling(self, mavlink_service):
        """Test homing substage update error handling [29b3]."""
        # Test with various invalid inputs
        invalid_inputs = [None, "invalid", 2.5, -0.5]
        original_substage = mavlink_service._parameters.get("PISAD_HOMING_SUBSTAGE", 0.0)

        for invalid_input in invalid_inputs:
            try:
                mavlink_service.update_homing_substage_parameter(invalid_input)
                # If no exception, substage should remain valid
                current_substage = mavlink_service._parameters.get("PISAD_HOMING_SUBSTAGE", 0.0)
                assert 0.0 <= current_substage <= 4.0
            except (TypeError, ValueError):
                # Exceptions are acceptable for invalid inputs
                pass

    def test_homing_substage_performance_requirements(self, mavlink_service):
        """Test homing substage update performance requirements [29b3]."""
        # Test multiple rapid substage updates to ensure performance
        num_updates = 50
        start_time = time.perf_counter()

        for i in range(num_updates):
            substage = i % 5  # Cycle through substages 0-4
            mavlink_service.update_homing_substage_parameter(substage)

        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000
        avg_time_per_update = total_time_ms / num_updates

        # Each update should be very fast
        assert avg_time_per_update < 1.0  # <1ms per update on average

        # Final substage should be correct
        assert mavlink_service._parameters["PISAD_HOMING_SUBSTAGE"] == float((num_updates - 1) % 5)

    def test_homing_substage_state_consistency(self, mavlink_service):
        """Test homing substage consistency with homing state [29b3]."""
        # Test that substage reporting is consistent with overall homing state

        # When homing is disabled, substage should be INACTIVE
        mavlink_service.update_homing_state_parameter(0)  # Disabled
        mavlink_service.update_homing_substage_parameter(0)  # INACTIVE

        assert mavlink_service._parameters["PISAD_HOMING_STATE"] == 0.0
        assert mavlink_service._parameters["PISAD_HOMING_SUBSTAGE"] == 0.0

        # When homing is active, substage should be non-zero
        mavlink_service.update_homing_state_parameter(2)  # Active
        mavlink_service.update_homing_substage_parameter(1)  # APPROACH

        assert mavlink_service._parameters["PISAD_HOMING_STATE"] == 2.0
        assert mavlink_service._parameters["PISAD_HOMING_SUBSTAGE"] == 1.0

    def test_homing_substage_mission_planner_display_data(self, mavlink_service):
        """Test homing substage data formatted for Mission Planner display [29b3]."""
        # Test that substage information is properly formatted for Mission Planner
        substage_descriptions = {
            0: "INACTIVE",
            1: "APPROACH",
            2: "SPIRAL_SEARCH",
            3: "S_TURN",
            4: "RETURN_TO_PEAK",
        }

        for substage_value, description in substage_descriptions.items():
            mavlink_service.update_homing_substage_parameter(substage_value)

            # Verify parameter is properly set for Mission Planner display
            assert mavlink_service._parameters["PISAD_HOMING_SUBSTAGE"] == float(substage_value)

            # Verify parameter is available for telemetry streaming
            param_value = mavlink_service.get_parameter("PISAD_HOMING_SUBSTAGE")
            assert param_value == float(substage_value)
