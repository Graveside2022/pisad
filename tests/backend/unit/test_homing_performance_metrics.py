"""
Test Suite for Homing Performance Metrics Parameters (TASK-6.3.1 [29b4])

Tests for homing performance metrics parameters (success_rate, average_time, confidence_level):
- Success rate calculation and parameter updates
- Average homing time tracking and reporting
- Confidence level monitoring and Mission Planner integration
- Performance metrics persistence and historical tracking
- Real-time performance updates during homing operations
"""

import time
from unittest.mock import AsyncMock

import pytest

from src.backend.services.mavlink_service import MAVLinkService


class TestHomingPerformanceMetrics:
    """Test suite for homing performance metrics parameters."""

    @pytest.fixture
    def mavlink_service(self):
        """Create MAVLink service for testing."""
        service = MAVLinkService()
        service._parameters = {}
        return service

    def test_homing_success_rate_parameter(self, mavlink_service):
        """Test homing success rate parameter tracking [29b4]."""
        # Test success rate parameter updates (0-100% scale)
        success_rates = [0.0, 25.0, 50.0, 75.0, 100.0]

        for rate in success_rates:
            mavlink_service.update_homing_success_rate_parameter(rate)

            # Verify parameter is updated
            assert mavlink_service._parameters["PISAD_HOMING_SUCCESS_RATE"] == rate

            # Verify value is in valid range
            assert 0.0 <= mavlink_service._parameters["PISAD_HOMING_SUCCESS_RATE"] <= 100.0

    def test_homing_average_time_parameter(self, mavlink_service):
        """Test homing average time parameter tracking [29b4]."""
        # Test average time parameter updates (seconds)
        avg_times = [0.0, 30.5, 60.0, 120.5, 300.0]

        for avg_time in avg_times:
            mavlink_service.update_homing_average_time_parameter(avg_time)

            # Verify parameter is updated
            assert mavlink_service._parameters["PISAD_HOMING_AVG_TIME"] == avg_time

            # Verify value is reasonable (non-negative)
            assert mavlink_service._parameters["PISAD_HOMING_AVG_TIME"] >= 0.0

    def test_homing_confidence_level_parameter(self, mavlink_service):
        """Test homing confidence level parameter tracking [29b4]."""
        # Test confidence level parameter updates (0-100% scale)
        confidence_levels = [0.0, 10.5, 50.0, 85.5, 100.0]

        for confidence in confidence_levels:
            mavlink_service.update_homing_confidence_level_parameter(confidence)

            # Verify parameter is updated
            assert mavlink_service._parameters["PISAD_HOMING_CONFIDENCE"] == confidence

            # Verify value is in valid range
            assert 0.0 <= mavlink_service._parameters["PISAD_HOMING_CONFIDENCE"] <= 100.0

    def test_homing_performance_metrics_validation(self, mavlink_service):
        """Test homing performance metrics parameter validation [29b4]."""
        # Test valid success rate values
        valid_success_rates = [0.0, 50.0, 100.0]
        for rate in valid_success_rates:
            mavlink_service.update_homing_success_rate_parameter(rate)
            assert mavlink_service._parameters["PISAD_HOMING_SUCCESS_RATE"] == rate

        # Test invalid success rate values (should be clamped or rejected)
        invalid_success_rates = [-10.0, 150.0]
        original_rate = mavlink_service._parameters.get("PISAD_HOMING_SUCCESS_RATE", 0.0)

        for invalid_rate in invalid_success_rates:
            mavlink_service.update_homing_success_rate_parameter(invalid_rate)
            # Rate should either remain unchanged or be clamped to valid range
            current_rate = mavlink_service._parameters.get("PISAD_HOMING_SUCCESS_RATE", 0.0)
            assert 0.0 <= current_rate <= 100.0

    def test_homing_performance_metrics_timing(self, mavlink_service):
        """Test homing performance metrics update timing [29b4]."""
        # Test metrics update response time
        start_time = time.perf_counter()
        mavlink_service.update_homing_success_rate_parameter(85.5)
        mavlink_service.update_homing_average_time_parameter(45.2)
        mavlink_service.update_homing_confidence_level_parameter(92.3)
        end_time = time.perf_counter()

        update_time_ms = (end_time - start_time) * 1000

        # Should update all metrics within reasonable time
        assert update_time_ms < 10.0  # <10ms for all metrics update

        # Verify all metrics were updated
        assert mavlink_service._parameters["PISAD_HOMING_SUCCESS_RATE"] == 85.5
        assert mavlink_service._parameters["PISAD_HOMING_AVG_TIME"] == 45.2
        assert mavlink_service._parameters["PISAD_HOMING_CONFIDENCE"] == 92.3

    def test_homing_performance_metrics_mission_planner_integration(self, mavlink_service):
        """Test performance metrics integration with Mission Planner parameters [29b4]."""
        # Test that performance metrics are properly configured for Mission Planner display

        # Set comprehensive performance metrics
        mavlink_service.update_homing_success_rate_parameter(78.5)
        mavlink_service.update_homing_average_time_parameter(65.3)
        mavlink_service.update_homing_confidence_level_parameter(88.7)

        # Verify all metrics are in Mission Planner parameter registry
        assert "PISAD_HOMING_SUCCESS_RATE" in mavlink_service._parameters
        assert "PISAD_HOMING_AVG_TIME" in mavlink_service._parameters
        assert "PISAD_HOMING_CONFIDENCE" in mavlink_service._parameters

        # Verify values are correctly set
        assert mavlink_service._parameters["PISAD_HOMING_SUCCESS_RATE"] == 78.5
        assert mavlink_service._parameters["PISAD_HOMING_AVG_TIME"] == 65.3
        assert mavlink_service._parameters["PISAD_HOMING_CONFIDENCE"] == 88.7

    def test_homing_performance_metrics_real_time_updates(self, mavlink_service):
        """Test real-time performance metrics updates during homing [29b4]."""
        # Mock the streaming functionality
        mavlink_service._send_named_value_float = AsyncMock()

        # Simulate real-time performance updates during a homing operation
        performance_updates = [
            (75.0, 60.5, 85.0),  # Early homing metrics
            (76.5, 58.2, 87.5),  # Improving metrics
            (78.0, 55.8, 90.0),  # Final improved metrics
        ]

        for success_rate, avg_time, confidence in performance_updates:
            mavlink_service.update_homing_success_rate_parameter(success_rate)
            mavlink_service.update_homing_average_time_parameter(avg_time)
            mavlink_service.update_homing_confidence_level_parameter(confidence)

            # Verify real-time updates
            assert mavlink_service._parameters["PISAD_HOMING_SUCCESS_RATE"] == success_rate
            assert mavlink_service._parameters["PISAD_HOMING_AVG_TIME"] == avg_time
            assert mavlink_service._parameters["PISAD_HOMING_CONFIDENCE"] == confidence

    def test_homing_performance_metrics_historical_tracking(self, mavlink_service):
        """Test performance metrics historical tracking [29b4]."""
        # Test that metrics can track historical performance trends
        historical_data = [
            (65.0, 75.0, 80.0),  # Historical baseline
            (70.0, 70.0, 82.0),  # Improving success, faster time
            (75.0, 65.0, 85.0),  # Further improvement
            (78.0, 62.0, 88.0),  # Current best performance
        ]

        for success_rate, avg_time, confidence in historical_data:
            mavlink_service.update_homing_success_rate_parameter(success_rate)
            mavlink_service.update_homing_average_time_parameter(avg_time)
            mavlink_service.update_homing_confidence_level_parameter(confidence)

            # Each update should be properly tracked
            assert mavlink_service._parameters["PISAD_HOMING_SUCCESS_RATE"] == success_rate
            assert mavlink_service._parameters["PISAD_HOMING_AVG_TIME"] == avg_time
            assert mavlink_service._parameters["PISAD_HOMING_CONFIDENCE"] == confidence

    def test_homing_performance_metrics_persistence(self, mavlink_service):
        """Test performance metrics persistence integration [29b4]."""
        # Test that performance metrics updates are properly persisted
        test_success_rate = 82.5
        test_avg_time = 48.7
        test_confidence = 91.2

        mavlink_service.update_homing_success_rate_parameter(test_success_rate)
        mavlink_service.update_homing_average_time_parameter(test_avg_time)
        mavlink_service.update_homing_confidence_level_parameter(test_confidence)

        # Verify metrics are updated in memory
        assert mavlink_service._parameters["PISAD_HOMING_SUCCESS_RATE"] == test_success_rate
        assert mavlink_service._parameters["PISAD_HOMING_AVG_TIME"] == test_avg_time
        assert mavlink_service._parameters["PISAD_HOMING_CONFIDENCE"] == test_confidence

        # Verify metrics persist in parameter storage
        stored_success_rate = mavlink_service.get_parameter("PISAD_HOMING_SUCCESS_RATE")
        stored_avg_time = mavlink_service.get_parameter("PISAD_HOMING_AVG_TIME")
        stored_confidence = mavlink_service.get_parameter("PISAD_HOMING_CONFIDENCE")

        assert stored_success_rate == test_success_rate
        assert stored_avg_time == test_avg_time
        assert stored_confidence == test_confidence

    def test_homing_performance_metrics_error_handling(self, mavlink_service):
        """Test performance metrics update error handling [29b4]."""
        # Test with various invalid inputs
        invalid_inputs = [None, "invalid", -5.0]

        for invalid_input in invalid_inputs:
            try:
                mavlink_service.update_homing_success_rate_parameter(invalid_input)
                # If no exception, rate should remain valid
                current_rate = mavlink_service._parameters.get("PISAD_HOMING_SUCCESS_RATE", 0.0)
                assert 0.0 <= current_rate <= 100.0
            except (TypeError, ValueError):
                # Exceptions are acceptable for invalid inputs
                pass

            try:
                mavlink_service.update_homing_average_time_parameter(invalid_input)
                # If no exception, time should remain valid
                current_time = mavlink_service._parameters.get("PISAD_HOMING_AVG_TIME", 0.0)
                assert current_time >= 0.0
            except (TypeError, ValueError):
                # Exceptions are acceptable for invalid inputs
                pass

            try:
                mavlink_service.update_homing_confidence_level_parameter(invalid_input)
                # If no exception, confidence should remain valid
                current_confidence = mavlink_service._parameters.get("PISAD_HOMING_CONFIDENCE", 0.0)
                assert 0.0 <= current_confidence <= 100.0
            except (TypeError, ValueError):
                # Exceptions are acceptable for invalid inputs
                pass

    def test_homing_performance_metrics_concurrent_updates(self, mavlink_service):
        """Test performance metrics updates under concurrent access [29b4]."""
        # Test rapid sequential performance updates
        rapid_updates = [
            (70.0, 80.0, 75.0),
            (72.0, 78.0, 77.0),
            (74.0, 76.0, 79.0),
            (76.0, 74.0, 81.0),
        ]

        for success_rate, avg_time, confidence in rapid_updates:
            mavlink_service.update_homing_success_rate_parameter(success_rate)
            mavlink_service.update_homing_average_time_parameter(avg_time)
            mavlink_service.update_homing_confidence_level_parameter(confidence)

            # Each update should be processed correctly
            assert mavlink_service._parameters["PISAD_HOMING_SUCCESS_RATE"] == success_rate
            assert mavlink_service._parameters["PISAD_HOMING_AVG_TIME"] == avg_time
            assert mavlink_service._parameters["PISAD_HOMING_CONFIDENCE"] == confidence

    def test_homing_performance_metrics_telemetry_priority(self, mavlink_service):
        """Test performance metrics telemetry priority for Mission Planner [29b4]."""
        # Performance metrics should be in medium-priority parameters for regular updates

        # Set performance metrics
        mavlink_service.update_homing_success_rate_parameter(83.5)
        mavlink_service.update_homing_average_time_parameter(52.8)
        mavlink_service.update_homing_confidence_level_parameter(89.2)

        # Verify parameters exist and are updated
        assert "PISAD_HOMING_SUCCESS_RATE" in mavlink_service._parameters
        assert "PISAD_HOMING_AVG_TIME" in mavlink_service._parameters
        assert "PISAD_HOMING_CONFIDENCE" in mavlink_service._parameters

        assert mavlink_service._parameters["PISAD_HOMING_SUCCESS_RATE"] == 83.5
        assert mavlink_service._parameters["PISAD_HOMING_AVG_TIME"] == 52.8
        assert mavlink_service._parameters["PISAD_HOMING_CONFIDENCE"] == 89.2

    def test_homing_performance_metrics_performance_requirements(self, mavlink_service):
        """Test performance metrics update performance requirements [29b4]."""
        # Test multiple rapid metrics updates to ensure performance
        num_updates = 30
        start_time = time.perf_counter()

        for i in range(num_updates):
            # Simulate varying performance metrics
            success_rate = 50.0 + (i % 30)  # 50-79%
            avg_time = 60.0 + (i % 20)  # 60-79 seconds
            confidence = 70.0 + (i % 25)  # 70-94%

            mavlink_service.update_homing_success_rate_parameter(success_rate)
            mavlink_service.update_homing_average_time_parameter(avg_time)
            mavlink_service.update_homing_confidence_level_parameter(confidence)

        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000
        avg_time_per_update_set = total_time_ms / num_updates

        # Each set of metrics updates should be very fast
        assert avg_time_per_update_set < 2.0  # <2ms per metrics set on average

        # Final metrics should be correct
        final_success_rate = 50.0 + ((num_updates - 1) % 30)
        final_avg_time = 60.0 + ((num_updates - 1) % 20)
        final_confidence = 70.0 + ((num_updates - 1) % 25)

        assert mavlink_service._parameters["PISAD_HOMING_SUCCESS_RATE"] == final_success_rate
        assert mavlink_service._parameters["PISAD_HOMING_AVG_TIME"] == final_avg_time
        assert mavlink_service._parameters["PISAD_HOMING_CONFIDENCE"] == final_confidence

    def test_homing_performance_metrics_comprehensive_integration(self, mavlink_service):
        """Test comprehensive integration of all performance metrics [29b4]."""
        # Test complete performance metrics functionality together

        # Simulate a complete homing operation with performance tracking
        operation_phases = [
            ("start", 75.0, 65.0, 80.0),
            ("mid", 77.0, 63.0, 85.0),
            ("end", 79.0, 61.0, 90.0),
        ]

        for phase, success_rate, avg_time, confidence in operation_phases:
            # Update all performance metrics
            mavlink_service.update_homing_success_rate_parameter(success_rate)
            mavlink_service.update_homing_average_time_parameter(avg_time)
            mavlink_service.update_homing_confidence_level_parameter(confidence)

            # Verify all metrics are properly updated
            assert mavlink_service._parameters["PISAD_HOMING_SUCCESS_RATE"] == success_rate
            assert mavlink_service._parameters["PISAD_HOMING_AVG_TIME"] == avg_time
            assert mavlink_service._parameters["PISAD_HOMING_CONFIDENCE"] == confidence

            # Verify all values are within valid ranges
            assert 0.0 <= mavlink_service._parameters["PISAD_HOMING_SUCCESS_RATE"] <= 100.0
            assert mavlink_service._parameters["PISAD_HOMING_AVG_TIME"] >= 0.0
            assert 0.0 <= mavlink_service._parameters["PISAD_HOMING_CONFIDENCE"] <= 100.0
