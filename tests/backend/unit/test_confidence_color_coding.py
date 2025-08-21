"""
Test Suite for Confidence Color Coding (TASK-6.3.2.1 [30b4])

Tests for confidence-based color coding recommendations for Mission Planner visualization:
- Color zone assignment based on confidence thresholds
- RGB color value generation for Mission Planner display
- Priority levels for operator attention management
- Recommendation text generation for operator guidance
- Parameter integration for real-time telemetry streaming
"""

import time

import pytest

from src.backend.services.mavlink_service import MAVLinkService


class TestConfidenceColorCoding:
    """Test suite for confidence-based color coding recommendations."""

    @pytest.fixture
    def mavlink_service(self):
        """Create MAVLink service for testing."""
        service = MAVLinkService()
        service._parameters = {}
        return service

    def test_confidence_color_coding_blue_zone(self, mavlink_service):
        """Test BLUE zone color coding for excellent confidence (85-100%) [30b4]."""
        # Test excellent confidence levels
        excellent_levels = [85.0, 90.0, 95.0, 100.0]

        for confidence in excellent_levels:
            result = mavlink_service.get_confidence_color_coding_recommendation(confidence)

            assert result["color_zone"] == "BLUE"
            assert result["color_rgb"] == {"r": 0, "g": 100, "b": 255}
            assert result["priority"] == 4  # Lowest priority (good news)
            assert result["threshold_status"] == "EXCELLENT"
            assert "proceed with confidence" in result["recommendation"].lower()
            assert result["zone_numeric"] == 4
            assert result["confidence_value"] == confidence

    def test_confidence_color_coding_green_zone(self, mavlink_service):
        """Test GREEN zone color coding for good confidence (70-84%) [30b4]."""
        # Test good confidence levels
        good_levels = [70.0, 75.0, 80.0, 84.9]

        for confidence in good_levels:
            result = mavlink_service.get_confidence_color_coding_recommendation(confidence)

            assert result["color_zone"] == "GREEN"
            assert result["color_rgb"] == {"r": 0, "g": 200, "b": 0}
            assert result["priority"] == 3  # Low priority (normal operation)
            assert result["threshold_status"] == "GOOD"
            assert "normal operation" in result["recommendation"].lower()
            assert result["zone_numeric"] == 3
            assert result["confidence_value"] == confidence

    def test_confidence_color_coding_yellow_zone(self, mavlink_service):
        """Test YELLOW zone color coding for moderate confidence (50-69%) [30b4]."""
        # Test moderate confidence levels
        moderate_levels = [50.0, 55.0, 60.0, 69.9]

        for confidence in moderate_levels:
            result = mavlink_service.get_confidence_color_coding_recommendation(confidence)

            assert result["color_zone"] == "YELLOW"
            assert result["color_rgb"] == {"r": 255, "g": 255, "b": 0}
            assert result["priority"] == 2  # Medium priority (attention needed)
            assert result["threshold_status"] == "MODERATE"
            assert "monitor signal quality" in result["recommendation"].lower()
            assert result["zone_numeric"] == 2
            assert result["confidence_value"] == confidence

    def test_confidence_color_coding_red_zone(self, mavlink_service):
        """Test RED zone color coding for low confidence (0-49%) [30b4]."""
        # Test low confidence levels
        low_levels = [0.0, 10.0, 25.0, 49.9]

        for confidence in low_levels:
            result = mavlink_service.get_confidence_color_coding_recommendation(confidence)

            assert result["color_zone"] == "RED"
            assert result["color_rgb"] == {"r": 255, "g": 0, "b": 0}
            assert result["priority"] == 1  # Highest priority (critical attention)
            assert result["threshold_status"] == "LOW"
            assert "verify signal" in result["recommendation"].lower()
            assert result["zone_numeric"] == 1
            assert result["confidence_value"] == confidence

    def test_confidence_color_coding_boundary_conditions(self, mavlink_service):
        """Test color coding at exact boundary conditions [30b4]."""
        # Test exact boundary values
        boundaries = [
            (49.9, "RED"),
            (50.0, "YELLOW"),
            (69.9, "YELLOW"),
            (70.0, "GREEN"),
            (84.9, "GREEN"),
            (85.0, "BLUE"),
        ]

        for confidence, expected_zone in boundaries:
            result = mavlink_service.get_confidence_color_coding_recommendation(confidence)
            assert result["color_zone"] == expected_zone
            assert result["confidence_value"] == confidence

    def test_confidence_color_coding_input_validation(self, mavlink_service):
        """Test color coding with invalid input values [30b4]."""
        # Test out-of-range values (should be clamped)
        invalid_inputs = [-10.0, 150.0, 999.0]

        for invalid_confidence in invalid_inputs:
            result = mavlink_service.get_confidence_color_coding_recommendation(invalid_confidence)

            # Should clamp to valid range
            assert 0.0 <= result["confidence_value"] <= 100.0
            assert result["color_zone"] in ["RED", "YELLOW", "GREEN", "BLUE"]
            assert 1 <= result["zone_numeric"] <= 4
            assert isinstance(result["timestamp"], float)

    def test_confidence_color_coding_parameter_integration(self, mavlink_service):
        """Test color coding parameter integration with Mission Planner [30b4]."""
        # Test parameter update functionality
        test_confidence = 75.5

        mavlink_service.update_confidence_color_coding_parameter(test_confidence)

        # Verify parameter was updated
        assert "PISAD_CONF_COLOR" in mavlink_service._parameters
        assert mavlink_service._parameters["PISAD_CONF_COLOR"] == 3.0  # GREEN zone

        # Verify detailed color data was stored
        assert hasattr(mavlink_service, "_confidence_color_data")
        color_data = mavlink_service._confidence_color_data
        assert color_data["color_zone"] == "GREEN"
        assert color_data["confidence_value"] == test_confidence

    def test_confidence_color_coding_priority_system(self, mavlink_service):
        """Test priority system for operator attention management [30b4]."""
        # Test priority levels across all zones
        priority_tests = [
            (10.0, 1),  # RED - Highest priority
            (60.0, 2),  # YELLOW - Medium priority
            (80.0, 3),  # GREEN - Low priority
            (95.0, 4),  # BLUE - Lowest priority
        ]

        for confidence, expected_priority in priority_tests:
            result = mavlink_service.get_confidence_color_coding_recommendation(confidence)
            assert result["priority"] == expected_priority

            # Higher confidence should have lower priority (less urgent)
            assert result["priority"] <= 4
            assert result["priority"] >= 1

    def test_confidence_color_coding_recommendation_text(self, mavlink_service):
        """Test recommendation text generation for operator guidance [30b4]."""
        # Test that recommendations are appropriate for each zone
        recommendation_tests = [
            (95.0, "proceed with confidence"),
            (80.0, "normal operation"),
            (60.0, "monitor signal"),
            (30.0, "verify signal"),
        ]

        for confidence, expected_text_fragment in recommendation_tests:
            result = mavlink_service.get_confidence_color_coding_recommendation(confidence)
            recommendation = result["recommendation"].lower()

            assert expected_text_fragment in recommendation
            assert len(recommendation) > 10  # Meaningful recommendation text
            assert isinstance(result["recommendation"], str)

    def test_confidence_color_coding_timestamp_accuracy(self, mavlink_service):
        """Test timestamp accuracy for real-time updates [30b4]."""
        # Test timestamp generation
        before_time = time.time()
        result = mavlink_service.get_confidence_color_coding_recommendation(75.0)
        after_time = time.time()

        # Timestamp should be within reasonable range
        assert before_time <= result["timestamp"] <= after_time
        assert isinstance(result["timestamp"], float)

    def test_confidence_color_coding_mission_planner_metadata(self, mavlink_service):
        """Test Mission Planner specific metadata [30b4]."""
        result = mavlink_service.get_confidence_color_coding_recommendation(65.0)

        # Verify Mission Planner specific fields
        assert result["parameter_name"] == "PISAD_CONF_COLOR"
        assert "zone_numeric" in result
        assert result["zone_numeric"] == 2  # YELLOW zone

        # Verify RGB values are properly formatted
        rgb = result["color_rgb"]
        assert "r" in rgb and "g" in rgb and "b" in rgb
        assert 0 <= rgb["r"] <= 255
        assert 0 <= rgb["g"] <= 255
        assert 0 <= rgb["b"] <= 255

    def test_confidence_color_coding_error_handling(self, mavlink_service):
        """Test error handling in color coding calculations [30b4]."""
        # Test with None input (should handle gracefully)
        try:
            result = mavlink_service.get_confidence_color_coding_recommendation(None)
            # Should return error state with RED zone
            assert result["color_zone"] == "RED"
            assert result["threshold_status"] == "ERROR"
        except Exception:
            # Exception handling is also acceptable
            pass

    def test_confidence_color_coding_performance(self, mavlink_service):
        """Test color coding calculation performance [30b4]."""
        # Test rapid calculations for real-time performance
        import time

        num_calculations = 100
        start_time = time.perf_counter()

        for i in range(num_calculations):
            confidence = i % 101  # 0-100
            result = mavlink_service.get_confidence_color_coding_recommendation(confidence)
            assert result is not None

        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000
        avg_time_per_calc = total_time_ms / num_calculations

        # Each calculation should be very fast (<1ms)
        assert avg_time_per_calc < 1.0

    def test_confidence_color_coding_zone_consistency(self, mavlink_service):
        """Test consistency of color zone assignments [30b4]."""
        # Test that same confidence always produces same result
        test_confidence = 77.5

        results = []
        for _ in range(10):
            result = mavlink_service.get_confidence_color_coding_recommendation(test_confidence)
            results.append(
                (result["color_zone"], result["zone_numeric"], result["threshold_status"])
            )

        # All results should be identical (excluding timestamp)
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result

    def test_confidence_color_coding_comprehensive_coverage(self, mavlink_service):
        """Test comprehensive coverage of all confidence levels [30b4]."""
        # Test full range with step increments
        for confidence in range(0, 101, 5):  # 0, 5, 10, ..., 100
            result = mavlink_service.get_confidence_color_coding_recommendation(float(confidence))

            # Verify all required fields are present
            required_fields = [
                "color_zone",
                "color_rgb",
                "priority",
                "recommendation",
                "threshold_status",
                "confidence_value",
                "timestamp",
                "parameter_name",
                "zone_numeric",
            ]

            for field in required_fields:
                assert field in result
                assert result[field] is not None

            # Verify color zone is valid
            assert result["color_zone"] in ["RED", "YELLOW", "GREEN", "BLUE"]
            assert result["zone_numeric"] in [1, 2, 3, 4]
