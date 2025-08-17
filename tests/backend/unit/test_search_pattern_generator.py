"""Unit tests for SearchPatternGenerator service.

Tests expanding square search patterns per PRD-FR2.
"""

import pytest

from src.backend.services.search_pattern_generator import SearchPatternGenerator


class TestSearchPatternGenerator:
    """Test search pattern generation service."""

    @pytest.fixture
    def pattern_generator(self):
        """Provide SearchPatternGenerator instance."""
        return SearchPatternGenerator()

    def test_pattern_generator_initialization(self, pattern_generator):
        """Test SearchPatternGenerator initializes correctly."""
        assert pattern_generator.spacing > 0
        assert pattern_generator.max_radius > 0
        assert isinstance(pattern_generator.waypoints, list)

    def test_generate_expanding_square_pattern(self, pattern_generator):
        """Test expanding square pattern generation per PRD-FR2."""
        center_lat, center_lon = 37.7749, -122.4194  # San Francisco

        waypoints = pattern_generator.generate_expanding_square(
            center_lat=center_lat,
            center_lon=center_lon,
            spacing=50,  # 50m between waypoints
            max_radius=200,  # 200m max radius
        )

        assert len(waypoints) > 0
        assert isinstance(waypoints, list)

        # First waypoint should be near center
        first_wp = waypoints[0]
        assert abs(first_wp["lat"] - center_lat) < 0.001
        assert abs(first_wp["lon"] - center_lon) < 0.001

    def test_pattern_velocity_configuration(self, pattern_generator):
        """Test search velocity configuration per PRD-FR2 (5-10 m/s)."""
        # Test valid velocity range
        for velocity in [5.0, 7.5, 10.0]:
            pattern_generator.set_search_velocity(velocity)
            assert pattern_generator.search_velocity == velocity

        # Test invalid velocities are rejected
        with pytest.raises(ValueError):
            pattern_generator.set_search_velocity(15.0)  # Too fast

        with pytest.raises(ValueError):
            pattern_generator.set_search_velocity(3.0)  # Too slow

    def test_search_area_boundaries(self, pattern_generator):
        """Test search area boundary enforcement."""
        # Define search boundaries
        boundary = {"north": 37.8000, "south": 37.7500, "east": -122.4000, "west": -122.4500}

        pattern_generator.set_boundaries(boundary)

        waypoints = pattern_generator.generate_expanding_square(
            center_lat=37.7749,
            center_lon=-122.4194,
            spacing=50,
            max_radius=500,  # Large radius that would exceed boundaries
        )

        # All waypoints should be within boundaries
        for wp in waypoints:
            assert boundary["south"] <= wp["lat"] <= boundary["north"]
            assert boundary["west"] <= wp["lon"] <= boundary["east"]

    def test_pattern_progress_tracking(self, pattern_generator):
        """Test search pattern progress tracking."""
        waypoints = pattern_generator.generate_expanding_square(
            center_lat=37.7749, center_lon=-122.4194, spacing=100, max_radius=300
        )

        # Initially 0% progress
        progress = pattern_generator.get_progress()
        assert progress == 0.0

        # Mark some waypoints as visited
        pattern_generator.mark_waypoint_visited(0)
        pattern_generator.mark_waypoint_visited(1)

        progress = pattern_generator.get_progress()
        assert 0 < progress <= 100

    def test_pattern_preview_generation(self, pattern_generator):
        """Test pattern preview for UI display."""
        waypoints = pattern_generator.generate_expanding_square(
            center_lat=37.7749, center_lon=-122.4194, spacing=75, max_radius=225
        )

        preview_data = pattern_generator.get_pattern_preview()

        assert isinstance(preview_data, dict)
        assert "waypoints" in preview_data
        assert "total_distance" in preview_data
        assert "estimated_time" in preview_data

    def test_mission_planner_compatibility(self, pattern_generator):
        """Test Mission Planner waypoint format compatibility."""
        waypoints = pattern_generator.generate_expanding_square(
            center_lat=37.7749, center_lon=-122.4194, spacing=60, max_radius=180
        )

        mp_format = pattern_generator.export_mission_planner_format()

        assert isinstance(mp_format, str)
        assert "QGC WPL" in mp_format or "VERSION" in mp_format

    def test_pattern_pause_resume(self, pattern_generator):
        """Test pattern pause and resume functionality."""
        waypoints = pattern_generator.generate_expanding_square(
            center_lat=37.7749, center_lon=-122.4194, spacing=80, max_radius=240
        )

        # Start pattern
        pattern_generator.start_pattern()
        assert pattern_generator.is_active() is True

        # Pause pattern
        pattern_generator.pause_pattern()
        assert pattern_generator.is_paused() is True

        # Resume pattern
        pattern_generator.resume_pattern()
        assert pattern_generator.is_active() is True

    def test_spiral_pattern_alternative(self, pattern_generator):
        """Test alternative spiral search pattern."""
        waypoints = pattern_generator.generate_spiral_pattern(
            center_lat=37.7749, center_lon=-122.4194, spacing=40, max_radius=160
        )

        assert len(waypoints) > 0

        # Check that spiral covers area efficiently
        total_distance = pattern_generator.calculate_total_distance(waypoints)
        assert total_distance > 0

    def test_adaptive_spacing_based_on_terrain(self, pattern_generator):
        """Test adaptive spacing based on terrain complexity."""
        # Test dense pattern for complex terrain
        dense_waypoints = pattern_generator.generate_adaptive_pattern(
            center_lat=37.7749, center_lon=-122.4194, terrain_complexity="high", base_spacing=50
        )

        # Test sparse pattern for simple terrain
        sparse_waypoints = pattern_generator.generate_adaptive_pattern(
            center_lat=37.7749, center_lon=-122.4194, terrain_complexity="low", base_spacing=50
        )

        # Dense should have more waypoints
        assert len(dense_waypoints) > len(sparse_waypoints)
