"""Unit tests for search pattern generator."""

import pytest

from src.backend.services.search_pattern_generator import (
    CenterRadiusBoundary,
    CornerBoundary,
    PatternType,
    SearchPatternGenerator,
    Waypoint,
    haversine_distance,
    offset_coordinate,
)


class TestSearchPatternGenerator:
    """Test search pattern generation."""

    @pytest.fixture
    def generator(self):
        """Create pattern generator."""
        return SearchPatternGenerator(default_altitude=50.0)

    @pytest.fixture
    def center_boundary(self):
        """Create center-radius boundary."""
        return CenterRadiusBoundary(center_lat=37.7749, center_lon=-122.4194, radius=500.0)

    @pytest.fixture
    def corner_boundary(self):
        """Create corner boundary (square)."""
        return CornerBoundary(
            corners=[(37.770, -122.425), (37.780, -122.425), (37.780, -122.415), (37.770, -122.415)]
        )

    def test_haversine_distance(self):
        """Test distance calculation."""
        # Known distance: ~111km between 1 degree latitude
        dist = haversine_distance(0, 0, 1, 0)
        assert 111000 < dist < 112000

        # Same point
        dist = haversine_distance(37.7749, -122.4194, 37.7749, -122.4194)
        assert dist == 0

    def test_offset_coordinate(self):
        """Test coordinate offset calculation."""
        # Move 100m north
        new_lat, new_lon = offset_coordinate(37.7749, -122.4194, 100, 0)
        assert new_lat > 37.7749
        assert abs(new_lon - (-122.4194)) < 0.00001

        # Move 100m east
        new_lat, new_lon = offset_coordinate(37.7749, -122.4194, 0, 100)
        assert abs(new_lat - 37.7749) < 0.00001
        assert new_lon > -122.4194

    def test_center_radius_boundary_contains(self, center_boundary):
        """Test center-radius boundary containment."""
        # Center point should be inside
        assert center_boundary.contains_point(37.7749, -122.4194)

        # Point 400m away should be inside
        lat, lon = offset_coordinate(37.7749, -122.4194, 400, 0)
        assert center_boundary.contains_point(lat, lon)

        # Point 600m away should be outside
        lat, lon = offset_coordinate(37.7749, -122.4194, 600, 0)
        assert not center_boundary.contains_point(lat, lon)

    def test_corner_boundary_contains(self, corner_boundary):
        """Test polygon boundary containment."""
        # Center should be inside
        assert corner_boundary.contains_point(37.775, -122.420)

        # Points clearly inside the boundary (not on edges)
        assert corner_boundary.contains_point(37.775, -122.420)
        assert corner_boundary.contains_point(37.772, -122.422)

        # Outside points
        assert not corner_boundary.contains_point(37.765, -122.420)
        assert not corner_boundary.contains_point(37.775, -122.430)

    def test_generate_expanding_square_pattern(self, generator, center_boundary):
        """Test expanding square pattern generation."""
        pattern = generator.generate_pattern(
            pattern_type=PatternType.EXPANDING_SQUARE,
            spacing=75.0,
            velocity=7.0,
            boundary=center_boundary,
        )

        assert pattern.pattern_type == PatternType.EXPANDING_SQUARE
        assert pattern.spacing == 75.0
        assert pattern.velocity == 7.0
        assert len(pattern.waypoints) > 0
        assert pattern.total_waypoints == len(pattern.waypoints)
        assert pattern.state == "IDLE"
        assert pattern.completed_waypoints == 0
        assert pattern.progress_percent == 0.0

        # First waypoint should be at center
        first_wp = pattern.waypoints[0]
        assert abs(first_wp.latitude - center_boundary.center_lat) < 0.0001
        assert abs(first_wp.longitude - center_boundary.center_lon) < 0.0001
        assert first_wp.altitude == 50.0

        # All waypoints should be within boundary
        for wp in pattern.waypoints:
            assert center_boundary.contains_point(wp.latitude, wp.longitude)

    def test_generate_spiral_pattern(self, generator, center_boundary):
        """Test spiral pattern generation."""
        pattern = generator.generate_pattern(
            pattern_type=PatternType.SPIRAL,
            spacing=80.0,
            velocity=6.0,
            boundary=center_boundary,
            altitude=75.0,
        )

        assert pattern.pattern_type == PatternType.SPIRAL
        assert pattern.spacing == 80.0
        assert pattern.velocity == 6.0
        assert len(pattern.waypoints) > 0

        # All waypoints should have correct altitude
        for wp in pattern.waypoints:
            assert wp.altitude == 75.0
            assert center_boundary.contains_point(wp.latitude, wp.longitude)

    def test_generate_lawnmower_pattern(self, generator, corner_boundary):
        """Test lawnmower pattern generation."""
        pattern = generator.generate_pattern(
            pattern_type=PatternType.LAWNMOWER,
            spacing=100.0,
            velocity=10.0,
            boundary=corner_boundary,
        )

        assert pattern.pattern_type == PatternType.LAWNMOWER
        assert pattern.spacing == 100.0
        assert pattern.velocity == 10.0
        assert len(pattern.waypoints) > 0

        # All waypoints should be within boundary
        for wp in pattern.waypoints:
            assert corner_boundary.contains_point(wp.latitude, wp.longitude)

    def test_spacing_validation(self, generator, center_boundary):
        """Test spacing parameter validation."""
        # Too small
        with pytest.raises(ValueError, match="Spacing must be between 50-100m"):
            generator.generate_pattern(
                pattern_type=PatternType.EXPANDING_SQUARE,
                spacing=40.0,
                velocity=7.0,
                boundary=center_boundary,
            )

        # Too large
        with pytest.raises(ValueError, match="Spacing must be between 50-100m"):
            generator.generate_pattern(
                pattern_type=PatternType.EXPANDING_SQUARE,
                spacing=110.0,
                velocity=7.0,
                boundary=center_boundary,
            )

    def test_velocity_validation(self, generator, center_boundary):
        """Test velocity parameter validation."""
        # Too slow
        with pytest.raises(ValueError, match="Velocity must be between 5-10 m/s"):
            generator.generate_pattern(
                pattern_type=PatternType.EXPANDING_SQUARE,
                spacing=75.0,
                velocity=4.0,
                boundary=center_boundary,
            )

        # Too fast
        with pytest.raises(ValueError, match="Velocity must be between 5-10 m/s"):
            generator.generate_pattern(
                pattern_type=PatternType.EXPANDING_SQUARE,
                spacing=75.0,
                velocity=11.0,
                boundary=center_boundary,
            )

    def test_boundary_validation(self, generator):
        """Test boundary validation."""
        # Invalid latitude
        with pytest.raises(ValueError, match="Invalid latitude"):
            boundary = CenterRadiusBoundary(center_lat=91.0, center_lon=-122.4194, radius=500.0)
            generator.validate_boundary(boundary)

        # Invalid longitude
        with pytest.raises(ValueError, match="Invalid longitude"):
            boundary = CenterRadiusBoundary(center_lat=37.7749, center_lon=-181.0, radius=500.0)
            generator.validate_boundary(boundary)

        # Invalid radius
        with pytest.raises(ValueError, match="Radius must be positive"):
            boundary = CenterRadiusBoundary(center_lat=37.7749, center_lon=-122.4194, radius=-100.0)
            generator.validate_boundary(boundary)

        # Too few corners
        with pytest.raises(ValueError, match="at least 3 corners"):
            boundary = CornerBoundary(corners=[(37.770, -122.425), (37.780, -122.425)])
            generator.validate_boundary(boundary)

    def test_waypoint_to_dict(self):
        """Test waypoint serialization."""
        wp = Waypoint(index=5, latitude=37.7749, longitude=-122.4194, altitude=50.0)

        data = wp.to_dict()
        assert data["index"] == 5
        assert data["lat"] == 37.7749
        assert data["lon"] == -122.4194
        assert data["alt"] == 50.0

    def test_search_pattern_to_dict(self, generator, center_boundary):
        """Test search pattern serialization."""
        pattern = generator.generate_pattern(
            pattern_type=PatternType.EXPANDING_SQUARE,
            spacing=75.0,
            velocity=7.0,
            boundary=center_boundary,
        )

        data = pattern.to_dict()
        assert data["pattern_id"] == pattern.id
        assert data["pattern_type"] == "expanding_square"
        assert data["spacing"] == 75.0
        assert data["velocity"] == 7.0
        assert data["state"] == "IDLE"
        assert len(data["waypoints"]) == pattern.total_waypoints
        assert isinstance(data["created_at"], str)

    def test_estimated_time_calculation(self, generator, center_boundary):
        """Test estimated time calculation based on distance and velocity."""
        pattern = generator.generate_pattern(
            pattern_type=PatternType.EXPANDING_SQUARE,
            spacing=75.0,
            velocity=5.0,  # 5 m/s
            boundary=center_boundary,
        )

        # Calculate total distance manually
        total_distance = 0
        waypoints = pattern.waypoints
        for i in range(1, len(waypoints)):
            total_distance += haversine_distance(
                waypoints[i - 1].latitude,
                waypoints[i - 1].longitude,
                waypoints[i].latitude,
                waypoints[i].longitude,
            )

        # Estimated time should be distance / velocity
        expected_time = total_distance / 5.0
        assert abs(pattern.estimated_time_remaining - expected_time) < 1.0
