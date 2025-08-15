"""Comprehensive tests for search pattern generator."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from src.backend.services.search_pattern_generator import (
    CenterRadiusBoundary,
    CornerBoundary,
    PatternType,
    SearchPattern,
    SearchPatternGenerator,
    Waypoint,
    haversine_distance,
    offset_coordinate,
)


class TestWaypoint:
    """Test Waypoint dataclass."""

    def test_waypoint_creation(self):
        """Test creating a waypoint."""
        wp = Waypoint(index=1, latitude=40.7128, longitude=-74.0060, altitude=50.0)
        assert wp.index == 1
        assert wp.latitude == 40.7128
        assert wp.longitude == -74.0060
        assert wp.altitude == 50.0

    def test_waypoint_to_dict(self):
        """Test converting waypoint to dictionary."""
        wp = Waypoint(index=2, latitude=51.5074, longitude=-0.1278, altitude=100.0)
        result = wp.to_dict()

        assert result == {"index": 2, "lat": 51.5074, "lon": -0.1278, "alt": 100.0}


class TestCenterRadiusBoundary:
    """Test CenterRadiusBoundary class."""

    def test_boundary_creation(self):
        """Test creating a center-radius boundary."""
        boundary = CenterRadiusBoundary(center_lat=40.7128, center_lon=-74.0060, radius=1000.0)
        assert boundary.center_lat == 40.7128
        assert boundary.center_lon == -74.0060
        assert boundary.radius == 1000.0

    def test_contains_point_inside(self):
        """Test point inside boundary."""
        boundary = CenterRadiusBoundary(center_lat=40.7128, center_lon=-74.0060, radius=1000.0)
        # Point very close to center
        assert boundary.contains_point(40.7129, -74.0061) is True

    def test_contains_point_outside(self):
        """Test point outside boundary."""
        boundary = CenterRadiusBoundary(center_lat=40.7128, center_lon=-74.0060, radius=1000.0)
        # Point far from center
        assert boundary.contains_point(41.7128, -73.0060) is False

    def test_contains_point_on_boundary(self):
        """Test point exactly on boundary."""
        boundary = CenterRadiusBoundary(center_lat=0.0, center_lon=0.0, radius=1000.0)
        # Point approximately 1000m north
        lat_offset = 1000.0 / 111320  # meters to degrees
        assert boundary.contains_point(lat_offset, 0.0) is True


class TestCornerBoundary:
    """Test CornerBoundary class."""

    def test_boundary_creation(self):
        """Test creating a corner boundary."""
        corners = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
        boundary = CornerBoundary(corners=corners)
        assert boundary.corners == corners

    def test_contains_point_inside_square(self):
        """Test point inside square boundary."""
        corners = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
        boundary = CornerBoundary(corners=corners)
        assert boundary.contains_point(0.5, 0.5) is True

    def test_contains_point_outside_square(self):
        """Test point outside square boundary."""
        corners = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
        boundary = CornerBoundary(corners=corners)
        assert boundary.contains_point(2.0, 2.0) is False

    def test_contains_point_on_edge(self):
        """Test point on edge of boundary."""
        corners = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
        boundary = CornerBoundary(corners=corners)
        assert boundary.contains_point(0.5, 1.0) is True

    def test_contains_point_triangle(self):
        """Test point inside triangle boundary."""
        corners = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        boundary = CornerBoundary(corners=corners)
        assert boundary.contains_point(0.5, 0.25) is True
        assert boundary.contains_point(0.0, 1.0) is False

    def test_contains_point_complex_polygon(self):
        """Test point inside complex polygon."""
        corners = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0)]
        boundary = CornerBoundary(corners=corners)
        assert boundary.contains_point(0.5, 0.5) is True
        assert boundary.contains_point(1.5, 1.5) is False
        assert boundary.contains_point(0.5, 1.5) is True


class TestSearchPattern:
    """Test SearchPattern dataclass."""

    def test_pattern_creation(self):
        """Test creating a search pattern."""
        waypoints = [Waypoint(0, 40.0, -74.0, 50.0), Waypoint(1, 40.001, -74.001, 50.0)]
        boundary = CenterRadiusBoundary(40.0, -74.0, 1000.0)

        pattern = SearchPattern(
            id="test-id",
            pattern_type=PatternType.SPIRAL,
            spacing=75.0,
            velocity=7.5,
            boundary=boundary,
            waypoints=waypoints,
            total_waypoints=2,
            completed_waypoints=0,
            state="IDLE",
            progress_percent=0.0,
            estimated_time_remaining=100.0,
            created_at=datetime.now(UTC),
            started_at=None,
            paused_at=None,
        )

        assert pattern.id == "test-id"
        assert pattern.pattern_type == PatternType.SPIRAL
        assert pattern.spacing == 75.0
        assert pattern.velocity == 7.5
        assert len(pattern.waypoints) == 2

    def test_pattern_to_dict(self):
        """Test converting pattern to dictionary."""
        waypoints = [Waypoint(0, 40.0, -74.0, 50.0)]
        boundary = CenterRadiusBoundary(40.0, -74.0, 1000.0)
        now = datetime.now(UTC)

        pattern = SearchPattern(
            id="test-id",
            pattern_type=PatternType.LAWNMOWER,
            spacing=60.0,
            velocity=8.0,
            boundary=boundary,
            waypoints=waypoints,
            total_waypoints=1,
            completed_waypoints=0,
            state="EXECUTING",
            progress_percent=25.5,
            estimated_time_remaining=75.0,
            created_at=now,
            started_at=now,
            paused_at=None,
        )

        result = pattern.to_dict()

        assert result["pattern_id"] == "test-id"
        assert result["pattern_type"] == "lawnmower"
        assert result["spacing"] == 60.0
        assert result["velocity"] == 8.0
        assert len(result["waypoints"]) == 1
        assert result["state"] == "EXECUTING"
        assert result["progress_percent"] == 25.5
        assert result["started_at"] == now.isoformat()
        assert result["paused_at"] is None


class TestUtilityFunctions:
    """Test utility functions."""

    def test_haversine_distance_zero(self):
        """Test distance between same points."""
        distance = haversine_distance(40.7128, -74.0060, 40.7128, -74.0060)
        assert distance == 0.0

    def test_haversine_distance_known(self):
        """Test distance between known points."""
        # Approximately 1 degree latitude difference (~111km)
        distance = haversine_distance(40.0, -74.0, 41.0, -74.0)
        assert 110000 < distance < 112000

    def test_haversine_distance_longitude(self):
        """Test distance along longitude."""
        # 1 degree longitude at equator
        distance = haversine_distance(0.0, 0.0, 0.0, 1.0)
        assert 110000 < distance < 112000

    def test_offset_coordinate_north(self):
        """Test offsetting coordinate north."""
        new_lat, new_lon = offset_coordinate(40.0, -74.0, 1000.0, 0.0)
        assert new_lat > 40.0
        assert abs(new_lon - (-74.0)) < 1e-6

    def test_offset_coordinate_east(self):
        """Test offsetting coordinate east."""
        new_lat, new_lon = offset_coordinate(40.0, -74.0, 0.0, 1000.0)
        assert abs(new_lat - 40.0) < 1e-6
        assert new_lon > -74.0

    def test_offset_coordinate_diagonal(self):
        """Test offsetting coordinate diagonally."""
        new_lat, new_lon = offset_coordinate(40.0, -74.0, 1000.0, 1000.0)
        assert new_lat > 40.0
        assert new_lon > -74.0

    def test_offset_coordinate_negative(self):
        """Test offsetting coordinate with negative distances."""
        new_lat, new_lon = offset_coordinate(40.0, -74.0, -1000.0, -1000.0)
        assert new_lat < 40.0
        assert new_lon < -74.0


class TestSearchPatternGenerator:
    """Test SearchPatternGenerator class."""

    def test_generator_initialization(self):
        """Test initializing pattern generator."""
        generator = SearchPatternGenerator(default_altitude=75.0)
        assert generator.default_altitude == 75.0

    def test_generator_default_altitude(self):
        """Test default altitude."""
        generator = SearchPatternGenerator()
        assert generator.default_altitude == 50.0

    def test_generate_pattern_invalid_spacing_low(self):
        """Test pattern generation with invalid low spacing."""
        generator = SearchPatternGenerator()
        boundary = CenterRadiusBoundary(40.0, -74.0, 1000.0)

        with pytest.raises(ValueError, match="Spacing must be between 50-100m"):
            generator.generate_pattern(
                PatternType.SPIRAL, spacing=30.0, velocity=7.5, boundary=boundary
            )

    def test_generate_pattern_invalid_spacing_high(self):
        """Test pattern generation with invalid high spacing."""
        generator = SearchPatternGenerator()
        boundary = CenterRadiusBoundary(40.0, -74.0, 1000.0)

        with pytest.raises(ValueError, match="Spacing must be between 50-100m"):
            generator.generate_pattern(
                PatternType.SPIRAL, spacing=150.0, velocity=7.5, boundary=boundary
            )

    def test_generate_pattern_invalid_velocity_low(self):
        """Test pattern generation with invalid low velocity."""
        generator = SearchPatternGenerator()
        boundary = CenterRadiusBoundary(40.0, -74.0, 1000.0)

        with pytest.raises(ValueError, match="Velocity must be between 5-10 m/s"):
            generator.generate_pattern(
                PatternType.SPIRAL, spacing=75.0, velocity=3.0, boundary=boundary
            )

    def test_generate_pattern_invalid_velocity_high(self):
        """Test pattern generation with invalid high velocity."""
        generator = SearchPatternGenerator()
        boundary = CenterRadiusBoundary(40.0, -74.0, 1000.0)

        with pytest.raises(ValueError, match="Velocity must be between 5-10 m/s"):
            generator.generate_pattern(
                PatternType.SPIRAL, spacing=75.0, velocity=15.0, boundary=boundary
            )

    @patch("src.backend.services.search_pattern_generator.uuid.uuid4")
    @patch("src.backend.services.search_pattern_generator.datetime")
    def test_generate_expanding_square_pattern(self, mock_datetime, mock_uuid):
        """Test generating expanding square pattern."""
        mock_uuid.return_value = "test-uuid"
        mock_now = MagicMock()
        mock_datetime.now.return_value = mock_now

        generator = SearchPatternGenerator()
        boundary = CenterRadiusBoundary(40.0, -74.0, 500.0)

        pattern = generator.generate_pattern(
            PatternType.EXPANDING_SQUARE,
            spacing=50.0,
            velocity=7.5,
            boundary=boundary,
            altitude=100.0,
        )

        assert pattern.id == "test-uuid"
        assert pattern.pattern_type == PatternType.EXPANDING_SQUARE
        assert pattern.spacing == 50.0
        assert pattern.velocity == 7.5
        assert len(pattern.waypoints) > 0
        assert pattern.waypoints[0].altitude == 100.0
        assert pattern.state == "IDLE"
        assert pattern.created_at == mock_now

    @patch("src.backend.services.search_pattern_generator.uuid.uuid4")
    def test_generate_spiral_pattern(self, mock_uuid):
        """Test generating spiral pattern."""
        mock_uuid.return_value = "test-uuid"

        generator = SearchPatternGenerator()
        boundary = CenterRadiusBoundary(40.0, -74.0, 300.0)

        pattern = generator.generate_pattern(
            PatternType.SPIRAL, spacing=60.0, velocity=8.0, boundary=boundary
        )

        assert pattern.pattern_type == PatternType.SPIRAL
        assert len(pattern.waypoints) > 0
        # Spiral should start near center
        first_wp = pattern.waypoints[0]
        distance_from_center = haversine_distance(
            40.0, -74.0, first_wp.latitude, first_wp.longitude
        )
        assert distance_from_center < 100  # Within 100m of center

    @patch("src.backend.services.search_pattern_generator.uuid.uuid4")
    def test_generate_lawnmower_pattern(self, mock_uuid):
        """Test generating lawnmower pattern."""
        mock_uuid.return_value = "test-uuid"

        generator = SearchPatternGenerator()
        boundary = CenterRadiusBoundary(40.0, -74.0, 400.0)

        pattern = generator.generate_pattern(
            PatternType.LAWNMOWER, spacing=75.0, velocity=6.0, boundary=boundary
        )

        assert pattern.pattern_type == PatternType.LAWNMOWER
        assert len(pattern.waypoints) > 0
        # Lawnmower should have waypoints in parallel tracks
        assert pattern.estimated_time_remaining > 0

    def test_generate_pattern_with_polygon_boundary(self):
        """Test pattern generation with polygon boundary."""
        generator = SearchPatternGenerator()
        # Square boundary around origin
        corners = [(-0.01, -0.01), (-0.01, 0.01), (0.01, 0.01), (0.01, -0.01)]
        boundary = CornerBoundary(corners=corners)

        pattern = generator.generate_pattern(
            PatternType.LAWNMOWER, spacing=50.0, velocity=7.5, boundary=boundary
        )

        assert len(pattern.waypoints) > 0
        # All waypoints should be within boundary
        for wp in pattern.waypoints:
            assert boundary.contains_point(wp.latitude, wp.longitude)

    def test_expanding_square_with_polygon(self):
        """Test expanding square with polygon boundary."""
        generator = SearchPatternGenerator()
        corners = [(40.0, -74.0), (40.0, -73.99), (40.01, -73.99), (40.01, -74.0)]
        boundary = CornerBoundary(corners=corners)

        pattern = generator.generate_pattern(
            PatternType.EXPANDING_SQUARE, spacing=50.0, velocity=7.5, boundary=boundary
        )

        assert pattern.pattern_type == PatternType.EXPANDING_SQUARE
        assert len(pattern.waypoints) > 0
        # Center should be approximately at (40.005, -73.995)
        center_wp = pattern.waypoints[0]
        assert 40.004 < center_wp.latitude < 40.006
        assert -73.996 < center_wp.longitude < -73.994

    def test_spiral_with_polygon(self):
        """Test spiral with polygon boundary."""
        generator = SearchPatternGenerator()
        # Triangle boundary
        corners = [(40.0, -74.0), (40.01, -74.0), (40.005, -73.99)]
        boundary = CornerBoundary(corners=corners)

        pattern = generator.generate_pattern(
            PatternType.SPIRAL, spacing=50.0, velocity=7.5, boundary=boundary
        )

        assert pattern.pattern_type == PatternType.SPIRAL
        assert len(pattern.waypoints) > 0
        # All waypoints should be within triangle
        for wp in pattern.waypoints:
            assert boundary.contains_point(wp.latitude, wp.longitude)

    def test_calculate_total_distance_empty(self):
        """Test calculating distance with no waypoints."""
        generator = SearchPatternGenerator()
        distance = generator._calculate_total_distance([])
        assert distance == 0.0

    def test_calculate_total_distance_single(self):
        """Test calculating distance with single waypoint."""
        generator = SearchPatternGenerator()
        waypoints = [Waypoint(0, 40.0, -74.0, 50.0)]
        distance = generator._calculate_total_distance(waypoints)
        assert distance == 0.0

    def test_calculate_total_distance_multiple(self):
        """Test calculating distance with multiple waypoints."""
        generator = SearchPatternGenerator()
        waypoints = [
            Waypoint(0, 40.0, -74.0, 50.0),
            Waypoint(1, 40.01, -74.0, 50.0),
            Waypoint(2, 40.01, -74.01, 50.0),
        ]
        distance = generator._calculate_total_distance(waypoints)
        assert distance > 0
        # Should be roughly 2km (1km north + 1km west)
        assert 1500 < distance < 2500

    def test_validate_boundary_center_radius_valid(self):
        """Test validating valid center-radius boundary."""
        generator = SearchPatternGenerator()
        boundary = CenterRadiusBoundary(40.0, -74.0, 1000.0)
        assert generator.validate_boundary(boundary) is True

    def test_validate_boundary_invalid_latitude(self):
        """Test validating boundary with invalid latitude."""
        generator = SearchPatternGenerator()
        boundary = CenterRadiusBoundary(95.0, -74.0, 1000.0)

        with pytest.raises(ValueError, match="Invalid latitude"):
            generator.validate_boundary(boundary)

    def test_validate_boundary_invalid_longitude(self):
        """Test validating boundary with invalid longitude."""
        generator = SearchPatternGenerator()
        boundary = CenterRadiusBoundary(40.0, -200.0, 1000.0)

        with pytest.raises(ValueError, match="Invalid longitude"):
            generator.validate_boundary(boundary)

    def test_validate_boundary_invalid_radius(self):
        """Test validating boundary with invalid radius."""
        generator = SearchPatternGenerator()
        boundary = CenterRadiusBoundary(40.0, -74.0, -100.0)

        with pytest.raises(ValueError, match="Radius must be positive"):
            generator.validate_boundary(boundary)

    def test_validate_boundary_polygon_too_few_corners(self):
        """Test validating polygon with too few corners."""
        generator = SearchPatternGenerator()
        boundary = CornerBoundary(corners=[(40.0, -74.0), (40.1, -74.1)])

        with pytest.raises(ValueError, match="at least 3 corners"):
            generator.validate_boundary(boundary)

    def test_validate_boundary_polygon_invalid_coords(self):
        """Test validating polygon with invalid coordinates."""
        generator = SearchPatternGenerator()
        boundary = CornerBoundary(corners=[(40.0, -74.0), (40.1, -74.1), (100.0, -74.0)])

        with pytest.raises(ValueError, match="Invalid latitude"):
            generator.validate_boundary(boundary)

    def test_pattern_state_transitions(self):
        """Test pattern state transitions."""
        generator = SearchPatternGenerator()
        boundary = CenterRadiusBoundary(40.0, -74.0, 500.0)

        pattern = generator.generate_pattern(
            PatternType.SPIRAL, spacing=50.0, velocity=7.5, boundary=boundary
        )

        # Initial state
        assert pattern.state == "IDLE"
        assert pattern.completed_waypoints == 0
        assert pattern.progress_percent == 0.0

        # Simulate execution
        pattern.state = "EXECUTING"
        pattern.started_at = datetime.now(UTC)
        pattern.completed_waypoints = 5
        pattern.progress_percent = (5 / pattern.total_waypoints) * 100

        assert pattern.state == "EXECUTING"
        assert pattern.completed_waypoints == 5
        assert pattern.progress_percent > 0

    def test_pattern_time_estimation(self):
        """Test pattern time estimation."""
        generator = SearchPatternGenerator()
        boundary = CenterRadiusBoundary(40.0, -74.0, 500.0)

        pattern = generator.generate_pattern(
            PatternType.LAWNMOWER,
            spacing=50.0,
            velocity=5.0,  # 5 m/s
            boundary=boundary,
        )

        # Calculate expected time
        total_distance = generator._calculate_total_distance(pattern.waypoints)
        expected_time = total_distance / 5.0

        assert abs(pattern.estimated_time_remaining - expected_time) < 0.01

    def test_expanding_square_coverage(self):
        """Test expanding square covers area effectively."""
        generator = SearchPatternGenerator()
        boundary = CenterRadiusBoundary(40.0, -74.0, 200.0)

        pattern = generator.generate_pattern(
            PatternType.EXPANDING_SQUARE, spacing=50.0, velocity=7.5, boundary=boundary
        )

        # Check waypoints expand outward
        distances = []
        for wp in pattern.waypoints:
            dist = haversine_distance(40.0, -74.0, wp.latitude, wp.longitude)
            distances.append(dist)

        # Generally increasing distances (with some variation due to square shape)
        avg_first_half = sum(distances[: len(distances) // 2]) / (len(distances) // 2)
        avg_second_half = sum(distances[len(distances) // 2 :]) / (
            len(distances) - len(distances) // 2
        )
        assert avg_second_half >= avg_first_half

    def test_spiral_density(self):
        """Test spiral pattern density."""
        generator = SearchPatternGenerator()
        boundary = CenterRadiusBoundary(40.0, -74.0, 300.0)

        pattern = generator.generate_pattern(
            PatternType.SPIRAL, spacing=50.0, velocity=7.5, boundary=boundary
        )

        # Check waypoint spacing
        for i in range(1, min(10, len(pattern.waypoints))):
            dist = haversine_distance(
                pattern.waypoints[i - 1].latitude,
                pattern.waypoints[i - 1].longitude,
                pattern.waypoints[i].latitude,
                pattern.waypoints[i].longitude,
            )
            # Spacing should be roughly consistent
            assert 20 < dist < 90  # Allow some variation in spiral patterns

    def test_lawnmower_parallel_tracks(self):
        """Test lawnmower creates parallel tracks."""
        generator = SearchPatternGenerator()
        boundary = CenterRadiusBoundary(40.0, -74.0, 200.0)

        pattern = generator.generate_pattern(
            PatternType.LAWNMOWER, spacing=50.0, velocity=7.5, boundary=boundary
        )

        # Waypoints should alternate direction
        if len(pattern.waypoints) >= 4:
            # Check that tracks are roughly parallel
            wp = pattern.waypoints
            # First track direction
            track1_dir = wp[1].longitude - wp[0].longitude
            # Second track direction (should be opposite)
            track2_dir = wp[3].longitude - wp[2].longitude
            # Directions should be opposite
            assert track1_dir * track2_dir < 0
