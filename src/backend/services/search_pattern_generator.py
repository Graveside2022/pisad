"""Search pattern generation service for SAR operations."""

import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal


class PatternType(Enum):
    """Search pattern types."""

    EXPANDING_SQUARE = "expanding_square"
    SPIRAL = "spiral"
    LAWNMOWER = "lawnmower"


@dataclass
class Waypoint:
    """Waypoint for search pattern."""

    index: int
    latitude: float
    longitude: float
    altitude: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "lat": self.latitude,
            "lon": self.longitude,
            "alt": self.altitude,
        }


@dataclass
class CenterRadiusBoundary:
    """Search boundary defined by center point and radius."""

    center_lat: float
    center_lon: float
    radius: float  # meters

    def contains_point(self, lat: float, lon: float) -> bool:
        """Check if point is within boundary."""
        distance = haversine_distance(self.center_lat, self.center_lon, lat, lon)
        return distance <= self.radius


@dataclass
class CornerBoundary:
    """Search boundary defined by corner coordinates."""

    corners: list[tuple[float, float]]  # [(lat, lon), ...]

    def contains_point(self, lat: float, lon: float) -> bool:
        """Check if point is within polygon using ray casting algorithm."""
        n = len(self.corners)
        inside = False

        p1_lat, p1_lon = self.corners[0]
        for i in range(1, n + 1):
            p2_lat, p2_lon = self.corners[i % n]
            if (
                lon > min(p1_lon, p2_lon)
                and lon <= max(p1_lon, p2_lon)
                and lat <= max(p1_lat, p2_lat)
            ):
                if p1_lon != p2_lon:
                    lat_inters = (lon - p1_lon) * (p2_lat - p1_lat) / (p2_lon - p1_lon) + p1_lat
                if p1_lat == p2_lat or lat <= lat_inters:
                    inside = not inside
            p1_lat, p1_lon = p2_lat, p2_lon

        return inside


@dataclass
class SearchPattern:
    """Search pattern data model."""

    id: str
    pattern_type: PatternType
    spacing: float  # meters between tracks (50-100m)
    velocity: float  # search velocity in m/s (5-10)
    boundary: CenterRadiusBoundary | CornerBoundary
    waypoints: list[Waypoint]
    total_waypoints: int
    completed_waypoints: int
    state: Literal["IDLE", "EXECUTING", "PAUSED", "COMPLETED"]
    progress_percent: float
    estimated_time_remaining: float  # seconds
    created_at: datetime
    started_at: datetime | None
    paused_at: datetime | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "pattern_id": self.id,
            "pattern_type": self.pattern_type.value,
            "spacing": self.spacing,
            "velocity": self.velocity,
            "waypoints": [wp.to_dict() for wp in self.waypoints],
            "total_waypoints": self.total_waypoints,
            "completed_waypoints": self.completed_waypoints,
            "state": self.state,
            "progress_percent": self.progress_percent,
            "estimated_time_remaining": self.estimated_time_remaining,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "paused_at": self.paused_at.isoformat() if self.paused_at else None,
        }


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two GPS coordinates in meters."""
    R = 6371000  # Earth radius in meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def offset_coordinate(
    lat: float, lon: float, distance_north: float, distance_east: float
) -> tuple[float, float]:
    """Offset a coordinate by specified distances in meters."""
    R = 6371000  # Earth radius in meters

    # Calculate new latitude
    new_lat = lat + (distance_north / R) * (180 / math.pi)

    # Calculate new longitude
    new_lon = lon + (distance_east / R) * (180 / math.pi) / math.cos(math.radians(lat))

    return new_lat, new_lon


class SearchPatternGenerator:
    """Generate search patterns for SAR operations."""

    def __init__(self, default_altitude: float = 50.0):
        """Initialize pattern generator.

        Args:
            default_altitude: Default altitude for waypoints in meters
        """
        self.default_altitude = default_altitude

    def generate_pattern(
        self,
        pattern_type: PatternType,
        spacing: float,
        velocity: float,
        boundary: CenterRadiusBoundary | CornerBoundary,
        altitude: float | None = None,
    ) -> SearchPattern:
        """Generate search pattern based on type and parameters.

        Args:
            pattern_type: Type of search pattern
            spacing: Distance between search tracks in meters (50-100m)
            velocity: Search velocity in m/s (5-10)
            boundary: Search area boundary
            altitude: Flight altitude in meters (uses default if not specified)

        Returns:
            SearchPattern with generated waypoints

        Raises:
            ValueError: If parameters are out of valid ranges
        """
        # Validate parameters
        if not 50 <= spacing <= 100:
            raise ValueError(f"Spacing must be between 50-100m, got {spacing}m")
        if not 5 <= velocity <= 10:
            raise ValueError(f"Velocity must be between 5-10 m/s, got {velocity} m/s")

        alt = altitude or self.default_altitude

        # Generate waypoints based on pattern type
        if pattern_type == PatternType.EXPANDING_SQUARE:
            waypoints = self._generate_expanding_square(spacing, boundary, alt)
        elif pattern_type == PatternType.SPIRAL:
            waypoints = self._generate_spiral(spacing, boundary, alt)
        elif pattern_type == PatternType.LAWNMOWER:
            waypoints = self._generate_lawnmower(spacing, boundary, alt)
        else:
            raise ValueError(f"Unsupported pattern type: {pattern_type}")

        # Calculate total distance and estimated time
        total_distance = self._calculate_total_distance(waypoints)
        estimated_time = total_distance / velocity

        # Create search pattern
        pattern = SearchPattern(
            id=str(uuid.uuid4()),
            pattern_type=pattern_type,
            spacing=spacing,
            velocity=velocity,
            boundary=boundary,
            waypoints=waypoints,
            total_waypoints=len(waypoints),
            completed_waypoints=0,
            state="IDLE",
            progress_percent=0.0,
            estimated_time_remaining=estimated_time,
            created_at=datetime.now(UTC),
            started_at=None,
            paused_at=None,
        )

        return pattern

    def _generate_expanding_square(
        self, spacing: float, boundary: CenterRadiusBoundary | CornerBoundary, altitude: float
    ) -> list[Waypoint]:
        """Generate expanding square pattern waypoints.

        The pattern starts from center and expands outward in a square spiral.
        """
        waypoints = []
        index = 0

        # Get center point
        if isinstance(boundary, CenterRadiusBoundary):
            center_lat = boundary.center_lat
            center_lon = boundary.center_lon
            max_size = boundary.radius * 2
        else:
            # Calculate center of polygon
            lats = [c[0] for c in boundary.corners]
            lons = [c[1] for c in boundary.corners]
            center_lat = sum(lats) / len(lats)
            center_lon = sum(lons) / len(lons)
            # Estimate max size
            max_size = max(
                haversine_distance(min(lats), min(lons), max(lats), max(lons)),
                haversine_distance(min(lats), max(lons), max(lats), min(lons)),
            )

        # Start from center
        current_lat = center_lat
        current_lon = center_lon
        waypoints.append(Waypoint(index, current_lat, current_lon, altitude))
        index += 1

        # Generate expanding square
        leg_length = spacing
        direction = 0  # 0=East, 1=North, 2=West, 3=South

        while leg_length < max_size:
            # Do two legs of same length, then increase
            for _ in range(2):
                # Calculate movement based on direction
                distance_north: float
                distance_east: float
                if direction == 0:  # East
                    distance_north, distance_east = 0.0, leg_length
                elif direction == 1:  # North
                    distance_north, distance_east = leg_length, 0.0
                elif direction == 2:  # West
                    distance_north, distance_east = 0.0, -leg_length
                else:  # South
                    distance_north, distance_east = -leg_length, 0.0

                # Move to next waypoint
                new_lat, new_lon = offset_coordinate(
                    current_lat, current_lon, distance_north, distance_east
                )

                # Check if within boundary
                if boundary.contains_point(new_lat, new_lon):
                    waypoints.append(Waypoint(index, new_lat, new_lon, altitude))
                    index += 1
                    current_lat, current_lon = new_lat, new_lon

                # Turn left (counter-clockwise)
                direction = (direction + 1) % 4

            # Increase leg length
            leg_length += spacing

        return waypoints

    def _generate_spiral(
        self, spacing: float, boundary: CenterRadiusBoundary | CornerBoundary, altitude: float
    ) -> list[Waypoint]:
        """Generate Archimedean spiral pattern waypoints."""
        waypoints: list[Waypoint] = []
        index = 0

        # Get center and max radius
        if isinstance(boundary, CenterRadiusBoundary):
            center_lat = boundary.center_lat
            center_lon = boundary.center_lon
            max_radius = boundary.radius
        else:
            # Calculate center and max radius for polygon
            lats = [c[0] for c in boundary.corners]
            lons = [c[1] for c in boundary.corners]
            center_lat = sum(lats) / len(lats)
            center_lon = sum(lons) / len(lons)
            max_radius = max(
                haversine_distance(center_lat, center_lon, lat, lon)
                for lat, lon in boundary.corners
            )

        # Generate spiral points
        theta: float = 0.0  # Angle in radians
        theta_step = spacing / 50  # Adjust for point density

        while True:
            # Archimedean spiral: r = a + b*theta
            r = spacing * theta / (2 * math.pi)

            if r > max_radius:
                break

            # Convert polar to offset distances
            distance_north = r * math.cos(theta)
            distance_east = r * math.sin(theta)

            # Calculate waypoint position
            wp_lat, wp_lon = offset_coordinate(
                center_lat, center_lon, distance_north, distance_east
            )

            # Check if within boundary and sufficient distance from last waypoint
            if boundary.contains_point(wp_lat, wp_lon) and (
                not waypoints
                or haversine_distance(
                    waypoints[-1].latitude, waypoints[-1].longitude, wp_lat, wp_lon
                )
                >= spacing * 0.5
            ):
                waypoints.append(Waypoint(index, wp_lat, wp_lon, altitude))
                index += 1

            theta += theta_step

        return waypoints

    def _generate_lawnmower(
        self, spacing: float, boundary: CenterRadiusBoundary | CornerBoundary, altitude: float
    ) -> list[Waypoint]:
        """Generate lawnmower (back-and-forth) pattern waypoints."""
        waypoints = []
        index = 0

        # Get bounding box
        if isinstance(boundary, CenterRadiusBoundary):
            # Create square bounding box for circle
            min_lat = boundary.center_lat - (boundary.radius / 111320)
            max_lat = boundary.center_lat + (boundary.radius / 111320)
            min_lon = boundary.center_lon - (
                boundary.radius / (111320 * math.cos(math.radians(boundary.center_lat)))
            )
            max_lon = boundary.center_lon + (
                boundary.radius / (111320 * math.cos(math.radians(boundary.center_lat)))
            )
        else:
            # Get bounding box from polygon corners
            lats = [c[0] for c in boundary.corners]
            lons = [c[1] for c in boundary.corners]
            min_lat, max_lat = min(lats), max(lats)
            min_lon, max_lon = min(lons), max(lons)

        # Generate parallel tracks from south to north
        current_lat = min_lat
        direction = 1  # 1=East, -1=West

        while current_lat <= max_lat:
            # Find the actual start and end points within the boundary
            # Sample points along the longitude to find entry/exit points
            track_points = []
            
            # Sample 100 points along the longitude range
            for i in range(100):
                test_lon = min_lon + (max_lon - min_lon) * i / 99
                if boundary.contains_point(current_lat, test_lon):
                    track_points.append(test_lon)
            
            # If we have valid points on this latitude, add waypoints
            if len(track_points) >= 2:
                if direction == 1:
                    # Moving east
                    start_lon = min(track_points)
                    end_lon = max(track_points)
                else:
                    # Moving west
                    start_lon = max(track_points)
                    end_lon = min(track_points)
                
                # Add waypoints
                waypoints.append(Waypoint(index, current_lat, start_lon, altitude))
                index += 1
                waypoints.append(Waypoint(index, current_lat, end_lon, altitude))
                index += 1
                
                direction *= -1  # Reverse direction for next track

            # Move to next track
            current_lat += spacing / 111320  # Convert meters to degrees

        return waypoints

    def _calculate_total_distance(self, waypoints: list[Waypoint]) -> float:
        """Calculate total distance between all waypoints."""
        if len(waypoints) < 2:
            return 0.0

        total = 0.0
        for i in range(1, len(waypoints)):
            total += haversine_distance(
                waypoints[i - 1].latitude,
                waypoints[i - 1].longitude,
                waypoints[i].latitude,
                waypoints[i].longitude,
            )

        return total

    def validate_boundary(self, boundary: CenterRadiusBoundary | CornerBoundary) -> bool:
        """Validate boundary parameters.

        Returns:
            True if boundary is valid

        Raises:
            ValueError: If boundary is invalid
        """
        if isinstance(boundary, CenterRadiusBoundary):
            if not -90 <= boundary.center_lat <= 90:
                raise ValueError(f"Invalid latitude: {boundary.center_lat}")
            if not -180 <= boundary.center_lon <= 180:
                raise ValueError(f"Invalid longitude: {boundary.center_lon}")
            if boundary.radius <= 0:
                raise ValueError(f"Radius must be positive: {boundary.radius}")
        else:
            if len(boundary.corners) < 3:
                raise ValueError("Polygon must have at least 3 corners")
            for lat, lon in boundary.corners:
                if not -90 <= lat <= 90:
                    raise ValueError(f"Invalid latitude: {lat}")
                if not -180 <= lon <= 180:
                    raise ValueError(f"Invalid longitude: {lon}")

        return True
