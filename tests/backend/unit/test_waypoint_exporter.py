"""Tests for waypoint export service."""

import json

import pytest

from src.backend.services.search_pattern_generator import Waypoint
from src.backend.services.waypoint_exporter import WaypointExporter


class TestWaypointExporterQGC:
    """Test QGroundControl waypoint export."""

    def test_export_qgc_wpl_empty(self):
        """Test exporting empty waypoint list."""
        result = WaypointExporter.export_qgc_wpl([], home_lat=47.5, home_lon=-122.3)
        
        lines = result.split("\n")
        assert lines[0] == "QGC WPL 110"
        assert len(lines) == 2  # Header + home
        assert "47.5" in lines[1]
        assert "-122.3" in lines[1]

    def test_export_qgc_wpl_with_waypoints(self):
        """Test exporting waypoints in QGC WPL format."""
        waypoints = [
            Waypoint(index=0, latitude=47.501, longitude=-122.301, altitude=50.0),
            Waypoint(index=1, latitude=47.502, longitude=-122.302, altitude=60.0),
            Waypoint(index=2, latitude=47.503, longitude=-122.303, altitude=70.0),
        ]
        
        result = WaypointExporter.export_qgc_wpl(waypoints, home_lat=47.5, home_lon=-122.3)
        
        lines = result.split("\n")
        assert lines[0] == "QGC WPL 110"
        assert len(lines) == 5  # Header + home + 3 waypoints
        
        # Check home waypoint
        home_parts = lines[1].split("\t")
        assert home_parts[0] == "0"  # Index
        assert home_parts[1] == "1"  # Current
        assert "47.5" in home_parts[8]  # Lat
        assert "-122.3" in home_parts[9]  # Lon
        
        # Check first waypoint
        wp1_parts = lines[2].split("\t")
        assert wp1_parts[0] == "1"  # Index
        assert "47.5010000" in wp1_parts[8]  # Lat
        assert "-122.3010000" in wp1_parts[9]  # Lon
        assert "50.0" in wp1_parts[10]  # Alt

    def test_export_qgc_wpl_format_precision(self):
        """Test coordinate precision in QGC export."""
        waypoints = [
            Waypoint(index=0, latitude=47.123456789, longitude=-122.987654321, altitude=123.456),
        ]
        
        result = WaypointExporter.export_qgc_wpl(waypoints)
        
        lines = result.split("\n")
        wp_parts = lines[2].split("\t")
        
        # Check 7 decimal places for lat/lon
        assert "47.1234568" in wp_parts[8]
        assert "-122.9876543" in wp_parts[9]
        # Check 1 decimal place for altitude
        assert "123.5" in wp_parts[10]


class TestWaypointExporterMissionPlanner:
    """Test Mission Planner waypoint export."""

    def test_export_mission_planner_empty(self):
        """Test exporting empty waypoint list for Mission Planner."""
        result = WaypointExporter.export_mission_planner([])
        
        lines = result.split("\n")
        assert lines[0] == "QGC WPL 110"
        assert len(lines) == 1  # Header only

    def test_export_mission_planner_with_waypoints(self):
        """Test exporting waypoints in Mission Planner format."""
        waypoints = [
            Waypoint(index=0, latitude=47.501, longitude=-122.301, altitude=50.0),
            Waypoint(index=1, latitude=47.502, longitude=-122.302, altitude=60.0),
        ]
        
        result = WaypointExporter.export_mission_planner(waypoints)
        
        lines = result.split("\n")
        assert lines[0] == "QGC WPL 110"
        assert len(lines) == 3  # Header + 2 waypoints
        
        # Check first waypoint
        wp1_parts = lines[1].split("\t")
        assert wp1_parts[0] == "0"  # Index starts at 0
        assert wp1_parts[3] == "16"  # MAV_CMD_NAV_WAYPOINT
        assert "47.501" in lines[1]
        assert "-122.301" in lines[1]


class TestWaypointExporterKML:
    """Test KML format export."""

    def test_export_kml_empty(self):
        """Test exporting empty waypoint list as KML."""
        result = WaypointExporter.export_kml([], name="Test Mission")
        
        assert '<?xml version="1.0" encoding="UTF-8"?>' in result
        assert '<kml' in result
        assert '<name>Test Mission</name>' in result
        assert '<Folder>' in result
        assert '</kml>' in result

    def test_export_kml_with_waypoints(self):
        """Test exporting waypoints in KML format."""
        waypoints = [
            Waypoint(index=0, latitude=47.501, longitude=-122.301, altitude=50.0),
            Waypoint(index=1, latitude=47.502, longitude=-122.302, altitude=60.0),
        ]
        
        result = WaypointExporter.export_kml(waypoints, pattern_name="Search Pattern")
        
        assert '<name>Search Pattern</name>' in result
        assert '<Placemark>' in result
        assert '<name>Waypoint 1</name>' in result
        assert '<Point>' in result
        assert '<coordinates>-122.301,47.501,50.0</coordinates>' in result
        assert '<LineString>' in result
        assert '-122.301,47.501,50.0' in result
        assert '-122.302,47.502,60.0' in result

    def test_export_kml_custom_description(self):
        """Test KML export with custom description."""
        waypoints = [Waypoint(47.5, -122.3, 100.0)]
        
        result = WaypointExporter.export_kml(
            waypoints, 
            name="Test", 
            description="Custom description"
        )
        
        assert '<description>Custom description</description>' in result


class TestWaypointExporterGeoJSON:
    """Test GeoJSON format export."""

    def test_export_geojson_empty(self):
        """Test exporting empty waypoint list as GeoJSON."""
        result = WaypointExporter.export_geojson([])
        data = json.loads(result)
        
        assert data["type"] == "FeatureCollection"
        assert data["features"] == []

    def test_export_geojson_with_waypoints(self):
        """Test exporting waypoints in GeoJSON format."""
        waypoints = [
            Waypoint(47.501, -122.301, 50.0),
            Waypoint(47.502, -122.302, 60.0),
        ]
        
        result = WaypointExporter.export_geojson(waypoints)
        data = json.loads(result)
        
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) == 3  # 2 points + 1 linestring
        
        # Check first point
        point = data["features"][0]
        assert point["type"] == "Feature"
        assert point["geometry"]["type"] == "Point"
        assert point["geometry"]["coordinates"] == [-122.301, 47.501, 50.0]
        assert point["properties"]["name"] == "Waypoint 1"
        
        # Check linestring
        linestring = data["features"][2]
        assert linestring["geometry"]["type"] == "LineString"
        assert len(linestring["geometry"]["coordinates"]) == 2


class TestWaypointExporterCSV:
    """Test CSV format export."""

    def test_export_csv_empty(self):
        """Test exporting empty waypoint list as CSV."""
        result = WaypointExporter.export_csv([])
        
        lines = result.strip().split("\n")
        assert lines[0] == "index,latitude,longitude,altitude"
        assert len(lines) == 1  # Header only

    def test_export_csv_with_waypoints(self):
        """Test exporting waypoints in CSV format."""
        waypoints = [
            Waypoint(47.501, -122.301, 50.0),
            Waypoint(47.502, -122.302, 60.0),
            Waypoint(47.503, -122.303, 70.0),
        ]
        
        result = WaypointExporter.export_csv(waypoints)
        
        lines = result.strip().split("\n")
        assert lines[0] == "index,latitude,longitude,altitude"
        assert len(lines) == 4  # Header + 3 waypoints
        
        # Check first waypoint
        assert lines[1] == "1,47.501,-122.301,50.0"
        assert lines[2] == "2,47.502,-122.302,60.0"
        assert lines[3] == "3,47.503,-122.303,70.0"


class TestWaypointExporterUtilities:
    """Test utility methods."""

    def test_calculate_total_distance_empty(self):
        """Test calculating distance for empty waypoint list."""
        distance = WaypointExporter.calculate_total_distance([])
        assert distance == 0.0

    def test_calculate_total_distance_single(self):
        """Test calculating distance for single waypoint."""
        waypoints = [Waypoint(47.5, -122.3, 50.0)]
        distance = WaypointExporter.calculate_total_distance(waypoints)
        assert distance == 0.0

    def test_calculate_total_distance_multiple(self):
        """Test calculating total distance for multiple waypoints."""
        waypoints = [
            Waypoint(47.500, -122.300, 50.0),
            Waypoint(47.501, -122.301, 50.0),
            Waypoint(47.502, -122.302, 50.0),
        ]
        
        distance = WaypointExporter.calculate_total_distance(waypoints)
        
        # Distance should be positive and reasonable
        assert distance > 0
        assert distance < 1000  # Less than 1km for these close waypoints

    def test_estimate_flight_time(self):
        """Test flight time estimation."""
        waypoints = [
            Waypoint(47.500, -122.300, 50.0),
            Waypoint(47.510, -122.310, 50.0),
        ]
        
        # Default speed (5 m/s)
        time_default = WaypointExporter.estimate_flight_time(waypoints)
        assert time_default > 0
        
        # Custom speed (10 m/s) - should be half the time
        time_fast = WaypointExporter.estimate_flight_time(waypoints, speed_ms=10.0)
        assert time_fast > 0
        assert time_fast < time_default

    def test_validate_waypoints_empty(self):
        """Test validating empty waypoint list."""
        is_valid, errors = WaypointExporter.validate_waypoints([])
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_waypoints_valid(self):
        """Test validating valid waypoints."""
        waypoints = [
            Waypoint(47.5, -122.3, 50.0),
            Waypoint(47.6, -122.4, 100.0),
        ]
        
        is_valid, errors = WaypointExporter.validate_waypoints(waypoints)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_waypoints_invalid_coords(self):
        """Test validating waypoints with invalid coordinates."""
        waypoints = [
            Waypoint(91.0, -122.3, 50.0),  # Invalid latitude > 90
            Waypoint(47.5, -181.0, 50.0),  # Invalid longitude < -180
        ]
        
        is_valid, errors = WaypointExporter.validate_waypoints(waypoints)
        assert is_valid is False
        assert len(errors) >= 2
        assert any("latitude" in err.lower() for err in errors)
        assert any("longitude" in err.lower() for err in errors)

    def test_validate_waypoints_invalid_altitude(self):
        """Test validating waypoints with invalid altitude."""
        waypoints = [
            Waypoint(47.5, -122.3, -50.0),  # Negative altitude
            Waypoint(47.5, -122.3, 10000.0),  # Too high
        ]
        
        is_valid, errors = WaypointExporter.validate_waypoints(waypoints)
        assert is_valid is False
        assert len(errors) >= 1
        assert any("altitude" in err.lower() for err in errors)