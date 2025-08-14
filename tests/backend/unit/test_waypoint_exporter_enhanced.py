"""Enhanced tests for waypoint export service with complete coverage."""

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
        
        # Check first waypoint
        wp1_parts = lines[2].split("\t")
        assert wp1_parts[0] == "1"  # Index
        assert "47.5010000" in wp1_parts[8]  # Lat
        assert "-122.3010000" in wp1_parts[9]  # Lon
        assert "50.0" in wp1_parts[10]  # Alt

    def test_export_qgc_wpl_default_home(self):
        """Test QGC export with default home position."""
        waypoints = [
            Waypoint(index=0, latitude=47.501, longitude=-122.301, altitude=50.0),
        ]
        
        result = WaypointExporter.export_qgc_wpl(waypoints)
        
        lines = result.split("\n")
        home_parts = lines[1].split("\t")
        assert home_parts[8] == "0.0"  # Default home lat
        assert home_parts[9] == "0.0"  # Default home lon


class TestWaypointExporterMissionPlanner:
    """Test Mission Planner waypoint export."""

    def test_export_mission_planner_empty(self):
        """Test exporting empty waypoint list."""
        result = WaypointExporter.export_mission_planner([])
        assert result == ""

    def test_export_mission_planner_with_waypoints(self):
        """Test exporting waypoints in Mission Planner format."""
        waypoints = [
            Waypoint(index=0, latitude=47.501, longitude=-122.301, altitude=50.0),
            Waypoint(index=1, latitude=47.502, longitude=-122.302, altitude=60.0),
        ]
        
        result = WaypointExporter.export_mission_planner(waypoints)
        
        lines = result.split("\n")
        assert len(lines) == 2
        
        # Check first waypoint (current flag = 1)
        wp1_parts = lines[0].split(",")
        assert wp1_parts[0] == "0"  # Index
        assert wp1_parts[1] == "1"  # Current
        assert "47.5010000" in wp1_parts[8]  # Lat
        
        # Check second waypoint (current flag = 0)
        wp2_parts = lines[1].split(",")
        assert wp2_parts[0] == "1"  # Index
        assert wp2_parts[1] == "0"  # Not current

    def test_export_mission_planner_single_waypoint(self):
        """Test exporting single waypoint."""
        waypoints = [
            Waypoint(index=0, latitude=47.5, longitude=-122.3, altitude=100.0),
        ]
        
        result = WaypointExporter.export_mission_planner(waypoints)
        
        lines = result.split("\n")
        assert len(lines) == 1
        wp_parts = lines[0].split(",")
        assert wp_parts[1] == "1"  # Current flag for first waypoint


class TestWaypointExporterMAVLinkJSON:
    """Test MAVLink JSON export."""

    def test_export_mavlink_json_empty(self):
        """Test exporting empty waypoint list as JSON."""
        result = WaypointExporter.export_mavlink_json([])
        
        assert result["fileType"] == "Plan"
        assert result["version"] == 1
        assert result["mission"]["items"] == []
        assert result["mission"]["vehicleType"] == 2  # Copter

    def test_export_mavlink_json_with_waypoints(self):
        """Test exporting waypoints as MAVLink JSON."""
        waypoints = [
            Waypoint(index=0, latitude=47.501, longitude=-122.301, altitude=50.0),
            Waypoint(index=1, latitude=47.502, longitude=-122.302, altitude=60.0),
        ]
        
        result = WaypointExporter.export_mavlink_json(waypoints)
        
        assert result["fileType"] == "Plan"
        assert len(result["mission"]["items"]) == 2
        
        # Check first item
        item1 = result["mission"]["items"][0]
        assert item1["command"] == 16  # NAV_WAYPOINT
        assert item1["frame"] == 3  # GLOBAL_RELATIVE_ALT
        assert item1["params"][4] == 47.501  # Lat
        assert item1["params"][5] == -122.301  # Lon
        assert item1["params"][6] == 50.0  # Alt

    def test_export_mavlink_json_mission_metadata(self):
        """Test MAVLink JSON mission metadata."""
        result = WaypointExporter.export_mavlink_json([])
        
        mission = result["mission"]
        assert mission["cruiseSpeed"] == 10.0
        assert mission["hoverSpeed"] == 5.0
        assert mission["firmwareType"] == 12  # ArduCopter
        assert mission["vehicleType"] == 2  # Copter
        assert mission["plannedHomePosition"] is None


class TestWaypointExporterKML:
    """Test KML export."""

    def test_export_kml_empty(self):
        """Test exporting empty waypoint list as KML."""
        result = WaypointExporter.export_kml([])
        
        assert '<?xml version="1.0" encoding="UTF-8"?>' in result
        assert '<kml xmlns="http://www.opengis.net/kml/2.2">' in result
        assert '<name>Search Pattern</name>' in result
        assert '</kml>' in result

    def test_export_kml_with_waypoints(self):
        """Test exporting waypoints in KML format."""
        waypoints = [
            Waypoint(index=0, latitude=47.501, longitude=-122.301, altitude=50.0),
            Waypoint(index=1, latitude=47.502, longitude=-122.302, altitude=60.0),
        ]
        
        result = WaypointExporter.export_kml(waypoints, pattern_name="Test Pattern")
        
        assert '<name>Test Pattern</name>' in result
        assert '<Placemark>' in result
        assert '<name>WP0</name>' in result
        assert '<name>WP1</name>' in result
        assert '<coordinates>-122.301,47.501,50.0</coordinates>' in result
        assert '<LineString>' in result
        assert '<altitudeMode>relativeToGround</altitudeMode>' in result

    def test_export_kml_styles(self):
        """Test KML style definitions."""
        result = WaypointExporter.export_kml([])
        
        # Check waypoint style
        assert '<Style id="waypointStyle">' in result
        assert '<IconStyle>' in result
        assert 'placemark_circle.png' in result
        
        # Check path style
        assert '<Style id="pathStyle">' in result
        assert '<LineStyle>' in result
        assert '<color>ff0000ff</color>' in result

    def test_export_kml_path_generation(self):
        """Test KML path generation from waypoints."""
        waypoints = [
            Waypoint(index=0, latitude=47.500, longitude=-122.300, altitude=50.0),
            Waypoint(index=1, latitude=47.501, longitude=-122.301, altitude=50.0),
            Waypoint(index=2, latitude=47.502, longitude=-122.302, altitude=50.0),
        ]
        
        result = WaypointExporter.export_kml(waypoints)
        
        # Check path contains all waypoints
        assert '<name>Search Path</name>' in result
        assert '-122.300,47.500,50.0' in result
        assert '-122.301,47.501,50.0' in result
        assert '-122.302,47.502,50.0' in result


class TestWaypointImporter:
    """Test waypoint import functionality."""

    def test_import_qgc_wpl_valid(self):
        """Test importing valid QGC WPL format."""
        content = """QGC WPL 110
0	1	0	16	0	0	0	0	47.5	-122.3	0	1
1	0	3	16	0	0	0	0	47.501	-122.301	50.0	1
2	0	3	16	0	0	0	0	47.502	-122.302	60.0	1
"""
        
        waypoints = WaypointExporter.import_qgc_wpl(content)
        
        assert len(waypoints) == 2  # Excludes home position
        assert waypoints[0].latitude == 47.501
        assert waypoints[0].longitude == -122.301
        assert waypoints[0].altitude == 50.0
        assert waypoints[1].latitude == 47.502

    def test_import_qgc_wpl_no_header(self):
        """Test importing WPL without header."""
        content = "1	0	3	16	0	0	0	0	47.501	-122.301	50.0	1"
        
        with pytest.raises(ValueError, match="Invalid QGC WPL format"):
            WaypointExporter.import_qgc_wpl(content)

    def test_import_qgc_wpl_invalid_line(self):
        """Test importing WPL with invalid line."""
        content = """QGC WPL 110
0	1	0	16	0	0	0	0	47.5	-122.3	0	1
1	0	3	16	INVALID
"""
        
        with pytest.raises(ValueError, match="Invalid waypoint line"):
            WaypointExporter.import_qgc_wpl(content)

    def test_import_qgc_wpl_empty_lines(self):
        """Test importing WPL with empty lines."""
        content = """QGC WPL 110
0	1	0	16	0	0	0	0	47.5	-122.3	0	1

1	0	3	16	0	0	0	0	47.501	-122.301	50.0	1

"""
        
        waypoints = WaypointExporter.import_qgc_wpl(content)
        assert len(waypoints) == 1
        assert waypoints[0].latitude == 47.501


class TestWaypointValidation:
    """Test waypoint validation."""

    def test_validate_waypoints_valid(self):
        """Test validation with valid waypoints."""
        waypoints = [
            Waypoint(index=0, latitude=47.5, longitude=-122.3, altitude=50.0),
            Waypoint(index=1, latitude=-33.8, longitude=151.2, altitude=100.0),
        ]
        
        assert WaypointExporter.validate_waypoints(waypoints) is True

    def test_validate_waypoints_empty(self):
        """Test validation with empty list."""
        with pytest.raises(ValueError, match="No waypoints provided"):
            WaypointExporter.validate_waypoints([])

    def test_validate_waypoints_invalid_latitude(self):
        """Test validation with invalid latitude."""
        waypoints = [
            Waypoint(index=0, latitude=91.0, longitude=-122.3, altitude=50.0),
        ]
        
        with pytest.raises(ValueError, match="Invalid latitude"):
            WaypointExporter.validate_waypoints(waypoints)
        
        waypoints = [
            Waypoint(index=0, latitude=-91.0, longitude=-122.3, altitude=50.0),
        ]
        
        with pytest.raises(ValueError, match="Invalid latitude"):
            WaypointExporter.validate_waypoints(waypoints)

    def test_validate_waypoints_invalid_longitude(self):
        """Test validation with invalid longitude."""
        waypoints = [
            Waypoint(index=0, latitude=47.5, longitude=181.0, altitude=50.0),
        ]
        
        with pytest.raises(ValueError, match="Invalid longitude"):
            WaypointExporter.validate_waypoints(waypoints)
        
        waypoints = [
            Waypoint(index=0, latitude=47.5, longitude=-181.0, altitude=50.0),
        ]
        
        with pytest.raises(ValueError, match="Invalid longitude"):
            WaypointExporter.validate_waypoints(waypoints)

    def test_validate_waypoints_invalid_altitude(self):
        """Test validation with invalid altitude."""
        waypoints = [
            Waypoint(index=0, latitude=47.5, longitude=-122.3, altitude=-50.0),
        ]
        
        with pytest.raises(ValueError, match="Altitude.*out of range"):
            WaypointExporter.validate_waypoints(waypoints)
        
        waypoints = [
            Waypoint(index=0, latitude=47.5, longitude=-122.3, altitude=1001.0),
        ]
        
        with pytest.raises(ValueError, match="Altitude.*out of range"):
            WaypointExporter.validate_waypoints(waypoints)

    def test_validate_waypoints_boundary_values(self):
        """Test validation with boundary values."""
        waypoints = [
            Waypoint(index=0, latitude=90.0, longitude=180.0, altitude=0.0),
            Waypoint(index=1, latitude=-90.0, longitude=-180.0, altitude=1000.0),
        ]
        
        assert WaypointExporter.validate_waypoints(waypoints) is True


class TestWaypointExporterEdgeCases:
    """Test edge cases and error handling."""

    def test_export_with_high_precision_coordinates(self):
        """Test export with high precision coordinates."""
        waypoints = [
            Waypoint(index=0, latitude=47.123456789, longitude=-122.987654321, altitude=123.456),
        ]
        
        # QGC format
        qgc_result = WaypointExporter.export_qgc_wpl(waypoints)
        assert "47.1234568" in qgc_result or "47.123456" in qgc_result
        
        # Mission Planner format
        mp_result = WaypointExporter.export_mission_planner(waypoints)
        assert "47.123456" in mp_result
        
        # KML format
        kml_result = WaypointExporter.export_kml(waypoints)
        assert "47.123456789" in kml_result

    def test_export_large_waypoint_list(self):
        """Test export with large number of waypoints."""
        waypoints = [
            Waypoint(index=i, latitude=47.5 + i*0.001, longitude=-122.3 + i*0.001, altitude=50.0 + i)
            for i in range(100)
        ]
        
        # All formats should handle large lists
        qgc_result = WaypointExporter.export_qgc_wpl(waypoints)
        assert len(qgc_result.split("\n")) == 102  # Header + home + 100 waypoints
        
        mp_result = WaypointExporter.export_mission_planner(waypoints)
        assert len(mp_result.split("\n")) == 100
        
        json_result = WaypointExporter.export_mavlink_json(waypoints)
        assert len(json_result["mission"]["items"]) == 100
        
        kml_result = WaypointExporter.export_kml(waypoints)
        assert kml_result.count("<Placemark>") == 101  # 100 waypoints + 1 path

    def test_export_with_special_characters_in_name(self):
        """Test KML export with special characters in pattern name."""
        waypoints = [
            Waypoint(index=0, latitude=47.5, longitude=-122.3, altitude=50.0),
        ]
        
        # Test with XML special characters
        result = WaypointExporter.export_kml(waypoints, pattern_name="Search & Rescue <Test>")
        assert "<name>Search &amp; Rescue &lt;Test&gt;</name>" in result or "<name>Search & Rescue <Test></name>" in result