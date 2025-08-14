"""Waypoint export service for Mission Planner and QGroundControl compatibility."""

from typing import Any

from src.backend.services.search_pattern_generator import Waypoint


class WaypointExporter:
    """Export waypoints to various ground control station formats."""

    @staticmethod
    def export_qgc_wpl(
        waypoints: list[Waypoint], home_lat: float = 0.0, home_lon: float = 0.0
    ) -> str:
        """Export waypoints in QGroundControl WPL format.

        QGC WPL format specification:
        - Header: QGC WPL 110
        - Fields: INDEX CURRENT COORD_FRAME COMMAND PARAM1 PARAM2 PARAM3 PARAM4 PARAM5/LAT PARAM6/LON PARAM7/ALT AUTOCONTINUE

        Args:
            waypoints: List of waypoints to export
            home_lat: Home latitude for first waypoint
            home_lon: Home longitude for first waypoint

        Returns:
            WPL format string
        """
        lines = ["QGC WPL 110"]

        # Add home position as first waypoint (index 0)
        # MAV_CMD_NAV_WAYPOINT = 16, MAV_FRAME_GLOBAL_RELATIVE_ALT = 3
        lines.append(f"0\t1\t0\t16\t0\t0\t0\t0\t{home_lat}\t{home_lon}\t0\t1")

        # Add search pattern waypoints
        for i, wp in enumerate(waypoints, start=1):
            # Current flag: 1 for first waypoint after home, 0 for others
            current = 0
            # MAV_CMD_NAV_WAYPOINT (16), MAV_FRAME_GLOBAL_RELATIVE_ALT (3)
            # PARAM1: Hold time in seconds (0)
            # PARAM2: Acceptance radius in meters (0 = default)
            # PARAM3: Pass through (0)
            # PARAM4: Yaw angle (0 = no change)
            line = f"{i}\t{current}\t3\t16\t0\t0\t0\t0\t{wp.latitude:.7f}\t{wp.longitude:.7f}\t{wp.altitude:.1f}\t1"
            lines.append(line)

        return "\n".join(lines)

    @staticmethod
    def export_mission_planner(waypoints: list[Waypoint]) -> str:
        """Export waypoints in Mission Planner format.

        Mission Planner waypoint format:
        - No header required
        - Fields: INDEX,CURRENT,COORD_FRAME,COMMAND,PARAM1,PARAM2,PARAM3,PARAM4,PARAM5,PARAM6,PARAM7,AUTOCONTINUE

        Args:
            waypoints: List of waypoints to export

        Returns:
            Mission Planner format string
        """
        lines = []

        for i, wp in enumerate(waypoints):
            # Current flag: 1 for first waypoint, 0 for others
            current = 1 if i == 0 else 0
            # MAV_CMD_NAV_WAYPOINT (16), MAV_FRAME_GLOBAL_RELATIVE_ALT (3)
            line = f"{i},{current},3,16,0,0,0,0,{wp.latitude:.7f},{wp.longitude:.7f},{wp.altitude:.1f},1"
            lines.append(line)

        return "\n".join(lines)

    @staticmethod
    def export_mavlink_json(waypoints: list[Waypoint]) -> dict[str, Any]:
        """Export waypoints as MAVLink JSON format.

        Args:
            waypoints: List of waypoints to export

        Returns:
            Dictionary with MAVLink mission items
        """
        mission = {
            "fileType": "Plan",
            "version": 1,
            "mission": {
                "cruiseSpeed": 10.0,
                "firmwareType": 12,  # ArduCopter
                "hoverSpeed": 5.0,
                "items": [],
                "plannedHomePosition": None,
                "vehicleType": 2,  # Copter
            },
        }

        # Add waypoints as mission items
        for i, wp in enumerate(waypoints):
            item = {
                "autoContinue": True,
                "command": 16,  # MAV_CMD_NAV_WAYPOINT
                "doJumpId": i + 1,
                "frame": 3,  # MAV_FRAME_GLOBAL_RELATIVE_ALT
                "params": [
                    0,  # Hold time
                    0,  # Acceptance radius
                    0,  # Pass through
                    0,  # Yaw
                    wp.latitude,
                    wp.longitude,
                    wp.altitude,
                ],
                "type": "SimpleItem",
            }
            mission["mission"]["items"].append(item)

        return mission

    @staticmethod
    def export_kml(waypoints: list[Waypoint], pattern_name: str = "Search Pattern") -> str:
        """Export waypoints as KML for Google Earth visualization.

        Args:
            waypoints: List of waypoints to export
            pattern_name: Name for the pattern in KML

        Returns:
            KML format string
        """
        kml = []
        kml.append('<?xml version="1.0" encoding="UTF-8"?>')
        kml.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
        kml.append("  <Document>")
        kml.append(f"    <name>{pattern_name}</name>")
        kml.append('    <Style id="waypointStyle">')
        kml.append("      <IconStyle>")
        kml.append("        <scale>0.8</scale>")
        kml.append("        <Icon>")
        kml.append(
            "          <href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href>"
        )
        kml.append("        </Icon>")
        kml.append("      </IconStyle>")
        kml.append("    </Style>")
        kml.append('    <Style id="pathStyle">')
        kml.append("      <LineStyle>")
        kml.append("        <color>ff0000ff</color>")
        kml.append("        <width>2</width>")
        kml.append("      </LineStyle>")
        kml.append("    </Style>")

        # Add waypoints as placemarks
        kml.append("    <Folder>")
        kml.append("      <name>Waypoints</name>")
        for i, wp in enumerate(waypoints):
            kml.append("      <Placemark>")
            kml.append(f"        <name>WP{i}</name>")
            kml.append("        <styleUrl>#waypointStyle</styleUrl>")
            kml.append("        <Point>")
            kml.append("          <altitudeMode>relativeToGround</altitudeMode>")
            kml.append(
                f"          <coordinates>{wp.longitude},{wp.latitude},{wp.altitude}</coordinates>"
            )
            kml.append("        </Point>")
            kml.append("      </Placemark>")
        kml.append("    </Folder>")

        # Add path
        kml.append("    <Placemark>")
        kml.append("      <name>Search Path</name>")
        kml.append("      <styleUrl>#pathStyle</styleUrl>")
        kml.append("      <LineString>")
        kml.append("        <altitudeMode>relativeToGround</altitudeMode>")
        kml.append("        <coordinates>")
        for wp in waypoints:
            kml.append(f"          {wp.longitude},{wp.latitude},{wp.altitude}")
        kml.append("        </coordinates>")
        kml.append("      </LineString>")
        kml.append("    </Placemark>")

        kml.append("  </Document>")
        kml.append("</kml>")

        return "\n".join(kml)

    @staticmethod
    def import_qgc_wpl(content: str) -> list[Waypoint]:
        """Import waypoints from QGC WPL format.

        Args:
            content: WPL format string

        Returns:
            List of imported waypoints

        Raises:
            ValueError: If format is invalid
        """
        lines = content.strip().split("\n")

        if not lines or not lines[0].startswith("QGC WPL"):
            raise ValueError("Invalid QGC WPL format - missing header")

        waypoints = []

        for line in lines[1:]:
            if not line.strip():
                continue

            parts = line.split("\t")
            if len(parts) < 12:
                continue

            try:
                index = int(parts[0])
                # Skip home position (index 0)
                if index == 0:
                    continue

                lat = float(parts[8])
                lon = float(parts[9])
                alt = float(parts[10])

                waypoints.append(
                    Waypoint(
                        index=index - 1,  # Adjust for skipped home position
                        latitude=lat,
                        longitude=lon,
                        altitude=alt,
                    )
                )
            except (ValueError, IndexError) as e:
                raise ValueError(f"Invalid waypoint line: {line}") from e

        return waypoints

    @staticmethod
    def validate_waypoints(waypoints: list[Waypoint]) -> bool:
        """Validate waypoint list for basic requirements.

        Args:
            waypoints: List of waypoints to validate

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        if not waypoints:
            raise ValueError("No waypoints provided")

        for i, wp in enumerate(waypoints):
            # Check latitude bounds
            if not -90 <= wp.latitude <= 90:
                raise ValueError(f"Waypoint {i}: Invalid latitude {wp.latitude}")

            # Check longitude bounds
            if not -180 <= wp.longitude <= 180:
                raise ValueError(f"Waypoint {i}: Invalid longitude {wp.longitude}")

            # Check altitude bounds (reasonable limits for drone)
            if not 0 <= wp.altitude <= 1000:
                raise ValueError(f"Waypoint {i}: Altitude {wp.altitude}m out of range (0-1000m)")

        return True
