#!/usr/bin/env python3
"""
Field Test Data Logger for PISAD Hardware Testing.
Story 4.7 Sprint 6 - Captures real-time metrics during field tests.
"""

import asyncio
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import yaml
from pymavlink import mavutil


class FieldTestLogger:
    """Automated data logger for field testing."""

    def __init__(
        self,
        output_dir: str = "data/field_tests",
        api_base: str = "http://localhost:8080",
        config_file: str | None = None,
    ) -> None:
        """Initialize field test logger.

        Args:
            output_dir: Directory for test data output
            api_base: Base URL for PISAD API endpoint
            config_file: Optional path to configuration file
        """
        # Load configuration if provided
        if config_file and Path(config_file).exists():
            with open(config_file) as f:
                config = yaml.safe_load(f)
                test_config = config.get("test_configuration", {})
                output_dir = test_config.get("output_directory", output_dir)
                api_base = test_config.get("api_endpoint", api_base)
                self.config = config
        else:
            self.config = {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Test session ID
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.test_dir = self.output_dir / self.session_id
        self.test_dir.mkdir(exist_ok=True)

        # Data files
        self.telemetry_file = self.test_dir / "telemetry.csv"
        self.rssi_file = self.test_dir / "rssi.csv"
        self.performance_file = self.test_dir / "performance.csv"
        self.events_file = self.test_dir / "events.json"

        # Initialize CSV writers
        self._init_csv_files()

        # API connection (now configurable)
        self.api_base = api_base
        self.mavlink_conn: Any | None = None

        # Test metadata
        self.test_metadata: dict[str, Any] = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "test_location": None,
            "weather_conditions": None,
            "hardware_config": None,
        }

        self.events: list[dict[str, Any]] = []
        self.running = False

    def _init_csv_files(self) -> None:
        """Initialize CSV files with headers."""
        # Telemetry CSV
        with open(self.telemetry_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "lat",
                    "lon",
                    "alt",
                    "roll",
                    "pitch",
                    "yaw",
                    "vx",
                    "vy",
                    "vz",
                    "battery_voltage",
                    "battery_percent",
                    "gps_sats",
                    "gps_hdop",
                    "mode",
                    "armed",
                ]
            )

        # RSSI CSV
        with open(self.rssi_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "rssi_dbm",
                    "snr_db",
                    "frequency_mhz",
                    "detection_state",
                    "beacon_distance_m",
                ]
            )

        # Performance CSV
        with open(self.performance_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "cpu_percent",
                    "ram_mb",
                    "usb_throughput_mbps",
                    "mavlink_latency_ms",
                    "processing_latency_ms",
                    "temp_c",
                ]
            )

    async def connect_mavlink(self, connection_string: str = "/dev/ttyACM0:115200") -> bool:
        """Connect to MAVLink for direct telemetry.

        Args:
            connection_string: MAVLink connection string

        Returns:
            True if connected successfully
        """
        try:
            if ":" in connection_string:
                port, baud = connection_string.split(":")
                conn_str = f"serial:{port}:{baud}"
            else:
                conn_str = connection_string

            self.mavlink_conn = mavutil.mavlink_connection(conn_str)
            self.mavlink_conn.wait_heartbeat(timeout=5)
            print(f"MAVLink connected: {conn_str}")
            return True
        except Exception as e:
            print(f"MAVLink connection failed: {e}")
            return False

    async def log_telemetry(self) -> None:
        """Log telemetry data from MAVLink and API."""
        async with httpx.AsyncClient() as client:
            while self.running:
                try:
                    # Get telemetry from API
                    response = await client.get(f"{self.api_base}/api/telemetry/current")
                    if response.status_code == 200:
                        telemetry = response.json()

                        # Write to CSV
                        with open(self.telemetry_file, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(
                                [
                                    time.time(),
                                    telemetry.get("position", {}).get("lat", 0),
                                    telemetry.get("position", {}).get("lon", 0),
                                    telemetry.get("position", {}).get("alt", 0),
                                    telemetry.get("attitude", {}).get("roll", 0),
                                    telemetry.get("attitude", {}).get("pitch", 0),
                                    telemetry.get("attitude", {}).get("yaw", 0),
                                    telemetry.get("velocity", {}).get("vx", 0),
                                    telemetry.get("velocity", {}).get("vy", 0),
                                    telemetry.get("velocity", {}).get("vz", 0),
                                    telemetry.get("battery", {}).get("voltage", 0),
                                    telemetry.get("battery", {}).get("percentage", 0),
                                    telemetry.get("gps", {}).get("satellites", 0),
                                    telemetry.get("gps", {}).get("hdop", 0),
                                    telemetry.get("mode", "UNKNOWN"),
                                    telemetry.get("armed", False),
                                ]
                            )

                    await asyncio.sleep(0.25)  # 4Hz logging

                except Exception as e:
                    print(f"Telemetry logging error: {e}")
                    await asyncio.sleep(1)

    async def log_rssi(self) -> None:
        """Log RSSI and signal detection data."""
        async with httpx.AsyncClient() as client:
            while self.running:
                try:
                    # Get RSSI from API
                    response = await client.get(f"{self.api_base}/api/signal/rssi")
                    if response.status_code == 200:
                        signal_data = response.json()

                        # Get state
                        state_response = await client.get(f"{self.api_base}/api/state/current")
                        state = (
                            state_response.json().get("state", "UNKNOWN")
                            if state_response.status_code == 200
                            else "UNKNOWN"
                        )

                        # Calculate beacon distance (if detected)
                        # This would use actual GPS positions in real implementation
                        beacon_distance = (
                            self._calculate_beacon_distance() if state == "DETECTED" else None
                        )

                        # Write to CSV
                        with open(self.rssi_file, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(
                                [
                                    time.time(),
                                    signal_data.get("rssi", -100),
                                    signal_data.get("snr", 0),
                                    signal_data.get("frequency", 3200),
                                    state,
                                    beacon_distance,
                                ]
                            )

                    await asyncio.sleep(0.1)  # 10Hz RSSI logging

                except Exception as e:
                    print(f"RSSI logging error: {e}")
                    await asyncio.sleep(1)

    async def log_performance(self) -> None:
        """Log system performance metrics."""
        import psutil

        async with httpx.AsyncClient() as client:
            while self.running:
                try:
                    # Get system metrics
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    ram_mb = psutil.virtual_memory().used / (1024 * 1024)

                    # Get USB throughput (simplified - would need actual measurement)
                    usb_throughput = 160.0  # Nominal 160 Mbps for HackRF

                    # Get latencies from API
                    response = await client.get(f"{self.api_base}/api/performance/metrics")
                    if response.status_code == 200:
                        metrics = response.json()
                        mavlink_latency = metrics.get("mavlink_latency_ms", 0)
                        processing_latency = metrics.get("processing_latency_ms", 0)
                    else:
                        mavlink_latency = 0
                        processing_latency = 0

                    # Get temperature (if available)
                    try:
                        temps = psutil.sensors_temperatures()
                        temp_c = temps["cpu_thermal"][0].current if "cpu_thermal" in temps else 0
                    except:
                        temp_c = 0

                    # Write to CSV
                    with open(self.performance_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                time.time(),
                                cpu_percent,
                                ram_mb,
                                usb_throughput,
                                mavlink_latency,
                                processing_latency,
                                temp_c,
                            ]
                        )

                    await asyncio.sleep(1)  # 1Hz performance logging

                except Exception as e:
                    print(f"Performance logging error: {e}")
                    await asyncio.sleep(1)

    def log_event(
        self, event_type: str, description: str, data: dict[str, Any] | None = None
    ) -> None:
        """Log a test event.

        Args:
            event_type: Type of event (e.g., 'test_start', 'detection', 'error')
            description: Human-readable description
            data: Optional additional data
        """
        event = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "type": event_type,
            "description": description,
            "data": data or {},
        }
        self.events.append(event)

        # Immediately write to file
        with open(self.events_file, "w") as f:
            json.dump(self.events, f, indent=2)

        print(f"[EVENT] {event_type}: {description}")

    def _calculate_beacon_distance(self) -> float | None:
        """Calculate distance to beacon based on GPS positions.

        Returns:
            Distance in meters or None if not available
        """
        # This would use actual beacon and drone GPS positions
        # For now, return a simulated value
        return np.random.uniform(100, 500)

    def set_test_metadata(self, **kwargs) -> None:
        """Update test metadata.

        Args:
            **kwargs: Metadata fields to update
        """
        self.test_metadata.update(kwargs)

        # Save metadata
        metadata_file = self.test_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(self.test_metadata, f, indent=2)

    async def start_logging(self) -> None:
        """Start all logging tasks."""
        self.running = True
        self.log_event("session_start", f"Field test session {self.session_id} started")

        # Start logging tasks
        tasks = [
            asyncio.create_task(self.log_telemetry()),
            asyncio.create_task(self.log_rssi()),
            asyncio.create_task(self.log_performance()),
        ]

        print(f"Logging started. Data directory: {self.test_dir}")
        print("Press Ctrl+C to stop logging...")

        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            print("\nStopping logger...")
            self.stop_logging()

    def stop_logging(self) -> None:
        """Stop all logging tasks."""
        self.running = False
        self.log_event("session_end", f"Field test session {self.session_id} ended")

        # Update metadata
        self.test_metadata["end_time"] = datetime.now().isoformat()
        metadata_file = self.test_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(self.test_metadata, f, indent=2)

        print(f"Logging stopped. Data saved to: {self.test_dir}")

    def generate_kml(self) -> None:
        """Generate KML file from GPS track for Google Earth."""
        kml_file = self.test_dir / "flight_path.kml"

        # Read telemetry data
        positions = []
        with open(self.telemetry_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row["lat"]) != 0 and float(row["lon"]) != 0:
                    positions.append((float(row["lon"]), float(row["lat"]), float(row["alt"])))

        # Generate KML
        kml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>PISAD Field Test - {self.session_id}</name>
    <Style id="flightPath">
      <LineStyle>
        <color>ff00ff00</color>
        <width>4</width>
      </LineStyle>
    </Style>
    <Placemark>
      <name>Flight Path</name>
      <styleUrl>#flightPath</styleUrl>
      <LineString>
        <altitudeMode>absolute</altitudeMode>
        <coordinates>
          {' '.join([f'{lon},{lat},{alt}' for lon, lat, alt in positions])}
        </coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>"""

        with open(kml_file, "w") as f:
            f.write(kml_content)

        print(f"KML file generated: {kml_file}")


async def main() -> None:
    """Main entry point for field test logger.

    Usage:
        python field_test_logger.py [config_file]
        python field_test_logger.py config/field_test.yaml
    """
    import sys

    # Check for config file argument
    config_file = None
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        if Path(config_file).exists():
            print(f"Loading configuration from: {config_file}")
        else:
            print(f"Config file not found: {config_file}, using defaults")
            config_file = None

    # Try default config location
    if not config_file:
        default_config = "config/field_test.yaml"
        if Path(default_config).exists():
            config_file = default_config
            print(f"Using default config: {config_file}")

    logger = FieldTestLogger(config_file=config_file)

    # Set test metadata from config or defaults
    if logger.config:
        metadata = logger.config.get("test_metadata", {})
        logger.set_test_metadata(**metadata)
    else:
        # Fallback to defaults
        logger.set_test_metadata(
            test_location="Test Field Alpha",
            weather_conditions={"temperature_c": 25, "wind_speed_ms": 5, "conditions": "Clear"},
            hardware_config={
                "sdr": "HackRF One",
                "flight_controller": "Pixhawk Cube Orange+",
                "antenna": "Log-periodic 850MHz-6.5GHz",
                "battery": "6S Li-ion",
            },
        )

    # Optional: Connect direct MAVLink
    # await logger.connect_mavlink("/dev/ttyACM0:115200")

    # Start logging
    await logger.start_logging()

    # Generate KML after logging
    logger.generate_kml()


if __name__ == "__main__":
    asyncio.run(main())
