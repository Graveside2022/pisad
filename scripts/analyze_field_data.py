#!/usr/bin/env python3
"""
Field Test Data Analysis Script for PISAD.
Story 4.7 Sprint 6 - Processes and analyzes field test data.
"""

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml


class FieldDataAnalyzer:
    """Analyzes field test data and generates reports."""

    def __init__(self, test_dir: str, config_file: str | None = None) -> None:
        """Initialize analyzer with test directory.

        Args:
            test_dir: Directory containing test data
            config_file: Optional path to field test configuration
        """
        self.test_dir = Path(test_dir)

        # Data files
        self.telemetry_file = self.test_dir / "telemetry.csv"
        self.rssi_file = self.test_dir / "rssi.csv"
        self.performance_file = self.test_dir / "performance.csv"
        self.events_file = self.test_dir / "events.json"
        self.metadata_file = self.test_dir / "metadata.json"

        # Load test requirements from config
        self.requirements: dict[str, Any] = {}
        if config_file and Path(config_file).exists():
            with open(config_file) as f:
                config = yaml.safe_load(f)
                self.requirements = config.get("test_requirements", {})
        else:
            # Default requirements if no config
            self.requirements = {
                "critical": {
                    "detection_range_m": 500,
                    "false_positive_rate": 0.05,
                    "flight_time_minutes": 25,
                    "emergency_stop_ms": 500,
                },
                "performance": {
                    "cpu_usage_percent": 30,
                    "ram_usage_mb": 500,
                    "mavlink_latency_ms": 50,
                    "processing_latency_ms": 100,
                },
            }

        # Results
        self.results: dict[str, Any] = {}

    def load_metadata(self) -> dict[str, Any]:
        """Load test metadata.

        Returns:
            Test metadata dictionary
        """
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                return json.load(f)
        return {}

    def analyze_telemetry(self) -> dict[str, Any]:
        """Analyze telemetry data.

        Returns:
            Telemetry analysis results
        """
        results = {
            "total_distance_m": 0,
            "max_altitude_m": 0,
            "avg_altitude_m": 0,
            "flight_time_s": 0,
            "position_drift_m": 0,
            "battery_consumption": 0,
            "gps_quality": {},
        }

        positions = []
        altitudes = []
        battery_voltages = []
        gps_sats = []
        gps_hdops = []
        timestamps = []

        with open(self.telemetry_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamps.append(float(row["timestamp"]))
                positions.append((float(row["lat"]), float(row["lon"])))
                altitudes.append(float(row["alt"]))
                battery_voltages.append(float(row["battery_voltage"]))
                gps_sats.append(int(row["gps_sats"]))
                gps_hdops.append(float(row["gps_hdop"]))

        if timestamps:
            # Flight time
            results["flight_time_s"] = timestamps[-1] - timestamps[0]

            # Distance calculation
            total_distance = 0
            for i in range(1, len(positions)):
                dist = self._haversine_distance(positions[i - 1], positions[i])
                total_distance += dist
            results["total_distance_m"] = total_distance

            # Altitude stats
            results["max_altitude_m"] = max(altitudes)
            results["avg_altitude_m"] = np.mean(altitudes)

            # Position drift (for hover tests)
            if len(positions) > 10:
                hover_positions = positions[:10]  # First 10 samples
                center = np.mean(hover_positions, axis=0)
                drifts = [self._haversine_distance(pos, center) for pos in hover_positions]
                results["position_drift_m"] = max(drifts)

            # Battery consumption
            if battery_voltages:
                results["battery_consumption"] = battery_voltages[0] - battery_voltages[-1]

            # GPS quality
            results["gps_quality"] = {
                "avg_satellites": np.mean(gps_sats),
                "min_satellites": min(gps_sats),
                "avg_hdop": np.mean(gps_hdops),
                "max_hdop": max(gps_hdops),
            }

        return results

    def analyze_rssi(self) -> dict[str, Any]:
        """Analyze RSSI and detection data.

        Returns:
            RSSI analysis results
        """
        results = {
            "max_detection_range_m": 0,
            "avg_rssi_dbm": 0,
            "detection_success_rate": 0,
            "false_positives": 0,
            "mean_time_to_detection_s": 0,
            "signal_statistics": {},
        }

        rssi_values = []
        snr_values = []
        detection_states = []
        beacon_distances = []
        timestamps = []

        with open(self.rssi_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamps.append(float(row["timestamp"]))
                rssi_values.append(float(row["rssi_dbm"]))
                snr_values.append(float(row["snr_db"]))
                detection_states.append(row["detection_state"])
                if row["beacon_distance_m"]:
                    beacon_distances.append(float(row["beacon_distance_m"]))

        if rssi_values:
            # RSSI statistics
            results["avg_rssi_dbm"] = np.mean(rssi_values)
            results["signal_statistics"] = {
                "rssi_min": min(rssi_values),
                "rssi_max": max(rssi_values),
                "rssi_std": np.std(rssi_values),
                "snr_avg": np.mean(snr_values),
                "snr_max": max(snr_values),
            }

            # Detection range
            if beacon_distances:
                results["max_detection_range_m"] = max(beacon_distances)

            # Detection success rate
            detected_count = detection_states.count("DETECTED")
            total_count = len(detection_states)
            if total_count > 0:
                results["detection_success_rate"] = detected_count / total_count

            # False positives (detections when RSSI < threshold)
            threshold = -70  # dBm
            false_positives = sum(
                1
                for i, state in enumerate(detection_states)
                if state == "DETECTED" and rssi_values[i] < threshold
            )
            results["false_positives"] = false_positives

            # Mean time to detection
            detection_times = []
            in_detection = False
            detection_start = 0

            for i, state in enumerate(detection_states):
                if state == "SEARCHING" and not in_detection:
                    detection_start = timestamps[i]
                    in_detection = True
                elif state == "DETECTED" and in_detection:
                    detection_times.append(timestamps[i] - detection_start)
                    in_detection = False

            if detection_times:
                results["mean_time_to_detection_s"] = np.mean(detection_times)

        return results

    def analyze_performance(self) -> dict[str, Any]:
        """Analyze system performance metrics.

        Returns:
            Performance analysis results
        """
        results = {
            "cpu_usage": {},
            "ram_usage": {},
            "latencies": {},
            "temperature": {},
            "requirements_met": {},
        }

        cpu_values = []
        ram_values = []
        mavlink_latencies = []
        processing_latencies = []
        temperatures = []

        with open(self.performance_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                cpu_values.append(float(row["cpu_percent"]))
                ram_values.append(float(row["ram_mb"]))
                mavlink_latencies.append(float(row["mavlink_latency_ms"]))
                processing_latencies.append(float(row["processing_latency_ms"]))
                temperatures.append(float(row["temp_c"]))

        if cpu_values:
            # CPU statistics
            results["cpu_usage"] = {
                "avg": np.mean(cpu_values),
                "max": max(cpu_values),
                "p95": np.percentile(cpu_values, 95),
            }

            # RAM statistics
            results["ram_usage"] = {
                "avg": np.mean(ram_values),
                "max": max(ram_values),
                "p95": np.percentile(ram_values, 95),
            }

            # Latency statistics
            results["latencies"] = {
                "mavlink_avg_ms": np.mean(mavlink_latencies),
                "mavlink_max_ms": max(mavlink_latencies),
                "processing_avg_ms": np.mean(processing_latencies),
                "processing_max_ms": max(processing_latencies),
            }

            # Temperature statistics
            if any(t > 0 for t in temperatures):
                valid_temps = [t for t in temperatures if t > 0]
                results["temperature"] = {"avg_c": np.mean(valid_temps), "max_c": max(valid_temps)}

            # Check requirements
            results["requirements_met"] = {
                "cpu_under_30_percent": results["cpu_usage"]["avg"] < 30,
                "ram_under_500_mb": results["ram_usage"]["avg"] < 500,
                "mavlink_under_50_ms": results["latencies"]["mavlink_avg_ms"] < 50,
                "processing_under_100_ms": results["latencies"]["processing_avg_ms"] < 100,
            }

        return results

    def analyze_events(self) -> dict[str, Any]:
        """Analyze test events.

        Returns:
            Event analysis results
        """
        results = {
            "total_events": 0,
            "errors": 0,
            "warnings": 0,
            "emergency_stops": 0,
            "mode_changes": 0,
            "event_timeline": [],
        }

        if self.events_file.exists():
            with open(self.events_file) as f:
                events = json.load(f)

            results["total_events"] = len(events)

            for event in events:
                event_type = event.get("type", "")

                if "error" in event_type.lower():
                    results["errors"] += 1
                elif "warning" in event_type.lower():
                    results["warnings"] += 1
                elif "emergency" in event_type.lower():
                    results["emergency_stops"] += 1
                elif "mode" in event_type.lower():
                    results["mode_changes"] += 1

                # Build timeline
                results["event_timeline"].append(
                    {
                        "time": event.get("datetime", ""),
                        "type": event_type,
                        "description": event.get("description", ""),
                    }
                )

        return results

    def generate_plots(self) -> None:
        """Generate analysis plots."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Field Test Analysis - {self.test_dir.name}")

        # Plot 1: Altitude Profile
        timestamps = []
        altitudes = []
        with open(self.telemetry_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamps.append(float(row["timestamp"]))
                altitudes.append(float(row["alt"]))

        if timestamps:
            times = [(t - timestamps[0]) / 60 for t in timestamps]  # Convert to minutes
            axes[0, 0].plot(times, altitudes)
            axes[0, 0].set_xlabel("Time (min)")
            axes[0, 0].set_ylabel("Altitude (m)")
            axes[0, 0].set_title("Altitude Profile")
            axes[0, 0].grid(True)

        # Plot 2: RSSI Over Time
        rssi_times = []
        rssi_values = []
        with open(self.rssi_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rssi_times.append(float(row["timestamp"]))
                rssi_values.append(float(row["rssi_dbm"]))

        if rssi_times:
            times = [(t - rssi_times[0]) / 60 for t in rssi_times]
            axes[0, 1].plot(times, rssi_values)
            axes[0, 1].axhline(y=-70, color="r", linestyle="--", label="Detection Threshold")
            axes[0, 1].set_xlabel("Time (min)")
            axes[0, 1].set_ylabel("RSSI (dBm)")
            axes[0, 1].set_title("Signal Strength")
            axes[0, 1].legend()
            axes[0, 1].grid(True)

        # Plot 3: Battery Voltage
        battery_times = []
        battery_voltages = []
        with open(self.telemetry_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                battery_times.append(float(row["timestamp"]))
                battery_voltages.append(float(row["battery_voltage"]))

        if battery_times:
            times = [(t - battery_times[0]) / 60 for t in battery_times]
            axes[0, 2].plot(times, battery_voltages)
            axes[0, 2].axhline(y=19.2, color="y", linestyle="--", label="Low Voltage")
            axes[0, 2].axhline(y=18.0, color="r", linestyle="--", label="Critical")
            axes[0, 2].set_xlabel("Time (min)")
            axes[0, 2].set_ylabel("Voltage (V)")
            axes[0, 2].set_title("Battery Voltage")
            axes[0, 2].legend()
            axes[0, 2].grid(True)

        # Plot 4: CPU Usage
        cpu_times = []
        cpu_values = []
        with open(self.performance_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                cpu_times.append(float(row["timestamp"]))
                cpu_values.append(float(row["cpu_percent"]))

        if cpu_times:
            times = [(t - cpu_times[0]) / 60 for t in cpu_times]
            axes[1, 0].plot(times, cpu_values)
            axes[1, 0].axhline(y=30, color="r", linestyle="--", label="Target Max")
            axes[1, 0].set_xlabel("Time (min)")
            axes[1, 0].set_ylabel("CPU Usage (%)")
            axes[1, 0].set_title("CPU Performance")
            axes[1, 0].legend()
            axes[1, 0].grid(True)

        # Plot 5: GPS Quality
        gps_times = []
        gps_sats = []
        with open(self.telemetry_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                gps_times.append(float(row["timestamp"]))
                gps_sats.append(int(row["gps_sats"]))

        if gps_times:
            times = [(t - gps_times[0]) / 60 for t in gps_times]
            axes[1, 1].plot(times, gps_sats)
            axes[1, 1].axhline(y=8, color="r", linestyle="--", label="Min Required")
            axes[1, 1].set_xlabel("Time (min)")
            axes[1, 1].set_ylabel("Satellites")
            axes[1, 1].set_title("GPS Satellites")
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        # Plot 6: Flight Path (if GPS data available)
        lats = []
        lons = []
        with open(self.telemetry_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                lat = float(row["lat"])
                lon = float(row["lon"])
                if lat != 0 and lon != 0:
                    lats.append(lat)
                    lons.append(lon)

        if lats and lons:
            axes[1, 2].plot(lons, lats, "b-", alpha=0.5)
            axes[1, 2].plot(lons[0], lats[0], "go", markersize=10, label="Start")
            axes[1, 2].plot(lons[-1], lats[-1], "ro", markersize=10, label="End")
            axes[1, 2].set_xlabel("Longitude")
            axes[1, 2].set_ylabel("Latitude")
            axes[1, 2].set_title("Flight Path")
            axes[1, 2].legend()
            axes[1, 2].grid(True)

        plt.tight_layout()
        plot_file = self.test_dir / "analysis_plots.png"
        plt.savefig(plot_file, dpi=150)
        print(f"Plots saved to: {plot_file}")
        plt.close()

    def _haversine_distance(self, pos1: tuple[float, float], pos2: tuple[float, float]) -> float:
        """Calculate distance between two GPS coordinates.

        Args:
            pos1: (lat, lon) tuple
            pos2: (lat, lon) tuple

        Returns:
            Distance in meters
        """
        lat1, lon1 = pos1
        lat2, lon2 = pos2

        R = 6371000  # Earth radius in meters
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)

        a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return R * c

    def generate_report(self) -> None:
        """Generate comprehensive test report."""
        # Run all analyses
        metadata = self.load_metadata()
        telemetry_results = self.analyze_telemetry()
        rssi_results = self.analyze_rssi()
        performance_results = self.analyze_performance()
        event_results = self.analyze_events()

        # Generate plots
        self.generate_plots()

        # Create report
        report = {
            "metadata": metadata,
            "telemetry": telemetry_results,
            "signal_detection": rssi_results,
            "performance": performance_results,
            "events": event_results,
            "summary": {
                "test_passed": self._evaluate_pass_fail(
                    telemetry_results, rssi_results, performance_results
                ),
                "key_metrics": {
                    "flight_time_minutes": telemetry_results.get("flight_time_s", 0) / 60,
                    "max_detection_range_m": rssi_results.get("max_detection_range_m", 0),
                    "detection_success_rate_percent": rssi_results.get("detection_success_rate", 0)
                    * 100,
                    "avg_cpu_percent": performance_results.get("cpu_usage", {}).get("avg", 0),
                    "avg_ram_mb": performance_results.get("ram_usage", {}).get("avg", 0),
                    "total_errors": event_results.get("errors", 0),
                },
            },
        }

        # Save report
        report_file = self.test_dir / "test_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        # Generate markdown summary
        self._generate_markdown_summary(report)

        print(f"Report generated: {report_file}")

    def _evaluate_pass_fail(
        self, telemetry: dict[str, Any], rssi: dict[str, Any], performance: dict[str, Any]
    ) -> bool:
        """Evaluate overall test pass/fail status.

        Args:
            telemetry: Telemetry analysis results
            rssi: RSSI analysis results
            performance: Performance analysis results

        Returns:
            True if test passed
        """
        # Use configured requirements
        critical = self.requirements.get("critical", {})
        perf_req = self.requirements.get("performance", {})

        # Calculate false positive threshold (5% of 1 hour = 3 events)
        max_false_positives = int(3600 * critical.get("false_positive_rate", 0.05) / 60)

        # Critical requirements check
        requirements = {
            "detection_range": rssi.get("max_detection_range_m", 0)
            >= critical.get("detection_range_m", 500),
            "false_positive_rate": rssi.get("false_positives", 0) < max_false_positives,
            "flight_time": telemetry.get("flight_time_s", 0)
            >= (critical.get("flight_time_minutes", 25) * 60),
            "cpu_usage": performance.get("cpu_usage", {}).get("avg", 100)
            < perf_req.get("cpu_usage_percent", 30),
            "ram_usage": performance.get("ram_usage", {}).get("avg", 1000)
            < perf_req.get("ram_usage_mb", 500),
            "mavlink_latency": performance.get("latencies", {}).get("mavlink_avg_ms", 100)
            < perf_req.get("mavlink_latency_ms", 50),
        }

        return all(requirements.values())

    def _generate_markdown_summary(self, report: dict[str, Any]) -> None:
        """Generate markdown summary report.

        Args:
            report: Complete test report dictionary
        """
        summary_file = self.test_dir / "SUMMARY.md"

        with open(summary_file, "w") as f:
            f.write(f"# Field Test Report - {report['metadata'].get('session_id', 'Unknown')}\n\n")
            f.write(f"**Date**: {report['metadata'].get('start_time', 'Unknown')}\n")
            f.write(f"**Location**: {report['metadata'].get('test_location', 'Unknown')}\n\n")

            f.write("## Test Result: ")
            if report["summary"]["test_passed"]:
                f.write("✅ **PASSED**\n\n")
            else:
                f.write("❌ **FAILED**\n\n")

            f.write("## Key Metrics\n\n")
            metrics = report["summary"]["key_metrics"]
            f.write(f"- **Flight Time**: {metrics['flight_time_minutes']:.1f} minutes\n")
            f.write(f"- **Max Detection Range**: {metrics['max_detection_range_m']:.0f} meters\n")
            f.write(
                f"- **Detection Success Rate**: {metrics['detection_success_rate_percent']:.1f}%\n"
            )
            f.write(f"- **Average CPU Usage**: {metrics['avg_cpu_percent']:.1f}%\n")
            f.write(f"- **Average RAM Usage**: {metrics['avg_ram_mb']:.0f} MB\n")
            f.write(f"- **Total Errors**: {metrics['total_errors']}\n\n")

            f.write("## Performance Requirements\n\n")
            perf = report["performance"].get("requirements_met", {})
            for req, met in perf.items():
                status = "✅" if met else "❌"
                f.write(f"- {status} {req.replace('_', ' ').title()}\n")

            f.write("\n## Files Generated\n\n")
            f.write("- `telemetry.csv` - Flight telemetry data\n")
            f.write("- `rssi.csv` - Signal strength data\n")
            f.write("- `performance.csv` - System performance metrics\n")
            f.write("- `events.json` - Test event log\n")
            f.write("- `test_report.json` - Complete analysis results\n")
            f.write("- `analysis_plots.png` - Performance visualization\n")
            f.write("- `flight_path.kml` - GPS track for Google Earth\n")

        print(f"Summary generated: {summary_file}")


def main() -> None:
    """Main entry point for analysis script.

    Usage:
        python analyze_field_data.py <test_directory> [config_file]
        python analyze_field_data.py data/field_tests/20250815_143022
        python analyze_field_data.py data/field_tests/20250815_143022 config/field_test.yaml
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyze_field_data.py <test_directory> [config_file]")
        print("Example: python analyze_field_data.py data/field_tests/20250815_143022")
        print(
            "Example: python analyze_field_data.py data/field_tests/20250815_143022 config/field_test.yaml"
        )
        sys.exit(1)

    test_dir = sys.argv[1]

    if not Path(test_dir).exists():
        print(f"Error: Test directory '{test_dir}' not found")
        sys.exit(1)

    # Check for config file
    config_file = None
    if len(sys.argv) > 2:
        config_file = sys.argv[2]
        if Path(config_file).exists():
            print(f"Using configuration: {config_file}")
        else:
            print(f"Config file not found: {config_file}, using defaults")
            config_file = None

    # Try default config if not specified
    if not config_file:
        default_config = "config/field_test.yaml"
        if Path(default_config).exists():
            config_file = default_config
            print(f"Using default config: {config_file}")

    print(f"Analyzing test data from: {test_dir}")

    analyzer = FieldDataAnalyzer(test_dir, config_file=config_file)
    analyzer.generate_report()

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
