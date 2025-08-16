"""Telemetry recording service for field tests.

Records all telemetry data during test flights for post-analysis.
"""

import asyncio
import contextlib
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.services.mavlink_service import MAVLinkService
from backend.services.signal_processor import SignalProcessor
from backend.services.state_machine import StateMachine
from backend.utils.logging import get_logger
from src.backend.core.exceptions import (
    PISADException,
)

logger = get_logger(__name__)


@dataclass
class TelemetryFrame:
    """Single telemetry data frame."""

    timestamp: datetime
    # Position
    latitude: float
    longitude: float
    altitude: float
    relative_altitude: float
    # Velocity
    groundspeed: float
    airspeed: float
    climb_rate: float
    # Attitude
    roll: float
    pitch: float
    yaw: float
    heading: float
    # System
    battery_percent: float
    battery_voltage: float
    battery_current: float
    gps_satellites: int
    gps_hdop: float
    # Signal
    rssi_dbm: float
    snr_db: float
    beacon_detected: bool
    # State
    system_state: str
    armed: bool
    mode: str
    # Performance
    cpu_percent: float
    memory_percent: float
    temperature_c: float


class TelemetryRecorder:
    """Service for recording telemetry during field tests."""

    def __init__(
        self,
        mavlink_service: MAVLinkService,
        signal_processor: SignalProcessor,
        state_machine: StateMachine,
        record_rate_hz: float = 10.0,
    ):
        """Initialize telemetry recorder.

        Args:
            mavlink_service: MAVLink communication service
            signal_processor: Signal processing service
            state_machine: System state machine
            record_rate_hz: Recording rate in Hz
        """
        self.mavlink = mavlink_service
        self.signal_processor = signal_processor
        self.state_machine = state_machine
        self.record_rate_hz = record_rate_hz

        self.recording = False
        self.record_task: asyncio.Task | None = None
        self.telemetry_buffer: list[TelemetryFrame] = []
        self.max_buffer_size = 36000  # ~1 hour at 10Hz to prevent memory overflow
        self.session_id: str | None = None
        self.output_dir = Path("data/telemetry")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def start_recording(self, session_id: str | None = None) -> str:
        """Start telemetry recording.

        Args:
            session_id: Optional session identifier

        Returns:
            Session ID for this recording
        """
        if self.recording:
            logger.warning("Recording already in progress")
            return self.session_id or ""

        # Generate session ID if not provided
        if session_id is None:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            session_id = f"telemetry_{timestamp}"

        self.session_id = session_id
        self.telemetry_buffer.clear()
        self.recording = True

        # Start recording task
        self.record_task = asyncio.create_task(self._recording_loop())

        logger.info(f"Started telemetry recording: {session_id}")
        return session_id

    async def stop_recording(self) -> Path | None:
        """Stop telemetry recording and save data.

        Returns:
            Path to saved telemetry file
        """
        if not self.recording:
            logger.warning("No recording in progress")
            return None

        self.recording = False

        # Wait for recording task to complete
        if self.record_task:
            self.record_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.record_task

        # Save telemetry data
        if self.telemetry_buffer and self.session_id:
            file_path = await self.save_telemetry()
            logger.info(f"Stopped recording: {len(self.telemetry_buffer)} frames saved")
            return file_path

        logger.warning("No telemetry data to save")
        return None

    async def _recording_loop(self):
        """Main recording loop."""
        interval = 1.0 / self.record_rate_hz

        try:
            while self.recording:
                # Collect telemetry frame
                frame = await self._collect_telemetry_frame()
                if frame:
                    # Check buffer size limit
                    if len(self.telemetry_buffer) >= self.max_buffer_size:
                        logger.warning(
                            f"Telemetry buffer full ({self.max_buffer_size} frames), saving to disk"
                        )
                        await self.save_telemetry()
                        self.telemetry_buffer.clear()

                    self.telemetry_buffer.append(frame)

                    # Log periodically
                    if len(self.telemetry_buffer) % 100 == 0:
                        logger.debug(f"Recorded {len(self.telemetry_buffer)} frames")

                await asyncio.sleep(interval)

        except asyncio.CancelledError:
            logger.info("Recording loop cancelled")
        except PISADException as e:
            logger.error(f"Recording error: {e}")
            self.recording = False

    async def _collect_telemetry_frame(self) -> TelemetryFrame | None:
        """Collect single telemetry frame.

        Returns:
            Telemetry frame or None if collection failed
        """
        try:
            # Get MAVLink telemetry
            mav_data = await self.mavlink.get_telemetry()
            if not mav_data:
                return None

            # Get system performance metrics
            import psutil

            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent

            # Try to get temperature (Pi specific)
            temperature = 0.0
            try:
                with open("/sys/class/thermal/thermal_zone0/temp") as f:
                    temperature = float(f.read()) / 1000.0
            except Exception:
                pass

            # Create telemetry frame
            frame = TelemetryFrame(
                timestamp=datetime.now(UTC),
                # Position
                latitude=mav_data.get("latitude", 0.0),
                longitude=mav_data.get("longitude", 0.0),
                altitude=mav_data.get("altitude", 0.0),
                relative_altitude=mav_data.get("relative_altitude", 0.0),
                # Velocity
                groundspeed=mav_data.get("groundspeed", 0.0),
                airspeed=mav_data.get("airspeed", 0.0),
                climb_rate=mav_data.get("climb_rate", 0.0),
                # Attitude
                roll=mav_data.get("roll", 0.0),
                pitch=mav_data.get("pitch", 0.0),
                yaw=mav_data.get("yaw", 0.0),
                heading=mav_data.get("heading", 0.0),
                # System
                battery_percent=mav_data.get("battery_percent", 0.0),
                battery_voltage=mav_data.get("battery_voltage", 0.0),
                battery_current=mav_data.get("battery_current", 0.0),
                gps_satellites=mav_data.get("gps_satellites", 0),
                gps_hdop=mav_data.get("gps_hdop", 99.9),
                # Signal
                rssi_dbm=self.signal_processor.current_rssi,
                snr_db=self.signal_processor.current_snr,
                beacon_detected=self.signal_processor.beacon_detected,
                # State
                system_state=self.state_machine.current_state,
                armed=mav_data.get("armed", False),
                mode=mav_data.get("mode", "UNKNOWN"),
                # Performance
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                temperature_c=temperature,
            )

            return frame

        except PISADException as e:
            logger.error(f"Failed to collect telemetry frame: {e}")
            return None

    async def save_telemetry(self, format: str = "both") -> Path:
        """Save telemetry data to file.

        Args:
            format: Output format ("json", "csv", or "both")

        Returns:
            Path to saved file(s)
        """
        if not self.session_id:
            raise ValueError("No session ID set")

        base_path = self.output_dir / self.session_id

        # Save as JSON
        if format in ["json", "both"]:
            json_path = base_path.with_suffix(".json")
            await self._save_json(json_path)

        # Save as CSV
        if format in ["csv", "both"]:
            csv_path = base_path.with_suffix(".csv")
            await self._save_csv(csv_path)

        # Return primary path
        if format == "json":
            return base_path.with_suffix(".json")
        elif format == "csv":
            return base_path.with_suffix(".csv")
        else:
            return base_path.with_suffix(".json")

    async def _save_json(self, path: Path):
        """Save telemetry as JSON.

        Args:
            path: Output file path
        """
        data = {
            "session_id": self.session_id,
            "start_time": (
                self.telemetry_buffer[0].timestamp.isoformat() if self.telemetry_buffer else None
            ),
            "end_time": (
                self.telemetry_buffer[-1].timestamp.isoformat() if self.telemetry_buffer else None
            ),
            "frame_count": len(self.telemetry_buffer),
            "record_rate_hz": self.record_rate_hz,
            "frames": [asdict(frame) for frame in self.telemetry_buffer],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved JSON telemetry to {path}")

    async def _save_csv(self, path: Path):
        """Save telemetry as CSV.

        Args:
            path: Output file path
        """
        if not self.telemetry_buffer:
            return

        # Get field names from first frame
        field_names = list(asdict(self.telemetry_buffer[0]).keys())

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()

            for frame in self.telemetry_buffer:
                row = asdict(frame)
                # Convert datetime to string
                row["timestamp"] = row["timestamp"].isoformat()
                writer.writerow(row)

        logger.info(f"Saved CSV telemetry to {path}")

    def get_statistics(self) -> dict[str, Any]:
        """Get recording statistics.

        Returns:
            Statistics dictionary
        """
        if not self.telemetry_buffer:
            return {"frame_count": 0}

        # Calculate statistics
        rssi_values = [f.rssi_dbm for f in self.telemetry_buffer if f.rssi_dbm > -120]
        battery_values = [f.battery_percent for f in self.telemetry_buffer]
        altitude_values = [f.altitude for f in self.telemetry_buffer]
        speed_values = [f.groundspeed for f in self.telemetry_buffer]

        stats = {
            "frame_count": len(self.telemetry_buffer),
            "duration_s": (
                (
                    self.telemetry_buffer[-1].timestamp - self.telemetry_buffer[0].timestamp
                ).total_seconds()
                if len(self.telemetry_buffer) > 1
                else 0
            ),
            "rssi": {
                "min": min(rssi_values) if rssi_values else -120,
                "max": max(rssi_values) if rssi_values else -120,
                "avg": sum(rssi_values) / len(rssi_values) if rssi_values else -120,
            },
            "battery": {
                "min": min(battery_values) if battery_values else 0,
                "max": max(battery_values) if battery_values else 0,
                "avg": sum(battery_values) / len(battery_values) if battery_values else 0,
            },
            "altitude": {
                "min": min(altitude_values) if altitude_values else 0,
                "max": max(altitude_values) if altitude_values else 0,
            },
            "speed": {
                "min": min(speed_values) if speed_values else 0,
                "max": max(speed_values) if speed_values else 0,
                "avg": sum(speed_values) / len(speed_values) if speed_values else 0,
            },
            "detection_percentage": (
                sum(1 for f in self.telemetry_buffer if f.beacon_detected)
                / len(self.telemetry_buffer)
                * 100
            ),
        }

        return stats

    async def export_for_analysis(self, output_path: Path | None = None) -> Path:
        """Export telemetry data formatted for analysis tools.

        Args:
            output_path: Optional output path

        Returns:
            Path to exported file
        """
        if not self.telemetry_buffer:
            raise ValueError("No telemetry data to export")

        if output_path is None:
            output_path = self.output_dir / f"{self.session_id}_analysis.csv"

        # Rex/Sherlock: Use async I/O to avoid blocking event loop

        # Prepare header
        header = (
            "# PiSAD Telemetry Data Export\n"
            f"# Session: {self.session_id}\n"
            f"# Frames: {len(self.telemetry_buffer)}\n"
            "#\n"
        )

        # Write data
        field_names = [
            "timestamp",
            "latitude",
            "longitude",
            "altitude_m",
            "groundspeed_mps",
            "heading_deg",
            "rssi_dbm",
            "beacon_detected",
            "system_state",
            "battery_percent",
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            # Write header comments
            f.write(header)

            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()

            for frame in self.telemetry_buffer:
                writer.writerow(
                    {
                        "timestamp": frame.timestamp.isoformat(),
                        "latitude": frame.latitude,
                        "longitude": frame.longitude,
                        "altitude_m": frame.altitude,
                        "groundspeed_mps": frame.groundspeed,
                        "heading_deg": frame.heading,
                        "rssi_dbm": frame.rssi_dbm,
                        "beacon_detected": int(frame.beacon_detected),
                        "system_state": frame.system_state,
                        "battery_percent": frame.battery_percent,
                    }
                )

        logger.info(f"Exported analysis data to {output_path}")
        return output_path
