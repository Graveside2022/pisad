"""Mission replay service for telemetry data playback."""

import asyncio
import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from src.backend.utils.logging import get_logger

logger = get_logger(__name__)


class PlaybackState(str, Enum):
    """Playback states for mission replay."""

    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"


class PlaybackSpeed(float, Enum):
    """Supported playback speeds."""

    QUARTER = 0.25
    HALF = 0.5
    NORMAL = 1.0
    DOUBLE = 2.0
    QUAD = 4.0
    MAX = 10.0


class ReplayEvent(BaseModel):
    """Replay event with synchronized data."""

    timestamp: datetime
    telemetry: dict[str, Any]
    signal_detections: list[dict[str, Any]] = Field(default_factory=list)
    state_changes: list[dict[str, Any]] = Field(default_factory=list)


class MissionReplayService:
    """Service for replaying mission telemetry and signal data."""

    def __init__(self) -> None:
        """Initialize the mission replay service."""
        self.state = PlaybackState.STOPPED
        self.speed = PlaybackSpeed.NORMAL
        self.current_position = 0
        self.timeline: list[ReplayEvent] = []
        self.mission_id: UUID | None = None
        self._playback_task: asyncio.Task[None] | None = None
        self._websocket_handler = None

    async def load_mission_data(
        self,
        mission_id: UUID,
        telemetry_file: Path,
        detections_file: Path | None = None,
        state_file: Path | None = None,
    ) -> bool:
        """
        Load telemetry and signal data for a mission.

        Args:
            mission_id: Mission identifier
            telemetry_file: Path to telemetry CSV file
            detections_file: Optional path to signal detections file
            state_file: Optional path to state machine transitions file

        Returns:
            True if data loaded successfully
        """
        try:
            self.mission_id = mission_id
            self.timeline = []
            self.current_position = 0

            # Load telemetry data
            telemetry_frames = await self._load_telemetry(telemetry_file)

            # Load signal detections if available
            detections = {}
            if detections_file and detections_file.exists():
                detections = await self._load_detections(detections_file)

            # Load state transitions if available
            state_changes = {}
            if state_file and state_file.exists():
                state_changes = await self._load_state_changes(state_file)

            # Build synchronized timeline
            for frame in telemetry_frames:
                timestamp = frame["timestamp"]
                # Find detections and state changes within a small time window (1 second)
                frame_detections = []
                frame_state_changes = []

                for det_time, det_list in detections.items():
                    if abs((det_time - timestamp).total_seconds()) < 1:
                        frame_detections.extend(det_list)

                for state_time, state_list in state_changes.items():
                    if abs((state_time - timestamp).total_seconds()) < 1:
                        frame_state_changes.extend(state_list)

                event = ReplayEvent(
                    timestamp=timestamp,
                    telemetry=frame,
                    signal_detections=frame_detections,
                    state_changes=frame_state_changes,
                )
                self.timeline.append(event)

            # Sort timeline by timestamp
            self.timeline.sort(key=lambda e: e.timestamp)

            logger.info(f"Loaded mission {mission_id} with {len(self.timeline)} events")
            return True

        except Exception as e:
            logger.error(f"Failed to load mission data: {e}")
            return False

    async def _load_telemetry(self, file_path: Path) -> list[dict[str, Any]]:
        """Load telemetry data from CSV file."""
        frames = []
        try:
            import csv

            with open(file_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Parse timestamp
                    row["timestamp"] = datetime.fromisoformat(row["timestamp"])
                    # Convert numeric fields
                    for key in [
                        "latitude",
                        "longitude",
                        "altitude",
                        "groundspeed",
                        "airspeed",
                        "climb_rate",
                        "roll",
                        "pitch",
                        "yaw",
                        "heading",
                        "battery_percent",
                        "battery_voltage",
                        "battery_current",
                        "rssi_dbm",
                        "snr_db",
                        "cpu_percent",
                        "memory_percent",
                        "temperature_c",
                    ]:
                        if row.get(key):
                            row[key] = float(row[key])
                    # Convert boolean fields
                    for key in ["beacon_detected", "armed"]:
                        if key in row:
                            row[key] = row[key].lower() == "true"
                    frames.append(row)
        except Exception as e:
            logger.error(f"Error loading telemetry: {e}")
            raise  # Re-raise the exception so load_mission_data can catch it
        return frames

    async def _load_detections(self, file_path: Path) -> dict[datetime, list[Any]]:
        """Load signal detection events."""
        detections: dict[datetime, list[Any]] = {}
        try:
            with open(file_path) as f:
                data = json.load(f)
                for detection in data:
                    timestamp = datetime.fromisoformat(detection["timestamp"])
                    if timestamp not in detections:
                        detections[timestamp] = []
                    detections[timestamp].append(detection)
        except Exception as e:
            logger.error(f"Error loading detections: {e}")
        return detections

    async def _load_state_changes(self, file_path: Path) -> dict[datetime, list[Any]]:
        """Load state machine transitions."""
        state_changes: dict[datetime, list[Any]] = {}
        try:
            with open(file_path) as f:
                data = json.load(f)
                for change in data:
                    timestamp = datetime.fromisoformat(change["timestamp"])
                    if timestamp not in state_changes:
                        state_changes[timestamp] = []
                    state_changes[timestamp].append(change)
        except Exception as e:
            logger.error(f"Error loading state changes: {e}")
        return state_changes

    async def play(self) -> None:
        """Start or resume playback."""
        if self.state == PlaybackState.PLAYING:
            return

        self.state = PlaybackState.PLAYING
        if self._playback_task:
            self._playback_task.cancel()

        self._playback_task = asyncio.create_task(self._playback_loop())
        logger.info(f"Started playback at position {self.current_position}")

    async def pause(self) -> None:
        """Pause playback."""
        if self.state != PlaybackState.PLAYING:
            return

        self.state = PlaybackState.PAUSED
        if self._playback_task:
            self._playback_task.cancel()
            self._playback_task = None
        logger.info(f"Paused playback at position {self.current_position}")

    async def stop(self) -> None:
        """Stop playback and reset position."""
        self.state = PlaybackState.STOPPED
        self.current_position = 0
        if self._playback_task:
            self._playback_task.cancel()
            self._playback_task = None
        logger.info("Stopped playback")

    async def seek(self, position: int) -> None:
        """
        Seek to specific position in timeline.

        Args:
            position: Timeline index to seek to
        """
        if 0 <= position < len(self.timeline):
            self.current_position = position
            await self._send_event(self.timeline[position])
            logger.info(f"Seeked to position {position}")

    async def set_speed(self, speed: PlaybackSpeed) -> None:
        """
        Set playback speed.

        Args:
            speed: Playback speed multiplier
        """
        self.speed = speed
        logger.info(f"Set playback speed to {speed}x")

    def set_websocket_handler(self, handler) -> None:
        """
        Set WebSocket handler for streaming events.

        Args:
            handler: WebSocket connection handler
        """
        self._websocket_handler = handler

    async def _playback_loop(self) -> None:
        """Main playback loop."""
        try:
            while self.state == PlaybackState.PLAYING and self.current_position < len(
                self.timeline
            ):
                current_event = self.timeline[self.current_position]
                await self._send_event(current_event)

                # Calculate delay to next event
                if self.current_position + 1 < len(self.timeline):
                    next_event = self.timeline[self.current_position + 1]
                    time_diff = (next_event.timestamp - current_event.timestamp).total_seconds()
                    delay = time_diff / self.speed
                    await asyncio.sleep(delay)

                self.current_position += 1

            # Playback completed
            if self.current_position >= len(self.timeline):
                await self.stop()
                logger.info("Playback completed")

        except asyncio.CancelledError:
            logger.debug("Playback loop cancelled")
        except Exception as e:
            logger.error(f"Error in playback loop: {e}")
            await self.stop()

    async def _send_event(self, event: ReplayEvent) -> None:
        """Send replay event via WebSocket."""
        if self._websocket_handler:
            try:
                await self._websocket_handler.send_json(
                    {
                        "type": "replay_event",
                        "timestamp": event.timestamp.isoformat(),
                        "telemetry": event.telemetry,
                        "signal_detections": event.signal_detections,
                        "state_changes": event.state_changes,
                        "position": self.current_position,
                        "total": len(self.timeline),
                    }
                )
            except Exception as e:
                logger.error(f"Failed to send WebSocket event: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get current replay status."""
        return {
            "mission_id": str(self.mission_id) if self.mission_id else None,
            "state": self.state.value,
            "speed": self.speed,
            "position": self.current_position,
            "total": len(self.timeline),
            "progress": (self.current_position / len(self.timeline) * 100 if self.timeline else 0),
        }

    def get_timeline_range(self) -> tuple[datetime, datetime] | None:
        """Get timeline start and end timestamps."""
        if not self.timeline:
            return None
        return (self.timeline[0].timestamp, self.timeline[-1].timestamp)
