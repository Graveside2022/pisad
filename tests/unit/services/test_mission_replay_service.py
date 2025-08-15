"""Unit tests for mission replay service."""

import asyncio
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from src.backend.services.mission_replay_service import (
    MissionReplayService,

pytestmark = pytest.mark.serial
    PlaybackSpeed,
    PlaybackState,
)


@pytest.fixture
def replay_service():
    """Create a mission replay service instance."""
    return MissionReplayService()


@pytest.fixture
def temp_telemetry_file(tmp_path):
    """Create a temporary telemetry CSV file."""
    file_path = tmp_path / "telemetry.csv"
    headers = [
        "timestamp",
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
        "beacon_detected",
        "system_state",
        "armed",
        "mode",
        "cpu_percent",
        "memory_percent",
        "temperature_c",
    ]

    base_time = datetime.now()
    rows = []
    for i in range(10):
        timestamp = base_time + timedelta(seconds=i)
        rows.append(
            {
                "timestamp": timestamp.isoformat(),
                "latitude": 47.6062 + i * 0.0001,
                "longitude": -122.3321 + i * 0.0001,
                "altitude": 100.0 + i,
                "groundspeed": 5.0,
                "airspeed": 5.5,
                "climb_rate": 0.5,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "heading": 90.0,
                "battery_percent": 90.0 - i,
                "battery_voltage": 12.6,
                "battery_current": 5.0,
                "rssi_dbm": -70.0 + i,
                "snr_db": 10.0,
                "beacon_detected": i > 5,
                "system_state": "SEARCHING",
                "armed": True,
                "mode": "AUTO",
                "cpu_percent": 45.0,
                "memory_percent": 30.0,
                "temperature_c": 55.0,
            }
        )

    with open(file_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    return file_path


@pytest.fixture
def temp_detections_file(tmp_path):
    """Create a temporary detections JSON file."""
    file_path = tmp_path / "detections.json"
    base_time = datetime.now()

    detections = []
    for i in range(3, 8):
        timestamp = base_time + timedelta(seconds=i)
        detections.append(
            {
                "timestamp": timestamp.isoformat(),
                "frequency": 121500000,
                "rssi": -65.0 + i,
                "snr": 12.0,
                "confidence": 85.0,
                "location": {"lat": 47.6062 + i * 0.0001, "lon": -122.3321 + i * 0.0001},
                "state": "SEARCHING",
            }
        )

    with open(file_path, "w") as f:
        json.dump(detections, f)

    return file_path


@pytest.fixture
def temp_state_file(tmp_path):
    """Create a temporary state transitions JSON file."""
    file_path = tmp_path / "states.json"
    base_time = datetime.now()

    states = []
    for i in [0, 3, 6]:
        timestamp = base_time + timedelta(seconds=i)
        states.append(
            {
                "timestamp": timestamp.isoformat(),
                "from_state": "IDLE" if i == 0 else "SEARCHING",
                "to_state": "SEARCHING" if i == 0 else "APPROACHING",
                "trigger": "start" if i == 0 else "beacon_detected",
            }
        )

    with open(file_path, "w") as f:
        json.dump(states, f)

    return file_path


@pytest.mark.asyncio
async def test_load_mission_data(
    replay_service, temp_telemetry_file, temp_detections_file, temp_state_file
):
    """Test loading mission data from files."""
    mission_id = uuid4()

    result = await replay_service.load_mission_data(
        mission_id=mission_id,
        telemetry_file=temp_telemetry_file,
        detections_file=temp_detections_file,
        state_file=temp_state_file,
    )

    assert result is True
    assert replay_service.mission_id == mission_id
    assert len(replay_service.timeline) == 10
    assert replay_service.current_position == 0

    # Check first event
    first_event = replay_service.timeline[0]
    assert isinstance(first_event.timestamp, datetime)
    assert first_event.telemetry["latitude"] == pytest.approx(47.6062, rel=1e-4)
    assert len(first_event.state_changes) == 1

    # Check event with detections
    detection_event = replay_service.timeline[3]
    assert len(detection_event.signal_detections) == 1
    assert detection_event.signal_detections[0]["frequency"] == 121500000


@pytest.mark.asyncio
async def test_load_telemetry_only(replay_service, temp_telemetry_file):
    """Test loading only telemetry data without detections or states."""
    mission_id = uuid4()

    result = await replay_service.load_mission_data(
        mission_id=mission_id, telemetry_file=temp_telemetry_file
    )

    assert result is True
    assert len(replay_service.timeline) == 10
    assert all(len(e.signal_detections) == 0 for e in replay_service.timeline)
    assert all(len(e.state_changes) == 0 for e in replay_service.timeline)


@pytest.mark.asyncio
async def test_playback_controls(replay_service, temp_telemetry_file):
    """Test play, pause, stop controls."""
    mission_id = uuid4()
    await replay_service.load_mission_data(
        mission_id=mission_id, telemetry_file=temp_telemetry_file
    )

    # Test play
    await replay_service.play()
    assert replay_service.state == PlaybackState.PLAYING

    # Give it a moment to advance
    await asyncio.sleep(0.1)

    # Test pause
    await replay_service.pause()
    assert replay_service.state == PlaybackState.PAUSED
    paused_position = replay_service.current_position

    # Wait and verify position hasn't changed
    await asyncio.sleep(0.1)
    assert replay_service.current_position == paused_position

    # Test stop
    await replay_service.stop()
    assert replay_service.state == PlaybackState.STOPPED
    assert replay_service.current_position == 0


@pytest.mark.asyncio
async def test_seek(replay_service, temp_telemetry_file):
    """Test seeking to specific position."""
    mission_id = uuid4()
    await replay_service.load_mission_data(
        mission_id=mission_id, telemetry_file=temp_telemetry_file
    )

    # Seek to middle
    await replay_service.seek(5)
    assert replay_service.current_position == 5

    # Seek to start
    await replay_service.seek(0)
    assert replay_service.current_position == 0

    # Seek to end
    await replay_service.seek(9)
    assert replay_service.current_position == 9

    # Test invalid seek
    await replay_service.seek(100)
    assert replay_service.current_position == 9  # Should not change


@pytest.mark.asyncio
async def test_set_speed(replay_service):
    """Test setting playback speed."""
    await replay_service.set_speed(PlaybackSpeed.DOUBLE)
    assert replay_service.speed == PlaybackSpeed.DOUBLE

    await replay_service.set_speed(PlaybackSpeed.HALF)
    assert replay_service.speed == PlaybackSpeed.HALF

    await replay_service.set_speed(PlaybackSpeed.NORMAL)
    assert replay_service.speed == PlaybackSpeed.NORMAL


@pytest.mark.asyncio
async def test_get_status(replay_service, temp_telemetry_file):
    """Test getting replay status."""
    mission_id = uuid4()
    await replay_service.load_mission_data(
        mission_id=mission_id, telemetry_file=temp_telemetry_file
    )

    status = replay_service.get_status()
    assert status["mission_id"] == str(mission_id)
    assert status["state"] == "stopped"
    assert status["speed"] == PlaybackSpeed.NORMAL
    assert status["position"] == 0
    assert status["total"] == 10
    assert status["progress"] == 0

    # Seek and check status
    await replay_service.seek(5)
    status = replay_service.get_status()
    assert status["position"] == 5
    assert status["progress"] == 50


@pytest.mark.asyncio
async def test_get_timeline_range(replay_service, temp_telemetry_file):
    """Test getting timeline range."""
    # Empty timeline
    range_result = replay_service.get_timeline_range()
    assert range_result is None

    # Loaded timeline
    mission_id = uuid4()
    await replay_service.load_mission_data(
        mission_id=mission_id, telemetry_file=temp_telemetry_file
    )

    range_result = replay_service.get_timeline_range()
    assert range_result is not None
    start, end = range_result
    assert isinstance(start, datetime)
    assert isinstance(end, datetime)
    assert end > start


@pytest.mark.asyncio
async def test_websocket_handler(replay_service, temp_telemetry_file):
    """Test WebSocket event handling."""

    class MockWebSocketHandler:
        def __init__(self):
            self.events = []

        async def send_json(self, data):
            self.events.append(data)

    handler = MockWebSocketHandler()
    replay_service.set_websocket_handler(handler)

    mission_id = uuid4()
    await replay_service.load_mission_data(
        mission_id=mission_id, telemetry_file=temp_telemetry_file
    )

    # Seek to trigger event send
    await replay_service.seek(3)

    assert len(handler.events) == 1
    event = handler.events[0]
    assert event["type"] == "replay_event"
    assert event["position"] == 3
    assert event["total"] == 10
    assert "telemetry" in event
    assert "signal_detections" in event
    assert "state_changes" in event


@pytest.mark.asyncio
async def test_playback_completion(replay_service, temp_telemetry_file):
    """Test that playback stops when reaching the end."""
    mission_id = uuid4()
    await replay_service.load_mission_data(
        mission_id=mission_id, telemetry_file=temp_telemetry_file
    )

    # Set very high speed to complete quickly
    await replay_service.set_speed(PlaybackSpeed.MAX)

    # Start from near end
    await replay_service.seek(8)
    await replay_service.play()

    # Wait for completion
    await asyncio.sleep(0.5)

    # Should be stopped at the end
    assert replay_service.state == PlaybackState.STOPPED
    assert replay_service.current_position == 0  # Reset after stop


@pytest.mark.asyncio
async def test_load_invalid_file(replay_service):
    """Test handling of invalid file paths."""
    mission_id = uuid4()
    invalid_path = Path("/nonexistent/file.csv")

    result = await replay_service.load_mission_data(
        mission_id=mission_id, telemetry_file=invalid_path
    )

    assert result is False
    assert len(replay_service.timeline) == 0
