"""Tests for telemetry recording service."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.backend.services.telemetry_recorder import TelemetryFrame, TelemetryRecorder


@pytest.fixture
def mock_mavlink_service():
    """Create a mock MAVLink service."""
    mock = MagicMock()
    mock.get_telemetry = MagicMock(
        return_value={
            "position": {"lat": 47.5, "lon": -122.3, "alt": 100.0},
            "attitude": {"roll": 5.0, "pitch": -2.0, "yaw": 45.0},
            "battery": {"voltage": 12.6, "current": 5.0, "percentage": 75.0},
            "gps": {"fix_type": 3, "satellites": 12, "hdop": 1.5},
            "flight_mode": "GUIDED",
            "armed": True,
        }
    )
    return mock


@pytest.fixture
def mock_signal_processor():
    """Create a mock signal processor."""
    mock = MagicMock()
    mock.get_current_rssi = MagicMock(return_value=-75.0)
    mock.get_snr = MagicMock(return_value=20.0)
    mock.beacon_detected = False
    return mock


@pytest.fixture
def mock_state_machine():
    """Create a mock state machine."""
    mock = MagicMock()
    mock.get_state_string = MagicMock(return_value="SEARCHING")
    return mock


@pytest.fixture
def telemetry_recorder(mock_mavlink_service, mock_signal_processor, mock_state_machine, tmp_path):
    """Create a telemetry recorder instance with mocked dependencies."""
    recorder = TelemetryRecorder(
        mavlink_service=mock_mavlink_service,
        signal_processor=mock_signal_processor,
        state_machine=mock_state_machine,
        record_rate_hz=10.0,
    )
    # Use temp directory for output
    recorder.output_dir = tmp_path / "telemetry"
    recorder.output_dir.mkdir(parents=True, exist_ok=True)
    return recorder


class TestTelemetryRecorderInit:
    """Test telemetry recorder initialization."""

    def test_init_with_defaults(
        self, mock_mavlink_service, mock_signal_processor, mock_state_machine
    ):
        """Test initialization with default parameters."""
        recorder = TelemetryRecorder(
            mavlink_service=mock_mavlink_service,
            signal_processor=mock_signal_processor,
            state_machine=mock_state_machine,
        )
        
        assert recorder.mavlink == mock_mavlink_service
        assert recorder.signal_processor == mock_signal_processor
        assert recorder.state_machine == mock_state_machine
        assert recorder.record_rate_hz == 10.0
        assert recorder.recording is False
        assert recorder.record_task is None
        assert len(recorder.telemetry_buffer) == 0
        assert recorder.max_buffer_size == 36000
        assert recorder.session_id is None

    def test_init_with_custom_rate(
        self, mock_mavlink_service, mock_signal_processor, mock_state_machine
    ):
        """Test initialization with custom recording rate."""
        recorder = TelemetryRecorder(
            mavlink_service=mock_mavlink_service,
            signal_processor=mock_signal_processor,
            state_machine=mock_state_machine,
            record_rate_hz=20.0,
        )
        
        assert recorder.record_rate_hz == 20.0

    def test_output_dir_creation(
        self, mock_mavlink_service, mock_signal_processor, mock_state_machine, tmp_path
    ):
        """Test output directory creation."""
        recorder = TelemetryRecorder(
            mavlink_service=mock_mavlink_service,
            signal_processor=mock_signal_processor,
            state_machine=mock_state_machine,
        )
        recorder.output_dir = tmp_path / "test_telemetry"
        recorder.output_dir.mkdir(parents=True, exist_ok=True)
        
        assert recorder.output_dir.exists()
        assert recorder.output_dir.is_dir()


class TestRecordingControl:
    """Test recording start/stop functionality."""

    @pytest.mark.asyncio
    async def test_start_recording_with_session_id(self, telemetry_recorder):
        """Test starting recording with provided session ID."""
        session_id = await telemetry_recorder.start_recording("test_session_001")
        
        assert session_id == "test_session_001"
        assert telemetry_recorder.recording is True
        assert telemetry_recorder.session_id == "test_session_001"
        assert telemetry_recorder.record_task is not None
        assert len(telemetry_recorder.telemetry_buffer) == 0
        
        # Clean up
        await telemetry_recorder.stop_recording()

    @pytest.mark.asyncio
    async def test_start_recording_auto_session_id(self, telemetry_recorder):
        """Test starting recording with auto-generated session ID."""
        with patch("src.backend.services.telemetry_recorder.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"
            
            session_id = await telemetry_recorder.start_recording()
            
            assert session_id == "telemetry_20240101_120000"
            assert telemetry_recorder.recording is True
            assert telemetry_recorder.session_id == "telemetry_20240101_120000"
        
        # Clean up
        await telemetry_recorder.stop_recording()

    @pytest.mark.asyncio
    async def test_start_recording_already_recording(self, telemetry_recorder):
        """Test starting recording when already recording."""
        # Start first recording
        session_id1 = await telemetry_recorder.start_recording("session1")
        
        # Try to start second recording
        session_id2 = await telemetry_recorder.start_recording("session2")
        
        assert session_id2 == session_id1
        assert telemetry_recorder.session_id == "session1"
        
        # Clean up
        await telemetry_recorder.stop_recording()

    @pytest.mark.asyncio
    async def test_stop_recording_success(self, telemetry_recorder):
        """Test stopping recording successfully."""
        # Start recording
        await telemetry_recorder.start_recording("test_session")
        
        # Add some mock data
        telemetry_recorder.telemetry_buffer.append(
            TelemetryFrame(
                timestamp=datetime.now(UTC),
                latitude=47.5,
                longitude=-122.3,
                altitude=100.0,
                relative_altitude=50.0,
                groundspeed=10.0,
                airspeed=12.0,
                climb_rate=2.0,
                roll=5.0,
                pitch=-2.0,
                yaw=45.0,
                heading=45.0,
                battery_percent=75.0,
                battery_voltage=12.6,
                battery_current=5.0,
                gps_satellites=12,
                gps_hdop=1.5,
                rssi_dbm=-75.0,
                snr_db=20.0,
                beacon_detected=False,
                system_state="SEARCHING",
                armed=True,
                mode="GUIDED",
                cpu_percent=45.0,
                memory_percent=60.0,
                temperature_c=55.0,
            )
        )
        
        # Mock save_telemetry
        with patch.object(telemetry_recorder, "save_telemetry", new_callable=AsyncMock) as mock_save:
            mock_save.return_value = Path("/tmp/test.csv")
            
            result = await telemetry_recorder.stop_recording()
            
            assert result == Path("/tmp/test.csv")
            assert telemetry_recorder.recording is False
            mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_recording_not_recording(self, telemetry_recorder):
        """Test stopping recording when not recording."""
        result = await telemetry_recorder.stop_recording()
        
        assert result is None
        assert telemetry_recorder.recording is False

    @pytest.mark.asyncio
    async def test_stop_recording_no_data(self, telemetry_recorder):
        """Test stopping recording with no data collected."""
        # Start recording
        await telemetry_recorder.start_recording("test_session")
        
        # Stop immediately without collecting data
        result = await telemetry_recorder.stop_recording()
        
        assert result is None
        assert telemetry_recorder.recording is False


class TestDataCollection:
    """Test telemetry data collection."""

    @pytest.mark.asyncio
    async def test_collect_telemetry_frame(self, telemetry_recorder):
        """Test collecting a single telemetry frame."""
        with patch("psutil.cpu_percent", return_value=45.0), \
             patch("psutil.virtual_memory") as mock_mem:
            
            mock_mem.return_value.percent = 60.0
            
            frame = await telemetry_recorder._collect_telemetry_frame()
            
            assert isinstance(frame, TelemetryFrame)
            assert frame.latitude == 47.5
            assert frame.longitude == -122.3
            assert frame.altitude == 100.0
            assert frame.rssi_dbm == -75.0
            assert frame.snr_db == 20.0
            assert frame.system_state == "SEARCHING"
            assert frame.armed is True
            assert frame.mode == "GUIDED"
            assert frame.cpu_percent == 45.0
            assert frame.memory_percent == 60.0

    @pytest.mark.asyncio
    async def test_collect_telemetry_with_temperature(self, telemetry_recorder):
        """Test collecting telemetry with CPU temperature."""
        with patch("psutil.cpu_percent", return_value=45.0), \
             patch("psutil.virtual_memory") as mock_mem, \
             patch("builtins.open", create=True) as mock_open:
            
            mock_mem.return_value.percent = 60.0
            mock_open.return_value.__enter__.return_value.read.return_value = "55000"
            
            frame = await telemetry_recorder._collect_telemetry_frame()
            
            assert frame.temperature_c == 55.0

    @pytest.mark.asyncio
    async def test_collect_telemetry_no_temperature(self, telemetry_recorder):
        """Test collecting telemetry when temperature unavailable."""
        with patch("psutil.cpu_percent", return_value=45.0), \
             patch("psutil.virtual_memory") as mock_mem, \
             patch("builtins.open", side_effect=FileNotFoundError):
            
            mock_mem.return_value.percent = 60.0
            
            frame = await telemetry_recorder._collect_telemetry_frame()
            
            assert frame.temperature_c == 0.0

    @pytest.mark.asyncio
    async def test_recording_loop(self, telemetry_recorder):
        """Test the recording loop collects data at correct rate."""
        telemetry_recorder.record_rate_hz = 100.0  # Fast rate for testing
        
        # Start recording
        await telemetry_recorder.start_recording("test_session")
        
        # Let it run briefly
        await asyncio.sleep(0.05)
        
        # Stop recording
        await telemetry_recorder.stop_recording()
        
        # Should have collected some frames
        assert len(telemetry_recorder.telemetry_buffer) > 0
        assert len(telemetry_recorder.telemetry_buffer) < 10  # Not too many

    @pytest.mark.asyncio
    async def test_buffer_overflow_protection(self, telemetry_recorder):
        """Test buffer overflow protection."""
        telemetry_recorder.max_buffer_size = 5
        
        # Add frames beyond buffer limit
        for i in range(10):
            frame = TelemetryFrame(
                timestamp=datetime.now(UTC),
                latitude=47.5 + i,
                longitude=-122.3,
                altitude=100.0,
                relative_altitude=50.0,
                groundspeed=10.0,
                airspeed=12.0,
                climb_rate=2.0,
                roll=5.0,
                pitch=-2.0,
                yaw=45.0,
                heading=45.0,
                battery_percent=75.0,
                battery_voltage=12.6,
                battery_current=5.0,
                gps_satellites=12,
                gps_hdop=1.5,
                rssi_dbm=-75.0,
                snr_db=20.0,
                beacon_detected=False,
                system_state="SEARCHING",
                armed=True,
                mode="GUIDED",
                cpu_percent=45.0,
                memory_percent=60.0,
                temperature_c=55.0,
            )
            
            # Simulate what recording loop does
            if len(telemetry_recorder.telemetry_buffer) >= telemetry_recorder.max_buffer_size:
                telemetry_recorder.telemetry_buffer.pop(0)
            telemetry_recorder.telemetry_buffer.append(frame)
        
        assert len(telemetry_recorder.telemetry_buffer) == 5
        # Oldest frames should be removed
        assert telemetry_recorder.telemetry_buffer[0].latitude == 47.5 + 5


class TestDataSaving:
    """Test telemetry data saving functionality."""

    @pytest.mark.asyncio
    async def test_save_telemetry_csv(self, telemetry_recorder):
        """Test saving telemetry data to CSV."""
        telemetry_recorder.session_id = "test_session"
        
        # Add test data
        frames = [
            TelemetryFrame(
                timestamp=datetime.now(UTC),
                latitude=47.5 + i * 0.001,
                longitude=-122.3 + i * 0.001,
                altitude=100.0 + i,
                relative_altitude=50.0 + i,
                groundspeed=10.0 + i,
                airspeed=12.0 + i,
                climb_rate=2.0,
                roll=5.0,
                pitch=-2.0,
                yaw=45.0 + i,
                heading=45.0 + i,
                battery_percent=75.0 - i,
                battery_voltage=12.6,
                battery_current=5.0,
                gps_satellites=12,
                gps_hdop=1.5,
                rssi_dbm=-75.0 + i,
                snr_db=20.0 - i,
                beacon_detected=i > 2,
                system_state="SEARCHING",
                armed=True,
                mode="GUIDED",
                cpu_percent=45.0,
                memory_percent=60.0,
                temperature_c=55.0,
            )
            for i in range(3)
        ]
        telemetry_recorder.telemetry_buffer = frames
        
        # Save telemetry
        file_path = await telemetry_recorder.save_telemetry()
        
        assert file_path.exists()
        assert file_path.suffix == ".csv"
        assert "test_session" in str(file_path)
        
        # Verify CSV content
        import csv
        
        with open(file_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            assert len(rows) == 3
            assert float(rows[0]["latitude"]) == 47.5
            assert float(rows[1]["latitude"]) == 47.501
            assert rows[2]["beacon_detected"] == "True"

    @pytest.mark.asyncio
    async def test_save_telemetry_json(self, telemetry_recorder):
        """Test saving telemetry data to JSON."""
        telemetry_recorder.session_id = "test_session"
        
        # Add test data
        frame = TelemetryFrame(
            timestamp=datetime.now(UTC),
            latitude=47.5,
            longitude=-122.3,
            altitude=100.0,
            relative_altitude=50.0,
            groundspeed=10.0,
            airspeed=12.0,
            climb_rate=2.0,
            roll=5.0,
            pitch=-2.0,
            yaw=45.0,
            heading=45.0,
            battery_percent=75.0,
            battery_voltage=12.6,
            battery_current=5.0,
            gps_satellites=12,
            gps_hdop=1.5,
            rssi_dbm=-75.0,
            snr_db=20.0,
            beacon_detected=False,
            system_state="SEARCHING",
            armed=True,
            mode="GUIDED",
            cpu_percent=45.0,
            memory_percent=60.0,
            temperature_c=55.0,
        )
        telemetry_recorder.telemetry_buffer = [frame]
        
        # Save as JSON
        file_path = await telemetry_recorder.save_telemetry(format="json")
        
        assert file_path.exists()
        assert file_path.suffix == ".json"
        
        # Verify JSON content
        with open(file_path, "r") as f:
            data = json.load(f)
            
            assert "metadata" in data
            assert "frames" in data
            assert len(data["frames"]) == 1
            assert data["frames"][0]["latitude"] == 47.5
            assert data["frames"][0]["rssi_dbm"] == -75.0

    @pytest.mark.asyncio
    async def test_save_empty_buffer(self, telemetry_recorder):
        """Test saving with empty buffer."""
        telemetry_recorder.session_id = "test_session"
        telemetry_recorder.telemetry_buffer = []
        
        file_path = await telemetry_recorder.save_telemetry()
        
        # Should still create file with headers
        assert file_path.exists()
        
        # Verify CSV has headers but no data
        with open(file_path, "r") as f:
            lines = f.readlines()
            assert len(lines) == 1  # Header only


class TestMetadataAndAnalysis:
    """Test metadata generation and analysis."""

    def test_generate_metadata(self, telemetry_recorder):
        """Test metadata generation."""
        telemetry_recorder.session_id = "test_session"
        
        # Add test frames
        frames = [
            TelemetryFrame(
                timestamp=datetime.now(UTC),
                latitude=47.5 + i * 0.001,
                longitude=-122.3,
                altitude=100.0 + i * 10,
                relative_altitude=50.0,
                groundspeed=10.0,
                airspeed=12.0,
                climb_rate=2.0,
                roll=5.0,
                pitch=-2.0,
                yaw=45.0,
                heading=45.0,
                battery_percent=75.0 - i * 5,
                battery_voltage=12.6,
                battery_current=5.0,
                gps_satellites=12,
                gps_hdop=1.5,
                rssi_dbm=-75.0 + i * 5,
                snr_db=20.0,
                beacon_detected=i == 2,
                system_state="SEARCHING",
                armed=True,
                mode="GUIDED",
                cpu_percent=45.0,
                memory_percent=60.0,
                temperature_c=55.0,
            )
            for i in range(3)
        ]
        telemetry_recorder.telemetry_buffer = frames
        
        metadata = telemetry_recorder._generate_metadata()
        
        assert metadata["session_id"] == "test_session"
        assert metadata["total_frames"] == 3
        assert metadata["recording_duration_s"] > 0
        assert metadata["avg_rssi_dbm"] == -70.0
        assert metadata["min_rssi_dbm"] == -75.0
        assert metadata["max_rssi_dbm"] == -65.0
        assert metadata["beacon_detections"] == 1
        assert metadata["max_altitude_m"] == 120.0
        assert metadata["total_distance_m"] >= 0

    def test_calculate_distance(self, telemetry_recorder):
        """Test distance calculation between GPS coordinates."""
        # Test known distance
        # Seattle to Portland approximately 280 km
        dist = telemetry_recorder._calculate_distance(
            47.6062, -122.3321,  # Seattle
            45.5152, -122.6784   # Portland
        )
        
        assert 270000 < dist < 290000  # Within reasonable range

    def test_calculate_distance_same_point(self, telemetry_recorder):
        """Test distance calculation for same point."""
        dist = telemetry_recorder._calculate_distance(
            47.5, -122.3,
            47.5, -122.3
        )
        
        assert dist == 0.0


class TestExportAndPlayback:
    """Test data export and playback functionality."""

    @pytest.mark.asyncio
    async def test_export_to_kml(self, telemetry_recorder):
        """Test exporting telemetry to KML format."""
        telemetry_recorder.session_id = "test_session"
        
        # Add test frames
        frames = [
            TelemetryFrame(
                timestamp=datetime.now(UTC),
                latitude=47.5 + i * 0.01,
                longitude=-122.3 + i * 0.01,
                altitude=100.0 + i * 10,
                relative_altitude=50.0,
                groundspeed=10.0,
                airspeed=12.0,
                climb_rate=2.0,
                roll=5.0,
                pitch=-2.0,
                yaw=45.0,
                heading=45.0,
                battery_percent=75.0,
                battery_voltage=12.6,
                battery_current=5.0,
                gps_satellites=12,
                gps_hdop=1.5,
                rssi_dbm=-75.0 + i * 5,
                snr_db=20.0,
                beacon_detected=False,
                system_state="SEARCHING",
                armed=True,
                mode="GUIDED",
                cpu_percent=45.0,
                memory_percent=60.0,
                temperature_c=55.0,
            )
            for i in range(3)
        ]
        telemetry_recorder.telemetry_buffer = frames
        
        kml_path = await telemetry_recorder.export_to_kml()
        
        assert kml_path.exists()
        assert kml_path.suffix == ".kml"
        
        # Verify KML structure
        content = kml_path.read_text()
        assert "<?xml version" in content
        assert "<kml" in content
        assert "<Placemark>" in content
        assert "<LineString>" in content
        assert "<coordinates>" in content
        assert "47.5,-122.3,100" in content

    @pytest.mark.asyncio
    async def test_load_telemetry(self, telemetry_recorder):
        """Test loading telemetry from file."""
        # First save some data
        telemetry_recorder.session_id = "test_session"
        
        frame = TelemetryFrame(
            timestamp=datetime.now(UTC),
            latitude=47.5,
            longitude=-122.3,
            altitude=100.0,
            relative_altitude=50.0,
            groundspeed=10.0,
            airspeed=12.0,
            climb_rate=2.0,
            roll=5.0,
            pitch=-2.0,
            yaw=45.0,
            heading=45.0,
            battery_percent=75.0,
            battery_voltage=12.6,
            battery_current=5.0,
            gps_satellites=12,
            gps_hdop=1.5,
            rssi_dbm=-75.0,
            snr_db=20.0,
            beacon_detected=False,
            system_state="SEARCHING",
            armed=True,
            mode="GUIDED",
            cpu_percent=45.0,
            memory_percent=60.0,
            temperature_c=55.0,
        )
        telemetry_recorder.telemetry_buffer = [frame]
        
        file_path = await telemetry_recorder.save_telemetry()
        
        # Clear buffer and load
        telemetry_recorder.telemetry_buffer.clear()
        loaded = await telemetry_recorder.load_telemetry(file_path)
        
        assert loaded is True
        assert len(telemetry_recorder.telemetry_buffer) == 1
        assert telemetry_recorder.telemetry_buffer[0].latitude == 47.5
        assert telemetry_recorder.telemetry_buffer[0].rssi_dbm == -75.0

    @pytest.mark.asyncio
    async def test_load_invalid_file(self, telemetry_recorder):
        """Test loading from invalid file."""
        result = await telemetry_recorder.load_telemetry(Path("/nonexistent/file.csv"))
        
        assert result is False
        assert len(telemetry_recorder.telemetry_buffer) == 0


class TestIntegration:
    """Test integration with other services."""

    @pytest.mark.asyncio
    async def test_recording_with_state_changes(
        self, telemetry_recorder, mock_state_machine
    ):
        """Test recording handles state changes correctly."""
        # Simulate state changes during recording
        states = ["IDLE", "SEARCHING", "DETECTING", "HOMING"]
        state_index = 0
        
        def get_state():
            nonlocal state_index
            state = states[state_index % len(states)]
            state_index += 1
            return state
        
        mock_state_machine.get_state_string.side_effect = get_state
        
        # Start recording
        await telemetry_recorder.start_recording("state_test")
        
        # Collect a few frames
        for _ in range(4):
            frame = await telemetry_recorder._collect_telemetry_frame()
            telemetry_recorder.telemetry_buffer.append(frame)
        
        # Verify different states were recorded
        states_recorded = {frame.system_state for frame in telemetry_recorder.telemetry_buffer}
        assert len(states_recorded) == 4
        
        await telemetry_recorder.stop_recording()

    @pytest.mark.asyncio
    async def test_recording_with_signal_changes(
        self, telemetry_recorder, mock_signal_processor
    ):
        """Test recording handles signal changes correctly."""
        # Simulate signal strength changes
        rssi_values = [-100.0, -80.0, -60.0, -40.0]
        rssi_index = 0
        
        def get_rssi():
            nonlocal rssi_index
            rssi = rssi_values[rssi_index % len(rssi_values)]
            rssi_index += 1
            return rssi
        
        mock_signal_processor.get_current_rssi.side_effect = get_rssi
        
        # Collect frames
        frames = []
        for _ in range(4):
            frame = await telemetry_recorder._collect_telemetry_frame()
            frames.append(frame)
        
        # Verify RSSI changes were captured
        assert frames[0].rssi_dbm == -100.0
        assert frames[1].rssi_dbm == -80.0
        assert frames[2].rssi_dbm == -60.0
        assert frames[3].rssi_dbm == -40.0

    @pytest.mark.asyncio
    async def test_recording_handles_service_errors(
        self, telemetry_recorder, mock_mavlink_service
    ):
        """Test recording handles service errors gracefully."""
        # Make MAVLink service raise exception
        mock_mavlink_service.get_telemetry.side_effect = Exception("MAVLink error")
        
        # Should still collect frame with default values
        frame = await telemetry_recorder._collect_telemetry_frame()
        
        assert isinstance(frame, TelemetryFrame)
        # Should have default/fallback values
        assert frame.latitude == 0.0
        assert frame.longitude == 0.0