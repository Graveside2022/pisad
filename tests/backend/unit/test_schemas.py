"""Unit tests for data schemas.

Tests all dataclass schemas for proper structure and validation.
"""

from datetime import UTC, datetime
from uuid import UUID

from src.backend.models.schemas import (
    BeaconConfiguration,
    DetectionEvent,
    FieldTestMetrics,
    RSSIReading,
    SafetyEvent,
    SDRConfig,
    SDRStatus,
    SignalDetection,
    SystemState,
)


class TestSDRConfig:
    """Test SDR configuration schema."""

    def test_sdr_config_defaults(self):
        """Test SDR configuration with default values."""
        config = SDRConfig()

        assert config.frequency == 2.437e9  # 2.437 GHz
        assert config.sampleRate == 2e6  # 2 Msps
        assert config.gain == "AUTO"
        assert config.bandwidth == 2e6  # 2 MHz
        assert config.buffer_size == 1024
        assert config.device_args == ""

    def test_sdr_config_custom_values(self):
        """Test SDR configuration with custom values."""
        config = SDRConfig(
            frequency=433920000,
            sampleRate=1000000,
            gain=30,
            bandwidth=1000000,
            buffer_size=2048,
            device_args="driver=hackrf",
        )

        assert config.frequency == 433920000
        assert config.sampleRate == 1000000
        assert config.gain == 30
        assert config.bandwidth == 1000000
        assert config.buffer_size == 2048
        assert config.device_args == "driver=hackrf"

    def test_sdr_config_gain_types(self):
        """Test SDR configuration supports different gain types."""
        # Numeric gain
        config1 = SDRConfig(gain=25)
        assert config1.gain == 25

        # Auto gain
        config2 = SDRConfig(gain="AUTO")
        assert config2.gain == "AUTO"


class TestSDRStatus:
    """Test SDR status schema."""

    def test_sdr_status_defaults(self):
        """Test SDR status with default values."""
        status = SDRStatus(status="CONNECTED")

        assert status.status == "CONNECTED"
        assert status.device_name is None
        assert status.driver is None
        assert status.stream_active is False
        assert status.samples_per_second == 0.0
        assert status.buffer_overflows == 0
        assert status.last_error is None
        assert status.temperature is None

    def test_sdr_status_all_fields(self):
        """Test SDR status with all fields populated."""
        status = SDRStatus(
            status="CONNECTED",
            device_name="HackRF One",
            driver="hackrf",
            stream_active=True,
            samples_per_second=2000000.0,
            buffer_overflows=5,
            last_error="Temporary timeout",
            temperature=45.2,
        )

        assert status.status == "CONNECTED"
        assert status.device_name == "HackRF One"
        assert status.driver == "hackrf"
        assert status.stream_active is True
        assert status.samples_per_second == 2000000.0
        assert status.buffer_overflows == 5
        assert status.last_error == "Temporary timeout"
        assert status.temperature == 45.2

    def test_sdr_status_valid_states(self):
        """Test SDR status accepts valid state values."""
        for state in ["CONNECTED", "DISCONNECTED", "ERROR"]:
            status = SDRStatus(status=state)
            assert status.status == state


class TestSystemState:
    """Test system state schema."""

    def test_system_state_defaults(self):
        """Test system state with default values."""
        state = SystemState()

        assert state.sdr_status == "DISCONNECTED"
        assert state.processing_active is False
        assert state.tracking_enabled is False
        assert state.mavlink_connected is False
        assert state.flight_mode == "UNKNOWN"
        assert state.battery_percent == 0.0
        assert state.gps_status == "NO_FIX"
        assert state.homing_enabled is False
        assert state.safety_interlocks is None
        assert state.search_pattern_id is None
        assert state.search_substate == "IDLE"


class TestRSSIReading:
    """Test RSSI reading schema."""

    def test_rssi_reading_creation(self):
        """Test RSSI reading with all fields."""
        timestamp = datetime.now(UTC)
        reading = RSSIReading(
            value=-45.5, snr=15.2, confidence=0.85, frequency=433920000, timestamp=timestamp
        )

        assert reading.value == -45.5
        assert reading.snr == 15.2
        assert reading.confidence == 0.85
        assert reading.frequency == 433920000
        assert reading.timestamp == timestamp

    def test_rssi_reading_defaults(self):
        """Test RSSI reading with minimal required fields."""
        reading = RSSIReading(value=-50.0, snr=10.0, confidence=0.7, frequency=868000000)

        assert reading.value == -50.0
        assert reading.snr == 10.0
        assert reading.confidence == 0.7
        assert reading.frequency == 868000000
        # Timestamp should be auto-generated if not provided
        assert isinstance(reading.timestamp, datetime)


class TestDetectionEvent:
    """Test detection event schema."""

    def test_detection_event_creation(self):
        """Test detection event with all fields."""
        event_id = UUID("12345678-1234-5678-9012-123456789012")
        timestamp = datetime.now(UTC)

        event = DetectionEvent(
            id=event_id,
            rssi=-35.0,
            snr=20.5,
            confidence=0.95,
            frequency=2437000000,
            location=(37.7749, -122.4194),
            timestamp=timestamp,
        )

        assert event.id == event_id
        assert event.rssi == -35.0
        assert event.snr == 20.5
        assert event.confidence == 0.95
        assert event.frequency == 2437000000
        assert event.location == (37.7749, -122.4194)
        assert event.timestamp == timestamp

    def test_detection_event_auto_fields(self):
        """Test detection event with auto-generated fields."""
        event = DetectionEvent(rssi=-40.0, snr=18.0, confidence=0.88, frequency=433920000)

        assert event.rssi == -40.0
        assert event.snr == 18.0
        assert event.confidence == 0.88
        assert event.frequency == 433920000
        # ID and timestamp should be auto-generated
        assert isinstance(event.id, UUID)
        assert isinstance(event.timestamp, datetime)


class TestSignalDetection:
    """Test signal detection schema."""

    def test_signal_detection_creation(self):
        """Test signal detection with all fields."""
        timestamp = datetime.now(UTC)

        detection = SignalDetection(
            rssi_dbm=-45.5,
            snr_db=15.2,
            confidence=0.85,
            frequency_hz=433920000,
            timestamp=timestamp,
        )

        assert detection.rssi_dbm == -45.5
        assert detection.snr_db == 15.2
        assert detection.confidence == 0.85
        assert detection.frequency_hz == 433920000
        assert detection.timestamp == timestamp


class TestSafetyEvent:
    """Test safety event schema."""

    def test_safety_event_creation(self):
        """Test safety event with all fields."""
        event_id = UUID("12345678-1234-5678-9012-123456789012")
        timestamp = datetime.now(UTC)

        event = SafetyEvent(
            id=event_id,
            event_type="BATTERY_LOW",
            severity="HIGH",
            message="Battery voltage below threshold",
            source="battery_monitor",
            timestamp=timestamp,
        )

        assert event.id == event_id
        assert event.event_type == "BATTERY_LOW"
        assert event.severity == "HIGH"
        assert event.message == "Battery voltage below threshold"
        assert event.source == "battery_monitor"
        assert event.timestamp == timestamp


class TestBeaconConfiguration:
    """Test beacon configuration schema."""

    def test_beacon_configuration_creation(self):
        """Test beacon configuration with all fields."""
        config = BeaconConfiguration(
            name="Test Beacon",
            frequency=433920000,
            power_dbm=10,
            modulation="FM",
            bandwidth=25000,
            enabled=True,
        )

        assert config.name == "Test Beacon"
        assert config.frequency == 433920000
        assert config.power_dbm == 10
        assert config.modulation == "FM"
        assert config.bandwidth == 25000
        assert config.enabled is True


class TestFieldTestMetrics:
    """Test field test metrics schema."""

    def test_field_test_metrics_creation(self):
        """Test field test metrics with all fields."""
        timestamp = datetime.now(UTC)

        metrics = FieldTestMetrics(
            test_id="TEST_001",
            detection_range_m=450.0,
            approach_accuracy_m=12.5,
            time_to_locate_s=125.3,
            false_positive_rate=0.02,
            signal_strength_dbm=-38.2,
            timestamp=timestamp,
        )

        assert metrics.test_id == "TEST_001"
        assert metrics.detection_range_m == 450.0
        assert metrics.approach_accuracy_m == 12.5
        assert metrics.time_to_locate_s == 125.3
        assert metrics.false_positive_rate == 0.02
        assert metrics.signal_strength_dbm == -38.2
        assert metrics.timestamp == timestamp
