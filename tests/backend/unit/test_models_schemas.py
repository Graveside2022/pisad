"""Unit tests for data model schemas."""

from dataclasses import asdict
from datetime import UTC, datetime

from src.backend.models.schemas import (
    BeaconConfiguration,
    ConfigProfile,
    DetectionEvent,
    FieldTestMetrics,
    GeofenceConfig,
    HomingConfig,
    RSSIReading,
    SafetyEvent,
    SDRConfig,
    SDRStatus,
    SignalConfig,
    SignalDetection,
    SystemState,
)


class TestSDRConfig:
    """Test SDRConfig schema."""

    def test_sdr_config_creation(self):
        """Test creating SDR config."""
        config = SDRConfig(
            frequency=433.92e6,
            sampleRate=2.048e6,
            gain=40,
            bandwidth=2e6,
            buffer_size=2048,
            device_args="rtl=0",
        )
        assert config.frequency == 433.92e6
        assert config.sampleRate == 2.048e6
        assert config.gain == 40
        assert config.buffer_size == 2048

    def test_sdr_config_defaults(self):
        """Test SDR config default values."""
        config = SDRConfig()
        assert config.frequency == 2.437e9
        assert config.sampleRate == 2e6
        assert config.gain == "AUTO"
        assert config.bandwidth == 2e6
        assert config.buffer_size == 1024
        assert config.device_args == ""

    def test_sdr_config_as_dict(self):
        """Test converting SDR config to dict."""
        config = SDRConfig(frequency=433.92e6, sampleRate=2.048e6, gain=40)
        config_dict = asdict(config)
        assert config_dict["frequency"] == 433.92e6
        assert config_dict["sampleRate"] == 2.048e6
        assert config_dict["gain"] == 40


class TestSDRStatus:
    """Test SDRStatus schema."""

    def test_sdr_status_creation(self):
        """Test creating SDR status."""
        status = SDRStatus(
            status="CONNECTED",
            device_name="RTL-SDR",
            driver="rtlsdr",
            stream_active=True,
            samples_per_second=2.048e6,
            buffer_overflows=5,
            last_error=None,
            temperature=45.5,
        )
        assert status.status == "CONNECTED"
        assert status.device_name == "RTL-SDR"
        assert status.stream_active is True
        assert status.samples_per_second == 2.048e6
        assert status.temperature == 45.5

    def test_sdr_status_error_state(self):
        """Test SDR status in error state."""
        status = SDRStatus(status="ERROR", last_error="Device not found")
        assert status.status == "ERROR"
        assert status.last_error == "Device not found"
        assert status.stream_active is False


class TestSystemState:
    """Test SystemState schema."""

    def test_system_state_creation(self):
        """Test creating system state."""
        state = SystemState(
            sdr_status="CONNECTED",
            processing_active=True,
            tracking_enabled=True,
            mavlink_connected=True,
            flight_mode="GUIDED",
            battery_percent=85.5,
            gps_status="3D_FIX",
            homing_enabled=True,
            gradient_confidence=75.0,
            target_heading=135.0,
        )
        assert state.sdr_status == "CONNECTED"
        assert state.processing_active is True
        assert state.battery_percent == 85.5
        assert state.gradient_confidence == 75.0

    def test_system_state_defaults(self):
        """Test system state default values."""
        state = SystemState()
        assert state.sdr_status == "DISCONNECTED"
        assert state.processing_active is False
        assert state.flight_mode == "UNKNOWN"
        assert state.gps_status == "NO_FIX"
        assert state.search_substate == "IDLE"
        assert state.homing_substage == "IDLE"

    def test_system_state_post_init(self):
        """Test system state post-initialization."""
        state = SystemState()
        assert state.errors == []
        assert state.safety_interlocks is not None
        assert "mode_check" in state.safety_interlocks
        assert "battery_check" in state.safety_interlocks


class TestSignalConfig:
    """Test SignalConfig schema."""

    def test_signal_config_creation(self):
        """Test creating signal config."""
        config = SignalConfig(
            fftSize=2048, ewmaAlpha=0.2, triggerThreshold=-55.0, dropThreshold=-65.0
        )
        assert config.fftSize == 2048
        assert config.ewmaAlpha == 0.2
        assert config.triggerThreshold == -55.0

    def test_signal_config_defaults(self):
        """Test signal config default values."""
        config = SignalConfig()
        assert config.fftSize == 1024
        assert config.ewmaAlpha == 0.1
        assert config.triggerThreshold == -60.0
        assert config.dropThreshold == -70.0


class TestHomingConfig:
    """Test HomingConfig schema."""

    def test_homing_config_creation(self):
        """Test creating homing config."""
        config = HomingConfig(
            forwardVelocityMax=10.0,
            yawRateMax=1.5,
            approachVelocity=3.0,
            signalLossTimeout=10.0,
            algorithmMode="SIMPLE",
            gradientWindowSize=20,
            approachThreshold=-45.0,
        )
        assert config.forwardVelocityMax == 10.0
        assert config.yawRateMax == 1.5
        assert config.algorithmMode == "SIMPLE"
        assert config.gradientWindowSize == 20

    def test_homing_config_defaults(self):
        """Test homing config default values."""
        config = HomingConfig()
        assert config.forwardVelocityMax == 5.0
        assert config.yawRateMax == 1.0
        assert config.algorithmMode == "GRADIENT"
        assert config.plateauVariance == 2.0


class TestGeofenceConfig:
    """Test GeofenceConfig schema."""

    def test_geofence_config_creation(self):
        """Test creating geofence config."""
        config = GeofenceConfig(
            enabled=True, center_lat=37.7749, center_lon=-122.4194, radius_meters=500.0
        )
        assert config.enabled is True
        assert config.center_lat == 37.7749
        assert config.center_lon == -122.4194
        assert config.radius_meters == 500.0

    def test_geofence_config_disabled(self):
        """Test disabled geofence config."""
        config = GeofenceConfig()
        assert config.enabled is False
        assert config.center_lat is None
        assert config.center_lon is None
        assert config.radius_meters == 100.0


class TestConfigProfile:
    """Test ConfigProfile schema."""

    def test_config_profile_creation(self):
        """Test creating config profile."""
        profile = ConfigProfile(
            id="test-id", name="Test Profile", description="Test description", isDefault=True
        )
        assert profile.id == "test-id"
        assert profile.name == "Test Profile"
        assert profile.isDefault is True

    def test_config_profile_post_init(self):
        """Test config profile post-initialization."""
        profile = ConfigProfile(id=None, name="Test Profile")
        assert profile.id is not None  # Should be auto-generated
        assert profile.sdrConfig is not None
        assert profile.signalConfig is not None
        assert profile.homingConfig is not None
        assert profile.geofenceConfig is not None
        assert profile.createdAt is not None
        assert profile.updatedAt is not None

    def test_config_profile_with_custom_configs(self):
        """Test config profile with custom configurations."""
        sdr_config = SDRConfig(frequency=915e6)
        signal_config = SignalConfig(fftSize=4096)

        profile = ConfigProfile(
            id="custom", name="Custom Profile", sdrConfig=sdr_config, signalConfig=signal_config
        )
        assert profile.sdrConfig.frequency == 915e6
        assert profile.signalConfig.fftSize == 4096


class TestRSSIReading:
    """Test RSSIReading schema."""

    def test_rssi_reading_creation(self):
        """Test creating RSSI reading."""
        now = datetime.now(UTC)
        reading = RSSIReading(timestamp=now, rssi=-75.5, noise_floor=-95.0, detection_id="det-123")
        assert reading.timestamp == now
        assert reading.rssi == -75.5
        assert reading.noise_floor == -95.0
        assert reading.detection_id == "det-123"

    def test_rssi_reading_without_detection(self):
        """Test RSSI reading without detection."""
        now = datetime.now(UTC)
        reading = RSSIReading(timestamp=now, rssi=-85.0, noise_floor=-95.0)
        assert reading.detection_id is None


class TestSignalDetection:
    """Test SignalDetection schema."""

    def test_signal_detection_creation(self):
        """Test creating signal detection."""
        now = datetime.now(UTC)
        detection = SignalDetection(
            id="det-456",
            timestamp=now,
            frequency=433.92e6,
            rssi=-65.0,
            snr=25.0,
            confidence=95.5,
            location={"lat": 37.7749, "lon": -122.4194, "alt": 100},
            state="homing",
        )
        assert detection.id == "det-456"
        assert detection.frequency == 433.92e6
        assert detection.rssi == -65.0
        assert detection.confidence == 95.5
        assert detection.location["lat"] == 37.7749

    def test_signal_detection_defaults(self):
        """Test signal detection default values."""
        now = datetime.now(UTC)
        detection = SignalDetection(
            id="det-789", timestamp=now, frequency=433e6, rssi=-70.0, snr=20.0, confidence=80.0
        )
        assert detection.state == "active"
        assert detection.location is None


class TestDetectionEvent:
    """Test DetectionEvent schema."""

    def test_detection_event_creation(self):
        """Test creating detection event."""
        now = datetime.now(UTC)
        event = DetectionEvent(
            id="evt-123",
            timestamp=now,
            frequency=915e6,
            rssi=-55.0,
            snr=30.0,
            confidence=99.0,
            location={"lat": 40.7128, "lon": -74.0060},
            state="tracking",
        )
        assert event.id == "evt-123"
        assert event.frequency == 915e6
        assert event.rssi == -55.0
        assert event.state == "tracking"


class TestSafetyEvent:
    """Test SafetyEvent schema."""

    def test_safety_event_creation(self):
        """Test creating safety event."""
        event = SafetyEvent(
            event_type="emergency_stop",
            trigger="low_battery",
            details={"battery_voltage": 10.5, "threshold": 11.0},
            resolved=False,
        )
        assert event.event_type == "emergency_stop"
        assert event.trigger == "low_battery"
        assert event.details["battery_voltage"] == 10.5
        assert event.resolved is False

    def test_safety_event_defaults(self):
        """Test safety event default values."""
        event = SafetyEvent()
        assert event.id is not None  # Auto-generated UUID
        assert event.timestamp is not None  # Auto-generated timestamp
        assert event.event_type == "interlock_triggered"
        assert event.trigger == "mode_change"
        assert event.details == {}
        assert event.resolved is False

    def test_safety_event_resolved(self):
        """Test resolved safety event."""
        event = SafetyEvent(event_type="safety_override", trigger="operator_command", resolved=True)
        assert event.resolved is True


class TestBeaconConfiguration:
    """Test BeaconConfiguration schema."""

    def test_beacon_config_creation(self):
        """Test creating beacon configuration."""
        config = BeaconConfiguration(
            frequency_hz=915_000_000,
            power_dbm=20.0,
            modulation="FSK",
            spreading_factor=10,
            bandwidth_hz=250000,
            coding_rate=6,
            pulse_rate_hz=2.0,
            pulse_duration_ms=200,
        )
        assert config.frequency_hz == 915_000_000
        assert config.power_dbm == 20.0
        assert config.modulation == "FSK"
        assert config.pulse_rate_hz == 2.0

    def test_beacon_config_defaults(self):
        """Test beacon configuration defaults."""
        config = BeaconConfiguration()
        assert config.frequency_hz == 433_000_000
        assert config.power_dbm == 10.0
        assert config.modulation == "LoRa"
        assert config.spreading_factor == 7
        assert config.bandwidth_hz == 125000


class TestFieldTestMetrics:
    """Test FieldTestMetrics schema."""

    def test_field_test_metrics_creation(self):
        """Test creating field test metrics."""
        metrics = FieldTestMetrics(
            test_id="test-001",
            beacon_power_dbm=10.0,
            detection_range_m=500.0,
            approach_accuracy_m=5.0,
            time_to_locate_s=120.0,
            transition_latency_ms=50.0,
            environmental_conditions={"wind_speed_mps": 5.0, "temperature_c": 20.0},
            safety_events=["geofence_warning", "battery_low"],
            success=True,
            max_rssi_dbm=-45.0,
            min_rssi_dbm=-85.0,
            avg_rssi_dbm=-65.0,
            signal_loss_count=2,
        )
        assert metrics.test_id == "test-001"
        assert metrics.detection_range_m == 500.0
        assert metrics.success is True
        assert metrics.signal_loss_count == 2
        assert len(metrics.safety_events) == 2

    def test_field_test_metrics_defaults(self):
        """Test field test metrics default values."""
        metrics = FieldTestMetrics(
            test_id="test-002",
            beacon_power_dbm=10.0,
            detection_range_m=100.0,
            approach_accuracy_m=10.0,
            time_to_locate_s=60.0,
            transition_latency_ms=100.0,
            environmental_conditions={},
            safety_events=[],
            success=False,
        )
        assert metrics.max_rssi_dbm == -120.0
        assert metrics.min_rssi_dbm == 0.0
        assert metrics.avg_rssi_dbm == -60.0
        assert metrics.signal_loss_count == 0
        assert metrics.state_transitions == []

    def test_field_test_metrics_with_transitions(self):
        """Test field test metrics with state transitions."""
        transitions = [
            {"from": "IDLE", "to": "SEARCHING", "timestamp": "2024-01-01T10:00:00"},
            {"from": "SEARCHING", "to": "HOMING", "timestamp": "2024-01-01T10:05:00"},
        ]
        metrics = FieldTestMetrics(
            test_id="test-003",
            beacon_power_dbm=10.0,
            detection_range_m=200.0,
            approach_accuracy_m=3.0,
            time_to_locate_s=90.0,
            transition_latency_ms=75.0,
            environmental_conditions={},
            safety_events=[],
            success=True,
            state_transitions=transitions,
        )
        assert len(metrics.state_transitions) == 2
        assert metrics.state_transitions[0]["from"] == "IDLE"
        assert metrics.state_transitions[1]["to"] == "HOMING"
