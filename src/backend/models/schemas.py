"""Data models and schemas for the PISAD backend."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID, uuid4


@dataclass
class SDRConfig:
    """SDR configuration parameters."""

    frequency: float = 2.437e9  # Hz (2.437 GHz default)
    sampleRate: float = 2e6  # Samples per second (2 Msps default)
    gain: float | str = "AUTO"  # dB or 'AUTO'
    bandwidth: float = 2e6  # Hz (2 MHz default)
    buffer_size: int = 1024  # Samples per buffer
    device_args: str = ""  # Additional device arguments


@dataclass
class SDRStatus:
    """SDR device status information."""

    status: Literal["CONNECTED", "DISCONNECTED", "ERROR"]
    device_name: str | None = None
    driver: str | None = None
    stream_active: bool = False
    samples_per_second: float = 0.0
    buffer_overflows: int = 0
    last_error: str | None = None
    temperature: float | None = None  # Celsius if available


@dataclass
class SystemState:
    """Overall system state."""

    sdr_status: Literal["CONNECTED", "DISCONNECTED", "ERROR"] = "DISCONNECTED"
    processing_active: bool = False
    tracking_enabled: bool = False
    mavlink_connected: bool = False  # MAVLink connection status
    flight_mode: str = "UNKNOWN"  # Current flight controller mode
    battery_percent: float = 0.0  # Battery percentage
    gps_status: Literal["NO_FIX", "2D_FIX", "3D_FIX", "RTK"] = "NO_FIX"  # GPS fix status
    homing_enabled: bool = False  # Operator activation status for homing
    safety_interlocks: dict[str, bool] | None = None  # Status of all safety checks
    search_pattern_id: str | None = None  # Active search pattern ID
    search_substate: Literal["IDLE", "EXECUTING", "PAUSED"] = (
        "IDLE"  # Search pattern execution state
    )
    homing_substage: Literal["IDLE", "GRADIENT_CLIMB", "SAMPLING", "APPROACH", "HOLDING"] = (
        "IDLE"  # Homing algorithm substage
    )
    gradient_confidence: float = 0.0  # Gradient calculation confidence (0-100%)
    target_heading: float = 0.0  # Computed optimal heading in degrees
    last_update: datetime | None = None
    errors: list[str] | None = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []
        if self.safety_interlocks is None:
            self.safety_interlocks = {
                "mode_check": False,
                "battery_check": False,
                "geofence_check": False,
                "signal_check": False,
                "operator_check": False,
            }


@dataclass
class SignalConfig:
    """Signal processing configuration parameters."""

    fftSize: int = 1024  # FFT size for spectral analysis
    ewmaAlpha: float = 0.1  # EWMA smoothing factor
    triggerThreshold: float = -60.0  # dBm threshold to trigger detection
    dropThreshold: float = -70.0  # dBm threshold to drop detection


@dataclass
class HomingConfig:
    """Homing behavior configuration parameters."""

    forwardVelocityMax: float = 5.0  # m/s maximum forward velocity
    yawRateMax: float = 1.0  # rad/s maximum yaw rate
    approachVelocity: float = 2.0  # m/s approach velocity
    signalLossTimeout: float = 5.0  # seconds before declaring signal lost
    # Gradient algorithm parameters
    algorithmMode: str = "GRADIENT"  # "SIMPLE" or "GRADIENT"
    gradientWindowSize: int = 10  # RSSI history window size (samples)
    gradientMinSNR: float = 10.0  # Minimum SNR for gradient calculation (dB)
    samplingTurnRadius: float = 10.0  # S-turn radius for sampling (meters)
    samplingDuration: float = 5.0  # Duration of sampling maneuver (seconds)
    approachThreshold: float = -50.0  # RSSI threshold for approach mode (dBm)
    plateauVariance: float = 2.0  # RSSI variance for plateau detection
    velocityScaleFactor: float = 0.1  # Scaling factor for velocity commands


@dataclass
class GeofenceConfig:
    """Geofence configuration for safety boundary."""

    enabled: bool = False
    center_lat: float | None = None  # Center latitude
    center_lon: float | None = None  # Center longitude
    radius_meters: float = 100.0  # Radius in meters


@dataclass
class ConfigProfile:
    """System configuration profile."""

    id: str
    name: str
    description: str = ""
    sdrConfig: SDRConfig | None = None
    signalConfig: SignalConfig | None = None
    homingConfig: HomingConfig | None = None
    geofenceConfig: GeofenceConfig | None = None
    isDefault: bool = False
    createdAt: datetime | None = None
    updatedAt: datetime | None = None

    def __post_init__(self) -> None:
        if not hasattr(self, "id") or self.id is None:
            self.id = str(uuid4())
        if self.sdrConfig is None:
            self.sdrConfig = SDRConfig()
        if self.signalConfig is None:
            self.signalConfig = SignalConfig()
        if self.homingConfig is None:
            self.homingConfig = HomingConfig()
        if self.geofenceConfig is None:
            self.geofenceConfig = GeofenceConfig()
        if self.createdAt is None:
            self.createdAt = datetime.now()
        if self.updatedAt is None:
            self.updatedAt = datetime.now()


@dataclass
class RSSIReading:
    """Time-series RSSI data for real-time visualization."""

    timestamp: datetime  # Microsecond precision timestamp
    rssi: float  # Signal strength in dBm
    noise_floor: float  # Estimated noise floor in dBm
    snr: float = 0.0  # Signal-to-noise ratio in dB
    detection_id: str | None = None  # Associated detection event (nullable)


@dataclass
class SignalDetection:
    """Records RF signal detection events."""

    id: str  # UUID identifier
    timestamp: datetime  # UTC timestamp of detection
    frequency: float  # Center frequency in Hz
    rssi: float  # Signal strength in dBm
    snr: float  # Signal-to-noise ratio in dB
    confidence: float  # Detection confidence percentage (0-100)
    location: dict[str, Any] | None = None  # GPS coordinates if available
    state: str = "active"  # System state during detection


@dataclass
class DetectionEvent:
    """Signal detection event for internal processing."""

    id: str  # UUID identifier
    timestamp: datetime  # UTC timestamp of detection
    frequency: float  # Center frequency in Hz
    rssi: float  # Signal strength in dBm
    snr: float  # Signal-to-noise ratio in dB
    confidence: float  # Detection confidence percentage (0-100)
    location: dict[str, Any] | None = None  # GPS coordinates if available
    state: str = "active"  # System state during detection


@dataclass
class SafetyEvent:
    """Safety event for tracking interlock and emergency events."""

    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    event_type: str = (
        "interlock_triggered"  # "interlock_triggered", "emergency_stop", "safety_override"
    )
    trigger: str = "mode_change"  # "mode_change", "low_battery", "signal_loss", etc.
    details: dict[str, Any] = field(default_factory=dict)  # Additional context
    resolved: bool = False  # Whether issue was resolved


@dataclass
class BeaconConfiguration:
    """Configuration for test beacon transmitter."""

    frequency_hz: float = 433_000_000
    power_dbm: float = 10.0
    modulation: str = "LoRa"
    spreading_factor: int = 7
    bandwidth_hz: float = 125000
    coding_rate: int = 5
    pulse_rate_hz: float = 1.0
    pulse_duration_ms: float = 100


@dataclass
class FieldTestMetrics:
    """Metrics collected during field testing."""

    test_id: str
    beacon_power_dbm: float
    detection_range_m: float
    approach_accuracy_m: float
    time_to_locate_s: float
    transition_latency_ms: float
    environmental_conditions: dict[str, Any]
    safety_events: list[str]
    success: bool
    max_rssi_dbm: float = -120.0
    min_rssi_dbm: float = 0.0
    avg_rssi_dbm: float = -60.0
    signal_loss_count: int = 0
    state_transitions: list[dict[str, Any]] = field(default_factory=list)
