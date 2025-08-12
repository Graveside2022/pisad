# Data Models

## SignalDetection
**Purpose:** Records RF signal detection events with metadata for analysis and replay

**Key Attributes:**
- id: UUID - Unique identifier for detection event
- timestamp: datetime - UTC timestamp of detection
- frequency: float - Center frequency in Hz
- rssi: float - Signal strength in dBm
- snr: float - Signal-to-noise ratio in dB
- confidence: float - Detection confidence percentage (0-100)
- location: JSON - GPS coordinates if available
- state: string - System state during detection (SEARCHING, DETECTING, HOMING)

### TypeScript Interface
```typescript
interface SignalDetection {
  id: string;
  timestamp: string; // ISO 8601
  frequency: number;
  rssi: number;
  snr: number;
  confidence: number;
  location?: {
    lat: number;
    lon: number;
    alt: number;
  };
  state: 'IDLE' | 'SEARCHING' | 'DETECTING' | 'HOMING' | 'HOLDING';
}
```

### Relationships
- Has many RSSIReadings (time series data)
- Belongs to one Mission

## RSSIReading
**Purpose:** Time-series RSSI data for real-time visualization and gradient analysis

**Key Attributes:**
- timestamp: datetime - Microsecond precision timestamp
- rssi: float - Signal strength in dBm
- noise_floor: float - Estimated noise floor in dBm
- detection_id: UUID - Associated detection event (nullable)

### TypeScript Interface
```typescript
interface RSSIReading {
  timestamp: string; // ISO 8601 with microseconds
  rssi: number;
  noiseFloor: number;
  detectionId?: string;
}
```

### Relationships
- Belongs to SignalDetection (optional)
- Used for real-time streaming (not always persisted)

## ConfigProfile
**Purpose:** Stores SDR and system configuration profiles for different beacon types

**Key Attributes:**
- id: UUID - Profile identifier
- name: string - Profile name (e.g., "WiFi Beacon", "LoRa Tracker")
- sdr_config: JSON - SDR settings (frequency, sample_rate, gain)
- signal_config: JSON - Signal processing parameters
- homing_config: JSON - Homing behavior parameters
- is_default: boolean - Default profile flag
- created_at: datetime - Profile creation time
- updated_at: datetime - Last modification time

### TypeScript Interface
```typescript
interface ConfigProfile {
  id: string;
  name: string;
  sdrConfig: {
    frequency: number;
    sampleRate: number;
    gain: number;
    bandwidth: number;
  };
  signalConfig: {
    fftSize: number;
    ewmaAlpha: number;
    triggerThreshold: number;
    dropThreshold: number;
  };
  homingConfig: {
    forwardVelocityMax: number;
    yawRateMax: number;
    approachVelocity: number;
    signalLossTimeout: number;
  };
  isDefault: boolean;
  createdAt: string;
  updatedAt: string;
}
```

### Relationships
- Used by multiple Missions
- Can be cloned/modified

## SystemState
**Purpose:** Real-time system state for UI synchronization and safety monitoring

**Key Attributes:**
- current_state: enum - State machine state
- homing_enabled: boolean - Homing activation status
- flight_mode: string - Current flight controller mode
- battery_percent: float - Battery remaining
- gps_status: string - GPS fix status
- mavlink_connected: boolean - MAVLink connection status
- sdr_status: string - SDR hardware status
- safety_interlocks: JSON - Status of all safety checks

### TypeScript Interface
```typescript
interface SystemState {
  currentState: 'IDLE' | 'SEARCHING' | 'DETECTING' | 'HOMING' | 'HOLDING';
  homingEnabled: boolean;
  flightMode: string;
  batteryPercent: number;
  gpsStatus: 'NO_FIX' | '2D_FIX' | '3D_FIX' | 'RTK';
  mavlinkConnected: boolean;
  sdrStatus: 'CONNECTED' | 'DISCONNECTED' | 'ERROR';
  safetyInterlocks: {
    modeCheck: boolean;
    batteryCheck: boolean;
    geofenceCheck: boolean;
    signalCheck: boolean;
    operatorCheck: boolean;
  };
}
```

### Relationships
- Singleton in-memory state
- Broadcast via WebSocket

## Mission
**Purpose:** Groups related flights and detections for analysis and reporting

**Key Attributes:**
- id: UUID - Mission identifier
- name: string - Mission name
- start_time: datetime - Mission start
- end_time: datetime - Mission end (nullable)
- search_area: JSON - GeoJSON polygon
- profile_id: UUID - Configuration profile used
- total_detections: integer - Detection count
- notes: text - Operator notes

### TypeScript Interface
```typescript
interface Mission {
  id: string;
  name: string;
  startTime: string;
  endTime?: string;
  searchArea?: GeoJSON.Polygon;
  profileId: string;
  totalDetections: number;
  notes?: string;
}
```

### Relationships
- Has many SignalDetections
- Uses one ConfigProfile
