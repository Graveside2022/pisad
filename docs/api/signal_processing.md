# Signal Processing and Safety Command Pipeline Documentation

## Overview

This document describes the signal processing algorithms and safety command pipeline implemented in Phase 3 of Story 4.3, covering PRD requirements FR6, FR7, FR8, FR12, and NFR7.

## Signal Processing Components

### 1. Noise Floor Estimation (FR6)

The system uses a **10th percentile method** for noise floor estimation, which provides robust performance in the presence of intermittent signals.

**Algorithm:**
```python
def update_noise_floor(readings: list[float]) -> float:
    """Calculate noise floor using 10th percentile of RSSI history."""
    if len(readings) < 10:
        return current_noise_floor  # Need minimum samples

    return np.percentile(readings, 10)
```

**Rationale:**
- The 10th percentile represents the noise level when no signal is present
- More robust than minimum value (which could be an outlier)
- Less affected by intermittent signals than mean/median

### 2. Debounced State Transitions (FR7)

The signal state controller implements hysteresis with two thresholds to prevent false triggers:

- **Trigger Threshold**: 12 dB SNR - Signal must exceed this to start detection
- **Drop Threshold**: 6 dB SNR - Signal must fall below this to lose detection

**State Machine:**
```
NO_SIGNAL → RISING (SNR ≥ 12dB)
RISING → CONFIRMED (after 0.5s confirmation time)
CONFIRMED → FALLING (SNR < 6dB)
FALLING → LOST (after 1.0s drop time)
LOST → NO_SIGNAL (after 2.0s timeout)
```

**Benefits:**
- Prevents brief noise spikes from triggering false positives
- Maintains lock during temporary signal dips
- Provides smooth transitions for flight control

### 3. Signal Quality Filtering

#### EWMA (Exponentially Weighted Moving Average)
```python
filtered_value = α * new_value + (1 - α) * previous_value
```
- Default α = 0.3 for responsive yet smooth filtering
- Reduces high-frequency noise while preserving signal trends

#### Simple Moving Average
- Maintains buffer of last 100 RSSI readings
- Used for trend analysis and gradient calculation

### 4. Anomaly Detection (NFR7)

Statistical anomaly detection maintains <5% false positive rate using Z-score analysis:

```python
def detect_anomaly(snr: float) -> bool:
    mean = np.mean(signal_history)
    std = np.std(signal_history)
    z_score = abs((snr - mean) / std)

    return z_score > 3.0  # 3-sigma threshold
```

**False Positive Prevention:**
- Anomalous signals receive 0% confidence score
- Requires stable signal history (≥10 samples)
- Tracks false positive rate continuously

## Safety Command Pipeline

### 1. Command Priority System

Commands are prioritized to ensure critical operations execute first:

| Priority | Level | Use Case | Example Commands |
|----------|-------|----------|-----------------|
| 0 | EMERGENCY | Immediate safety response | Emergency Stop, RTL |
| 1 | CRITICAL | Safety-critical operations | Geofence violation response |
| 2 | HIGH | Important operations | Start/stop homing |
| 3 | NORMAL | Regular commands | Waypoint navigation |
| 4 | LOW | Background tasks | Telemetry updates |

### 2. Safety Interlock System

All commands (except emergency) must pass safety checks:

```python
Safety Checks:
├── Mode Check (GUIDED mode required)
├── Operator Activation (manual enable)
├── Signal Quality (SNR > threshold)
├── Battery Level (> 20%)
└── Geofence Boundary (FR8)
```

### 3. Command Validation & Sanitization

Each command type has specific validators:

**Position Commands:**
- Latitude: -90° to +90°
- Longitude: -180° to +180°
- Altitude: 0 to 500m

**Velocity Commands:**
- Maximum velocity: 20 m/s per axis
- Vector magnitude check

**Mode Commands:**
- Whitelist of valid flight modes
- Transition validation

### 4. Rate Limiting

Prevents command flooding and ensures stable operation:
- Default: 10 commands/second
- Configurable per deployment
- Emergency commands bypass rate limiting

### 5. Emergency Stop (100ms Requirement)

Emergency commands execute immediately with minimal checks:

```python
async def execute_emergency_stop():
    start = time.perf_counter()

    # Bypass queue
    await mavlink.emergency_stop()
    await safety_system.emergency_stop()

    execution_time = (time.perf_counter() - start) * 1000
    assert execution_time < 100  # Must complete in 100ms
```

### 6. Geofence Enforcement (FR8)

Automatic boundary enforcement prevents operations outside safe area:

```python
class GeofenceCheck:
    def check_position(lat, lon) -> bool:
        distance = haversine(lat, lon, center_lat, center_lon)
        return distance <= radius_meters
```

**Enforcement Actions:**
- Commands blocked if they would violate geofence
- Automatic RTL if drone exits geofence
- Override only in emergency situations

### 7. Command Audit Logging (FR12)

Comprehensive logging for all state transitions and commands:

```python
@dataclass
class CommandAuditEntry:
    command_id: str
    timestamp: datetime
    command_type: str
    priority: int
    source: str
    safety_status: dict[str, bool]
    execution_time_ms: float
    success: bool
    error: str | None
```

**Logged Information:**
- All command submissions
- Safety check results
- Execution times
- Success/failure status
- State transitions
- Signal detection events

## Integration Architecture

```
┌─────────────────┐     ┌──────────────────┐
│ SDR Service     │────>│ Signal Processor │
└─────────────────┘     └──────────────────┘
                               │
                               v
                    ┌──────────────────────┐
                    │ Signal State         │
                    │ Controller           │
                    └──────────────────────┘
                               │
                               v
                    ┌──────────────────────┐
                    │ State Machine        │
                    └──────────────────────┘
                               │
                               v
                    ┌──────────────────────┐     ┌─────────────────┐
                    │ Command Pipeline     │────>│ MAVLink Service │
                    └──────────────────────┘     └─────────────────┘
                               ┑
                    ┌──────────────────────┐
                    │ Safety Interlocks    │
                    └──────────────────────┘
```

## Performance Metrics

### Signal Processing
- Noise floor update rate: 100 Hz
- State transition latency: < 10ms
- False positive rate: < 5%
- Signal confirmation time: 500ms
- Signal drop timeout: 1000ms

### Command Pipeline
- Emergency stop response: < 100ms
- Command validation: < 5ms
- Rate limit: 10 commands/second
- Queue capacity: 100 commands
- Audit log retention: 10,000 entries

## Configuration Parameters

```python
# Signal Processing
TRIGGER_THRESHOLD = 12.0  # dB SNR to trigger detection
DROP_THRESHOLD = 6.0      # dB SNR to drop detection
CONFIRMATION_TIME = 0.5   # seconds to confirm signal
DROP_TIME = 1.0          # seconds before declaring lost
EWMA_ALPHA = 0.3         # EWMA smoothing factor
ANOMALY_Z_SCORE = 3.0    # Standard deviations for anomaly

# Command Pipeline
RATE_LIMIT = 10.0        # commands per second
MAX_QUEUE_SIZE = 100     # maximum queued commands
EMERGENCY_TIMEOUT = 0.1  # 100ms for emergency commands
GEOFENCE_RADIUS = 500    # meters from center
MIN_BATTERY = 20.0       # minimum battery percentage
```

## Testing & Validation

### Unit Tests
- Signal state transitions with various SNR patterns
- False positive rate validation
- Anomaly detection accuracy
- Command validation for all types
- Safety interlock scenarios

### Integration Tests
- End-to-end signal detection to command execution
- Emergency stop timing verification
- Geofence enforcement validation
- Audit logging completeness
- Rate limiting effectiveness

### Performance Tests
- 100ms emergency stop requirement
- 10-second startup time
- Signal processing latency
- Command queue throughput
- Memory usage under load

## Safety Considerations

1. **Fail-Safe Defaults**: System defaults to safe states on error
2. **Emergency Override**: Emergency commands always execute
3. **Redundant Checks**: Multiple safety layers prevent unsafe operations
4. **Audit Trail**: Complete logging for incident analysis
5. **Graceful Degradation**: System remains operational with degraded sensors

## Future Enhancements

- Machine learning for signal pattern recognition
- Adaptive thresholds based on environment
- Predictive geofence warnings
- Command rollback capability
- Multi-drone coordination safety
