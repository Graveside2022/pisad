# PISAD API Documentation

This directory contains the complete API documentation for the PISAD (Precision Intelligent Search and Rescue Drone) system.

## API Overview

PISAD provides a comprehensive REST API built with FastAPI, offering real-time access to system state, configuration, telemetry, and control endpoints.

### Base URL
```
http://<pi-ip>:8080/api
```

### Authentication
Currently, the API uses optional API key authentication (configurable):
```bash
# Headers (if API_KEY_ENABLED=true in config)
Authorization: Bearer <api-key>
```

## API Endpoints

### Core System

#### Health & Status
```
GET  /api/health          - System health check
GET  /api/system/status   - Detailed system status
GET  /api/system/config   - Current system configuration
POST /api/system/config   - Update system configuration
```

#### Telemetry & State
```
GET  /api/telemetry       - Real-time MAVLink telemetry
GET  /api/state           - Current system state machine status
POST /api/state/transition - Force state transition (testing only)
```

### Signal Processing

#### RSSI & Detection
```
GET  /api/analytics/rssi     - Real-time RSSI data
GET  /api/analytics/spectrum - Spectrum analysis data
GET  /api/detections        - Signal detection history
POST /api/detections/clear  - Clear detection history
```

#### SDR Configuration
```
GET  /api/sdr/config      - Current SDR configuration
POST /api/sdr/config      - Update SDR parameters
GET  /api/sdr/status      - SDR hardware status
```

### Navigation & Control

#### Homing Control
```
POST /api/homing/enable   - Enable homing mode
POST /api/homing/disable  - Disable homing mode
GET  /api/homing/status   - Current homing status
```

#### Search Patterns
```
GET  /api/search/patterns - Available search patterns
POST /api/search/start    - Start search pattern
POST /api/search/stop     - Stop current search
```

### Field Testing

#### Test Management
```
GET  /api/testing/runs        - List test runs
POST /api/testing/runs        - Create new test run
GET  /api/testing/runs/{id}   - Get test run details
POST /api/testing/runs/{id}/start - Start test run
POST /api/testing/runs/{id}/stop  - Stop test run
```

#### Beacon Simulation
```
POST /api/testing/beacon/start - Start beacon simulation
POST /api/testing/beacon/stop  - Stop beacon simulation
GET  /api/testing/beacon/status - Beacon simulator status
```

### WebSocket Endpoints

Real-time data streaming:
```
WS /ws/rssi          - Real-time RSSI updates (10Hz)
WS /ws/telemetry     - MAVLink telemetry stream (2Hz)
WS /ws/state         - State machine updates
WS /ws/detections    - Signal detection events
```

## Data Models

### System State
```json
{
  "current_state": "IDLE|SEARCHING|DETECTING|HOMING|HOLDING|EMERGENCY",
  "timestamp": "2025-08-21T10:30:00Z",
  "state_duration": 12.5,
  "previous_state": "SEARCHING",
  "transition_reason": "signal_detected",
  "homing_enabled": false,
  "safety_interlocks": {
    "flight_mode_ok": true,
    "battery_ok": true,
    "signal_present": false,
    "geofence_ok": true
  }
}
```

### RSSI Data
```json
{
  "rssi_dbm": -65.2,
  "noise_floor": -85.1,
  "snr_db": 19.9,
  "confidence": 87.5,
  "timestamp": "2025-08-21T10:30:00.123Z",
  "frequency_hz": 433920000,
  "bandwidth_hz": 10000,
  "detection_active": true
}
```

### Telemetry Data
```json
{
  "position": {
    "latitude": 37.7749,
    "longitude": -122.4194,
    "altitude_msl": 120.5,
    "altitude_rel": 50.0
  },
  "velocity": {
    "north": 2.1,
    "east": -1.3,
    "down": 0.1
  },
  "orientation": {
    "roll": 0.02,
    "pitch": -0.01,
    "yaw": 1.57
  },
  "system": {
    "flight_mode": "GUIDED",
    "armed": true,
    "battery_voltage": 24.2,
    "battery_remaining": 85,
    "gps_fix": "3D_FIX",
    "satellites": 12
  },
  "timestamp": "2025-08-21T10:30:00.456Z"
}
```

### Configuration Schema
```json
{
  "sdr": {
    "frequency": 433920000,
    "sample_rate": 2048000,
    "gain": 30,
    "ppm_correction": 0
  },
  "signal": {
    "rssi_threshold": -70.0,
    "averaging_window": 10,
    "trigger_threshold": 12.0,
    "drop_threshold": 6.0
  },
  "homing": {
    "forward_velocity_max": 5.0,
    "yaw_rate_max": 0.5,
    "approach_velocity": 1.0,
    "signal_loss_timeout": 10.0
  },
  "safety": {
    "velocity_max_mps": 2.0,
    "interlock_enabled": true,
    "emergency_stop_gpio": 23
  }
}
```

## Error Handling

### HTTP Status Codes
- `200` - Success
- `201` - Created
- `400` - Bad Request (validation error)
- `401` - Unauthorized (API key required)
- `403` - Forbidden (safety interlock active)
- `404` - Not Found
- `409` - Conflict (operation not allowed in current state)
- `422` - Unprocessable Entity (validation error)
- `500` - Internal Server Error
- `503` - Service Unavailable (hardware error)

### Error Response Format
```json
{
  "error": {
    "code": "SAFETY_INTERLOCK_ACTIVE",
    "message": "Cannot enable homing: drone not in GUIDED mode",
    "details": {
      "current_mode": "STABILIZE",
      "required_mode": "GUIDED",
      "safety_check": "flight_mode_validation"
    },
    "timestamp": "2025-08-21T10:30:00Z"
  }
}
```

## Rate Limiting

Default rate limits (configurable):
- General API: 100 requests/minute
- WebSocket connections: 10 concurrent
- Telemetry endpoints: 200 requests/minute

## Examples

### Enable Homing Mode
```bash
curl -X POST http://192.168.1.100:8080/api/homing/enable \
  -H "Content-Type: application/json" \
  -d '{
    "confirmation": true,
    "operator_id": "operator_1"
  }'
```

### Get Real-time RSSI
```javascript
const ws = new WebSocket('ws://192.168.1.100:8080/ws/rssi');
ws.onmessage = (event) => {
  const rssiData = JSON.parse(event.data);
  console.log(`RSSI: ${rssiData.rssi_dbm} dBm`);
};
```

### Start Field Test
```bash
curl -X POST http://192.168.1.100:8080/api/testing/runs \
  -H "Content-Type: application/json" \
  -d '{
    "test_type": "detection_range",
    "beacon_config": {
      "frequency": 433920000,
      "power_dbm": 10,
      "pulse_width": 0.001
    },
    "search_pattern": "expanding_square",
    "operator": "test_operator_1"
  }'
```

## Monitoring & Metrics

### Prometheus Metrics
Access at `/metrics` endpoint:

```
# HELP pisad_rssi_processing_seconds Time spent processing RSSI
# TYPE pisad_rssi_processing_seconds histogram
pisad_rssi_processing_seconds_bucket{le="0.01"} 142
pisad_rssi_processing_seconds_bucket{le="0.05"} 456
pisad_rssi_processing_seconds_bucket{le="0.1"} 567
pisad_rssi_processing_seconds_bucket{le="+Inf"} 567

# HELP pisad_mavlink_latency_seconds MAVLink communication latency
# TYPE pisad_mavlink_latency_seconds histogram
pisad_mavlink_latency_seconds_bucket{le="0.01"} 234
pisad_mavlink_latency_seconds_bucket{le="0.05"} 445
pisad_mavlink_latency_seconds_bucket{le="0.1"} 456
pisad_mavlink_latency_seconds_bucket{le="+Inf"} 456

# HELP pisad_state_transitions_total Number of state transitions
# TYPE pisad_state_transitions_total counter
pisad_state_transitions_total{from="IDLE",to="SEARCHING"} 12
pisad_state_transitions_total{from="SEARCHING",to="DETECTING"} 8
pisad_state_transitions_total{from="DETECTING",to="HOMING"} 5

# HELP pisad_detections_total Number of signal detections
# TYPE pisad_detections_total counter
pisad_detections_total{confidence="high"} 15
pisad_detections_total{confidence="medium"} 23
pisad_detections_total{confidence="low"} 45
```

## Security Considerations

### API Key Management
```bash
# Generate new API key
curl -X POST http://localhost:8080/api/auth/generate-key \
  -H "Authorization: Bearer <admin-key>"

# Rotate API key
curl -X PUT http://localhost:8080/api/auth/rotate-key \
  -H "Authorization: Bearer <current-key>"
```

### CORS Configuration
Configure allowed origins in `config/default.yaml`:
```yaml
API_CORS_ENABLED: true
API_CORS_ORIGINS: 
  - "http://localhost:3000"
  - "http://192.168.1.0/24"
```

### Network Security
- Use HTTPS in production deployments
- Configure firewall to restrict API access
- Consider VPN for remote operations
- Monitor access logs for suspicious activity

## Development & Testing

### OpenAPI Specification
Interactive API documentation available at:
- Swagger UI: `http://<pi-ip>:8080/docs`
- ReDoc: `http://<pi-ip>:8080/redoc`
- OpenAPI JSON: `http://<pi-ip>:8080/openapi.json`

### API Testing
```bash
# Run API integration tests
uv run pytest tests/backend/integration/test_api_*.py

# Test specific endpoint
uv run pytest tests/backend/integration/test_api_homing.py -v

# Generate API documentation
uv run python scripts/generate_openapi.py
```

### Mock Services
For development without hardware:
```bash
# Start with mock SDR
export USE_MOCK_HARDWARE=true
uv run python -m src.backend.main

# Start with SITL simulation
export MAVLINK_CONNECTION="tcp://localhost:5760"
uv run python -m src.backend.main
```

## Version Information

- **API Version**: 1.0.0
- **OpenAPI Version**: 3.0.0
- **Last Updated**: 2025-08-21
- **Compatibility**: PISAD v1.0.0+

For detailed implementation information, see the individual service documentation files in this directory.