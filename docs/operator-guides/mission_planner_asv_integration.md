# Mission Planner ASV Integration - Operator Guide

**SUBTASK-6.1.3.3 [20d] - Operator documentation for RF control within Mission Planner workflow**

## Overview

This guide provides comprehensive instructions for operating PISAD's ASV-enhanced RF capabilities through Mission Planner. The integration provides professional-grade signal processing with intuitive Mission Planner controls.

## Prerequisites

- Mission Planner 1.3.80 or later
- PISAD system connected via MAVLink (Cube Orange+)
- ASV integration enabled (completed via Story 6.1)
- Valid HackRF One SDR hardware connected

## Quick Start

### 1. Initial Connection

1. Connect to your drone via Mission Planner as normal
2. Verify PISAD parameters appear in the parameter list:
   - `PISAD_FREQ_PROF` - Frequency Profile Selection
   - `PISAD_FREQ_HZ` - Custom Frequency (Hz)
   - `PISAD_HOMING_EN` - Homing Mode Enable

### 2. Basic Frequency Control

**Emergency Beacon Mode (406 MHz):**
```
Set PISAD_FREQ_PROF = 0
```

**Aviation Beacon Mode (121.5 MHz):**
```
Set PISAD_FREQ_PROF = 1
```

**Custom Frequency Mode:**
```
Set PISAD_FREQ_PROF = 2
Set PISAD_FREQ_HZ = [frequency in Hz]
Example: 406025000 for 406.025 MHz
```

### 3. Enable ASV Homing

```
Set PISAD_HOMING_EN = 1
```

To disable homing:
```
Set PISAD_HOMING_EN = 0
```

## ASV Telemetry Display

### Real-Time Parameters

Mission Planner displays these ASV telemetry values in real-time:

| Parameter | Description | Units | Range |
|-----------|-------------|-------|--------|
| `ASV_BEARING` | Signal bearing from ASV | degrees | 0-360 |
| `ASV_CONFIDENCE` | Bearing confidence | percent | 0-100 |
| `ASV_PRECISION` | Expected precision | degrees | 0.5-20 |
| `ASV_SIG_QUAL` | Signal quality score | percent | 0-100 |
| `ASV_SIG_RSSI` | Signal strength | dBm | -120 to -30 |
| `ASV_INTERF` | Interference detected | flag | 0=No, 1=Yes |
| `ASV_SIG_TYPE` | Signal classification | code | See table below |

### Signal Classification Codes

| Code | Classification | Description |
|------|----------------|-------------|
| 0 | UNKNOWN | Unclassified signal |
| 1 | CONTINUOUS | Continuous beacon |
| 2 | FM_CHIRP | FM chirp beacon |
| 3 | FM_CHIRP_WEAK | Weak FM chirp |
| 4 | INTERFERENCE | Interference detected |
| 5 | BEACON_121_5 | 121.5 MHz ELT |
| 6 | BEACON_406 | 406 MHz ELT |
| 7 | AVIATION | Aviation signal |
| 8 | MULTIPATH | Multipath signal |
| 9 | SPURIOUS | Spurious emission |

### Signal Quality Indicators

| Parameter | Description | Units | Interpretation |
|-----------|-------------|--------|---------------|
| `ASV_RSSI_TREND` | RSSI trend | dB/s | Positive=improving, Negative=degrading |
| `ASV_STABILITY` | Signal stability | percent | Higher=more stable |
| `ASV_FREQ_DRIFT` | Frequency drift | Hz | Drift from center frequency |
| `ASV_MULTIPATH` | Multipath severity | percent | Higher=more multipath |

### Detection Events

| Parameter | Description | Values |
|-----------|-------------|---------|
| `ASV_DET_EVENT` | Detection event type | 1=Detection, 2=Signal Lost, 8=Beacon Confirmed |
| `ASV_DET_STR` | Detection strength | 0-100% |
| `ASV_ANALYZER` | ASV analyzer source | 1=Professional, 2=Standard, 3=Enhanced |

## Operational Workflows

### Emergency Beacon Search

1. **Set Emergency Mode:**
   ```
   PISAD_FREQ_PROF = 0
   ```

2. **Enable Homing:**
   ```
   PISAD_HOMING_EN = 1
   ```

3. **Monitor Telemetry:**
   - Watch `ASV_CONFIDENCE` for signal quality
   - Monitor `ASV_BEARING` for direction
   - Check `ASV_SIG_TYPE` for beacon confirmation (5 or 6)

4. **Follow ASV Guidance:**
   - High confidence (>70%): Follow bearing directly
   - Medium confidence (30-70%): Use search patterns
   - Low confidence (<30%): Expand search area

### Aviation ELT Search

1. **Set Aviation Mode:**
   ```
   PISAD_FREQ_PROF = 1
   ```

2. **Monitor for 121.5 MHz signals:**
   - `ASV_SIG_TYPE = 5` indicates 121.5 MHz ELT
   - Use precision (`ASV_PRECISION`) to judge bearing accuracy

### Custom Frequency Operations

1. **Configure Custom Frequency:**
   ```
   PISAD_FREQ_PROF = 2
   PISAD_FREQ_HZ = [target frequency in Hz]
   ```

2. **Frequency Examples:**
   - 406.025 MHz: `406025000`
   - 121.500 MHz: `121500000`
   - 243.000 MHz: `243000000`

3. **Verify Frequency Setting:**
   - Check status messages for "Custom frequency changed"
   - Monitor `ASV_SIG_TYPE` for expected signal classification

## Troubleshooting

### No ASV Telemetry

**Symptoms:** ASV_* parameters not updating
**Solutions:**
1. Verify MAVLink connection is active
2. Check PISAD system status
3. Confirm HackRF One is connected
4. Restart PISAD services if needed

### Invalid Parameter Values

**Symptoms:** Parameter changes rejected
**Common Issues:**
- `PISAD_FREQ_HZ` out of range (1 MHz - 6 GHz)
- `PISAD_FREQ_PROF` not 0, 1, or 2
- `PISAD_HOMING_EN` not 0 or 1

### Low Signal Confidence

**Symptoms:** `ASV_CONFIDENCE` consistently low
**Actions:**
1. Check for interference (`ASV_INTERF = 1`)
2. Monitor signal stability (`ASV_STABILITY`)
3. Consider altitude or position changes
4. Check for multipath issues (`ASV_MULTIPATH`)

### Frequency Drift Issues

**Symptoms:** High `ASV_FREQ_DRIFT` values
**Solutions:**
1. Check temperature stability
2. Verify frequency calibration
3. Monitor for Doppler effects in moving platforms

## Performance Requirements

### Response Times
- Parameter changes: <50ms
- Bearing updates: <100ms
- Detection events: <500ms

### Accuracy Specifications
- Bearing precision: ±2° (high-quality signals)
- Frequency accuracy: ±100 Hz
- RSSI precision: ±1 dBm

## Safety Considerations

### Operator Override Authority
- Mission Planner retains full parameter control
- Safety systems preserve operator override capability
- Emergency stops function independently of ASV processing

### Validation Checks
- All frequency changes validated against hardware limits
- Invalid parameters automatically rejected
- Status messages confirm all parameter changes

### Backup Procedures
- Manual frequency switching remains available
- Standard beacon detection continues without ASV
- Flight safety systems unaffected by ASV status

## Advanced Features

### Parameter Callbacks
The system supports parameter change callbacks for advanced integrations:
- Custom frequency switching logic
- Automated search pattern adjustments
- Integration with external navigation systems

### Signal Classification
ASV provides detailed signal analysis:
- Chirp characteristic detection
- Interference source identification
- Signal quality assessment
- Multipath analysis

### Telemetry Integration
ASV telemetry integrates with Mission Planner:
- Real-time parameter display
- Status message notifications
- Detection event logging
- Signal trend analysis

## Support Information

### Log Files
ASV integration logs to standard PISAD log files:
- Parameter changes logged with timestamps
- Signal analysis results recorded
- Error conditions documented

### Diagnostic Parameters
For troubleshooting, monitor these additional values:
- Connection status in MAVLink telemetry
- ASV processing health indicators
- Hardware detection status

### Contact Information
For technical support with ASV integration:
- Review PISAD system logs
- Check MAVLink connection quality
- Verify hardware connections
- Contact PISAD support with log files

---

**Document Version:** 1.0  
**Compatible with:** PISAD Story 6.1 ASV Integration  
**Last Updated:** August 2025  
**Operator Training Required:** Basic Mission Planner proficiency