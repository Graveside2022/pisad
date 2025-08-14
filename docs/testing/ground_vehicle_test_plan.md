# Ground Vehicle RF Validation Test Plan

## Executive Summary

This document defines the test methodology for validating PiSAD RF detection and homing capabilities using ground vehicles prior to flight testing. The plan ensures comprehensive system validation while maintaining safety and reproducibility.

## Test Equipment Requirements

### Required Hardware

| Equipment         | Specification                           | Purpose             | Quantity |
| ----------------- | --------------------------------------- | ------------------- | -------- |
| RF Beacon         | 433 MHz, 100mW output                   | Signal source       | 1        |
| Spectrum Analyzer | 400-500 MHz range, -120 dBm sensitivity | Signal verification | 1        |
| Test Vehicle      | Ground vehicle with payload mount       | Platform            | 1        |
| GPS Logger        | <3m accuracy, 10Hz update               | Position tracking   | 2        |
| Power Meter       | RF power measurement                    | Calibration         | 1        |
| Laptop            | Running PiSAD interface                 | Control station     | 1        |
| Raspberry Pi 5    | With SDR and antennas                   | Detection system    | 1        |
| Battery Pack      | 12V, >5Ah capacity                      | Power supply        | 1        |
| Weather Station   | Wind, temp, humidity                    | Environmental data  | 1        |

### Required Software

- PiSAD system (latest version)
- GPS logging software
- Spectrum analyzer software
- Data analysis tools (Python/Excel)
- MAVLink ground station (Mission Planner/QGroundControl)

## Test Course Setup

### Course Layout

```
Start Point (0m)
    |
    | --- 50m --- Waypoint 1 (Signal strength measurement)
    |
    | --- 100m -- Waypoint 2 (Signal strength measurement)
    |
    | --- 200m -- Waypoint 3 (Signal strength measurement)
    |
    | --- 300m -- Waypoint 4 (Detection threshold test)
    |
    | --- 500m -- Waypoint 5 (Maximum range test)
    |
    L --- Turn Point --- Cross-track test path (perpendicular)
```

### Measurement Points

1. **Static Test Points**: 0m, 50m, 100m, 200m, 300m, 500m
2. **Dynamic Test Paths**: Straight line, figure-8, random walk
3. **Cross-Track Points**: 90° offset at 100m intervals
4. **Elevation Points**: If terrain permits, test at different elevations

### Safety Zones

- Establish 10m safety buffer around test vehicle
- Define emergency stop zones
- Mark spectator areas minimum 20m from test path
- Ensure clear line of sight for entire course

## RF Signal Measurement Procedures

### Pre-Test Calibration

1. **Beacon Calibration**
   - Verify beacon frequency: 433.000 MHz ± 1 kHz
   - Measure output power: 100mW (20 dBm) ± 1 dB
   - Check modulation: Continuous wave (CW) or specified pattern
   - Record antenna type and orientation

2. **SDR Calibration**
   - Set center frequency: 433 MHz
   - Configure sample rate: 2.4 MHz
   - Set gain: Auto or fixed at 30 dB
   - Verify noise floor: < -100 dBm

3. **System Synchronization**
   - Sync all GPS devices to UTC
   - Verify MAVLink connection
   - Confirm WebSocket data stream
   - Test emergency stop functionality

### Signal Strength Measurements

1. **Static Measurements**

   ```
   For each measurement point:
   1. Position vehicle at marked waypoint
   2. Record GPS coordinates
   3. Measure RSSI for 60 seconds
   4. Calculate mean, std dev, min, max
   5. Record spectrum analyzer reading
   6. Note environmental conditions
   ```

2. **Dynamic Measurements**

   ```
   During vehicle movement:
   1. Maintain constant speed (5 mph / 8 km/h)
   2. Log RSSI at 10 Hz
   3. Log GPS position at 10 Hz
   4. Record vehicle heading
   5. Note terrain features
   ```

3. **Interference Testing**
   ```
   At each test point:
   1. Baseline measurement (beacon only)
   2. Add WiFi interference (2.4 GHz)
   3. Add cellular interference (LTE)
   4. Add vehicle electronics noise
   5. Record degradation levels
   ```

## Performance Criteria

### Minimum Acceptable Performance

| Metric              | Requirement         | Measurement Method              |
| ------------------- | ------------------- | ------------------------------- |
| Detection Range     | >300m line-of-sight | GPS distance at first detection |
| Signal Threshold    | 6 dB SNR            | RSSI measurement                |
| Detection Latency   | <1 second           | Timestamp comparison            |
| False Positive Rate | <1%                 | Count over test duration        |
| Bearing Accuracy    | ±15°                | Compared to GPS bearing         |
| Update Rate         | >5 Hz               | WebSocket message rate          |
| System Uptime       | >95%                | Total run time / test time      |

### Performance Goals

| Metric              | Goal     | Rationale                      |
| ------------------- | -------- | ------------------------------ |
| Detection Range     | >500m    | Extended operational range     |
| Signal Threshold    | 3 dB SNR | Better weak signal performance |
| Detection Latency   | <500ms   | Faster response                |
| False Positive Rate | <0.1%    | Higher reliability             |
| Bearing Accuracy    | ±10°     | More precise homing            |
| Update Rate         | >10 Hz   | Smoother tracking              |

## Vehicle Test Parameters

### Speed Profiles

1. **Slow Speed**: 5 mph (8 km/h) - Initial validation
2. **Medium Speed**: 15 mph (24 km/h) - Typical operation
3. **High Speed**: 25 mph (40 km/h) - Stress testing
4. **Variable Speed**: 5-25 mph - Real-world simulation

### Distance Parameters

1. **Close Range**: 0-100m - Strong signal validation
2. **Medium Range**: 100-300m - Operational range
3. **Long Range**: 300-500m - Maximum capability
4. **Beyond Range**: >500m - Loss of signal behavior

### Movement Patterns

1. **Linear Approach**: Direct path to beacon
2. **Circular Orbit**: Constant distance, varying bearing
3. **Spiral Pattern**: Decreasing distance approach
4. **Random Walk**: Unpredictable movement
5. **Stop-and-Go**: Intermittent movement

## Environmental Recording

### Required Environmental Data

| Parameter           | Unit    | Recording Frequency | Acceptable Range     |
| ------------------- | ------- | ------------------- | -------------------- |
| Temperature         | °C      | Every 5 minutes     | -10 to +50           |
| Humidity            | %       | Every 5 minutes     | 0 to 100             |
| Wind Speed          | m/s     | Every minute        | 0 to 15              |
| Wind Direction      | degrees | Every minute        | 0 to 360             |
| Precipitation       | mm/hr   | Continuous          | Note if present      |
| Solar Radiation     | W/m²    | Every 10 minutes    | For interference     |
| Barometric Pressure | hPa     | Every 10 minutes    | Record for reference |

### Environmental Test Conditions

1. **Ideal Conditions**
   - Clear weather
   - Wind <5 m/s
   - Temperature 15-25°C
   - No precipitation

2. **Adverse Conditions** (separate tests)
   - Rain/moisture
   - High wind (10-15 m/s)
   - Temperature extremes
   - Fog/reduced visibility

## Data Collection Template

### Test Run Header

```yaml
Test_ID: GV_YYYYMMDD_NNN
Date: YYYY-MM-DD
Start_Time: HH:MM:SS UTC
End_Time: HH:MM:SS UTC
Operator: [Name]
Location: [GPS Coordinates]
Weather: [Summary]
System_Version: [PiSAD version]
```

### Measurement Record

```csv
timestamp,gps_lat,gps_lon,distance_m,rssi_db,snr_db,bearing_deg,speed_mps,heading_deg,temp_c,wind_mps
```

### Test Results Summary

```yaml
Detection_Range_Max: [meters]
Detection_Range_Avg: [meters]
RSSI_at_100m: [dB]
RSSI_at_300m: [dB]
SNR_Threshold_Met: [Yes/No]
False_Positives: [count]
System_Failures: [count]
Test_Result: [PASS/FAIL]
```

## Test Execution Procedure

### Pre-Test Checklist

- [ ] Weather conditions within limits
- [ ] Test area cleared and secured
- [ ] All equipment powered and functioning
- [ ] GPS fix acquired (all devices)
- [ ] Communications established
- [ ] Data logging started
- [ ] Emergency procedures reviewed
- [ ] Safety equipment ready

### Test Sequence

1. **System Startup** (15 minutes)
   - Power on all systems
   - Wait for GPS lock
   - Verify MAVLink connection
   - Start data logging
   - Perform function checks

2. **Calibration Run** (10 minutes)
   - Known distance measurement
   - Verify RSSI readings
   - Confirm GPS accuracy
   - Test emergency stop

3. **Test Execution** (60-90 minutes)
   - Static point measurements
   - Dynamic path following
   - Interference testing
   - Edge case scenarios

4. **Data Verification** (10 minutes)
   - Confirm data saved
   - Quick field analysis
   - Note any anomalies
   - Backup data files

### Post-Test Procedure

1. Safe system shutdown
2. Data backup (3 copies)
3. Equipment inspection
4. Preliminary analysis
5. Test report generation

## Success Criteria

### Test Pass Requirements

1. Detection range exceeds 300m in clear conditions
2. SNR remains above 6 dB within operational range
3. No critical system failures during test
4. False positive rate below 1%
5. All safety interlocks function correctly
6. Data collection >90% complete

### Test Fail Conditions

1. Detection range less than 300m
2. System crash or hang requiring reboot
3. Safety interlock failure
4. Data loss >10%
5. Unable to maintain stable detection
6. Environmental conditions out of range

## Risk Mitigation

### Identified Risks

| Risk              | Likelihood | Impact | Mitigation                       |
| ----------------- | ---------- | ------ | -------------------------------- |
| RF Interference   | Medium     | High   | Spectrum scan before test        |
| Equipment Failure | Low        | High   | Backup equipment ready           |
| Weather Changes   | Medium     | Medium | Monitor forecast, abort criteria |
| GPS Loss          | Low        | High   | Multiple GPS units               |
| Data Corruption   | Low        | High   | Real-time backup                 |
| Vehicle Collision | Low        | High   | Safety zones, spotters           |

### Abort Criteria

- Wind exceeds 15 m/s
- Lightning within 10 km
- Equipment malfunction affecting safety
- Loss of vehicle control
- Interference preventing operation
- Team member injury

## Reporting

### Required Deliverables

1. Test execution report
2. Raw data files (CSV/JSON)
3. Analysis summary with graphs
4. Anomaly investigation reports
5. Recommendations for flight testing
6. Updated system configuration

### Report Distribution

- Engineering team
- Safety officer
- Project manager
- Test team members
- Archive for compliance
