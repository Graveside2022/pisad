# Pre-Flight Safety Checklist

## Overview

This checklist must be completed before every flight test involving the PiSAD system. All items must pass for a GO decision. Any NO-GO item requires resolution before proceeding.

**Date**: \***\*\_\_\_\*\*** **Time**: \***\*\_\_\_\*\*** **Location**: \***\*\_\_\_\*\***  
**Pilot**: \***\*\_\_\_\*\*** **Safety Officer**: \***\*\_\_\_\*\*** **Test ID**: \***\*\_\_\_\*\***

## 1. Hardware Verification

### Power Systems

- [ ] **Battery Voltage**: Main battery >11.4V (3S) or >15.2V (4S)
- [ ] **Pi Power Supply**: 5V supply connected and stable
- [ ] **SDR Power**: USB power verified, no overcurrent warnings
- [ ] **Backup Battery**: Charged and connected (if applicable)
- [ ] **Power Connectors**: All connectors secure, no exposed wires
- [ ] **Current Draw**: Within expected range (<2A for Pi + SDR)

**Status**: [ ] GO [ ] NO-GO **Notes**: \***\*\_\_\_\*\***

### Connections & Mounting

- [ ] **SDR Connection**: RTL-SDR firmly connected to USB 3.0 port
- [ ] **Antenna Mounting**: 433 MHz antenna secure, no damage
- [ ] **Antenna Orientation**: Vertical orientation maintained
- [ ] **SMA Connectors**: Finger-tight plus 1/4 turn
- [ ] **MAVLink Cable**: Connected to flight controller TELEM port
- [ ] **Raspberry Pi Mount**: Secure, vibration dampening in place
- [ ] **Cable Management**: All cables secured, no prop interference

**Status**: [ ] GO [ ] NO-GO **Notes**: \***\*\_\_\_\*\***

### Physical Inspection

- [ ] **Antenna Condition**: No kinks, breaks, or damage
- [ ] **SDR Heat**: Normal temperature, heatsink attached
- [ ] **Pi Temperature**: <60°C at startup
- [ ] **SD Card**: Seated properly, no corruption warnings
- [ ] **LED Indicators**: Power and activity LEDs normal
- [ ] **Enclosure**: Sealed properly if weatherproofing required

**Status**: [ ] GO [ ] NO-GO **Notes**: \***\*\_\_\_\*\***

## 2. Software Version Confirmation

### System Versions

- [ ] **PiSAD Version**: \***\*\_\*\*** (record version)
- [ ] **ArduPilot Version**: \***\*\_\*\*** (record version)
- [ ] **Python Version**: 3.11+ confirmed
- [ ] **FastAPI Running**: Service status active
- [ ] **Frontend Build**: Latest build deployed
- [ ] **Database**: SQLite accessible and not corrupted

### Version Compatibility Check

- [ ] **MAVLink Protocol**: Version 2.0 confirmed
- [ ] **WebSocket Protocol**: Compatible with frontend version
- [ ] **Configuration Files**: Latest version loaded
- [ ] **Safety Parameters**: Match current flight plan

**Status**: [ ] GO [ ] NO-GO **Notes**: \***\*\_\_\_\*\***

## 3. Configuration Validation

### System Configuration

- [ ] **SDR Frequency**: Set to 433.000 MHz
- [ ] **Sample Rate**: 2.4 MHz configured
- [ ] **Gain Setting**: Auto or predetermined value
- [ ] **Detection Threshold**: 6 dB SNR configured
- [ ] **Geofence**: Boundaries set appropriately
- [ ] **Battery Threshold**: 20% minimum configured

### Flight Controller Settings

- [ ] **Flight Modes**: GUIDED mode available
- [ ] **Failsafe**: RTL configured and tested
- [ ] **Geofence FC**: Enabled and boundaries set
- [ ] **Battery Failsafe**: Set to RTL at 21%
- [ ] **RC Failsafe**: Configured for signal loss
- [ ] **EKF Status**: Healthy, GPS 3D fix

**Status**: [ ] GO [ ] NO-GO **Notes**: \***\*\_\_\_\*\***

## 4. Communication Link Tests

### MAVLink Communication

- [ ] **Connection Status**: HEARTBEAT received
- [ ] **Message Rate**: >1 Hz confirmed
- [ ] **Telemetry Data**: All fields populated
- [ ] **Mode Changes**: GUIDED mode switch successful
- [ ] **Command Response**: Velocity commands acknowledged
- [ ] **Latency**: <100ms round trip

**Test Result**: \***\*\_\_\_\*\***  
**Status**: [ ] GO [ ] NO-GO

### WebSocket Communication

- [ ] **Connection**: Frontend connected to backend
- [ ] **RSSI Stream**: Real-time updates visible
- [ ] **Status Updates**: System state changes reflected
- [ ] **Command Transmission**: Button presses registered
- [ ] **Latency**: <50ms for UI updates

**Test Result**: \***\*\_\_\_\*\***  
**Status**: [ ] GO [ ] NO-GO

### Ground Station Link

- [ ] **Telemetry Radio**: Connected and paired
- [ ] **Signal Strength**: >50% at launch point
- [ ] **Mission Planner**: Connected and receiving data
- [ ] **Video Feed**: FPV operational (if applicable)
- [ ] **Range Test**: Communication at 100m confirmed

**Test Result**: \***\*\_\_\_\*\***  
**Status**: [ ] GO [ ] NO-GO

## 5. Safety Interlock Verification

### Individual Interlock Tests

- [ ] **Mode Check**: Blocks commands in non-GUIDED modes
- [ ] **Operator Enable**: Homing requires manual activation
- [ ] **Battery Check**: Disables at <20% battery
- [ ] **Signal Check**: Auto-disable after 10s weak signal
- [ ] **Geofence Check**: Prevents movement outside boundary
- [ ] **Emergency Stop**: Immediately halts all commands

### Interlock Integration Test

- [ ] **Multiple Triggers**: System handles concurrent failures
- [ ] **Recovery**: Normal operation resumes after clearing
- [ ] **Logging**: All safety events recorded
- [ ] **Notifications**: UI shows safety status correctly

**Test Result**: \***\*\_\_\_\*\***  
**Status**: [ ] GO [ ] NO-GO

## 6. SDR Functionality Checks

### Signal Detection

- [ ] **Noise Floor**: <-90 dBm baseline
- [ ] **Test Signal**: Beacon detected at known distance
- [ ] **RSSI Values**: Reasonable range (-30 to -90 dBm)
- [ ] **SNR Calculation**: Positive values for strong signals
- [ ] **Bearing Estimation**: Approximate direction correct
- [ ] **Update Rate**: >5 Hz measurement rate

### Performance Validation

- [ ] **CPU Usage**: <50% with SDR running
- [ ] **Memory Usage**: <1GB for PiSAD process
- [ ] **Temperature**: SDR <70°C after 5 min operation
- [ ] **USB Stability**: No disconnections in 10 min test
- [ ] **Data Pipeline**: No dropped samples warning

**Test Result**: \***\*\_\_\_\*\***  
**Status**: [ ] GO [ ] NO-GO

## 7. Battery and Power Verification

### Battery Health

- [ ] **Cell Balance**: All cells within 0.05V
- [ ] **Voltage Sag**: <0.5V under load
- [ ] **Capacity**: Sufficient for planned mission + 30%
- [ ] **Temperature**: Battery not hot or swollen
- [ ] **Connector**: No burn marks or damage
- [ ] **Charge Rate**: C-rating appropriate for aircraft

### Power Distribution

- [ ] **BEC Output**: 5V ± 0.25V under load
- [ ] **Current Sensor**: Calibrated and reading correctly
- [ ] **Voltage Telemetry**: Matches multimeter reading
- [ ] **Power Module**: No excessive heat
- [ ] **Redundancy**: Backup power system tested

**Measured Values**:

- Main Battery: **\_**V
- Pi Supply: **\_**V
- Current Draw: **\_**A

**Status**: [ ] GO [ ] NO-GO

## 8. Emergency Stop Test

### E-Stop Functionality

- [ ] **Physical Button**: Accessible and functioning
- [ ] **Software E-Stop**: UI button works
- [ ] **Response Time**: <500ms to stop commands
- [ ] **State Preservation**: System remains stable
- [ ] **Recovery Process**: Clear and documented
- [ ] **Team Awareness**: All members know E-stop procedure

### Emergency Procedures Review

- [ ] **Abort Locations**: Identified and clear
- [ ] **Recovery Plan**: Documented for each failure
- [ ] **Communication**: Channel and callsigns confirmed
- [ ] **Medical Kit**: Present and accessible
- [ ] **Fire Extinguisher**: Available for LiPo fires
- [ ] **Contact Info**: Emergency services numbers ready

**Status**: [ ] GO [ ] NO-GO

## 9. Environmental Conditions

### Weather Assessment

- [ ] **Wind Speed**: <15 mph (24 km/h)
- [ ] **Precipitation**: None present or forecast
- [ ] **Visibility**: >1 mile (1.6 km)
- [ ] **Cloud Ceiling**: Above planned altitude
- [ ] **Temperature**: Within operating range
- [ ] **Lightning**: No activity within 10 miles

### Site Conditions

- [ ] **Airspace**: Clear and authorized
- [ ] **Ground Hazards**: Area checked and safe
- [ ] **RF Environment**: Spectrum scan completed
- [ ] **GPS Satellites**: >8 satellites visible
- [ ] **Magnetic Interference**: Compass calibrated

**Current Conditions**:

- Wind: **\_** mph from **\_**
- Temp: **\_**°C
- Humidity: **\_**%

**Status**: [ ] GO [ ] NO-GO

## 10. Final Go/No-Go Decision

### System Readiness Summary

- [ ] Hardware Verification: **GO / NO-GO**
- [ ] Software Verification: **GO / NO-GO**
- [ ] Configuration Valid: **GO / NO-GO**
- [ ] Communications Good: **GO / NO-GO**
- [ ] Safety Systems Ready: **GO / NO-GO**
- [ ] SDR Functional: **GO / NO-GO**
- [ ] Power Systems Good: **GO / NO-GO**
- [ ] E-Stop Tested: **GO / NO-GO**
- [ ] Weather Acceptable: **GO / NO-GO**

### Team Readiness

- [ ] **Pilot Ready**: Confirmed
- [ ] **Safety Officer**: Confirmed
- [ ] **Ground Station**: Confirmed
- [ ] **Spotters**: In position
- [ ] **Test Plan**: Reviewed and understood

## FINAL DECISION

**FLIGHT TEST IS:** [ ] **GO** [ ] **NO-GO**

**Pilot Signature**: **\*\***\_\_\_**\*\*** **Date/Time**: **\*\***\_\_\_**\*\***

**Safety Officer Signature**: **\*\***\_\_\_**\*\*** **Date/Time**: **\*\***\_\_\_**\*\***

**Reason for NO-GO (if applicable)**: **\*\***\_\_\_**\*\***

---

## Post-Flight Actions

- [ ] System powered down safely
- [ ] Data downloaded and backed up
- [ ] Anomalies documented
- [ ] Equipment inspected for damage
- [ ] Batteries discharged to storage voltage
- [ ] Lessons learned recorded
- [ ] Next test planning initiated

**Post-Flight Notes**: **\*\***\_\_\_**\*\***
