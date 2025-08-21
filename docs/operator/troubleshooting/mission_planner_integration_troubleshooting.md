# Mission Planner Integration Troubleshooting Guide

## Overview

This troubleshooting guide provides systematic problem resolution for common Mission Planner RF integration issues. Use this guide to quickly diagnose and resolve configuration, parameter, and operational problems.

**Troubleshooting Philosophy:** Systematic approach from basic to advanced, documenting all steps taken.

---

## Quick Diagnostic Checklist

### System Status Verification (Complete in 2 minutes)

**Connection Status:**
- [ ] Mission Planner shows "Connected" status
- [ ] Heartbeat messages received from flight controller
- [ ] PISAD parameters visible in Full Parameter Tree
- [ ] Parameter synchronization active (not "Sync Required")

**Parameter Access:**
- [ ] PISAD parameters editable (not grayed out)
- [ ] Parameter changes save successfully
- [ ] No parameter write permission errors
- [ ] Parameter values within expected ranges

**System Health:**
- [ ] `PISAD_RF_HEALTH` >50%
- [ ] No critical system warnings displayed
- [ ] Telemetry updates flowing normally
- [ ] No hardware disconnection alerts

**Basic Functionality:**
- [ ] RF frequency parameters accept changes
- [ ] Homing enable parameter functional
- [ ] Emergency disable command available
- [ ] Status messages appearing in Mission Planner

---

## Parameter Access Issues

### Problem: PISAD Parameters Not Visible

**Symptoms:**
- PISAD parameters missing from Full Parameter Tree
- Search for "PISAD" returns no results
- Parameter synchronization shows incomplete

**Diagnostic Steps:**
1. **Verify Connection:**
   - Check Mission Planner connection status
   - Confirm MAVLink heartbeat messages received
   - Verify baud rate matches flight controller settings (57600/115200)

2. **Force Parameter Refresh:**
   - Click "Refresh Params" in Full Parameter Tree
   - Wait for complete synchronization (may take 30-60 seconds)
   - Check for parameter update progress indicator

3. **Check PISAD System Status:**
   - Verify PISAD services running on Raspberry Pi
   - Check USB connection between Pi and flight controller
   - Confirm power supply to PISAD system adequate

**Resolution Steps:**
1. **Restart Parameter Synchronization:**
   ```
   In Mission Planner:
   1. Disconnect from flight controller
   2. Wait 10 seconds
   3. Reconnect to flight controller
   4. Navigate to Full Parameter Tree
   5. Wait for automatic parameter sync
   ```

2. **PISAD Service Restart:**
   ```
   On Raspberry Pi:
   sudo systemctl restart pisad
   sudo systemctl status pisad
   ```

3. **Hardware Connection Check:**
   - Verify USB cable connection secure
   - Check for USB power issues
   - Test with different USB port if available

### Problem: Parameters Grayed Out (Read-Only)

**Symptoms:**
- PISAD parameters visible but cannot be modified
- Right-click shows no "Modify" option
- Parameter changes rejected or ignored

**Diagnostic Steps:**
1. **Check Connection Authority:**
   - Verify Mission Planner has parameter write permissions
   - Confirm no other GCS applications connected
   - Check for safety switch armed state if required

2. **Verify Parameter Types:**
   - Some parameters are intentionally read-only (status parameters)
   - Confirm attempting to modify writable parameters
   - Check parameter documentation for access level

**Resolution Steps:**
1. **Enable Parameter Modification:**
   - Ensure safety switch disarmed if equipped
   - Verify no parameter protection enabled
   - Check for firmware version compatibility

2. **Parameter Reset:**
   - Try resetting specific parameter to default
   - Use "Reset to Default" option if available
   - Restart Mission Planner if permissions issue persists

### Problem: Parameter Changes Not Saving

**Symptoms:**
- Parameter modification accepted but reverts to previous value
- Changes appear temporarily then disappear
- Parameter validation errors

**Diagnostic Steps:**
1. **Check Parameter Ranges:**
   - Verify new value within acceptable range
   - Confirm parameter format correct (integer vs. float)
   - Check for parameter validation rules

2. **Monitor Save Process:**
   - Watch for parameter save confirmation
   - Check for error messages during save
   - Verify parameter write completion

**Resolution Steps:**
1. **Parameter Validation:**
   - Use only values within documented ranges
   - For `PISAD_RF_FREQ`: 1,000,000 to 6,000,000,000 Hz
   - For `PISAD_RF_PROFILE`: 0, 1, 2, or 3 only
   - For boolean parameters: 0 or 1 only

2. **Force Parameter Write:**
   - Right-click parameter and select "Write"
   - Verify write completion before proceeding
   - Check parameter value after write operation

---

## Frequency Configuration Errors

### Problem: Frequency Validation Failures

**Symptoms:**
- Frequency parameter shows red highlighting
- "Invalid frequency" error messages
- Frequency changes rejected by system

**Diagnostic Steps:**
1. **Range Validation:**
   - Check frequency within 1MHz - 6GHz range
   - Verify optimal range 24MHz - 1.75GHz for best performance
   - Confirm frequency format (Hz, not MHz)

2. **Regulatory Compliance:**
   - Verify frequency legal in operational location
   - Check for emergency frequency restrictions
   - Confirm no conflicting radio services

**Resolution Steps:**
1. **Frequency Format Correction:**
   ```
   Correct formats:
   - 406MHz = 406000000 (406 million Hz)
   - 121.5MHz = 121500000 (121.5 million Hz)
   - 433.92MHz = 433920000 (433.92 million Hz)
   ```

2. **Use Standard Profiles:**
   - Set `PISAD_RF_PROFILE` to standard profile (0, 1, or 2)
   - Allow automatic frequency configuration
   - Verify profile selection before manual frequency entry

### Problem: Profile Switching Not Working

**Symptoms:**
- Profile parameter changes but frequency doesn't update
- Automatic configuration not occurring
- Profile switch appears but no system response

**Diagnostic Steps:**
1. **Verify Profile Parameter:**
   - Confirm `PISAD_RF_PROFILE` accepts new value
   - Check for parameter synchronization delay
   - Monitor for automatic frequency updates

2. **Check System Response:**
   - Watch for frequency and bandwidth parameter changes
   - Monitor system status messages
   - Verify profile switch confirmation

**Resolution Steps:**
1. **Manual Profile Configuration:**
   ```
   Emergency Profile (0):
   - PISAD_RF_FREQ = 406000000
   - PISAD_RF_BW = 25000
   
   Aviation Profile (1):
   - PISAD_RF_FREQ = 121500000
   - PISAD_RF_BW = 50000
   
   SAR Profile (2):
   - PISAD_RF_FREQ = 162025000
   - PISAD_RF_BW = 25000
   ```

2. **System Restart:**
   - Restart PISAD service to refresh profile configuration
   - Reconnect Mission Planner to reload parameters
   - Verify profile switching after restart

---

## Homing Activation Problems

### Problem: Homing Won't Activate

**Symptoms:**
- `PISAD_HOMING_EN` set to 1 but `PISAD_HOMING_STATE` remains 0
- MAV_CMD_USER_1 command has no effect
- "Homing activation failed" messages

**Diagnostic Steps:**
1. **Check Prerequisites:**
   - Flight controller in GUIDED mode
   - Valid signal detected (`PISAD_SIG_CONF` >50%)
   - Battery level >20%
   - GPS lock obtained
   - No geofence violations

2. **Signal Quality Assessment:**
   - Verify `PISAD_BEARING` showing consistent direction
   - Check `PISAD_BEAR_CONF` >70%
   - Confirm signal classification appropriate
   - Monitor interference levels

**Resolution Steps:**
1. **Mode Verification:**
   ```
   In Mission Planner Flight Mode:
   1. Ensure mode shows "GUIDED"
   2. If not GUIDED, switch to GUIDED mode
   3. Wait for mode change confirmation
   4. Retry homing activation
   ```

2. **Signal Enhancement:**
   - Relocate aircraft for better signal reception
   - Wait for signal confidence to improve
   - Try different altitude for signal clarity
   - Check for interference sources

### Problem: Homing Activates But No Movement

**Symptoms:**
- `PISAD_HOMING_STATE` shows 2 (Active)
- No aircraft movement toward signal
- Velocity commands not affecting flight path

**Diagnostic Steps:**
1. **Velocity Command Verification:**
   - Check for velocity command transmission
   - Verify flight controller receiving commands
   - Monitor flight mode stability

2. **Flight Controller Integration:**
   - Confirm MAVLink communication active
   - Check for flight controller parameter conflicts
   - Verify autopilot accepting velocity commands

**Resolution Steps:**
1. **Manual Test:**
   - Try manual GUIDED mode navigation
   - Test velocity commands from Mission Planner
   - Verify basic autopilot functionality

2. **System Integration Check:**
   - Restart MAVLink communication
   - Verify parameter synchronization
   - Check for system integration errors

---

## Emergency Procedure Issues

### Problem: Emergency Disable Not Working

**Symptoms:**
- MAV_CMD_USER_2 command ignored
- Homing continues after emergency command
- No response to emergency disable

**Diagnostic Steps:**
1. **Command Verification:**
   - Confirm MAV_CMD_USER_2 executed correctly
   - Check command parameter format
   - Verify command transmission to flight controller

2. **System Response Check:**
   - Monitor for emergency disable confirmation
   - Check system status after command
   - Verify RF processing cessation

**Resolution Steps:**
1. **Alternative Override Methods:**
   ```
   Immediate overrides (try in order):
   1. Flight mode change to MANUAL
   2. Flight mode change to STABILIZE
   3. RC transmitter override
   4. Power cycle PISAD system
   ```

2. **Hardware Intervention:**
   - Use RC transmitter manual override
   - Land aircraft immediately
   - Power cycle entire system

### Problem: Mode Change Override Not Working

**Symptoms:**
- Flight mode change doesn't stop homing
- Homing continues in non-GUIDED modes
- Mode change ignored by PISAD system

**Diagnostic Steps:**
1. **Mode Change Verification:**
   - Confirm mode change accepted by flight controller
   - Check actual flight mode versus displayed mode
   - Verify mode change transmission

2. **System Integration:**
   - Check PISAD monitoring of flight mode
   - Verify mode change detection
   - Monitor system response to mode changes

**Resolution Steps:**
1. **Force Mode Change:**
   - Use RC transmitter to force mode change
   - Try multiple mode changes in sequence
   - Verify physical RC override capability

2. **Emergency Landing:**
   - If overrides fail, execute emergency landing
   - Use RC transmitter direct control
   - Land in nearest safe area

---

## System Performance Issues

### Problem: Slow Parameter Updates

**Symptoms:**
- Parameter changes take >5 seconds to apply
- Telemetry updates delayed or intermittent
- System response sluggish

**Diagnostic Steps:**
1. **Communication Performance:**
   - Check MAVLink packet loss rate
   - Monitor telemetry bandwidth utilization
   - Verify connection stability

2. **System Loading:**
   - Check PISAD system CPU usage
   - Monitor memory utilization
   - Verify system not overloaded

**Resolution Steps:**
1. **Communication Optimization:**
   - Reduce telemetry update rates
   - Check for bandwidth saturation
   - Use higher baud rate if supported

2. **System Optimization:**
   - Restart PISAD services to clear memory
   - Check for background processes consuming resources
   - Verify adequate system cooling

### Problem: Telemetry Data Inconsistent

**Symptoms:**
- Parameter values jumping erratically
- Signal strength readings inconsistent
- Bearing information unstable

**Diagnostic Steps:**
1. **Data Quality Assessment:**
   - Monitor raw vs. processed parameter values
   - Check for data filtering issues
   - Verify sensor data consistency

2. **System Health Check:**
   - Monitor `PISAD_RF_HEALTH` parameter
   - Check for hardware malfunction indicators
   - Verify antenna system integrity

**Resolution Steps:**
1. **System Calibration:**
   - Restart signal processing services
   - Recalibrate antenna system if possible
   - Check for environmental interference

2. **Hardware Verification:**
   - Inspect antenna connections
   - Check HackRF USB connection
   - Verify power supply stability

---

## Resolution Documentation Template

### Incident Recording Format

**Problem Description:**
- Date/Time:
- Operator:
- Aircraft/System ID:
- Problem symptoms:

**Diagnostic Steps Taken:**
- [ ] Step 1: [Description and result]
- [ ] Step 2: [Description and result]
- [ ] Step 3: [Description and result]

**Resolution Applied:**
- Method used:
- Result:
- Time to resolution:
- Preventive measures:

**Follow-up Actions:**
- Documentation updated:
- Training requirements:
- System modifications needed:
- Prevention measures implemented:

---

## Escalation Procedures

### When to Escalate

**Level 1 - Operator Resolution (0-15 minutes):**
- Use this troubleshooting guide
- Try standard resolution procedures
- Document problem and steps taken

**Level 2 - Technical Support (15-30 minutes):**
- Contact PISAD technical support
- Provide detailed problem description
- Include system logs and configuration

**Level 3 - Emergency Response (>30 minutes or safety issue):**
- Contact emergency support line
- Implement emergency procedures
- Prioritize aircraft and personnel safety

### Support Contact Information

**Technical Support:**
- Primary: [Support Phone/Email]
- Emergency: [Emergency Contact]
- Documentation: [Support Portal]

**Hardware Support:**
- HackRF Support: [Vendor Contact]
- Flight Controller: [Autopilot Support]
- Mission Planner: [Community Support]

---

**Document Version:** 1.0  
**Last Updated:** 2025-08-20  
**Coverage:** Parameter access, frequency configuration, homing activation, emergency procedures, performance issues  
**Resolution Success Rate Target:** >90% issues resolved using this guide