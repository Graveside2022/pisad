# Emergency Procedures for PiSAD System Failures

## Quick Reference Decision Tree

```
SYSTEM FAILURE DETECTED
         |
    Is aircraft airborne?
    /            \
   YES            NO
    |             |
    v             v
Execute RTL    Power cycle
    |          and debug
    v
Monitor descent
    |
Land safely -> Investigate
```

## 1. Loss of MAVLink Communication

### Immediate Actions (0-10 seconds)

1. **ANNOUNCE**: "MAVLink lost" to all team members
2. **CHECK**: Ground station telemetry for alternate link
3. **ATTEMPT**: Single reconnection via UI button
4. **MONITOR**: Aircraft behavior via visual/FPV

### If Communication Not Restored (10-30 seconds)

1. **SWITCH**: RC pilot takes manual control
2. **MODE**: Change to STABILIZE or LOITER via RC
3. **DISABLE**: PiSAD homing via emergency stop
4. **RTL**: Initiate return-to-launch if safe

### Recovery Procedure

```python
# MAVLink Recovery Sequence
1. Check USB/serial connection physical
2. Restart MAVLink service:
   sudo systemctl restart pisad-mavlink
3. Verify baud rate: 57600
4. Check SYSID_THISMAV matches
5. Monitor /var/log/pisad/mavlink.log
```

### Post-Incident Actions

- Document time of failure
- Save system logs
- Check for USB disconnection
- Verify flight controller health
- Review telemetry logs for cause

## 2. Unintended Homing Activation

### Immediate Actions

1. **ANNOUNCE**: "Unintended homing" loudly
2. **PRESS**: Emergency stop button (physical or UI)
3. **OVERRIDE**: RC pilot switch to STABILIZE
4. **OBSERVE**: Confirm vehicle stopped moving

### Containment

1. **HOLD**: Maintain manual control
2. **ALTITUDE**: Climb to safe altitude if needed
3. **POSITION**: Move to safe area away from obstacles
4. **DISABLE**: Turn off PiSAD homing system

### Investigation Checklist

- [ ] Check operator enable button state
- [ ] Review safety interlock status
- [ ] Verify signal strength readings
- [ ] Check for false positive detection
- [ ] Review command history log

### Safe Recovery

```bash
# Disable homing and investigate
curl -X POST http://pisad.local:8000/api/control/emergency-stop
# Check system state
curl http://pisad.local:8000/api/system/status
# Review recent commands
tail -n 100 /var/log/pisad/commands.log
```

## 3. Signal Processor Crash

### Detection Indicators

- RSSI values frozen or missing
- WebSocket disconnection
- High CPU alert
- Process not responding

### Immediate Response

1. **NOTIFY**: "Signal processor down"
2. **DISABLE**: Homing capability immediately
3. **MAINTAIN**: Current flight mode
4. **PREPARE**: For manual recovery

### Recovery Sequence

```bash
# Check process status
ps aux | grep signal_processor
# Attempt restart
sudo systemctl restart pisad-signal
# If failed, hard restart
sudo killall -9 python3
sudo systemctl start pisad-signal
# Verify operation
curl http://pisad.local:8000/api/signal/status
```

### Fallback Procedure

If processor won't restart:

1. Continue mission without homing
2. Use manual search pattern
3. Land when battery at 25%
4. Troubleshoot on ground

## 4. SDR Hardware Failure

### Symptoms

- "SDR not found" error
- RSSI reading -999 or null
- USB overcurrent warning
- Kernel USB errors in dmesg

### Immediate Actions

1. **ANNOUNCE**: "SDR failure"
2. **ABORT**: Homing operations
3. **NOTE**: Last known signal location
4. **SWITCH**: To manual search mode

### Diagnostic Steps

```bash
# Check USB devices
lsusb | grep RTL
# Check kernel messages
dmesg | tail -20
# Test SDR directly
rtl_test -t
# Check USB power
cat /sys/bus/usb/devices/*/power/level
```

### Field Recovery Attempts

1. **Unplug/Replug**: SDR from USB port
2. **Try Different Port**: USB 2.0 if 3.0 fails
3. **Check Temperature**: Let cool if overheated
4. **Swap SDR**: If spare available
5. **Reset USB Bus**:
   ```bash
   sudo modprobe -r rtl2832
   sudo modprobe rtl2832
   ```

## 5. Battery Critical Level

### Warning Thresholds

- 25%: First warning - prepare to land
- 22%: Start RTL immediately
- 20%: PiSAD auto-disable
- 18%: Emergency land now

### Response Procedure

1. **At 25%**: Announce "Battery warning"
2. **At 22%**: Initiate RTL
3. **At 20%**: Confirm PiSAD disabled
4. **Monitor**: Voltage sag during RTL
5. **Land**: At home or nearest safe spot

### Emergency Landing Protocol

If battery critical during homing:

1. **STOP**: All autonomous movement
2. **DESCEND**: Controlled vertical descent
3. **ANNOUNCE**: "Emergency landing"
4. **CLEAR**: Ensure landing zone safe
5. **LAND**: Priority on aircraft safety

## 6. GPS Loss During Homing

### Initial Response

1. **DETECT**: "GPS: No Fix" warning
2. **HOLD**: Current position (LOITER)
3. **ALTITUDE**: Maintain or climb
4. **WAIT**: 30 seconds for recovery

### If GPS Not Recovered

1. **SWITCH**: To STABILIZE mode
2. **NAVIGATE**: Using visual references
3. **USE**: Compass heading if available
4. **FPV**: Navigate using video feed
5. **LAND**: At nearest safe location

### GPS Recovery Procedure

```python
GPS Recovery Steps:
1. Check satellite count
2. Verify antenna connection
3. Move away from interference
4. Wait for HDOP < 2.0
5. Ensure 3D fix before continuing
```

## 7. Manual Override Instructions

### Override Activation Methods

#### Method 1: RC Transmitter

1. **SWITCH**: Mode switch to STABILIZE
2. **THROTTLE**: Take control immediately
3. **CONFIRM**: PiSAD commands stopped

#### Method 2: Ground Station

1. **CLICK**: Emergency Stop in UI
2. **CONFIRM**: Dialog if presented
3. **VERIFY**: State changed to IDLE

#### Method 3: SSH Emergency

```bash
ssh pi@pisad.local
curl -X POST http://localhost:8000/api/control/emergency-stop
```

#### Method 4: Physical Power

1. **LOCATE**: Pi power switch
2. **DISCONNECT**: Power to Pi only
3. **MAINTAIN**: Flight controller power
4. **FLY**: Manual only

### Override Priority Levels

1. **Level 1**: RC pilot override (instant)
2. **Level 2**: Ground station command (< 1s)
3. **Level 3**: SSH/terminal (< 5s)
4. **Level 4**: Physical disconnect (< 10s)

## 8. Communication Protocols

### Emergency Callouts

| Situation        | Callout               | Response Required          |
| ---------------- | --------------------- | -------------------------- |
| System failure   | "[Component] FAILURE" | "ACKNOWLEDGED"             |
| Taking control   | "TAKING MANUAL"       | "MANUAL CONFIRMED"         |
| Emergency stop   | "E-STOP E-STOP"       | All: "E-STOP ACKNOWLEDGED" |
| Landing          | "EMERGENCY LANDING"   | "AREA CLEAR"               |
| Battery critical | "BATTERY CRITICAL"    | "RTL IMMEDIATELY"          |
| Fire/Smoke       | "FIRE FIRE FIRE"      | "LANDING NOW"              |

### Team Roles During Emergency

**Pilot in Command (PIC)**

- Final decision authority
- Executes emergency procedures
- Communicates with ATC if required

**Safety Officer**

- Calls out emergencies
- Monitors all systems
- Documents incidents
- Ensures area safety

**Ground Station Operator**

- Executes E-stop if needed
- Monitors telemetry
- Assists with navigation
- Records data

**Visual Observer**

- Maintains visual contact
- Calls out obstacles
- Confirms aircraft state
- Assists with landing

## 9. Decision Trees

### System Failure Decision Tree

```
FAILURE DETECTED
       |
  Safety Critical?
    /      \
   YES      NO
    |        |
    v        v
E-STOP    Continue
    |     with caution
    v        |
  RTL     Monitor
    |        |
    v        v
  Land    Complete
         mission
```

### Communication Loss Tree

```
COMMS LOST
      |
   Which link?
   /    |    \
  RC  MAVLink WebSocket
  |     |        |
LAND   RTL    Continue
NOW           (no homing)
```

### Power Failure Tree

```
POWER ISSUE
      |
  Which system?
   /       \
Flight    PiSAD
  |         |
LAND    Disable
ASAP    homing
```

## 10. Post-Emergency Procedures

### Immediate Actions (0-5 minutes)

1. **SAFE**: Ensure aircraft secured
2. **POWER**: Disconnect batteries
3. **PRESERVE**: Don't clear logs
4. **DOCUMENT**: Initial observations
5. **PHOTO**: System state/damage

### Short Term (5-30 minutes)

1. **DOWNLOAD**: All logs immediately
2. **BACKUP**: Create 3 copies
3. **INTERVIEW**: Team members
4. **TIMELINE**: Create event sequence
5. **SECURE**: Equipment from weather

### Investigation (Same day)

1. **LOGS**: Analyze all system logs
2. **REPLAY**: Telemetry data
3. **TEST**: Suspected components
4. **REPORT**: Initial findings
5. **DECIDE**: Equipment disposition

### Reporting Requirements

```yaml
Incident Report Must Include:
  - Date/Time/Location
  - Weather conditions
  - System configuration
  - Failure description
  - Actions taken
  - Result/damage
  - Root cause (if known)
  - Recommendations
  - Logs attached
```

## Emergency Contact Information

### On-Site Emergency

- **Site Safety Officer**: [Name] - [Phone]
- **Medical Emergency**: 911
- **Fire Department**: [Local number]
- **Hospital**: [Name and address]

### Technical Support

- **Lead Developer**: [Name] - [Phone]
- **Systems Engineer**: [Name] - [Phone]
- **Project Manager**: [Name] - [Phone]

### Regulatory

- **FAA Incident**: 1-844-FLY-MY-UA
- **Insurance**: [Company] - [Policy#]
- **Legal Contact**: [Name] - [Phone]

## Training Requirements

All team members must:

1. Read this document completely
2. Practice E-stop procedures
3. Know their emergency role
4. Participate in drills
5. Sign acknowledgment

**Last Drill Date**: \***\*\_\_\_\*\***
**Next Drill Due**: \***\*\_\_\_\*\***

---

**Document Version**: 1.0  
**Last Updated**: 2025-08-12  
**Review Frequency**: Before each test session
