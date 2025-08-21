# Homing Workflow Procedures

## Overview

This document provides comprehensive procedures for activating, monitoring, and controlling PISAD homing operations through Mission Planner. Follow these workflows to ensure safe and effective autonomous RF beacon homing.

**Safety Priority:** Operator maintains full override authority at all times (PRD-FR11)

---

## Pre-Activation Safety Checklist

### Critical Safety Verification (Complete ALL items before activation)

**Flight Controller Status:**
- [ ] Flight controller in **GUIDED** mode (required for velocity commands)
- [ ] Aircraft armed and ready for autonomous flight
- [ ] GPS position lock with HDOP <2.0
- [ ] Battery level >20% remaining (automatic safety threshold)
- [ ] No critical system warnings or errors displayed

**RF Signal Verification:**
- [ ] Valid signal detected with `PISAD_SIG_CONF` >50%
- [ ] Signal classification appropriate for target type
- [ ] `PISAD_BEARING` showing consistent direction (±10° variation)
- [ ] `PISAD_BEAR_CONF` >70% for reliable bearing calculation
- [ ] No excessive interference (`PISAD_INTERFERENCE` <50%)

**Operational Environment:**
- [ ] Clear airspace with no conflicting traffic
- [ ] Within approved geofence boundaries
- [ ] Weather conditions suitable for autonomous flight
- [ ] Emergency landing areas identified and accessible
- [ ] Communication with ground control established

**Mission Planner Configuration:**
- [ ] PISAD parameters synchronized and displaying valid data
- [ ] Telemetry connection stable with <1% packet loss
- [ ] Safety override procedures reviewed and accessible
- [ ] Emergency disable command (MAV_CMD_USER_2) readily available

### Signal Quality Assessment

**Minimum Signal Requirements:**
- **RSSI:** >-90dBm for initial detection, >-80dBm preferred for homing
- **SNR:** >12dB for reliable signal processing
- **Confidence:** >50% required, >70% recommended for optimal performance
- **Bearing Stability:** ±15° variation over 30-second period

**Signal Quality Indicators:**
- `PISAD_RSSI_ZONE`: 2 (Yellow) minimum, 3 (Green) or 4 (Blue) preferred
- `PISAD_SIG_CLASS`: Should match expected beacon type
- `PISAD_HUD_QUAL`: >60% for reliable homing operations
- `PISAD_HUD_TREND`: Stable (0) or improving (+1) signal trend

---

## Homing Activation Procedures

### Method 1: MAV_CMD_USER_1 Command Execution

**Step 1: Command Access**
1. In Mission Planner, navigate to **Flight Data** tab
2. Click **Actions** panel in lower right area
3. Select **Custom Commands** from dropdown menu
4. Choose **MAV_CMD_USER_1** from command list

**Step 2: Homing Enable Command**
1. Set **Param1** = `1` (Enable homing)
2. Leave other parameters at default (0)
3. Click **Execute Command** button
4. Monitor status area for confirmation message

**Step 3: Activation Confirmation**
1. Verify `PISAD_HOMING_STATE` changes from `0` to `1` (Armed)
2. Status message appears: "PISAD Homing Armed - Signal Acquired"
3. `PISAD_HOMING_EN` parameter shows `1` (Enabled)
4. System begins signal direction analysis

**Step 4: Homing Engagement**
1. When bearing confidence sufficient, `PISAD_HOMING_STATE` advances to `2` (Active)
2. `PISAD_HOMING_SUBSTAGE` shows active algorithm (1=APPROACH initially)
3. Velocity commands begin sending to flight controller
4. Aircraft movement toward signal source commences

### Method 2: Parameter Direct Modification

**Alternative Activation:**
1. Navigate to **CONFIG/TUNING** → **Full Parameter Tree**
2. Locate `PISAD_HOMING_EN` parameter
3. Change value from `0` to `1`
4. System performs same safety checks and activation sequence

### Activation Safety Interlocks

**Automatic Safety Checks (System verifies before activation):**
- Flight mode must be GUIDED (all other modes rejected)
- Valid signal detection with adequate confidence
- GPS position lock verified
- Battery level above critical threshold
- No geofence violations detected
- System health parameters within normal ranges

**Activation Failure Reasons:**
- **Mode Error:** Flight controller not in GUIDED mode
- **Signal Error:** Insufficient signal confidence or bearing accuracy
- **Battery Error:** Battery level <20% remaining
- **GPS Error:** No GPS lock or poor HDOP
- **System Error:** PISAD system health issues detected

---

## Homing Status Monitoring

### Real-Time Status Parameters

**Primary Status Indicators:**
- `PISAD_HOMING_STATE`: Overall homing operation status
  - **0 = Disabled:** No homing operation active
  - **1 = Armed:** Ready for homing, analyzing signal direction
  - **2 = Active:** Actively homing, sending velocity commands
  - **3 = Lost:** Signal lost, executing search pattern

- `PISAD_HOMING_SUBSTAGE`: Active homing algorithm
  - **0 = INACTIVE:** No algorithm running
  - **1 = APPROACH:** Direct approach to signal source
  - **2 = SPIRAL_SEARCH:** Expanding spiral when signal lost
  - **3 = S_TURN:** S-turn pattern for signal reacquisition
  - **4 = RETURN_TO_PEAK:** Return to strongest signal location

### Monitoring Procedures

**Continuous Monitoring Requirements:**
1. **Status Display:** Keep Mission Planner parameter display visible
2. **Signal Strength:** Monitor `PISAD_SIG_CONF` for signal quality
3. **Bearing Information:** Watch `PISAD_BEARING` and `PISAD_BEAR_CONF`
4. **Flight Parameters:** Maintain awareness of altitude, battery, GPS status
5. **Communication:** Ensure continuous telemetry connection

**Progress Tracking:**
- Monitor `PISAD_RSSI_SMOOTH` for signal strength improvement
- Watch `PISAD_RSSI_GRAD` for positive gradient (approaching source)
- Observe `PISAD_HUD_TREND` for signal improvement indication
- Track flight path and estimated time to target

### Homing Algorithm Progression

**APPROACH Substage (Substage 1):**
- Direct flight toward calculated signal bearing
- Velocity scaled based on signal confidence and strength
- Continuous bearing updates as aircraft moves
- Transition to SPIRAL_SEARCH if signal lost

**SPIRAL_SEARCH Substage (Substage 2):**
- Expanding spiral pattern centered on last known signal location
- Spiral radius increases until signal reacquired
- Returns to APPROACH when signal confidence restored
- Maximum spiral radius limited by geofence boundaries

**S_TURN Substage (Substage 3):**
- Side-to-side search pattern for signal direction refinement
- Used when signal detected but bearing unclear
- Narrow S-turns to establish signal gradient
- Transitions to APPROACH when bearing confidence achieved

**RETURN_TO_PEAK Substage (Substage 4):**
- Returns to location of strongest previously detected signal
- Activated when extended search fails to find signal
- Serves as fallback to prevent aircraft from wandering
- Provides opportunity for signal reacquisition

### Performance Metrics Monitoring

**Real-Time Performance Indicators:**
- `PISAD_HOMING_SUCCESS_RATE`: Running average of successful completions
- `PISAD_HOMING_AVG_TIME`: Average time for homing completion
- `PISAD_HOMING_CONFIDENCE`: Current operation confidence level

**Operational Decision Points:**
- **High Confidence (>80%):** Continue homing operation
- **Medium Confidence (50-80%):** Monitor closely, prepare for intervention
- **Low Confidence (<50%):** Consider manual intervention or abort

---

## Operator Decision Points

### When to Intervene

**Signal Quality Degradation:**
- Signal confidence drops below 30% for >30 seconds
- Signal bearing becomes erratic (>45° variation)
- Interference level exceeds 75%
- Signal classification changes unexpectedly

**Flight Safety Concerns:**
- Aircraft approaching terrain or obstacles
- Battery level approaching critical (25%)
- GPS accuracy degrading (HDOP >3.0)
- Weather conditions deteriorating

**Mission Effectiveness Issues:**
- Homing operation exceeding reasonable time limits
- Aircraft circling without progress toward target
- Multiple false signals causing confusion
- Target area reached but no visual confirmation

### Intervention Procedures

**Gradual Intervention (Maintain homing with adjustments):**
1. Adjust flight altitude for better signal reception
2. Modify search pattern parameters if accessible
3. Change to different frequency profile if multiple beacons possible
4. Relocate search area based on operator assessment

**Immediate Intervention (Take manual control):**
1. Change flight mode from GUIDED to MANUAL or STABILIZE
2. Assume direct flight control via RC transmitter
3. Navigate manually toward suspected target location
4. Land aircraft for inspection or system diagnosis

---

## Homing Completion and Deactivation

### Automatic Completion

**Target Acquisition Indicators:**
- Signal strength reaches maximum level (plateau effect)
- `PISAD_HOMING_STATE` automatically transitions to `0` (Disabled)
- Aircraft enters position hold mode over target location
- Status message: "PISAD Homing Complete - Target Acquired"

**Automatic Completion Criteria:**
- RSSI reaches local maximum with stable reading
- Aircraft within estimated beacon proximity (typically 10-50m)
- Signal strength plateau maintained for configured duration
- Bearing calculation indicates target directly below aircraft

### Manual Deactivation

**Normal Manual Deactivation:**
1. Execute MAV_CMD_USER_1 with Param1 = `0` (Disable)
2. Or change `PISAD_HOMING_EN` parameter to `0`
3. Verify `PISAD_HOMING_STATE` returns to `0` (Disabled)
4. Aircraft maintains current position in GUIDED mode

**Flight Mode Change Deactivation:**
1. Use Mission Planner flight mode selector
2. Change from GUIDED to MANUAL, STABILIZE, or RTL
3. Homing automatically deactivates within 100ms
4. Resume manual flight control immediately

### Post-Homing Procedures

**Target Area Assessment:**
1. Maintain position hold over computed target location
2. Conduct visual inspection of area below aircraft
3. Take photographs or video for target documentation
4. Record GPS coordinates of target location

**System State Verification:**
1. Confirm all velocity commands ceased
2. Verify `PISAD_HOMING_STATE` = `0` (Disabled)
3. Check system health parameters for normal operation
4. Document homing performance metrics

**Mission Continuation or Completion:**
1. If target found: Execute recovery or documentation procedures
2. If target not found: Consider expanded search or alternate methods
3. Assess battery and weather for continued operations
4. Plan return to launch or alternate landing site

---

## Homing Abort and Emergency Procedures

### Emergency Homing Abort

**Immediate Abort (MAV_CMD_USER_2):**
1. Click **Actions** → **Custom Commands** → **MAV_CMD_USER_2**
2. Enter any parameter value (command is immediate)
3. Click **Execute Command**
4. **Response time: <100ms guaranteed**

**Emergency Abort Effects:**
- All RF processing immediately halted
- Velocity commands terminated instantly
- `PISAD_HOMING_STATE` forced to `0` (Disabled)
- Aircraft maintains last commanded position/velocity
- System requires restart to resume RF operations

### Recovery from Abort

**System Recovery Procedure:**
1. Assess aircraft state and position
2. Switch to manual flight mode for direct control
3. Navigate to safe area for system restart
4. Power cycle PISAD system if required
5. Re-establish parameter synchronization
6. Complete pre-activation checklist before resuming

**Incident Documentation:**
- Record time and circumstances of abort
- Document aircraft position and status at abort
- Note any system errors or unusual behavior
- File incident report for safety analysis

---

**Document Version:** 1.0  
**Last Updated:** 2025-08-20  
**Safety Compliance:** PRD-FR11 (operator override), PRD-FR14 (explicit activation), PRD-FR16 (500ms disable)  
**Performance Standards:** <100ms emergency response, <2s homing transition