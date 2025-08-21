# PISAD RF Field Reference - Quick Reference Cards

## Overview

This document provides quick reference cards designed for field use during PISAD RF operations. These cards are formatted for printing and lamination for use in cockpit environments where quick access to critical information is essential.

**Usage:** Print individual cards on durable material, laminate for field use, organize in flight bag or cockpit reference binder.

---

## **CARD 1: EMERGENCY PROCEDURES**

```
╔══════════════════════════════════════════════════════════════════╗
║                    PISAD RF EMERGENCY PROCEDURES                ║
║                        ** CRITICAL SAFETY **                   ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  🚨 IMMEDIATE RF DISABLE (MAV_CMD_USER_2)                       ║
║     1. Actions → Custom Commands → MAV_CMD_USER_2               ║
║     2. Enter ANY parameter value                                ║
║     3. Execute Command                                          ║
║     ⚡ RESPONSE: <100ms GUARANTEED                               ║
║                                                                  ║
║  🚨 FLIGHT MODE OVERRIDE                                        ║
║     1. Flight Mode Selector → MANUAL/STABILIZE                 ║
║     2. Resume RC control immediately                            ║
║     ⚡ RESPONSE: <100ms automatic disable                        ║
║                                                                  ║
║  ⚠️  WHEN TO USE EMERGENCY PROCEDURES:                          ║
║     • Aircraft safety threat                                    ║
║     • System malfunction/erratic commands                      ║
║     • Communication loss uncertainty                            ║
║     • Immediate manual control needed                           ║
║     • Airspace conflict requiring clearance                     ║
║                                                                  ║
║  📞 EMERGENCY CONTACTS:                                          ║
║     Mission Control: [RADIO FREQ]                              ║
║     SAR Command: [PHONE/RADIO]                                  ║
║     ATC Emergency: 121.5 MHz                                   ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## **CARD 2: HOMING ACTIVATION CHECKLIST**

```
╔══════════════════════════════════════════════════════════════════╗
║                    HOMING ACTIVATION CHECKLIST                  ║
║                      ** SAFETY CRITICAL **                     ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  ✅ PRE-ACTIVATION REQUIREMENTS (ALL MUST BE MET):              ║
║     □ Flight mode: GUIDED (required)                           ║
║     □ Signal confidence: PISAD_SIG_CONF ≥50%                   ║
║     □ Bearing confidence: PISAD_BEAR_CONF ≥70%                 ║
║     □ Battery level: >25% remaining                            ║
║     □ GPS lock: HDOP <3.0                                      ║
║     □ Weather: Within operational limits                        ║
║     □ Geofence: Within boundaries                              ║
║                                                                  ║
║  🎯 ACTIVATION PROCEDURE:                                        ║
║     1. Actions → Custom Commands → MAV_CMD_USER_1              ║
║     2. Param1 = 1 (Enable homing)                             ║
║     3. Execute Command                                         ║
║     4. Verify PISAD_HOMING_STATE = 1 (Armed)                   ║
║     5. Monitor for automatic progression to 2 (Active)         ║
║                                                                  ║
║  👁️  CONTINUOUS MONITORING:                                     ║
║     • PISAD_HOMING_SUBSTAGE (algorithm status)                ║
║     • PISAD_BEARING (target direction)                        ║
║     • PISAD_RSSI_SMOOTH (signal strength)                     ║
║     • Aircraft position and safety                            ║
║                                                                  ║
║  🛑 IMMEDIATE ABORT IF:                                         ║
║     • Signal confidence drops <30%                            ║
║     • Aircraft approaching obstacles                           ║
║     • Battery <25% remaining                                   ║
║     • Weather deteriorating                                    ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## **CARD 3: PARAMETER QUICK REFERENCE**

```
╔══════════════════════════════════════════════════════════════════╗
║                     PISAD PARAMETER REFERENCE                   ║
║                     ** MISSION CRITICAL **                     ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  📡 RF CONFIGURATION:                                           ║
║     PISAD_RF_FREQ: Target frequency (Hz)                       ║
║     PISAD_RF_PROFILE: 0=Emerg, 1=Avn, 2=Marine, 3=Training    ║
║     PISAD_RF_HEALTH: System health % (>70% required)           ║
║                                                                  ║
║  📊 SIGNAL STATUS:                                              ║
║     PISAD_SIG_CONF: Signal confidence % (≥50% for homing)      ║
║     PISAD_RSSI: Signal strength (dBm)                          ║
║     PISAD_BEARING: Signal direction (degrees)                  ║
║     PISAD_BEAR_CONF: Bearing confidence % (≥70% for homing)    ║
║                                                                  ║
║  🎯 HOMING STATUS:                                              ║
║     PISAD_HOMING_EN: 0=Disabled, 1=Enabled                     ║
║     PISAD_HOMING_STATE: 0=Off, 1=Armed, 2=Active, 3=Lost      ║
║     PISAD_HOMING_SUBSTAGE: 0=None, 1=Approach, 2=Spiral,      ║
║                           3=S-Turn, 4=Return                   ║
║                                                                  ║
║  📈 PERFORMANCE INDICATORS:                                     ║
║     PISAD_RSSI_ZONE: 1=Red, 2=Yellow, 3=Green, 4=Blue         ║
║     PISAD_HUD_QUAL: Signal quality % for HUD display           ║
║     PISAD_HUD_TREND: -1=Weak, 0=Stable, +1=Strong             ║
║                                                                  ║
║  ⚙️  EMERGENCY FREQUENCIES:                                     ║
║     406.000 MHz: International emergency beacon                ║
║     121.500 MHz: Aviation emergency                            ║
║     162.025 MHz: Marine emergency (EPIRB)                      ║
║     243.000 MHz: Military emergency (restricted)               ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## **CARD 4: MISSION PLANNER NAVIGATION**

```
╔══════════════════════════════════════════════════════════════════╗
║                  MISSION PLANNER PISAD NAVIGATION               ║
║                        ** QUICK ACCESS **                      ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  📋 PARAMETER ACCESS:                                           ║
║     CONFIG/TUNING → Full Parameter Tree                        ║
║     Search: "PISAD" to filter parameters                       ║
║     Right-click parameter → Modify value                       ║
║                                                                  ║
║  🎮 CUSTOM COMMANDS:                                            ║
║     Flight Data → Actions Panel → Custom Commands              ║
║     MAV_CMD_USER_1: Homing Enable/Disable                      ║
║       - Param1=1: Enable homing                               ║
║       - Param1=0: Disable homing                              ║
║     MAV_CMD_USER_2: Emergency RF Disable                       ║
║       - Any parameter: Immediate disable                       ║
║                                                                  ║
║  📊 TELEMETRY DISPLAY:                                          ║
║     Flight Data → Status Panel                                 ║
║     Custom screens can display PISAD parameters                ║
║     Quick view: Double-click parameter in tree                 ║
║                                                                  ║
║  🔧 FLIGHT MODE CONTROL:                                        ║
║     Flight Mode dropdown (upper toolbar)                       ║
║     GUIDED: Required for PISAD homing                         ║
║     MANUAL/STABILIZE: Immediate override                       ║
║                                                                  ║
║  📱 STATUS MESSAGES:                                            ║
║     Messages panel shows PISAD status updates                  ║
║     Filter by "PISAD" for RF-specific messages                ║
║                                                                  ║
║  💾 DATA LOGGING:                                               ║
║     Automatic logging of all parameters                        ║
║     .tlog files contain complete mission data                  ║
║     Use MAVLink Inspector for real-time data                   ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## **CARD 5: TROUBLESHOOTING QUICK FIXES**

```
╔══════════════════════════════════════════════════════════════════╗
║                  TROUBLESHOOTING QUICK FIXES                    ║
║                      ** IMMEDIATE FIXES **                     ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  🔧 PARAMETERS NOT VISIBLE:                                     ║
║     1. Full Parameter Tree → Refresh Params                    ║
║     2. Wait 60 seconds for complete sync                       ║
║     3. If failed: Disconnect/Reconnect Mission Planner         ║
║                                                                  ║
║  🔧 PARAMETERS GRAYED OUT:                                      ║
║     1. Check aircraft armed/disarmed state                     ║
║     2. Verify no other GCS connected                           ║
║     3. Right-click → Write to force parameter                  ║
║                                                                  ║
║  🔧 HOMING WON'T ACTIVATE:                                      ║
║     1. Verify flight mode = GUIDED                             ║
║     2. Check signal confidence ≥50%                            ║
║     3. Verify bearing confidence ≥70%                          ║
║     4. Check battery >25%                                      ║
║                                                                  ║
║  🔧 SIGNAL DETECTION ISSUES:                                    ║
║     1. Check antenna connections                               ║
║     2. Verify frequency setting correct                        ║
║     3. Check for interference sources                          ║
║     4. Try different altitude                                  ║
║                                                                  ║
║  🔧 POOR SIGNAL QUALITY:                                        ║
║     1. Increase altitude if safe                               ║
║     2. Change location to avoid terrain blocking               ║
║     3. Check for electromagnetic interference                   ║
║     4. Verify antenna not damaged                              ║
║                                                                  ║
║  🔧 SYSTEM RESTART PROCEDURE:                                   ║
║     1. Land aircraft safely                                    ║
║     2. SSH to Pi: sudo systemctl restart pisad                ║
║     3. Reconnect Mission Planner                               ║
║     4. Verify parameter sync before flight                     ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## **CARD 6: FREQUENCY REFERENCE**

```
╔══════════════════════════════════════════════════════════════════╗
║                      FREQUENCY REFERENCE                        ║
║                    ** REGULATORY CRITICAL **                   ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  🌍 INTERNATIONAL EMERGENCY FREQUENCIES:                        ║
║     406.000-406.100 MHz: COSPAS-SARSAT (Primary)              ║
║     121.500 MHz: Aeronautical Emergency (Secondary)            ║
║     243.000 MHz: Military Emergency (Authorization Required)    ║
║                                                                  ║
║  🚢 MARINE EMERGENCY FREQUENCIES:                               ║
║     162.025 MHz: EPIRB Primary                                 ║
║     156.525 MHz: Marine Emergency (Channel 70)                 ║
║     156.800 MHz: Marine VHF Channel 16                        ║
║                                                                  ║
║  ✈️  AVIATION EMERGENCY FREQUENCIES:                            ║
║     121.500 MHz: Civil Aviation Emergency                      ║
║     243.000 MHz: Military Aviation Emergency                   ║
║     282.800 MHz: SARSAT Downlink                              ║
║                                                                  ║
║  🎓 TRAINING FREQUENCIES (ISM BANDS):                           ║
║     433.920 MHz: ISM Band (Training use only)                 ║
║     915.000 MHz: ISM Band (US training)                       ║
║     868.000 MHz: ISM Band (EU training)                       ║
║                                                                  ║
║  ⚠️  FREQUENCY COORDINATION:                                    ║
║     • Verify legal operation in current location               ║
║     • Coordinate with emergency services                       ║
║     • Check NOTAMs for frequency restrictions                  ║
║     • Emergency frequencies have absolute priority             ║
║                                                                  ║
║  📝 FORMAT CONVERSION:                                          ║
║     406 MHz = 406,000,000 Hz (Parameter entry)                ║
║     121.5 MHz = 121,500,000 Hz                                ║
║     433.92 MHz = 433,920,000 Hz                               ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## **CARD 7: SAFETY DECISION MATRIX**

```
╔══════════════════════════════════════════════════════════════════╗
║                     SAFETY DECISION MATRIX                      ║
║                       ** USE FOR ALL DECISIONS **              ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  🟢 CONTINUE OPERATION:                                         ║
║     ✓ Signal confidence >50%                                   ║
║     ✓ Battery >30%                                             ║
║     ✓ Weather within limits                                    ║
║     ✓ Aircraft systems normal                                  ║
║     ✓ Clear airspace                                           ║
║                                                                  ║
║  🟡 PROCEED WITH CAUTION:                                       ║
║     ⚠️ Signal confidence 30-50%                                ║
║     ⚠️ Battery 25-30%                                          ║
║     ⚠️ Weather at limits                                       ║
║     ⚠️ Minor system degradation                                ║
║     ⚠️ Increased monitoring required                           ║
║                                                                  ║
║  🔴 ABORT OPERATION:                                           ║
║     ❌ Signal confidence <30%                                  ║
║     ❌ Battery <25%                                            ║
║     ❌ Weather exceeding limits                                ║
║     ❌ System malfunction                                      ║
║     ❌ Safety threat identified                                ║
║                                                                  ║
║  🚨 EMERGENCY RESPONSE:                                         ║
║     🆘 Immediate safety threat                                 ║
║     🆘 System providing dangerous commands                     ║
║     🆘 Loss of aircraft control                                ║
║     🆘 Medical emergency                                       ║
║     🆘 Execute MAV_CMD_USER_2 immediately                      ║
║                                                                  ║
║  👤 OPERATOR AUTHORITY:                                         ║
║     You have COMPLETE override authority                        ║
║     Safety decisions supersede mission goals                    ║
║     When in doubt, choose conservative action                   ║
║     Document all safety decisions                              ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## **PRINTING AND USAGE INSTRUCTIONS**

### **Card Printing Specifications**

**Print Settings:**
- **Paper Size:** 8.5" x 11" (standard letter)
- **Orientation:** Portrait
- **Print Quality:** High quality/Best
- **Color:** Black and white acceptable
- **Font Size:** Keep minimum 10pt for cockpit readability

**Durability Preparation:**
- **Lamination:** 5-mil lamination minimum for field use
- **Card Stock:** Heavy cardstock (110lb minimum) if not laminating
- **Corner Rounding:** Optional corner rounding for durability
- **Hole Punching:** 3-hole punch for binder organization

### **Cockpit Organization**

**Primary Location:**
- Cockpit reference binder or flight bag
- Quick access pocket for emergency cards
- Kneeboard pocket for immediate reference
- Mission Planner workspace area

**Emergency Access:**
- Card 1 (Emergency Procedures) in immediate reach
- Card 7 (Safety Decision Matrix) readily visible
- Consider taping emergency card to instrument panel
- Ensure readable under cockpit lighting conditions

### **Field Usage Guidelines**

**Pre-Mission:**
- Review all cards during crew briefing
- Verify emergency contact information current
- Check frequency assignments for mission area
- Confirm parameter values for mission type

**During Mission:**
- Keep emergency procedures card immediately accessible
- Use parameter card for real-time reference
- Refer to troubleshooting card for immediate issues
- Apply safety decision matrix for all major decisions

**Post-Mission:**
- Note any cards that need updates
- Report errors or improvements needed
- Update contact information as required
- Document usage effectiveness for improvement

---

**Document Version:** 1.0  
**Last Updated:** 2025-08-21  
**Print Verification:** Test print recommended before field deployment  
**Update Schedule:** Monthly review, immediate update for safety-critical changes  
**Distribution:** All PISAD operators, flight crews, training facilities