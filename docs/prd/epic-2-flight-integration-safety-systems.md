# Epic 2: Flight Integration & Safety Systems

**Goal:** Integrate the signal processing payload with the flight control system via MAVLink, implementing comprehensive safety protocols and operator-controlled homing activation. This epic transforms the standalone signal detector into a flight-ready payload with proper safeguards and clear operational boundaries.

## Story 2.1: MAVLink Communication Foundation

**As a** drone operator,  
**I want** reliable bi-directional communication between payload and flight controller,  
**so that** the payload can receive telemetry and send velocity commands when authorized.

**Acceptance Criteria:**
1. MAVLink 2.0 connection established over serial (/dev/ttyACM0 or /dev/ttyAMA0)
2. Heartbeat messages exchanged at 1Hz with connection monitoring
3. Flight telemetry received and parsed (position, altitude, battery, mode, GPS status)
4. Velocity command sending implemented (SET_POSITION_TARGET_LOCAL_NED) but disabled by default
5. Connection status displayed in web UI with automatic reconnection on failure
6. MAVLink message logging for debugging with configurable verbosity
7. SITL testing environment configured for development without hardware

## Story 2.2: Safety Interlock System

**As a** safety officer,  
**I want** multiple independent safety mechanisms,  
**so that** the payload never compromises flight safety or operator control.

**Acceptance Criteria:**
1. Mode monitor detects flight mode changes within 100ms and stops velocity commands if not GUIDED
2. Operator activation required via web UI "Enable Homing" button before any velocity commands sent
3. Automatic homing disable after 10 seconds of signal loss (<6 dB SNR)
4. Geofence boundary check before sending any movement commands
5. Battery monitor disables homing if battery <20% remaining
6. Emergency stop accessible via web UI that immediately ceases all commands
7. All safety events logged with timestamp and trigger reason

## Story 2.3: Operator Control Interface

**As a** drone operator,  
**I want** clear and intuitive controls for activating and monitoring homing behavior,  
**so that** I maintain situational awareness and positive control.

**Acceptance Criteria:**
1. Large, prominent "Enable/Disable Homing" toggle button with clear state indication
2. Visual confirmation required before enabling homing (popup or slide-to-confirm)
3. Current homing state displayed prominently with color coding (gray=disabled, green=enabled, red=active)
4. Override instructions clearly displayed: "To regain control: Switch flight mode in Mission Planner"
5. Auto-disable conditions listed visibly (signal loss, mode change, low battery)
6. Homing activation/deactivation sends STATUSTEXT to GCS for logging
7. UI prevents homing activation unless drone is in GUIDED mode and armed

## Story 2.4: Telemetry Integration & Reporting

**As a** mission commander,  
**I want** payload status visible in my primary GCS,  
**so that** I have unified situational awareness without switching interfaces.

**Acceptance Criteria:**
1. RSSI value streamed to GCS via NAMED_VALUE_FLOAT at 2Hz
2. Homing state sent via STATUSTEXT on any state change
3. Detection events sent as STATUSTEXT with signal strength and confidence
4. Custom MAVLink messages documented for future GCS plugin development
5. Telemetry rate configurable to prevent bandwidth saturation
6. QGroundControl and Mission Planner both display payload telemetry correctly
7. Payload health status (CPU, memory, SDR) sent every 10 seconds

## Story 2.5: Ground Testing & Safety Validation

**As a** test engineer,  
**I want** comprehensive ground testing procedures,  
**so that** I can validate all safety systems before flight testing.

**Acceptance Criteria:**
1. Bench test procedure validates all safety interlocks with simulated signals
2. Ground vehicle test plan documents RF validation methodology
3. Safety checklist created for pre-flight payload verification
4. Emergency procedures documented for all failure modes
5. SITL test suite runs through all safety scenarios automatically
6. Hardware-in-loop test validates timing of safety responses <500ms
7. Test results logged and archived for safety audit trail
