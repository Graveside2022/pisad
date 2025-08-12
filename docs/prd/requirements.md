# Requirements

## Functional

- **FR1:** The system shall autonomously detect RF beacons at minimum 500m range with >12 dB SNR threshold
- **FR2:** The drone shall execute expanding square search patterns at configurable velocities between 5-10 m/s
- **FR3:** The system shall be capable of transitioning to HOMING behavior within 2 seconds of beacon detection when homing mode is activated by the operator
- **FR4:** The drone shall navigate toward detected signals using RSSI gradient climbing with forward velocity and yaw-rate control
- **FR5:** The system shall support both GUIDED (GPS-assisted) and GUIDED_NOGPS (GPS-denied) autonomous flight modes
- **FR6:** The system shall compute real-time RSSI with EWMA filtering and noise floor estimation using 10th percentile
- **FR7:** The system shall implement debounced state transitions with configurable trigger (12dB) and drop (6dB) thresholds
- **FR8:** The drone shall maintain flight within designated geofence boundaries with automatic enforcement
- **FR9:** The system shall stream RSSI telemetry to ground control station via MAVLink NAMED_VALUE_FLOAT messages
- **FR10:** The system shall execute automatic return-to-launch (RTL) or LOITER on communication loss or low battery
- **FR11:** The operator shall maintain full override capability through primary GCS, with GCS mode changes immediately overriding any payload-initiated commands
- **FR12:** The system shall log all state transitions and signal detections for post-mission analysis
- **FR13:** The SDR sensing suite shall initialize automatically on system boot and continuously monitor for RF signals
- **FR14:** The operator shall explicitly activate homing mode through the payload UI, which enables the system to send velocity commands to the flight controller
- **FR15:** The system shall immediately cease sending velocity commands when the flight controller mode changes from GUIDED to any other mode
- **FR16:** The payload UI shall provide a prominent "Disable Homing" control that stops all velocity commands within 500ms
- **FR17:** The system shall automatically disable homing mode after 10 seconds of signal loss and notify the operator

## Non Functional

- **NFR1:** The system shall maintain MAVLink communication with <1% packet loss at 115200-921600 baud
- **NFR2:** Signal processing latency shall not exceed 100ms per RSSI computation cycle
- **NFR3:** The system shall achieve minimum 25 minutes flight endurance with full payload (Pi + SDR)
- **NFR4:** Power consumption shall not exceed 2.5A @ 5V continuous for companion computer and SDR
- **NFR5:** The system shall operate in temperatures from -10°C to +45°C
- **NFR6:** The system shall tolerate wind conditions up to 15 m/s sustained, 20 m/s gusts
- **NFR7:** False positive detection rate shall remain below 5% in controlled conditions
- **NFR8:** The system shall achieve 90% successful homing rate once signal is acquired in open field
- **NFR9:** Mean time between failures (MTBF) shall exceed 10 flight hours
- **NFR10:** System deployment by single operator shall be achievable in under 15 minutes
- **NFR11:** The codebase shall follow modular architecture with clear interfaces between SDR, signal processing, navigation, and flight control
- **NFR12:** All safety-critical functions shall execute with deterministic timing using AsyncIO architecture
- **NFR13:** The system shall visually indicate homing state in both payload UI and GCS telemetry messages
