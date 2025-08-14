# Epic 1: Foundation & Core Signal Processing

**Goal:** Establish the foundational infrastructure and core signal processing capabilities that will serve as the backbone of the RF-homing system. This epic delivers a working SDR-based signal detection system with real-time monitoring via web interface, proving the technical feasibility of RF beacon detection even before flight integration.

## Story 1.1: Project Setup & Development Environment

**As a** developer,  
**I want** a properly configured development environment with all dependencies,  
**so that** I can build, test, and deploy the RF-homing payload software consistently.

**Acceptance Criteria:**

1. Git repository initialized with .gitignore for Python projects and README.md with project overview
2. Python 3.10+ virtual environment configured with requirements.txt including: asyncio, numpy, scipy, SoapySDR, FastAPI/Flask, pytest
3. Basic project structure created: /src, /tests, /config, /docs, /web/static, /web/templates
4. Systemd service file template created for auto-start on boot
5. Configuration system implemented using YAML with default config file at /config/default.yaml
6. Basic logging setup with rotation configured to prevent SD card overflow
7. Development environment documented in /docs/setup.md with step-by-step instructions

## Story 1.2: SDR Hardware Interface Layer

**As a** system operator,  
**I want** the software to reliably interface with SDR hardware,  
**so that** I can capture RF signals regardless of using HackRF or USRP devices.

**Acceptance Criteria:**

1. SoapySDR wrapper class implemented with device enumeration and selection
2. SDR initialization with configurable sample rate (2 Msps default), center frequency (2.437 GHz default), and gain settings
3. Continuous IQ sample streaming implemented using async generator pattern
4. Graceful error handling for device disconnection/reconnection
5. Hardware abstraction validated with both HackRF One and USRP B205mini (if available)
6. SDR health monitoring with periodic status checks every 5 seconds
7. Manual test utility created for validating SDR connectivity and streaming

## Story 1.3: Signal Processing Pipeline

**As a** signal analyst,  
**I want** real-time RSSI computation from IQ samples,  
**so that** I can detect and measure RF beacon signal strength.

**Acceptance Criteria:**

1. FFT-based RSSI computation implemented processing 1024-sample blocks
2. EWMA filter implemented with configurable alpha parameter (default 0.3)
3. Noise floor estimation using 10th percentile method over 1-second window
4. Signal confidence scoring based on SNR threshold (>12 dB for detection)
5. Processing pipeline achieves <100ms latency per computation cycle
6. Signal detection events logged with timestamp, frequency, RSSI, and confidence
7. Unit tests validate RSSI calculation accuracy with known test signals

## Story 1.4: Web-Based Payload UI Foundation

**As a** system operator,  
**I want** a web interface to monitor signal detection in real-time,  
**so that** I can observe payload operation without additional software installation.

**Acceptance Criteria:**

1. FastAPI/Flask web server starts automatically on boot at port 8080
2. Main dashboard displays current RSSI value updated via WebSocket at 10Hz minimum
3. Time-series graph shows RSSI history for last 60 seconds
4. Current SDR configuration displayed (frequency, sample rate, gain)
5. Signal detection log shows last 10 detection events with timestamps
6. System health indicators show CPU usage, memory, SDR status
7. Interface responsive on desktop (1920x1080) and tablet (1024x768) resolutions

## Story 1.5: Configuration Management & Persistence

**As a** field operator,  
**I want** to save and load different configuration profiles,  
**so that** I can quickly adapt to different beacon types and environments.

**Acceptance Criteria:**

1. Configuration profiles stored as YAML files in /config/profiles/
2. Web UI provides "Save Profile" and "Load Profile" functionality
3. Configurable parameters include: frequency, sample rate, gain, thresholds, filter settings
4. Default profile loads automatically on system startup
5. Configuration changes via UI immediately apply to running SDR process
6. Profile switching logs events for troubleshooting
7. At least 3 preset profiles provided (WiFi beacon, LoRa beacon, custom)
