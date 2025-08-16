# RF-Homing SAR Drone Fullstack Architecture Document

## Introduction

This document outlines the complete fullstack architecture for RF-Homing SAR Drone, including backend systems, frontend implementation, and their integration. It serves as the single source of truth for AI-driven development, ensuring consistency across the entire technology stack.

This unified approach combines what would traditionally be separate backend and frontend architecture documents, streamlining the development process for modern fullstack applications where these concerns are increasingly intertwined.

### Starter Template or Existing Project

N/A - Greenfield project

### Change Log

| Date       | Version | Description                            | Author              |
| ---------- | ------- | -------------------------------------- | ------------------- |
| 2025-08-12 | 1.0     | Initial architecture document creation | Winston (Architect) |
| 2025-01-15 | 1.1     | Added backend implementation status verification | Sentinel |
| 2025-08-16 | 2.0     | Major optimization update after Story 4.9 completion | Winston & Sentinel |

## Backend Implementation Status - OPTIMIZED (v2.0)

### Post-Optimization Status (2025-08-16)

**Backend Framework: 85% COMPLETE - Optimized & Refactored**
- **Lines of Code**: 22,097 lines of Python backend code (65 files)
- **Test Coverage**: 67% overall, 100% for safety-critical components
- **Test Suite**: 1,388 tests (94% passing after optimization)
- **Complexity Reduction**: 86% (from 142 to <20 cyclomatic complexity per component)
- **Performance Gains**: 97.7% CPU reduction in signal processing, 60% faster database queries

**Implemented Components:**

#### Core Infrastructure ✅
- `src/backend/core/app.py` - FastAPI application with CORS, WebSocket, Prometheus
- `src/backend/core/config.py` - YAML-based configuration system
- `src/backend/core/dependencies.py` - Dependency injection

#### API Routes (All Implemented) ✅
- `/api/analytics/*` - Performance analytics endpoints
- `/api/config/*` - Configuration management
- `/api/detections/*` - Detection logging
- `/api/health/*` - Health check endpoints
- `/api/search/*` - Search pattern control
- `/api/state/*` - State management
- `/api/system/*` - System control
- `/api/telemetry/*` - Telemetry streaming
- `/api/testing/*` - Test result endpoints
- `/ws` - WebSocket for real-time updates

#### Services (21 Services Implemented) ✅
- `beacon_simulator.py` - Beacon signal simulation
- `command_pipeline.py` - Command processing pipeline
- `config_service.py` - Configuration management
- `field_test_service.py` - Field test orchestration
- `homing_algorithm.py` - Gradient climbing implementation
- `homing_controller.py` - Homing state control
- `mavlink_service.py` - MAVLink communication
- `mission_replay_service.py` - Mission playback
- `performance_analytics.py` - Performance analysis
- `recommendations_engine.py` - AI recommendations
- `report_generator.py` - PDF/JSON/CSV reports
- `sdr_service.py` - SDR hardware interface
- `search_pattern_generator.py` - Search patterns
- `signal_processor.py` - RSSI/SNR processing
- `signal_processor_integration.py` - Signal integration
- `signal_state_controller.py` - Signal state management
- `state_integration.py` - State machine integration
- `state_machine.py` - Flight state management
- `telemetry_recorder.py` - Telemetry recording
- `waypoint_exporter.py` - Waypoint export

#### Database Models ✅
- `models/database.py` - SQLAlchemy models
- `models/schemas.py` - Pydantic schemas

#### Utilities ✅
- `utils/logging.py` - Logging configuration
- `utils/safety.py` - Safety interlock system
- `utils/test_logger.py` - Test logging utilities

### What's Missing (15% Remaining)

1. **Hardware Integration**
   - SDR hardware mocks need improvement
   - MAVLink SITL integration incomplete
   - GPIO pin mappings not configured

2. **Deployment Configuration**
   - `deployment/pisad.service` needs creation
   - Hardware configuration file missing

3. **Test Fixes Needed**
   - 60 failing tests (mostly hardware mocks)
   - 9 import errors
   - Some API status code mismatches

### Architecture Alignment

The implementation follows the planned architecture precisely:
- **Modular Monolith**: Single Python process with distinct service modules ✅
- **Event-Driven AsyncIO**: All services use async/await patterns ✅
- **WebSocket Real-time**: Binary protocol for 10Hz updates ✅
- **Repository Pattern**: Abstract data access implemented ✅
- **State Machine Pattern**: Explicit state transitions with safety ✅
- **Component-Based UI**: React components with TypeScript ✅

### Performance Metrics (Post-Optimization v2.0)

- **API Response Time**: <50ms for all endpoints (improved from <100ms)
- **WebSocket Latency**: <5ms for real-time updates (improved from <10ms)
- **Memory Usage**: ~150MB for full application (reduced from ~200MB)
- **CPU Usage**: <5% on Raspberry Pi 5 (dramatically reduced from <20%)
- **Startup Time**: <3 seconds to full operational (improved from <5 seconds)
- **Signal Processing**: 0.5ms per update (97.7% improvement)
- **Memory Stability**: <10MB/hour growth (99.7% leak prevention)

## High Level Architecture

### Technical Summary

The RF-Homing SAR Drone system implements a modular monolith architecture deployed on Raspberry Pi 5, combining a Python AsyncIO backend for real-time SDR signal processing and MAVLink communication with a React/TypeScript frontend for operator control. The backend runs as a systemd service providing continuous RF sensing and safety-critical flight control interfaces, while the frontend delivers real-time signal visualization and homing activation controls via WebSocket. All components run locally on the Pi 5 without external dependencies, ensuring operation in disconnected field environments while achieving the PRD goal of 70% search time reduction through autonomous RF beacon detection and gradient-based homing.

### Platform and Infrastructure Choice

**Platform:** Self-hosted Raspberry Pi 5 (Edge Computing)
**Key Services:** Local Python services (FastAPI, AsyncIO, SoapySDR), Local web server, SQLite for persistence
**Deployment Host and Regions:** Single Raspberry Pi 5 unit mounted on drone, no cloud regions

### Repository Structure

**Structure:** Monorepo
**Monorepo Tool:** Python workspace with shared packages
**Package Organization:** Unified Python backend with embedded static frontend, shared TypeScript types

### High Level Architecture Diagram

```mermaid
graph TB
    subgraph "Operator Interface"
        Browser[Web Browser<br/>Desktop/Tablet]
    end

    subgraph "Raspberry Pi 5"
        subgraph "Frontend"
            React[React SPA<br/>MUI Components]
            WS_Client[WebSocket Client]
        end

        subgraph "Backend Services"
            FastAPI[FastAPI Server<br/>Port 8080]
            WS_Server[WebSocket Server]
            SDR_Service[SDR Service<br/>SoapySDR]
            Signal_Proc[Signal Processing<br/>NumPy/SciPy]
            MAVLink[MAVLink Manager]
            State_Machine[State Machine<br/>AsyncIO]
        end

        subgraph "Data Layer"
            SQLite[(SQLite DB<br/>Logs/Profiles)]
            Config[YAML Configs]
        end
    end

    subgraph "Hardware Interfaces"
        SDR_HW[HackRF/USRP B205mini<br/>3.2 GHz Wideband SDR]
        FC[Flight Controller<br/>Serial MAVLink]
    end

    subgraph "External Systems"
        GCS[Mission Planner<br/>QGroundControl]
    end

    Browser <-->|HTTP/WS| React
    React <--> WS_Client
    WS_Client <-->|Binary Protocol| WS_Server
    WS_Server <--> FastAPI
    FastAPI <--> Signal_Proc
    FastAPI <--> State_Machine
    Signal_Proc <--> SDR_Service
    SDR_Service <--> SDR_HW
    State_Machine <--> MAVLink
    MAVLink <--> FC
    FC <--> GCS
    FastAPI <--> SQLite
    FastAPI <--> Config
```

### Architectural Patterns (v2.0 - Optimized)

- **Decomposed State Machine:** 6 focused classes replacing god object - _Rationale:_ Reduced complexity from 142 to <20 per component
- **Circuit Breaker Pattern:** Automatic failure recovery with exponential backoff - _Rationale:_ Prevents cascade failures, 30s recovery window
- **Sliding Window Algorithms:** O(1) percentile calculations for signal processing - _Rationale:_ 97.7% CPU reduction, real-time performance
- **Async I/O Patterns:** AsyncFileIO and AsyncDatabasePool - _Rationale:_ Event loop never blocked >10ms
- **Bounded Queue Pattern:** Memory-safe queue management - _Rationale:_ 99.7% memory leak prevention
- **YAML Inheritance:** Configuration inheritance with circular detection - _Rationale:_ 53% config duplication reduction
- **Property-Based Testing:** Hypothesis framework for invariant validation - _Rationale:_ 100% safety invariant coverage
- **Contract Testing:** Schemathesis for API contract validation - _Rationale:_ Automated API compliance checking

**Original Patterns (Retained):**
- **Modular Monolith:** Single Python process with distinct service modules
- **Event-Driven AsyncIO:** Non-blocking async operations throughout
- **WebSocket Real-time Updates:** Binary protocol for RSSI streaming
- **Repository Pattern:** Abstract data access for configs and logs
- **Component-Based UI:** Reusable React components with TypeScript
- **Operator-Activated Control:** Explicit activation required for autonomous behaviors

## Tech Stack

## Hardware Specifications

### SDR Hardware Requirements

**Beacon Signal Characteristics:**
- **Center Frequency**: 3.2 GHz (software configurable 850 MHz - 6.5 GHz)
- **Modulation Type**: FM with pulse generation
- **Signal Pattern**: Radar-like pulses (similar to weather radar)
- **Bandwidth**: 2-5 MHz wideband signal
- **Beacon Source**: Software-generated for testing, manual frequency selection for operations
- **Antenna**: Log-periodic directional antenna (850 MHz - 6500 MHz)

**SDR Hardware Selection:**
| SDR Model | Frequency Range | Sample Rate | Price | Status |
|-----------|----------------|-------------|-------|-------------|
| RTL-SDR v3 | 24 MHz - 1.7 GHz | 2.4 Msps | $40 | ❌ Cannot reach 3.2 GHz |
| HackRF One | 1 MHz - 6 GHz | 20 Msps | $350 | ✅ PRIMARY - Connected (USB Bus 003) |
| USRP B205mini | 70 MHz - 6 GHz | 30.72 Msps | $1,365 | Alternative if needed |

**Minimum Requirements:**
- Frequency coverage: 3.0 - 3.5 GHz minimum
- Sample rate: ≥10 Msps (Nyquist for 5 MHz bandwidth)
- USB 3.0 interface (mandatory for wideband data rates)
- Gain control: 0-40 dB minimum

### Signal Processing Implications

**Pulse Detection Requirements:**
- Fast rise time detection (<1 μs)
- Pulse width measurement capability
- PRF calculation from pulse timing
- FM demodulation for pulse envelope
- Matched filter for pulse detection

**Data Rate Calculations:**
- 5 MHz bandwidth × 2 (I+Q) × 2 bytes/sample = 20 MB/s minimum
- 20 Msps × 2 (I+Q) × 2 bytes = 80 MB/s for HackRF
- USB 3.0 required (USB 2.0 max ~40 MB/s insufficient)

### MAVLink/Flight Controller Interface

**Hardware Configuration:**
- **Flight Controller**: Pixhawk 4 with Cube Orange+ (Connected)
- **Connection Type**: Serial over USB (auto-discovery)
- **Device Path**: `/dev/ttyACM0` (CubeOrange+ via USB)
- **Baud Rate**: 115200 bps
- **Flow Control**: None (3-wire: TX, RX, GND)
- **MAVLink Version**: v1.0 and v2.0 (auto-detect)
- **Python Library**: pymavlink v2.4.49 (installed via uv)

**MAVLink Message Set (Required):**
| Message | Direction | Rate | Purpose |
|---------|-----------|------|---------|
| HEARTBEAT | Bidirectional | 1 Hz | Connection monitoring |
| GLOBAL_POSITION_INT | From FC | 4 Hz | GPS position |
| ATTITUDE | From FC | 10 Hz | Orientation data |
| VFR_HUD | From FC | 4 Hz | Speed and altitude |
| COMMAND_LONG | To FC | As needed | Mode changes |
| SET_POSITION_TARGET_LOCAL_NED | To FC | 10 Hz | Velocity commands |
| STATUSTEXT | From FC | As needed | Status messages |
| RC_CHANNELS | From FC | 2 Hz | RC override detection |

**Auto-Discovery Implementation:**
```python
# Scan for Pixhawk on common USB serial ports
# CubeOrange+ detected at /dev/ttyACM0 (primary and secondary)
SERIAL_PORTS = [
    "/dev/ttyACM0",  # CubeOrange+ primary (confirmed)
    "/dev/ttyACM1",  # CubeOrange+ secondary (confirmed)
    "/dev/ttyUSB0", "/dev/ttyUSB1",  # Fallback options
    "/dev/serial/by-id/*Cube*",
    "/dev/serial/by-id/*Pixhawk*"
]
```

### System Timing Requirements

**Safety-Critical Timing Constraints:**
| Operation | Maximum Latency | Purpose |
|-----------|----------------|---------|
| Mode Change Detection | <100 ms | Detect GUIDED mode transitions |
| Emergency Stop Response | <500 ms | Halt all autonomous movement |
| MAVLink Heartbeat Timeout | 10 seconds | Connection loss detection |
| Command Acknowledgment | 1000 ms | Command confirmation timeout |

**Signal Processing Rates:**
| Process | Rate | Period | Rationale |
|---------|------|--------|-----------|
| RSSI Sampling | 1 Hz | 1000 ms | Power efficiency, sufficient for homing |
| Gradient Calculation | 1 Hz | 1000 ms | Matches RSSI sampling rate |
| State Machine Update | 10 Hz | 100 ms | Responsive to mode changes |
| MAVLink Telemetry | 4-10 Hz | 100-250 ms | Per message type requirements |
| WebSocket Updates | 1 Hz | 1000 ms | UI responsiveness |

**Beacon Pulse Parameters:**
- **PRF**: TBD (to be determined during beacon development)
- **Pulse Width**: TBD
- **Detection Window**: Configurable based on beacon specs

**Processing Efficiency Notes:**
- 1 Hz RSSI processing significantly reduces CPU load on Pi 5
- Allows more headroom for pulse detection algorithms
- Buffer 1 second of IQ samples for batch processing
- Use moving average for gradient smoothing

### Hardware Abstraction Layer (HAL)

**SDR Interface (HackRF One):**
```python
# Using pyhackrf library (installed via uv)
class HackRFInterface:
    def open(self) -> bool:
        """Open HackRF device via USB"""

    def set_freq(self, freq: float) -> bool:
        """Set center frequency in Hz (3.2e9 for 3.2 GHz)"""

    def set_sample_rate(self, rate: float) -> bool:
        """Set sample rate in Hz (20e6 for 20 Msps)"""

    def set_lna_gain(self, gain: int) -> bool:
        """Set LNA gain 0-40 dB in 8dB steps"""

    def set_vga_gain(self, gain: int) -> bool:
        """Set VGA gain 0-62 dB in 2dB steps"""

    def start_rx(self) -> bool:
        """Start receiving IQ samples"""

    def read_samples(self, num_samples: int) -> np.ndarray:
        """Returns complex64 numpy array (complex float32)"""
        # Format: np.complex64 with I + jQ
        # Sample rate: 20 Msps
        # Bandwidth: 5 MHz (filtered in software)
```

**MAVLink Interface (Cube Orange+ via pymavlink):**
```python
# Using pymavlink library
class MAVLinkInterface:
    def connect(self, device: str, baud: int = 115200):
        """Auto-discover and connect to Cube Orange"""

    def send_velocity_ned(self, vn: float, ve: float, vd: float):
        """Send velocity in NED frame (North-East-Down)
        Units: meters per second (m/s)
        Frame: MAV_FRAME_LOCAL_NED (body frame)"""

    def get_position(self) -> Tuple[float, float, float]:
        """Get GPS position from GLOBAL_POSITION_INT
        Returns: (lat, lon, alt) in degrees and meters"""

    def get_battery(self) -> float:
        """Get battery voltage from SYS_STATUS
        Returns: Voltage in volts (6S Li-ion: 18-25.2V)"""
```

**GPS Interface (Here4 via UAVCAN):**
```python
# Here4 data accessed through MAVLink messages
# UAVCAN handles low-level communication
# We read via MAVLink GLOBAL_POSITION_INT and GPS_RAW_INT
class GPSInterface:
    def get_fix_type(self) -> int:
        """0=No GPS, 1=No Fix, 2=2D, 3=3D, 4=DGPS, 5=RTK Float, 6=RTK Fixed"""

    def get_satellites(self) -> int:
        """Number of visible satellites (minimum 8 required)"""

    def get_hdop(self) -> float:
        """Horizontal dilution of precision (must be < 2.0)"""
```

### Safety Interlock System

**Battery Management:**
- **Battery Type**: 6S6P Li-ion (Amprius cells)
- **Nominal Voltage**: 22.2V (3.7V/cell)
- **Low Battery Threshold**: 19.2V (3.2V/cell - 20% capacity)
- **Critical Battery**: 18.0V (3.0V/cell - emergency RTL)
- **Full Charge**: 25.2V (4.2V/cell)

**GPS/GNSS Requirements (Here4 CAN GPS):**
- **GPS Module**: Here4 on CAN bus
- **Minimum Satellites**: 8 (recommended by Here4)
- **HDOP Threshold**: < 2.0 (standard for autonomous ops)
- **3D Fix Required**: Yes (no operations in 2D fix)
- **RTK Support**: Available if base station configured

**Geofence Configuration:**
- **Type**: ArduPilot/PX4 standard geofence
- **Configuration**: Via Mission Planner or QGC
- **Shape**: Cylindrical or Polygonal (per GCS selection)
- **Action on Breach**: RTL or Land (configurable)
- **Integration**: Read fence parameters via MAVLink

**Signal Loss Handling:**
- **Primary Action**: Switch to SEARCHING state
- **Behavior**: Resume expanding square search pattern
- **Operator Control**: Full manual override available
- **Auto-Recovery**: Re-enter HOMING if signal reacquired
- **Timeout**: After 30 seconds, prompt operator for RTL

**RC Override & Mode Detection:**
- **Manual Override Trigger**: ANY stick movement during GUIDED
- **Mode Override**: ANY mode change from GUIDED
- **Detection Method**: Monitor RC_CHANNELS_RAW via MAVLink
- **Deadband**: ±50 PWM units from center (1500)
- **Response**: Immediate transition to IDLE state
- **Notification**: Alert operator via UI and MAVLink STATUSTEXT

### Hardware Auto-Detection System

**Automatic Hardware Discovery:**
```python
class HardwareAutoDetect:
    async def detect_all(self) -> dict:
        """Auto-detect all connected hardware"""
        return {
            "hackrf": await self.detect_hackrf(),
            "cube_orange": await self.detect_cube(),
            "status": "operational" if all_found else "degraded"
        }

    async def detect_hackrf(self) -> dict:
        """Detect HackRF via USB enumeration"""
        # Check USB Bus 003 for HackRF (idVendor=1d50, idProduct=6089)
        # Use pyhackrf library for device access

    async def detect_cube(self) -> dict:
        """Detect Cube Orange+ via serial probe"""
        # Primary: /dev/ttyACM0 (confirmed)
        # Secondary: /dev/ttyACM1 (confirmed)
        # Send HEARTBEAT and wait for response
```

**Hardware Health Monitoring:**
- Check every 5 seconds for device presence
- Auto-reconnect on disconnect
- Graceful degradation if hardware unavailable
- Alert operator of hardware status changes

### Real-Time Performance Metrics

**System Performance Dashboard:**
```python
class PerformanceMetrics:
    """Real-time performance monitoring"""

    # CPU Metrics
    cpu_usage: float  # Current CPU usage (target <30%)
    cpu_temp: float   # CPU temperature (°C)

    # Memory Metrics
    ram_used: float   # RAM usage in MB
    ram_percent: float  # RAM usage percentage

    # SDR Performance
    sample_rate_actual: float  # Actual vs configured
    sample_drops: int  # Dropped samples count
    usb_throughput: float  # MB/s USB data rate

    # Processing Performance
    rssi_latency: float  # RSSI computation time (ms)
    fft_time: float  # FFT processing time (ms)

    # MAVLink Performance
    mavlink_latency: float  # Command round-trip (ms)
    mavlink_packet_loss: float  # Packet loss percentage

    # WebSocket Performance
    ws_latency: float  # WebSocket round-trip (ms)
    ws_connections: int  # Active connections
```

**Performance Monitoring Implementation:**
```python
# Real-time metrics collection
async def collect_metrics():
    """Collect performance metrics at 1 Hz"""
    metrics = PerformanceMetrics()

    # CPU usage via psutil
    metrics.cpu_usage = psutil.cpu_percent(interval=0.1)
    metrics.cpu_temp = read_cpu_temp()  # Pi-specific

    # Memory usage
    mem = psutil.virtual_memory()
    metrics.ram_used = mem.used / 1024 / 1024
    metrics.ram_percent = mem.percent

    # SDR performance from HackRF
    metrics.sample_rate_actual = hackrf.get_sample_rate()
    metrics.sample_drops = hackrf.get_drops()

    # Processing timing
    with Timer() as t:
        rssi = compute_rssi(samples)
    metrics.rssi_latency = t.elapsed_ms

    return metrics
```

**Performance Targets:**
| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| CPU Usage | <30% | >50% | >80% |
| RAM Usage | <500MB | >700MB | >900MB |
| RSSI Latency | <100ms | >200ms | >500ms |
| Sample Drops | 0/sec | >10/sec | >100/sec |
| MAVLink Latency | <50ms | >100ms | >500ms |
| USB Throughput | 80MB/s | <40MB/s | <20MB/s |

## Hardware Dependencies

### SDR Hardware Abstraction
- **SoapySDR**: Vendor-neutral SDR interface library
  - Provides unified API for multiple SDR devices
  - Installed via system packages: `soapysdr-tools`, `libsoapysdr-dev`, `python3-soapysdr`
  - Device modules: `soapysdr-module-hackrf`, `soapysdr-module-rtlsdr`

### Supported SDR Devices
- **Primary**: HackRF One (850 MHz - 6 GHz, 20 Msps)
- **Alternative**: RTL-SDR (24 MHz - 1.7 GHz, 2.4 Msps)
- **Future**: USRP, LimeSDR, PlutoSDR via SoapySDR modules

### Technology Stack Table

| Category               | Technology                   | Version         | Purpose                                    | Rationale                                                                      |
| ---------------------- | ---------------------------- | --------------- | ------------------------------------------ | ------------------------------------------------------------------------------ |
| SDR Abstraction        | SoapySDR                     | 0.8.1           | Hardware-agnostic SDR interface            | Supports multiple SDR devices through unified API                              |
| Python Package Manager | uv                           | Latest          | Fast Python package and project management | 10-100x faster than pip, handles Python versions, modern dependency resolution |
| Python Runner          | uvx                          | Latest          | Run Python apps in isolated environments   | Zero-config execution of Python tools and scripts                              |
| Frontend Language      | TypeScript                   | 5.9.2           | Type-safe frontend development             | Prevents runtime errors in safety-critical UI                                  |
| Frontend Framework     | React                        | 18.3.1          | UI component framework                     | Proven reliability, extensive ecosystem                                        |
| UI Component Library   | MUI (Material-UI)            | 7.3.1           | Pre-built components                       | Professional emergency services aesthetic, accessibility                       |
| State Management       | React Context + useReducer   | Built-in        | Local state management                     | Minimal overhead for Pi 5, no Redux complexity                                 |
| Backend Language       | Python                       | 3.11-3.13       | Backend services and signal processing     | AsyncIO support, extensive SDR libraries, managed by uv                        |
| Backend Framework      | FastAPI                      | 0.116.1         | REST API and WebSocket server              | High performance async, automatic OpenAPI docs                                 |
| API Style              | REST + WebSocket             | HTTP/1.1 + WS   | API communication                          | REST for commands, WebSocket for real-time data                                |
| Database               | SQLite                       | 3.50.4          | Local data persistence                     | Zero-configuration, file-based for Pi SD card                                  |
| Cache                  | In-Memory (Python dict)      | N/A             | Caching RSSI data                          | Minimal overhead, no Redis needed                                              |
| File Storage           | Local Filesystem             | ext4            | Config and log storage                     | Direct SD card access                                                          |
| Authentication         | API Key (local)              | N/A             | Simple auth for local network              | No external auth needed for field ops                                          |
| Frontend Testing       | Jest + React Testing Library | 30.0.5 / 16.3.0 | Component and unit testing                 | Standard React testing stack                                                   |
| Backend Testing        | pytest + pytest-asyncio      | 8.4.1 / 1.1.0   | Async Python testing                       | Handles AsyncIO testing                                                        |
| Backend Linting        | Ruff                         | 0.8.6           | Fast Python linter and formatter           | 10-100x faster than flake8/black, drop-in replacement                          |
| E2E Testing            | Playwright                   | 1.54.2          | End-to-end testing                         | Works headless on Pi                                                           |
| Build Tool             | Vite                         | 7.1.1           | Frontend bundling                          | Faster than Webpack on Pi 5                                                    |
| Bundler                | esbuild (via Vite)           | 0.25.8          | JavaScript bundling                        | Optimized for ARM processors                                                   |
| IaC Tool               | Ansible                      | 11.8.0          | Pi deployment automation                   | Idempotent configuration management                                            |
| CI/CD                  | GitHub Actions               | N/A             | Automated testing and deployment           | Free for open source                                                           |
| Monitoring             | Custom logging + Prometheus  | 3.5.0 LTS       | Metrics and logging                        | Lightweight, local metrics                                                     |
| Logging                | Python logging + systemd     | Built-in        | Structured logging                         | Integrated with systemd journal                                                |
| CSS Framework          | MUI sx prop                  | 7.3.1           | Styling via sx prop                        | 95% MUI, 5% custom CSS for performance                                         |

### Additional Dependencies

#### MAVLink/Drone Control Stack

| Category             | Technology        | Version | Purpose                             | Rationale                                                                                                        |
| -------------------- | ----------------- | ------- | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Ground Station       | MAVProxy          | 1.8.74  | MAVLink ground control station      | Industry standard, Python-based, integrates with mission planners                                                |
| Drone SDK            | DroneKit          | 2.9.2   | High-level drone control API        | Simplifies MAVLink commands, vehicle abstraction                                                                 |
| MAVLink Library      | pymavlink         | 2.4.49  | MAVLink protocol implementation     | Core protocol library, included with MAVProxy                                                                    |
| Computer Vision      | OpenCV (headless) | 4.12.0  | Image processing for visual markers | Headless version for server deployment                                                                           |
| Serial Communication | PySerial          | 3.5     | Serial port communication           | Required for MAVLink over serial/USB                                                                             |
| Companion Libraries  | future            | 1.0.0   | Python 2/3 compatibility            | Required by DroneKit for legacy support (Note: Python 3.13+ compatibility issue with collections.MutableMapping) |

#### SDR (Software Defined Radio) Stack

| Category             | Technology        | Version      | Purpose                    | Rationale                                                    |
| -------------------- | ----------------- | ------------ | -------------------------- | ------------------------------------------------------------ |
| SDR Hardware Support | HackRF Host Tools | git-c5d63b97 | HackRF One SDR control     | Built from source for latest features and ARM64 optimization |
| SDR Library          | libhackrf         | 0.9          | HackRF library interface   | Core library for HackRF hardware access                      |
| Python SDR Interface | pyhackrf          | 0.2.0        | HackRF Python bindings     | Primary interface installed via uv                           |
| Python SDR Fallback  | pyrtlsdr          | 0.3.0        | RTL-SDR Python bindings    | Support for RTL-SDR dongles as alternative hardware          |
| Signal Processing    | NumPy             | 2.2.1        | Numerical computing        | Core array operations for signal processing                  |
| Scientific Computing | SciPy             | 1.15.1       | Advanced signal processing | FFT, filters, signal analysis algorithms                     |

#### System GUI Dependencies (Installed via apt)

| Category           | Technology     | Version | Purpose                        | Rationale                                     |
| ------------------ | -------------- | ------- | ------------------------------ | --------------------------------------------- |
| GUI Framework      | wxPython       | 4.2.0   | MAVProxy GUI support           | Required for MAVProxy map and console windows |
| Plotting Library   | Matplotlib     | 3.10.5  | Real-time signal visualization | Used by MAVProxy for graphs and system GUI    |
| OpenCV GUI Support | python3-opencv | System  | OpenCV with GUI support        | System package for full OpenCV functionality  |

#### Build Dependencies

| Category       | Technology       | Version | Purpose                            | Rationale                                        |
| -------------- | ---------------- | ------- | ---------------------------------- | ------------------------------------------------ |
| C++ Compiler   | g++              | 12.2.0  | Compile native extensions          | Required for building HackRF and Python packages |
| Build System   | CMake            | 3.25.1  | Cross-platform build configuration | Required for HackRF compilation                  |
| USB Library    | libusb-1.0-0-dev | 1.0.26  | USB device communication           | Required for SDR hardware communication          |
| FFT Library    | libfftw3-dev     | 3.3.10  | Fast Fourier Transform             | Optimized FFT for signal processing              |
| Package Config | pkg-config       | 1.8.1   | Library configuration              | Manages compiler flags for libraries             |

## Data Models

### SignalDetection

**Purpose:** Records RF signal detection events with metadata for analysis and replay

**Key Attributes:**

- id: UUID - Unique identifier for detection event
- timestamp: datetime - UTC timestamp of detection
- frequency: float - Center frequency in Hz
- rssi: float - Signal strength in dBm
- snr: float - Signal-to-noise ratio in dB
- confidence: float - Detection confidence percentage (0-100)
- location: JSON - GPS coordinates if available
- state: string - System state during detection (SEARCHING, DETECTING, HOMING)

#### TypeScript Interface

```typescript
interface SignalDetection {
  id: string;
  timestamp: string; // ISO 8601
  frequency: number;
  rssi: number;
  snr: number;
  confidence: number;
  location?: {
    lat: number;
    lon: number;
    alt: number;
  };
  state: "IDLE" | "SEARCHING" | "DETECTING" | "HOMING" | "HOLDING";
}
```

#### Relationships

- Has many RSSIReadings (time series data)
- Belongs to one Mission

### RSSIReading

**Purpose:** Time-series RSSI data for real-time visualization and gradient analysis

**Key Attributes:**

- timestamp: datetime - Microsecond precision timestamp
- rssi: float - Signal strength in dBm
- noise_floor: float - Estimated noise floor in dBm
- detection_id: UUID - Associated detection event (nullable)

#### TypeScript Interface

```typescript
interface RSSIReading {
  timestamp: string; // ISO 8601 with microseconds
  rssi: number;
  noiseFloor: number;
  detectionId?: string;
}
```

#### Relationships

- Belongs to SignalDetection (optional)
- Used for real-time streaming (not always persisted)

### ConfigProfile

**Purpose:** Stores SDR and system configuration profiles for different beacon types

**Key Attributes:**

- id: UUID - Profile identifier
- name: string - Profile name (e.g., "WiFi Beacon", "LoRa Tracker")
- sdr_config: JSON - SDR settings (frequency, sample_rate, gain)
- signal_config: JSON - Signal processing parameters
- homing_config: JSON - Homing behavior parameters
- is_default: boolean - Default profile flag
- created_at: datetime - Profile creation time
- updated_at: datetime - Last modification time

#### TypeScript Interface

```typescript
interface ConfigProfile {
  id: string;
  name: string;
  sdrConfig: {
    frequency: number;
    sampleRate: number;
    gain: number;
    bandwidth: number;
  };
  signalConfig: {
    fftSize: number;
    ewmaAlpha: number;
    triggerThreshold: number;
    dropThreshold: number;
  };
  homingConfig: {
    forwardVelocityMax: number;
    yawRateMax: number;
    approachVelocity: number;
    signalLossTimeout: number;
  };
  isDefault: boolean;
  createdAt: string;
  updatedAt: string;
}
```

#### Relationships

- Used by multiple Missions
- Can be cloned/modified

### SystemState

**Purpose:** Real-time system state for UI synchronization and safety monitoring

**Key Attributes:**

- current_state: enum - State machine state
- homing_enabled: boolean - Homing activation status
- flight_mode: string - Current flight controller mode
- battery_percent: float - Battery remaining
- gps_status: string - GPS fix status
- mavlink_connected: boolean - MAVLink connection status
- sdr_status: string - SDR hardware status
- safety_interlocks: JSON - Status of all safety checks

#### TypeScript Interface

```typescript
interface SystemState {
  currentState: "IDLE" | "SEARCHING" | "DETECTING" | "HOMING" | "HOLDING";
  homingEnabled: boolean;
  flightMode: string;
  batteryPercent: number;
  gpsStatus: "NO_FIX" | "2D_FIX" | "3D_FIX" | "RTK";
  mavlinkConnected: boolean;
  sdrStatus: "CONNECTED" | "DISCONNECTED" | "ERROR";
  safetyInterlocks: {
    modeCheck: boolean;
    batteryCheck: boolean;
    geofenceCheck: boolean;
    signalCheck: boolean;
    operatorCheck: boolean;
  };
}
```

#### Relationships

- Singleton in-memory state
- Broadcast via WebSocket

### Mission

**Purpose:** Groups related flights and detections for analysis and reporting

**Key Attributes:**

- id: UUID - Mission identifier
- name: string - Mission name
- start_time: datetime - Mission start
- end_time: datetime - Mission end (nullable)
- search_area: JSON - GeoJSON polygon
- profile_id: UUID - Configuration profile used
- total_detections: integer - Detection count
- notes: text - Operator notes

#### TypeScript Interface

```typescript
interface Mission {
  id: string;
  name: string;
  startTime: string;
  endTime?: string;
  searchArea?: GeoJSON.Polygon;
  profileId: string;
  totalDetections: number;
  notes?: string;
}
```

#### Relationships

- Has many SignalDetections
- Uses one ConfigProfile

## API Specification

### REST API Specification

```yaml
openapi: 3.0.0
info:
  title: RF-Homing SAR Drone API
  version: 1.0.0
  description: REST API for RF homing payload control and monitoring
servers:
  - url: http://localhost:8080/api
    description: Local Raspberry Pi server

paths:
  /system/status:
    get:
      summary: Get current system status
      responses:
        200:
          description: System status
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/SystemState"

  /system/homing:
    post:
      summary: Enable/disable homing mode
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                enabled:
                  type: boolean
                confirmationToken:
                  type: string
      responses:
        200:
          description: Homing state changed
        403:
          description: Safety interlock blocked activation

  /config/profiles:
    get:
      summary: List configuration profiles
      responses:
        200:
          description: List of profiles
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: "#/components/schemas/ConfigProfile"

    post:
      summary: Create new profile
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/ConfigProfile"
      responses:
        201:
          description: Profile created

  /config/profiles/{id}:
    put:
      summary: Update profile
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/ConfigProfile"
      responses:
        200:
          description: Profile updated

  /config/profiles/{id}/activate:
    post:
      summary: Activate a profile
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
      responses:
        200:
          description: Profile activated

  /detections:
    get:
      summary: Get detection history
      parameters:
        - name: limit
          in: query
          schema:
            type: integer
            default: 100
        - name: since
          in: query
          schema:
            type: string
            format: date-time
      responses:
        200:
          description: Detection events
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: "#/components/schemas/SignalDetection"

  /missions:
    get:
      summary: List missions
      responses:
        200:
          description: List of missions

    post:
      summary: Start new mission
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                name:
                  type: string
                profileId:
                  type: string
      responses:
        201:
          description: Mission started

  /missions/{id}/end:
    post:
      summary: End active mission
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
      responses:
        200:
          description: Mission ended

  /search/pattern:
    post:
      summary: Configure search pattern
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                pattern:
                  type: string
                  enum: [expanding_square, spiral, lawnmower]
                spacing:
                  type: number
                velocity:
                  type: number
                bounds:
                  type: object
      responses:
        200:
          description: Pattern configured

components:
  schemas:
    SystemState:
      # As defined in data models
    ConfigProfile:
      # As defined in data models
    SignalDetection:
      # As defined in data models
```

### WebSocket Events

```typescript
// WebSocket message types for real-time communication
interface WSMessage {
  type: "rssi" | "state" | "detection" | "telemetry" | "error";
  timestamp: string;
  data: any;
}

// RSSI update (10Hz)
interface RSSIUpdate {
  type: "rssi";
  data: {
    rssi: number;
    noiseFloor: number;
    snr: number;
    confidence: number;
  };
}

// State change notification
interface StateUpdate {
  type: "state";
  data: SystemState;
}

// Detection event
interface DetectionEvent {
  type: "detection";
  data: SignalDetection;
}

// Telemetry update (2Hz)
interface TelemetryUpdate {
  type: "telemetry";
  data: {
    position: { lat: number; lon: number; alt: number };
    battery: number;
    flightMode: string;
    velocity: { forward: number; yaw: number };
  };
}
```

## Optimized Component Architecture (v2.0)

### Refactored State Machine Components

The original 1,104-line state machine has been decomposed into 6 focused components:

1. **StateTransitionManager** (~200 lines)
   - Manages state transitions with validation
   - Enforces transition rules and guards
   - Maintains transition history

2. **StateValidator** (~150 lines)
   - Validates state preconditions
   - Checks safety interlocks
   - Ensures system consistency

3. **StateEventHandler** (~200 lines)
   - Handles state entry/exit events
   - Manages event subscriptions
   - Broadcasts state changes

4. **StatePersistence** (~150 lines)
   - Saves/restores state to database
   - Handles crash recovery
   - Maintains audit trail

5. **StateHistory** (~100 lines)
   - Tracks state transition history
   - Provides rollback capability
   - Generates state analytics

6. **StateContext** (~150 lines)
   - Manages state-specific data
   - Provides state encapsulation
   - Handles context switching

### New Performance Components

**NoiseEstimator** (161 lines)
- O(1) percentile calculations using sliding window
- 97.7% CPU reduction in signal processing
- Real-time noise floor estimation

**CircuitBreaker** (308 lines)
- Automatic failure recovery
- Exponential backoff strategy
- 3-failure threshold, 30s recovery

**AsyncFileIO** (New)
- Non-blocking file operations
- Event loop protection (<10ms)
- Async context managers

**AsyncDatabasePool** (New)
- Connection pooling
- Async query execution
- Transaction management

## Components

### Core Service

**Responsibility:** Main application orchestrator managing lifecycle of all services and handling graceful shutdown

**Key Interfaces:**

- Service registration and dependency injection
- Graceful shutdown coordination
- Configuration loading and validation
- Health check aggregation

**Dependencies:** All other services

**Technology Stack:** Python AsyncIO with dependency injection pattern

### SDR Service

**Responsibility:** Hardware abstraction layer for SDR devices, managing device lifecycle and IQ sample streaming

**Key Interfaces:**

- `initialize(config: SDRConfig)` - Device initialization
- `stream_iq() -> AsyncGenerator` - IQ sample streaming
- `set_frequency(freq: float)` - Frequency tuning
- `get_status() -> SDRStatus` - Health monitoring

**Dependencies:** SoapySDR, hardware drivers

**Technology Stack:** Python SoapySDR bindings with async wrapper

### Signal Processing Service

**Responsibility:** Real-time RSSI computation, noise floor estimation, and signal detection logic

**Key Interfaces:**

- `process_iq(samples: np.array) -> RSSIReading` - RSSI calculation
- `update_noise_floor(readings: List[float])` - Noise estimation
- `detect_signal(rssi: float) -> DetectionEvent` - Detection logic
- `compute_gradient(history: List[RSSIReading]) -> Vector` - Gradient for homing

**Dependencies:** NumPy, SciPy, SDR Service

**Technology Stack:** NumPy/SciPy with optimized FFT operations

### MAVLink Service

**Responsibility:** Bidirectional communication with flight controller, command sending, and telemetry reception

**Key Interfaces:**

- `connect(port: str, baud: int)` - Serial connection
- `send_velocity_command(vx: float, yaw_rate: float)` - Movement commands
- `get_telemetry() -> FlightTelemetry` - Current telemetry
- `subscribe_mode_changes() -> AsyncIterator` - Mode monitoring

**Dependencies:** pymavlink, serial interface

**Technology Stack:** pymavlink with AsyncIO serial wrapper

### State Machine Service

**Responsibility:** Manages operational states (IDLE, SEARCHING, HOMING) with safety-enforced transitions

**Key Interfaces:**

- `transition_to(state: State) -> bool` - Request state change
- `get_current_state() -> State` - Current state query
- `register_guard(guard: SafetyGuard)` - Add safety check
- `get_allowed_transitions() -> List[State]` - Valid transitions

**Dependencies:** MAVLink Service, Signal Processing Service

**Technology Stack:** Python state machine with async guards

### Homing Controller

**Responsibility:** Implements gradient climbing algorithm and generates velocity commands for beacon approach

**Key Interfaces:**

- `enable_homing(confirmation: str) -> bool` - Activate homing
- `compute_velocity(gradient: Vector) -> VelocityCommand` - Gradient to velocity
- `execute_sampling_maneuver()` - S-turn for gradient sampling
- `disable_homing(reason: str)` - Deactivate homing

**Dependencies:** State Machine Service, MAVLink Service

**Technology Stack:** Python control algorithms with NumPy

### Web API Service

**Responsibility:** FastAPI server providing REST endpoints and WebSocket connections for frontend

**Key Interfaces:**

- REST endpoints as defined in API specification
- WebSocket connection management
- Static file serving for React app
- CORS and security middleware

**Dependencies:** All backend services

**Technology Stack:** FastAPI with Uvicorn ASGI server

### Frontend Dashboard Component

**Responsibility:** Main operational view with real-time RSSI display and system status

**Key Interfaces:**

- WebSocket subscription for real-time updates
- Homing activation control
- Safety status display
- Detection event log

**Dependencies:** WebSocket Service, API Client

**Technology Stack:** React with MUI components, recharts for graphs

### Frontend Configuration Component

**Responsibility:** SDR and profile configuration management interface

**Key Interfaces:**

- Profile CRUD operations
- Real-time parameter adjustment
- Profile activation
- Settings validation

**Dependencies:** REST API Client

**Technology Stack:** React with MUI form components

### Component Diagrams

```mermaid
graph LR
    subgraph "Frontend Components"
        Dashboard[Dashboard<br/>Component]
        Config[Configuration<br/>Component]
        Homing[Homing Control<br/>Component]
        WSClient[WebSocket<br/>Client]
        APIClient[API<br/>Client]
    end

    subgraph "Backend Services"
        WebAPI[Web API<br/>Service]
        Core[Core<br/>Service]
        SDR[SDR<br/>Service]
        Signal[Signal Processing<br/>Service]
        MAVLink[MAVLink<br/>Service]
        State[State Machine<br/>Service]
        HomingCtrl[Homing<br/>Controller]
    end

    subgraph "Data Layer"
        DB[(SQLite)]
        Cache[Memory<br/>Cache]
    end

    Dashboard --> WSClient
    Dashboard --> APIClient
    Config --> APIClient
    Homing --> APIClient
    Homing --> WSClient

    WSClient -.->|WS| WebAPI
    APIClient -.->|REST| WebAPI

    WebAPI --> Core
    Core --> SDR
    Core --> Signal
    Core --> MAVLink
    Core --> State
    Core --> HomingCtrl

    Signal --> SDR
    HomingCtrl --> State
    HomingCtrl --> MAVLink
    State --> MAVLink

    WebAPI --> DB
    WebAPI --> Cache
    Signal --> Cache
```

## External APIs

Since this system operates in disconnected field environments, no external APIs are required. All functionality is self-contained on the Raspberry Pi 5.

## Core Workflows

### Signal Detection and Homing Activation Workflow

```mermaid
sequenceDiagram
    participant Op as Operator
    participant UI as React UI
    participant WS as WebSocket
    participant API as FastAPI
    participant SM as State Machine
    participant SP as Signal Processor
    participant SDR as SDR Service
    participant HC as Homing Controller
    participant MAV as MAVLink Service
    participant FC as Flight Controller

    Note over SDR,SP: Continuous signal monitoring (10Hz)
    SDR->>SP: IQ samples stream
    SP->>SP: Compute RSSI
    SP->>WS: RSSI update
    WS->>UI: Update display

    alt Signal Detected (SNR > 12dB)
        SP->>SM: Signal detected event
        SM->>SM: Transition to DETECTING
        SM->>API: State change notification
        API->>WS: State update
        WS->>UI: Show detection alert
        UI->>UI: Enable homing button

        Op->>UI: Click Enable Homing
        UI->>UI: Show confirmation slider
        Op->>UI: Slide to confirm
        UI->>API: POST /system/homing {enabled: true}

        API->>SM: Request HOMING state
        SM->>SM: Check safety interlocks
        alt All interlocks pass
            SM->>SM: Transition to HOMING
            SM->>HC: Activate homing
            HC->>SP: Get gradient
            SP->>HC: Return gradient vector
            HC->>HC: Compute velocity
            HC->>MAV: Send velocity command
            MAV->>FC: SET_POSITION_TARGET_LOCAL_NED
            FC->>FC: Execute movement

            loop While signal maintained
                SP->>HC: Updated gradient
                HC->>MAV: Adjusted velocity
                MAV->>FC: New commands
            end
        else Interlock failed
            SM->>API: Blocked (reason)
            API->>UI: Error response
            UI->>Op: Show blocked reason
        end
    end

    alt Signal Lost > 10s
        SP->>SM: Signal lost timeout
        SM->>HC: Disable homing
        HC->>MAV: Stop commands
        SM->>SM: Transition to SEARCHING
        SM->>API: State change
        API->>WS: State update
        WS->>UI: Update display
    end
```

### Configuration Profile Loading Workflow

```mermaid
sequenceDiagram
    participant Op as Operator
    participant UI as React UI
    participant API as FastAPI
    participant DB as SQLite
    participant Core as Core Service
    participant SDR as SDR Service
    participant SP as Signal Processor

    Op->>UI: Select profile
    UI->>API: GET /config/profiles
    API->>DB: Query profiles
    DB->>API: Return profiles
    API->>UI: Profile list
    UI->>Op: Display profiles

    Op->>UI: Choose "LoRa Beacon"
    UI->>API: POST /config/profiles/{id}/activate
    API->>DB: Load profile details
    DB->>API: Profile config

    API->>Core: Apply configuration
    Core->>SDR: Update SDR settings
    SDR->>SDR: Reconfigure hardware
    SDR->>Core: Confirmation

    Core->>SP: Update processing params
    SP->>SP: Reset filters
    SP->>Core: Confirmation

    Core->>API: Profile activated
    API->>UI: Success response
    UI->>Op: Show active profile
```

## Database Schema

```sql
-- Configuration profiles for different beacon types
CREATE TABLE config_profiles (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    name TEXT NOT NULL UNIQUE,
    sdr_config JSON NOT NULL,
    signal_config JSON NOT NULL,
    homing_config JSON NOT NULL,
    is_default BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Mission tracking
CREATE TABLE missions (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    name TEXT NOT NULL,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP NULL,
    search_area JSON NULL,
    profile_id TEXT REFERENCES config_profiles(id),
    total_detections INTEGER DEFAULT 0,
    notes TEXT NULL
);

-- Signal detection events
CREATE TABLE signal_detections (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    mission_id TEXT REFERENCES missions(id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    frequency REAL NOT NULL,
    rssi REAL NOT NULL,
    snr REAL NOT NULL,
    confidence REAL NOT NULL,
    location JSON NULL,
    state TEXT NOT NULL CHECK(state IN ('IDLE','SEARCHING','DETECTING','HOMING','HOLDING'))
);

-- Time-series RSSI data (optional persistence for analysis)
CREATE TABLE rssi_readings (
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    rssi REAL NOT NULL,
    noise_floor REAL NOT NULL,
    detection_id TEXT NULL REFERENCES signal_detections(id)
);

-- Indexes for performance
CREATE INDEX idx_detections_mission ON signal_detections(mission_id);
CREATE INDEX idx_detections_timestamp ON signal_detections(timestamp);
CREATE INDEX idx_rssi_detection ON rssi_readings(detection_id);
CREATE INDEX idx_rssi_timestamp ON rssi_readings(timestamp);

-- Triggers for updated_at
CREATE TRIGGER update_profile_timestamp
AFTER UPDATE ON config_profiles
BEGIN
    UPDATE config_profiles SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;
```

## Frontend Architecture

### Component Architecture

#### Component Organization

```
src/
├── components/
│   ├── common/
│   │   ├── SafetyToggle.tsx
│   │   ├── StatusBadge.tsx
│   │   └── AlertBanner.tsx
│   ├── dashboard/
│   │   ├── SignalMeter.tsx
│   │   ├── RSSIGraph.tsx
│   │   ├── DetectionLog.tsx
│   │   └── Dashboard.tsx
│   ├── homing/
│   │   ├── HomingControl.tsx
│   │   ├── SafetyInterlocks.tsx
│   │   └── VelocityVisualizer.tsx
│   └── config/
│       ├── ProfileManager.tsx
│       ├── SDRSettings.tsx
│       └── ConfigPanel.tsx
├── hooks/
│   ├── useWebSocket.ts
│   ├── useSystemState.ts
│   └── useRSSIData.ts
├── services/
│   ├── api.ts
│   ├── websocket.ts
│   └── config.ts
└── types/
    └── index.ts
```

#### Component Template

```typescript
// Example: SignalMeter.tsx
import React, { memo } from 'react';
import { Box, Typography, LinearProgress } from '@mui/material';
import { SignalCellularAlt } from '@mui/icons-material';

interface SignalMeterProps {
  rssi: number;
  noiseFloor: number;
  snr: number;
  size?: 'small' | 'medium' | 'large';
}

export const SignalMeter = memo<SignalMeterProps>(({
  rssi,
  noiseFloor,
  snr,
  size = 'medium'
}) => {
  const signalPercent = Math.max(0, Math.min(100, (rssi + 100) * 2));
  const color = snr > 12 ? 'success' : snr > 6 ? 'warning' : 'error';

  return (
    <Box sx={{
      p: 2,
      border: 1,
      borderColor: `${color}.main`,
      borderRadius: 1
    }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <SignalCellularAlt color={color} />
        <Typography variant="h4" sx={{ fontFamily: 'Roboto Mono' }}>
          {rssi.toFixed(1)} dBm
        </Typography>
      </Box>
      <LinearProgress
        variant="determinate"
        value={signalPercent}
        color={color}
        sx={{ mt: 1, height: 8 }}
      />
      <Typography variant="caption" sx={{ color: 'text.secondary' }}>
        SNR: {snr.toFixed(1)} dB | Noise: {noiseFloor.toFixed(1)} dBm
      </Typography>
    </Box>
  );
});
```

### State Management Architecture

#### State Structure

```typescript
// Global application state
interface AppState {
  system: SystemState;
  rssi: {
    current: RSSIReading;
    history: RSSIReading[];
    isStreaming: boolean;
  };
  detections: SignalDetection[];
  config: {
    profiles: ConfigProfile[];
    activeProfile: ConfigProfile | null;
  };
  ui: {
    homingConfirmation: boolean;
    alertQueue: Alert[];
    connectionStatus: "connected" | "connecting" | "disconnected";
  };
}

// Context provider
export const AppContext = React.createContext<{
  state: AppState;
  dispatch: React.Dispatch<Action>;
}>({} as any);
```

#### State Management Patterns

- Use React Context for global state (system status, active profile)
- Use local component state for UI-only concerns (form inputs, modals)
- Use useReducer for complex state updates (homing activation flow)
- Implement optimistic updates with rollback for better UX
- Memoize expensive computations (gradient calculations)

### Routing Architecture

#### Route Organization

```
/                    # Dashboard (default)
/homing             # Homing control panel
/config             # Configuration management
/config/profiles    # Profile management
/config/sdr        # SDR settings
/missions          # Mission history
/missions/:id      # Mission detail/replay
```

#### Protected Route Pattern

```typescript
// No authentication needed - all routes accessible locally
// But enforce safety checks for certain operations
import { Navigate, Outlet } from 'react-router-dom';
import { useSystemState } from '../hooks/useSystemState';

export const SafetyRoute: React.FC = () => {
  const { isConnected, isSafeMode } = useSystemState();

  if (!isConnected) {
    return <Navigate to="/" replace />;
  }

  return (
    <>
      {isSafeMode && (
        <Alert severity="warning">
          System in safe mode - some functions disabled
        </Alert>
      )}
      <Outlet />
    </>
  );
};
```

### Frontend Services Layer

#### API Client Setup

```typescript
// services/api.ts
import axios from "axios";

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8080/api";

export const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 5000,
  headers: {
    "Content-Type": "application/json",
  },
});

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.code === "ECONNABORTED") {
      console.error("Request timeout");
    }
    return Promise.reject(error);
  },
);
```

#### Service Example

```typescript
// services/config.service.ts
import { apiClient } from "./api";
import { ConfigProfile } from "../types";

export const configService = {
  async getProfiles(): Promise<ConfigProfile[]> {
    const { data } = await apiClient.get("/config/profiles");
    return data;
  },

  async activateProfile(id: string): Promise<void> {
    await apiClient.post(`/config/profiles/${id}/activate`);
  },

  async saveProfile(profile: Partial<ConfigProfile>): Promise<ConfigProfile> {
    const { data } = await apiClient.post("/config/profiles", profile);
    return data;
  },
};
```

## Backend Architecture

### Service Architecture

#### Service Organization

```
src/
├── core/
│   ├── __init__.py
│   ├── app.py              # FastAPI app initialization
│   ├── config.py           # Configuration management
│   └── dependencies.py     # Dependency injection
├── services/
│   ├── __init__.py
│   ├── sdr_service.py      # SDR hardware interface
│   ├── signal_processor.py # Signal processing
│   ├── mavlink_service.py  # MAVLink communication
│   ├── state_machine.py    # State management
│   └── homing_controller.py # Homing algorithms
├── api/
│   ├── __init__.py
│   ├── routes/
│   │   ├── system.py       # System control endpoints
│   │   ├── config.py       # Configuration endpoints
│   │   └── missions.py     # Mission management
│   └── websocket.py        # WebSocket handler
├── models/
│   ├── __init__.py
│   ├── database.py         # SQLite models
│   └── schemas.py          # Pydantic schemas
└── utils/
    ├── __init__.py
    ├── logging.py          # Logging configuration
    └── safety.py           # Safety interlock checks
```

#### Service Template

```python
# services/sdr_service.py
import asyncio
from typing import AsyncGenerator, Optional
import numpy as np
import SoapySDR
from dataclasses import dataclass

@dataclass
class SDRConfig:
    frequency: float = 2.437e9
    sample_rate: float = 2e6
    gain: float = 30.0

class SDRService:
    def __init__(self, config: SDRConfig):
        self.config = config
        self.device: Optional[SoapySDR.Device] = None
        self.stream = None
        self._running = False

    async def initialize(self) -> bool:
        """Initialize SDR hardware"""
        try:
            # Find and create device
            args = dict(driver="hackrf")
            self.device = SoapySDR.Device(args)

            # Configure device
            self.device.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, self.config.sample_rate)
            self.device.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, self.config.frequency)
            self.device.setGain(SoapySDR.SOAPY_SDR_RX, 0, self.config.gain)

            # Setup stream
            self.stream = self.device.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
            self.device.activateStream(self.stream)
            self._running = True

            return True
        except Exception as e:
            logger.error(f"SDR initialization failed: {e}")
            return False

    async def stream_iq(self) -> AsyncGenerator[np.ndarray, None]:
        """Stream IQ samples asynchronously"""
        buffer = np.zeros(1024, dtype=np.complex64)

        while self._running:
            # Non-blocking read with asyncio
            await asyncio.sleep(0)  # Yield control

            status = self.device.readStream(self.stream, [buffer], len(buffer))
            if status.ret > 0:
                yield buffer[:status.ret].copy()

    async def shutdown(self):
        """Clean shutdown"""
        self._running = False
        if self.stream:
            self.device.deactivateStream(self.stream)
            self.device.closeStream(self.stream)
```

### Database Architecture

#### Schema Design

```python
# models/database.py
from sqlalchemy import create_engine, Column, String, Float, JSON, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import uuid
from datetime import datetime

Base = declarative_base()

class ConfigProfile(Base):
    __tablename__ = 'config_profiles'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, unique=True)
    sdr_config = Column(JSON, nullable=False)
    signal_config = Column(JSON, nullable=False)
    homing_config = Column(JSON, nullable=False)
    is_default = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class SignalDetection(Base):
    __tablename__ = 'signal_detections'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    mission_id = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    frequency = Column(Float, nullable=False)
    rssi = Column(Float, nullable=False)
    snr = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    location = Column(JSON, nullable=True)
    state = Column(String, nullable=False)

# Database setup
engine = create_engine('sqlite:///rf_homing.db')
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)
```

#### Data Access Layer

```python
# services/repository.py
from typing import List, Optional
from sqlalchemy.orm import Session
from models.database import ConfigProfile, SignalDetection

class ConfigRepository:
    def __init__(self, db: Session):
        self.db = db

    async def get_profiles(self) -> List[ConfigProfile]:
        return self.db.query(ConfigProfile).all()

    async def get_profile(self, profile_id: str) -> Optional[ConfigProfile]:
        return self.db.query(ConfigProfile).filter(
            ConfigProfile.id == profile_id
        ).first()

    async def create_profile(self, profile_data: dict) -> ConfigProfile:
        profile = ConfigProfile(**profile_data)
        self.db.add(profile)
        self.db.commit()
        self.db.refresh(profile)
        return profile

    async def activate_profile(self, profile_id: str) -> bool:
        # Deactivate all profiles
        self.db.query(ConfigProfile).update({'is_default': False})

        # Activate selected profile
        profile = await self.get_profile(profile_id)
        if profile:
            profile.is_default = True
            self.db.commit()
            return True
        return False
```

### Authentication and Authorization

#### Auth Flow

```mermaid
sequenceDiagram
    participant Browser
    participant Frontend
    participant API
    participant Core

    Note over Browser,Core: Local network only - no external auth

    Browser->>Frontend: Access UI
    Frontend->>API: Check connection
    API->>Frontend: System status
    Frontend->>Browser: Display dashboard

    Browser->>Frontend: Request homing activation
    Frontend->>Frontend: Show confirmation UI
    Browser->>Frontend: Confirm action
    Frontend->>API: POST /system/homing
    API->>Core: Validate safety interlocks

    alt Safety checks pass
        Core->>API: Authorized
        API->>Frontend: Success
        Frontend->>Browser: Homing active
    else Safety check fails
        Core->>API: Blocked (reason)
        API->>Frontend: Error
        Frontend->>Browser: Show blocked reason
    end
```

#### Middleware/Guards

```python
# api/middleware.py
from fastapi import HTTPException, Depends
from typing import Optional

class SafetyInterlock:
    """Middleware to enforce safety checks on critical operations"""

    def __init__(self, state_machine, mavlink_service):
        self.state_machine = state_machine
        self.mavlink = mavlink_service

    async def check_homing_allowed(self) -> dict:
        """Verify all conditions for homing activation"""
        checks = {
            'mode_check': False,
            'battery_check': False,
            'signal_check': False,
            'geofence_check': False
        }

        # Check flight mode is GUIDED
        telemetry = await self.mavlink.get_telemetry()
        checks['mode_check'] = telemetry.flight_mode == 'GUIDED'

        # Check battery > 20%
        checks['battery_check'] = telemetry.battery_percent > 20

        # Check signal is detected
        state = self.state_machine.current_state
        checks['signal_check'] = state in ['DETECTING', 'HOMING']

        # Check within geofence (simplified)
        checks['geofence_check'] = True  # Would check actual bounds

        all_pass = all(checks.values())
        return {
            'allowed': all_pass,
            'checks': checks
        }

# Usage in routes
@router.post("/system/homing")
async def enable_homing(
    enabled: bool,
    safety: SafetyInterlock = Depends()
):
    if enabled:
        result = await safety.check_homing_allowed()
        if not result['allowed']:
            raise HTTPException(
                status_code=403,
                detail=f"Safety interlock blocked: {result['checks']}"
            )
    # Proceed with homing activation
```

## Unified Project Structure

```
rf-homing-sar-drone/
├── .github/                    # CI/CD workflows
│   └── workflows/
│       ├── ci.yaml            # Test on push
│       └── release.yaml       # Build and package
├── src/                       # Main source code
│   ├── backend/              # Python backend
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── app.py        # FastAPI initialization
│   │   │   ├── config.py     # Configuration loader
│   │   │   └── dependencies.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── sdr_service.py
│   │   │   ├── signal_processor.py
│   │   │   ├── mavlink_service.py
│   │   │   ├── state_machine.py
│   │   │   └── homing_controller.py
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes/
│   │   │   │   ├── system.py
│   │   │   │   ├── config.py
│   │   │   │   └── missions.py
│   │   │   ├── websocket.py
│   │   │   └── middleware.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── database.py
│   │   │   └── schemas.py
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── logging.py
│   │   │   └── safety.py
│   │   └── main.py          # Entry point
│   └── frontend/            # React frontend
│       ├── src/
│       │   ├── components/
│       │   │   ├── common/
│       │   │   ├── dashboard/
│       │   │   ├── homing/
│       │   │   └── config/
│       │   ├── hooks/
│       │   ├── services/
│       │   ├── contexts/
│       │   ├── types/
│       │   ├── theme/
│       │   ├── App.tsx
│       │   └── main.tsx
│       ├── public/
│       │   └── index.html
│       ├── package.json
│       ├── tsconfig.json
│       └── vite.config.ts
├── shared/                   # Shared types (if needed)
│   └── types.ts
├── config/                   # Configuration files
│   ├── default.yaml         # Default configuration
│   └── profiles/            # Beacon profiles
│       ├── wifi_beacon.yaml
│       ├── lora_beacon.yaml
│       └── custom.yaml
├── scripts/                  # Utility scripts
│   ├── setup.sh             # Initial setup
│   ├── deploy.sh            # Deployment script
│   └── test_sdr.py         # SDR test utility
├── tests/                   # Test suites
│   ├── backend/
│   │   ├── unit/
│   │   ├── integration/
│   │   └── conftest.py
│   ├── frontend/
│   │   ├── components/
│   │   └── setup.ts
│   └── e2e/
│       └── homing_flow.spec.ts
├── deployment/              # Deployment configs
│   ├── systemd/
│   │   └── rf-homing.service
│   └── ansible/
│       └── playbook.yaml
├── docs/                    # Documentation
│   ├── prd.md
│   ├── front-end-spec.md
│   ├── architecture.md
│   ├── setup.md
│   └── api.md
├── .env.example            # Environment template
├── docker-compose.yaml     # Optional containerization
├── pyproject.toml          # Python project config (uv configuration)
├── uv.lock                # Locked dependencies for reproducibility
├── .python-version        # Python version for uv
├── package.json            # Root package.json
└── README.md
```

## Development Workflow

### Local Development Setup

#### Prerequisites

```bash
# System requirements
# Raspberry Pi OS 64-bit (Bookworm) or Ubuntu 22.04 LTS
# uv (Python package manager)
# Node.js 20.19+
# Git

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    nodejs \
    npm \
    libhackrf-dev \
    libusb-1.0-0-dev \
    soapysdr-tools \
    python3-soapysdr

# Install SoapySDR modules
sudo apt-get install -y \
    soapysdr-module-hackrf \
    soapysdr-module-uhd
```

#### Initial Setup

```bash
# Clone repository
git clone https://github.com/your-org/rf-homing-sar-drone.git
cd rf-homing-sar-drone

# Install Python version and create environment with uv
uv python install 3.13.5
uv venv --python 3.13.5

# Install Python dependencies
uv sync --all-extras

# Install frontend dependencies
cd src/frontend
npm install
cd ../..

# Initialize database
uv run python scripts/init_db.py

# Copy environment template
cp .env.example .env
# Edit .env with your settings
```

#### Development Commands

```bash
# Start all services
uv run ./scripts/dev.sh

# Start frontend only
cd src/frontend && npm run dev

# Start backend only
uv run uvicorn backend.main:app --reload

# Run tests
# Backend tests with uv
uv run pytest tests/backend

# Or use uvx for one-off commands
uvx --from . pytest tests/backend

# Frontend tests
cd src/frontend && npm test

# E2E tests
npm run test:e2e

# Code formatting with ruff
uv run ruff format src/
uv run ruff check --fix src/

# Type checking
uv run mypy src/
```

### Environment Configuration

#### Required Environment Variables

```bash
# Frontend (.env.local)
VITE_API_URL=http://localhost:8080
VITE_WS_URL=ws://localhost:8080/ws
VITE_UPDATE_RATE=100  # ms between updates

# Backend (.env)
# SDR Configuration
SDR_DRIVER=hackrf
SDR_FREQUENCY=2437000000
SDR_SAMPLE_RATE=2000000
SDR_GAIN=30

# MAVLink Configuration
MAVLINK_PORT=/dev/ttyACM0
MAVLINK_BAUD=115200
MAVLINK_SYSTEM_ID=255
MAVLINK_COMPONENT_ID=1

# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8080
LOG_LEVEL=INFO

# Shared
NODE_ENV=development
DATABASE_PATH=./rf_homing.db
```

## Deployment Architecture

### Deployment Strategy

**Frontend Deployment:**

- **Platform:** Static files served by FastAPI
- **Build Command:** `npm run build`
- **Output Directory:** `dist/`
- **CDN/Edge:** N/A (local only)

**Backend Deployment:**

- **Platform:** Systemd service on Raspberry Pi
- **Build Command:** `pip install -r requirements.txt`
- **Deployment Method:** Ansible playbook

### CI/CD Pipeline

```yaml
# .github/workflows/ci.yaml
name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test-backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.13"
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Install dependencies
        run: |
          uv sync --all-extras
          uv pip install pytest==8.4.1 pytest-asyncio==1.1.0 pytest-cov
      - name: Run tests
        run: uv run pytest tests/backend --cov=src/backend

  test-frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: "20"
      - name: Install and test
        run: |
          cd src/frontend
          npm ci
          npm run test
          npm run build

  build-release:
    needs: [test-backend, test-frontend]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build release package
        run: |
          ./scripts/build-release.sh
      - uses: actions/upload-artifact@v3
        with:
          name: rf-homing-release
          path: dist/
```

### Environments

| Environment | Frontend URL          | Backend URL               | Purpose           |
| ----------- | --------------------- | ------------------------- | ----------------- |
| Development | http://localhost:5173 | http://localhost:8080     | Local development |
| Testing     | http://pi.local       | http://pi.local:8080      | Pi testing        |
| Production  | http://192.168.1.100  | http://192.168.1.100:8080 | Field deployment  |

## Security and Performance

### Security Requirements

**Frontend Security:**

- CSP Headers: `default-src 'self'; ws-src 'self' ws://localhost:*`
- XSS Prevention: React automatic escaping, input sanitization
- Secure Storage: No sensitive data in localStorage, session only

**Backend Security:**

- Input Validation: Pydantic schemas for all inputs
- Rate Limiting: 100 requests/minute per IP
- CORS Policy: Allow only localhost and configured IPs

**Authentication Security:**

- Token Storage: N/A (local network only)
- Session Management: WebSocket session timeout after 30 min idle
- Password Policy: N/A (no user accounts)

### Performance Optimization

**Frontend Performance:**

- Bundle Size Target: < 500KB gzipped
- Loading Strategy: Code splitting by route
- Caching Strategy: Service worker for static assets

**Backend Performance:**

- Response Time Target: < 50ms for API calls
- Database Optimization: Indexes on timestamp fields
- Caching Strategy: In-memory cache for RSSI data (10 sec TTL)

## Testing Strategy

### Testing Pyramid

```
          E2E Tests
         /        \
    Integration Tests
       /            \
  Frontend Unit  Backend Unit
```

### Complete Test File Inventory

#### Backend Unit Tests (35 files in `tests/backend/unit/`)

```
tests/backend/unit/
├── test_api_analytics.py              # Analytics API endpoint tests
├── test_api_state.py                  # State API endpoint tests
├── test_config.py                     # Configuration system tests
├── test_config_service.py             # Configuration service tests
├── test_database_models.py            # Database model tests
├── test_field_test_service.py         # Field test service tests
├── test_homing_algorithm_comprehensive.py    # Homing algorithm tests
├── test_homing_controller_comprehensive.py   # Homing controller tests
├── test_homing_debug_mode.py          # Homing debug mode tests
├── test_homing_parameters_api.py      # Homing parameters API tests
├── test_homing_performance.py         # Homing performance tests
├── test_main.py                       # Main entry point tests
├── test_mavlink_service.py            # MAVLink service tests
├── test_mission_replay_service.py     # Mission replay tests
├── test_models_schemas.py             # Data model schema tests
├── test_performance_analytics.py      # Performance analytics tests
├── test_recommendations_engine.py     # Recommendations engine tests
├── test_report_generator.py           # Report generation tests
├── test_safety.py                     # Safety system tests
├── test_sdr_service.py                # SDR service tests
├── test_search_pattern_generator.py   # Search pattern tests
├── test_signal_processor.py           # Signal processing tests
├── test_state_integration.py          # State integration tests
├── test_state_machine_additional.py   # Additional state machine tests
├── test_state_machine_comprehensive.py # Comprehensive state tests
├── test_state_machine_enhanced.py     # Enhanced state machine tests
├── test_state_machine_entry_exit.py   # State entry/exit tests
├── test_state_override_api.py         # State override API tests
├── test_state_persistence.py          # State persistence tests
├── test_telemetry_api.py              # Telemetry API tests
├── test_telemetry.py                  # Telemetry service tests
├── test_telemetry_recorder.py         # Telemetry recorder tests
├── test_utils_safety.py               # Safety utility tests
├── test_waypoint_exporter_enhanced.py # Enhanced waypoint tests
└── test_waypoint_exporter.py          # Waypoint exporter tests
```

#### Backend Integration Tests (19 files in `tests/backend/integration/`)

```
tests/backend/integration/
├── test_analytics_api.py              # Analytics API integration
├── test_analytics_export.py           # Analytics export integration
├── test_api_auth.py                   # Authentication integration
├── test_api_config.py                 # Config API integration
├── test_api_missions.py               # Missions API integration
├── test_api_system.py                 # System API integration
├── test_app.py                        # FastAPI app integration
├── test_config_routes.py              # Config routes integration
├── test_field_test_service.py         # Field test integration
├── test_gcs_integration.py            # GCS integration tests
├── test_homing_integration.py         # Homing system integration
├── test_mavlink_integration.py        # MAVLink integration
├── test_safety_integration.py         # Safety system integration
├── test_safety_system.py              # Safety system tests
├── test_search_api.py                 # Search API integration
├── test_websocket.py                  # WebSocket integration
├── test_websocket_state_events.py     # WebSocket state events
└── conftest.py                        # Integration test fixtures
```

#### SITL Tests (7 files in `tests/backend/sitl_disabled/`)

```
tests/backend/sitl_disabled/
├── test_ardupilot_sitl_integration.py # ArduPilot SITL integration
├── test_beacon_detection_scenario.py  # Beacon detection scenarios
├── test_homing_approach_scenario.py   # Homing approach scenarios
├── test_homing_behavior.py            # Homing behavior tests
├── test_mission_abort_scenario.py     # Mission abort scenarios
├── test_safety_interlock_scenario.py  # Safety interlock scenarios
└── test_safety_scenarios.py           # Safety system scenarios
```

#### Frontend Tests

```
tests/frontend/
├── components/
│   ├── SignalMeter.test.tsx
│   ├── HomingControl.test.tsx
│   └── SafetyToggle.test.tsx
├── hooks/
│   └── useWebSocket.test.ts
└── services/
    └── api.test.ts
```

#### E2E Tests

```
tests/e2e/
├── homing_activation.spec.ts
├── profile_management.spec.ts
└── signal_detection.spec.ts
```

### Test Coverage Summary

#### Current Test Statistics (as of 2025-08-14)

- **Total Unit Tests:** 740 tests across 35 files
- **Total Integration Tests:** 171 tests across 19 files
- **Total SITL Tests:** 36 tests across 7 files
- **Pass Rate:** 718/740 unit tests passing (97.0%)
- **Backend Coverage:** ~52% (when tests run without timeout)
- **Target Coverage:** 60% minimum

#### Key Test Files by Component

**Core Services:**

- Signal Processing: `test_signal_processor.py` (47 tests)
- MAVLink Service: `test_mavlink_service.py` (53 tests)
- SDR Service: `test_sdr_service.py` (18 tests)
- State Machine: `test_state_machine_enhanced.py` (32 tests)

**Homing System:**

- Algorithm: `test_homing_algorithm_comprehensive.py` (22 tests)
- Controller: `test_homing_controller_comprehensive.py` (21 tests)
- Debug Mode: `test_homing_debug_mode.py` (6 tests)
- Performance: `test_homing_performance.py` (7 tests)

**API Endpoints:**

- State API: `test_api_state.py` (44 tests)
- Analytics API: `test_api_analytics.py` (16 tests)
- Telemetry API: `test_telemetry_api.py` (15 tests)

**Safety & Field Testing:**

- Safety System: `test_safety.py` (28 tests)
- Safety Utils: `test_utils_safety.py` (51 tests)
- Field Test Service: `test_field_test_service.py` (48 tests)

### Test Examples

#### Frontend Component Test

```typescript
// SignalMeter.test.tsx
import { render, screen } from '@testing-library/react';
import { SignalMeter } from '../src/components/dashboard/SignalMeter';

describe('SignalMeter', () => {
  it('displays correct RSSI value', () => {
    render(<SignalMeter rssi={-75} noiseFloor={-95} snr={20} />);
    expect(screen.getByText('-75.0 dBm')).toBeInTheDocument();
  });

  it('shows success color for high SNR', () => {
    const { container } = render(
      <SignalMeter rssi={-60} noiseFloor={-95} snr={35} />
    );
    expect(container.querySelector('[color="success"]')).toBeTruthy();
  });
});
```

#### Backend API Test

```python
# test_api_routes.py
import pytest
from fastapi.testclient import TestClient
from src.backend.main import app

client = TestClient(app)

def test_system_status():
    response = client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert "currentState" in data
    assert "homingEnabled" in data

@pytest.mark.asyncio
async def test_homing_activation_blocked():
    # Test with safety interlock failure
    response = client.post("/api/system/homing", json={
        "enabled": True,
        "confirmationToken": "test"
    })
    assert response.status_code == 403
    assert "Safety interlock" in response.json()["detail"]
```

#### E2E Test

```typescript
// homing_activation.spec.ts
import { test, expect } from "@playwright/test";

test("homing activation flow", async ({ page }) => {
  await page.goto("http://localhost:5173");

  // Wait for signal detection
  await page.waitForSelector('[data-testid="signal-detected"]');

  // Click enable homing
  await page.click('[data-testid="enable-homing-btn"]');

  // Confirm activation
  await page
    .locator('[data-testid="confirm-slider"]')
    .dragTo(page.locator('[data-testid="slider-end"]'));

  // Verify homing active
  await expect(page.locator('[data-testid="homing-status"]')).toHaveText(
    "HOMING ACTIVE",
  );
});
```

## Coding Standards

### Critical Fullstack Rules

- **Type Sharing:** Always define shared types in shared/types.ts and import
- **API Calls:** Never make direct HTTP calls - use the service layer
- **Environment Variables:** Access only through config objects, never process.env directly
- **Error Handling:** All API routes must use the standard error handler
- **State Updates:** Never mutate state directly - use proper state management patterns
- **WebSocket Messages:** Use binary protocol for RSSI data, JSON for control messages
- **Safety Checks:** All velocity commands must pass through safety interlock validation
- **Logging:** Use structured logging with correlation IDs for request tracking

### Naming Conventions

| Element          | Frontend             | Backend     | Example                      |
| ---------------- | -------------------- | ----------- | ---------------------------- |
| Components       | PascalCase           | -           | `SignalMeter.tsx`            |
| Hooks            | camelCase with 'use' | -           | `useSystemState.ts`          |
| API Routes       | -                    | kebab-case  | `/api/system-status`         |
| Database Tables  | -                    | snake_case  | `signal_detections`          |
| Config Keys      | -                    | UPPER_SNAKE | `SDR_FREQUENCY`              |
| WebSocket Events | camelCase            | snake_case  | `rssiUpdate` / `rssi_update` |

## Error Handling Strategy

### Error Flow

```mermaid
sequenceDiagram
    participant UI
    participant API
    participant Service
    participant Safety

    UI->>API: Request homing activation
    API->>Safety: Check interlocks
    Safety-->>API: Interlock failed
    API->>API: Format error response
    API->>UI: 403 Forbidden + details
    UI->>UI: Display error banner
    UI->>UI: Log to console
    UI->>UI: Auto-dismiss after 5s
```

### Error Response Format

```typescript
interface ApiError {
  error: {
    code: string;
    message: string;
    details?: Record<string, any>;
    timestamp: string;
    requestId: string;
  };
}
```

### Frontend Error Handling

```typescript
// hooks/useApiError.ts
export const useApiError = () => {
  const { dispatch } = useContext(AppContext);

  const handleError = (error: AxiosError<ApiError>) => {
    const message =
      error.response?.data?.error?.message || "An unexpected error occurred";

    dispatch({
      type: "ADD_ALERT",
      payload: {
        severity: "error",
        message,
        autoHide: true,
      },
    });

    // Log for debugging
    console.error("[API Error]", error.response?.data);
  };

  return { handleError };
};
```

### Backend Error Handling

```python
# api/middleware.py
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import uuid
import traceback
from datetime import datetime

async def error_handler(request: Request, exc: Exception):
    request_id = str(uuid.uuid4())

    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": f"HTTP_{exc.status_code}",
                    "message": exc.detail,
                    "timestamp": datetime.utcnow().isoformat(),
                    "requestId": request_id
                }
            }
        )

    # Log unexpected errors
    logger.error(f"Request {request_id}: {traceback.format_exc()}")

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An internal error occurred",
                "timestamp": datetime.utcnow().isoformat(),
                "requestId": request_id
            }
        }
    )
```

## Monitoring and Observability

### Monitoring Stack

- **Frontend Monitoring:** Browser console logs + local performance API
- **Backend Monitoring:** Prometheus metrics endpoint
- **Error Tracking:** Local log aggregation with systemd journal
- **Performance Monitoring:** Custom metrics via Prometheus

### Key Metrics

**Frontend Metrics:**

- Core Web Vitals (FCP, LCP, CLS)
- JavaScript errors count
- API response times (p50, p95, p99)
- WebSocket connection drops

**Backend Metrics:**

- Request rate (req/s)
- Error rate (4xx, 5xx)
- Response time (p50, p95, p99)
- SDR sample processing latency
- MAVLink command latency
- State transition frequency
- Signal detection rate

## Technical Debt and Future Work (v2.0)

### Remaining Technical Debt (15% of system)

#### 1. Hardware Integration (35% of remaining debt)
- **Issue:** Physical hardware not yet integrated
- **Impact:** Cannot validate real-world performance
- **Resolution:** Story 2.5 (Ground Testing) blocked on hardware
- **Mitigation:** HAL interfaces ready, comprehensive mocks in place

#### 2. API Security (25% of remaining debt)
- **Issue:** No authentication/authorization implemented
- **Impact:** System only safe for local network deployment
- **Resolution:** Story 4.5 (API Security) not started
- **Mitigation:** Rate limiting and CORS configured

#### 3. Test Coverage Gap (20% of remaining debt)
- **Issue:** 67% coverage vs 85% target for safety-critical
- **Impact:** Some edge cases may not be tested
- **Resolution:** Stories 4.2, 4.6 partially complete
- **Mitigation:** 100% coverage on refactored safety components

#### 4. Deployment Configuration (15% of remaining debt)
- **Issue:** Production deployment configs incomplete
- **Impact:** Manual deployment required
- **Resolution:** Create systemd service files
- **Mitigation:** Development deployment works

#### 5. Frontend Optimization (5% of remaining debt)
- **Issue:** Service worker and PWA not configured
- **Impact:** No offline capability
- **Resolution:** Low priority enhancement
- **Mitigation:** Bundle already optimized to <1.5MB

### Migration Path for Remaining Work

1. **Phase 1: Hardware Integration** (Story 2.5)
   - Connect physical HackRF and Pixhawk
   - Validate HAL interfaces
   - Run ground tests

2. **Phase 2: Field Testing** (Story 3.4)
   - Real beacon detection tests
   - GPS-denied scenarios
   - Performance validation

3. **Phase 3: API Security** (Story 4.5)
   - Implement JWT authentication
   - Add role-based access control
   - Complete OpenAPI documentation

4. **Phase 4: Production Deployment**
   - Create systemd services
   - Configure production environment
   - Implement monitoring

## Checklist Results Report

The architecture document is now complete with all updated package versions from the framework.txt file. The key updates include:

- TypeScript 5.9.2
- React 18.3.1
- MUI 7.3.1
- Python 3.13.5
- FastAPI 0.116.1
- SQLite 3.50.4
- Jest 30.0.5
- React Testing Library 16.3.0
- pytest 8.4.1
- pytest-asyncio 1.1.0
- Playwright 1.54.2
- Vite 7.1.1
- esbuild 0.25.8
- Ansible 11.8.0
- Prometheus 3.5.0 LTS

All versions are production-stable and suitable for the safety-critical drone system.
