# Technical Assumptions

## Repository Structure: Monorepo
Single repository containing all components (flight control configs, SDR processing, web UI, documentation) to simplify version control and ensure synchronized updates across the tightly-coupled payload system.

## Service Architecture
**Modular monolith** with clear separation between:
- **Core Services** (always running on boot):
  - SDR sensing daemon (Python asyncio service)
  - Signal processing pipeline (NumPy/SciPy)
  - MAVLink communication manager
  - Web server for payload UI (FastAPI/Flask)
- **Control Modules** (activated on demand):
  - Homing velocity controller
  - Search pattern generator
  - State machine manager
- All services communicate via async message passing within single Python process for deterministic timing

## Testing Requirements
**Comprehensive testing pyramid** essential for safety-critical system:
- **Unit tests** for signal processing algorithms, state machines, safety interlocks
- **Integration tests** for MAVLink communication, SDR hardware interface
- **Hardware-in-loop (HIL)** testing with real SDR and simulated flight controller
- **Software-in-loop (SITL)** testing for full mission scenarios
- **Manual test utilities** for signal injection, RSSI simulation, state forcing

## Additional Technical Assumptions and Requests
- **Python 3.10+** as primary language for rapid prototyping and extensive library support
- **AsyncIO architecture** throughout for concurrent SDR sampling and flight control
- **SoapySDR** for hardware abstraction supporting both USRP and HackRF
- **FastAPI or Flask** for lightweight web server hosting payload UI
- **WebSocket** for real-time UI updates without polling
- **Docker containerization** optional but recommended for reproducible builds
- **Git with semantic versioning** for all code and configuration
- **Ubuntu 22.04 LTS** or Raspberry Pi OS 64-bit as target OS
- **MAVLink 2.0 protocol** for all drone communication
- **No cloud dependencies** - fully operational in disconnected environments
- **Configuration via YAML/JSON** files for field-adjustable parameters
- **Systemd service** management for automatic startup and recovery
- **Logging to local storage** with rotation to prevent filling SD card
