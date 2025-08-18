# User Interface Design Goals

## Overall UX Vision

A focused web-based payload control interface hosted on the Raspberry Pi 5, accessible via browser at the companion computer's IP address. This interface specifically controls the RF homing payload (SDR operations, signal processing, homing behaviors) while deliberately NOT duplicating primary flight control functions that remain in Mission Planner/QGC. The UI emphasizes real-time signal intelligence and payload-specific controls with clear separation of concerns between platform and payload operations.

## Key Interaction Paradigms

- **Payload-centric controls** for RF detection parameters, frequency selection, and homing behavior tuning with immediate application
- **Real-time signal visualization** with live spectrum waterfall displays (3Hz updates), RSSI graphs, and confidence metrics
- **Interactive frequency selection** via clickable waterfall plot to set beacon target frequencies
- **Manual frequency control** allowing operators to set waterfall center frequency (850 MHz - 6.5 GHz range)
- **Flight controller integration** with explicit GUIDED mode switching and prominent homing enable/disable controls
- **State synchronization** showing current homing state and MAVLink command status
- **Read-only flight telemetry** displaying relevant platform data (altitude, battery, flight mode) without duplicating GCS controls
- **Clear visual boundary** between payload controls (editable) and platform status (read-only)

## Core Screens and Views

- **Payload Dashboard** - Main view with live spectrum waterfall, RSSI graph, signal strength meter, frequency controls, detection state
- **Spectrum Waterfall Display** - Interactive 5MHz bandwidth waterfall (±2.5MHz around center) with clickable frequency selection, 3Hz update rate optimized for Raspberry Pi 5 performance
- **SDR Configuration Panel** - Manual center frequency input (850 MHz - 6.5 GHz), sensitivity threshold controls, bandwidth settings with immediate application
- **Flight Controller Interface** - GUIDED mode switching, homing enable/disable emergency button, MAVLink command history (last 5 human-readable commands)
- **Homing Parameters** - Trigger/drop thresholds, gradient climbing settings, search pattern dimensions
- **Detection History** - Log of detection events with timestamps, signal strength, confidence scores, beacon target frequencies
- **System Health** - SDR status, CPU/memory usage, MAVLink connection health, processing latency, HackRF One connection status

## Accessibility: WCAG AA

Ensuring compliance for emergency response personnel who may have visual or motor impairments, including high-contrast mode, keyboard navigation, and screen reader support for all critical functions.

## Branding

Clean, professional interface aligned with emergency services aesthetic - high contrast, clear typography, avoiding decorative elements. Color palette focused on operational clarity: blues for normal operation, amber for warnings, red for critical alerts.

## Target Device and Platforms: Web Responsive

Primary: Desktop/laptop browser accessing Pi5 web server (1920x1080 minimum)
Secondary: Tablet browser for field operations (iPad/Surface Pro)
Emergency: Mobile phone browser for status monitoring only

## Technical Implementation Requirements

### Spectrum Visualization
- **Library**: Plotly.js with react-plotly.js for interactive waterfall displays
- **Performance**: Canvas/WebGL rendering optimized for Raspberry Pi 5 compute constraints
- **Update Rate**: 3Hz spectrum updates (333ms intervals) via WebSocket streaming
- **Bandwidth**: 5MHz spectrum window (configurable center frequency ±2.5MHz)
- **Interaction**: Click-to-set frequency selection on waterfall plot

### Frequency Control
- **Center Frequency**: Manual operator input with immediate application (850 MHz - 6.5 GHz range per PRD-FR1)
- **Beacon Target**: Independent frequency selection via waterfall click interaction
- **Presets**: Quick-select buttons for common frequencies (433 MHz, 915 MHz, 2.4 GHz)
- **Validation**: Real-time input validation with HackRF One frequency range constraints

### Flight Controller Integration
- **Mode Control**: GUIDED mode switching capability (manual mode switching via primary GCS)
- **Safety Controls**: Prominent emergency homing disable button (per PRD-FR16: <500ms response)
- **Command History**: Last 5 human-readable MAVLink commands with timestamps
- **Status Display**: Real-time flight mode, battery, GPS status, and homing state indicators

### Performance Optimization
- **WebSocket Efficiency**: Spectrum data streaming optimized for 3Hz update rate
- **Memory Management**: Rolling buffer for waterfall history (60-second window)
- **Compute Resource Awareness**: Efficient FFT processing and canvas rendering for Raspberry Pi 5
- **Immediate Response**: All operator inputs apply instantly without confirmation dialogs
