# PISAD Deployment Guide

## System Requirements

- Raspberry Pi 5 (8GB RAM recommended)
- HackRF One SDR connected via USB
- Pixhawk 4 with Cube Orange+ connected via USB
- Log-periodic antenna (850 MHz - 6.5 GHz)

## Installation Steps

### 1. Install System Dependencies

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python and system dependencies
sudo apt install python3.11 python3.11-venv python3-pip -y
sudo apt install libusb-1.0-0-dev libfftw3-dev -y

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Setup User and Permissions

```bash
# Create pisad user if not exists
sudo useradd -m -s /bin/bash pisad

# Add user to required groups for hardware access
sudo usermod -a -G dialout,plugdev pisad

# Set permissions for USB devices
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
```

### 3. Clone and Setup Project

```bash
# Switch to pisad user
sudo su - pisad

# Clone repository
git clone https://github.com/your-org/pisad.git
cd pisad

# Install Python dependencies with uv
uv sync

# Create required directories
mkdir -p data logs
```

### 4. Configure Hardware

The system will automatically detect hardware on startup and load the appropriate configuration:

- **With Hardware**: Uses `config/hardware.yaml`
- **Without Hardware**: Falls back to `config/hardware-mock.yaml`
- **Force Mock**: Set `USE_MOCK_HARDWARE=true` environment variable

### 5. Install Systemd Service

```bash
# Copy service file
sudo cp deployment/pisad.service /etc/systemd/system/

# Reload systemd daemon
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable pisad.service

# Start the service
sudo systemctl start pisad.service
```

### 6. Service Management

```bash
# Check service status
sudo systemctl status pisad.service

# View logs
sudo journalctl -u pisad -f

# Restart service
sudo systemctl restart pisad.service

# Stop service
sudo systemctl stop pisad.service

# Disable auto-start
sudo systemctl disable pisad.service
```

## Hardware Configuration

### HackRF One Setup

1. Connect HackRF One to USB 3.0 port
2. Verify detection: `lsusb | grep HackRF`
3. Expected: `Bus 003 Device 003: ID 1d50:6089 OpenMoko, Inc. HackRF One`

### Cube Orange+ Setup

1. Connect Cube Orange+ via USB
2. Verify detection: `ls -la /dev/ttyACM*`
3. Expected: `/dev/ttyACM0` and `/dev/ttyACM1` present

### Antenna Setup

1. Connect log-periodic antenna to HackRF SMA port
2. Point antenna in search direction
3. Ensure clear line of sight

## Performance Monitoring

The service includes real-time performance monitoring:

- **Target CPU Usage**: < 30% on Raspberry Pi 5
- **Target Memory**: < 512MB
- **RSSI Update Rate**: 1 Hz (power efficient)
- **MAVLink Telemetry**: 4 Hz

Monitor performance via:
- WebSocket endpoint: `ws://localhost:8080/ws`
- REST API: `http://localhost:8080/api/system/metrics`

## Troubleshooting

### Hardware Not Detected

```bash
# Check USB devices
lsusb

# Check serial ports
ls -la /dev/ttyACM*

# Check user permissions
groups pisad

# Force mock mode for testing
export USE_MOCK_HARDWARE=true
sudo systemctl restart pisad
```

### Service Won't Start

```bash
# Check service logs
sudo journalctl -u pisad -n 50

# Test manual startup
cd /home/pisad/projects/pisad
/home/pisad/.local/bin/uv run uvicorn src.backend.core.app:app --host 0.0.0.0 --port 8080

# Verify Python environment
/home/pisad/.local/bin/uv run python --version
```

### High Resource Usage

1. Check performance metrics: `http://localhost:8080/api/system/metrics`
2. Adjust RSSI update rate in `config/hardware.yaml`
3. Reduce telemetry rate if needed
4. Monitor with: `htop` or `btop`

## Security Notes

The systemd service includes security hardening:

- Runs as non-root user (pisad)
- Limited filesystem access (ProtectSystem=strict)
- Device access restricted to required hardware
- Memory and CPU quotas enforced
- No new privileges after start

## Production Checklist

- [ ] Hardware connected and detected
- [ ] Service installed and enabled
- [ ] Logs show successful startup
- [ ] WebSocket clients can connect
- [ ] RSSI values updating at 1 Hz
- [ ] MAVLink telemetry streaming at 4 Hz
- [ ] CPU usage below 30%
- [ ] Memory usage below 512MB
- [ ] Auto-restart on failure working
- [ ] GPS lock achieved (8+ satellites)