# PISAD Setup Guide

This guide provides step-by-step instructions for setting up the PISAD (Portable Interferometric Signal Analysis Device) development environment on a Raspberry Pi 5.

## System Requirements

### Hardware

- Raspberry Pi 5 (4GB or 8GB RAM recommended)
- HackRF One SDR (primary) or RTL-SDR USB dongles (alternative)
- MicroSD card (32GB minimum, 64GB recommended)
- Power supply (5V 5A USB-C)
- Ethernet cable or WiFi connection

### Software

- Raspberry Pi OS Lite (64-bit) or Full
- Python 3.11+ (managed by uv)
- Git
- uv (Python package and environment manager)
- SDR drivers and libraries

## Installation Steps

### 1. System Preparation

Update the system and install basic dependencies:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git build-essential cmake libusb-1.0-0-dev pkg-config curl
```

### 2. Install uv

Install uv, the modern Python package and project manager:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH (add this to your ~/.bashrc or ~/.zshrc)
source $HOME/.cargo/env

# Verify installation
uv --version
```

### 3. SDR Driver Installation

Install SDR drivers and libraries based on your hardware:

#### For HackRF One (Primary):

```bash
# Install HackRF tools and libraries
sudo apt install -y hackrf libhackrf-dev

# Install SoapySDR (SDR abstraction library)
sudo apt install -y soapysdr-tools libsoapysdr-dev python3-soapysdr

# Install SoapySDR HackRF module
sudo apt install -y soapysdr-module-hackrf

# Test HackRF detection
hackrf_info
SoapySDRUtil --find="driver=hackrf"
```

#### For RTL-SDR (Alternative):

```bash
# Install RTL-SDR tools
sudo apt install -y rtl-sdr librtlsdr-dev

# Install SoapySDR RTL-SDR module
sudo apt install -y soapysdr-module-rtlsdr

# Blacklist DVB-T drivers to prevent conflicts
echo 'blacklist dvb_usb_rtl28xxu' | sudo tee /etc/modprobe.d/blacklist-rtl.conf
sudo modprobe -r dvb_usb_rtl28xxu

# Test RTL-SDR detection
rtl_test -t
SoapySDRUtil --find="driver=rtlsdr"
```

#### Python Integration with uv:

Since SoapySDR Python bindings are installed system-wide, you need to make them available in your uv environment:

```bash
# Create virtual environment with system site packages access
uv venv --system-site-packages --python=3.11

# Or if venv already exists, create symlinks
ln -sf /usr/lib/python3/dist-packages/SoapySDR.py .venv/lib/python*/site-packages/
ln -sf /usr/lib/python3/dist-packages/_SoapySDR.*.so .venv/lib/python*/site-packages/

# Test Python import
uv run python -c "import SoapySDR; print(SoapySDR.Device.enumerate())"
```

### 4. Project Setup

Clone the repository and set up the Python environment with uv:

```bash
# Clone the repository (replace with your repository URL)
git clone https://github.com/yourusername/pisad.git
cd pisad

# Install the recommended Python version
uv python install 3.13.5

# Create virtual environment with uv
uv venv --python 3.13.5

# Install all dependencies (including dev dependencies)
uv sync --all-extras

# Or install only production dependencies
uv sync

# To run commands in the uv environment
uv run python --version

# For development with hot reload
uv run --dev uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Configuration

The application uses YAML configuration files located in the `config/` directory.

#### Default Configuration

The default configuration is provided in `config/default.yaml`. This file contains all available settings with sensible defaults.

#### Custom Configuration

To create a custom configuration:

```bash
# Copy default configuration
cp config/default.yaml config/custom.yaml

# Edit with your preferred editor
nano config/custom.yaml
```

Key configuration sections:

- **SDR Settings**: Frequency, sample rate, gain
- **Signal Processing**: RSSI thresholds, averaging windows
- **Safety**: Velocity limits, interlock settings
- **Logging**: Log levels, file paths, rotation settings

#### Environment Variables

Configuration can be overridden using environment variables with the `PISAD_` prefix:

```bash
export PISAD_APP_PORT=8080
export PISAD_SDR_FREQUENCY=434000000
export PISAD_LOG_LEVEL=DEBUG
```

### 6. Directory Structure

The project follows this structure:

```
pisad/
├── src/
│   ├── backend/         # Python backend code
│   │   ├── core/        # Core functionality
│   │   ├── services/    # Business logic services
│   │   ├── api/         # FastAPI routes
│   │   ├── models/      # Data models
│   │   └── utils/       # Utility functions
│   └── frontend/        # React frontend (future)
├── config/              # Configuration files
│   ├── default.yaml     # Default configuration
│   └── profiles/        # Configuration profiles
├── tests/               # Test files
│   └── backend/
│       ├── unit/        # Unit tests
│       └── integration/ # Integration tests
├── deployment/          # Deployment configurations
│   └── systemd/         # Systemd service files
├── scripts/             # Utility scripts
├── logs/                # Application logs
├── docs/                # Documentation
├── pyproject.toml       # Project configuration and dependencies
├── uv.lock             # Locked dependency versions
└── .python-version     # Python version for uv
```

### 7. Running the Application

#### Development Mode

```bash
# Run with uv (automatically handles environment)
uv run uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Or use the project script
uv run pisad-server --dev

# Run with specific Python version
uv run --python 3.13.5 uvicorn backend.main:app --reload
```

#### Production Mode

```bash
# Run without reload
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Or use the project script
uv run pisad-server
```

#### Using uvx for Quick Commands

```bash
# Run one-off commands with uvx (no environment activation needed)
uvx --from . pisad --help

# Run tests
uvx --from . pytest

# Format code
uvx --from . black src/ tests/

# Lint code
uvx --from . ruff check src/
```

### 8. Systemd Service Installation

To run PISAD as a system service:

```bash
# Copy service file
sudo cp deployment/systemd/rf-homing.service /etc/systemd/system/

# Create pisad user (if not exists)
sudo useradd -r -s /bin/false pisad
sudo chown -R pisad:pisad /home/pisad/projects/pisad

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable rf-homing.service

# Start the service
sudo systemctl start rf-homing.service

# Check service status
sudo systemctl status rf-homing.service

# View logs
sudo journalctl -u rf-homing.service -f
```

### 9. Testing

Run the test suite to verify installation:

```bash
# Run all tests with uv
uv run pytest

# Run with coverage
uv run pytest --cov=backend tests/

# Run specific test file
uv run pytest tests/backend/unit/test_config.py

# Run tests with different Python version
uv run --python 3.12 pytest

# Quick test with uvx
uvx --from . pytest -v
```

### 10. Development Tools

Use development tools with uv:

```bash
# Code formatting
uv run black src/ tests/

# Linting with ruff (faster than flake8)
uv run ruff check src/ tests/
uv run ruff check --fix src/  # Auto-fix issues

# Type checking
uv run mypy src/

# Pre-commit hooks
uv run pre-commit install
uv run pre-commit run --all-files

# Interactive Python with project context
uv run ipython

# Debug with ipdb
uv run python -m ipdb src/backend/main.py
```

## ArduPilot SITL Setup (For MAVLink Testing)

ArduPilot SITL (Software In The Loop) allows testing MAVLink communication without physical hardware. This is essential for developing and testing drone integration features.

### Quick Start

The project includes a helper script for SITL setup:

```bash
# Quick start - installs, builds, and starts SITL
python3 scripts/sitl_setup.py quick

# Or run individual commands:
python3 scripts/sitl_setup.py install  # Clone and install ArduPilot
python3 scripts/sitl_setup.py build    # Build SITL for copter
python3 scripts/sitl_setup.py start    # Start SITL simulation
python3 scripts/sitl_setup.py test     # Test MAVLink connection
python3 scripts/sitl_setup.py stop     # Stop SITL
```

### Manual SITL Installation

If you prefer manual installation:

#### 1. Install Prerequisites

```bash
# Install required packages
sudo apt install -y git python3-pip python3-dev python3-numpy
sudo apt install -y python3-wxgtk4.0 python3-matplotlib python3-lxml
sudo apt install -y python3-pygame python3-opencv python3-yaml

# Install MAVProxy (optional but recommended for debugging)
pip3 install MAVProxy
```

#### 2. Clone ArduPilot

```bash
# Clone ArduPilot repository
cd ~
git clone https://github.com/ArduPilot/ardupilot.git
cd ardupilot

# Update submodules
git submodule update --init --recursive
```

#### 3. Install ArduPilot Prerequisites

```bash
# Run the prerequisites installation script
cd ~/ardupilot
Tools/environment_install/install-prereqs-ubuntu.sh -y

# Reload your shell configuration
. ~/.profile
```

#### 4. Build SITL

```bash
# Configure and build for SITL
cd ~/ardupilot
./waf configure --board sitl
./waf copter  # Build for copter (or plane, rover, sub)
```

#### 5. Run SITL

```bash
# Start SITL with TCP output for PISAD
cd ~/ardupilot/ArduCopter
sim_vehicle.py -v ArduCopter -L -35.363261,149.165230,584,90 \
    --out tcp:127.0.0.1:5760 \
    --out tcp:127.0.0.1:14550 \
    --console --map
```

### Connecting PISAD to SITL

Configure PISAD to connect to SITL by setting the MAVLink device path:

```python
# In your configuration or code:
mavlink_service = MAVLinkService(
    device_path="tcp:127.0.0.1:5760",  # SITL TCP connection
    baud_rate=115200  # Ignored for TCP
)
```

Or set via environment variable:

```bash
export PISAD_MAVLINK_DEVICE="tcp:127.0.0.1:5760"
```

### SITL Connection Modes

The MAVLink service supports both hardware and SITL connections:

- **Hardware (Serial)**: `/dev/ttyACM0` or `/dev/ttyAMA0`
- **SITL (TCP)**: `tcp:127.0.0.1:5760`
- **UDP**: `udp:127.0.0.1:14550` (for MAVProxy bridging)

### Testing MAVLink Connection

```bash
# Test with pymavlink directly
python3 -c "
from pymavlink import mavutil
conn = mavutil.mavlink_connection('tcp:127.0.0.1:5760')
msg = conn.wait_heartbeat(timeout=5)
print(f'Connected to system {msg.get_srcSystem()}' if msg else 'No heartbeat')
"

# Or use the SITL setup script
python3 scripts/sitl_setup.py test
```

### SITL Commands

Common MAVProxy commands for testing (when connected to SITL):

```bash
# Connect MAVProxy to SITL
mavproxy.py --master tcp:127.0.0.1:5760 --out udp:127.0.0.1:14551

# In MAVProxy console:
arm throttle        # Arm the vehicle
mode guided         # Switch to guided mode
takeoff 10          # Takeoff to 10 meters
land                # Land the vehicle
disarm              # Disarm motors
```

### Troubleshooting SITL

#### SITL Won't Start

- Check if port 5760 is already in use: `lsof -i :5760`
- Kill existing SITL processes: `pkill -f sim_vehicle.py`
- Check ArduPilot build: `cd ~/ardupilot && ./waf copter`

#### No Heartbeat from SITL

- Verify SITL is running: `ps aux | grep sim_vehicle`
- Check network connectivity: `nc -zv 127.0.0.1 5760`
- Review SITL console output for errors

#### MAVLink Connection Issues

- Enable debug logging in MAVLink service
- Check firewall settings: `sudo ufw status`
- Verify pymavlink installation: `pip3 show pymavlink`

## Troubleshooting

### RTL-SDR Not Detected

If `rtl_test` doesn't detect your SDR:

1. Check USB connection
2. Verify blacklist is applied: `lsmod | grep dvb`
3. Check permissions: `ls -la /dev/bus/usb/`
4. Add user to plugdev group: `sudo usermod -a -G plugdev $USER`

### Python Import Errors

With uv, imports should work automatically, but if you have issues:

```bash
# Ensure you're using uv run
uv run python -c "import backend.core.config"

# Check Python version
uv run python --version

# Reinstall dependencies
uv sync --reinstall
```

### Permission Denied Errors

For GPIO access or system resources:

```bash
# Add user to gpio group
sudo usermod -a -G gpio pisad

# For development, add your user too
sudo usermod -a -G gpio $USER
```

### Service Won't Start

Check the service logs:

```bash
sudo journalctl -u rf-homing.service -n 50
```

Common issues:

- Wrong file paths in service file
- Missing Python dependencies
- Permission issues with log/config directories

## Performance Tuning

### Raspberry Pi 5 Optimizations

1. **CPU Governor**: Set to performance mode

   ```bash
   echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   ```

2. **GPU Memory Split**: Reduce GPU memory for headless operation

   ```bash
   # Add to /boot/config.txt
   gpu_mem=16
   ```

3. **Disable Unnecessary Services**:
   ```bash
   sudo systemctl disable bluetooth
   sudo systemctl disable avahi-daemon
   ```

### SDR Optimizations

1. **USB Buffer Size**: Increase for better performance

   ```bash
   # Add to /etc/modprobe.d/rtlsdr.conf
   options rtl2832 buffers=16 buffer_size=262144
   ```

2. **Sample Rate**: Adjust based on your needs (lower = less CPU)
   - Edit `config/default.yaml`: `SDR_SAMPLE_RATE`

## Security Considerations

1. **API Key**: Enable for production
   - Set `API_KEY_ENABLED: true` in config
   - Generate secure key: `openssl rand -hex 32`

2. **Firewall**: Configure for production

   ```bash
   sudo ufw allow 22/tcp  # SSH
   sudo ufw allow 8000/tcp  # API
   sudo ufw enable
   ```

3. **File Permissions**: Secure sensitive files
   ```bash
   chmod 600 config/*.yaml
   chmod 700 logs/
   ```

## Next Steps

- Review the [Architecture Documentation](architecture.md)
- Explore the [API Documentation](http://localhost:8000/docs) when running
- Check the [Testing Strategy](architecture/testing-strategy.md)
- Review [Coding Standards](architecture/coding-standards.md)

## Support

For issues or questions:

- Check the [GitHub Issues](https://github.com/yourusername/pisad/issues)
- Review logs in `/home/pisad/projects/pisad/logs/`
- Enable debug mode: `export PISAD_LOG_LEVEL=DEBUG`
