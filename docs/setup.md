# PISAD Setup Guide

This guide provides step-by-step instructions for setting up the PISAD (Portable Interferometric Signal Analysis Device) development environment on a Raspberry Pi 5.

## System Requirements

### Hardware
- Raspberry Pi 5 (4GB or 8GB RAM recommended)
- RTL-SDR USB dongles (1 or 2 for interferometry)
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

Install RTL-SDR drivers and libraries:

```bash
# Install RTL-SDR tools
sudo apt install -y rtl-sdr librtlsdr-dev

# Install SoapySDR (SDR abstraction library)
sudo apt install -y soapysdr-tools python3-soapysdr

# Blacklist DVB-T drivers to prevent conflicts
echo 'blacklist dvb_usb_rtl28xxu' | sudo tee /etc/modprobe.d/blacklist-rtl.conf
sudo modprobe -r dvb_usb_rtl28xxu

# Test RTL-SDR detection
rtl_test -t
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