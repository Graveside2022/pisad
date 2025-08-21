# PISAD Development Guide

This guide provides comprehensive instructions for setting up a development environment and contributing to the PISAD project.

## Prerequisites

### System Requirements
- **OS**: Ubuntu 22.04 LTS, macOS 12+, or Windows 11 with WSL2
- **Python**: 3.11, 3.12, or 3.13
- **Node.js**: 20.x LTS or higher
- **Git**: 2.30+ with large file support
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 20GB free space

### Hardware (Optional for Full Testing)
- **SDR Device**: HackRF One, RTL-SDR, or compatible
- **Flight Controller**: Pixhawk 4, Cube Orange+, or SITL simulator
- **GPS Module**: For position testing
- **RF Beacon**: 433MHz test transmitter

## Development Environment Setup

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/pisad.git
cd pisad
```

### 2. Install System Dependencies

#### Ubuntu/Debian
```bash
# Update package list
sudo apt update

# Install build tools
sudo apt install -y build-essential git curl wget

# Install Python development tools
sudo apt install -y python3-dev python3-pip python3-venv

# Install SDR libraries (optional for hardware testing)
sudo apt install -y libusb-1.0-0-dev librtlsdr-dev soapysdr-tools

# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

#### macOS
```bash
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.11 node@20 git
brew install --cask docker  # For containerized testing
```

#### Windows (WSL2)
```bash
# In WSL2 Ubuntu environment
sudo apt update
sudo apt install -y build-essential python3-dev python3-pip nodejs npm git
```

### 3. Install UV Package Manager
```bash
# Install UV (ultra-fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Verify installation
uv --version
```

### 4. Set Up Python Environment
```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows

# Install all dependencies including dev tools
uv sync --all-extras

# Verify installation
uv pip list | grep -E "(pytest|black|ruff|mypy)"
```

### 5. Set Up Frontend Environment
```bash
cd src/frontend
npm install
npm run build
cd ../..
```

### 6. Configure Development Environment
```bash
# Copy environment template
cp .env.example .env

# Edit configuration for development
nano .env
```

Example `.env` for development:
```bash
# Development configuration
PISAD_CONFIG_PROFILE=development
PISAD_APP_ENV=development
PISAD_LOG_LEVEL=DEBUG

# Mock hardware for testing without physical devices
USE_MOCK_HARDWARE=true
PISAD_DEV_MOCK_SDR=true

# SITL connection (if using ArduPilot SITL)
PISAD_HARDWARE_MAVLINK_CONNECTION=tcp://localhost:5760

# Database
PISAD_DB_PATH=data/pisad_dev.db

# Enable debug features
PISAD_DEV_HOT_RELOAD=true
PISAD_DEV_DEBUG_MODE=true
```

## Development Workflow

### Code Quality Standards

#### 1. Code Formatting & Linting
```bash
# Format Python code
uv run black src/ tests/

# Lint Python code
uv run ruff check src/ tests/ --fix

# Type checking
uv run mypy src/

# Format/lint frontend
cd src/frontend
npm run lint
npm run format
cd ../..

# Run all quality checks
npx trunk check --all
```

#### 2. Testing
```bash
# Run unit tests
uv run pytest tests/backend/unit/ -v

# Run integration tests
uv run pytest tests/backend/integration/ -v

# Run with coverage
uv run pytest tests/backend/ --cov=src/backend --cov-report=html

# Run frontend tests
cd src/frontend && npm test

# Run performance tests
uv run pytest tests/performance/ -v

# Run PRD compliance tests
uv run pytest tests/prd/ -v
```

#### 3. Pre-commit Hooks
```bash
# Install pre-commit hooks
uv run pre-commit install

# Run hooks manually
uv run pre-commit run --all-files

# Skip hooks (emergency only)
git commit --no-verify
```

### Development Server

#### Backend Development
```bash
# Start development server with hot reload
uv run uvicorn src.backend.core.app:app --reload --host 0.0.0.0 --port 8080

# Or use the main entry point
uv run python -m src.backend.main

# With specific configuration
PISAD_CONFIG_PROFILE=development uv run python -m src.backend.main
```

#### Frontend Development
```bash
cd src/frontend
npm run dev  # Starts Vite dev server on port 3000
```

#### Full Stack Development
```bash
# Terminal 1: Backend
uv run uvicorn src.backend.core.app:app --reload --port 8080

# Terminal 2: Frontend  
cd src/frontend && npm run dev

# Access:
# - Backend API: http://localhost:8080
# - Frontend: http://localhost:3000
# - API Docs: http://localhost:8080/docs
```

### Testing with Hardware

#### SITL Simulation (Recommended)
```bash
# Install ArduPilot SITL
pip install MAVProxy

# Start SITL simulator
sim_vehicle.py -v ArduCopter --console --map

# In another terminal, start PISAD
PISAD_HARDWARE_MAVLINK_CONNECTION=tcp://localhost:5760 uv run python -m src.backend.main
```

#### Real Hardware Testing
```bash
# Connect via serial
PISAD_HARDWARE_MAVLINK_CONNECTION=serial:///dev/ttyACM0:115200 uv run python -m src.backend.main

# Connect via network
PISAD_HARDWARE_MAVLINK_CONNECTION=tcp://192.168.1.100:5760 uv run python -m src.backend.main
```

## Architecture Overview

### Backend Structure
```
src/backend/
├── core/           # Core application setup
│   ├── app.py     # FastAPI application
│   ├── config.py  # Configuration management
│   └── exceptions.py
├── services/       # Business logic services
│   ├── state_machine.py
│   ├── mavlink_service.py
│   ├── signal_processor.py
│   └── homing_controller.py
├── api/           # REST API endpoints
│   └── routes/
├── models/        # Data models and schemas
├── hal/           # Hardware abstraction layer
└── utils/         # Utility functions
```

### Key Services

#### State Machine (`services/state_machine.py`)
- Manages system operational states
- Handles state transitions and validation
- Implements safety interlocks

#### MAVLink Service (`services/mavlink_service.py`)
- Communicates with flight controller
- Streams telemetry data
- Sends velocity commands when authorized

#### Signal Processor (`services/signal_processor.py`)
- Processes SDR data streams
- Computes RSSI and signal metrics
- Detects beacon signals

#### Homing Controller (`services/homing_controller.py`)
- Implements gradient climbing algorithm
- Generates velocity commands
- Manages approach behaviors

### Configuration System
```
config/
├── default.yaml      # Base configuration
├── development.yaml  # Development overrides
├── production.yaml   # Production settings
└── profiles/         # User-defined profiles
```

## Contributing Guidelines

### Branch Strategy
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "feat: add new feature"

# Push and create PR
git push origin feature/your-feature-name
```

### Commit Message Format
```
type(scope): description

feat(api): add new homing endpoint
fix(state): resolve transition race condition
docs(readme): update installation instructions
test(unit): add state machine test coverage
refactor(signal): optimize RSSI computation
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

### Pull Request Process
1. **Create Feature Branch**: Branch from `main`
2. **Implement Changes**: Follow coding standards
3. **Add Tests**: Maintain >80% coverage for new code
4. **Update Documentation**: Update relevant docs
5. **Run Quality Checks**: All checks must pass
6. **Create PR**: Use PR template
7. **Code Review**: Address reviewer feedback
8. **Merge**: Squash and merge to `main`

### Code Review Checklist
- [ ] Code follows style guidelines
- [ ] Tests added for new functionality
- [ ] Documentation updated
- [ ] No breaking changes without migration path
- [ ] Security considerations addressed
- [ ] Performance impact assessed
- [ ] Error handling implemented

## Debugging

### Logging Configuration
```python
# Enable debug logging
import logging
logging.getLogger('src.backend').setLevel(logging.DEBUG)

# Enable specific component logging
logging.getLogger('src.backend.services.state_machine').setLevel(logging.DEBUG)
```

### Common Debug Commands
```bash
# View real-time logs
uv run python -m src.backend.main | tee debug.log

# Check service health
curl http://localhost:8080/api/health

# View system status
curl http://localhost:8080/api/system/status

# Monitor state machine
curl http://localhost:8080/api/state

# Check telemetry
curl http://localhost:8080/api/telemetry
```

### Performance Profiling
```bash
# Memory profiling
uv run python -m memory_profiler src/backend/main.py

# CPU profiling
uv run py-spy top --pid $(pgrep -f "python.*main.py")

# Line-by-line profiling
uv run python -m line_profiler -v src/backend/services/signal_processor.py
```

### Testing Utilities
```bash
# Run specific test with output
uv run pytest tests/backend/unit/test_state_machine.py::TestStateMachine::test_idle_to_searching -v -s

# Test with debugging
uv run pytest --pdb tests/backend/unit/test_state_machine.py

# Run tests matching pattern
uv run pytest -k "test_homing" -v

# Generate coverage report
uv run pytest --cov=src/backend --cov-report=html tests/backend/
```

## IDE Configuration

### VS Code Settings
```json
{
  "python.defaultInterpreterPath": "./.venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests/backend"],
  "files.exclude": {
    "**/__pycache__": true,
    "**/.pytest_cache": true,
    "**/htmlcov": true
  }
}
```

### PyCharm Configuration
1. **Interpreter**: Set to `.venv/bin/python`
2. **Test Runner**: Configure pytest
3. **Code Style**: Import black and ruff configurations
4. **Run Configurations**: Add backend/frontend run configs

## Common Issues & Solutions

### Import Errors
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH=/path/to/pisad/src:$PYTHONPATH

# Or use uv run for all Python commands
uv run python -m src.backend.main
```

### Test Failures
```bash
# Clear pytest cache
rm -rf .pytest_cache

# Regenerate coverage data
rm -f .coverage coverage.xml

# Run tests with fresh environment
uv sync --all-extras
uv run pytest tests/backend/unit/ -v
```

### Performance Issues
```bash
# Check resource usage
htop

# Monitor async tasks
uv run python -c "
import asyncio
print('Event loop debug mode:', asyncio.get_event_loop().get_debug())
"

# Profile startup time
time uv run python -m src.backend.main --check-config
```

### Hardware Connection Issues
```bash
# Check USB devices
lsusb

# Test SDR connection
rtl_test -t

# Check serial permissions
ls -la /dev/ttyACM*
sudo usermod -a -G dialout $USER
```

## Resources

### Documentation
- [Architecture Documentation](docs/architecture/)
- [API Reference](docs/api/)
- [Testing Guide](docs/testing/)
- [Deployment Guide](docs/deployment/)

### External Dependencies
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [ArduPilot MAVLink](https://ardupilot.org/dev/docs/mavlink-basics.html)
- [SoapySDR Documentation](https://github.com/pothosware/SoapySDR/wiki)
- [pytest Documentation](https://docs.pytest.org/)

### Community
- GitHub Issues: Report bugs and request features
- Discussions: Ask questions and share ideas
- Wiki: Additional documentation and examples

---

For questions about development setup or contribution process, please create an issue or reach out to the maintainers.