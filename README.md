# PISAD - Portable Interferometric Signal Analysis Device

[![CI Pipeline](https://github.com/yourusername/pisad/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/pisad/actions/workflows/ci.yml)
[![Backend Coverage](https://codecov.io/gh/yourusername/pisad/branch/main/graph/badge.svg?flag=backend)](https://codecov.io/gh/yourusername/pisad)
[![Frontend Coverage](https://codecov.io/gh/yourusername/pisad/branch/main/graph/badge.svg?flag=frontend)](https://codecov.io/gh/yourusername/pisad)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![TypeScript](https://img.shields.io/badge/typescript-5.8%2B-blue.svg)](https://www.typescriptlang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

PISAD is a portable RF homing payload system designed for emergency services operations. The system provides real-time radio frequency signal detection, analysis, and localization capabilities using Software Defined Radio (SDR) technology on a Raspberry Pi 5 platform.

### Key Features

- **Real-time RF Signal Detection**: Continuous monitoring and analysis of radio frequency signals
- **Interferometric Processing**: Advanced signal processing for accurate direction finding
- **Portable Design**: Ruggedized system suitable for field operations
- **Web-based Interface**: Real-time visualization and control through responsive web UI
- **Autonomous Operation**: Automatic signal acquisition and tracking
- **Safety Integration**: Built-in safety interlocks for vehicle integration

### System Components

- **Hardware**: Raspberry Pi 5 with RTL-SDR dongles
- **Backend**: Python 3.13.5 with FastAPI for async processing
- **Frontend**: React 18.3.1 with TypeScript for type-safe UI
- **Database**: SQLite for local data persistence
- **Real-time Communication**: WebSocket for streaming RSSI data

### Use Cases

- Emergency beacon localization
- Search and rescue operations
- Radio interference tracking
- Signal strength mapping
- RF environment monitoring

## Getting Started

See [docs/setup.md](docs/setup.md) for detailed installation and configuration instructions.

## Architecture

The system follows a modular architecture with clear separation between:

- Signal processing core
- Web API layer
- User interface
- Data persistence

See [docs/architecture.md](docs/architecture.md) for detailed architecture documentation.

## Development

This project uses:

- Python virtual environments for dependency management
- pytest for testing
- systemd for service management
- GitHub Actions for CI/CD

## License

[License information to be added]

## Contact

[Contact information to be added]
