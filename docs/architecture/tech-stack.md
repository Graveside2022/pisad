# Tech Stack

## Technology Stack Table

| Category               | Technology                   | Version         | Purpose                                    | Rationale                                                                      |
| ---------------------- | ---------------------------- | --------------- | ------------------------------------------ | ------------------------------------------------------------------------------ |
| Python Package Manager | uv                           | Latest          | Fast Python package and project management | 10-100x faster than pip, handles Python versions, modern dependency resolution |
| Python Runner          | uvx                          | Latest          | Run Python apps in isolated environments   | Zero-config execution of Python tools and scripts                              |
| Frontend Language      | TypeScript                   | 5.9.2           | Type-safe frontend development             | Prevents runtime errors in safety-critical UI                                  |
| Frontend Framework     | React                        | 18.3.1          | UI component framework                     | Proven reliability, extensive ecosystem                                        |
| UI Component Library   | MUI (Material-UI)            | 7.3.1           | Pre-built components with Grid             | Professional emergency services aesthetic, accessibility, modern Grid system   |
| State Management       | React Context + useReducer   | Built-in        | Local state management                     | Minimal overhead for Pi 5, no Redux complexity                                 |
| Backend Language       | Python                       | 3.11-3.13       | Backend services and signal processing     | AsyncIO support, extensive SDR libraries, managed by uv                        |
| Backend Framework      | FastAPI                      | 0.116.1         | REST API and WebSocket server              | High performance async, automatic OpenAPI docs                                 |
| ASGI Server            | Uvicorn                      | 0.35.0          | ASGI web server for FastAPI                | Production-ready, supports HTTP/1.1 and WebSockets                             |
| HTTP Client            | HTTPX                        | 0.28.1          | Async HTTP client for backend              | Modern async/await support, HTTP/2, connection pooling                         |
| Settings Management    | Pydantic Settings            | 2.10.1          | Configuration and env management           | Type-safe settings with validation, .env file support                          |
| API Style              | REST + WebSocket             | HTTP/1.1 + WS   | API communication                          | REST for commands, WebSocket for real-time data                                |
| Database               | SQLite                       | 3.50.4          | Local data persistence                     | Zero-configuration, file-based for Pi SD card                                  |
| Cache                  | In-Memory (Python dict)      | N/A             | Caching RSSI data                          | Minimal overhead, no Redis needed                                              |
| File Storage           | Local Filesystem             | ext4            | Config and log storage                     | Direct SD card access                                                          |
| Authentication         | API Key (local)              | N/A             | Simple auth for local network              | No external auth needed for field ops                                          |
| Frontend Testing       | Jest + React Testing Library | 30.0.5 / 16.3.0 | Component and unit testing                 | Standard React testing stack                                                   |
| Data Visualization     | Matplotlib                   | 3.10.5          | Signal plotting and analysis               | Industry standard for scientific plotting, RSSI graphs                         |
| PDF Generation         | ReportLab                    | 4.4.3           | Field test reports and documentation       | Generate PDF reports with telemetry data and graphs                            |
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
