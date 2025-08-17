# Backend Architecture

## Service Architecture

### Service Organization **UPDATED v3.0**

```
src/backend/
├── core/
│   ├── __init__.py
│   ├── app.py              # FastAPI app with WebSocket & monitoring
│   ├── config.py           # YAML configuration with inheritance
│   ├── config_enhanced.py  # Enhanced config loader
│   ├── dependencies.py     # Dependency injection framework
│   └── exceptions.py       # Custom exception hierarchy
├── services/ (25+ Services)
│   ├── __init__.py
│   ├── beacon_simulator.py      # RF beacon signal simulation
│   ├── command_pipeline.py      # Command processing pipeline
│   ├── config_service.py        # Configuration management
│   ├── field_test_service.py    # Field testing orchestration
│   ├── hardware_detector.py     # Hardware auto-detection **NEW**
│   ├── homing_algorithm.py      # Gradient climbing algorithms
│   ├── homing_controller.py     # Homing state control
│   ├── mavlink_service.py       # MAVLink + SITL communication
│   ├── performance_analytics.py # Performance analysis **NEW**
│   ├── performance_monitor.py   # Real-time monitoring **NEW**
│   ├── recommendations_engine.py# AI recommendations **NEW**
│   ├── report_generator.py      # PDF/JSON report generation
│   ├── safety_manager.py        # Safety interlock management **NEW**
│   ├── sdr_service.py           # SDR hardware interface
│   ├── search_pattern_generator.py # Search pattern algorithms
│   ├── signal_processor.py      # FFT-based RSSI processing
│   ├── signal_processor_integration.py # Signal integration **NEW**
│   ├── signal_state_controller.py # Signal state management **NEW**
│   ├── state_integration.py     # State machine integration **NEW**
│   ├── state_machine.py         # 5-state flight management
│   ├── telemetry_recorder.py    # Telemetry recording
│   └── waypoint_exporter.py     # Mission Planner export
├── hal/ (Hardware Abstraction Layer) **NEW**
│   ├── __init__.py
│   ├── beacon_generator.py      # Test beacon generation
│   ├── hackrf_interface.py      # HackRF One interface
│   ├── mavlink_interface.py     # MAVLink hardware interface
│   ├── mock_hackrf.py          # Hardware mocking
│   └── sitl_interface.py       # SITL testing interface
├── api/
│   ├── __init__.py
│   ├── routes/ (15+ Route Files)
│   │   ├── analytics.py         # Performance analytics **NEW**
│   │   ├── config.py           # Configuration management
│   │   ├── detections.py       # Detection logging
│   │   ├── health.py           # Health monitoring
│   │   ├── search.py           # Search pattern control
│   │   ├── state.py            # State management
│   │   ├── static.py           # Static file serving
│   │   ├── system.py           # System control + safety
│   │   ├── telemetry.py        # Telemetry streaming
│   │   └── testing.py          # Test result endpoints **NEW**
│   ├── websocket.py            # Real-time WebSocket handler
│   └── middleware.py           # CORS and security middleware
├── models/
│   ├── __init__.py
│   ├── database.py             # SQLAlchemy models + metrics
│   └── schemas.py              # Pydantic validation schemas
└── utils/
    ├── __init__.py
    ├── async_io_helpers.py      # AsyncIO utilities **NEW**
    ├── circuit_breaker.py       # Circuit breaker pattern **NEW**
    ├── logging.py              # Structured logging + systemd
    ├── noise_estimator.py      # Signal noise estimation **NEW**
    ├── safety.py               # Multi-layer safety interlocks
    ├── test_logger.py          # Test execution logging **NEW**
    ├── test_metrics.py         # Performance benchmarking **NEW**
    └── yaml_inheritance.py     # YAML config inheritance **NEW**
```

### Service Template

```python

```
