# Backend Architecture

## Service Architecture

### Service Organization
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

### Service Template
```python