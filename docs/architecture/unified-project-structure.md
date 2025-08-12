# Unified Project Structure

```
rf-homing-sar-drone/
├── .github/                    # CI/CD workflows
│   └── workflows/
│       ├── ci.yaml            # Test on push
│       └── release.yaml       # Build and package
├── src/                       # Main source code
│   ├── backend/              # Python backend
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── app.py        # FastAPI initialization
│   │   │   ├── config.py     # Configuration loader
│   │   │   └── dependencies.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── sdr_service.py
│   │   │   ├── signal_processor.py
│   │   │   ├── mavlink_service.py
│   │   │   ├── state_machine.py
│   │   │   └── homing_controller.py
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes/
│   │   │   │   ├── system.py
│   │   │   │   ├── config.py
│   │   │   │   └── missions.py
│   │   │   ├── websocket.py
│   │   │   └── middleware.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── database.py
│   │   │   └── schemas.py
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── logging.py
│   │   │   └── safety.py
│   │   ├── main.py          # Entry point
│   │   └── requirements.txt
│   └── frontend/            # React frontend
│       ├── src/
│       │   ├── components/
│       │   │   ├── common/
│       │   │   ├── dashboard/
│       │   │   ├── homing/
│       │   │   └── config/
│       │   ├── hooks/
│       │   ├── services/
│       │   ├── contexts/
│       │   ├── types/
│       │   ├── theme/
│       │   ├── App.tsx
│       │   └── main.tsx
│       ├── public/
│       │   └── index.html
│       ├── package.json
│       ├── tsconfig.json
│       └── vite.config.ts
├── shared/                   # Shared types (if needed)
│   └── types.ts
├── config/                   # Configuration files
│   ├── default.yaml         # Default configuration
│   └── profiles/            # Beacon profiles
│       ├── wifi_beacon.yaml
│       ├── lora_beacon.yaml
│       └── custom.yaml
├── scripts/                  # Utility scripts
│   ├── setup.sh             # Initial setup
│   ├── deploy.sh            # Deployment script
│   └── test_sdr.py         # SDR test utility
├── tests/                   # Test suites
│   ├── backend/
│   │   ├── unit/
│   │   ├── integration/
│   │   └── conftest.py
│   ├── frontend/
│   │   ├── components/
│   │   └── setup.ts
│   └── e2e/
│       └── homing_flow.spec.ts
├── deployment/              # Deployment configs
│   ├── systemd/
│   │   └── rf-homing.service
│   └── ansible/
│       └── playbook.yaml
├── docs/                    # Documentation
│   ├── prd.md
│   ├── front-end-spec.md
│   ├── architecture.md
│   ├── setup.md
│   └── api.md
├── .env.example            # Environment template
├── docker-compose.yaml     # Optional containerization
├── pyproject.toml          # Python project config
├── package.json            # Root package.json
└── README.md
```
