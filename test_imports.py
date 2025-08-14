#!/usr/bin/env python3
"""Test all installed dependencies for PISAD project"""

import importlib


def test_import(module_name, description=""):
    """Test if a module can be imported"""
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"✓ {module_name:30} {version:15} {description}")
        return True
    except ImportError as e:
        print(f"✗ {module_name:30} FAILED: {e}")
        return False
    except Exception as e:
        print(f"✗ {module_name:30} ERROR: {e}")
        return False


print("=" * 70)
print("PISAD Dependencies Test")
print("=" * 70)

# Core Python dependencies
print("\n--- Core Dependencies ---")
test_import("numpy", "Numerical computing")
test_import("scipy", "Scientific computing")
test_import("matplotlib", "Plotting library")

# MAVLink/Drone dependencies
print("\n--- MAVLink/Drone Dependencies ---")
test_import("dronekit", "DroneKit API")
test_import("pymavlink", "MAVLink protocol")
test_import("MAVProxy", "MAVProxy ground station")

# SDR dependencies
print("\n--- SDR Dependencies ---")
test_import("pyrtlsdr", "RTL-SDR support")

# Computer Vision
print("\n--- Computer Vision ---")
test_import("cv2", "OpenCV")

# Serial Communication
print("\n--- Serial Communication ---")
test_import("serial", "PySerial")

# Web Framework
print("\n--- Web Framework ---")
test_import("fastapi", "FastAPI framework")
test_import("uvicorn", "ASGI server")
test_import("websockets", "WebSocket support")
test_import("pydantic", "Data validation")

# Database
print("\n--- Database ---")
test_import("sqlalchemy", "SQL toolkit")
test_import("aiosqlite", "Async SQLite")

# Testing
print("\n--- Testing ---")
test_import("pytest", "Testing framework")
test_import("pytest_asyncio", "Async testing")
test_import("pytest_cov", "Coverage plugin")
test_import("hypothesis", "Property-based testing")

# Development Tools
print("\n--- Development Tools ---")
test_import("black", "Code formatter")
test_import("ruff", "Linter")
test_import("mypy", "Type checker")
test_import("ipdb", "Debugger")

print("\n" + "=" * 70)
print("Test complete!")
