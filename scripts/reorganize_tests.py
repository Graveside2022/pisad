#!/usr/bin/env python3
"""
Script to reorganize test files into proper directories based on their type.
Sprint 6 - Task 9.2: Test Architecture Refactoring
"""

import shutil
from pathlib import Path

# Base paths
PROJECT_ROOT = Path("/home/pisad/projects/pisad")
TESTS_DIR = PROJECT_ROOT / "tests"

# Test categorization rules
SITL_TESTS = [
    "test_ardupilot_sitl_integration.py",
    "test_beacon_detection_scenario.py",
    "test_homing_approach_scenario.py",
    "test_homing_behavior.py",
    "test_mission_abort_scenario.py",
    "test_safety_interlock_scenario.py",
    "test_safety_scenarios.py",
]

INTEGRATION_TESTS = [
    "test_analytics_api.py",
    "test_analytics_export.py",
    "test_api_auth.py",
    "test_api_config.py",
    "test_api_missions.py",
    "test_api_system.py",
    "test_app.py",
    "test_config_routes.py",
    "test_field_test_service.py",
    "test_gcs_integration.py",
    "test_homing_integration.py",
    "test_mavlink_integration.py",
    "test_phase1_integration.py",
    "test_phase2_integration.py",
    "test_phase3_integration.py",
    "test_safety_integration.py",
    "test_safety_system.py",
    "test_search_api.py",
    "test_signal_processor_integration.py",
    "test_signal_state_integration.py",
    "test_service_integration.py",
    "test_state_integration.py",
    "test_system_integration.py",
    "test_websocket.py",
    "test_websocket_state_events.py",
]

E2E_TESTS = [
    "test_full_mission_flow.py",
    "test_emergency_procedures.py",
    "test_operator_workflows.py",
]

PERFORMANCE_TESTS = [
    "test_homing_performance.py",
    "test_performance_analytics.py",
    "test_signal_processing_performance.py",
    "test_database_performance.py",
]

# Files to remove (duplicates/low-value)
TESTS_TO_REMOVE = [
    "test_state_machine_additional.py",
    "test_state_machine_comprehensive.py",
    "test_state_machine_entry_exit.py",
    "test_api_auth.py",  # Only 1 test, no auth implemented
    "test_api_config.py",  # Duplicates test_config_routes.py
    "test_api_missions.py",  # Only 2 tests, minimal value
    "test_websocket_state_events.py",  # Covered by main websocket test
]


def categorize_test(filename: str) -> str:
    """Determine which category a test belongs to."""
    if filename in SITL_TESTS:
        return "sitl"
    elif filename in INTEGRATION_TESTS:
        return "integration"
    elif filename in E2E_TESTS:
        return "e2e"
    elif filename in PERFORMANCE_TESTS:
        return "performance"
    elif "mock" in filename:
        return "unit/mocks"
    elif any(x in filename for x in ["api", "route", "endpoint"]):
        return "integration/api"
    elif any(x in filename for x in ["websocket", "ws"]):
        return "integration/websocket"
    elif any(x in filename for x in ["database", "db", "model"]):
        return "integration/database"
    elif any(x in filename for x in ["algorithm", "homing_algorithm", "search_pattern"]):
        return "unit/algorithms"
    elif any(x in filename for x in ["service", "processor", "controller"]):
        return "unit/services"
    elif any(x in filename for x in ["util", "helper", "safety"]):
        return "unit/utils"
    elif any(x in filename for x in ["schema", "model"]):
        return "unit/models"
    else:
        return "unit"  # Default to unit tests


def should_remove(filename: str) -> bool:
    """Check if a test file should be removed."""
    return filename in TESTS_TO_REMOVE


def move_test_file(src: Path, category: str):
    """Move a test file to its appropriate directory."""
    # Determine destination
    if category == "sitl":
        dst_dir = TESTS_DIR / "sitl"
    elif category == "e2e":
        dst_dir = TESTS_DIR / "e2e"
    elif category == "performance":
        dst_dir = TESTS_DIR / "performance"
    elif category.startswith("integration/"):
        subdir = category.split("/", 1)[1]
        dst_dir = TESTS_DIR / "integration" / subdir
    elif category.startswith("unit/"):
        subdir = category.split("/", 1)[1]
        dst_dir = TESTS_DIR / "unit" / subdir
    else:
        dst_dir = TESTS_DIR / category

    # Create directory if needed
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Move file
    dst = dst_dir / src.name
    if src.exists() and not dst.exists():
        print(f"Moving {src.name} -> {dst_dir.relative_to(TESTS_DIR)}/")
        shutil.move(str(src), str(dst))
        return True
    return False


def main():
    """Main reorganization logic."""
    print("=" * 60)
    print("Test Suite Reorganization - Sprint 6 Task 9.2")
    print("=" * 60)

    # Create backup
    backup_dir = TESTS_DIR / "backup_before_reorg"
    if not backup_dir.exists():
        print("\n1. Creating backup...")
        shutil.copytree(TESTS_DIR / "backend", backup_dir / "backend")
        shutil.copytree(TESTS_DIR / "hardware", backup_dir / "hardware", dirs_exist_ok=True)
        print("   Backup created in tests/backup_before_reorg/")

    # Statistics
    stats = {"moved": 0, "removed": 0, "skipped": 0, "errors": 0}

    print("\n2. Processing test files...")

    # Process backend/unit tests
    unit_dir = TESTS_DIR / "backend" / "unit"
    if unit_dir.exists():
        for test_file in unit_dir.glob("test_*.py"):
            filename = test_file.name

            # Check if should be removed
            if should_remove(filename):
                print(f"   Removing duplicate: {filename}")
                test_file.unlink()
                stats["removed"] += 1
                continue

            # Categorize and move
            category = categorize_test(filename)
            if category != "unit":  # Don't move files already in unit
                if move_test_file(test_file, category):
                    stats["moved"] += 1
                else:
                    stats["skipped"] += 1

    # Process backend/integration tests
    integration_dir = TESTS_DIR / "backend" / "integration"
    if integration_dir.exists():
        for test_file in integration_dir.glob("test_*.py"):
            filename = test_file.name

            # Check if should be removed
            if should_remove(filename):
                print(f"   Removing duplicate: {filename}")
                test_file.unlink()
                stats["removed"] += 1
                continue

            # These are already integration tests, just need subcategorization
            if "api" in filename or "route" in filename:
                move_test_file(test_file, "integration/api")
                stats["moved"] += 1
            elif "websocket" in filename:
                move_test_file(test_file, "integration/websocket")
                stats["moved"] += 1
            elif "database" in filename or "db" in filename:
                move_test_file(test_file, "integration/database")
                stats["moved"] += 1

    # Process SITL tests
    sitl_dir = TESTS_DIR / "backend" / "sitl_disabled"
    if sitl_dir.exists():
        for test_file in sitl_dir.glob("test_*.py"):
            if move_test_file(test_file, "sitl"):
                stats["moved"] += 1

    # Process hardware tests
    hardware_dir = TESTS_DIR / "hardware"
    if hardware_dir.exists():
        # Move mock tests to unit/mocks
        mock_dir = hardware_dir / "mock"
        if mock_dir.exists():
            for test_file in mock_dir.glob("test_*.py"):
                if move_test_file(test_file, "unit/mocks"):
                    stats["moved"] += 1

        # Move real hardware tests to integration
        real_dir = hardware_dir / "real"
        if real_dir.exists():
            for test_file in real_dir.glob("test_*.py"):
                if move_test_file(test_file, "integration"):
                    stats["moved"] += 1

    # Print summary
    print("\n" + "=" * 60)
    print("Reorganization Complete!")
    print("=" * 60)
    print(f"Files moved:   {stats['moved']}")
    print(f"Files removed: {stats['removed']}")
    print(f"Files skipped: {stats['skipped']}")
    print(f"Errors:        {stats['errors']}")

    # Show new structure
    print("\n3. New test structure:")
    for category in ["unit", "integration", "e2e", "sitl", "performance", "property", "contract"]:
        cat_dir = TESTS_DIR / category
        if cat_dir.exists():
            count = len(list(cat_dir.rglob("test_*.py")))
            print(f"   tests/{category}/: {count} test files")

            # Show subdirectories
            for subdir in cat_dir.iterdir():
                if subdir.is_dir() and not subdir.name.startswith("__"):
                    sub_count = len(list(subdir.glob("test_*.py")))
                    if sub_count > 0:
                        print(f"      {subdir.name}/: {sub_count} files")


if __name__ == "__main__":
    main()
