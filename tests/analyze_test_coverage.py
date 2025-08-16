#!/usr/bin/env python3
"""
Sprint 8 - Day 1-2: Comprehensive PRD Test Analysis
Maps every existing test to PRD requirements or marks for deletion
"""

import re
from pathlib import Path

# PRD Functional Requirements
FR_REQUIREMENTS = {
    "FR1": "RF beacon detection 500m range",
    "FR2": "Expanding square search patterns",
    "FR3": "State transitions within 2 seconds",
    "FR4": "RSSI gradient climbing navigation",
    "FR5": "GUIDED and GUIDED_NOGPS modes",
    "FR6": "RSSI computation with EWMA",
    "FR7": "Debounced state transitions",
    "FR8": "Geofence boundaries",
    "FR9": "MAVLink telemetry streaming",
    "FR10": "Auto RTL/LOITER",
    "FR11": "Operator override",
    "FR12": "State logging",
    "FR13": "SDR auto-init",
    "FR14": "Operator homing activation",
    "FR15": "Cease velocity on mode change",
    "FR16": "Disable homing control",
    "FR17": "Auto-disable after signal loss",
}

# PRD Non-Functional Requirements
NFR_REQUIREMENTS = {
    "NFR1": "MAVLink <1% packet loss",
    "NFR2": "Signal processing <100ms",
    "NFR3": "25 min flight endurance",
    "NFR4": "Power <2.5A @ 5V",
    "NFR5": "Temperature -10°C to +45°C",
    "NFR6": "Wind tolerance 15 m/s",
    "NFR7": "False positive <5%",
    "NFR8": "90% homing success",
    "NFR9": "MTBF >10 hours",
    "NFR10": "Deploy <15 minutes",
    "NFR11": "Modular architecture",
    "NFR12": "Deterministic timing",
    "NFR13": "Visual state indication",
}


def analyze_test_file(filepath: Path) -> tuple[set[str], bool]:
    """Analyze a test file for PRD requirement coverage"""
    with open(filepath) as f:
        content = f.read().lower()

    covered_reqs = set()
    is_prd_test = False

    # Check for PRD-critical keywords
    prd_keywords = {
        "signal_processor|rssi|ewma|noise_floor": ["FR1", "FR6", "NFR2"],
        "sdr_service|hackrf|soapy": ["FR1", "FR13"],
        "state_machine|state_transition": ["FR3", "FR7", "FR15"],
        "search_pattern|expanding_square": ["FR2"],
        "homing_algorithm|gradient_climbing": ["FR4"],
        "homing_controller|homing_mode": ["FR14", "FR15", "FR16"],
        "mavlink|telemetry|named_value_float": ["FR9", "NFR1"],
        "safety|interlock|emergency_stop": ["FR10", "FR11", "FR15", "FR16", "FR17"],
        "geofence|boundary": ["FR8"],
        "guided|guided_nogps": ["FR5"],
        "debounce|trigger_threshold|drop_threshold": ["FR7"],
        "performance|latency|async": ["NFR2", "NFR12"],
    }

    for pattern, reqs in prd_keywords.items():
        if re.search(pattern, content):
            covered_reqs.update(reqs)
            is_prd_test = True

    # Check for non-PRD patterns (mark for deletion)
    non_prd_patterns = [
        "analytics",
        "report_generator",
        "recommendations",
        "frontend",
        "dashboard",
        "ui_",
        "component",
        "ci_cd",
        "deployment",
        "docker",
        "mission_replay",
        "test_logger",
        "test_utilities",
        "mock_test",
        "dummy_",
        "example_",
    ]

    for pattern in non_prd_patterns:
        if pattern in str(filepath).lower() or pattern in content:
            is_prd_test = False
            break

    return covered_reqs, is_prd_test


def main():
    """Analyze all test files and generate deletion list"""
    test_dir = Path("/home/pisad/projects/pisad/tests")

    prd_tests = {}
    delete_tests = []
    coverage_map = {}

    # Scan all test files
    for test_file in test_dir.rglob("test_*.py"):
        # Skip backup directories
        if "backup" in str(test_file) or "__pycache__" in str(test_file):
            continue

        rel_path = test_file.relative_to(test_dir)
        covered_reqs, is_prd = analyze_test_file(test_file)

        if is_prd and covered_reqs:
            prd_tests[str(rel_path)] = list(covered_reqs)
            for req in covered_reqs:
                if req not in coverage_map:
                    coverage_map[req] = []
                coverage_map[req].append(str(rel_path))
        else:
            delete_tests.append(str(rel_path))

    # Calculate statistics
    all_reqs = set(FR_REQUIREMENTS.keys()) | set(NFR_REQUIREMENTS.keys())
    covered_reqs = set(coverage_map.keys())
    missing_reqs = all_reqs - covered_reqs

    print("=" * 80)
    print("SPRINT 8 - PRD TEST ANALYSIS REPORT")
    print("=" * 80)
    print("\n📊 STATISTICS:")
    print(f"  Total PRD Requirements: {len(all_reqs)}")
    print(
        f"  Covered Requirements: {len(covered_reqs)} ({len(covered_reqs)/len(all_reqs)*100:.1f}%)"
    )
    print(f"  Missing Requirements: {len(missing_reqs)}")
    print(f"  PRD-Aligned Tests: {len(prd_tests)}")
    print(f"  Tests to Delete: {len(delete_tests)}")

    print("\n❌ MISSING PRD REQUIREMENTS:")
    for req in sorted(missing_reqs):
        desc = FR_REQUIREMENTS.get(req) or NFR_REQUIREMENTS.get(req)
        print(f"  {req}: {desc}")

    print("\n✅ TESTS TO KEEP (PRD-aligned):")
    for test, reqs in sorted(prd_tests.items())[:10]:
        print(f"  {test}: {', '.join(reqs)}")
    if len(prd_tests) > 10:
        print(f"  ... and {len(prd_tests)-10} more")

    print("\n🗑️ TESTS TO DELETE (Non-PRD):")
    for test in sorted(delete_tests)[:10]:
        print(f"  {test}")
    if len(delete_tests) > 10:
        print(f"  ... and {len(delete_tests)-10} more")

    # Generate deletion script
    with open("/home/pisad/projects/pisad/tests/delete_non_prd_tests.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Sprint 8 - Delete non-PRD tests\n")
        f.write("# Generated by analyze_test_coverage.py\n\n")
        for test in delete_tests:
            f.write(f"rm -f tests/{test}\n")

    print("\n📝 ACTIONS REQUIRED:")
    print("  1. Review and execute: tests/delete_non_prd_tests.sh")
    print(f"  2. Create tests for {len(missing_reqs)} missing requirements")
    print("  3. Verify remaining tests actually run")

    return len(covered_reqs), len(missing_reqs), len(prd_tests), len(delete_tests)


if __name__ == "__main__":
    covered, missing, keep, delete = main()
