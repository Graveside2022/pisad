#!/usr/bin/env python3
"""
Sprint 8 - Refined PRD Test Analysis
More accurate mapping of tests to PRD requirements
"""

import re
from pathlib import Path

# Key PRD mappings with specific patterns
PRD_TEST_PATTERNS = {
    # Safety-critical tests (multiple FR/NFR requirements)
    "hara_coverage": ["FR10", "FR11", "FR14", "FR15", "FR16", "FR17", "NFR12"],
    "safety_integration": ["FR10", "FR11", "FR14", "FR15", "FR16", "FR17", "NFR12"],
    "safety_system": ["FR10", "FR11", "FR15", "FR16", "NFR12"],
    "emergency_stop": ["FR11", "FR16", "NFR12"],
    # Signal processing (FR1, FR6, NFR2)
    "signal_processor": ["FR1", "FR6", "NFR2"],
    "sdr_service": ["FR1", "FR13", "NFR2"],
    "rssi": ["FR6", "NFR2"],
    "ewma": ["FR6"],
    "noise_floor|noise_estimator": ["FR6"],
    # State machine (FR3, FR7, FR15)
    "state_machine|state_transition": ["FR3", "FR7", "FR15"],
    "state_persistence": ["FR12"],
    "debounce": ["FR7"],
    # Navigation (FR2, FR4)
    "search_pattern": ["FR2"],
    "homing_algorithm|gradient_climbing": ["FR4"],
    "homing_controller": ["FR4", "FR14", "FR15", "FR16"],
    "waypoint": ["FR2"],
    # MAVLink (FR9, NFR1)
    "mavlink": ["FR9", "NFR1"],
    "telemetry": ["FR9"],
    # Geofence (FR8)
    "geofence|boundary": ["FR8"],
    # Flight modes (FR5)
    "guided|guided_nogps": ["FR5"],
    # Performance (NFR2, NFR12)
    "async|asyncio": ["NFR12"],
    "performance|latency|benchmark": ["NFR2", "NFR12"],
    "circuit_breaker": ["NFR12"],
    "memory_bounded": ["NFR12"],
    # SITL tests (integration)
    "sitl": ["FR9", "FR10", "FR11", "NFR1"],
    "beacon_detection": ["FR1", "FR6", "NFR2"],
    "homing_approach": ["FR4"],
    "mission_abort": ["FR10", "FR11"],
}

# Tests that are definitely NOT in PRD scope
NON_PRD_PATTERNS = [
    "analytics",
    "report_generator",
    "recommendations",
    "ci_cd",
    "deployment",
    "flaky_detector",
    "test_logger",
    "coverage_boost",
    "phase1_integration",  # This is about test phases, not PRD phases
    "phase2_coverage",
    "test_main",  # Meta-testing
    "mock_test_",
    "dummy_",
    "example_",
]


def analyze_file_detailed(filepath: Path) -> tuple[set[str], bool, str]:
    """Detailed analysis of test file"""
    filename = filepath.name
    content = filepath.read_text().lower() if filepath.exists() else ""

    covered_reqs = set()
    is_prd = False
    reason = ""

    # Check against NON-PRD patterns first
    for pattern in NON_PRD_PATTERNS:
        if pattern in filename.lower() or pattern in content:
            return set(), False, f"Non-PRD: {pattern}"

    # Check for PRD patterns
    for pattern, reqs in PRD_TEST_PATTERNS.items():
        if re.search(pattern, filename.lower()) or re.search(pattern, content):
            covered_reqs.update(reqs)
            is_prd = True
            reason = f"PRD: {pattern}"

    # Special cases - keep these even if not explicitly mapped
    keep_patterns = [
        "integration",  # Integration tests often validate multiple PRD requirements
        "hardware",  # Hardware tests validate real PRD requirements
        "critical",  # Critical path tests
        "validation",  # Validation tests
    ]

    for pattern in keep_patterns:
        if pattern in str(filepath).lower():
            is_prd = True
            if not reason:
                reason = f"Keep: {pattern}"

    return covered_reqs, is_prd, reason


def main():
    """Refined analysis of test coverage"""
    test_dir = Path("/home/pisad/projects/pisad/tests")

    keep_tests = {}
    delete_tests = {}
    coverage_map = {}

    for test_file in test_dir.rglob("test_*.py"):
        if "backup" in str(test_file) or "__pycache__" in str(test_file):
            continue

        rel_path = test_file.relative_to(test_dir)
        covered_reqs, is_prd, reason = analyze_file_detailed(test_file)

        if is_prd:
            keep_tests[str(rel_path)] = {"reqs": list(covered_reqs), "reason": reason}
            for req in covered_reqs:
                if req not in coverage_map:
                    coverage_map[req] = []
                coverage_map[req].append(str(rel_path))
        else:
            delete_tests[str(rel_path)] = reason

    # Generate refined deletion script
    with open("/home/pisad/projects/pisad/tests/delete_non_prd_refined.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Sprint 8 - Refined deletion of non-PRD tests\n")
        f.write("# SAFETY: Keep HARA tests, integration tests, hardware tests\n\n")

        for test, reason in sorted(delete_tests.items()):
            # Double-check we're not deleting safety-critical tests
            if "safety" not in test and "hara" not in test and "critical" not in test:
                f.write(f"# {reason}\n")
                f.write(f"rm -f tests/{test}\n")

    # Calculate statistics
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
    all_reqs = set(FR_REQUIREMENTS.keys()) | set(NFR_REQUIREMENTS.keys())
    covered_reqs = set(coverage_map.keys())
    missing_reqs = all_reqs - covered_reqs

    print("=" * 80)
    print("REFINED PRD TEST ANALYSIS")
    print("=" * 80)
    print("\n📊 STATISTICS:")
    print(f"  Total PRD Requirements: {len(all_reqs)}")
    print(
        f"  Covered Requirements: {len(covered_reqs)} ({len(covered_reqs)/len(all_reqs)*100:.1f}%)"
    )
    print(f"  Tests to Keep: {len(keep_tests)}")
    print(f"  Tests to Delete: {len(delete_tests)}")

    print("\n✅ KEY TESTS TO KEEP:")
    important_tests = [k for k in keep_tests if "safety" in k or "hara" in k or "critical" in k]
    for test in important_tests[:5]:
        info = keep_tests[test]
        print(f"  {test}: {info['reason']}")

    print("\n🗑️ DEFINITELY DELETE (Non-PRD):")
    definite_deletes = [
        k for k, v in delete_tests.items() if "analytics" in v or "report" in v or "ci_cd" in v
    ]
    for test in definite_deletes[:10]:
        print(f"  {test}: {delete_tests[test]}")

    print("\n❌ MISSING REQUIREMENTS:")
    for req in sorted(missing_reqs)[:10]:
        print(f"  {req}")

    return covered_reqs, missing_reqs, keep_tests, delete_tests


if __name__ == "__main__":
    covered, missing, keep, delete = main()
