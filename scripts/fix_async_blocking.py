#!/usr/bin/env python3
"""
Fix all blocking I/O operations in async functions.

Sprint 6 Day 7 - Task 6: Fix Async Blocking Operations
Authors: Rex & Sherlock
"""

import re
from pathlib import Path

# Files identified with blocking I/O in async functions
FILES_TO_FIX = [
    "src/backend/services/telemetry_recorder.py",
    "src/backend/services/field_test_service.py",
    "src/backend/services/mission_replay_service.py",
    "src/backend/services/report_generator.py",
    "src/backend/api/routes/analytics.py",
    "src/backend/api/routes/system.py",
    "src/backend/api/routes/health.py",
]

# Patterns to detect and fix
BLOCKING_PATTERNS = [
    (r"with open\((.*?)\) as f:", "async file I/O"),
    (r"sqlite3\.connect\((.*?)\)", "async database"),
    (r"f\.read\(\)", "async read"),
    (r"f\.write\((.*?)\)", "async write"),
    (r"json\.dump\((.*?), f", "async JSON write"),
    (r"json\.load\(f\)", "async JSON read"),
    (r"csv\.DictWriter\(f,", "async CSV write"),
    (r"time\.sleep\((.*?)\)", "asyncio.sleep"),
]


def analyze_file(file_path: Path) -> list[tuple[int, str, str]]:
    """Analyze file for blocking operations in async functions."""
    issues = []

    with open(file_path) as f:
        lines = f.readlines()

    in_async_func = False
    async_indent = 0

    for i, line in enumerate(lines):
        # Check if entering async function
        if re.match(r"^(\s*)async def ", line):
            in_async_func = True
            async_indent = len(line) - len(line.lstrip())

        # Check if leaving function
        elif in_async_func and line.strip() and not line[0].isspace():
            in_async_func = False
        elif in_async_func:
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= async_indent and line.strip():
                in_async_func = False

        # Check for blocking patterns in async context
        if in_async_func:
            for pattern, issue_type in BLOCKING_PATTERNS:
                if re.search(pattern, line):
                    issues.append((i + 1, line.strip(), issue_type))

    return issues


def generate_fix_report() -> str:
    """Generate report of all blocking I/O issues."""
    report = []
    report.append("# Async Blocking I/O Analysis Report")
    report.append("## Sprint 6 Day 7 - Task 6")
    report.append("")

    total_issues = 0

    for file_path in FILES_TO_FIX:
        path = Path(file_path)
        if not path.exists():
            continue

        issues = analyze_file(path)
        if issues:
            report.append(f"\n### {file_path}")
            report.append(f"Found {len(issues)} blocking operations:")
            for line_num, code, issue_type in issues:
                report.append(f"- Line {line_num}: {issue_type}")
                report.append("  ```python")
                report.append(f"  {code}")
                report.append("  ```")
            total_issues += len(issues)

    report.append("\n## Summary")
    report.append(f"- Total blocking operations found: {total_issues}")
    report.append(f"- Files affected: {len(FILES_TO_FIX)}")

    return "\n".join(report)


if __name__ == "__main__":
    report = generate_fix_report()
    print(report)

    # Save report
    report_path = Path("docs/async_blocking_analysis.md")
    report_path.write_text(report)
    print(f"\nReport saved to {report_path}")
