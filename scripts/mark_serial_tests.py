#!/usr/bin/env python3
"""
Identify and mark tests that require serial execution.

This script scans test files to identify tests that:
- Use shared databases
- Write to files
- Use global state
- Have timing dependencies
"""

import re
import sys
from pathlib import Path


def find_serial_tests(test_dir: Path) -> list[tuple[Path, list[str]]]:
    """
    Find tests that need serial execution.

    Returns:
        List of (file_path, [reasons]) tuples
    """
    serial_tests = []

    # Patterns that indicate serial execution needed
    patterns = {
        r"tempfile\.|TemporaryDirectory|NamedTemporaryFile": "temp files",
        r'open\([^,]*,\s*["\']w': "file writes",
        r'\.db["\']|sqlite3\.connect': "database access",
        r"global\s+\w+": "global state",
        r"time\.sleep\(|asyncio\.sleep\(": "timing dependencies",
        r"/tmp/|/var/tmp/": "shared temp directory",
        r"shutil\.rmtree|os\.remove": "file deletion",
        r"subprocess\.": "subprocess calls",
        r"@pytest\.mark\.serial": "already marked serial",
    }

    for test_file in test_dir.rglob("test_*.py"):
        if "backup" in str(test_file):
            continue

        reasons = []
        content = test_file.read_text()

        for pattern, reason in patterns.items():
            if re.search(pattern, content):
                reasons.append(reason)

        if reasons and "already marked serial" not in reasons:
            serial_tests.append((test_file, reasons))

    return serial_tests


def mark_test_serial(file_path: Path) -> bool:
    """
    Add @pytest.mark.serial to test file if not already present.

    Returns:
        True if file was modified
    """
    content = file_path.read_text()

    # Check if already has serial mark at file level
    if "@pytest.mark.serial" in content:
        return False

    # Add import if needed
    if "import pytest" not in content:
        content = "import pytest\n" + content

    # Add file-level mark after imports
    lines = content.split("\n")
    import_end = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            import_end = i + 1
        elif import_end > 0 and line and not line.startswith("#"):
            break

    # Insert the mark
    lines.insert(import_end + 1, "\npytestmark = pytest.mark.serial")

    file_path.write_text("\n".join(lines))
    return True


def main():
    """Main entry point."""
    test_dir = Path("tests")

    if not test_dir.exists():
        print(f"Error: {test_dir} directory not found")
        sys.exit(1)

    print("Scanning for tests that require serial execution...")
    serial_tests = find_serial_tests(test_dir)

    if not serial_tests:
        print("No tests requiring serial execution found.")
        return

    print(f"\nFound {len(serial_tests)} test files requiring serial execution:\n")

    for file_path, reasons in serial_tests:
        relative_path = file_path.relative_to(test_dir)
        print(f"  {relative_path}:")
        for reason in reasons:
            print(f"    - {reason}")

    print("\nMarking tests as serial...")
    modified_count = 0

    for file_path, _ in serial_tests:
        if mark_test_serial(file_path):
            modified_count += 1
            print(f"  ✓ Marked {file_path.name}")

    print(f"\nModified {modified_count} files.")
    print("\nTo run tests with parallel execution:")
    print("  pytest -n 8")
    print("\nTo run only serial tests:")
    print("  pytest -m serial")
    print("\nTo run only parallel-safe tests:")
    print("  pytest -m 'not serial' -n 8")


if __name__ == "__main__":
    main()
