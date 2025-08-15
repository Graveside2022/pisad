#!/usr/bin/env python3
"""Batch refactor all generic exceptions to specific ones."""

import re
from pathlib import Path


def get_exception_type(context: str) -> str:
    """Determine the appropriate exception type based on context."""

    # Map keywords to exception types
    mappings = {
        # Hardware related
        ("sdr", "hackrf", "device", "hardware"): "SDRError",
        (
            "mavlink",
            "heartbeat",
            "flight",
            "mode",
            "mission",
            "waypoint",
            "velocity",
        ): "MAVLinkError",
        # State machine related
        ("state", "transition", "state_machine"): "StateTransitionError",
        # Config related
        ("config", "profile", "settings", "yaml"): "ConfigurationError",
        # Safety related
        ("safety", "interlock", "emergency"): "SafetyInterlockError",
        # Callback related
        ("callback", "handler", "listener"): "CallbackError",
        # Database related
        ("database", "sqlite", "insert", "update", "delete", "select"): "DatabaseError",
        # Signal processing
        ("signal", "rssi", "noise", "fft", "processing"): "SignalProcessingError",
    }

    context_lower = context.lower()

    for keywords, exception_type in mappings.items():
        if any(keyword in context_lower for keyword in keywords):
            return exception_type

    # Default to PISADException for unknown contexts
    return "PISADException"


def refactor_file(file_path: Path) -> tuple[int, int]:
    """Refactor exceptions in a single file.

    Returns:
        Tuple of (original_count, refactored_count)
    """

    content = file_path.read_text()
    original_count = len(re.findall(r"except Exception", content))

    if original_count == 0:
        return 0, 0

    # Check if exceptions are imported
    if "from src.backend.core.exceptions import" not in content:
        # Add import at the beginning after other imports
        import_line = "from src.backend.core.exceptions import (\n    PISADException, SignalProcessingError, MAVLinkError,\n    StateTransitionError, HardwareError, SDRError,\n    ConfigurationError, SafetyInterlockError, CallbackError,\n    DatabaseError\n)\n"

        # Find the last import line
        import_pattern = r"^(from|import)\s+.*$"
        lines = content.split("\n")
        last_import = 0
        for i, line in enumerate(lines):
            if re.match(import_pattern, line):
                last_import = i

        lines.insert(last_import + 1, "")
        lines.insert(last_import + 2, import_line)
        content = "\n".join(lines)

    # Pattern to match generic exceptions with context
    pattern = r'except Exception as e:\n(\s+)logger\.(error|warning|info)\(f?"([^"]+)"'

    def replace_exception(match):
        """Replace with appropriate exception type."""
        indent = match.group(1)
        level = match.group(2)
        message = match.group(3)

        exception_type = get_exception_type(message)

        return f'except {exception_type} as e:\n{indent}logger.{level}(f"{message}"'

    # Replace all generic exceptions
    modified = re.sub(pattern, replace_exception, content)

    # Handle raise Exception patterns
    raise_pattern = r'raise Exception\(f?"([^"]+)"\)'

    def replace_raise(match):
        """Replace raise Exception with appropriate type."""
        message = match.group(1)
        exception_type = get_exception_type(message)
        return f'raise {exception_type}(f"{message}")'

    modified = re.sub(raise_pattern, replace_raise, modified)

    # Handle specific database exceptions
    modified = re.sub(r"except sqlite3\.\w+ as e:", r"except DatabaseError as e:", modified)

    # Count remaining generic exceptions
    refactored_count = len(re.findall(r"except Exception", modified))

    # Write back if changes were made
    if modified != content:
        file_path.write_text(modified)

    return original_count, refactored_count


def main():
    """Main refactoring function."""

    # Find all Python files with generic exceptions
    backend_path = Path("src/backend")

    files_to_refactor = []
    for file_path in backend_path.rglob("*.py"):
        if file_path.name == "__init__.py":
            continue

        content = file_path.read_text()
        if "except Exception" in content:
            files_to_refactor.append(file_path)

    print(f"Found {len(files_to_refactor)} files with generic exceptions")

    total_original = 0
    total_refactored = 0

    for file_path in files_to_refactor:
        original, refactored = refactor_file(file_path)
        total_original += original
        total_refactored += refactored

        if original > 0:
            print(f"  {file_path.relative_to('src/backend')}: {original} -> {refactored}")

    print(f"\nTotal: {total_original} generic exceptions -> {total_refactored} remaining")
    print(f"Success rate: {100 * (total_original - total_refactored) / total_original:.1f}%")


if __name__ == "__main__":
    main()
