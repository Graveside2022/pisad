#!/usr/bin/env python3
"""Script to refactor generic exceptions in MAVLink service."""

import re
from pathlib import Path


def refactor_mavlink_exceptions():
    """Refactor all generic exceptions in MAVLink service."""

    file_path = Path("src/backend/services/mavlink_service.py")
    content = file_path.read_text()

    # Map of context patterns to specific exceptions
    replacements = [
        # Callback errors
        (
            r'except Exception as e:\n(\s+)logger\.error\(f"Error in (\w+) callback: \{e\}"\)',
            r'except CallbackError as e:\n\1logger.error(f"Error in \2 callback: {e}")',
            "CallbackError",
        ),
        # Connection errors
        (
            r'except Exception as e:\n(\s+)logger\.error\(f"(Failed to connect|Connection \w+): \{e\}"\)',
            r'except MAVLinkError as e:\n\1logger.error(f"\2: {e}")',
            "MAVLinkError",
        ),
        # Command sending errors
        (
            r'except Exception as e:\n(\s+)logger\.error\(f"Failed to (send|upload|start|pause|resume|stop|get) (\w+): \{e\}"\)',
            r'except MAVLinkError as e:\n\1logger.error(f"Failed to \2 \3: {e}")',
            "MAVLinkError",
        ),
        # Heartbeat and monitoring errors
        (
            r'except Exception as e:\n(\s+)logger\.error\(f"Error (sending heartbeat|receiving message): \{e\}"\)',
            r'except MAVLinkError as e:\n\1logger.error(f"Error \2: {e}")',
            "MAVLinkError",
        ),
        # Safety check errors
        (
            r'except Exception as e:\n(\s+)logger\.error\(f"Error in safety check callback: \{e\}"\)',
            r'except SafetyInterlockError as e:\n\1logger.error(f"Error in safety check callback: {e}")',
            "SafetyInterlockError",
        ),
        # Generic connection/monitor errors
        (
            r'except Exception as e:\n(\s+)logger\.error\(f"(\w+ \w+) error: \{e\}"\)',
            r'except MAVLinkError as e:\n\1logger.error(f"\2 error: {e}")',
            "MAVLinkError",
        ),
        # Replace raise Exception with raise MAVLinkError
        (r'raise Exception\("([^"]+)"\)', r'raise MAVLinkError("\1")', "MAVLinkError"),
    ]

    # Apply replacements
    modified = content
    for pattern, replacement, exception_type in replacements:
        modified = re.sub(pattern, replacement, modified)

    # Count remaining generic exceptions
    remaining = len(re.findall(r"except Exception", modified))

    # Write back the modified content
    file_path.write_text(modified)

    print("Refactored MAVLink service exceptions")
    print(f"Remaining generic exceptions: {remaining}")

    return remaining == 0


if __name__ == "__main__":
    success = refactor_mavlink_exceptions()
    exit(0 if success else 1)
