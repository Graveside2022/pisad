#!/usr/bin/env python3
"""Generate OpenAPI specification from FastAPI app."""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backend.core.app import create_app


def generate_openapi_spec(output_file: str = "openapi.json"):
    """Generate OpenAPI specification and save to file."""
    app = create_app()

    # Get OpenAPI schema
    openapi_schema = app.openapi()

    # Save to file
    output_path = Path(output_file)
    with open(output_path, "w") as f:
        json.dump(openapi_schema, f, indent=2)

    print(f"OpenAPI specification saved to {output_path}")
    print(f"API Title: {openapi_schema.get('info', {}).get('title')}")
    print(f"API Version: {openapi_schema.get('info', {}).get('version')}")
    print(f"Number of paths: {len(openapi_schema.get('paths', {}))}")

    return output_path


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "openapi.json"
    generate_openapi_spec(output)
