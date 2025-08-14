"""
Main entry point for PISAD backend server.
"""

import logging
import sys
from pathlib import Path

import uvicorn

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.backend.core.config import get_config  # noqa: E402
from src.backend.utils.logging import setup_logging  # noqa: E402


def main() -> None:
    """Run the PISAD backend server."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Get configuration
    config = get_config()

    logger.info(
        f"Starting {config.app.APP_NAME} server on {config.app.APP_HOST}:{config.app.APP_PORT}"
    )

    # Run uvicorn server
    uvicorn.run(
        "src.backend.core.app:app",
        host=config.app.APP_HOST,
        port=config.app.APP_PORT,
        reload=config.development.DEV_HOT_RELOAD,
        log_level=config.logging.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
