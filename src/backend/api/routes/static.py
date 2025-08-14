"""
Static file serving routes for React frontend.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "PISAD API"}


@router.get("/static/{file_path:path}")
async def serve_static_file(file_path: str) -> FileResponse:
    """
    Serve static files from the frontend build directory.

    Args:
        file_path: Path to the static file

    Returns:
        The requested static file

    Raises:
        HTTPException: If file not found
    """
    # Get the frontend build directory
    frontend_build_path = Path(__file__).parent.parent.parent.parent / "frontend" / "dist"
    file_full_path = frontend_build_path / file_path

    if not file_full_path.exists() or not file_full_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_full_path)
