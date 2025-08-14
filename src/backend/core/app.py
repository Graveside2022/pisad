"""
FastAPI application setup with CORS middleware and static file serving.
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.backend.core.config import get_config
from src.backend.core.dependencies import get_service_manager

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    config = get_config()

    app = FastAPI(
        title=config.app.APP_NAME,
        version=config.app.APP_VERSION,
        description="PISAD - Pi-based Signal Analysis & Detection System",
    )

    # Configure CORS middleware
    if config.api.API_CORS_ENABLED:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.api.API_CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.info(f"CORS enabled with origins: {config.api.API_CORS_ORIGINS}")

    # Register API routes FIRST (before static files)
    from src.backend.api.routes import (
        analytics,
        config,
        detections,
        health,
        search,
        state,
        static,
        system,
        telemetry,
        testing,
    )

    app.include_router(system.router, prefix="/api", tags=["system"])
    app.include_router(health.router, prefix="/api", tags=["health"])  # Health endpoints
    app.include_router(detections.router, prefix="/api", tags=["detections"])
    app.include_router(analytics.router, tags=["analytics"])  # Already has /api/analytics prefix
    app.include_router(
        config.router, prefix="/api", tags=["config"]
    )  # Has /config prefix, needs /api
    app.include_router(state.router, tags=["state"])  # Already has /api/state prefix
    app.include_router(telemetry.router, tags=["telemetry"])  # Already has /api/telemetry prefix
    app.include_router(search.router, tags=["search"])  # Already has /api/search prefix
    app.include_router(static.router, prefix="/api", tags=["static"])
    app.include_router(testing.router, tags=["testing"])  # Already has /api/testing prefix

    # Register WebSocket endpoint
    from src.backend.api import websocket

    app.include_router(websocket.router)

    # Mount static files for React frontend AFTER API routes
    # Frontend build directory will be at src/frontend/dist
    frontend_build_path = Path(__file__).parent.parent.parent / "frontend" / "dist"
    if frontend_build_path.exists():
        app.mount("/", StaticFiles(directory=str(frontend_build_path), html=True), name="static")
        logger.info(f"Serving static files from {frontend_build_path}")
    else:
        logger.warning(f"Frontend build directory not found at {frontend_build_path}")

    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup."""
        logger.info(f"Starting {config.app.APP_NAME} v{config.app.APP_VERSION}")
        logger.info(f"Environment: {config.app.APP_ENV}")
        logger.info(f"Listening on {config.app.APP_HOST}:{config.app.APP_PORT}")

        # Initialize all services
        try:
            service_manager = get_service_manager()
            await service_manager.initialize_services()
            logger.info("All services initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            raise

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        logger.info("Shutting down application")

        # Shutdown all services
        try:
            service_manager = get_service_manager()
            await service_manager.shutdown_services()
            logger.info("All services shutdown successfully")
        except Exception as e:
            logger.error(f"Error during service shutdown: {e}")

    return app


# Create application instance
app = create_app()
