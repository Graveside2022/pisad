"""
FastAPI application setup with CORS middleware and static file serving.
"""

import logging
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from prometheus_client import Gauge
from prometheus_fastapi_instrumentator import Instrumentator

from src.backend.core.config import get_config
from src.backend.core.dependencies import get_service_manager

logger = logging.getLogger(__name__)

# Prometheus metric for startup time
STARTUP_TIME_GAUGE = Gauge(
    "pisad_startup_time_seconds", "Time taken for service to start in seconds"
)


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
        detections,
        health,
        search,
        state,
        static,
        system,
        telemetry,
        testing,
    )
    from src.backend.api.routes import (
        config as config_routes,
    )

    app.include_router(system.router, prefix="/api", tags=["system"])
    app.include_router(health.router, prefix="/api", tags=["health"])  # Health endpoints
    app.include_router(detections.router, prefix="/api", tags=["detections"])
    app.include_router(analytics.router, tags=["analytics"])  # Already has /api/analytics prefix
    app.include_router(
        config_routes.router, prefix="/api", tags=["config"]
    )  # Has /config prefix, needs /api
    app.include_router(state.router, tags=["state"])  # Already has /api/state prefix
    app.include_router(telemetry.router, tags=["telemetry"])  # Already has /api/telemetry prefix
    app.include_router(search.router, tags=["search"])  # Already has /api/search prefix
    app.include_router(static.router, prefix="/api", tags=["static"])
    app.include_router(testing.router, tags=["testing"])  # Already has /api/testing prefix

    # Register WebSocket endpoint
    from src.backend.api import websocket

    app.include_router(websocket.router)

    # Setup Prometheus metrics
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics"],
        env_var_name="ENABLE_METRICS",
        inprogress_name="pisad_inprogress",
        inprogress_labels=True,
    )

    # Add custom metrics for PISAD requirements
    @instrumentator.add
    def mavlink_latency(info: Any) -> None:
        """Track MAVLink packet latency (NFR1: <1% packet loss)"""
        from prometheus_client import Histogram

        MAVLINK_LATENCY = Histogram(
            "pisad_mavlink_latency_seconds",
            "MAVLink communication latency in seconds",
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0),
        )

        if info.modified_handler == "/api/telemetry" or info.modified_handler == "/api/state":
            MAVLINK_LATENCY.observe(info.modified_duration)

    @instrumentator.add
    def rssi_processing_time(info: Any) -> None:
        """Track RSSI processing time (NFR2: <100ms latency)"""
        from prometheus_client import Histogram

        RSSI_PROCESSING = Histogram(
            "pisad_rssi_processing_seconds",
            "RSSI computation time in seconds",
            buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0),
        )

        if info.modified_handler == "/api/analytics/rssi" or "signal" in info.modified_handler:
            RSSI_PROCESSING.observe(info.modified_duration)

    # Instrument the app
    instrumentator.instrument(app).expose(app, endpoint="/metrics", tags=["monitoring"])
    logger.info("Prometheus metrics enabled at /metrics endpoint")

    # Mount static files for React frontend AFTER API routes
    # Frontend build directory will be at src/frontend/dist
    frontend_build_path = Path(__file__).parent.parent.parent / "frontend" / "dist"
    if frontend_build_path.exists():
        app.mount("/", StaticFiles(directory=str(frontend_build_path), html=True), name="static")
        logger.info(f"Serving static files from {frontend_build_path}")
    else:
        logger.warning(f"Frontend build directory not found at {frontend_build_path}")

    @app.on_event("startup")
    async def startup_event() -> None:
        """Initialize services on startup."""
        start_time = time.time()

        logger.info(f"Starting {config.app.APP_NAME} v{config.app.APP_VERSION}")
        logger.info(f"Environment: {config.app.APP_ENV}")
        logger.info(f"Listening on {config.app.APP_HOST}:{config.app.APP_PORT}")

        # Initialize all services
        try:
            service_manager = get_service_manager()
            await service_manager.initialize_services()
            logger.info("All services initialized successfully")

            # Calculate and log startup time
            startup_duration = time.time() - start_time
            startup_duration_ms = startup_duration * 1000
            logger.info(f"Service started in {startup_duration_ms:.2f}ms")

            # Update Prometheus metric
            STARTUP_TIME_GAUGE.set(startup_duration)

        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            raise

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
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
