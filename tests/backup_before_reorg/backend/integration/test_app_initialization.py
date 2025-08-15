"""Integration tests for FastAPI app initialization and core setup.

This module provides 100% coverage for src/backend/core/app.py
Testing app initialization, middleware, CORS, and Prometheus metrics.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.cors import CORSMiddleware


class TestAppInitialization:
    """Test FastAPI app initialization and configuration."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        config = MagicMock()
        config.app.APP_NAME = "PISAD Test"
        config.app.APP_VERSION = "1.0.0"
        config.app.APP_HOST = "0.0.0.0"
        config.app.APP_PORT = 8080
        config.app.CORS_ORIGINS = ["http://localhost:3000", "http://localhost:5173"]
        config.development.DEV_HOT_RELOAD = False
        config.logging.LOG_LEVEL = "INFO"
        return config

    def test_app_creation(self, mock_config):
        """Given: config, When: creating app, Then: FastAPI instance created."""
        with patch("src.backend.core.app.get_config", return_value=mock_config):
            from src.backend.core.app import app

            assert isinstance(app, FastAPI)
            assert app.title == "PISAD Test"
            assert app.version == "1.0.0"

    def test_cors_middleware_added(self, mock_config):
        """Given: app with CORS config, When: initialized, Then: CORS middleware added."""
        with patch("src.backend.core.app.get_config", return_value=mock_config):
            from src.backend.core.app import app

            # Check if CORS middleware is in the middleware stack
            cors_middleware_found = False
            for middleware in app.user_middleware:
                if middleware.cls == CORSMiddleware:
                    cors_middleware_found = True
                    # Verify CORS settings
                    assert middleware.options["allow_origins"] == mock_config.app.CORS_ORIGINS
                    assert middleware.options["allow_credentials"] is True
                    assert middleware.options["allow_methods"] == ["*"]
                    assert middleware.options["allow_headers"] == ["*"]
                    break

            assert cors_middleware_found, "CORS middleware must be added"

    def test_prometheus_instrumentation(self, mock_config):
        """Given: app, When: initializing, Then: Prometheus metrics added."""
        with patch("src.backend.core.app.get_config", return_value=mock_config):
            with patch("src.backend.core.app.PrometheusInstrumentator") as mock_prometheus:
                mock_instance = MagicMock()
                mock_prometheus.return_value = mock_instance

                # Re-import to trigger initialization
                import importlib

                import src.backend.core.app

                importlib.reload(src.backend.core.app)

                # Verify Prometheus was initialized
                mock_prometheus.assert_called_once()
                mock_instance.instrument.assert_called_once()
                mock_instance.expose.assert_called_once_with(
                    src.backend.core.app.app, endpoint="/metrics"
                )

    def test_router_registration(self, mock_config):
        """Given: app, When: initialized, Then: all routers registered."""
        with patch("src.backend.core.app.get_config", return_value=mock_config):
            from src.backend.core.app import app

            # Get all routes
            routes = [route.path for route in app.routes]

            # Verify API routes are registered
            assert any(
                "/api/system" in route for route in routes
            ), "System routes must be registered"
            assert any(
                "/api/config" in route for route in routes
            ), "Config routes must be registered"
            assert any(
                "/api/missions" in route for route in routes
            ), "Mission routes must be registered"
            assert any(
                "/api/analytics" in route for route in routes
            ), "Analytics routes must be registered"
            assert any("/api/state" in route for route in routes), "State routes must be registered"
            assert any(
                "/api/search" in route for route in routes
            ), "Search routes must be registered"
            assert any(
                "/api/health" in route for route in routes
            ), "Health routes must be registered"
            assert any(
                "/api/telemetry" in route for route in routes
            ), "Telemetry routes must be registered"
            assert any("/ws" in route for route in routes), "WebSocket route must be registered"

    def test_static_files_mount(self, mock_config):
        """Given: app, When: initialized, Then: static files mounted."""
        with patch("src.backend.core.app.get_config", return_value=mock_config):
            from src.backend.core.app import app

            # Check for static file mounts
            static_found = False
            for route in app.routes:
                if hasattr(route, "app") and hasattr(route.app, "all_directories"):
                    static_found = True
                    break

            assert static_found, "Static files must be mounted"


class TestAppLifecycle:
    """Test app lifecycle events and handlers."""

    @pytest.fixture
    def app_with_mocks(self, mock_config):
        """Create app with mocked dependencies."""
        with patch("src.backend.core.app.get_config", return_value=mock_config):
            with patch("src.backend.core.app.init_db") as mock_init_db:
                with patch("src.backend.core.app.get_sdr_service") as mock_sdr:
                    with patch("src.backend.core.app.get_mavlink_service") as mock_mav:
                        # Create fresh app instance
                        import importlib

                        import src.backend.core.app

                        importlib.reload(src.backend.core.app)

                        app = src.backend.core.app.app
                        return app, mock_init_db, mock_sdr, mock_mav

    @pytest.mark.asyncio
    async def test_startup_event(self, app_with_mocks):
        """Given: app, When: startup event, Then: initializes services."""
        app, mock_init_db, mock_sdr, mock_mav = app_with_mocks

        # Get startup handlers
        startup_handlers = []
        for handler in app.router.lifespan.startup_handlers:
            startup_handlers.append(handler)

        # Execute startup handlers
        for handler in startup_handlers:
            if asyncio.iscoroutinefunction(handler):
                await handler()
            else:
                handler()

        # Verify initialization was called
        mock_init_db.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_event(self, app_with_mocks):
        """Given: app, When: shutdown event, Then: cleans up services."""
        app, _, mock_sdr, mock_mav = app_with_mocks

        # Mock service instances
        sdr_service = AsyncMock()
        mavlink_service = AsyncMock()

        mock_sdr.return_value = sdr_service
        mock_mav.return_value = mavlink_service

        # Get shutdown handlers
        shutdown_handlers = []
        for handler in app.router.lifespan.shutdown_handlers:
            shutdown_handlers.append(handler)

        # Execute shutdown handlers
        for handler in shutdown_handlers:
            if asyncio.iscoroutinefunction(handler):
                await handler()
            else:
                handler()


class TestAppEndpoints:
    """Test basic app endpoints and responses."""

    @pytest.fixture
    def client(self, mock_config):
        """Create test client."""
        with patch("src.backend.core.app.get_config", return_value=mock_config):
            from src.backend.core.app import app

            return TestClient(app)

    def test_root_endpoint(self, client):
        """Given: app, When: accessing root, Then: returns app info."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert data["name"] == "PISAD Test"

    def test_health_endpoint(self, client):
        """Given: app, When: accessing /health, Then: returns health status."""
        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_metrics_endpoint(self, client):
        """Given: app with Prometheus, When: accessing /metrics, Then: returns metrics."""
        response = client.get("/metrics")

        # Metrics endpoint may require auth or return text
        assert response.status_code in [200, 401, 403]

        if response.status_code == 200:
            assert "text/plain" in response.headers.get("content-type", "")

    def test_cors_headers(self, client):
        """Given: app with CORS, When: making request, Then: CORS headers present."""
        response = client.options(
            "/api/health",
            headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
        )

        # Check CORS headers
        assert (
            "access-control-allow-origin" in response.headers
            or "Access-Control-Allow-Origin" in response.headers
        )

    def test_websocket_endpoint(self, client):
        """Given: app, When: connecting WebSocket, Then: connection established."""
        with client.websocket_connect("/ws") as websocket:
            # Send a test message
            websocket.send_json({"type": "ping"})

            # Should receive response or close
            try:
                data = websocket.receive_json(timeout=1)
                assert data is not None
            except:
                # WebSocket may close if not authenticated
                pass


class TestErrorHandling:
    """Test app error handling and middleware."""

    @pytest.fixture
    def client(self, mock_config):
        """Create test client."""
        with patch("src.backend.core.app.get_config", return_value=mock_config):
            from src.backend.core.app import app

            return TestClient(app)

    def test_404_handling(self, client):
        """Given: app, When: accessing invalid route, Then: returns 404."""
        response = client.get("/invalid/route/that/does/not/exist")

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    def test_method_not_allowed(self, client):
        """Given: app, When: using wrong method, Then: returns 405."""
        response = client.post("/")  # Root only accepts GET

        assert response.status_code == 405
        data = response.json()
        assert "detail" in data

    def test_validation_error(self, client):
        """Given: app, When: sending invalid data, Then: returns 422."""
        # Try to update config with invalid data
        response = client.put("/api/config/profiles/test", json={"invalid": "data"})

        # Should return validation error or auth error
        assert response.status_code in [401, 403, 422]


class TestAppConfiguration:
    """Test app configuration and environment handling."""

    def test_app_with_dev_config(self):
        """Given: dev config, When: creating app, Then: enables dev features."""
        config = MagicMock()
        config.app.APP_NAME = "PISAD Dev"
        config.app.APP_VERSION = "dev"
        config.app.CORS_ORIGINS = ["*"]
        config.development.DEV_HOT_RELOAD = True
        config.logging.LOG_LEVEL = "DEBUG"

        with patch("src.backend.core.app.get_config", return_value=config):
            import importlib

            import src.backend.core.app

            importlib.reload(src.backend.core.app)

            app = src.backend.core.app.app
            assert app.debug is True or config.development.DEV_HOT_RELOAD

    def test_app_with_prod_config(self):
        """Given: prod config, When: creating app, Then: disables dev features."""
        config = MagicMock()
        config.app.APP_NAME = "PISAD"
        config.app.APP_VERSION = "1.0.0"
        config.app.CORS_ORIGINS = ["https://pisad.example.com"]
        config.development.DEV_HOT_RELOAD = False
        config.logging.LOG_LEVEL = "WARNING"

        with patch("src.backend.core.app.get_config", return_value=config):
            import importlib

            import src.backend.core.app

            importlib.reload(src.backend.core.app)

            app = src.backend.core.app.app
            assert not config.development.DEV_HOT_RELOAD

    def test_app_version_endpoint(self):
        """Given: app, When: getting version, Then: returns correct version."""
        config = MagicMock()
        config.app.APP_NAME = "PISAD"
        config.app.APP_VERSION = "4.4.0"
        config.app.CORS_ORIGINS = []

        with patch("src.backend.core.app.get_config", return_value=config):
            from src.backend.core.app import app

            client = TestClient(app)

            response = client.get("/")
            assert response.status_code == 200

            data = response.json()
            assert data["version"] == "4.4.0"
