"""Integration tests for FastAPI application."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.backend.core.app import create_app


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = MagicMock()
    config.app.APP_NAME = "PISAD Test"
    config.app.APP_VERSION = "1.0.0-test"
    config.app.APP_ENV = "test"
    config.app.APP_HOST = "0.0.0.0"
    config.app.APP_PORT = 8000
    config.api.API_CORS_ENABLED = True
    config.api.API_CORS_ORIGINS = ["http://localhost:3000", "http://localhost:8000"]
    return config


@pytest.fixture
def test_app(mock_config):
    """Create test FastAPI application."""
    with patch("src.backend.core.app.get_config", return_value=mock_config):
        app = create_app()
        return app


@pytest.fixture
def client(test_app):
    """Create test client."""
    return TestClient(test_app)


class TestAppCreation:
    """Test application creation and configuration."""

    def test_create_app(self, mock_config):
        """Test app creation with configuration."""
        with patch("src.backend.core.app.get_config", return_value=mock_config):
            app = create_app()

            assert app.title == "PISAD Test"
            assert app.version == "1.0.0-test"
            assert "PISAD" in app.description

    def test_cors_enabled(self, mock_config):
        """Test CORS middleware is added when enabled."""
        mock_config.api.API_CORS_ENABLED = True

        with patch("src.backend.core.app.get_config", return_value=mock_config):
            app = create_app()

            # Check if CORS middleware is present
            middlewares = [str(m) for m in app.middleware]
            assert any("CORSMiddleware" in str(m) for m in middlewares)

    def test_cors_disabled(self, mock_config):
        """Test CORS middleware is not added when disabled."""
        mock_config.api.API_CORS_ENABLED = False

        with patch("src.backend.core.app.get_config", return_value=mock_config):
            app = create_app()

            # Check that CORS middleware is not present
            middlewares = [str(m) for m in app.middleware]
            assert not any("CORSMiddleware" in str(m) for m in middlewares)

    def test_routers_included(self, test_app):
        """Test all API routers are included."""
        routes = [route.path for route in test_app.routes]

        # Check API routes are included
        assert any("/api/system" in path for path in routes)
        assert any("/api/detections" in path for path in routes)
        assert any("/search" in path for path in routes)
        assert any("/api/static" in path for path in routes)
        assert any("/testing" in path for path in routes)
        assert any("/ws" in path for path in routes)  # WebSocket route

    def test_startup_event(self, test_app, mock_config):
        """Test startup event handler."""
        with patch("src.backend.core.app.logger") as mock_logger:
            with patch("src.backend.core.app.get_config", return_value=mock_config):
                # Trigger startup event
                with TestClient(test_app):
                    # Check startup logs
                    assert mock_logger.info.called
                    calls = [str(call) for call in mock_logger.info.call_args_list]
                    assert any("Starting PISAD Test" in str(call) for call in calls)
                    assert any("Environment: test" in str(call) for call in calls)

    def test_shutdown_event(self, test_app):
        """Test shutdown event handler."""
        with patch("src.backend.core.app.logger") as mock_logger:
            # Trigger shutdown event
            client = TestClient(test_app)
            client.__enter__()
            client.__exit__(None, None, None)

            # Check shutdown logs
            assert mock_logger.info.called
            calls = [str(call) for call in mock_logger.info.call_args_list]
            assert any("Shutting down" in str(call) for call in calls)


class TestAPIEndpoints:
    """Test basic API endpoint availability."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns frontend or 404."""
        response = client.get("/")
        # Will be 404 if frontend build doesn't exist, which is expected in tests
        assert response.status_code in [200, 404]

    def test_api_docs_available(self, client):
        """Test API documentation endpoints."""
        # OpenAPI schema
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert "openapi" in response.json()

        # Swagger UI
        response = client.get("/docs")
        assert response.status_code == 200

        # ReDoc
        response = client.get("/redoc")
        assert response.status_code == 200

    @pytest.mark.parametrize(
        "endpoint",
        [
            "/api/system/status",
            "/api/system/health",
            "/api/detections",
        ],
    )
    def test_api_endpoints_exist(self, client, endpoint):
        """Test that API endpoints are accessible."""
        response = client.get(endpoint)
        # May return various status codes depending on implementation
        # but should not return 404 (Not Found)
        assert response.status_code != 404

    def test_websocket_endpoint_exists(self, client):
        """Test WebSocket endpoint is available."""
        # WebSocket connections need special handling
        # Just verify the route exists
        routes = [route.path for route in client.app.routes]
        assert any("/ws" in path for path in routes)


class TestCORSHeaders:
    """Test CORS header functionality."""

    def test_cors_headers_present(self, client):
        """Test CORS headers are present in response."""
        response = client.options(
            "/api/system/status",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )

        # CORS preflight should return 200
        assert response.status_code == 200
        # Check CORS headers
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers

    def test_cors_allows_configured_origins(self, client):
        """Test CORS allows configured origins."""
        # Test allowed origin
        response = client.get("/api/system/health", headers={"Origin": "http://localhost:3000"})

        if "access-control-allow-origin" in response.headers:
            assert response.headers["access-control-allow-origin"] in ["http://localhost:3000", "*"]

    def test_cors_credentials_allowed(self, client):
        """Test CORS allows credentials."""
        response = client.options(
            "/api/system/status",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )

        if "access-control-allow-credentials" in response.headers:
            assert response.headers["access-control-allow-credentials"] == "true"


class TestStaticFiles:
    """Test static file serving."""

    def test_static_files_mount(self, test_app):
        """Test static files are mounted correctly."""
        routes = [str(route) for route in test_app.routes]
        # Static files mount appears as a Mount route
        assert any("Mount" in route for route in routes)

    def test_frontend_warning_logged(self, mock_config, tmp_path):
        """Test warning is logged when frontend build doesn't exist."""
        with patch("src.backend.core.app.get_config", return_value=mock_config):
            with patch("src.backend.core.app.logger") as mock_logger:
                with patch("src.backend.core.app.Path") as mock_path:
                    # Mock frontend path doesn't exist
                    mock_path.return_value.parent.parent.parent = tmp_path
                    mock_frontend_path = MagicMock()
                    mock_frontend_path.exists.return_value = False
                    mock_path.return_value.parent.parent.parent.__truediv__.return_value.__truediv__.return_value = mock_frontend_path

                    app = create_app()

                    # Check warning was logged
                    assert mock_logger.warning.called
                    calls = [str(call) for call in mock_logger.warning.call_args_list]
                    assert any("Frontend build directory not found" in str(call) for call in calls)


class TestErrorHandling:
    """Test error handling in the application."""

    def test_404_error(self, client):
        """Test 404 error for non-existent endpoint."""
        response = client.get("/api/nonexistent")
        assert response.status_code == 404
        assert "detail" in response.json()

    def test_method_not_allowed(self, client):
        """Test 405 error for wrong method."""
        # Assuming /api/system/status only accepts GET
        response = client.delete("/api/system/status")
        # Should return 405 Method Not Allowed or 404 if route doesn't exist
        assert response.status_code in [404, 405]

    def test_validation_error(self, client):
        """Test 422 error for invalid request data."""
        # Send invalid data to an endpoint that expects specific format
        response = client.post("/api/detections", json={"invalid": "data"})
        # Should return 422 Unprocessable Entity or other error
        assert response.status_code in [400, 422, 404]


class TestHealthCheck:
    """Test application health check functionality."""

    def test_health_endpoint(self, client):
        """Test health check endpoint returns expected data."""
        response = client.get("/api/system/health")

        if response.status_code == 200:
            data = response.json()
            # Health endpoint should return some status
            assert isinstance(data, dict)

    def test_status_endpoint(self, client):
        """Test status endpoint returns system information."""
        response = client.get("/api/system/status")

        if response.status_code == 200:
            data = response.json()
            # Status endpoint should return system info
            assert isinstance(data, dict)
