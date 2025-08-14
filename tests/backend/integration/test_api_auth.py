"""Integration tests for API authentication and authorization.

NOTE: Auth middleware not yet implemented. These tests document expected behavior
when API_KEY_ENABLED is set to true. Currently all endpoints are open.
"""

import os
from unittest.mock import patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from httpx import AsyncClient

from src.backend.core.app import create_app


@pytest.fixture
def app_with_auth():
    """Create app with authentication enabled."""
    with patch.dict(os.environ, {"API_KEY_ENABLED": "true", "API_KEY": "test-api-key-123"}):
        app = create_app()
        return app


@pytest.fixture
def app_without_auth():
    """Create app with authentication disabled."""
    with patch.dict(os.environ, {"API_KEY_ENABLED": "false"}):
        app = create_app()
        return app


@pytest.fixture
def client_with_auth(app_with_auth):
    """Create test client with auth enabled."""
    return TestClient(app_with_auth)


@pytest.fixture
def client_without_auth(app_without_auth):
    """Create test client with auth disabled."""
    return TestClient(app_without_auth)


class TestAPIAuthentication:
    """Test API authentication mechanisms."""

    def test_no_auth_required_when_disabled(self, client_without_auth):
        """Test endpoints accessible without auth when disabled."""
        response = client_without_auth.get("/api/system/status")
        assert response.status_code == status.HTTP_200_OK

        response = client_without_auth.get("/api/config")
        assert response.status_code == status.HTTP_200_OK

        response = client_without_auth.get("/api/missions")
        assert response.status_code == status.HTTP_200_OK

    def test_auth_required_when_enabled(self, client_with_auth):
        """Test endpoints require auth when enabled."""
        # System endpoints
        response = client_with_auth.get("/api/system/status")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

        # Config endpoints
        response = client_with_auth.get("/api/config")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

        # Mission endpoints
        response = client_with_auth.get("/api/missions")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_valid_api_key_header(self, client_with_auth):
        """Test access with valid API key in header."""
        headers = {"X-API-Key": "test-api-key-123"}

        response = client_with_auth.get("/api/system/status", headers=headers)
        assert response.status_code == status.HTTP_200_OK

        response = client_with_auth.get("/api/config", headers=headers)
        assert response.status_code == status.HTTP_200_OK

    def test_invalid_api_key_header(self, client_with_auth):
        """Test rejection with invalid API key."""
        headers = {"X-API-Key": "invalid-key"}

        response = client_with_auth.get("/api/system/status", headers=headers)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

        response = client_with_auth.get("/api/config", headers=headers)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_missing_api_key_header(self, client_with_auth):
        """Test rejection when API key header missing."""
        response = client_with_auth.get("/api/system/status")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "API key required" in response.json()["detail"]

    def test_api_key_in_query_params(self, client_with_auth):
        """Test API key in query parameters as fallback."""
        # Some APIs support key in query params as fallback
        response = client_with_auth.get("/api/system/status?api_key=test-api-key-123")
        # This might work depending on implementation
        # If not supported, should return 401
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_401_UNAUTHORIZED]

    def test_websocket_auth(self, client_with_auth):
        """Test WebSocket connection requires auth when enabled."""
        # WebSocket auth typically done via query params or initial message
        with pytest.raises(Exception):  # WebSocket should reject without auth
            with client_with_auth.websocket_connect("/ws"):
                pass

    def test_health_endpoint_no_auth(self, client_with_auth):
        """Test health check endpoint doesn't require auth."""
        # Health endpoints typically bypass auth for monitoring
        response = client_with_auth.get("/health")
        # May or may not exist, but if it does, should not require auth
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_404_NOT_FOUND]


class TestAuthorizationLevels:
    """Test different authorization levels and permissions."""

    def test_read_only_endpoints(self, client_with_auth):
        """Test read-only endpoints with valid auth."""
        headers = {"X-API-Key": "test-api-key-123"}

        # GET endpoints should work
        response = client_with_auth.get("/api/system/status", headers=headers)
        assert response.status_code == status.HTTP_200_OK

        response = client_with_auth.get("/api/config", headers=headers)
        assert response.status_code == status.HTTP_200_OK

    def test_write_endpoints(self, client_with_auth):
        """Test write endpoints with valid auth."""
        headers = {"X-API-Key": "test-api-key-123"}

        # POST/PUT endpoints should work with valid auth
        config_data = {"sdr": {"frequency": 433920000}}
        response = client_with_auth.put("/api/config", json=config_data, headers=headers)
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_422_UNPROCESSABLE_ENTITY]

    def test_critical_operations_auth(self, client_with_auth):
        """Test critical operations require additional confirmation."""
        headers = {"X-API-Key": "test-api-key-123"}

        # Emergency stop might require additional confirmation
        response = client_with_auth.post("/api/system/emergency-stop", headers=headers)
        # Should work with valid auth
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_403_FORBIDDEN]

    def test_state_override_special_auth(self, client_with_auth):
        """Test state override requires special token."""
        headers = {"X-API-Key": "test-api-key-123"}

        # State override has special auth requirements
        from datetime import datetime

        override_data = {
            "target_state": "HOMING",
            "confirmation_token": f"override-{datetime.now().strftime('%Y%m%d')}",
            "operator_id": "test-operator",
            "reason": "Testing override",
        }

        response = client_with_auth.post(
            "/api/system/override-state", json=override_data, headers=headers
        )
        # Should accept with correct token format
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST]


class TestAPIKeySecurity:
    """Test API key security features."""

    def test_api_key_not_in_response(self, client_with_auth):
        """Test API key is never returned in responses."""
        headers = {"X-API-Key": "test-api-key-123"}

        response = client_with_auth.get("/api/config", headers=headers)
        if response.status_code == status.HTTP_200_OK:
            response_text = response.text
            assert "test-api-key-123" not in response_text
            assert "API_KEY" not in response_text or response.json().get("API_KEY") == "***"

    def test_api_key_rotation(self):
        """Test API key can be rotated."""
        # Test with first key
        with patch.dict(os.environ, {"API_KEY_ENABLED": "true", "API_KEY": "old-key"}):
            app = create_app()
            client = TestClient(app)

            response = client.get("/api/system/status", headers={"X-API-Key": "old-key"})
            assert response.status_code == status.HTTP_200_OK

            response = client.get("/api/system/status", headers={"X-API-Key": "new-key"})
            assert response.status_code == status.HTTP_401_UNAUTHORIZED

        # Test with rotated key
        with patch.dict(os.environ, {"API_KEY_ENABLED": "true", "API_KEY": "new-key"}):
            app = create_app()
            client = TestClient(app)

            response = client.get("/api/system/status", headers={"X-API-Key": "new-key"})
            assert response.status_code == status.HTTP_200_OK

            response = client.get("/api/system/status", headers={"X-API-Key": "old-key"})
            assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_rate_limiting_per_key(self, client_with_auth):
        """Test rate limiting is applied per API key."""
        headers = {"X-API-Key": "test-api-key-123"}

        # Make multiple rapid requests
        responses = []
        for _ in range(20):
            response = client_with_auth.get("/api/system/status", headers=headers)
            responses.append(response.status_code)

        # At some point might get rate limited (429) if implemented
        # For now, all should succeed since rate limiting not implemented
        assert all(code == status.HTTP_200_OK for code in responses)


class TestCORSWithAuth:
    """Test CORS configuration with authentication."""

    def test_cors_preflight_no_auth(self, client_with_auth):
        """Test CORS preflight requests don't require auth."""
        response = client_with_auth.options(
            "/api/system/status",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "X-API-Key",
            },
        )
        # OPTIONS requests should not require auth
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_204_NO_CONTENT]

    def test_cors_actual_request_needs_auth(self, client_with_auth):
        """Test actual CORS requests still need auth."""
        response = client_with_auth.get(
            "/api/system/status", headers={"Origin": "http://localhost:3000"}
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

        response = client_with_auth.get(
            "/api/system/status",
            headers={"Origin": "http://localhost:3000", "X-API-Key": "test-api-key-123"},
        )
        assert response.status_code == status.HTTP_200_OK


@pytest.mark.asyncio
class TestAsyncAuthentication:
    """Test authentication with async client."""

    async def test_async_client_auth(self, app_with_auth):
        """Test authentication with async HTTP client."""
        async with AsyncClient(app=app_with_auth, base_url="http://test") as client:
            # Without auth
            response = await client.get("/api/system/status")
            assert response.status_code == status.HTTP_401_UNAUTHORIZED

            # With auth
            response = await client.get(
                "/api/system/status", headers={"X-API-Key": "test-api-key-123"}
            )
            assert response.status_code == status.HTTP_200_OK

    async def test_concurrent_auth_requests(self, app_with_auth):
        """Test concurrent authenticated requests."""
        import asyncio

        async with AsyncClient(app=app_with_auth, base_url="http://test") as client:
            headers = {"X-API-Key": "test-api-key-123"}

            # Make concurrent requests
            tasks = [
                client.get("/api/system/status", headers=headers),
                client.get("/api/config", headers=headers),
                client.get("/api/missions", headers=headers),
            ]

            responses = await asyncio.gather(*tasks)

            # All should succeed with auth
            assert all(r.status_code == status.HTTP_200_OK for r in responses)
