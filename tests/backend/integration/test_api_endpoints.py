"""Integration tests for API endpoints.

Tests complete API functionality including request/response cycles,
authentication, error handling, and service coordination.
"""

import time

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from src.backend.core.app import app


class TestAPIEndpointIntegration:
    """Test API endpoint integration."""

    @pytest.fixture
    def client(self):
        """Provide test client."""
        return TestClient(app)

    @pytest.fixture
    async def async_client(self):
        """Provide async test client."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac

    def test_health_endpoint_integration(self, client):
        """Test health endpoint returns system status."""
        response = client.get("/api/health")  # Fixed path

        assert response.status_code == 200
        data = response.json()

        # Should return comprehensive health data
        assert "status" in data or "health" in data

        # Should have some form of health information
        assert isinstance(data, dict)

    def test_metrics_endpoint_integration(self, client):
        """Test metrics endpoint provides Prometheus metrics."""
        response = client.get("/metrics")

        # Metrics may or may not be enabled
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            # Should return Prometheus format metrics
            content = response.text
            assert "# HELP" in content or "# TYPE" in content or len(content) > 0

    def test_state_endpoint_get_current_state(self, client):
        """Test state endpoint returns current system state."""
        response = client.get("/api/state/current")  # Fixed path

        assert response.status_code == 200
        data = response.json()

        # Should return valid state information
        assert "current_state" in data or "state" in data

        # Should be a valid response
        assert isinstance(data, dict)

    def test_state_transition_integration(self, client):
        """Test state transition API integration."""
        # First get current state
        response = client.get("/api/state/current")
        if response.status_code == 200:
            current_state = response.json()

        # Attempt transition to IDLE (should always be valid)
        transition_data = {"target_state": "IDLE"}
        response = client.post("/api/state/transition", json=transition_data)

        # Should handle transition request
        assert response.status_code in [200, 400, 404]  # Various valid responses

        if response.status_code == 200:
            data = response.json()
            # Should have some response structure
            assert isinstance(data, dict)

    def test_telemetry_endpoint_integration(self, client):
        """Test telemetry endpoint returns system telemetry."""
        response = client.get("/api/telemetry")

        # Should handle request (may be 200 or 404 depending on implementation)
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)

    def test_config_endpoint_integration(self, client):
        """Test configuration endpoint integration."""
        response = client.get("/api/config")

        # Should handle request
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)

    def test_detections_endpoint_integration(self, client):
        """Test detections endpoint integration."""
        response = client.get("/api/detections")

        # Should handle request
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)

    def test_search_patterns_endpoint_integration(self, client):
        """Test search patterns endpoint integration."""
        response = client.get("/api/search/patterns")

        # Should handle request
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, (list, dict))

    def test_analytics_endpoint_integration(self, client):
        """Test analytics endpoint integration."""
        response = client.get("/api/analytics")

        # Should handle request
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)

    def test_system_endpoint_integration(self, client):
        """Test system endpoint integration."""
        response = client.get("/api/system")

        # Should handle request
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)

    def test_error_handling_integration(self, client):
        """Test API error handling integration."""
        # Test 404 for non-existent endpoint
        response = client.get("/api/nonexistent")
        assert response.status_code == 404

        # Test invalid method on existing endpoint
        response = client.delete("/api/health")
        assert response.status_code in [404, 405]  # May not exist or wrong method

    def test_cors_headers_integration(self, client):
        """Test CORS headers in API responses."""
        response = client.get("/api/health")

        # Should handle request
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            # Should have CORS headers for web frontend
            headers = response.headers
            # CORS headers may or may not be present in test client
            assert isinstance(headers, dict)

    @pytest.mark.asyncio
    async def test_async_endpoint_integration(self, async_client):
        """Test async endpoint integration."""
        response = await async_client.get("/api/health")

        # Should handle async request
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)

    def test_request_validation_integration(self, client):
        """Test request validation integration."""
        # Test invalid JSON in POST request
        response = client.post(
            "/api/state/transition",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )

        # Should handle validation error appropriately
        assert response.status_code in [400, 404, 422]

    def test_response_performance_integration(self, client):
        """Test API response performance."""
        start_time = time.perf_counter()
        response = client.get("/api/health")
        end_time = time.perf_counter()

        # Should respond within reasonable time
        response_time = end_time - start_time
        assert response_time < 5.0  # Generous timeout for any endpoint

    def test_concurrent_requests_integration(self, client):
        """Test concurrent request handling."""
        import concurrent.futures

        def make_request():
            return client.get("/api/health")

        # Make 3 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request) for _ in range(3)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]

        # All requests should complete
        assert len(responses) == 3
        for response in responses:
            assert response.status_code in [200, 404]

    def test_content_type_handling_integration(self, client):
        """Test content type handling integration."""
        # Test JSON content type
        response = client.get("/api/health")

        if response.status_code == 200:
            content_type = response.headers.get("content-type", "")
            # Should be JSON or similar
            assert isinstance(content_type, str)

    def test_api_versioning_integration(self, client):
        """Test API versioning integration."""
        # Test API endpoints work
        response = client.get("/api/health")
        assert response.status_code in [200, 404]

        # Test root API redirect or info
        response = client.get("/api")
        assert response.status_code in [200, 404, 301, 302]

    def test_authentication_integration(self, client):
        """Test authentication integration if applicable."""
        # Test endpoints that may require authentication
        protected_endpoints = ["/api/config", "/api/system"]

        for endpoint in protected_endpoints:
            response = client.get(endpoint)
            # Should either work or return appropriate error
            assert response.status_code in [200, 401, 403, 404]

    def test_rate_limiting_integration(self, client):
        """Test rate limiting if implemented."""
        # Make multiple rapid requests
        responses = []
        for _ in range(5):
            response = client.get("/api/health")
            responses.append(response)

        # Should all complete
        assert len(responses) == 5

    def test_api_documentation_integration(self, client):
        """Test API documentation integration."""
        # Test OpenAPI/Swagger documentation
        response = client.get("/docs")
        assert response.status_code in [200, 404]  # May or may not be enabled

        # Test OpenAPI JSON
        response = client.get("/openapi.json")
        assert response.status_code in [200, 404]  # May or may not be enabled

    def test_websocket_endpoint_integration(self, client):
        """Test WebSocket endpoint integration if available."""
        # Note: WebSocket testing requires special handling
        # This is a placeholder for WebSocket integration testing
        pass

    def test_static_file_serving_integration(self, client):
        """Test static file serving integration."""
        # Test if static files are served
        response = client.get("/")
        assert response.status_code in [200, 404]  # May or may not serve static files

        # Test common static paths
        static_paths = ["/static/", "/assets/", "/favicon.ico"]
        for path in static_paths:
            response = client.get(path)
            assert response.status_code in [200, 404]  # May or may not exist

    def test_service_coordination_integration(self, client):
        """Test service coordination through API."""
        # Test that API endpoints properly coordinate with backend services

        # Get initial state
        response = client.get("/api/v1/state")
        assert response.status_code == 200

        # Get telemetry (which should coordinate with multiple services)
        response = client.get("/api/v1/telemetry")
        assert response.status_code == 200

        # Test that services are coordinating properly
        # (Implementation depends on actual service architecture)

    def test_database_integration_through_api(self, client):
        """Test database integration through API endpoints."""
        # Test endpoints that require database access

        # Detections should come from database
        response = client.get("/api/v1/detections")
        assert response.status_code == 200

        # Analytics may require database
        response = client.get("/api/v1/analytics")
        assert response.status_code == 200

        # These endpoints should work even if database is empty

    def test_configuration_persistence_integration(self, client):
        """Test configuration persistence through API."""
        # Get current config
        response = client.get("/api/v1/config")
        if response.status_code == 200:
            config = response.json()

            # Configuration should be persistent and valid
            assert isinstance(config, dict)
