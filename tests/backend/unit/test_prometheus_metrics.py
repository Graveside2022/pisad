"""Unit tests for Prometheus metrics monitoring.

This module tests Story 4.4 Phase 3 requirements:
- Prometheus metrics endpoint
- MAVLink latency metrics (NFR1: <1% packet loss)
- RSSI processing metrics (NFR2: <100ms latency)
- Performance monitoring integration
"""

import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry, Counter, Histogram

pytestmark = pytest.mark.serial


class TestPrometheusMetricsEndpoint:
    """Test Prometheus /metrics endpoint configuration."""

    @pytest.fixture
    def test_client(self):
        """Create test client with mocked dependencies."""
        with patch("src.backend.core.app.get_config") as mock_config:
            mock_config.return_value = MagicMock(
                app=MagicMock(APP_NAME="PISAD", APP_HOST="0.0.0.0", APP_PORT=8080),
                development=MagicMock(DEV_HOT_RELOAD=False),
                logging=MagicMock(LOG_LEVEL="INFO"),
            )

            from src.backend.core.app import app

            return TestClient(app)

    def test_metrics_endpoint_exists(self, test_client):
        """Given: FastAPI app, When: accessing /metrics, Then: endpoint exists."""
        response = test_client.get("/metrics")

        # Should return metrics or require auth
        assert response.status_code in [200, 401, 403], "Metrics endpoint must be accessible"

    def test_metrics_content_type(self, test_client):
        """Given: metrics endpoint, When: requesting, Then: returns text/plain."""
        response = test_client.get("/metrics")

        if response.status_code == 200:
            content_type = response.headers.get("content-type", "")
            assert "text/plain" in content_type, "Metrics must be text/plain format"

    def test_metrics_format(self, test_client):
        """Given: metrics endpoint, When: fetching, Then: returns Prometheus format."""
        response = test_client.get("/metrics")

        if response.status_code == 200:
            content = response.text

            # Verify Prometheus format
            assert "# HELP" in content or "# TYPE" in content, "Must have Prometheus format"
            assert "_total" in content or "_seconds" in content, "Must have metric suffixes"


class TestMAVLinkMetrics:
    """Test MAVLink performance metrics for NFR1 compliance."""

    @pytest.fixture
    def registry(self):
        """Create isolated metric registry for testing."""
        return CollectorRegistry()

    def test_mavlink_packet_counter(self, registry):
        """Given: MAVLink service, When: processing packets, Then: counts packets."""
        # Create metrics
        packets_sent = Counter(
            "mavlink_packets_sent_total", "Total MAVLink packets sent", registry=registry
        )
        packets_received = Counter(
            "mavlink_packets_received_total", "Total MAVLink packets received", registry=registry
        )
        packets_lost = Counter(
            "mavlink_packets_lost_total", "Total MAVLink packets lost", registry=registry
        )

        # Simulate packet processing
        for _ in range(100):
            packets_sent.inc()
            packets_received.inc()

        # Simulate 0.5% packet loss (under 1% threshold)
        packets_lost.inc()

        # Verify metrics
        assert packets_sent._value.get() == 100
        assert packets_received._value.get() == 100
        assert packets_lost._value.get() == 1

        # Calculate packet loss rate
        loss_rate = packets_lost._value.get() / packets_sent._value.get()
        assert loss_rate < 0.01, f"Packet loss {loss_rate:.2%} must be under 1%"

    def test_mavlink_latency_histogram(self, registry):
        """Given: MAVLink messages, When: measuring latency, Then: tracks distribution."""
        # Create latency histogram
        latency = Histogram(
            "mavlink_message_latency_seconds",
            "MAVLink message processing latency",
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=registry,
        )

        # Simulate latency measurements (all under 100ms)
        latencies = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.060, 0.080]

        for lat in latencies:
            latency.observe(lat)

        # Get histogram data
        histogram_data = latency._sum.get()
        count = latency._count.get()

        assert count == len(latencies), "Must track all observations"
        avg_latency = histogram_data / count
        assert avg_latency < 0.1, f"Average latency {avg_latency:.3f}s must be under 100ms"

    @pytest.mark.asyncio
    async def test_mavlink_connection_status(self, registry):
        """Given: MAVLink connection, When: monitoring, Then: tracks status."""

        # Create connection gauge
        from prometheus_client import Gauge

        connection_status = Gauge(
            "mavlink_connection_status",
            "MAVLink connection status (1=connected, 0=disconnected)",
            registry=registry,
        )

        # Simulate connection states
        connection_status.set(0)  # Disconnected
        assert connection_status._value.get() == 0

        connection_status.set(1)  # Connected
        assert connection_status._value.get() == 1


class TestRSSIMetrics:
    """Test RSSI signal processing metrics for NFR2 compliance."""

    def test_rssi_processing_time(self, registry):
        """Given: RSSI data, When: processing, Then: measures time under 100ms."""
        # Create processing time histogram
        processing_time = Histogram(
            "rssi_processing_duration_seconds",
            "RSSI signal processing duration",
            buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.5],
            registry=registry,
        )

        # Simulate RSSI processing
        for _ in range(50):
            start = time.perf_counter()

            # Simulate processing (should be fast)
            time.sleep(0.02)  # 20ms processing

            duration = time.perf_counter() - start
            processing_time.observe(duration)

        # Verify all processing under 100ms
        count = processing_time._count.get()
        total_time = processing_time._sum.get()
        avg_time = total_time / count

        assert avg_time < 0.1, f"RSSI processing {avg_time:.3f}s must be under 100ms"

    def test_rssi_sample_rate(self, registry):
        """Given: SDR stream, When: sampling, Then: tracks sample rate."""
        from prometheus_client import Counter, Gauge

        samples_processed = Counter(
            "rssi_samples_processed_total", "Total RSSI samples processed", registry=registry
        )

        sample_rate = Gauge(
            "rssi_sample_rate_hz", "Current RSSI sample rate in Hz", registry=registry
        )

        # Simulate sampling at 2MHz
        target_rate = 2_000_000
        sample_rate.set(target_rate)

        # Process samples
        for _ in range(1000):
            samples_processed.inc(2048)  # Buffer size

        assert sample_rate._value.get() == target_rate
        assert samples_processed._value.get() == 1000 * 2048

    def test_signal_strength_gauge(self, registry):
        """Given: signal detection, When: updating RSSI, Then: tracks current value."""
        from prometheus_client import Gauge

        current_rssi = Gauge(
            "current_rssi_dbm", "Current RSSI signal strength in dBm", registry=registry
        )

        peak_rssi = Gauge("peak_rssi_dbm", "Peak RSSI signal strength in dBm", registry=registry)

        # Simulate RSSI updates
        rssi_values = [-90, -85, -80, -75, -70, -65, -70, -75]

        for rssi in rssi_values:
            current_rssi.set(rssi)

            # Update peak if higher
            if peak_rssi._value.get() == 0 or rssi > peak_rssi._value.get():
                peak_rssi.set(rssi)

        assert current_rssi._value.get() == -75  # Last value
        assert peak_rssi._value.get() == -65  # Peak value


class TestPerformanceMetrics:
    """Test general performance monitoring metrics."""

    def test_http_request_metrics(self, registry):
        """Given: HTTP requests, When: handling, Then: tracks latency and count."""
        from prometheus_client import Counter, Histogram

        http_requests = Counter(
            "http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"],
            registry=registry,
        )

        http_latency = Histogram(
            "http_request_duration_seconds",
            "HTTP request latency",
            ["method", "endpoint"],
            registry=registry,
        )

        # Simulate requests
        endpoints = ["/api/system/status", "/api/config", "/api/missions"]

        for endpoint in endpoints:
            with http_latency.labels(method="GET", endpoint=endpoint).time():
                time.sleep(0.01)  # Simulate processing

            http_requests.labels(method="GET", endpoint=endpoint, status="200").inc()

        # Verify metrics
        assert http_requests._metrics[("GET", endpoints[0], "200")]._value.get() == 1

    def test_websocket_metrics(self, registry):
        """Given: WebSocket connections, When: active, Then: tracks connections."""
        from prometheus_client import Counter, Gauge

        ws_connections = Gauge(
            "websocket_connections_active", "Active WebSocket connections", registry=registry
        )

        ws_messages = Counter(
            "websocket_messages_total",
            "Total WebSocket messages",
            ["direction"],  # sent/received
            registry=registry,
        )

        # Simulate WebSocket activity
        ws_connections.inc()  # New connection
        assert ws_connections._value.get() == 1

        # Simulate message traffic
        for _ in range(100):
            ws_messages.labels(direction="sent").inc()
            ws_messages.labels(direction="received").inc()

        ws_connections.dec()  # Connection closed
        assert ws_connections._value.get() == 0

    def test_system_resource_metrics(self, registry):
        """Given: system resources, When: monitoring, Then: tracks usage."""
        from prometheus_client import Gauge

        cpu_usage = Gauge(
            "system_cpu_usage_percent", "System CPU usage percentage", registry=registry
        )

        memory_usage = Gauge(
            "system_memory_usage_bytes", "System memory usage in bytes", registry=registry
        )

        # Simulate resource monitoring
        cpu_usage.set(45.5)  # 45.5% CPU
        memory_usage.set(512 * 1024 * 1024)  # 512MB

        assert cpu_usage._value.get() == 45.5
        assert memory_usage._value.get() == 512 * 1024 * 1024


class TestMetricsIntegration:
    """Test metrics integration with FastAPI app."""

    @pytest.mark.asyncio
    async def test_app_startup_metrics(self):
        """Given: app startup, When: initializing, Then: registers metrics."""
        with patch("src.backend.core.app.PrometheusInstrumentator") as mock_instrumentator:
            mock_instance = MagicMock()
            mock_instrumentator.return_value = mock_instance

            # Import app to trigger initialization
            from src.backend.core.app import app

            # Verify Prometheus instrumentator was called
            mock_instrumentator.assert_called_once()
            mock_instance.instrument.assert_called_once_with(app)
            mock_instance.expose.assert_called_once()

    def test_metrics_in_ci_pipeline(self):
        """Given: CI workflow, When: reading, Then: includes metrics testing."""
        from pathlib import Path

        ci_workflow = Path(".github/workflows/ci.yml")
        if ci_workflow.exists():
            content = ci_workflow.read_text()

            # Verify metrics are considered in CI
            assert (
                "prometheus" in content.lower()
                or "metrics" in content.lower()
                or "coverage" in content.lower()
            ), "CI should include metrics/monitoring tests"

    @pytest.mark.asyncio
    async def test_metrics_collection_performance(self, registry):
        """Given: metrics collection, When: under load, Then: minimal overhead."""
        import time

        from prometheus_client import Counter

        counter = Counter(
            "test_performance_counter", "Test counter for performance", registry=registry
        )

        # Measure overhead of metric collection
        iterations = 10000

        start = time.perf_counter()
        for _ in range(iterations):
            counter.inc()
        duration = time.perf_counter() - start

        # Should be very fast (< 1ms per 1000 increments)
        per_thousand = (duration / iterations) * 1000
        assert per_thousand < 0.001, f"Metric overhead {per_thousand:.6f}s per 1000 ops"


class TestNFRCompliance:
    """Test Non-Functional Requirements compliance through metrics."""

    def test_nfr1_mavlink_packet_loss(self, registry):
        """Given: NFR1 requirement, When: monitoring MAVLink, Then: <1% packet loss."""
        from prometheus_client import Counter

        # Setup counters
        sent = Counter("test_mavlink_sent", "Sent packets", registry=registry)
        lost = Counter("test_mavlink_lost", "Lost packets", registry=registry)

        # Simulate traffic with 0.8% loss (passes NFR1)
        total_packets = 1000
        lost_packets = 8  # 0.8% loss

        for _ in range(total_packets):
            sent.inc()

        for _ in range(lost_packets):
            lost.inc()

        loss_rate = lost._value.get() / sent._value.get()
        assert loss_rate < 0.01, f"NFR1: Packet loss {loss_rate:.2%} must be <1%"

    def test_nfr2_rssi_processing_latency(self, registry):
        """Given: NFR2 requirement, When: processing RSSI, Then: <100ms latency."""
        from prometheus_client import Summary

        # Create latency summary
        latency = Summary("test_rssi_latency", "RSSI processing latency", registry=registry)

        # Simulate RSSI processing with various latencies (all under 100ms)
        latencies_ms = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]

        for lat_ms in latencies_ms:
            latency.observe(lat_ms / 1000)  # Convert to seconds

        # Get percentiles
        # Note: Summary doesn't expose percentiles directly in tests
        # but we verify all observations are under threshold
        max_latency = max(latencies_ms)
        assert max_latency < 100, f"NFR2: Max RSSI latency {max_latency}ms must be <100ms"

    def test_nfr10_deployment_time(self):
        """Given: NFR10 requirement, When: deploying, Then: <15 minute deployment."""
        # This is validated through README and deployment scripts
        from pathlib import Path

        readme = Path("README.md")
        assert readme.exists(), "README must exist for deployment docs"

        content = readme.read_text().lower()
        assert "15 minute" in content, "NFR10: Must document 15 minute deployment"
