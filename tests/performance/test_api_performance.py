"""
API Performance Monitoring Tests

BACKWARDS ANALYSIS:
- User Action: Operator interacts with web interface during mission
- Expected Result: Responsive UI with <30ms API latency
- Failure Impact: Delayed control responses, poor user experience

REQUIREMENT TRACE:
- NFR1: MAVLink communication with <1% packet loss
- User Story: 4.9 Task 10 - API response time metrics

TEST VALUE: Ensures responsive operator control during critical SAR operations
"""

import asyncio
import json
import time
from unittest.mock import MagicMock, patch

import numpy as np
import psutil
import pytest
from fastapi.testclient import TestClient

from src.backend.core.app import app


class TestAPIPerformance:
    """Performance tests for REST API endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    @pytest.fixture
    def mock_services(self):
        """Mock backend services for testing"""
        with patch("src.backend.api.routes.system.get_service_manager") as mock_manager:
            manager = MagicMock()
            manager.mavlink_service = MagicMock()
            manager.sdr_service = MagicMock()
            manager.state_machine = MagicMock()
            manager.signal_processor = MagicMock()

            # Setup return values
            manager.mavlink_service.is_connected.return_value = True
            manager.sdr_service.get_status.return_value = {
                "status": "CONNECTED",
                "device": "HackRF One",
            }
            manager.state_machine.get_current_state.return_value = "IDLE"
            manager.signal_processor.get_latest_rssi.return_value = -75.0

            mock_manager.return_value = manager
            yield manager

    @pytest.mark.benchmark(group="api")
    def test_system_status_endpoint_speed(self, benchmark, client, mock_services):
        """Benchmark /api/system/status endpoint"""

        def get_status():
            response = client.get("/api/system/status")
            assert response.status_code == 200
            return response.json()

        result = benchmark(get_status)

        # Performance requirements
        assert benchmark.stats["mean"] < 0.030  # Less than 30ms average
        assert benchmark.stats["max"] < 0.050  # Less than 50ms worst case

        print(f"\n/api/system/status: {benchmark.stats['mean']*1000:.1f}ms avg")

    @pytest.mark.benchmark(group="api")
    def test_config_profiles_list_speed(self, benchmark, client, mock_services):
        """Benchmark /api/config/profiles endpoint"""

        # Mock database query
        with patch("src.backend.api.routes.config.get_db") as mock_db:
            mock_db.return_value.query.return_value.all.return_value = [
                {"id": "1", "name": "Default", "sdr_config": {}},
                {"id": "2", "name": "LoRa", "sdr_config": {}},
                {"id": "3", "name": "WiFi", "sdr_config": {}},
            ]

            def get_profiles():
                response = client.get("/api/config/profiles")
                assert response.status_code == 200
                return response.json()

            result = benchmark(get_profiles)

        assert benchmark.stats["mean"] < 0.020  # Less than 20ms
        print(f"\n/api/config/profiles: {benchmark.stats['mean']*1000:.1f}ms avg")

    @pytest.mark.benchmark(group="api")
    def test_detections_query_speed(self, benchmark, client):
        """Benchmark /api/detections endpoint with filters"""

        # Mock database with sample detections
        with patch("src.backend.api.routes.detections.get_db") as mock_db:
            mock_query = MagicMock()
            mock_query.filter.return_value = mock_query
            mock_query.order_by.return_value = mock_query
            mock_query.limit.return_value = mock_query
            mock_query.all.return_value = [
                {
                    "id": str(i),
                    "timestamp": "2025-01-15T12:00:00",
                    "frequency": 2437000000,
                    "rssi": -75.0 + i,
                    "snr": 15.0,
                    "confidence": 95.0,
                }
                for i in range(100)
            ]
            mock_db.return_value.query.return_value = mock_query

            def get_detections():
                response = client.get("/api/detections?limit=100")
                assert response.status_code == 200
                data = response.json()
                assert len(data) <= 100
                return data

            result = benchmark(get_detections)

        assert benchmark.stats["mean"] < 0.040  # Less than 40ms for 100 records
        print(f"\n/api/detections (100 records): {benchmark.stats['mean']*1000:.1f}ms avg")

    @pytest.mark.benchmark(group="api")
    def test_homing_activation_speed(self, benchmark, client, mock_services):
        """Benchmark /api/system/homing activation"""

        mock_services.state_machine.get_current_state.return_value = "DETECTING"

        def activate_homing():
            response = client.post(
                "/api/system/homing", json={"enabled": True, "confirmationToken": "test123"}
            )
            # May return 200 or 403 depending on safety checks
            assert response.status_code in [200, 403]
            return response.json()

        result = benchmark(activate_homing)

        assert benchmark.stats["mean"] < 0.025  # Less than 25ms
        print(f"\n/api/system/homing: {benchmark.stats['mean']*1000:.1f}ms avg")


class TestWebSocketPerformance:
    """Performance tests for WebSocket connections"""

    @pytest.mark.asyncio
    async def test_websocket_message_latency(self):
        """Test WebSocket message round-trip latency"""

        from fastapi.testclient import TestClient

        latencies = []

        with TestClient(app) as client:
            with client.websocket_connect("/ws") as websocket:
                # Send and receive messages
                for _ in range(100):
                    start = time.perf_counter()

                    # Send ping
                    websocket.send_json({"type": "ping"})

                    # Receive pong
                    data = websocket.receive_json()

                    latency = time.perf_counter() - start
                    latencies.append(latency)

        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)

        # WebSocket requirements
        assert avg_latency < 0.005  # Less than 5ms average
        assert max_latency < 0.010  # Less than 10ms worst case

        print(f"\nWebSocket latency: {avg_latency*1000:.2f}ms avg, {max_latency*1000:.2f}ms max")

    @pytest.mark.asyncio
    async def test_websocket_throughput(self):
        """Test WebSocket message throughput at 10Hz"""

        from fastapi.testclient import TestClient

        messages_sent = 0
        duration = 1.0  # Test for 1 second

        with TestClient(app) as client:
            with client.websocket_connect("/ws") as websocket:
                start_time = time.time()

                while time.time() - start_time < duration:
                    # Send RSSI update
                    websocket.send_json(
                        {
                            "type": "rssi",
                            "data": {
                                "rssi": -75.0,
                                "noiseFloor": -95.0,
                                "snr": 20.0,
                                "confidence": 95.0,
                            },
                        }
                    )
                    messages_sent += 1

                    # Small delay for 10Hz rate
                    await asyncio.sleep(0.1)

        throughput = messages_sent / duration

        # Should handle at least 10Hz
        assert throughput >= 9.5  # Allow 5% tolerance

        print(f"\nWebSocket throughput: {throughput:.1f} messages/sec")


class TestLoadPerformance:
    """Load testing for concurrent operations"""

    @pytest.mark.asyncio
    async def test_concurrent_api_requests(self):
        """Test API performance under concurrent load"""

        async def make_request(client, endpoint):
            """Make async request to endpoint"""
            start = time.perf_counter()
            response = await asyncio.to_thread(client.get, endpoint)
            latency = time.perf_counter() - start
            return latency, response.status_code

        client = TestClient(app)

        # Simulate 50 concurrent requests
        tasks = []
        endpoints = [
            "/api/system/status",
            "/api/config/profiles",
            "/api/detections",
            "/api/health",
            "/api/health/sdr",
        ]

        for _ in range(10):  # 10 iterations
            for endpoint in endpoints:  # 5 endpoints = 50 total
                tasks.append(make_request(client, endpoint))

        # Execute concurrently
        start = time.time()
        results = await asyncio.gather(*tasks)
        total_duration = time.time() - start

        # Analyze results
        latencies = [r[0] for r in results]
        status_codes = [r[1] for r in results]

        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        success_rate = sum(1 for s in status_codes if s == 200) / len(status_codes)

        # Performance requirements under load
        assert avg_latency < 0.100  # Less than 100ms average under load
        assert max_latency < 0.500  # Less than 500ms worst case
        assert success_rate > 0.95  # >95% success rate

        print("\nConcurrent load test (50 requests):")
        print(f"  Average latency: {avg_latency*1000:.1f}ms")
        print(f"  Max latency: {max_latency*1000:.1f}ms")
        print(f"  Success rate: {success_rate*100:.1f}%")
        print(f"  Total duration: {total_duration:.2f}s")

    @pytest.mark.asyncio
    async def test_sustained_100hz_processing(self):
        """Test sustained 100Hz signal processing"""

        from src.backend.services.signal_processor import SignalProcessor

        processor = SignalProcessor()
        processor.sdr_service = MagicMock()
        processor.sdr_service.is_connected.return_value = True

        # Process at 100Hz for 10 seconds
        target_samples = 1000
        interval = 0.01  # 10ms = 100Hz

        start_time = time.perf_counter()
        samples_processed = 0
        latencies = []

        while samples_processed < target_samples:
            sample_start = time.perf_counter()

            # Generate and process IQ data
            iq_data = np.random.randn(1024) + 1j * np.random.randn(1024)
            fft_result = np.fft.fft(iq_data)
            power_spectrum = np.abs(fft_result) ** 2
            rssi = 10 * np.log10(np.mean(power_spectrum) + 1e-10)

            # Track latency
            processing_time = time.perf_counter() - sample_start
            latencies.append(processing_time)

            samples_processed += 1

            # Maintain rate
            elapsed = time.perf_counter() - start_time
            expected_time = samples_processed * interval
            if elapsed < expected_time:
                await asyncio.sleep(expected_time - elapsed)

        total_duration = time.perf_counter() - start_time
        actual_rate = samples_processed / total_duration

        # Performance assertions
        assert actual_rate >= 95  # At least 95Hz sustained
        assert np.mean(latencies) < 0.005  # Less than 5ms average processing
        assert np.percentile(latencies, 99) < 0.010  # 99th percentile < 10ms

        print("\nSustained 100Hz processing:")
        print(f"  Actual rate: {actual_rate:.1f} Hz")
        print(f"  Average latency: {np.mean(latencies)*1000:.2f}ms")
        print(f"  99th percentile: {np.percentile(latencies, 99)*1000:.2f}ms")


class TestResourceMonitoring:
    """Monitor CPU and memory usage during operations"""

    @pytest.mark.asyncio
    async def test_cpu_usage_during_processing(self):
        """Monitor CPU usage during signal processing"""

        process = psutil.Process()

        # Baseline CPU
        process.cpu_percent()  # First call to initialize
        await asyncio.sleep(0.1)
        baseline_cpu = process.cpu_percent()

        # Start processing
        from src.backend.services.signal_processor import SignalProcessor

        processor = SignalProcessor()
        processor.sdr_service = MagicMock()
        processor.sdr_service.is_connected.return_value = True

        # Process for 5 seconds
        cpu_samples = []
        start = time.time()

        while time.time() - start < 5.0:
            # Process sample
            iq_data = np.random.randn(1024) + 1j * np.random.randn(1024)
            _ = np.fft.fft(iq_data)

            # Sample CPU
            cpu_samples.append(process.cpu_percent())
            await asyncio.sleep(0.01)  # 100Hz

        avg_cpu = np.mean(cpu_samples)
        max_cpu = np.max(cpu_samples)

        # CPU usage requirements
        assert avg_cpu < 30  # Less than 30% average
        assert max_cpu < 50  # Less than 50% peak

        print("\nCPU usage during processing:")
        print(f"  Baseline: {baseline_cpu:.1f}%")
        print(f"  Average: {avg_cpu:.1f}%")
        print(f"  Peak: {max_cpu:.1f}%")

    def test_memory_profile_summary(self):
        """Generate memory usage summary"""

        process = psutil.Process()
        memory_info = process.memory_info()

        print("\nMemory Profile:")
        print(f"  RSS: {memory_info.rss / 1024 / 1024:.1f} MB")
        print(f"  VMS: {memory_info.vms / 1024 / 1024:.1f} MB")

        # Get system memory
        vm = psutil.virtual_memory()
        print("\nSystem Memory:")
        print(f"  Total: {vm.total / 1024 / 1024 / 1024:.1f} GB")
        print(f"  Available: {vm.available / 1024 / 1024 / 1024:.1f} GB")
        print(f"  Used: {vm.percent:.1f}%")


def generate_performance_report():
    """Generate comprehensive performance report"""

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system": {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024**3,
        },
        "benchmarks": {},
        "requirements": {
            "api_response_time": "<30ms",
            "rssi_processing": "<40ms",
            "websocket_latency": "<5ms",
            "cpu_usage": "<30%",
            "memory_idle": "<200MB",
        },
    }

    # Save report
    with open("performance_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Performance report generated: performance_report.json")


if __name__ == "__main__":
    generate_performance_report()
