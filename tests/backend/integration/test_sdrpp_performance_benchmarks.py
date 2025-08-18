"""
Performance Benchmarking Tests for SDR++ TCP Communication

Tests validate PRD-NFR2 requirement: <100ms signal processing latency.
Using pytest-benchmark for accurate performance measurement with statistical analysis.

PRD References:
- NFR2: Signal processing latency (<100ms per computation cycle)
- NFR8: 90% successful homing rate (success rate requirement)
- FR9: Enhanced telemetry streaming with dual-SDR coordination

Test Coverage:
- Round-trip TCP message latency measurement
- RSSI request/response performance validation
- JSON message serialization/deserialization performance  
- Multi-client concurrent performance impact
- Network latency vs processing latency separation

Following TDD methodology - RED phase: Writing failing tests first.
"""

import asyncio
import json
import socket
import time
from unittest.mock import MagicMock
import pytest
from src.backend.services.sdrpp_bridge_service import SDRPPBridgeService
from src.backend.services.signal_processor import SignalProcessor


class TestSDRPPPerformanceBenchmarks:
    """Performance benchmarking tests for TCP communication latency validation."""

    @pytest.fixture
    def unused_tcp_port(self):
        """Provide unused TCP port for testing."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    @pytest.fixture
    async def sdrpp_service_with_real_processor(self, unused_tcp_port):
        """Provide SDRPPBridgeService with realistic signal processor mock."""
        service = SDRPPBridgeService()
        service.port = unused_tcp_port
        
        # Create realistic signal processor mock with authentic methods only
        mock_processor = MagicMock(spec=SignalProcessor)
        mock_processor.get_current_rssi.return_value = -55.2
        mock_processor.get_current_snr.return_value = 12.5  # >12dB SNR threshold
        service.set_signal_processor(mock_processor)
        
        yield service
        
        if service.running:
            await service.stop()

    @pytest.mark.asyncio
    async def test_rssi_request_response_latency_benchmark(self, sdrpp_service_with_real_processor):
        """RED: Benchmark RSSI request/response round-trip latency - must be <100ms."""
        service = sdrpp_service_with_real_processor
        await service.start()
        
        async def rssi_request_response():
            """Measure single RSSI request/response cycle."""
            # Connect client
            reader, writer = await asyncio.open_connection(service.host, service.port)
            
            # Send RSSI request
            request = {
                "type": "rssi_update",
                "timestamp": "2025-08-18T23:59:59Z",
                "data": {},
                "sequence": 1
            }
            
            start_time = time.perf_counter()
            
            message_bytes = json.dumps(request).encode() + b'\n'
            writer.write(message_bytes)
            await writer.drain()
            
            # Read response
            response_data = await reader.readline()
            response = json.loads(response_data.decode().strip())
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            
            # Cleanup
            writer.close()
            await writer.wait_closed()
            
            # Verify valid response
            assert response["type"] == "rssi_update"
            assert "data" in response
            assert "rssi" in response["data"]
            
            return latency_ms
        
        # Benchmark the operation using async approach
        async def benchmark_wrapper():
            return await rssi_request_response()
        
        # Run benchmark and measure latency manually since pytest-benchmark has async issues
        latencies = []
        for _ in range(10):
            latency = await benchmark_wrapper()
            latencies.append(latency)
        
        result = sum(latencies) / len(latencies)  # Average latency
        
        # PRD-NFR2 requirement: <100ms latency
        assert result < 100.0, f"Latency {result:.2f}ms exceeds 100ms requirement"

    @pytest.mark.asyncio 
    async def test_json_message_processing_performance(self, benchmark, sdrpp_service_with_real_processor):
        """RED: Benchmark JSON message parsing and handling performance."""
        service = sdrpp_service_with_real_processor
        
        def json_processing_benchmark():
            """Measure JSON message parsing performance."""
            # Test realistic message payload
            test_message = {
                "type": "rssi_update",
                "timestamp": "2025-08-18T23:59:59Z",
                "data": {
                    "rssi": -55.2,
                    "snr": 12.5,
                    "confidence": 0.95
                },
                "sequence": 12345
            }
            
            start_time = time.perf_counter()
            
            # Serialize to JSON
            json_str = json.dumps(test_message)
            
            # Parse message (using service's method)
            parsed = service._parse_message(json_str)
            
            end_time = time.perf_counter()
            processing_time_ms = (end_time - start_time) * 1000
            
            # Verify parsing worked
            assert parsed is not None
            assert parsed["type"] == "rssi_update"
            
            return processing_time_ms
        
        # Benchmark JSON processing - should be very fast (<1ms)
        result = benchmark.pedantic(json_processing_benchmark, rounds=100, iterations=10)
        
        # JSON processing should be negligible compared to network latency
        assert result < 1.0, f"JSON processing {result:.2f}ms too slow for real-time operation"

    @pytest.mark.asyncio
    async def test_concurrent_client_performance_impact(self, benchmark, sdrpp_service_with_real_processor):
        """RED: Test performance impact of multiple concurrent clients."""
        service = sdrpp_service_with_real_processor
        await service.start()
        
        async def concurrent_clients_benchmark():
            """Measure performance with multiple concurrent clients."""
            num_clients = 5
            clients = []
            
            # Connect multiple clients
            for i in range(num_clients):
                reader, writer = await asyncio.open_connection(service.host, service.port)
                clients.append((reader, writer))
            
            start_time = time.perf_counter()
            
            # Send requests from all clients simultaneously
            tasks = []
            for i, (reader, writer) in enumerate(clients):
                async def client_request(r, w, client_id):
                    request = {
                        "type": "rssi_update",
                        "timestamp": f"2025-08-18T23:59:{client_id:02d}Z",
                        "data": {},
                        "sequence": client_id
                    }
                    
                    w.write(json.dumps(request).encode() + b'\n')
                    await w.drain()
                    
                    response_data = await r.readline()
                    response = json.loads(response_data.decode().strip())
                    
                    assert response["type"] == "rssi_update"
                    return response
                
                tasks.append(client_request(reader, writer, i))
            
            # Wait for all responses
            responses = await asyncio.gather(*tasks)
            
            end_time = time.perf_counter()
            total_time_ms = (end_time - start_time) * 1000
            
            # Cleanup
            for reader, writer in clients:
                writer.close()
                await writer.wait_closed()
            
            # Verify all responses
            assert len(responses) == num_clients
            
            # Return average time per client
            return total_time_ms / num_clients
        
        # Benchmark concurrent operation
        result = await benchmark.pedantic(concurrent_clients_benchmark, rounds=5, iterations=1)
        
        # Even with multiple clients, per-client latency should stay <100ms
        assert result < 100.0, f"Concurrent client latency {result:.2f}ms exceeds requirement"

    @pytest.mark.asyncio
    async def test_sustained_operation_performance_degradation(self, benchmark, sdrpp_service_with_real_processor):
        """RED: Test for performance degradation during sustained operation."""
        service = sdrpp_service_with_real_processor
        await service.start()
        
        async def sustained_operation_benchmark():
            """Measure performance stability during sustained operation."""
            reader, writer = await asyncio.open_connection(service.host, service.port)
            
            latencies = []
            num_requests = 50  # Sustained operation test
            
            for i in range(num_requests):
                request = {
                    "type": "rssi_update", 
                    "timestamp": f"2025-08-18T23:59:{i:02d}Z",
                    "data": {},
                    "sequence": i
                }
                
                start_time = time.perf_counter()
                
                writer.write(json.dumps(request).encode() + b'\n')
                await writer.drain()
                
                response_data = await reader.readline()
                response = json.loads(response_data.decode().strip())
                
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                
                assert response["type"] == "rssi_update"
                
                # Small delay between requests to simulate realistic operation
                await asyncio.sleep(0.01)  # 10ms between requests
            
            # Cleanup
            writer.close()
            await writer.wait_closed()
            
            # Calculate performance metrics
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            min_latency = min(latencies)
            
            # Check for performance degradation (last 10 requests vs first 10)
            early_avg = sum(latencies[:10]) / 10
            late_avg = sum(latencies[-10:]) / 10
            degradation_factor = late_avg / early_avg
            
            # Performance should not degrade significantly
            assert degradation_factor < 2.0, f"Performance degraded by factor of {degradation_factor:.2f}"
            assert max_latency < 150.0, f"Peak latency {max_latency:.2f}ms too high"
            
            return avg_latency
        
        # Benchmark sustained operation
        result = await benchmark.pedantic(sustained_operation_benchmark, rounds=3, iterations=1)
        
        # Average latency during sustained operation must meet PRD requirement
        assert result < 100.0, f"Sustained operation latency {result:.2f}ms exceeds 100ms requirement"