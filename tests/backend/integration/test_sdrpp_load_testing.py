"""
Load Testing Framework for SDR++ TCP Communication

Tests validate sustained TCP operation under concurrent client connections.
Following TDD methodology with RED-GREEN-REFACTOR cycles.

PRD References:
- NFR2: Signal processing latency (<100ms per computation cycle)
- NFR8: 90% successful homing rate (success rate requirement)
- NFR9: MTBF >10 hours (connection stability under load)

Test Coverage:
- Concurrent client connection infrastructure with asyncio
- Sustained operation testing with multiple client simulation
- Performance monitoring for latency degradation under load
- Connection stability and memory usage validation
- Automatic reconnection resilience testing under load

Following TDD methodology - RED phase: Writing failing tests first.
"""

import asyncio
import json
import socket
import time
import psutil
from unittest.mock import MagicMock
import pytest
from src.backend.services.sdrpp_bridge_service import SDRPPBridgeService
from src.backend.services.signal_processor import SignalProcessor


class TestSDRPPLoadTesting:
    """Load testing framework for concurrent TCP client connections."""

    @pytest.fixture
    def unused_tcp_port(self):
        """Provide unused TCP port for testing."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    @pytest.fixture
    async def load_test_service(self, unused_tcp_port):
        """Provide SDRPPBridgeService configured for load testing."""
        # RED: This will fail - need to verify service can handle load testing setup
        service = SDRPPBridgeService()
        service.port = unused_tcp_port
        
        # Mock signal processor for load testing
        mock_processor = MagicMock(spec=SignalProcessor)
        mock_processor.get_current_rssi.return_value = -45.0
        mock_processor.get_current_snr.return_value = 15.0
        service.set_signal_processor(mock_processor)
        
        await service.start()
        yield service
        await service.stop()

    @pytest.mark.asyncio
    async def test_concurrent_client_connection_infrastructure(self, load_test_service):
        """RED: Test concurrent client connection setup with asyncio task management."""
        # This will fail initially - need concurrent connection infrastructure
        service = load_test_service
        
        # Test should create multiple concurrent connections
        concurrent_clients = 5
        client_tasks = []
        
        # Create concurrent client connection tasks
        for i in range(concurrent_clients):
            task = asyncio.create_task(self._create_test_client(service.port, f"client_{i}"))
            client_tasks.append(task)
        
        # Wait for all clients to connect
        connected_clients = await asyncio.gather(*client_tasks, return_exceptions=True)
        
        # Verify all clients connected successfully
        assert len(connected_clients) == concurrent_clients
        for client_result in connected_clients:
            assert not isinstance(client_result, Exception), f"Client connection failed: {client_result}"
        
        # Verify service tracks all clients
        assert len(service.clients) == concurrent_clients

    async def _create_test_client(self, port, client_id):
        """Helper method to create individual test client connection."""
        # RED: This will fail - need client connection implementation
        reader, writer = await asyncio.open_connection('127.0.0.1', port)
        
        # Send identification message
        message = {
            "type": "heartbeat",
            "client_id": client_id,
            "timestamp": time.time()
        }
        writer.write(json.dumps(message).encode() + b'\n')
        await writer.drain()
        
        return {"reader": reader, "writer": writer, "client_id": client_id}

    @pytest.mark.asyncio
    async def test_sustained_tcp_operation_with_multiple_clients(self, load_test_service):
        """RED: Test sustained operation with multiple client load simulation."""
        # This will fail initially - need sustained operation testing
        service = load_test_service
        
        # Create sustained load with multiple clients
        num_clients = 3
        test_duration = 5  # seconds
        
        # Start concurrent clients with sustained message traffic
        client_tasks = []
        for i in range(num_clients):
            task = asyncio.create_task(
                self._sustained_client_load(service.port, f"load_client_{i}", test_duration)
            )
            client_tasks.append(task)
        
        # Monitor service during sustained load
        start_time = time.time()
        while time.time() - start_time < test_duration:
            # Verify service remains responsive
            assert service.running is True
            assert len(service.clients) <= num_clients
            await asyncio.sleep(0.1)
        
        # Wait for all clients to complete
        results = await asyncio.gather(*client_tasks, return_exceptions=True)
        
        # Verify sustained operation success
        for result in results:
            assert not isinstance(result, Exception), f"Sustained load failed: {result}"
        
        # Verify service is still operational
        assert service.running is True

    async def _sustained_client_load(self, port, client_id, duration):
        """Helper method for sustained client load simulation."""
        # RED: This will fail - need sustained load implementation
        reader, writer = await asyncio.open_connection('127.0.0.1', port)
        
        start_time = time.time()
        message_count = 0
        
        try:
            while time.time() - start_time < duration:
                # Send RSSI update messages continuously
                message = {
                    "type": "rssi_update",
                    "client_id": client_id,
                    "sequence": message_count,
                    "timestamp": time.time()
                }
                writer.write(json.dumps(message).encode() + b'\n')
                await writer.drain()
                
                # Read response to maintain message flow
                response_data = await reader.readline()
                if response_data:
                    response = json.loads(response_data.decode().strip())
                    assert response["type"] == "rssi_update"
                
                message_count += 1
                await asyncio.sleep(0.1)  # 10Hz message rate
        
        finally:
            writer.close()
            await writer.wait_closed()
        
        return {"client_id": client_id, "messages_sent": message_count}

    @pytest.mark.asyncio
    async def test_performance_monitoring_under_load(self, load_test_service):
        """RED: Test performance monitoring for latency degradation under concurrent load."""
        # This will help validate PRD-NFR2 (<100ms latency) under load
        service = load_test_service
        
        num_clients = 5
        test_duration = 3  # seconds
        latency_measurements = []
        
        # Create concurrent clients with latency measurement
        client_tasks = []
        for i in range(num_clients):
            task = asyncio.create_task(
                self._measure_client_latency(service.port, f"perf_client_{i}", test_duration)
            )
            client_tasks.append(task)
        
        # Collect latency measurements from all clients
        results = await asyncio.gather(*client_tasks, return_exceptions=True)
        
        # Verify all clients completed successfully
        for result in results:
            assert not isinstance(result, Exception), f"Performance test failed: {result}"
            latency_measurements.extend(result["latencies"])
        
        # Validate PRD-NFR2 requirement: <100ms latency
        if latency_measurements:
            avg_latency = sum(latency_measurements) / len(latency_measurements)
            max_latency = max(latency_measurements)
            
            assert avg_latency < 0.1, f"Average latency {avg_latency:.3f}s exceeds 100ms"
            assert max_latency < 0.15, f"Max latency {max_latency:.3f}s significantly exceeds threshold"

    async def _measure_client_latency(self, port, client_id, duration):
        """Helper method to measure round-trip latency for individual client."""
        reader, writer = await asyncio.open_connection('127.0.0.1', port)
        
        latencies = []
        start_time = time.time()
        message_count = 0
        
        try:
            while time.time() - start_time < duration:
                # Measure round-trip latency
                send_time = time.time()
                
                message = {
                    "type": "rssi_update",
                    "client_id": client_id,
                    "sequence": message_count,
                    "timestamp": send_time
                }
                writer.write(json.dumps(message).encode() + b'\n')
                await writer.drain()
                
                # Read response and calculate latency
                response_data = await reader.readline()
                receive_time = time.time()
                
                if response_data:
                    latency = receive_time - send_time
                    latencies.append(latency)
                
                message_count += 1
                await asyncio.sleep(0.05)  # 20Hz measurement rate
        
        finally:
            writer.close()
            await writer.wait_closed()
        
        return {"client_id": client_id, "latencies": latencies, "messages": message_count}

    @pytest.mark.asyncio
    async def test_connection_stability_and_memory_usage(self, load_test_service):
        """RED: Test connection stability and memory usage during extended load."""
        service = load_test_service
        
        # Monitor initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create sustained load with connection cycling
        num_cycles = 3
        clients_per_cycle = 4
        cycle_duration = 2  # seconds
        
        for cycle in range(num_cycles):
            # Start clients for this cycle
            client_tasks = []
            for i in range(clients_per_cycle):
                task = asyncio.create_task(
                    self._cycling_client_load(service.port, f"cycle_{cycle}_client_{i}", cycle_duration)
                )
                client_tasks.append(task)
            
            # Wait for cycle to complete
            results = await asyncio.gather(*client_tasks, return_exceptions=True)
            
            # Verify cycle completed successfully
            for result in results:
                assert not isinstance(result, Exception), f"Connection stability test failed: {result}"
            
            # Check memory usage hasn't grown excessively
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_growth = current_memory - initial_memory
            
            # Allow some memory growth but not excessive (threshold: 50MB)
            assert memory_growth < 50, f"Memory growth {memory_growth:.1f}MB exceeds threshold after cycle {cycle}"
            
            # Brief pause between cycles
            await asyncio.sleep(0.5)
        
        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_growth = final_memory - initial_memory
        assert total_growth < 100, f"Total memory growth {total_growth:.1f}MB exceeds acceptable threshold"

    async def _cycling_client_load(self, port, client_id, duration):
        """Helper method for cycling client connections to test stability."""
        reader, writer = await asyncio.open_connection('127.0.0.1', port)
        
        start_time = time.time()
        message_count = 0
        
        try:
            while time.time() - start_time < duration:
                message = {
                    "type": "heartbeat",
                    "client_id": client_id,
                    "timestamp": time.time()
                }
                writer.write(json.dumps(message).encode() + b'\n')
                await writer.drain()
                
                message_count += 1
                await asyncio.sleep(0.1)  # 10Hz rate
        
        finally:
            writer.close()
            await writer.wait_closed()
        
        return {"client_id": client_id, "messages_sent": message_count}

    @pytest.mark.asyncio
    async def test_automatic_reconnection_resilience_under_load(self, load_test_service):
        """RED: Test automatic client disconnection and reconnection resilience under load."""
        service = load_test_service
        
        # Create clients that will disconnect and reconnect
        num_clients = 3
        test_duration = 4  # seconds
        
        client_tasks = []
        for i in range(num_clients):
            task = asyncio.create_task(
                self._reconnecting_client_load(service.port, f"reconnect_client_{i}", test_duration)
            )
            client_tasks.append(task)
        
        # Wait for all reconnecting clients to complete
        results = await asyncio.gather(*client_tasks, return_exceptions=True)
        
        # Verify resilience test completed successfully
        for result in results:
            assert not isinstance(result, Exception), f"Reconnection resilience test failed: {result}"
            # Verify each client successfully reconnected at least once
            assert result["reconnections"] > 0, f"Client {result['client_id']} did not reconnect"

    async def _reconnecting_client_load(self, port, client_id, duration):
        """Helper method for clients that disconnect and reconnect during test."""
        start_time = time.time()
        reconnections = 0
        total_messages = 0
        
        while time.time() - start_time < duration:
            try:
                # Connect
                reader, writer = await asyncio.open_connection('127.0.0.1', port)
                connection_start = time.time()
                message_count = 0
                
                # Send messages for a short period
                while (time.time() - connection_start < 1.0 and 
                       time.time() - start_time < duration):
                    message = {
                        "type": "heartbeat",
                        "client_id": client_id,
                        "timestamp": time.time()
                    }
                    writer.write(json.dumps(message).encode() + b'\n')
                    await writer.drain()
                    
                    message_count += 1
                    total_messages += 1
                    await asyncio.sleep(0.1)
                
                # Disconnect intentionally
                writer.close()
                await writer.wait_closed()
                reconnections += 1
                
                # Brief pause before reconnecting
                await asyncio.sleep(0.2)
                
            except Exception as e:
                # Handle connection errors gracefully
                await asyncio.sleep(0.1)
                continue
        
        return {
            "client_id": client_id, 
            "reconnections": reconnections,
            "total_messages": total_messages
        }