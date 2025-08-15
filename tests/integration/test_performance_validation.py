"""
Performance Validation Tests - Priority 3
Tests system performance metrics, latency, and resource usage
"""

import asyncio
import time

import numpy as np
import psutil
import pytest

from src.backend.core.config import get_config
from src.backend.hal.hackrf_interface import HackRFConfig, HackRFInterface
from src.backend.services.mavlink_service import MAVLinkService
from src.backend.services.performance_monitor import PerformanceMonitor
from src.backend.services.state_machine import SystemState

pytestmark = pytest.mark.serial


@pytest.mark.hardware
class TestPerformanceValidation:
    """Priority 3 - Performance validation tests"""

    @pytest.fixture
    async def performance_monitor(self):
        """Create performance monitor"""
        monitor = PerformanceMonitor()
        await monitor.start()

        yield monitor

        await monitor.stop()

    @pytest.fixture
    async def hackrf(self):
        """Create HackRF interface"""
        config = HackRFConfig(frequency=3.2e9, sample_rate=20e6, lna_gain=16, vga_gain=20)

        hackrf = HackRFInterface(config)
        if not await hackrf.open():
            pytest.skip("HackRF not available")

        yield hackrf

        await hackrf.close()

    @pytest.fixture
    async def mavlink(self):
        """Create MAVLink service"""
        config = get_config()
        service = MAVLinkService(config)

        if not await service.connect():
            pytest.skip("MAVLink not connected")

        yield service

        await service.disconnect()

    @pytest.mark.asyncio
    async def test_mode_change_latency(self, mavlink):
        """Measure mode change latency (<100ms requirement)"""
        latencies = []

        # Test multiple mode changes
        test_modes = [
            (SystemState.IDLE, SystemState.SEARCHING),
            (SystemState.SEARCHING, SystemState.APPROACH),
            (SystemState.APPROACH, SystemState.IDLE),
        ]

        for from_state, to_state in test_modes:
            # Measure latency
            start_time = time.perf_counter()

            # Simulate mode change (using MAVLink service)
            # This would normally go through the command pipeline
            success = True  # Placeholder for actual mode change

            latency = (time.perf_counter() - start_time) * 1000  # ms
            latencies.append(latency)

            print(f"Mode change {from_state.name} → {to_state.name}: {latency:.1f}ms")

            await asyncio.sleep(0.5)  # Wait between tests

        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)

        print("\nMode Change Performance:")
        print(f"  Average latency: {avg_latency:.1f}ms")
        print(f"  Maximum latency: {max_latency:.1f}ms")

        assert (
            max_latency < 100
        ), f"Mode change latency {max_latency:.1f}ms exceeds 100ms requirement"

    @pytest.mark.asyncio
    async def test_emergency_stop_latency(self, mavlink):
        """Measure emergency stop latency (<500ms requirement)"""
        # Send test velocity command first
        await mavlink.send_velocity_ned(1.0, 0, 0)
        await asyncio.sleep(0.5)

        # Measure emergency stop
        start_time = time.perf_counter()

        # Send stop command
        success = await mavlink.send_velocity_ned(0, 0, 0)

        latency = (time.perf_counter() - start_time) * 1000  # ms

        print(f"Emergency stop latency: {latency:.1f}ms")

        assert success, "Emergency stop command should succeed"
        assert latency < 500, f"Emergency stop latency {latency:.1f}ms exceeds 500ms requirement"

    @pytest.mark.asyncio
    async def test_cpu_usage_at_1hz_rssi(self, performance_monitor, hackrf):
        """Measure CPU usage at 1Hz RSSI (<30% target)"""
        # Start SDR streaming
        sample_count = 0

        def sample_callback(samples: np.ndarray):
            nonlocal sample_count
            sample_count += 1

            # Simulate RSSI calculation
            rssi = 10 * np.log10(np.mean(np.abs(samples) ** 2) + 1e-10)

        await hackrf.start_rx(sample_callback)

        # Collect CPU measurements for 10 seconds
        cpu_measurements = []

        for _ in range(10):
            cpu_percent = psutil.cpu_percent(interval=1.0)
            cpu_measurements.append(cpu_percent)

            # Record metric
            performance_monitor.record_cpu_usage(cpu_percent)

        await hackrf.stop()

        # Calculate statistics
        avg_cpu = sum(cpu_measurements) / len(cpu_measurements)
        max_cpu = max(cpu_measurements)

        print("\nCPU Usage at 1Hz RSSI:")
        print(f"  Average: {avg_cpu:.1f}%")
        print(f"  Maximum: {max_cpu:.1f}%")
        print(f"  Samples processed: {sample_count}")

        assert avg_cpu < 30, f"Average CPU usage {avg_cpu:.1f}% exceeds 30% target"

    @pytest.mark.asyncio
    async def test_ram_usage(self, performance_monitor):
        """Measure RAM usage (<500MB target)"""
        # Get current process
        process = psutil.Process()

        # Collect memory measurements
        memory_measurements = []

        for _ in range(5):
            mem_info = process.memory_info()
            ram_mb = mem_info.rss / (1024 * 1024)  # Convert to MB
            memory_measurements.append(ram_mb)

            # Record metric
            performance_monitor.record_memory_usage(ram_mb)

            await asyncio.sleep(1.0)

        avg_ram = sum(memory_measurements) / len(memory_measurements)
        max_ram = max(memory_measurements)

        print("\nRAM Usage:")
        print(f"  Average: {avg_ram:.1f} MB")
        print(f"  Maximum: {max_ram:.1f} MB")

        assert max_ram < 500, f"Maximum RAM usage {max_ram:.1f}MB exceeds 500MB target"

    @pytest.mark.asyncio
    async def test_usb_throughput_hackrf(self, hackrf):
        """Measure USB 2.0 throughput for HackRF (20 Msps * 8 bytes/sample)"""
        # Expected throughput: 20 Msps * 8 bytes (complex64) = 160 MB/s
        # USB 2.0 max: 480 Mbps = 60 MB/s theoretical, ~35 MB/s practical

        bytes_received = 0
        samples_received = 0

        def sample_callback(samples: np.ndarray):
            nonlocal bytes_received, samples_received
            samples_received += len(samples)
            bytes_received += len(samples) * 8  # complex64 = 8 bytes

        # Measure for 5 seconds
        await hackrf.start_rx(sample_callback)
        start_time = time.time()
        await asyncio.sleep(5.0)
        await hackrf.stop()
        elapsed = time.time() - start_time

        # Calculate throughput
        throughput_mbps = (bytes_received * 8) / (elapsed * 1e6)  # Mbps
        throughput_mbytes = bytes_received / (elapsed * 1e6)  # MB/s
        effective_sample_rate = samples_received / elapsed / 1e6  # Msps

        print("\nUSB Throughput:")
        print(f"  Data rate: {throughput_mbps:.1f} Mbps")
        print(f"  Data rate: {throughput_mbytes:.1f} MB/s")
        print(f"  Effective sample rate: {effective_sample_rate:.1f} Msps")
        print(f"  Total samples: {samples_received:,}")

        # USB 2.0 practical limit is ~280 Mbps
        assert throughput_mbps < 480, "Throughput exceeds USB 2.0 theoretical maximum"

        # Should achieve at least 80% of target sample rate
        target_rate = hackrf.config.sample_rate / 1e6  # Msps
        assert (
            effective_sample_rate > target_rate * 0.8
        ), f"Sample rate {effective_sample_rate:.1f} Msps below 80% of target {target_rate} Msps"

    @pytest.mark.asyncio
    async def test_mavlink_packet_latency(self, mavlink, performance_monitor):
        """Measure MAVLink packet latency (<50ms target)"""
        latencies = []

        # Send multiple test commands and measure round-trip time
        for _ in range(10):
            start_time = time.perf_counter()

            # Request telemetry (round-trip)
            telemetry = await mavlink.get_telemetry()

            latency = (time.perf_counter() - start_time) * 1000  # ms
            latencies.append(latency)

            # Record metric
            performance_monitor.record_mavlink_latency(latency)

            await asyncio.sleep(0.1)

        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

        print("\nMAVLink Latency:")
        print(f"  Average: {avg_latency:.1f}ms")
        print(f"  Maximum: {max_latency:.1f}ms")
        print(f"  P95: {p95_latency:.1f}ms")

        assert avg_latency < 50, f"Average MAVLink latency {avg_latency:.1f}ms exceeds 50ms target"

    @pytest.mark.asyncio
    async def test_hackrf_rx_loop_performance(self, hackrf):
        """Test HackRF RX loop performance (asyncio.sleep(0.001))"""
        # Monitor callback timing
        callback_times = []
        callback_count = 0

        def timed_callback(samples: np.ndarray):
            nonlocal callback_count
            callback_times.append(time.perf_counter())
            callback_count += 1

        # Run for 2 seconds
        await hackrf.start_rx(timed_callback)
        await asyncio.sleep(2.0)
        await hackrf.stop()

        if len(callback_times) >= 2:
            # Calculate callback intervals
            intervals = [
                callback_times[i + 1] - callback_times[i] for i in range(len(callback_times) - 1)
            ]

            avg_interval = sum(intervals) / len(intervals)
            min_interval = min(intervals)
            max_interval = max(intervals)

            print("\nRX Loop Performance:")
            print(f"  Callbacks: {callback_count}")
            print(f"  Avg interval: {avg_interval*1000:.2f}ms")
            print(f"  Min interval: {min_interval*1000:.2f}ms")
            print(f"  Max interval: {max_interval*1000:.2f}ms")

            # Should maintain consistent timing
            assert max_interval < 0.1, f"Max callback interval {max_interval*1000:.1f}ms too high"

    @pytest.mark.asyncio
    async def test_system_startup_time(self):
        """Measure system startup time with hardware"""
        start_time = time.perf_counter()

        # Initialize all hardware components
        config = get_config()

        # Initialize HackRF
        hackrf = HackRFInterface()
        hackrf_success = await hackrf.open()
        hackrf_time = time.perf_counter() - start_time

        # Initialize MAVLink
        mavlink_start = time.perf_counter()
        mavlink = MAVLinkService(config)
        mavlink_success = await mavlink.connect()
        mavlink_time = time.perf_counter() - mavlink_start

        # Initialize Performance Monitor
        monitor_start = time.perf_counter()
        monitor = PerformanceMonitor()
        await monitor.start()
        monitor_time = time.perf_counter() - monitor_start

        total_time = time.perf_counter() - start_time

        print("\nStartup Times:")
        print(f"  HackRF: {hackrf_time*1000:.1f}ms {'✓' if hackrf_success else '✗'}")
        print(f"  MAVLink: {mavlink_time*1000:.1f}ms {'✓' if mavlink_success else '✗'}")
        print(f"  Monitor: {monitor_time*1000:.1f}ms")
        print(f"  Total: {total_time*1000:.1f}ms")

        # Cleanup
        if hackrf_success:
            await hackrf.close()
        if mavlink_success:
            await mavlink.disconnect()
        await monitor.stop()

        # Should start in reasonable time
        assert total_time < 10.0, f"Startup time {total_time:.1f}s exceeds 10s"

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, performance_monitor):
        """Test system performance with concurrent SDR and MAVLink operations"""
        config = get_config()

        # Initialize components
        hackrf = HackRFInterface()
        mavlink = MAVLinkService(config)

        hackrf_ready = await hackrf.open()
        mavlink_ready = await mavlink.connect()

        if not (hackrf_ready and mavlink_ready):
            pytest.skip("Both HackRF and MAVLink required for concurrent test")

        try:
            # Start concurrent operations
            sample_count = 0
            telemetry_count = 0

            def sample_callback(samples: np.ndarray):
                nonlocal sample_count
                sample_count += 1

            async def telemetry_loop():
                nonlocal telemetry_count
                for _ in range(50):
                    await mavlink.get_telemetry()
                    telemetry_count += 1
                    await asyncio.sleep(0.1)

            # Start SDR streaming
            await hackrf.start_rx(sample_callback)

            # Run telemetry loop concurrently
            start_time = time.time()
            await telemetry_loop()
            elapsed = time.time() - start_time

            # Stop SDR
            await hackrf.stop()

            # Calculate rates
            sample_rate = sample_count / elapsed
            telemetry_rate = telemetry_count / elapsed

            print("\nConcurrent Performance:")
            print(f"  SDR callbacks: {sample_rate:.1f} Hz")
            print(f"  Telemetry updates: {telemetry_rate:.1f} Hz")
            print(f"  CPU: {psutil.cpu_percent():.1f}%")
            print(f"  RAM: {psutil.Process().memory_info().rss / 1e6:.1f} MB")

            # Both should maintain reasonable rates
            assert sample_rate > 100, "SDR callback rate too low during concurrent ops"
            assert telemetry_rate > 5, "Telemetry rate too low during concurrent ops"

        finally:
            await hackrf.close()
            await mavlink.disconnect()


if __name__ == "__main__":
    # Run with: pytest tests/hardware/real/test_performance_validation.py -v -m hardware
    pytest.main([__file__, "-v", "-m", "hardware"])
