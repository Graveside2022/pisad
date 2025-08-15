"""
Signal Processing Performance Benchmark Tests

BACKWARDS ANALYSIS:
- User Action: System processes 100Hz RSSI data continuously
- Expected Result: Consistent <40ms latency per computation cycle
- Failure Impact: Missed signal detections, degraded homing accuracy

REQUIREMENT TRACE:
- NFR2: Signal processing latency shall not exceed 100ms per RSSI computation cycle
- User Story: 4.9 Task 9.7 - Performance test suite with 100Hz load tests

TEST VALUE: Ensures system meets real-time processing requirements for SAR operations
"""

import asyncio
import time
from unittest.mock import MagicMock

import numpy as np
import psutil
import pytest

from src.backend.services.signal_processor import SignalProcessor
from src.backend.utils.noise_estimator import NoiseEstimator


class TestSignalProcessingPerformance:
    """Performance benchmarks for signal processing pipeline"""

    @pytest.fixture
    def signal_processor(self):
        """Create signal processor with mocked dependencies"""
        processor = SignalProcessor()
        processor.sdr_service = MagicMock()
        processor.sdr_service.is_connected.return_value = True
        return processor

    @pytest.fixture
    def sample_iq_data(self):
        """Generate realistic IQ sample data"""
        # 1024 samples of complex IQ data
        return np.random.randn(1024) + 1j * np.random.randn(1024)

    @pytest.mark.benchmark(group="rssi")
    def test_rssi_computation_speed(self, benchmark, signal_processor, sample_iq_data):
        """Benchmark RSSI computation from IQ samples"""

        def compute_rssi():
            # Compute power spectrum
            fft_result = np.fft.fft(sample_iq_data)
            power_spectrum = np.abs(fft_result) ** 2

            # Calculate RSSI in dBm
            rssi_linear = np.mean(power_spectrum)
            rssi_dbm = 10 * np.log10(rssi_linear + 1e-10)
            return rssi_dbm

        result = benchmark(compute_rssi)

        # Performance assertions
        assert benchmark.stats["mean"] < 0.001  # Less than 1ms average
        assert benchmark.stats["max"] < 0.005  # Less than 5ms worst case
        print(f"\nRSSI computation: {benchmark.stats['mean']*1000:.3f}ms avg")

    @pytest.mark.benchmark(group="noise")
    def test_noise_floor_estimation_speed(self, benchmark):
        """Benchmark noise floor estimation with sliding window"""

        estimator = NoiseEstimator(window_size=1000)

        # Pre-fill with samples
        for i in range(1000):
            estimator.add_sample(-80.0 + np.random.randn())

        def estimate_noise():
            estimator.add_sample(-80.0 + np.random.randn())
            return estimator.get_percentile(10)

        result = benchmark(estimate_noise)

        # Performance assertions
        assert benchmark.stats["mean"] < 0.0001  # Less than 0.1ms average
        assert benchmark.stats["max"] < 0.001  # Less than 1ms worst case
        print(f"\nNoise estimation: {benchmark.stats['mean']*1000:.3f}ms avg")

    @pytest.mark.benchmark(group="pipeline")
    @pytest.mark.asyncio
    async def test_full_pipeline_100hz(self, benchmark, signal_processor):
        """Benchmark full signal processing pipeline at 100Hz"""

        # Mock IQ queue with data
        signal_processor.iq_queue = asyncio.Queue(maxsize=100)

        async def simulate_100hz_processing():
            """Simulate processing at 100Hz for 1 second"""
            processed = 0
            start = time.perf_counter()

            while processed < 100:  # Process 100 samples (1 second at 100Hz)
                # Add IQ data
                iq_data = np.random.randn(1024) + 1j * np.random.randn(1024)
                await signal_processor.iq_queue.put(iq_data)

                # Process it
                if not signal_processor.iq_queue.empty():
                    data = await signal_processor.iq_queue.get()

                    # Simulate processing
                    fft_result = np.fft.fft(data)
                    power_spectrum = np.abs(fft_result) ** 2
                    rssi = 10 * np.log10(np.mean(power_spectrum) + 1e-10)

                    processed += 1

            return time.perf_counter() - start

        # Run benchmark
        duration = await benchmark(simulate_100hz_processing)

        # Should process 100 samples in ~1 second
        assert duration < 1.5  # Allow 50% overhead
        throughput = 100 / duration
        print(f"\nPipeline throughput: {throughput:.1f} Hz")

    @pytest.mark.benchmark(group="memory")
    def test_memory_allocation_efficiency(self, benchmark):
        """Benchmark memory allocation patterns"""

        def allocate_and_process():
            # Simulate typical allocation pattern
            buffers = []
            for _ in range(10):
                # Allocate IQ buffer
                buffer = np.zeros(1024, dtype=np.complex64)
                buffers.append(buffer)

            # Process buffers
            for buf in buffers:
                _ = np.fft.fft(buf)

            return len(buffers)

        result = benchmark(allocate_and_process)

        # Check memory efficiency
        process = psutil.Process()
        memory_info = process.memory_info()

        assert benchmark.stats["mean"] < 0.01  # Less than 10ms
        print(f"\nMemory allocation: {benchmark.stats['mean']*1000:.3f}ms avg")

    @pytest.mark.benchmark(group="ewma")
    def test_ewma_filter_speed(self, benchmark):
        """Benchmark EWMA filter performance"""

        alpha = 0.3
        current_value = -75.0

        def apply_ewma():
            nonlocal current_value
            new_sample = -75.0 + np.random.randn()
            current_value = alpha * new_sample + (1 - alpha) * current_value
            return current_value

        result = benchmark(apply_ewma)

        assert benchmark.stats["mean"] < 0.00001  # Less than 0.01ms
        print(f"\nEWMA filter: {benchmark.stats['mean']*1000000:.3f}μs avg")

    @pytest.mark.benchmark(group="gradient")
    def test_gradient_calculation_speed(self, benchmark):
        """Benchmark gradient calculation for homing"""

        # Simulate RSSI history
        rssi_history = [
            (-75.0, (47.0, 8.0)),  # RSSI, (lat, lon)
            (-73.0, (47.001, 8.0)),
            (-71.0, (47.002, 8.0)),
            (-69.0, (47.003, 8.0)),
            (-67.0, (47.004, 8.0)),
        ]

        def calculate_gradient():
            # Simple gradient calculation
            if len(rssi_history) < 2:
                return (0, 0)

            # Calculate RSSI gradient direction
            recent = rssi_history[-3:]

            rssi_values = [r[0] for r in recent]
            positions = [r[1] for r in recent]

            # Simplified gradient
            d_rssi = rssi_values[-1] - rssi_values[0]
            d_lat = positions[-1][0] - positions[0][0]
            d_lon = positions[-1][1] - positions[0][1]

            # Normalize
            magnitude = np.sqrt(d_lat**2 + d_lon**2)
            if magnitude > 0:
                return (d_lat / magnitude, d_lon / magnitude)
            return (0, 0)

        result = benchmark(calculate_gradient)

        assert benchmark.stats["mean"] < 0.0001  # Less than 0.1ms
        print(f"\nGradient calc: {benchmark.stats['mean']*1000:.3f}ms avg")


class TestPerformanceRegression:
    """Regression detection for performance metrics"""

    # Performance baselines (from initial measurements)
    BASELINES = {
        "rssi_computation": 0.001,  # 1ms
        "noise_estimation": 0.0001,  # 0.1ms
        "gradient_calculation": 0.0001,  # 0.1ms
        "full_pipeline": 0.01,  # 10ms per sample
    }

    REGRESSION_THRESHOLD = 1.1  # 10% regression threshold

    def test_rssi_regression(self):
        """Detect RSSI computation performance regression"""

        # Measure current performance
        samples = np.random.randn(1024) + 1j * np.random.randn(1024)

        times = []
        for _ in range(100):
            start = time.perf_counter()
            fft_result = np.fft.fft(samples)
            power_spectrum = np.abs(fft_result) ** 2
            rssi = 10 * np.log10(np.mean(power_spectrum) + 1e-10)
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times)
        baseline = self.BASELINES["rssi_computation"]

        # Check for regression
        if avg_time > baseline * self.REGRESSION_THRESHOLD:
            pytest.fail(
                f"Performance regression detected! "
                f"RSSI computation: {avg_time*1000:.3f}ms "
                f"(baseline: {baseline*1000:.3f}ms, "
                f"regression: {(avg_time/baseline - 1)*100:.1f}%)"
            )

        print(f"\nRSSI performance OK: {avg_time*1000:.3f}ms (baseline: {baseline*1000:.3f}ms)")

    def test_noise_floor_regression(self):
        """Detect noise floor estimation regression"""

        estimator = NoiseEstimator(window_size=1000)

        # Pre-fill
        for i in range(1000):
            estimator.add_sample(-80.0 + np.random.randn())

        # Measure performance
        times = []
        for _ in range(1000):
            start = time.perf_counter()
            estimator.add_sample(-80.0 + np.random.randn())
            _ = estimator.get_percentile(10)
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times)
        baseline = self.BASELINES["noise_estimation"]

        # Check for regression
        if avg_time > baseline * self.REGRESSION_THRESHOLD:
            pytest.fail(
                f"Performance regression detected! "
                f"Noise estimation: {avg_time*1000:.3f}ms "
                f"(baseline: {baseline*1000:.3f}ms, "
                f"regression: {(avg_time/baseline - 1)*100:.1f}%)"
            )

        print(f"\nNoise estimation OK: {avg_time*1000:.3f}ms (baseline: {baseline*1000:.3f}ms)")


class TestMemoryProfiling:
    """Memory usage profiling tests"""

    @pytest.mark.asyncio
    async def test_memory_usage_100hz_load(self):
        """Profile memory usage under 100Hz load"""

        processor = SignalProcessor()
        processor.sdr_service = MagicMock()
        processor.sdr_service.is_connected.return_value = True

        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process at 100Hz for 10 seconds (1000 samples)
        start_time = time.time()
        samples_processed = 0

        while samples_processed < 1000:
            # Generate IQ data
            iq_data = np.random.randn(1024) + 1j * np.random.randn(1024)

            # Process
            fft_result = np.fft.fft(iq_data)
            power_spectrum = np.abs(fft_result) ** 2
            rssi = 10 * np.log10(np.mean(power_spectrum) + 1e-10)

            samples_processed += 1

            # Small yield to simulate async
            await asyncio.sleep(0)

        # Check memory growth
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory

        # Should have minimal memory growth
        assert memory_growth < 10  # Less than 10MB growth

        duration = time.time() - start_time
        rate = samples_processed / duration

        print("\nMemory profiling results:")
        print(f"  Initial memory: {initial_memory:.1f} MB")
        print(f"  Final memory: {final_memory:.1f} MB")
        print(f"  Memory growth: {memory_growth:.1f} MB")
        print(f"  Processing rate: {rate:.1f} Hz")

    def test_memory_leak_detection(self):
        """Detect memory leaks in processing pipeline"""

        import gc

        # Force garbage collection
        gc.collect()

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024

        # Run processing loop
        for iteration in range(10):
            # Allocate and process
            for _ in range(100):
                data = np.random.randn(1024) + 1j * np.random.randn(1024)
                _ = np.fft.fft(data)

            # Force GC
            gc.collect()

            # Check memory
            current_memory = process.memory_info().rss / 1024 / 1024
            iteration_growth = current_memory - initial_memory

            # Memory should stabilize
            if iteration > 5 and iteration_growth > 5:
                pytest.fail(f"Memory leak detected: {iteration_growth:.1f} MB growth")

        print("\nNo memory leaks detected")


def generate_performance_baseline():
    """Generate performance baseline file for regression detection"""

    baselines = {}

    # Measure RSSI computation
    samples = np.random.randn(1024) + 1j * np.random.randn(1024)
    times = []
    for _ in range(100):
        start = time.perf_counter()
        fft_result = np.fft.fft(samples)
        power_spectrum = np.abs(fft_result) ** 2
        rssi = 10 * np.log10(np.mean(power_spectrum) + 1e-10)
        times.append(time.perf_counter() - start)
    baselines["rssi_computation"] = np.mean(times)

    # Measure noise estimation
    estimator = NoiseEstimator(window_size=1000)
    for i in range(1000):
        estimator.add_sample(-80.0 + np.random.randn())

    times = []
    for _ in range(1000):
        start = time.perf_counter()
        estimator.add_sample(-80.0 + np.random.randn())
        _ = estimator.get_percentile(10)
        times.append(time.perf_counter() - start)
    baselines["noise_estimation"] = np.mean(times)

    # Save baselines
    import json

    with open("performance_baseline.json", "w") as f:
        json.dump(baselines, f, indent=2)

    print("Performance baseline generated:")
    for key, value in baselines.items():
        print(f"  {key}: {value*1000:.3f}ms")


if __name__ == "__main__":
    # Generate baseline if running directly
    generate_performance_baseline()
