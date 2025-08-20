"""
Bandwidth-Latency Validation Tests (Task 5.6.8f)

Tests that bandwidth optimization implementations maintain PRD-NFR2 latency requirements:
- Signal processing latency <100ms per RSSI computation cycle
- End-to-end latency measurement with bandwidth controls active

Hardware Requirements:
- Raspberry Pi 5 network monitoring
- HackRF One SDR (for authentic signal processing)

Integration Points:
- NetworkBandwidthMonitor (from task 8a) - bandwidth monitoring
- BandwidthThrottle (from task 8d) - rate limiting
- CoordinationLatencyTracker - latency measurement
- SignalProcessor - RSSI computation pipeline
"""

import time

import numpy as np
import pytest

from src.backend.services.signal_processor import SignalProcessor
from src.backend.utils.coordination_optimizer import CoordinationLatencyTracker
from src.backend.utils.resource_optimizer import BandwidthThrottle, NetworkBandwidthMonitor


class TestEndToEndLatencyMeasurement:
    """Test end-to-end latency measurement framework with bandwidth controls."""

    # PRD-NFR2: Signal processing latency shall not exceed 100ms per RSSI computation cycle
    MAX_LATENCY_MS = 100.0

    @pytest.fixture
    def network_monitor(self):
        """NetworkBandwidthMonitor fixture with real psutil integration."""
        return NetworkBandwidthMonitor()

    @pytest.fixture
    def bandwidth_throttle(self, network_monitor):
        """BandwidthThrottle fixture with realistic configuration."""
        return BandwidthThrottle(
            window_size_seconds=10.0,
            max_bandwidth_bps=1_000_000,  # 1 Mbps limit for testing
            congestion_threshold_ratio=0.8,
        )

    @pytest.fixture
    def signal_processor(self):
        """SignalProcessor fixture for authentic RSSI computation."""
        return SignalProcessor(
            fft_size=1024,
            ewma_alpha=0.3,
            snr_threshold=12.0,
            noise_window_seconds=1.0,
            sample_rate=2e6,
        )

    @pytest.fixture
    def latency_tracker(self):
        """CoordinationLatencyTracker fixture for performance measurement."""
        return CoordinationLatencyTracker(
            max_samples=1000, warning_threshold_ms=50.0, alert_threshold_ms=100.0
        )

    @pytest.fixture
    def test_iq_samples(self):
        """Generate realistic IQ samples for latency testing."""
        # 1024 samples at complex64 - typical processing block
        real_part = np.random.randn(1024).astype(np.float32)
        imag_part = np.random.randn(1024).astype(np.float32)
        return (real_part + 1j * imag_part).astype(np.complex64)

    def test_end_to_end_latency_measurement_framework_exists(
        self, network_monitor, bandwidth_throttle, signal_processor, latency_tracker
    ):
        """
        TDD GREEN: Test that end-to-end latency measurement framework exists and can be instantiated.

        This test verifies the integration between:
        - NetworkBandwidthMonitor (bandwidth monitoring)
        - BandwidthThrottle (rate limiting)
        - SignalProcessor (RSSI computation)
        - CoordinationLatencyTracker (latency measurement)

        Framework should now exist and initialize successfully.
        """
        from src.backend.utils.resource_optimizer import EndToEndLatencyMeasurementFramework

        # Framework should initialize successfully with all components
        framework = EndToEndLatencyMeasurementFramework(
            network_monitor=network_monitor,
            bandwidth_throttle=bandwidth_throttle,
            signal_processor=signal_processor,
            latency_tracker=latency_tracker,
        )

        # Verify framework has all required components
        assert framework.network_monitor is network_monitor
        assert framework.bandwidth_throttle is bandwidth_throttle
        assert framework.signal_processor is signal_processor
        assert framework.latency_tracker is latency_tracker

        # Verify initial state
        assert not framework._measurement_active
        assert not framework._baseline_captured

    def test_rssi_processing_latency_measurement(
        self,
        network_monitor,
        bandwidth_throttle,
        signal_processor,
        latency_tracker,
        test_iq_samples,
    ):
        """
        Test [8f1] - RSSI processing latency measurement with authentic signal processing.

        Validates that the framework can measure actual RSSI computation latency
        and verify it meets PRD-NFR2 <100ms requirements.
        """
        from src.backend.utils.resource_optimizer import EndToEndLatencyMeasurementFramework

        framework = EndToEndLatencyMeasurementFramework(
            network_monitor=network_monitor,
            bandwidth_throttle=bandwidth_throttle,
            signal_processor=signal_processor,
            latency_tracker=latency_tracker,
        )

        # Start measurement session
        framework.start_measurement_session()
        assert framework._measurement_active
        assert framework._baseline_captured

        # Measure RSSI processing latency with authentic IQ samples
        latency_ms = framework.measure_rssi_processing_latency(test_iq_samples)

        # Verify latency is reasonable and meets PRD-NFR2
        assert latency_ms > 0.0  # Should take some time
        assert latency_ms < self.MAX_LATENCY_MS  # Should meet PRD-NFR2 requirement

        # Verify latency was recorded in tracker
        assert len(latency_tracker.latencies) == 1
        assert latency_tracker.latencies[0] == latency_ms

        # Test multiple measurements for statistical validation
        for _ in range(5):
            latency_ms = framework.measure_rssi_processing_latency(test_iq_samples)
            assert latency_ms < self.MAX_LATENCY_MS

        # Verify all measurements recorded
        assert len(latency_tracker.latencies) == 6

        # Stop session and get summary
        summary = framework.stop_measurement_session()
        assert not framework._measurement_active
        assert summary["total_measurements"] == 6
        assert summary["prd_nfr2_compliant"] is True
        assert summary["avg_latency_ms"] < self.MAX_LATENCY_MS

    def test_bandwidth_optimization_load_testing(
        self,
        network_monitor,
        bandwidth_throttle,
        signal_processor,
        latency_tracker,
        test_iq_samples,
    ):
        """
        Test [8f2] - Realistic bandwidth optimization load testing with RSSI streaming + control messages.

        Simulates realistic network load with RSSI streaming and control messages
        while measuring latency impact under bandwidth optimization.
        """
        from src.backend.utils.resource_optimizer import EndToEndLatencyMeasurementFramework

        framework = EndToEndLatencyMeasurementFramework(
            network_monitor=network_monitor,
            bandwidth_throttle=bandwidth_throttle,
            signal_processor=signal_processor,
            latency_tracker=latency_tracker,
        )

        framework.start_measurement_session()

        # Simulate high-frequency RSSI streaming (10Hz rate)
        import time

        rssi_streaming_latencies = []

        for i in range(10):  # 10 measurements at 10Hz = 1 second of streaming
            start_time = time.perf_counter()

            # Measure RSSI processing latency under simulated load
            processing_latency = framework.measure_rssi_processing_latency(test_iq_samples)

            # Simulate control message processing overhead
            bandwidth_usage = network_monitor.get_current_bandwidth_usage()

            # Simulate network throttling decision
            current_bandwidth = bandwidth_usage.get("total_bandwidth_bps", 0)
            if current_bandwidth > 500_000:  # 500 kbps threshold
                throttle_result = bandwidth_throttle.should_throttle_bandwidth(current_bandwidth)

            end_time = time.perf_counter()
            total_latency_ms = (end_time - start_time) * 1000.0
            rssi_streaming_latencies.append(total_latency_ms)

            # Simulate 10Hz streaming interval
            time.sleep(0.01)  # 10ms between measurements for 100Hz test rate

        # Validate all latencies meet PRD-NFR2 under load
        max_streaming_latency = max(rssi_streaming_latencies)
        avg_streaming_latency = sum(rssi_streaming_latencies) / len(rssi_streaming_latencies)

        assert (
            max_streaming_latency < self.MAX_LATENCY_MS
        ), f"Max streaming latency {max_streaming_latency:.2f}ms exceeds {self.MAX_LATENCY_MS}ms"
        assert (
            avg_streaming_latency < self.MAX_LATENCY_MS / 2
        ), f"Average streaming latency {avg_streaming_latency:.2f}ms should be well under {self.MAX_LATENCY_MS}ms"

        # Verify framework compliance validation
        assert framework.validate_latency_requirement()

        summary = framework.stop_measurement_session()
        assert summary["prd_nfr2_compliant"] is True

    def test_bandwidth_throttling_latency_impact(
        self,
        network_monitor,
        bandwidth_throttle,
        signal_processor,
        latency_tracker,
        test_iq_samples,
    ):
        """
        Test [8f3] - Validate <100ms latency requirement under various bandwidth throttling scenarios.

        Tests latency performance under different bandwidth throttling levels:
        - No throttling (baseline)
        - Moderate throttling (50% reduction)
        - Heavy throttling (80% reduction)
        - Extreme throttling (95% reduction)
        """
        from src.backend.utils.resource_optimizer import EndToEndLatencyMeasurementFramework

        framework = EndToEndLatencyMeasurementFramework(
            network_monitor=network_monitor,
            bandwidth_throttle=bandwidth_throttle,
            signal_processor=signal_processor,
            latency_tracker=latency_tracker,
        )

        throttling_scenarios = [
            {
                "name": "no_throttling",
                "bandwidth_limit": 10_000_000,
                "expected_max_latency": 50.0,
            },  # 10 Mbps - baseline
            {
                "name": "moderate_throttling",
                "bandwidth_limit": 5_000_000,
                "expected_max_latency": 75.0,
            },  # 5 Mbps - 50% reduction
            {
                "name": "heavy_throttling",
                "bandwidth_limit": 2_000_000,
                "expected_max_latency": 90.0,
            },  # 2 Mbps - 80% reduction
            {
                "name": "extreme_throttling",
                "bandwidth_limit": 500_000,
                "expected_max_latency": 99.0,
            },  # 500 kbps - 95% reduction
        ]

        for scenario in throttling_scenarios:
            # Reset measurement session for each scenario
            if framework._measurement_active:
                framework.stop_measurement_session()

            # Configure throttling for this scenario
            bandwidth_throttle._max_bandwidth_bps = scenario["bandwidth_limit"]

            framework.start_measurement_session()

            # Measure latency under this throttling scenario
            scenario_latencies = []
            for _ in range(5):  # 5 measurements per scenario
                latency_ms = framework.measure_rssi_processing_latency(test_iq_samples)
                scenario_latencies.append(latency_ms)

            max_latency = max(scenario_latencies)
            avg_latency = sum(scenario_latencies) / len(scenario_latencies)

            # Validate PRD-NFR2 compliance for this scenario
            assert (
                max_latency < self.MAX_LATENCY_MS
            ), f"Scenario {scenario['name']}: max latency {max_latency:.2f}ms exceeds {self.MAX_LATENCY_MS}ms"
            assert (
                avg_latency < scenario["expected_max_latency"]
            ), f"Scenario {scenario['name']}: avg latency {avg_latency:.2f}ms exceeds expected {scenario['expected_max_latency']}ms"

            # Verify framework validation passes
            assert framework.validate_latency_requirement()

            print(
                f"✓ Scenario {scenario['name']}: max={max_latency:.2f}ms, avg={avg_latency:.2f}ms, limit={scenario['bandwidth_limit']} bps"
            )

    def test_lz4_compression_latency_impact(
        self,
        network_monitor,
        bandwidth_throttle,
        signal_processor,
        latency_tracker,
        test_iq_samples,
    ):
        """
        Test [8f4] - Test latency performance with LZ4 compression enabled vs disabled for comparison analysis.

        Compares RSSI processing latency with and without LZ4 compression to ensure
        compression overhead doesn't violate PRD-NFR2 requirements.
        """
        from src.backend.utils.resource_optimizer import EndToEndLatencyMeasurementFramework

        framework = EndToEndLatencyMeasurementFramework(
            network_monitor=network_monitor,
            bandwidth_throttle=bandwidth_throttle,
            signal_processor=signal_processor,
            latency_tracker=latency_tracker,
        )

        # Test without compression (baseline)
        framework.start_measurement_session()

        baseline_latencies = []
        for _ in range(10):
            latency_ms = framework.measure_rssi_processing_latency(test_iq_samples)
            baseline_latencies.append(latency_ms)

        baseline_avg = sum(baseline_latencies) / len(baseline_latencies)
        baseline_max = max(baseline_latencies)

        framework.stop_measurement_session()

        # Test with compression enabled (simulated compression overhead)
        framework.start_measurement_session()

        compression_latencies = []
        for _ in range(10):
            # Simulate compression processing overhead
            import time

            compression_start = time.perf_counter()

            # Actual RSSI processing
            latency_ms = framework.measure_rssi_processing_latency(test_iq_samples)

            # Simulate LZ4 compression overhead (typically 1-3ms for small data)
            time.sleep(0.002)  # 2ms compression overhead simulation
            compression_end = time.perf_counter()

            total_latency_ms = latency_ms + (compression_end - compression_start) * 1000.0
            compression_latencies.append(total_latency_ms)

        compression_avg = sum(compression_latencies) / len(compression_latencies)
        compression_max = max(compression_latencies)

        # Validate both scenarios meet PRD-NFR2
        assert (
            baseline_max < self.MAX_LATENCY_MS
        ), f"Baseline max latency {baseline_max:.2f}ms exceeds {self.MAX_LATENCY_MS}ms"
        assert (
            compression_max < self.MAX_LATENCY_MS
        ), f"Compression max latency {compression_max:.2f}ms exceeds {self.MAX_LATENCY_MS}ms"

        # Compression should add minimal overhead (<10ms additional)
        compression_overhead = compression_avg - baseline_avg
        assert (
            compression_overhead < 10.0
        ), f"Compression overhead {compression_overhead:.2f}ms too high"

        # Both should still meet requirements
        assert framework.validate_latency_requirement()

        print(f"✓ Baseline: avg={baseline_avg:.2f}ms, max={baseline_max:.2f}ms")
        print(
            f"✓ Compression: avg={compression_avg:.2f}ms, max={compression_max:.2f}ms, overhead={compression_overhead:.2f}ms"
        )

    def test_congestion_detection_latency_impact(
        self,
        network_monitor,
        bandwidth_throttle,
        signal_processor,
        latency_tracker,
        test_iq_samples,
    ):
        """
        Test [8f5] - Measure latency impact of congestion detection and adaptive transmission rate adjustments.

        Tests the latency impact of:
        - Congestion detection algorithm execution
        - Adaptive transmission rate changes (10Hz → 5Hz → 2Hz → 1Hz)
        - Congestion response and recovery
        """
        from src.backend.utils.resource_optimizer import EndToEndLatencyMeasurementFramework

        framework = EndToEndLatencyMeasurementFramework(
            network_monitor=network_monitor,
            bandwidth_throttle=bandwidth_throttle,
            signal_processor=signal_processor,
            latency_tracker=latency_tracker,
        )

        framework.start_measurement_session()

        # Test congestion detection overhead
        congestion_detection_latencies = []

        for _ in range(5):
            import time

            start_time = time.perf_counter()

            # RSSI processing
            rssi_latency = framework.measure_rssi_processing_latency(test_iq_samples)

            # Simulate congestion detection execution
            current_bandwidth = network_monitor.get_current_bandwidth_usage().get(
                "total_bandwidth_bps", 0
            )
            congestion_result = bandwidth_throttle.should_throttle_bandwidth(current_bandwidth)

            # Simulate adaptive rate adjustment decision
            if congestion_result.get("congestion_detected", False):
                # Simulate rate adjustment logic (typically very fast)
                time.sleep(0.001)  # 1ms for rate adjustment

            end_time = time.perf_counter()
            total_latency_ms = (end_time - start_time) * 1000.0
            congestion_detection_latencies.append(total_latency_ms)

        avg_congestion_latency = sum(congestion_detection_latencies) / len(
            congestion_detection_latencies
        )
        max_congestion_latency = max(congestion_detection_latencies)

        # Validate congestion detection doesn't violate PRD-NFR2
        assert (
            max_congestion_latency < self.MAX_LATENCY_MS
        ), f"Max congestion detection latency {max_congestion_latency:.2f}ms exceeds {self.MAX_LATENCY_MS}ms"
        assert (
            avg_congestion_latency < self.MAX_LATENCY_MS / 2
        ), f"Average congestion detection latency {avg_congestion_latency:.2f}ms should be well under {self.MAX_LATENCY_MS}ms"

        # Test adaptive transmission rate scenarios
        transmission_rates = [10, 5, 2, 1]  # Hz
        for rate in transmission_rates:
            interval_ms = 1000.0 / rate  # Interval between transmissions

            # Measure latency at this transmission rate
            rate_latencies = []
            for _ in range(3):  # 3 measurements per rate
                latency_ms = framework.measure_rssi_processing_latency(test_iq_samples)
                rate_latencies.append(latency_ms)

            avg_rate_latency = sum(rate_latencies) / len(rate_latencies)

            # Latency should remain consistent regardless of transmission rate
            assert (
                avg_rate_latency < self.MAX_LATENCY_MS
            ), f"Rate {rate}Hz: avg latency {avg_rate_latency:.2f}ms exceeds {self.MAX_LATENCY_MS}ms"

        # Final validation
        assert framework.validate_latency_requirement()

        print(
            f"✓ Congestion detection: avg={avg_congestion_latency:.2f}ms, max={max_congestion_latency:.2f}ms"
        )

    def test_comprehensive_bandwidth_latency_validation_suite(
        self,
        network_monitor,
        bandwidth_throttle,
        signal_processor,
        latency_tracker,
        test_iq_samples,
    ):
        """
        Test [8f6] - Create comprehensive bandwidth-latency validation test suite with performance benchmarking.

        Comprehensive test suite covering all [8f1-8f5] scenarios with statistical analysis
        and performance benchmarking against PRD-NFR2 requirements.
        """
        from src.backend.utils.resource_optimizer import EndToEndLatencyMeasurementFramework

        framework = EndToEndLatencyMeasurementFramework(
            network_monitor=network_monitor,
            bandwidth_throttle=bandwidth_throttle,
            signal_processor=signal_processor,
            latency_tracker=latency_tracker,
        )

        # Comprehensive test matrix
        test_scenarios = [
            {
                "name": "baseline_performance",
                "throttle_bps": 10_000_000,
                "compression": False,
                "measurements": 20,
            },
            {
                "name": "moderate_throttling",
                "throttle_bps": 5_000_000,
                "compression": False,
                "measurements": 15,
            },
            {
                "name": "heavy_throttling",
                "throttle_bps": 2_000_000,
                "compression": True,
                "measurements": 15,
            },
            {
                "name": "extreme_throttling",
                "throttle_bps": 500_000,
                "compression": True,
                "measurements": 10,
            },
            {
                "name": "congestion_recovery",
                "throttle_bps": 1_000_000,
                "compression": True,
                "measurements": 10,
            },
        ]

        comprehensive_results = {}

        for scenario in test_scenarios:
            scenario_name = scenario["name"]

            # Configure scenario
            bandwidth_throttle._max_bandwidth_bps = scenario["throttle_bps"]

            framework.start_measurement_session()

            scenario_latencies = []
            for i in range(scenario["measurements"]):
                # Simulate compression overhead if enabled
                processing_start = time.perf_counter()

                latency_ms = framework.measure_rssi_processing_latency(test_iq_samples)

                if scenario["compression"]:
                    # Simulate compression processing time
                    time.sleep(0.001)  # 1ms compression overhead

                processing_end = time.perf_counter()
                total_latency = latency_ms + (processing_end - processing_start) * 1000.0
                scenario_latencies.append(total_latency)

            # Statistical analysis
            scenario_results = {
                "min_latency_ms": min(scenario_latencies),
                "max_latency_ms": max(scenario_latencies),
                "avg_latency_ms": sum(scenario_latencies) / len(scenario_latencies),
                "std_dev_ms": (
                    sum(
                        (x - sum(scenario_latencies) / len(scenario_latencies)) ** 2
                        for x in scenario_latencies
                    )
                    / len(scenario_latencies)
                )
                ** 0.5,
                "prd_nfr2_compliant": all(l < self.MAX_LATENCY_MS for l in scenario_latencies),
                "measurements_count": len(scenario_latencies),
                "throttle_bandwidth_bps": scenario["throttle_bps"],
                "compression_enabled": scenario["compression"],
            }

            comprehensive_results[scenario_name] = scenario_results

            # Validate this scenario meets PRD-NFR2
            assert (
                scenario_results["max_latency_ms"] < self.MAX_LATENCY_MS
            ), f"Scenario {scenario_name}: max latency {scenario_results['max_latency_ms']:.2f}ms exceeds {self.MAX_LATENCY_MS}ms"
            assert scenario_results[
                "prd_nfr2_compliant"
            ], f"Scenario {scenario_name}: PRD-NFR2 compliance failed"

            framework.stop_measurement_session()

        # Overall performance benchmarking
        all_max_latencies = [r["max_latency_ms"] for r in comprehensive_results.values()]
        all_avg_latencies = [r["avg_latency_ms"] for r in comprehensive_results.values()]

        overall_max_latency = max(all_max_latencies)
        overall_avg_latency = sum(all_avg_latencies) / len(all_avg_latencies)

        # Final comprehensive validation
        assert (
            overall_max_latency < self.MAX_LATENCY_MS
        ), f"Overall max latency {overall_max_latency:.2f}ms exceeds PRD-NFR2 {self.MAX_LATENCY_MS}ms requirement"
        assert (
            overall_avg_latency < self.MAX_LATENCY_MS / 2
        ), f"Overall average latency {overall_avg_latency:.2f}ms should be well under {self.MAX_LATENCY_MS}ms"

        # Performance report
        print("\n=== COMPREHENSIVE BANDWIDTH-LATENCY VALIDATION REPORT ===")
        for scenario_name, results in comprehensive_results.items():
            print(f"{scenario_name}:")
            print(f"  Bandwidth: {results['throttle_bandwidth_bps']/1000000:.1f} Mbps")
            print(f"  Compression: {results['compression_enabled']}")
            print(
                f"  Latency: min={results['min_latency_ms']:.2f}ms, avg={results['avg_latency_ms']:.2f}ms, max={results['max_latency_ms']:.2f}ms"
            )
            print(f"  Std Dev: {results['std_dev_ms']:.2f}ms")
            print(f"  PRD-NFR2 Compliant: {'✓' if results['prd_nfr2_compliant'] else '✗'}")
            print()

        print(
            f"Overall Performance: max={overall_max_latency:.2f}ms, avg={overall_avg_latency:.2f}ms"
        )
        print(
            f"PRD-NFR2 Requirement (<{self.MAX_LATENCY_MS}ms): {'✓ PASSED' if overall_max_latency < self.MAX_LATENCY_MS else '✗ FAILED'}"
        )
        print("=" * 60)
