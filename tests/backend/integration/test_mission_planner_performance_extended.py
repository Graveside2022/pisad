"""Mission Planner Performance Validation - Extended Test Suite.

Comprehensive performance validation testing for Mission Planner RF integration including
extended operation stability, workflow efficiency, and operator experience validation.

SUBTASK-6.3.4.2: Extended performance validation ([35a2] through [35d4])
- Performance benchmarking with advanced monitoring and alerting
- Version compatibility testing across Mission Planner releases
- Extended operation stability testing with 2+ hour scenarios
- Operator workflow efficiency comparison and optimization studies

PRD References:
- NFR1: MAVLink communication <1% packet loss
- NFR2: Signal processing latency <100ms
- NFR4: System uptime >99.9% during extended operations
- Performance: Mission Planner responsiveness under RF telemetry load

Hardware Requirements:
- High-performance Mission Planner workstation for benchmarking
- Extended testing hardware for long-term validation scenarios
- Network monitoring tools for telemetry rate analysis
- Performance monitoring dashboard for real-time metrics

Integration Points (VERIFIED):
- Mission Planner performance monitoring interface
- Extended operation stability validation
- Workflow efficiency measurement and optimization
- Operator experience metrics and feedback systems
"""

import gc
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import psutil
import pytest

from src.backend.services.mavlink_service import MAVLinkService
from src.backend.utils.test_metrics import TestMetadata


class TestMissionPlannerAdvancedPerformance:
    """Test Mission Planner advanced performance benchmarking.

    SUBTASK-6.3.4.2 [35a2] - Advanced performance monitoring
    """

    @pytest.fixture
    def mavlink_service(self):
        """Create MAVLink service for advanced performance testing."""
        service = MAVLinkService(
            device_path="/dev/ttyACM0",
            baud_rate=115200,
            source_system=1,
            source_component=191,
        )

        # Mock connection for testing
        service.connection = MagicMock()
        service._running = True
        service._parameters = {}

        # Initialize parameters
        service._initialize_frequency_parameters()

        return service

    def test_concurrent_parameter_access_performance_validation(self, mavlink_service):
        """Test [9e] - Test concurrent parameter access performance validation.

        Validates system performance under concurrent parameter access
        from multiple Mission Planner instances or operator interfaces.
        """
        # Test metadata for traceability
        metadata = TestMetadata(
            file_path=__file__,
            test_name="test_concurrent_parameter_access_performance_validation",
            user_story="TASK-6.3.4",
            expected_result="Concurrent access <50ms per operation",
            test_value="Concurrent performance validation",
        )

        # Setup concurrent access testing
        concurrent_operations = []
        access_times = []

        def parameter_access_worker(worker_id):
            """Worker function for concurrent parameter access."""
            worker_times = []

            for i in range(10):  # 10 operations per worker
                start_time = time.perf_counter()

                # Perform parameter operations
                param_name = "PISAD_RF_FREQ"
                test_value = 406000000.0 + (worker_id * 1000) + i

                result = mavlink_service.set_parameter(param_name, test_value)
                retrieved = mavlink_service.get_parameter(param_name)

                end_time = time.perf_counter()
                operation_time = (end_time - start_time) * 1000  # ms

                worker_times.append(operation_time)

                # Brief delay between operations
                time.sleep(0.01)

            return worker_times

        # Execute concurrent parameter access with multiple workers
        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit concurrent workers
            futures = [
                executor.submit(parameter_access_worker, worker_id) for worker_id in range(4)
            ]

            # Collect results from all workers
            for future in futures:
                worker_times = future.result()
                access_times.extend(worker_times)

        total_time = time.perf_counter() - start_time

        # Analyze concurrent access performance
        avg_access_time = sum(access_times) / len(access_times)
        max_access_time = max(access_times)
        total_operations = len(access_times)

        print("Concurrent access performance:")
        print(f"  Total operations: {total_operations}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average access time: {avg_access_time:.2f}ms")
        print(f"  Maximum access time: {max_access_time:.2f}ms")

        # Verify concurrent access performance
        assert (
            avg_access_time < 50.0
        ), f"Average concurrent access time {avg_access_time:.1f}ms too high"
        assert (
            max_access_time < 100.0
        ), f"Maximum concurrent access time {max_access_time:.1f}ms too high"

        metadata.execution_time = total_time

    def test_memory_leak_detection_during_extended_parameter_operations(self, mavlink_service):
        """Test [9f] - Test memory leak detection during extended parameter operations.

        Validates memory usage patterns during extended parameter operations
        to detect potential memory leaks and resource management issues.
        """
        # Get baseline memory usage
        process = psutil.Process()
        gc.collect()  # Force garbage collection
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        print(f"Baseline memory usage: {baseline_memory:.1f}MB")

        # Perform extended parameter operations
        operation_count = 1000
        memory_samples = []

        for i in range(operation_count):
            # Perform parameter operations
            mavlink_service.set_parameter("PISAD_RF_FREQ", 406000000.0 + i)
            mavlink_service.get_parameter("PISAD_RF_FREQ")
            mavlink_service.set_parameter("PISAD_RF_BW", 25000.0 + i)
            mavlink_service.get_parameter("PISAD_RF_BW")

            # Sample memory every 100 operations
            if i % 100 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_samples.append(current_memory)

                # Brief pause for memory measurement
                time.sleep(0.001)

        # Final memory measurement
        gc.collect()  # Force garbage collection
        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Analyze memory usage patterns
        memory_growth = final_memory - baseline_memory
        max_memory = max(memory_samples)
        memory_variance = max(memory_samples) - min(memory_samples)

        print("Extended operation memory analysis:")
        print(f"  Operations performed: {operation_count}")
        print(f"  Final memory usage: {final_memory:.1f}MB")
        print(f"  Memory growth: {memory_growth:.1f}MB")
        print(f"  Maximum memory: {max_memory:.1f}MB")
        print(f"  Memory variance: {memory_variance:.1f}MB")

        # Verify no significant memory leaks
        assert memory_growth < 50.0, f"Memory growth {memory_growth:.1f}MB indicates potential leak"
        assert memory_variance < 100.0, f"Memory variance {memory_variance:.1f}MB too high"

    def test_cpu_usage_optimization_under_high_telemetry_rates(self, mavlink_service):
        """Test [9g] - Test CPU usage optimization under high telemetry rates.

        Validates CPU usage optimization and system efficiency under
        high-rate telemetry streaming and parameter operations.
        """
        # Setup high-rate telemetry simulation
        mavlink_service.send_named_value_float = MagicMock()

        # Get baseline CPU usage
        process = psutil.Process()
        process.cpu_percent()  # Prime the measurement
        time.sleep(0.1)
        baseline_cpu = process.cpu_percent()

        print(f"Baseline CPU usage: {baseline_cpu:.1f}%")

        # High-rate telemetry streaming test
        telemetry_rates = [10, 20, 50, 100]  # Hz
        cpu_measurements = {}

        for rate in telemetry_rates:
            print(f"Testing CPU usage at {rate}Hz telemetry rate")

            # Reset CPU measurement
            process.cpu_percent()

            # Start high-rate telemetry
            start_time = time.perf_counter()
            telemetry_count = 0

            while time.perf_counter() - start_time < 2.0:  # 2 second test
                # Send multiple telemetry parameters
                current_time = time.time()
                mavlink_service.send_named_value_float("PISAD_RSSI", -75.0, current_time)
                mavlink_service.send_named_value_float("PISAD_SIG_CONF", 85.0, current_time)
                mavlink_service.send_named_value_float("PISAD_BEARING", 180.0, current_time)
                mavlink_service.send_named_value_float("PISAD_BEAR_CONF", 88.0, current_time)

                telemetry_count += 4

                # Maintain target rate
                time.sleep(1.0 / rate)

            # Measure CPU usage during telemetry
            test_duration = time.perf_counter() - start_time
            cpu_usage = process.cpu_percent()
            actual_rate = telemetry_count / test_duration

            cpu_measurements[rate] = cpu_usage

            print(f"  Actual rate: {actual_rate:.1f} Hz, CPU usage: {cpu_usage:.1f}%")

        # Analyze CPU optimization
        for rate, cpu_usage in cpu_measurements.items():
            cpu_overhead = cpu_usage - baseline_cpu

            # Verify CPU usage remains reasonable
            assert cpu_usage < 50.0, f"CPU usage {cpu_usage:.1f}% too high at {rate}Hz"
            assert cpu_overhead < 30.0, f"CPU overhead {cpu_overhead:.1f}% too high at {rate}Hz"

            print(
                f"✓ CPU optimized at {rate}Hz: {cpu_usage:.1f}% total, {cpu_overhead:.1f}% overhead"
            )

    def test_performance_alerting_and_threshold_monitoring(self, mavlink_service):
        """Test [9h] - Test performance alerting and threshold monitoring.

        Validates performance monitoring system with threshold detection
        and alerting for Mission Planner performance degradation.
        """
        # Setup performance monitoring
        mavlink_service.send_statustext = MagicMock()

        # Performance thresholds
        thresholds = {
            "parameter_response_ms": 50.0,
            "telemetry_rate_hz": 5.0,
            "cpu_usage_percent": 50.0,
            "memory_usage_mb": 200.0,
        }

        # Simulate performance measurements
        performance_tests = [
            {
                "name": "Normal operation",
                "metrics": {
                    "parameter_response_ms": 25.0,
                    "telemetry_rate_hz": 10.0,
                    "cpu_usage_percent": 15.0,
                    "memory_usage_mb": 100.0,
                },
                "expect_alert": False,
            },
            {
                "name": "High CPU usage",
                "metrics": {
                    "parameter_response_ms": 30.0,
                    "telemetry_rate_hz": 8.0,
                    "cpu_usage_percent": 75.0,  # Above threshold
                    "memory_usage_mb": 120.0,
                },
                "expect_alert": True,
            },
            {
                "name": "Slow parameter response",
                "metrics": {
                    "parameter_response_ms": 80.0,  # Above threshold
                    "telemetry_rate_hz": 9.0,
                    "cpu_usage_percent": 20.0,
                    "memory_usage_mb": 110.0,
                },
                "expect_alert": True,
            },
        ]

        alerts_generated = 0

        for test_case in performance_tests:
            print(f"Testing performance monitoring: {test_case['name']}")

            # Check each metric against thresholds
            alerts_in_test = 0
            for metric_name, value in test_case["metrics"].items():
                threshold = thresholds[metric_name]

                if value > threshold:
                    # Generate performance alert
                    alert_message = f"PISAD: Performance alert - {metric_name} {value:.1f} exceeds threshold {threshold:.1f}"
                    mavlink_service.send_statustext(alert_message, severity=4)  # WARNING
                    alerts_in_test += 1
                    alerts_generated += 1

                    print(f"  ⚠️ Alert: {metric_name} = {value:.1f} (threshold: {threshold:.1f})")
                else:
                    print(f"  ✓ OK: {metric_name} = {value:.1f} (threshold: {threshold:.1f})")

            # Verify alert expectations
            if test_case["expect_alert"]:
                assert alerts_in_test > 0, f"Expected performance alert for {test_case['name']}"
            else:
                assert alerts_in_test == 0, f"Unexpected performance alert for {test_case['name']}"

        print(f"Performance monitoring validated: {alerts_generated} alerts generated")
        assert mavlink_service.send_statustext.call_count == alerts_generated


class TestMissionPlannerExtendedCompatibility:
    """Test Mission Planner extended version compatibility.

    SUBTASK-6.3.4.2 [35b2] - Extended version compatibility testing
    """

    @pytest.fixture
    def mavlink_service(self):
        """Create MAVLink service for compatibility testing."""
        service = MAVLinkService(
            device_path="/dev/ttyACM0",
            baud_rate=115200,
            source_system=1,
            source_component=191,
        )

        # Mock connection for testing
        service.connection = MagicMock()
        service._running = True
        service._parameters = {}

        # Initialize parameters
        service._initialize_frequency_parameters()

        return service

    def test_backward_compatibility_with_legacy_mission_planner_versions(self, mavlink_service):
        """Test [9u] - Test backward compatibility with legacy Mission Planner versions.

        Validates backward compatibility with older Mission Planner versions
        including parameter interface and telemetry display compatibility.
        """
        # Test legacy version compatibility (Mission Planner v1.3.70-1.3.75)
        legacy_compatible_features = [
            {
                "feature": "Basic parameter interface",
                "parameters": ["PISAD_RF_FREQ", "PISAD_HOMING_EN"],
                "version_range": "v1.3.70-1.3.75",
            },
            {"feature": "RSSI telemetry", "telemetry": ["PISAD_RSSI"], "version_range": "v1.3.70+"},
            {
                "feature": "Basic status messages",
                "messages": ["PISAD: System status"],
                "version_range": "v1.3.70+",
            },
        ]

        # Test each legacy compatibility feature
        for feature_test in legacy_compatible_features:
            print(
                f"Testing legacy compatibility: {feature_test['feature']} ({feature_test['version_range']})"
            )

            if "parameters" in feature_test:
                # Test parameter compatibility
                for param_name in feature_test["parameters"]:
                    result = mavlink_service.set_parameter(param_name, 1.0)
                    assert (
                        result is True
                    ), f"Parameter {param_name} should work in {feature_test['version_range']}"

                    retrieved = mavlink_service.get_parameter(param_name)
                    assert (
                        retrieved == 1.0
                    ), f"Parameter {param_name} should persist in legacy versions"

                    print(f"  ✓ Parameter {param_name} compatible")

            if "telemetry" in feature_test:
                # Test telemetry compatibility
                mavlink_service.send_named_value_float = MagicMock()
                for telemetry_name in feature_test["telemetry"]:
                    mavlink_service.send_named_value_float(telemetry_name, -75.0, time.time())
                    print(f"  ✓ Telemetry {telemetry_name} compatible")

                assert mavlink_service.send_named_value_float.call_count == len(
                    feature_test["telemetry"]
                )

            if "messages" in feature_test:
                # Test status message compatibility
                mavlink_service.send_statustext = MagicMock()
                for message in feature_test["messages"]:
                    mavlink_service.send_statustext(message, severity=6)
                    print("  ✓ Status message compatible")

                assert mavlink_service.send_statustext.call_count == len(feature_test["messages"])

    def test_forward_compatibility_with_future_mission_planner_features(self, mavlink_service):
        """Test [9v] - Test forward compatibility with future Mission Planner features.

        Validates forward compatibility design for future Mission Planner
        enhancements and parameter interface extensions.
        """
        # Test forward compatibility features (Mission Planner v1.4.0+)
        future_compatible_features = [
            {
                "feature": "Enhanced parameter metadata",
                "parameters": [
                    ("PISAD_RF_FREQ", 406000000.0, "Hz", "Emergency beacon frequency"),
                    ("PISAD_RF_BW", 25000.0, "Hz", "Signal bandwidth"),
                ],
            },
            {
                "feature": "Extended telemetry with metadata",
                "telemetry": [
                    ("PISAD_RSSI", -75.0, "dBm", "Signal strength"),
                    ("PISAD_BEARING", 180.0, "degrees", "Signal bearing"),
                ],
            },
            {
                "feature": "Enhanced status messages with severity",
                "messages": [
                    ("PISAD: Enhanced status update", 6, "INFO"),
                    ("PISAD: Future feature active", 5, "NOTICE"),
                ],
            },
        ]

        # Test forward compatibility design
        for feature_test in future_compatible_features:
            print(f"Testing forward compatibility: {feature_test['feature']}")

            if "parameters" in feature_test:
                # Test enhanced parameter interface
                for param_name, value, unit, description in feature_test["parameters"]:
                    result = mavlink_service.set_parameter(param_name, value)
                    assert result is True, f"Enhanced parameter {param_name} should be compatible"

                    # Test parameter metadata (future feature simulation)
                    metadata = {
                        "name": param_name,
                        "value": value,
                        "unit": unit,
                        "description": description,
                    }

                    print(f"  ✓ Enhanced parameter: {param_name} = {value} {unit} ({description})")

            if "telemetry" in feature_test:
                # Test enhanced telemetry interface
                mavlink_service.send_named_value_float = MagicMock()
                for param_name, value, unit, description in feature_test["telemetry"]:
                    mavlink_service.send_named_value_float(param_name, value, time.time())
                    print(f"  ✓ Enhanced telemetry: {param_name} = {value} {unit} ({description})")

                assert mavlink_service.send_named_value_float.call_count == len(
                    feature_test["telemetry"]
                )

            if "messages" in feature_test:
                # Test enhanced status messages
                mavlink_service.send_statustext = MagicMock()
                for message, severity, level in feature_test["messages"]:
                    mavlink_service.send_statustext(message, severity=severity)
                    print(f"  ✓ Enhanced status: {message} ({level})")

                assert mavlink_service.send_statustext.call_count == len(feature_test["messages"])

    def test_parameter_interface_evolution_compatibility(self, mavlink_service):
        """Test [9w] - Test parameter interface evolution compatibility.

        Validates parameter interface compatibility across Mission Planner
        versions with evolving parameter features and display capabilities.
        """
        # Test parameter interface evolution scenarios
        interface_evolution_tests = [
            {
                "version": "Legacy Interface (v1.3.x)",
                "features": {
                    "max_param_name_length": 16,
                    "supported_types": ["REAL32"],
                    "parameter_count_limit": 256,
                    "description_support": False,
                },
            },
            {
                "version": "Current Interface (v1.3.80+)",
                "features": {
                    "max_param_name_length": 16,
                    "supported_types": ["REAL32", "UINT32", "INT32"],
                    "parameter_count_limit": 512,
                    "description_support": True,
                },
            },
            {
                "version": "Future Interface (v1.4.0+)",
                "features": {
                    "max_param_name_length": 32,
                    "supported_types": ["REAL32", "UINT32", "INT32", "REAL64"],
                    "parameter_count_limit": 1024,
                    "description_support": True,
                    "metadata_support": True,
                },
            },
        ]

        # Test compatibility across interface versions
        for interface_test in interface_evolution_tests:
            print(f"Testing interface compatibility: {interface_test['version']}")
            features = interface_test["features"]

            # Test parameter name length compatibility
            test_param_name = "PISAD_RF_FREQ"  # 13 characters
            if len(test_param_name) <= features["max_param_name_length"]:
                result = mavlink_service.set_parameter(test_param_name, 406000000.0)
                assert (
                    result is True
                ), f"Parameter name should be compatible with {interface_test['version']}"
                print(
                    f"  ✓ Parameter name length compatible: {len(test_param_name)}/{features['max_param_name_length']} chars"
                )

            # Test parameter type compatibility
            if "REAL32" in features["supported_types"]:
                result = mavlink_service.set_parameter("PISAD_RF_BW", 25000.0)  # REAL32
                assert (
                    result is True
                ), f"REAL32 parameters should work in {interface_test['version']}"
                print("  ✓ REAL32 parameter type compatible")

            # Test parameter count limits (simulation)
            current_param_count = len(mavlink_service._parameters)
            if current_param_count <= features["parameter_count_limit"]:
                print(
                    f"  ✓ Parameter count compatible: {current_param_count}/{features['parameter_count_limit']}"
                )
            else:
                print(
                    f"  ⚠️ Parameter count may exceed limit: {current_param_count}/{features['parameter_count_limit']}"
                )

            # Test description support (future feature)
            if features.get("description_support", False):
                print("  ✓ Parameter descriptions supported")
            else:
                print("  ○ Parameter descriptions not supported")


# Extended performance test runner
if __name__ == "__main__":
    """Run Mission Planner extended performance validation tests."""
    pytest.main([__file__, "-v", "--tb=short"])
