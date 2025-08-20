"""System Stability Validation Tests for Enhanced Algorithm Performance.

SUBTASK-6.2.4.2: Safety and Integration Validation - System Stability Validation [27d1-27d4]

This test suite validates system stability and robustness of enhanced algorithms
under various stress conditions, failure scenarios, and long-term operation.

Test Coverage:
- [27d1] System stability validation under sustained enhanced algorithm load
- [27d2] Error recovery validation during enhanced processing failures
- [27d3] Graceful degradation validation when resource limits approached
- [27d4] Long-term stability validation with continuous enhanced processing

PRD Requirements Validated:
- PRD-NFR4: System stability under load
- PRD-FR17: Error recovery mechanisms
- PRD-NFR6: Graceful degradation capabilities
- System reliability for extended operation
"""

import asyncio
import gc
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

from src.backend.services.asv_integration.asv_confidence_based_homing import (
    ASVConfidenceBasedHoming,
    DynamicThresholdConfig,
)
from src.backend.services.asv_integration.asv_enhanced_signal_processor import (
    ASVEnhancedSignalProcessor,
)

logger = logging.getLogger(__name__)


@dataclass
class SystemStabilityScenario:
    """Test scenario for system stability validation."""

    name: str
    description: str
    test_duration_minutes: float
    load_intensity: str  # sustained, burst, variable, extreme
    failure_injection_rate: float  # failures per minute
    resource_stress_level: float  # 0.0-1.0
    expected_uptime_percentage: float
    expected_recovery_time_seconds: float


@dataclass
class SystemStabilityMetrics:
    """Metrics collected during system stability validation."""

    total_test_duration_seconds: float
    uptime_seconds: float
    downtime_events: List[Dict[str, Any]]
    error_events: List[Dict[str, Any]]
    recovery_events: List[Dict[str, Any]]
    resource_exhaustion_events: List[Dict[str, Any]]
    graceful_degradation_events: List[Dict[str, Any]]
    performance_degradation_percentage: float
    memory_leak_detected: bool
    cpu_spike_events: int
    successful_operations: int
    failed_operations: int
    average_response_time_ms: float
    max_response_time_ms: float


class TestEnhancedAlgorithmSystemStabilityValidation:
    """Test suite for validating enhanced algorithm system stability."""

    @pytest.fixture
    async def enhanced_signal_processor(self):
        """Create ASV enhanced signal processor for stability testing."""
        processor = ASVEnhancedSignalProcessor()
        await processor.initialize()
        # Configure for stability monitoring
        await processor.enable_stability_monitoring()
        yield processor
        await processor.shutdown()

    @pytest.fixture
    async def confidence_based_homing(self, enhanced_signal_processor):
        """Create confidence-based homing system for stability testing."""
        homing = ASVConfidenceBasedHoming(
            asv_processor=enhanced_signal_processor, threshold_config=DynamicThresholdConfig()
        )
        yield homing

    @pytest.fixture
    def system_stability_scenarios(self) -> List[SystemStabilityScenario]:
        """Generate comprehensive system stability test scenarios."""
        return [
            SystemStabilityScenario(
                name="sustained_load_stability_validation",
                description="Sustained load stability for extended operation",
                test_duration_minutes=5.0,  # 5 minutes sustained load
                load_intensity="sustained",
                failure_injection_rate=0.5,  # 0.5 failures per minute
                resource_stress_level=0.7,
                expected_uptime_percentage=98.0,
                expected_recovery_time_seconds=2.0,
            ),
            SystemStabilityScenario(
                name="burst_load_stability_validation",
                description="Burst load stability with periodic high intensity",
                test_duration_minutes=3.0,  # 3 minutes burst testing
                load_intensity="burst",
                failure_injection_rate=1.0,  # 1 failure per minute
                resource_stress_level=0.8,
                expected_uptime_percentage=95.0,
                expected_recovery_time_seconds=3.0,
            ),
            SystemStabilityScenario(
                name="variable_load_stability_validation",
                description="Variable load stability with changing conditions",
                test_duration_minutes=4.0,  # 4 minutes variable load
                load_intensity="variable",
                failure_injection_rate=0.75,  # 0.75 failures per minute
                resource_stress_level=0.6,
                expected_uptime_percentage=96.5,
                expected_recovery_time_seconds=2.5,
            ),
            SystemStabilityScenario(
                name="extreme_stress_stability_validation",
                description="Extreme stress stability under maximum load",
                test_duration_minutes=2.5,  # 2.5 minutes extreme stress
                load_intensity="extreme",
                failure_injection_rate=2.0,  # 2 failures per minute
                resource_stress_level=0.9,
                expected_uptime_percentage=92.0,
                expected_recovery_time_seconds=5.0,
            ),
        ]

    @pytest.mark.asyncio
    async def test_system_stability_validation_under_sustained_enhanced_algorithm_load(
        self, enhanced_signal_processor, confidence_based_homing, system_stability_scenarios
    ):
        """[27d1] System stability validation under sustained enhanced algorithm load."""

        logger.info("Starting system stability validation under sustained enhanced algorithm load")

        stability_validation_results = []

        for scenario in system_stability_scenarios:
            logger.info(f"Testing system stability scenario: {scenario.name}")

            # Initialize stability monitoring
            stability_monitor = SystemStabilityMonitor(scenario.name)
            await stability_monitor.start_monitoring()

            try:
                # Run sustained enhanced algorithm load with stability monitoring
                metrics = await self._run_sustained_enhanced_algorithm_load_with_monitoring(
                    enhanced_signal_processor, confidence_based_homing, scenario, stability_monitor
                )

                # Validate system stability requirements
                uptime_percentage = (
                    metrics.uptime_seconds / metrics.total_test_duration_seconds
                ) * 100

                assert uptime_percentage >= scenario.expected_uptime_percentage, (
                    f"System uptime {uptime_percentage:.1f}% below "
                    f"expected {scenario.expected_uptime_percentage:.1f}% for {scenario.name}"
                )

                # Validate recovery time requirements
                if metrics.recovery_events:
                    avg_recovery_time = sum(
                        event["recovery_time_seconds"] for event in metrics.recovery_events
                    ) / len(metrics.recovery_events)

                    assert avg_recovery_time <= scenario.expected_recovery_time_seconds, (
                        f"Average recovery time {avg_recovery_time:.1f}s exceeds "
                        f"expected {scenario.expected_recovery_time_seconds:.1f}s for {scenario.name}"
                    )

                # Validate system performance didn't degrade excessively
                assert metrics.performance_degradation_percentage <= 30.0, (
                    f"Performance degradation {metrics.performance_degradation_percentage:.1f}% "
                    f"exceeds 30% threshold for {scenario.name}"
                )

                # Validate no memory leaks were detected
                assert (
                    not metrics.memory_leak_detected
                ), f"Memory leak detected during {scenario.name}"

                stability_validation_results.append(
                    {
                        "scenario": scenario.name,
                        "uptime_percentage": uptime_percentage,
                        "avg_recovery_time_seconds": avg_recovery_time
                        if metrics.recovery_events
                        else 0.0,
                        "performance_degradation": metrics.performance_degradation_percentage,
                        "error_event_count": len(metrics.error_events),
                        "successful_operations": metrics.successful_operations,
                        "failed_operations": metrics.failed_operations,
                        "load_intensity": scenario.load_intensity,
                    }
                )

                logger.info(
                    f"Stability validation for {scenario.name}: "
                    f"Uptime: {uptime_percentage:.1f}%, "
                    f"Performance degradation: {metrics.performance_degradation_percentage:.1f}%, "
                    f"Success rate: {metrics.successful_operations/(metrics.successful_operations + metrics.failed_operations + 1)*100:.1f}%"
                )

            finally:
                await stability_monitor.stop_monitoring()

            # Cool down and garbage collection between scenarios
            gc.collect()
            await asyncio.sleep(10.0)  # 10 second cool down

        # Validate overall system stability performance
        overall_min_uptime = min(
            result["uptime_percentage"] for result in stability_validation_results
        )
        overall_max_degradation = max(
            result["performance_degradation"] for result in stability_validation_results
        )
        overall_success_rate = (
            sum(
                result["successful_operations"]
                / max(1, result["successful_operations"] + result["failed_operations"])
                for result in stability_validation_results
            )
            / len(stability_validation_results)
            * 100
        )

        # Overall system stability requirements
        assert (
            overall_min_uptime >= 90.0
        ), f"Minimum system uptime {overall_min_uptime:.1f}% below 90% stability requirement"

        assert (
            overall_max_degradation <= 35.0
        ), f"Maximum performance degradation {overall_max_degradation:.1f}% exceeds 35% stability threshold"

        assert (
            overall_success_rate >= 85.0
        ), f"Overall success rate {overall_success_rate:.1f}% below 85% stability requirement"

        logger.info(
            f"System stability validation completed: "
            f"Min uptime: {overall_min_uptime:.1f}%, "
            f"Max degradation: {overall_max_degradation:.1f}%, "
            f"Success rate: {overall_success_rate:.1f}%"
        )

    async def _run_sustained_enhanced_algorithm_load_with_monitoring(
        self,
        signal_processor: ASVEnhancedSignalProcessor,
        confidence_homing: ASVConfidenceBasedHoming,
        scenario: SystemStabilityScenario,
        stability_monitor: "SystemStabilityMonitor",
    ) -> SystemStabilityMetrics:
        """Run sustained enhanced algorithm load with comprehensive stability monitoring."""

        test_duration_seconds = scenario.test_duration_minutes * 60.0
        end_time = time.perf_counter() + test_duration_seconds

        # Configure load based on scenario intensity
        load_configs = {
            "sustained": {"base_frequency": 100.0, "variance": 0.1, "concurrent_tasks": 4},
            "burst": {"base_frequency": 50.0, "variance": 0.8, "concurrent_tasks": 8},
            "variable": {"base_frequency": 75.0, "variance": 0.5, "concurrent_tasks": 6},
            "extreme": {"base_frequency": 200.0, "variance": 0.3, "concurrent_tasks": 12},
        }

        load_config = load_configs[scenario.load_intensity]

        # Start sustained load tasks
        load_tasks = []
        for i in range(load_config["concurrent_tasks"]):
            task = asyncio.create_task(
                self._sustained_load_task(
                    signal_processor, confidence_homing, end_time, load_config, stability_monitor, i
                )
            )
            load_tasks.append(task)

        # Start failure injection task
        failure_injection_task = asyncio.create_task(
            self._failure_injection_task(
                signal_processor,
                confidence_homing,
                end_time,
                scenario.failure_injection_rate,
                stability_monitor,
            )
        )
        load_tasks.append(failure_injection_task)

        # Start resource stress task
        resource_stress_task = asyncio.create_task(
            self._resource_stress_task(end_time, scenario.resource_stress_level, stability_monitor)
        )
        load_tasks.append(resource_stress_task)

        # Wait for all load tasks to complete
        await asyncio.gather(*load_tasks, return_exceptions=True)

        # Get final stability metrics
        return await stability_monitor.get_final_metrics()

    async def _sustained_load_task(
        self,
        signal_processor: ASVEnhancedSignalProcessor,
        confidence_homing: ASVConfidenceBasedHoming,
        end_time: float,
        load_config: Dict[str, Any],
        stability_monitor: "SystemStabilityMonitor",
        task_id: int,
    ) -> None:
        """Run sustained load task for system stability validation."""

        base_frequency = load_config["base_frequency"]
        variance = load_config["variance"]

        operation_count = 0

        while time.perf_counter() < end_time:
            try:
                operation_start = time.perf_counter()

                # Create load-appropriate signal data
                frequency_multiplier = 1.0 + random.uniform(-variance, variance)
                processing_frequency = base_frequency * frequency_multiplier
                processing_interval = 1.0 / processing_frequency

                signal_data = {
                    "signal_strength_dbm": -70.0 - (task_id * 5) + random.uniform(-10, 10),
                    "frequency_hz": 433.92e6,
                    "bearing_deg": (time.perf_counter() * 3 + task_id * 45) % 360,
                    "interference_detected": random.random() < 0.3,
                    "multipath_detected": random.random() < 0.2,
                    "noise_floor_dbm": -120.0,
                    "sample_rate_hz": 2.4e6,
                    "task_id": task_id,
                    "operation_id": operation_count,
                }

                # Perform enhanced processing
                bearing_calc = await signal_processor.calculate_enhanced_bearing(signal_data)
                decision = confidence_homing.evaluate_confidence_based_decision(bearing_calc)

                # Record successful operation
                operation_time = (time.perf_counter() - operation_start) * 1000
                await stability_monitor.record_successful_operation(
                    task_id, operation_count, operation_time
                )

                # Validate operation results
                if not (0 <= bearing_calc.bearing_deg <= 360):
                    await stability_monitor.record_error_event(
                        f"Invalid bearing result: {bearing_calc.bearing_deg}", task_id
                    )

                operation_count += 1
                await asyncio.sleep(processing_interval)

            except Exception as e:
                # Record failed operation
                await stability_monitor.record_failed_operation(task_id, operation_count, str(e))
                operation_count += 1
                await asyncio.sleep(0.1)  # Brief pause after failure

    async def _failure_injection_task(
        self,
        signal_processor: ASVEnhancedSignalProcessor,
        confidence_homing: ASVConfidenceBasedHoming,
        end_time: float,
        failure_rate: float,
        stability_monitor: "SystemStabilityMonitor",
    ) -> None:
        """Inject controlled failures for stability testing."""

        failure_interval = 60.0 / failure_rate  # Convert failures per minute to interval
        failure_count = 0

        while time.perf_counter() < end_time:
            await asyncio.sleep(failure_interval + random.uniform(-1.0, 1.0))

            if time.perf_counter() >= end_time:
                break

            failure_type = random.choice(
                [
                    "processor_exception",
                    "confidence_timeout",
                    "resource_exhaustion",
                    "configuration_corruption",
                    "network_failure",
                ]
            )

            try:
                await self._inject_controlled_failure(
                    signal_processor, confidence_homing, failure_type, stability_monitor
                )
                failure_count += 1

            except Exception as e:
                logger.warning(f"Failure injection error: {e}")

    async def _inject_controlled_failure(
        self,
        signal_processor: ASVEnhancedSignalProcessor,
        confidence_homing: ASVConfidenceBasedHoming,
        failure_type: str,
        stability_monitor: "SystemStabilityMonitor",
    ) -> None:
        """Inject specific type of controlled failure."""

        failure_start = time.perf_counter()

        if failure_type == "processor_exception":
            # Inject processing exception
            await signal_processor.inject_test_failure("calculation_exception")
            await stability_monitor.record_error_event(
                "Processor exception injected", "failure_injection"
            )

        elif failure_type == "confidence_timeout":
            # Inject confidence decision timeout
            await confidence_homing.configure_test_latency(timeout_ms=2000)
            await stability_monitor.record_error_event(
                "Confidence timeout injected", "failure_injection"
            )
            # Reset after timeout
            await asyncio.sleep(2.5)
            await confidence_homing.reset_test_configuration()

        elif failure_type == "resource_exhaustion":
            # Inject resource exhaustion
            await signal_processor.inject_test_failure("memory_exhaustion")
            await stability_monitor.record_resource_exhaustion_event(
                "Memory exhaustion injected", "failure_injection"
            )

        elif failure_type == "configuration_corruption":
            # Inject configuration corruption
            await confidence_homing.configure_threshold_parameters(
                DynamicThresholdConfig(
                    high_quality_threshold=999.0,  # Invalid configuration
                    moderate_quality_threshold=-1.0,
                    low_quality_threshold=0.0,
                )
            )
            await stability_monitor.record_error_event(
                "Configuration corruption injected", "failure_injection"
            )
            # Reset configuration after brief period
            await asyncio.sleep(1.0)
            await confidence_homing.configure_threshold_parameters(DynamicThresholdConfig())

        elif failure_type == "network_failure":
            # Simulate network failure (mock service disruption)
            await stability_monitor.record_error_event(
                "Network failure simulated", "failure_injection"
            )

        # Record recovery event if system recovers
        recovery_time = time.perf_counter() - failure_start
        await stability_monitor.record_recovery_event(failure_type, recovery_time)

    async def _resource_stress_task(
        self, end_time: float, stress_level: float, stability_monitor: "SystemStabilityMonitor"
    ) -> None:
        """Apply controlled resource stress for stability testing."""

        stress_cycle = 0

        while time.perf_counter() < end_time:
            try:
                # CPU stress
                cpu_stress_duration = 0.1 * stress_level
                cpu_stress_start = time.perf_counter()
                while time.perf_counter() - cpu_stress_start < cpu_stress_duration:
                    # CPU intensive operation
                    _ = sum(i * i for i in range(10000))

                # Memory stress
                if stress_level > 0.6:
                    memory_stress_data = []
                    for _ in range(int(1000 * stress_level)):
                        memory_stress_data.append([random.random() for _ in range(100)])

                    # Hold memory briefly then release
                    await asyncio.sleep(0.05)
                    del memory_stress_data

                    # Check for potential memory issues
                    import psutil

                    memory_percent = psutil.virtual_memory().percent
                    if memory_percent > 85.0:
                        await stability_monitor.record_resource_exhaustion_event(
                            f"High memory usage: {memory_percent:.1f}%", "resource_stress"
                        )

                stress_cycle += 1
                await asyncio.sleep(0.5)  # Stress cycle every 500ms

            except Exception as e:
                await stability_monitor.record_error_event(
                    f"Resource stress error: {e}", "resource_stress"
                )
                await asyncio.sleep(1.0)

    @pytest.mark.asyncio
    async def test_error_recovery_validation_during_enhanced_processing_failures(
        self, enhanced_signal_processor, confidence_based_homing
    ):
        """[27d2] Error recovery validation during enhanced processing failures."""

        logger.info("Testing error recovery validation during enhanced processing failures")

        # Test various error recovery scenarios
        error_recovery_scenarios = [
            {
                "name": "processor_crash_recovery",
                "description": "Recovery from signal processor crash",
                "failure_type": "processor_crash",
                "expected_recovery_time_ms": 3000.0,
            },
            {
                "name": "confidence_system_failure_recovery",
                "description": "Recovery from confidence system failure",
                "failure_type": "confidence_failure",
                "expected_recovery_time_ms": 2000.0,
            },
            {
                "name": "memory_leak_recovery",
                "description": "Recovery from memory leak detection",
                "failure_type": "memory_leak",
                "expected_recovery_time_ms": 5000.0,
            },
            {
                "name": "configuration_corruption_recovery",
                "description": "Recovery from configuration corruption",
                "failure_type": "config_corruption",
                "expected_recovery_time_ms": 1500.0,
            },
        ]

        error_recovery_results = []

        for scenario in error_recovery_scenarios:
            logger.info(f"Testing error recovery scenario: {scenario['name']}")

            # Initialize recovery monitoring
            recovery_start = time.perf_counter()

            # Inject specific failure
            await self._inject_error_for_recovery_testing(
                enhanced_signal_processor, confidence_homing, scenario["failure_type"]
            )

            # Monitor recovery process
            recovery_successful = await self._monitor_error_recovery(
                enhanced_signal_processor, confidence_homing, scenario["expected_recovery_time_ms"]
            )

            recovery_time_ms = (time.perf_counter() - recovery_start) * 1000

            # Validate recovery was successful
            assert (
                recovery_successful
            ), f"Error recovery failed for {scenario['name']} after {recovery_time_ms:.1f}ms"

            # Validate recovery time was acceptable
            assert recovery_time_ms <= scenario["expected_recovery_time_ms"], (
                f"Error recovery took {recovery_time_ms:.1f}ms, exceeds "
                f"expected {scenario['expected_recovery_time_ms']:.1f}ms for {scenario['name']}"
            )

            error_recovery_results.append(
                {
                    "scenario": scenario["name"],
                    "recovery_time_ms": recovery_time_ms,
                    "recovery_successful": recovery_successful,
                    "failure_type": scenario["failure_type"],
                }
            )

            logger.info(
                f"Error recovery for {scenario['name']}: "
                f"Successful: {recovery_successful}, Time: {recovery_time_ms:.1f}ms"
            )

            # Reset system state for next test
            await self._reset_system_state(enhanced_signal_processor, confidence_homing)
            await asyncio.sleep(2.0)  # Cool down between tests

        # Validate overall error recovery performance
        max_recovery_time = max(result["recovery_time_ms"] for result in error_recovery_results)
        recovery_success_rate = (
            sum(1 for result in error_recovery_results if result["recovery_successful"])
            / len(error_recovery_results)
            * 100
        )

        # Error recovery should be reliable and timely
        assert (
            max_recovery_time <= 6000.0
        ), f"Maximum error recovery time {max_recovery_time:.1f}ms exceeds 6000ms threshold"

        assert (
            recovery_success_rate >= 95.0
        ), f"Error recovery success rate {recovery_success_rate:.1f}% below 95% requirement"

        logger.info(
            f"Error recovery validation completed: "
            f"Max recovery time: {max_recovery_time:.1f}ms, Success rate: {recovery_success_rate:.1f}%"
        )
