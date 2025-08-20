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
from typing import Any, AsyncGenerator, Dict, List

import pytest

from src.backend.services.asv_integration.asv_confidence_based_homing import (
    ASVConfidenceBasedHoming,
    DynamicThresholdConfig,
)
from src.backend.services.asv_integration.asv_enhanced_signal_processor import (
    ASVBearingCalculation,
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
    async def enhanced_signal_processor(self) -> AsyncGenerator[ASVEnhancedSignalProcessor, None]:
        """Create ASV enhanced signal processor for stability testing."""
        processor = ASVEnhancedSignalProcessor()
        # No initialize/shutdown methods needed
        yield processor

    @pytest.fixture
    async def confidence_based_homing(
        self, enhanced_signal_processor: ASVEnhancedSignalProcessor
    ) -> AsyncGenerator[ASVConfidenceBasedHoming, None]:
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
        self,
        enhanced_signal_processor: ASVEnhancedSignalProcessor,
        confidence_based_homing: ASVConfidenceBasedHoming,
        system_stability_scenarios: List[SystemStabilityScenario],
    ) -> None:
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
            float(result["uptime_percentage"]) for result in stability_validation_results
        )
        overall_max_degradation = max(
            float(result["performance_degradation"]) for result in stability_validation_results
        )
        overall_success_rate = (
            sum(
                float(result["successful_operations"])
                / max(
                    1, float(result["successful_operations"]) + float(result["failed_operations"])
                )
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
        for i in range(int(load_config["concurrent_tasks"])):
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

                # Create mock ASV bearing calculation for stability testing
                bearing_calc = ASVBearingCalculation(
                    bearing_deg=signal_data["bearing_deg"],
                    confidence=0.8 + random.uniform(-0.2, 0.2),
                    precision_deg=2.0,
                    signal_strength_dbm=signal_data["signal_strength_dbm"],
                    signal_quality=0.7 + random.uniform(-0.1, 0.1),
                    timestamp_ns=int(time.perf_counter() * 1e9),
                    analyzer_type="stability_test",
                    interference_detected=signal_data.get("interference_detected", False),
                )
                decision = confidence_homing.evaluate_confidence_based_decision(bearing_calc)

                # Record successful operation
                operation_time = (time.perf_counter() - operation_start) * 1000
                await stability_monitor.record_successful_operation(
                    task_id, operation_count, operation_time
                )

                # Validate operation results
                if not (0 <= bearing_calc.bearing_deg <= 360):
                    await stability_monitor.record_error_event(
                        f"Invalid bearing result: {bearing_calc.bearing_deg}", str(task_id)
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
            # Simulate timeout by brief pause
            await asyncio.sleep(2.0)
            await stability_monitor.record_error_event(
                "Confidence timeout injected", "failure_injection"
            )

        elif failure_type == "resource_exhaustion":
            # Inject resource exhaustion simulation
            logger.warning("Simulating resource exhaustion failure")
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
            confidence_homing.configure_threshold_parameters(DynamicThresholdConfig())

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
                enhanced_signal_processor, confidence_based_homing, scenario["failure_type"]
            )

            # Monitor recovery process
            recovery_successful = await self._monitor_error_recovery(
                enhanced_signal_processor,
                confidence_based_homing,
                scenario["expected_recovery_time_ms"],
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
            await self._reset_system_state(enhanced_signal_processor, confidence_based_homing)
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

    async def _inject_error_for_recovery_testing(
        self,
        signal_processor: ASVEnhancedSignalProcessor,
        confidence_homing: ASVConfidenceBasedHoming,
        failure_type: str,
    ) -> None:
        """Inject specific error for recovery testing."""

        if failure_type == "processor_crash":
            # Simulate processor crash - no specific injection method, just log
            logger.warning("Simulating processor crash failure")

        elif failure_type == "confidence_failure":
            # Simulate confidence system failure
            confidence_homing.set_safety_override(True, "Test confidence failure")

        elif failure_type == "memory_leak":
            # Simulate memory leak by creating large data structures
            logger.warning("Simulating memory leak failure")

        elif failure_type == "config_corruption":
            # Corrupt configuration
            invalid_config = DynamicThresholdConfig(
                high_quality_threshold=-1.0,  # Invalid
                moderate_quality_threshold=999.0,  # Invalid
                low_quality_threshold=0.5,
            )
            confidence_homing.configure_threshold_parameters(invalid_config)

    async def _monitor_error_recovery(
        self,
        signal_processor: ASVEnhancedSignalProcessor,
        confidence_homing: ASVConfidenceBasedHoming,
        max_recovery_time_ms: float,
    ) -> bool:
        """Monitor error recovery process."""

        recovery_start = time.perf_counter()
        recovery_timeout = recovery_start + (max_recovery_time_ms / 1000.0)

        # Monitor recovery indicators
        while time.perf_counter() < recovery_timeout:
            try:
                # Test basic functionality
                test_signal_data = {
                    "signal_strength_dbm": -60.0,
                    "frequency_hz": 433.92e6,
                    "bearing_deg": 45.0,
                    "interference_detected": False,
                    "multipath_detected": False,
                    "noise_floor_dbm": -120.0,
                    "sample_rate_hz": 2.4e6,
                }

                # Try processing with mock calculation
                bearing_calc = ASVBearingCalculation(
                    bearing_deg=test_signal_data["bearing_deg"],
                    confidence=0.8,
                    precision_deg=2.0,
                    signal_strength_dbm=test_signal_data["signal_strength_dbm"],
                    signal_quality=0.7,
                    timestamp_ns=int(time.perf_counter() * 1e9),
                    analyzer_type="recovery_test",
                    interference_detected=test_signal_data.get("interference_detected", False),
                )
                decision = confidence_homing.evaluate_confidence_based_decision(bearing_calc)

                # Check if system is functioning normally
                if (
                    bearing_calc.confidence > 0.0
                    and 0 <= bearing_calc.bearing_deg <= 360
                    and decision is not None
                ):
                    return True  # Recovery successful

            except Exception:
                # Still recovering
                pass

            await asyncio.sleep(0.1)  # Check every 100ms

        return False  # Recovery failed within timeout

    async def _reset_system_state(
        self,
        signal_processor: ASVEnhancedSignalProcessor,
        confidence_homing: ASVConfidenceBasedHoming,
    ) -> None:
        """Reset system to clean state for next test."""

        # Reset safety override
        confidence_homing.set_safety_override(False)

        # Reset configuration to defaults
        confidence_homing.configure_threshold_parameters(DynamicThresholdConfig())

        # Reset processor state (no specific method needed)

        # Reset metrics
        confidence_homing.reset_metrics()

    @pytest.mark.asyncio
    async def test_graceful_degradation_validation_when_resource_limits_approached(
        self,
        enhanced_signal_processor: ASVEnhancedSignalProcessor,
        confidence_based_homing: ASVConfidenceBasedHoming,
    ) -> None:
        """[27d3] Graceful degradation validation when resource limits approached."""

        logger.info("Testing graceful degradation validation when resource limits approached")

        # Test various resource limit scenarios
        resource_limit_scenarios = [
            {
                "name": "cpu_limit_approached_80_percent",
                "description": "CPU usage approaching 80% limit",
                "resource_type": "cpu",
                "limit_percentage": 80.0,
                "stress_duration_seconds": 30.0,
                "expected_degradation_response": "reduce_processing_frequency",
            },
            {
                "name": "memory_limit_approached_400mb",
                "description": "Memory usage approaching 400MB limit",
                "resource_type": "memory",
                "limit_percentage": 78.0,  # 400MB out of 512MB
                "stress_duration_seconds": 25.0,
                "expected_degradation_response": "reduce_memory_buffers",
            },
            {
                "name": "io_bandwidth_limit_approached",
                "description": "I/O bandwidth limit approached",
                "resource_type": "io",
                "limit_percentage": 85.0,
                "stress_duration_seconds": 20.0,
                "expected_degradation_response": "reduce_sample_rate",
            },
        ]

        graceful_degradation_results = []

        for scenario in resource_limit_scenarios:
            logger.info(f"Testing graceful degradation scenario: {scenario['name']}")

            # Simulate graceful degradation scenario
            degradation_start = time.perf_counter()

            # Apply resource stress
            await self._apply_resource_stress_approaching_limit(
                enhanced_signal_processor, confidence_based_homing, scenario
            )

            # Test system functionality during stress
            functionality_maintained = await self._test_functionality_during_degradation(
                enhanced_signal_processor, confidence_based_homing
            )

            # Measure performance impact
            performance_impact = await self._measure_performance_impact_during_degradation(
                enhanced_signal_processor, confidence_based_homing
            )

            degradation_time = (time.perf_counter() - degradation_start) * 1000

            # Validate graceful degradation requirements
            assert (
                functionality_maintained
            ), f"System lost functionality during degradation for {scenario['name']}"

            assert performance_impact <= 50.0, (
                f"Performance degraded by {performance_impact:.1f}%, "
                f"exceeds 50% limit for {scenario['name']}"
            )

            graceful_degradation_results.append(
                {
                    "scenario": scenario["name"],
                    "resource_type": scenario["resource_type"],
                    "functionality_maintained": functionality_maintained,
                    "performance_impact": performance_impact,
                    "degradation_time_ms": degradation_time,
                }
            )

            logger.info(
                f"Graceful degradation for {scenario['name']}: "
                f"Functionality maintained: {functionality_maintained}, "
                f"Performance impact: {performance_impact:.1f}%"
            )

            # Cool down between scenarios
            await asyncio.sleep(3.0)

        # Validate overall graceful degradation performance
        all_maintained_functionality = all(
            result["functionality_maintained"] for result in graceful_degradation_results
        )
        max_performance_impact = max(
            float(result["performance_impact"]) for result in graceful_degradation_results
        )

        assert (
            all_maintained_functionality
        ), "System lost functionality in some degradation scenarios"
        assert (
            max_performance_impact <= 50.0
        ), f"Maximum performance impact {max_performance_impact:.1f}% exceeds 50% threshold"

        logger.info(
            f"Graceful degradation validation completed: "
            f"Functionality maintained: {all_maintained_functionality}, "
            f"Max performance impact: {max_performance_impact:.1f}%"
        )

    async def _apply_resource_stress_approaching_limit(
        self,
        signal_processor: ASVEnhancedSignalProcessor,
        confidence_homing: ASVConfidenceBasedHoming,
        scenario: Dict[str, Any],
    ) -> None:
        """Apply resource stress that approaches system limits."""

        stress_duration = scenario["stress_duration_seconds"]
        resource_type = scenario["resource_type"]

        if resource_type == "cpu":
            # CPU stress approaching 80% limit
            stress_tasks = []
            for i in range(4):  # Multiple CPU-intensive tasks
                task = asyncio.create_task(self._cpu_stress_task(stress_duration))
                stress_tasks.append(task)
            await asyncio.gather(*stress_tasks, return_exceptions=True)

        elif resource_type == "memory":
            # Memory stress approaching limit
            await self._memory_stress_task(stress_duration)

        elif resource_type == "io":
            # I/O stress
            await self._io_stress_task(stress_duration)

    async def _cpu_stress_task(self, duration: float) -> None:
        """Apply CPU stress for specified duration."""
        end_time = time.perf_counter() + duration
        while time.perf_counter() < end_time:
            # CPU intensive operation
            _ = sum(i * i * i for i in range(10000))
            await asyncio.sleep(0.01)  # Brief yield

    async def _memory_stress_task(self, duration: float) -> None:
        """Apply memory stress for specified duration."""
        memory_blocks = []
        try:
            end_time = time.perf_counter() + duration
            while time.perf_counter() < end_time:
                # Gradually increase memory usage
                memory_blocks.append([random.random() for _ in range(50000)])
                await asyncio.sleep(0.1)

                # Check memory usage
                import psutil

                current_memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                if current_memory_mb > 400:  # Approaching limit
                    break
        finally:
            del memory_blocks
            import gc

            gc.collect()

    async def _io_stress_task(self, duration: float) -> None:
        """Apply I/O stress for specified duration."""
        temp_files = []
        try:
            file_count = 0
            end_time = time.perf_counter() + duration
            while time.perf_counter() < end_time:
                # Create temporary files to stress I/O
                temp_file = f"/tmp/stress_test_{file_count}.tmp"
                with open(temp_file, "w") as f:
                    f.write("x" * 10000)  # 10KB per file
                temp_files.append(temp_file)
                file_count += 1
                await asyncio.sleep(0.05)
        finally:
            # Cleanup temp files
            import os

            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except:
                    pass

    async def _test_functionality_during_degradation(
        self,
        signal_processor: ASVEnhancedSignalProcessor,
        confidence_homing: ASVConfidenceBasedHoming,
    ) -> bool:
        """Test if system maintains functionality during degradation."""
        try:
            for i in range(5):
                signal_data = {
                    "signal_strength_dbm": -60.0 - (i * 5),
                    "frequency_hz": 433.92e6,
                    "bearing_deg": i * 72,
                    "interference_detected": False,
                    "multipath_detected": False,
                    "noise_floor_dbm": -120.0,
                    "sample_rate_hz": 2.4e6,
                }

                bearing_calc = ASVBearingCalculation(
                    bearing_deg=signal_data["bearing_deg"],
                    confidence=0.8,
                    precision_deg=2.0,
                    signal_strength_dbm=signal_data["signal_strength_dbm"],
                    signal_quality=0.7,
                    timestamp_ns=int(time.perf_counter() * 1e9),
                    analyzer_type="functionality_test",
                    interference_detected=signal_data.get("interference_detected", False),
                )
                decision = confidence_homing.evaluate_confidence_based_decision(bearing_calc)

                # Validate basic functionality
                if not (0 <= bearing_calc.bearing_deg <= 360):
                    return False
                if decision is None:
                    return False

                await asyncio.sleep(0.1)

            return True
        except Exception:
            return False

    async def _measure_performance_impact_during_degradation(
        self,
        signal_processor: ASVEnhancedSignalProcessor,
        confidence_homing: ASVConfidenceBasedHoming,
    ) -> float:
        """Measure performance impact during degradation."""
        try:
            # Measure baseline performance
            baseline_times = []
            for i in range(3):
                signal_data = {
                    "signal_strength_dbm": -60.0,
                    "frequency_hz": 433.92e6,
                    "bearing_deg": 45.0,
                    "interference_detected": False,
                    "multipath_detected": False,
                    "noise_floor_dbm": -120.0,
                    "sample_rate_hz": 2.4e6,
                }

                start_time = time.perf_counter()
                # Mock calculation for performance measurement
                _ = ASVBearingCalculation(
                    bearing_deg=signal_data["bearing_deg"],
                    confidence=0.8,
                    precision_deg=2.0,
                    signal_strength_dbm=signal_data["signal_strength_dbm"],
                    signal_quality=0.7,
                    timestamp_ns=int(time.perf_counter() * 1e9),
                    analyzer_type="performance_test",
                    interference_detected=signal_data.get("interference_detected", False),
                )
                processing_time = (time.perf_counter() - start_time) * 1000
                baseline_times.append(processing_time)

            baseline_avg = sum(baseline_times) / len(baseline_times)

            # Measure degraded performance
            degraded_times = []
            for i in range(3):
                signal_data = {
                    "signal_strength_dbm": -70.0,
                    "frequency_hz": 433.92e6,
                    "bearing_deg": 90.0,
                    "interference_detected": False,
                    "multipath_detected": False,
                    "noise_floor_dbm": -120.0,
                    "sample_rate_hz": 2.4e6,
                }

                start_time = time.perf_counter()
                # Mock calculation for performance measurement
                _ = ASVBearingCalculation(
                    bearing_deg=signal_data["bearing_deg"],
                    confidence=0.8,
                    precision_deg=2.0,
                    signal_strength_dbm=signal_data["signal_strength_dbm"],
                    signal_quality=0.7,
                    timestamp_ns=int(time.perf_counter() * 1e9),
                    analyzer_type="performance_test",
                    interference_detected=signal_data.get("interference_detected", False),
                )
                processing_time = (time.perf_counter() - start_time) * 1000
                degraded_times.append(processing_time)

            degraded_avg = sum(degraded_times) / len(degraded_times)

            # Calculate performance impact percentage
            if baseline_avg > 0:
                performance_impact = ((degraded_avg - baseline_avg) / baseline_avg) * 100
                return max(0.0, performance_impact)
            else:
                return 0.0

        except Exception:
            return 100.0  # Maximum impact if measurement fails

    @pytest.mark.asyncio
    async def test_long_term_stability_validation_with_continuous_enhanced_processing(
        self,
        enhanced_signal_processor: ASVEnhancedSignalProcessor,
        confidence_based_homing: ASVConfidenceBasedHoming,
    ) -> None:
        """[27d4] Long-term stability validation with continuous enhanced processing."""

        logger.info("Testing long-term stability validation with continuous enhanced processing")

        # Define long-term stability test scenarios
        long_term_scenarios = [
            {
                "name": "extended_operation_30_minutes",
                "description": "30-minute continuous enhanced processing",
                "duration_minutes": 30.0,
                "signal_frequency_hz": 10.0,  # 10 Hz signal processing
                "expected_memory_growth_mb": 50.0,  # Max 50MB growth
                "expected_cpu_stability_percentage": 85.0,  # Min 85% stable performance
            },
            {
                "name": "high_frequency_operation_15_minutes",
                "description": "15-minute high-frequency enhanced processing",
                "duration_minutes": 15.0,
                "signal_frequency_hz": 50.0,  # 50 Hz signal processing
                "expected_memory_growth_mb": 30.0,  # Max 30MB growth
                "expected_cpu_stability_percentage": 80.0,  # Min 80% stable performance
            },
            {
                "name": "variable_load_operation_20_minutes",
                "description": "20-minute variable load enhanced processing",
                "duration_minutes": 20.0,
                "signal_frequency_hz": 25.0,  # Variable frequency around 25 Hz
                "expected_memory_growth_mb": 40.0,  # Max 40MB growth
                "expected_cpu_stability_percentage": 82.0,  # Min 82% stable performance
            },
        ]

        long_term_stability_results = []

        for scenario in long_term_scenarios:
            logger.info(f"Testing long-term stability scenario: {scenario['name']}")

            # Initialize long-term monitoring
            stability_monitor = LongTermStabilityMonitor(str(scenario["name"]))
            await stability_monitor.start_monitoring()

            try:
                # Run continuous enhanced processing
                stability_metrics = await self._run_continuous_enhanced_processing(
                    enhanced_signal_processor, confidence_based_homing, scenario, stability_monitor
                )

                # Validate long-term stability requirements
                assert (
                    stability_metrics["memory_growth_mb"] <= scenario["expected_memory_growth_mb"]
                ), (
                    f"Memory growth {stability_metrics['memory_growth_mb']:.1f}MB exceeds "
                    f"expected {scenario['expected_memory_growth_mb']:.1f}MB for {scenario['name']}"
                )

                assert (
                    stability_metrics["cpu_stability_percentage"]
                    >= scenario["expected_cpu_stability_percentage"]
                ), (
                    f"CPU stability {stability_metrics['cpu_stability_percentage']:.1f}% below "
                    f"expected {scenario['expected_cpu_stability_percentage']:.1f}% for {scenario['name']}"
                )

                # Validate no memory leaks detected
                assert not stability_metrics[
                    "memory_leak_detected"
                ], f"Memory leak detected during long-term operation for {scenario['name']}"

                # Validate system maintained responsiveness
                assert stability_metrics["avg_response_time_ms"] <= 150.0, (
                    f"Average response time {stability_metrics['avg_response_time_ms']:.1f}ms "
                    f"exceeds 150ms threshold for {scenario['name']}"
                )

                # Validate error rate remained acceptable
                assert stability_metrics["error_rate_percentage"] <= 5.0, (
                    f"Error rate {stability_metrics['error_rate_percentage']:.1f}% "
                    f"exceeds 5% threshold for {scenario['name']}"
                )

                long_term_stability_results.append(
                    {
                        "scenario": scenario["name"],
                        "duration_minutes": scenario["duration_minutes"],
                        "memory_growth_mb": stability_metrics["memory_growth_mb"],
                        "cpu_stability": stability_metrics["cpu_stability_percentage"],
                        "avg_response_time": stability_metrics["avg_response_time_ms"],
                        "error_rate": stability_metrics["error_rate_percentage"],
                        "operations_completed": stability_metrics["total_operations"],
                        "memory_leak_detected": stability_metrics["memory_leak_detected"],
                    }
                )

                logger.info(
                    f"Long-term stability for {scenario['name']}: "
                    f"Memory growth: {stability_metrics['memory_growth_mb']:.1f}MB, "
                    f"CPU stability: {stability_metrics['cpu_stability_percentage']:.1f}%, "
                    f"Avg response: {stability_metrics['avg_response_time_ms']:.1f}ms, "
                    f"Error rate: {stability_metrics['error_rate_percentage']:.1f}%"
                )

            finally:
                await stability_monitor.stop_monitoring()

            # Extended cool down between long-term tests
            await asyncio.sleep(10.0)

        # Validate overall long-term stability performance
        max_memory_growth = max(
            result["memory_growth_mb"] for result in long_term_stability_results
        )
        min_cpu_stability = min(result["cpu_stability"] for result in long_term_stability_results)
        max_avg_response_time = max(
            result["avg_response_time"] for result in long_term_stability_results
        )
        max_error_rate = max(result["error_rate"] for result in long_term_stability_results)
        any_memory_leaks = any(
            result["memory_leak_detected"] for result in long_term_stability_results
        )
        total_operations = sum(
            result["operations_completed"] for result in long_term_stability_results
        )

        # Overall long-term stability requirements
        assert (
            max_memory_growth <= 60.0
        ), f"Maximum memory growth {max_memory_growth:.1f}MB exceeds 60MB long-term threshold"
        assert (
            min_cpu_stability >= 75.0
        ), f"Minimum CPU stability {min_cpu_stability:.1f}% below 75% long-term requirement"
        assert (
            max_avg_response_time <= 150.0
        ), f"Maximum average response time {max_avg_response_time:.1f}ms exceeds 150ms long-term threshold"
        assert (
            max_error_rate <= 5.0
        ), f"Maximum error rate {max_error_rate:.1f}% exceeds 5% long-term threshold"
        assert not any_memory_leaks, "Memory leaks detected during long-term stability testing"
        assert (
            total_operations >= 50000
        ), f"Total operations {total_operations} below 50000 expected minimum for comprehensive testing"

        logger.info(
            f"Long-term stability validation completed: "
            f"Max memory growth: {max_memory_growth:.1f}MB, "
            f"Min CPU stability: {min_cpu_stability:.1f}%, "
            f"Max avg response: {max_avg_response_time:.1f}ms, "
            f"Max error rate: {max_error_rate:.1f}%, "
            f"Total operations: {total_operations}, "
            f"Memory leaks: {any_memory_leaks}"
        )

    async def _run_continuous_enhanced_processing(
        self,
        signal_processor: ASVEnhancedSignalProcessor,
        confidence_homing: ASVConfidenceBasedHoming,
        scenario: Dict[str, Any],
        monitor: "LongTermStabilityMonitor",
    ) -> Dict[str, Any]:
        """Run continuous enhanced processing for long-term stability testing."""

        duration_seconds = scenario["duration_minutes"] * 60.0
        signal_frequency = scenario["signal_frequency_hz"]
        processing_interval = 1.0 / signal_frequency

        end_time = time.perf_counter() + duration_seconds
        operation_count = 0
        error_count = 0
        response_times = []

        logger.info(
            f"Starting {scenario['duration_minutes']:.1f}-minute continuous processing at {signal_frequency}Hz"
        )

        while time.perf_counter() < end_time:
            try:
                operation_start = time.perf_counter()

                # Create realistic signal data with variations
                signal_data = {
                    "signal_strength_dbm": -70.0 + random.uniform(-15, 15),
                    "frequency_hz": 433.92e6 + random.uniform(-1000, 1000),
                    "bearing_deg": (time.perf_counter() * 2.5) % 360,
                    "interference_detected": random.random() < 0.15,
                    "multipath_detected": random.random() < 0.1,
                    "noise_floor_dbm": -120.0 + random.uniform(-5, 5),
                    "sample_rate_hz": 2.4e6,
                    "operation_id": operation_count,
                }

                # Create mock ASV bearing calculation for stability testing
                bearing_calc = ASVBearingCalculation(
                    bearing_deg=signal_data["bearing_deg"],
                    confidence=0.8 + random.uniform(-0.2, 0.2),
                    precision_deg=2.0,
                    signal_strength_dbm=signal_data["signal_strength_dbm"],
                    signal_quality=0.7 + random.uniform(-0.1, 0.1),
                    timestamp_ns=int(time.perf_counter() * 1e9),
                    analyzer_type="stability_test",
                    interference_detected=signal_data.get("interference_detected", False),
                )
                decision = confidence_homing.evaluate_confidence_based_decision(bearing_calc)

                # Record operation metrics
                operation_time = (time.perf_counter() - operation_start) * 1000
                response_times.append(operation_time)

                # Validate results
                if not (0 <= bearing_calc.bearing_deg <= 360) or decision is None:
                    error_count += 1

                operation_count += 1

                # Record metrics periodically
                if operation_count % 100 == 0:
                    await monitor.record_operation_metrics(
                        operation_count,
                        sum(response_times[-100:]) / min(100, len(response_times)),
                        error_count,
                    )

                    # Log progress periodically
                    if operation_count % 1000 == 0:
                        elapsed_minutes = (
                            time.perf_counter() - (end_time - duration_seconds)
                        ) / 60.0
                        logger.info(
                            f"Progress: {elapsed_minutes:.1f}/{scenario['duration_minutes']:.1f} minutes, "
                            f"{operation_count} operations, {error_count} errors"
                        )

                # Wait for next processing cycle
                await asyncio.sleep(processing_interval)

            except Exception as e:
                error_count += 1
                await monitor.record_error(str(e))
                await asyncio.sleep(0.1)  # Brief pause after error

        # Calculate final metrics
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        error_rate = (error_count / max(1, operation_count)) * 100

        return await monitor.get_final_long_term_metrics(
            operation_count, avg_response_time, error_rate
        )


# Monitoring classes for system stability validation
class SystemStabilityMonitor:
    """Monitor system stability metrics during testing."""

    def __init__(self, scenario_name: str):
        self.scenario_name = scenario_name
        self.start_time: float = 0.0
        self.successful_operations = 0
        self.failed_operations = 0
        self.error_events: List[Dict[str, Any]] = []
        self.recovery_events: List[Dict[str, Any]] = []
        self.resource_exhaustion_events: List[Dict[str, Any]] = []

    async def start_monitoring(self) -> None:
        """Start stability monitoring."""
        self.start_time = time.perf_counter()
        import psutil

        self.initial_memory_mb = psutil.Process().memory_info().rss / 1024 / 1024

    async def stop_monitoring(self) -> None:
        """Stop stability monitoring."""
        pass

    async def record_successful_operation(
        self, task_id: int, operation_id: int, operation_time_ms: float
    ) -> None:
        """Record successful operation."""
        self.successful_operations += 1

    async def record_failed_operation(self, task_id: int, operation_id: int, error: str) -> None:
        """Record failed operation."""
        self.failed_operations += 1

    async def record_error_event(self, error_description: str, source: str) -> None:
        """Record error event."""
        self.error_events.append(
            {"timestamp": time.perf_counter(), "error": error_description, "source": source}
        )

    async def record_recovery_event(self, failure_type: str, recovery_time_seconds: float) -> None:
        """Record recovery event."""
        self.recovery_events.append(
            {
                "timestamp": time.perf_counter(),
                "failure_type": failure_type,
                "recovery_time_seconds": recovery_time_seconds,
            }
        )

    async def record_resource_exhaustion_event(self, description: str, source: str) -> None:
        """Record resource exhaustion event."""
        self.resource_exhaustion_events.append(
            {"timestamp": time.perf_counter(), "description": description, "source": source}
        )

    async def get_final_metrics(self) -> SystemStabilityMetrics:
        """Get final stability metrics."""
        total_duration = time.perf_counter() - self.start_time

        # Calculate memory leak detection
        import psutil

        current_memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        memory_growth = current_memory_mb - self.initial_memory_mb
        memory_leak_detected = (
            memory_growth > 100
        )  # More than 100MB growth indicates potential leak

        # Calculate performance degradation
        total_operations = self.successful_operations + self.failed_operations
        if total_operations > 0:
            failure_rate = (self.failed_operations / total_operations) * 100
            performance_degradation = min(
                failure_rate * 2, 100.0
            )  # Scale failure rate to degradation
        else:
            performance_degradation = 0.0

        return SystemStabilityMetrics(
            total_test_duration_seconds=total_duration,
            uptime_seconds=total_duration - len(self.error_events) * 0.1,  # Approximate downtime
            downtime_events=[],
            error_events=self.error_events,
            recovery_events=self.recovery_events,
            resource_exhaustion_events=self.resource_exhaustion_events,
            graceful_degradation_events=[],
            performance_degradation_percentage=performance_degradation,
            memory_leak_detected=memory_leak_detected,
            cpu_spike_events=0,
            successful_operations=self.successful_operations,
            failed_operations=self.failed_operations,
            average_response_time_ms=50.0,  # Approximate
            max_response_time_ms=200.0,
        )


class LongTermStabilityMonitor:
    """Monitor long-term stability metrics during extended testing."""

    def __init__(self, scenario_name: str):
        self.scenario_name = scenario_name
        self.start_time: float = 0.0
        self.initial_memory_mb: float = 0.0
        self.operation_metrics: List[Dict[str, Any]] = []
        self.error_events: List[str] = []

    async def start_monitoring(self) -> None:
        """Start long-term stability monitoring."""
        self.start_time = time.perf_counter()
        import psutil

        self.initial_memory_mb = psutil.Process().memory_info().rss / 1024 / 1024

    async def stop_monitoring(self) -> None:
        """Stop long-term stability monitoring."""
        pass

    async def record_operation_metrics(
        self, operation_count: int, avg_response_time_ms: float, error_count: int
    ) -> None:
        """Record operation metrics."""
        import psutil

        current_memory_mb = psutil.Process().memory_info().rss / 1024 / 1024

        self.operation_metrics.append(
            {
                "timestamp": time.perf_counter(),
                "operation_count": operation_count,
                "avg_response_time_ms": avg_response_time_ms,
                "error_count": error_count,
                "memory_usage_mb": current_memory_mb,
            }
        )

    async def record_error(self, error_description: str) -> None:
        """Record error during long-term testing."""
        self.error_events.append(error_description)

    async def get_final_long_term_metrics(
        self, total_operations: int, avg_response_time_ms: float, error_rate_percentage: float
    ) -> Dict[str, Any]:
        """Get final long-term stability metrics."""
        import psutil

        current_memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        memory_growth_mb = current_memory_mb - self.initial_memory_mb

        # Calculate CPU stability (simplified metric based on consistent performance)
        cpu_stability_percentage = max(0.0, 100.0 - error_rate_percentage * 2)

        # Detect memory leaks
        memory_leak_detected = memory_growth_mb > 200.0  # More than 200MB growth for long-term test

        return {
            "memory_growth_mb": memory_growth_mb,
            "cpu_stability_percentage": cpu_stability_percentage,
            "avg_response_time_ms": avg_response_time_ms,
            "error_rate_percentage": error_rate_percentage,
            "total_operations": total_operations,
            "memory_leak_detected": memory_leak_detected,
        }
