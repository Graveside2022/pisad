"""Safety Integration Validation Tests for Enhanced Algorithm Performance.

SUBTASK-6.2.4.2: Safety and Integration Validation - Child subtasks [27a1-27d4]

This test suite validates that enhanced algorithms preserve all safety mechanisms
and meet safety requirements under various operational conditions.

Test Coverage:
- [27a1-27a4] Safety interlock validation with enhanced algorithms active
- [27b1-27b4] Integration timing validation with other system components
- [27c1-27c4] Resource usage validation during enhanced processing
- [27d1-27d4] System stability validation under enhanced algorithm load

PRD Requirements Validated:
- PRD-FR16: <500ms emergency stop response time
- PRD-NFR3: Safety authority override mechanisms
- PRD-NFR4: System stability under load
- PRD-NFR5: Resource usage constraints
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

from src.backend.services.asv_integration.asv_confidence_based_homing import (
    ASVConfidenceBasedHoming,
    DynamicThresholdConfig,
)
from src.backend.services.asv_integration.asv_enhanced_signal_processor import (
    ASVBearingCalculation,
    ASVEnhancedSignalProcessor,
)
from src.backend.services.emergency_stop_coordinator import EmergencyStopCoordinator
from src.backend.services.safety_authority_manager import (
    SafetyAuthorityLevel,
    SafetyAuthorityManager,
)

logger = logging.getLogger(__name__)


@dataclass
class SafetyTestScenario:
    """Test scenario for safety integration validation."""

    name: str
    description: str
    enhanced_algorithms_active: bool
    expected_emergency_stop_time_ms: float
    expected_safety_override_response_ms: float
    system_load_factor: float  # 0.0-1.0 representing system load
    concurrent_operations: int


@dataclass
class SafetyValidationMetrics:
    """Metrics collected during safety validation tests."""

    emergency_stop_time_ms: float
    safety_override_response_ms: float
    enhanced_processing_impact_ms: float
    resource_usage_percent: Dict[str, float]
    interlock_response_times_ms: List[float]
    authority_validation_time_ms: float


class TestEnhancedAlgorithmSafetyIntegrationValidation:
    """Test suite for validating safety integration with enhanced algorithms."""

    @pytest.fixture
    async def safety_authority_manager(self):
        """Create safety authority manager for testing."""
        manager = SafetyAuthorityManager()
        yield manager
        await manager.shutdown()

    @pytest.fixture
    async def emergency_stop_coordinator(self, safety_authority_manager):
        """Create emergency stop coordinator for safety testing."""
        coordinator = EmergencyStopCoordinator()
        # Configure with safety authority manager
        coordinator.set_safety_authority_manager(safety_authority_manager)
        yield coordinator
        await coordinator.shutdown()

    @pytest.fixture
    async def enhanced_signal_processor(self):
        """Create ASV enhanced signal processor for testing."""
        processor = ASVEnhancedSignalProcessor()
        await processor.initialize()
        yield processor
        await processor.shutdown()

    @pytest.fixture
    async def confidence_based_homing(self, enhanced_signal_processor):
        """Create confidence-based homing system for testing."""
        homing = ASVConfidenceBasedHoming(
            asv_processor=enhanced_signal_processor, threshold_config=DynamicThresholdConfig()
        )
        yield homing

    @pytest.fixture
    def safety_test_scenarios(self) -> List[SafetyTestScenario]:
        """Generate comprehensive safety test scenarios."""
        return [
            SafetyTestScenario(
                name="baseline_safety_with_enhanced_inactive",
                description="Baseline safety response with enhanced algorithms inactive",
                enhanced_algorithms_active=False,
                expected_emergency_stop_time_ms=400.0,  # Well under 500ms requirement
                expected_safety_override_response_ms=100.0,
                system_load_factor=0.3,
                concurrent_operations=1,
            ),
            SafetyTestScenario(
                name="safety_with_enhanced_active_low_load",
                description="Safety response with enhanced algorithms active, low system load",
                enhanced_algorithms_active=True,
                expected_emergency_stop_time_ms=450.0,  # Allow slight increase but <500ms
                expected_safety_override_response_ms=120.0,
                system_load_factor=0.4,
                concurrent_operations=2,
            ),
            SafetyTestScenario(
                name="safety_with_enhanced_active_high_load",
                description="Safety response with enhanced algorithms active, high system load",
                enhanced_algorithms_active=True,
                expected_emergency_stop_time_ms=480.0,  # Maximum acceptable under load
                expected_safety_override_response_ms=150.0,
                system_load_factor=0.8,
                concurrent_operations=4,
            ),
            SafetyTestScenario(
                name="safety_during_intensive_asv_processing",
                description="Safety response during intensive ASV signal processing",
                enhanced_algorithms_active=True,
                expected_emergency_stop_time_ms=470.0,
                expected_safety_override_response_ms=130.0,
                system_load_factor=0.7,
                concurrent_operations=3,
            ),
            SafetyTestScenario(
                name="safety_with_confidence_fallbacks_active",
                description="Safety response while confidence-based fallbacks are active",
                enhanced_algorithms_active=True,
                expected_emergency_stop_time_ms=460.0,
                expected_safety_override_response_ms=140.0,
                system_load_factor=0.6,
                concurrent_operations=3,
            ),
        ]

    @pytest.mark.asyncio
    async def test_emergency_stop_timing_with_enhanced_algorithms_active(
        self,
        emergency_stop_coordinator,
        enhanced_signal_processor,
        confidence_based_homing,
        safety_test_scenarios,
    ):
        """[27a1] Verify enhanced algorithms don't impact <500ms emergency stop requirement (PRD-FR16)."""

        logger.info("Starting emergency stop timing validation with enhanced algorithms")

        validation_results = []

        for scenario in safety_test_scenarios:
            logger.info(f"Testing scenario: {scenario.name}")

            # Configure enhanced algorithms based on scenario
            if scenario.enhanced_algorithms_active:
                await self._activate_enhanced_algorithms(
                    enhanced_signal_processor, confidence_based_homing, scenario.system_load_factor
                )

            # Simulate system load
            load_tasks = []
            if scenario.concurrent_operations > 1:
                load_tasks = await self._simulate_system_load(
                    scenario.concurrent_operations - 1, scenario.system_load_factor
                )

            try:
                # Measure emergency stop response time
                metrics = await self._measure_emergency_stop_response(
                    emergency_stop_coordinator, scenario
                )

                # Validate emergency stop timing requirement
                assert metrics.emergency_stop_time_ms <= 500.0, (
                    f"Emergency stop time {metrics.emergency_stop_time_ms}ms exceeds "
                    f"500ms requirement for scenario {scenario.name}"
                )

                # Validate expected performance for scenario
                assert metrics.emergency_stop_time_ms <= scenario.expected_emergency_stop_time_ms, (
                    f"Emergency stop time {metrics.emergency_stop_time_ms}ms exceeds "
                    f"expected {scenario.expected_emergency_stop_time_ms}ms for scenario {scenario.name}"
                )

                validation_results.append(
                    {
                        "scenario": scenario.name,
                        "emergency_stop_time_ms": metrics.emergency_stop_time_ms,
                        "enhanced_active": scenario.enhanced_algorithms_active,
                        "system_load": scenario.system_load_factor,
                        "validation_passed": True,
                    }
                )

                logger.info(
                    f"Scenario {scenario.name}: Emergency stop in {metrics.emergency_stop_time_ms:.1f}ms "
                    f"(enhanced={'active' if scenario.enhanced_algorithms_active else 'inactive'})"
                )

            finally:
                # Clean up load simulation tasks
                for task in load_tasks:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        # Verify that enhanced algorithms don't significantly degrade emergency stop timing
        baseline_time = next(
            r["emergency_stop_time_ms"] for r in validation_results if not r["enhanced_active"]
        )
        enhanced_times = [
            r["emergency_stop_time_ms"] for r in validation_results if r["enhanced_active"]
        ]

        max_enhanced_time = max(enhanced_times)
        timing_degradation = max_enhanced_time - baseline_time

        # Allow up to 80ms additional latency for enhanced algorithms (16% of 500ms requirement)
        assert timing_degradation <= 80.0, (
            f"Enhanced algorithms cause {timing_degradation:.1f}ms timing degradation, "
            f"exceeding acceptable 80ms threshold"
        )

        logger.info(
            f"Emergency stop validation completed: "
            f"Baseline: {baseline_time:.1f}ms, "
            f"Enhanced max: {max_enhanced_time:.1f}ms, "
            f"Degradation: {timing_degradation:.1f}ms"
        )

    async def _activate_enhanced_algorithms(
        self,
        signal_processor: ASVEnhancedSignalProcessor,
        confidence_homing: ASVConfidenceBasedHoming,
        load_factor: float,
    ) -> None:
        """Activate enhanced algorithms with specified load factor."""
        # Simulate enhanced processing activation
        await signal_processor.configure_enhanced_processing(enabled=True)

        # Configure confidence-based homing with appropriate thresholds
        config = DynamicThresholdConfig(
            high_quality_threshold=0.2 + (load_factor * 0.1),
            moderate_quality_threshold=0.5 + (load_factor * 0.1),
            low_quality_threshold=0.7 + (load_factor * 0.1),
        )
        confidence_homing.configure_threshold_parameters(config)

    async def _simulate_system_load(
        self, concurrent_operations: int, load_factor: float
    ) -> List[asyncio.Task]:
        """Simulate system load with concurrent operations."""
        load_tasks = []

        for i in range(concurrent_operations):
            # Create CPU-intensive tasks that simulate enhanced processing load
            task = asyncio.create_task(self._cpu_intensive_simulation(load_factor, i))
            load_tasks.append(task)

        # Allow tasks to start and establish load
        await asyncio.sleep(0.05)

        return load_tasks

    async def _cpu_intensive_simulation(self, load_factor: float, task_id: int) -> None:
        """Simulate CPU-intensive enhanced algorithm processing."""
        base_duration = 0.1 * load_factor  # Base load duration

        while True:
            # Simulate ASV analysis computation
            start_time = time.perf_counter()
            iterations = int(10000 * load_factor)  # Scale iterations with load

            # CPU-bound simulation
            for _ in range(iterations):
                # Simulate signal processing calculations
                result = sum(i * 0.001 for i in range(100))

            # Ensure minimum processing time
            elapsed = time.perf_counter() - start_time
            if elapsed < base_duration:
                await asyncio.sleep(base_duration - elapsed)

            # Yield control periodically
            await asyncio.sleep(0.001)

    async def _measure_emergency_stop_response(
        self, emergency_stop_coordinator: EmergencyStopCoordinator, scenario: SafetyTestScenario
    ) -> SafetyValidationMetrics:
        """Measure emergency stop response time with detailed metrics."""

        # Reset coordinator to known state
        await emergency_stop_coordinator.reset_emergency_state()

        # Start timing measurement
        start_time = time.perf_counter()

        # Trigger emergency stop
        emergency_reason = f"Safety validation test: {scenario.name}"
        await emergency_stop_coordinator.trigger_emergency_stop(emergency_reason)

        # Measure time until emergency stop is fully active
        while not await emergency_stop_coordinator.is_emergency_active():
            await asyncio.sleep(0.001)  # 1ms polling interval

        emergency_stop_time_ms = (time.perf_counter() - start_time) * 1000

        # Measure safety override response
        override_start = time.perf_counter()
        await emergency_stop_coordinator.request_safety_override(
            SafetyAuthorityLevel.EMERGENCY, "Test safety override"
        )
        safety_override_response_ms = (time.perf_counter() - override_start) * 1000

        # Collect resource usage metrics
        resource_metrics = await self._collect_resource_metrics()

        return SafetyValidationMetrics(
            emergency_stop_time_ms=emergency_stop_time_ms,
            safety_override_response_ms=safety_override_response_ms,
            enhanced_processing_impact_ms=0.0,  # Will be calculated by caller
            resource_usage_percent=resource_metrics,
            interlock_response_times_ms=[emergency_stop_time_ms],
            authority_validation_time_ms=safety_override_response_ms,
        )

    async def _collect_resource_metrics(self) -> Dict[str, float]:
        """Collect system resource usage metrics."""
        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())

            # CPU usage (percentage)
            cpu_percent = process.cpu_percent(interval=0.1)

            # Memory usage (percentage of system memory)
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()

            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_mb": memory_info.rss / 1024 / 1024,
            }

        except ImportError:
            logger.warning("psutil not available for resource metrics")
            return {"cpu_percent": 0.0, "memory_percent": 0.0, "memory_mb": 0.0}

    @pytest.mark.asyncio
    async def test_safety_interlock_preservation_during_enhanced_processing(
        self, safety_authority_manager, enhanced_signal_processor, confidence_based_homing
    ):
        """[27a2] Test safety interlock preservation during enhanced signal processing."""

        logger.info("Testing safety interlock preservation during enhanced processing")

        # Activate enhanced algorithms
        await enhanced_signal_processor.configure_enhanced_processing(enabled=True)

        # Test safety interlocks under various enhanced processing scenarios
        test_scenarios = [
            {
                "name": "intensive_asv_analysis",
                "signal_strength": -70.0,
                "interference_present": True,
                "processing_complexity": "high",
            },
            {
                "name": "confidence_based_fallback_active",
                "signal_strength": -90.0,
                "interference_present": False,
                "processing_complexity": "medium",
            },
            {
                "name": "adaptive_search_pattern_active",
                "signal_strength": -60.0,
                "interference_present": True,
                "processing_complexity": "high",
            },
        ]

        interlock_response_times = []

        for scenario in test_scenarios:
            logger.info(f"Testing interlock preservation: {scenario['name']}")

            # Simulate enhanced processing scenario
            processing_task = asyncio.create_task(
                self._simulate_enhanced_processing_scenario(
                    enhanced_signal_processor, confidence_based_homing, scenario
                )
            )

            # Wait for processing to stabilize
            await asyncio.sleep(0.1)

            # Test safety interlock response during processing
            interlock_start = time.perf_counter()

            # Trigger safety interlock
            await safety_authority_manager.request_emergency_authority(
                SafetyAuthorityLevel.CRITICAL, f"Interlock test during {scenario['name']}"
            )

            # Verify interlock is immediately effective
            assert (
                await safety_authority_manager.is_emergency_authority_active()
            ), f"Safety interlock not immediately active during {scenario['name']}"

            interlock_time_ms = (time.perf_counter() - interlock_start) * 1000
            interlock_response_times.append(interlock_time_ms)

            # Verify enhanced processing is properly suspended
            assert await self._verify_enhanced_processing_suspended(
                enhanced_signal_processor
            ), f"Enhanced processing not properly suspended during interlock for {scenario['name']}"

            # Clean up
            processing_task.cancel()
            try:
                await processing_task
            except asyncio.CancelledError:
                pass

            # Reset safety state
            await safety_authority_manager.release_emergency_authority()
            await asyncio.sleep(0.05)  # Allow state to reset

            logger.info(
                f"Interlock preserved for {scenario['name']}: {interlock_time_ms:.1f}ms response"
            )

        # Validate all interlock response times are acceptable
        max_interlock_time = max(interlock_response_times)
        avg_interlock_time = sum(interlock_response_times) / len(interlock_response_times)

        # Safety interlocks should respond faster than emergency stop (within 200ms)
        assert (
            max_interlock_time <= 200.0
        ), f"Safety interlock response time {max_interlock_time:.1f}ms exceeds 200ms threshold"

        assert (
            avg_interlock_time <= 100.0
        ), f"Average safety interlock response time {avg_interlock_time:.1f}ms exceeds 100ms threshold"

        logger.info(
            f"Safety interlock preservation validated: "
            f"Max: {max_interlock_time:.1f}ms, Avg: {avg_interlock_time:.1f}ms"
        )

    async def _simulate_enhanced_processing_scenario(
        self,
        signal_processor: ASVEnhancedSignalProcessor,
        confidence_homing: ASVConfidenceBasedHoming,
        scenario: Dict[str, Any],
    ) -> None:
        """Simulate enhanced processing scenario for interlock testing."""

        # Create mock signal data based on scenario
        signal_data = self._create_mock_signal_data(
            signal_strength=scenario["signal_strength"],
            interference_present=scenario["interference_present"],
        )

        complexity_iterations = {"high": 100, "medium": 50, "low": 25}

        iterations = complexity_iterations.get(scenario["processing_complexity"], 50)

        for i in range(iterations):
            # Simulate intensive ASV analysis
            bearing_calc = await signal_processor.calculate_enhanced_bearing(signal_data)

            # Simulate confidence-based decision making
            decision = confidence_homing.evaluate_confidence_based_decision(bearing_calc)

            # Add processing delay to simulate real workload
            await asyncio.sleep(0.01)  # 10ms processing per iteration

    def _create_mock_signal_data(
        self, signal_strength: float, interference_present: bool
    ) -> Dict[str, Any]:
        """Create mock signal data for testing scenarios."""
        return {
            "signal_strength_dbm": signal_strength,
            "frequency_hz": 433.92e6,
            "bearing_deg": 45.0,
            "interference_detected": interference_present,
            "noise_floor_dbm": -120.0,
            "sample_rate_hz": 2.4e6,
        }

    async def _verify_enhanced_processing_suspended(
        self, signal_processor: ASVEnhancedSignalProcessor
    ) -> bool:
        """Verify that enhanced processing is properly suspended during safety interlock."""
        # Check if enhanced processing responds to safety interlock
        try:
            # This should either complete very quickly (if suspended) or timeout
            processing_status = await asyncio.wait_for(
                signal_processor.get_processing_status(),
                timeout=0.05,  # 50ms timeout
            )

            # If we get here, check that processing is in safe/suspended state
            return processing_status.get("suspended_for_safety", False)

        except asyncio.TimeoutError:
            # If processing is hanging, it's not properly responding to interlocks
            logger.error("Enhanced processing not responding to safety interlock")
            return False

    @pytest.mark.asyncio
    async def test_safety_authority_override_mechanisms_with_asv_integration_active(
        self, safety_authority_manager, enhanced_signal_processor, confidence_based_homing
    ):
        """[27a3] Validate safety authority override mechanisms with ASV integration active."""

        logger.info("Testing safety authority override mechanisms with ASV integration")

        # Activate enhanced ASV integration
        await enhanced_signal_processor.configure_enhanced_processing(enabled=True)

        # Test various safety authority levels and override scenarios
        authority_test_scenarios = [
            {
                "level": SafetyAuthorityLevel.COMMUNICATION,
                "expected_response_ms": 50.0,
                "description": "Communication level override during ASV processing",
            },
            {
                "level": SafetyAuthorityLevel.NAVIGATION,
                "expected_response_ms": 75.0,
                "description": "Navigation level override during enhanced homing",
            },
            {
                "level": SafetyAuthorityLevel.CRITICAL,
                "expected_response_ms": 100.0,
                "description": "Critical level override during confidence decisions",
            },
            {
                "level": SafetyAuthorityLevel.EMERGENCY,
                "expected_response_ms": 25.0,
                "description": "Emergency level override (highest priority)",
            },
        ]

        override_response_times = {}

        for scenario in authority_test_scenarios:
            logger.info(f"Testing authority override: {scenario['description']}")

            # Start enhanced processing to create realistic load
            processing_task = asyncio.create_task(
                self._continuous_enhanced_processing(
                    enhanced_signal_processor, confidence_based_homing
                )
            )

            await asyncio.sleep(0.05)  # Let processing establish

            # Measure authority override response time
            override_start = time.perf_counter()

            await safety_authority_manager.request_emergency_authority(
                scenario["level"], f"Test override: {scenario['description']}"
            )

            # Verify authority is immediately active
            while not await safety_authority_manager.is_emergency_authority_active():
                await asyncio.sleep(0.001)

            override_time_ms = (time.perf_counter() - override_start) * 1000
            override_response_times[scenario["level"]] = override_time_ms

            # Verify enhanced processing respects authority
            assert await self._verify_authority_respected(
                enhanced_signal_processor, scenario["level"]
            ), f"Enhanced processing not respecting {scenario['level']} authority"

            # Verify confidence decisions are suspended
            test_bearing = ASVBearingCalculation(
                bearing_deg=90.0,
                confidence=0.8,
                precision_deg=1.5,
                signal_strength_dbm=-65.0,
                interference_detected=False,
                signal_quality=0.9,
            )

            decision = confidence_based_homing.evaluate_confidence_based_decision(test_bearing)

            # During safety authority override, should not proceed with homing
            assert (
                not decision.proceed_with_homing
            ), f"Confidence decisions not respecting {scenario['level']} authority override"
            assert (
                decision.safety_override_reason is not None
            ), f"Safety override reason not properly set for {scenario['level']}"

            # Clean up
            processing_task.cancel()
            try:
                await processing_task
            except asyncio.CancelledError:
                pass

            await safety_authority_manager.release_emergency_authority()
            await asyncio.sleep(0.02)

            # Validate response time meets requirements
            assert override_time_ms <= scenario["expected_response_ms"], (
                f"{scenario['level']} authority override took {override_time_ms:.1f}ms, "
                f"expected <= {scenario['expected_response_ms']}ms"
            )

            logger.info(
                f"Authority override validated for {scenario['level']}: {override_time_ms:.1f}ms"
            )

        # Validate authority hierarchy (Emergency should be fastest)
        emergency_time = override_response_times[SafetyAuthorityLevel.EMERGENCY]
        communication_time = override_response_times[SafetyAuthorityLevel.COMMUNICATION]

        assert emergency_time <= communication_time, (
            f"Emergency authority ({emergency_time:.1f}ms) not faster than "
            f"Communication authority ({communication_time:.1f}ms)"
        )

        max_override_time = max(override_response_times.values())
        assert (
            max_override_time <= 100.0
        ), f"Maximum authority override time {max_override_time:.1f}ms exceeds 100ms threshold"

    async def _continuous_enhanced_processing(
        self,
        signal_processor: ASVEnhancedSignalProcessor,
        confidence_homing: ASVConfidenceBasedHoming,
    ) -> None:
        """Continuously run enhanced processing for authority testing."""

        signal_data = {
            "signal_strength_dbm": -75.0,
            "frequency_hz": 433.92e6,
            "bearing_deg": 135.0,
            "interference_detected": False,
            "noise_floor_dbm": -120.0,
            "sample_rate_hz": 2.4e6,
        }

        while True:
            bearing_calc = await signal_processor.calculate_enhanced_bearing(signal_data)
            decision = confidence_homing.evaluate_confidence_based_decision(bearing_calc)
            await asyncio.sleep(0.02)  # 20ms processing cycle

    async def _verify_authority_respected(
        self, signal_processor: ASVEnhancedSignalProcessor, authority_level: SafetyAuthorityLevel
    ) -> bool:
        """Verify enhanced processing properly respects safety authority."""
        try:
            # Try to get processing status with timeout
            status = await asyncio.wait_for(signal_processor.get_processing_status(), timeout=0.1)

            # Check if processing is in authority-compliant state
            authority_compliant = status.get("authority_level_respected", False)
            current_authority = status.get("current_authority_level", None)

            return authority_compliant and current_authority == authority_level.value

        except asyncio.TimeoutError:
            logger.error(f"Enhanced processing not responding to {authority_level} authority")
            return False

    @pytest.mark.asyncio
    async def test_emergency_fallback_to_basic_homing_when_enhanced_algorithms_fail(
        self, enhanced_signal_processor, confidence_based_homing
    ):
        """[27a4] Test emergency fallback to basic homing when enhanced algorithms fail."""

        logger.info("Testing emergency fallback to basic homing")

        # Test various failure scenarios that should trigger fallback
        failure_scenarios = [
            {
                "name": "asv_processor_exception",
                "description": "ASV signal processor throws exception",
                "failure_type": "processor_exception",
            },
            {
                "name": "confidence_decision_timeout",
                "description": "Confidence decision taking too long",
                "failure_type": "decision_timeout",
            },
            {
                "name": "critical_resource_exhaustion",
                "description": "System resources critically low",
                "failure_type": "resource_exhaustion",
            },
            {
                "name": "enhanced_algorithm_corruption",
                "description": "Enhanced algorithm outputs corrupted",
                "failure_type": "output_corruption",
            },
        ]

        fallback_response_times = []

        for scenario in failure_scenarios:
            logger.info(f"Testing fallback scenario: {scenario['name']}")

            # Simulate the specific failure condition
            await self._simulate_enhanced_algorithm_failure(
                enhanced_signal_processor, confidence_based_homing, scenario["failure_type"]
            )

            # Measure time to detect failure and switch to basic homing
            fallback_start = time.perf_counter()

            # Create test signal for processing
            test_signal_data = {
                "signal_strength_dbm": -80.0,
                "frequency_hz": 433.92e6,
                "bearing_deg": 270.0,
                "interference_detected": False,
                "noise_floor_dbm": -120.0,
                "sample_rate_hz": 2.4e6,
            }

            # Attempt enhanced processing (should fail and fallback)
            try:
                bearing_result = await asyncio.wait_for(
                    signal_processor.calculate_enhanced_bearing(test_signal_data),
                    timeout=2.0,  # Allow time for failure detection and fallback
                )

                # Check if result indicates fallback was used
                assert (
                    bearing_result.processing_method == "basic_fallback"
                ), f"Failed to fallback to basic processing for {scenario['name']}"

                fallback_time_ms = (time.perf_counter() - fallback_start) * 1000
                fallback_response_times.append(fallback_time_ms)

                # Verify fallback produces valid results
                assert (
                    0 <= bearing_result.bearing_deg <= 360
                ), f"Basic fallback produced invalid bearing: {bearing_result.bearing_deg}°"

                assert (
                    bearing_result.confidence > 0.0
                ), f"Basic fallback produced zero confidence: {bearing_result.confidence}"

                # Verify fallback is responsive (should complete within 500ms)
                assert (
                    fallback_time_ms <= 500.0
                ), f"Fallback response took {fallback_time_ms:.1f}ms, exceeds 500ms threshold"

                logger.info(
                    f"Fallback successful for {scenario['name']}: {fallback_time_ms:.1f}ms, "
                    f"bearing={bearing_result.bearing_deg:.1f}°, confidence={bearing_result.confidence:.2f}"
                )

            except asyncio.TimeoutError:
                pytest.fail(
                    f"Enhanced algorithm failure did not trigger fallback within timeout for {scenario['name']}"
                )

            # Reset system state for next test
            await self._reset_enhanced_algorithm_state(
                enhanced_signal_processor, confidence_based_homing
            )
            await asyncio.sleep(0.1)

        # Validate fallback performance across all scenarios
        max_fallback_time = max(fallback_response_times)
        avg_fallback_time = sum(fallback_response_times) / len(fallback_response_times)

        # Fallback should be fast to maintain system responsiveness
        assert (
            max_fallback_time <= 300.0
        ), f"Maximum fallback time {max_fallback_time:.1f}ms exceeds 300ms threshold"

        assert (
            avg_fallback_time <= 200.0
        ), f"Average fallback time {avg_fallback_time:.1f}ms exceeds 200ms threshold"

        logger.info(
            f"Emergency fallback validation completed: "
            f"Max: {max_fallback_time:.1f}ms, Avg: {avg_fallback_time:.1f}ms"
        )

    async def _simulate_enhanced_algorithm_failure(
        self,
        signal_processor: ASVEnhancedSignalProcessor,
        confidence_homing: ASVConfidenceBasedHoming,
        failure_type: str,
    ) -> None:
        """Simulate various types of enhanced algorithm failures."""

        if failure_type == "processor_exception":
            # Inject a failure condition that will cause processor to throw exception
            await signal_processor.inject_test_failure("calculation_exception")

        elif failure_type == "decision_timeout":
            # Configure confidence homing to simulate slow decision making
            await confidence_homing.configure_test_latency(timeout_ms=1000)

        elif failure_type == "resource_exhaustion":
            # Simulate resource exhaustion condition
            await signal_processor.inject_test_failure("memory_exhaustion")

        elif failure_type == "output_corruption":
            # Inject output corruption that should trigger fallback
            await signal_processor.inject_test_failure("output_corruption")

    async def _reset_enhanced_algorithm_state(
        self,
        signal_processor: ASVEnhancedSignalProcessor,
        confidence_homing: ASVConfidenceBasedHoming,
    ) -> None:
        """Reset enhanced algorithm state after failure testing."""
        await signal_processor.clear_test_failures()
        await confidence_homing.reset_test_configuration()
        await signal_processor.reinitialize_enhanced_processing()

    # ================================================================================
    # [27b1-27b4] Integration Timing Validation with Other System Components
    # ================================================================================

    @pytest.mark.asyncio
    async def test_integration_timing_validation_with_mavlink_service_during_enhanced_processing(
        self, enhanced_signal_processor, confidence_based_homing
    ):
        """[27b1] Integration timing validation with mavlink_service during enhanced processing."""

        logger.info("Testing integration timing validation with MAVLink service")

        # Import MAVLink service for integration testing
        from src.backend.services.mavlink_service import MAVLinkService

        # Initialize MAVLink service
        mavlink_service = MAVLinkService()
        await mavlink_service.initialize()

        try:
            # Configure enhanced processing
            await enhanced_signal_processor.configure_enhanced_processing(enabled=True)

            # Test various integration scenarios with timing measurement
            integration_scenarios = [
                {
                    "name": "mavlink_homing_commands_with_enhanced_active",
                    "description": "MAVLink homing commands while enhanced algorithms active",
                    "command_frequency_hz": 10.0,
                    "test_duration_seconds": 5.0,
                    "expected_max_latency_ms": 50.0,
                },
                {
                    "name": "mavlink_status_updates_during_asv_processing",
                    "description": "MAVLink status updates during intensive ASV processing",
                    "command_frequency_hz": 20.0,
                    "test_duration_seconds": 3.0,
                    "expected_max_latency_ms": 75.0,
                },
                {
                    "name": "mavlink_emergency_commands_with_confidence_decisions",
                    "description": "MAVLink emergency commands with confidence-based decisions active",
                    "command_frequency_hz": 5.0,
                    "test_duration_seconds": 2.0,
                    "expected_max_latency_ms": 30.0,
                },
            ]

            integration_timing_results = []

            for scenario in integration_scenarios:
                logger.info(f"Testing integration scenario: {scenario['name']}")

                # Start enhanced processing load
                processing_task = asyncio.create_task(
                    self._sustained_enhanced_processing(
                        enhanced_signal_processor,
                        confidence_based_homing,
                        duration_seconds=scenario["test_duration_seconds"],
                    )
                )

                # Measure MAVLink service response times during enhanced processing
                command_latencies = await self._measure_mavlink_integration_timing(
                    mavlink_service,
                    scenario["command_frequency_hz"],
                    scenario["test_duration_seconds"],
                )

                # Wait for processing task to complete
                await processing_task

                # Analyze timing results
                max_latency = max(command_latencies)
                avg_latency = sum(command_latencies) / len(command_latencies)
                p95_latency = sorted(command_latencies)[int(len(command_latencies) * 0.95)]

                # Validate timing requirements
                assert max_latency <= scenario["expected_max_latency_ms"], (
                    f"MAVLink integration max latency {max_latency:.1f}ms exceeds "
                    f"expected {scenario['expected_max_latency_ms']}ms for {scenario['name']}"
                )

                integration_timing_results.append(
                    {
                        "scenario": scenario["name"],
                        "max_latency_ms": max_latency,
                        "avg_latency_ms": avg_latency,
                        "p95_latency_ms": p95_latency,
                        "command_count": len(command_latencies),
                    }
                )

                logger.info(
                    f"MAVLink integration timing for {scenario['name']}: "
                    f"Max: {max_latency:.1f}ms, Avg: {avg_latency:.1f}ms, "
                    f"P95: {p95_latency:.1f}ms ({len(command_latencies)} commands)"
                )

                await asyncio.sleep(0.2)  # Cool down between scenarios

            # Validate overall integration timing performance
            overall_max_latency = max(
                result["max_latency_ms"] for result in integration_timing_results
            )
            overall_avg_latency = sum(
                result["avg_latency_ms"] for result in integration_timing_results
            ) / len(integration_timing_results)

            # MAVLink integration should remain responsive during enhanced processing
            assert (
                overall_max_latency <= 75.0
            ), f"Overall MAVLink integration max latency {overall_max_latency:.1f}ms exceeds 75ms threshold"

            assert (
                overall_avg_latency <= 40.0
            ), f"Overall MAVLink integration avg latency {overall_avg_latency:.1f}ms exceeds 40ms threshold"

            logger.info(
                f"MAVLink integration validation completed: "
                f"Overall max: {overall_max_latency:.1f}ms, Overall avg: {overall_avg_latency:.1f}ms"
            )

        finally:
            await mavlink_service.shutdown()

    async def _sustained_enhanced_processing(
        self,
        signal_processor: ASVEnhancedSignalProcessor,
        confidence_homing: ASVConfidenceBasedHoming,
        duration_seconds: float,
    ) -> None:
        """Run sustained enhanced processing for integration testing."""

        end_time = time.perf_counter() + duration_seconds

        signal_variations = [
            {"strength": -60.0, "bearing": 0.0, "interference": False},
            {"strength": -80.0, "bearing": 90.0, "interference": True},
            {"strength": -70.0, "bearing": 180.0, "interference": False},
            {"strength": -90.0, "bearing": 270.0, "interference": True},
        ]

        variation_index = 0

        while time.perf_counter() < end_time:
            signal_data = {
                "signal_strength_dbm": signal_variations[variation_index]["strength"],
                "frequency_hz": 433.92e6,
                "bearing_deg": signal_variations[variation_index]["bearing"],
                "interference_detected": signal_variations[variation_index]["interference"],
                "noise_floor_dbm": -120.0,
                "sample_rate_hz": 2.4e6,
            }

            # Intensive processing cycle
            bearing_calc = await signal_processor.calculate_enhanced_bearing(signal_data)
            decision = confidence_homing.evaluate_confidence_based_decision(bearing_calc)

            # Rotate through signal variations
            variation_index = (variation_index + 1) % len(signal_variations)

            # Processing frequency ~50Hz
            await asyncio.sleep(0.02)

    async def _measure_mavlink_integration_timing(
        self, mavlink_service, command_frequency_hz: float, duration_seconds: float
    ) -> List[float]:
        """Measure MAVLink service integration timing during enhanced processing."""

        command_interval = 1.0 / command_frequency_hz
        end_time = time.perf_counter() + duration_seconds
        command_latencies = []

        command_types = [
            "set_position_target",
            "get_vehicle_status",
            "set_mode",
            "arm_disarm",
            "get_attitude",
        ]

        command_index = 0

        while time.perf_counter() < end_time:
            # Measure command response time
            command_start = time.perf_counter()

            command_type = command_types[command_index % len(command_types)]

            try:
                # Execute MAVLink command (with timeout to prevent hanging)
                await asyncio.wait_for(
                    self._execute_mavlink_command(mavlink_service, command_type),
                    timeout=0.5,  # 500ms timeout per command
                )

                command_latency_ms = (time.perf_counter() - command_start) * 1000
                command_latencies.append(command_latency_ms)

            except asyncio.TimeoutError:
                # Command timed out - record maximum penalty latency
                command_latencies.append(500.0)  # 500ms penalty
                logger.warning(f"MAVLink command {command_type} timed out")

            command_index += 1

            # Wait for next command interval
            await asyncio.sleep(command_interval)

        return command_latencies

    async def _execute_mavlink_command(self, mavlink_service, command_type: str) -> Any:
        """Execute specific MAVLink command for timing measurement."""

        if command_type == "set_position_target":
            return await mavlink_service.set_position_target(lat=37.7749, lon=-122.4194, alt=10.0)
        elif command_type == "get_vehicle_status":
            return await mavlink_service.get_vehicle_status()
        elif command_type == "set_mode":
            return await mavlink_service.set_mode("GUIDED")
        elif command_type == "arm_disarm":
            return await mavlink_service.arm_disarm(arm=True)
        elif command_type == "get_attitude":
            return await mavlink_service.get_attitude()
        else:
            # Default command
            return await mavlink_service.heartbeat()

    @pytest.mark.asyncio
    async def test_hackrf_one_sdr_coordination_timing_with_enhanced_algorithms_active(
        self, enhanced_signal_processor, confidence_based_homing
    ):
        """[27b2] HackRF One SDR coordination timing with enhanced algorithms active."""

        logger.info("Testing HackRF One SDR coordination timing with enhanced algorithms")

        # Import HackRF coordinator for integration testing
        from src.backend.services.asv_integration.asv_hackrf_coordinator import ASVHackRFCoordinator

        # Initialize HackRF coordinator
        hackrf_coordinator = ASVHackRFCoordinator()
        await hackrf_coordinator.initialize()

        try:
            # Configure enhanced algorithms
            await enhanced_signal_processor.configure_enhanced_processing(enabled=True)

            # Test SDR coordination scenarios with enhanced algorithms active
            sdr_coordination_scenarios = [
                {
                    "name": "frequency_sweeping_with_enhanced_analysis",
                    "description": "HackRF frequency sweeping with enhanced signal analysis",
                    "sweep_range_mhz": [433.0, 435.0],
                    "sweep_step_khz": 100.0,
                    "expected_coordination_latency_ms": 20.0,
                },
                {
                    "name": "gain_adjustment_during_asv_processing",
                    "description": "Dynamic gain adjustment during intensive ASV processing",
                    "gain_range_db": [0, 40],
                    "adjustment_frequency_hz": 5.0,
                    "expected_coordination_latency_ms": 15.0,
                },
                {
                    "name": "sample_rate_changes_with_confidence_decisions",
                    "description": "Sample rate changes coordinated with confidence decisions",
                    "sample_rates_hz": [1e6, 2.4e6, 5e6],
                    "change_frequency_hz": 2.0,
                    "expected_coordination_latency_ms": 30.0,
                },
            ]

            sdr_coordination_results = []

            for scenario in sdr_coordination_scenarios:
                logger.info(f"Testing SDR coordination: {scenario['name']}")

                # Start enhanced processing load
                processing_task = asyncio.create_task(
                    self._intensive_asv_processing_load(
                        enhanced_signal_processor, confidence_based_homing, duration_seconds=5.0
                    )
                )

                # Measure HackRF coordination timing
                coordination_latencies = await self._measure_hackrf_coordination_timing(
                    hackrf_coordinator, scenario
                )

                await processing_task

                # Analyze coordination timing
                max_latency = max(coordination_latencies)
                avg_latency = sum(coordination_latencies) / len(coordination_latencies)
                p90_latency = sorted(coordination_latencies)[
                    int(len(coordination_latencies) * 0.90)
                ]

                # Validate coordination timing requirements
                assert max_latency <= scenario["expected_coordination_latency_ms"], (
                    f"HackRF coordination max latency {max_latency:.1f}ms exceeds "
                    f"expected {scenario['expected_coordination_latency_ms']}ms for {scenario['name']}"
                )

                sdr_coordination_results.append(
                    {
                        "scenario": scenario["name"],
                        "max_latency_ms": max_latency,
                        "avg_latency_ms": avg_latency,
                        "p90_latency_ms": p90_latency,
                        "operation_count": len(coordination_latencies),
                    }
                )

                logger.info(
                    f"HackRF coordination timing for {scenario['name']}: "
                    f"Max: {max_latency:.1f}ms, Avg: {avg_latency:.1f}ms, "
                    f"P90: {p90_latency:.1f}ms ({len(coordination_latencies)} operations)"
                )

                await asyncio.sleep(0.1)  # Cool down between scenarios

            # Validate overall SDR coordination performance
            overall_max_latency = max(
                result["max_latency_ms"] for result in sdr_coordination_results
            )
            overall_avg_latency = sum(
                result["avg_latency_ms"] for result in sdr_coordination_results
            ) / len(sdr_coordination_results)

            # HackRF coordination should remain responsive during enhanced processing
            assert (
                overall_max_latency <= 40.0
            ), f"Overall HackRF coordination max latency {overall_max_latency:.1f}ms exceeds 40ms threshold"

            assert (
                overall_avg_latency <= 20.0
            ), f"Overall HackRF coordination avg latency {overall_avg_latency:.1f}ms exceeds 20ms threshold"

            logger.info(
                f"HackRF coordination validation completed: "
                f"Overall max: {overall_max_latency:.1f}ms, Overall avg: {overall_avg_latency:.1f}ms"
            )

        finally:
            await hackrf_coordinator.shutdown()

    async def _intensive_asv_processing_load(
        self,
        signal_processor: ASVEnhancedSignalProcessor,
        confidence_homing: ASVConfidenceBasedHoming,
        duration_seconds: float,
    ) -> None:
        """Create intensive ASV processing load for coordination testing."""

        end_time = time.perf_counter() + duration_seconds

        # Complex signal scenarios for intensive processing
        complex_scenarios = [
            {"strength": -85.0, "bearing": 45.0, "interference": True, "multipath": True},
            {"strength": -95.0, "bearing": 135.0, "interference": True, "multipath": False},
            {"strength": -75.0, "bearing": 225.0, "interference": False, "multipath": True},
            {"strength": -100.0, "bearing": 315.0, "interference": True, "multipath": True},
        ]

        scenario_index = 0

        while time.perf_counter() < end_time:
            scenario = complex_scenarios[scenario_index % len(complex_scenarios)]

            # Create complex signal data requiring intensive processing
            signal_data = {
                "signal_strength_dbm": scenario["strength"],
                "frequency_hz": 433.92e6,
                "bearing_deg": scenario["bearing"],
                "interference_detected": scenario["interference"],
                "multipath_detected": scenario["multipath"],
                "noise_floor_dbm": -120.0,
                "sample_rate_hz": 2.4e6,
                "analysis_complexity": "maximum",  # Request most intensive analysis
            }

            # Perform intensive enhanced processing
            bearing_calc = await signal_processor.calculate_enhanced_bearing(signal_data)
            decision = confidence_homing.evaluate_confidence_based_decision(bearing_calc)

            # Multiple confidence evaluations to increase load
            for _ in range(3):
                _ = confidence_homing.evaluate_confidence_based_decision(bearing_calc)

            scenario_index += 1

            # High frequency processing ~100Hz
            await asyncio.sleep(0.01)

    async def _measure_hackrf_coordination_timing(
        self, hackrf_coordinator, scenario: Dict[str, Any]
    ) -> List[float]:
        """Measure HackRF coordination timing during enhanced processing."""

        coordination_latencies = []

        if "sweep_range_mhz" in scenario:
            # Frequency sweeping coordination timing
            freq_start, freq_end = scenario["sweep_range_mhz"]
            step_khz = scenario["sweep_step_khz"]

            current_freq = freq_start
            while current_freq <= freq_end:
                coord_start = time.perf_counter()

                try:
                    await asyncio.wait_for(
                        hackrf_coordinator.set_frequency(current_freq * 1e6),  # Convert to Hz
                        timeout=0.1,
                    )

                    coord_latency_ms = (time.perf_counter() - coord_start) * 1000
                    coordination_latencies.append(coord_latency_ms)

                except asyncio.TimeoutError:
                    coordination_latencies.append(100.0)  # Timeout penalty

                current_freq += step_khz / 1000.0  # Convert kHz step to MHz
                await asyncio.sleep(0.05)  # Brief pause between frequency changes

        elif "gain_range_db" in scenario:
            # Gain adjustment coordination timing
            gain_min, gain_max = scenario["gain_range_db"]
            adjustment_interval = 1.0 / scenario["adjustment_frequency_hz"]

            test_duration = 3.0  # 3 seconds of gain adjustments
            end_time = time.perf_counter() + test_duration

            while time.perf_counter() < end_time:
                # Alternate between min and max gain
                target_gain = gain_min if len(coordination_latencies) % 2 == 0 else gain_max

                coord_start = time.perf_counter()

                try:
                    await asyncio.wait_for(hackrf_coordinator.set_gain(target_gain), timeout=0.05)

                    coord_latency_ms = (time.perf_counter() - coord_start) * 1000
                    coordination_latencies.append(coord_latency_ms)

                except asyncio.TimeoutError:
                    coordination_latencies.append(50.0)  # Timeout penalty

                await asyncio.sleep(adjustment_interval)

        elif "sample_rates_hz" in scenario:
            # Sample rate change coordination timing
            sample_rates = scenario["sample_rates_hz"]
            change_interval = 1.0 / scenario["change_frequency_hz"]

            test_duration = 4.0  # 4 seconds of sample rate changes
            end_time = time.perf_counter() + test_duration
            rate_index = 0

            while time.perf_counter() < end_time:
                target_rate = sample_rates[rate_index % len(sample_rates)]

                coord_start = time.perf_counter()

                try:
                    await asyncio.wait_for(
                        hackrf_coordinator.set_sample_rate(target_rate), timeout=0.1
                    )

                    coord_latency_ms = (time.perf_counter() - coord_start) * 1000
                    coordination_latencies.append(coord_latency_ms)

                except asyncio.TimeoutError:
                    coordination_latencies.append(100.0)  # Timeout penalty

                rate_index += 1
                await asyncio.sleep(change_interval)

        return coordination_latencies

    @pytest.mark.asyncio
    async def test_sitl_interface_response_validation_under_enhanced_algorithm_load(
        self, enhanced_signal_processor, confidence_based_homing
    ):
        """[27b3] SITL interface response validation under enhanced algorithm load."""

        logger.info("Testing SITL interface response validation under enhanced algorithm load")

        # Import SITL bridge for integration testing
        from src.backend.services.sdrpp_bridge_service import SDRPPBridgeService

        # Initialize SITL/SDRPP bridge service
        sitl_bridge = SDRPPBridgeService()
        await sitl_bridge.initialize()

        try:
            # Configure enhanced algorithms
            await enhanced_signal_processor.configure_enhanced_processing(enabled=True)

            # Test SITL interface scenarios under enhanced algorithm load
            sitl_response_scenarios = [
                {
                    "name": "signal_updates_during_enhanced_processing",
                    "description": "SITL signal updates while enhanced algorithms processing",
                    "update_frequency_hz": 15.0,
                    "test_duration_seconds": 4.0,
                    "expected_max_response_ms": 40.0,
                },
                {
                    "name": "bearing_data_requests_under_asv_load",
                    "description": "SITL bearing data requests during intensive ASV processing",
                    "request_frequency_hz": 25.0,
                    "test_duration_seconds": 3.0,
                    "expected_max_response_ms": 60.0,
                },
                {
                    "name": "configuration_changes_with_confidence_active",
                    "description": "SITL configuration changes while confidence decisions active",
                    "change_frequency_hz": 3.0,
                    "test_duration_seconds": 5.0,
                    "expected_max_response_ms": 80.0,
                },
            ]

            sitl_response_results = []

            for scenario in sitl_response_scenarios:
                logger.info(f"Testing SITL scenario: {scenario['name']}")

                # Start maximum enhanced processing load
                processing_task = asyncio.create_task(
                    self._maximum_enhanced_processing_load(
                        enhanced_signal_processor,
                        confidence_based_homing,
                        duration_seconds=scenario["test_duration_seconds"],
                    )
                )

                # Measure SITL interface response times
                response_latencies = await self._measure_sitl_interface_timing(
                    sitl_bridge, scenario
                )

                await processing_task

                # Analyze SITL response timing
                max_response = max(response_latencies)
                avg_response = sum(response_latencies) / len(response_latencies)
                p95_response = sorted(response_latencies)[int(len(response_latencies) * 0.95)]

                # Validate SITL response requirements
                assert max_response <= scenario["expected_max_response_ms"], (
                    f"SITL interface max response {max_response:.1f}ms exceeds "
                    f"expected {scenario['expected_max_response_ms']}ms for {scenario['name']}"
                )

                sitl_response_results.append(
                    {
                        "scenario": scenario["name"],
                        "max_response_ms": max_response,
                        "avg_response_ms": avg_response,
                        "p95_response_ms": p95_response,
                        "interaction_count": len(response_latencies),
                    }
                )

                logger.info(
                    f"SITL interface timing for {scenario['name']}: "
                    f"Max: {max_response:.1f}ms, Avg: {avg_response:.1f}ms, "
                    f"P95: {p95_response:.1f}ms ({len(response_latencies)} interactions)"
                )

                await asyncio.sleep(0.15)  # Cool down between scenarios

            # Validate overall SITL interface performance
            overall_max_response = max(
                result["max_response_ms"] for result in sitl_response_results
            )
            overall_avg_response = sum(
                result["avg_response_ms"] for result in sitl_response_results
            ) / len(sitl_response_results)

            # SITL interface should remain responsive during enhanced processing
            assert (
                overall_max_response <= 80.0
            ), f"Overall SITL interface max response {overall_max_response:.1f}ms exceeds 80ms threshold"

            assert (
                overall_avg_response <= 45.0
            ), f"Overall SITL interface avg response {overall_avg_response:.1f}ms exceeds 45ms threshold"

            logger.info(
                f"SITL interface validation completed: "
                f"Overall max: {overall_max_response:.1f}ms, Overall avg: {overall_avg_response:.1f}ms"
            )

        finally:
            await sitl_bridge.shutdown()

    async def _maximum_enhanced_processing_load(
        self,
        signal_processor: ASVEnhancedSignalProcessor,
        confidence_homing: ASVConfidenceBasedHoming,
        duration_seconds: float,
    ) -> None:
        """Create maximum enhanced processing load for SITL testing."""

        end_time = time.perf_counter() + duration_seconds

        # Most challenging signal scenarios for maximum processing load
        max_load_scenarios = [
            {
                "strength": -105.0,
                "bearing": 22.5,
                "interference": True,
                "multipath": True,
                "fading": True,
            },
            {
                "strength": -110.0,
                "bearing": 67.5,
                "interference": True,
                "multipath": True,
                "fading": True,
            },
            {
                "strength": -100.0,
                "bearing": 112.5,
                "interference": True,
                "multipath": True,
                "fading": False,
            },
            {
                "strength": -108.0,
                "bearing": 157.5,
                "interference": True,
                "multipath": True,
                "fading": True,
            },
            {
                "strength": -95.0,
                "bearing": 202.5,
                "interference": True,
                "multipath": True,
                "fading": True,
            },
            {
                "strength": -115.0,
                "bearing": 247.5,
                "interference": True,
                "multipath": True,
                "fading": False,
            },
            {
                "strength": -102.0,
                "bearing": 292.5,
                "interference": True,
                "multipath": True,
                "fading": True,
            },
            {
                "strength": -112.0,
                "bearing": 337.5,
                "interference": True,
                "multipath": True,
                "fading": True,
            },
        ]

        scenario_index = 0

        while time.perf_counter() < end_time:
            scenario = max_load_scenarios[scenario_index % len(max_load_scenarios)]

            # Create extremely challenging signal data
            signal_data = {
                "signal_strength_dbm": scenario["strength"],
                "frequency_hz": 433.92e6,
                "bearing_deg": scenario["bearing"],
                "interference_detected": scenario["interference"],
                "multipath_detected": scenario["multipath"],
                "fading_present": scenario["fading"],
                "noise_floor_dbm": -120.0,
                "sample_rate_hz": 5e6,  # Maximum sample rate for processing load
                "analysis_complexity": "extreme",  # Request most intensive analysis
                "correlation_length": 16384,  # Maximum correlation processing
                "doppler_compensation": True,
            }

            # Perform maximum enhanced processing
            bearing_calc = await signal_processor.calculate_enhanced_bearing(signal_data)

            # Multiple confidence evaluations with different thresholds
            for threshold_adjust in [0.0, 0.1, 0.2]:
                adjusted_calc = ASVBearingCalculation(
                    bearing_deg=bearing_calc.bearing_deg,
                    confidence=max(0.0, bearing_calc.confidence - threshold_adjust),
                    precision_deg=bearing_calc.precision_deg,
                    signal_strength_dbm=bearing_calc.signal_strength_dbm,
                    interference_detected=bearing_calc.interference_detected,
                    signal_quality=bearing_calc.signal_quality,
                )
                _ = confidence_homing.evaluate_confidence_based_decision(adjusted_calc)

            scenario_index += 1

            # Maximum frequency processing ~200Hz
            await asyncio.sleep(0.005)

    async def _measure_sitl_interface_timing(
        self, sitl_bridge, scenario: Dict[str, Any]
    ) -> List[float]:
        """Measure SITL interface timing during enhanced processing."""

        interaction_interval = 1.0 / scenario["request_frequency_hz"]
        test_duration = scenario["test_duration_seconds"]
        end_time = time.perf_counter() + test_duration
        response_latencies = []

        interaction_types = [
            "get_signal_strength",
            "get_bearing_data",
            "set_frequency",
            "get_spectrum_data",
            "update_configuration",
        ]

        interaction_index = 0

        while time.perf_counter() < end_time:
            interaction_type = interaction_types[interaction_index % len(interaction_types)]

            response_start = time.perf_counter()

            try:
                # Execute SITL interface interaction
                await asyncio.wait_for(
                    self._execute_sitl_interaction(sitl_bridge, interaction_type),
                    timeout=0.2,  # 200ms timeout per interaction
                )

                response_latency_ms = (time.perf_counter() - response_start) * 1000
                response_latencies.append(response_latency_ms)

            except asyncio.TimeoutError:
                # Interaction timed out - record penalty
                response_latencies.append(200.0)  # 200ms penalty
                logger.warning(f"SITL interaction {interaction_type} timed out")

            interaction_index += 1

            # Wait for next interaction interval
            await asyncio.sleep(interaction_interval)

        return response_latencies

    async def _execute_sitl_interaction(self, sitl_bridge, interaction_type: str) -> Any:
        """Execute specific SITL interface interaction for timing measurement."""

        if interaction_type == "get_signal_strength":
            return await sitl_bridge.get_signal_strength()
        elif interaction_type == "get_bearing_data":
            return await sitl_bridge.get_bearing_data()
        elif interaction_type == "set_frequency":
            return await sitl_bridge.set_frequency(433.92e6)
        elif interaction_type == "get_spectrum_data":
            return await sitl_bridge.get_spectrum_data()
        elif interaction_type == "update_configuration":
            return await sitl_bridge.update_configuration({"gain": 20})
        else:
            # Default interaction
            return await sitl_bridge.get_status()

    @pytest.mark.asyncio
    async def test_cross_service_communication_latency_during_asv_processing_peaks(
        self, enhanced_signal_processor, confidence_based_homing
    ):
        """[27b4] Cross-service communication latency during ASV processing peaks."""

        logger.info("Testing cross-service communication latency during ASV processing peaks")

        # Import services for cross-service communication testing
        from src.backend.services.mavlink_service import MAVLinkService
        from src.backend.services.sdrpp_bridge_service import SDRPPBridgeService
        from src.backend.services.signal_processor import SignalProcessor

        # Initialize all services for cross-service communication
        services = {
            "mavlink": MAVLinkService(),
            "sdrpp_bridge": SDRPPBridgeService(),
            "signal_processor": SignalProcessor(),
        }

        # Initialize all services
        for service in services.values():
            await service.initialize()

        try:
            # Configure enhanced algorithms for peak processing load
            await enhanced_signal_processor.configure_enhanced_processing(enabled=True)

            # Test cross-service communication scenarios during ASV processing peaks
            cross_service_scenarios = [
                {
                    "name": "mavlink_to_sdr_coordination_under_peak_load",
                    "description": "MAVLink to SDR coordination during peak ASV processing",
                    "communication_frequency_hz": 8.0,
                    "test_duration_seconds": 6.0,
                    "expected_max_latency_ms": 100.0,
                    "services": ["mavlink", "sdrpp_bridge"],
                },
                {
                    "name": "signal_processor_to_enhanced_coordination",
                    "description": "Signal processor to enhanced processor coordination",
                    "communication_frequency_hz": 12.0,
                    "test_duration_seconds": 4.0,
                    "expected_max_latency_ms": 75.0,
                    "services": ["signal_processor", "enhanced"],
                },
                {
                    "name": "full_service_mesh_communication_peak_load",
                    "description": "Full service mesh communication during ASV processing peaks",
                    "communication_frequency_hz": 6.0,
                    "test_duration_seconds": 8.0,
                    "expected_max_latency_ms": 150.0,
                    "services": ["mavlink", "sdrpp_bridge", "signal_processor", "enhanced"],
                },
            ]

            cross_service_results = []

            for scenario in cross_service_scenarios:
                logger.info(f"Testing cross-service scenario: {scenario['name']}")

                # Start peak ASV processing load
                processing_task = asyncio.create_task(
                    self._peak_asv_processing_load(
                        enhanced_signal_processor,
                        confidence_based_homing,
                        duration_seconds=scenario["test_duration_seconds"],
                    )
                )

                # Measure cross-service communication latencies
                communication_latencies = await self._measure_cross_service_communication_timing(
                    services, enhanced_signal_processor, scenario
                )

                await processing_task

                # Analyze cross-service communication timing
                max_comm_latency = max(communication_latencies)
                avg_comm_latency = sum(communication_latencies) / len(communication_latencies)
                p99_comm_latency = sorted(communication_latencies)[
                    int(len(communication_latencies) * 0.99)
                ]

                # Validate cross-service communication requirements
                assert max_comm_latency <= scenario["expected_max_latency_ms"], (
                    f"Cross-service communication max latency {max_comm_latency:.1f}ms exceeds "
                    f"expected {scenario['expected_max_latency_ms']}ms for {scenario['name']}"
                )

                cross_service_results.append(
                    {
                        "scenario": scenario["name"],
                        "max_latency_ms": max_comm_latency,
                        "avg_latency_ms": avg_comm_latency,
                        "p99_latency_ms": p99_comm_latency,
                        "communication_count": len(communication_latencies),
                    }
                )

                logger.info(
                    f"Cross-service communication timing for {scenario['name']}: "
                    f"Max: {max_comm_latency:.1f}ms, Avg: {avg_comm_latency:.1f}ms, "
                    f"P99: {p99_comm_latency:.1f}ms ({len(communication_latencies)} communications)"
                )

                await asyncio.sleep(0.2)  # Cool down between scenarios

            # Validate overall cross-service communication performance
            overall_max_latency = max(result["max_latency_ms"] for result in cross_service_results)
            overall_avg_latency = sum(
                result["avg_latency_ms"] for result in cross_service_results
            ) / len(cross_service_results)

            # Cross-service communication should remain efficient during ASV processing peaks
            assert (
                overall_max_latency <= 150.0
            ), f"Overall cross-service communication max latency {overall_max_latency:.1f}ms exceeds 150ms threshold"

            assert (
                overall_avg_latency <= 80.0
            ), f"Overall cross-service communication avg latency {overall_avg_latency:.1f}ms exceeds 80ms threshold"

            logger.info(
                f"Cross-service communication validation completed: "
                f"Overall max: {overall_max_latency:.1f}ms, Overall avg: {overall_avg_latency:.1f}ms"
            )

        finally:
            # Shutdown all services
            for service in services.values():
                await service.shutdown()

    async def _peak_asv_processing_load(
        self,
        signal_processor: ASVEnhancedSignalProcessor,
        confidence_homing: ASVConfidenceBasedHoming,
        duration_seconds: float,
    ) -> None:
        """Create peak ASV processing load for cross-service communication testing."""

        end_time = time.perf_counter() + duration_seconds

        # Peak processing scenarios that stress all ASV components
        peak_scenarios = [
            {
                "strength": -110.0,
                "bearing": 15.0,
                "interference": True,
                "multipath": True,
                "doppler": 5.0,
            },
            {
                "strength": -115.0,
                "bearing": 45.0,
                "interference": True,
                "multipath": True,
                "doppler": -3.2,
            },
            {
                "strength": -105.0,
                "bearing": 75.0,
                "interference": True,
                "multipath": True,
                "doppler": 8.1,
            },
            {
                "strength": -108.0,
                "bearing": 105.0,
                "interference": True,
                "multipath": True,
                "doppler": -6.5,
            },
            {
                "strength": -112.0,
                "bearing": 135.0,
                "interference": True,
                "multipath": True,
                "doppler": 4.7,
            },
            {
                "strength": -118.0,
                "bearing": 165.0,
                "interference": True,
                "multipath": True,
                "doppler": -2.8,
            },
            {
                "strength": -103.0,
                "bearing": 195.0,
                "interference": True,
                "multipath": True,
                "doppler": 7.3,
            },
            {
                "strength": -120.0,
                "bearing": 225.0,
                "interference": True,
                "multipath": True,
                "doppler": -9.1,
            },
            {
                "strength": -106.0,
                "bearing": 255.0,
                "interference": True,
                "multipath": True,
                "doppler": 6.2,
            },
            {
                "strength": -114.0,
                "bearing": 285.0,
                "interference": True,
                "multipath": True,
                "doppler": -4.5,
            },
            {
                "strength": -111.0,
                "bearing": 315.0,
                "interference": True,
                "multipath": True,
                "doppler": 3.8,
            },
            {
                "strength": -117.0,
                "bearing": 345.0,
                "interference": True,
                "multipath": True,
                "doppler": -7.6,
            },
        ]

        scenario_index = 0

        while time.perf_counter() < end_time:
            scenario = peak_scenarios[scenario_index % len(peak_scenarios)]

            # Create peak load signal data
            signal_data = {
                "signal_strength_dbm": scenario["strength"],
                "frequency_hz": 433.92e6,
                "bearing_deg": scenario["bearing"],
                "interference_detected": scenario["interference"],
                "multipath_detected": scenario["multipath"],
                "doppler_shift_hz": scenario["doppler"],
                "noise_floor_dbm": -125.0,
                "sample_rate_hz": 10e6,  # Maximum sample rate for peak load
                "analysis_complexity": "peak",  # Peak analysis complexity
                "correlation_length": 32768,  # Maximum correlation processing
                "doppler_compensation": True,
                "advanced_filtering": True,
                "spectral_analysis": True,
            }

            # Perform peak enhanced processing with multiple parallel operations
            bearing_calc = await signal_processor.calculate_enhanced_bearing(signal_data)

            # Multiple concurrent confidence evaluations to maximize load
            confidence_tasks = []
            for threshold_adjust in [0.0, 0.05, 0.1, 0.15, 0.2]:
                adjusted_calc = ASVBearingCalculation(
                    bearing_deg=bearing_calc.bearing_deg
                    + (threshold_adjust * 10),  # Slight bearing variation
                    confidence=max(0.0, bearing_calc.confidence - threshold_adjust),
                    precision_deg=bearing_calc.precision_deg + threshold_adjust,
                    signal_strength_dbm=bearing_calc.signal_strength_dbm,
                    interference_detected=bearing_calc.interference_detected,
                    signal_quality=max(0.0, bearing_calc.signal_quality - threshold_adjust),
                )
                # Create concurrent confidence evaluation tasks
                confidence_tasks.append(
                    asyncio.create_task(
                        asyncio.to_thread(
                            confidence_homing.evaluate_confidence_based_decision, adjusted_calc
                        )
                    )
                )

            # Wait for all confidence evaluations to complete (peak load)
            await asyncio.gather(*confidence_tasks)

            scenario_index += 1

            # Peak frequency processing ~250Hz
            await asyncio.sleep(0.004)

    async def _measure_cross_service_communication_timing(
        self,
        services: Dict[str, Any],
        enhanced_signal_processor: ASVEnhancedSignalProcessor,
        scenario: Dict[str, Any],
    ) -> List[float]:
        """Measure cross-service communication timing during peak ASV processing."""

        communication_interval = 1.0 / scenario["communication_frequency_hz"]
        test_duration = scenario["test_duration_seconds"]
        end_time = time.perf_counter() + test_duration
        communication_latencies = []

        # Define cross-service communication patterns
        communication_patterns = {
            ("mavlink", "sdrpp_bridge"): [
                "coordinate_frequency_change",
                "synchronize_signal_data",
                "update_position_from_bearing",
            ],
            ("signal_processor", "enhanced"): [
                "share_processing_results",
                "coordinate_analysis_parameters",
                "synchronize_confidence_data",
            ],
            ("mavlink", "signal_processor"): [
                "update_vehicle_position",
                "coordinate_homing_commands",
                "share_navigation_data",
            ],
            ("sdrpp_bridge", "enhanced"): [
                "coordinate_sdr_parameters",
                "share_spectrum_data",
                "synchronize_processing_state",
            ],
        }

        # Get relevant service pairs for this scenario
        relevant_pairs = []
        scenario_services = scenario["services"]

        if "enhanced" in scenario_services:
            # Replace "enhanced" with actual enhanced_signal_processor
            scenario_services = [
                s if s != "enhanced" else "enhanced_signal_processor" for s in scenario_services
            ]

        for i, service1 in enumerate(scenario_services):
            for service2 in scenario_services[i + 1 :]:
                service1_key = service1 if service1 != "enhanced_signal_processor" else "enhanced"
                service2_key = service2 if service2 != "enhanced_signal_processor" else "enhanced"

                # Check both directions for communication patterns
                if (service1_key, service2_key) in communication_patterns:
                    relevant_pairs.append(
                        (service1, service2, communication_patterns[(service1_key, service2_key)])
                    )
                elif (service2_key, service1_key) in communication_patterns:
                    relevant_pairs.append(
                        (service1, service2, communication_patterns[(service2_key, service1_key)])
                    )

        communication_index = 0

        while time.perf_counter() < end_time:
            if not relevant_pairs:
                # Fallback to basic service interaction timing
                communication_latencies.append(await self._measure_basic_service_interaction())
                await asyncio.sleep(communication_interval)
                continue

            # Select communication pair and pattern
            pair_info = relevant_pairs[communication_index % len(relevant_pairs)]
            service1_name, service2_name, patterns = pair_info
            pattern = patterns[communication_index % len(patterns)]

            # Measure cross-service communication latency
            comm_start = time.perf_counter()

            try:
                # Execute cross-service communication
                await asyncio.wait_for(
                    self._execute_cross_service_communication(
                        services, enhanced_signal_processor, service1_name, service2_name, pattern
                    ),
                    timeout=0.3,  # 300ms timeout per communication
                )

                comm_latency_ms = (time.perf_counter() - comm_start) * 1000
                communication_latencies.append(comm_latency_ms)

            except asyncio.TimeoutError:
                # Communication timed out - record penalty
                communication_latencies.append(300.0)  # 300ms penalty
                logger.warning(
                    f"Cross-service communication {pattern} between {service1_name} and {service2_name} timed out"
                )

            communication_index += 1

            # Wait for next communication interval
            await asyncio.sleep(communication_interval)

        return communication_latencies

    async def _measure_basic_service_interaction(self) -> float:
        """Measure basic service interaction latency as fallback."""
        start = time.perf_counter()
        await asyncio.sleep(0.001)  # Simulate basic interaction
        return (time.perf_counter() - start) * 1000

    async def _execute_cross_service_communication(
        self,
        services: Dict[str, Any],
        enhanced_signal_processor: ASVEnhancedSignalProcessor,
        service1_name: str,
        service2_name: str,
        communication_pattern: str,
    ) -> Any:
        """Execute specific cross-service communication pattern."""

        # Get service instances
        service1 = (
            services.get(service1_name)
            if service1_name != "enhanced_signal_processor"
            else enhanced_signal_processor
        )
        service2 = (
            services.get(service2_name)
            if service2_name != "enhanced_signal_processor"
            else enhanced_signal_processor
        )

        if communication_pattern == "coordinate_frequency_change":
            # MAVLink to SDRPP frequency coordination
            await service1.request_frequency_change(433.95e6)
            return await service2.acknowledge_frequency_change(433.95e6)

        elif communication_pattern == "synchronize_signal_data":
            # Cross-service signal data synchronization
            signal_data = await service1.get_current_signal_data()
            return await service2.update_signal_context(signal_data)

        elif communication_pattern == "update_position_from_bearing":
            # Update position based on bearing calculations
            bearing_data = await service2.get_latest_bearing()
            return await service1.update_position_estimate(bearing_data)

        elif communication_pattern == "share_processing_results":
            # Signal processor to enhanced processor result sharing
            processing_results = await service1.get_processing_results()
            return await service2.integrate_processing_results(processing_results)

        elif communication_pattern == "coordinate_analysis_parameters":
            # Cross-processor analysis parameter coordination
            analysis_params = await service1.get_analysis_parameters()
            return await service2.update_analysis_parameters(analysis_params)

        elif communication_pattern == "synchronize_confidence_data":
            # Confidence data synchronization between processors
            confidence_data = await service2.get_confidence_metrics()
            return await service1.update_confidence_context(confidence_data)

        elif communication_pattern == "update_vehicle_position":
            # MAVLink vehicle position updates
            position = await service1.get_vehicle_position()
            return await service2.update_signal_source_estimate(position)

        elif communication_pattern == "coordinate_homing_commands":
            # MAVLink and signal processor homing coordination
            homing_vector = await service2.get_homing_vector()
            return await service1.execute_homing_command(homing_vector)

        elif communication_pattern == "share_navigation_data":
            # Navigation data sharing
            nav_data = await service1.get_navigation_data()
            return await service2.update_navigation_context(nav_data)

        elif communication_pattern == "coordinate_sdr_parameters":
            # SDRPP and enhanced processor SDR parameter coordination
            sdr_params = await service1.get_sdr_parameters()
            return await service2.optimize_sdr_parameters(sdr_params)

        elif communication_pattern == "share_spectrum_data":
            # Spectrum data sharing between SDRPP and enhanced processor
            spectrum_data = await service1.get_spectrum_analysis()
            return await service2.integrate_spectrum_data(spectrum_data)

        elif communication_pattern == "synchronize_processing_state":
            # Processing state synchronization
            processing_state = await service2.get_processing_state()
            return await service1.synchronize_state(processing_state)

        else:
            # Default communication pattern
            await service1.ping()
            return await service2.ping()
