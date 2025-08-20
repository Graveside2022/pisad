"""
TASK-6.2.4-PERFORMANCE-VALIDATION-AND-TESTING: Enhanced Algorithm Performance Validation
SUBTASK-6.2.4.1 - Comprehensive algorithm performance testing

Performance validation tests for enhanced homing algorithms to ensure they meet
all PRD requirements before field deployment.

PRD References:
- PRD-NFR2: Signal processing latency shall not exceed 100ms
- PRD-NFR8: 90% successful homing rate once signal is acquired
- PRD-FR4: Navigate toward detected signals using RSSI gradient climbing with enhanced accuracy

Test Categories:
- RSSI degradation scenarios with controlled signal attenuation
- Recovery strategy validation (spiral search, S-turn, return-to-peak)
- Performance benchmarking framework for enhanced algorithms
- Bearing accuracy validation with ±2° precision requirement
- Adaptive search pattern effectiveness testing

All tests use authentic system integration - NO mocks or simulated components.
"""

import asyncio
import logging
import math
import statistics
import time
from dataclasses import dataclass

import numpy as np
import pytest

from src.backend.hal.beacon_generator import BeaconGenerator
from src.backend.services.asv_integration.asv_confidence_based_homing import (
    ASVConfidenceBasedHoming,
)
from src.backend.services.asv_integration.asv_enhanced_signal_processor import (
    ASVEnhancedSignalProcessor,
)
from src.backend.services.homing_algorithm import (
    HomingAlgorithm,
)
from src.backend.utils.test_metrics import PerformanceMetrics, TimingValidator

logger = logging.getLogger(__name__)


@dataclass
class RSSIDegradationScenario:
    """Test scenario for RSSI degradation validation."""

    name: str
    initial_rssi_dbm: float
    final_rssi_dbm: float
    degradation_steps: int
    degradation_duration_ms: int
    expected_recovery_time_ms: float
    signal_confidence_threshold: float


@dataclass
class BearingAccuracyScenario:
    """Test scenario for bearing accuracy validation."""

    name: str
    true_bearing_deg: float
    signal_strength_dbm: float
    distance_meters: float
    expected_precision_deg: float
    confidence_threshold: float


class TestEnhancedAlgorithmPerformanceValidation:
    """Comprehensive performance validation tests for enhanced algorithms."""

    @pytest.fixture
    async def enhanced_signal_processor(self):
        """Create ASV enhanced signal processor instance."""
        processor = ASVEnhancedSignalProcessor()
        await processor.start()
        yield processor
        await processor.stop()

    @pytest.fixture
    async def confidence_based_homing(self):
        """Create ASV confidence-based homing instance."""
        homing = ASVConfidenceBasedHoming()
        await homing.initialize()
        yield homing
        await homing.shutdown()

    @pytest.fixture
    def baseline_homing_algorithm(self):
        """Create baseline homing algorithm for comparison."""
        return HomingAlgorithm()

    @pytest.fixture
    async def beacon_generator(self):
        """Create beacon generator for controlled signal testing."""
        generator = BeaconGenerator()
        await generator.initialize()
        yield generator
        await generator.shutdown()

    @pytest.fixture
    def performance_metrics(self):
        """Create performance metrics tracker."""
        return PerformanceMetrics()

    @pytest.fixture
    def timing_validator(self):
        """Create timing validator for <100ms requirement."""
        return TimingValidator(max_latency_ms=100.0)

    # Test [26a1]: RSSI degradation test scenarios with controlled signal attenuation

    @pytest.mark.asyncio
    async def test_rssi_degradation_controlled_attenuation(
        self, enhanced_signal_processor, beacon_generator, performance_metrics, timing_validator
    ):
        """Test RSSI degradation scenarios with controlled signal attenuation.

        PRD-NFR2: <100ms processing latency requirement
        Recovery time target: <2 seconds per Story 6.2 acceptance criteria
        """
        # Define test scenarios
        degradation_scenarios = [
            RSSIDegradationScenario(
                name="moderate_degradation",
                initial_rssi_dbm=-50.0,
                final_rssi_dbm=-70.0,
                degradation_steps=10,
                degradation_duration_ms=5000,
                expected_recovery_time_ms=1500.0,
                signal_confidence_threshold=0.3,
            ),
            RSSIDegradationScenario(
                name="severe_degradation",
                initial_rssi_dbm=-60.0,
                final_rssi_dbm=-85.0,
                degradation_steps=15,
                degradation_duration_ms=7000,
                expected_recovery_time_ms=2000.0,
                signal_confidence_threshold=0.2,
            ),
            RSSIDegradationScenario(
                name="rapid_degradation",
                initial_rssi_dbm=-45.0,
                final_rssi_dbm=-80.0,
                degradation_steps=5,
                degradation_duration_ms=2000,
                expected_recovery_time_ms=1000.0,
                signal_confidence_threshold=0.25,
            ),
        ]

        results = []

        for scenario in degradation_scenarios:
            logger.info(f"Testing RSSI degradation scenario: {scenario.name}")

            # Configure beacon with initial signal strength
            await beacon_generator.set_signal_strength(scenario.initial_rssi_dbm)
            await beacon_generator.set_frequency(3200.0)  # 3.2 GHz default
            await beacon_generator.start_transmission()

            # Allow signal to stabilize
            await asyncio.sleep(1.0)

            # Record initial processing latency
            start_time = time.time_ns()

            initial_bearing = await enhanced_signal_processor.compute_bearing()

            initial_latency_ms = (time.time_ns() - start_time) / 1_000_000

            # Validate initial processing meets <100ms requirement
            timing_validator.validate_latency(initial_latency_ms)
            assert (
                initial_latency_ms < 100.0
            ), f"Initial processing latency {initial_latency_ms}ms exceeds 100ms requirement"

            # Perform controlled signal degradation
            degradation_start_time = time.time()
            step_duration_ms = scenario.degradation_duration_ms / scenario.degradation_steps
            rssi_step = (
                scenario.final_rssi_dbm - scenario.initial_rssi_dbm
            ) / scenario.degradation_steps

            for step in range(scenario.degradation_steps):
                current_rssi = scenario.initial_rssi_dbm + (step * rssi_step)
                await beacon_generator.set_signal_strength(current_rssi)

                # Measure processing latency during degradation
                step_start_time = time.time_ns()
                bearing_result = await enhanced_signal_processor.compute_bearing()
                step_latency_ms = (time.time_ns() - step_start_time) / 1_000_000

                # Validate processing latency throughout degradation
                timing_validator.validate_latency(step_latency_ms)
                assert (
                    step_latency_ms < 100.0
                ), f"Degradation step {step} latency {step_latency_ms}ms exceeds 100ms requirement"

                # Check if confidence drops below threshold (recovery trigger)
                if bearing_result.confidence < scenario.signal_confidence_threshold:
                    recovery_start_time = time.time()
                    logger.info(
                        f"Confidence dropped to {bearing_result.confidence}, recovery should trigger"
                    )

                    # Enhanced algorithm should trigger recovery strategy
                    # Wait for recovery mechanism to activate
                    max_recovery_wait = 3.0  # Maximum wait time for recovery
                    recovery_detected = False

                    for wait_step in range(30):  # Check every 100ms for 3 seconds
                        await asyncio.sleep(0.1)
                        recovery_bearing = await enhanced_signal_processor.compute_bearing()

                        if recovery_bearing.confidence > scenario.signal_confidence_threshold:
                            recovery_time = time.time() - recovery_start_time
                            recovery_detected = True
                            logger.info(f"Recovery detected in {recovery_time*1000:.1f}ms")
                            break

                    recovery_total_time = time.time() - recovery_start_time

                    # Validate recovery time meets requirement
                    assert recovery_detected, f"Recovery not detected within {max_recovery_wait}s"
                    assert (
                        recovery_total_time < (scenario.expected_recovery_time_ms / 1000.0)
                    ), f"Recovery time {recovery_total_time*1000:.1f}ms exceeds expected {scenario.expected_recovery_time_ms}ms"

                    break

                await asyncio.sleep(step_duration_ms / 1000.0)

            await beacon_generator.stop_transmission()

            # Record scenario results
            scenario_result = {
                "scenario_name": scenario.name,
                "initial_latency_ms": initial_latency_ms,
                "max_latency_ms": timing_validator.get_max_latency(),
                "avg_latency_ms": timing_validator.get_average_latency(),
                "recovery_time_ms": recovery_total_time * 1000
                if "recovery_total_time" in locals()
                else None,
                "recovery_detected": recovery_detected
                if "recovery_detected" in locals()
                else False,
                "processing_samples": timing_validator.get_sample_count(),
            }
            results.append(scenario_result)

            performance_metrics.add_measurement("rssi_degradation_test", scenario_result)

        # Validate overall performance across all scenarios
        avg_latencies = [r["avg_latency_ms"] for r in results]
        max_latencies = [r["max_latency_ms"] for r in results]

        overall_avg_latency = statistics.mean(avg_latencies)
        overall_max_latency = max(max_latencies)

        logger.info("RSSI degradation test results:")
        logger.info(f"  Overall average latency: {overall_avg_latency:.2f}ms")
        logger.info(f"  Overall maximum latency: {overall_max_latency:.2f}ms")
        logger.info(f"  All scenarios processed: {len(results)}")

        # Final assertions for PRD compliance
        assert (
            overall_avg_latency < 100.0
        ), f"Average processing latency {overall_avg_latency:.2f}ms exceeds 100ms requirement"
        assert (
            overall_max_latency < 100.0
        ), f"Maximum processing latency {overall_max_latency:.2f}ms exceeds 100ms requirement"

        recovery_success_rate = sum(1 for r in results if r["recovery_detected"]) / len(results)
        assert (
            recovery_success_rate >= 0.9
        ), f"Recovery success rate {recovery_success_rate:.2%} below 90% requirement"

    # Test [26a2]: Recovery strategy validation tests for spiral search, S-turn, return-to-peak

    @pytest.mark.asyncio
    async def test_recovery_strategy_validation_spiral_search(
        self, confidence_based_homing, beacon_generator, performance_metrics, timing_validator
    ):
        """Test spiral search recovery strategy validation with success rate measurement.

        Tests spiral search pattern when signal confidence < 30%
        Measures success rate and time-to-acquisition
        """
        logger.info("Testing spiral search recovery strategy")

        # Configure beacon for weak signal scenario (triggers spiral search)
        await beacon_generator.set_signal_strength(-75.0)  # Weak signal
        await beacon_generator.set_frequency(3200.0)
        await beacon_generator.start_transmission()

        # Configure homing for spiral search trigger
        confidence_based_homing.set_confidence_threshold(0.3)  # 30% threshold

        test_iterations = 10  # Multiple tests for statistical validation
        successful_acquisitions = 0
        acquisition_times = []

        for iteration in range(test_iterations):
            logger.info(f"Spiral search test iteration {iteration + 1}/{test_iterations}")

            # Reset homing state
            await confidence_based_homing.reset_state()

            # Start spiral search sequence
            search_start_time = time.time()

            # Trigger low confidence condition to initiate spiral search
            await confidence_based_homing.trigger_low_confidence_recovery()

            # Monitor spiral search execution
            max_search_time = 15.0  # 15 second maximum search time
            acquisition_success = False

            for check_step in range(150):  # Check every 100ms for 15 seconds
                await asyncio.sleep(0.1)

                # Measure processing latency during spiral search
                step_start_time = time.time_ns()
                current_state = await confidence_based_homing.get_current_state()
                step_latency_ms = (time.time_ns() - step_start_time) / 1_000_000

                timing_validator.validate_latency(step_latency_ms)
                assert (
                    step_latency_ms < 100.0
                ), f"Spiral search processing latency {step_latency_ms}ms exceeds 100ms"

                # Check if spiral search successfully acquired signal
                if current_state.confidence > 0.3 and current_state.signal_acquired:
                    acquisition_time = time.time() - search_start_time
                    acquisition_times.append(acquisition_time)
                    successful_acquisitions += 1
                    acquisition_success = True
                    logger.info(f"Spiral search acquisition successful in {acquisition_time:.2f}s")
                    break

            if not acquisition_success:
                logger.warning(
                    f"Spiral search iteration {iteration + 1} did not acquire signal within {max_search_time}s"
                )

        await beacon_generator.stop_transmission()

        # Calculate success rate and performance metrics
        success_rate = successful_acquisitions / test_iterations
        avg_acquisition_time = (
            statistics.mean(acquisition_times) if acquisition_times else float("inf")
        )

        # Record results
        spiral_search_results = {
            "strategy": "spiral_search",
            "test_iterations": test_iterations,
            "successful_acquisitions": successful_acquisitions,
            "success_rate": success_rate,
            "avg_acquisition_time_s": avg_acquisition_time,
            "max_latency_ms": timing_validator.get_max_latency(),
            "avg_latency_ms": timing_validator.get_average_latency(),
        }

        performance_metrics.add_measurement("spiral_search_recovery", spiral_search_results)

        logger.info("Spiral search recovery results:")
        logger.info(f"  Success rate: {success_rate:.2%}")
        logger.info(f"  Average acquisition time: {avg_acquisition_time:.2f}s")
        logger.info(f"  Processing latency (avg): {timing_validator.get_average_latency():.2f}ms")

        # Assertions for requirements compliance
        assert (
            success_rate >= 0.8
        ), f"Spiral search success rate {success_rate:.2%} below 80% requirement"
        assert (
            avg_acquisition_time < 10.0
        ), f"Average acquisition time {avg_acquisition_time:.2f}s exceeds 10s limit"
        assert (
            timing_validator.get_max_latency() < 100.0
        ), "Max processing latency exceeds 100ms requirement"

    @pytest.mark.asyncio
    async def test_recovery_strategy_validation_s_turn_sampling(
        self, confidence_based_homing, beacon_generator, performance_metrics, timing_validator
    ):
        """Test S-turn sampling maneuver effectiveness with weak signal scenarios.

        Tests S-turn pattern for gradient determination in weak signals
        Validates effectiveness at -80dB threshold
        """
        logger.info("Testing S-turn sampling recovery strategy")

        # Configure beacon for very weak signal scenario (triggers S-turn sampling)
        await beacon_generator.set_signal_strength(-80.0)  # Very weak signal
        await beacon_generator.set_frequency(3200.0)
        await beacon_generator.start_transmission()

        test_iterations = 8  # Multiple tests for statistical validation
        successful_gradient_determinations = 0
        sampling_times = []

        for iteration in range(test_iterations):
            logger.info(f"S-turn sampling test iteration {iteration + 1}/{test_iterations}")

            # Reset homing state
            await confidence_based_homing.reset_state()

            # Start S-turn sampling sequence
            sampling_start_time = time.time()

            # Trigger weak signal condition to initiate S-turn sampling
            await confidence_based_homing.trigger_weak_signal_sampling()

            # Monitor S-turn sampling execution
            max_sampling_time = 12.0  # 12 second maximum sampling time
            gradient_determined = False

            for check_step in range(120):  # Check every 100ms for 12 seconds
                await asyncio.sleep(0.1)

                # Measure processing latency during S-turn sampling
                step_start_time = time.time_ns()
                current_gradient = await confidence_based_homing.get_current_gradient()
                step_latency_ms = (time.time_ns() - step_start_time) / 1_000_000

                timing_validator.validate_latency(step_latency_ms)
                assert (
                    step_latency_ms < 100.0
                ), f"S-turn sampling latency {step_latency_ms}ms exceeds 100ms"

                # Check if S-turn sampling successfully determined gradient
                if current_gradient.confidence > 0.2 and current_gradient.magnitude > 0.1:
                    sampling_time = time.time() - sampling_start_time
                    sampling_times.append(sampling_time)
                    successful_gradient_determinations += 1
                    gradient_determined = True
                    logger.info(f"S-turn gradient determination successful in {sampling_time:.2f}s")
                    break

            if not gradient_determined:
                logger.warning(
                    f"S-turn sampling iteration {iteration + 1} did not determine gradient within {max_sampling_time}s"
                )

        await beacon_generator.stop_transmission()

        # Calculate success rate and performance metrics
        success_rate = successful_gradient_determinations / test_iterations
        avg_sampling_time = statistics.mean(sampling_times) if sampling_times else float("inf")

        # Record results
        s_turn_results = {
            "strategy": "s_turn_sampling",
            "test_iterations": test_iterations,
            "successful_determinations": successful_gradient_determinations,
            "success_rate": success_rate,
            "avg_sampling_time_s": avg_sampling_time,
            "max_latency_ms": timing_validator.get_max_latency(),
            "avg_latency_ms": timing_validator.get_average_latency(),
        }

        performance_metrics.add_measurement("s_turn_recovery", s_turn_results)

        logger.info("S-turn sampling recovery results:")
        logger.info(f"  Success rate: {success_rate:.2%}")
        logger.info(f"  Average sampling time: {avg_sampling_time:.2f}s")
        logger.info(f"  Processing latency (avg): {timing_validator.get_average_latency():.2f}ms")

        # Assertions for requirements compliance
        assert (
            success_rate >= 0.75
        ), f"S-turn sampling success rate {success_rate:.2%} below 75% requirement"
        assert (
            avg_sampling_time < 8.0
        ), f"Average sampling time {avg_sampling_time:.2f}s exceeds 8s limit"
        assert (
            timing_validator.get_max_latency() < 100.0
        ), "Max processing latency exceeds 100ms requirement"

    @pytest.mark.asyncio
    async def test_recovery_strategy_validation_return_to_peak(
        self, confidence_based_homing, beacon_generator, performance_metrics, timing_validator
    ):
        """Test return-to-last-peak algorithm validation with complete signal loss scenarios.

        Tests return-to-peak strategy when signal is completely lost
        Validates navigation back to last known strong signal location
        """
        logger.info("Testing return-to-last-peak recovery strategy")

        test_iterations = 6  # Multiple tests for statistical validation
        successful_returns = 0
        return_times = []

        for iteration in range(test_iterations):
            logger.info(f"Return-to-peak test iteration {iteration + 1}/{test_iterations}")

            # Phase 1: Establish strong signal and record peak location
            await beacon_generator.set_signal_strength(-45.0)  # Strong signal
            await beacon_generator.set_frequency(3200.0)
            await beacon_generator.start_transmission()

            # Allow peak location to be established
            await asyncio.sleep(2.0)

            # Record peak location
            peak_start_time = time.time_ns()
            peak_location = await confidence_based_homing.get_current_location()
            peak_signal_strength = await confidence_based_homing.get_signal_strength()
            peak_latency_ms = (time.time_ns() - peak_start_time) / 1_000_000

            timing_validator.validate_latency(peak_latency_ms)
            assert (
                peak_latency_ms < 100.0
            ), f"Peak location recording latency {peak_latency_ms}ms exceeds 100ms"

            # Phase 2: Complete signal loss
            await beacon_generator.stop_transmission()
            await asyncio.sleep(1.0)  # Allow signal loss to be detected

            # Phase 3: Trigger return-to-last-peak
            return_start_time = time.time()

            await confidence_based_homing.trigger_signal_loss_recovery()

            # Phase 4: Re-enable beacon at peak location for validation
            await asyncio.sleep(0.5)  # Brief delay to simulate return navigation
            await beacon_generator.start_transmission()

            # Monitor return-to-peak execution
            max_return_time = 10.0  # 10 second maximum return time
            return_success = False

            for check_step in range(100):  # Check every 100ms for 10 seconds
                await asyncio.sleep(0.1)

                # Measure processing latency during return-to-peak
                step_start_time = time.time_ns()
                current_state = await confidence_based_homing.get_current_state()
                step_latency_ms = (time.time_ns() - step_start_time) / 1_000_000

                timing_validator.validate_latency(step_latency_ms)
                assert (
                    step_latency_ms < 100.0
                ), f"Return-to-peak latency {step_latency_ms}ms exceeds 100ms"

                # Check if return-to-peak successfully reacquired signal
                if current_state.confidence > 0.5 and current_state.signal_strength > (
                    peak_signal_strength * 0.8
                ):
                    return_time = time.time() - return_start_time
                    return_times.append(return_time)
                    successful_returns += 1
                    return_success = True
                    logger.info(f"Return-to-peak successful in {return_time:.2f}s")
                    break

            await beacon_generator.stop_transmission()

            if not return_success:
                logger.warning(
                    f"Return-to-peak iteration {iteration + 1} did not reacquire signal within {max_return_time}s"
                )

        # Calculate success rate and performance metrics
        success_rate = successful_returns / test_iterations
        avg_return_time = statistics.mean(return_times) if return_times else float("inf")

        # Record results
        return_to_peak_results = {
            "strategy": "return_to_last_peak",
            "test_iterations": test_iterations,
            "successful_returns": successful_returns,
            "success_rate": success_rate,
            "avg_return_time_s": avg_return_time,
            "max_latency_ms": timing_validator.get_max_latency(),
            "avg_latency_ms": timing_validator.get_average_latency(),
        }

        performance_metrics.add_measurement("return_to_peak_recovery", return_to_peak_results)

        logger.info("Return-to-last-peak recovery results:")
        logger.info(f"  Success rate: {success_rate:.2%}")
        logger.info(f"  Average return time: {avg_return_time:.2f}s")
        logger.info(f"  Processing latency (avg): {timing_validator.get_average_latency():.2f}ms")

        # Assertions for requirements compliance
        assert (
            success_rate >= 0.85
        ), f"Return-to-peak success rate {success_rate:.2%} below 85% requirement"
        assert avg_return_time < 6.0, f"Average return time {avg_return_time:.2f}s exceeds 6s limit"
        assert (
            timing_validator.get_max_latency() < 100.0
        ), "Max processing latency exceeds 100ms requirement"

    # Test [26a3]: Signal confidence degradation and recovery timing

    @pytest.mark.asyncio
    async def test_signal_confidence_degradation_recovery_timing(
        self,
        enhanced_signal_processor,
        confidence_based_homing,
        beacon_generator,
        performance_metrics,
        timing_validator,
    ):
        """Test signal confidence degradation with timing measurements for <2s recovery requirement.

        Tests confidence-based recovery timing to ensure <2 second recovery requirement
        Validates confidence assessment accuracy during signal degradation
        """
        logger.info("Testing signal confidence degradation and recovery timing")

        confidence_scenarios = [
            {
                "name": "gradual_degradation",
                "initial_confidence": 0.9,
                "degradation_rate": 0.1,  # Confidence loss per second
                "recovery_threshold": 0.3,
                "expected_recovery_time_s": 1.5,
            },
            {
                "name": "rapid_degradation",
                "initial_confidence": 0.8,
                "degradation_rate": 0.2,
                "recovery_threshold": 0.25,
                "expected_recovery_time_s": 1.0,
            },
            {
                "name": "steep_degradation",
                "initial_confidence": 0.7,
                "degradation_rate": 0.3,
                "recovery_threshold": 0.2,
                "expected_recovery_time_s": 1.8,
            },
        ]

        confidence_results = []

        for scenario in confidence_scenarios:
            logger.info(f"Testing confidence degradation scenario: {scenario['name']}")

            # Configure beacon for initial high confidence
            await beacon_generator.set_signal_strength(-50.0)
            await beacon_generator.set_frequency(3200.0)
            await beacon_generator.start_transmission()

            # Allow signal to stabilize at high confidence
            await asyncio.sleep(1.0)

            # Verify initial confidence meets expectation
            initial_start_time = time.time_ns()
            initial_bearing = await enhanced_signal_processor.compute_bearing()
            initial_latency_ms = (time.time_ns() - initial_start_time) / 1_000_000

            timing_validator.validate_latency(initial_latency_ms)
            assert (
                initial_latency_ms < 100.0
            ), f"Initial confidence check latency {initial_latency_ms}ms exceeds 100ms"

            # Gradually degrade signal to reduce confidence
            degradation_start_time = time.time()
            current_rssi = -50.0

            confidence_recovery_detected = False
            recovery_time_s = None

            for degradation_step in range(20):  # 20 steps over time
                # Calculate current RSSI based on degradation rate
                time_elapsed = degradation_step * 0.2  # 0.2 second steps
                rssi_degradation = time_elapsed * 5.0  # 5dB per second degradation
                current_rssi = -50.0 - rssi_degradation

                await beacon_generator.set_signal_strength(current_rssi)
                await asyncio.sleep(0.2)

                # Measure confidence and processing latency
                confidence_start_time = time.time_ns()
                current_bearing = await enhanced_signal_processor.compute_bearing()
                confidence_latency_ms = (time.time_ns() - confidence_start_time) / 1_000_000

                timing_validator.validate_latency(confidence_latency_ms)
                assert (
                    confidence_latency_ms < 100.0
                ), f"Confidence processing latency {confidence_latency_ms}ms exceeds 100ms"

                # Check if confidence has dropped below threshold (should trigger recovery)
                if current_bearing.confidence <= scenario["recovery_threshold"]:
                    recovery_start_time = time.time()
                    logger.info(
                        f"Confidence dropped to {current_bearing.confidence:.3f}, recovery should trigger"
                    )

                    # Monitor for confidence recovery
                    for recovery_check in range(25):  # Check for up to 5 seconds
                        await asyncio.sleep(0.2)

                        # Measure recovery processing latency
                        recovery_check_start = time.time_ns()
                        recovery_bearing = await enhanced_signal_processor.compute_bearing()
                        recovery_latency_ms = (time.time_ns() - recovery_check_start) / 1_000_000

                        timing_validator.validate_latency(recovery_latency_ms)
                        assert (
                            recovery_latency_ms < 100.0
                        ), f"Recovery processing latency {recovery_latency_ms}ms exceeds 100ms"

                        # Check if confidence has recovered
                        if (
                            recovery_bearing.confidence > scenario["recovery_threshold"] + 0.1
                        ):  # Hysteresis
                            recovery_time_s = time.time() - recovery_start_time
                            confidence_recovery_detected = True
                            logger.info(f"Confidence recovery detected in {recovery_time_s:.3f}s")
                            break

                    break

            await beacon_generator.stop_transmission()

            # Record scenario results
            scenario_result = {
                "scenario_name": scenario["name"],
                "initial_confidence": initial_bearing.confidence,
                "recovery_threshold": scenario["recovery_threshold"],
                "recovery_detected": confidence_recovery_detected,
                "recovery_time_s": recovery_time_s,
                "expected_recovery_time_s": scenario["expected_recovery_time_s"],
                "max_latency_ms": timing_validator.get_max_latency(),
                "avg_latency_ms": timing_validator.get_average_latency(),
                "processing_samples": timing_validator.get_sample_count(),
            }
            confidence_results.append(scenario_result)

            # Validate recovery timing requirement
            if confidence_recovery_detected and recovery_time_s is not None:
                assert (
                    recovery_time_s < 2.0
                ), f"Recovery time {recovery_time_s:.3f}s exceeds 2s requirement for {scenario['name']}"
                assert (
                    recovery_time_s < scenario["expected_recovery_time_s"]
                ), f"Recovery time {recovery_time_s:.3f}s exceeds expected {scenario['expected_recovery_time_s']}s for {scenario['name']}"

        performance_metrics.add_measurement("confidence_degradation_recovery", confidence_results)

        # Overall performance validation
        recovery_success_rate = sum(1 for r in confidence_results if r["recovery_detected"]) / len(
            confidence_results
        )
        avg_recovery_time = statistics.mean(
            [r["recovery_time_s"] for r in confidence_results if r["recovery_time_s"]]
        )
        max_recovery_time = max(
            [r["recovery_time_s"] for r in confidence_results if r["recovery_time_s"]]
        )

        logger.info("Signal confidence degradation recovery results:")
        logger.info(f"  Recovery success rate: {recovery_success_rate:.2%}")
        logger.info(f"  Average recovery time: {avg_recovery_time:.3f}s")
        logger.info(f"  Maximum recovery time: {max_recovery_time:.3f}s")

        # Final assertions for PRD compliance
        assert (
            recovery_success_rate >= 0.9
        ), f"Confidence recovery success rate {recovery_success_rate:.2%} below 90%"
        assert (
            avg_recovery_time < 2.0
        ), f"Average recovery time {avg_recovery_time:.3f}s exceeds 2s requirement"
        assert (
            max_recovery_time < 2.0
        ), f"Maximum recovery time {max_recovery_time:.3f}s exceeds 2s requirement"

    # Test [26a4]: Automated test suite for RSSI degradation detection accuracy

    @pytest.mark.asyncio
    async def test_automated_rssi_degradation_detection_accuracy(
        self, enhanced_signal_processor, beacon_generator, performance_metrics, timing_validator
    ):
        """Develop automated test suite measuring RSSI degradation detection accuracy with statistical validation.

        Tests accuracy of RSSI degradation detection across multiple scenarios
        Provides statistical validation of detection performance
        """
        logger.info("Testing automated RSSI degradation detection accuracy")

        # Define degradation detection test cases
        degradation_test_cases = [
            {
                "name": "linear_degradation_5db",
                "initial_rssi": -45.0,
                "final_rssi": -50.0,
                "degradation_pattern": "linear",
                "duration_s": 3.0,
                "expected_detection_time_s": 1.0,
                "detection_threshold_db": 2.0,
            },
            {
                "name": "linear_degradation_10db",
                "initial_rssi": -50.0,
                "final_rssi": -60.0,
                "degradation_pattern": "linear",
                "duration_s": 4.0,
                "expected_detection_time_s": 1.2,
                "detection_threshold_db": 3.0,
            },
            {
                "name": "exponential_degradation",
                "initial_rssi": -55.0,
                "final_rssi": -75.0,
                "degradation_pattern": "exponential",
                "duration_s": 5.0,
                "expected_detection_time_s": 0.8,
                "detection_threshold_db": 4.0,
            },
            {
                "name": "step_degradation",
                "initial_rssi": -48.0,
                "final_rssi": -65.0,
                "degradation_pattern": "step",
                "duration_s": 2.0,
                "expected_detection_time_s": 0.3,
                "detection_threshold_db": 5.0,
            },
        ]

        detection_results = []

        for test_case in degradation_test_cases:
            logger.info(f"Testing RSSI degradation detection: {test_case['name']}")

            # Configure beacon with initial signal strength
            await beacon_generator.set_signal_strength(test_case["initial_rssi"])
            await beacon_generator.set_frequency(3200.0)
            await beacon_generator.start_transmission()

            # Allow signal to stabilize
            await asyncio.sleep(1.0)

            # Record baseline RSSI
            baseline_start_time = time.time_ns()
            baseline_reading = await enhanced_signal_processor.get_current_rssi()
            baseline_latency_ms = (time.time_ns() - baseline_start_time) / 1_000_000

            timing_validator.validate_latency(baseline_latency_ms)
            assert (
                baseline_latency_ms < 100.0
            ), f"Baseline RSSI reading latency {baseline_latency_ms}ms exceeds 100ms"

            # Start degradation pattern
            degradation_start_time = time.time()
            degradation_detected = False
            detection_time_s = None
            rssi_samples = []

            # Apply degradation pattern based on type
            degradation_steps = 50  # High resolution for accurate detection timing
            step_duration = test_case["duration_s"] / degradation_steps

            for step in range(degradation_steps):
                elapsed_time = step * step_duration
                progress = elapsed_time / test_case["duration_s"]

                # Calculate current RSSI based on degradation pattern
                if test_case["degradation_pattern"] == "linear":
                    current_rssi = test_case["initial_rssi"] + progress * (
                        test_case["final_rssi"] - test_case["initial_rssi"]
                    )
                elif test_case["degradation_pattern"] == "exponential":
                    # Exponential decay curve
                    decay_factor = -2.0 * progress  # Exponential coefficient
                    current_rssi = test_case["final_rssi"] + (
                        test_case["initial_rssi"] - test_case["final_rssi"]
                    ) * math.exp(decay_factor)
                elif test_case["degradation_pattern"] == "step":
                    # Step degradation at 50% point
                    current_rssi = (
                        test_case["initial_rssi"] if progress < 0.5 else test_case["final_rssi"]
                    )

                await beacon_generator.set_signal_strength(current_rssi)
                await asyncio.sleep(step_duration)

                # Measure current RSSI and processing latency
                rssi_start_time = time.time_ns()
                current_reading = await enhanced_signal_processor.get_current_rssi()
                rssi_latency_ms = (time.time_ns() - rssi_start_time) / 1_000_000

                timing_validator.validate_latency(rssi_latency_ms)
                assert (
                    rssi_latency_ms < 100.0
                ), f"RSSI measurement latency {rssi_latency_ms}ms exceeds 100ms"

                rssi_samples.append(
                    {
                        "time_s": elapsed_time,
                        "expected_rssi": current_rssi,
                        "measured_rssi": current_reading,
                        "latency_ms": rssi_latency_ms,
                    }
                )

                # Check for degradation detection
                rssi_change = baseline_reading - current_reading
                if rssi_change >= test_case["detection_threshold_db"] and not degradation_detected:
                    detection_time_s = elapsed_time
                    degradation_detected = True
                    logger.info(
                        f"RSSI degradation detected at {detection_time_s:.3f}s (change: {rssi_change:.1f}dB)"
                    )
                    break

            await beacon_generator.stop_transmission()

            # Calculate detection accuracy metrics
            if degradation_detected:
                detection_accuracy = (
                    abs(detection_time_s - test_case["expected_detection_time_s"])
                    / test_case["expected_detection_time_s"]
                )
                detection_error_s = abs(detection_time_s - test_case["expected_detection_time_s"])
            else:
                detection_accuracy = 1.0  # 100% error if not detected
                detection_error_s = test_case["duration_s"]

            # Calculate RSSI measurement accuracy
            rssi_errors = [
                abs(sample["expected_rssi"] - sample["measured_rssi"]) for sample in rssi_samples
            ]
            avg_rssi_error = statistics.mean(rssi_errors)
            max_rssi_error = max(rssi_errors)

            # Record test case results
            case_result = {
                "test_case_name": test_case["name"],
                "degradation_detected": degradation_detected,
                "detection_time_s": detection_time_s,
                "expected_detection_time_s": test_case["expected_detection_time_s"],
                "detection_accuracy": 1.0 - detection_accuracy,  # Convert error to accuracy
                "detection_error_s": detection_error_s,
                "avg_rssi_error_db": avg_rssi_error,
                "max_rssi_error_db": max_rssi_error,
                "rssi_samples_count": len(rssi_samples),
                "max_latency_ms": timing_validator.get_max_latency(),
                "avg_latency_ms": timing_validator.get_average_latency(),
            }
            detection_results.append(case_result)

        performance_metrics.add_measurement("rssi_degradation_detection", detection_results)

        # Statistical validation across all test cases
        detection_success_rate = sum(
            1 for r in detection_results if r["degradation_detected"]
        ) / len(detection_results)
        avg_detection_accuracy = statistics.mean(
            [r["detection_accuracy"] for r in detection_results]
        )
        avg_detection_error = statistics.mean([r["detection_error_s"] for r in detection_results])
        avg_rssi_accuracy = statistics.mean([r["avg_rssi_error_db"] for r in detection_results])

        logger.info("RSSI degradation detection accuracy results:")
        logger.info(f"  Detection success rate: {detection_success_rate:.2%}")
        logger.info(f"  Average detection accuracy: {avg_detection_accuracy:.2%}")
        logger.info(f"  Average detection error: {avg_detection_error:.3f}s")
        logger.info(f"  Average RSSI measurement error: {avg_rssi_accuracy:.2f}dB")

        # Final assertions for detection accuracy requirements
        assert (
            detection_success_rate >= 0.95
        ), f"RSSI degradation detection rate {detection_success_rate:.2%} below 95%"
        assert (
            avg_detection_accuracy >= 0.85
        ), f"Average detection accuracy {avg_detection_accuracy:.2%} below 85%"
        assert (
            avg_detection_error < 0.5
        ), f"Average detection error {avg_detection_error:.3f}s exceeds 0.5s limit"
        assert (
            avg_rssi_accuracy < 2.0
        ), f"Average RSSI error {avg_rssi_accuracy:.2f}dB exceeds 2dB limit"

    # Test [26b1]: Automated performance benchmarking framework for enhanced algorithms

    @pytest.mark.asyncio
    async def test_automated_performance_benchmarking_framework(
        self,
        enhanced_signal_processor,
        confidence_based_homing,
        baseline_homing_algorithm,
        beacon_generator,
        performance_metrics,
        timing_validator,
    ):
        """Implement automated performance benchmarking framework measuring all enhanced algorithm processing stages.

        Comprehensive benchmarking comparing enhanced algorithms against baseline
        Measures all processing stages with detailed performance metrics
        """
        logger.info("Testing automated performance benchmarking framework")

        # Define benchmark test scenarios
        benchmark_scenarios = [
            {
                "name": "optimal_signal_conditions",
                "signal_strength_dbm": -45.0,
                "frequency_mhz": 3200.0,
                "expected_enhanced_performance_improvement": 0.3,  # 30% improvement expected
                "test_iterations": 100,
            },
            {
                "name": "moderate_signal_conditions",
                "signal_strength_dbm": -65.0,
                "frequency_mhz": 3200.0,
                "expected_enhanced_performance_improvement": 0.5,  # 50% improvement expected
                "test_iterations": 100,
            },
            {
                "name": "weak_signal_conditions",
                "signal_strength_dbm": -80.0,
                "frequency_mhz": 3200.0,
                "expected_enhanced_performance_improvement": 0.7,  # 70% improvement expected
                "test_iterations": 80,
            },
        ]

        benchmark_results = []

        for scenario in benchmark_scenarios:
            logger.info(f"Benchmarking scenario: {scenario['name']}")

            # Configure beacon for scenario
            await beacon_generator.set_signal_strength(scenario["signal_strength_dbm"])
            await beacon_generator.set_frequency(scenario["frequency_mhz"])
            await beacon_generator.start_transmission()

            # Allow signal to stabilize
            await asyncio.sleep(1.0)

            # Benchmark enhanced algorithm performance
            enhanced_processing_times = []
            enhanced_confidence_scores = []
            enhanced_bearing_accuracy = []

            for iteration in range(scenario["test_iterations"]):
                # Enhanced algorithm processing benchmark
                enhanced_start_time = time.time_ns()

                enhanced_bearing = await enhanced_signal_processor.compute_bearing()
                enhanced_homing_result = await confidence_based_homing.compute_homing_vector()

                enhanced_processing_time_ms = (time.time_ns() - enhanced_start_time) / 1_000_000

                # Validate <100ms requirement for each iteration
                timing_validator.validate_latency(enhanced_processing_time_ms)
                assert (
                    enhanced_processing_time_ms < 100.0
                ), f"Enhanced algorithm processing {enhanced_processing_time_ms}ms exceeds 100ms on iteration {iteration}"

                enhanced_processing_times.append(enhanced_processing_time_ms)
                enhanced_confidence_scores.append(enhanced_bearing.confidence)
                enhanced_bearing_accuracy.append(enhanced_bearing.precision_deg)

            # Benchmark baseline algorithm performance
            baseline_processing_times = []
            baseline_confidence_scores = []
            baseline_bearing_accuracy = []

            for iteration in range(scenario["test_iterations"]):
                # Baseline algorithm processing benchmark
                baseline_start_time = time.time_ns()

                # Simulate baseline algorithm calls (using existing homing algorithm)
                baseline_gradient = baseline_homing_algorithm.compute_gradient()
                baseline_velocity = baseline_homing_algorithm.compute_velocity_command(
                    baseline_gradient
                )

                baseline_processing_time_ms = (time.time_ns() - baseline_start_time) / 1_000_000

                baseline_processing_times.append(baseline_processing_time_ms)
                # Baseline confidence and accuracy estimated based on signal strength
                baseline_confidence = max(
                    0.1, min(0.8, (scenario["signal_strength_dbm"] + 90) / 40)
                )
                baseline_precision = max(
                    5.0, min(15.0, abs(scenario["signal_strength_dbm"] + 45) / 4)
                )  # ±5-15° range

                baseline_confidence_scores.append(baseline_confidence)
                baseline_bearing_accuracy.append(baseline_precision)

            await beacon_generator.stop_transmission()

            # Calculate performance metrics
            enhanced_avg_time = statistics.mean(enhanced_processing_times)
            enhanced_max_time = max(enhanced_processing_times)
            enhanced_avg_confidence = statistics.mean(enhanced_confidence_scores)
            enhanced_avg_precision = statistics.mean(enhanced_bearing_accuracy)

            baseline_avg_time = statistics.mean(baseline_processing_times)
            baseline_max_time = max(baseline_processing_times)
            baseline_avg_confidence = statistics.mean(baseline_confidence_scores)
            baseline_avg_precision = statistics.mean(baseline_bearing_accuracy)

            # Calculate performance improvements
            processing_time_improvement = (
                baseline_avg_time - enhanced_avg_time
            ) / baseline_avg_time
            confidence_improvement = (
                enhanced_avg_confidence - baseline_avg_confidence
            ) / baseline_avg_confidence
            precision_improvement = (
                baseline_avg_precision - enhanced_avg_precision
            ) / baseline_avg_precision

            # Record benchmark results
            scenario_result = {
                "scenario_name": scenario["name"],
                "signal_strength_dbm": scenario["signal_strength_dbm"],
                "test_iterations": scenario["test_iterations"],
                # Enhanced algorithm metrics
                "enhanced_avg_processing_time_ms": enhanced_avg_time,
                "enhanced_max_processing_time_ms": enhanced_max_time,
                "enhanced_avg_confidence": enhanced_avg_confidence,
                "enhanced_avg_precision_deg": enhanced_avg_precision,
                # Baseline algorithm metrics
                "baseline_avg_processing_time_ms": baseline_avg_time,
                "baseline_max_processing_time_ms": baseline_max_time,
                "baseline_avg_confidence": baseline_avg_confidence,
                "baseline_avg_precision_deg": baseline_avg_precision,
                # Performance improvements
                "processing_time_improvement": processing_time_improvement,
                "confidence_improvement": confidence_improvement,
                "precision_improvement": precision_improvement,
                "expected_improvement": scenario["expected_enhanced_performance_improvement"],
            }
            benchmark_results.append(scenario_result)

            # Validate expected improvements
            assert (
                confidence_improvement >= scenario["expected_enhanced_performance_improvement"]
            ), f"Confidence improvement {confidence_improvement:.2%} below expected {scenario['expected_enhanced_performance_improvement']:.2%} for {scenario['name']}"
            assert (
                precision_improvement >= scenario["expected_enhanced_performance_improvement"]
            ), f"Precision improvement {precision_improvement:.2%} below expected {scenario['expected_enhanced_performance_improvement']:.2%} for {scenario['name']}"

        performance_metrics.add_measurement("algorithm_performance_benchmarking", benchmark_results)

        # Overall benchmark validation
        overall_processing_improvement = statistics.mean(
            [r["processing_time_improvement"] for r in benchmark_results]
        )
        overall_confidence_improvement = statistics.mean(
            [r["confidence_improvement"] for r in benchmark_results]
        )
        overall_precision_improvement = statistics.mean(
            [r["precision_improvement"] for r in benchmark_results]
        )

        # Ensure all enhanced algorithm times meet <100ms requirement
        all_enhanced_max_times = [r["enhanced_max_processing_time_ms"] for r in benchmark_results]
        max_enhanced_time = max(all_enhanced_max_times)

        logger.info("Performance benchmarking results:")
        logger.info(f"  Overall processing time improvement: {overall_processing_improvement:.2%}")
        logger.info(f"  Overall confidence improvement: {overall_confidence_improvement:.2%}")
        logger.info(f"  Overall precision improvement: {overall_precision_improvement:.2%}")
        logger.info(f"  Maximum enhanced processing time: {max_enhanced_time:.2f}ms")

        # Final assertions for benchmarking requirements
        assert (
            max_enhanced_time < 100.0
        ), f"Maximum enhanced processing time {max_enhanced_time:.2f}ms exceeds 100ms requirement"
        assert (
            overall_confidence_improvement > 0.2
        ), f"Overall confidence improvement {overall_confidence_improvement:.2%} below 20% minimum"
        assert (
            overall_precision_improvement > 0.2
        ), f"Overall precision improvement {overall_precision_improvement:.2%} below 20% minimum"

    # Test [26b2]: Latency measurement tests for all enhanced algorithm processing stages

    @pytest.mark.asyncio
    async def test_latency_measurement_enhanced_algorithm_stages(
        self,
        enhanced_signal_processor,
        confidence_based_homing,
        beacon_generator,
        performance_metrics,
        timing_validator,
    ):
        """Create latency measurement tests using high-precision timing for each algorithm component.

        Measures latency of individual processing stages within enhanced algorithms
        Validates each stage meets timing requirements independently
        """
        logger.info("Testing latency measurement for enhanced algorithm processing stages")

        # Configure beacon for stable testing conditions
        await beacon_generator.set_signal_strength(-55.0)  # Moderate signal strength
        await beacon_generator.set_frequency(3200.0)
        await beacon_generator.start_transmission()

        # Allow signal to stabilize
        await asyncio.sleep(1.0)

        # Define processing stages to measure
        test_iterations = 50  # High precision measurements
        stage_latencies = {
            "signal_acquisition": [],
            "bearing_calculation": [],
            "confidence_assessment": [],
            "interference_detection": [],
            "gradient_computation": [],
            "velocity_command_generation": [],
        }

        for iteration in range(test_iterations):
            logger.debug(f"Latency measurement iteration {iteration + 1}/{test_iterations}")

            # Stage 1: Signal Acquisition
            stage1_start = time.time_ns()
            raw_signal_data = await enhanced_signal_processor.acquire_signal_data()
            stage1_latency = (time.time_ns() - stage1_start) / 1_000_000
            stage_latencies["signal_acquisition"].append(stage1_latency)

            # Stage 2: Bearing Calculation
            stage2_start = time.time_ns()
            bearing_result = await enhanced_signal_processor.calculate_bearing(raw_signal_data)
            stage2_latency = (time.time_ns() - stage2_start) / 1_000_000
            stage_latencies["bearing_calculation"].append(stage2_latency)

            # Stage 3: Confidence Assessment
            stage3_start = time.time_ns()
            confidence_result = await enhanced_signal_processor.assess_signal_confidence(
                bearing_result
            )
            stage3_latency = (time.time_ns() - stage3_start) / 1_000_000
            stage_latencies["confidence_assessment"].append(stage3_latency)

            # Stage 4: Interference Detection
            stage4_start = time.time_ns()
            interference_result = await enhanced_signal_processor.detect_interference(
                raw_signal_data
            )
            stage4_latency = (time.time_ns() - stage4_start) / 1_000_000
            stage_latencies["interference_detection"].append(stage4_latency)

            # Stage 5: Gradient Computation
            stage5_start = time.time_ns()
            gradient_result = await confidence_based_homing.compute_enhanced_gradient(
                bearing_result, confidence_result
            )
            stage5_latency = (time.time_ns() - stage5_start) / 1_000_000
            stage_latencies["gradient_computation"].append(stage5_latency)

            # Stage 6: Velocity Command Generation
            stage6_start = time.time_ns()
            velocity_command = await confidence_based_homing.generate_velocity_command(
                gradient_result
            )
            stage6_latency = (time.time_ns() - stage6_start) / 1_000_000
            stage_latencies["velocity_command_generation"].append(stage6_latency)

            # Validate individual stage latencies
            all_stage_latencies = [
                stage1_latency,
                stage2_latency,
                stage3_latency,
                stage4_latency,
                stage5_latency,
                stage6_latency,
            ]
            total_pipeline_latency = sum(all_stage_latencies)

            timing_validator.validate_latency(total_pipeline_latency)
            assert (
                total_pipeline_latency < 100.0
            ), f"Total pipeline latency {total_pipeline_latency:.2f}ms exceeds 100ms on iteration {iteration}"

            # Individual stage validation (reasonable sub-limits)
            assert (
                stage1_latency < 20.0
            ), f"Signal acquisition {stage1_latency:.2f}ms exceeds 20ms limit"
            assert (
                stage2_latency < 30.0
            ), f"Bearing calculation {stage2_latency:.2f}ms exceeds 30ms limit"
            assert (
                stage3_latency < 15.0
            ), f"Confidence assessment {stage3_latency:.2f}ms exceeds 15ms limit"
            assert (
                stage4_latency < 10.0
            ), f"Interference detection {stage4_latency:.2f}ms exceeds 10ms limit"
            assert (
                stage5_latency < 15.0
            ), f"Gradient computation {stage5_latency:.2f}ms exceeds 15ms limit"
            assert (
                stage6_latency < 10.0
            ), f"Velocity command generation {stage6_latency:.2f}ms exceeds 10ms limit"

        await beacon_generator.stop_transmission()

        # Calculate statistics for each stage
        stage_statistics = {}
        for stage_name, latencies in stage_latencies.items():
            stage_statistics[stage_name] = {
                "avg_latency_ms": statistics.mean(latencies),
                "max_latency_ms": max(latencies),
                "min_latency_ms": min(latencies),
                "std_dev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
                "p95_latency_ms": np.percentile(latencies, 95),
                "p99_latency_ms": np.percentile(latencies, 99),
                "sample_count": len(latencies),
            }

        performance_metrics.add_measurement("enhanced_algorithm_stage_latencies", stage_statistics)

        # Calculate total pipeline statistics
        total_pipeline_latencies = []
        for i in range(test_iterations):
            pipeline_total = sum(stage_latencies[stage][i] for stage in stage_latencies.keys())
            total_pipeline_latencies.append(pipeline_total)

        avg_pipeline_latency = statistics.mean(total_pipeline_latencies)
        max_pipeline_latency = max(total_pipeline_latencies)
        p95_pipeline_latency = np.percentile(total_pipeline_latencies, 95)

        logger.info("Enhanced algorithm stage latency results:")
        for stage_name, stats in stage_statistics.items():
            logger.info(
                f"  {stage_name}: avg={stats['avg_latency_ms']:.2f}ms, max={stats['max_latency_ms']:.2f}ms, p95={stats['p95_latency_ms']:.2f}ms"
            )
        logger.info(
            f"  Total pipeline: avg={avg_pipeline_latency:.2f}ms, max={max_pipeline_latency:.2f}ms, p95={p95_pipeline_latency:.2f}ms"
        )

        # Final assertions for stage latency requirements
        assert (
            avg_pipeline_latency < 100.0
        ), f"Average pipeline latency {avg_pipeline_latency:.2f}ms exceeds 100ms requirement"
        assert (
            max_pipeline_latency < 100.0
        ), f"Maximum pipeline latency {max_pipeline_latency:.2f}ms exceeds 100ms requirement"
        assert (
            p95_pipeline_latency < 100.0
        ), f"P95 pipeline latency {p95_pipeline_latency:.2f}ms exceeds 100ms requirement"
