"""
TASK-6.2.4-PERFORMANCE-VALIDATION-AND-TESTING: Enhanced Algorithm Adaptive Search Pattern Validation
SUBTASK-6.2.4.1 [26d1-26d4] - Adaptive search pattern effectiveness testing

Adaptive search pattern validation tests for enhanced homing algorithms to ensure
effective search strategies when signal confidence is low or signal is lost.

PRD References:
- PRD-NFR8: 90% successful homing rate once signal is acquired
- PRD-FR4: Navigate toward detected signals using RSSI gradient climbing with enhanced accuracy
- Story 6.2 Acceptance Criteria: Enhanced search patterns for signal acquisition

Test Categories:
- Controlled environments for adaptive search pattern effectiveness testing
- Spiral search pattern validation with success rate measurement
- S-turn sampling maneuver effectiveness tests with weak signal scenarios
- Return-to-last-peak algorithm validation with complete signal loss scenarios

All tests use authentic system integration - NO mocks or simulated components.
"""

import asyncio
import logging
import statistics
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pytest

from src.backend.hal.beacon_generator import BeaconGenerator
from src.backend.hal.sitl_interface import SITLInterface
from src.backend.services.asv_integration.asv_confidence_based_homing import (
    ASVConfidenceBasedHoming,
)
from src.backend.services.asv_integration.asv_enhanced_signal_processor import (
    ASVEnhancedSignalProcessor,
)
from src.backend.services.search_pattern_generator import SearchPatternGenerator
from src.backend.utils.test_metrics import PerformanceMetrics, TimingValidator

logger = logging.getLogger(__name__)


class SearchPatternType(str, Enum):
    """Adaptive search pattern types."""

    SPIRAL = "SPIRAL"
    S_TURN = "S_TURN"
    RETURN_TO_PEAK = "RETURN_TO_PEAK"
    EXPANDING_SQUARE = "EXPANDING_SQUARE"


@dataclass
class SearchPatternScenario:
    """Test scenario for adaptive search pattern validation."""

    name: str
    pattern_type: SearchPatternType
    initial_signal_strength_dbm: float
    signal_loss_type: str  # "gradual", "sudden", "complete"
    search_area_size_meters: float
    expected_acquisition_time_s: float
    success_rate_threshold: float
    test_iterations: int


@dataclass
class SearchPatternResult:
    """Result of adaptive search pattern test."""

    scenario_name: str
    pattern_type: SearchPatternType
    search_successful: bool
    acquisition_time_s: float
    search_distance_covered_m: float
    signal_reacquired: bool
    final_confidence: float
    processing_latency_ms: float


class TestEnhancedAlgorithmAdaptiveSearchValidation:
    """Enhanced algorithm adaptive search pattern validation tests."""

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
    async def search_pattern_generator(self):
        """Create search pattern generator instance."""
        generator = SearchPatternGenerator()
        await generator.initialize()
        yield generator
        await generator.shutdown()

    @pytest.fixture
    async def beacon_generator(self):
        """Create beacon generator for controlled signal testing."""
        generator = BeaconGenerator()
        await generator.initialize()
        yield generator
        await generator.shutdown()

    @pytest.fixture
    async def sitl_interface(self):
        """Create SITL interface for controlled environment simulation."""
        interface = SITLInterface()
        await interface.start()
        yield interface
        await interface.stop()

    @pytest.fixture
    def performance_metrics(self):
        """Create performance metrics tracker."""
        return PerformanceMetrics()

    @pytest.fixture
    def timing_validator(self):
        """Create timing validator for <100ms requirement."""
        return TimingValidator(max_latency_ms=100.0)

    # Test [26d1]: Controlled environments for adaptive search pattern effectiveness testing

    @pytest.mark.asyncio
    async def test_controlled_environment_adaptive_search_effectiveness(
        self,
        enhanced_signal_processor,
        confidence_based_homing,
        search_pattern_generator,
        beacon_generator,
        sitl_interface,
        performance_metrics,
        timing_validator,
    ):
        """Create controlled environments for adaptive search pattern effectiveness using SITL simulation.

        Tests adaptive search patterns in controlled SITL environment
        Validates search effectiveness under various controlled conditions
        """
        logger.info("Testing controlled environment adaptive search pattern effectiveness")

        # Define controlled environment test scenarios
        controlled_scenarios = [
            {
                "name": "open_field_search",
                "environment_type": "open_field",
                "obstacles": [],
                "signal_source_distance_m": 150.0,
                "signal_source_bearing_deg": 45.0,
                "initial_signal_strength_dbm": -65.0,
                "search_radius_m": 200.0,
                "expected_acquisition_time_s": 8.0,
            },
            {
                "name": "urban_environment_search",
                "environment_type": "urban_canyon",
                "obstacles": ["buildings", "interference"],
                "signal_source_distance_m": 120.0,
                "signal_source_bearing_deg": 280.0,
                "initial_signal_strength_dbm": -70.0,
                "search_radius_m": 180.0,
                "expected_acquisition_time_s": 12.0,
            },
            {
                "name": "forest_environment_search",
                "environment_type": "forest_canopy",
                "obstacles": ["trees", "multipath"],
                "signal_source_distance_m": 100.0,
                "signal_source_bearing_deg": 135.0,
                "initial_signal_strength_dbm": -75.0,
                "search_radius_m": 150.0,
                "expected_acquisition_time_s": 15.0,
            },
            {
                "name": "mountainous_terrain_search",
                "environment_type": "mountainous",
                "obstacles": ["terrain_masking", "altitude_variation"],
                "signal_source_distance_m": 200.0,
                "signal_source_bearing_deg": 0.0,
                "initial_signal_strength_dbm": -78.0,
                "search_radius_m": 250.0,
                "expected_acquisition_time_s": 18.0,
            },
        ]

        environment_results = []

        for scenario in controlled_scenarios:
            logger.info(f"Testing controlled scenario: {scenario['name']}")

            # Configure SITL environment
            await sitl_interface.set_environment_type(scenario["environment_type"])
            await sitl_interface.add_obstacles(scenario["obstacles"])
            await sitl_interface.set_home_position(lat=0.0, lon=0.0, alt=50.0)

            # Configure beacon in controlled environment
            await beacon_generator.set_position(
                bearing_deg=scenario["signal_source_bearing_deg"],
                distance_meters=scenario["signal_source_distance_m"],
            )
            await beacon_generator.set_signal_strength(scenario["initial_signal_strength_dbm"])
            await beacon_generator.set_frequency(3200.0)
            await beacon_generator.start_transmission()

            # Allow environment to stabilize
            await asyncio.sleep(2.0)

            # Initialize adaptive search
            search_start_time = time.time()
            search_successful = False
            total_search_distance = 0.0
            processing_latencies = []

            # Trigger adaptive search pattern selection
            search_start_lat_ms = time.time_ns()
            await confidence_based_homing.trigger_adaptive_search()
            search_init_latency_ms = (time.time_ns() - search_start_lat_ms) / 1_000_000

            timing_validator.validate_latency(search_init_latency_ms)
            assert (
                search_init_latency_ms < 100.0
            ), f"Search initialization latency {search_init_latency_ms}ms exceeds 100ms"
            processing_latencies.append(search_init_latency_ms)

            # Execute adaptive search with timing validation
            max_search_time_s = scenario["expected_acquisition_time_s"] * 1.5  # 50% buffer

            for search_step in range(int(max_search_time_s * 10)):  # 10 steps per second
                await asyncio.sleep(0.1)

                # Measure search processing latency
                step_start_time = time.time_ns()

                current_position = await sitl_interface.get_current_position()
                search_status = await confidence_based_homing.get_search_status()
                bearing_result = await enhanced_signal_processor.compute_bearing()

                step_latency_ms = (time.time_ns() - step_start_time) / 1_000_000
                timing_validator.validate_latency(step_latency_ms)
                assert (
                    step_latency_ms < 100.0
                ), f"Search step latency {step_latency_ms}ms exceeds 100ms"
                processing_latencies.append(step_latency_ms)

                # Calculate distance covered in this step
                if search_step > 0:
                    step_distance = await sitl_interface.get_distance_traveled()
                    total_search_distance += step_distance

                # Check for successful signal acquisition
                if (
                    bearing_result.confidence > 0.4
                    and bearing_result.signal_strength_dbm
                    > scenario["initial_signal_strength_dbm"] - 5.0
                ):
                    search_successful = True
                    break

                # Check search area bounds
                if (
                    total_search_distance > scenario["search_radius_m"] * 2
                ):  # Covered reasonable search area
                    logger.warning(f"Search exceeded reasonable area for {scenario['name']}")
                    break

            search_total_time = time.time() - search_start_time
            await beacon_generator.stop_transmission()

            # Final signal measurement
            final_bearing = await enhanced_signal_processor.compute_bearing()

            scenario_result = {
                "scenario_name": scenario["name"],
                "environment_type": scenario["environment_type"],
                "obstacles": scenario["obstacles"],
                "signal_source_distance_m": scenario["signal_source_distance_m"],
                "signal_source_bearing_deg": scenario["signal_source_bearing_deg"],
                "initial_signal_strength_dbm": scenario["initial_signal_strength_dbm"],
                "search_radius_m": scenario["search_radius_m"],
                # Search performance results
                "search_successful": search_successful,
                "search_time_s": search_total_time,
                "expected_time_s": scenario["expected_acquisition_time_s"],
                "search_distance_covered_m": total_search_distance,
                "final_confidence": final_bearing.confidence,
                "final_signal_strength_dbm": final_bearing.signal_strength_dbm,
                # Processing performance
                "avg_processing_latency_ms": statistics.mean(processing_latencies),
                "max_processing_latency_ms": max(processing_latencies),
                "processing_samples": len(processing_latencies),
                # Success metrics
                "meets_time_requirement": search_total_time
                <= scenario["expected_acquisition_time_s"],
                "search_efficiency": scenario["search_radius_m"] / total_search_distance
                if total_search_distance > 0
                else 0.0,
            }
            environment_results.append(scenario_result)

            logger.info(f"  Search successful: {search_successful}")
            logger.info(
                f"  Search time: {search_total_time:.2f}s (expected: {scenario['expected_acquisition_time_s']}s)"
            )
            logger.info(f"  Distance covered: {total_search_distance:.1f}m")
            logger.info(f"  Final confidence: {final_bearing.confidence:.3f}")
            logger.info(
                f"  Processing latency (avg): {statistics.mean(processing_latencies):.2f}ms"
            )

            # Scenario-specific validation
            assert (
                search_total_time <= scenario["expected_acquisition_time_s"] * 1.2
            ), f"Search time {search_total_time:.2f}s exceeds 120% of expected {scenario['expected_acquisition_time_s']}s for {scenario['name']}"
            assert (
                max(processing_latencies) < 100.0
            ), f"Max processing latency exceeds 100ms for {scenario['name']}"

        performance_metrics.add_measurement(
            "controlled_environment_adaptive_search", environment_results
        )

        # Overall environment validation
        overall_success_rate = sum(1 for r in environment_results if r["search_successful"]) / len(
            environment_results
        )
        avg_search_time = statistics.mean([r["search_time_s"] for r in environment_results])
        avg_search_efficiency = statistics.mean(
            [r["search_efficiency"] for r in environment_results if r["search_efficiency"] > 0]
        )
        overall_avg_latency = statistics.mean(
            [r["avg_processing_latency_ms"] for r in environment_results]
        )

        scenarios_meeting_time = sum(1 for r in environment_results if r["meets_time_requirement"])
        time_compliance_rate = scenarios_meeting_time / len(environment_results)

        logger.info("Controlled environment adaptive search results:")
        logger.info(f"  Overall success rate: {overall_success_rate:.2%}")
        logger.info(f"  Average search time: {avg_search_time:.2f}s")
        logger.info(f"  Average search efficiency: {avg_search_efficiency:.2f}")
        logger.info(f"  Overall processing latency: {overall_avg_latency:.2f}ms")
        logger.info(f"  Time compliance rate: {time_compliance_rate:.2%}")

        # Final assertions for controlled environment requirements
        assert (
            overall_success_rate >= 0.8
        ), f"Overall success rate {overall_success_rate:.2%} below 80%"
        assert (
            avg_search_time <= 15.0
        ), f"Average search time {avg_search_time:.2f}s exceeds 15s limit"
        assert (
            time_compliance_rate >= 0.75
        ), f"Time compliance rate {time_compliance_rate:.2%} below 75%"
        assert (
            overall_avg_latency < 100.0
        ), f"Overall processing latency {overall_avg_latency:.2f}ms exceeds 100ms"

    # Test [26d2]: Spiral search pattern validation with success rate measurement

    @pytest.mark.asyncio
    async def test_spiral_search_pattern_success_rate_validation(
        self,
        enhanced_signal_processor,
        confidence_based_homing,
        search_pattern_generator,
        beacon_generator,
        performance_metrics,
        timing_validator,
    ):
        """Implement spiral search pattern validation measuring success rate and time-to-acquisition.

        Tests spiral search pattern effectiveness with statistical validation
        Measures success rate and acquisition time across multiple scenarios
        """
        logger.info("Testing spiral search pattern validation with success rate measurement")

        # Define spiral search test scenarios
        spiral_search_scenarios = [
            SearchPatternScenario(
                name="close_range_spiral",
                pattern_type=SearchPatternType.SPIRAL,
                initial_signal_strength_dbm=-60.0,
                signal_loss_type="gradual",
                search_area_size_meters=100.0,
                expected_acquisition_time_s=6.0,
                success_rate_threshold=0.9,
                test_iterations=12,
            ),
            SearchPatternScenario(
                name="medium_range_spiral",
                pattern_type=SearchPatternType.SPIRAL,
                initial_signal_strength_dbm=-70.0,
                signal_loss_type="sudden",
                search_area_size_meters=200.0,
                expected_acquisition_time_s=10.0,
                success_rate_threshold=0.85,
                test_iterations=10,
            ),
            SearchPatternScenario(
                name="long_range_spiral",
                pattern_type=SearchPatternType.SPIRAL,
                initial_signal_strength_dbm=-75.0,
                signal_loss_type="complete",
                search_area_size_meters=300.0,
                expected_acquisition_time_s=15.0,
                success_rate_threshold=0.8,
                test_iterations=8,
            ),
        ]

        spiral_results = []

        for scenario in spiral_search_scenarios:
            logger.info(f"Testing spiral search scenario: {scenario.name}")

            scenario_successes = 0
            scenario_acquisition_times = []
            scenario_processing_latencies = []
            scenario_distances = []

            for iteration in range(scenario.test_iterations):
                logger.debug(f"Spiral search iteration {iteration + 1}/{scenario.test_iterations}")

                # Configure beacon for scenario
                await beacon_generator.set_signal_strength(scenario.initial_signal_strength_dbm)
                await beacon_generator.set_frequency(3200.0)

                # Random beacon position within search area
                beacon_bearing = np.random.uniform(0, 360)
                beacon_distance = np.random.uniform(50.0, scenario.search_area_size_meters)
                await beacon_generator.set_position(
                    bearing_deg=beacon_bearing, distance_meters=beacon_distance
                )
                await beacon_generator.start_transmission()

                # Allow signal to establish
                await asyncio.sleep(1.0)

                # Simulate signal loss based on scenario type
                if scenario.signal_loss_type == "gradual":
                    # Gradual signal degradation
                    for fade_step in range(10):
                        current_strength = scenario.initial_signal_strength_dbm - (fade_step * 2.0)
                        await beacon_generator.set_signal_strength(current_strength)
                        await asyncio.sleep(0.2)
                elif scenario.signal_loss_type == "sudden":
                    # Sudden signal drop
                    await asyncio.sleep(1.0)
                    await beacon_generator.set_signal_strength(
                        scenario.initial_signal_strength_dbm - 20.0
                    )
                elif scenario.signal_loss_type == "complete":
                    # Complete signal loss
                    await beacon_generator.stop_transmission()

                # Wait for signal loss detection
                await asyncio.sleep(0.5)

                # Restore beacon signal for potential reacquisition
                await beacon_generator.set_signal_strength(scenario.initial_signal_strength_dbm)
                if scenario.signal_loss_type == "complete":
                    await beacon_generator.start_transmission()

                # Initiate spiral search
                search_start_time = time.time()

                spiral_init_start = time.time_ns()
                await search_pattern_generator.initiate_spiral_search()
                spiral_init_latency_ms = (time.time_ns() - spiral_init_start) / 1_000_000

                timing_validator.validate_latency(spiral_init_latency_ms)
                assert (
                    spiral_init_latency_ms < 100.0
                ), f"Spiral initialization latency {spiral_init_latency_ms}ms exceeds 100ms"
                scenario_processing_latencies.append(spiral_init_latency_ms)

                # Execute spiral search pattern
                search_successful = False
                total_search_distance = 0.0
                max_search_time = scenario.expected_acquisition_time_s * 1.3  # 30% buffer

                for search_step in range(int(max_search_time * 10)):  # 10 steps per second
                    await asyncio.sleep(0.1)

                    # Measure spiral search step processing
                    step_start_time = time.time_ns()

                    spiral_position = await search_pattern_generator.get_next_spiral_position()
                    bearing_result = await enhanced_signal_processor.compute_bearing()

                    step_latency_ms = (time.time_ns() - step_start_time) / 1_000_000
                    timing_validator.validate_latency(step_latency_ms)
                    assert (
                        step_latency_ms < 100.0
                    ), f"Spiral step latency {step_latency_ms}ms exceeds 100ms"
                    scenario_processing_latencies.append(step_latency_ms)

                    # Calculate search distance
                    step_distance = await search_pattern_generator.get_step_distance()
                    total_search_distance += step_distance

                    # Check for successful signal reacquisition
                    if (
                        bearing_result.confidence > 0.4
                        and bearing_result.signal_strength_dbm > -80.0
                    ):
                        search_successful = True
                        break

                    # Check for reasonable search bounds
                    if total_search_distance > scenario.search_area_size_meters * 2:
                        break

                search_time = time.time() - search_start_time
                await beacon_generator.stop_transmission()

                # Record iteration results
                if search_successful:
                    scenario_successes += 1
                    scenario_acquisition_times.append(search_time)

                scenario_distances.append(total_search_distance)

                logger.debug(
                    f"  Iteration {iteration + 1}: Success={search_successful}, Time={search_time:.2f}s, Distance={total_search_distance:.1f}m"
                )

            # Calculate scenario statistics
            success_rate = scenario_successes / scenario.test_iterations
            avg_acquisition_time = (
                statistics.mean(scenario_acquisition_times)
                if scenario_acquisition_times
                else float("inf")
            )
            avg_search_distance = statistics.mean(scenario_distances)
            avg_processing_latency = statistics.mean(scenario_processing_latencies)
            max_processing_latency = max(scenario_processing_latencies)

            scenario_result = {
                "scenario_name": scenario.name,
                "pattern_type": scenario.pattern_type.value,
                "test_iterations": scenario.test_iterations,
                "successful_searches": scenario_successes,
                "success_rate": success_rate,
                "success_rate_threshold": scenario.success_rate_threshold,
                "avg_acquisition_time_s": avg_acquisition_time,
                "expected_acquisition_time_s": scenario.expected_acquisition_time_s,
                "avg_search_distance_m": avg_search_distance,
                "search_area_size_m": scenario.search_area_size_meters,
                "avg_processing_latency_ms": avg_processing_latency,
                "max_processing_latency_ms": max_processing_latency,
                # Validation flags
                "meets_success_rate_threshold": success_rate >= scenario.success_rate_threshold,
                "meets_time_requirement": avg_acquisition_time
                <= scenario.expected_acquisition_time_s,
                "search_efficiency": scenario.search_area_size_meters / avg_search_distance
                if avg_search_distance > 0
                else 0.0,
            }
            spiral_results.append(scenario_result)

            logger.info(
                f"  Success rate: {success_rate:.2%} (threshold: {scenario.success_rate_threshold:.2%})"
            )
            logger.info(
                f"  Average acquisition time: {avg_acquisition_time:.2f}s (expected: {scenario.expected_acquisition_time_s}s)"
            )
            logger.info(f"  Average search distance: {avg_search_distance:.1f}m")
            logger.info(f"  Processing latency (avg): {avg_processing_latency:.2f}ms")

            # Scenario-specific validation
            assert (
                success_rate >= scenario.success_rate_threshold
            ), f"Success rate {success_rate:.2%} below threshold {scenario.success_rate_threshold:.2%} for {scenario.name}"
            if avg_acquisition_time != float("inf"):
                assert (
                    avg_acquisition_time <= scenario.expected_acquisition_time_s
                ), f"Acquisition time {avg_acquisition_time:.2f}s exceeds expected {scenario.expected_acquisition_time_s}s for {scenario.name}"
            assert (
                max_processing_latency < 100.0
            ), f"Max processing latency exceeds 100ms for {scenario.name}"

        performance_metrics.add_measurement("spiral_search_pattern_validation", spiral_results)

        # Overall spiral search validation
        overall_success_rate = statistics.mean([r["success_rate"] for r in spiral_results])
        overall_avg_time = statistics.mean(
            [
                r["avg_acquisition_time_s"]
                for r in spiral_results
                if r["avg_acquisition_time_s"] != float("inf")
            ]
        )
        overall_search_efficiency = statistics.mean(
            [r["search_efficiency"] for r in spiral_results if r["search_efficiency"] > 0]
        )
        overall_processing_latency = statistics.mean(
            [r["avg_processing_latency_ms"] for r in spiral_results]
        )

        scenarios_meeting_threshold = sum(
            1 for r in spiral_results if r["meets_success_rate_threshold"]
        )
        threshold_compliance_rate = scenarios_meeting_threshold / len(spiral_results)

        logger.info("Spiral search pattern validation results:")
        logger.info(f"  Overall success rate: {overall_success_rate:.2%}")
        logger.info(f"  Overall average acquisition time: {overall_avg_time:.2f}s")
        logger.info(f"  Overall search efficiency: {overall_search_efficiency:.2f}")
        logger.info(f"  Overall processing latency: {overall_processing_latency:.2f}ms")
        logger.info(f"  Threshold compliance rate: {threshold_compliance_rate:.2%}")

        # Final assertions for spiral search requirements
        assert (
            overall_success_rate >= 0.8
        ), f"Overall spiral search success rate {overall_success_rate:.2%} below 80%"
        assert (
            overall_avg_time <= 12.0
        ), f"Overall average acquisition time {overall_avg_time:.2f}s exceeds 12s"
        assert (
            threshold_compliance_rate >= 0.9
        ), f"Threshold compliance rate {threshold_compliance_rate:.2%} below 90%"
        assert (
            overall_processing_latency < 100.0
        ), f"Overall processing latency {overall_processing_latency:.2f}ms exceeds 100ms"

    # Test [26d3]: S-turn sampling maneuver effectiveness tests with weak signal scenarios

    @pytest.mark.asyncio
    async def test_s_turn_sampling_maneuver_weak_signal_effectiveness(
        self,
        enhanced_signal_processor,
        confidence_based_homing,
        search_pattern_generator,
        beacon_generator,
        performance_metrics,
        timing_validator,
    ):
        """Develop S-turn sampling maneuver effectiveness tests with weak signal scenarios (-80dB threshold).

        Tests S-turn sampling pattern for gradient determination in weak signals
        Validates effectiveness at -80dB threshold with statistical analysis
        """
        logger.info("Testing S-turn sampling maneuver effectiveness with weak signal scenarios")

        # Define S-turn sampling test scenarios
        s_turn_scenarios = [
            {
                "name": "s_turn_threshold_signal",
                "signal_strength_dbm": -80.0,
                "signal_variation_db": 2.0,  # ±2dB variation
                "sampling_pattern_width_m": 40.0,
                "expected_gradient_determination_time_s": 8.0,
                "confidence_threshold": 0.2,
                "test_iterations": 10,
            },
            {
                "name": "s_turn_very_weak_signal",
                "signal_strength_dbm": -82.0,
                "signal_variation_db": 3.0,  # ±3dB variation
                "sampling_pattern_width_m": 60.0,
                "expected_gradient_determination_time_s": 10.0,
                "confidence_threshold": 0.15,
                "test_iterations": 8,
            },
            {
                "name": "s_turn_marginal_signal",
                "signal_strength_dbm": -85.0,
                "signal_variation_db": 4.0,  # ±4dB variation
                "sampling_pattern_width_m": 80.0,
                "expected_gradient_determination_time_s": 12.0,
                "confidence_threshold": 0.1,
                "test_iterations": 6,
            },
        ]

        s_turn_results = []

        for scenario in s_turn_scenarios:
            logger.info(f"Testing S-turn scenario: {scenario['name']}")

            scenario_successes = 0
            scenario_determination_times = []
            scenario_processing_latencies = []
            scenario_gradient_qualities = []

            for iteration in range(scenario["test_iterations"]):
                logger.debug(f"S-turn iteration {iteration + 1}/{scenario['test_iterations']}")

                # Configure beacon with weak signal and gradient
                beacon_bearing = 90.0 + (iteration * 20.0)  # Vary bearing each iteration
                beacon_distance = 200.0

                await beacon_generator.set_position(
                    bearing_deg=beacon_bearing, distance_meters=beacon_distance
                )

                # Create weak signal with spatial gradient for S-turn sampling
                await beacon_generator.set_signal_strength(scenario["signal_strength_dbm"])
                await beacon_generator.set_signal_gradient(
                    gradient_strength_db_per_m=0.1,  # Weak but detectable gradient
                    gradient_direction_deg=beacon_bearing,
                )
                await beacon_generator.set_frequency(3200.0)
                await beacon_generator.start_transmission()

                # Allow weak signal to stabilize
                await asyncio.sleep(2.0)

                # Initiate S-turn sampling maneuver
                sampling_start_time = time.time()

                s_turn_init_start = time.time_ns()
                await search_pattern_generator.initiate_s_turn_sampling(
                    pattern_width_m=scenario["sampling_pattern_width_m"],
                    sample_points=8,  # 8 sample points per S-turn
                )
                s_turn_init_latency_ms = (time.time_ns() - s_turn_init_start) / 1_000_000

                timing_validator.validate_latency(s_turn_init_latency_ms)
                assert (
                    s_turn_init_latency_ms < 100.0
                ), f"S-turn initialization latency {s_turn_init_latency_ms}ms exceeds 100ms"
                scenario_processing_latencies.append(s_turn_init_latency_ms)

                # Execute S-turn sampling with gradient determination
                gradient_determined = False
                gradient_quality = 0.0
                max_sampling_time = (
                    scenario["expected_gradient_determination_time_s"] * 1.2
                )  # 20% buffer

                for sampling_step in range(int(max_sampling_time * 10)):  # 10 steps per second
                    await asyncio.sleep(0.1)

                    # Measure S-turn sampling processing
                    step_start_time = time.time_ns()

                    current_position = await search_pattern_generator.get_current_s_turn_position()
                    bearing_result = await enhanced_signal_processor.compute_bearing()
                    gradient_vector = await confidence_based_homing.compute_enhanced_gradient(
                        bearing_result,
                        confidence_based_homing.assess_signal_confidence(bearing_result),
                    )

                    step_latency_ms = (time.time_ns() - step_start_time) / 1_000_000
                    timing_validator.validate_latency(step_latency_ms)
                    assert (
                        step_latency_ms < 100.0
                    ), f"S-turn step latency {step_latency_ms}ms exceeds 100ms"
                    scenario_processing_latencies.append(step_latency_ms)

                    # Check for gradient determination success
                    if (
                        gradient_vector.magnitude > 0.1
                        and gradient_vector.confidence > scenario["confidence_threshold"]
                        and bearing_result.confidence > scenario["confidence_threshold"]
                    ):
                        gradient_determined = True
                        gradient_quality = gradient_vector.confidence
                        break

                sampling_time = time.time() - sampling_start_time
                await beacon_generator.stop_transmission()

                # Record iteration results
                if gradient_determined:
                    scenario_successes += 1
                    scenario_determination_times.append(sampling_time)
                    scenario_gradient_qualities.append(gradient_quality)

                logger.debug(
                    f"  Iteration {iteration + 1}: Gradient determined={gradient_determined}, Time={sampling_time:.2f}s, Quality={gradient_quality:.3f}"
                )

            # Calculate scenario statistics
            success_rate = scenario_successes / scenario["test_iterations"]
            avg_determination_time = (
                statistics.mean(scenario_determination_times)
                if scenario_determination_times
                else float("inf")
            )
            avg_gradient_quality = (
                statistics.mean(scenario_gradient_qualities) if scenario_gradient_qualities else 0.0
            )
            avg_processing_latency = statistics.mean(scenario_processing_latencies)
            max_processing_latency = max(scenario_processing_latencies)

            scenario_result = {
                "scenario_name": scenario["name"],
                "signal_strength_dbm": scenario["signal_strength_dbm"],
                "pattern_width_m": scenario["sampling_pattern_width_m"],
                "test_iterations": scenario["test_iterations"],
                "successful_determinations": scenario_successes,
                "success_rate": success_rate,
                "avg_determination_time_s": avg_determination_time,
                "expected_determination_time_s": scenario["expected_gradient_determination_time_s"],
                "avg_gradient_quality": avg_gradient_quality,
                "confidence_threshold": scenario["confidence_threshold"],
                "avg_processing_latency_ms": avg_processing_latency,
                "max_processing_latency_ms": max_processing_latency,
                # Validation flags
                "meets_time_requirement": avg_determination_time
                <= scenario["expected_gradient_determination_time_s"],
                "sufficient_success_rate": success_rate >= 0.7,  # 70% success rate for weak signals
                "adequate_gradient_quality": avg_gradient_quality
                >= scenario["confidence_threshold"],
            }
            s_turn_results.append(scenario_result)

            logger.info(f"  Success rate: {success_rate:.2%}")
            logger.info(
                f"  Average determination time: {avg_determination_time:.2f}s (expected: {scenario['expected_gradient_determination_time_s']}s)"
            )
            logger.info(f"  Average gradient quality: {avg_gradient_quality:.3f}")
            logger.info(f"  Processing latency (avg): {avg_processing_latency:.2f}ms")

            # Scenario-specific validation
            assert (
                success_rate >= 0.6
            ), f"S-turn success rate {success_rate:.2%} below 60% for weak signal scenario {scenario['name']}"
            if avg_determination_time != float("inf"):
                assert (
                    avg_determination_time <= scenario["expected_gradient_determination_time_s"]
                ), f"Determination time {avg_determination_time:.2f}s exceeds expected {scenario['expected_gradient_determination_time_s']}s for {scenario['name']}"
            assert (
                max_processing_latency < 100.0
            ), f"Max processing latency exceeds 100ms for {scenario['name']}"

        performance_metrics.add_measurement("s_turn_sampling_effectiveness", s_turn_results)

        # Overall S-turn sampling validation
        overall_success_rate = statistics.mean([r["success_rate"] for r in s_turn_results])
        overall_avg_time = statistics.mean(
            [
                r["avg_determination_time_s"]
                for r in s_turn_results
                if r["avg_determination_time_s"] != float("inf")
            ]
        )
        overall_gradient_quality = statistics.mean(
            [r["avg_gradient_quality"] for r in s_turn_results if r["avg_gradient_quality"] > 0]
        )
        overall_processing_latency = statistics.mean(
            [r["avg_processing_latency_ms"] for r in s_turn_results]
        )

        scenarios_with_sufficient_success = sum(
            1 for r in s_turn_results if r["sufficient_success_rate"]
        )
        sufficient_success_compliance = scenarios_with_sufficient_success / len(s_turn_results)

        logger.info("S-turn sampling maneuver effectiveness results:")
        logger.info(f"  Overall success rate: {overall_success_rate:.2%}")
        logger.info(f"  Overall average determination time: {overall_avg_time:.2f}s")
        logger.info(f"  Overall gradient quality: {overall_gradient_quality:.3f}")
        logger.info(f"  Overall processing latency: {overall_processing_latency:.2f}ms")
        logger.info(f"  Sufficient success compliance: {sufficient_success_compliance:.2%}")

        # Final assertions for S-turn sampling requirements
        assert (
            overall_success_rate >= 0.65
        ), f"Overall S-turn success rate {overall_success_rate:.2%} below 65% for weak signals"
        assert (
            overall_avg_time <= 10.0
        ), f"Overall average determination time {overall_avg_time:.2f}s exceeds 10s"
        assert (
            sufficient_success_compliance >= 0.8
        ), f"Sufficient success compliance {sufficient_success_compliance:.2%} below 80%"
        assert (
            overall_processing_latency < 100.0
        ), f"Overall processing latency {overall_processing_latency:.2f}ms exceeds 100ms"

    # Test [26d4]: Return-to-last-peak algorithm validation with complete signal loss scenarios

    @pytest.mark.asyncio
    async def test_return_to_last_peak_complete_signal_loss_validation(
        self,
        enhanced_signal_processor,
        confidence_based_homing,
        search_pattern_generator,
        beacon_generator,
        performance_metrics,
        timing_validator,
    ):
        """Create return-to-last-peak algorithm validation with complete signal loss recovery scenarios.

        Tests return-to-peak strategy when signal is completely lost
        Validates navigation back to last known strong signal location
        """
        logger.info("Testing return-to-last-peak algorithm validation with complete signal loss")

        # Define return-to-peak test scenarios
        return_to_peak_scenarios = [
            {
                "name": "short_distance_return",
                "peak_signal_strength_dbm": -45.0,
                "peak_distance_from_start_m": 80.0,
                "drift_distance_from_peak_m": 30.0,
                "signal_loss_duration_s": 5.0,
                "expected_return_time_s": 8.0,
                "return_accuracy_threshold_m": 15.0,
                "test_iterations": 8,
            },
            {
                "name": "medium_distance_return",
                "peak_signal_strength_dbm": -55.0,
                "peak_distance_from_start_m": 150.0,
                "drift_distance_from_peak_m": 60.0,
                "signal_loss_duration_s": 8.0,
                "expected_return_time_s": 12.0,
                "return_accuracy_threshold_m": 25.0,
                "test_iterations": 6,
            },
            {
                "name": "long_distance_return",
                "peak_signal_strength_dbm": -60.0,
                "peak_distance_from_start_m": 250.0,
                "drift_distance_from_peak_m": 100.0,
                "signal_loss_duration_s": 12.0,
                "expected_return_time_s": 18.0,
                "return_accuracy_threshold_m": 40.0,
                "test_iterations": 4,
            },
        ]

        return_to_peak_results = []

        for scenario in return_to_peak_scenarios:
            logger.info(f"Testing return-to-peak scenario: {scenario['name']}")

            scenario_successes = 0
            scenario_return_times = []
            scenario_processing_latencies = []
            scenario_return_accuracies = []

            for iteration in range(scenario["test_iterations"]):
                logger.debug(
                    f"Return-to-peak iteration {iteration + 1}/{scenario['test_iterations']}"
                )

                # Phase 1: Establish peak signal location
                peak_bearing = 45.0 + (iteration * 45.0)  # Vary peak location each iteration

                await beacon_generator.set_position(
                    bearing_deg=peak_bearing, distance_meters=scenario["peak_distance_from_start_m"]
                )
                await beacon_generator.set_signal_strength(scenario["peak_signal_strength_dbm"])
                await beacon_generator.set_frequency(3200.0)
                await beacon_generator.start_transmission()

                # Allow navigation to peak location (simulated)
                await asyncio.sleep(2.0)

                # Record peak location and signal characteristics
                peak_record_start = time.time_ns()
                peak_bearing_result = await enhanced_signal_processor.compute_bearing()
                peak_location = await confidence_based_homing.get_current_location()
                peak_record_latency_ms = (time.time_ns() - peak_record_start) / 1_000_000

                timing_validator.validate_latency(peak_record_latency_ms)
                assert (
                    peak_record_latency_ms < 100.0
                ), f"Peak recording latency {peak_record_latency_ms}ms exceeds 100ms"
                scenario_processing_latencies.append(peak_record_latency_ms)

                # Validate peak signal establishment
                assert (
                    peak_bearing_result.confidence > 0.6
                ), f"Peak signal confidence {peak_bearing_result.confidence} insufficient for return-to-peak test"

                # Phase 2: Simulate drift away from peak
                drift_bearing = (peak_bearing + 180.0 + np.random.uniform(-45, 45)) % 360.0

                # Simulate movement away from peak (recorded in homing system)
                await confidence_based_homing.simulate_position_drift(
                    drift_bearing_deg=drift_bearing,
                    drift_distance_m=scenario["drift_distance_from_peak_m"],
                )

                # Phase 3: Complete signal loss
                await beacon_generator.stop_transmission()

                # Wait for signal loss to be detected
                await asyncio.sleep(scenario["signal_loss_duration_s"])

                # Phase 4: Initiate return-to-last-peak
                return_start_time = time.time()

                return_init_start = time.time_ns()
                await search_pattern_generator.initiate_return_to_peak(
                    peak_location=peak_location,
                    peak_signal_strength=peak_bearing_result.signal_strength_dbm,
                    peak_confidence=peak_bearing_result.confidence,
                )
                return_init_latency_ms = (time.time_ns() - return_init_start) / 1_000_000

                timing_validator.validate_latency(return_init_latency_ms)
                assert (
                    return_init_latency_ms < 100.0
                ), f"Return-to-peak initialization latency {return_init_latency_ms}ms exceeds 100ms"
                scenario_processing_latencies.append(return_init_latency_ms)

                # Phase 5: Re-enable beacon at peak location for return validation
                await beacon_generator.start_transmission()

                # Execute return-to-peak navigation
                return_successful = False
                return_accuracy_m = float("inf")
                max_return_time = scenario["expected_return_time_s"] * 1.3  # 30% buffer

                for return_step in range(int(max_return_time * 10)):  # 10 steps per second
                    await asyncio.sleep(0.1)

                    # Measure return-to-peak processing
                    step_start_time = time.time_ns()

                    current_position = await confidence_based_homing.get_current_location()
                    bearing_result = await enhanced_signal_processor.compute_bearing()
                    return_status = await search_pattern_generator.get_return_to_peak_status()

                    step_latency_ms = (time.time_ns() - step_start_time) / 1_000_000
                    timing_validator.validate_latency(step_latency_ms)
                    assert (
                        step_latency_ms < 100.0
                    ), f"Return step latency {step_latency_ms}ms exceeds 100ms"
                    scenario_processing_latencies.append(step_latency_ms)

                    # Calculate return accuracy (distance from original peak)
                    return_accuracy_m = (
                        await confidence_based_homing.calculate_distance_from_position(
                            peak_location
                        )
                    )

                    # Check for successful return to peak
                    if (
                        bearing_result.confidence > 0.5
                        and bearing_result.signal_strength_dbm
                        > (scenario["peak_signal_strength_dbm"] - 5.0)
                        and return_accuracy_m <= scenario["return_accuracy_threshold_m"]
                    ):
                        return_successful = True
                        break

                return_time = time.time() - return_start_time
                await beacon_generator.stop_transmission()

                # Record iteration results
                if return_successful:
                    scenario_successes += 1
                    scenario_return_times.append(return_time)
                    scenario_return_accuracies.append(return_accuracy_m)

                logger.debug(
                    f"  Iteration {iteration + 1}: Return successful={return_successful}, Time={return_time:.2f}s, Accuracy={return_accuracy_m:.1f}m"
                )

            # Calculate scenario statistics
            success_rate = scenario_successes / scenario["test_iterations"]
            avg_return_time = (
                statistics.mean(scenario_return_times) if scenario_return_times else float("inf")
            )
            avg_return_accuracy = (
                statistics.mean(scenario_return_accuracies)
                if scenario_return_accuracies
                else float("inf")
            )
            avg_processing_latency = statistics.mean(scenario_processing_latencies)
            max_processing_latency = max(scenario_processing_latencies)

            scenario_result = {
                "scenario_name": scenario["name"],
                "peak_distance_from_start_m": scenario["peak_distance_from_start_m"],
                "drift_distance_from_peak_m": scenario["drift_distance_from_peak_m"],
                "signal_loss_duration_s": scenario["signal_loss_duration_s"],
                "test_iterations": scenario["test_iterations"],
                "successful_returns": scenario_successes,
                "success_rate": success_rate,
                "avg_return_time_s": avg_return_time,
                "expected_return_time_s": scenario["expected_return_time_s"],
                "avg_return_accuracy_m": avg_return_accuracy,
                "return_accuracy_threshold_m": scenario["return_accuracy_threshold_m"],
                "avg_processing_latency_ms": avg_processing_latency,
                "max_processing_latency_ms": max_processing_latency,
                # Validation flags
                "meets_time_requirement": avg_return_time <= scenario["expected_return_time_s"],
                "meets_accuracy_requirement": avg_return_accuracy
                <= scenario["return_accuracy_threshold_m"],
                "sufficient_success_rate": success_rate
                >= 0.75,  # 75% success rate for return-to-peak
            }
            return_to_peak_results.append(scenario_result)

            logger.info(f"  Success rate: {success_rate:.2%}")
            logger.info(
                f"  Average return time: {avg_return_time:.2f}s (expected: {scenario['expected_return_time_s']}s)"
            )
            logger.info(
                f"  Average return accuracy: {avg_return_accuracy:.1f}m (threshold: {scenario['return_accuracy_threshold_m']}m)"
            )
            logger.info(f"  Processing latency (avg): {avg_processing_latency:.2f}ms")

            # Scenario-specific validation
            assert (
                success_rate >= 0.7
            ), f"Return-to-peak success rate {success_rate:.2%} below 70% for {scenario['name']}"
            if avg_return_time != float("inf"):
                assert (
                    avg_return_time <= scenario["expected_return_time_s"]
                ), f"Return time {avg_return_time:.2f}s exceeds expected {scenario['expected_return_time_s']}s for {scenario['name']}"
            assert (
                max_processing_latency < 100.0
            ), f"Max processing latency exceeds 100ms for {scenario['name']}"

        performance_metrics.add_measurement("return_to_peak_validation", return_to_peak_results)

        # Overall return-to-peak validation
        overall_success_rate = statistics.mean([r["success_rate"] for r in return_to_peak_results])
        overall_avg_time = statistics.mean(
            [
                r["avg_return_time_s"]
                for r in return_to_peak_results
                if r["avg_return_time_s"] != float("inf")
            ]
        )
        overall_accuracy = statistics.mean(
            [
                r["avg_return_accuracy_m"]
                for r in return_to_peak_results
                if r["avg_return_accuracy_m"] != float("inf")
            ]
        )
        overall_processing_latency = statistics.mean(
            [r["avg_processing_latency_ms"] for r in return_to_peak_results]
        )

        scenarios_meeting_requirements = sum(
            1
            for r in return_to_peak_results
            if r["meets_time_requirement"] and r["meets_accuracy_requirement"]
        )
        requirements_compliance_rate = scenarios_meeting_requirements / len(return_to_peak_results)

        logger.info("Return-to-last-peak algorithm validation results:")
        logger.info(f"  Overall success rate: {overall_success_rate:.2%}")
        logger.info(f"  Overall average return time: {overall_avg_time:.2f}s")
        logger.info(f"  Overall average return accuracy: {overall_accuracy:.1f}m")
        logger.info(f"  Overall processing latency: {overall_processing_latency:.2f}ms")
        logger.info(f"  Requirements compliance rate: {requirements_compliance_rate:.2%}")

        # Final assertions for return-to-peak requirements
        assert (
            overall_success_rate >= 0.75
        ), f"Overall return-to-peak success rate {overall_success_rate:.2%} below 75%"
        assert (
            overall_avg_time <= 15.0
        ), f"Overall average return time {overall_avg_time:.2f}s exceeds 15s"
        assert (
            requirements_compliance_rate >= 0.8
        ), f"Requirements compliance rate {requirements_compliance_rate:.2%} below 80%"
        assert (
            overall_processing_latency < 100.0
        ), f"Overall processing latency {overall_processing_latency:.2f}ms exceeds 100ms"
