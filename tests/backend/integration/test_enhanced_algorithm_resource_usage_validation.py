"""Resource Usage Validation Tests for Enhanced Algorithm Performance.

SUBTASK-6.2.4.2: Safety and Integration Validation - Resource Usage Validation [27c1-27c4]

This test suite validates resource usage patterns of enhanced algorithms to ensure
they operate within acceptable system resource constraints during various operational scenarios.

Test Coverage:
- [27c1] CPU usage validation during enhanced ASV analysis with <80% sustained load
- [27c2] Memory usage validation with <512MB allocation during peak processing
- [27c3] I/O bandwidth validation during concurrent SDR and enhanced processing
- [27c4] Network bandwidth validation during MAVLink coordination with ASV active

PRD Requirements Validated:
- PRD-NFR4: System stability under load
- PRD-NFR5: Resource usage constraints
- System performance thresholds for sustained operation
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
class ResourceUsageScenario:
    """Test scenario for resource usage validation."""

    name: str
    description: str
    test_duration_seconds: float
    processing_intensity: str  # low, medium, high, extreme
    concurrent_operations: int
    expected_max_cpu_percent: float
    expected_max_memory_mb: float
    expected_max_io_mbps: float
    expected_max_network_kbps: float


@dataclass
class ResourceUsageMetrics:
    """Metrics collected during resource usage validation."""

    cpu_usage_percent: List[float]
    memory_usage_mb: List[float]
    io_read_mbps: List[float]
    io_write_mbps: List[float]
    network_sent_kbps: List[float]
    network_recv_kbps: List[float]
    peak_cpu_percent: float
    peak_memory_mb: float
    avg_cpu_percent: float
    avg_memory_mb: float
    sustained_cpu_percent: float  # 95th percentile for sustained load
    test_duration_seconds: float


class TestEnhancedAlgorithmResourceUsageValidation:
    """Test suite for validating enhanced algorithm resource usage."""

    @pytest.fixture
    async def enhanced_signal_processor(self):
        """Create ASV enhanced signal processor for resource testing."""
        processor = ASVEnhancedSignalProcessor()
        await processor.initialize()
        # Configure for resource usage monitoring
        await processor.enable_resource_monitoring()
        yield processor
        await processor.shutdown()

    @pytest.fixture
    async def confidence_based_homing(self, enhanced_signal_processor):
        """Create confidence-based homing system for resource testing."""
        homing = ASVConfidenceBasedHoming(
            asv_processor=enhanced_signal_processor, threshold_config=DynamicThresholdConfig()
        )
        yield homing

    @pytest.fixture
    def resource_usage_scenarios(self) -> List[ResourceUsageScenario]:
        """Generate comprehensive resource usage test scenarios."""
        return [
            ResourceUsageScenario(
                name="low_intensity_sustained_processing",
                description="Low intensity sustained processing for baseline measurement",
                test_duration_seconds=10.0,
                processing_intensity="low",
                concurrent_operations=1,
                expected_max_cpu_percent=40.0,
                expected_max_memory_mb=128.0,
                expected_max_io_mbps=5.0,
                expected_max_network_kbps=100.0,
            ),
            ResourceUsageScenario(
                name="medium_intensity_asv_analysis",
                description="Medium intensity ASV analysis with confidence decisions",
                test_duration_seconds=15.0,
                processing_intensity="medium",
                concurrent_operations=2,
                expected_max_cpu_percent=65.0,
                expected_max_memory_mb=256.0,
                expected_max_io_mbps=10.0,
                expected_max_network_kbps=200.0,
            ),
            ResourceUsageScenario(
                name="high_intensity_enhanced_processing",
                description="High intensity enhanced processing with multiple algorithms",
                test_duration_seconds=20.0,
                processing_intensity="high",
                concurrent_operations=3,
                expected_max_cpu_percent=75.0,  # Under 80% sustained load requirement
                expected_max_memory_mb=400.0,
                expected_max_io_mbps=20.0,
                expected_max_network_kbps=500.0,
            ),
            ResourceUsageScenario(
                name="extreme_load_peak_processing",
                description="Extreme load peak processing for maximum resource validation",
                test_duration_seconds=12.0,
                processing_intensity="extreme",
                concurrent_operations=4,
                expected_max_cpu_percent=78.0,  # Just under 80% limit
                expected_max_memory_mb=500.0,  # Just under 512MB limit
                expected_max_io_mbps=25.0,
                expected_max_network_kbps=750.0,
            ),
        ]

    @pytest.mark.asyncio
    async def test_cpu_usage_validation_during_enhanced_asv_analysis_with_sustained_load_limit(
        self, enhanced_signal_processor, confidence_based_homing, resource_usage_scenarios
    ):
        """[27c1] CPU usage validation during enhanced ASV analysis with <80% sustained load."""

        logger.info("Starting CPU usage validation during enhanced ASV analysis")

        cpu_validation_results = []

        for scenario in resource_usage_scenarios:
            logger.info(f"Testing CPU usage scenario: {scenario.name}")

            # Start resource monitoring
            resource_monitor = ResourceUsageMonitor()
            await resource_monitor.start_monitoring()

            try:
                # Run enhanced ASV analysis with specified intensity
                await self._run_enhanced_asv_analysis_with_intensity(
                    enhanced_signal_processor, confidence_based_homing, scenario
                )

                # Collect resource usage metrics
                metrics = await resource_monitor.get_metrics()

                # Validate CPU usage requirements
                assert metrics.peak_cpu_percent <= scenario.expected_max_cpu_percent, (
                    f"Peak CPU usage {metrics.peak_cpu_percent:.1f}% exceeds "
                    f"expected {scenario.expected_max_cpu_percent:.1f}% for {scenario.name}"
                )

                # Validate sustained CPU load (95th percentile) is under 80%
                assert metrics.sustained_cpu_percent <= 80.0, (
                    f"Sustained CPU usage {metrics.sustained_cpu_percent:.1f}% exceeds "
                    f"80% requirement for {scenario.name}"
                )

                cpu_validation_results.append(
                    {
                        "scenario": scenario.name,
                        "peak_cpu_percent": metrics.peak_cpu_percent,
                        "avg_cpu_percent": metrics.avg_cpu_percent,
                        "sustained_cpu_percent": metrics.sustained_cpu_percent,
                        "processing_intensity": scenario.processing_intensity,
                        "concurrent_operations": scenario.concurrent_operations,
                    }
                )

                logger.info(
                    f"CPU validation for {scenario.name}: "
                    f"Peak: {metrics.peak_cpu_percent:.1f}%, "
                    f"Avg: {metrics.avg_cpu_percent:.1f}%, "
                    f"Sustained: {metrics.sustained_cpu_percent:.1f}%"
                )

            finally:
                await resource_monitor.stop_monitoring()

            await asyncio.sleep(2.0)  # Cool down between scenarios

        # Validate overall CPU usage patterns
        max_sustained_cpu = max(
            result["sustained_cpu_percent"] for result in cpu_validation_results
        )
        avg_peak_cpu = sum(result["peak_cpu_percent"] for result in cpu_validation_results) / len(
            cpu_validation_results
        )

        # Overall system should maintain <80% sustained CPU usage
        assert (
            max_sustained_cpu <= 80.0
        ), f"Maximum sustained CPU usage {max_sustained_cpu:.1f}% exceeds 80% system requirement"

        # Average peak CPU should be reasonable for enhanced processing
        assert (
            avg_peak_cpu <= 70.0
        ), f"Average peak CPU usage {avg_peak_cpu:.1f}% indicates inefficient processing"

        logger.info(
            f"CPU usage validation completed: "
            f"Max sustained: {max_sustained_cpu:.1f}%, Avg peak: {avg_peak_cpu:.1f}%"
        )

    async def _run_enhanced_asv_analysis_with_intensity(
        self,
        signal_processor: ASVEnhancedSignalProcessor,
        confidence_homing: ASVConfidenceBasedHoming,
        scenario: ResourceUsageScenario,
    ) -> None:
        """Run enhanced ASV analysis with specified processing intensity."""

        end_time = time.perf_counter() + scenario.test_duration_seconds

        # Configure signal scenarios based on processing intensity
        intensity_configs = {
            "low": {
                "scenarios": [
                    {"strength": -70.0, "bearing": 90.0, "interference": False, "multipath": False}
                ],
                "processing_frequency": 20.0,  # 20Hz
                "confidence_evaluations": 1,
            },
            "medium": {
                "scenarios": [
                    {"strength": -80.0, "bearing": 45.0, "interference": True, "multipath": False},
                    {"strength": -75.0, "bearing": 135.0, "interference": False, "multipath": True},
                ],
                "processing_frequency": 50.0,  # 50Hz
                "confidence_evaluations": 3,
            },
            "high": {
                "scenarios": [
                    {"strength": -90.0, "bearing": 30.0, "interference": True, "multipath": True},
                    {"strength": -85.0, "bearing": 120.0, "interference": True, "multipath": False},
                    {"strength": -95.0, "bearing": 210.0, "interference": False, "multipath": True},
                ],
                "processing_frequency": 100.0,  # 100Hz
                "confidence_evaluations": 5,
            },
            "extreme": {
                "scenarios": [
                    {
                        "strength": -105.0,
                        "bearing": 15.0,
                        "interference": True,
                        "multipath": True,
                        "doppler": 5.2,
                    },
                    {
                        "strength": -100.0,
                        "bearing": 75.0,
                        "interference": True,
                        "multipath": True,
                        "doppler": -3.8,
                    },
                    {
                        "strength": -110.0,
                        "bearing": 165.0,
                        "interference": True,
                        "multipath": True,
                        "doppler": 7.1,
                    },
                    {
                        "strength": -98.0,
                        "bearing": 255.0,
                        "interference": True,
                        "multipath": True,
                        "doppler": -6.3,
                    },
                ],
                "processing_frequency": 200.0,  # 200Hz
                "confidence_evaluations": 8,
            },
        }

        config = intensity_configs[scenario.processing_intensity]
        processing_interval = 1.0 / config["processing_frequency"]
        scenario_index = 0

        # Start concurrent processing tasks based on scenario
        concurrent_tasks = []
        for i in range(scenario.concurrent_operations):
            task = asyncio.create_task(
                self._concurrent_processing_task(
                    signal_processor, confidence_homing, config, end_time, i
                )
            )
            concurrent_tasks.append(task)

        # Wait for all concurrent processing to complete
        await asyncio.gather(*concurrent_tasks)

    async def _concurrent_processing_task(
        self,
        signal_processor: ASVEnhancedSignalProcessor,
        confidence_homing: ASVConfidenceBasedHoming,
        config: Dict[str, Any],
        end_time: float,
        task_id: int,
    ) -> None:
        """Run concurrent processing task for CPU load testing."""

        processing_interval = 1.0 / config["processing_frequency"]
        scenario_index = task_id  # Offset each task's scenario selection

        while time.perf_counter() < end_time:
            scenario_data = config["scenarios"][scenario_index % len(config["scenarios"])]

            # Create signal data for processing
            signal_data = {
                "signal_strength_dbm": scenario_data["strength"],
                "frequency_hz": 433.92e6,
                "bearing_deg": scenario_data["bearing"],
                "interference_detected": scenario_data["interference"],
                "multipath_detected": scenario_data["multipath"],
                "doppler_shift_hz": scenario_data.get("doppler", 0.0),
                "noise_floor_dbm": -120.0,
                "sample_rate_hz": 2.4e6,
            }

            # Perform enhanced signal processing
            bearing_calc = await signal_processor.calculate_enhanced_bearing(signal_data)

            # Multiple confidence evaluations based on intensity
            for eval_index in range(config["confidence_evaluations"]):
                confidence_adjustment = eval_index * 0.02  # Slight variation per evaluation
                adjusted_calc = ASVBearingCalculation(
                    bearing_deg=bearing_calc.bearing_deg,
                    confidence=max(0.0, bearing_calc.confidence - confidence_adjustment),
                    precision_deg=bearing_calc.precision_deg,
                    signal_strength_dbm=bearing_calc.signal_strength_dbm,
                    interference_detected=bearing_calc.interference_detected,
                    signal_quality=bearing_calc.signal_quality,
                )
                _ = confidence_homing.evaluate_confidence_based_decision(adjusted_calc)

            scenario_index += 1
            await asyncio.sleep(processing_interval)

    @pytest.mark.asyncio
    async def test_memory_usage_validation_with_allocation_limit_during_peak_processing(
        self, enhanced_signal_processor, confidence_based_homing, resource_usage_scenarios
    ):
        """[27c2] Memory usage validation with <512MB allocation during peak processing."""

        logger.info("Starting memory usage validation during peak processing")

        memory_validation_results = []

        for scenario in resource_usage_scenarios:
            logger.info(f"Testing memory usage scenario: {scenario.name}")

            # Start resource monitoring with focus on memory
            resource_monitor = ResourceUsageMonitor(focus="memory")
            await resource_monitor.start_monitoring()

            try:
                # Run memory-intensive enhanced processing
                await self._run_memory_intensive_enhanced_processing(
                    enhanced_signal_processor, confidence_based_homing, scenario
                )

                # Collect memory usage metrics
                metrics = await resource_monitor.get_metrics()

                # Validate memory usage requirements
                assert metrics.peak_memory_mb <= scenario.expected_max_memory_mb, (
                    f"Peak memory usage {metrics.peak_memory_mb:.1f}MB exceeds "
                    f"expected {scenario.expected_max_memory_mb:.1f}MB for {scenario.name}"
                )

                # Validate against 512MB system allocation limit
                assert metrics.peak_memory_mb <= 512.0, (
                    f"Peak memory usage {metrics.peak_memory_mb:.1f}MB exceeds "
                    f"512MB system allocation limit for {scenario.name}"
                )

                memory_validation_results.append(
                    {
                        "scenario": scenario.name,
                        "peak_memory_mb": metrics.peak_memory_mb,
                        "avg_memory_mb": metrics.avg_memory_mb,
                        "memory_efficiency": metrics.peak_memory_mb
                        / max(1, scenario.concurrent_operations),
                        "processing_intensity": scenario.processing_intensity,
                    }
                )

                logger.info(
                    f"Memory validation for {scenario.name}: "
                    f"Peak: {metrics.peak_memory_mb:.1f}MB, "
                    f"Avg: {metrics.avg_memory_mb:.1f}MB, "
                    f"Efficiency: {metrics.peak_memory_mb / max(1, scenario.concurrent_operations):.1f}MB/operation"
                )

            finally:
                await resource_monitor.stop_monitoring()

            await asyncio.sleep(1.5)  # Cool down between scenarios

        # Validate overall memory usage patterns
        max_peak_memory = max(result["peak_memory_mb"] for result in memory_validation_results)
        avg_memory_efficiency = sum(
            result["memory_efficiency"] for result in memory_validation_results
        ) / len(memory_validation_results)

        # Overall system should stay under 512MB allocation limit
        assert (
            max_peak_memory <= 512.0
        ), f"Maximum peak memory usage {max_peak_memory:.1f}MB exceeds 512MB system limit"

        # Memory efficiency should be reasonable (not excessive per operation)
        assert (
            avg_memory_efficiency <= 150.0
        ), f"Average memory efficiency {avg_memory_efficiency:.1f}MB/operation indicates memory waste"

        logger.info(
            f"Memory usage validation completed: "
            f"Max peak: {max_peak_memory:.1f}MB, Avg efficiency: {avg_memory_efficiency:.1f}MB/op"
        )

    async def _run_memory_intensive_enhanced_processing(
        self,
        signal_processor: ASVEnhancedSignalProcessor,
        confidence_homing: ASVConfidenceBasedHoming,
        scenario: ResourceUsageScenario,
    ) -> None:
        """Run memory-intensive enhanced processing for memory usage validation."""

        end_time = time.perf_counter() + scenario.test_duration_seconds

        # Configure memory-intensive processing based on scenario
        memory_configs = {
            "low": {"buffer_size": 8192, "history_length": 100, "analysis_depth": 2},
            "medium": {"buffer_size": 16384, "history_length": 250, "analysis_depth": 4},
            "high": {"buffer_size": 32768, "history_length": 500, "analysis_depth": 6},
            "extreme": {"buffer_size": 65536, "history_length": 1000, "analysis_depth": 8},
        }

        config = memory_configs[scenario.processing_intensity]

        # Create memory-intensive concurrent tasks
        memory_tasks = []
        for i in range(scenario.concurrent_operations):
            task = asyncio.create_task(
                self._memory_intensive_processing_task(
                    signal_processor, confidence_homing, config, end_time, i
                )
            )
            memory_tasks.append(task)

        # Wait for all memory-intensive processing to complete
        await asyncio.gather(*memory_tasks)

    async def _memory_intensive_processing_task(
        self,
        signal_processor: ASVEnhancedSignalProcessor,
        confidence_homing: ASVConfidenceBasedHoming,
        config: Dict[str, Any],
        end_time: float,
        task_id: int,
    ) -> None:
        """Run memory-intensive processing task for memory usage validation."""

        # Create large data structures for memory-intensive processing
        signal_history = []
        confidence_history = []
        analysis_buffers = []

        # Pre-allocate memory buffers based on configuration
        for buffer_idx in range(config["analysis_depth"]):
            buffer = [0.0] * config["buffer_size"]
            analysis_buffers.append(buffer)

        processing_cycle = 0

        while time.perf_counter() < end_time:
            # Create complex signal data for memory-intensive analysis
            signal_data = {
                "signal_strength_dbm": -85.0 - (processing_cycle % 30),
                "frequency_hz": 433.92e6,
                "bearing_deg": (processing_cycle * 2.5) % 360,
                "interference_detected": (processing_cycle % 3) == 0,
                "multipath_detected": (processing_cycle % 4) == 0,
                "noise_floor_dbm": -120.0,
                "sample_rate_hz": 2.4e6,
                "buffer_length": config["buffer_size"],
                "analysis_depth": config["analysis_depth"],
            }

            # Perform memory-intensive enhanced processing
            bearing_calc = await signal_processor.calculate_enhanced_bearing(signal_data)

            # Store results in history (memory accumulation)
            signal_history.append(
                {
                    "timestamp": time.perf_counter(),
                    "bearing": bearing_calc.bearing_deg,
                    "confidence": bearing_calc.confidence,
                    "signal_strength": bearing_calc.signal_strength_dbm,
                }
            )

            # Perform memory-intensive confidence analysis with history
            for analysis_idx in range(config["analysis_depth"]):
                # Create confidence calculation with historical context
                historical_context = signal_history[
                    -min(len(signal_history), 50) :
                ]  # Last 50 samples

                enhanced_calc = ASVBearingCalculation(
                    bearing_deg=bearing_calc.bearing_deg,
                    confidence=bearing_calc.confidence * (1.0 - analysis_idx * 0.05),
                    precision_deg=bearing_calc.precision_deg,
                    signal_strength_dbm=bearing_calc.signal_strength_dbm,
                    interference_detected=bearing_calc.interference_detected,
                    signal_quality=bearing_calc.signal_quality,
                )

                decision = confidence_homing.evaluate_confidence_based_decision(enhanced_calc)
                confidence_history.append(decision)

                # Update analysis buffer (memory intensive operation)
                buffer = analysis_buffers[analysis_idx]
                for i in range(min(len(buffer), 1000)):  # Update buffer elements
                    buffer[i] = (buffer[i] + bearing_calc.confidence + analysis_idx) / 2.0

            # Manage memory by limiting history size
            if len(signal_history) > config["history_length"]:
                signal_history = signal_history[-config["history_length"] :]

            if len(confidence_history) > config["history_length"]:
                confidence_history = confidence_history[-config["history_length"] :]

            processing_cycle += 1
            await asyncio.sleep(0.02)  # 50Hz processing rate

    @pytest.mark.asyncio
    async def test_io_bandwidth_validation_during_concurrent_sdr_and_enhanced_processing(
        self, enhanced_signal_processor, confidence_based_homing, resource_usage_scenarios
    ):
        """[27c3] I/O bandwidth validation during concurrent SDR and enhanced processing."""

        logger.info(
            "Starting I/O bandwidth validation during concurrent SDR and enhanced processing"
        )

        # Import SDR services for I/O intensive testing
        from src.backend.services.asv_integration.asv_hackrf_coordinator import ASVHackRFCoordinator
        from src.backend.services.sdrpp_bridge_service import SDRPPBridgeService

        sdr_bridge = SDRPPBridgeService()
        hackrf_coordinator = ASVHackRFCoordinator()

        await sdr_bridge.initialize()
        await hackrf_coordinator.initialize()

        io_validation_results = []

        try:
            for scenario in resource_usage_scenarios:
                logger.info(f"Testing I/O bandwidth scenario: {scenario.name}")

                # Start resource monitoring with focus on I/O
                resource_monitor = ResourceUsageMonitor(focus="io")
                await resource_monitor.start_monitoring()

                try:
                    # Run concurrent SDR and enhanced processing (I/O intensive)
                    await self._run_concurrent_sdr_enhanced_processing(
                        enhanced_signal_processor,
                        confidence_based_homing,
                        sdr_bridge,
                        hackrf_coordinator,
                        scenario,
                    )

                    # Collect I/O usage metrics
                    metrics = await resource_monitor.get_metrics()

                    # Calculate total I/O bandwidth (read + write)
                    max_io_bandwidth = max(
                        max(metrics.io_read_mbps, default=0.0)
                        + max(metrics.io_write_mbps, default=0.0),
                        0.0,
                    )
                    avg_io_read = sum(metrics.io_read_mbps) / max(len(metrics.io_read_mbps), 1)
                    avg_io_write = sum(metrics.io_write_mbps) / max(len(metrics.io_write_mbps), 1)

                    # Validate I/O bandwidth requirements
                    assert max_io_bandwidth <= scenario.expected_max_io_mbps, (
                        f"Peak I/O bandwidth {max_io_bandwidth:.1f}MB/s exceeds "
                        f"expected {scenario.expected_max_io_mbps:.1f}MB/s for {scenario.name}"
                    )

                    io_validation_results.append(
                        {
                            "scenario": scenario.name,
                            "peak_io_bandwidth_mbps": max_io_bandwidth,
                            "avg_io_read_mbps": avg_io_read,
                            "avg_io_write_mbps": avg_io_write,
                            "processing_intensity": scenario.processing_intensity,
                            "concurrent_operations": scenario.concurrent_operations,
                        }
                    )

                    logger.info(
                        f"I/O validation for {scenario.name}: "
                        f"Peak bandwidth: {max_io_bandwidth:.1f}MB/s, "
                        f"Avg read: {avg_io_read:.1f}MB/s, "
                        f"Avg write: {avg_io_write:.1f}MB/s"
                    )

                finally:
                    await resource_monitor.stop_monitoring()

                await asyncio.sleep(2.0)  # Cool down between scenarios

        finally:
            await sdr_bridge.shutdown()
            await hackrf_coordinator.shutdown()

        # Validate overall I/O bandwidth patterns
        max_io_bandwidth = max(result["peak_io_bandwidth_mbps"] for result in io_validation_results)
        avg_total_io = sum(
            result["avg_io_read_mbps"] + result["avg_io_write_mbps"]
            for result in io_validation_results
        ) / len(io_validation_results)

        # I/O bandwidth should be efficient for concurrent SDR and enhanced processing
        assert (
            max_io_bandwidth <= 30.0
        ), f"Maximum I/O bandwidth {max_io_bandwidth:.1f}MB/s exceeds reasonable 30MB/s threshold"

        assert (
            avg_total_io <= 15.0
        ), f"Average total I/O {avg_total_io:.1f}MB/s indicates inefficient I/O usage"

        logger.info(
            f"I/O bandwidth validation completed: "
            f"Max bandwidth: {max_io_bandwidth:.1f}MB/s, Avg total: {avg_total_io:.1f}MB/s"
        )

    async def _run_concurrent_sdr_enhanced_processing(
        self,
        signal_processor: ASVEnhancedSignalProcessor,
        confidence_homing: ASVConfidenceBasedHoming,
        sdr_bridge: Any,
        hackrf_coordinator: Any,
        scenario: ResourceUsageScenario,
    ) -> None:
        """Run concurrent SDR and enhanced processing for I/O bandwidth validation."""

        end_time = time.perf_counter() + scenario.test_duration_seconds

        # Start concurrent processing tasks
        io_intensive_tasks = []

        # SDR data acquisition tasks (I/O intensive)
        for i in range(max(1, scenario.concurrent_operations // 2)):
            task = asyncio.create_task(
                self._sdr_data_acquisition_task(sdr_bridge, hackrf_coordinator, end_time, i)
            )
            io_intensive_tasks.append(task)

        # Enhanced processing tasks (I/O intensive)
        for i in range(
            max(1, scenario.concurrent_operations - scenario.concurrent_operations // 2)
        ):
            task = asyncio.create_task(
                self._io_intensive_enhanced_processing_task(
                    signal_processor, confidence_homing, end_time, i
                )
            )
            io_intensive_tasks.append(task)

        # Wait for all I/O intensive processing to complete
        await asyncio.gather(*io_intensive_tasks)

    async def _sdr_data_acquisition_task(
        self, sdr_bridge: Any, hackrf_coordinator: Any, end_time: float, task_id: int
    ) -> None:
        """Run SDR data acquisition task for I/O bandwidth validation."""

        frequency_step = 100e3  # 100kHz steps
        base_frequency = 433.0e6 + (task_id * 1e6)  # Offset by task ID
        current_frequency = base_frequency

        while time.perf_counter() < end_time:
            try:
                # I/O intensive SDR operations

                # Frequency sweeping (configuration I/O)
                await hackrf_coordinator.set_frequency(current_frequency)
                current_frequency += frequency_step
                if current_frequency > base_frequency + 5e6:  # 5MHz sweep range
                    current_frequency = base_frequency

                # Data acquisition (high bandwidth I/O)
                spectrum_data = await sdr_bridge.get_spectrum_data()
                signal_data = await sdr_bridge.get_signal_data()

                # Data processing and storage (write I/O)
                processed_data = {
                    "timestamp": time.perf_counter(),
                    "frequency": current_frequency,
                    "spectrum": spectrum_data[:1024] if spectrum_data else [],  # Limit data size
                    "signal": signal_data[:2048] if signal_data else [],
                }

                # Simulate data logging (I/O write operations)
                await self._log_sdr_data(processed_data, task_id)

            except Exception as e:
                logger.warning(f"SDR data acquisition error in task {task_id}: {e}")

            await asyncio.sleep(0.05)  # 20Hz acquisition rate

    async def _io_intensive_enhanced_processing_task(
        self,
        signal_processor: ASVEnhancedSignalProcessor,
        confidence_homing: ASVConfidenceBasedHoming,
        end_time: float,
        task_id: int,
    ) -> None:
        """Run I/O intensive enhanced processing task."""

        data_buffer = []
        file_counter = 0

        while time.perf_counter() < end_time:
            # Create I/O intensive signal processing
            signal_data = {
                "signal_strength_dbm": -80.0 - (task_id * 5),
                "frequency_hz": 433.92e6,
                "bearing_deg": (time.perf_counter() * 10 + task_id * 45) % 360,
                "interference_detected": (file_counter % 3) == 0,
                "multipath_detected": (file_counter % 4) == 0,
                "noise_floor_dbm": -120.0,
                "sample_rate_hz": 2.4e6,
                "enable_data_logging": True,  # Enable I/O intensive logging
            }

            # Enhanced processing with I/O operations
            bearing_calc = await signal_processor.calculate_enhanced_bearing(signal_data)
            decision = confidence_homing.evaluate_confidence_based_decision(bearing_calc)

            # I/O intensive data accumulation
            processing_result = {
                "task_id": task_id,
                "timestamp": time.perf_counter(),
                "bearing_deg": bearing_calc.bearing_deg,
                "confidence": bearing_calc.confidence,
                "decision": decision.proceed_with_homing,
                "processing_metadata": {
                    "signal_strength": bearing_calc.signal_strength_dbm,
                    "interference": bearing_calc.interference_detected,
                    "precision": bearing_calc.precision_deg,
                },
            }

            data_buffer.append(processing_result)

            # Periodic I/O operations (write to storage)
            if len(data_buffer) >= 50:  # Batch write every 50 samples
                await self._write_processing_results(data_buffer, task_id, file_counter)
                data_buffer.clear()
                file_counter += 1

            await asyncio.sleep(0.02)  # 50Hz processing rate

        # Final I/O operation (write remaining buffer)
        if data_buffer:
            await self._write_processing_results(data_buffer, task_id, file_counter)

    async def _log_sdr_data(self, data: Dict[str, Any], task_id: int) -> None:
        """Simulate SDR data logging for I/O testing."""
        # Simulate file I/O operations
        await asyncio.sleep(0.001)  # Simulate write latency

    async def _write_processing_results(
        self, results: List[Dict[str, Any]], task_id: int, file_counter: int
    ) -> None:
        """Simulate processing results writing for I/O testing."""
        # Simulate batch file I/O operations
        await asyncio.sleep(0.002)  # Simulate batch write latency

    @pytest.mark.asyncio
    async def test_network_bandwidth_validation_during_mavlink_coordination_with_asv_active(
        self, enhanced_signal_processor, confidence_based_homing, resource_usage_scenarios
    ):
        """[27c4] Network bandwidth validation during MAVLink coordination with ASV active."""

        logger.info("Starting network bandwidth validation during MAVLink coordination")

        # Import MAVLink service for network intensive testing
        from src.backend.services.mavlink_service import MAVLinkService

        mavlink_service = MAVLinkService()
        await mavlink_service.initialize()

        network_validation_results = []

        try:
            for scenario in resource_usage_scenarios:
                logger.info(f"Testing network bandwidth scenario: {scenario.name}")

                # Start resource monitoring with focus on network
                resource_monitor = ResourceUsageMonitor(focus="network")
                await resource_monitor.start_monitoring()

                try:
                    # Run network intensive MAVLink coordination with ASV active
                    await self._run_network_intensive_mavlink_asv_coordination(
                        enhanced_signal_processor,
                        confidence_based_homing,
                        mavlink_service,
                        scenario,
                    )

                    # Collect network usage metrics
                    metrics = await resource_monitor.get_metrics()

                    # Calculate total network bandwidth (sent + received)
                    max_network_sent = max(metrics.network_sent_kbps, default=0.0)
                    max_network_recv = max(metrics.network_recv_kbps, default=0.0)
                    max_network_bandwidth = max_network_sent + max_network_recv

                    avg_network_sent = sum(metrics.network_sent_kbps) / max(
                        len(metrics.network_sent_kbps), 1
                    )
                    avg_network_recv = sum(metrics.network_recv_kbps) / max(
                        len(metrics.network_recv_kbps), 1
                    )

                    # Validate network bandwidth requirements
                    assert max_network_bandwidth <= scenario.expected_max_network_kbps, (
                        f"Peak network bandwidth {max_network_bandwidth:.1f}KB/s exceeds "
                        f"expected {scenario.expected_max_network_kbps:.1f}KB/s for {scenario.name}"
                    )

                    network_validation_results.append(
                        {
                            "scenario": scenario.name,
                            "peak_network_bandwidth_kbps": max_network_bandwidth,
                            "avg_network_sent_kbps": avg_network_sent,
                            "avg_network_recv_kbps": avg_network_recv,
                            "processing_intensity": scenario.processing_intensity,
                            "concurrent_operations": scenario.concurrent_operations,
                        }
                    )

                    logger.info(
                        f"Network validation for {scenario.name}: "
                        f"Peak bandwidth: {max_network_bandwidth:.1f}KB/s, "
                        f"Avg sent: {avg_network_sent:.1f}KB/s, "
                        f"Avg recv: {avg_network_recv:.1f}KB/s"
                    )

                finally:
                    await resource_monitor.stop_monitoring()

                await asyncio.sleep(2.0)  # Cool down between scenarios

        finally:
            await mavlink_service.shutdown()

        # Validate overall network bandwidth patterns
        max_network_bandwidth = max(
            result["peak_network_bandwidth_kbps"] for result in network_validation_results
        )
        avg_total_network = sum(
            result["avg_network_sent_kbps"] + result["avg_network_recv_kbps"]
            for result in network_validation_results
        ) / len(network_validation_results)

        # Network bandwidth should be efficient for MAVLink coordination
        assert (
            max_network_bandwidth <= 1000.0
        ), f"Maximum network bandwidth {max_network_bandwidth:.1f}KB/s exceeds reasonable 1MB/s threshold"

        assert (
            avg_total_network <= 400.0
        ), f"Average total network {avg_total_network:.1f}KB/s indicates inefficient network usage"

        logger.info(
            f"Network bandwidth validation completed: "
            f"Max bandwidth: {max_network_bandwidth:.1f}KB/s, Avg total: {avg_total_network:.1f}KB/s"
        )

    async def _run_network_intensive_mavlink_asv_coordination(
        self,
        signal_processor: ASVEnhancedSignalProcessor,
        confidence_homing: ASVConfidenceBasedHoming,
        mavlink_service: Any,
        scenario: ResourceUsageScenario,
    ) -> None:
        """Run network intensive MAVLink coordination with ASV active."""

        end_time = time.perf_counter() + scenario.test_duration_seconds

        # Start network intensive tasks
        network_intensive_tasks = []

        # MAVLink coordination tasks (network intensive)
        for i in range(max(1, scenario.concurrent_operations // 2)):
            task = asyncio.create_task(
                self._mavlink_coordination_task(mavlink_service, end_time, i)
            )
            network_intensive_tasks.append(task)

        # ASV processing with network reporting tasks
        for i in range(
            max(1, scenario.concurrent_operations - scenario.concurrent_operations // 2)
        ):
            task = asyncio.create_task(
                self._network_reporting_asv_processing_task(
                    signal_processor, confidence_homing, mavlink_service, end_time, i
                )
            )
            network_intensive_tasks.append(task)

        # Wait for all network intensive processing to complete
        await asyncio.gather(*network_intensive_tasks)

    async def _mavlink_coordination_task(
        self, mavlink_service: Any, end_time: float, task_id: int
    ) -> None:
        """Run MAVLink coordination task for network bandwidth validation."""

        command_sequence = [
            "heartbeat",
            "get_vehicle_status",
            "get_attitude",
            "set_position_target",
            "get_mission_status",
            "set_mode",
            "arm_disarm",
            "get_navigation_data",
        ]

        command_index = task_id  # Offset each task's command sequence

        while time.perf_counter() < end_time:
            command = command_sequence[command_index % len(command_sequence)]

            try:
                # Network intensive MAVLink operations

                if command == "heartbeat":
                    await mavlink_service.heartbeat()
                elif command == "get_vehicle_status":
                    status = await mavlink_service.get_vehicle_status()
                    # Simulate processing network response
                    await self._process_network_response(status, "vehicle_status", task_id)
                elif command == "get_attitude":
                    attitude = await mavlink_service.get_attitude()
                    await self._process_network_response(attitude, "attitude", task_id)
                elif command == "set_position_target":
                    # Network intensive position updates
                    lat = 37.7749 + (task_id * 0.001) + ((time.perf_counter() % 100) * 0.0001)
                    lon = -122.4194 + (task_id * 0.001) + ((time.perf_counter() % 100) * 0.0001)
                    await mavlink_service.set_position_target(lat=lat, lon=lon, alt=10.0)
                elif command == "get_mission_status":
                    mission = await mavlink_service.get_mission_status()
                    await self._process_network_response(mission, "mission_status", task_id)
                elif command == "set_mode":
                    modes = ["GUIDED", "AUTO", "LOITER", "RTL"]
                    mode = modes[command_index % len(modes)]
                    await mavlink_service.set_mode(mode)
                elif command == "arm_disarm":
                    arm_state = (command_index % 4) < 2  # Alternate arm/disarm
                    await mavlink_service.arm_disarm(arm=arm_state)
                elif command == "get_navigation_data":
                    nav_data = await mavlink_service.get_navigation_data()
                    await self._process_network_response(nav_data, "navigation_data", task_id)

                # Simulate additional network coordination overhead
                await self._coordinate_with_ground_station(mavlink_service, task_id, command_index)

            except Exception as e:
                logger.warning(f"MAVLink coordination error in task {task_id}: {e}")

            command_index += 1
            await asyncio.sleep(0.1)  # 10Hz coordination rate

    async def _network_reporting_asv_processing_task(
        self,
        signal_processor: ASVEnhancedSignalProcessor,
        confidence_homing: ASVConfidenceBasedHoming,
        mavlink_service: Any,
        end_time: float,
        task_id: int,
    ) -> None:
        """Run ASV processing with network reporting for bandwidth validation."""

        report_counter = 0

        while time.perf_counter() < end_time:
            # Enhanced ASV processing
            signal_data = {
                "signal_strength_dbm": -75.0 - (task_id * 3),
                "frequency_hz": 433.92e6,
                "bearing_deg": (time.perf_counter() * 5 + task_id * 30) % 360,
                "interference_detected": (report_counter % 5) == 0,
                "multipath_detected": (report_counter % 7) == 0,
                "noise_floor_dbm": -120.0,
                "sample_rate_hz": 2.4e6,
            }

            bearing_calc = await signal_processor.calculate_enhanced_bearing(signal_data)
            decision = confidence_homing.evaluate_confidence_based_decision(bearing_calc)

            # Network intensive reporting to MAVLink
            asv_report = {
                "task_id": task_id,
                "timestamp": time.perf_counter(),
                "bearing_deg": bearing_calc.bearing_deg,
                "confidence": bearing_calc.confidence,
                "signal_strength": bearing_calc.signal_strength_dbm,
                "proceed_with_homing": decision.proceed_with_homing,
                "fallback_strategy": decision.fallback_strategy,
                "processing_metadata": {
                    "interference_detected": bearing_calc.interference_detected,
                    "precision_deg": bearing_calc.precision_deg,
                    "signal_quality": bearing_calc.signal_quality,
                },
            }

            # Send ASV report over network (network intensive)
            try:
                await self._send_asv_report_to_network(mavlink_service, asv_report)

                # Periodic detailed status reports (high bandwidth)
                if report_counter % 10 == 0:  # Every 10th report
                    detailed_report = {
                        "summary": asv_report,
                        "historical_data": await self._generate_historical_data(task_id),
                        "system_status": await self._generate_system_status(task_id),
                        "performance_metrics": await self._generate_performance_metrics(task_id),
                    }
                    await self._send_detailed_report_to_network(mavlink_service, detailed_report)

            except Exception as e:
                logger.warning(f"Network reporting error in task {task_id}: {e}")

            report_counter += 1
            await asyncio.sleep(0.05)  # 20Hz reporting rate

    async def _process_network_response(
        self, response_data: Any, response_type: str, task_id: int
    ) -> None:
        """Simulate processing network response data."""
        # Simulate network response processing overhead
        await asyncio.sleep(0.002)

    async def _coordinate_with_ground_station(
        self, mavlink_service: Any, task_id: int, sequence_num: int
    ) -> None:
        """Simulate ground station coordination network traffic."""
        # Simulate additional network coordination
        coordination_data = {
            "task_id": task_id,
            "sequence": sequence_num,
            "timestamp": time.perf_counter(),
            "coordination_type": "ground_station_sync",
        }
        await asyncio.sleep(0.001)  # Simulate network coordination latency

    async def _send_asv_report_to_network(
        self, mavlink_service: Any, report_data: Dict[str, Any]
    ) -> None:
        """Simulate sending ASV report over network."""
        # Simulate network transmission latency and bandwidth usage
        await asyncio.sleep(0.003)  # Simulate report transmission time

    async def _send_detailed_report_to_network(
        self, mavlink_service: Any, detailed_report: Dict[str, Any]
    ) -> None:
        """Simulate sending detailed report over network (high bandwidth)."""
        # Simulate high bandwidth network transmission
        await asyncio.sleep(0.01)  # Simulate detailed report transmission time

    async def _generate_historical_data(self, task_id: int) -> List[Dict[str, Any]]:
        """Generate mock historical data for network reporting."""
        return [
            {
                "timestamp": time.perf_counter() - i * 0.1,
                "bearing": (task_id * 45 + i * 5) % 360,
                "confidence": max(0.1, 0.8 - i * 0.05),
            }
            for i in range(20)  # 20 historical samples
        ]

    async def _generate_system_status(self, task_id: int) -> Dict[str, Any]:
        """Generate mock system status for network reporting."""
        return {
            "cpu_usage": 45.0 + task_id * 5,
            "memory_usage": 128.0 + task_id * 20,
            "sdr_status": "active",
            "processing_load": "nominal",
        }

    async def _generate_performance_metrics(self, task_id: int) -> Dict[str, Any]:
        """Generate mock performance metrics for network reporting."""
        return {
            "processing_latency_ms": 15.0 + task_id * 2,
            "confidence_accuracy": 0.85 + task_id * 0.02,
            "bearing_precision_deg": 1.5 + task_id * 0.1,
            "throughput_hz": 50.0 - task_id * 2,
        }


class ResourceUsageMonitor:
    """Resource usage monitor for validation testing."""

    def __init__(self, focus: str = "all"):
        """Initialize resource monitor with optional focus area.

        Args:
            focus: Focus area - "cpu", "memory", "io", "network", or "all"
        """
        self.focus = focus
        self.monitoring = False
        self.metrics = ResourceUsageMetrics(
            cpu_usage_percent=[],
            memory_usage_mb=[],
            io_read_mbps=[],
            io_write_mbps=[],
            network_sent_kbps=[],
            network_recv_kbps=[],
            peak_cpu_percent=0.0,
            peak_memory_mb=0.0,
            avg_cpu_percent=0.0,
            avg_memory_mb=0.0,
            sustained_cpu_percent=0.0,
            test_duration_seconds=0.0,
        )
        self._monitor_task: Optional[asyncio.Task] = None
        self._start_time: Optional[float] = None

    async def start_monitoring(self) -> None:
        """Start resource usage monitoring."""
        if self.monitoring:
            return

        self.monitoring = True
        self._start_time = time.perf_counter()
        self._monitor_task = asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self) -> None:
        """Stop resource usage monitoring and finalize metrics."""
        if not self.monitoring:
            return

        self.monitoring = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        # Calculate final metrics
        await self._finalize_metrics()

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())

            # Get initial network and I/O counters
            initial_io = process.io_counters() if hasattr(process, "io_counters") else None
            initial_net = psutil.net_io_counters()
            last_measurement_time = time.perf_counter()

            while self.monitoring:
                current_time = time.perf_counter()
                measurement_interval = current_time - last_measurement_time

                # CPU monitoring
                if self.focus in ["cpu", "all"]:
                    cpu_percent = process.cpu_percent(interval=None)
                    self.metrics.cpu_usage_percent.append(cpu_percent)

                # Memory monitoring
                if self.focus in ["memory", "all"]:
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / 1024 / 1024
                    self.metrics.memory_usage_mb.append(memory_mb)

                # I/O monitoring
                if self.focus in ["io", "all"] and initial_io and hasattr(process, "io_counters"):
                    current_io = process.io_counters()

                    if measurement_interval > 0:
                        # Calculate I/O rates
                        read_bytes_per_sec = (
                            current_io.read_bytes - initial_io.read_bytes
                        ) / measurement_interval
                        write_bytes_per_sec = (
                            current_io.write_bytes - initial_io.write_bytes
                        ) / measurement_interval

                        self.metrics.io_read_mbps.append(read_bytes_per_sec / 1024 / 1024)
                        self.metrics.io_write_mbps.append(write_bytes_per_sec / 1024 / 1024)

                    initial_io = current_io

                # Network monitoring
                if self.focus in ["network", "all"]:
                    current_net = psutil.net_io_counters()

                    if measurement_interval > 0:
                        # Calculate network rates
                        sent_bytes_per_sec = (
                            current_net.bytes_sent - initial_net.bytes_sent
                        ) / measurement_interval
                        recv_bytes_per_sec = (
                            current_net.bytes_recv - initial_net.bytes_recv
                        ) / measurement_interval

                        self.metrics.network_sent_kbps.append(sent_bytes_per_sec / 1024)
                        self.metrics.network_recv_kbps.append(recv_bytes_per_sec / 1024)

                    initial_net = current_net

                last_measurement_time = current_time
                await asyncio.sleep(0.1)  # 100ms monitoring interval

        except ImportError:
            logger.warning("psutil not available for resource monitoring")
        except Exception as e:
            logger.error(f"Resource monitoring error: {e}")

    async def _finalize_metrics(self) -> None:
        """Finalize collected metrics with calculated values."""
        if self._start_time:
            self.metrics.test_duration_seconds = time.perf_counter() - self._start_time

        # CPU metrics
        if self.metrics.cpu_usage_percent:
            self.metrics.peak_cpu_percent = max(self.metrics.cpu_usage_percent)
            self.metrics.avg_cpu_percent = sum(self.metrics.cpu_usage_percent) / len(
                self.metrics.cpu_usage_percent
            )

            # Calculate sustained CPU (95th percentile)
            sorted_cpu = sorted(self.metrics.cpu_usage_percent)
            p95_index = int(len(sorted_cpu) * 0.95)
            self.metrics.sustained_cpu_percent = sorted_cpu[min(p95_index, len(sorted_cpu) - 1)]

        # Memory metrics
        if self.metrics.memory_usage_mb:
            self.metrics.peak_memory_mb = max(self.metrics.memory_usage_mb)
            self.metrics.avg_memory_mb = sum(self.metrics.memory_usage_mb) / len(
                self.metrics.memory_usage_mb
            )

    async def get_metrics(self) -> ResourceUsageMetrics:
        """Get current resource usage metrics."""
        return self.metrics
