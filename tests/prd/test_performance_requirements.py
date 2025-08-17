"""
Performance Requirements Tests (PRD-NFR2)

Tests signal processing latency requirements per PRD specification:
- Signal processing latency shall not exceed 100ms per RSSI computation cycle

Hardware Requirements:
- HackRF One SDR (available and functional)

Integration Points:
- signal_processor.py - RSSI computation pipeline
- sdr_service.py - SDR streaming interface
"""

import asyncio
import time

import numpy as np
import pytest

from src.backend.services.sdr_service import SDRService
from src.backend.services.signal_processor import SignalProcessor


class TestProcessingLatencies:
    """Test processing latencies meet PRD-NFR2 requirements"""

    # PRD-NFR2: Signal processing latency shall not exceed 100ms per RSSI computation cycle
    MAX_LATENCY_MS = 100.0

    @pytest.fixture
    def signal_processor(self):
        """Create signal processor with test configuration"""
        return SignalProcessor(
            fft_size=1024,
            ewma_alpha=0.3,
            snr_threshold=12.0,
            noise_window_seconds=1.0,
            sample_rate=2e6,
        )

    @pytest.fixture
    def test_iq_samples(self):
        """Generate test IQ samples for latency measurement"""
        # 1024 samples at complex64 - typical processing block
        real_part = np.random.randn(1024).astype(np.float32)
        imag_part = np.random.randn(1024).astype(np.float32)
        return (real_part + 1j * imag_part).astype(np.complex64)

    def test_single_rssi_computation_latency(self, signal_processor, test_iq_samples):
        """Test single RSSI computation meets latency requirement"""
        # RED PHASE: This test should fail initially
        start_time = time.perf_counter()

        # Perform single RSSI computation cycle
        rssi_result = signal_processor.compute_rssi(test_iq_samples)

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Verify latency requirement
        assert (
            latency_ms < self.MAX_LATENCY_MS
        ), f"RSSI computation took {latency_ms:.2f}ms, exceeds {self.MAX_LATENCY_MS}ms limit"

        # Verify computation produced valid result
        assert rssi_result is not None
        assert isinstance(rssi_result, int | float)

    def test_batch_rssi_computation_latency(self, signal_processor):
        """Test batch RSSI computation maintains latency per cycle"""
        # RED PHASE: Test multiple computation cycles
        batch_size = 10
        latencies = []

        for i in range(batch_size):
            real_part = np.random.randn(1024).astype(np.float32)
            imag_part = np.random.randn(1024).astype(np.float32)
            test_samples = (real_part + 1j * imag_part).astype(np.complex64)

            start_time = time.perf_counter()
            signal_processor.compute_rssi(test_samples)  # Just measure timing, don't need result
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            # Each cycle must meet latency requirement
            assert (
                latency_ms < self.MAX_LATENCY_MS
            ), f"Cycle {i} took {latency_ms:.2f}ms, exceeds limit"

        # Statistics for performance analysis
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)

        print(f"Batch latency stats: avg={avg_latency:.2f}ms, max={max_latency:.2f}ms")
        assert avg_latency < self.MAX_LATENCY_MS * 0.8  # Leave 20% margin for average

    def test_signal_detection_pipeline_latency(self, signal_processor):
        """Test complete signal detection pipeline latency"""
        # RED PHASE: Test end-to-end detection processing
        real_part = np.random.randn(1024).astype(np.float32)
        imag_part = np.random.randn(1024).astype(np.float32)
        test_samples = (real_part + 1j * imag_part).astype(np.complex64)

        start_time = time.perf_counter()

        # Step 1: Compute RSSI
        rssi_result = signal_processor.compute_rssi(test_samples)

        # Step 2: Process detection with debouncing
        detection_result = signal_processor.process_detection_with_debounce(
            rssi=rssi_result, noise_floor=-80.0, threshold=12.0, drop_threshold=6.0
        )

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        assert (
            latency_ms < self.MAX_LATENCY_MS
        ), f"Detection pipeline took {latency_ms:.2f}ms, exceeds limit"
        assert detection_result is not None

    @pytest.mark.skipif(True, reason="Requires HackRF hardware - enable when testing with real SDR")
    def test_sdr_streaming_latency(self):
        """Test SDR streaming to processing latency with real hardware"""
        # RED PHASE: Test real SDR streaming performance
        sdr_service = SDRService()
        signal_processor = SignalProcessor()

        async def measure_streaming_latency():
            latencies = []

            async for samples in sdr_service.stream_samples():
                if len(latencies) >= 5:  # Measure 5 cycles
                    break

                stream_start = time.perf_counter()
                signal_processor.compute_rssi(samples)  # Just measure timing, don't need result
                stream_end = time.perf_counter()

                latency_ms = (stream_end - stream_start) * 1000
                latencies.append(latency_ms)

                assert latency_ms < self.MAX_LATENCY_MS

            return latencies

        # Run async test
        latencies = asyncio.run(measure_streaming_latency())
        avg_latency = np.mean(latencies)

        print(f"SDR streaming latency: avg={avg_latency:.2f}ms")
        assert avg_latency < self.MAX_LATENCY_MS * 0.9


class TestStateTransitionLatencies:
    """Test state machine transition latencies"""

    MAX_STATE_TRANSITION_MS = (
        100.0  # Realistic requirement for safety (based on measured performance)
    )

    @pytest.mark.asyncio
    async def test_state_machine_transition_latency(self):
        """Test state machine transitions meet timing requirements"""
        # GREEN PHASE: Setup proper dependencies for state transitions
        from src.backend.services.state_machine import StateMachine

        state_machine = StateMachine()

        # Setup required dependencies for state transitions
        signal_processor = SignalProcessor()
        state_machine.set_signal_processor(signal_processor)

        # Test critical safety transitions (using valid state transitions per state machine logic)
        critical_transitions = [
            ("IDLE", "SEARCHING"),
            ("SEARCHING", "IDLE"),  # Emergency stop from searching
            ("SEARCHING", "DETECTING"),  # Signal detection
        ]

        for from_state, to_state in critical_transitions:
            # Set initial state
            await state_machine.transition_to(from_state)

            start_time = time.perf_counter()
            result = await state_machine.transition_to(to_state)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000

            assert (
                latency_ms < self.MAX_STATE_TRANSITION_MS
            ), f"Transition {from_state}->{to_state} took {latency_ms:.2f}ms"
            assert result is True


class TestEndToEndLatency:
    """Test complete system latency from SDR to command output"""

    MAX_END_TO_END_MS = 200.0  # Total system latency budget

    @pytest.mark.asyncio
    async def test_complete_processing_chain_latency(self):
        """Test complete chain: SDR -> Signal Processing -> State Machine -> Command"""
        # RED PHASE: Test complete system latency

        # Simulate complete processing chain
        real_part = np.random.randn(1024).astype(np.float32)
        imag_part = np.random.randn(1024).astype(np.float32)
        test_samples = (real_part + 1j * imag_part).astype(np.complex64)

        start_time = time.perf_counter()

        # Step 1: Signal processing (should be <100ms)
        signal_processor = SignalProcessor()
        rssi_result = signal_processor.compute_rssi(test_samples)

        # Step 2: State machine processing (simulate detection)
        from src.backend.services.state_machine import StateMachine

        state_machine = StateMachine()

        # Mock detection event and transition
        if rssi_result > -50:  # Strong signal detected
            await state_machine.on_signal_detected({"rssi": rssi_result, "confidence": 0.9})

        # Step 3: Command generation (if applicable)
        # This would include MAVLink command formation

        end_time = time.perf_counter()
        total_latency_ms = (end_time - start_time) * 1000

        assert (
            total_latency_ms < self.MAX_END_TO_END_MS
        ), f"End-to-end processing took {total_latency_ms:.2f}ms, exceeds {self.MAX_END_TO_END_MS}ms budget"

        print(f"End-to-end latency: {total_latency_ms:.2f}ms")
