"""Tests for SUBTASK-6.2.1.3: Enhanced signal processing integration with ASV algorithms.

SUBTASK-6.2.1.3 Test Implementation:
- [23a1-23a4]: ASV Doppler compensation integration
- [23b1-23b4]: ASV interference rejection algorithms
- [23c1-23c4]: ASV signal classification system
- [23d1-23d4]: Confidence-weighted bearing fusion

This test module validates enhanced signal processing with ASV professional algorithms
maintains <100ms processing latency while adding Doppler compensation, interference
rejection, signal classification, and multi-signal bearing fusion capabilities.
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.backend.services.asv_integration.asv_enhanced_signal_processor import (
    ASVBearingCalculation,
)
from src.backend.services.signal_processor import SignalProcessor
from src.backend.utils.doppler_compensation import DopplerCompensator, PlatformVelocity
from src.backend.utils.logging import get_logger

logger = get_logger(__name__)


class TestEnhancedSignalProcessingIntegration:
    """Test suite for SUBTASK-6.2.1.3 enhanced ASV signal processing integration."""

    @pytest.fixture
    def platform_velocity(self) -> PlatformVelocity:
        """Create test platform velocity for Doppler compensation testing."""
        return PlatformVelocity(
            vx_ms=15.0,  # 15 m/s north velocity
            vy_ms=10.0,  # 10 m/s east velocity
            vz_ms=-2.0,  # 2 m/s descent
            ground_speed_ms=18.0,  # Ground speed magnitude
        )

    @pytest.fixture
    def mock_asv_analyzer(self) -> MagicMock:
        """Create mock ASV analyzer for testing integration."""
        analyzer = MagicMock()
        analyzer.compute_precise_bearing.return_value = ASVBearingCalculation(
            bearing_deg=45.0,
            confidence=0.8,
            precision_deg=2.0,
            signal_strength_dbm=-65.0,
            signal_quality=0.85,
            timestamp_ns=int(time.time() * 1e9),
            analyzer_type="ASV_PROFESSIONAL",
            interference_detected=False,
            signal_classification="FM_CHIRP",
        )
        return analyzer

    @pytest.fixture
    def enhanced_signal_processor(self, mock_asv_analyzer) -> SignalProcessor:
        """Create enhanced signal processor with ASV integration."""
        processor = SignalProcessor(
            fft_size=1024,
            ewma_alpha=0.3,
            snr_threshold=12.0,
            sample_rate=2.048e6,
        )
        # Add platform velocity support for Doppler compensation
        processor.platform_velocity = None
        processor.doppler_compensator = DopplerCompensator()
        processor.asv_analyzer = mock_asv_analyzer
        return processor

    # [23a1] Update SignalProcessor to accept platform velocity from MAVLink telemetry
    def test_signal_processor_accepts_platform_velocity(
        self, enhanced_signal_processor, platform_velocity
    ):
        """Test [23a1]: SignalProcessor accepts and stores platform velocity from MAVLink."""
        # ARRANGE: Enhanced signal processor ready
        processor = enhanced_signal_processor

        # ACT: Set platform velocity from MAVLink telemetry
        processor.set_platform_velocity(platform_velocity)

        # ASSERT: Platform velocity stored correctly
        assert processor.platform_velocity is not None
        assert processor.platform_velocity.vx_ms == 15.0
        assert processor.platform_velocity.vy_ms == 10.0
        assert processor.platform_velocity.vz_ms == -2.0
        assert processor.platform_velocity.ground_speed_ms == 18.0

    def test_platform_velocity_updates_in_real_time(self, enhanced_signal_processor):
        """Test [23a1]: Platform velocity updates from MAVLink telemetry in real-time."""
        # ARRANGE: Enhanced signal processor with initial velocity
        processor = enhanced_signal_processor
        initial_velocity = PlatformVelocity(5.0, 5.0, 0.0, 7.0)
        processor.set_platform_velocity(initial_velocity)

        # ACT: Update with new MAVLink telemetry
        updated_velocity = PlatformVelocity(20.0, 15.0, -3.0, 25.0)
        processor.set_platform_velocity(updated_velocity)

        # ASSERT: Velocity updated correctly
        assert processor.platform_velocity.vx_ms == 20.0
        assert processor.platform_velocity.vy_ms == 15.0
        assert processor.platform_velocity.ground_speed_ms == 25.0

    # [23a2] Integrate DopplerCompensator.compensate_frequency() in RSSI computation pipeline
    def test_doppler_compensation_integration_in_rssi_pipeline(
        self, enhanced_signal_processor, platform_velocity
    ):
        """Test [23a2]: Doppler compensation integrated in RSSI computation pipeline."""
        # ARRANGE: Signal processor with platform velocity and test IQ data
        processor = enhanced_signal_processor
        processor.set_platform_velocity(platform_velocity)

        # Create test IQ data at 2.4 GHz (beacon frequency)
        test_iq_data = (np.random.random(1024) + 1j * np.random.random(1024)).astype(
            np.complex64
        ) * 0.1  # Low amplitude signal
        test_frequency = 2.4e9  # 2.4 GHz beacon

        # ACT: Process IQ data with Doppler compensation
        with patch.object(processor, "_compute_rssi_with_doppler_compensation") as mock_compute:
            mock_compute.return_value = -65.0  # Compensated RSSI
            rssi_reading = processor.process_iq_samples_with_doppler(test_iq_data, test_frequency)

        # ASSERT: Doppler compensation applied in pipeline
        mock_compute.assert_called_once()
        assert rssi_reading is not None
        # Doppler shift should be calculated for moving platform
        expected_doppler_shift = processor.doppler_compensator.calculate_doppler_shift(
            platform_velocity,
            test_frequency,
            45.0,  # Assuming 45° bearing
        )
        assert abs(expected_doppler_shift) > 0  # Non-zero Doppler shift expected

    def test_doppler_compensation_processing_latency_requirement(
        self, enhanced_signal_processor, platform_velocity
    ):
        """Test [23a2]: Doppler compensation maintains <100ms processing latency."""
        # ARRANGE: Signal processor with Doppler compensation enabled
        processor = enhanced_signal_processor
        processor.set_platform_velocity(platform_velocity)
        test_iq_data = (np.random.random(1024) + 1j * np.random.random(1024)).astype(
            np.complex64
        ) * 0.1

        # ACT: Measure processing time with Doppler compensation
        start_time = time.perf_counter()
        for _ in range(10):  # Process multiple samples for average
            processor.process_iq_samples_with_doppler(test_iq_data, 2.4e9)
        end_time = time.perf_counter()

        # ASSERT: Processing latency < 100ms requirement (PRD-NFR2)
        average_latency_ms = ((end_time - start_time) / 10) * 1000
        assert (
            average_latency_ms < 100.0
        ), f"Doppler compensation latency {average_latency_ms:.1f}ms exceeds 100ms requirement"

    # [23a3] Add Doppler-compensated frequency tracking to detection events
    def test_doppler_compensated_frequency_tracking_in_detection_events(
        self, enhanced_signal_processor, platform_velocity
    ):
        """Test [23a3]: Detection events include Doppler-compensated frequency tracking."""
        # ARRANGE: Signal processor with platform velocity and detection callback
        processor = enhanced_signal_processor
        processor.set_platform_velocity(platform_velocity)
        detected_events = []

        def capture_detection(event):
            detected_events.append(event)

        processor.add_detection_callback(capture_detection)

        # Create high-SNR test data to trigger detection
        test_iq_data = (np.random.random(1024) + 1j * np.random.random(1024)).astype(
            np.complex64
        ) * 2.0  # High amplitude
        test_frequency = 2.4e9  # 2.4 GHz beacon

        # ACT: Process IQ data with Doppler compensation and trigger detection
        rssi_reading = processor.process_iq_samples_with_doppler(test_iq_data, test_frequency)

        # ASSERT: Detection event includes Doppler compensation data
        assert len(detected_events) >= 1, "Expected detection event to be generated"
        detection_event = detected_events[0]

        # Verify original frequency is preserved
        assert detection_event.frequency == test_frequency

        # Verify Doppler-compensated frequency is calculated
        assert detection_event.doppler_compensated_frequency is not None
        assert (
            detection_event.doppler_compensated_frequency != test_frequency
        )  # Should be different due to platform velocity

        # Verify Doppler shift is calculated
        assert detection_event.doppler_shift_hz is not None
        assert (
            abs(detection_event.doppler_shift_hz) > 0
        )  # Non-zero shift expected for moving platform

        # Verify relationship: compensated_frequency = original_frequency - doppler_shift
        expected_compensated_frequency = test_frequency - detection_event.doppler_shift_hz
        assert (
            abs(detection_event.doppler_compensated_frequency - expected_compensated_frequency)
            < 1.0
        )  # Within 1 Hz tolerance

    def test_doppler_frequency_tracking_accuracy_validation(
        self, enhanced_signal_processor, platform_velocity
    ):
        """Test [23a3]: Doppler frequency tracking accuracy with known platform velocity."""
        # ARRANGE: Signal processor with known platform velocity
        processor = enhanced_signal_processor
        processor.set_platform_velocity(platform_velocity)
        detected_events = []

        def capture_detection(event):
            detected_events.append(event)

        processor.add_detection_callback(capture_detection)

        # Use known test parameters
        test_frequency = 433e6  # 433 MHz beacon frequency
        test_iq_data = (np.random.random(1024) + 1j * np.random.random(1024)).astype(
            np.complex64
        ) * 3.0

        # ACT: Process with Doppler compensation
        processor.process_iq_samples_with_doppler(test_iq_data, test_frequency)

        # ASSERT: Doppler calculations match expected physics
        assert len(detected_events) >= 1
        detection_event = detected_events[0]

        # Calculate expected Doppler shift manually for validation
        # For 45° bearing and given platform velocity
        expected_doppler_shift = processor.doppler_compensator.calculate_doppler_shift(
            platform_velocity, test_frequency, 45.0
        )

        # Verify calculated values match expected
        assert abs(detection_event.doppler_shift_hz - expected_doppler_shift) < 0.1  # Within 0.1 Hz

        # Verify Doppler-compensated frequency is reasonable
        expected_compensated_freq = test_frequency - expected_doppler_shift
        assert abs(detection_event.doppler_compensated_frequency - expected_compensated_freq) < 0.1

    # [23a4] Validate Doppler compensation accuracy with moving platform test scenarios
    def test_doppler_compensation_accuracy_various_velocities(self, enhanced_signal_processor):
        """Test [23a4]: Doppler compensation accuracy across various platform velocities."""
        # ARRANGE: Test scenarios with different platform velocities
        test_scenarios = [
            # Scenario 1: Low speed patrol
            PlatformVelocity(vx_ms=5.0, vy_ms=3.0, vz_ms=0.0, ground_speed_ms=5.83),
            # Scenario 2: Medium speed approach
            PlatformVelocity(vx_ms=15.0, vy_ms=10.0, vz_ms=-2.0, ground_speed_ms=18.0),
            # Scenario 3: High speed transit
            PlatformVelocity(vx_ms=25.0, vy_ms=20.0, vz_ms=-5.0, ground_speed_ms=32.0),
            # Scenario 4: Stationary hover
            PlatformVelocity(vx_ms=0.0, vy_ms=0.0, vz_ms=0.0, ground_speed_ms=0.0),
        ]

        test_frequency = 433e6  # 433 MHz beacon
        processor = enhanced_signal_processor

        for i, velocity in enumerate(test_scenarios):
            # ACT: Set velocity and process samples
            processor.set_platform_velocity(velocity)
            detected_events = []

            def capture_detection(event):
                detected_events.append(event)

            processor.add_detection_callback(capture_detection)

            # High amplitude to ensure detection
            test_iq_data = (np.random.random(1024) + 1j * np.random.random(1024)).astype(
                np.complex64
            ) * 3.0
            processor.process_iq_samples_with_doppler(test_iq_data, test_frequency)

            # ASSERT: Doppler calculations are accurate for each scenario
            assert len(detected_events) >= 1, f"Scenario {i+1}: Expected detection event"
            event = detected_events[-1]  # Get latest event

            # Calculate expected Doppler shift using physics
            expected_shift = processor.doppler_compensator.calculate_doppler_shift(
                velocity,
                test_frequency,
                45.0,  # 45° bearing assumption
            )

            # Verify accuracy within 1% tolerance for non-zero velocities
            if velocity.ground_speed_ms > 0.1:  # Non-stationary
                relative_error = abs(event.doppler_shift_hz - expected_shift) / abs(expected_shift)
                assert (
                    relative_error < 0.01
                ), f"Scenario {i+1}: Doppler shift error {relative_error:.3f} exceeds 1%"
            else:  # Stationary case
                assert (
                    abs(event.doppler_shift_hz) < 0.001
                ), f"Scenario {i+1}: Expected near-zero Doppler shift for stationary platform"

            # Clean up callback for next iteration
            processor.remove_detection_callback(capture_detection)

    def test_doppler_compensation_accuracy_various_frequencies(
        self, enhanced_signal_processor, platform_velocity
    ):
        """Test [23a4]: Doppler compensation accuracy across various RF frequencies."""
        # ARRANGE: Test frequencies common in SAR operations
        test_frequencies = [
            144e6,  # 144 MHz (VHF)
            433e6,  # 433 MHz (UHF ISM)
            868e6,  # 868 MHz (EU ISM)
            915e6,  # 915 MHz (US ISM)
            1.575e9,  # GPS L1
            2.4e9,  # 2.4 GHz ISM
        ]

        processor = enhanced_signal_processor
        processor.set_platform_velocity(platform_velocity)

        for freq in test_frequencies:
            # ACT: Process samples at each frequency
            detected_events = []

            def capture_detection(event):
                detected_events.append(event)

            processor.add_detection_callback(capture_detection)

            test_iq_data = (np.random.random(1024) + 1j * np.random.random(1024)).astype(
                np.complex64
            ) * 3.0
            processor.process_iq_samples_with_doppler(test_iq_data, freq)

            # ASSERT: Doppler compensation scales correctly with frequency
            assert len(detected_events) >= 1, f"Expected detection at {freq/1e6:.1f} MHz"
            event = detected_events[-1]

            # Calculate expected values
            expected_shift = processor.doppler_compensator.calculate_doppler_shift(
                platform_velocity, freq, 45.0
            )
            expected_compensated = freq - expected_shift

            # Verify frequency scaling (higher frequencies = larger Doppler shifts)
            shift_ratio = event.doppler_shift_hz / freq
            expected_ratio = expected_shift / freq
            assert (
                abs(shift_ratio - expected_ratio) < 1e-9
            ), f"Frequency scaling error at {freq/1e6:.1f} MHz"

            # Verify compensated frequency accuracy
            compensation_error = abs(event.doppler_compensated_frequency - expected_compensated)
            assert (
                compensation_error < 0.1
            ), f"Compensation error {compensation_error:.3f} Hz at {freq/1e6:.1f} MHz"

            processor.remove_detection_callback(capture_detection)

    def test_doppler_compensation_accuracy_various_bearings(
        self, enhanced_signal_processor, platform_velocity
    ):
        """Test [23a4]: Doppler compensation accuracy with various bearing scenarios."""
        # ARRANGE: Test different bearing angles (currently using fixed 45° in implementation)
        # This test validates the current implementation and documents future enhancement needs
        test_bearings = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]  # Every 45°
        test_frequency = 433e6

        processor = enhanced_signal_processor
        processor.set_platform_velocity(platform_velocity)

        # ACT & ASSERT: Current implementation uses fixed 45° bearing
        # Future subtasks will enhance this with actual bearing calculation
        detected_events = []

        def capture_detection(event):
            detected_events.append(event)

        processor.add_detection_callback(capture_detection)

        test_iq_data = (np.random.random(1024) + 1j * np.random.random(1024)).astype(
            np.complex64
        ) * 3.0
        processor.process_iq_samples_with_doppler(test_iq_data, test_frequency)

        assert len(detected_events) >= 1, "Expected detection event"
        event = detected_events[0]

        # Verify current implementation uses 45° bearing assumption
        expected_shift_45deg = processor.doppler_compensator.calculate_doppler_shift(
            platform_velocity, test_frequency, 45.0
        )

        assert (
            abs(event.doppler_shift_hz - expected_shift_45deg) < 0.001
        ), "Current implementation should use 45° bearing"

        # Document expected behavior for future enhancement
        # When bearing calculation is enhanced in future subtasks, Doppler shift should vary with bearing
        logger.info(
            "FUTURE ENHANCEMENT: Doppler compensation will be enhanced with real bearing calculation in subsequent subtasks"
        )

    def test_doppler_compensation_performance_under_realistic_conditions(
        self, enhanced_signal_processor
    ):
        """Test [23a4]: Doppler compensation performance under realistic SAR mission conditions."""
        # ARRANGE: Realistic SAR mission scenario
        # Simulate drone moving at 20 m/s northeast with slight descent
        mission_velocity = PlatformVelocity(
            vx_ms=14.14,  # North component (20 m/s * cos(45°))
            vy_ms=14.14,  # East component (20 m/s * sin(45°))
            vz_ms=-1.0,  # Slight descent
            ground_speed_ms=20.0,
        )

        processor = enhanced_signal_processor
        processor.set_platform_velocity(mission_velocity)

        # Multiple beacon frequencies for comprehensive validation
        beacon_frequencies = [144e6, 433e6, 868e6]  # Common SAR beacon frequencies

        for freq in beacon_frequencies:
            # ACT: Process realistic signal with noise
            detected_events = []

            def capture_detection(event):
                detected_events.append(event)

            processor.add_detection_callback(capture_detection)

            # Simulate realistic signal with some noise
            signal_amplitude = 2.5  # Strong enough for detection
            noise_level = 0.1
            test_iq_data = (
                np.random.random(1024) * signal_amplitude
                + np.random.normal(0, noise_level, 1024)
                + 1j
                * (
                    np.random.random(1024) * signal_amplitude
                    + np.random.normal(0, noise_level, 1024)
                )
            ).astype(np.complex64)

            # Measure processing time
            start_time = time.perf_counter()
            rssi_reading = processor.process_iq_samples_with_doppler(test_iq_data, freq)
            processing_time = time.perf_counter() - start_time

            # ASSERT: Performance and accuracy requirements met
            assert (
                processing_time < 0.1
            ), f"Processing time {processing_time*1000:.1f}ms exceeds 100ms requirement"
            assert len(detected_events) >= 1, f"Expected detection at {freq/1e6:.1f} MHz"

            event = detected_events[-1]

            # Validate Doppler compensation results
            expected_shift = processor.doppler_compensator.calculate_doppler_shift(
                mission_velocity, freq, 45.0
            )

            # Should have measurable Doppler shift for 20 m/s ground speed
            # At 144 MHz: 20 m/s * cos(45°) * 144e6 / 3e8 ≈ 6.8 Hz
            # At 433 MHz: 20 m/s * cos(45°) * 433e6 / 3e8 ≈ 20.4 Hz
            # At 868 MHz: 20 m/s * cos(45°) * 868e6 / 3e8 ≈ 40.8 Hz
            expected_min_shift = 20.0 * 0.707 * freq / 3e8  # Conservative estimate
            assert (
                abs(event.doppler_shift_hz) > expected_min_shift * 0.8
            ), f"Expected Doppler shift >{expected_min_shift*0.8:.1f}Hz at {freq/1e6:.1f} MHz, got {abs(event.doppler_shift_hz):.1f}Hz"
            assert (
                abs(event.doppler_shift_hz - expected_shift) < 1.0
            ), f"Doppler shift accuracy error at {freq/1e6:.1f} MHz"

            # Verify RSSI reading is valid
            assert rssi_reading.rssi > -120.0, "Expected valid RSSI reading"
            assert rssi_reading.snr > processor.snr_threshold, "Expected detection-level SNR"

            processor.remove_detection_callback(capture_detection)
