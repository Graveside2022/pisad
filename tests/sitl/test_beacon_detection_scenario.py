"""SITL test scenario for beacon detection with simulated RSSI patterns."""

import asyncio
import math
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.backend.services.beacon_simulator import BeaconSimulator
from src.backend.services.signal_processor import SignalProcessor
from src.backend.services.state_machine import StateMachine, SystemState

pytestmark = pytest.mark.serial


class TestBeaconDetectionScenario:
    """Test beacon detection scenario with simulated RSSI patterns."""

    @pytest.fixture
    def beacon_simulator(self):
        """Create beacon simulator."""
        return BeaconSimulator(
            beacon_lat=37.4419,
            beacon_lon=-122.1430,
            transmit_power=5.0,  # 5W beacon
            frequency=406025000,
            pulse_duration=0.5,
        )

    @pytest.fixture
    def signal_processor(self):
        """Create signal processor."""
        return SignalProcessor(
            fft_size=1024, ewma_alpha=0.3, snr_threshold=12.0, sample_rate=2.048e6
        )

    @pytest.fixture
    def state_machine(self):
        """Create state machine."""
        sm = StateMachine(enable_persistence=False)
        return sm

    @pytest.fixture
    def mock_mavlink_service(self):
        """Mock MAVLink service with simulated telemetry."""
        mock = MagicMock()
        mock.is_connected = MagicMock(return_value=True)

        # Simulate drone position updates
        self.current_position = {
            "lat": 37.4420,  # Start 100m north of beacon
            "lon": -122.1430,
            "alt": 50.0,
            "heading": 180.0,
            "groundspeed": 5.0,
        }

        async def get_telemetry():
            return self.current_position

        mock.get_telemetry = AsyncMock(side_effect=get_telemetry)
        mock.send_velocity_command = AsyncMock(return_value=True)
        return mock

    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two GPS coordinates in meters."""
        R = 6371000  # Earth radius in meters

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (
            math.sin(delta_lat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def simulate_rssi(self, distance: float, beacon_power: float = 5.0) -> float:
        """Simulate RSSI based on distance using free-space path loss."""
        if distance < 1:
            distance = 1  # Minimum distance to avoid log(0)

        # Free-space path loss model
        frequency = 406.025  # MHz
        path_loss = 20 * math.log10(distance) + 20 * math.log10(frequency) - 27.55

        # Convert beacon power to dBm (5W = 37 dBm)
        tx_power_dbm = 10 * math.log10(beacon_power * 1000)

        # Calculate RSSI
        rssi = tx_power_dbm - path_loss

        # Add some noise
        noise = np.random.normal(0, 2)
        rssi += noise

        # Clamp to realistic range
        return max(-120.0, min(-30.0, rssi))

    @pytest.mark.asyncio
    async def test_beacon_detection_approaching(
        self, signal_processor, state_machine, beacon_simulator
    ):
        """Test beacon detection as drone approaches."""
        await signal_processor.start()
        await state_machine.start()

        # Simulate drone approaching beacon
        distances = [500, 400, 300, 200, 100, 50, 20, 10]  # meters

        for distance in distances:
            # Simulate RSSI based on distance
            rssi = self.simulate_rssi(distance)

            # Update signal processor with simulated RSSI
            signal_processor._current_rssi = rssi
            signal_processor.rssi_history.append(rssi)

            # Check for detection
            detection = await signal_processor.detect_signal(rssi)

            # At close distances, should detect signal
            if distance < 200:
                assert detection is not None, f"Should detect signal at {distance}m"
                assert detection.rssi == rssi
                assert detection.snr > signal_processor.snr_threshold

            # Simulate time passing
            await asyncio.sleep(0.1)

        await signal_processor.stop()
        await state_machine.stop()

    @pytest.mark.asyncio
    async def test_beacon_detection_with_state_transitions(
        self, signal_processor, state_machine, beacon_simulator, mock_mavlink_service
    ):
        """Test state transitions during beacon detection."""
        # Set up dependencies
        state_machine.set_signal_processor(signal_processor)
        state_machine.set_mavlink_service(mock_mavlink_service)

        await signal_processor.start()
        await state_machine.start()

        # Start in SEARCHING state
        await state_machine.transition_to(SystemState.SEARCHING)
        assert state_machine.get_current_state() == SystemState.SEARCHING

        # Simulate weak signal detection
        weak_rssi = self.simulate_rssi(400)  # Far from beacon
        signal_processor._current_rssi = weak_rssi
        await signal_processor.detect_signal(weak_rssi)

        # Should remain in SEARCHING with weak signal
        assert state_machine.get_current_state() == SystemState.SEARCHING

        # Simulate strong signal detection
        strong_rssi = self.simulate_rssi(50)  # Close to beacon
        signal_processor._current_rssi = strong_rssi
        signal_processor.noise_floor = -100.0  # Set noise floor

        detection = await signal_processor.detect_signal(strong_rssi)
        assert detection is not None

        # Should transition to DETECTING
        await state_machine.handle_detection(strong_rssi, detection.confidence)
        assert state_machine.get_current_state() == SystemState.DETECTING

        # Enable homing
        state_machine.enable_homing(True)

        # With strong signal and homing enabled, should transition to HOMING
        if state_machine._homing_enabled and detection.snr > 15.0:
            await state_machine.transition_to(SystemState.HOMING)
            assert state_machine.get_current_state() == SystemState.HOMING

        await signal_processor.stop()
        await state_machine.stop()

    @pytest.mark.asyncio
    async def test_beacon_detection_intermittent_signal(self, signal_processor, state_machine):
        """Test handling of intermittent beacon signal."""
        await signal_processor.start()
        await state_machine.start()

        # Simulate intermittent signal (beacon pulses)
        signal_pattern = [
            (-95.0, True),  # Signal detected
            (-110.0, False),  # No signal
            (-93.0, True),  # Signal detected
            (-108.0, False),  # No signal
            (-91.0, True),  # Signal detected
        ]

        detections = []
        for rssi, should_detect in signal_pattern:
            signal_processor._current_rssi = rssi
            signal_processor.noise_floor = -105.0

            detection = await signal_processor.detect_signal(rssi)

            if should_detect:
                assert detection is not None
                detections.append(detection)
            else:
                assert detection is None

            await asyncio.sleep(0.5)  # Beacon pulse interval

        # Should have detected multiple pulses
        assert len(detections) == 3

        await signal_processor.stop()
        await state_machine.stop()

    @pytest.mark.asyncio
    async def test_beacon_detection_with_movement(
        self, signal_processor, beacon_simulator, mock_mavlink_service
    ):
        """Test beacon detection while drone is moving."""
        await signal_processor.start()

        # Simulate drone flying search pattern
        search_positions = [
            (37.4420, -122.1430),  # North
            (37.4419, -122.1431),  # West
            (37.4418, -122.1430),  # South
            (37.4419, -122.1429),  # East
            (37.4419, -122.1430),  # Center (beacon location)
        ]

        strongest_rssi = -120.0
        best_position = None

        for lat, lon in search_positions:
            # Update drone position
            mock_mavlink_service.current_position = {
                "lat": lat,
                "lon": lon,
                "alt": 50.0,
                "heading": 0.0,
                "groundspeed": 5.0,
            }

            # Calculate distance to beacon
            distance = self.calculate_distance(
                lat, lon, beacon_simulator.beacon_lat, beacon_simulator.beacon_lon
            )

            # Simulate RSSI
            rssi = self.simulate_rssi(distance)
            signal_processor._current_rssi = rssi

            # Track strongest signal
            if rssi > strongest_rssi:
                strongest_rssi = rssi
                best_position = (lat, lon)

            # Check for detection
            detection = await signal_processor.detect_signal(rssi)

            # Should detect when at beacon location
            if distance < 10:
                assert detection is not None
                assert detection.confidence > 90.0

            await asyncio.sleep(1.0)  # Time at each position

        # Best position should be at or near beacon
        assert best_position is not None
        best_distance = self.calculate_distance(
            best_position[0],
            best_position[1],
            beacon_simulator.beacon_lat,
            beacon_simulator.beacon_lon,
        )
        assert best_distance < 50  # Within 50m of beacon

        await signal_processor.stop()

    @pytest.mark.asyncio
    async def test_beacon_detection_noise_rejection(self, signal_processor):
        """Test rejection of noise and false signals."""
        await signal_processor.start()

        # Set realistic noise floor
        signal_processor.noise_floor = -95.0

        # Simulate various noise levels
        noise_samples = np.random.normal(-95, 5, 100)  # Noise around noise floor

        false_detections = 0
        for noise_rssi in noise_samples:
            signal_processor._current_rssi = float(noise_rssi)
            detection = await signal_processor.detect_signal(float(noise_rssi))

            if detection is not None:
                false_detections += 1

        # Should have very few false detections (< 5%)
        false_positive_rate = false_detections / len(noise_samples)
        assert false_positive_rate < 0.05

        await signal_processor.stop()
