"""
Factory fixtures for test data generation.

Using factory-boy to create reusable test fixtures that reduce
test setup complexity and improve maintainability.
"""

import random
from datetime import datetime, timedelta

import factory
import factory.fuzzy
import numpy as np
from factory import Factory, LazyAttribute, SubFactory, Trait

from tests.hardware.mock.base import MockHardwareConfig
from tests.hardware.mock.mavlink_mock import MockMAVLinkInterface
from tests.hardware.mock.sdr_mock import MockSDRInterface


class MockHardwareConfigFactory(Factory):
    """Factory for MockHardwareConfig instances."""

    class Meta:
        model = MockHardwareConfig

    device_id = factory.Sequence(lambda n: f"mock_device_{n}")
    sample_rate = factory.fuzzy.FuzzyFloat(1e6, 20e6)
    frequency = factory.fuzzy.FuzzyFloat(850e6, 6.5e9)
    gain = factory.fuzzy.FuzzyFloat(0, 60)
    enabled = True
    simulate_errors = False
    error_rate = 0.0
    response_delay = 0.0

    class Params:
        """Factory traits for common configurations."""

        faulty = Trait(simulate_errors=True, error_rate=0.5)
        slow = Trait(response_delay=0.1)
        hackrf = Trait(device_id="hackrf_one", sample_rate=20e6, frequency=3.2e9, gain=30.0)
        rtlsdr = Trait(device_id="rtl_sdr", sample_rate=2.4e6, frequency=1.09e9, gain=40.0)


class MockSDRFactory(Factory):
    """Factory for MockSDRInterface instances."""

    class Meta:
        model = MockSDRInterface

    config = SubFactory(MockHardwareConfigFactory)

    @factory.post_generation
    def connect(obj, create, extracted, **kwargs):
        """Optionally connect the SDR after creation."""
        if extracted:
            import asyncio

            asyncio.run(obj.connect())


class MockMAVLinkFactory(Factory):
    """Factory for MockMAVLinkInterface instances."""

    class Meta:
        model = MockMAVLinkInterface

    config = SubFactory(MockHardwareConfigFactory, device_id="mavlink_fc")

    @factory.post_generation
    def telemetry(obj, create, extracted, **kwargs):
        """Set initial telemetry values."""
        if extracted:
            obj.telemetry.update(extracted)

    class Params:
        """Traits for common MAVLink states."""

        armed = Trait(
            telemetry=LazyAttribute(
                lambda o: {**o.telemetry, "armed": True, "flight_mode": "GUIDED"}
            )
        )
        low_battery = Trait(
            telemetry=LazyAttribute(
                lambda o: {**o.telemetry, "battery_voltage": 18.5, "battery_percent": 15.0}
            )
        )
        no_gps = Trait(
            telemetry=LazyAttribute(lambda o: {**o.telemetry, "gps_fix": 0, "satellites": 0})
        )


class SignalDetectionFactory(Factory):
    """Factory for SignalDetection instances."""

    class Meta:
        model = dict  # Using dict since SignalDetection is a Pydantic model

    id = factory.Faker("uuid4")
    timestamp = factory.Faker("date_time_between", start_date="-1h", end_date="now")
    frequency = factory.fuzzy.FuzzyFloat(3.19e9, 3.21e9)
    rssi = factory.fuzzy.FuzzyFloat(-100, -40)
    snr = factory.fuzzy.FuzzyFloat(0, 40)
    confidence = factory.fuzzy.FuzzyFloat(0, 100)
    location = LazyAttribute(
        lambda o: {
            "lat": random.uniform(-90, 90),
            "lon": random.uniform(-180, 180),
            "alt": random.uniform(0, 500),
        }
    )
    state = factory.fuzzy.FuzzyChoice(["IDLE", "SEARCHING", "DETECTING", "HOMING", "HOLDING"])

    class Params:
        """Traits for specific detection scenarios."""

        strong_signal = Trait(
            rssi=factory.fuzzy.FuzzyFloat(-60, -40),
            snr=factory.fuzzy.FuzzyFloat(20, 40),
            confidence=factory.fuzzy.FuzzyFloat(80, 100),
            state="DETECTING",
        )
        weak_signal = Trait(
            rssi=factory.fuzzy.FuzzyFloat(-100, -80),
            snr=factory.fuzzy.FuzzyFloat(0, 10),
            confidence=factory.fuzzy.FuzzyFloat(0, 30),
            state="SEARCHING",
        )
        homing = Trait(state="HOMING", confidence=factory.fuzzy.FuzzyFloat(70, 100))


class ConfigProfileFactory(Factory):
    """Factory for ConfigProfile instances."""

    class Meta:
        model = dict

    id = factory.Faker("uuid4")
    name = factory.Sequence(lambda n: f"Profile_{n}")
    sdr_config = LazyAttribute(
        lambda o: {"frequency": 3.2e9, "sample_rate": 2e6, "gain": 30, "bandwidth": 5e6}
    )
    signal_config = LazyAttribute(
        lambda o: {
            "fft_size": 1024,
            "ewma_alpha": 0.3,
            "trigger_threshold": 12,
            "drop_threshold": 6,
        }
    )
    homing_config = LazyAttribute(
        lambda o: {
            "forward_velocity_max": 10.0,
            "yaw_rate_max": 45.0,
            "approach_velocity": 2.0,
            "signal_loss_timeout": 10.0,
        }
    )
    is_default = False
    created_at = factory.Faker("date_time_between", start_date="-30d", end_date="now")
    updated_at = factory.LazyAttribute(lambda o: o.created_at)

    class Params:
        """Traits for specific profile types."""

        wifi_beacon = Trait(
            name="WiFi Beacon",
            sdr_config=LazyAttribute(
                lambda o: {"frequency": 2.437e9, "sample_rate": 20e6, "gain": 40, "bandwidth": 20e6}
            ),
        )
        lora_beacon = Trait(
            name="LoRa Beacon",
            sdr_config=LazyAttribute(
                lambda o: {"frequency": 915e6, "sample_rate": 1e6, "gain": 30, "bandwidth": 125e3}
            ),
        )
        default = Trait(name="Default Profile", is_default=True)


class RSSIReadingFactory(Factory):
    """Factory for RSSI reading data."""

    class Meta:
        model = dict

    timestamp = factory.Faker("date_time_between", start_date="-1m", end_date="now")
    rssi = factory.fuzzy.FuzzyFloat(-100, -40)
    noise_floor = factory.fuzzy.FuzzyFloat(-110, -90)
    detection_id = factory.Faker("uuid4")

    @factory.lazy_attribute
    def snr(self):
        """Calculate SNR from RSSI and noise floor."""
        return self.rssi - self.noise_floor


class SystemStateFactory(Factory):
    """Factory for SystemState instances."""

    class Meta:
        model = dict

    current_state = factory.fuzzy.FuzzyChoice(
        ["IDLE", "SEARCHING", "DETECTING", "HOMING", "HOLDING"]
    )
    homing_enabled = False
    flight_mode = factory.fuzzy.FuzzyChoice(["MANUAL", "GUIDED", "AUTO", "LOITER", "RTL"])
    battery_percent = factory.fuzzy.FuzzyFloat(20, 100)
    gps_status = factory.fuzzy.FuzzyChoice(["NO_FIX", "2D_FIX", "3D_FIX", "RTK"])
    mavlink_connected = True
    sdr_status = factory.fuzzy.FuzzyChoice(["CONNECTED", "DISCONNECTED", "ERROR"])
    safety_interlocks = LazyAttribute(
        lambda o: {
            "mode_check": o.flight_mode == "GUIDED",
            "battery_check": o.battery_percent > 20,
            "geofence_check": True,
            "signal_check": o.current_state in ["DETECTING", "HOMING"],
            "operator_check": o.homing_enabled,
        }
    )

    class Params:
        """Traits for specific system states."""

        ready_for_homing = Trait(
            current_state="DETECTING",
            homing_enabled=False,
            flight_mode="GUIDED",
            battery_percent=85,
            gps_status="3D_FIX",
            mavlink_connected=True,
            sdr_status="CONNECTED",
        )
        homing_active = Trait(current_state="HOMING", homing_enabled=True, flight_mode="GUIDED")
        emergency = Trait(battery_percent=15, flight_mode="RTL")


def create_test_signal_data(duration_seconds: int = 60, sample_rate: int = 10) -> np.ndarray:
    """
    Create realistic test signal data for testing.

    Args:
        duration_seconds: Duration of signal in seconds
        sample_rate: Samples per second

    Returns:
        Array of RSSI values with realistic characteristics
    """
    num_samples = duration_seconds * sample_rate
    t = np.linspace(0, duration_seconds, num_samples)

    # Base noise floor
    noise_floor = -95 + np.random.randn(num_samples) * 2

    # Add signal with varying strength
    signal_strength = -60 + 10 * np.sin(2 * np.pi * 0.1 * t)  # Slow variation
    signal_strength += 5 * np.sin(2 * np.pi * 1 * t)  # Faster variation

    # Add occasional signal spikes
    spike_indices = np.random.choice(num_samples, size=int(num_samples * 0.05), replace=False)
    signal_strength[spike_indices] += np.random.uniform(10, 20, size=len(spike_indices))

    # Combine signal and noise
    rssi = np.maximum(noise_floor, signal_strength + np.random.randn(num_samples) * 3)

    return rssi


def create_test_telemetry_sequence(duration_seconds: int = 60) -> list:
    """
    Create a sequence of telemetry updates for testing.

    Args:
        duration_seconds: Duration of telemetry sequence

    Returns:
        List of telemetry dictionaries with timestamps
    """
    telemetry_sequence = []
    start_time = datetime.now()

    for i in range(duration_seconds):
        timestamp = start_time + timedelta(seconds=i)

        telemetry = {
            "timestamp": timestamp.isoformat(),
            "battery_voltage": 22.2 - (i * 0.01),  # Slowly decreasing
            "battery_percent": max(20, 100 - (i * 0.5)),
            "altitude": min(100, i * 2),  # Climbing
            "latitude": 37.7749 + (i * 0.0001),
            "longitude": -122.4194 + (i * 0.0001),
            "heading": (i * 5) % 360,
            "groundspeed": min(10, i * 0.2),
            "satellites": min(15, 8 + (i // 10)),
        }

        telemetry_sequence.append(telemetry)

    return telemetry_sequence
