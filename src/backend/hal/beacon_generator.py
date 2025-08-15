"""
Software Beacon Generator for HackRF
Generates FM pulse signals for testing without physical beacons
"""

import asyncio
import logging
from dataclasses import dataclass

import numpy as np
import yaml

from src.backend.core.exceptions import (
    MAVLinkError,
)

logger = logging.getLogger(__name__)


@dataclass
class BeaconConfig:
    """Beacon signal configuration"""

    frequency: float = 3.2e9  # Center frequency (Hz)
    pulse_width: float = 0.001  # Pulse duration (seconds)
    pulse_period: float = 0.1  # Time between pulses (seconds)
    deviation: float = 5e3  # FM deviation (Hz)
    power_dbm: float = -10  # Output power (dBm)
    hop_enabled: bool = False  # Frequency hopping
    hop_frequencies: list[float] | None = None  # Hop sequence
    hop_dwell: float = 1.0  # Seconds per frequency


class BeaconGenerator:
    """Generate beacon signals for transmission"""

    def __init__(self, config: BeaconConfig, sample_rate: float = 20e6):
        self.config = config
        self.sample_rate = sample_rate
        self._running = False
        self._hop_index = 0

    def generate_pulse(self, duration: float) -> np.ndarray:
        """Generate single FM pulse"""
        num_samples = int(duration * self.sample_rate)
        t = np.arange(num_samples) / self.sample_rate

        # Simple FM modulation - rectangular pulse
        # Scale to 8-bit range for HackRF (-127 to 127)
        amplitude = 10 ** (self.config.power_dbm / 20) * 127

        # Generate carrier with FM modulation
        phase = 2 * np.pi * self.config.deviation * t
        signal = amplitude * np.exp(1j * phase)

        return signal.astype(np.complex64)

    def generate_pattern(self, total_duration: float) -> np.ndarray:
        """Generate radar-like pulse pattern"""
        total_samples = int(total_duration * self.sample_rate)
        pattern = np.zeros(total_samples, dtype=np.complex64)

        pulse_samples = int(self.config.pulse_width * self.sample_rate)
        period_samples = int(self.config.pulse_period * self.sample_rate)

        # Generate pulses at regular intervals
        pulse = self.generate_pulse(self.config.pulse_width)

        # Place pulses in pattern
        idx = 0
        while idx + pulse_samples < total_samples:
            pattern[idx : idx + pulse_samples] = pulse
            idx += period_samples

        return pattern

    def get_next_frequency(self) -> float:
        """Get next frequency for hopping"""
        if not self.config.hop_enabled or not self.config.hop_frequencies:
            return self.config.frequency

        freq = self.config.hop_frequencies[self._hop_index]
        self._hop_index = (self._hop_index + 1) % len(self.config.hop_frequencies)
        return freq

    async def start_transmission(self, hackrf_interface):
        """Start beacon transmission via HackRF"""
        if not hackrf_interface or not hackrf_interface.device:
            logger.error("HackRF not initialized")
            return False

        self._running = True
        logger.info("Starting beacon transmission")

        try:
            while self._running:
                # Handle frequency hopping
                if self.config.hop_enabled:
                    freq = self.get_next_frequency()
                    await hackrf_interface.set_freq(freq)
                    logger.debug(f"Hopped to {freq/1e9:.3f} GHz")

                # Generate and transmit pattern
                pattern = self.generate_pattern(self.config.hop_dwell)

                # Convert complex64 to int8 I/Q for HackRF
                iq_int8 = np.zeros(len(pattern) * 2, dtype=np.int8)
                iq_int8[0::2] = np.real(pattern).astype(np.int8)
                iq_int8[1::2] = np.imag(pattern).astype(np.int8)

                # Would transmit via hackrf.start_tx() here
                # For now, just simulate the timing
                await asyncio.sleep(self.config.hop_dwell)

        except MAVLinkError as e:
            logger.error(f"Beacon transmission error: {e}")
            self._running = False
            return False

    def stop_transmission(self):
        """Stop beacon transmission"""
        self._running = False
        logger.info("Beacon transmission stopped")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "BeaconGenerator":
        """Load configuration from YAML file"""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        beacon_data = data.get("beacon", {})

        # Parse configuration
        config = BeaconConfig(
            frequency=beacon_data.get("frequency", 3.2e9),
            pulse_width=beacon_data.get("pulse_width", 0.001),
            pulse_period=beacon_data.get("pulse_period", 0.1),
            deviation=beacon_data.get("deviation", 5e3),
            power_dbm=beacon_data.get("power_dbm", -10),
            hop_enabled=beacon_data.get("hop_enabled", False),
            hop_frequencies=beacon_data.get("hop_frequencies"),
            hop_dwell=beacon_data.get("hop_dwell", 1.0),
        )

        # Convert frequency range to list if specified
        if "frequency_range" in beacon_data:
            min_freq = beacon_data["frequency_range"]["min"]
            max_freq = beacon_data["frequency_range"]["max"]
            num_channels = beacon_data.get("hop_channels", 10)

            config.hop_frequencies = np.linspace(min_freq, max_freq, num_channels).tolist()
            config.hop_enabled = True

        return cls(config)


# Auto-configuration for testing
def create_test_beacon() -> BeaconGenerator:
    """Create beacon generator with test configuration"""
    config = BeaconConfig(
        frequency=3.2e9,  # 3.2 GHz
        pulse_width=0.001,  # 1ms pulse
        pulse_period=0.1,  # 10 Hz PRF
        deviation=5e3,  # 5 kHz FM deviation
        power_dbm=-10,  # Low power for testing
        hop_enabled=True,
        hop_frequencies=[
            850e6,  # 850 MHz
            1.8e9,  # 1.8 GHz
            3.2e9,  # 3.2 GHz
            5.8e9,  # 5.8 GHz
            6.5e9,  # 6.5 GHz
        ],
        hop_dwell=2.0,  # 2 seconds per frequency
    )

    return BeaconGenerator(config)
