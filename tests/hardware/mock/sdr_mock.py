"""
Mock SDR hardware interface for testing.
"""

from collections.abc import AsyncGenerator
from typing import Any

import numpy as np

from tests.hardware.mock.base import MockHardwareConfig, MockHardwareInterface


class MockSDRInterface(MockHardwareInterface):
    """Mock implementation of SDR hardware interface."""

    def __init__(self, config: MockHardwareConfig | None = None):
        """Initialize mock SDR with optional configuration."""
        super().__init__(config)
        self.sample_buffer = None
        self.buffer_size = 1024

    async def connect(self) -> bool:
        """Connect to the mock SDR device."""
        self._track_call("connect")
        await self._simulate_delay()

        if self._should_fail():
            self._error_count += 1
            return False

        self._is_connected = True
        return True

    async def disconnect(self) -> bool:
        """Disconnect from the mock SDR device."""
        self._track_call("disconnect")
        await self._simulate_delay()

        if self._should_fail():
            self._error_count += 1
            return False

        self._is_connected = False
        self._is_streaming = False
        return True

    async def configure(self, **kwargs) -> bool:
        """Configure the mock SDR device."""
        self._track_call("configure")
        await self._simulate_delay()

        if self._should_fail():
            self._error_count += 1
            return False

        # Update configuration
        if "sample_rate" in kwargs:
            self.config.sample_rate = kwargs["sample_rate"]
        if "frequency" in kwargs:
            self.config.frequency = kwargs["frequency"]
        if "gain" in kwargs:
            self.config.gain = kwargs["gain"]

        return True

    async def start_streaming(self) -> bool:
        """Start data streaming from the mock SDR."""
        self._track_call("start_streaming")
        await self._simulate_delay()

        if not self._is_connected:
            return False

        if self._should_fail():
            self._error_count += 1
            return False

        self._is_streaming = True
        return True

    async def stop_streaming(self) -> bool:
        """Stop data streaming from the mock SDR."""
        self._track_call("stop_streaming")
        await self._simulate_delay()

        if self._should_fail():
            self._error_count += 1
            return False

        self._is_streaming = False
        return True

    async def get_status(self) -> dict[str, Any]:
        """Get current status of the mock SDR."""
        self._track_call("get_status")
        await self._simulate_delay()

        return {
            "connected": self._is_connected,
            "streaming": self._is_streaming,
            "sample_rate": self.config.sample_rate,
            "frequency": self.config.frequency,
            "gain": self.config.gain,
            "error_count": self._error_count,
            "device_id": self.config.device_id,
        }

    async def stream_iq_samples(self) -> AsyncGenerator[np.ndarray, None]:
        """Stream mock IQ samples."""
        self._track_call("stream_iq_samples")

        while self._is_streaming:
            await self._simulate_delay()

            if self._should_fail():
                self._error_count += 1
                raise RuntimeError("Mock streaming error")

            # Generate mock IQ samples with configurable signal
            samples = self._generate_mock_samples()
            yield samples

    def _generate_mock_samples(self) -> np.ndarray:
        """Generate mock IQ samples with realistic characteristics."""
        # Create complex samples with noise and optional signal
        noise = np.random.randn(self.buffer_size) + 1j * np.random.randn(self.buffer_size)
        noise *= 0.1  # Scale noise

        # Add a mock signal if enabled
        if self.config.enabled:
            t = np.arange(self.buffer_size) / self.config.sample_rate
            signal_freq = 100e3  # 100 kHz offset from center
            signal = 0.5 * np.exp(1j * 2 * np.pi * signal_freq * t)
            samples = signal + noise
        else:
            samples = noise

        return samples.astype(np.complex64)
