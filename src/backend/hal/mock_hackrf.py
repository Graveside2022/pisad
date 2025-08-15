"""
Mock HackRF interface for testing without hardware
"""

import logging
import threading
import time
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)


# Mock pyhackrf module
class MockHackRF:
    """Mock HackRF device"""

    def __init__(self) -> None:
        self.board_id = "Mock HackRF"
        self.version_string = "1.0.0-mock"
        self.frequency = 3200000000  # 3.2 GHz default
        self.sample_rate = 20000000  # 20 Msps default
        self.lna_gain = 16
        self.vga_gain = 20
        self.amp_enabled = False
        self._rx_callback: Callable[[bytes], None] | None = None
        self._running = False
        self.connected = False
        self.is_streaming = False
        self._stream_thread: threading.Thread | None = None

    def open(self) -> int:
        """Open mock device."""
        self.connected = True
        return 0

    def close(self) -> int:
        """Close mock device."""
        if self.is_streaming:
            self.stop()
        self.connected = False
        return 0

    def set_freq(self, freq: int) -> int:
        """Set frequency in Hz."""
        if freq < 850000000 or freq > 6500000000:
            return -1  # Invalid frequency
        self.frequency = freq
        return 0

    def set_sample_rate(self, rate: int) -> int:
        """Set sample rate in Hz."""
        if rate < 2000000 or rate > 20000000:
            return -1  # Invalid sample rate
        self.sample_rate = rate
        return 0

    def set_lna_gain(self, gain: int) -> int:
        """Set LNA gain (0-40 dB in 8dB steps)."""
        # Round to nearest 8dB step
        self.lna_gain = min(40, max(0, (gain // 8) * 8))
        return 0

    def set_vga_gain(self, gain: int) -> int:
        """Set VGA gain (0-62 dB in 2dB steps)."""
        # Round to nearest 2dB step
        self.vga_gain = min(62, max(0, (gain // 2) * 2))
        return 0

    def set_amp_enable(self, enable: bool) -> int:
        """Enable/disable RF amplifier."""
        self.amp_enabled = enable
        return 0

    def start_rx(self, callback: Callable[[bytes], None]) -> int:
        """Start receiving IQ samples."""
        if not callback:
            return -1
        if self.is_streaming:
            return -1  # Already streaming

        self._rx_callback = callback
        self._running = True
        self.is_streaming = True

        # Start streaming thread
        self._stream_thread = threading.Thread(target=self._stream_samples)
        self._stream_thread.daemon = True
        self._stream_thread.start()

        logger.info("Mock HackRF RX started")
        return 0

    def stop(self) -> int:
        """Stop receiving."""
        self._running = False
        self.is_streaming = False
        if self._stream_thread:
            self._stream_thread.join(timeout=1.0)
            self._stream_thread = None
        self._rx_callback = None
        logger.info("Mock HackRF RX stopped")
        return 0

    def stop_rx(self) -> None:
        """Legacy stop method."""
        self.stop()

    def _stream_samples(self) -> None:
        """Generate and stream mock IQ samples."""
        samples_per_buffer = 16384

        while self._running and self._rx_callback:
            try:
                # Generate mock IQ samples (complex sinusoid with noise)
                t = np.arange(samples_per_buffer) / self.sample_rate
                freq_offset = 0.1  # Normalized frequency
                signal = np.exp(2j * np.pi * freq_offset * t)

                # Add noise based on gain settings
                noise_level = 0.1 * np.exp(-(self.lna_gain + self.vga_gain) / 40.0)
                noise = (
                    np.random.randn(samples_per_buffer) + 1j * np.random.randn(samples_per_buffer)
                ) * noise_level
                signal += noise

                # Convert to interleaved float32
                iq = np.zeros(samples_per_buffer * 2, dtype=np.float32)
                iq[0::2] = signal.real
                iq[1::2] = signal.imag

                # Call callback with samples
                self._rx_callback(iq.tobytes())

                # Simulate sample rate
                time.sleep(samples_per_buffer / self.sample_rate)

            except Exception as e:
                logger.debug(f"Mock streaming error: {e}")
                continue

    def get_device_info(self) -> dict:
        """Get device information."""
        return {
            "device": "MockHackRF",
            "serial": "MOCK123456",
            "version": "1.0.0",
            "api_version": "1.0",
            "status": "connected" if self.connected else "disconnected",
        }


def pyhackrf_init() -> None:
    """Mock initialization"""
    logger.info("Mock pyhackrf initialized")


def pyhackrf_exit() -> None:
    """Mock cleanup"""
    logger.info("Mock pyhackrf exited")


# Export the mock classes and functions
HackRF = MockHackRF
