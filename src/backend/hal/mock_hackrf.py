"""
Mock HackRF interface for testing without hardware
"""

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


# Mock pyhackrf module
class MockHackRF:
    """Mock HackRF device"""

    def __init__(self) -> None:
        self.board_id = "Mock HackRF"
        self.version_string = "1.0.0-mock"
        self.frequency = 3.2e9
        self.sample_rate = 20e6
        self.lna_gain = 16
        self.vga_gain = 20
        self.amp_enabled = False
        self._rx_callback = None
        self._running = False

    def set_freq(self, freq: float) -> None:
        self.frequency = freq

    def set_sample_rate(self, rate: float) -> None:
        self.sample_rate = rate

    def set_lna_gain(self, gain: int) -> None:
        self.lna_gain = gain

    def set_vga_gain(self, gain: int) -> None:
        self.vga_gain = gain

    def set_amp_enable(self, enable: bool) -> None:
        self.amp_enabled = enable

    def start_rx(self, callback: Callable[[bytes], None]) -> None:
        self._rx_callback = callback
        self._running = True
        logger.info("Mock HackRF RX started")

    def stop_rx(self) -> None:
        self._running = False
        self._rx_callback = None
        logger.info("Mock HackRF RX stopped")

    def close(self) -> None:
        self._running = False
        logger.info("Mock HackRF closed")


def pyhackrf_init() -> None:
    """Mock initialization"""
    logger.info("Mock pyhackrf initialized")


def pyhackrf_exit() -> None:
    """Mock cleanup"""
    logger.info("Mock pyhackrf exited")


# Export the mock classes and functions
HackRF = MockHackRF
