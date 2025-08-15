"""
HackRF Hardware Abstraction Layer
Uses pyhackrf library (installed via uv)
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Try to import real hackrf (pyhackrf 0.2.0 installs as 'hackrf'), fall back to mock
try:
    import hackrf as pyhackrf

    logger.info("Using real hackrf library (pyhackrf 0.2.0)")
except ImportError:
    logger.warning("hackrf not available, using mock for testing")
    from src.backend.hal import mock_hackrf as pyhackrf


@dataclass
class HackRFConfig:
    """HackRF configuration parameters"""

    frequency: float = 3.2e9  # 3.2 GHz for beacon
    sample_rate: float = 20e6  # 20 Msps
    lna_gain: int = 16  # 0-40 dB in 8dB steps
    vga_gain: int = 20  # 0-62 dB in 2dB steps
    amp_enable: bool = False  # RF amplifier


class HackRFInterface:
    """Hardware interface for HackRF One SDR"""

    def __init__(self, config: HackRFConfig):
        self.config = config
        self.device: pyhackrf.HackRF | None = None
        self._running = False
        self._callback: Callable | None = None

    async def open(self) -> bool:
        """Open HackRF device via USB"""
        try:
            # Initialize pyhackrf
            pyhackrf.pyhackrf_init()

            # Open first available HackRF
            self.device = pyhackrf.HackRF()

            logger.info("HackRF opened successfully")
            logger.info(f"Board ID: {self.device.board_id}")
            logger.info(f"Version: {self.device.version_string}")

            return True

        except Exception as e:
            logger.error(f"Failed to open HackRF: {e}")
            return False

    async def set_freq(self, freq: float) -> bool:
        """Set center frequency in Hz (3.2e9 for 3.2 GHz)"""
        if not self.device:
            return False

        try:
            self.device.set_freq(int(freq))
            self.config.frequency = freq
            logger.info(f"Frequency set to {freq/1e9:.3f} GHz")
            return True
        except Exception as e:
            logger.error(f"Failed to set frequency: {e}")
            return False

    async def set_sample_rate(self, rate: float) -> bool:
        """Set sample rate in Hz (20e6 for 20 Msps)"""
        if not self.device:
            return False

        try:
            self.device.set_sample_rate(rate)
            self.config.sample_rate = rate
            logger.info(f"Sample rate set to {rate/1e6:.1f} Msps")
            return True
        except Exception as e:
            logger.error(f"Failed to set sample rate: {e}")
            return False

    async def set_lna_gain(self, gain: int) -> bool:
        """Set LNA gain 0-40 dB in 8dB steps"""
        if not self.device:
            return False

        # Round to nearest 8dB step
        gain = (gain // 8) * 8
        gain = max(0, min(40, gain))

        try:
            self.device.set_lna_gain(gain)
            self.config.lna_gain = gain
            logger.info(f"LNA gain set to {gain} dB")
            return True
        except Exception as e:
            logger.error(f"Failed to set LNA gain: {e}")
            return False

    async def set_vga_gain(self, gain: int) -> bool:
        """Set VGA gain 0-62 dB in 2dB steps"""
        if not self.device:
            return False

        # Round to nearest 2dB step
        gain = (gain // 2) * 2
        gain = max(0, min(62, gain))

        try:
            self.device.set_vga_gain(gain)
            self.config.vga_gain = gain
            logger.info(f"VGA gain set to {gain} dB")
            return True
        except Exception as e:
            logger.error(f"Failed to set VGA gain: {e}")
            return False

    async def set_amp_enable(self, enable: bool) -> bool:
        """Enable/disable RF amplifier"""
        if not self.device:
            return False

        try:
            self.device.set_amp_enable(enable)
            self.config.amp_enable = enable
            logger.info(f"RF amplifier {'enabled' if enable else 'disabled'}")
            return True
        except Exception as e:
            logger.error(f"Failed to set amplifier: {e}")
            return False

    async def configure(self) -> bool:
        """Apply all configuration settings"""
        if not self.device:
            return False

        success = True
        success &= await self.set_freq(self.config.frequency)
        success &= await self.set_sample_rate(self.config.sample_rate)
        success &= await self.set_lna_gain(self.config.lna_gain)
        success &= await self.set_vga_gain(self.config.vga_gain)
        success &= await self.set_amp_enable(self.config.amp_enable)

        return success

    def _rx_callback(self, hackrf_transfer):
        """Internal callback for receiving samples"""
        if self._callback:
            # Convert bytes to complex float32
            bytes_data = bytes(
                hackrf_transfer.contents.buffer[: hackrf_transfer.contents.valid_length]
            )

            # HackRF provides signed 8-bit I/Q samples
            iq_bytes = np.frombuffer(bytes_data, dtype=np.int8)

            # Convert to complex float32 (-1.0 to 1.0 range)
            iq_samples = iq_bytes.astype(np.float32) / 127.0

            # Interleaved I/Q to complex
            samples = iq_samples[0::2] + 1j * iq_samples[1::2]

            # Call user callback
            self._callback(samples)

        return 0  # Continue receiving

    async def start_rx(self, callback: Callable) -> bool:
        """Start receiving IQ samples"""
        if not self.device or self._running:
            return False

        try:
            self._callback = callback
            self._running = True

            # Start RX with callback
            self.device.start_rx(self._rx_callback)

            logger.info("HackRF RX started")
            return True

        except Exception as e:
            logger.error(f"Failed to start RX: {e}")
            self._running = False
            return False

    async def stop_rx(self) -> bool:
        """Stop receiving"""
        if not self.device or not self._running:
            return False

        try:
            self.device.stop_rx()
            self._running = False
            self._callback = None

            logger.info("HackRF RX stopped")
            return True

        except Exception as e:
            logger.error(f"Failed to stop RX: {e}")
            return False

    async def read_samples(self, num_samples: int) -> np.ndarray:
        """Read samples synchronously (blocking)"""
        if not self.device:
            return np.array([], dtype=np.complex64)

        samples_buffer = []
        samples_needed = num_samples

        def collect_samples(samples):
            nonlocal samples_needed
            samples_buffer.extend(samples[:samples_needed])
            samples_needed -= len(samples)

        # Start RX with collection callback
        await self.start_rx(collect_samples)

        # Wait for samples
        while samples_needed > 0:
            await asyncio.sleep(0.001)

        # Stop RX
        await self.stop_rx()

        return np.array(samples_buffer, dtype=np.complex64)

    async def close(self):
        """Close HackRF device"""
        if self._running:
            await self.stop_rx()

        if self.device:
            try:
                self.device.close()
                logger.info("HackRF closed")
            except (AttributeError, Exception) as e:
                logger.warning(f"Error closing HackRF: {e}")
            finally:
                self.device = None

        # Cleanup pyhackrf
        pyhackrf.pyhackrf_exit()

    async def get_info(self) -> dict:
        """Get device information"""
        if not self.device:
            return {"status": "disconnected"}

        return {
            "status": "connected",
            "board_id": self.device.board_id,
            "version": self.device.version_string,
            "frequency": self.config.frequency,
            "sample_rate": self.config.sample_rate,
            "lna_gain": self.config.lna_gain,
            "vga_gain": self.config.vga_gain,
            "amp_enabled": self.config.amp_enable,
        }


# Auto-detection function
async def auto_detect_hackrf() -> HackRFInterface | None:
    """Auto-detect and initialize HackRF"""
    try:
        config = HackRFConfig()
        hackrf = HackRFInterface(config)

        if await hackrf.open():
            if await hackrf.configure():
                logger.info("HackRF auto-detected and configured")
                return hackrf
            else:
                await hackrf.close()

    except Exception as e:
        logger.error(f"HackRF auto-detection failed: {e}")

    return None
