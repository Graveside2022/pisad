"""
HackRF Interface with correct API for hackrf module (pyhackrf 0.2.0)
Compatible with the actual hackrf Python module that's installed
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Try to import real hackrf module, fall back to mock
try:
    import hackrf
    HACKRF_AVAILABLE = True
    logger.info("Using real hackrf library")
except ImportError:
    HACKRF_AVAILABLE = False
    logger.warning("hackrf not available, using mock for testing")
    from src.backend.hal import mock_hackrf as hackrf


@dataclass
class HackRFConfig:
    """HackRF configuration parameters"""

    frequency: float = 3.2e9  # 3.2 GHz for beacon
    sample_rate: float = 20e6  # 20 Msps
    lna_gain: int = 16  # 0-40 dB in 8dB steps
    vga_gain: int = 20  # 0-62 dB in 2dB steps
    amp_enable: bool = False  # RF amplifier
    min_frequency: float = 850e6  # 850 MHz minimum
    max_frequency: float = 6.5e9  # 6.5 GHz maximum


class HackRFInterface:
    """Interface for HackRF One SDR using correct hackrf module API"""

    def __init__(self, config: HackRFConfig | None = None):
        self.config = config or HackRFConfig()
        self.device: hackrf.HackRF | None = None
        self._running = False
        self._callback: Callable[[bytes], None] | None = None
        self._rx_active = False

    async def open(self) -> bool:
        """Open HackRF device via USB"""
        try:
            # Create HackRF instance
            self.device = hackrf.HackRF()
            
            # The hackrf module automatically opens on creation
            # Check if device is opened
            if not hasattr(self.device, 'device_opened') or not self.device.device_opened:
                # Try to explicitly open
                try:
                    result = self.device.open()
                    if result != 0:
                        logger.error(f"Failed to open HackRF, error code: {result}")
                        return False
                except Exception as e:
                    logger.warning(f"HackRF open raised exception (may already be open): {e}")
                    # Continue - device might be auto-opened

            # Get device info
            try:
                serial = self.device.get_serial_no()
                logger.info(f"HackRF opened successfully - Serial: {serial}")
            except:
                logger.info("HackRF opened (serial number unavailable)")

            # Set initial configuration
            await self.set_freq(self.config.frequency)
            await self.set_sample_rate(self.config.sample_rate)
            await self.set_lna_gain(self.config.lna_gain)
            await self.set_vga_gain(self.config.vga_gain)
            
            if self.config.amp_enable:
                await self.set_amp_enable(True)

            return True

        except Exception as e:
            logger.error(f"Failed to open HackRF: {e}")
            self.device = None
            return False

    async def set_freq(self, freq: float) -> bool:
        """Set center frequency in Hz (3.2e9 for 3.2 GHz)"""
        if not self.device:
            return False

        try:
            # The hackrf module uses set_freq with Hz as integer
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
            self.device.set_sample_rate(int(rate))
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
            if enable:
                self.device.enable_amp()
            else:
                self.device.disable_amp()
            self.config.amp_enable = enable
            logger.info(f"RF amplifier {'enabled' if enable else 'disabled'}")
            return True
        except Exception as e:
            logger.error(f"Failed to set amplifier: {e}")
            return False

    async def start_rx(self, callback: Callable[[np.ndarray], None]) -> bool:
        """Start receiving IQ samples"""
        if not self.device or self._rx_active:
            return False

        try:
            self._callback = callback
            self._running = True
            self._rx_active = True
            
            # Start receiving in a background task
            asyncio.create_task(self._rx_loop())
            
            logger.info("Started RX streaming")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start RX: {e}")
            self._running = False
            self._rx_active = False
            return False

    async def _rx_loop(self):
        """Background loop for receiving samples"""
        try:
            while self._running and self.device:
                # Read samples (returns numpy array of complex64)
                samples = self.device.read_samples(16384)  # Read 16k samples
                
                if samples is not None and len(samples) > 0 and self._callback:
                    # Convert to complex float32 numpy array if needed
                    if not isinstance(samples, np.ndarray):
                        samples = np.array(samples, dtype=np.complex64)
                    
                    # Call the callback with samples
                    self._callback(samples)
                
                # Small delay to prevent CPU spinning
                await asyncio.sleep(0.001)
                
        except Exception as e:
            logger.error(f"RX loop error: {e}")
        finally:
            self._rx_active = False

    async def stop(self) -> bool:
        """Stop receiving"""
        if not self.device:
            return False

        try:
            self._running = False
            
            # Stop RX if using the streaming API
            if hasattr(self.device, 'stop_rx'):
                self.device.stop_rx()
            
            # Wait for RX loop to finish
            for _ in range(10):
                if not self._rx_active:
                    break
                await asyncio.sleep(0.1)
            
            logger.info("Stopped RX streaming")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop RX: {e}")
            return False

    async def close(self) -> None:
        """Close HackRF device"""
        if self.device:
            try:
                await self.stop()
                self.device.close()
                logger.info("HackRF closed")
            except Exception as e:
                logger.warning(f"Error closing HackRF: {e}")
            finally:
                self.device = None

    async def get_info(self) -> dict[str, any]:
        """Get device information"""
        if not self.device:
            return {"status": "Not connected"}

        info = {
            "status": "Connected",
            "frequency": self.config.frequency,
            "sample_rate": self.config.sample_rate,
            "lna_gain": self.config.lna_gain,
            "vga_gain": self.config.vga_gain,
            "amp_enabled": self.config.amp_enable,
        }

        try:
            info["serial_number"] = self.device.get_serial_no()
        except:
            info["serial_number"] = "Unknown"

        return info


async def auto_detect_hackrf() -> HackRFInterface | None:
    """Auto-detect and initialize HackRF if available"""
    if not HACKRF_AVAILABLE:
        logger.warning("hackrf module not available")
        return None
        
    try:
        hackrf_interface = HackRFInterface()
        if await hackrf_interface.open():
            logger.info("HackRF auto-detected and initialized")
            
            # Test basic functionality
            info = await hackrf_interface.get_info()
            logger.info(f"HackRF info: {info}")
            
            return hackrf_interface
        else:
            logger.warning("HackRF device not found or failed to open")
            return None
            
    except Exception as e:
        logger.error(f"HackRF auto-detection failed: {e}")
        return None


# For backward compatibility during migration
async def test_hackrf_hardware():
    """Test if real HackRF hardware works"""
    hackrf = await auto_detect_hackrf()
    if hackrf:
        print("✅ HackRF hardware detected and working!")
        await hackrf.close()
        return True
    else:
        print("❌ HackRF hardware not available")
        return False


if __name__ == "__main__":
    # Test the interface
    asyncio.run(test_hackrf_hardware())