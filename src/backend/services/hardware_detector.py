"""
Hardware Auto-Detection Service
Detects and initializes connected hardware at startup
"""

import asyncio
import logging
from dataclasses import dataclass

from src.backend.core.exceptions import (
    SDRError,
)
from src.backend.hal.hackrf_interface import HackRFInterface, auto_detect_hackrf
from src.backend.hal.mavlink_interface import MAVLinkInterface, auto_detect_cube_orange

logger = logging.getLogger(__name__)


@dataclass
class HardwareStatus:
    """Status of detected hardware"""

    sdr_connected: bool = False
    sdr_type: str = "none"
    sdr_info: dict | None = None

    flight_controller_connected: bool = False
    flight_controller_type: str = "none"
    flight_controller_info: dict | None = None

    timestamp: float = 0.0


class HardwareDetector:
    """Auto-detect and manage hardware connections"""

    def __init__(self):
        self.hackrf: HackRFInterface | None = None
        self.cube_orange: MAVLinkInterface | None = None
        self.status = HardwareStatus()
        self._retry_delay = 1.0  # Start with 1 second
        self._max_retry_delay = 30.0  # Max 30 seconds

    async def detect_sdr(self) -> bool:
        """Detect and initialize SDR hardware"""
        logger.info("Detecting SDR hardware...")

        # Try HackRF first (primary SDR)
        self.hackrf = await auto_detect_hackrf()
        if self.hackrf:
            self.status.sdr_connected = True
            self.status.sdr_type = "hackrf"
            self.status.sdr_info = await self.hackrf.get_info()
            logger.info(f"HackRF detected: {self.status.sdr_info}")
            return True

        # Could try other SDRs here (RTL-SDR, USRP, etc.)
        logger.warning("No SDR hardware detected")
        self.status.sdr_connected = False
        self.status.sdr_type = "none"
        return False

    async def detect_flight_controller(self) -> bool:
        """Detect and initialize flight controller"""
        logger.info("Detecting flight controller...")

        # Try Cube Orange+ via MAVLink
        self.cube_orange = await auto_detect_cube_orange()
        if self.cube_orange:
            self.status.flight_controller_connected = True
            self.status.flight_controller_type = "cube_orange"
            self.status.flight_controller_info = await self.cube_orange.get_info()
            logger.info(f"Cube Orange+ detected: {self.status.flight_controller_info}")
            return True

        logger.warning("No flight controller detected")
        self.status.flight_controller_connected = False
        self.status.flight_controller_type = "none"
        return False

    async def detect_all(self) -> HardwareStatus:
        """Detect all hardware components"""
        logger.info("Starting hardware auto-detection...")

        # Detect in parallel for speed
        sdr_task = asyncio.create_task(self.detect_sdr())
        fc_task = asyncio.create_task(self.detect_flight_controller())

        sdr_found = await sdr_task
        fc_found = await fc_task

        self.status.timestamp = asyncio.get_event_loop().time()

        # Log summary
        if sdr_found and fc_found:
            logger.info("All hardware detected successfully")
        elif sdr_found or fc_found:
            logger.warning("Partial hardware detection - running in degraded mode")
        else:
            logger.error("No hardware detected - running in simulation mode")

        return self.status

    async def monitor_connections(self):
        """Monitor hardware connections with exponential backoff retry"""
        while True:
            try:
                # Check SDR connection
                if self.hackrf and not await self._check_sdr_alive():
                    logger.warning("SDR connection lost, attempting reconnect...")
                    if not await self.detect_sdr():
                        self._retry_delay = min(self._retry_delay * 2, self._max_retry_delay)
                    else:
                        self._retry_delay = 1.0  # Reset on success

                # Check flight controller connection
                if self.cube_orange and not await self._check_fc_alive():
                    logger.warning("Flight controller connection lost, attempting reconnect...")
                    if not await self.detect_flight_controller():
                        self._retry_delay = min(self._retry_delay * 2, self._max_retry_delay)
                    else:
                        self._retry_delay = 1.0  # Reset on success

                await asyncio.sleep(self._retry_delay)

            except SDRError as e:
                logger.error(f"Hardware monitor error: {e}")
                await asyncio.sleep(self._retry_delay)

    async def _check_sdr_alive(self) -> bool:
        """Check if SDR is still responsive"""
        if not self.hackrf:
            return False

        try:
            info = await self.hackrf.get_info()
            return info.get("status") == "connected"
        except (AttributeError, KeyError, TimeoutError) as e:
            logger.debug(f"SDR check failed: {e}")
            return False

    async def _check_fc_alive(self) -> bool:
        """Check if flight controller is still responsive"""
        if not self.cube_orange:
            return False

        try:
            info = await self.cube_orange.get_info()
            return info.get("status") == "connected"
        except (AttributeError, KeyError, TimeoutError) as e:
            logger.debug(f"MAVLink check failed: {e}")
            return False

    def get_sdr(self) -> HackRFInterface | None:
        """Get SDR interface if available"""
        return self.hackrf if self.status.sdr_connected else None

    def get_flight_controller(self) -> MAVLinkInterface | None:
        """Get flight controller interface if available"""
        return self.cube_orange if self.status.flight_controller_connected else None

    async def shutdown(self):
        """Clean shutdown of hardware"""
        logger.info("Shutting down hardware connections...")

        if self.hackrf:
            await self.hackrf.close()

        if self.cube_orange:
            await self.cube_orange.close()

        logger.info("Hardware shutdown complete")
