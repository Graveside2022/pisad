"""Integration module for Signal Processor and SDR Service.

This module connects the Signal Processing Service with the SDR Service
to enable real-time RF signal analysis from IQ sample streams.
"""

import asyncio
import contextlib
from collections.abc import AsyncGenerator
from typing import Any

from src.backend.core.exceptions import (
    PISADException,
    SignalProcessingError,
)
from src.backend.models.schemas import RSSIReading, SDRConfig
from src.backend.services.sdr_service import SDRService
from src.backend.services.signal_processor import SignalProcessor
from src.backend.utils.logging import get_logger

logger = get_logger(__name__)


class SignalProcessorIntegration:
    """Integrates Signal Processor with SDR Service for real-time processing."""

    def __init__(
        self, signal_processor: SignalProcessor | None = None, sdr_service: SDRService | None = None
    ):
        """Initialize integration service.

        Args:
            signal_processor: SignalProcessor instance (creates new if None)
            sdr_service: SDRService instance (creates new if None)
        """
        self.signal_processor = signal_processor or SignalProcessor()
        self.sdr_service = sdr_service or SDRService()
        self.process_task: asyncio.Task[None] | None = None
        self.is_running = False

    async def start(self, sdr_config: SDRConfig | None = None) -> None:
        """Start integrated signal processing pipeline.

        Args:
            sdr_config: Optional SDR configuration
        """
        if self.is_running:
            logger.warning("Integration already running")
            return

        try:
            # Initialize SDR service
            await self.sdr_service.initialize(sdr_config)

            # Start signal processor
            await self.signal_processor.start()

            # Start processing task
            self.is_running = True
            self.process_task = asyncio.create_task(self._process_iq_stream())

            logger.info("Signal processing integration started")

        except PISADException as e:
            logger.error(f"Failed to start integration: {e}")
            await self.stop()
            raise

    async def _process_iq_stream(self) -> None:
        """Process IQ samples from SDR stream."""
        consecutive_errors = 0
        max_consecutive_errors = 5

        try:
            async for iq_samples in self.sdr_service.stream_iq():
                if not self.is_running:
                    break

                # Add samples to processing queue
                try:
                    self.signal_processor.iq_queue.put_nowait(iq_samples)
                    consecutive_errors = 0  # Reset error counter on success
                except asyncio.QueueFull:
                    # Drop samples if queue is full
                    logger.warning("Processing queue full, dropping samples")
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(
                            f"Too many consecutive queue full errors ({consecutive_errors}), potential processing bottleneck"
                        )

        except asyncio.CancelledError:
            logger.info("IQ processing task cancelled")
        except SignalProcessingError as e:
            logger.error(f"Error in IQ processing: {e}", exc_info=True)
            self.is_running = False

    async def stop(self) -> None:
        """Stop integrated signal processing pipeline."""
        if not self.is_running:
            return

        self.is_running = False

        # Cancel processing task
        if self.process_task and not self.process_task.done():
            self.process_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.process_task

        # Stop services
        await self.signal_processor.stop()
        await self.sdr_service.shutdown()

        logger.info("Signal processing integration stopped")

    async def get_status(self) -> dict[str, Any]:
        """Get integrated status from both services.

        Returns:
            Combined status dictionary
        """
        sdr_status = self.sdr_service.get_status()
        processor_status = await self.signal_processor.get_status()

        return {
            "integration_running": self.is_running,
            "sdr": {
                "status": sdr_status.status,
                "device": sdr_status.device_name,
                "driver": sdr_status.driver,
                "stream_active": sdr_status.stream_active,
                "samples_per_second": sdr_status.samples_per_second,
                "buffer_overflows": sdr_status.buffer_overflows,
                "temperature": sdr_status.temperature,
            },
            "processor": processor_status,
        }

    def get_rssi_stream(self) -> AsyncGenerator[RSSIReading, None]:
        """Get RSSI reading stream from signal processor.

        Returns:
            AsyncGenerator of RSSIReading objects
        """
        return self.signal_processor.stream_rssi()

    def set_frequency(self, frequency: float) -> None:
        """Set SDR center frequency.

        Args:
            frequency: Center frequency in Hz

        Raises:
            ValueError: If frequency is invalid (negative or zero)
        """
        if frequency <= 0:
            raise ValueError(f"Invalid frequency: {frequency} Hz. Must be positive.")

        self.sdr_service.set_frequency(frequency)
        logger.info(f"Frequency updated to {frequency/1e9:.3f} GHz")

    def get_noise_floor(self) -> float:
        """Get current noise floor estimate.

        Returns:
            Current noise floor in dBm
        """
        return self.signal_processor.get_noise_floor()
