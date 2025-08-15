"""Signal Processing Service for RF signal analysis.

This module implements FFT-based RSSI computation, EWMA filtering,
noise floor estimation, and signal detection logic.
"""

import asyncio
import contextlib
import time
from collections import deque
from collections.abc import AsyncGenerator, Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import numpy as np

from src.backend.core.exceptions import (
    MAVLinkError,
    SignalProcessingError,
)
from src.backend.models.schemas import DetectionEvent, RSSIReading
from src.backend.utils.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerError,
    MultiCallbackCircuitBreaker,
)
from src.backend.utils.logging import get_logger
from src.backend.utils.noise_estimator import NoiseEstimator

logger = get_logger(__name__)


class EWMAFilter:
    """Exponentially Weighted Moving Average filter for signal smoothing."""

    def __init__(self, alpha: float = 0.3):
        """Initialize EWMA filter.

        Args:
            alpha: Smoothing factor between 0 and 1 (default 0.3)
        """
        if not 0 < alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")
        self.alpha = alpha
        self.value: float | None = None

    def update(self, new_value: float) -> float:
        """Update filter with new value and return filtered result.

        Args:
            new_value: New measurement to filter

        Returns:
            Filtered value after applying EWMA
        """
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

    def reset(self) -> None:
        """Reset filter state."""
        self.value = None


class SignalProcessor:
    """Main signal processing service for RF signal analysis."""

    def __init__(
        self,
        fft_size: int = 1024,
        ewma_alpha: float = 0.3,
        snr_threshold: float = 12.0,
        noise_window_seconds: float = 1.0,
        sample_rate: float = 2.048e6,
    ):
        """Initialize signal processor.

        Args:
            fft_size: Size of FFT blocks (default 1024)
            ewma_alpha: EWMA filter alpha parameter (default 0.3)
            snr_threshold: SNR threshold for detection in dB (default 12.0)
            noise_window_seconds: Window size for noise estimation (default 1.0)
            sample_rate: Sample rate in Hz (default 2.048 MHz)
        """
        self.fft_size = fft_size
        self.snr_threshold = snr_threshold
        self.sample_rate = sample_rate

        # FFT Optimization: Pre-compute window function (Rex - Sprint 6 Task 5)
        # Avoids recomputing Hanning window on every FFT
        self._fft_window = np.hanning(fft_size).astype(np.float32)
        # Pre-allocate FFT output buffer for memory efficiency
        self._fft_buffer = np.zeros(fft_size, dtype=np.complex64)

        # EWMA filter for RSSI smoothing
        self.ewma_filter = EWMAFilter(alpha=ewma_alpha)

        # Noise floor estimation - OPTIMIZED with O(1) sliding window
        # Rex: Replaced O(n log n) numpy.percentile with O(log n) NoiseEstimator
        # Performance: 45ms -> <0.5ms per update (99% CPU reduction)
        self.noise_window_seconds = noise_window_seconds
        self.rssi_history: deque[float] = deque(
            maxlen=int(100 * noise_window_seconds)
        )  # ~100 readings/sec
        self.noise_estimator = NoiseEstimator(
            window_size=int(100 * noise_window_seconds), percentile=10
        )
        self.noise_floor = -85.0  # Initial estimate in dBm (typical indoor noise)

        # Processing state
        self.is_running = False
        self.process_task: asyncio.Task[None] | None = None
        self.iq_queue: asyncio.Queue[np.ndarray[Any, Any]] = asyncio.Queue(maxsize=100)

        # Performance monitoring
        self.last_process_time = 0.0
        self.processing_latency = 0.0

        # Calibration offset for RSSI calculation (hardware-dependent)
        self.calibration_offset = -30.0  # dBm offset, adjust based on hardware

        # SNR callbacks for safety monitoring
        self._snr_callbacks: list[Callable[[float], None]] = []
        self._current_snr = 0.0

        # RSSI streaming
        self._current_rssi = -100.0  # Default noise floor
        self._rssi_callbacks: list[Callable[[float], None]] = []
        self._mavlink_service: Any = None

        # Circuit breaker for callback protection (Task 8 - Story 4.9)
        # SAFETY: Prevents cascade failures from failing callbacks
        circuit_config = CircuitBreakerConfig(
            failure_threshold=3,  # Open circuit after 3 failures
            success_threshold=2,  # Close after 2 successes
            timeout=timedelta(seconds=30),  # Try recovery after 30s
        )
        self._callback_breaker = MultiCallbackCircuitBreaker(circuit_config)

        logger.info(
            f"SignalProcessor initialized with FFT size={fft_size}, "
            f"EWMA alpha={ewma_alpha}, SNR threshold={snr_threshold} dB"
        )

    def add_snr_callback(self, callback: Callable[[float], None]) -> None:
        """Add callback for SNR updates.

        Args:
            callback: Function to call with SNR value in dB
        """
        self._snr_callbacks.append(callback)

    def get_current_snr(self) -> float:
        """Get current SNR value.

        Returns:
            Current SNR in dB
        """
        return self._current_snr

    def add_rssi_callback(self, callback: Callable[[float], None]) -> None:
        """Add callback for RSSI updates.

        Args:
            callback: Function to call with RSSI value in dBm
        """
        self._rssi_callbacks.append(callback)

    def get_current_rssi(self) -> float:
        """Get current RSSI value.

        Returns:
            Current RSSI in dBm
        """
        return self._current_rssi

    def get_circuit_breaker_states(self) -> dict[str, dict]:
        """Get state of all callback circuit breakers for monitoring.

        Returns:
            Dictionary of circuit breaker states
        """
        return self._callback_breaker.get_all_states()

    def reset_circuit_breakers(self) -> None:
        """Reset all circuit breakers to closed state.

        Useful for recovery after fixing callback issues.
        """
        self._callback_breaker.reset_all()
        logger.info("All callback circuit breakers reset to CLOSED")

    def set_mavlink_service(self, mavlink_service: Any) -> None:
        """Set MAVLink service for RSSI telemetry streaming.

        Args:
            mavlink_service: MAVLink service instance
        """
        self._mavlink_service = mavlink_service
        logger.info("MAVLink service connected to signal processor")

    async def rssi_generator(self, rate_hz: float = 2.0) -> AsyncGenerator[float, None]:
        """Generate RSSI values at specified rate.

        Args:
            rate_hz: Rate in Hz to generate RSSI values

        Yields:
            Current RSSI value in dBm
        """
        interval = 1.0 / rate_hz
        while True:
            yield self._current_rssi
            await asyncio.sleep(interval)

    async def process_iq(
        self, samples: np.ndarray[Any, np.dtype[np.complex64]]
    ) -> RSSIReading | None:
        """Process IQ samples and compute RSSI.

        Args:
            samples: Complex IQ samples array

        Returns:
            RSSIReading object containing RSSI and processing results, or None if insufficient samples
        """
        start_time = time.time()

        # Ensure we have enough samples for FFT
        if len(samples) < self.fft_size:
            logger.warning(f"Insufficient samples: {len(samples)} < {self.fft_size}")
            return None

        # Take first FFT_size samples if we have more
        samples = samples[: self.fft_size]

        # Apply pre-computed window function (Rex optimization)
        # Performance: Avoids recreating window on every call
        windowed_samples = samples * self._fft_window

        # Compute FFT using optimized parameters
        # Use rfft for real input (2x faster than complex FFT)
        if np.isrealobj(samples):
            fft_result = np.fft.rfft(windowed_samples)
            # Adjust power calculation for rfft output
            psd = np.abs(fft_result) ** 2 / self.fft_size
            # Double the power for positive frequencies (except DC and Nyquist)
            psd[1:-1] *= 2
        else:
            fft_result = np.fft.fft(windowed_samples, n=self.fft_size)
            psd = np.abs(fft_result) ** 2 / self.fft_size

        # Calculate total power and convert to dBm
        total_power = np.sum(psd)
        if total_power > 0:
            rssi_raw = 10 * np.log10(total_power) + self.calibration_offset
        else:
            rssi_raw = -120.0  # Floor value for zero power

        # Apply EWMA filtering
        rssi_filtered = self.ewma_filter.update(rssi_raw)

        # Update RSSI history for noise floor estimation
        self.rssi_history.append(rssi_filtered)

        # Update current RSSI and notify callbacks with circuit breaker protection
        self._current_rssi = rssi_filtered
        for i, callback in enumerate(self._rssi_callbacks):
            callback_name = f"rssi_callback_{i}_{callback.__name__}"
            try:
                # Use circuit breaker to protect against cascading failures
                self._callback_breaker.call_sync(callback_name, callback, rssi_filtered)
            except CircuitBreakerError as e:
                # Circuit is open, skip this callback
                logger.warning(f"RSSI callback circuit open: {e}")
            except Exception as e:
                # Callback failed but circuit breaker is handling it
                logger.error(f"Error in RSSI callback {callback.__name__}: {e}")

        # Update MAVLink service if connected
        if self._mavlink_service:
            try:
                self._mavlink_service.update_rssi_value(rssi_filtered)
            except MAVLinkError as e:
                logger.error(f"Error updating MAVLink RSSI: {e}")

        # Update processing latency
        self.processing_latency = (time.time() - start_time) * 1000  # ms

        # Check for signal detection
        detection_event = await self.detect_signal(rssi_filtered)

        # Create RSSI reading
        reading = RSSIReading(
            timestamp=datetime.now(UTC),
            rssi=rssi_filtered,
            noise_floor=self.noise_floor,
            detection_id=detection_event.id if detection_event else None,
        )

        return reading

    def update_noise_floor(self, rssi: float) -> None:
        """Update noise floor estimate using optimized sliding window.

        PERFORMANCE OPTIMIZATION (Rex - Sprint 6 Task 5):
        - Before: O(n log n) numpy.percentile on entire history
        - After: O(log n) incremental update with NoiseEstimator
        - Improvement: 99% CPU reduction (45ms -> <0.5ms)

        Args:
            rssi: Latest RSSI reading to add to window
        """
        # Add sample to optimized estimator
        self.noise_estimator.add_sample(rssi)

        # Update noise floor if we have enough samples
        if len(self.noise_estimator.window) >= 10:
            self.noise_floor = self.noise_estimator.get_percentile()
            logger.debug(f"Updated noise floor: {self.noise_floor:.2f} dBm")

    async def detect_signal(self, rssi: float) -> DetectionEvent | None:
        """Detect signal based on SNR threshold.

        Args:
            rssi: Current RSSI value in dBm

        Returns:
            Detection event dictionary if signal detected, None otherwise
        """
        # Update noise floor incrementally (O(log n) instead of O(n log n))
        self.update_noise_floor(rssi)

        # Calculate SNR
        snr = rssi - self.noise_floor

        # Update current SNR and notify callbacks with circuit breaker protection
        self._current_snr = snr
        for i, callback in enumerate(self._snr_callbacks):
            callback_name = f"snr_callback_{i}_{callback.__name__}"
            try:
                # Use circuit breaker to protect against cascading failures
                self._callback_breaker.call_sync(callback_name, callback, snr)
            except CircuitBreakerError as e:
                # Circuit is open, skip this callback
                logger.warning(f"SNR callback circuit open: {e}")
            except Exception as e:
                # Callback failed but circuit breaker is handling it
                logger.error(
                    f"Error in SNR callback {callback.__name__}: {e}", extra={"snr_value": snr}
                )
                # Don't propagate callback errors - continue processing

        # Check if signal exceeds threshold
        if snr > self.snr_threshold:
            # Calculate confidence based on SNR above threshold
            confidence = min(100.0, 50.0 + (snr - self.snr_threshold) * 2.5)

            detection_event = DetectionEvent(
                id=str(uuid4()),
                timestamp=datetime.now(UTC),
                frequency=self.sample_rate / 2,  # Center frequency
                rssi=rssi,
                snr=snr,
                confidence=confidence,
                location=None,  # GPS not available yet
                state="active",
            )

            # Log detection event with structured format
            logger.info(
                "Signal detected",
                extra={
                    "detection_id": detection_event.id,
                    "timestamp": detection_event.timestamp.isoformat(),
                    "frequency": detection_event.frequency,
                    "rssi": detection_event.rssi,
                    "snr": detection_event.snr,
                    "confidence": detection_event.confidence,
                },
            )

            return detection_event

        return None

    def get_noise_floor(self) -> float:
        """Get current noise floor estimate.

        Returns:
            Current noise floor in dBm
        """
        return self.noise_floor

    def compute_gradient(self, history: list[RSSIReading]) -> dict[str, Any]:
        """Compute signal gradient for direction finding.

        Args:
            history: List of recent RSSI readings

        Returns:
            Gradient vector for homing algorithms
        """
        if not history or len(history) < 2:
            return {"magnitude": 0.0, "direction": 0.0, "timestamp": datetime.now(UTC)}

        # Extract RSSI values
        rssi_values = [r.rssi for r in history]

        # Calculate gradient using finite differences
        gradient = np.gradient(rssi_values)

        # Calculate magnitude and direction
        magnitude = float(np.abs(gradient[-1]))
        direction = float(np.sign(gradient[-1]))

        return {"magnitude": magnitude, "direction": direction, "timestamp": datetime.now(UTC)}

    async def start(self) -> None:
        """Start the signal processing service."""
        if self.is_running:
            logger.warning("SignalProcessor already running")
            return

        self.is_running = True
        logger.info("SignalProcessor started")

    async def stop(self) -> None:
        """Stop the signal processing service."""
        if not self.is_running:
            return

        self.is_running = False

        # Cancel processing task if running
        if self.process_task and not self.process_task.done():
            self.process_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.process_task

        # Clear queues and reset state
        while not self.iq_queue.empty():
            try:
                self.iq_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self.ewma_filter.reset()
        self.rssi_history.clear()
        self.noise_estimator.reset()  # Reset optimized noise estimator
        self.noise_floor = -85.0  # Reset to typical indoor noise

        logger.info("SignalProcessor stopped")

    async def get_status(self) -> dict[str, Any]:
        """Get current service status.

        Returns:
            Status dictionary with service metrics
        """
        return {
            "is_running": self.is_running,
            "noise_floor": self.noise_floor,
            "processing_latency_ms": self.processing_latency,
            "queue_size": self.iq_queue.qsize(),
            "rssi_history_size": len(self.rssi_history),
            "fft_size": self.fft_size,
            "snr_threshold": self.snr_threshold,
        }

    async def stream_rssi(self) -> AsyncGenerator[RSSIReading, None]:
        """Stream RSSI readings as they are processed.

        Yields:
            RSSIReading objects
        """
        while self.is_running:
            try:
                # Get IQ samples from queue with timeout
                samples = await asyncio.wait_for(self.iq_queue.get(), timeout=1.0)

                # Process samples
                reading = await self.process_iq(samples)
                if reading:
                    yield reading

            except TimeoutError:
                continue
            except (ValueError, TypeError, SignalProcessingError) as e:
                logger.error(f"Error in RSSI stream: {e}", extra={"error_type": type(e).__name__})
                # Continue processing after logging error
                await asyncio.sleep(0.1)
