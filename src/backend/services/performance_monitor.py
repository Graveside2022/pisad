"""
Real-Time Performance Monitoring Service
Tracks CPU, RAM, SDR performance, and MAVLink latency
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field

import psutil

from src.backend.core.exceptions import (
    PISADException,
)

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """System performance metrics"""

    # System resources
    cpu_percent: float = 0.0
    ram_mb: float = 0.0
    ram_percent: float = 0.0

    # SDR metrics
    sdr_sample_rate: float = 0.0
    sdr_samples_dropped: int = 0
    sdr_buffer_usage: float = 0.0

    # MAVLink metrics
    mavlink_latency_ms: float = 0.0
    mavlink_packet_loss: float = 0.0
    mavlink_messages_per_sec: float = 0.0

    # Processing metrics
    rssi_update_rate: float = 0.0
    state_machine_rate: float = 0.0
    websocket_clients: int = 0

    # Timing
    timestamp: float = field(default_factory=time.time)


class PerformanceMonitor:
    """Monitor system and application performance"""

    def __init__(self) -> None:
        self.metrics = PerformanceMetrics()
        self._running = False
        self._monitor_task: asyncio.Task | None = None
        self._sdr_samples_total = 0
        self._sdr_samples_dropped = 0
        self._mavlink_messages = 0
        self._mavlink_last_msg_time = 0.0
        self._rssi_updates = 0
        self._state_updates = 0
        self._update_interval = 1.0  # Update every second

    async def start(self) -> None:
        """Start performance monitoring"""
        self._running = True
        logger.info("Performance monitoring started")

        # Start monitoring task
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop(self) -> None:
        """Stop performance monitoring"""
        self._running = False
        logger.info("Performance monitoring stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self._running:
            try:
                # Collect metrics
                await self._update_system_metrics()
                await self._calculate_rates()

                # Log if metrics exceed thresholds
                self._check_thresholds()

                await asyncio.sleep(self._update_interval)

            except PISADException as e:
                logger.error(f"Performance monitor error: {e}")
                await asyncio.sleep(self._update_interval)

    async def _update_system_metrics(self) -> None:
        """Update system resource metrics"""
        # CPU usage (target < 30% for Pi 5)
        self.metrics.cpu_percent = psutil.cpu_percent(interval=0.1)

        # Memory usage
        mem = psutil.virtual_memory()
        self.metrics.ram_mb = mem.used / (1024 * 1024)
        self.metrics.ram_percent = mem.percent

        # Update timestamp
        self.metrics.timestamp = time.time()

    async def _calculate_rates(self) -> None:
        """Calculate update rates"""
        # RSSI update rate (target 1 Hz)
        if self._rssi_updates > 0:
            self.metrics.rssi_update_rate = self._rssi_updates / self._update_interval
            self._rssi_updates = 0

        # State machine rate (target 10 Hz)
        if self._state_updates > 0:
            self.metrics.state_machine_rate = self._state_updates / self._update_interval
            self._state_updates = 0

        # MAVLink messages per second
        if self._mavlink_messages > 0:
            self.metrics.mavlink_messages_per_sec = self._mavlink_messages / self._update_interval
            self._mavlink_messages = 0

    def _check_thresholds(self) -> None:
        """Check and log threshold violations"""
        # CPU threshold (30% for Pi 5)
        if self.metrics.cpu_percent > 30:
            logger.warning(f"High CPU usage: {self.metrics.cpu_percent:.1f}%")

        # RAM threshold (500 MB)
        if self.metrics.ram_mb > 500:
            logger.warning(f"High RAM usage: {self.metrics.ram_mb:.1f} MB")

        # SDR sample drops
        if self.metrics.sdr_samples_dropped > 0:
            logger.warning(f"SDR samples dropped: {self.metrics.sdr_samples_dropped}")

        # MAVLink latency (50ms threshold)
        if self.metrics.mavlink_latency_ms > 50:
            logger.warning(f"High MAVLink latency: {self.metrics.mavlink_latency_ms:.1f} ms")

        # RSSI rate (should be ~1 Hz)
        if self.metrics.rssi_update_rate > 0 and abs(self.metrics.rssi_update_rate - 1.0) > 0.2:
            logger.warning(f"RSSI rate off target: {self.metrics.rssi_update_rate:.2f} Hz")

    # Public methods for components to report metrics

    def record_sdr_sample(self, num_samples: int, dropped: int = 0) -> None:
        """Record SDR sample processing"""
        self._sdr_samples_total += num_samples
        self._sdr_samples_dropped += dropped
        self.metrics.sdr_samples_dropped = self._sdr_samples_dropped

        # Calculate sample rate
        if self._sdr_samples_total > 0:
            self.metrics.sdr_sample_rate = self._sdr_samples_total / self._update_interval

    def record_sdr_buffer_usage(self, usage_percent: float) -> None:
        """Record SDR buffer usage percentage"""
        self.metrics.sdr_buffer_usage = usage_percent

    def record_mavlink_message(self, latency_ms: float = 0) -> None:
        """Record MAVLink message received"""
        self._mavlink_messages += 1

        if latency_ms > 0:
            # Moving average of latency
            alpha = 0.1
            self.metrics.mavlink_latency_ms = (
                alpha * latency_ms + (1 - alpha) * self.metrics.mavlink_latency_ms
            )

        self._mavlink_last_msg_time = time.time()

    def record_mavlink_packet_loss(self, loss_percent: float) -> None:
        """Record MAVLink packet loss percentage"""
        self.metrics.mavlink_packet_loss = loss_percent

    def record_rssi_update(self) -> None:
        """Record RSSI processing update"""
        self._rssi_updates += 1

    def record_state_update(self) -> None:
        """Record state machine update"""
        self._state_updates += 1

    def record_websocket_clients(self, count: int) -> None:
        """Record number of WebSocket clients"""
        self.metrics.websocket_clients = count

    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        return self.metrics

    def get_metrics_dict(self) -> dict:
        """Get metrics as dictionary for WebSocket/API"""
        return {
            "cpu_percent": round(self.metrics.cpu_percent, 1),
            "ram_mb": round(self.metrics.ram_mb, 1),
            "ram_percent": round(self.metrics.ram_percent, 1),
            "sdr_sample_rate": round(self.metrics.sdr_sample_rate, 0),
            "sdr_samples_dropped": self.metrics.sdr_samples_dropped,
            "sdr_buffer_usage": round(self.metrics.sdr_buffer_usage, 1),
            "mavlink_latency_ms": round(self.metrics.mavlink_latency_ms, 1),
            "mavlink_packet_loss": round(self.metrics.mavlink_packet_loss, 1),
            "mavlink_messages_per_sec": round(self.metrics.mavlink_messages_per_sec, 1),
            "rssi_update_rate": round(self.metrics.rssi_update_rate, 2),
            "state_machine_rate": round(self.metrics.state_machine_rate, 1),
            "websocket_clients": self.metrics.websocket_clients,
            "timestamp": self.metrics.timestamp,
        }

    def get_summary(self) -> str:
        """Get performance summary string"""
        return (
            f"CPU: {self.metrics.cpu_percent:.1f}% | "
            f"RAM: {self.metrics.ram_mb:.0f}MB | "
            f"SDR: {self.metrics.sdr_sample_rate:.0f} sps | "
            f"MAVLink: {self.metrics.mavlink_latency_ms:.1f}ms | "
            f"RSSI: {self.metrics.rssi_update_rate:.1f}Hz"
        )
