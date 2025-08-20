"""
Performance Monitor for TCP Latency and Resource Optimization

SUBTASK-5.6.1.4 - Adaptive performance monitoring with automatic degradation detection.
Real-time performance metrics collection and threshold-based alerting.

PRD References:
- AC5.6.3: Automatic performance monitoring detects degradation and triggers fallback
- NFR2: Signal processing latency <100ms maintained through monitoring
"""

import asyncio
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

try:
    import psutil

    _psutil_available = True
except ImportError:
    psutil = None
    _psutil_available = False

from src.backend.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceThresholds:
    """Performance threshold configuration for monitoring."""

    tcp_latency_warning_ms: float = 30.0  # Warning at 30ms
    tcp_latency_critical_ms: float = 50.0  # Critical at 50ms (PRD limit)
    processing_latency_warning_ms: float = 75.0  # Warning at 75ms
    processing_latency_critical_ms: float = 100.0  # Critical at 100ms (PRD limit)
    cpu_usage_warning_percent: float = 70.0  # Warning at 70% CPU
    cpu_usage_critical_percent: float = 85.0  # Critical at 85% CPU
    memory_usage_warning_percent: float = 75.0  # Warning at 75% memory
    memory_usage_critical_percent: float = 90.0  # Critical at 90% memory

    # SUBTASK-5.6.2.4 [9b] - Extended resource monitoring thresholds
    disk_usage_warning_percent: float = 80.0  # Warning at 80% disk
    disk_usage_critical_percent: float = 95.0  # Critical at 95% disk
    temperature_warning_celsius: float = 70.0  # Warning at 70°C
    temperature_critical_celsius: float = 85.0  # Critical at 85°C
    network_usage_warning_mbps: float = 50.0  # Warning at 50 Mbps
    network_usage_critical_mbps: float = 80.0  # Critical at 80 Mbps

    def adjust_for_operational_context(self, context: str) -> None:
        """
        SUBTASK-5.6.2.4 [9b] - Adjust thresholds based on operational context.

        Args:
            context: Operational context ('mission_critical', 'normal', 'testing')
        """
        if context == "mission_critical":
            # Lower thresholds for mission critical operations
            self.cpu_usage_critical_percent = 80.0
            self.memory_usage_critical_percent = 85.0
            self.temperature_critical_celsius = 80.0
        elif context == "normal":
            # Standard thresholds for normal operations
            self.cpu_usage_critical_percent = 85.0
            self.memory_usage_critical_percent = 90.0
            self.temperature_critical_celsius = 85.0
        elif context == "testing":
            # Relaxed thresholds for testing
            self.cpu_usage_critical_percent = 95.0
            self.memory_usage_critical_percent = 95.0
            self.temperature_critical_celsius = 90.0


@dataclass
class PerformanceAlert:
    """Performance alert with severity and details."""

    level: str  # "warning" or "critical"
    metric: str  # "tcp_latency", "processing_latency", "cpu_usage", "memory_usage"
    threshold: float
    measured_value: float
    message: str
    timestamp: float = field(default_factory=time.time)


class AdaptivePerformanceMonitor:
    """
    SUBTASK-5.6.1.4 [4a-4f] - Adaptive performance monitoring with degradation detection.

    Provides real-time performance monitoring, threshold-based alerting,
    and automatic performance baseline adjustment.
    """

    def __init__(self, thresholds: PerformanceThresholds | None = None) -> None:
        """Initialize performance monitor with configurable thresholds."""
        self.thresholds = thresholds or PerformanceThresholds()

        # Performance history with configurable retention
        self._tcp_latency_history: deque[float] = deque(maxlen=1000)
        self._processing_latency_history: deque[float] = deque(maxlen=1000)
        self._cpu_history: deque[float] = deque(maxlen=100)
        self._memory_history: deque[float] = deque(maxlen=100)

        # SUBTASK-5.6.2.4 [9a] - Extended resource monitoring history
        self._disk_usage_history: deque[float] = deque(maxlen=100)
        self._temperature_history: deque[float] = deque(maxlen=100)
        self._network_usage_history: deque[float] = deque(maxlen=100)

        # Performance baselines for adaptive thresholds
        self._tcp_baseline_ms: float | None = None
        self._processing_baseline_ms: float | None = None
        self._baseline_sample_count = 0
        self._baseline_established = False

        # Alert management
        self._active_alerts: list[PerformanceAlert] = []
        self._alert_callbacks: list[Callable[[PerformanceAlert], None]] = []

        # Monitoring state
        self._monitoring_active = False
        self._monitoring_task: asyncio.Task[None] | None = None
        if _psutil_available and psutil is not None:
            self._last_network_stats = psutil.net_io_counters()
        else:
            self._last_network_stats = None

        # Performance degradation detection
        self._degradation_detected = False
        self._degradation_start_time: float | None = None
        self._consecutive_violations = 0
        self._violation_threshold = 3  # Trigger after 3 consecutive violations

        # SUBTASK-5.6.2.4 [9f] - Structured logging support
        self._structured_logging_enabled = False
        self._structured_log_entries: list[dict[str, Any]] = []

        logger.info(
            "AdaptivePerformanceMonitor initialized with TCP latency limit %sms",
            self.thresholds.tcp_latency_critical_ms,
        )

    def record_tcp_latency(self, latency_ms: float) -> None:
        """
        SUBTASK-5.6.1.4 [4b] - Record TCP latency with threshold checking.

        Args:
            latency_ms: TCP round-trip latency in milliseconds
        """
        self._tcp_latency_history.append(latency_ms)

        # Update baseline if not established
        if not self._baseline_established:
            self._update_baseline(latency_ms, "tcp")

        # Check thresholds and generate alerts
        self._check_tcp_latency_threshold(latency_ms)

        # Update degradation detection
        self._check_performance_degradation()

    def record_processing_latency(self, latency_ms: float) -> None:
        """
        SUBTASK-5.6.1.4 [4c] - Record processing latency with baseline tracking.

        Args:
            latency_ms: Processing latency in milliseconds
        """
        self._processing_latency_history.append(latency_ms)

        # Update baseline if not established
        if not self._baseline_established:
            self._update_baseline(latency_ms, "processing")

        # Check thresholds
        self._check_processing_latency_threshold(latency_ms)

        # Update degradation detection
        self._check_performance_degradation()

    def is_performance_degraded(self) -> bool:
        """Check if performance degradation has been detected."""
        return self._degradation_detected

    def get_performance_summary(self) -> dict[str, Any]:
        """
        SUBTASK-5.6.1.4 [4f] - Get comprehensive performance summary with trends.

        Returns:
            Performance summary with statistics and trend analysis
        """
        # Calculate statistics
        tcp_stats = self._calculate_statistics(self._tcp_latency_history)
        processing_stats = self._calculate_statistics(self._processing_latency_history)

        # Determine performance status
        performance_status = self._determine_performance_status()

        return {
            "statistics": {
                "tcp_latency": tcp_stats,
                "processing_latency": processing_stats,
            },
            "baselines": {
                "tcp_baseline_ms": self._tcp_baseline_ms,
                "processing_baseline_ms": self._processing_baseline_ms,
                "baseline_established": self._baseline_established,
            },
            "performance_status": performance_status,
            "active_alerts": len(self._active_alerts),
            "degradation_detected": self._degradation_detected,
            "monitoring_active": self._monitoring_active,
            "thresholds": {
                "tcp_critical_ms": self.thresholds.tcp_latency_critical_ms,
                "processing_critical_ms": self.thresholds.processing_latency_critical_ms,
            },
        }

    def get_average_latency(self) -> float | None:
        """Get average TCP latency over recent samples."""
        return self._get_recent_average(self._tcp_latency_history, 10)

    def get_latency_trend(self) -> dict[str, Any]:
        """Get latency trend analysis."""
        if not self._tcp_latency_history:
            return {"trend": "insufficient_data"}

        recent_half = len(self._tcp_latency_history) // 2
        if recent_half == 0:
            return {"trend": "insufficient_data"}

        history_list = list(self._tcp_latency_history)
        older_avg = sum(history_list[:recent_half]) / recent_half
        newer_avg = sum(history_list[recent_half:]) / (len(history_list) - recent_half)

        return {
            "older_avg_ms": older_avg,
            "newer_avg_ms": newer_avg,
            "trend": "improving" if newer_avg < older_avg else "degrading",
        }

    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None:
        """Add callback function for performance alerts."""
        self._alert_callbacks.append(callback)

    def _update_baseline(self, value: float, metric_type: str) -> None:
        """
        SUBTASK-5.6.1.4 [4d] - Update performance baseline with adaptive adjustment.

        Args:
            value: New measurement value
            metric_type: Type of metric ("tcp" or "processing")
        """
        self._baseline_sample_count += 1

        if metric_type == "tcp":
            if self._tcp_baseline_ms is None:
                self._tcp_baseline_ms = value
            else:
                # Exponential moving average with alpha=0.1
                self._tcp_baseline_ms = 0.9 * self._tcp_baseline_ms + 0.1 * value

        elif metric_type == "processing":
            if self._processing_baseline_ms is None:
                self._processing_baseline_ms = value
            else:
                self._processing_baseline_ms = (
                    0.9 * self._processing_baseline_ms + 0.1 * value
                )

        # Establish baseline after sufficient samples
        if self._baseline_sample_count >= 50:
            self._baseline_established = True
            logger.info(
                "Performance baseline established: TCP %s ms, Processing %s ms",
                f"{self._tcp_baseline_ms:.1f}" if self._tcp_baseline_ms else "N/A",
                (
                    f"{self._processing_baseline_ms:.1f}"
                    if self._processing_baseline_ms
                    else "N/A"
                ),
            )

    def _check_tcp_latency_threshold(self, latency_ms: float) -> None:
        """Check TCP latency against thresholds and generate alerts."""
        if latency_ms >= self.thresholds.tcp_latency_critical_ms:
            self._create_alert(
                "critical",
                "tcp_latency",
                self.thresholds.tcp_latency_critical_ms,
                latency_ms,
                f"TCP latency {latency_ms:.1f}ms exceeds critical threshold of {self.thresholds.tcp_latency_critical_ms}ms",
            )
        elif latency_ms >= self.thresholds.tcp_latency_warning_ms:
            self._create_alert(
                "warning",
                "tcp_latency",
                self.thresholds.tcp_latency_warning_ms,
                latency_ms,
                f"TCP latency {latency_ms:.1f}ms exceeds warning threshold of {self.thresholds.tcp_latency_warning_ms}ms",
            )

    def _check_processing_latency_threshold(self, latency_ms: float) -> None:
        """Check processing latency against thresholds."""
        if latency_ms >= self.thresholds.processing_latency_critical_ms:
            self._create_alert(
                "critical",
                "processing_latency",
                self.thresholds.processing_latency_critical_ms,
                latency_ms,
                f"Processing latency {latency_ms:.1f}ms exceeds critical threshold of {self.thresholds.processing_latency_critical_ms}ms",
            )
        elif latency_ms >= self.thresholds.processing_latency_warning_ms:
            self._create_alert(
                "warning",
                "processing_latency",
                self.thresholds.processing_latency_warning_ms,
                latency_ms,
                f"Processing latency {latency_ms:.1f}ms exceeds warning threshold of {self.thresholds.processing_latency_warning_ms}ms",
            )

    def _create_alert(
        self,
        level: str,
        metric: str,
        threshold: float,
        measured_value: float,
        message: str,
    ) -> None:
        """Create and dispatch performance alert."""
        alert = PerformanceAlert(
            level=level,
            metric=metric,
            threshold=threshold,
            measured_value=measured_value,
            message=message,
        )

        self._active_alerts.append(alert)

        # Keep only recent alerts (last 100)
        if len(self._active_alerts) > 100:
            self._active_alerts.pop(0)

        # Dispatch to callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error("Error in alert callback: %s", e)

        # Log the alert
        log_method = logger.critical if level == "critical" else logger.warning
        log_method("Performance alert: %s", message)

    def _check_performance_degradation(self) -> None:
        """
        SUBTASK-5.6.1.4 [4c] - Check for performance degradation patterns.

        Detects sustained performance issues across multiple metrics.
        """
        # Get recent averages for key metrics
        recent_tcp = self._get_recent_average(self._tcp_latency_history, 10)
        recent_processing = self._get_recent_average(
            self._processing_latency_history, 10
        )

        # Check if any critical thresholds are being violated
        violations = 0
        if recent_tcp and recent_tcp >= self.thresholds.tcp_latency_critical_ms:
            violations += 1
        if (
            recent_processing
            and recent_processing >= self.thresholds.processing_latency_critical_ms
        ):
            violations += 1

        if violations > 0:
            self._consecutive_violations += 1
            if self._consecutive_violations >= self._violation_threshold:
                if not self._degradation_detected:
                    self._degradation_detected = True
                    self._degradation_start_time = time.time()
                    logger.critical(
                        "Performance degradation detected - %d consecutive violations",
                        self._consecutive_violations,
                    )
        else:
            # Reset violation counter if no violations
            self._consecutive_violations = 0
            if self._degradation_detected:
                # Check if degradation has been resolved
                degradation_duration = time.time() - (self._degradation_start_time or 0)
                if degradation_duration > 10:  # 10 second recovery period
                    self._degradation_detected = False
                    self._degradation_start_time = None
                    logger.info(
                        "Performance degradation resolved after %s seconds",
                        f"{degradation_duration:.1f}",
                    )

    def _get_recent_average(self, history: deque[float], samples: int) -> float | None:
        """Get average of recent samples from history."""
        if not history:
            return None
        recent_samples = list(history)[-samples:]
        return sum(recent_samples) / len(recent_samples) if recent_samples else None

    def _calculate_statistics(self, history: deque[float]) -> dict[str, float]:
        """Calculate statistical summary of performance history."""
        if not history:
            return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}

        values = list(history)
        count = len(values)
        mean = sum(values) / count
        min_val = min(values)
        max_val = max(values)

        # Calculate standard deviation
        if count > 1:
            variance = sum((x - mean) ** 2 for x in values) / count
            std_dev = variance**0.5
        else:
            std_dev = 0.0

        return {
            "count": count,
            "mean": round(mean, 2),
            "min": round(min_val, 2),
            "max": round(max_val, 2),
            "std": round(std_dev, 2),
        }

    def _determine_performance_status(self) -> str:
        """Determine overall performance status."""
        if self._degradation_detected:
            return "degraded"

        recent_alerts = self._active_alerts[-10:] if self._active_alerts else []
        critical_alerts = [a for a in recent_alerts if a.level == "critical"]
        if critical_alerts:
            return "critical"

        warning_alerts = [a for a in recent_alerts if a.level == "warning"]
        if warning_alerts:
            return "warning"

        return "optimal"
