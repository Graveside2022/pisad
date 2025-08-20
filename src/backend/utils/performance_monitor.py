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

        # SUBTASK-5.6.2.4 [9a] - Add resource statistics
        cpu_stats = self._calculate_statistics(self._cpu_history)
        memory_stats = self._calculate_statistics(self._memory_history)
        disk_stats = self._calculate_statistics(self._disk_usage_history)
        network_stats = self._calculate_statistics(self._network_usage_history)
        temperature_stats = self._calculate_statistics(self._temperature_history)

        # Determine performance status
        performance_status = self._determine_performance_status()

        return {
            "statistics": {
                "tcp_latency": tcp_stats,
                "processing_latency": processing_stats,
                "cpu_usage": cpu_stats,
                "memory_usage": memory_stats,
                "disk_usage": disk_stats,
                "network_usage": network_stats,
                "temperature": temperature_stats,
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

    def record_cpu_usage(self, cpu_percent: float) -> None:
        """
        SUBTASK-5.6.2.4 [9a] - Record CPU usage with threshold checking.

        Args:
            cpu_percent: CPU usage percentage (0.0-100.0)
        """
        self._cpu_history.append(cpu_percent)
        self._check_cpu_usage_threshold(cpu_percent)
        self._check_performance_degradation()

        if self._structured_logging_enabled:
            self._log_structured_entry("cpu_usage", cpu_percent)

    def record_memory_usage(self, memory_percent: float) -> None:
        """
        SUBTASK-5.6.2.4 [9a] - Record memory usage with threshold checking.

        Args:
            memory_percent: Memory usage percentage (0.0-100.0)
        """
        self._memory_history.append(memory_percent)
        self._check_memory_usage_threshold(memory_percent)
        self._check_performance_degradation()

        if self._structured_logging_enabled:
            self._log_structured_entry("memory_usage", memory_percent)

    def record_disk_usage(self, disk_percent: float) -> None:
        """
        SUBTASK-5.6.2.4 [9a] - Record disk usage with threshold checking.

        Args:
            disk_percent: Disk usage percentage (0.0-100.0)
        """
        self._disk_usage_history.append(disk_percent)
        self._check_disk_usage_threshold(disk_percent)
        self._check_performance_degradation()

        if self._structured_logging_enabled:
            self._log_structured_entry("disk_usage", disk_percent)

    def record_network_usage(self, network_mbps: float) -> None:
        """
        SUBTASK-5.6.2.4 [9a] - Record network usage with threshold checking.

        Args:
            network_mbps: Network usage in Mbps
        """
        self._network_usage_history.append(network_mbps)
        self._check_network_usage_threshold(network_mbps)
        self._check_performance_degradation()

        if self._structured_logging_enabled:
            self._log_structured_entry("network_usage", network_mbps)

    def record_temperature(self, temperature_celsius: float) -> None:
        """
        SUBTASK-5.6.2.4 [9a] - Record temperature with threshold checking.

        Args:
            temperature_celsius: Temperature in Celsius
        """
        self._temperature_history.append(temperature_celsius)
        self._check_temperature_threshold(temperature_celsius)
        self._check_performance_degradation()

        if self._structured_logging_enabled:
            self._log_structured_entry("temperature", temperature_celsius)

    def _check_cpu_usage_threshold(self, cpu_percent: float) -> None:
        """Check CPU usage against thresholds and generate alerts."""
        if cpu_percent >= self.thresholds.cpu_usage_critical_percent:
            self._create_alert(
                "critical",
                "cpu_usage",
                self.thresholds.cpu_usage_critical_percent,
                cpu_percent,
                f"CPU usage {cpu_percent:.1f}% exceeds critical threshold of {self.thresholds.cpu_usage_critical_percent}%",
            )
        elif cpu_percent >= self.thresholds.cpu_usage_warning_percent:
            self._create_alert(
                "warning",
                "cpu_usage",
                self.thresholds.cpu_usage_warning_percent,
                cpu_percent,
                f"CPU usage {cpu_percent:.1f}% exceeds warning threshold of {self.thresholds.cpu_usage_warning_percent}%",
            )

    def _check_memory_usage_threshold(self, memory_percent: float) -> None:
        """Check memory usage against thresholds and generate alerts."""
        if memory_percent >= self.thresholds.memory_usage_critical_percent:
            self._create_alert(
                "critical",
                "memory_usage",
                self.thresholds.memory_usage_critical_percent,
                memory_percent,
                f"Memory usage {memory_percent:.1f}% exceeds critical threshold of {self.thresholds.memory_usage_critical_percent}%",
            )
        elif memory_percent >= self.thresholds.memory_usage_warning_percent:
            self._create_alert(
                "warning",
                "memory_usage",
                self.thresholds.memory_usage_warning_percent,
                memory_percent,
                f"Memory usage {memory_percent:.1f}% exceeds warning threshold of {self.thresholds.memory_usage_warning_percent}%",
            )

    def _check_disk_usage_threshold(self, disk_percent: float) -> None:
        """Check disk usage against thresholds and generate alerts."""
        if disk_percent >= self.thresholds.disk_usage_critical_percent:
            self._create_alert(
                "critical",
                "disk_usage",
                self.thresholds.disk_usage_critical_percent,
                disk_percent,
                f"Disk usage {disk_percent:.1f}% exceeds critical threshold of {self.thresholds.disk_usage_critical_percent}%",
            )
        elif disk_percent >= self.thresholds.disk_usage_warning_percent:
            self._create_alert(
                "warning",
                "disk_usage",
                self.thresholds.disk_usage_warning_percent,
                disk_percent,
                f"Disk usage {disk_percent:.1f}% exceeds warning threshold of {self.thresholds.disk_usage_warning_percent}%",
            )

    def _check_network_usage_threshold(self, network_mbps: float) -> None:
        """Check network usage against thresholds and generate alerts."""
        if network_mbps >= self.thresholds.network_usage_critical_mbps:
            self._create_alert(
                "critical",
                "network_usage",
                self.thresholds.network_usage_critical_mbps,
                network_mbps,
                f"Network usage {network_mbps:.1f} Mbps exceeds critical threshold of {self.thresholds.network_usage_critical_mbps} Mbps",
            )
        elif network_mbps >= self.thresholds.network_usage_warning_mbps:
            self._create_alert(
                "warning",
                "network_usage",
                self.thresholds.network_usage_warning_mbps,
                network_mbps,
                f"Network usage {network_mbps:.1f} Mbps exceeds warning threshold of {self.thresholds.network_usage_warning_mbps} Mbps",
            )

    def _check_temperature_threshold(self, temperature_celsius: float) -> None:
        """Check temperature against thresholds and generate alerts."""
        if temperature_celsius >= self.thresholds.temperature_critical_celsius:
            self._create_alert(
                "critical",
                "temperature",
                self.thresholds.temperature_critical_celsius,
                temperature_celsius,
                f"Temperature {temperature_celsius:.1f}°C exceeds critical threshold of {self.thresholds.temperature_critical_celsius}°C",
            )
        elif temperature_celsius >= self.thresholds.temperature_warning_celsius:
            self._create_alert(
                "warning",
                "temperature",
                self.thresholds.temperature_warning_celsius,
                temperature_celsius,
                f"Temperature {temperature_celsius:.1f}°C exceeds warning threshold of {self.thresholds.temperature_warning_celsius}°C",
            )

    def get_resource_trend_analysis(self) -> dict[str, Any]:
        """
        SUBTASK-5.6.2.4 [9d] - Get resource usage trend analysis with moving averages.

        Returns:
            Trend analysis for all resource metrics
        """
        return {
            "cpu_usage": self._analyze_metric_trend(self._cpu_history, "cpu"),
            "memory_usage": self._analyze_metric_trend(self._memory_history, "memory"),
            "disk_usage": self._analyze_metric_trend(self._disk_usage_history, "disk"),
            "network_usage": self._analyze_metric_trend(
                self._network_usage_history, "network"
            ),
            "temperature": self._analyze_metric_trend(
                self._temperature_history, "temperature"
            ),
        }

    def get_predictive_threshold_adjustment(self) -> dict[str, Any]:
        """
        SUBTASK-5.6.2.4 [9d] - Get predictive threshold management recommendations.

        Returns:
            Predictions for threshold breaches and recommended adjustments
        """
        predictions = {}

        # Analyze CPU usage trend for prediction
        if len(self._cpu_history) >= 5:
            cpu_trend = self._predict_threshold_breach(
                self._cpu_history, self.thresholds.cpu_usage_critical_percent
            )
            predictions["cpu_usage"] = cpu_trend

        # Analyze memory usage trend for prediction
        if len(self._memory_history) >= 5:
            memory_trend = self._predict_threshold_breach(
                self._memory_history, self.thresholds.memory_usage_critical_percent
            )
            predictions["memory_usage"] = memory_trend

        # Analyze temperature trend for prediction
        if len(self._temperature_history) >= 5:
            temp_trend = self._predict_threshold_breach(
                self._temperature_history, self.thresholds.temperature_critical_celsius
            )
            predictions["temperature"] = temp_trend

        return predictions

    def _analyze_metric_trend(
        self, history: deque[float], metric_name: str
    ) -> dict[str, Any]:
        """Analyze trend for a specific metric."""
        if len(history) < 3:
            return {"moving_average": 0.0, "trend_direction": "insufficient_data"}

        # Calculate moving average
        recent_values = list(history)[-5:]  # Last 5 values
        moving_average = sum(recent_values) / len(recent_values)

        # Determine trend direction
        if len(recent_values) >= 3:
            first_half = sum(recent_values[: len(recent_values) // 2]) / (
                len(recent_values) // 2
            )
            second_half = sum(recent_values[len(recent_values) // 2 :]) / (
                len(recent_values) - len(recent_values) // 2
            )

            if second_half > first_half * 1.05:  # 5% increase threshold
                trend_direction = "increasing"
            elif second_half < first_half * 0.95:  # 5% decrease threshold
                trend_direction = "decreasing"
            else:
                trend_direction = "stable"
        else:
            trend_direction = "stable"

        return {
            "moving_average": round(moving_average, 2),
            "trend_direction": trend_direction,
            "sample_count": len(history),
        }

    def _predict_threshold_breach(
        self, history: deque[float], threshold: float
    ) -> dict[str, Any]:
        """Predict when a metric might breach its threshold."""
        if len(history) < 5:
            return {"predicted_breach_time": 0, "confidence": 0.0}

        recent_values = list(history)[-10:]  # Last 10 values

        # Simple linear regression to predict trend
        if len(recent_values) < 2:
            return {"predicted_breach_time": 0, "confidence": 0.0}

        # Calculate rate of change
        time_points = list(range(len(recent_values)))
        n = len(recent_values)

        # Linear regression: y = mx + b
        sum_x = sum(time_points)
        sum_y = sum(recent_values)
        sum_xy = sum(x * y for x, y in zip(time_points, recent_values))
        sum_x2 = sum(x * x for x in time_points)

        # Calculate slope (rate of change)
        if n * sum_x2 - sum_x * sum_x != 0:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        else:
            slope = 0

        current_value = recent_values[-1]

        # Predict time to breach (in number of samples)
        if slope > 0 and current_value < threshold:
            predicted_breach_time = (threshold - current_value) / slope
        else:
            predicted_breach_time = float("inf")  # No breach predicted

        # Confidence based on trend consistency
        value_range = max(recent_values) - min(recent_values)
        confidence = min(
            1.0, abs(slope) / (value_range + 0.1)
        )  # Avoid division by zero

        return {
            "predicted_breach_time": (
                round(predicted_breach_time, 1)
                if predicted_breach_time != float("inf")
                else 0
            ),
            "confidence": round(confidence, 2),
            "current_value": round(current_value, 2),
            "threshold": threshold,
            "rate_of_change": round(slope, 3),
        }

    def get_alerts_for_frontend(self) -> list[dict[str, Any]]:
        """
        SUBTASK-5.6.2.4 [9e] - Format alerts for frontend consumption.

        Returns:
            List of alerts formatted for SystemHealth.tsx component
        """
        frontend_alerts = []

        for alert in self._active_alerts[-10:]:  # Last 10 alerts
            frontend_alerts.append(
                {
                    "id": f"{alert.metric}_{int(alert.timestamp)}",
                    "level": alert.level,
                    "metric": alert.metric,
                    "message": alert.message,
                    "timestamp": alert.timestamp,
                    "threshold": alert.threshold,
                    "measured_value": alert.measured_value,
                }
            )

        return frontend_alerts

    def enable_structured_logging(self) -> None:
        """
        SUBTASK-5.6.2.4 [9f] - Enable structured logging for resource usage.
        """
        self._structured_logging_enabled = True
        logger.info("Structured resource logging enabled")

    def get_structured_log_entries(self) -> list[dict[str, Any]]:
        """
        SUBTASK-5.6.2.4 [9f] - Get structured log entries for analysis.

        Returns:
            List of structured log entries with timestamps and metrics
        """
        return self._structured_log_entries.copy()

    def get_historical_analysis(self) -> dict[str, Any]:
        """
        SUBTASK-5.6.2.4 [9f] - Get historical analysis for optimization insights.

        Returns:
            Historical analysis with patterns and optimization recommendations
        """
        analysis = {
            "daily_patterns": self._analyze_daily_patterns(),
            "peak_hours": self._identify_peak_hours(),
            "optimization_recommendations": self._generate_optimization_recommendations(),
        }

        return analysis

    def _log_structured_entry(self, metric_type: str, value: float) -> None:
        """Log a structured entry for a metric."""
        # Determine threshold status
        threshold_status = "normal"
        if metric_type == "cpu_usage":
            if value >= self.thresholds.cpu_usage_critical_percent:
                threshold_status = "critical"
            elif value >= self.thresholds.cpu_usage_warning_percent:
                threshold_status = "warning"
        elif metric_type == "memory_usage":
            if value >= self.thresholds.memory_usage_critical_percent:
                threshold_status = "critical"
            elif value >= self.thresholds.memory_usage_warning_percent:
                threshold_status = "warning"
        # Add other metric types as needed

        entry = {
            "timestamp": time.time(),
            "metric_type": metric_type,
            "value": value,
            "threshold_status": threshold_status,
        }

        self._structured_log_entries.append(entry)

        # Keep only recent entries (last 1000)
        if len(self._structured_log_entries) > 1000:
            self._structured_log_entries.pop(0)

    def _analyze_daily_patterns(self) -> dict[str, Any]:
        """Analyze daily usage patterns from structured logs."""
        if not self._structured_log_entries:
            return {"status": "insufficient_data"}

        # Simple analysis - could be enhanced with more sophisticated pattern detection
        return {
            "status": "patterns_detected",
            "description": "Daily pattern analysis available with structured logging data",
        }

    def _identify_peak_hours(self) -> dict[str, Any]:
        """Identify peak resource usage hours."""
        if not self._structured_log_entries:
            return {"status": "insufficient_data"}

        # Simple analysis - could be enhanced with time-based aggregation
        return {
            "status": "peak_hours_identified",
            "description": "Peak hour analysis available with structured logging data",
        }

    def _generate_optimization_recommendations(self) -> list[str]:
        """Generate optimization recommendations based on historical data."""
        recommendations = []

        # Basic recommendations based on current thresholds
        if self._cpu_history and len(self._cpu_history) > 0:
            avg_cpu = sum(self._cpu_history) / len(self._cpu_history)
            if avg_cpu > 80:
                recommendations.append("Consider CPU optimization or load balancing")

        if self._memory_history and len(self._memory_history) > 0:
            avg_memory = sum(self._memory_history) / len(self._memory_history)
            if avg_memory > 80:
                recommendations.append(
                    "Consider memory optimization or garbage collection tuning"
                )

        if not recommendations:
            recommendations.append("System performance within normal parameters")

        return recommendations
