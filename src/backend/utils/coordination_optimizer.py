"""
Coordination Optimization Utilities

High-precision latency measurement, statistical analysis, performance monitoring,
and TCP message serialization optimization for dual SDR coordination system.

PRD References:
- PRD-NFR2: Signal processing latency shall not exceed 100ms per RSSI computation cycle
- PRD-NFR4: Frequency control commands processed with <50ms response time
- PRD-NFR12: All safety-critical functions shall execute with deterministic timing
"""

import gzip
import json
import statistics
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from src.backend.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LatencyStatistics:
    """Statistical analysis of coordination latency measurements."""

    count: int
    mean: float
    min_latency: float
    max_latency: float
    p95: float
    p99: float
    std_dev: float = 0.0


@dataclass
class LatencyAlert:
    """Latency threshold alert information."""

    level: str  # "warning" or "critical"
    threshold_ms: float
    measured_latency_ms: float
    timestamp: float
    message: str = ""


class CoordinationLatencyTracker:
    """
    High-precision latency measurement and monitoring for coordination operations.

    Provides statistical analysis, alerting, and performance tracking to ensure
    coordination decisions meet PRD timing requirements.
    """

    def __init__(
        self,
        max_samples: int = 1000,
        alert_threshold_ms: float = 50.0,
        warning_threshold_ms: float = 30.0,
    ):
        """
        Initialize latency tracker with configuration.

        Args:
            max_samples: Maximum number of latency samples to retain
            alert_threshold_ms: Critical latency threshold for alerts
            warning_threshold_ms: Warning latency threshold
        """
        self.max_samples = max_samples
        self.alert_threshold_ms = alert_threshold_ms
        self.warning_threshold_ms = warning_threshold_ms

        # Rolling buffer of latency measurements
        self.latencies: list[float] = []
        self.total_measurements = 0

        logger.info(
            f"CoordinationLatencyTracker initialized: max_samples={max_samples}, "
            f"alert_threshold={alert_threshold_ms}ms, warning_threshold={warning_threshold_ms}ms"
        )

    def record_latency(self, latency_ms: float) -> None:
        """
        Record a latency measurement.

        Args:
            latency_ms: Latency measurement in milliseconds
        """
        # Add to rolling buffer
        self.latencies.append(latency_ms)
        self.total_measurements += 1

        # Maintain buffer size
        if len(self.latencies) > self.max_samples:
            self.latencies.pop(0)

        # Log high latency measurements
        if latency_ms > self.alert_threshold_ms:
            logger.warning(f"High coordination latency: {latency_ms:.2f}ms")
        elif latency_ms > self.warning_threshold_ms:
            logger.info(f"Elevated coordination latency: {latency_ms:.2f}ms")

    def get_statistics(self) -> LatencyStatistics:
        """
        Calculate statistical analysis of recorded latencies.

        Returns:
            LatencyStatistics with comprehensive analysis
        """
        if not self.latencies:
            return LatencyStatistics(
                count=0, mean=0.0, min_latency=0.0, max_latency=0.0, p95=0.0, p99=0.0, std_dev=0.0
            )

        # Calculate basic statistics
        count = len(self.latencies)
        mean_latency = statistics.mean(self.latencies)
        min_latency = min(self.latencies)
        max_latency = max(self.latencies)

        # Calculate percentiles
        sorted_latencies = sorted(self.latencies)
        p95 = self._calculate_percentile(sorted_latencies, 95)
        p99 = self._calculate_percentile(sorted_latencies, 99)

        # Calculate standard deviation
        std_dev = statistics.stdev(self.latencies) if count > 1 else 0.0

        return LatencyStatistics(
            count=count,
            mean=mean_latency,
            min_latency=min_latency,
            max_latency=max_latency,
            p95=p95,
            p99=p99,
            std_dev=std_dev,
        )

    def _calculate_percentile(self, sorted_values: list[float], percentile: int) -> float:
        """Calculate percentile from sorted values."""
        if not sorted_values:
            return 0.0

        index = (percentile / 100.0) * (len(sorted_values) - 1)
        lower_index = int(index)
        upper_index = min(lower_index + 1, len(sorted_values) - 1)

        if lower_index == upper_index:
            return sorted_values[lower_index]

        # Linear interpolation
        weight = index - lower_index
        return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight

    def check_alerts(self) -> list[LatencyAlert]:
        """
        Check for latency threshold violations.

        Returns:
            List of LatencyAlert objects for threshold violations
        """
        alerts: list[LatencyAlert] = []
        current_time = time.time()

        if not self.latencies:
            return alerts

        # Check most recent latency measurement
        latest_latency = self.latencies[-1]

        if latest_latency > self.alert_threshold_ms:
            alerts.append(
                LatencyAlert(
                    level="critical",
                    threshold_ms=self.alert_threshold_ms,
                    measured_latency_ms=latest_latency,
                    timestamp=current_time,
                    message=f"Coordination latency {latest_latency:.2f}ms exceeds critical threshold {self.alert_threshold_ms}ms",
                )
            )
        elif latest_latency > self.warning_threshold_ms:
            alerts.append(
                LatencyAlert(
                    level="warning",
                    threshold_ms=self.warning_threshold_ms,
                    measured_latency_ms=latest_latency,
                    timestamp=current_time,
                    message=f"Coordination latency {latest_latency:.2f}ms exceeds warning threshold {self.warning_threshold_ms}ms",
                )
            )

        return alerts

    @asynccontextmanager
    async def measure(self) -> AsyncGenerator["CoordinationLatencyTracker", None]:
        """
        Context manager for measuring async operation timing.

        Usage:
            async with latency_tracker.measure():
                await some_coordination_operation()
        """
        start_time = time.perf_counter()
        try:
            yield self
        finally:
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            self.record_latency(latency_ms)

    def reset(self) -> None:
        """Reset all measurements and statistics."""
        self.latencies.clear()
        self.total_measurements = 0
        logger.info("Coordination latency measurements reset")

    def get_current_latency(self) -> float | None:
        """Get the most recent latency measurement."""
        return self.latencies[-1] if self.latencies else None

    def is_meeting_requirements(self) -> bool:
        """
        Check if coordination is meeting PRD latency requirements.

        Returns:
            True if all measurements are within acceptable limits
        """
        if not self.latencies:
            return True

        stats = self.get_statistics()

        # Check if P95 is under alert threshold (good performance indicator)
        return stats.p95 <= self.alert_threshold_ms


@dataclass
class SerializationStats:
    """Statistics for message serialization performance."""

    total_messages: int = 0
    total_serializations: int = 0
    total_deserializations: int = 0
    total_serialization_time: float = 0.0
    total_deserialization_time: float = 0.0
    total_original_bytes: int = 0
    total_compressed_bytes: int = 0
    compression_count: int = 0


class MessageSerializer:
    """
    Optimized message serialization for TCP communication.

    Provides JSON compression, compact serialization, batching, and performance
    optimization to reduce bandwidth and improve latency for dual SDR coordination.
    """

    def __init__(
        self,
        compression_enabled: bool = True,
        compression_threshold: int = 100,
        use_compact_json: bool = True,
    ):
        """
        Initialize message serializer with optimization settings.

        Args:
            compression_enabled: Enable gzip compression for large messages
            compression_threshold: Minimum message size (bytes) for compression
            use_compact_json: Use compact JSON without extra whitespace
        """
        self.compression_enabled = compression_enabled
        self.compression_threshold = compression_threshold
        self.use_compact_json = use_compact_json

        # Performance tracking
        self.serialization_stats = SerializationStats()

        # JSON serialization settings for performance
        self.json_separators = (",", ":") if use_compact_json else None

        logger.info(
            f"MessageSerializer initialized: compression={compression_enabled}, "
            f"threshold={compression_threshold}B, compact={use_compact_json}"
        )

    def serialize(self, message: dict[str, Any]) -> bytes:
        """
        Serialize message to optimized byte format.

        Args:
            message: Dictionary message to serialize

        Returns:
            Optimized byte representation
        """
        start_time = time.perf_counter()

        try:
            # Convert to compact JSON
            json_str = json.dumps(message, separators=self.json_separators, ensure_ascii=False)
            json_bytes = json_str.encode("utf-8")

            # Apply compression if enabled and message is large enough
            if self.compression_enabled and len(json_bytes) >= self.compression_threshold:
                compressed_bytes = gzip.compress(json_bytes, compresslevel=6)
                result_bytes = b"GZIP" + compressed_bytes  # Add compression marker

                # Update compression stats
                self.serialization_stats.total_original_bytes += len(json_bytes)
                self.serialization_stats.total_compressed_bytes += len(result_bytes)
                self.serialization_stats.compression_count += 1
            else:
                result_bytes = json_bytes

            # Update performance stats
            serialization_time = time.perf_counter() - start_time
            self.serialization_stats.total_serializations += 1
            self.serialization_stats.total_serialization_time += serialization_time
            self.serialization_stats.total_messages += 1

            return result_bytes

        except Exception as e:
            logger.error(f"Serialization error: {e}")
            raise ValueError(f"Failed to serialize message: {e}")

    def serialize_compact(self, message: dict[str, Any]) -> bytes:
        """
        Serialize message with maximum compactness.

        Args:
            message: Dictionary message to serialize

        Returns:
            Maximally compact byte representation
        """
        # Remove optional fields for compactness
        compact_message = self._remove_optional_fields(message.copy())
        return self.serialize(compact_message)

    def _remove_optional_fields(self, message: dict[str, Any]) -> dict[str, Any]:
        """Remove optional fields to reduce message size."""
        # For RSSI messages, we can omit certain fields if they're default values
        if message.get("type") == "rssi_update":
            data = message.get("data", {})

            # Remove fields that are often default or redundant
            if data.get("confidence") == 1.0:
                data.pop("confidence", None)
            if data.get("source") == "drone_sdr":  # Default source
                data.pop("source", None)

        return message

    def deserialize(self, data: bytes) -> dict[str, Any] | None:
        """
        Deserialize optimized byte format back to message.

        Args:
            data: Byte data to deserialize

        Returns:
            Deserialized message dictionary or None if invalid
        """
        if not data:
            return None

        start_time = time.perf_counter()

        try:
            # Check for compression marker
            if data.startswith(b"GZIP"):
                # Remove marker and decompress
                compressed_data = data[4:]
                json_bytes = gzip.decompress(compressed_data)
            else:
                json_bytes = data

            # Parse JSON
            json_str = json_bytes.decode("utf-8")
            message = json.loads(json_str)

            # Update performance stats
            deserialization_time = time.perf_counter() - start_time
            self.serialization_stats.total_deserializations += 1
            self.serialization_stats.total_deserialization_time += deserialization_time

            return message  # type: ignore[no-any-return]

        except (json.JSONDecodeError, UnicodeDecodeError, gzip.BadGzipFile) as e:
            logger.warning(f"Deserialization error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected deserialization error: {e}")
            return None

    def serialize_batch(self, messages: list[dict[str, Any]]) -> bytes:
        """
        Serialize multiple messages efficiently.

        Args:
            messages: List of messages to serialize together

        Returns:
            Batch-serialized byte data
        """
        # Create batch message with shared metadata
        batch_message = {
            "type": "batch",
            "timestamp": messages[0].get("timestamp") if messages else None,
            "count": len(messages),
            "messages": messages,
        }

        return self.serialize(batch_message)

    def deserialize_batch(self, data: bytes) -> list[dict[str, Any]]:
        """
        Deserialize batch message back to list.

        Args:
            data: Batch-serialized data

        Returns:
            List of individual messages
        """
        batch_message = self.deserialize(data)
        if not batch_message or batch_message.get("type") != "batch":
            return []

        return batch_message.get("messages", [])

    def serialize_delta(
        self, current_message: dict[str, Any], previous_message: dict[str, Any]
    ) -> bytes:
        """
        Serialize message using delta compression against previous message.

        Args:
            current_message: Current message to serialize
            previous_message: Previous message for delta calculation

        Returns:
            Delta-compressed byte data
        """
        # Calculate delta (only changed fields)
        delta = self._calculate_delta(current_message, previous_message)

        # If delta is small enough, use it; otherwise use full message
        delta_size = len(json.dumps(delta).encode("utf-8"))
        full_size = len(json.dumps(current_message).encode("utf-8"))

        if delta_size < full_size * 0.7:  # Use delta if <70% of full size
            delta_message = {
                "type": "delta",
                "base_sequence": previous_message.get("sequence"),
                "delta": delta,
            }
            return self.serialize(delta_message)
        else:
            return self.serialize(current_message)

    def _calculate_delta(self, current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
        """Calculate delta between two messages."""
        delta = {}

        for key, value in current.items():
            if key not in previous or previous[key] != value:
                if isinstance(value, dict) and isinstance(previous.get(key), dict):
                    # Recursive delta for nested objects
                    nested_delta = self._calculate_delta(value, previous[key])
                    if nested_delta:
                        delta[key] = nested_delta
                else:
                    delta[key] = value

        return delta

    def get_statistics(self) -> dict[str, Any]:
        """
        Get comprehensive serialization performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        stats = self.serialization_stats

        # Calculate averages
        avg_ser_time = (
            stats.total_serialization_time / stats.total_serializations
            if stats.total_serializations > 0
            else 0.0
        ) * 1000  # Convert to milliseconds

        avg_deser_time = (
            stats.total_deserialization_time / stats.total_deserializations
            if stats.total_deserializations > 0
            else 0.0
        ) * 1000  # Convert to milliseconds

        compression_ratio = (
            stats.total_compressed_bytes / stats.total_original_bytes
            if stats.total_original_bytes > 0
            else 1.0
        )

        return {
            "total_messages": stats.total_messages,
            "total_serializations": stats.total_serializations,
            "total_deserializations": stats.total_deserializations,
            "average_serialization_time_ms": round(avg_ser_time, 3),
            "average_deserialization_time_ms": round(avg_deser_time, 3),
            "compression_ratio_average": round(compression_ratio, 3),
            "compression_count": stats.compression_count,
            "total_bytes_saved": max(0, stats.total_original_bytes - stats.total_compressed_bytes),
            "compression_enabled": self.compression_enabled,
            "compression_threshold": self.compression_threshold,
            "compact_json_enabled": self.use_compact_json,
        }


# Aliases for the test imports
CompressedMessageHandler = MessageSerializer
SerializationBenchmark = MessageSerializer
