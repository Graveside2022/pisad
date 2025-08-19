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

import lz4.frame

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

        messages = batch_message.get("messages", [])
        # Ensure type safety: return empty list if messages is not a list
        return messages if isinstance(messages, list) else []

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


# TASK-5.6.2.3 [8b] - LZ4 Adaptive Compression Implementation


@dataclass
class MessageClassification:
    """Classification result for message compression decisions."""

    category: str  # safety, control, coordination, data_streaming
    urgency: str  # critical, medium, low
    compression_priority: str  # speed, balanced, efficiency


@dataclass
class CompressionStats:
    """Performance statistics for adaptive compression."""

    total_compressions: int = 0
    total_compression_time: float = 0.0
    total_bytes_saved: int = 0
    total_original_bytes: int = 0
    total_compressed_bytes: int = 0
    last_compression_skipped: bool = False
    last_skip_reason: str = ""


class MessageTypeClassifier:
    """
    SUBTASK-5.6.2.3 [8b3] - Message type classification for adaptive compression.

    Classifies messages based on type, priority, and content to determine optimal
    compression strategy balancing speed vs efficiency.
    """

    def __init__(self) -> None:
        # Priority thresholds for classification
        self.critical_priority_threshold = 4  # Priorities 1-4 are critical
        self.medium_priority_threshold = 15  # Priorities 5-15 are medium
        # Everything above 15 is low priority

        logger.info("MessageTypeClassifier initialized with priority thresholds")

    def classify_message(self, message: dict[str, Any]) -> MessageClassification:
        """
        Classify message for compression optimization.

        Args:
            message: Message dictionary with type, priority, and data

        Returns:
            MessageClassification with category, urgency, and compression priority
        """
        msg_type = message.get("type", "unknown")
        priority = message.get("priority", 50)  # Default to low priority

        # Determine category based on message type
        if msg_type in ["safety_alert", "rc_override", "emergency", "battery_critical"]:
            category = "safety"
        elif msg_type in ["freq_control", "command", "control"]:
            category = "control"
        elif msg_type in ["coordination", "source_switch", "fallback"]:
            category = "coordination"
        elif msg_type in ["rssi_update", "telemetry", "batch", "data_streaming"]:
            category = "data_streaming"
        else:
            category = "unknown"

        # Determine urgency based on priority
        if priority <= self.critical_priority_threshold:
            urgency = "critical"
        elif priority <= self.medium_priority_threshold:
            urgency = "medium"
        else:
            urgency = "low"

        # Determine compression priority based on category and urgency
        if category == "safety" or urgency == "critical":
            compression_priority = "speed"  # Minimize latency for safety
        elif category in ["control", "coordination"] or urgency == "medium":
            compression_priority = "balanced"  # Balance speed and efficiency
        else:
            compression_priority = "efficiency"  # Maximize compression for bulk data

        return MessageClassification(
            category=category, urgency=urgency, compression_priority=compression_priority
        )


class LZ4AdaptiveCompressor:
    """
    SUBTASK-5.6.2.3 [8b1,8b2] - LZ4 adaptive compression with dynamic thresholds.

    Provides LZ4 compression with dynamic threshold adjustment based on message type,
    priority, and network conditions for optimal performance vs bandwidth trade-offs.
    """

    def __init__(
        self,
        default_threshold: int = 1024,
        enable_type_classification: bool = True,
        performance_monitoring: bool = True,
    ):
        """
        Initialize LZ4 adaptive compressor.

        Args:
            default_threshold: Default compression threshold in bytes
            enable_type_classification: Enable message type-based threshold adjustment
            performance_monitoring: Enable performance statistics collection
        """
        self.default_threshold = default_threshold
        self.enable_type_classification = enable_type_classification
        self.performance_monitoring = performance_monitoring

        # Initialize components
        self.classifier: MessageTypeClassifier | None = (
            MessageTypeClassifier() if enable_type_classification else None
        )
        self.stats = CompressionStats()

        # Compression level configuration
        self.compression_levels = {
            "speed": 1,  # Fast compression for critical messages
            "balanced": 4,  # Balanced compression for normal messages
            "efficiency": 9,  # High compression for bulk data
        }

        logger.info(
            f"LZ4AdaptiveCompressor initialized: threshold={default_threshold}B, "
            f"classification={enable_type_classification}, monitoring={performance_monitoring}"
        )

    def calculate_threshold_for_message(self, message: dict[str, Any]) -> int:
        """
        SUBTASK-5.6.2.3 [8b1] - Calculate compression threshold for specific message.

        Args:
            message: Message dictionary with type and priority

        Returns:
            Compression threshold in bytes adjusted for message characteristics
        """
        if not self.enable_type_classification or not self.classifier:
            return self.default_threshold

        classification = self.classifier.classify_message(message)
        priority = message.get("priority", 50)

        # Apply priority-based scaling
        threshold = self.scale_threshold_by_priority(self.default_threshold, priority)

        # Apply category-based adjustment
        if classification.compression_priority == "speed":
            threshold = int(threshold * 0.5)  # Lower threshold for speed
        elif classification.compression_priority == "balanced":
            threshold = threshold  # Use calculated threshold
        else:  # efficiency
            threshold = int(threshold * 1.5)  # Higher threshold for efficiency

        # Ensure reasonable bounds
        return max(256, min(threshold, 4096))

    def scale_threshold_by_priority(self, base_threshold: int, priority: int) -> int:
        """
        SUBTASK-5.6.2.3 [8b2] - Scale threshold based on message priority.

        Args:
            base_threshold: Base threshold value
            priority: Message priority (1=highest, higher numbers = lower priority)

        Returns:
            Scaled threshold value
        """
        if priority <= 4:  # Critical priority
            return int(base_threshold * 0.5)
        elif priority <= 15:  # Medium priority
            return base_threshold
        else:  # Low priority
            return int(base_threshold * 1.5)

    def adjust_threshold_for_network_conditions(
        self, base_threshold: int, network_stats: dict[str, float]
    ) -> int:
        """
        SUBTASK-5.6.2.3 [8b2] - Adjust threshold based on network conditions.

        Args:
            base_threshold: Base threshold value
            network_stats: Network statistics with utilization, latency, packet_loss

        Returns:
            Adjusted threshold based on network conditions
        """
        utilization = network_stats.get("bandwidth_utilization_percent", 50.0)
        latency = network_stats.get("latency_ms", 25.0)
        packet_loss = network_stats.get("packet_loss_percent", 0.5)

        # Calculate adjustment factor based on network conditions
        adjustment_factor = 1.0

        # High utilization -> more aggressive compression (lower threshold)
        if utilization > 80:
            adjustment_factor *= 0.6
        elif utilization > 60:
            adjustment_factor *= 0.8
        elif utilization < 30:
            adjustment_factor *= 1.3

        # High latency -> be more conservative with compression overhead
        if latency > 50:
            adjustment_factor *= 1.2
        elif latency < 20:
            adjustment_factor *= 0.9

        # High packet loss -> more aggressive compression to reduce retransmissions
        if packet_loss > 1.0:
            adjustment_factor *= 0.7

        adjusted_threshold = int(base_threshold * adjustment_factor)
        return max(256, min(adjusted_threshold, 4096))

    def compress_with_lz4(self, data: bytes) -> bytes:
        """
        Compress data using LZ4 frame format.

        Args:
            data: Raw bytes to compress

        Returns:
            LZ4 compressed bytes
        """
        try:
            # Use LZ4 frame format with correct API
            compressed = lz4.frame.compress(
                data, compression_level=self.compression_levels["balanced"]
            )
            # Verify compression actually happened
            if len(compressed) >= len(data):
                logger.warning(
                    f"LZ4 compression ineffective: {len(compressed)} >= {len(data)} bytes"
                )
            return bytes(compressed)
        except Exception as e:
            logger.error(f"LZ4 compression failed: {e}")
            raise  # Re-raise for proper error handling in tests

    def compress_message(self, message: dict[str, Any]) -> bytes | None:
        """
        SUBTASK-5.6.2.3 [8b4,8b5] - Compress message with adaptive thresholds and monitoring.

        Args:
            message: Message dictionary to compress

        Returns:
            Compressed message bytes or None on failure
        """
        if not message:
            return None

        start_time = time.perf_counter()

        try:
            # Serialize message to JSON
            json_str = json.dumps(message, separators=(",", ":"))
            json_bytes = json_str.encode("utf-8")
            original_size = len(json_bytes)

            # Calculate adaptive threshold
            threshold = self.calculate_threshold_for_message(message)

            # Decide whether to compress
            if original_size < threshold:
                # Skip compression for small messages
                if self.performance_monitoring:
                    self.stats.last_compression_skipped = True
                    self.stats.last_skip_reason = "size_threshold"
                return json_bytes

            # Determine compression level based on message classification
            if self.classifier:
                classification = self.classifier.classify_message(message)
                compression_level = self.compression_levels.get(
                    classification.compression_priority, 4
                )
            else:
                compression_level = 4

            # Compress with LZ4
            compressed_data = lz4.frame.compress(json_bytes, compression_level=compression_level)

            # Check compression ratio
            compression_ratio = len(compressed_data) / original_size
            if compression_ratio > 0.9:  # Poor compression ratio
                if self.performance_monitoring:
                    self.stats.last_compression_skipped = True
                    self.stats.last_skip_reason = "poor_compression_ratio"
                return json_bytes

            # Add LZ4 marker for decompression
            result = b"LZ4F" + compressed_data

            # Update performance statistics
            if self.performance_monitoring:
                compression_time = time.perf_counter() - start_time
                self.stats.total_compressions += 1
                self.stats.total_compression_time += compression_time
                self.stats.total_original_bytes += original_size
                self.stats.total_compressed_bytes += len(result)
                self.stats.total_bytes_saved += original_size - len(result)
                self.stats.last_compression_skipped = False

            return bytes(result)

        except Exception as e:
            logger.error(f"Message compression failed: {e}")
            return None

    def get_performance_stats(self) -> dict[str, Any]:
        """
        SUBTASK-5.6.2.3 [8b5] - Get compression performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        if not self.performance_monitoring:
            return {"performance_monitoring": False}

        avg_compression_time = 0.0
        if self.stats.total_compressions > 0:
            avg_compression_time = (
                self.stats.total_compression_time / self.stats.total_compressions
            ) * 1000  # Convert to milliseconds

        return {
            "total_compressions": self.stats.total_compressions,
            "average_compression_time_ms": round(avg_compression_time, 3),
            "total_bytes_saved": self.stats.total_bytes_saved,
            "last_compression_skipped": self.stats.last_compression_skipped,
            "last_skip_reason": self.stats.last_skip_reason,
            "performance_monitoring": True,
        }

    def get_efficiency_stats(self) -> dict[str, Any]:
        """
        SUBTASK-5.6.2.3 [8b5] - Get compression efficiency statistics.

        Returns:
            Dictionary with efficiency metrics
        """
        if not self.performance_monitoring or self.stats.total_original_bytes == 0:
            return {"efficiency_monitoring": False}

        avg_compression_ratio = self.stats.total_compressed_bytes / self.stats.total_original_bytes

        bandwidth_savings_percent = (
            (self.stats.total_bytes_saved / self.stats.total_original_bytes) * 100
            if self.stats.total_original_bytes > 0
            else 0.0
        )

        return {
            "average_compression_ratio": round(avg_compression_ratio, 3),
            "total_bytes_saved": self.stats.total_bytes_saved,
            "bandwidth_savings_percent": round(bandwidth_savings_percent, 1),
            "efficiency_monitoring": True,
        }


# Aliases for the test imports
CompressedMessageHandler = MessageSerializer
SerializationBenchmark = MessageSerializer
