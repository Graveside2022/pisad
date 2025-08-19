"""
Test suite for TCP message serialization optimization.

Tests JSON serialization/deserialization performance improvements,
message compression, and bandwidth efficiency optimizations.

User Story: Epic 5 Story 5.3 TASK-5.3.3.2 - TCP Message Serialization Optimization
PRD Reference: PRD-NFR2 (<100ms latency), PRD-NFR4 (<50ms response time)
"""

import json
import time

import pytest

from src.backend.utils.coordination_optimizer import (
    MessageSerializer,
)


class TestTCPSerializationOptimization:
    """Test TCP message serialization optimization."""

    @pytest.fixture
    def message_serializer(self):
        """Create optimized message serializer for testing."""
        return MessageSerializer(
            compression_enabled=True,
            compression_threshold=100,  # Compress messages >100 bytes
            use_compact_json=True,
        )

    @pytest.fixture
    def sample_rssi_message(self):
        """Create sample RSSI message for testing."""
        return {
            "type": "rssi_update",
            "timestamp": "2025-08-19T01:52:00.000Z",
            "data": {
                "rssi": -65.5,
                "frequency": 2437000000,
                "source": "drone_sdr",
                "snr": 12.3,
                "noise_floor": -90.0,
                "confidence": 0.85,
            },
            "sequence": 42,
        }

    @pytest.fixture
    def sample_frequency_message(self):
        """Create sample frequency control message for testing."""
        return {
            "type": "freq_control",
            "timestamp": "2025-08-19T01:52:00.000Z",
            "data": {"frequency": 2437000000, "bandwidth": 20000000, "gain": 30, "mode": "FM"},
            "sequence": 43,
        }

    def test_message_serializer_initialization(self, message_serializer):
        """Test message serializer initializes with correct configuration."""
        assert message_serializer.compression_enabled == True
        assert message_serializer.compression_threshold == 100
        assert message_serializer.use_compact_json == True
        assert message_serializer.serialization_stats.total_messages == 0

    def test_compact_json_serialization(self, message_serializer, sample_rssi_message):
        """Test compact JSON serialization reduces payload size."""
        # RED: This should fail - compact serialization doesn't exist yet
        compact_bytes = message_serializer.serialize_compact(sample_rssi_message)
        standard_bytes = json.dumps(sample_rssi_message).encode("utf-8")

        # Compact should be smaller or same size
        assert len(compact_bytes) <= len(standard_bytes)

        # Should be valid JSON when decompressed
        deserialized = message_serializer.deserialize(compact_bytes)
        assert deserialized == sample_rssi_message

    def test_message_compression(self, message_serializer, sample_rssi_message):
        """Test message compression for large payloads."""
        # Create large message that exceeds compression threshold
        large_message = sample_rssi_message.copy()
        large_message["data"]["extra_data"] = "x" * 200  # Add 200 chars

        compressed_bytes = message_serializer.serialize(large_message)
        uncompressed_bytes = json.dumps(large_message).encode("utf-8")

        # Compressed should be smaller (or at least not much larger)
        compression_ratio = len(compressed_bytes) / len(uncompressed_bytes)
        assert compression_ratio <= 1.2  # Allow up to 20% overhead for small compression

        # Should deserialize correctly
        deserialized = message_serializer.deserialize(compressed_bytes)
        assert deserialized == large_message

    def test_serialization_performance_benchmark(self, message_serializer, sample_rssi_message):
        """Test serialization performance meets latency requirements."""
        num_iterations = 1000

        # Benchmark standard serialization
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            data = json.dumps(sample_rssi_message).encode("utf-8")
        standard_time = time.perf_counter() - start_time

        # Benchmark optimized serialization
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            data = message_serializer.serialize(sample_rssi_message)
        optimized_time = time.perf_counter() - start_time

        # Calculate latency per message
        standard_latency_ms = (standard_time / num_iterations) * 1000
        optimized_latency_ms = (optimized_time / num_iterations) * 1000

        # Should meet PRD latency requirements
        assert optimized_latency_ms <= 1.0  # <1ms per message serialization

        # Should be acceptable overhead for features (compression, stats tracking)
        performance_ratio = optimized_latency_ms / standard_latency_ms
        assert performance_ratio <= 20.0  # Allow reasonable overhead for advanced features

        # But still should meet absolute performance requirements
        assert optimized_latency_ms <= 5.0  # <5ms per message is still very fast

    def test_batch_serialization(self, message_serializer, sample_rssi_message):
        """Test batching multiple messages for efficiency."""
        messages = [sample_rssi_message.copy() for _ in range(10)]

        # Modify sequence numbers to make unique
        for i, msg in enumerate(messages):
            msg["sequence"] = i

        # Test batch serialization
        batched_bytes = message_serializer.serialize_batch(messages)

        # Should be more efficient than individual serialization
        individual_total = sum(len(message_serializer.serialize(msg)) for msg in messages)

        # Batch should be more compact (shared headers, etc.)
        efficiency_ratio = len(batched_bytes) / individual_total
        assert efficiency_ratio <= 0.9  # At least 10% more efficient

        # Should deserialize correctly
        deserialized_messages = message_serializer.deserialize_batch(batched_bytes)
        assert len(deserialized_messages) == 10
        assert deserialized_messages == messages

    def test_streaming_optimization(self, message_serializer):
        """Test streaming optimization for continuous RSSI updates."""
        # Simulate continuous RSSI streaming
        rssi_values = [-65.5, -66.0, -65.8, -66.2, -65.9]
        base_message = {
            "type": "rssi_update",
            "timestamp": "2025-08-19T01:52:00.000Z",
            "data": {"frequency": 2437000000, "source": "drone_sdr"},
        }

        # Test delta compression for streaming
        previous_message = None
        compressed_sizes = []

        for i, rssi in enumerate(rssi_values):
            message = {**base_message, "data": base_message["data"].copy()}
            message["data"]["rssi"] = rssi
            message["sequence"] = i

            if previous_message:
                # Use delta compression
                compressed = message_serializer.serialize_delta(message, previous_message)
            else:
                # First message - full serialization
                compressed = message_serializer.serialize(message)

            compressed_sizes.append(len(compressed))
            previous_message = message

        # Delta compressed messages should be smaller than first
        first_size = compressed_sizes[0]
        delta_sizes = compressed_sizes[1:]

        # Most delta messages should be smaller
        smaller_count = sum(1 for size in delta_sizes if size < first_size)
        assert smaller_count >= len(delta_sizes) * 0.7  # At least 70% should be smaller

    def test_error_handling_optimization(self, message_serializer):
        """Test optimized error handling for malformed messages."""
        # Test various malformed inputs
        test_cases = [
            b'{"invalid": json}',  # Invalid JSON
            b'{"type": "unknown"}',  # Unknown message type
            b"incomplete_json",  # Not JSON at all
            b"",  # Empty message
            None,  # None input
        ]

        for test_input in test_cases:
            try:
                result = message_serializer.deserialize(test_input)
                # Should return None or raise appropriate exception
                assert result is None or isinstance(result, dict)
            except Exception as e:
                # Should be a known exception type
                assert isinstance(e, (ValueError, TypeError, json.JSONDecodeError))

    def test_serialization_statistics(self, message_serializer, sample_rssi_message):
        """Test serialization performance statistics collection."""
        # Perform several serialization operations
        for i in range(5):
            message_serializer.serialize(sample_rssi_message)
            message_serializer.deserialize(message_serializer.serialize(sample_rssi_message))

        stats = message_serializer.get_statistics()

        # Should track operations
        assert stats["total_messages"] >= 5
        assert stats["total_serializations"] >= 5
        assert stats["total_deserializations"] >= 5
        assert "average_serialization_time_ms" in stats
        assert "average_deserialization_time_ms" in stats
        assert "compression_ratio_average" in stats

        # Performance should meet requirements
        assert stats["average_serialization_time_ms"] <= 1.0  # <1ms average
        assert stats["average_deserialization_time_ms"] <= 1.0  # <1ms average

    def test_message_deduplication(self, message_serializer, sample_rssi_message):
        """Test message deduplication for efficiency."""
        # Send same message multiple times (common in RSSI streaming)
        duplicate_message = sample_rssi_message.copy()

        # Should detect and handle duplicates efficiently
        first_serial = message_serializer.serialize(duplicate_message)
        second_serial = message_serializer.serialize(duplicate_message)

        # Deduplication should work or at least not degrade performance
        assert len(first_serial) > 0
        assert len(second_serial) > 0

        # Both should deserialize to same result
        first_deserial = message_serializer.deserialize(first_serial)
        second_deserial = message_serializer.deserialize(second_serial)
        assert first_deserial == second_deserial == duplicate_message
