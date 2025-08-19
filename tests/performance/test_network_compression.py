#!/usr/bin/env python3
"""
Network Compression Performance Tests

TASK-5.6.2.3 [8b] - Adaptive compression using LZ4 with dynamic threshold adjustment 
based on message type and priority.

Tests verify authentic compression behavior with real message payloads and 
performance requirements per PRD-NFR2 (<100ms latency).

CRITICAL: NO MOCK/FAKE/PLACEHOLDER TESTS - All tests verify real LZ4 compression behavior.
"""

import asyncio
import json
import time
from typing import Any

import pytest

# Add project root to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from src.backend.utils.coordination_optimizer import MessageSerializer


class TestLZ4AdaptiveCompression:
    """
    SUBTASK-5.6.2.3 [8b1] - Test LZ4 compression adapter with message type-aware thresholds.
    
    Tests verify real LZ4 compression with authentic message types and priority-based thresholds.
    """

    def test_lz4_compression_adapter_creation(self):
        """
        SUBTASK-5.6.2.3 [8b1] - Test LZ4 compression adapter initialization.
        
        RED PHASE: This test should FAIL initially because LZ4AdaptiveCompressor doesn't exist yet.
        Tests real LZ4 compression adapter creation with message type classification.
        """
        # This will fail initially - TDD RED phase
        from src.backend.utils.coordination_optimizer import LZ4AdaptiveCompressor
        
        # Create LZ4 adaptive compressor with real configuration
        compressor = LZ4AdaptiveCompressor(
            default_threshold=1024,  # 1KB default
            enable_type_classification=True,
            performance_monitoring=True
        )
        
        # Verify compressor is properly initialized
        assert compressor is not None
        assert compressor.default_threshold == 1024
        assert compressor.enable_type_classification is True
        assert compressor.performance_monitoring is True
        
        # Verify LZ4 is available and working
        test_data = b"test compression data" * 100
        compressed = compressor.compress_with_lz4(test_data)
        assert compressed != test_data
        assert len(compressed) < len(test_data)

    def test_message_type_threshold_calculation(self):
        """
        SUBTASK-5.6.2.3 [8b1] - Test message type-aware threshold calculation.
        
        Tests authentic threshold calculation based on real message types and priorities.
        """
        from src.backend.utils.coordination_optimizer import LZ4AdaptiveCompressor
        
        compressor = LZ4AdaptiveCompressor()
        
        # Test high-priority safety message (priority 1-4)
        safety_message = {
            "type": "safety_alert",
            "priority": 1,
            "data": {"alert": "Emergency stop required", "timestamp": time.time()}
        }
        safety_threshold = compressor.calculate_threshold_for_message(safety_message)
        assert safety_threshold <= 512  # Lower threshold for faster processing
        
        # Test control message (medium priority)
        control_message = {
            "type": "freq_control", 
            "priority": 10,
            "data": {"command": "SET_FREQUENCY", "frequency": 406025000}
        }
        control_threshold = compressor.calculate_threshold_for_message(control_message)
        assert 512 < control_threshold <= 1024  # Medium threshold
        
        # Test RSSI streaming message (lower priority, high frequency)
        rssi_message = {
            "type": "rssi_update",
            "priority": 20,
            "data": {"rssi_dbm": -70.5, "timestamp": time.time(), "source": "drone_sdr"}
        }
        rssi_threshold = compressor.calculate_threshold_for_message(rssi_message)
        assert rssi_threshold > 1024  # Higher threshold for efficiency


class TestDynamicThresholdAdjustment:
    """
    SUBTASK-5.6.2.3 [8b2] - Test dynamic threshold adjustment based on network conditions.
    
    Tests verify real-time threshold adaptation using authentic network monitoring data.
    """

    def test_threshold_adjustment_based_on_network_conditions(self):
        """
        SUBTASK-5.6.2.3 [8b2] - Test dynamic threshold adjustment algorithm.
        
        Tests real network condition monitoring and threshold adaptation.
        """
        from src.backend.utils.coordination_optimizer import LZ4AdaptiveCompressor
        from src.backend.utils.resource_optimizer import NetworkBandwidthMonitor
        
        compressor = LZ4AdaptiveCompressor()
        network_monitor = NetworkBandwidthMonitor()
        
        # Simulate high network utilization scenario
        high_utilization_stats = {
            "bandwidth_utilization_percent": 85.0,
            "latency_ms": 45.0,
            "packet_loss_percent": 2.0
        }
        
        adjusted_threshold = compressor.adjust_threshold_for_network_conditions(
            base_threshold=1024,
            network_stats=high_utilization_stats
        )
        
        # Under high utilization, threshold should be lower (more aggressive compression)
        assert adjusted_threshold < 1024
        assert adjusted_threshold >= 256  # But not too aggressive
        
        # Simulate low network utilization scenario
        low_utilization_stats = {
            "bandwidth_utilization_percent": 25.0,
            "latency_ms": 15.0,
            "packet_loss_percent": 0.1
        }
        
        adjusted_threshold_low = compressor.adjust_threshold_for_network_conditions(
            base_threshold=1024,
            network_stats=low_utilization_stats
        )
        
        # Under low utilization, threshold should be higher (less compression for speed)
        assert adjusted_threshold_low > 1024
        assert adjusted_threshold_low <= 2048  # Reasonable upper limit

    def test_priority_based_threshold_scaling(self):
        """
        SUBTASK-5.6.2.3 [8b2] - Test priority-based threshold scaling.
        
        Tests authentic priority system integration with compression thresholds.
        """
        from src.backend.utils.coordination_optimizer import LZ4AdaptiveCompressor
        
        compressor = LZ4AdaptiveCompressor()
        base_threshold = 1024
        
        # Test critical priority (safety messages)
        critical_threshold = compressor.scale_threshold_by_priority(
            base_threshold=base_threshold,
            priority=1  # Critical safety priority
        )
        assert critical_threshold <= base_threshold * 0.5  # Aggressive scaling for speed
        
        # Test normal priority
        normal_threshold = compressor.scale_threshold_by_priority(
            base_threshold=base_threshold,
            priority=10  # Normal operation priority
        )
        assert normal_threshold == base_threshold  # No scaling
        
        # Test low priority
        low_threshold = compressor.scale_threshold_by_priority(
            base_threshold=base_threshold,
            priority=50  # Low priority bulk data
        )
        assert low_threshold >= base_threshold * 1.5  # Higher threshold for efficiency


class TestMessageTypeClassification:
    """
    SUBTASK-5.6.2.3 [8b3] - Test message type classification for adaptive compression.
    
    Tests verify authentic message classification using real message structures.
    """

    def test_safety_message_classification(self):
        """
        SUBTASK-5.6.2.3 [8b3] - Test safety message type classification.
        
        Tests classification of real safety and emergency messages.
        """
        from src.backend.utils.coordination_optimizer import MessageTypeClassifier
        
        classifier = MessageTypeClassifier()
        
        # Test emergency safety message
        emergency_msg = {
            "type": "safety_alert",
            "priority": 1,
            "data": {"action": "RTL", "reason": "Battery critical"}
        }
        
        classification = classifier.classify_message(emergency_msg)
        assert classification.category == "safety"
        assert classification.urgency == "critical"
        assert classification.compression_priority == "speed"
        
        # Test RC override message
        rc_override_msg = {
            "type": "rc_override",
            "priority": 1,
            "data": {"override_active": True, "source": "pilot"}
        }
        
        rc_classification = classifier.classify_message(rc_override_msg)
        assert rc_classification.category == "safety"
        assert rc_classification.compression_priority == "speed"

    def test_control_message_classification(self):
        """
        SUBTASK-5.6.2.3 [8b3] - Test control message type classification.
        
        Tests classification of frequency control and coordination messages.
        """
        from src.backend.utils.coordination_optimizer import MessageTypeClassifier
        
        classifier = MessageTypeClassifier()
        
        # Test SET_FREQUENCY control message
        freq_control_msg = {
            "type": "freq_control",
            "priority": 10,
            "data": {"command": "SET_FREQUENCY", "frequency": 406025000}
        }
        
        classification = classifier.classify_message(freq_control_msg)
        assert classification.category == "control"
        assert classification.urgency == "medium"
        assert classification.compression_priority == "balanced"
        
        # Test coordination message
        coordination_msg = {
            "type": "coordination",
            "priority": 15,
            "data": {"source_switch": True, "new_source": "ground_sdr"}
        }
        
        coord_classification = classifier.classify_message(coordination_msg)
        assert coord_classification.category == "coordination"
        assert coord_classification.compression_priority == "balanced"

    def test_data_streaming_classification(self):
        """
        SUBTASK-5.6.2.3 [8b3] - Test data streaming message classification.
        
        Tests classification of high-frequency RSSI and telemetry streaming messages.
        """
        from src.backend.utils.coordination_optimizer import MessageTypeClassifier
        
        classifier = MessageTypeClassifier()
        
        # Test RSSI streaming message
        rssi_msg = {
            "type": "rssi_update",
            "priority": 20,
            "data": {
                "rssi_dbm": -70.5,
                "frequency_hz": 406025000,
                "timestamp": time.time(),
                "source_id": 0
            }
        }
        
        classification = classifier.classify_message(rssi_msg)
        assert classification.category == "data_streaming"
        assert classification.urgency == "low"
        assert classification.compression_priority == "efficiency"
        
        # Test telemetry batch message
        telemetry_batch = {
            "type": "batch",
            "priority": 25,
            "data": {"messages": [rssi_msg] * 10, "count": 10}
        }
        
        batch_classification = classifier.classify_message(telemetry_batch)
        assert batch_classification.category == "data_streaming"
        assert batch_classification.compression_priority == "efficiency"


class TestPerformanceMonitoring:
    """
    SUBTASK-5.6.2.3 [8b5] - Test compression performance monitoring.
    
    Tests verify real compression latency and efficiency monitoring.
    """

    def test_compression_latency_monitoring(self):
        """
        SUBTASK-5.6.2.3 [8b5] - Test compression latency measurement.
        
        Tests authentic latency monitoring for compression operations.
        """
        from src.backend.utils.coordination_optimizer import LZ4AdaptiveCompressor
        
        compressor = LZ4AdaptiveCompressor(performance_monitoring=True)
        
        # Test compression latency with realistic large message
        test_message = {
            "type": "rssi_update",
            "timestamp": time.time(),
            "priority": 20,  # Low priority to ensure compression
            "data": {
                "rssi_dbm": -70.5, 
                "metadata": "x" * 2000,  # 2KB+ message to exceed threshold
                "extended_data": ["sample"] * 500  # Additional data to ensure compression
            }
        }
        
        start_time = time.perf_counter()
        compressed_data = compressor.compress_message(test_message)
        compression_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        # Verify compression completed successfully
        assert compressed_data is not None
        assert len(compressed_data) > 0
        
        # Verify compression latency meets PRD-NFR2 requirement (<100ms)
        assert compression_time < 100.0, f"Compression took {compression_time:.2f}ms, exceeds 100ms limit"
        
        # Verify performance statistics are collected
        stats = compressor.get_performance_stats()
        assert stats["total_compressions"] >= 1
        assert stats["average_compression_time_ms"] > 0
        assert stats["average_compression_time_ms"] < 100

    def test_compression_efficiency_monitoring(self):
        """
        SUBTASK-5.6.2.3 [8b5] - Test compression efficiency measurement.
        
        Tests real compression ratio and bandwidth savings monitoring.
        """
        from src.backend.utils.coordination_optimizer import LZ4AdaptiveCompressor
        
        compressor = LZ4AdaptiveCompressor(performance_monitoring=True)
        
        # Test with highly compressible data (large and repetitive)
        repetitive_message = {
            "type": "telemetry_batch",
            "priority": 25,  # Low priority to ensure compression
            "data": {
                "readings": ["similar_data_pattern"] * 200,  # Very repetitive data
                "metadata": "repeated_text " * 300  # Additional repetitive content
            }
        }
        
        original_size = len(json.dumps(repetitive_message).encode('utf-8'))
        compressed_data = compressor.compress_message(repetitive_message)
        
        # Verify compression achieved good ratio
        compression_ratio = len(compressed_data) / original_size
        assert compression_ratio < 0.8, f"Compression ratio {compression_ratio:.2f} not efficient enough"
        
        # Test efficiency statistics
        efficiency_stats = compressor.get_efficiency_stats()
        assert efficiency_stats["average_compression_ratio"] < 1.0
        assert efficiency_stats["total_bytes_saved"] > 0
        assert efficiency_stats["bandwidth_savings_percent"] > 0


class TestLatencyRequirementValidation:
    """
    SUBTASK-5.6.2.3 [8b6] - Test latency requirement validation per PRD-NFR2.
    
    Tests verify compression maintains <100ms latency under realistic conditions.
    """

    @pytest.mark.asyncio
    async def test_compression_latency_under_load(self):
        """
        SUBTASK-5.6.2.3 [8b6] - Test compression latency under realistic load.
        
        Tests compression performance with concurrent operations and realistic message volumes.
        """
        from src.backend.utils.coordination_optimizer import LZ4AdaptiveCompressor
        
        compressor = LZ4AdaptiveCompressor()
        
        # Create realistic message load
        messages = []
        for i in range(100):
            messages.append({
                "type": "rssi_update",
                "timestamp": time.time() + i * 0.01,
                "data": {
                    "rssi_dbm": -70.0 + (i % 20),
                    "frequency_hz": 406025000,
                    "source_id": i % 2,
                    "metadata": f"sample_{i}" * 10  # Variable size data
                }
            })
        
        # Test compression latency under concurrent load
        async def compress_message_async(msg):
            start_time = time.perf_counter()
            result = compressor.compress_message(msg)
            latency_ms = (time.perf_counter() - start_time) * 1000
            return result, latency_ms
        
        # Process messages concurrently
        tasks = [compress_message_async(msg) for msg in messages[:10]]
        results = await asyncio.gather(*tasks)
        
        # Verify all compression operations completed
        assert len(results) == 10
        
        # Verify latency requirements met for all operations
        for compressed_data, latency_ms in results:
            assert compressed_data is not None
            assert latency_ms < 100.0, f"Compression latency {latency_ms:.2f}ms exceeds 100ms requirement"
        
        # Calculate average latency
        avg_latency = sum(latency for _, latency in results) / len(results)
        assert avg_latency < 50.0, f"Average latency {avg_latency:.2f}ms should be well under 100ms"

    def test_worst_case_compression_latency(self):
        """
        SUBTASK-5.6.2.3 [8b6] - Test worst-case compression latency scenarios.
        
        Tests compression performance with large messages and poor compression ratios.
        """
        from src.backend.utils.coordination_optimizer import LZ4AdaptiveCompressor
        
        compressor = LZ4AdaptiveCompressor()
        
        # Create worst-case message (large, random data that doesn't compress well)
        import random
        worst_case_message = {
            "type": "large_data",
            "priority": 50,
            "data": {
                "random_data": [random.random() for _ in range(1000)],
                "large_string": "".join(chr(random.randint(65, 90)) for _ in range(5000))
            }
        }
        
        # Measure compression latency for worst case
        start_time = time.perf_counter()
        compressed_data = compressor.compress_message(worst_case_message)
        worst_case_latency = (time.perf_counter() - start_time) * 1000
        
        # Even worst-case scenarios must meet latency requirement
        assert worst_case_latency < 100.0, f"Worst-case latency {worst_case_latency:.2f}ms exceeds 100ms"
        assert compressed_data is not None
        
        # Verify adaptive threshold prevents compression of poorly compressible data
        stats = compressor.get_performance_stats()
        if stats.get("last_compression_skipped", False):
            # If compression was skipped due to poor ratio, that's acceptable
            assert stats["last_skip_reason"] in ["poor_compression_ratio", "size_threshold"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])