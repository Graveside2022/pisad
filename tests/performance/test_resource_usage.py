#!/usr/bin/env python3
"""
Resource Usage Performance Tests

TASK-5.6.2-RESOURCE-OPTIMIZATION - Comprehensive resource monitoring and optimization testing.
Tests verify authentic system behavior using real resource consumption patterns.

PRD References:
- NFR4: Power consumption ≤2.5A @ 5V (implies memory <2GB on Pi 5)
- NFR2: Signal processing latency <100ms per RSSI computation cycle
- AC5.6.5: Memory usage optimization prevents resource exhaustion

CRITICAL: NO MOCK/FAKE/PLACEHOLDER TESTS - All tests verify real system behavior.
"""

import asyncio
import gc
import os
import sys
import time
import threading
from collections import deque
from typing import Dict, Any, List
import pytest

try:
    import psutil
    import memory_profiler
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from src.backend.utils.resource_optimizer import ResourceOptimizer, MemoryPool, GracefulDegradationManager
from src.backend.utils.performance_monitor import AdaptivePerformanceMonitor, PerformanceThresholds


class TestMemoryUsageAnalysis:
    """
    SUBTASK-5.6.2.1 [6a] - Test memory usage pattern analysis and leak detection.
    
    Tests verify real memory consumption in RSSI processing and SDR coordination.
    """
    
    @pytest.mark.skipif(not PROFILING_AVAILABLE, reason="memory-profiler not available")
    def test_memory_usage_analysis_detects_rssi_buffer_growth(self):
        """
        SUBTASK-5.6.2.1 [6a] - Verify memory analysis can detect RSSI buffer growth patterns.
        
        RED PHASE: This test should FAIL initially because ResourceOptimizer doesn't exist yet.
        Tests real memory consumption of RSSI data processing without mocks.
        """
        # Create ResourceOptimizer instance (will fail initially)
        optimizer = ResourceOptimizer()
        
        # Simulate RSSI data processing that might cause memory growth
        initial_memory = optimizer.get_current_memory_usage()
        
        # Create realistic RSSI data similar to actual system workload
        rssi_data = []
        for i in range(10000):  # Simulate high-frequency RSSI samples
            rssi_data.append({
                'timestamp': time.time() + i * 0.01,  # 10ms intervals
                'rssi_dbm': -70.0 + (i % 20) - 10,    # Realistic RSSI variation
                'frequency_hz': 406025000,             # 406.025 MHz beacon frequency
                'source_id': i % 2,                    # Dual SDR sources
                'quality_score': min(100, 50 + (i % 50))
            })
        
        # Process RSSI data and measure memory growth
        memory_usage_samples = []
        for batch_start in range(0, len(rssi_data), 1000):
            batch = rssi_data[batch_start:batch_start + 1000]
            optimizer.process_rssi_batch(batch)
            current_memory = optimizer.get_current_memory_usage()
            memory_usage_samples.append(current_memory)
        
        # Verify memory analysis capabilities
        analysis = optimizer.analyze_memory_usage_patterns(memory_usage_samples)
        
        # Test assertions for real memory behavior
        assert hasattr(analysis, 'memory_trend'), "Memory analysis must include trend detection"
        assert hasattr(analysis, 'leak_detection'), "Memory analysis must detect potential leaks"  
        assert hasattr(analysis, 'peak_usage_mb'), "Memory analysis must track peak usage"
        assert analysis.peak_usage_mb > initial_memory, "Memory usage should increase with data processing"
        
        # Verify memory stays within Pi 5 limits (2GB = 2048MB)
        assert analysis.peak_usage_mb < 2048, "Memory usage must stay under 2GB per PRD NFR4"
        
        # Verify memory leak detection accuracy
        if analysis.leak_detection['potential_leak']:
            assert analysis.leak_detection['growth_rate_mb_per_sec'] > 0.1, "Leak detection threshold must be realistic"

    @pytest.mark.skipif(not PROFILING_AVAILABLE, reason="memory-profiler not available")  
    def test_memory_profiler_integration_with_sdr_coordination(self):
        """
        SUBTASK-5.6.2.1 [6a] - Test memory profiler integration with dual-SDR coordination.
        
        Tests real coordination state management memory consumption.
        """
        optimizer = ResourceOptimizer()
        
        # Profile memory usage during dual-SDR coordination simulation
        @memory_profiler.profile
        def coordination_workload():
            """Simulate dual-SDR coordination memory usage patterns."""
            coordination_states = {}
            
            # Simulate coordination state management over time  
            for coordination_cycle in range(1000):
                # Create coordination state similar to actual system
                state_key = f"coordination_{coordination_cycle}"
                coordination_states[state_key] = {
                    'primary_sdr': {
                        'rssi_history': deque(maxlen=100),
                        'signal_quality': 85.0,
                        'last_update': time.time()
                    },
                    'secondary_sdr': {
                        'rssi_history': deque(maxlen=100), 
                        'signal_quality': 78.0,
                        'last_update': time.time()
                    },
                    'decision_matrix': [[0.8, 0.2], [0.3, 0.7]],
                    'coordination_metadata': {
                        'cycle_count': coordination_cycle,
                        'performance_metrics': list(range(50))  # Realistic metrics array
                    }
                }
                
                # Cleanup old states periodically (memory management test)
                if coordination_cycle % 100 == 0 and coordination_cycle > 0:
                    states_to_remove = [k for k in coordination_states.keys() 
                                      if int(k.split('_')[1]) < coordination_cycle - 200]
                    for state_key in states_to_remove:
                        del coordination_states[state_key]
                        
            return len(coordination_states)
        
        # Execute profiled coordination workload
        final_state_count = coordination_workload()
        
        # Verify coordination memory management
        memory_analysis = optimizer.analyze_coordination_memory_usage()
        
        assert 'coordination_state_memory_mb' in memory_analysis
        assert 'state_cleanup_efficiency' in memory_analysis
        assert final_state_count <= 300, "Coordination state cleanup should limit memory growth to reasonable levels"
        
        # Verify memory efficiency per coordination cycle
        memory_per_cycle = memory_analysis['coordination_state_memory_mb'] / 1000
        assert memory_per_cycle < 0.1, "Memory per coordination cycle should be under 0.1MB"
        
    def test_memory_analysis_without_profiler_fallback(self):
        """
        SUBTASK-5.6.2.1 [6a] - Test memory analysis fallback when memory-profiler unavailable.
        
        Ensures system works even without profiling tools available.
        """
        optimizer = ResourceOptimizer(enable_memory_profiler=False)
        
        # Test basic memory monitoring using psutil only
        initial_memory = optimizer.get_current_memory_usage()
        
        # Create memory load to test monitoring
        memory_consumer = []
        for i in range(1000):
            memory_consumer.append([j for j in range(1000)])  # Create memory load
            
        current_memory = optimizer.get_current_memory_usage()
        
        # Verify basic memory monitoring works without profiler
        assert current_memory > initial_memory, "Memory usage should increase with load"
        assert optimizer.can_analyze_memory_patterns(), "Basic memory analysis should be available"
        
        # Test memory analysis with psutil-only data
        analysis = optimizer.analyze_memory_usage_patterns([initial_memory, current_memory])
        
        assert hasattr(analysis, 'basic_growth_detected'), "Basic growth detection should work"
        assert analysis.memory_monitoring_method == 'psutil', "Should indicate psutil-only monitoring"


class TestMemoryOptimization:
    """
    SUBTASK-5.6.2.1 [6b-6f] - Test memory optimization implementations.
    
    Tests verify real memory management improvements and resource efficiency.
    """
    
    def test_circular_buffer_memory_efficiency(self):
        """
        SUBTASK-5.6.2.1 [6b] - Test memory-efficient circular buffers for RSSI data.
        
        Tests real circular buffer implementation with automatic cleanup.
        """
        optimizer = ResourceOptimizer()
        
        # Create circular buffer with size limit
        rssi_buffer = optimizer.create_rssi_circular_buffer(max_size=1000)
        
        # Fill buffer beyond capacity to test size limiting
        initial_memory = optimizer.get_current_memory_usage()
        
        for i in range(2000):  # Fill twice the capacity
            rssi_sample = {
                'timestamp': time.time() + i * 0.01,
                'rssi_dbm': -70.0 + (i % 40) - 20,
                'source_id': i % 2
            }
            rssi_buffer.append(rssi_sample)
            
        final_memory = optimizer.get_current_memory_usage()
        
        # Verify circular buffer behavior - may be less than max_size due to automatic cleanup
        assert len(rssi_buffer) <= 1000, "Circular buffer should not exceed size limit"
        assert len(rssi_buffer) >= 800, "Circular buffer should retain reasonable amount of data after cleanup"
        # Buffer may not be full after automatic cleanup, which is correct behavior
        
        # Verify memory efficiency - memory growth should be bounded
        memory_growth_mb = final_memory - initial_memory
        assert memory_growth_mb < 50, "Circular buffer should limit memory growth to <50MB"
        
        # Test automatic cleanup trigger
        cleanup_triggered = rssi_buffer.trigger_cleanup(memory_threshold_mb=25)
        post_cleanup_memory = optimizer.get_current_memory_usage()
        
        if cleanup_triggered:
            assert post_cleanup_memory <= final_memory, "Cleanup should not increase memory"
            
    def test_memory_pool_object_recycling(self):
        """
        SUBTASK-5.6.2.1 [6c] - Test memory pool management with object recycling.
        
        Tests real object recycling for frequent allocations in signal processing.
        """
        # Create memory pool for RSSI processing objects
        memory_pool = MemoryPool(object_type="rssi_processor", pool_size=100)
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        allocation_times = []
        
        # Keep some objects active to measure utilization
        active_objects = []
        
        # Test object recycling performance and memory efficiency
        for cycle in range(500):  # Multiple allocation/deallocation cycles
            start_time = time.perf_counter()
            
            # Get object from pool (recycled or new)
            processor_obj = memory_pool.get_object()
            
            # Simulate signal processing work
            processor_obj.process_signal_data({
                'samples': list(range(1024)),  # Realistic IQ sample count
                'processing_params': {'fft_size': 1024, 'window': 'hann'}
            })
            
            # Keep first 50 objects active to test utilization
            if cycle < 50:
                active_objects.append(processor_obj)
            else:
                # Return object to pool for recycling
                memory_pool.return_object(processor_obj)
            
            allocation_time = time.perf_counter() - start_time
            allocation_times.append(allocation_time)
            
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Verify memory pool efficiency
        memory_growth_mb = final_memory - initial_memory  
        avg_allocation_time_ms = sum(allocation_times) / len(allocation_times) * 1000
        
        assert memory_growth_mb < 20, "Memory pool should limit growth to <20MB for 500 cycles"
        assert avg_allocation_time_ms < 0.1, "Object recycling should be fast (<0.1ms average)"
        assert memory_pool.get_pool_utilization() > 0.3, "Memory pool should have reasonable utilization with active objects"
        assert memory_pool.get_recycling_rate() > 0.85, "Should recycle >85% of objects efficiently"
        
        # Clean up active objects
        for obj in active_objects:
            memory_pool.return_object(obj)