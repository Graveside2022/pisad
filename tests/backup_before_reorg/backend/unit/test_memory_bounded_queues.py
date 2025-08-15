"""
Test Memory Usage with Bounded Queues

BACKWARDS ANALYSIS:
- User Action: System processes signals at 100Hz continuously
- Expected Result: Memory usage remains stable without growth
- Failure Impact: Memory exhaustion crashes Pi after ~1 hour

REQUIREMENT TRACE:
- User Story: 4.9 - Task 7 (Implement Bounded Queues)

TEST VALUE: Prevents 2.9GB memory leak in 1 hour at 100Hz
"""

import asyncio
import gc
import time

import numpy as np
import psutil
import pytest

from src.backend.services.signal_processor import SignalProcessor


class TestBoundedQueueMemory:
    """Test that bounded queues prevent memory leaks."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)  # 10 second timeout for safety
    async def test_memory_stable_at_100hz(self):
        """Verify memory doesn't grow unbounded at 100Hz processing rate."""

        # Get initial memory usage
        process = psutil.Process()
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create processor with bounded queues
        processor = SignalProcessor()

        # Add some callbacks to simulate real usage
        def rssi_callback(value):
            # Simulate some processing
            _ = value * 2

        def snr_callback(value):
            # Simulate some processing
            _ = value + 10

        processor.add_rssi_callback(rssi_callback)
        processor.add_snr_callback(snr_callback)

        # Process at 100Hz for 5 seconds (500 samples)
        samples_processed = 0
        start_time = time.time()
        target_rate = 100  # Hz
        interval = 1.0 / target_rate

        while samples_processed < 500:
            # Generate IQ samples
            samples = np.random.randn(1024) + 1j * np.random.randn(1024)

            # Process samples
            await processor.process_iq(samples)

            samples_processed += 1

            # Maintain 100Hz rate
            elapsed = time.time() - start_time
            expected_time = samples_processed * interval
            if expected_time > elapsed:
                await asyncio.sleep(expected_time - elapsed)

        # Check memory after processing
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory

        # Log results
        print(f"Initial memory: {initial_memory:.1f} MB")
        print(f"Final memory: {final_memory:.1f} MB")
        print(f"Memory growth: {memory_growth:.1f} MB")
        print(f"Samples processed: {samples_processed}")
        print(f"Processing rate: {samples_processed / (time.time() - start_time):.1f} Hz")

        # Memory should not grow more than 10MB for 500 samples
        # Without bounded queues, it would grow by ~40MB
        assert memory_growth < 10, f"Memory grew by {memory_growth:.1f} MB (should be <10 MB)"

    @pytest.mark.asyncio
    async def test_queue_backpressure(self):
        """Test that bounded queues handle backpressure correctly."""

        # Create processor with known queue size
        processor = SignalProcessor()

        # Verify the IQ queue has maxsize set
        assert processor.iq_queue.maxsize == 100

        # Fill the queue to capacity
        samples = np.random.randn(1024) + 1j * np.random.randn(1024)

        # Queue should handle backpressure without growing unbounded
        tasks = []
        for i in range(150):  # Try to add more than maxsize
            # These should queue up but not cause memory explosion
            task = asyncio.create_task(processor.iq_queue.put(samples))
            tasks.append(task)

        # Let some tasks complete
        await asyncio.sleep(0.1)

        # Cancel remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Queue size should never exceed maxsize
        assert processor.iq_queue.qsize() <= processor.iq_queue.maxsize

    def test_all_queues_bounded(self):
        """Verify all AsyncIO queues in the system are bounded."""

        # Check command pipeline queue
        from src.backend.services.command_pipeline import CommandPipeline

        pipeline = CommandPipeline(mavlink_service=None)
        assert pipeline.command_queue.maxsize > 0, "Command queue should be bounded"

        # Check signal processor queue
        processor = SignalProcessor()
        assert processor.iq_queue.maxsize > 0, "IQ queue should be bounded"

        # Check async database pool queue
        from src.backend.utils.async_io_helpers import AsyncDatabasePool

        pool = AsyncDatabasePool("test.db", pool_size=5)
        assert pool._available.maxsize > 0, "Database pool queue should be bounded"

        print("✅ All queues are properly bounded")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
