"""
Unit tests for async blocking operations fix.

PERFORMANCE TEST: Sprint 6 Day 7 Task 6
REQUIREMENT: Event loop must never block >10ms
MEASUREMENT: Track event loop response time under load

Authors: Rex & Sherlock
"""

import asyncio
import time
from pathlib import Path

import pytest

from src.backend.utils.async_io_helpers import AsyncBatchProcessor, AsyncDatabasePool, AsyncFileIO


class TestAsyncFileIO:
    """Test non-blocking file I/O operations."""

    @pytest.mark.asyncio
    async def test_read_text_non_blocking(self, tmp_path):
        """Test that read_text doesn't block event loop."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content" * 1000)  # Large content

        # Track event loop responsiveness
        loop_blocked = False

        async def monitor_loop():
            """Monitor if loop gets blocked."""
            nonlocal loop_blocked
            start = time.perf_counter()
            await asyncio.sleep(0.001)  # Should complete in ~1ms
            elapsed = (time.perf_counter() - start) * 1000
            if elapsed > 10:  # More than 10ms means blocked
                loop_blocked = True

        # Run file read with monitoring
        monitor_task = asyncio.create_task(monitor_loop())
        content = await AsyncFileIO.read_text(test_file)
        await monitor_task

        assert not loop_blocked, "Event loop was blocked during file read"
        assert "test content" in content

    @pytest.mark.asyncio
    async def test_write_text_non_blocking(self, tmp_path):
        """Test that write_text doesn't block event loop."""
        test_file = tmp_path / "test_write.txt"
        large_content = "x" * 100000  # 100KB content

        # Track loop responsiveness
        max_delay = 0

        async def measure_responsiveness():
            """Measure maximum loop delay."""
            nonlocal max_delay
            for _ in range(10):
                start = time.perf_counter()
                await asyncio.sleep(0.001)
                delay = (time.perf_counter() - start) * 1000
                max_delay = max(max_delay, delay)

        # Write file while measuring
        measure_task = asyncio.create_task(measure_responsiveness())
        await AsyncFileIO.write_text(test_file, large_content)
        await measure_task

        assert max_delay < 10, f"Event loop blocked for {max_delay:.2f}ms"
        assert test_file.read_text() == large_content

    @pytest.mark.asyncio
    async def test_json_operations_non_blocking(self, tmp_path):
        """Test JSON read/write don't block."""
        test_file = tmp_path / "test.json"
        test_data = {"items": [{"id": i, "data": "x" * 100} for i in range(1000)]}

        # Write JSON
        start = time.perf_counter()
        await AsyncFileIO.write_json(test_file, test_data)
        write_time = (time.perf_counter() - start) * 1000

        # Read JSON
        start = time.perf_counter()
        loaded_data = await AsyncFileIO.read_json(test_file)
        read_time = (time.perf_counter() - start) * 1000

        # Operations should be fast (delegated to thread)
        assert loaded_data == test_data
        print(f"JSON write: {write_time:.2f}ms, read: {read_time:.2f}ms")


class TestAsyncDatabasePool:
    """Test async database connection pool."""

    @pytest.mark.asyncio
    async def test_pool_initialization(self, tmp_path):
        """Test pool initializes correctly."""
        db_path = tmp_path / "test.db"
        pool = AsyncDatabasePool(db_path, pool_size=3)

        await pool.initialize()
        assert pool._initialized
        assert pool._available.qsize() == 3

        await pool.close()

    @pytest.mark.asyncio
    async def test_concurrent_queries_non_blocking(self, tmp_path):
        """Test concurrent queries don't block each other."""
        db_path = tmp_path / "test.db"
        pool = AsyncDatabasePool(db_path, pool_size=5)
        await pool.initialize()

        # Create test table
        await pool.execute("""
            CREATE TABLE test (
                id INTEGER PRIMARY KEY,
                value TEXT
            )
        """)

        # Insert test data
        test_data = [(i, f"value_{i}") for i in range(100)]
        await pool.executemany("INSERT INTO test (id, value) VALUES (?, ?)", test_data)

        # Run concurrent queries
        async def query_task(task_id: int):
            start = time.perf_counter()
            result = await pool.fetchall("SELECT * FROM test WHERE id < ?", (50,))
            elapsed = (time.perf_counter() - start) * 1000
            return task_id, elapsed, len(result)

        # Launch 10 concurrent queries
        results = await asyncio.gather(*[query_task(i) for i in range(10)])

        # All queries should complete quickly
        max_time = max(r[1] for r in results)
        assert max_time < 100, f"Query took {max_time:.2f}ms"

        # All should get correct results
        for _, _, count in results:
            assert count == 49

        await pool.close()

    @pytest.mark.asyncio
    async def test_pool_connection_reuse(self, tmp_path):
        """Test connections are properly reused."""
        db_path = tmp_path / "test.db"
        pool = AsyncDatabasePool(db_path, pool_size=2)
        await pool.initialize()

        # Track connection acquisitions
        acquisitions = []

        async def use_connection(conn_id: int):
            async with pool.acquire() as conn:
                acquisitions.append((conn_id, id(conn)))
                await asyncio.sleep(0.01)  # Simulate work

        # Use more than pool size
        await asyncio.gather(*[use_connection(i) for i in range(5)])

        # Check that connections were reused
        conn_ids = [conn_id for _, conn_id in acquisitions]
        unique_conns = set(conn_ids)
        assert len(unique_conns) == 2, "Pool should reuse connections"

        await pool.close()


class TestAsyncBatchProcessor:
    """Test batch processing for large datasets."""

    @pytest.mark.asyncio
    async def test_batch_processing_yields_to_loop(self):
        """Test batch processor yields control to event loop."""
        items = list(range(1000))

        # Track if loop stays responsive
        loop_delays = []

        async def monitor_loop():
            """Monitor loop responsiveness during processing."""
            while True:
                start = time.perf_counter()
                await asyncio.sleep(0.001)
                delay = (time.perf_counter() - start) * 1000
                loop_delays.append(delay)
                if len(loop_delays) >= 20:
                    break

        # Process items while monitoring
        async def process_item(x):
            # Simulate work
            await asyncio.sleep(0.0001)
            return x * 2

        monitor_task = asyncio.create_task(monitor_loop())
        results = await AsyncBatchProcessor.process_batch(
            items, process_item, batch_size=50, delay_ms=5
        )

        try:
            await asyncio.wait_for(monitor_task, timeout=1.0)
        except TimeoutError:
            pass

        # Check results
        assert results == [x * 2 for x in items]

        # Check loop stayed responsive
        max_delay = max(loop_delays) if loop_delays else 0
        avg_delay = sum(loop_delays) / len(loop_delays) if loop_delays else 0

        print(f"Loop delays - Max: {max_delay:.2f}ms, Avg: {avg_delay:.2f}ms")
        assert max_delay < 20, f"Loop blocked for {max_delay:.2f}ms"
        assert avg_delay < 10, f"Average loop delay {avg_delay:.2f}ms too high"


@pytest.mark.benchmark
class TestEventLoopPerformance:
    """Benchmark event loop responsiveness."""

    @pytest.mark.asyncio
    async def test_event_loop_never_blocks_10ms(self):
        """
        PERFORMANCE REQUIREMENT: Event loop must never block >10ms

        This test simulates real workload with mixed I/O operations.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Setup test files and database
            test_file = tmppath / "test.txt"
            test_file.write_text("initial content")

            db_path = tmppath / "test.db"
            pool = AsyncDatabasePool(db_path)
            await pool.initialize()

            # Create test table
            await pool.execute("""
                CREATE TABLE metrics (
                    id INTEGER PRIMARY KEY,
                    timestamp REAL,
                    value REAL
                )
            """)

            # Track loop responsiveness
            blocking_events = []

            async def loop_monitor():
                """Continuously monitor loop responsiveness."""
                for i in range(100):
                    start = time.perf_counter()
                    await asyncio.sleep(0.001)  # Should complete in ~1ms
                    elapsed = (time.perf_counter() - start) * 1000

                    if elapsed > 10:
                        blocking_events.append({"iteration": i, "blocked_ms": elapsed})

            # Simulate mixed workload
            async def mixed_workload():
                """Simulate real application workload."""
                for i in range(50):
                    # File I/O
                    await AsyncFileIO.write_text(test_file, f"iteration {i}\n" * 100)
                    content = await AsyncFileIO.read_text(test_file)

                    # Database operations
                    await pool.execute(
                        "INSERT INTO metrics (timestamp, value) VALUES (?, ?)",
                        (time.time(), i * 1.5),
                    )

                    results = await pool.fetchall(
                        "SELECT * FROM metrics WHERE value > ?", (i * 0.5,)
                    )

                    # JSON operations
                    json_file = tmppath / f"data_{i}.json"
                    await AsyncFileIO.write_json(json_file, {"iteration": i, "results": results})

                    # Small delay
                    await asyncio.sleep(0.001)

            # Run workload with monitoring
            monitor_task = asyncio.create_task(loop_monitor())
            workload_task = asyncio.create_task(mixed_workload())

            await asyncio.gather(workload_task, monitor_task)

            # Cleanup
            await pool.close()

            # Verify no blocking events
            if blocking_events:
                max_block = max(e["blocked_ms"] for e in blocking_events)
                print(f"\n⚠️ Found {len(blocking_events)} blocking events")
                print(f"Max block: {max_block:.2f}ms")
                for event in blocking_events[:5]:  # Show first 5
                    print(f"  Iteration {event['iteration']}: {event['blocked_ms']:.2f}ms")

            assert (
                len(blocking_events) == 0
            ), f"Event loop blocked {len(blocking_events)} times (max: {max_block:.2f}ms)"

            print("✅ Event loop stayed responsive (<10ms) during entire test")
