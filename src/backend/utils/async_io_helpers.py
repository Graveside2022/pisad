"""
Async I/O helpers for non-blocking file and database operations.

PERFORMANCE OPTIMIZATION: Story 4.9 Sprint 6 Task 6
PROBLEM: Blocking I/O in async functions blocks event loop
SOLUTION: Use asyncio.to_thread for file I/O and async database pool

Author: Rex & Sherlock
Date: 2025-08-15
"""

import asyncio
import csv
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)


class AsyncFileIO:
    """Non-blocking file I/O operations for async contexts."""

    @staticmethod
    async def read_text(path: Path | str) -> str:
        """Read text file without blocking event loop.

        Performance: Offloads to thread pool, event loop stays responsive.
        """

        def _read():
            with open(path) as f:
                return f.read()

        return await asyncio.to_thread(_read)

    @staticmethod
    async def write_text(path: Path | str, content: str) -> None:
        """Write text file without blocking event loop."""

        def _write():
            with open(path, "w") as f:
                f.write(content)

        await asyncio.to_thread(_write)

    @staticmethod
    async def read_json(path: Path | str) -> dict[str, Any]:
        """Read JSON file without blocking."""

        def _read_json():
            with open(path) as f:
                return json.load(f)

        return await asyncio.to_thread(_read_json)

    @staticmethod
    async def write_json(path: Path | str, data: dict[str, Any], indent: int = 2) -> None:
        """Write JSON file without blocking."""

        def _write_json():
            with open(path, "w") as f:
                json.dump(data, f, indent=indent, default=str)

        await asyncio.to_thread(_write_json)

    @staticmethod
    async def write_csv(
        path: Path | str, rows: list[dict[str, Any]], fieldnames: list[str]
    ) -> None:
        """Write CSV file without blocking."""

        def _write_csv():
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        await asyncio.to_thread(_write_csv)

    @staticmethod
    async def append_text(path: Path | str, content: str) -> None:
        """Append text to file without blocking."""

        def _append():
            with open(path, "a") as f:
                f.write(content)

        await asyncio.to_thread(_append)


class AsyncDatabasePool:
    """
    Async database connection pool for non-blocking queries.

    PERFORMANCE:
    - Connection pooling reduces overhead
    - Async queries don't block event loop
    - Prepared statements prevent SQL injection
    """

    def __init__(self, db_path: Path | str, pool_size: int = 5):
        """Initialize async database pool.

        Args:
            db_path: Path to SQLite database
            pool_size: Number of connections in pool
        """
        self.db_path = str(db_path)
        self.pool_size = pool_size
        self._pool: list[aiosqlite.Connection] = []
        self._available: asyncio.Queue = asyncio.Queue(maxsize=pool_size)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize connection pool."""
        if self._initialized:
            return

        for _ in range(self.pool_size):
            conn = await aiosqlite.connect(self.db_path)
            conn.row_factory = aiosqlite.Row
            self._pool.append(conn)
            await self._available.put(conn)

        self._initialized = True
        logger.info(f"Initialized async database pool with {self.pool_size} connections")

    async def close(self) -> None:
        """Close all connections in pool."""
        for conn in self._pool:
            await conn.close()
        self._pool.clear()
        self._initialized = False

    @asynccontextmanager
    async def acquire(self):
        """Acquire connection from pool."""
        if not self._initialized:
            await self.initialize()

        conn = await self._available.get()
        try:
            yield conn
        finally:
            await self._available.put(conn)

    async def execute(self, query: str, params: tuple = ()) -> None:
        """Execute query without returning results."""
        async with self.acquire() as conn:
            await conn.execute(query, params)
            await conn.commit()

    async def fetchone(self, query: str, params: tuple = ()) -> dict[str, Any] | None:
        """Fetch single row."""
        async with self.acquire() as conn:
            async with conn.execute(query, params) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None

    async def fetchall(self, query: str, params: tuple = ()) -> list[dict[str, Any]]:
        """Fetch all rows."""
        async with self.acquire() as conn:
            async with conn.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    async def executemany(self, query: str, params_list: list[tuple]) -> None:
        """Execute query multiple times with different parameters."""
        async with self.acquire() as conn:
            await conn.executemany(query, params_list)
            await conn.commit()


class AsyncBatchProcessor:
    """
    Process items in batches to avoid blocking event loop.

    Use for processing large datasets without blocking.
    """

    @staticmethod
    async def process_batch(
        items: list[Any], processor: callable, batch_size: int = 100, delay_ms: int = 10
    ) -> list[Any]:
        """Process items in batches with delays.

        Args:
            items: Items to process
            processor: Async function to process each item
            batch_size: Items per batch
            delay_ms: Delay between batches (ms)

        Returns:
            Processed results
        """
        results = []

        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]

            # Process batch
            if asyncio.iscoroutinefunction(processor):
                batch_results = await asyncio.gather(*[processor(item) for item in batch])
            else:
                # Run sync function in thread
                batch_results = await asyncio.gather(
                    *[asyncio.to_thread(processor, item) for item in batch]
                )

            results.extend(batch_results)

            # Yield to event loop
            if i + batch_size < len(items):
                await asyncio.sleep(delay_ms / 1000)

        return results


# Singleton database pool instance
_db_pool: AsyncDatabasePool | None = None


async def get_db_pool(db_path: Path | str) -> AsyncDatabasePool:
    """Get or create singleton database pool.

    Args:
        db_path: Path to database

    Returns:
        Async database pool instance
    """
    global _db_pool

    if _db_pool is None:
        _db_pool = AsyncDatabasePool(db_path)
        await _db_pool.initialize()

    return _db_pool


async def close_db_pool() -> None:
    """Close singleton database pool."""
    global _db_pool

    if _db_pool:
        await _db_pool.close()
        _db_pool = None
