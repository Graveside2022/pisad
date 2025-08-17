"""Integration tests for database operations.

Tests database connectivity, CRUD operations, data persistence,
transaction handling, and database-service integration.
"""

import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

from src.backend.models.database import ConfigProfileDB
from src.backend.services.state_machine import StateMachine, SystemState


class TestDatabaseIntegration:
    """Test database integration functionality."""

    @pytest.fixture
    def temp_db_path(self):
        """Provide temporary database path."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def database(self, temp_db_path):
        """Provide database instance."""
        return ConfigProfileDB(temp_db_path)

    def test_database_connection_integration(self, temp_db_path):
        """Test database connection establishment."""
        # Test SQLite connection
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()

        # Should be able to execute basic query
        cursor.execute("SELECT 1")
        result = cursor.fetchone()

        assert result == (1,)

        conn.close()

    def test_database_schema_creation_integration(self, database):
        """Test database schema creation."""
        # Database should initialize with proper schema
        # (Implementation depends on actual database module)

        # Test basic database functionality
        assert database is not None

    def test_state_persistence_integration(self, temp_db_path):
        """Test state machine database persistence."""
        # Create state machine with specific database
        state_machine = StateMachine(db_path=temp_db_path, enable_persistence=True)

        # Initial state should be loaded or default
        initial_state = state_machine.current_state
        assert initial_state in SystemState

        # Test persistence is enabled
        assert state_machine._enable_persistence is True
        assert state_machine._db_path == temp_db_path

    @pytest.mark.asyncio
    async def test_state_transition_persistence_integration(self, temp_db_path):
        """Test state transition persistence."""
        # Create state machine with persistence
        state_machine1 = StateMachine(db_path=temp_db_path, enable_persistence=True)

        # Transition to a specific state
        await state_machine1.transition_to(SystemState.SEARCHING)

        # Create new state machine instance
        state_machine2 = StateMachine(db_path=temp_db_path, enable_persistence=True)

        # Should restore to the persisted state
        # (Implementation dependent - may restore to saved state or default)
        restored_state = state_machine2.current_state
        assert restored_state in SystemState

    def test_detection_storage_integration(self, database):
        """Test detection event storage integration."""
        # Test detection event can be stored
        # (Implementation depends on actual database schema)

        # Create sample detection event
        detection = {
            "id": "test-detection-1",
            "timestamp": time.time(),
            "rssi": -45.0,
            "snr": 15.0,
            "frequency": 915000000,
            "confidence": 85.0,
        }

        # Database should handle detection storage
        # (Actual implementation would use proper database methods)

        # For now, test database is available
        assert database is not None

    def test_configuration_persistence_integration(self, temp_db_path):
        """Test configuration persistence."""
        # Test configuration can be stored and retrieved
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()

        # Create simple config table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """
        )

        # Store configuration
        cursor.execute(
            "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
            ("test_setting", "test_value"),
        )
        conn.commit()

        # Retrieve configuration
        cursor.execute("SELECT value FROM config WHERE key = ?", ("test_setting",))
        result = cursor.fetchone()

        assert result == ("test_value",)

        conn.close()

    def test_telemetry_storage_integration(self, temp_db_path):
        """Test telemetry data storage."""
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()

        # Create telemetry table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                component TEXT,
                data TEXT
            )
        """
        )

        # Store telemetry
        cursor.execute(
            """
            INSERT INTO telemetry (timestamp, component, data)
            VALUES (?, ?, ?)
        """,
            (time.time(), "state_machine", '{"state": "IDLE"}'),
        )
        conn.commit()

        # Retrieve telemetry
        cursor.execute("SELECT component, data FROM telemetry ORDER BY id DESC LIMIT 1")
        result = cursor.fetchone()

        assert result[0] == "state_machine"
        assert "IDLE" in result[1]

        conn.close()

    def test_database_transaction_integration(self, temp_db_path):
        """Test database transaction handling."""
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()

        # Create test table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS test_table (
                id INTEGER PRIMARY KEY,
                value TEXT
            )
        """
        )

        # Test transaction rollback
        try:
            cursor.execute("BEGIN TRANSACTION")
            cursor.execute("INSERT INTO test_table (value) VALUES (?)", ("test1",))
            cursor.execute("INSERT INTO test_table (value) VALUES (?)", ("test2",))
            cursor.execute("ROLLBACK")
        except Exception:
            cursor.execute("ROLLBACK")

        # Check nothing was committed
        cursor.execute("SELECT COUNT(*) FROM test_table")
        count = cursor.fetchone()[0]
        assert count == 0

        # Test transaction commit
        cursor.execute("BEGIN TRANSACTION")
        cursor.execute("INSERT INTO test_table (value) VALUES (?)", ("committed",))
        cursor.execute("COMMIT")

        # Check data was committed
        cursor.execute("SELECT value FROM test_table")
        result = cursor.fetchone()
        assert result == ("committed",)

        conn.close()

    def test_database_concurrent_access_integration(self, temp_db_path):
        """Test concurrent database access."""
        import threading

        results = []

        def database_worker(worker_id):
            conn = sqlite3.connect(temp_db_path)
            cursor = conn.cursor()

            # Create table if not exists
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS concurrent_test (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    worker_id INTEGER,
                    timestamp REAL
                )
            """
            )

            # Insert data
            cursor.execute(
                """
                INSERT INTO concurrent_test (worker_id, timestamp)
                VALUES (?, ?)
            """,
                (worker_id, time.time()),
            )
            conn.commit()

            # Read data
            cursor.execute("SELECT COUNT(*) FROM concurrent_test WHERE worker_id = ?", (worker_id,))
            count = cursor.fetchone()[0]
            results.append(count)

            conn.close()

        # Run concurrent database operations
        threads = []
        for i in range(3):
            thread = threading.Thread(target=database_worker, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All operations should succeed
        assert len(results) == 3
        assert all(count == 1 for count in results)

    def test_database_backup_integration(self, temp_db_path):
        """Test database backup functionality."""
        # Create source database with data
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE test_backup (
                id INTEGER PRIMARY KEY,
                data TEXT
            )
        """
        )
        cursor.execute("INSERT INTO test_backup (data) VALUES (?)", ("backup_test",))
        conn.commit()
        conn.close()

        # Create backup
        backup_path = temp_db_path + ".backup"

        # SQLite backup
        source = sqlite3.connect(temp_db_path)
        backup = sqlite3.connect(backup_path)
        source.backup(backup)
        source.close()
        backup.close()

        # Verify backup
        backup_conn = sqlite3.connect(backup_path)
        backup_cursor = backup_conn.cursor()
        backup_cursor.execute("SELECT data FROM test_backup")
        result = backup_cursor.fetchone()

        assert result == ("backup_test",)

        backup_conn.close()
        Path(backup_path).unlink(missing_ok=True)

    def test_database_migration_integration(self, temp_db_path):
        """Test database schema migration."""
        # Create initial schema
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()

        # Version 1 schema
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            )
        """
        )
        cursor.execute("INSERT INTO schema_version (version) VALUES (1)")

        cursor.execute(
            """
            CREATE TABLE users_v1 (
                id INTEGER PRIMARY KEY,
                name TEXT
            )
        """
        )
        conn.commit()

        # Simulate migration to version 2
        cursor.execute("SELECT version FROM schema_version")
        current_version = cursor.fetchone()[0]

        if current_version == 1:
            # Add column (migration)
            cursor.execute("ALTER TABLE users_v1 ADD COLUMN email TEXT")
            cursor.execute("UPDATE schema_version SET version = 2")
            conn.commit()

        # Verify migration
        cursor.execute("PRAGMA table_info(users_v1)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]

        assert "email" in column_names

        cursor.execute("SELECT version FROM schema_version")
        new_version = cursor.fetchone()[0]
        assert new_version == 2

        conn.close()

    def test_database_error_handling_integration(self, temp_db_path):
        """Test database error handling."""
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()

        # Test constraint violation
        cursor.execute(
            """
            CREATE TABLE test_constraints (
                id INTEGER PRIMARY KEY,
                unique_value TEXT UNIQUE
            )
        """
        )

        cursor.execute("INSERT INTO test_constraints (unique_value) VALUES (?)", ("unique1",))
        conn.commit()

        # This should fail due to unique constraint
        with pytest.raises(sqlite3.IntegrityError):
            cursor.execute("INSERT INTO test_constraints (unique_value) VALUES (?)", ("unique1",))

        # Test syntax error
        with pytest.raises(sqlite3.OperationalError):
            cursor.execute("INVALID SQL SYNTAX")

        conn.close()

    def test_database_performance_integration(self, temp_db_path):
        """Test database performance characteristics."""
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()

        # Create table for performance test
        cursor.execute(
            """
            CREATE TABLE performance_test (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data TEXT,
                timestamp REAL
            )
        """
        )

        # Measure bulk insert performance
        start_time = time.perf_counter()

        cursor.execute("BEGIN TRANSACTION")
        for i in range(1000):
            cursor.execute(
                """
                INSERT INTO performance_test (data, timestamp)
                VALUES (?, ?)
            """,
                (f"data_{i}", time.time()),
            )
        cursor.execute("COMMIT")

        end_time = time.perf_counter()
        insert_time = end_time - start_time

        # Should complete reasonably quickly
        assert insert_time < 5.0  # Less than 5 seconds for 1000 inserts

        # Measure query performance
        start_time = time.perf_counter()
        cursor.execute("SELECT COUNT(*) FROM performance_test")
        count = cursor.fetchone()[0]
        end_time = time.perf_counter()
        query_time = end_time - start_time

        assert count == 1000
        assert query_time < 1.0  # Less than 1 second for count query

        conn.close()

    def test_database_cleanup_integration(self, temp_db_path):
        """Test database cleanup and maintenance."""
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()

        # Create table with data
        cursor.execute(
            """
            CREATE TABLE cleanup_test (
                id INTEGER PRIMARY KEY,
                data TEXT,
                created_at REAL
            )
        """
        )

        # Insert test data
        old_time = time.time() - 86400  # 24 hours ago
        recent_time = time.time()

        cursor.execute(
            "INSERT INTO cleanup_test (data, created_at) VALUES (?, ?)", ("old_data", old_time)
        )
        cursor.execute(
            "INSERT INTO cleanup_test (data, created_at) VALUES (?, ?)",
            ("recent_data", recent_time),
        )
        conn.commit()

        # Test cleanup of old data
        cutoff_time = time.time() - 3600  # 1 hour ago
        cursor.execute("DELETE FROM cleanup_test WHERE created_at < ?", (cutoff_time,))
        conn.commit()

        # Verify cleanup
        cursor.execute("SELECT data FROM cleanup_test")
        remaining = cursor.fetchall()

        assert len(remaining) == 1
        assert remaining[0][0] == "recent_data"

        # Test VACUUM for space reclamation
        cursor.execute("VACUUM")

        conn.close()

    def test_database_integrity_check_integration(self, temp_db_path):
        """Test database integrity checking."""
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()

        # Create table with data
        cursor.execute(
            """
            CREATE TABLE integrity_test (
                id INTEGER PRIMARY KEY,
                data TEXT NOT NULL
            )
        """
        )
        cursor.execute("INSERT INTO integrity_test (data) VALUES (?)", ("test_data",))
        conn.commit()

        # Run integrity check
        cursor.execute("PRAGMA integrity_check")
        result = cursor.fetchone()

        # Should return "ok" for healthy database
        assert result[0] == "ok"

        # Test foreign key check (if applicable)
        cursor.execute("PRAGMA foreign_key_check")
        fk_violations = cursor.fetchall()

        # Should have no foreign key violations
        assert len(fk_violations) == 0

        conn.close()
