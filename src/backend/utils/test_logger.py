"""Test result logging and archival system.

Provides a centralized system for logging, storing, and retrieving
test results with full traceability and audit trail support.
"""

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any


class TestType(Enum):
    """Types of tests."""

    UNIT = "unit"
    INTEGRATION = "integration"
    SITL = "sitl"
    HIL = "hil"
    E2E = "e2e"
    SAFETY = "safety"
    PERFORMANCE = "performance"
    FIELD = "field"


class TestStatus(Enum):
    """Test execution status."""

    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"
    TIMEOUT = "timeout"


@dataclass
class TestResult:
    """Individual test result."""

    test_name: str
    test_type: TestType
    status: TestStatus
    duration_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    error_message: str | None = None
    stack_trace: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestRun:
    """Complete test run with multiple results."""

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    test_type: TestType = TestType.UNIT
    environment: str = "development"
    system_config: dict[str, Any] = field(default_factory=dict)
    results: list[TestResult] = field(default_factory=list)
    total_duration_ms: float = 0.0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0

    def add_result(self, result: TestResult) -> None:
        """Add a test result to the run.

        Args:
            result: Test result to add
        """
        self.results.append(result)
        self.total_duration_ms += result.duration_ms

        if result.status == TestStatus.PASS:
            self.passed += 1
        elif result.status == TestStatus.FAIL:
            self.failed += 1
        elif result.status == TestStatus.ERROR:
            self.errors += 1
        elif result.status == TestStatus.SKIP:
            self.skipped += 1

    def get_summary(self) -> dict[str, Any]:
        """Get test run summary.

        Returns:
            Summary dictionary
        """
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "test_type": self.test_type.value,
            "environment": self.environment,
            "total_tests": len(self.results),
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "total_duration_ms": self.total_duration_ms,
            "success_rate": ((self.passed / len(self.results) * 100) if self.results else 0),
        }


class TestLogger:
    """Test result logger with database storage."""

    def __init__(self, db_path: str = "test_results.db"):
        """Initialize test logger.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS test_runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    test_type TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    system_config TEXT,
                    total_tests INTEGER,
                    passed INTEGER,
                    failed INTEGER,
                    errors INTEGER,
                    skipped INTEGER,
                    total_duration_ms REAL,
                    success_rate REAL
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    test_name TEXT NOT NULL,
                    test_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    duration_ms REAL,
                    timestamp TEXT NOT NULL,
                    error_message TEXT,
                    stack_trace TEXT,
                    metadata TEXT,
                    FOREIGN KEY (run_id) REFERENCES test_runs(run_id)
                )
            """
            )

            conn.execute("CREATE INDEX IF NOT EXISTS idx_run_id ON test_results(run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON test_runs(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_test_type ON test_runs(test_type)")

    def log_test_run(self, test_run: TestRun) -> None:
        """Log a complete test run.

        Args:
            test_run: Test run to log
        """
        with sqlite3.connect(self.db_path) as conn:
            # Insert test run
            summary = test_run.get_summary()
            conn.execute(
                """
                INSERT INTO test_runs (
                    run_id, timestamp, test_type, environment,
                    system_config, total_tests, passed, failed,
                    errors, skipped, total_duration_ms, success_rate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    test_run.run_id,
                    test_run.timestamp.isoformat(),
                    test_run.test_type.value,
                    test_run.environment,
                    json.dumps(test_run.system_config),
                    len(test_run.results),
                    test_run.passed,
                    test_run.failed,
                    test_run.errors,
                    test_run.skipped,
                    test_run.total_duration_ms,
                    summary["success_rate"],
                ),
            )

            # Insert individual results
            for result in test_run.results:
                conn.execute(
                    """
                    INSERT INTO test_results (
                        run_id, test_name, test_type, status,
                        duration_ms, timestamp, error_message,
                        stack_trace, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        test_run.run_id,
                        result.test_name,
                        result.test_type.value,
                        result.status.value,
                        result.duration_ms,
                        result.timestamp.isoformat(),
                        result.error_message,
                        result.stack_trace,
                        json.dumps(result.metadata),
                    ),
                )

    def get_test_run(self, run_id: str) -> dict[str, Any] | None:
        """Get a specific test run.

        Args:
            run_id: Test run ID

        Returns:
            Test run data or None
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get run summary
            cursor.execute("SELECT run_id, timestamp, test_type, environment, system_config, total_tests, passed, failed, errors, skipped
                          FROM test_runs WHERE run_id = ?", (run_id,))
            run_row = cursor.fetchone()

            if not run_row:
                return None

            # Get test results
            cursor.execute(
                "SELECT id, run_id, test_name, test_type, status, duration_ms, timestamp, error_message, stack_trace, metadata
                 FROM test_results WHERE run_id = ? ORDER BY timestamp",
                (run_id,),
            )
            results = [dict(row) for row in cursor.fetchall()]

            return {
                "run": dict(run_row),
                "results": results,
            }

    def get_recent_runs(
        self, limit: int = 10, test_type: TestType | None = None
    ) -> list[dict[str, Any]]:
        """Get recent test runs.

        Args:
            limit: Maximum number of runs to return
            test_type: Filter by test type

        Returns:
            List of test run summaries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if test_type:
                cursor.execute(
                    """
                    SELECT run_id, timestamp, test_type, environment, system_config, total_tests, passed, failed, errors, skipped
                    FROM test_runs
                    WHERE test_type = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """,
                    (test_type.value, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT run_id, timestamp, test_type, environment, system_config, total_tests, passed, failed, errors, skipped
                    FROM test_runs
                    ORDER BY timestamp DESC
                    LIMIT ?
                """,
                    (limit,),
                )

            return [dict(row) for row in cursor.fetchall()]

    def get_statistics(self, since: datetime | None = None) -> dict[str, Any]:
        """Get test statistics.

        Args:
            since: Get statistics since this time

        Returns:
            Statistics dictionary
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            where_clause = ""
            params = []
            if since:
                where_clause = "WHERE timestamp >= ?"
                params.append(since.isoformat())

            # Overall statistics
            cursor.execute(
                f"""
                SELECT
                    COUNT(*) as total_runs,
                    SUM(total_tests) as total_tests,
                    SUM(passed) as total_passed,
                    SUM(failed) as total_failed,
                    SUM(errors) as total_errors,
                    AVG(success_rate) as avg_success_rate,
                    AVG(total_duration_ms) as avg_duration_ms
                FROM test_runs
                {where_clause}
            """,
                params,
            )
            overall = dict(cursor.fetchone())

            # By test type
            cursor.execute(
                f"""
                SELECT
                    test_type,
                    COUNT(*) as runs,
                    AVG(success_rate) as avg_success_rate
                FROM test_runs
                {where_clause}
                GROUP BY test_type
            """,
                params,
            )
            by_type = [dict(row) for row in cursor.fetchall()]

            return {
                "overall": overall,
                "by_type": by_type,
            }

    def export_to_json(self, run_id: str, output_path: str) -> None:
        """Export test run to JSON.

        Args:
            run_id: Test run ID
            output_path: Output file path
        """
        test_run = self.get_test_run(run_id)
        if test_run:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                json.dump(test_run, f, indent=2)

    def export_to_csv(self, run_id: str, output_path: str) -> None:
        """Export test run to CSV.

        Args:
            run_id: Test run ID
            output_path: Output file path
        """
        import csv

        test_run = self.get_test_run(run_id)
        if test_run:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)

            with open(output, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "test_name",
                        "test_type",
                        "status",
                        "duration_ms",
                        "timestamp",
                        "error_message",
                    ],
                )
                writer.writeheader()
                for result in test_run["results"]:
                    writer.writerow(
                        {
                            "test_name": result["test_name"],
                            "test_type": result["test_type"],
                            "status": result["status"],
                            "duration_ms": result["duration_ms"],
                            "timestamp": result["timestamp"],
                            "error_message": result.get("error_message", ""),
                        }
                    )

    def apply_retention_policy(self, days: int = 90) -> int:
        """Apply retention policy to remove old test results.

        Args:
            days: Number of days to retain

        Returns:
            Number of runs deleted
        """
        cutoff_date = datetime.now(UTC) - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get runs to delete
            cursor.execute(
                "SELECT run_id FROM test_runs WHERE timestamp < ?",
                (cutoff_date.isoformat(),),
            )
            runs_to_delete = [row[0] for row in cursor.fetchall()]

            if runs_to_delete:
                # Delete results first (foreign key)
                placeholders = ",".join("?" * len(runs_to_delete))
                cursor.execute(
                    f"DELETE FROM test_results WHERE run_id IN ({placeholders})",
                    runs_to_delete,
                )

                # Delete runs
                cursor.execute(
                    f"DELETE FROM test_runs WHERE run_id IN ({placeholders})",
                    runs_to_delete,
                )

        return len(runs_to_delete)


# Example usage and integration helpers
def create_test_run_from_pytest(pytest_report: Any) -> TestRun:
    """Create TestRun from pytest report.

    Args:
        pytest_report: Pytest report object

    Returns:
        TestRun instance
    """
    test_run = TestRun(
        test_type=TestType.UNIT,
        environment="ci" if os.getenv("CI") else "local",
    )

    # Parse pytest report and add results
    # This is a simplified example
    for item in pytest_report.items:
        status = TestStatus.PASS if item.passed else TestStatus.FAIL
        result = TestResult(
            test_name=item.nodeid,
            test_type=TestType.UNIT,
            status=status,
            duration_ms=item.duration * 1000,
        )
        test_run.add_result(result)

    return test_run
