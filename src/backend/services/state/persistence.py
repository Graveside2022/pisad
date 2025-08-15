"""State persistence to database."""

import json
import sqlite3
from datetime import datetime
from typing import Any

from src.backend.core.exceptions import DatabaseError
from src.backend.services.state.types import SystemState
from src.backend.utils.logging import get_logger

logger = get_logger(__name__)


class StatePersistence:
    """Handles state persistence to database.

    This class is responsible ONLY for:
    - Saving state changes to database
    - Restoring last known state
    - Managing database connections

    Cyclomatic complexity: <15
    """

    def __init__(self, db_path: str = "data/pisad.db", enabled: bool = True):
        """Initialize persistence with database path.

        Args:
            db_path: Path to SQLite database
            enabled: Whether persistence is enabled
        """
        self._db_path = db_path
        self._enabled = enabled
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Ensure state_history table exists."""
        if not self._enabled:
            return

        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS state_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        state TEXT NOT NULL,
                        previous_state TEXT,
                        timestamp TEXT NOT NULL,
                        reason TEXT,
                        metadata TEXT
                    )
                """)
                conn.commit()
                logger.debug("State history table ensured")
        except sqlite3.Error as e:
            logger.error(f"Failed to create state_history table: {e}")
            raise DatabaseError(f"Database initialization failed: {e}")

    def save_state(
        self,
        state: SystemState,
        previous_state: SystemState | None = None,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Save current state to database.

        Args:
            state: Current state to save
            previous_state: Previous state (optional)
            reason: Reason for state change (optional)
            metadata: Additional metadata (optional)

        Returns:
            True if saved successfully, False otherwise
        """
        if not self._enabled:
            return True

        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO state_history (state, previous_state, timestamp, reason, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        state.value,
                        previous_state.value if previous_state else None,
                        datetime.now().isoformat(),
                        reason,
                        json.dumps(metadata) if metadata else None,
                    ),
                )
                conn.commit()
                logger.debug(f"Saved state {state.value} to database")
                return True
        except sqlite3.Error as e:
            logger.error(f"Failed to save state: {e}")
            return False

    def restore_last_state(self) -> SystemState | None:
        """Restore the most recent state from database.

        Returns:
            Last known state or None if not found/disabled
        """
        if not self._enabled:
            return None

        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute("SELECT state FROM state_history ORDER BY id DESC LIMIT 1")
                row = cursor.fetchone()

                if row:
                    state_value = row[0]
                    state = SystemState(state_value)
                    logger.info(f"Restored last state: {state_value}")
                    return state

                logger.debug("No previous state found in database")
                return None
        except (sqlite3.Error, ValueError) as e:
            logger.error(f"Failed to restore state: {e}")
            return None

    def get_state_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent state history from database.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of state history entries
        """
        if not self._enabled:
            return []

        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT state, previous_state, timestamp, reason, metadata
                    FROM state_history
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                )

                history = []
                for row in cursor:
                    entry = {
                        "state": row["state"],
                        "previous_state": row["previous_state"],
                        "timestamp": row["timestamp"],
                        "reason": row["reason"],
                    }

                    if row["metadata"]:
                        try:
                            entry["metadata"] = json.loads(row["metadata"])
                        except json.JSONDecodeError:
                            entry["metadata"] = None
                    else:
                        entry["metadata"] = None

                    history.append(entry)

                return history
        except sqlite3.Error as e:
            logger.error(f"Failed to get state history: {e}")
            return []

    def clear_history(self) -> bool:
        """Clear all state history from database.

        Returns:
            True if cleared successfully, False otherwise
        """
        if not self._enabled:
            return True

        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("DELETE FROM state_history")
                conn.commit()
                logger.info("Cleared state history")
                return True
        except sqlite3.Error as e:
            logger.error(f"Failed to clear state history: {e}")
            return False

    def is_enabled(self) -> bool:
        """Check if persistence is enabled."""
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable persistence.

        Args:
            enabled: Whether persistence should be enabled
        """
        self._enabled = enabled
        logger.info(f"State persistence {'enabled' if enabled else 'disabled'}")
