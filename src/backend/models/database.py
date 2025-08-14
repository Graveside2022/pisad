"""Database models and table definitions for PISAD."""

import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ConfigProfileDB:
    """Database handler for configuration profiles."""

    def __init__(self, db_path: str = "data/pisad.db"):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database tables if they don't exist."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Create config_profiles table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS config_profiles (
                    id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    sdr_config TEXT,
                    signal_config TEXT,
                    homing_config TEXT,
                    is_default BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """
            )

            # Create index on name for faster lookups
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_profile_name
                ON config_profiles(name)
            """
            )

            conn.commit()
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def insert_profile(self, profile_data: dict[str, Any]) -> bool:
        """Insert a new configuration profile.

        Args:
            profile_data: Dictionary containing profile data

        Returns:
            True if inserted successfully, False otherwise
        """
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO config_profiles
                (id, name, description, sdr_config, signal_config, homing_config,
                 is_default, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    profile_data["id"],
                    profile_data["name"],
                    profile_data.get("description", ""),
                    json.dumps(profile_data.get("sdrConfig")),
                    json.dumps(profile_data.get("signalConfig")),
                    json.dumps(profile_data.get("homingConfig")),
                    profile_data.get("isDefault", False),
                    profile_data.get("createdAt", datetime.now(UTC).isoformat()),
                    profile_data.get("updatedAt", datetime.now(UTC).isoformat()),
                ),
            )

            conn.commit()
            return True

        except sqlite3.IntegrityError as e:
            logger.error(f"Profile with name '{profile_data['name']}' already exists: {e}")
            return False
        except Exception as e:
            logger.error(f"Error inserting profile: {e}")
            return False
        finally:
            if conn:
                conn.close()

    def update_profile(self, profile_id: str, profile_data: dict[str, Any]) -> bool:
        """Update an existing configuration profile.

        Args:
            profile_id: ID of the profile to update
            profile_data: Dictionary containing updated profile data

        Returns:
            True if updated successfully, False otherwise
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE config_profiles
                SET name=?, description=?, sdr_config=?, signal_config=?,
                    homing_config=?, is_default=?, updated_at=?
                WHERE id=?
            """,
                (
                    profile_data["name"],
                    profile_data.get("description", ""),
                    json.dumps(profile_data.get("sdrConfig")),
                    json.dumps(profile_data.get("signalConfig")),
                    json.dumps(profile_data.get("homingConfig")),
                    profile_data.get("isDefault", False),
                    datetime.now(UTC).isoformat(),
                    profile_id,
                ),
            )

            conn.commit()
            conn.close()
            return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Error updating profile: {e}")
            return False

    def get_profile(self, profile_id: str) -> dict[str, Any] | None:
        """Get a configuration profile by ID.

        Args:
            profile_id: ID of the profile to retrieve

        Returns:
            Profile data dictionary if found, None otherwise
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM config_profiles WHERE id=?
            """,
                (profile_id,),
            )

            row = cursor.fetchone()
            conn.close()

            if row:
                return self._row_to_dict(row)
            return None

        except Exception as e:
            logger.error(f"Error getting profile: {e}")
            return None

    def get_profile_by_name(self, name: str) -> dict[str, Any] | None:
        """Get a configuration profile by name.

        Args:
            name: Name of the profile to retrieve

        Returns:
            Profile data dictionary if found, None otherwise
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM config_profiles WHERE name=?
            """,
                (name,),
            )

            row = cursor.fetchone()
            conn.close()

            if row:
                return self._row_to_dict(row)
            return None

        except Exception as e:
            logger.error(f"Error getting profile by name: {e}")
            return None

    def list_profiles(self) -> list[dict[str, Any]]:
        """List all configuration profiles.

        Returns:
            List of profile data dictionaries
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM config_profiles ORDER BY name
            """
            )

            rows = cursor.fetchall()
            conn.close()

            return [self._row_to_dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Error listing profiles: {e}")
            return []

    def delete_profile(self, profile_id: str) -> bool:
        """Delete a configuration profile.

        Args:
            profile_id: ID of the profile to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                DELETE FROM config_profiles WHERE id=?
            """,
                (profile_id,),
            )

            conn.commit()
            conn.close()
            return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Error deleting profile: {e}")
            return False

    def set_default_profile(self, profile_id: str) -> bool:
        """Set a profile as the default.

        Args:
            profile_id: ID of the profile to set as default

        Returns:
            True if set successfully, False otherwise
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # First, unset all defaults
            cursor.execute(
                """
                UPDATE config_profiles SET is_default=0
            """
            )

            # Then set the new default
            cursor.execute(
                """
                UPDATE config_profiles SET is_default=1 WHERE id=?
            """,
                (profile_id,),
            )

            conn.commit()
            conn.close()
            return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Error setting default profile: {e}")
            return False

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert a database row to a dictionary.

        Args:
            row: SQLite row object

        Returns:
            Dictionary representation of the row
        """
        return {
            "id": row["id"],
            "name": row["name"],
            "description": row["description"],
            "sdrConfig": json.loads(row["sdr_config"]) if row["sdr_config"] else None,
            "signalConfig": json.loads(row["signal_config"]) if row["signal_config"] else None,
            "homingConfig": json.loads(row["homing_config"]) if row["homing_config"] else None,
            "isDefault": bool(row["is_default"]),
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
        }


class StateHistoryDB:
    """Database handler for state machine history persistence."""

    def __init__(self, db_path: str = "data/pisad.db"):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self) -> None:
        """Initialize state history table if it doesn't exist."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Create state_history table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS state_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    from_state TEXT NOT NULL,
                    to_state TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    reason TEXT,
                    operator_id TEXT,
                    action_duration_ms INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # Create current_state table for persistence
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS current_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    state TEXT NOT NULL,
                    previous_state TEXT NOT NULL,
                    homing_enabled BOOLEAN DEFAULT 0,
                    last_detection_time REAL DEFAULT 0,
                    detection_count INTEGER DEFAULT 0,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # Create indexes for better query performance
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_state_history_timestamp
                ON state_history(timestamp DESC)
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_state_history_states
                ON state_history(from_state, to_state)
                """
            )

            conn.commit()
        except Exception as e:
            logger.error(f"Error initializing state history database: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def save_state_change(
        self,
        from_state: str,
        to_state: str,
        timestamp: datetime,
        reason: str | None = None,
        operator_id: str | None = None,
        action_duration_ms: int | None = None,
    ) -> bool:
        """Save a state transition to the database.

        Args:
            from_state: The state transitioning from
            to_state: The state transitioning to
            timestamp: When the transition occurred
            reason: Optional reason for the transition
            operator_id: Optional operator ID for manual overrides
            action_duration_ms: Optional duration of entry/exit actions

        Returns:
            True if saved successfully, False otherwise
        """
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO state_history
                (from_state, to_state, timestamp, reason, operator_id, action_duration_ms)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    from_state,
                    to_state,
                    timestamp.isoformat(),
                    reason,
                    operator_id,
                    action_duration_ms,
                ),
            )

            conn.commit()
            return True

        except Exception as e:
            logger.error(f"Error saving state change: {e}")
            return False
        finally:
            if conn:
                conn.close()

    def save_current_state(
        self,
        state: str,
        previous_state: str,
        homing_enabled: bool = False,
        last_detection_time: float = 0.0,
        detection_count: int = 0,
    ) -> bool:
        """Save or update the current state for persistence.

        Args:
            state: Current state
            previous_state: Previous state
            homing_enabled: Whether homing is enabled
            last_detection_time: Last detection timestamp
            detection_count: Number of detections

        Returns:
            True if saved successfully, False otherwise
        """
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Use INSERT OR REPLACE to ensure we only have one row
            cursor.execute(
                """
                INSERT OR REPLACE INTO current_state
                (id, state, previous_state, homing_enabled, last_detection_time,
                 detection_count, updated_at)
                VALUES (1, ?, ?, ?, ?, ?, ?)
                """,
                (
                    state,
                    previous_state,
                    homing_enabled,
                    last_detection_time,
                    detection_count,
                    datetime.now(UTC).isoformat(),
                ),
            )

            conn.commit()
            return True

        except Exception as e:
            logger.error(f"Error saving current state: {e}")
            return False
        finally:
            if conn:
                conn.close()

    def restore_state(self) -> dict[str, Any] | None:
        """Restore the last saved state from the database.

        Returns:
            Dictionary with state information if found, None otherwise
        """
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM current_state WHERE id = 1
                """
            )

            row = cursor.fetchone()

            if row:
                return {
                    "state": row["state"],
                    "previous_state": row["previous_state"],
                    "homing_enabled": bool(row["homing_enabled"]),
                    "last_detection_time": row["last_detection_time"],
                    "detection_count": row["detection_count"],
                    "updated_at": row["updated_at"],
                }
            return None

        except Exception as e:
            logger.error(f"Error restoring state: {e}")
            return None
        finally:
            if conn:
                conn.close()

    def get_state_history(
        self, limit: int = 100, from_state: str | None = None, to_state: str | None = None
    ) -> list[dict[str, Any]]:
        """Get state transition history from the database.

        Args:
            limit: Maximum number of records to return
            from_state: Filter by from_state
            to_state: Filter by to_state

        Returns:
            List of state transition records
        """
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM state_history WHERE 1=1"
            params: list[Any] = []

            if from_state:
                query += " AND from_state = ?"
                params.append(from_state)

            if to_state:
                query += " AND to_state = ?"
                params.append(to_state)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [
                {
                    "id": row["id"],
                    "from_state": row["from_state"],
                    "to_state": row["to_state"],
                    "timestamp": row["timestamp"],
                    "reason": row["reason"],
                    "operator_id": row["operator_id"],
                    "action_duration_ms": row["action_duration_ms"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Error getting state history: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def cleanup_old_history(self, days_to_keep: int = 30) -> int:
        """Clean up old state history records.

        Args:
            days_to_keep: Number of days of history to keep

        Returns:
            Number of records deleted
        """
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cutoff_date = datetime.now(UTC).replace(day=datetime.now(UTC).day - days_to_keep)

            cursor.execute(
                """
                DELETE FROM state_history
                WHERE datetime(timestamp) < datetime(?)
                """,
                (cutoff_date.isoformat(),),
            )

            deleted = cursor.rowcount
            conn.commit()

            logger.info(f"Cleaned up {deleted} old state history records")
            return deleted

        except Exception as e:
            logger.error(f"Error cleaning up state history: {e}")
            return 0
        finally:
            if conn:
                conn.close()
