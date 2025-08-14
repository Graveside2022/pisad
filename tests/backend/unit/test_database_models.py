"""Comprehensive tests for database models to reach 60% coverage."""

import json
import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, mock_open
import uuid

import pytest

from src.backend.models.database import ConfigProfileDB, StateHistoryDB


class TestConfigProfileDB:
    """Test ConfigProfileDB class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        Path(db_path).unlink(missing_ok=True)

    @pytest.fixture
    def config_db(self, temp_db):
        """Create ConfigProfileDB instance with temp database."""
        return ConfigProfileDB(db_path=temp_db)

    @pytest.fixture
    def sample_profile(self):
        """Create sample profile data."""
        return {
            "id": str(uuid.uuid4()),
            "name": "Test Profile",
            "description": "Test Description",
            "sdrConfig": {"frequency": 433000000, "sampleRate": 2400000},
            "signalConfig": {"threshold": -80, "averaging": True},
            "homingConfig": {"algorithm": "gradient", "speed": 5.0},
            "isDefault": False,
            "createdAt": datetime.now(UTC).isoformat(),
            "updatedAt": datetime.now(UTC).isoformat(),
        }

    def test_init_database(self, temp_db):
        """Test database initialization."""
        db = ConfigProfileDB(db_path=temp_db)
        
        # Verify tables were created
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        conn.close()
        
        assert ("config_profiles",) in tables

    def test_init_database_creates_directory(self):
        """Test database initialization creates parent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "subdir" / "test.db"
            db = ConfigProfileDB(db_path=str(db_path))
            assert db_path.parent.exists()

    def test_init_database_error(self):
        """Test database initialization with error."""
        with patch("sqlite3.connect", side_effect=Exception("DB Error")):
            with pytest.raises(Exception, match="DB Error"):
                ConfigProfileDB(db_path="test.db")

    def test_insert_profile_success(self, config_db, sample_profile):
        """Test successful profile insertion."""
        result = config_db.insert_profile(sample_profile)
        assert result is True
        
        # Verify data was inserted
        conn = sqlite3.connect(str(config_db.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM config_profiles WHERE id=?", (sample_profile["id"],))
        row = cursor.fetchone()
        conn.close()
        
        assert row is not None
        assert row[1] == sample_profile["name"]  # name column

    def test_insert_profile_duplicate(self, config_db, sample_profile):
        """Test inserting duplicate profile."""
        config_db.insert_profile(sample_profile)
        result = config_db.insert_profile(sample_profile)
        assert result is False

    def test_insert_profile_exception(self, config_db, sample_profile):
        """Test profile insertion with exception."""
        with patch("sqlite3.connect", side_effect=Exception("Insert error")):
            result = config_db.insert_profile(sample_profile)
            assert result is False

    def test_get_profile_by_id(self, config_db, sample_profile):
        """Test getting profile by ID."""
        config_db.insert_profile(sample_profile)
        profile = config_db.get_profile(sample_profile["id"])
        
        assert profile is not None
        assert profile["name"] == sample_profile["name"]
        assert profile["description"] == sample_profile["description"]

    def test_get_profile_by_name(self, config_db, sample_profile):
        """Test getting profile by name."""
        config_db.insert_profile(sample_profile)
        profile = config_db.get_profile_by_name(sample_profile["name"])
        
        assert profile is not None
        assert profile["id"] == sample_profile["id"]

    def test_get_profile_not_found(self, config_db):
        """Test getting non-existent profile."""
        profile = config_db.get_profile("non_existent")
        assert profile is None

    def test_get_profile_exception(self, config_db):
        """Test getting profile with exception."""
        with patch("sqlite3.connect", side_effect=Exception("Get error")):
            profile = config_db.get_profile("test_id")
            assert profile is None

    def test_get_all_profiles(self, config_db, sample_profile):
        """Test getting all profiles."""
        # Insert multiple profiles
        profile1 = sample_profile.copy()
        profile2 = sample_profile.copy()
        profile2["id"] = str(uuid.uuid4())
        profile2["name"] = "Profile 2"
        
        config_db.insert_profile(profile1)
        config_db.insert_profile(profile2)
        
        profiles = config_db.get_all_profiles()
        assert len(profiles) == 2
        assert any(p["name"] == "Test Profile" for p in profiles)
        assert any(p["name"] == "Profile 2" for p in profiles)

    def test_get_all_profiles_empty(self, config_db):
        """Test getting all profiles when none exist."""
        profiles = config_db.get_all_profiles()
        assert profiles == []

    def test_get_all_profiles_exception(self, config_db):
        """Test getting all profiles with exception."""
        with patch("sqlite3.connect", side_effect=Exception("Get all error")):
            profiles = config_db.get_all_profiles()
            assert profiles == []

    def test_update_profile(self, config_db, sample_profile):
        """Test updating profile."""
        config_db.insert_profile(sample_profile)
        
        # Update profile
        sample_profile["description"] = "Updated Description"
        sample_profile["sdrConfig"]["frequency"] = 915000000
        
        result = config_db.update_profile(sample_profile["id"], sample_profile)
        assert result is True
        
        # Verify update
        updated = config_db.get_profile(sample_profile["id"])
        assert updated["description"] == "Updated Description"

    def test_update_profile_not_found(self, config_db, sample_profile):
        """Test updating non-existent profile."""
        result = config_db.update_profile("non_existent", sample_profile)
        assert result is False

    def test_update_profile_exception(self, config_db, sample_profile):
        """Test updating profile with exception."""
        with patch("sqlite3.connect", side_effect=Exception("Update error")):
            result = config_db.update_profile("test_id", sample_profile)
            assert result is False

    def test_delete_profile(self, config_db, sample_profile):
        """Test deleting profile."""
        config_db.insert_profile(sample_profile)
        
        result = config_db.delete_profile(sample_profile["id"])
        assert result is True
        
        # Verify deletion
        profile = config_db.get_profile(sample_profile["id"])
        assert profile is None

    def test_delete_profile_not_found(self, config_db):
        """Test deleting non-existent profile."""
        result = config_db.delete_profile("non_existent")
        assert result is True  # Returns True even if not found

    def test_delete_profile_exception(self, config_db):
        """Test deleting profile with exception."""
        with patch("sqlite3.connect", side_effect=Exception("Delete error")):
            result = config_db.delete_profile("test_id")
            assert result is False

    def test_set_default_profile(self, config_db, sample_profile):
        """Test setting default profile."""
        config_db.insert_profile(sample_profile)
        
        result = config_db.set_default_profile(sample_profile["id"])
        assert result is True
        
        # Verify default was set
        profile = config_db.get_profile(sample_profile["id"])
        assert profile["isDefault"] is True

    def test_set_default_profile_clears_others(self, config_db, sample_profile):
        """Test setting default profile clears other defaults."""
        # Insert two profiles
        profile1 = sample_profile.copy()
        profile2 = sample_profile.copy()
        profile2["id"] = str(uuid.uuid4())
        profile2["name"] = "Profile 2"
        
        config_db.insert_profile(profile1)
        config_db.insert_profile(profile2)
        
        # Set first as default
        config_db.set_default_profile(profile1["id"])
        # Set second as default
        config_db.set_default_profile(profile2["id"])
        
        # Verify only second is default
        p1 = config_db.get_profile(profile1["id"])
        p2 = config_db.get_profile(profile2["id"])
        assert p1["isDefault"] is False
        assert p2["isDefault"] is True

    def test_get_default_profile(self, config_db, sample_profile):
        """Test getting default profile."""
        sample_profile["isDefault"] = True
        config_db.insert_profile(sample_profile)
        
        default = config_db.get_default_profile()
        assert default is not None
        assert default["id"] == sample_profile["id"]

    def test_get_default_profile_none(self, config_db):
        """Test getting default profile when none exists."""
        default = config_db.get_default_profile()
        assert default is None


class TestStateHistoryDB:
    """Test StateHistoryDB class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        Path(db_path).unlink(missing_ok=True)

    @pytest.fixture
    def state_db(self, temp_db):
        """Create StateHistoryDB instance with temp database."""
        return StateHistoryDB(db_path=temp_db)

    @pytest.fixture
    def sample_state(self):
        """Create sample state transition data."""
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "previous_state": "IDLE",
            "new_state": "SEARCHING",
            "trigger": "user_command",
            "details": {"rssi": -85, "confidence": 0.75},
        }

    def test_init_database(self, temp_db):
        """Test database initialization."""
        db = StateHistoryDB(db_path=temp_db)
        
        # Verify tables were created
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        conn.close()
        
        assert ("state_history",) in tables

    def test_log_state_transition(self, state_db, sample_state):
        """Test logging state transition."""
        result = state_db.log_state_transition(
            sample_state["previous_state"],
            sample_state["new_state"],
            sample_state["trigger"],
            sample_state["details"]
        )
        assert result is True

    def test_log_state_transition_exception(self, state_db, sample_state):
        """Test logging state transition with exception."""
        with patch("sqlite3.connect", side_effect=Exception("Log error")):
            result = state_db.log_state_transition(
                sample_state["previous_state"],
                sample_state["new_state"],
                sample_state["trigger"],
                sample_state["details"]
            )
            assert result is False

    def test_get_recent_transitions(self, state_db, sample_state):
        """Test getting recent state transitions."""
        # Log multiple transitions
        states = ["IDLE", "SEARCHING", "DETECTING", "HOMING", "HOLDING"]
        for i in range(len(states) - 1):
            state_db.log_state_transition(
                states[i],
                states[i + 1],
                "test_trigger",
                {"iteration": i}
            )
        
        recent = state_db.get_recent_transitions(limit=3)
        assert len(recent) == 3
        # Should be ordered by timestamp descending
        assert recent[0]["new_state"] == "HOLDING"

    def test_get_recent_transitions_empty(self, state_db):
        """Test getting recent transitions when none exist."""
        recent = state_db.get_recent_transitions()
        assert recent == []

    def test_get_transitions_by_state(self, state_db):
        """Test getting transitions by specific state."""
        # Log various transitions
        state_db.log_state_transition("IDLE", "SEARCHING", "user", {})
        state_db.log_state_transition("SEARCHING", "DETECTING", "signal", {})
        state_db.log_state_transition("DETECTING", "SEARCHING", "lost", {})
        state_db.log_state_transition("SEARCHING", "IDLE", "timeout", {})
        
        # Get all transitions to SEARCHING
        transitions = state_db.get_transitions_by_state(new_state="SEARCHING")
        assert len(transitions) == 2
        assert all(t["new_state"] == "SEARCHING" for t in transitions)

    def test_get_transitions_by_state_from(self, state_db):
        """Test getting transitions from specific state."""
        # Log various transitions
        state_db.log_state_transition("IDLE", "SEARCHING", "user", {})
        state_db.log_state_transition("SEARCHING", "DETECTING", "signal", {})
        state_db.log_state_transition("SEARCHING", "IDLE", "timeout", {})
        
        # Get all transitions from SEARCHING
        transitions = state_db.get_transitions_by_state(previous_state="SEARCHING")
        assert len(transitions) == 2
        assert all(t["previous_state"] == "SEARCHING" for t in transitions)

    def test_get_transition_statistics(self, state_db):
        """Test getting transition statistics."""
        # Log multiple transitions
        state_db.log_state_transition("IDLE", "SEARCHING", "user", {})
        state_db.log_state_transition("SEARCHING", "DETECTING", "signal", {})
        state_db.log_state_transition("DETECTING", "HOMING", "lock", {})
        state_db.log_state_transition("HOMING", "HOLDING", "reached", {})
        state_db.log_state_transition("HOLDING", "IDLE", "complete", {})
        
        stats = state_db.get_transition_statistics()
        assert stats is not None
        assert "total_transitions" in stats
        assert "unique_states" in stats
        assert "most_common_transition" in stats
        assert stats["total_transitions"] == 5

    def test_get_state_duration_stats(self, state_db):
        """Test getting state duration statistics."""
        import time
        
        # Log transitions with delays
        state_db.log_state_transition("IDLE", "SEARCHING", "user", {})
        time.sleep(0.1)
        state_db.log_state_transition("SEARCHING", "DETECTING", "signal", {})
        time.sleep(0.1)
        state_db.log_state_transition("DETECTING", "HOMING", "lock", {})
        
        stats = state_db.get_state_duration_stats()
        assert stats is not None
        assert "SEARCHING" in stats
        assert "avg_duration_ms" in stats["SEARCHING"]

    def test_clear_old_history(self, state_db):
        """Test clearing old state history."""
        # Log old transitions
        old_time = datetime.now(UTC).timestamp() - 86400 * 8  # 8 days old
        
        # Manually insert old records
        conn = sqlite3.connect(str(state_db.db_path))
        cursor = conn.cursor()
        for i in range(5):
            cursor.execute(
                """
                INSERT INTO state_history 
                (timestamp, previous_state, new_state, trigger, details)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    datetime.fromtimestamp(old_time - i * 3600, UTC).isoformat(),
                    "IDLE",
                    "SEARCHING",
                    "old_trigger",
                    json.dumps({})
                )
            )
        conn.commit()
        conn.close()
        
        # Log recent transition
        state_db.log_state_transition("IDLE", "SEARCHING", "recent", {})
        
        # Clear transitions older than 7 days
        result = state_db.clear_old_history(days=7)
        assert result is True
        
        # Verify only recent transition remains
        recent = state_db.get_recent_transitions()
        assert len(recent) == 1
        assert recent[0]["trigger"] == "recent"

    def test_get_current_state(self, state_db):
        """Test getting current state from history."""
        # Log sequence of transitions
        state_db.log_state_transition("IDLE", "SEARCHING", "user", {})
        state_db.log_state_transition("SEARCHING", "DETECTING", "signal", {})
        state_db.log_state_transition("DETECTING", "HOMING", "lock", {})
        
        current = state_db.get_current_state()
        assert current == "HOMING"

    def test_get_current_state_no_history(self, state_db):
        """Test getting current state with no history."""
        current = state_db.get_current_state()
        assert current is None

    def test_export_history_to_csv(self, state_db):
        """Test exporting state history to CSV."""
        # Log some transitions
        state_db.log_state_transition("IDLE", "SEARCHING", "user", {})
        state_db.log_state_transition("SEARCHING", "DETECTING", "signal", {})
        
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            csv_path = f.name
        
        result = state_db.export_history_to_csv(csv_path)
        assert result is True
        assert Path(csv_path).exists()
        
        # Verify CSV content
        with open(csv_path, 'r') as f:
            content = f.read()
            assert "previous_state" in content
            assert "IDLE" in content
            assert "SEARCHING" in content
        
        # Cleanup
        Path(csv_path).unlink()

    def test_export_history_to_csv_exception(self, state_db):
        """Test exporting history with exception."""
        with patch("builtins.open", side_effect=Exception("Export error")):
            result = state_db.export_history_to_csv("test.csv")
            assert result is False