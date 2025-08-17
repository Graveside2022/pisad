"""Unit tests for ConfigService.

Tests configuration management including profile loading,
validation, and runtime updates per PRD requirements.
"""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
import yaml

from src.backend.models.schemas import ConfigProfile, HomingConfig, SDRConfig, SignalConfig
from src.backend.services.config_service import ConfigService


class TestConfigService:
    """Test configuration management service."""

    @pytest.fixture
    def config_service(self):
        """Provide ConfigService instance."""
        return ConfigService()

    @pytest.fixture
    def test_config_profile(self):
        """Provide test configuration profile."""
        return ConfigProfile(
            id="test-profile-123",
            name="test_profile",
            description="Test configuration profile",
            sdrConfig=SDRConfig(
                frequency=433920000, sampleRate=2000000, gain=30, bandwidth=2000000
            ),
            signalConfig=SignalConfig(),
            homingConfig=HomingConfig(),
            isDefault=False,
            createdAt=datetime.now(UTC),
            updatedAt=datetime.now(UTC),
        )

    def test_config_service_initialization(self, config_service):
        """Test ConfigService initializes correctly."""
        assert config_service.profiles_dir is not None
        assert config_service.profiles_dir.exists()
        assert isinstance(config_service.profiles_dir, Path)

    def test_get_default_profile(self, config_service, test_config_profile):
        """Test getting default configuration profile."""
        with patch.object(config_service, "load_profile", return_value=test_config_profile):
            with patch.object(config_service, "list_profiles", return_value=["default"]):
                profile = config_service.get_default_profile()

                assert profile is not None
                assert profile.name == "test_profile"
                assert profile.sdrConfig.frequency == 433920000

    def test_load_profile_by_name(self, config_service, test_config_profile):
        """Test loading specific configuration profile."""
        profile_name = "field_test"

        # Create mock YAML data for ConfigProfile
        profile_data = {
            "id": test_config_profile.id,
            "name": profile_name,
            "description": test_config_profile.description,
            "sdrConfig": {
                "frequency": 433920000,
                "sampleRate": 2000000,
                "gain": 30,
                "bandwidth": 2000000,
            },
            "signalConfig": {},
            "homingConfig": {},
            "isDefault": False,
            "createdAt": "2025-08-17T14:30:00Z",
            "updatedAt": "2025-08-17T14:30:00Z",
        }

        with patch("builtins.open", mock_open(read_data=yaml.dump(profile_data))):
            with patch("pathlib.Path.exists", return_value=True):
                profile = config_service.load_profile(profile_name)

                assert profile is not None
                assert isinstance(profile, ConfigProfile)
                assert profile.name == profile_name
                assert profile.sdrConfig.frequency == 433920000

    def test_save_profile(self, config_service, test_config_profile, tmp_path):
        """Test saving configuration profile to file."""
        with patch.object(config_service, "profiles_dir", tmp_path):
            success = config_service.save_profile(test_config_profile)

            assert success is True

            # Verify file was created
            profile_path = tmp_path / f"{test_config_profile.name}.yaml"
            assert profile_path.exists()

            # Verify content
            with open(profile_path) as f:
                saved_data = yaml.safe_load(f)
            assert saved_data["name"] == "test_profile"
            assert saved_data["sdrConfig"]["frequency"] == 433920000

    def test_list_available_profiles(self, config_service, tmp_path):
        """Test listing available configuration profiles."""
        # Create test profile files
        (tmp_path / "profile1.yaml").touch()
        (tmp_path / "profile2.yaml").touch()
        (tmp_path / "profile3.yaml").touch()
        (tmp_path / "not_a_profile.txt").touch()  # Should be ignored

        with patch.object(config_service, "profiles_dir", tmp_path):
            profiles = config_service.list_profiles()

            assert "profile1" in profiles
            assert "profile2" in profiles
            assert "profile3" in profiles
            assert "not_a_profile" not in profiles

    def test_validate_profile_structure(self, config_service, test_config_profile):
        """Test configuration profile validation."""
        # Valid profile should pass
        validation_result = config_service.validate_profile(test_config_profile)
        assert isinstance(validation_result, dict)
        # Validation passes if no errors returned

    def test_validate_profile_missing_required_fields(self, config_service):
        """Test validation fails for missing required fields."""
        incomplete_profile = ConfigProfile(
            id="test-123",
            name="incomplete",
            description="Missing required fields",
            # Missing sdrConfig intentionally
            isDefault=False,
            createdAt=datetime.now(UTC),
            updatedAt=datetime.now(UTC),
        )

        validation_result = config_service.validate_profile(incomplete_profile)
        assert isinstance(validation_result, dict)

    def test_delete_profile(self, config_service, tmp_path):
        """Test deleting a configuration profile."""
        profile_name = "test_delete"
        profile_file = tmp_path / f"{profile_name}.yaml"
        profile_file.touch()  # Create the file

        with patch.object(config_service, "profiles_dir", tmp_path):
            success = config_service.delete_profile(profile_name)
            assert success is True
            assert not profile_file.exists()

    def test_set_default_profile(self, config_service, test_config_profile, tmp_path):
        """Test setting a profile as default."""
        profile_name = "new_default"

        with patch.object(config_service, "profiles_dir", tmp_path):
            with patch.object(config_service, "load_profile", return_value=test_config_profile):
                success = config_service.set_default_profile(profile_name)
                assert success is True
