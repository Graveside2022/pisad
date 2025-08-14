"""Unit tests for the Configuration Service."""

import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from src.backend.models.schemas import (
    ConfigProfile,
    HomingConfig,
    SDRConfig,
    SignalConfig,
)
from src.backend.services.config_service import ConfigService


@pytest.fixture
def temp_profiles_dir():
    """Create a temporary directory for test profiles."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def config_service(temp_profiles_dir):
    """Create a ConfigService instance with temporary directory."""
    return ConfigService(profiles_dir=temp_profiles_dir)


@pytest.fixture
def sample_profile():
    """Create a sample configuration profile."""
    return ConfigProfile(
        id=str(uuid4()),
        name="test_profile",
        description="Test configuration profile",
        sdrConfig=SDRConfig(
            frequency=2437000000.0, sampleRate=2000000.0, gain=40, bandwidth=2000000.0
        ),
        signalConfig=SignalConfig(
            fftSize=1024, ewmaAlpha=0.1, triggerThreshold=-60.0, dropThreshold=-70.0
        ),
        homingConfig=HomingConfig(
            forwardVelocityMax=5.0, yawRateMax=1.0, approachVelocity=2.0, signalLossTimeout=5.0
        ),
        isDefault=False,
        createdAt=datetime.now(UTC),
        updatedAt=datetime.now(UTC),
    )


class TestConfigService:
    """Test suite for ConfigService."""

    def test_save_and_load_profile(self, config_service, sample_profile):
        """Test saving and loading a configuration profile."""
        # Save profile
        assert config_service.save_profile(sample_profile) is True

        # Load profile
        loaded_profile = config_service.load_profile(sample_profile.name)
        assert loaded_profile is not None
        assert loaded_profile.name == sample_profile.name
        assert loaded_profile.description == sample_profile.description
        assert loaded_profile.sdrConfig.frequency == sample_profile.sdrConfig.frequency
        assert loaded_profile.signalConfig.ewmaAlpha == sample_profile.signalConfig.ewmaAlpha
        assert (
            loaded_profile.homingConfig.forwardVelocityMax
            == sample_profile.homingConfig.forwardVelocityMax
        )

    def test_list_profiles(self, config_service, sample_profile):
        """Test listing all profiles."""
        # Initially empty
        assert config_service.list_profiles() == []

        # Save multiple profiles
        config_service.save_profile(sample_profile)

        profile2 = sample_profile
        profile2.id = str(uuid4())
        profile2.name = "test_profile_2"
        config_service.save_profile(profile2)

        # List profiles
        profiles = config_service.list_profiles()
        assert len(profiles) == 2
        assert "test_profile" in profiles
        assert "test_profile_2" in profiles

    def test_delete_profile(self, config_service, sample_profile):
        """Test deleting a profile."""
        # Save profile
        config_service.save_profile(sample_profile)
        assert sample_profile.name in config_service.list_profiles()

        # Delete profile
        assert config_service.delete_profile(sample_profile.name) is True
        assert sample_profile.name not in config_service.list_profiles()

        # Delete non-existent profile
        assert config_service.delete_profile("non_existent") is False

    def test_get_default_profile(self, config_service, sample_profile):
        """Test getting the default profile."""
        # No default initially
        assert config_service.get_default_profile() is None

        # Save profile as default
        sample_profile.isDefault = True
        config_service.save_profile(sample_profile)

        # Get default profile
        default = config_service.get_default_profile()
        assert default is not None
        assert default.name == sample_profile.name
        assert default.isDefault is True

    def test_set_default_profile(self, config_service):
        """Test setting a profile as default."""
        # Create and save first profile
        profile1 = ConfigProfile(
            id=str(uuid4()),
            name="test_profile_1",
            description="First test profile",
            sdrConfig=SDRConfig(
                frequency=2437000000.0, sampleRate=2000000.0, gain=40, bandwidth=2000000.0
            ),
            signalConfig=SignalConfig(
                fftSize=1024, ewmaAlpha=0.1, triggerThreshold=-60.0, dropThreshold=-70.0
            ),
            homingConfig=HomingConfig(
                forwardVelocityMax=5.0, yawRateMax=1.0, approachVelocity=2.0, signalLossTimeout=5.0
            ),
            isDefault=False,
            createdAt=datetime.now(UTC),
            updatedAt=datetime.now(UTC),
        )
        config_service.save_profile(profile1)

        # Create and save second profile
        profile2 = ConfigProfile(
            id=str(uuid4()),
            name="test_profile_2",
            description="Second test profile",
            sdrConfig=SDRConfig(
                frequency=2437000000.0, sampleRate=2000000.0, gain=40, bandwidth=2000000.0
            ),
            signalConfig=SignalConfig(
                fftSize=1024, ewmaAlpha=0.1, triggerThreshold=-60.0, dropThreshold=-70.0
            ),
            homingConfig=HomingConfig(
                forwardVelocityMax=5.0, yawRateMax=1.0, approachVelocity=2.0, signalLossTimeout=5.0
            ),
            isDefault=False,
            createdAt=datetime.now(UTC),
            updatedAt=datetime.now(UTC),
        )
        config_service.save_profile(profile2)

        # Set first as default
        assert config_service.set_default_profile(profile1.name) is True
        default = config_service.get_default_profile()
        assert default.name == profile1.name

        # Set second as default (should unset first)
        assert config_service.set_default_profile(profile2.name) is True
        default = config_service.get_default_profile()
        assert default.name == profile2.name

        # Check first is no longer default
        profile1_reloaded = config_service.load_profile(profile1.name)
        assert profile1_reloaded.isDefault is False

    def test_validate_profile(self, config_service, sample_profile):
        """Test profile validation."""
        # Valid profile
        validation = config_service.validate_profile(sample_profile)
        assert validation["valid"] is True
        assert len(validation["errors"]) == 0

        # Invalid frequency
        sample_profile.sdrConfig.frequency = 0.5e6  # Below 1 MHz
        validation = config_service.validate_profile(sample_profile)
        assert validation["valid"] is False
        assert "Frequency must be between 1 MHz and 6 GHz" in validation["errors"]

        # Invalid sample rate
        sample_profile.sdrConfig.frequency = 2.4e9  # Reset to valid
        sample_profile.sdrConfig.sampleRate = 0.1e6  # Below 0.25 Msps
        validation = config_service.validate_profile(sample_profile)
        assert validation["valid"] is False
        assert "Sample rate must be between 0.25 Msps and 20 Msps" in validation["errors"]

        # Invalid gain
        sample_profile.sdrConfig.sampleRate = 2e6  # Reset to valid
        sample_profile.sdrConfig.gain = 100  # Above 70 dB
        validation = config_service.validate_profile(sample_profile)
        assert validation["valid"] is False
        assert "Gain must be between -10 dB and 70 dB" in validation["errors"]

        # Invalid EWMA alpha
        sample_profile.sdrConfig.gain = 40  # Reset to valid
        sample_profile.signalConfig.ewmaAlpha = 1.5  # Above 1
        validation = config_service.validate_profile(sample_profile)
        assert validation["valid"] is False
        assert "EWMA alpha must be between 0 and 1" in validation["errors"]

        # Invalid thresholds
        sample_profile.signalConfig.ewmaAlpha = 0.1  # Reset to valid
        sample_profile.signalConfig.dropThreshold = -50  # Greater than trigger threshold
        sample_profile.signalConfig.triggerThreshold = -60
        validation = config_service.validate_profile(sample_profile)
        assert validation["valid"] is False
        assert "Drop threshold must be less than trigger threshold" in validation["errors"]

        # Invalid homing parameters
        sample_profile.signalConfig.dropThreshold = -70  # Reset to valid
        sample_profile.homingConfig.forwardVelocityMax = -1  # Negative
        validation = config_service.validate_profile(sample_profile)
        assert validation["valid"] is False
        assert "Forward velocity max must be positive" in validation["errors"]

    def test_load_nonexistent_profile(self, config_service):
        """Test loading a non-existent profile."""
        profile = config_service.load_profile("nonexistent")
        assert profile is None

    def test_yaml_file_structure(self, config_service, sample_profile):
        """Test that YAML files are created with correct structure."""
        config_service.save_profile(sample_profile)

        # Check file exists
        profile_path = Path(config_service.profiles_dir) / f"{sample_profile.name}.yaml"
        assert profile_path.exists()

        # Load and verify YAML content
        import yaml

        with open(profile_path) as f:
            data = yaml.safe_load(f)

        assert data["id"] == sample_profile.id
        assert data["name"] == sample_profile.name
        assert data["sdrConfig"]["frequency"] == sample_profile.sdrConfig.frequency
        assert data["signalConfig"]["ewmaAlpha"] == sample_profile.signalConfig.ewmaAlpha
        assert (
            data["homingConfig"]["forwardVelocityMax"]
            == sample_profile.homingConfig.forwardVelocityMax
        )
