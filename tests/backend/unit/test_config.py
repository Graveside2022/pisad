"""
Unit tests for configuration management.
"""

import os
import tempfile
import pytest
from pathlib import Path
import yaml
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from backend.core.config import (
    Config, ConfigLoader, get_config, reload_config,
    AppConfig, SDRConfig, SignalConfig, LoggingConfig
)


class TestConfigDataclasses:
    """Test configuration dataclass defaults."""
    
    def test_app_config_defaults(self):
        """Test AppConfig default values."""
        config = AppConfig()
        assert config.APP_NAME == "PISAD"
        assert config.APP_VERSION == "1.0.0"
        assert config.APP_ENV == "development"
        assert config.APP_HOST == "0.0.0.0"
        assert config.APP_PORT == 8000
    
    def test_sdr_config_defaults(self):
        """Test SDRConfig default values."""
        config = SDRConfig()
        assert config.SDR_FREQUENCY == 433920000
        assert config.SDR_SAMPLE_RATE == 2048000
        assert config.SDR_GAIN == 30
        assert config.SDR_PPM_CORRECTION == 0
        assert config.SDR_DEVICE_INDEX == 0
        assert config.SDR_BUFFER_SIZE == 16384
    
    def test_signal_config_defaults(self):
        """Test SignalConfig default values."""
        config = SignalConfig()
        assert config.SIGNAL_RSSI_THRESHOLD == -70.0
        assert config.SIGNAL_AVERAGING_WINDOW == 10
        assert config.SIGNAL_MIN_DURATION_MS == 100
        assert config.SIGNAL_MAX_GAP_MS == 50
    
    def test_logging_config_defaults(self):
        """Test LoggingConfig default values."""
        config = LoggingConfig()
        assert config.LOG_LEVEL == "INFO"
        assert config.LOG_FILE_MAX_BYTES == 10485760
        assert config.LOG_FILE_BACKUP_COUNT == 5
        assert config.LOG_ENABLE_CONSOLE == True
        assert config.LOG_ENABLE_FILE == True
        assert config.LOG_ENABLE_JOURNAL == True


class TestConfigLoader:
    """Test configuration loading functionality."""
    
    def test_load_default_config(self):
        """Test loading configuration with defaults."""
        loader = ConfigLoader()
        config = loader.load()
        
        assert isinstance(config, Config)
        assert config.app.APP_NAME == "PISAD"
        assert config.sdr.SDR_FREQUENCY == 433920000
        assert config.logging.LOG_LEVEL == "INFO"
    
    def test_load_from_yaml_file(self):
        """Test loading configuration from YAML file."""
        # Create temporary YAML config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_content = {
                'APP_NAME': 'TestApp',
                'APP_PORT': 9000,
                'SDR_FREQUENCY': 868000000,
                'LOG_LEVEL': 'DEBUG'
            }
            yaml.dump(yaml_content, f)
            temp_file = f.name
        
        try:
            loader = ConfigLoader(temp_file)
            config = loader.load()
            
            assert config.app.APP_NAME == 'TestApp'
            assert config.app.APP_PORT == 9000
            assert config.sdr.SDR_FREQUENCY == 868000000
            assert config.logging.LOG_LEVEL == 'DEBUG'
        finally:
            os.unlink(temp_file)
    
    def test_environment_override(self):
        """Test environment variable overrides."""
        # Set environment variables
        os.environ['PISAD_APP_PORT'] = '8080'
        os.environ['PISAD_SDR_GAIN'] = '40'
        os.environ['PISAD_LOG_LEVEL'] = 'WARNING'
        os.environ['PISAD_DEV_DEBUG_MODE'] = 'true'
        
        try:
            loader = ConfigLoader()
            config = loader.load()
            
            assert config.app.APP_PORT == 8080
            assert config.sdr.SDR_GAIN == 40
            assert config.logging.LOG_LEVEL == 'WARNING'
            assert config.development.DEV_DEBUG_MODE == True
        finally:
            # Clean up environment
            del os.environ['PISAD_APP_PORT']
            del os.environ['PISAD_SDR_GAIN']
            del os.environ['PISAD_LOG_LEVEL']
            del os.environ['PISAD_DEV_DEBUG_MODE']
    
    def test_boolean_conversion(self):
        """Test boolean value conversion from environment."""
        test_cases = [
            ('true', True),
            ('True', True),
            ('TRUE', True),
            ('1', True),
            ('yes', True),
            ('on', True),
            ('false', False),
            ('False', False),
            ('0', False),
            ('no', False),
            ('off', False),
        ]
        
        for env_value, expected in test_cases:
            os.environ['PISAD_DEV_MOCK_SDR'] = env_value
            
            try:
                loader = ConfigLoader()
                config = loader.load()
                assert config.development.DEV_MOCK_SDR == expected, f"Failed for {env_value}"
            finally:
                del os.environ['PISAD_DEV_MOCK_SDR']
    
    def test_list_conversion(self):
        """Test list value conversion from environment."""
        os.environ['PISAD_API_CORS_ORIGINS'] = 'http://localhost:3000, http://example.com, https://app.example.com'
        
        try:
            loader = ConfigLoader()
            config = loader.load()
            
            expected = ['http://localhost:3000', 'http://example.com', 'https://app.example.com']
            assert config.api.API_CORS_ORIGINS == expected
        finally:
            del os.environ['PISAD_API_CORS_ORIGINS']
    
    def test_invalid_type_conversion(self):
        """Test handling of invalid type conversions."""
        # Invalid integer
        os.environ['PISAD_APP_PORT'] = 'not_a_number'
        
        try:
            loader = ConfigLoader()
            config = loader.load()
            # Should keep default value on conversion error
            assert config.app.APP_PORT == 8000
        finally:
            del os.environ['PISAD_APP_PORT']
        
        # Invalid float
        os.environ['PISAD_SAFETY_VELOCITY_MAX_MPS'] = 'invalid_float'
        
        try:
            loader = ConfigLoader()
            config = loader.load()
            # Should keep default value on conversion error
            assert config.safety.SAFETY_VELOCITY_MAX_MPS == 2.0
        finally:
            del os.environ['PISAD_SAFETY_VELOCITY_MAX_MPS']
    
    def test_config_to_dict(self):
        """Test converting configuration to dictionary."""
        config = Config()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'app' in config_dict
        assert 'sdr' in config_dict
        assert 'logging' in config_dict
        
        assert config_dict['app']['APP_NAME'] == 'PISAD'
        assert config_dict['sdr']['SDR_FREQUENCY'] == 433920000


class TestConfigSingleton:
    """Test configuration singleton pattern."""
    
    def test_get_config_singleton(self):
        """Test that get_config returns the same instance."""
        config1 = get_config()
        config2 = get_config()
        
        assert config1 is config2
    
    def test_reload_config(self):
        """Test reloading configuration."""
        # Get initial config
        config1 = get_config()
        initial_port = config1.app.APP_PORT
        
        # Set environment variable
        os.environ['PISAD_APP_PORT'] = '9999'
        
        try:
            # Reload config
            config2 = reload_config()
            
            # Should be new instance with updated value
            assert config2.app.APP_PORT == 9999
            assert config2 is get_config()
        finally:
            del os.environ['PISAD_APP_PORT']
            # Reset for other tests
            reload_config()


class TestConfigEdgeCases:
    """Test edge cases and error handling."""
    
    def test_missing_config_file(self):
        """Test handling of missing configuration file."""
        loader = ConfigLoader('/non/existent/path/config.yaml')
        config = loader.load()
        
        # Should load with defaults
        assert config.app.APP_NAME == 'PISAD'
    
    def test_empty_yaml_file(self):
        """Test loading empty YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('')
            temp_file = f.name
        
        try:
            loader = ConfigLoader(temp_file)
            config = loader.load()
            
            # Should load with defaults
            assert config.app.APP_NAME == 'PISAD'
        finally:
            os.unlink(temp_file)
    
    def test_invalid_yaml_file(self):
        """Test handling of invalid YAML syntax."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('invalid: yaml: content: {')
            temp_file = f.name
        
        try:
            loader = ConfigLoader(temp_file)
            with pytest.raises(yaml.YAMLError):
                config = loader.load()
        finally:
            os.unlink(temp_file)
    
    def test_unknown_config_key(self):
        """Test handling of unknown configuration keys."""
        os.environ['PISAD_UNKNOWN_KEY'] = 'value'
        
        try:
            loader = ConfigLoader()
            config = loader.load()
            
            # Should not crash, just ignore unknown key
            assert not hasattr(config.app, 'UNKNOWN_KEY')
        finally:
            del os.environ['PISAD_UNKNOWN_KEY']