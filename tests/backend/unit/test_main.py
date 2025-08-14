"""Unit tests for main entry point."""

import pytest
from unittest.mock import MagicMock, patch
import sys

from src.backend.main import main


class TestMain:
    """Test main entry point functionality."""
    
    @patch('src.backend.main.uvicorn.run')
    @patch('src.backend.main.setup_logging')
    @patch('src.backend.main.get_config')
    def test_main_starts_server(self, mock_config, mock_logging, mock_uvicorn):
        """Test that main starts the uvicorn server."""
        # Setup mock config
        mock_config.return_value = MagicMock(
            app=MagicMock(
                APP_NAME="PISAD",
                APP_HOST="0.0.0.0",
                APP_PORT=8000
            ),
            development=MagicMock(
                DEV_HOT_RELOAD=True
            ),
            logging=MagicMock(
                LOG_LEVEL="INFO"
            )
        )
        
        # Call main
        main()
        
        # Check setup was called
        mock_logging.assert_called_once()
        mock_config.assert_called_once()
        
        # Check uvicorn was started with correct params
        mock_uvicorn.assert_called_once_with(
            "src.backend.core.app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    
    @patch('src.backend.main.uvicorn.run')
    @patch('src.backend.main.setup_logging')
    @patch('src.backend.main.get_config')
    def test_main_production_config(self, mock_config, mock_logging, mock_uvicorn):
        """Test main with production configuration."""
        # Setup mock config for production
        mock_config.return_value = MagicMock(
            app=MagicMock(
                APP_NAME="PISAD",
                APP_HOST="127.0.0.1",
                APP_PORT=8080
            ),
            development=MagicMock(
                DEV_HOT_RELOAD=False
            ),
            logging=MagicMock(
                LOG_LEVEL="WARNING"
            )
        )
        
        # Call main
        main()
        
        # Check uvicorn was started with production params
        mock_uvicorn.assert_called_once_with(
            "src.backend.core.app:app",
            host="127.0.0.1",
            port=8080,
            reload=False,
            log_level="warning"
        )
    
    @patch('src.backend.main.uvicorn.run')
    @patch('src.backend.main.setup_logging')
    @patch('src.backend.main.get_config')
    @patch('src.backend.main.logging.getLogger')
    def test_main_logs_startup(self, mock_logger, mock_config, mock_logging, mock_uvicorn):
        """Test that main logs startup message."""
        # Setup mocks
        mock_config.return_value = MagicMock(
            app=MagicMock(
                APP_NAME="PISAD",
                APP_HOST="0.0.0.0",
                APP_PORT=8000
            ),
            development=MagicMock(
                DEV_HOT_RELOAD=True
            ),
            logging=MagicMock(
                LOG_LEVEL="INFO"
            )
        )
        logger_instance = MagicMock()
        mock_logger.return_value = logger_instance
        
        # Call main
        main()
        
        # Check logger was obtained
        mock_logger.assert_called_once_with("src.backend.main")
        
        # Check info message was logged
        logger_instance.info.assert_called_once()
        call_args = logger_instance.info.call_args[0][0]
        assert "Starting PISAD server" in call_args
        assert "0.0.0.0:8000" in call_args
    
    def test_project_root_in_path(self):
        """Test that project root is added to sys.path."""
        # Check that the project root is in the path
        # This happens at module import time
        import src.backend.main
        # Path should contain the project root
        assert any("pisad" in str(p) for p in sys.path)
    
    @patch('src.backend.main.uvicorn.run')
    @patch('src.backend.main.setup_logging')
    @patch('src.backend.main.get_config')
    def test_main_handles_config_error(self, mock_config, mock_logging, mock_uvicorn):
        """Test main handles configuration errors."""
        # Setup mock to raise error
        mock_config.side_effect = Exception("Config error")
        
        # Call main - should raise the exception
        with pytest.raises(Exception) as exc_info:
            main()
        
        assert "Config error" in str(exc_info.value)
        
        # Uvicorn should not be called
        mock_uvicorn.assert_not_called()
    
    @patch('src.backend.main.uvicorn.run')
    @patch('src.backend.main.setup_logging')
    @patch('src.backend.main.get_config')
    def test_main_passes_log_level_lowercase(self, mock_config, mock_logging, mock_uvicorn):
        """Test that log level is converted to lowercase."""
        # Setup mock config with uppercase log level
        mock_config.return_value = MagicMock(
            app=MagicMock(
                APP_NAME="PISAD",
                APP_HOST="0.0.0.0",
                APP_PORT=8000
            ),
            development=MagicMock(
                DEV_HOT_RELOAD=False
            ),
            logging=MagicMock(
                LOG_LEVEL="DEBUG"
            )
        )
        
        # Call main
        main()
        
        # Check log level was lowercased
        call_kwargs = mock_uvicorn.call_args[1]
        assert call_kwargs["log_level"] == "debug"