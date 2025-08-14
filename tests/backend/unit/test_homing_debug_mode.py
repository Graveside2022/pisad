"""Tests for homing algorithm debug mode functionality."""

import logging
from unittest.mock import MagicMock, patch

from src.backend.services.homing_algorithm import HomingAlgorithm, set_debug_mode


class TestHomingDebugMode:
    """Test debug mode functionality for homing algorithm."""

    def test_set_debug_mode_enables_verbose_logging(self):
        """Test that enabling debug mode sets correct log level."""
        with patch.object(
            logging.getLogger("src.backend.services.homing_algorithm"), "setLevel"
        ) as mock_setlevel:
            set_debug_mode(True)
            mock_setlevel.assert_called_with(logging.DEBUG)

    def test_set_debug_mode_disables_verbose_logging(self):
        """Test that disabling debug mode sets correct log level."""
        with patch.object(
            logging.getLogger("src.backend.services.homing_algorithm"), "setLevel"
        ) as mock_setlevel:
            set_debug_mode(False)
            mock_setlevel.assert_called_with(logging.INFO)

    @patch("src.backend.services.homing_algorithm.get_config")
    def test_debug_mode_enabled_on_init_if_config_set(self, mock_get_config):
        """Test that debug mode is enabled during init if config flag is set."""
        mock_config = MagicMock()
        mock_config.development.DEV_DEBUG_MODE = True
        mock_config.homing.HOMING_GRADIENT_WINDOW_SIZE = 10
        mock_config.homing.HOMING_GRADIENT_MIN_SNR = 10.0
        mock_config.homing.HOMING_SAMPLING_TURN_RADIUS = 10.0
        mock_config.homing.HOMING_SAMPLING_DURATION = 5.0
        mock_config.homing.HOMING_APPROACH_THRESHOLD = -50.0
        mock_config.homing.HOMING_PLATEAU_VARIANCE = 2.0
        mock_config.homing.HOMING_VELOCITY_SCALE_FACTOR = 0.1
        mock_config.homing.HOMING_FORWARD_VELOCITY_MAX = 5.0
        mock_config.homing.HOMING_YAW_RATE_MAX = 0.5
        mock_get_config.return_value = mock_config

        with patch("src.backend.services.homing_algorithm.set_debug_mode") as mock_set_debug:
            _ = HomingAlgorithm()
            mock_set_debug.assert_called_once_with(True)

    @patch("src.backend.services.homing_algorithm.get_config")
    def test_debug_mode_not_enabled_on_init_if_config_false(self, mock_get_config):
        """Test that debug mode is not enabled during init if config flag is false."""
        mock_config = MagicMock()
        mock_config.development.DEV_DEBUG_MODE = False
        mock_config.homing.HOMING_GRADIENT_WINDOW_SIZE = 10
        mock_config.homing.HOMING_GRADIENT_MIN_SNR = 10.0
        mock_config.homing.HOMING_SAMPLING_TURN_RADIUS = 10.0
        mock_config.homing.HOMING_SAMPLING_DURATION = 5.0
        mock_config.homing.HOMING_APPROACH_THRESHOLD = -50.0
        mock_config.homing.HOMING_PLATEAU_VARIANCE = 2.0
        mock_config.homing.HOMING_VELOCITY_SCALE_FACTOR = 0.1
        mock_config.homing.HOMING_FORWARD_VELOCITY_MAX = 5.0
        mock_config.homing.HOMING_YAW_RATE_MAX = 0.5
        mock_get_config.return_value = mock_config

        with patch("src.backend.services.homing_algorithm.set_debug_mode") as mock_set_debug:
            _ = HomingAlgorithm()
            mock_set_debug.assert_not_called()

    @patch("src.backend.services.homing_algorithm._debug_mode_enabled", True)
    @patch("src.backend.services.homing_algorithm.get_config")
    def test_get_status_includes_debug_info_when_enabled(self, mock_get_config):
        """Test that get_status includes debug information when debug mode is enabled."""
        mock_config = MagicMock()
        mock_config.development.DEV_DEBUG_MODE = False
        mock_config.homing.HOMING_GRADIENT_WINDOW_SIZE = 10
        mock_config.homing.HOMING_GRADIENT_MIN_SNR = 10.0
        mock_config.homing.HOMING_SAMPLING_TURN_RADIUS = 10.0
        mock_config.homing.HOMING_SAMPLING_DURATION = 5.0
        mock_config.homing.HOMING_APPROACH_THRESHOLD = -50.0
        mock_config.homing.HOMING_PLATEAU_VARIANCE = 2.0
        mock_config.homing.HOMING_VELOCITY_SCALE_FACTOR = 0.1
        mock_config.homing.HOMING_FORWARD_VELOCITY_MAX = 5.0
        mock_config.homing.HOMING_YAW_RATE_MAX = 0.5
        mock_get_config.return_value = mock_config

        algorithm = HomingAlgorithm()

        # Add some samples
        algorithm.add_rssi_sample(-60.0, 10.0, 20.0, 90.0, 1.0)
        algorithm.add_rssi_sample(-58.0, 12.0, 22.0, 95.0, 2.0)
        algorithm.add_rssi_sample(-55.0, 14.0, 24.0, 100.0, 3.0)

        status = algorithm.get_status()

        assert "debug_mode" in status
        assert status["debug_mode"] is True
        assert "debug_info" in status
        assert "rssi_min" in status["debug_info"]
        assert "rssi_max" in status["debug_info"]
        assert "rssi_mean" in status["debug_info"]
        assert "rssi_variance" in status["debug_info"]
        assert "position_spread_x" in status["debug_info"]
        assert "position_spread_y" in status["debug_info"]
        assert "time_span" in status["debug_info"]

    @patch("src.backend.services.homing_algorithm._debug_mode_enabled", False)
    @patch("src.backend.services.homing_algorithm.get_config")
    def test_get_status_excludes_debug_info_when_disabled(self, mock_get_config):
        """Test that get_status excludes debug information when debug mode is disabled."""
        mock_config = MagicMock()
        mock_config.development.DEV_DEBUG_MODE = False
        mock_config.homing.HOMING_GRADIENT_WINDOW_SIZE = 10
        mock_config.homing.HOMING_GRADIENT_MIN_SNR = 10.0
        mock_config.homing.HOMING_SAMPLING_TURN_RADIUS = 10.0
        mock_config.homing.HOMING_SAMPLING_DURATION = 5.0
        mock_config.homing.HOMING_APPROACH_THRESHOLD = -50.0
        mock_config.homing.HOMING_PLATEAU_VARIANCE = 2.0
        mock_config.homing.HOMING_VELOCITY_SCALE_FACTOR = 0.1
        mock_config.homing.HOMING_FORWARD_VELOCITY_MAX = 5.0
        mock_config.homing.HOMING_YAW_RATE_MAX = 0.5
        mock_get_config.return_value = mock_config

        algorithm = HomingAlgorithm()

        # Add some samples
        algorithm.add_rssi_sample(-60.0, 10.0, 20.0, 90.0, 1.0)

        status = algorithm.get_status()

        assert "debug_mode" in status
        assert status["debug_mode"] is False
        assert "debug_info" not in status
