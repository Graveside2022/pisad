"""Unit tests for MAVLink Mission Planner integration.

Tests the MAVLink parameter interface for frequency control from Mission Planner,
implementing authentic Test-Driven Development with real system behavior validation.

PRD Reference: FR11 (Operator full override), FR9 (RSSI telemetry), FR14 (Operator control)
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from src.backend.services.mavlink_service import MAVLinkService


class TestMAVLinkMissionPlannerIntegration:
    """Test MAVLink Mission Planner parameter integration."""

    @pytest.fixture
    def mavlink_service(self):
        """Create MAVLink service for testing."""
        return MAVLinkService(
            device_path="/dev/ttyACM0",
            baud_rate=115200,
            source_system=1,
            source_component=191,
        )

    @pytest.fixture
    def mock_connection(self):
        """Mock MAVLink connection."""
        mock_conn = MagicMock()
        mock_conn.recv_match.return_value = None
        return mock_conn

    def test_parameter_interface_initialization(self, mavlink_service):
        """Test parameter interface is properly initialized."""
        # Verify parameter interface exists
        assert hasattr(mavlink_service, "_parameter_handlers")
        assert hasattr(mavlink_service, "set_parameter")
        assert hasattr(mavlink_service, "get_parameter")
        assert hasattr(mavlink_service, "request_parameter")

    def test_frequency_parameter_registration(self, mavlink_service):
        """Test frequency profile parameters are registered."""
        # Verify frequency parameters exist
        frequency_params = [
            "PISAD_FREQ_PROF",  # Current frequency profile (0=Emergency, 1=Aviation, 2=Custom)
            "PISAD_FREQ_HZ",  # Custom frequency in Hz
            "PISAD_HOMING_EN",  # Homing mode enable/disable
        ]

        for param_name in frequency_params:
            assert param_name in mavlink_service._parameter_handlers

    @patch("pymavlink.mavutil.mavlink_connection")
    def test_param_set_message_handling(self, mock_mavutil, mavlink_service, mock_connection):
        """Test PARAM_SET message handling from Mission Planner."""
        # Setup mock connection
        mock_mavutil.return_value = mock_connection
        mavlink_service.connection = mock_connection

        # Create mock PARAM_SET message
        param_set_msg = MagicMock()
        param_set_msg.get_type.return_value = "PARAM_SET"
        param_set_msg.param_id = "PISAD_FREQ_PROF"
        param_set_msg.param_value = 1.0  # Aviation profile
        param_set_msg.param_type = 2  # MAV_PARAM_TYPE_REAL32

        # Process parameter set
        result = mavlink_service._handle_param_set(param_set_msg)

        # Verify parameter was set
        assert result is True
        assert mavlink_service.get_parameter("PISAD_FREQ_PROF") == 1.0

    @patch("pymavlink.mavutil.mavlink_connection")
    def test_param_request_read_message_handling(
        self, mock_mavutil, mavlink_service, mock_connection
    ):
        """Test PARAM_REQUEST_READ message handling."""
        # Setup mock connection
        mock_mavutil.return_value = mock_connection
        mavlink_service.connection = mock_connection

        # Set a parameter value
        mavlink_service.set_parameter("PISAD_FREQ_HZ", 406000000.0)  # 406 MHz

        # Create mock PARAM_REQUEST_READ message
        param_request_msg = MagicMock()
        param_request_msg.get_type.return_value = "PARAM_REQUEST_READ"
        param_request_msg.param_id = "PISAD_FREQ_HZ"
        param_request_msg.param_index = -1

        # Process parameter request
        result = mavlink_service._handle_param_request_read(param_request_msg)

        # Verify parameter was sent back
        assert result is True
        # Verify PARAM_VALUE message was sent
        mock_connection.mav.param_value_send.assert_called_once()

    def test_frequency_profile_parameter_validation(self, mavlink_service):
        """Test frequency profile parameter validation."""
        # Test valid profile values (0=Emergency, 1=Aviation, 2=Custom)
        assert mavlink_service.set_parameter("PISAD_FREQ_PROF", 0.0) is True
        assert mavlink_service.set_parameter("PISAD_FREQ_PROF", 1.0) is True
        assert mavlink_service.set_parameter("PISAD_FREQ_PROF", 2.0) is True

        # Test invalid profile values
        assert mavlink_service.set_parameter("PISAD_FREQ_PROF", -1.0) is False
        assert mavlink_service.set_parameter("PISAD_FREQ_PROF", 3.0) is False
        assert mavlink_service.set_parameter("PISAD_FREQ_PROF", 999.0) is False

    def test_custom_frequency_parameter_validation(self, mavlink_service):
        """Test custom frequency parameter validation."""
        # Test valid frequencies within HackRF range (1 MHz - 6 GHz)
        assert mavlink_service.set_parameter("PISAD_FREQ_HZ", 1000000.0) is True  # 1 MHz
        assert mavlink_service.set_parameter("PISAD_FREQ_HZ", 406000000.0) is True  # 406 MHz
        assert mavlink_service.set_parameter("PISAD_FREQ_HZ", 6000000000.0) is True  # 6 GHz

        # Test invalid frequencies outside HackRF range
        assert (
            mavlink_service.set_parameter("PISAD_FREQ_HZ", 500000.0) is False
        )  # 500 kHz (too low)
        assert (
            mavlink_service.set_parameter("PISAD_FREQ_HZ", 7000000000.0) is False
        )  # 7 GHz (too high)

    def test_homing_enable_parameter_validation(self, mavlink_service):
        """Test homing mode enable parameter validation."""
        # Test valid boolean values
        assert mavlink_service.set_parameter("PISAD_HOMING_EN", 0.0) is True  # Disabled
        assert mavlink_service.set_parameter("PISAD_HOMING_EN", 1.0) is True  # Enabled

        # Test invalid values
        assert mavlink_service.set_parameter("PISAD_HOMING_EN", -1.0) is False
        assert mavlink_service.set_parameter("PISAD_HOMING_EN", 2.0) is False

    def test_parameter_change_callback_integration(self, mavlink_service):
        """Test parameter change callbacks for frequency switching."""
        # Setup callback to track parameter changes
        callback_calls = []

        def param_change_callback(param_name: str, value: float):
            callback_calls.append((param_name, value))

        mavlink_service.add_parameter_callback(param_change_callback)

        # Change frequency profile parameter
        mavlink_service.set_parameter("PISAD_FREQ_PROF", 1.0)  # Aviation profile

        # Verify callback was called
        assert len(callback_calls) == 1
        assert callback_calls[0] == ("PISAD_FREQ_PROF", 1.0)

    def test_response_time_requirement(self, mavlink_service):
        """Test <50ms response time requirement for frequency changes."""
        # Measure parameter processing time
        start_time = time.time()

        result = mavlink_service.set_parameter("PISAD_FREQ_PROF", 2.0)

        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000

        # Verify parameter was set successfully
        assert result is True

        # Verify response time is under 50ms requirement
        assert (
            processing_time_ms < 50.0
        ), f"Parameter processing took {processing_time_ms}ms, exceeds 50ms requirement"

    @patch("pymavlink.mavutil.mavlink_connection")
    def test_mission_planner_integration_workflow(
        self, mock_mavutil, mavlink_service, mock_connection
    ):
        """Test complete Mission Planner frequency control workflow."""
        # Setup mock connection
        mock_mavutil.return_value = mock_connection
        mavlink_service.connection = mock_connection

        # Step 1: Mission Planner requests current frequency profile
        param_request = MagicMock()
        param_request.get_type.return_value = "PARAM_REQUEST_READ"
        param_request.param_id = "PISAD_FREQ_PROF"
        param_request.param_index = -1

        result = mavlink_service._handle_param_request_read(param_request)
        assert result is True

        # Step 2: Mission Planner sets new frequency profile (Aviation)
        param_set = MagicMock()
        param_set.get_type.return_value = "PARAM_SET"
        param_set.param_id = "PISAD_FREQ_PROF"
        param_set.param_value = 1.0  # Aviation profile
        param_set.param_type = 2

        result = mavlink_service._handle_param_set(param_set)
        assert result is True
        assert mavlink_service.get_parameter("PISAD_FREQ_PROF") == 1.0

        # Step 3: Mission Planner enables homing mode
        homing_set = MagicMock()
        homing_set.get_type.return_value = "PARAM_SET"
        homing_set.param_id = "PISAD_HOMING_EN"
        homing_set.param_value = 1.0  # Enable homing
        homing_set.param_type = 2

        result = mavlink_service._handle_param_set(homing_set)
        assert result is True
        assert mavlink_service.get_parameter("PISAD_HOMING_EN") == 1.0
