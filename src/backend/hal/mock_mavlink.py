"""
Mock MAVLink interface for testing without hardware
"""

import logging
import time
from unittest.mock import Mock

logger = logging.getLogger(__name__)


class MockMAVLinkConnection:
    """Mock MAVLink connection for testing"""

    def __init__(self, device: str, baudrate: int = 115200):
        self.device = device
        self.baudrate = baudrate
        self.connected = False
        self.heartbeat_received = False
        self.mode_map = {
            "MANUAL": 0,
            "ACRO": 1,
            "ALT_HOLD": 2,
            "AUTO": 3,
            "GUIDED": 4,
            "LOITER": 5,
            "RTL": 6,
            "CIRCLE": 7,
            "LAND": 9,
            "DRIFT": 11,
            "SPORT": 13,
            "FLIP": 14,
            "AUTOTUNE": 15,
            "POSHOLD": 16,
            "BRAKE": 17,
            "THROW": 18,
            "AVOID_ADSB": 19,
            "GUIDED_NOGPS": 20,
            "SMART_RTL": 21,
        }
        self.current_mode = "MANUAL"
        self.armed = False
        self.mav = Mock()
        self._setup_mav_mock()

    def _setup_mav_mock(self):
        """Setup MAV mock methods"""
        self.mav.heartbeat_send = Mock()
        self.mav.set_position_target_local_ned_send = Mock()
        self.mav.statustext_send = Mock()
        self.mav.command_long_send = Mock()

    def wait_heartbeat(self, timeout: float = 3.0) -> Mock | None:
        """Mock heartbeat waiting"""
        if not self.connected:
            return None

        # Simulate heartbeat message
        heartbeat = Mock()
        heartbeat.autopilot = 3  # MAV_AUTOPILOT_ARDUPILOTMEGA
        heartbeat.get_srcSystem.return_value = 1
        heartbeat.custom_mode = self.mode_map.get(self.current_mode, 0)
        heartbeat.base_mode = 81  # MAV_MODE_FLAG_GUIDED_ENABLED | others

        self.heartbeat_received = True
        return heartbeat

    def recv_match(self, type: str, blocking: bool = False, timeout: float = 1.0) -> Mock | None:
        """Mock message receiving"""
        if not self.connected:
            return None

        if type == "HEARTBEAT":
            msg = Mock()
            msg.custom_mode = self.mode_map.get(self.current_mode, 0)
            msg.base_mode = 81
            return msg
        elif type == "GLOBAL_POSITION_INT":
            msg = Mock()
            msg.lat = 377749000  # 37.7749 * 1e7
            msg.lon = -1224194000  # -122.4194 * 1e7
            msg.alt = 30000  # 30m in mm
            return msg
        elif type == "SYS_STATUS":
            msg = Mock()
            msg.voltage_battery = 22000  # 22V in mV
            msg.current_battery = 500  # 5A in cA
            msg.battery_remaining = 75  # 75%
            return msg
        elif type == "RC_CHANNELS":
            msg = Mock()
            msg.chan1_raw = 1500  # Roll
            msg.chan2_raw = 1500  # Pitch
            msg.chan3_raw = 1500  # Throttle
            msg.chan4_raw = 1500  # Yaw
            msg.chan5_raw = 1500  # Mode
            msg.chan6_raw = 1500  # Aux1
            msg.chan7_raw = 1500  # Aux2
            msg.chan8_raw = 1500  # Aux3
            return msg
        elif type == "GPS_RAW_INT":
            msg = Mock()
            msg.fix_type = 3  # 3D fix
            msg.satellites_visible = 12
            msg.eph = 150  # HDOP * 100
            return msg
        elif type == "COMMAND_ACK":
            msg = Mock()
            msg.result = 0  # MAV_RESULT_ACCEPTED
            return msg

        return None

    def mode_mapping(self) -> dict:
        """Return mode mapping"""
        return self.mode_map if self.connected else None

    def close(self):
        """Close connection"""
        self.connected = False
        logger.info(f"Mock MAVLink connection to {self.device} closed")


class MockMAVUtil:
    """Mock mavutil module for testing"""

    class mavlink:
        """Mock MAVLink constants"""

        MAV_TYPE_GCS = 6
        MAV_AUTOPILOT_INVALID = 8
        POSITION_TARGET_TYPEMASK_X_IGNORE = 1
        POSITION_TARGET_TYPEMASK_Y_IGNORE = 2
        POSITION_TARGET_TYPEMASK_Z_IGNORE = 4
        POSITION_TARGET_TYPEMASK_AX_IGNORE = 2048
        POSITION_TARGET_TYPEMASK_AY_IGNORE = 4096
        POSITION_TARGET_TYPEMASK_AZ_IGNORE = 8192
        POSITION_TARGET_TYPEMASK_YAW_IGNORE = 1024
        MAV_FRAME_LOCAL_NED = 1
        MAV_CMD_DO_SET_MODE = 176
        MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 1
        MAV_CMD_COMPONENT_ARM_DISARM = 400
        MAV_RESULT_ACCEPTED = 0
        MAV_MODE_FLAG_GUIDED_ENABLED = 4

    @staticmethod
    def mavlink_connection(
        device: str, source_system: int = 255, source_component: int = 190
    ) -> MockMAVLinkConnection:
        """Create mock MAVLink connection"""
        connection = MockMAVLinkConnection(device)

        # Simulate connection success/failure based on device
        if "fail" in device.lower() or "invalid" in device.lower():
            connection.connected = False
        else:
            connection.connected = True

        return connection


# Mock the entire mavutil for testing
def create_mock_mavutil():
    """Create complete mock mavutil"""
    return MockMAVUtil()
