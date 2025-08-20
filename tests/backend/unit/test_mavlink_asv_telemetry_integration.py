"""Unit tests for MAVLink ASV enhanced telemetry integration.

Tests the enhanced RF telemetry capabilities including ASV signal classification,
bearing confidence, and signal quality indicators for Mission Planner display.

SUBTASK-6.1.3.2 [19a][19b][19c][19d] - Enhanced RF telemetry integration
"""

import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.backend.services.mavlink_service import ConnectionState, MAVLinkService


@dataclass
class ASVBearingCalculation:
    """Mock ASV bearing calculation for testing."""

    bearing_deg: float = 90.0
    confidence: float = 0.85
    precision_deg: float = 2.5
    signal_strength_dbm: float = -75.0
    signal_quality: float = 0.78
    timestamp_ns: int = field(default_factory=lambda: int(time.time_ns()))
    analyzer_type: str = "ASV_PROFESSIONAL"
    interference_detected: bool = False
    signal_classification: str = "FM_CHIRP"
    chirp_characteristics: dict[str, Any] | None = None
    interference_analysis: dict[str, Any] | None = None
    raw_asv_data: dict[str, Any] | None = None


class TestMAVLinkASVTelemetryIntegration:
    """Test MAVLink ASV enhanced telemetry integration."""

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

    @pytest.fixture
    def asv_bearing_calculation(self):
        """High-quality ASV bearing calculation."""
        return ASVBearingCalculation(
            bearing_deg=135.5,
            confidence=0.92,
            precision_deg=1.8,
            signal_strength_dbm=-68.5,
            signal_quality=0.89,
            analyzer_type="ASV_PROFESSIONAL",
            interference_detected=False,
            signal_classification="FM_CHIRP",
        )

    @pytest.fixture
    def asv_bearing_with_interference(self):
        """ASV bearing calculation with interference."""
        return ASVBearingCalculation(
            bearing_deg=200.0,
            confidence=0.35,
            precision_deg=12.0,
            signal_strength_dbm=-95.0,
            signal_quality=0.42,
            analyzer_type="ASV_ENHANCED",
            interference_detected=True,
            signal_classification="INTERFERENCE",
        )

    def test_asv_bearing_telemetry_transmission(
        self, mavlink_service, mock_connection, asv_bearing_calculation
    ):
        """Test ASV bearing telemetry transmission to Mission Planner."""
        # Setup mock connection
        mavlink_service.connection = mock_connection
        mavlink_service._running = True
        mavlink_service.state = ConnectionState.CONNECTED

        # Mock send_named_value_float method
        mavlink_service.send_named_value_float = MagicMock()
        mavlink_service.send_statustext = MagicMock()

        # Act
        mavlink_service.send_asv_bearing_telemetry(asv_bearing_calculation)

        # Assert - Verify all ASV telemetry values sent
        expected_calls = [
            ("ASV_BEARING", 135.5),
            ("ASV_CONFIDENCE", 92.0),  # Converted to percentage
            ("ASV_PRECISION", 1.8),
            ("ASV_SIG_QUAL", 89.0),  # Converted to percentage
            ("ASV_SIG_RSSI", -68.5),
            ("ASV_INTERF", 0.0),  # No interference
            ("ASV_SIG_TYPE", 2.0),  # FM_CHIRP classification code
        ]

        for param_name, expected_value in expected_calls:
            mavlink_service.send_named_value_float.assert_any_call(
                param_name, expected_value, pytest.approx(time.time(), abs=1.0)
            )

        # Verify status text was sent
        mavlink_service.send_statustext.assert_called_once()
        status_args = mavlink_service.send_statustext.call_args[0]
        assert "ASV: 135.5°" in status_args[0]
        assert "±1.8°" in status_args[0]
        assert "Conf:92%" in status_args[0]
        assert "FM_CHIRP" in status_args[0]

    def test_asv_bearing_telemetry_with_interference(
        self, mavlink_service, mock_connection, asv_bearing_with_interference
    ):
        """Test ASV bearing telemetry with interference detection."""
        # Setup
        mavlink_service.connection = mock_connection
        mavlink_service._running = True
        mavlink_service.state = ConnectionState.CONNECTED
        mavlink_service.send_named_value_float = MagicMock()
        mavlink_service.send_statustext = MagicMock()

        # Act
        mavlink_service.send_asv_bearing_telemetry(asv_bearing_with_interference)

        # Assert - Verify interference flag is set
        mavlink_service.send_named_value_float.assert_any_call(
            "ASV_INTERF", 1.0, pytest.approx(time.time(), abs=1.0)
        )

        # Verify INTERFERENCE classification code (4)
        mavlink_service.send_named_value_float.assert_any_call(
            "ASV_SIG_TYPE", 4.0, pytest.approx(time.time(), abs=1.0)
        )

    def test_asv_signal_quality_telemetry(self, mavlink_service, mock_connection):
        """Test ASV signal quality and trend telemetry."""
        # Setup
        mavlink_service.connection = mock_connection
        mavlink_service._running = True
        mavlink_service.state = ConnectionState.CONNECTED
        mavlink_service.send_named_value_float = MagicMock()

        signal_quality_data = {
            "rssi_trend": 2.5,  # Improving signal
            "signal_stability": 0.85,
            "frequency_drift_hz": -120.0,
            "multipath_severity": 0.15,
        }

        # Act
        mavlink_service.send_asv_signal_quality_telemetry(signal_quality_data)

        # Assert - Verify all signal quality metrics sent
        expected_calls = [
            ("ASV_RSSI_TREND", 2.5),
            ("ASV_STABILITY", 85.0),  # Converted to percentage
            ("ASV_FREQ_DRIFT", -120.0),
            ("ASV_MULTIPATH", 15.0),  # Converted to percentage
        ]

        for param_name, expected_value in expected_calls:
            mavlink_service.send_named_value_float.assert_any_call(
                param_name, expected_value, pytest.approx(time.time(), abs=1.0)
            )

    def test_asv_detection_event_telemetry(self, mavlink_service, mock_connection):
        """Test ASV detection event notification telemetry."""
        # Setup
        mavlink_service.connection = mock_connection
        mavlink_service._running = True
        mavlink_service.state = ConnectionState.CONNECTED
        mavlink_service.send_named_value_float = MagicMock()
        mavlink_service.send_statustext = MagicMock()

        detection_event = {
            "event_type": "BEACON_CONFIRMED",
            "detection_strength": 0.78,
            "analyzer_source": "ASV_PROFESSIONAL",
            "frequency_hz": 406025000,  # 406.025 MHz
        }

        # Act
        mavlink_service.send_asv_detection_event_telemetry(detection_event)

        # Assert - Verify detection event telemetry
        expected_calls = [
            ("ASV_DET_EVENT", 8.0),  # BEACON_CONFIRMED code
            ("ASV_DET_STR", 78.0),  # Converted to percentage
            ("ASV_ANALYZER", 1.0),  # ASV_PROFESSIONAL code
        ]

        for param_name, expected_value in expected_calls:
            mavlink_service.send_named_value_float.assert_any_call(
                param_name, expected_value, pytest.approx(time.time(), abs=1.0)
            )

        # Verify status message contains event details
        mavlink_service.send_statustext.assert_called_once()
        status_args = mavlink_service.send_statustext.call_args[0]
        assert "ASV BEACON_CONFIRMED" in status_args[0]
        assert "406.025MHz" in status_args[0]
        assert "Strength:78%" in status_args[0]
        assert "ASV_PROFESSIONAL" in status_args[0]

    def test_signal_classification_mapping(self, mavlink_service):
        """Test signal classification to numeric code mapping."""
        # Test known classifications
        assert mavlink_service._map_signal_classification_to_code("CONTINUOUS") == 1
        assert mavlink_service._map_signal_classification_to_code("FM_CHIRP") == 2
        assert mavlink_service._map_signal_classification_to_code("FM_CHIRP_WEAK") == 3
        assert mavlink_service._map_signal_classification_to_code("INTERFERENCE") == 4
        assert mavlink_service._map_signal_classification_to_code("BEACON_121_5") == 5
        assert mavlink_service._map_signal_classification_to_code("BEACON_406") == 6
        assert mavlink_service._map_signal_classification_to_code("AVIATION") == 7

        # Test unknown classification defaults to 0
        assert mavlink_service._map_signal_classification_to_code("UNKNOWN_TYPE") == 0

    def test_detection_event_mapping(self, mavlink_service):
        """Test detection event type to numeric code mapping."""
        # Test known event types
        assert mavlink_service._map_detection_event_to_code("DETECTION") == 1
        assert mavlink_service._map_detection_event_to_code("SIGNAL_LOST") == 2
        assert mavlink_service._map_detection_event_to_code("SIGNAL_IMPROVED") == 3
        assert mavlink_service._map_detection_event_to_code("INTERFERENCE_DETECTED") == 4
        assert mavlink_service._map_detection_event_to_code("BEACON_CONFIRMED") == 8

        # Test unknown event defaults to 0
        assert mavlink_service._map_detection_event_to_code("CUSTOM_EVENT") == 0

    def test_analyzer_source_mapping(self, mavlink_service):
        """Test ASV analyzer source to numeric code mapping."""
        # Test known analyzer sources
        assert mavlink_service._map_analyzer_source_to_code("ASV_PROFESSIONAL") == 1
        assert mavlink_service._map_analyzer_source_to_code("ASV_STANDARD") == 2
        assert mavlink_service._map_analyzer_source_to_code("ASV_ENHANCED") == 3
        assert mavlink_service._map_analyzer_source_to_code("HACKRF_DIRECT") == 4
        assert mavlink_service._map_analyzer_source_to_code("HYBRID_ASV") == 5

        # Test unknown source defaults to 0
        assert mavlink_service._map_analyzer_source_to_code("CUSTOM_ANALYZER") == 0

    def test_asv_telemetry_without_connection(self, mavlink_service, asv_bearing_calculation):
        """Test ASV telemetry gracefully handles no connection."""
        # Setup - No connection
        mavlink_service.connection = None
        mavlink_service.state = ConnectionState.DISCONNECTED

        # Act - Should not raise exceptions
        mavlink_service.send_asv_bearing_telemetry(asv_bearing_calculation)
        mavlink_service.send_asv_signal_quality_telemetry({"rssi_trend": 1.0})
        mavlink_service.send_asv_detection_event_telemetry({"event_type": "DETECTION"})

        # Assert - No exceptions raised (test passes if no exception)

    def test_asv_telemetry_error_handling(self, mavlink_service, mock_connection):
        """Test ASV telemetry error handling with malformed data."""
        # Setup
        mavlink_service.connection = mock_connection
        mavlink_service._running = True
        mavlink_service.state = ConnectionState.CONNECTED
        mavlink_service.send_named_value_float = MagicMock(side_effect=Exception("Telemetry error"))

        # Create malformed bearing calculation
        malformed_bearing = ASVBearingCalculation()
        malformed_bearing.bearing_deg = None  # Invalid data

        # Act - Should handle errors gracefully
        try:
            mavlink_service.send_asv_bearing_telemetry(malformed_bearing)
            # Test should not fail even with malformed data
        except Exception:
            pytest.fail("ASV telemetry should handle errors gracefully")
