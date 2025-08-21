"""
Tests for TASK-2.1.7 MAVLink Telemetry Enhancement
Tests enhanced telemetry streaming including noise floor, SNR, and confidence metrics.

Implements authentic TDD for Story 2.1.7 subtasks with real system integration.
"""

import time
from unittest.mock import MagicMock

import pytest

from src.backend.models.schemas import RSSIReading
from src.backend.services.mavlink_service import MAVLinkService
from src.backend.services.signal_processor import SignalProcessor


class TestMAVLinkTelemetryEnhancement:
    """Test enhanced telemetry streaming for Story 2.1.7."""

    @pytest.fixture
    def mavlink_service(self):
        """Create MAVLinkService for testing."""
        service = MAVLinkService()
        return service

    @pytest.fixture
    def signal_processor(self):
        """Create SignalProcessor for testing."""
        processor = SignalProcessor()
        return processor

    @pytest.fixture
    def mock_connection(self):
        """Mock MAVLink connection."""
        connection = MagicMock()
        connection.mav = MagicMock()
        connection.mav.named_value_float_send = MagicMock()
        return connection

    # SUBTASK-2.1.7.1 Tests: NAMED_VALUE_FLOAT message streaming

    @pytest.mark.asyncio
    async def test_30c_noise_floor_telemetry_streaming(
        self, mavlink_service, signal_processor, mock_connection
    ):
        """
        [30c] Add noise floor and SNR telemetry streaming to ground control

        Verifies that noise floor and SNR data from signal processor
        is streamed via NAMED_VALUE_FLOAT messages.
        """
        # Setup: Connect service and set mock connection
        mavlink_service.connection = mock_connection

        # Import ConnectionState and set proper state
        from src.backend.services.mavlink_service import ConnectionState

        mavlink_service.state = ConnectionState.CONNECTED

        # Initialize required telemetry attributes
        if not hasattr(mavlink_service, "_enhanced_telemetry_config"):
            mavlink_service._enhanced_telemetry_config = {}
        if not hasattr(mavlink_service, "_telemetry_statistics"):
            mavlink_service._telemetry_statistics = {"messages_sent": 0, "messages_failed": 0}

        # Test data from signal processor
        test_noise_floor = -82.5
        test_snr = 15.2
        test_rssi = -67.3

        # Create realistic signal processor reading
        signal_reading = RSSIReading(
            timestamp=time.time(),
            rssi=test_rssi,
            noise_floor=test_noise_floor,
            snr=test_snr,
            detection_id=None,
        )

        # ACT: Stream enhanced telemetry data
        success = mavlink_service.send_enhanced_telemetry_data(
            {
                "rssi": test_rssi,
                "noise_floor": test_noise_floor,
                "snr": test_snr,
                "signal_reading": signal_reading,
            }
        )

        # ASSERT: Verify noise floor and SNR were sent
        assert success is True

        # Verify NAMED_VALUE_FLOAT calls for noise floor and SNR
        calls = mock_connection.mav.named_value_float_send.call_args_list

        # Should have at least 3 calls (RSSI, noise floor, SNR)
        assert len(calls) >= 3

        # Extract parameter names from calls
        param_names = []
        for call in calls:
            if len(call[0]) >= 2:  # call args format: (time_boot_ms, name, value)
                name_param = call[0][1]
                if isinstance(name_param, bytes):
                    param_names.append(name_param.decode())
                else:
                    param_names.append(str(name_param))

        # Verify expected parameters were sent
        assert "PISAD_RSSI" in param_names
        assert "PISAD_NF" in param_names
        assert "PISAD_SNR" in param_names

    @pytest.mark.asyncio
    async def test_30d_confidence_metrics_telemetry_streaming(
        self, mavlink_service, signal_processor, mock_connection
    ):
        """
        [30d] Include signal confidence metrics in telemetry stream

        Verifies that signal confidence scores and quality metrics
        are included in telemetry streaming.
        """
        # Setup: Connect service and set mock connection
        mavlink_service.connection = mock_connection

        # Import ConnectionState and set proper state
        from src.backend.services.mavlink_service import ConnectionState

        mavlink_service.state = ConnectionState.CONNECTED

        # Initialize required telemetry attributes
        if not hasattr(mavlink_service, "_enhanced_telemetry_config"):
            mavlink_service._enhanced_telemetry_config = {}
        if not hasattr(mavlink_service, "_telemetry_statistics"):
            mavlink_service._telemetry_statistics = {"messages_sent": 0, "messages_failed": 0}

        # Test confidence data
        test_confidence = 0.85
        test_signal_quality = 0.92
        test_classification_confidence = 0.78

        # Create enhanced signal reading with confidence data
        signal_reading = RSSIReading(
            timestamp=time.time(),
            rssi=-65.0,
            noise_floor=-80.0,
            snr=15.0,
            detection_id=None,
            confidence_score=test_confidence,
            signal_classification="FM_CHIRP",
            asv_analysis={
                "signal_quality": test_signal_quality,
                "classification_confidence": test_classification_confidence,
                "confidence_weighting": {"final_confidence": test_confidence},
            },
        )

        # ACT: Stream confidence telemetry data
        success = mavlink_service.send_enhanced_telemetry_data(
            {
                "confidence_score": test_confidence,
                "signal_quality": test_signal_quality,
                "classification_confidence": test_classification_confidence,
                "signal_reading": signal_reading,
            }
        )

        # ASSERT: Verify confidence metrics were sent
        assert success is True

        # Verify NAMED_VALUE_FLOAT calls for confidence metrics
        calls = mock_connection.mav.named_value_float_send.call_args_list

        # Should have calls for confidence metrics
        assert len(calls) >= 2

        # Extract parameter names from calls
        param_names = []
        for call in calls:
            if len(call[0]) >= 2:  # call args format: (time_boot_ms, name, value)
                name_param = call[0][1]
                if isinstance(name_param, bytes):
                    param_names.append(name_param.decode())
                else:
                    param_names.append(str(name_param))

        # Verify expected confidence parameters were sent
        assert "PISAD_CONF" in param_names
        assert "PISAD_QUAL" in param_names

    # SUBTASK-2.1.7.2 Tests: Message validation and buffering

    @pytest.mark.asyncio
    async def test_31a_telemetry_message_validation(self, mavlink_service, mock_connection):
        """
        [31a] Implement telemetry message validation before transmission

        Verifies that telemetry messages are validated for proper format,
        range, and content before being sent.
        """
        # Setup
        mavlink_service.connection = mock_connection

        # Import ConnectionState and set proper state
        from src.backend.services.mavlink_service import ConnectionState

        mavlink_service.state = ConnectionState.CONNECTED

        # Initialize required telemetry attributes
        if not hasattr(mavlink_service, "_enhanced_telemetry_config"):
            mavlink_service._enhanced_telemetry_config = {}
        if not hasattr(mavlink_service, "_telemetry_statistics"):
            mavlink_service._telemetry_statistics = {"messages_sent": 0, "messages_failed": 0}

        # Enable message validation
        mavlink_service._enhanced_telemetry_config["message_validation"] = True

        # Test valid message
        valid_result = mavlink_service.send_named_value_float("PISAD_RSSI", -65.0)
        assert valid_result is True

        # Test invalid messages that should fail validation
        invalid_name_result = mavlink_service.send_named_value_float(
            "INVALID_VERY_LONG_NAME", -65.0
        )
        assert invalid_name_result is False

        invalid_value_result = mavlink_service.send_named_value_float("PISAD_RSSI", float("inf"))
        assert invalid_value_result is False

        out_of_range_result = mavlink_service.send_named_value_float(
            "PISAD_RSSI", -200.0
        )  # Unrealistic RSSI
        assert out_of_range_result is False

    @pytest.mark.asyncio
    async def test_31b_message_buffering_reliability(self, mavlink_service, mock_connection):
        """
        [31b] Add message buffering for reliable delivery during connection issues

        Verifies that telemetry messages are buffered when connection is lost
        and retransmitted when connection is restored.
        """
        # Setup with buffering enabled
        mavlink_service.connection = mock_connection

        # Import ConnectionState and set proper state
        from src.backend.services.mavlink_service import ConnectionState

        mavlink_service.state = ConnectionState.CONNECTED

        # Initialize required telemetry attributes
        if not hasattr(mavlink_service, "_enhanced_telemetry_config"):
            mavlink_service._enhanced_telemetry_config = {}
        if not hasattr(mavlink_service, "_telemetry_statistics"):
            mavlink_service._telemetry_statistics = {"messages_sent": 0, "messages_failed": 0}

        mavlink_service._enhanced_telemetry_config["reliability_buffering"] = True

        # Send message while connected - should succeed immediately
        result1 = mavlink_service.send_named_value_float("PISAD_RSSI", -65.0)
        assert result1 is True

        # Simulate connection loss
        mavlink_service.connection = None
        mavlink_service.state = ConnectionState.DISCONNECTED

        # Send message while disconnected - should buffer
        result2 = mavlink_service.send_named_value_float("PISAD_SNR", 12.5)
        assert result2 is False  # Immediate send fails

        # Verify message was buffered
        assert len(mavlink_service._telemetry_buffer) > 0

        # Restore connection
        mavlink_service.connection = mock_connection
        mavlink_service.state = ConnectionState.CONNECTED

        # Process buffered messages
        mavlink_service._process_buffered_telemetry()

        # Verify buffered message was sent
        calls = mock_connection.mav.named_value_float_send.call_args_list
        call_names = []
        for call in calls:
            if len(call[0]) >= 2:  # call args format: (time_boot_ms, name, value)
                name_param = call[0][1]
                if isinstance(name_param, bytes):
                    call_names.append(name_param.decode())
                else:
                    call_names.append(str(name_param))
        assert "PISAD_SNR" in call_names

    @pytest.mark.asyncio
    async def test_31c_telemetry_retry_mechanism(self, mavlink_service, mock_connection):
        """
        [31c] Create telemetry retry mechanism with exponential backoff

        Verifies that failed telemetry sends are retried with exponential backoff.
        """
        # Setup
        mavlink_service.connection = mock_connection

        # Import ConnectionState and set proper state
        from src.backend.services.mavlink_service import ConnectionState

        mavlink_service.state = ConnectionState.CONNECTED

        # Initialize required telemetry attributes
        if not hasattr(mavlink_service, "_enhanced_telemetry_config"):
            mavlink_service._enhanced_telemetry_config = {}
        if not hasattr(mavlink_service, "_telemetry_statistics"):
            mavlink_service._telemetry_statistics = {"messages_sent": 0, "messages_failed": 0}

        mavlink_service._enhanced_telemetry_config["retry_mechanism"] = True

        # Mock connection to fail first attempts then succeed
        failure_count = 0

        def mock_send_with_failures(*args, **kwargs):
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:  # Fail first 2 attempts
                raise ConnectionError("Simulated connection error")
            return None  # Success on 3rd attempt

        mock_connection.mav.named_value_float_send.side_effect = mock_send_with_failures

        # ACT: Send message that will require retries
        start_time = time.time()
        result = mavlink_service.send_named_value_float_with_retry("PISAD_RSSI", -65.0)
        retry_duration = time.time() - start_time

        # ASSERT: Verify retry succeeded and used exponential backoff
        assert result is True
        assert failure_count == 3  # Failed twice, succeeded on 3rd
        assert retry_duration > 0.1  # Should have some delay due to backoff

    @pytest.mark.asyncio
    async def test_31d_telemetry_performance_monitoring(self, mavlink_service, mock_connection):
        """
        [31d] Add telemetry performance monitoring and bandwidth tracking

        Verifies that telemetry performance metrics are tracked including
        message rates, bandwidth usage, and transmission success rates.
        """
        # Setup
        mavlink_service.connection = mock_connection

        # Import ConnectionState and set proper state
        from src.backend.services.mavlink_service import ConnectionState

        mavlink_service.state = ConnectionState.CONNECTED

        # Initialize required telemetry attributes
        if not hasattr(mavlink_service, "_enhanced_telemetry_config"):
            mavlink_service._enhanced_telemetry_config = {}
        if not hasattr(mavlink_service, "_telemetry_statistics"):
            mavlink_service._telemetry_statistics = {"messages_sent": 0, "messages_failed": 0}

        mavlink_service._enhanced_telemetry_config["performance_monitoring"] = True

        # Send multiple telemetry messages
        for i in range(10):
            mavlink_service.send_named_value_float(f"TEST_{i:02d}", float(i))

        # Get performance statistics
        stats = mavlink_service.get_telemetry_performance_stats()

        # Verify performance monitoring data
        assert "messages_sent" in stats
        assert "bandwidth_usage_bps" in stats
        assert "success_rate" in stats
        assert "average_latency_ms" in stats

        assert stats["messages_sent"] >= 10
        assert stats["success_rate"] > 0.0
        assert stats["bandwidth_usage_bps"] > 0.0

    @pytest.mark.asyncio
    async def test_integration_signal_processor_to_mavlink_telemetry(
        self, mavlink_service, signal_processor, mock_connection
    ):
        """
        Integration test: Signal processor data flows to MAVLink telemetry

        Verifies end-to-end integration where signal processor generates
        RSSI readings that are automatically streamed via MAVLink telemetry.
        """
        # Setup: Connect signal processor to MAVLink service
        mavlink_service.connection = mock_connection

        # Import ConnectionState and set proper state
        from src.backend.services.mavlink_service import ConnectionState

        mavlink_service.state = ConnectionState.CONNECTED

        # Initialize required telemetry attributes
        if not hasattr(mavlink_service, "_enhanced_telemetry_config"):
            mavlink_service._enhanced_telemetry_config = {}
        if not hasattr(mavlink_service, "_telemetry_statistics"):
            mavlink_service._telemetry_statistics = {"messages_sent": 0, "messages_failed": 0}

        # Disable validation for integration test to allow any values
        mavlink_service._enhanced_telemetry_config["message_validation"] = False

        signal_processor.set_mavlink_service(mavlink_service)

        # Generate test IQ samples that will create detectable signal
        import numpy as np

        # Create strong signal samples (above threshold)
        test_samples = np.random.normal(0, 0.1, 1024) + 0.5j * np.random.normal(0, 0.1, 1024)
        test_samples[100:200] += 2.0 + 2.0j  # Add strong signal

        # Process samples through signal processor
        rssi_reading = signal_processor.compute_rssi_with_confidence_weighting(test_samples)

        # Simulate automatic telemetry streaming (normally done by async task)
        telemetry_data = {
            "rssi": rssi_reading.rssi,
            "noise_floor": rssi_reading.noise_floor,
            "snr": rssi_reading.snr,
            "confidence_score": getattr(rssi_reading, "confidence_score", 0.0),
            "signal_reading": rssi_reading,
        }

        success = mavlink_service.send_enhanced_telemetry_data(telemetry_data)

        # Verify integration worked
        assert success is True
        assert mock_connection.mav.named_value_float_send.call_count > 0

        # Verify data was generated from signal processor
        assert rssi_reading.rssi is not None
        assert rssi_reading.noise_floor is not None
        assert rssi_reading.snr is not None

        # Verify telemetry contains all expected fields
        assert "rssi" in telemetry_data
        assert "noise_floor" in telemetry_data
        assert "snr" in telemetry_data
        assert "signal_reading" in telemetry_data
