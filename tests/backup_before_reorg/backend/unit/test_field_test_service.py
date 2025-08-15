"""Comprehensive tests for field test service with 60%+ coverage target."""

import json
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest
import yaml

from src.backend.models.schemas import BeaconConfiguration, FieldTestMetrics
from src.backend.services.field_test_service import (
    FieldTestConfig,
    FieldTestService,
    FieldTestStatus,
)
from src.backend.utils.test_logger import (
    TestLogger,
)


@pytest.fixture
def mock_test_logger():
    """Create mock test logger."""
    mock = MagicMock(spec=TestLogger)
    mock.log_test_run = MagicMock()
    return mock


@pytest.fixture
def mock_state_machine():
    """Create mock state machine."""
    mock = MagicMock()
    mock.current_state = "IDLE"
    mock.request_transition = AsyncMock(return_value=True)
    mock.emergency_stop = AsyncMock()
    return mock


@pytest.fixture
def mock_mavlink_service():
    """Create mock MAVLink service."""
    mock = MagicMock()
    mock.connected = True
    mock.get_telemetry = AsyncMock(
        return_value={
            "gps_status": "3D_FIX",
            "battery_percent": 80,
            "position": {"lat": -35.363261, "lon": 149.165230, "alt": 50},
            "armed": False,
        }
    )
    mock.upload_mission = AsyncMock(return_value=True)
    mock.start_mission = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_signal_processor():
    """Create mock signal processor."""
    mock = MagicMock()
    mock.current_rssi = -75.0
    mock.is_processing = True
    mock.process_samples = AsyncMock(return_value=-70.0)
    return mock


@pytest.fixture
def mock_safety_manager():
    """Create mock safety interlock system."""
    mock = MagicMock()
    mock.check_all_safety_interlocks = AsyncMock(
        return_value={
            "geofence_check": True,
            "battery_check": True,
            "signal_check": True,
        }
    )
    mock.emergency_stop = AsyncMock()
    return mock


@pytest.fixture
def field_test_service(
    mock_test_logger,
    mock_state_machine,
    mock_mavlink_service,
    mock_signal_processor,
    mock_safety_manager,
):
    """Create field test service with mocked dependencies."""
    service = FieldTestService(
        test_logger=mock_test_logger,
        state_machine=mock_state_machine,
        mavlink_service=mock_mavlink_service,
        signal_processor=mock_signal_processor,
        safety_manager=mock_safety_manager,
    )
    return service


@pytest.fixture
def beacon_config():
    """Create beacon configuration."""
    return BeaconConfiguration(
        frequency_hz=915000000,
        power_dbm=10,
        modulation="LoRa",
        spreading_factor=7,
        bandwidth_hz=125000,
        coding_rate=5,
        pulse_rate_hz=1.0,
        pulse_duration_ms=100,
    )


@pytest.fixture
def test_config(beacon_config):
    """Create test configuration."""
    return FieldTestConfig(
        test_name="Test Detection Range",
        test_type="detection_range",
        beacon_config=beacon_config,
        environmental_conditions={
            "temperature_c": 25,
            "humidity_pct": 60,
            "wind_speed_mps": 5,
        },
        test_distances_m=[100, 250, 500],
        start_distance_m=500,
        target_radius_m=50,
        repetitions=3,
    )


class TestFieldTestServiceInit:
    """Test field test service initialization."""

    def test_initialization(self, field_test_service):
        """Test service initialization."""
        assert field_test_service.test_logger is not None
        assert field_test_service.state_machine is not None
        assert field_test_service.mavlink is not None
        assert field_test_service.signal_processor is not None
        assert field_test_service.safety_manager is not None
        assert field_test_service.active_tests == {}
        assert field_test_service.test_results == {}
        assert field_test_service.rssi_samples == []
        assert field_test_service.max_rssi_samples == 10000
        assert field_test_service.state_transition_times == {}
        assert field_test_service.detection_timestamps == []
        assert field_test_service.max_detection_timestamps == 1000

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=yaml.dump({"profiles": {"default": {"frequency": 433.0, "power": 5}}}),
    )
    @patch("pathlib.Path.exists", return_value=True)
    def test_load_beacon_profiles_success(self, mock_exists, mock_file):
        """Test loading beacon profiles successfully."""
        service = FieldTestService(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())

        assert service.beacon_profiles == {"default": {"frequency": 433.0, "power": 5}}

    @patch("pathlib.Path.exists", return_value=False)
    def test_load_beacon_profiles_file_not_found(self, mock_exists):
        """Test loading beacon profiles when file doesn't exist."""
        service = FieldTestService(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())

        assert service.beacon_profiles == {}

    @patch("builtins.open", side_effect=Exception("Read error"))
    @patch("pathlib.Path.exists", return_value=True)
    def test_load_beacon_profiles_error(self, mock_exists, mock_file):
        """Test loading beacon profiles with error."""
        service = FieldTestService(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())

        assert service.beacon_profiles == {}


class TestPreflightValidation:
    """Test preflight checklist validation."""

    @pytest.mark.asyncio
    async def test_validate_preflight_checklist_all_pass(
        self, field_test_service, mock_mavlink_service, mock_safety_manager
    ):
        """Test preflight validation when all checks pass."""
        result = await field_test_service.validate_preflight_checklist()

        assert result["mavlink_connected"] is True
        assert result["gps_fix_valid"] is True
        assert result["battery_sufficient"] is True
        assert result["safety_interlocks_passed"] is True
        assert result["geofence_configured"] is True
        assert result["emergency_stop_ready"] is True
        assert result["signal_processor_active"] is True
        assert result["state_machine_ready"] is True

    @pytest.mark.asyncio
    async def test_validate_preflight_checklist_low_battery(
        self, field_test_service, mock_mavlink_service
    ):
        """Test preflight validation with low battery."""
        mock_mavlink_service.get_telemetry.return_value = {
            "battery_percent": 15,
            "gps_status": "3D_FIX",
        }

        result = await field_test_service.validate_preflight_checklist()

        assert result["battery_sufficient"] is False

    @pytest.mark.asyncio
    async def test_validate_preflight_checklist_no_gps(
        self, field_test_service, mock_mavlink_service
    ):
        """Test preflight validation without GPS lock."""
        mock_mavlink_service.get_telemetry.return_value = {
            "battery_percent": 80,
            "gps_status": "NO_FIX",
        }

        result = await field_test_service.validate_preflight_checklist()

        assert result["gps_fix_valid"] is False

    @pytest.mark.asyncio
    async def test_validate_preflight_checklist_no_mavlink(
        self, field_test_service, mock_mavlink_service
    ):
        """Test preflight validation with MAVLink disconnected."""
        mock_mavlink_service.connected = False

        result = await field_test_service.validate_preflight_checklist()

        assert result["mavlink_connected"] is False

    @pytest.mark.asyncio
    async def test_validate_preflight_checklist_state_error(
        self, field_test_service, mock_state_machine
    ):
        """Test preflight validation with state machine in error."""
        mock_state_machine.current_state = "ERROR"

        result = await field_test_service.validate_preflight_checklist()

        assert result["state_machine_ready"] is False

    @pytest.mark.asyncio
    async def test_validate_preflight_checklist_exception(
        self, field_test_service, mock_mavlink_service
    ):
        """Test preflight validation with exception."""
        mock_mavlink_service.get_telemetry.side_effect = Exception("Connection error")

        result = await field_test_service.validate_preflight_checklist()

        # Should return False for checks that failed due to exception
        assert result["gps_fix_valid"] is False
        assert result["battery_sufficient"] is False


class TestFieldTestExecution:
    """Test field test execution."""

    @pytest.mark.asyncio
    async def test_start_field_test_preflight_fail(
        self, field_test_service, test_config, mock_mavlink_service
    ):
        """Test starting field test with preflight failure."""
        mock_mavlink_service.get_telemetry.return_value = {
            "battery_percent": 15,
            "gps_status": "NO_FIX",
        }

        with pytest.raises(ValueError, match="Preflight checks failed"):
            await field_test_service.start_field_test(test_config)

    @pytest.mark.asyncio
    async def test_start_field_test_success(self, field_test_service, test_config):
        """Test successful field test start."""
        with patch.object(field_test_service, "_execute_test") as mock_execute:
            status = await field_test_service.start_field_test(test_config)

            assert status.test_id is not None
            assert status.phase == "setup"
            assert status.status == "running"
            assert status.total_iterations == 3
            assert status.metrics is not None
            assert status.test_id in field_test_service.active_tests

    @pytest.mark.asyncio
    async def test_execute_test_detection_range(self, field_test_service, test_config):
        """Test execute test for detection range type."""
        test_id = str(uuid.uuid4())
        field_test_service.active_tests[test_id] = FieldTestStatus(
            test_id=test_id,
            phase="setup",
            status="running",
            start_time=datetime.now(UTC),
        )

        with patch.object(field_test_service, "_execute_detection_range_test") as mock_detect:
            with patch.object(field_test_service, "_save_test_results") as mock_save:
                await field_test_service._execute_test(test_id, test_config)

                mock_detect.assert_called_once()
                mock_save.assert_called_once()
                assert field_test_service.active_tests[test_id].phase == "completed"
                assert field_test_service.active_tests[test_id].status == "completed"

    @pytest.mark.asyncio
    async def test_execute_test_approach_accuracy(self, field_test_service, test_config):
        """Test execute test for approach accuracy type."""
        test_id = str(uuid.uuid4())
        test_config.test_type = "approach_accuracy"
        field_test_service.active_tests[test_id] = FieldTestStatus(
            test_id=test_id,
            phase="setup",
            status="running",
            start_time=datetime.now(UTC),
        )

        with patch.object(field_test_service, "_execute_approach_accuracy_test") as mock_approach:
            with patch.object(field_test_service, "_save_test_results") as mock_save:
                await field_test_service._execute_test(test_id, test_config)

                mock_approach.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_test_state_transition(self, field_test_service, test_config):
        """Test execute test for state transition type."""
        test_id = str(uuid.uuid4())
        test_config.test_type = "state_transition"
        field_test_service.active_tests[test_id] = FieldTestStatus(
            test_id=test_id,
            phase="setup",
            status="running",
            start_time=datetime.now(UTC),
        )

        with patch.object(field_test_service, "_execute_state_transition_test") as mock_state:
            with patch.object(field_test_service, "_save_test_results") as mock_save:
                await field_test_service._execute_test(test_id, test_config)

                mock_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_test_safety_validation(self, field_test_service, test_config):
        """Test execute test for safety validation type."""
        test_id = str(uuid.uuid4())
        test_config.test_type = "safety_validation"
        field_test_service.active_tests[test_id] = FieldTestStatus(
            test_id=test_id,
            phase="setup",
            status="running",
            start_time=datetime.now(UTC),
        )

        with patch.object(field_test_service, "_execute_safety_validation_test") as mock_safety:
            with patch.object(field_test_service, "_save_test_results") as mock_save:
                await field_test_service._execute_test(test_id, test_config)

                mock_safety.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_test_exception(self, field_test_service, test_config):
        """Test execute test with exception."""
        test_id = str(uuid.uuid4())
        field_test_service.active_tests[test_id] = FieldTestStatus(
            test_id=test_id,
            phase="setup",
            status="running",
            start_time=datetime.now(UTC),
        )

        with patch.object(
            field_test_service, "_execute_detection_range_test", side_effect=Exception("Test error")
        ):
            await field_test_service._execute_test(test_id, test_config)

            assert field_test_service.active_tests[test_id].phase == "failed"
            assert field_test_service.active_tests[test_id].status == "failed"
            assert field_test_service.active_tests[test_id].error_message == "Test error"


class TestDetectionRangeTest:
    """Test detection range test execution."""

    @pytest.mark.asyncio
    async def test_execute_detection_range_test(
        self, field_test_service, test_config, mock_signal_processor
    ):
        """Test detection range test execution."""
        test_id = str(uuid.uuid4())
        field_test_service.active_tests[test_id] = FieldTestStatus(
            test_id=test_id,
            phase="detection",
            status="running",
            start_time=datetime.now(UTC),
            metrics=FieldTestMetrics(
                test_id=test_id,
                beacon_power_dbm=10,
                detection_range_m=0,
                approach_accuracy_m=0,
                time_to_locate_s=0,
                transition_latency_ms=0,
                environmental_conditions={},
                safety_events=[],
                success=False,
            ),
        )

        with patch.object(field_test_service, "_configure_beacon_distance") as mock_config:
            with patch.object(field_test_service, "_wait_for_detection") as mock_wait:
                mock_wait.return_value = True
                with patch.object(field_test_service, "_measure_rssi") as mock_measure:
                    with patch("asyncio.sleep", return_value=None):
                        mock_measure.return_value = -75.0

                        await field_test_service._execute_detection_range_test(test_id, test_config)

                        # Verify multiple distances were tested
                        assert mock_config.call_count == 9  # 3 distances * 3 repetitions
                        assert mock_wait.call_count == 9
                        assert mock_measure.call_count == 9

                        # Check RSSI samples were collected
                        assert len(field_test_service.rssi_samples) > 0
                        assert field_test_service.active_tests[test_id].current_rssi_dbm == -75.0
                        assert field_test_service.active_tests[test_id].beacon_detected is True

    @pytest.mark.asyncio
    async def test_execute_detection_range_test_circular_buffer(
        self, field_test_service, test_config
    ):
        """Test RSSI circular buffer behavior."""
        test_id = str(uuid.uuid4())
        field_test_service.active_tests[test_id] = FieldTestStatus(
            test_id=test_id,
            phase="detection",
            status="running",
            start_time=datetime.now(UTC),
            metrics=FieldTestMetrics(
                test_id=test_id,
                beacon_power_dbm=10,
                detection_range_m=0,
                approach_accuracy_m=0,
                time_to_locate_s=0,
                transition_latency_ms=0,
                environmental_conditions={},
                safety_events=[],
                success=False,
            ),
        )

        # Pre-fill samples to near max
        field_test_service.rssi_samples = [-80.0] * (field_test_service.max_rssi_samples - 1)

        with patch.object(field_test_service, "_configure_beacon_distance"):
            with patch.object(field_test_service, "_wait_for_detection", return_value=True):
                with patch.object(field_test_service, "_measure_rssi", return_value=-75.0):
                    with patch("asyncio.sleep", return_value=None):
                        await field_test_service._execute_detection_range_test(test_id, test_config)

                        # Should not exceed max samples
                        assert (
                            len(field_test_service.rssi_samples)
                            <= field_test_service.max_rssi_samples
                        )

    @pytest.mark.asyncio
    async def test_wait_for_detection_success(self, field_test_service, mock_signal_processor):
        """Test waiting for detection success."""
        mock_signal_processor.current_rssi = -75.0  # Above threshold

        with patch("asyncio.sleep", return_value=None):
            result = await field_test_service._wait_for_detection(timeout=1.0)

        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_detection_timeout(self, field_test_service, mock_signal_processor):
        """Test waiting for detection timeout."""
        mock_signal_processor.current_rssi = -120.0  # Below threshold

        with patch("asyncio.sleep", return_value=None):
            result = await field_test_service._wait_for_detection(timeout=0.1)

        assert result is False

    @pytest.mark.asyncio
    async def test_measure_rssi(self, field_test_service, mock_signal_processor):
        """Test RSSI measurement."""
        rssi = await field_test_service._measure_rssi()

        assert rssi == -75.0


class TestApproachAccuracyTest:
    """Test approach accuracy test."""

    @pytest.mark.asyncio
    async def test_execute_approach_accuracy_test(
        self, field_test_service, test_config, mock_state_machine
    ):
        """Test approach accuracy test execution."""
        test_id = str(uuid.uuid4())
        field_test_service.active_tests[test_id] = FieldTestStatus(
            test_id=test_id,
            phase="approach",
            status="running",
            start_time=datetime.now(UTC),
            metrics=FieldTestMetrics(
                test_id=test_id,
                beacon_power_dbm=10,
                detection_range_m=0,
                approach_accuracy_m=0,
                time_to_locate_s=0,
                transition_latency_ms=0,
                environmental_conditions={},
                safety_events=[],
                success=False,
            ),
        )

        with patch.object(field_test_service, "_configure_beacon_distance"):
            with patch.object(field_test_service, "_monitor_approach") as mock_monitor:
                mock_monitor.return_value = {"lat": 0.0, "lon": 0.0, "alt": 0.0}
                with patch.object(field_test_service, "_calculate_position_error") as mock_error:
                    mock_error.return_value = 45.0
                    with patch("asyncio.sleep", return_value=None):
                        await field_test_service._execute_approach_accuracy_test(
                            test_id, test_config
                        )

                    assert mock_state_machine.request_transition.call_count == 3  # repetitions
                    assert (
                        field_test_service.active_tests[test_id].metrics.approach_accuracy_m == 45.0
                    )

    @pytest.mark.asyncio
    async def test_monitor_approach_success(self, field_test_service, mock_mavlink_service):
        """Test approach monitoring success."""
        with patch("asyncio.sleep", return_value=None):
            position = await field_test_service._monitor_approach(50.0, timeout=1.0)

        assert "lat" in position
        assert "lon" in position
        assert "alt" in position

    @pytest.mark.asyncio
    async def test_monitor_approach_timeout(self, field_test_service, mock_mavlink_service):
        """Test approach monitoring timeout."""
        # Make telemetry return a position
        mock_mavlink_service.get_telemetry = AsyncMock(
            return_value={"position": {"lat": 0.0, "lon": 0.0, "alt": 0.0}}
        )

        with patch("asyncio.sleep", return_value=None):
            position = await field_test_service._monitor_approach(50.0, timeout=0.1)

        assert position == {"lat": 0.0, "lon": 0.0, "alt": 0.0}

    def test_calculate_position_error(self, field_test_service):
        """Test position error calculation."""
        position = {"lat": -35.363261, "lon": 149.165230, "alt": 50}
        error = field_test_service._calculate_position_error(position)

        assert error == 45.0  # Simplified calculation


class TestStateTransitionTest:
    """Test state transition test."""

    @pytest.mark.asyncio
    async def test_execute_state_transition_test(
        self, field_test_service, test_config, mock_state_machine
    ):
        """Test state transition test execution."""
        test_id = str(uuid.uuid4())
        field_test_service.active_tests[test_id] = FieldTestStatus(
            test_id=test_id,
            phase="detection",
            status="running",
            start_time=datetime.now(UTC),
            metrics=FieldTestMetrics(
                test_id=test_id,
                beacon_power_dbm=10,
                detection_range_m=0,
                approach_accuracy_m=0,
                time_to_locate_s=0,
                transition_latency_ms=0,
                environmental_conditions={},
                safety_events=[],
                success=False,
            ),
        )

        with patch.object(field_test_service, "_trigger_detection"):
            with patch("asyncio.sleep", return_value=None):
                await field_test_service._execute_state_transition_test(test_id, test_config)

            # Should test state transitions
            assert mock_state_machine.request_transition.call_count >= 6  # SEARCHING and HOMING * 3
            assert field_test_service.active_tests[test_id].metrics.transition_latency_ms > 0

    @pytest.mark.asyncio
    async def test_trigger_detection(self, field_test_service, mock_signal_processor):
        """Test trigger detection."""
        await field_test_service._trigger_detection()

        assert mock_signal_processor.current_rssi == -75.0


class TestSafetyValidationTest:
    """Test safety validation test."""

    @pytest.mark.asyncio
    async def test_execute_safety_validation_test(
        self, field_test_service, test_config, mock_state_machine, mock_safety_manager
    ):
        """Test safety validation test execution."""
        test_id = str(uuid.uuid4())
        field_test_service.active_tests[test_id] = FieldTestStatus(
            test_id=test_id,
            phase="detection",
            status="running",
            start_time=datetime.now(UTC),
            metrics=FieldTestMetrics(
                test_id=test_id,
                beacon_power_dbm=10,
                detection_range_m=0,
                approach_accuracy_m=0,
                time_to_locate_s=0,
                transition_latency_ms=0,
                environmental_conditions={},
                safety_events=[],
                success=False,
            ),
        )

        with patch.object(field_test_service, "_test_geofence_enforcement", return_value=True):
            with patch.object(field_test_service, "_test_battery_failsafe", return_value=True):
                with patch.object(
                    field_test_service, "_test_signal_loss_recovery", return_value=True
                ):
                    with patch("asyncio.sleep", return_value=None):
                        await field_test_service._execute_safety_validation_test(
                            test_id, test_config
                        )

                    assert mock_state_machine.request_transition.called
                    assert mock_safety_manager.emergency_stop.called

                    metrics = field_test_service.active_tests[test_id].metrics
                    assert "emergency_stop_success" in metrics.safety_events
                    assert "geofence_enforced" in metrics.safety_events
                    assert "battery_failsafe_triggered" in metrics.safety_events
                    assert "signal_loss_handled" in metrics.safety_events
                    assert metrics.success is True

    @pytest.mark.asyncio
    async def test_test_geofence_enforcement(self, field_test_service):
        """Test geofence enforcement test."""
        result = await field_test_service._test_geofence_enforcement()
        assert result is True

    @pytest.mark.asyncio
    async def test_test_battery_failsafe(self, field_test_service):
        """Test battery failsafe test."""
        result = await field_test_service._test_battery_failsafe()
        assert result is True

    @pytest.mark.asyncio
    async def test_test_signal_loss_recovery(self, field_test_service):
        """Test signal loss recovery test."""
        result = await field_test_service._test_signal_loss_recovery()
        assert result is True


class TestBeaconValidation:
    """Test beacon signal validation."""

    @pytest.mark.asyncio
    async def test_validate_beacon_signal_with_sdr(self, field_test_service, beacon_config):
        """Test beacon signal validation with SDR."""
        with patch("backend.services.sdr_service.SDRService") as MockSDR:
            mock_sdr = MagicMock()
            mock_sdr.initialize = AsyncMock()
            mock_sdr.set_frequency = MagicMock()
            mock_sdr.config.sampleRate = 2.4e6

            # Mock IQ samples stream
            async def mock_stream():
                import numpy as np

                for _ in range(2):
                    yield np.random.randn(1024) + 1j * np.random.randn(1024)

            mock_sdr.stream_iq = mock_stream
            mock_sdr.shutdown = AsyncMock()
            MockSDR.return_value = mock_sdr

            # Mock scipy.signal to avoid import timeout
            with patch.dict("sys.modules", {"scipy.signal": MagicMock()}):
                result = await field_test_service.validate_beacon_signal(beacon_config)

            assert "frequency_match" in result
            assert "power_level_match" in result
            assert "modulation_match" in result
            assert "measured_frequency_hz" in result
            assert "measured_power_dbm" in result
            assert "spectrum_data" in result
            assert "validation_passed" in result

    @pytest.mark.asyncio
    async def test_validate_beacon_signal_no_scipy(self, field_test_service, beacon_config):
        """Test beacon signal validation without SciPy."""
        with patch("backend.services.sdr_service.SDRService", side_effect=ImportError):
            result = await field_test_service.validate_beacon_signal(beacon_config)

            assert result["validation_passed"] is True
            assert result["frequency_match"] is True
            assert result["power_level_match"] is True
            assert result["modulation_match"] is True

    @pytest.mark.asyncio
    async def test_validate_beacon_signal_exception(self, field_test_service, beacon_config):
        """Test beacon signal validation with exception."""
        with patch("backend.services.sdr_service.SDRService") as MockSDR:
            MockSDR.side_effect = Exception("SDR Error")

            result = await field_test_service.validate_beacon_signal(beacon_config)

            assert "error" in result
            assert result["error"] == "SDR Error"

    @pytest.mark.asyncio
    async def test_configure_beacon_distance(self, field_test_service, beacon_config):
        """Test beacon distance configuration."""
        await field_test_service._configure_beacon_distance(100.0, beacon_config)
        # Should complete without error


class TestResultsManagement:
    """Test results saving and retrieval."""

    @pytest.mark.asyncio
    async def test_save_test_results(self, field_test_service, test_config, mock_test_logger):
        """Test saving test results."""
        test_id = str(uuid.uuid4())
        field_test_service.active_tests[test_id] = FieldTestStatus(
            test_id=test_id,
            phase="completed",
            status="completed",
            start_time=datetime.now(UTC),
            metrics=FieldTestMetrics(
                test_id=test_id,
                beacon_power_dbm=10,
                detection_range_m=500,
                approach_accuracy_m=45,
                time_to_locate_s=120,
                transition_latency_ms=250,
                environmental_conditions={},
                safety_events=["all_passed"],
                success=True,
            ),
        )

        with patch("builtins.open", mock_open()) as mock_file:
            with patch("pathlib.Path.mkdir"):
                await field_test_service._save_test_results(test_id, test_config)

                mock_test_logger.log_test_run.assert_called_once()
                mock_file.assert_called()

    @pytest.mark.asyncio
    async def test_get_test_status_active(self, field_test_service):
        """Test getting status of active test."""
        test_id = str(uuid.uuid4())
        field_test_service.active_tests[test_id] = FieldTestStatus(
            test_id=test_id,
            phase="detection",
            status="running",
            start_time=datetime.now(UTC),
        )

        status = await field_test_service.get_test_status(test_id)

        assert status is not None
        assert status.test_id == test_id
        assert status.status == "running"

    @pytest.mark.asyncio
    async def test_get_test_status_not_found(self, field_test_service):
        """Test getting status of non-existent test."""
        status = await field_test_service.get_test_status("non_existent")

        assert status is None

    @pytest.mark.asyncio
    async def test_get_test_metrics_active(self, field_test_service):
        """Test getting metrics for active test."""
        test_id = str(uuid.uuid4())
        metrics = FieldTestMetrics(
            test_id=test_id,
            beacon_power_dbm=10,
            detection_range_m=500,
            approach_accuracy_m=45,
            time_to_locate_s=120,
            transition_latency_ms=250,
            environmental_conditions={},
            safety_events=[],
            success=True,
        )
        field_test_service.active_tests[test_id] = FieldTestStatus(
            test_id=test_id,
            phase="completed",
            status="completed",
            start_time=datetime.now(UTC),
            metrics=metrics,
        )

        result = await field_test_service.get_test_metrics(test_id)

        assert result is not None
        assert result.test_id == test_id
        assert result.detection_range_m == 500

    @pytest.mark.asyncio
    async def test_get_test_metrics_from_file(self, field_test_service):
        """Test getting metrics from archived file."""
        test_id = str(uuid.uuid4())
        metrics_data = {
            "test_id": test_id,
            "beacon_power_dbm": 10,
            "detection_range_m": 500,
            "approach_accuracy_m": 45,
            "time_to_locate_s": 120,
            "transition_latency_ms": 250,
            "environmental_conditions": {},
            "safety_events": [],
            "success": True,
        }

        with patch("builtins.open", mock_open(read_data=json.dumps(metrics_data))):
            with patch("pathlib.Path.exists", return_value=True):
                result = await field_test_service.get_test_metrics(test_id)

                assert result is not None
                assert result.test_id == test_id

    @pytest.mark.asyncio
    async def test_get_test_metrics_not_found(self, field_test_service):
        """Test getting metrics for non-existent test."""
        with patch("pathlib.Path.exists", return_value=False):
            result = await field_test_service.get_test_metrics("non_existent")

            assert result is None

    @pytest.mark.asyncio
    async def test_export_test_data_json(self, field_test_service):
        """Test exporting test data as JSON."""
        test_id = str(uuid.uuid4())

        with patch.object(field_test_service, "get_test_metrics") as mock_get:
            mock_get.return_value = FieldTestMetrics(
                test_id=test_id,
                beacon_power_dbm=10,
                detection_range_m=500,
                approach_accuracy_m=45,
                time_to_locate_s=120,
                transition_latency_ms=250,
                environmental_conditions={},
                safety_events=[],
                success=True,
            )

            with patch("builtins.open", mock_open()) as mock_file:
                with patch("pathlib.Path.mkdir"):
                    result = await field_test_service.export_test_data(test_id, format="json")

                    assert result is not None
                    assert str(result).endswith("_export.json")
                    mock_file.assert_called()

    @pytest.mark.asyncio
    async def test_export_test_data_csv(self, field_test_service):
        """Test exporting test data as CSV."""
        test_id = str(uuid.uuid4())

        with patch.object(field_test_service, "get_test_metrics") as mock_get:
            mock_get.return_value = FieldTestMetrics(
                test_id=test_id,
                beacon_power_dbm=10,
                detection_range_m=500,
                approach_accuracy_m=45,
                time_to_locate_s=120,
                transition_latency_ms=250,
                environmental_conditions={},
                safety_events=[],
                success=True,
            )

            with patch("builtins.open", mock_open()) as mock_file:
                with patch("pathlib.Path.mkdir"):
                    result = await field_test_service.export_test_data(test_id, format="csv")

                    assert result is not None
                    assert str(result).endswith("_export.csv")

    @pytest.mark.asyncio
    async def test_export_test_data_invalid_format(self, field_test_service):
        """Test exporting test data with invalid format."""
        test_id = str(uuid.uuid4())

        with patch.object(field_test_service, "get_test_metrics") as mock_get:
            mock_get.return_value = FieldTestMetrics(
                test_id=test_id,
                beacon_power_dbm=10,
                detection_range_m=500,
                approach_accuracy_m=45,
                time_to_locate_s=120,
                transition_latency_ms=250,
                environmental_conditions={},
                safety_events=[],
                success=True,
            )

            result = await field_test_service.export_test_data(test_id, format="xml")

            assert result is None

    @pytest.mark.asyncio
    async def test_export_test_data_invalid_test_id(self, field_test_service):
        """Test exporting test data with invalid test ID."""
        result = await field_test_service.export_test_data("invalid-id", format="json")

        assert result is None

    @pytest.mark.asyncio
    async def test_export_test_data_no_metrics(self, field_test_service):
        """Test exporting test data when metrics not found."""
        test_id = str(uuid.uuid4())

        with patch.object(field_test_service, "get_test_metrics", return_value=None):
            result = await field_test_service.export_test_data(test_id, format="json")

            assert result is None
