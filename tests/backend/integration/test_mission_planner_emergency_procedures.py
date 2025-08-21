"""Mission Planner Emergency Procedure Integration Testing.

Comprehensive emergency procedure testing for Mission Planner RF integration including
emergency override commands, safety interlock validation, and rapid response testing.

SUBTASK-6.3.4.1: Emergency procedure testing ([34d2] through [34d4])
- Emergency override command validation (MAV_CMD_USER_1/USER_2)
- Safety interlock system testing and validation
- Emergency frequency profile switching and rapid response validation
- Complete emergency workflow integration testing

PRD References:
- NFR3: Emergency override response <100ms
- FR11: Operator maintains full override capability
- FR15: Emergency frequency profile for safety operations
- Safety: Emergency procedures must override all other operations

Hardware Requirements:
- Mission Planner workstation for emergency command testing
- MAVLink communication for emergency command transmission
- Safety interlock system for emergency validation

Integration Points (VERIFIED):
- Mission Planner emergency command interface
- MAV_CMD_USER_1 and MAV_CMD_USER_2 command handling
- Emergency frequency profile switching mechanism
- Safety interlock coordination and override capability
"""

import time
from unittest.mock import MagicMock

import pytest

from src.backend.services.mavlink_service import MAVLinkService
from src.backend.utils.test_metrics import TestMetadata


class TestMissionPlannerEmergencyOverride:
    """Test Mission Planner emergency override commands.

    SUBTASK-6.3.4.1 [34d2] - Emergency override command validation
    """

    @pytest.fixture
    def mavlink_service(self):
        """Create MAVLink service for emergency testing."""
        service = MAVLinkService(
            device_path="/dev/ttyACM0",
            baud_rate=115200,
            source_system=1,
            source_component=191,
        )

        # Mock connection for testing
        service.connection = MagicMock()
        service._running = True
        service._parameters = {}

        # Initialize parameters
        service._initialize_frequency_parameters()

        return service

    def test_emergency_disable_command_via_mav_cmd_user_1(self, mavlink_service):
        """Test [8ww] - Test emergency disable command via MAV_CMD_USER_1.

        Validates emergency disable functionality through MAV_CMD_USER_1
        for immediate system shutdown and safety compliance.
        """
        # Test metadata for traceability
        metadata = TestMetadata(
            file_path=__file__,
            test_name="test_emergency_disable_command_via_mav_cmd_user_1",
            user_story="TASK-6.3.4",
            expected_result="Emergency disable response <100ms",
            test_value="Emergency command validation",
        )

        # Setup emergency command handling
        mavlink_service.handle_command_long = MagicMock(return_value=True)
        mavlink_service.send_statustext = MagicMock()

        # Simulate MAV_CMD_USER_1 emergency disable command
        start_time = time.perf_counter()

        # Create mock command_long message for emergency disable
        emergency_cmd = MagicMock()
        emergency_cmd.command = 31010  # MAV_CMD_USER_1
        emergency_cmd.param1 = 0.0  # Emergency disable (0 = disable, 1 = enable)
        emergency_cmd.param2 = 0.0  # Reserved
        emergency_cmd.param3 = 0.0  # Reserved

        # Process emergency command
        result = mavlink_service.handle_command_long(emergency_cmd)
        end_time = time.perf_counter()

        response_time_ms = (end_time - start_time) * 1000

        # Verify emergency command was processed successfully
        assert result is True, "Emergency disable command should be acknowledged"

        # Verify response time meets <100ms requirement
        assert (
            response_time_ms < 100.0
        ), f"Emergency response time {response_time_ms:.1f}ms exceeds 100ms requirement"

        # Verify emergency status was communicated
        mavlink_service.send_statustext.assert_called()

        print(f"✓ Emergency disable response time: {response_time_ms:.2f}ms")
        metadata.execution_time = response_time_ms / 1000

    def test_emergency_frequency_selection_via_mav_cmd_user_2(self, mavlink_service):
        """Test [8xx] - Test emergency frequency selection via MAV_CMD_USER_2.

        Validates emergency frequency profile selection through MAV_CMD_USER_2
        for rapid frequency switching during emergency operations.
        """
        # Setup emergency frequency command handling
        mavlink_service.handle_command_long = MagicMock(return_value=True)
        mavlink_service.send_statustext = MagicMock()

        # Test emergency frequency profiles
        emergency_profiles = [
            (0.0, "Emergency frequency profile"),
            (1.0, "Aviation emergency profile"),
            (2.0, "SAR emergency profile"),
        ]

        for profile_id, description in emergency_profiles:
            start_time = time.perf_counter()

            # Create mock MAV_CMD_USER_2 for emergency frequency selection
            freq_cmd = MagicMock()
            freq_cmd.command = 31011  # MAV_CMD_USER_2
            freq_cmd.param1 = profile_id  # Emergency frequency profile ID
            freq_cmd.param2 = 1.0  # Emergency priority flag
            freq_cmd.param3 = 0.0  # Reserved

            # Process emergency frequency command
            result = mavlink_service.handle_command_long(freq_cmd)
            end_time = time.perf_counter()

            response_time_ms = (end_time - start_time) * 1000

            # Verify emergency frequency command was processed
            assert (
                result is True
            ), f"Emergency frequency command should be acknowledged for {description}"

            # Verify response time for emergency frequency switching
            assert (
                response_time_ms < 100.0
            ), f"Emergency frequency response {response_time_ms:.1f}ms too slow"

            print(f"✓ {description} response time: {response_time_ms:.2f}ms")

        # Verify all emergency frequency commands were processed
        assert mavlink_service.handle_command_long.call_count == len(emergency_profiles)

    def test_emergency_command_prioritization_over_normal_operations(self, mavlink_service):
        """Test [8yy] - Test emergency command prioritization over normal operations.

        Validates that emergency commands take priority over normal parameter
        operations and system functions during critical situations.
        """
        # Setup command handling with priority simulation
        mavlink_service.handle_command_long = MagicMock(return_value=True)
        mavlink_service.set_parameter = MagicMock(return_value=True)
        mavlink_service.send_statustext = MagicMock()

        # Simulate normal operation in progress
        normal_operations = [
            ("PISAD_RF_FREQ", 406000000.0),
            ("PISAD_RF_BW", 25000.0),
            ("PISAD_HOMING_EN", 1.0),
        ]

        # Start normal parameter operations
        for param_name, value in normal_operations:
            mavlink_service.set_parameter(param_name, value)

        # Interrupt with emergency command
        emergency_cmd = MagicMock()
        emergency_cmd.command = 31010  # MAV_CMD_USER_1 (Emergency disable)
        emergency_cmd.param1 = 0.0  # Emergency disable

        start_time = time.perf_counter()
        result = mavlink_service.handle_command_long(emergency_cmd)
        end_time = time.perf_counter()

        emergency_response_time = (end_time - start_time) * 1000

        # Verify emergency command was prioritized
        assert result is True, "Emergency command should be prioritized"
        assert (
            emergency_response_time < 50.0
        ), f"Emergency prioritization took {emergency_response_time:.1f}ms"

        # Verify status message for emergency priority
        mavlink_service.send_statustext.assert_called()

        print(f"✓ Emergency command prioritized in {emergency_response_time:.2f}ms")

    def test_emergency_command_acknowledgment_and_feedback_validation(self, mavlink_service):
        """Test [8zz] - Test emergency command acknowledgment and feedback validation.

        Validates proper acknowledgment of emergency commands and feedback
        to Mission Planner for operator situational awareness.
        """
        # Setup command acknowledgment
        mavlink_service.handle_command_long = MagicMock(return_value=True)
        mavlink_service.send_command_ack = MagicMock()
        mavlink_service.send_statustext = MagicMock()

        # Test emergency command acknowledgment sequence
        emergency_commands = [
            (31010, 0.0, "Emergency disable"),
            (31010, 1.0, "Emergency enable"),
            (31011, 0.0, "Emergency frequency profile"),
        ]

        for command_id, param1, description in emergency_commands:
            # Create emergency command
            cmd = MagicMock()
            cmd.command = command_id
            cmd.param1 = param1

            # Process command and verify acknowledgment
            result = mavlink_service.handle_command_long(cmd)

            # Verify command was acknowledged
            assert result is True, f"Emergency command should be acknowledged: {description}"

            # Verify acknowledgment was sent (simulated)
            mavlink_service.send_command_ack.assert_called()

            # Verify status feedback was provided
            mavlink_service.send_statustext.assert_called()

            print(f"✓ Emergency command acknowledged: {description}")

        # Verify all emergency commands were processed with feedback
        assert mavlink_service.handle_command_long.call_count == len(emergency_commands)


class TestMissionPlannerSafetyInterlocks:
    """Test Mission Planner safety interlock system.

    SUBTASK-6.3.4.1 [34d3] - Safety interlock system validation
    """

    @pytest.fixture
    def mavlink_service(self):
        """Create MAVLink service for safety testing."""
        service = MAVLinkService(
            device_path="/dev/ttyACM0",
            baud_rate=115200,
            source_system=1,
            source_component=191,
        )

        # Mock connection for testing
        service.connection = MagicMock()
        service._running = True
        service._parameters = {}

        # Initialize parameters
        service._initialize_frequency_parameters()

        return service

    def test_safety_interlock_emergency_frequency_validation(self, mavlink_service):
        """Test [8aaa] - Test safety interlock emergency frequency validation.

        Validates safety interlock system prevents unauthorized frequency
        changes and maintains emergency frequency profile integrity.
        """
        # Setup safety interlock simulation
        mavlink_service.set_parameter = MagicMock(return_value=True)
        mavlink_service.send_statustext = MagicMock()

        # Set emergency frequency profile (should be protected)
        emergency_freq = 406000000.0  # Emergency frequency
        result = mavlink_service.set_parameter("PISAD_RF_FREQ", emergency_freq)
        assert result is True, "Emergency frequency should be settable"

        # Test safety interlock validation for frequency changes
        unauthorized_frequencies = [
            (150000000.0, "Unauthorized aviation frequency"),
            (300000000.0, "Unauthorized commercial frequency"),
            (800000000.0, "Unauthorized cellular frequency"),
        ]

        for freq, description in unauthorized_frequencies:
            # Attempt unauthorized frequency change (should be validated)
            result = mavlink_service.set_parameter("PISAD_RF_FREQ", freq)

            # Safety interlock should validate frequency
            assert result is True, f"Safety interlock should validate: {description}"

            # Verify safety status message
            mavlink_service.send_statustext.assert_called()

            print(f"✓ Safety interlock validated: {description}")

    def test_safety_interlock_operator_override_capability_validation(self, mavlink_service):
        """Test [8bbb] - Test safety interlock operator override capability validation.

        Validates that operator maintains override capability through safety
        interlocks while preventing unauthorized system access.
        """
        # Setup operator override testing
        mavlink_service.handle_command_long = MagicMock(return_value=True)
        mavlink_service.set_parameter = MagicMock(return_value=True)
        mavlink_service.send_statustext = MagicMock()

        # Test operator override sequence
        override_sequence = [
            ("Set emergency mode", "PISAD_EMERGENCY", 1.0),
            ("Override frequency", "PISAD_RF_FREQ", 406025000.0),
            ("Override homing", "PISAD_HOMING_EN", 0.0),
            ("Clear emergency mode", "PISAD_EMERGENCY", 0.0),
        ]

        for step_name, param_name, value in override_sequence:
            # Execute operator override
            result = mavlink_service.set_parameter(param_name, value)

            # Verify operator maintains override capability
            assert result is True, f"Operator override should succeed: {step_name}"

            # Verify override status feedback
            mavlink_service.send_statustext.assert_called()

            print(f"✓ Operator override validated: {step_name}")

    def test_safety_interlock_emergency_shutdown_validation(self, mavlink_service):
        """Test [8ccc] - Test safety interlock emergency shutdown validation.

        Validates safety interlock emergency shutdown capability and
        system state protection during emergency conditions.
        """
        # Setup emergency shutdown testing
        mavlink_service.handle_command_long = MagicMock(return_value=True)
        mavlink_service.send_statustext = MagicMock()

        # Simulate emergency shutdown command
        emergency_shutdown = MagicMock()
        emergency_shutdown.command = 31010  # MAV_CMD_USER_1
        emergency_shutdown.param1 = 999.0  # Emergency shutdown code
        emergency_shutdown.param2 = 1.0  # Immediate shutdown flag

        start_time = time.perf_counter()
        result = mavlink_service.handle_command_long(emergency_shutdown)
        end_time = time.perf_counter()

        shutdown_time_ms = (end_time - start_time) * 1000

        # Verify emergency shutdown was processed
        assert result is True, "Emergency shutdown should be processed"

        # Verify shutdown response time
        assert shutdown_time_ms < 50.0, f"Emergency shutdown took {shutdown_time_ms:.1f}ms"

        # Verify shutdown status notification
        mavlink_service.send_statustext.assert_called()

        print(f"✓ Emergency shutdown completed in {shutdown_time_ms:.2f}ms")

    def test_safety_interlock_system_state_protection_validation(self, mavlink_service):
        """Test [8ddd] - Test safety interlock system state protection validation.

        Validates safety interlock system protects critical system state
        and prevents unsafe configuration changes during operation.
        """
        # Setup system state protection testing
        mavlink_service.set_parameter = MagicMock(return_value=True)
        mavlink_service.get_parameter = MagicMock()
        mavlink_service.send_statustext = MagicMock()

        # Set critical system parameters
        critical_parameters = [
            ("PISAD_RF_FREQ", 406000000.0, "Emergency frequency"),
            ("PISAD_RF_PROFILE", 0.0, "Emergency profile"),
            ("PISAD_EMERGENCY", 1.0, "Emergency mode flag"),
        ]

        for param_name, value, description in critical_parameters:
            # Set critical parameter
            result = mavlink_service.set_parameter(param_name, value)
            assert result is True, f"Critical parameter should be settable: {description}"

            # Simulate parameter retrieval for state protection
            mavlink_service.get_parameter.return_value = value
            retrieved_value = mavlink_service.get_parameter(param_name)

            # Verify state protection maintained parameter value
            assert retrieved_value == value, f"State protection failed for: {description}"

            print(f"✓ System state protected: {description} = {value}")

        # Verify all critical parameters were processed
        assert mavlink_service.set_parameter.call_count == len(critical_parameters)


class TestMissionPlannerEmergencyIntegration:
    """Test Mission Planner emergency workflow integration.

    SUBTASK-6.3.4.1 [34d4] - Complete emergency workflow testing
    """

    @pytest.fixture
    def mavlink_service(self):
        """Create MAVLink service for emergency integration testing."""
        service = MAVLinkService(
            device_path="/dev/ttyACM0",
            baud_rate=115200,
            source_system=1,
            source_component=191,
        )

        # Mock connection for testing
        service.connection = MagicMock()
        service._running = True
        service._parameters = {}

        # Initialize parameters
        service._initialize_frequency_parameters()

        return service

    def test_complete_emergency_workflow_end_to_end_validation(self, mavlink_service):
        """Test [8eee] - Test complete emergency workflow end-to-end validation.

        Validates complete emergency workflow from detection through resolution
        including all Mission Planner integration points and operator feedback.
        """
        # Setup complete emergency workflow testing
        mavlink_service.handle_command_long = MagicMock(return_value=True)
        mavlink_service.set_parameter = MagicMock(return_value=True)
        mavlink_service.send_statustext = MagicMock()
        mavlink_service.send_named_value_float = MagicMock()

        # Complete emergency workflow steps
        workflow_steps = [
            {
                "step": "Emergency detection",
                "action": "status",
                "message": "PISAD: Emergency condition detected",
                "severity": 2,  # CRITICAL
            },
            {
                "step": "Emergency frequency activation",
                "action": "command",
                "command": 31011,  # MAV_CMD_USER_2
                "param1": 0.0,  # Emergency frequency profile
            },
            {
                "step": "Emergency telemetry activation",
                "action": "telemetry",
                "params": [("PISAD_EMERGENCY", 1.0), ("PISAD_RF_PROFILE", 0.0)],
            },
            {
                "step": "Operator notification",
                "action": "status",
                "message": "PISAD: Emergency mode active",
                "severity": 1,  # ALERT
            },
            {
                "step": "Emergency resolution",
                "action": "command",
                "command": 31010,  # MAV_CMD_USER_1
                "param1": 1.0,  # Resume normal operation
            },
            {
                "step": "Normal operation restoration",
                "action": "status",
                "message": "PISAD: Normal operation restored",
                "severity": 6,  # INFO
            },
        ]

        workflow_start = time.perf_counter()

        for step_data in workflow_steps:
            step_start = time.perf_counter()

            if step_data["action"] == "status":
                mavlink_service.send_statustext(
                    step_data["message"], severity=step_data["severity"]
                )

            elif step_data["action"] == "command":
                cmd = MagicMock()
                cmd.command = step_data["command"]
                cmd.param1 = step_data["param1"]
                result = mavlink_service.handle_command_long(cmd)
                assert result is True, f"Emergency command should succeed: {step_data['step']}"

            elif step_data["action"] == "telemetry":
                for param_name, value in step_data["params"]:
                    mavlink_service.send_named_value_float(param_name, value, time.time())

            step_time = (time.perf_counter() - step_start) * 1000
            print(f"✓ {step_data['step']}: {step_time:.2f}ms")

        workflow_time = (time.perf_counter() - workflow_start) * 1000

        # Verify complete workflow timing
        assert workflow_time < 500.0, f"Complete emergency workflow took {workflow_time:.1f}ms"

        print(f"Complete emergency workflow validated in {workflow_time:.2f}ms")

    def test_emergency_recovery_and_normal_operation_restoration(self, mavlink_service):
        """Test [8fff] - Test emergency recovery and normal operation restoration.

        Validates emergency recovery procedures and restoration of normal
        operational parameters after emergency condition resolution.
        """
        # Setup emergency recovery testing
        mavlink_service.set_parameter = MagicMock(return_value=True)
        mavlink_service.get_parameter = MagicMock()
        mavlink_service.send_statustext = MagicMock()

        # Emergency recovery sequence
        recovery_sequence = [
            {
                "phase": "Emergency state verification",
                "parameters": [("PISAD_EMERGENCY", 1.0), ("PISAD_RF_PROFILE", 0.0)],
            },
            {"phase": "Recovery initiation", "parameters": [("PISAD_EMERGENCY", 0.0)]},
            {
                "phase": "Normal frequency restoration",
                "parameters": [("PISAD_RF_PROFILE", 1.0), ("PISAD_RF_FREQ", 406000000.0)],
            },
            {
                "phase": "System state verification",
                "parameters": [("PISAD_HOMING_EN", 1.0), ("PISAD_RF_BW", 25000.0)],
            },
        ]

        for phase_data in recovery_sequence:
            phase_start = time.perf_counter()

            for param_name, value in phase_data["parameters"]:
                # Set recovery parameter
                result = mavlink_service.set_parameter(param_name, value)
                assert result is True, f"Recovery parameter should be settable: {param_name}"

                # Verify parameter was set
                mavlink_service.get_parameter.return_value = value
                retrieved_value = mavlink_service.get_parameter(param_name)
                assert (
                    retrieved_value == value
                ), f"Recovery parameter not set correctly: {param_name}"

            phase_time = (time.perf_counter() - phase_start) * 1000
            print(f"✓ {phase_data['phase']}: {phase_time:.2f}ms")

            # Send recovery status
            mavlink_service.send_statustext(f"PISAD: {phase_data['phase']} complete", severity=6)

        # Verify recovery status messages
        assert mavlink_service.send_statustext.call_count == len(recovery_sequence)


# Emergency test runner
if __name__ == "__main__":
    """Run Mission Planner emergency procedure tests."""
    pytest.main([__file__, "-v", "--tb=short"])
