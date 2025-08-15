"""
SITL Validation Tests - Verify SITL configuration and interfaces.

Story 4.7 - Sprint 5: SITL Integration
These tests validate SITL configuration without requiring ArduPilot installation.
"""

import os
from pathlib import Path

import yaml

from src.backend.hal.sitl_interface import SITLInterface


class TestSITLValidation:
    """Validation tests for SITL configuration and interfaces."""

    def test_sitl_config_exists(self) -> None:
        """Test that SITL configuration file exists.

        AC: 2 - SITL configuration created
        """
        # GIVEN: Expected config path
        config_path = Path("config/sitl.yaml")

        # THEN: Config file should exist
        assert config_path.exists(), f"SITL config not found at {config_path}"

    def test_sitl_config_valid(self) -> None:
        """Test that SITL configuration is valid YAML.

        AC: 2 - Valid SITL configuration
        """
        # GIVEN: SITL config file
        config_path = Path("config/sitl.yaml")

        # WHEN: We load the configuration
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # THEN: Configuration should have required sections
        assert "sitl" in config
        sitl_config = config["sitl"]

        # Verify connection settings
        assert "connection" in sitl_config
        assert "primary" in sitl_config["connection"]
        assert sitl_config["connection"]["primary"] == "tcp:127.0.0.1:5760"

        # Verify vehicle settings
        assert "vehicle" in sitl_config
        assert sitl_config["vehicle"]["type"] == "copter"

        # Verify location settings
        assert "location" in sitl_config
        assert "lat" in sitl_config["location"]
        assert "lon" in sitl_config["location"]
        assert "alt" in sitl_config["location"]

        # Verify safety settings match Story 4.7 requirements
        assert "safety" in sitl_config
        safety = sitl_config["safety"]
        assert safety["battery_low_voltage"] == 19.2  # 6S Li-ion
        assert safety["battery_critical_voltage"] == 18.0
        assert safety["gps_min_satellites"] == 8
        assert safety["gps_max_hdop"] == 2.0
        assert safety["rc_override_threshold"] == 50
        assert safety["emergency_stop_time"] == 0.5

        # Verify performance targets
        assert "performance" in sitl_config
        perf = sitl_config["performance"]
        assert perf["mode_change_latency"] == 0.1  # <100ms
        assert perf["emergency_stop_latency"] == 0.5  # <500ms
        assert perf["mavlink_latency"] == 0.05  # <50ms

    def test_sitl_interface_initialization(self) -> None:
        """Test SITL interface can be initialized.

        AC: 2 - SITL interface created
        """
        # WHEN: We create a SITL interface
        sitl = SITLInterface()

        # THEN: Interface should be initialized with config
        assert sitl is not None
        assert sitl.config is not None
        assert not sitl.connected
        assert sitl.sitl_process is None

    def test_sitl_interface_config_loading(self) -> None:
        """Test SITL interface loads configuration correctly.

        AC: 2 - Configuration loaded correctly
        """
        # GIVEN: SITL interface
        sitl = SITLInterface()

        # THEN: Configuration should be loaded
        assert "connection" in sitl.config
        assert "vehicle" in sitl.config
        assert "location" in sitl.config

        # Verify connection settings
        assert sitl.config["connection"]["primary"] == "tcp:127.0.0.1:5760"
        assert sitl.config["connection"]["timeout"] == 10

        # Verify vehicle type
        assert sitl.config["vehicle"]["type"] == "copter"

    def test_sitl_interface_default_config(self) -> None:
        """Test SITL interface uses defaults when config missing.

        AC: 2 - Default configuration fallback
        """
        # GIVEN: SITL interface with non-existent config
        sitl = SITLInterface(config_path="nonexistent.yaml")

        # THEN: Should use default configuration
        assert sitl.config is not None
        assert "connection" in sitl.config
        assert sitl.config["connection"]["primary"] == "tcp:127.0.0.1:5760"

    def test_sitl_test_scenarios_configured(self) -> None:
        """Test that all required test scenarios are configured.

        AC: 5 - Test scenarios defined
        """
        # GIVEN: SITL config file
        config_path = Path("config/sitl.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # THEN: Test scenarios should be defined
        assert "test_scenarios" in config["sitl"]
        scenarios = config["sitl"]["test_scenarios"]

        # Verify required test scenarios
        assert "connection_test" in scenarios
        assert scenarios["connection_test"]["enabled"]

        assert "telemetry_test" in scenarios
        assert scenarios["telemetry_test"]["enabled"]
        assert scenarios["telemetry_test"]["expected_rate"] == 4  # Hz

        assert "command_test" in scenarios
        assert scenarios["command_test"]["enabled"]
        commands = scenarios["command_test"]["commands"]
        assert "arm" in commands
        assert "takeoff" in commands
        assert "land" in commands
        assert "disarm" in commands

        assert "safety_test" in scenarios
        assert scenarios["safety_test"]["enabled"]
        safety_tests = scenarios["safety_test"]["tests"]
        assert "emergency_stop" in safety_tests
        assert "rc_override" in safety_tests
        assert "battery_failsafe" in safety_tests
        assert "gps_failsafe" in safety_tests

        assert "performance_test" in scenarios
        assert scenarios["performance_test"]["enabled"]

    def test_sitl_setup_script_exists(self) -> None:
        """Test that SITL setup script exists.

        AC: 1 - Setup script available
        """
        # GIVEN: Expected script path
        script_path = Path("scripts/sitl_setup.py")

        # THEN: Script should exist
        assert script_path.exists(), f"SITL setup script not found at {script_path}"

        # AND: Script should be Python
        assert script_path.suffix == ".py"

    def test_sitl_test_runner_exists(self) -> None:
        """Test that SITL test runner script exists.

        AC: 1 - Test runner available
        """
        # GIVEN: Expected script path
        script_path = Path("scripts/run_sitl_tests.sh")

        # THEN: Script should exist
        assert script_path.exists(), f"SITL test runner not found at {script_path}"

        # AND: Script should be executable
        assert os.access(script_path, os.X_OK), "SITL test runner is not executable"

    def test_sitl_integration_tests_exist(self) -> None:
        """Test that SITL integration tests exist.

        AC: 5 - Integration tests created
        """
        # GIVEN: Expected test file path
        test_path = Path("tests/backend/integration/test_sitl_integration.py")

        # THEN: Test file should exist
        assert test_path.exists(), f"SITL integration tests not found at {test_path}"

        # AND: File should contain test classes
        with open(test_path) as f:
            content = f.read()
            assert "TestSITLIntegration" in content
            assert "test_sitl_basic_connection" in content
            assert "test_sitl_telemetry_streaming" in content
            assert "test_sitl_mode_changes" in content
            assert "test_sitl_arm_disarm" in content
            assert "test_sitl_takeoff_land" in content
            assert "test_sitl_velocity_commands" in content
            assert "test_sitl_emergency_stop" in content
            assert "test_sitl_battery_monitoring" in content
            assert "test_sitl_gps_requirements" in content

    def test_sitl_safety_requirements_match_story(self) -> None:
        """Test that SITL safety requirements match Story 4.7.

        AC: 5 - Safety requirements aligned
        """
        # GIVEN: SITL config
        config_path = Path("config/sitl.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        safety = config["sitl"]["safety"]

        # THEN: Safety requirements should match Story 4.7
        # From Story 4.7 Sprint 5 - Safety System Validation

        # SAFETY: Validate battery thresholds - HARA-PWR-001
        assert safety["battery_low_voltage"] == 19.2  # 6S Li-ion low threshold
        assert safety["battery_critical_voltage"] == 18.0  # 6S Li-ion critical

        # SAFETY: Validate GPS requirements - HARA-NAV-001
        assert safety["gps_min_satellites"] == 8  # GPS requirement
        assert safety["gps_max_hdop"] == 2.0  # HDOP requirement

        # SAFETY: Validate RC override - HARA-CTL-002
        assert safety["rc_override_threshold"] == 50  # PWM units

        # SAFETY: Validate emergency stop - HARA-CTL-001
        assert safety["emergency_stop_time"] == 0.5  # <500ms requirement

    def test_sitl_performance_requirements_match_story(self) -> None:
        """Test that SITL performance requirements match Story 4.7.

        AC: 8 - Performance requirements aligned
        """
        # GIVEN: SITL config
        config_path = Path("config/sitl.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        perf = config["sitl"]["performance"]

        # THEN: Performance requirements should match Story 4.7
        # From Story 4.7 Sprint 4 - Performance Validation
        assert perf["mode_change_latency"] == 0.1  # <100ms requirement
        assert perf["emergency_stop_latency"] == 0.5  # <500ms requirement
        assert perf["mavlink_latency"] == 0.05  # <50ms requirement
        assert perf["cpu_target"] == 30  # <30% CPU usage
        assert perf["ram_target"] == 500  # <500MB RAM usage
