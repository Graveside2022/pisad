"""
Test all safety interlocks with SDR++ coordination active.

Tests SUBTASK-5.5.1.4 implementation with step [1t].
Validates that ALL existing safety interlocks continue to function
correctly when SDR++ coordination is active.

This ensures complete safety preservation during coordination operations.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.backend.services.dual_sdr_coordinator import DualSDRCoordinator
from src.backend.utils.safety import SafetyInterlockSystem


class TestAllSafetyInterlocksWithCoordination:
    """Test all safety interlocks with coordination active."""

    @pytest.fixture
    async def safety_system(self):
        """Create safety interlock system."""
        safety = SafetyInterlockSystem()
        await safety.start_monitoring()
        yield safety
        await safety.stop_monitoring()

    @pytest.fixture
    def dual_coordinator(self):
        """Create dual coordinator for interlock testing."""
        coordinator = DualSDRCoordinator()

        coordinator.get_health_status = AsyncMock(
            return_value={
                "coordination_latency_ms": 35.0,
                "ground_connection_status": 0.9,
                "coordination_active": True,
            }
        )
        coordinator.trigger_emergency_override = AsyncMock(
            return_value={
                "emergency_override_active": True,
                "response_time_ms": 90.0,
                "source_switched_to": "drone",
            }
        )
        coordinator.get_ground_rssi = MagicMock(return_value=14.0)
        coordinator.get_drone_rssi = MagicMock(return_value=11.0)

        return coordinator

    @pytest.mark.asyncio
    async def test_mode_check_interlock_with_coordination(self, safety_system, dual_coordinator):
        """Test [1t] - Mode check interlock works with coordination active."""
        # Enable coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Test mode check in different modes
        modes_to_test = ["GUIDED", "STABILIZE", "LOITER", "RTL", "AUTO"]

        for mode in modes_to_test:
            # Update flight mode
            safety_system.update_flight_mode(mode)

            # Check safety
            safety_results = await safety_system.check_all_safety()

            # Mode check should work correctly
            if mode == "GUIDED":
                assert safety_results["mode"] is True, f"Mode check should pass for {mode}"
            else:
                assert safety_results["mode"] is False, f"Mode check should fail for {mode}"

            # Coordination checks should continue working
            assert "coordination_health" in safety_results
            assert "dual_source_signal" in safety_results

    @pytest.mark.asyncio
    async def test_operator_activation_interlock_with_coordination(
        self, safety_system, dual_coordinator
    ):
        """Test [1t] - Operator activation interlock works with coordination."""
        # Enable coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Test operator activation states
        safety_results = await safety_system.check_all_safety()
        assert safety_results["operator"] is False, "Operator check should fail when disabled"

        # Enable homing through safety system
        safety_system.update_flight_mode("GUIDED")  # Ensure mode is correct
        await safety_system.enable_homing()

        # Check safety again
        safety_results = await safety_system.check_all_safety()
        assert safety_results["operator"] is True, "Operator check should pass when enabled"

        # Disable homing
        await safety_system.disable_homing("Test disable")

        safety_results = await safety_system.check_all_safety()
        assert safety_results["operator"] is False, "Operator check should fail when disabled"

    @pytest.mark.asyncio
    async def test_signal_loss_interlock_with_coordination(self, safety_system, dual_coordinator):
        """Test [1t] - Signal loss interlock works with coordination."""
        # Enable coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Test with good signal
        safety_system.update_signal_snr(10.0)  # Above 6dB threshold
        safety_results = await safety_system.check_all_safety()
        assert safety_results["signal"] is True, "Signal check should pass with good SNR"

        # Test with poor signal
        safety_system.update_signal_snr(3.0)  # Below 6dB threshold
        safety_results = await safety_system.check_all_safety()
        assert safety_results["signal"] is False, "Signal check should fail with poor SNR"

        # Coordination checks should still function
        assert safety_results["coordination_health"] is True
        assert safety_results["dual_source_signal"] is True

    @pytest.mark.asyncio
    async def test_battery_interlock_with_coordination(self, safety_system, dual_coordinator):
        """Test [1t] - Battery interlock works with coordination."""
        # Enable coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Test with good battery
        safety_system.update_battery(75.0)  # Above 20% threshold
        safety_results = await safety_system.check_all_safety()
        assert safety_results["battery"] is True, "Battery check should pass with good level"

        # Test with low battery
        safety_system.update_battery(15.0)  # Below 20% threshold
        safety_results = await safety_system.check_all_safety()
        assert safety_results["battery"] is False, "Battery check should fail with low level"

        # Test critical battery
        safety_system.update_battery(5.0)  # Very low
        safety_results = await safety_system.check_all_safety()
        assert safety_results["battery"] is False, "Battery check should fail with critical level"

    @pytest.mark.asyncio
    async def test_geofence_interlock_with_coordination(self, safety_system, dual_coordinator):
        """Test [1t] - Geofence interlock works with coordination."""
        # Enable coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Setup geofence
        geofence_check = safety_system.checks["geofence"]
        geofence_check.set_geofence(
            center_lat=40.7128, center_lon=-74.0060, radius_meters=1000.0, altitude=100.0
        )

        # Test position inside geofence
        safety_system.update_position(40.7120, -74.0050, 50.0)  # Inside fence
        safety_results = await safety_system.check_all_safety()
        assert safety_results["geofence"] is True, "Geofence check should pass inside boundary"

        # Test position outside geofence (too far)
        safety_system.update_position(40.7200, -74.0200, 50.0)  # Outside fence
        safety_results = await safety_system.check_all_safety()
        assert safety_results["geofence"] is False, "Geofence check should fail outside boundary"

        # Test altitude violation
        safety_system.update_position(40.7120, -74.0050, 150.0)  # Too high
        safety_results = await safety_system.check_all_safety()
        assert (
            safety_results["geofence"] is False
        ), "Geofence check should fail with altitude violation"

    @pytest.mark.asyncio
    async def test_coordination_health_interlock_with_coordination(
        self, safety_system, dual_coordinator
    ):
        """Test [1t] - Coordination health interlock works correctly."""
        # Enable coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Test with healthy coordination
        safety_results = await safety_system.check_all_safety()
        assert safety_results["coordination_health"] is True, "Coordination health should pass"

        # Test with degraded coordination
        dual_coordinator.get_health_status = AsyncMock(
            return_value={
                "coordination_latency_ms": 150.0,  # Above 100ms threshold
                "ground_connection_status": 0.3,  # Poor quality
                "coordination_active": True,
            }
        )
        dual_coordinator.get_ground_rssi = MagicMock(return_value=12.0)

        safety_results = await safety_system.check_all_safety()
        assert safety_results["coordination_health"] is False, "Coordination health should fail"

    @pytest.mark.asyncio
    async def test_dual_source_signal_interlock_with_coordination(
        self, safety_system, dual_coordinator
    ):
        """Test [1t] - Dual source signal interlock works correctly."""
        # Enable coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Test with both sources good
        dual_coordinator.get_ground_rssi = MagicMock(return_value=12.0)  # Above 6dB
        dual_coordinator.get_drone_rssi = MagicMock(return_value=10.0)  # Above 6dB

        safety_results = await safety_system.check_all_safety()
        assert (
            safety_results["dual_source_signal"] is True
        ), "Dual source should pass with both good"

        # Test with one source failed
        dual_coordinator.get_ground_rssi = MagicMock(return_value=4.0)  # Below 6dB
        dual_coordinator.get_drone_rssi = MagicMock(return_value=10.0)  # Above 6dB

        safety_results = await safety_system.check_all_safety()
        assert safety_results["dual_source_signal"] is True, "Dual source should pass with one good"

        # Test with both sources failed
        dual_coordinator.get_ground_rssi = MagicMock(return_value=3.0)  # Below 6dB
        dual_coordinator.get_drone_rssi = MagicMock(return_value=2.0)  # Below 6dB

        safety_results = await safety_system.check_all_safety()
        assert (
            safety_results["dual_source_signal"] is False
        ), "Dual source should fail with both bad"

    @pytest.mark.asyncio
    async def test_all_interlocks_together_with_coordination(self, safety_system, dual_coordinator):
        """Test [1t] - All interlocks work together with coordination active."""
        # Enable coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Set up all systems for success
        safety_system.update_flight_mode("GUIDED")
        safety_system.update_battery(80.0)
        safety_system.update_signal_snr(12.0)

        # Enable operator
        await safety_system.enable_homing()

        # Setup geofence
        geofence_check = safety_system.checks["geofence"]
        geofence_check.set_geofence(40.7128, -74.0060, 1000.0)
        safety_system.update_position(40.7120, -74.0050, 50.0)

        # All checks should pass
        safety_results = await safety_system.check_all_safety()
        for check_name, result in safety_results.items():
            assert result is True, f"Check {check_name} should pass with good conditions"

        # Overall safety should pass
        is_safe = await safety_system.is_safe_to_proceed()
        assert is_safe is True, "Overall safety should pass with all good conditions"

    @pytest.mark.asyncio
    async def test_multiple_interlock_failures_with_coordination(
        self, safety_system, dual_coordinator
    ):
        """Test [1t] - Multiple interlock failures with coordination active."""
        # Enable coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Set up multiple failure conditions
        safety_system.update_flight_mode("STABILIZE")  # Wrong mode
        safety_system.update_battery(15.0)  # Low battery
        safety_system.update_signal_snr(3.0)  # Poor signal

        # Check results
        safety_results = await safety_system.check_all_safety()

        # Multiple checks should fail
        assert safety_results["mode"] is False, "Mode check should fail"
        assert safety_results["battery"] is False, "Battery check should fail"
        assert safety_results["signal"] is False, "Signal check should fail"
        assert safety_results["operator"] is False, "Operator check should fail (not enabled)"

        # Coordination checks should still work
        assert safety_results["coordination_health"] is True, "Coordination should still work"
        assert safety_results["dual_source_signal"] is True, "Dual source should still work"

        # Overall safety should fail
        is_safe = await safety_system.is_safe_to_proceed()
        assert is_safe is False, "Overall safety should fail with multiple failures"

    @pytest.mark.asyncio
    async def test_emergency_stop_affects_all_interlocks(self, safety_system, dual_coordinator):
        """Test [1t] - Emergency stop affects all interlocks with coordination."""
        # Enable coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Set up good conditions
        safety_system.update_flight_mode("GUIDED")
        safety_system.update_battery(80.0)
        safety_system.update_signal_snr(12.0)
        await safety_system.enable_homing()

        # Verify all good before emergency stop
        safety_results = await safety_system.check_all_safety()
        all_standard_good = all(
            safety_results[check] for check in ["mode", "operator", "signal", "battery"]
        )
        assert all_standard_good, "Standard checks should pass before emergency stop"

        # Trigger emergency stop
        await safety_system.emergency_stop("Test emergency stop")

        # All checks should now fail due to emergency stop
        safety_results = await safety_system.check_all_safety()
        for check_name, result in safety_results.items():
            assert result is False, f"Check {check_name} should fail during emergency stop"

        # Overall safety should fail
        is_safe = await safety_system.is_safe_to_proceed()
        assert is_safe is False, "Overall safety should fail during emergency stop"

    @pytest.mark.asyncio
    async def test_interlock_timing_with_coordination(self, safety_system, dual_coordinator):
        """Test [1t] - All interlock timing remains fast with coordination."""
        import time

        # Enable coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Test timing of all safety checks
        start_time = time.perf_counter()

        safety_results = await safety_system.check_all_safety()

        end_time = time.perf_counter()
        total_check_time_ms = (end_time - start_time) * 1000

        # All checks should complete quickly
        assert total_check_time_ms < 100.0, f"All safety checks took {total_check_time_ms:.1f}ms"

        # Should have all expected checks
        expected_checks = [
            "mode",
            "operator",
            "signal",
            "battery",
            "geofence",
            "coordination_health",
            "dual_source_signal",
        ]
        for check_name in expected_checks:
            assert check_name in safety_results, f"Missing check: {check_name}"

    @pytest.mark.asyncio
    async def test_interlock_independence_with_coordination(self, safety_system, dual_coordinator):
        """Test [1t] - Interlocks remain independent with coordination active."""
        # Enable coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)

        # Test that one interlock failure doesn't affect others
        safety_system.update_flight_mode("GUIDED")  # Good
        safety_system.update_battery(80.0)  # Good
        safety_system.update_signal_snr(12.0)  # Good
        await safety_system.enable_homing()  # Good

        # Fail only battery
        safety_system.update_battery(10.0)  # Bad

        safety_results = await safety_system.check_all_safety()

        # Only battery should fail
        assert safety_results["mode"] is True, "Mode should still pass"
        assert safety_results["operator"] is True, "Operator should still pass"
        assert safety_results["signal"] is True, "Signal should still pass"
        assert safety_results["battery"] is False, "Battery should fail"
        assert safety_results["coordination_health"] is True, "Coordination health should pass"
        assert safety_results["dual_source_signal"] is True, "Dual source should pass"

    @pytest.mark.asyncio
    async def test_coordination_disabled_interlock_behavior(self, safety_system, dual_coordinator):
        """Test [1t] - Standard interlocks work when coordination is disabled."""
        # Setup coordination but disable it
        safety_system.set_coordination_system(dual_coordinator, active=False)

        # Set up good standard conditions
        safety_system.update_flight_mode("GUIDED")
        safety_system.update_battery(80.0)
        safety_system.update_signal_snr(12.0)
        await safety_system.enable_homing()

        # Check safety
        safety_results = await safety_system.check_all_safety()

        # Standard checks should work normally
        assert safety_results["mode"] is True, "Mode check should work"
        assert safety_results["operator"] is True, "Operator check should work"
        assert safety_results["signal"] is True, "Signal check should work"
        assert safety_results["battery"] is True, "Battery check should work"

        # Coordination checks should be present but not active
        assert "coordination_health" in safety_results
        assert "dual_source_signal" in safety_results
