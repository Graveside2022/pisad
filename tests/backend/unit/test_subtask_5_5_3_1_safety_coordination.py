"""
Test Suite for SUBTASK-5.5.3.1: Enhance SafetyManager for SDR++ coordination awareness

Tests implement authentic TDD for safety architecture integration enhancements.

PRD References:
- PRD-AC5.5.4: Safety authority hierarchy maintained with coordination
- PRD-FR16: Emergency stop <500ms response time
- PRD-AC5.5.1: All existing PISAD safety interlocks remain active

Test Coverage:
- [8a] SDR++ coordination status monitoring methods
- [8b] Dual-system battery monitoring with coordination health awareness  
- [8c] Coordination health status integration into safety decision matrix
- [8d] Safety-aware source selection criteria with fallback triggers
- [8e] Safety event triggers for coordination system state changes
- [8f] Coordination latency monitoring to safety timing validation framework
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, Mock

from src.backend.services.safety_manager import SafetyManager
from src.backend.core.exceptions import SafetyInterlockError


class TestSafetyManagerCoordinationAwareness:
    """Test SafetyManager enhancements for SDR++ coordination awareness."""

    @pytest.fixture
    def safety_manager(self):
        """Create SafetyManager instance for testing."""
        return SafetyManager()

    def test_coordination_status_monitoring_integration_working(self, safety_manager):
        """
        Test [8a]: SDR++ coordination status monitoring methods to SafetyManager.
        
        TDD GREEN PHASE: Methods should now exist and work correctly.
        """
        # Test that coordination monitoring methods exist
        assert hasattr(safety_manager, 'get_coordination_status')
        assert hasattr(safety_manager, 'set_coordination_status')  
        assert hasattr(safety_manager, 'is_coordination_healthy')
        
        # Test default coordination status
        status = safety_manager.get_coordination_status()
        assert status["active"] == False
        assert status["healthy"] == True
        assert status["source"] == "drone"
        
        # Test setting coordination status
        new_status = {"active": True, "healthy": False, "source": "ground"}
        safety_manager.set_coordination_status(new_status)
        
        updated_status = safety_manager.get_coordination_status()
        assert updated_status["active"] == True
        assert updated_status["healthy"] == False
        assert updated_status["source"] == "ground"
        
        # Test health checking
        assert safety_manager.is_coordination_healthy() == False

    def test_dual_system_battery_monitoring_working(self, safety_manager):
        """
        Test [8b]: Dual-system battery monitoring with coordination health awareness.
        
        TDD GREEN PHASE: Enhanced battery monitoring should work correctly.
        """
        # Test that enhanced battery monitoring methods exist
        assert hasattr(safety_manager, 'check_dual_system_battery_status')
        assert hasattr(safety_manager, 'get_coordination_battery_health')
        
        # Test dual system battery monitoring
        dual_battery = safety_manager.check_dual_system_battery_status()
        assert "drone" in dual_battery
        assert "ground" in dual_battery
        assert "coordination_impact" in dual_battery
        assert dual_battery["coordination_impact"] == True  # healthy by default
        
        # Test coordination battery health
        battery_health = safety_manager.get_coordination_battery_health()
        assert "ground" in battery_health
        assert "drone" in battery_health
        assert battery_health["ground"] is None  # default state

    def test_coordination_health_integration_working(self, safety_manager):
        """
        Test [8c]: Coordination health status integration into safety decision matrix.
        
        TDD GREEN PHASE: Health integration should work correctly.
        """
        # Test that coordination health integration exists
        assert hasattr(safety_manager, 'include_coordination_in_decisions')
        
        # Test default state (coordination not active)
        assert safety_manager.include_coordination_in_decisions() == False
        failsafe = safety_manager.get_failsafe_action()
        assert "coordination_health" not in failsafe
        
        # Mock MAVLink to avoid GPS/battery failures interfering with test
        mock_mavlink = Mock()
        mock_mavlink.telemetry = {
            "gps": {"satellites": 10, "hdop": 1.0, "fix_type": 3},
            "battery": {"voltage": 22.0}  # Above low threshold
        }
        safety_manager.mavlink = mock_mavlink
        
        # Test with coordination active and healthy
        safety_manager.set_coordination_status({"active": True, "healthy": True})
        assert safety_manager.include_coordination_in_decisions() == True
        
        failsafe_healthy = safety_manager.get_failsafe_action()
        assert "coordination_health" in failsafe_healthy
        assert failsafe_healthy["coordination_health"] == True
        
        # Test with coordination active but unhealthy
        safety_manager.set_coordination_status({"active": True, "healthy": False})
        failsafe_unhealthy = safety_manager.get_failsafe_action()
        assert failsafe_unhealthy["priority"] == 4
        assert failsafe_unhealthy["action"] == "FALLBACK_DRONE"
        assert failsafe_unhealthy["coordination_health"] == False

    def test_safety_aware_source_selection_working(self, safety_manager):
        """
        Test [8d]: Safety-aware source selection criteria with fallback triggers.
        
        TDD GREEN PHASE: Source selection should work correctly.
        """
        # Test that source selection methods exist
        assert hasattr(safety_manager, 'evaluate_source_safety')
        assert hasattr(safety_manager, 'get_safe_source_recommendation')
        assert hasattr(safety_manager, 'trigger_source_fallback')
        
        # Test drone source safety (always safe)
        drone_safety = safety_manager.evaluate_source_safety("drone")
        assert drone_safety["safe"] == True
        assert drone_safety["reason"] == "primary_safety_authority"
        
        # Test ground source safety (depends on coordination health)
        ground_safety_healthy = safety_manager.evaluate_source_safety("ground")
        assert ground_safety_healthy["safe"] == True  # healthy by default
        
        # Test with unhealthy coordination
        safety_manager.set_coordination_status({"healthy": False})
        ground_safety_unhealthy = safety_manager.evaluate_source_safety("ground")
        assert ground_safety_unhealthy["safe"] == False
        
        # Test safe source recommendation
        safe_source = safety_manager.get_safe_source_recommendation()
        assert safe_source == "drone"  # Should fallback to drone when unhealthy
        
        # Test fallback triggering
        safety_manager.trigger_source_fallback("test_reason")
        status = safety_manager.get_coordination_status()
        assert status["source"] == "drone"
        assert status["fallback_reason"] == "test_reason"

    def test_safety_event_triggers_working(self, safety_manager):
        """
        Test [8e]: Safety event triggers for coordination system state changes.
        
        TDD GREEN PHASE: Event triggers should work correctly.
        """
        # Test that coordination event triggers exist
        assert hasattr(safety_manager, 'handle_coordination_state_change')
        assert hasattr(safety_manager, 'register_coordination_event_handler')
        
        # Test state change handling
        initial_status = safety_manager.get_coordination_status()
        assert initial_status["healthy"] == True
        
        # Trigger state change to unhealthy
        new_state = {"healthy": False, "active": True}
        safety_manager.handle_coordination_state_change(new_state)
        
        # Should have triggered fallback
        updated_status = safety_manager.get_coordination_status()
        assert updated_status["healthy"] == False
        assert updated_status["source"] == "drone"  # fallback triggered
        assert "fallback_reason" in updated_status
        
        # Test event handler registration (placeholder functionality)
        def dummy_handler(event):
            pass
        
        # Should not raise exception
        safety_manager.register_coordination_event_handler(dummy_handler)

    def test_coordination_latency_monitoring_working(self, safety_manager):
        """
        Test [8f]: Coordination latency monitoring to safety timing validation framework.
        
        TDD GREEN PHASE: Latency monitoring should work correctly.
        """
        # Test that latency monitoring methods exist
        assert hasattr(safety_manager, 'monitor_coordination_latency')
        assert hasattr(safety_manager, 'validate_coordination_timing')
        assert hasattr(safety_manager, 'get_coordination_latency_status')
        
        # Test latency monitoring
        safety_manager.monitor_coordination_latency(50.0)
        assert safety_manager.coordination_latency_ms == 50.0
        
        # Test timing validation (within limits)
        timing_valid = safety_manager.validate_coordination_timing()
        assert timing_valid["valid"] == True
        assert timing_valid["latency_ms"] == 50.0
        assert timing_valid["limit_ms"] == 100.0
        
        # Test with excessive latency
        safety_manager.monitor_coordination_latency(150.0)
        timing_invalid = safety_manager.validate_coordination_timing()
        assert timing_invalid["valid"] == False
        assert timing_invalid["latency_ms"] == 150.0
        
        # Test latency status
        latency_status = safety_manager.get_coordination_latency_status()
        assert latency_status["latency_ms"] == 150.0
        assert latency_status["within_limits"] == False
        assert latency_status["safety_impact"] == "warning"
        
        # Test emergency stop includes coordination latency when active
        safety_manager.set_coordination_status({"active": True})
        emergency_result = safety_manager.trigger_emergency_stop()
        assert 'coordination_latency_ms' in emergency_result
        assert emergency_result['coordination_latency_ms'] == 150.0