"""
Unit tests for Safety Manager Dependency Injection

Tests SUBTASK-5.5.3.4 [11a-11f] implementation - dependency injection of
SafetyAuthorityManager throughout coordination stack with proper lifecycle management.

TDD Phase: RED - Writing failing tests first for dependency injection
"""

from unittest.mock import MagicMock

import pytest

from src.backend.core.dependencies import ServiceManager
from src.backend.services.safety_authority_manager import (
    SafetyAuthorityManager,
)


class TestDualSDRCoordinatorInjection:
    """Test suite for SafetyManager injection into DualSDRCoordinator [11a]"""

    @pytest.fixture
    def service_manager(self):
        """Create service manager for testing"""
        return ServiceManager()

    @pytest.fixture
    def safety_authority_manager(self):
        """Create safety authority manager for testing"""
        return SafetyAuthorityManager()

    @pytest.mark.asyncio
    async def test_dual_sdr_coordinator_constructor_with_safety_manager(
        self, service_manager, safety_authority_manager
    ):
        """Test DualSDRCoordinator accepts SafetyAuthorityManager in constructor"""
        # TDD RED: This test should FAIL initially
        from src.backend.services.dual_sdr_coordinator import DualSDRCoordinator

        # Test constructor with safety_authority parameter
        coordinator = DualSDRCoordinator(
            safety_authority=safety_authority_manager, service_manager=service_manager
        )

        assert coordinator._safety_authority is not None
        assert coordinator._safety_authority == safety_authority_manager
        assert coordinator._service_manager == service_manager

    @pytest.mark.asyncio
    async def test_dual_sdr_coordinator_lifecycle_with_safety_manager(self, service_manager):
        """Test DualSDRCoordinator lifecycle management with safety manager"""
        # TDD RED: This test should FAIL initially
        from src.backend.services.dual_sdr_coordinator import DualSDRCoordinator

        # Initialize service manager to get safety authority
        await service_manager.initialize_services()
        safety_authority = service_manager.get_service("safety_authority")

        coordinator = DualSDRCoordinator(
            safety_authority=safety_authority, service_manager=service_manager
        )

        # Test that coordinator can access safety functions
        result = coordinator.validate_command_before_execution(
            command="test_command", params={"test": "data"}
        )

        assert result is not None
        assert "authorized" in result

    @pytest.mark.asyncio
    async def test_dual_sdr_coordinator_initialization_validation(self, service_manager):
        """Test DualSDRCoordinator validates SafetyManager on initialization"""
        # TDD RED: This test should FAIL initially
        from src.backend.services.dual_sdr_coordinator import DualSDRCoordinator

        # Test with invalid safety manager
        coordinator = DualSDRCoordinator(safety_authority=None, service_manager=service_manager)

        # Should have validation that safety_authority is required
        with pytest.raises(ValueError, match="SafetyAuthorityManager is required"):
            await coordinator.initialize()


class TestSDRPriorityManagerInjection:
    """Test suite for SafetyManager dependency in SDRPriorityManager [11b]"""

    @pytest.fixture
    def safety_authority_manager(self):
        """Create safety authority manager for testing"""
        return SafetyAuthorityManager()

    def test_sdr_priority_manager_constructor_with_safety_manager(self, safety_authority_manager):
        """Test SDRPriorityManager accepts SafetyAuthorityManager in constructor"""
        # TDD RED: This test should FAIL initially
        from src.backend.services.sdr_priority_manager import SDRPriorityManager

        priority_manager = SDRPriorityManager(
            coordinator=MagicMock(),
            safety_manager=MagicMock(),
            safety_authority=safety_authority_manager,
        )

        assert priority_manager._safety_authority is not None
        assert priority_manager._safety_authority == safety_authority_manager

    def test_sdr_priority_manager_initialization_validation(self, safety_authority_manager):
        """Test SDRPriorityManager validates SafetyManager during initialization"""
        # TDD RED: This test should FAIL initially
        from src.backend.services.sdr_priority_manager import SDRPriorityManager

        # Test initialization with safety authority
        priority_manager = SDRPriorityManager(
            coordinator=MagicMock(),
            safety_manager=MagicMock(),
            safety_authority=safety_authority_manager,
        )

        # Should validate safety authority during initialization
        validation_result = priority_manager.validate_safety_authority_integration()

        assert validation_result["safety_authority_valid"] is True
        assert validation_result["integration_ready"] is True

    def test_sdr_priority_manager_safety_validation_integration(self, safety_authority_manager):
        """Test SDRPriorityManager integrates safety validation correctly"""
        # TDD RED: This test should FAIL initially
        from src.backend.services.sdr_priority_manager import SDRPriorityManager

        priority_manager = SDRPriorityManager(
            coordinator=MagicMock(),
            safety_manager=MagicMock(),
            safety_authority=safety_authority_manager,
        )

        # Test that safety validation is properly integrated
        result = priority_manager.validate_priority_decision(
            decision_type="test_decision", details={"test": "data"}
        )

        assert "validation_result" in result
        assert result["validation_result"]["authorized"] in [True, False]


class TestSDRPPBridgeServiceInjection:
    """Test suite for SafetyManager integration in SDRPPBridge service [11c]"""

    @pytest.fixture
    def safety_authority_manager(self):
        """Create safety authority manager for testing"""
        return SafetyAuthorityManager()

    def test_sdrpp_bridge_service_constructor_with_safety_manager(self, safety_authority_manager):
        """Test SDRPPBridgeService accepts SafetyAuthorityManager in constructor"""
        # TDD RED: This test should FAIL initially
        from src.backend.services.sdrpp_bridge_service import SDRPPBridgeService

        bridge_service = SDRPPBridgeService(safety_authority=safety_authority_manager)

        assert bridge_service._safety_authority is not None
        assert bridge_service._safety_authority == safety_authority_manager

    def test_sdrpp_bridge_service_communication_monitoring(self, safety_authority_manager):
        """Test SDRPPBridgeService implements communication monitoring with safety"""
        # TDD RED: This test should FAIL initially
        from src.backend.services.sdrpp_bridge_service import SDRPPBridgeService

        bridge_service = SDRPPBridgeService(safety_authority=safety_authority_manager)

        # Test communication monitoring with safety integration
        monitoring_result = bridge_service.monitor_communication_safety(
            connection_status="active", last_heartbeat_ms=100
        )

        assert monitoring_result["safety_status"] is not None
        assert "communication_health" in monitoring_result
        assert "safety_authority_notified" in monitoring_result


class TestSafetyManagerFactory:
    """Test suite for SafetyManager factory pattern [11d]"""

    @pytest.mark.asyncio
    async def test_safety_manager_factory_creation(self):
        """Test SafetyManager factory pattern for coordination services"""
        # TDD RED: This test should FAIL initially
        from src.backend.core.dependencies import SafetyManagerFactory

        factory = SafetyManagerFactory()

        # Test factory can create safety-enabled coordination services
        coordinator = await factory.create_dual_sdr_coordinator()
        assert coordinator._safety_authority is not None

        priority_manager = await factory.create_sdr_priority_manager()
        assert priority_manager._safety_authority is not None

        bridge_service = await factory.create_sdrpp_bridge_service()
        assert bridge_service._safety_authority is not None

    @pytest.mark.asyncio
    async def test_safety_manager_factory_with_service_manager(self):
        """Test SafetyManager factory integrates with ServiceManager"""
        # TDD RED: This test should FAIL initially
        from src.backend.core.dependencies import SafetyManagerFactory

        service_manager = ServiceManager()
        await service_manager.initialize_services()

        factory = SafetyManagerFactory(service_manager=service_manager)

        # Factory should use existing SafetyAuthorityManager from ServiceManager
        coordinator = await factory.create_dual_sdr_coordinator()
        expected_safety_authority = service_manager.get_service("safety_authority")

        assert coordinator._safety_authority == expected_safety_authority


class TestSafetyManagerConfigValidation:
    """Test suite for SafetyManager configuration validation [11e]"""

    @pytest.fixture
    def service_manager(self):
        """Create service manager for testing"""
        return ServiceManager()

    @pytest.mark.asyncio
    async def test_safety_manager_config_validation(self, service_manager):
        """Test SafetyManager configuration validation"""
        # TDD RED: This test should FAIL initially
        await service_manager.initialize_services()

        safety_authority = service_manager.get_service("safety_authority")

        # Test configuration validation
        config_validation = safety_authority.validate_configuration()

        assert config_validation["config_valid"] is True
        assert "authority_levels_configured" in config_validation
        assert config_validation["authority_levels_configured"] == 6  # Should have 6 levels
        assert "emergency_response_ready" in config_validation

    @pytest.mark.asyncio
    async def test_safety_manager_health_checking(self, service_manager):
        """Test SafetyManager health checking capabilities"""
        # TDD RED: This test should FAIL initially
        await service_manager.initialize_services()

        safety_authority = service_manager.get_service("safety_authority")

        # Test health check functionality
        health_check = safety_authority.perform_health_check()

        assert health_check["health_status"] in ["healthy", "degraded", "critical"]
        assert "response_time_ms" in health_check
        assert health_check["response_time_ms"] < 100  # Should be fast
        assert "authority_levels_active" in health_check


class TestSafetyManagerLifecycle:
    """Test suite for SafetyManager lifecycle management [11f]"""

    @pytest.fixture
    def service_manager(self):
        """Create service manager for testing"""
        return ServiceManager()

    @pytest.mark.asyncio
    async def test_safety_manager_startup_lifecycle(self, service_manager):
        """Test SafetyManager lifecycle during system startup"""
        # TDD RED: This test should FAIL initially
        # Safety manager should be initialized first in startup sequence
        await service_manager.initialize_services()

        # Verify safety manager is available and ready
        safety_authority = service_manager.get_service("safety_authority")
        assert safety_authority is not None

        # Test that safety manager is ready for coordination services
        lifecycle_status = service_manager.get_safety_manager_lifecycle_status()

        assert lifecycle_status["initialized"] is True
        assert lifecycle_status["ready_for_coordination"] is True
        assert lifecycle_status["emergency_response_active"] is True

    @pytest.mark.asyncio
    async def test_safety_manager_shutdown_lifecycle(self, service_manager):
        """Test SafetyManager lifecycle during system shutdown"""
        # TDD RED: This test should FAIL initially
        await service_manager.initialize_services()

        # Test graceful shutdown with safety manager
        shutdown_result = await service_manager.shutdown_services_safely()

        assert shutdown_result["safety_shutdown_successful"] is True
        assert shutdown_result["emergency_functions_preserved"] is True
        assert "shutdown_order" in shutdown_result

    @pytest.mark.asyncio
    async def test_coordination_services_lifecycle_with_safety(self, service_manager):
        """Test coordination services lifecycle management with safety integration"""
        # TDD RED: This test should FAIL initially
        await service_manager.initialize_services()

        # Test that coordination services can be started with safety integration
        coordination_lifecycle = service_manager.start_coordination_services_with_safety()

        assert coordination_lifecycle["coordination_services_started"] is True
        assert coordination_lifecycle["safety_integration_active"] is True
        assert len(coordination_lifecycle["services_with_safety"]) >= 3  # At least 3 services
        assert "DualSDRCoordinator" in coordination_lifecycle["services_with_safety"]
        assert "SDRPriorityManager" in coordination_lifecycle["services_with_safety"]
        assert "SDRPPBridgeService" in coordination_lifecycle["services_with_safety"]
