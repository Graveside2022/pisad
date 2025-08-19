"""
Test safety integration analysis for Story 5.5 SUBTASK-5.5.1.1

Validates that the safety integration analysis correctly identifies
all existing safety components and required integration points.
"""

import pytest

from src.backend.services.safety_integration_analysis import (
    IntegrationPriority,
    SafetyIntegrationAnalyzer,
    safety_analyzer,
)


class TestSafetyIntegrationAnalysis:
    """Test safety integration analysis functionality."""

    def test_analyze_existing_safety_system(self):
        """Test [1a] - Map all existing safety checks to coordination components."""
        analyzer = SafetyIntegrationAnalyzer()

        # Run the analysis
        results = analyzer.analyze_existing_safety_system()

        # Validate results structure
        assert isinstance(results, dict)
        assert "safety_components" in results
        assert "coordination_components" in results
        assert "integration_points" in results
        assert "critical_integrations" in results

        # Validate expected safety components are identified
        assert results["safety_components"] >= 6  # At least 6 safety components
        assert results["coordination_components"] >= 3  # At least 3 coordination components
        assert results["critical_integrations"] >= 3  # At least 3 critical integrations

        # Validate specific safety components are mapped
        components = results["components"]
        assert "safety_interlock_system" in components
        assert "safety_manager" in components
        assert "mode_check" in components
        assert "battery_check" in components

        # Validate critical components have correct priority
        safety_interlock = components["safety_interlock_system"]
        assert safety_interlock.priority == IntegrationPriority.CRITICAL

        safety_manager = components["safety_manager"]
        assert safety_manager.priority == IntegrationPriority.CRITICAL

    def test_identify_integration_points(self):
        """Test [1b] - Identify integration points requiring safety manager dependency."""
        analyzer = SafetyIntegrationAnalyzer()
        analyzer.analyze_existing_safety_system()

        # Verify critical integration points are identified
        critical_points = [
            ip for ip in analyzer.integration_points if ip.priority == IntegrationPriority.CRITICAL
        ]

        assert len(critical_points) >= 3

        # Verify specific critical integrations
        integration_names = [ip.name for ip in critical_points]
        assert "emergency_stop_integration" in integration_names
        assert "communication_health_monitoring" in integration_names
        assert "safety_aware_source_selection" in integration_names

        # Verify integration points specify dependency injection
        emergency_stop_integration = next(
            ip for ip in critical_points if ip.name == "emergency_stop_integration"
        )
        assert emergency_stop_integration.integration_type == "dependency_injection"
        assert emergency_stop_integration.coordination_component == "dual_sdr_coordinator"

    def test_safety_authority_hierarchy(self):
        """Test [1c] - Document safety authority hierarchy with coordination awareness."""
        analyzer = SafetyIntegrationAnalyzer()

        hierarchy = analyzer.get_safety_authority_hierarchy()

        # Validate hierarchy structure
        assert isinstance(hierarchy, dict)
        assert len(hierarchy) == 6  # 6 levels of safety authority

        # Validate level 1 emergency stop
        level_1 = hierarchy["level_1_emergency_stop"]
        assert level_1["response_time"] == "<500ms"
        assert level_1["coordination_integration"] == "Must trigger through DualSDRCoordinator"

        # Validate level 5 communication monitoring
        level_5 = hierarchy["level_5_communication"]
        assert level_5["response_time"] == "<10s"
        assert level_5["coordination_integration"] == "Automatic drone-only fallback"

    def test_story_2_2_validation(self):
        """Test [1d] - Verify all Story 2.2 safety interlocks implementation status."""
        analyzer = SafetyIntegrationAnalyzer()

        validation = analyzer.validate_story_2_2_safety_interlocks()

        # Validate all expected interlocks are present
        assert validation["total_interlocks"] == 6
        assert validation["implemented_count"] == 6  # All should be implemented
        assert validation["integration_needed_count"] == 6  # All need integration

        interlocks = validation["interlocks"]

        # Verify specific interlocks
        assert "emergency_stop_system" in interlocks
        assert interlocks["emergency_stop_system"]["implemented"] is True
        assert "emergency_stop_system" in validation["critical_integrations"]

        # Verify mode check
        assert "mode_check" in interlocks
        assert interlocks["mode_check"]["functionality"] == "Validates flight mode is GUIDED"

        # Verify signal loss monitoring
        assert "signal_loss_monitoring" in interlocks
        assert (
            interlocks["signal_loss_monitoring"]["functionality"]
            == "10-second timeout with SNR monitoring"
        )

    def test_integration_architecture_diagram(self):
        """Test [1e] - Create safety integration architecture diagram."""
        analyzer = SafetyIntegrationAnalyzer()
        analyzer.analyze_existing_safety_system()

        architecture = analyzer.generate_integration_architecture_diagram()

        # Validate architecture structure
        assert "safety_layer" in architecture
        assert "coordination_layer" in architecture
        assert "integration_connections" in architecture
        assert "data_flows" in architecture

        # Validate safety layer
        safety_layer = architecture["safety_layer"]
        assert safety_layer["primary_coordinator"] == "SafetyInterlockSystem"
        assert safety_layer["emergency_response"] == "SafetyManager"

        # Validate coordination layer
        coordination_layer = architecture["coordination_layer"]
        assert coordination_layer["primary_coordinator"] == "DualSDRCoordinator"
        assert coordination_layer["communication"] == "SDRPPBridge"

        # Validate integration connections
        connections = architecture["integration_connections"]
        assert len(connections) >= 3

        # Verify emergency stop connection
        emergency_connection = next(
            conn
            for conn in connections
            if conn["from"] == "SafetyManager" and conn["to"] == "DualSDRCoordinator"
        )
        assert emergency_connection["purpose"] == "Emergency stop integration"

    def test_emergency_response_pathways(self):
        """Test [1f] - Map emergency response pathways through coordination system."""
        analyzer = SafetyIntegrationAnalyzer()

        pathways = analyzer.identify_emergency_response_pathways()

        # Validate pathways structure
        assert isinstance(pathways, list)
        assert len(pathways) >= 3

        # Verify operator emergency stop pathway
        operator_pathway = next(
            p for p in pathways if p["pathway_name"] == "operator_emergency_stop"
        )
        assert operator_pathway["response_time_requirement"] == "<500ms"
        assert operator_pathway["safety_authority_level"] == 1
        assert (
            "DualSDRCoordinator.trigger_emergency_override()"
            in operator_pathway["coordination_path"]
        )

        # Verify communication loss fallback pathway
        comm_loss_pathway = next(
            p for p in pathways if p["pathway_name"] == "communication_loss_fallback"
        )
        assert comm_loss_pathway["response_time_requirement"] == "<10s"
        assert comm_loss_pathway["trigger"] == "TCP connection loss >10s"

    def test_comprehensive_analysis_report(self):
        """Test complete analysis report generation."""
        analyzer = SafetyIntegrationAnalyzer()

        report = analyzer.generate_comprehensive_analysis_report()

        # Validate report structure
        assert "analysis_summary" in report
        assert "system_analysis" in report
        assert "safety_authority_hierarchy" in report
        assert "emergency_response_pathways" in report
        assert "integration_architecture" in report
        assert "story_2_2_validation" in report
        assert "next_steps" in report

        # Validate analysis summary
        summary = report["analysis_summary"]
        assert summary["subtask"] == "SUBTASK-5.5.1.1"
        assert summary["completion_status"] == "COMPLETE"
        assert summary["safety_components_analyzed"] >= 6
        assert summary["coordination_components_analyzed"] >= 3

        # Validate next steps
        next_steps = report["next_steps"]
        assert len(next_steps) == 3
        assert any("SUBTASK-5.5.1.2" in step for step in next_steps)

    def test_safety_analyzer_singleton(self):
        """Test that safety_analyzer singleton is properly initialized."""
        # Test that the global analyzer instance is available
        assert safety_analyzer is not None
        assert isinstance(safety_analyzer, SafetyIntegrationAnalyzer)

        # Test that it can run analysis
        results = safety_analyzer.analyze_existing_safety_system()
        assert results["safety_components"] >= 6

    @pytest.mark.asyncio
    async def test_integration_with_actual_safety_system(self):
        """Test integration with actual safety system components."""
        # This test validates that our analysis correctly identifies real components
        try:
            from src.backend.services.dual_sdr_coordinator import DualSDRCoordinator
            from src.backend.services.safety_manager import SafetyManager
            from src.backend.utils.safety import ModeCheck, SafetyInterlockSystem

            # Verify that the classes we identified actually exist
            assert SafetyInterlockSystem is not None
            assert ModeCheck is not None
            assert SafetyManager is not None
            assert DualSDRCoordinator is not None

            # Verify specific methods exist on safety components
            safety_system = SafetyInterlockSystem()
            assert hasattr(safety_system, "emergency_stop")
            assert hasattr(safety_system, "check_all_safety")
            assert hasattr(safety_system, "is_safe_to_proceed")

            safety_manager = SafetyManager()
            assert hasattr(safety_manager, "trigger_emergency_stop")
            assert hasattr(safety_manager, "check_battery_status")

        except ImportError as e:
            pytest.skip(f"Could not import safety components: {e}")

    def test_coordination_component_analysis(self):
        """Test analysis of coordination components needing safety integration."""
        analyzer = SafetyIntegrationAnalyzer()
        analyzer._map_coordination_components()

        # Verify all coordination components are identified
        assert len(analyzer.coordination_components) >= 3

        # Verify DualSDRCoordinator analysis
        dual_sdr = analyzer.coordination_components["dual_sdr_coordinator"]
        assert dual_sdr.safety_integration_needed is True
        assert dual_sdr.dependency_injection_required is True
        assert "set_safety_manager" in dual_sdr.safety_methods_to_add
        assert "emergency_safety_override" in dual_sdr.safety_methods_to_add

        # Verify SDRPriorityManager analysis
        priority_mgr = analyzer.coordination_components["sdr_priority_manager"]
        assert priority_mgr.safety_integration_needed is True
        assert "safety_aware_decision" in priority_mgr.safety_methods_to_add

        # Verify SDRPPBridge analysis
        sdrpp_bridge = analyzer.coordination_components["sdrpp_bridge"]
        assert sdrpp_bridge.safety_integration_needed is True
        assert "safety_communication_loss" in sdrpp_bridge.safety_methods_to_add
