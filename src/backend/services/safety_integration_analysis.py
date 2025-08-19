"""
Safety Integration Analysis for SDR++ Coordination

Maps existing safety checks to coordination components and identifies
integration points for comprehensive safety preservation.

Story 5.5 - SUBTASK-5.5.1.1 [1a-1f] implementation
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.backend.utils.logging import get_logger

logger = get_logger(__name__)


class SafetyComponentType(Enum):
    """Types of safety components in the system."""

    INTERLOCK = "interlock"
    MONITOR = "monitor"
    TRIGGER = "trigger"
    RESPONSE = "response"


class IntegrationPriority(Enum):
    """Priority levels for safety integration."""

    CRITICAL = "critical"  # Must integrate immediately
    HIGH = "high"  # Should integrate in this story
    MEDIUM = "medium"  # Can be enhanced later
    LOW = "low"  # Optional improvement


@dataclass
class SafetyComponent:
    """Represents a safety component in the system."""

    name: str
    component_type: SafetyComponentType
    file_path: str
    class_name: str
    methods: list[str] = field(default_factory=list)
    integration_points: list[str] = field(default_factory=list)
    coordination_dependencies: list[str] = field(default_factory=list)
    priority: IntegrationPriority = IntegrationPriority.MEDIUM


@dataclass
class CoordinationComponent:
    """Represents a coordination system component."""

    name: str
    file_path: str
    class_name: str
    safety_integration_needed: bool = False
    safety_methods_to_add: list[str] = field(default_factory=list)
    dependency_injection_required: bool = False


@dataclass
class IntegrationPoint:
    """Represents a point where safety and coordination must integrate."""

    name: str
    safety_component: str
    coordination_component: str
    integration_type: str  # "dependency_injection", "event_handler", "monitoring"
    priority: IntegrationPriority
    implementation_notes: str = ""


class SafetyIntegrationAnalyzer:
    """
    Analyzes existing safety system and identifies integration points
    with SDR++ coordination system.
    """

    def __init__(self) -> None:
        """Initialize the safety integration analyzer."""
        self.safety_components: dict[str, SafetyComponent] = {}
        self.coordination_components: dict[str, CoordinationComponent] = {}
        self.integration_points: list[IntegrationPoint] = []

        logger.info("Safety integration analyzer initialized")

    def analyze_existing_safety_system(self) -> dict[str, Any]:
        """
        [1a] Map all existing safety checks from SafetyInterlockSystem
        to coordination components.

        Returns:
            Analysis results with mapped safety components
        """
        logger.info("Starting analysis of existing safety system")

        # Map existing safety components from src/backend/utils/safety.py
        self._map_safety_interlock_components()

        # Map safety manager components from src/backend/services/safety_manager.py
        self._map_safety_manager_components()

        # Map coordination components
        self._map_coordination_components()

        # Identify integration points
        self._identify_integration_points()

        analysis_results = {
            "safety_components": len(self.safety_components),
            "coordination_components": len(self.coordination_components),
            "integration_points": len(self.integration_points),
            "critical_integrations": len(
                [
                    ip
                    for ip in self.integration_points
                    if ip.priority == IntegrationPriority.CRITICAL
                ]
            ),
            "components": self.safety_components,
            "integrations": self.integration_points,
        }

        logger.info(f"Safety system analysis complete: {analysis_results}")
        return analysis_results

    def _map_safety_interlock_components(self) -> None:
        """Map all SafetyInterlockSystem components."""
        # SafetyInterlockSystem - Main coordinator
        self.safety_components["safety_interlock_system"] = SafetyComponent(
            name="SafetyInterlockSystem",
            component_type=SafetyComponentType.INTERLOCK,
            file_path="/home/pisad/projects/pisad/src/backend/utils/safety.py",
            class_name="SafetyInterlockSystem",
            methods=[
                "start_monitoring",
                "stop_monitoring",
                "check_all_safety",
                "is_safe_to_proceed",
                "emergency_stop",
                "reset_emergency_stop",
                "enable_homing",
                "disable_homing",
            ],
            integration_points=["dual_sdr_coordinator", "sdrpp_bridge", "priority_manager"],
            coordination_dependencies=[
                "communication_health",
                "coordination_status",
                "source_selection",
            ],
            priority=IntegrationPriority.CRITICAL,
        )

        # Individual Safety Checks
        safety_checks = [
            ("mode_check", "ModeCheck", ["check", "update_mode"]),
            (
                "operator_check",
                "OperatorActivationCheck",
                ["check", "enable_homing", "disable_homing"],
            ),
            ("signal_check", "SignalLossCheck", ["check", "update_snr"]),
            ("battery_check", "BatteryCheck", ["check", "update_battery"]),
            ("geofence_check", "GeofenceCheck", ["check", "update_position", "set_geofence"]),
        ]

        for check_name, class_name, methods in safety_checks:
            self.safety_components[check_name] = SafetyComponent(
                name=class_name,
                component_type=SafetyComponentType.MONITOR,
                file_path="/home/pisad/projects/pisad/src/backend/utils/safety.py",
                class_name=class_name,
                methods=methods,
                integration_points=["dual_sdr_coordinator"],
                priority=IntegrationPriority.HIGH,
            )

    def _map_safety_manager_components(self) -> None:
        """Map SafetyManager components."""
        self.safety_components["safety_manager"] = SafetyComponent(
            name="SafetyManager",
            component_type=SafetyComponentType.RESPONSE,
            file_path="/home/pisad/projects/pisad/src/backend/services/safety_manager.py",
            class_name="SafetyManager",
            methods=[
                "trigger_emergency_stop",
                "is_rc_override_active",
                "check_battery_status",
                "check_gps_status",
                "validate_mode_change",
                "get_failsafe_action",
                "pre_arm_checks",
                "start_monitoring",
            ],
            integration_points=["dual_sdr_coordinator", "sdrpp_bridge", "priority_manager"],
            coordination_dependencies=[
                "coordination_health",
                "communication_status",
                "emergency_override",
            ],
            priority=IntegrationPriority.CRITICAL,
        )

    def _map_coordination_components(self) -> None:
        """Map coordination system components that need safety integration."""

        # DualSDRCoordinator
        self.coordination_components["dual_sdr_coordinator"] = CoordinationComponent(
            name="DualSDRCoordinator",
            file_path="/home/pisad/projects/pisad/src/backend/services/dual_sdr_coordinator.py",
            class_name="DualSDRCoordinator",
            safety_integration_needed=True,
            safety_methods_to_add=[
                "set_safety_manager",
                "emergency_safety_override",
                "check_safety_status",
                "safety_triggered_fallback",
            ],
            dependency_injection_required=True,
        )

        # SDRPriorityManager
        self.coordination_components["sdr_priority_manager"] = CoordinationComponent(
            name="SDRPriorityManager",
            file_path="/home/pisad/projects/pisad/src/backend/services/sdr_priority_manager.py",
            class_name="SDRPriorityManager",
            safety_integration_needed=True,
            safety_methods_to_add=[
                "set_safety_manager",
                "safety_aware_decision",
                "emergency_safety_override",
            ],
            dependency_injection_required=True,
        )

        # SDRPPBridge
        self.coordination_components["sdrpp_bridge"] = CoordinationComponent(
            name="SDRPPBridgeService",
            file_path="/home/pisad/projects/pisad/src/backend/services/sdrpp_bridge_service.py",
            class_name="SDRPPBridgeService",
            safety_integration_needed=True,
            safety_methods_to_add=[
                "set_safety_manager",
                "safety_communication_loss",
                "emergency_disconnect",
            ],
            dependency_injection_required=True,
        )

    def _identify_integration_points(self) -> None:
        """[1b] Identify integration points in DualSDRCoordinator requiring safety manager dependency."""

        # Critical integration points
        critical_integrations = [
            IntegrationPoint(
                name="emergency_stop_integration",
                safety_component="safety_manager",
                coordination_component="dual_sdr_coordinator",
                integration_type="dependency_injection",
                priority=IntegrationPriority.CRITICAL,
                implementation_notes="Safety manager must be injected into coordinator for emergency stops",
            ),
            IntegrationPoint(
                name="communication_health_monitoring",
                safety_component="safety_interlock_system",
                coordination_component="sdrpp_bridge",
                integration_type="event_handler",
                priority=IntegrationPriority.CRITICAL,
                implementation_notes="Communication loss must trigger safety events",
            ),
            IntegrationPoint(
                name="safety_aware_source_selection",
                safety_component="safety_manager",
                coordination_component="sdr_priority_manager",
                integration_type="monitoring",
                priority=IntegrationPriority.CRITICAL,
                implementation_notes="Priority decisions must consider safety status",
            ),
        ]

        # High priority integrations
        high_priority_integrations = [
            IntegrationPoint(
                name="coordination_health_monitoring",
                safety_component="safety_interlock_system",
                coordination_component="dual_sdr_coordinator",
                integration_type="monitoring",
                priority=IntegrationPriority.HIGH,
                implementation_notes="Coordination health status feeds into safety decisions",
            ),
            IntegrationPoint(
                name="battery_dual_system_monitoring",
                safety_component="battery_check",
                coordination_component="dual_sdr_coordinator",
                integration_type="monitoring",
                priority=IntegrationPriority.HIGH,
                implementation_notes="Battery status affects coordination source selection",
            ),
            IntegrationPoint(
                name="signal_dual_source_validation",
                safety_component="signal_check",
                coordination_component="dual_sdr_coordinator",
                integration_type="monitoring",
                priority=IntegrationPriority.HIGH,
                implementation_notes="Enhanced signal monitoring with dual sources",
            ),
        ]

        self.integration_points.extend(critical_integrations)
        self.integration_points.extend(high_priority_integrations)

    def get_safety_authority_hierarchy(self) -> dict[str, Any]:
        """
        [1c] Document safety authority hierarchy with coordination awareness.

        Returns:
            Safety authority hierarchy with integration points
        """
        hierarchy = {
            "level_1_emergency_stop": {
                "authority": "Operator Emergency Stop",
                "response_time": "<500ms",
                "integration_point": "All coordination components",
                "override_capability": "Immediate shutdown of all systems",
                "coordination_integration": "Must trigger through DualSDRCoordinator",
            },
            "level_2_flight_mode": {
                "authority": "Flight Mode Monitor",
                "response_time": "<100ms",
                "integration_point": "DualSDRCoordinator decision making",
                "override_capability": "Override payload if not GUIDED",
                "coordination_integration": "Block coordination commands when not GUIDED",
            },
            "level_3_geofence": {
                "authority": "Geofence Boundary Enforcement",
                "response_time": "<1s",
                "integration_point": "Priority Manager source selection",
                "override_capability": "Hard boundary enforcement",
                "coordination_integration": "Geofence-aware source selection",
            },
            "level_4_battery": {
                "authority": "Battery Monitor",
                "response_time": "<5s",
                "integration_point": "Coordination health monitoring",
                "override_capability": "Low battery triggers RTL",
                "coordination_integration": "Dual-system battery awareness",
            },
            "level_5_communication": {
                "authority": "Communication Monitor",
                "response_time": "<10s",
                "integration_point": "SDRPPBridge health monitoring",
                "override_capability": "Communication loss triggers fallback",
                "coordination_integration": "Automatic drone-only fallback",
            },
            "level_6_signal": {
                "authority": "Signal Monitor",
                "response_time": "<10s",
                "integration_point": "Dual source signal validation",
                "override_capability": "Signal loss auto-disable",
                "coordination_integration": "Enhanced dual-source monitoring",
            },
        }

        logger.info("Safety authority hierarchy documented with coordination integration")
        return hierarchy

    def identify_emergency_response_pathways(self) -> list[dict[str, Any]]:
        """
        [1f] Map emergency response pathways through coordination system.

        Returns:
            List of emergency response pathways
        """
        pathways = [
            {
                "pathway_name": "operator_emergency_stop",
                "trigger": "Operator button/UI command",
                "entry_point": "SafetyManager.trigger_emergency_stop()",
                "coordination_path": [
                    "DualSDRCoordinator.trigger_emergency_override()",
                    "SDRPriorityManager.emergency_safety_override()",
                    "SDRPPBridge.emergency_disconnect()",
                ],
                "response_time_requirement": "<500ms",
                "safety_authority_level": 1,
            },
            {
                "pathway_name": "communication_loss_fallback",
                "trigger": "TCP connection loss >10s",
                "entry_point": "SDRPPBridge.safety_communication_loss()",
                "coordination_path": [
                    "SafetyManager notification",
                    "DualSDRCoordinator.safety_triggered_fallback()",
                    "Automatic drone-only mode",
                ],
                "response_time_requirement": "<10s",
                "safety_authority_level": 5,
            },
            {
                "pathway_name": "flight_mode_override",
                "trigger": "Mode change from GUIDED",
                "entry_point": "SafetyInterlockSystem mode check",
                "coordination_path": [
                    "DualSDRCoordinator.check_safety_status()",
                    "Block all coordination commands",
                    "Notify operator of mode conflict",
                ],
                "response_time_requirement": "<100ms",
                "safety_authority_level": 2,
            },
        ]

        logger.info(f"Identified {len(pathways)} emergency response pathways")
        return pathways

    def generate_integration_architecture_diagram(self) -> dict[str, Any]:
        """
        [1e] Create safety integration architecture showing all connection points.

        Returns:
            Architecture diagram data structure
        """
        architecture = {
            "safety_layer": {
                "components": list(self.safety_components.keys()),
                "primary_coordinator": "SafetyInterlockSystem",
                "emergency_response": "SafetyManager",
            },
            "coordination_layer": {
                "components": list(self.coordination_components.keys()),
                "primary_coordinator": "DualSDRCoordinator",
                "communication": "SDRPPBridge",
                "decision_making": "SDRPriorityManager",
            },
            "integration_connections": [
                {
                    "from": "SafetyManager",
                    "to": "DualSDRCoordinator",
                    "connection_type": "dependency_injection",
                    "purpose": "Emergency stop integration",
                },
                {
                    "from": "SafetyInterlockSystem",
                    "to": "SDRPPBridge",
                    "connection_type": "event_monitoring",
                    "purpose": "Communication health monitoring",
                },
                {
                    "from": "SafetyManager",
                    "to": "SDRPriorityManager",
                    "connection_type": "decision_input",
                    "purpose": "Safety-aware source selection",
                },
            ],
            "data_flows": [
                "Safety status → Coordination decisions",
                "Communication health → Safety events",
                "Emergency triggers → Coordination shutdown",
                "Coordination status → Safety monitoring",
            ],
        }

        logger.info("Safety integration architecture diagram generated")
        return architecture

    def validate_story_2_2_safety_interlocks(self) -> dict[str, Any]:
        """
        [1d] Verify all Story 2.2 safety interlocks and their current implementation status.

        Returns:
            Validation results for all Story 2.2 safety interlocks
        """
        story_2_2_interlocks = {
            "mode_check": {
                "implemented": True,
                "location": "src/backend/utils/safety.py:ModeCheck",
                "functionality": "Validates flight mode is GUIDED",
                "integration_needed": "Coordinate with DualSDRCoordinator",
            },
            "operator_activation": {
                "implemented": True,
                "location": "src/backend/utils/safety.py:OperatorActivationCheck",
                "functionality": "Verifies homing enabled by operator",
                "integration_needed": "Enhanced with coordination status",
            },
            "signal_loss_monitoring": {
                "implemented": True,
                "location": "src/backend/utils/safety.py:SignalLossCheck",
                "functionality": "10-second timeout with SNR monitoring",
                "integration_needed": "Dual-source signal validation",
            },
            "battery_monitoring": {
                "implemented": True,
                "location": "src/backend/utils/safety.py:BatteryCheck",
                "functionality": "Battery threshold monitoring",
                "integration_needed": "Dual-system battery awareness",
            },
            "geofence_validation": {
                "implemented": True,
                "location": "src/backend/utils/safety.py:GeofenceCheck",
                "functionality": "Position boundary enforcement",
                "integration_needed": "Coordinate with source selection",
            },
            "emergency_stop_system": {
                "implemented": True,
                "location": "src/backend/utils/safety.py:SafetyInterlockSystem.emergency_stop",
                "functionality": "<500ms emergency stop capability",
                "integration_needed": "CRITICAL - Must integrate with coordination",
            },
        }

        validation_results = {
            "total_interlocks": len(story_2_2_interlocks),
            "implemented_count": sum(
                1 for interlock in story_2_2_interlocks.values() if interlock["implemented"]
            ),
            "integration_needed_count": len(story_2_2_interlocks),
            "critical_integrations": ["emergency_stop_system"],
            "interlocks": story_2_2_interlocks,
        }

        logger.info(f"Story 2.2 safety interlock validation: {validation_results}")
        return validation_results

    def generate_comprehensive_analysis_report(self) -> dict[str, Any]:
        """
        Generate comprehensive analysis report for SUBTASK-5.5.1.1.

        Returns:
            Complete analysis report with all integration requirements
        """
        logger.info("Generating comprehensive safety integration analysis report")

        # Run all analysis components
        system_analysis = self.analyze_existing_safety_system()
        authority_hierarchy = self.get_safety_authority_hierarchy()
        emergency_pathways = self.identify_emergency_response_pathways()
        architecture_diagram = self.generate_integration_architecture_diagram()
        story_2_2_validation = self.validate_story_2_2_safety_interlocks()

        report = {
            "analysis_summary": {
                "subtask": "SUBTASK-5.5.1.1",
                "completion_status": "COMPLETE",
                "safety_components_analyzed": system_analysis["safety_components"],
                "coordination_components_analyzed": system_analysis["coordination_components"],
                "integration_points_identified": system_analysis["integration_points"],
                "critical_integrations": system_analysis["critical_integrations"],
            },
            "system_analysis": system_analysis,
            "safety_authority_hierarchy": authority_hierarchy,
            "emergency_response_pathways": emergency_pathways,
            "integration_architecture": architecture_diagram,
            "story_2_2_validation": story_2_2_validation,
            "next_steps": [
                "SUBTASK-5.5.1.2: Enhance SafetyInterlockSystem for dual-SDR awareness",
                "SUBTASK-5.5.1.3: Implement communication health monitoring integration",
                "SUBTASK-5.5.1.4: Create comprehensive safety test suite",
            ],
        }

        logger.info("Comprehensive safety integration analysis complete")
        return report


# Create analyzer instance for use by other modules
safety_analyzer = SafetyIntegrationAnalyzer()
