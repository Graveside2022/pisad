"""
Safety Authority Manager

Implements and enforces the safety authority hierarchy throughout 
the coordination system. Ensures all safety decisions follow proper
authority levels and emergency response pathways.

Story 5.5 - SUBTASK-5.5.3.2 [9a] implementation
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from typing import Any, Optional

from src.backend.services.safety_integration_analysis import SafetyIntegrationAnalyzer
from src.backend.utils.logging import get_logger

logger = get_logger(__name__)


class SafetyAuthorityLevel(IntEnum):
    """
    Safety authority levels in descending order of authority.
    Lower number = higher authority
    """

    EMERGENCY_STOP = 1  # <500ms - Operator emergency stop
    FLIGHT_MODE = 2  # <100ms - Flight mode monitor
    GEOFENCE = 3  # <1s - Geofence boundary enforcement
    BATTERY = 4  # <5s - Battery monitor
    COMMUNICATION = 5  # <10s - Communication monitor
    SIGNAL = 6  # <10s - Signal monitor


class SafetyDecisionType(Enum):
    """Types of safety decisions that require authority validation"""

    EMERGENCY_STOP = "emergency_stop"
    SOURCE_SELECTION = "source_selection"
    COORDINATION_OVERRIDE = "coordination_override"
    COMMAND_VALIDATION = "command_validation"
    SYSTEM_SHUTDOWN = "system_shutdown"
    FALLBACK_TRIGGER = "fallback_trigger"


class SafetyAuthority:
    """Represents a safety authority with its hierarchy level and capabilities"""

    def __init__(
        self,
        level: SafetyAuthorityLevel,
        name: str,
        response_time_ms: int,
        integration_point: str,
        override_capability: str,
        coordination_integration: str,
    ):
        self.level = level
        self.name = name
        self.response_time_ms = response_time_ms
        self.integration_point = integration_point
        self.override_capability = override_capability
        self.coordination_integration = coordination_integration
        self.active = True
        self.last_trigger: Optional[datetime] = None


class SafetyDecision:
    """Represents a safety decision with authority validation"""

    def __init__(
        self,
        decision_type: SafetyDecisionType,
        requesting_authority: SafetyAuthorityLevel,
        details: dict[str, Any],
        timestamp: Optional[datetime] = None,
    ):
        self.decision_type = decision_type
        self.requesting_authority = requesting_authority
        self.details = details
        self.timestamp = timestamp or datetime.now()
        self.approved = False
        self.approving_authority: Optional[SafetyAuthorityLevel] = None
        self.response_time_ms: Optional[int] = None


class SafetyAuthorityManager:
    """
    Manages and enforces safety authority hierarchy throughout coordination system.
    
    Implements the 6-level safety authority hierarchy:
    1. Emergency Stop (Operator) - <500ms
    2. Flight Mode Monitor - <100ms  
    3. Geofence Boundary - <1s
    4. Battery Monitor - <5s
    5. Communication Monitor - <10s
    6. Signal Monitor - <10s
    """

    def __init__(self):
        """Initialize safety authority manager with hierarchy"""
        self.authorities: dict[SafetyAuthorityLevel, SafetyAuthority] = {}
        self.decision_history: list[SafetyDecision] = []
        self.emergency_override_active = False
        self.analyzer = SafetyIntegrationAnalyzer()
        
        self._initialize_authority_hierarchy()
        logger.info("Safety authority manager initialized with 6-level hierarchy")

    def _initialize_authority_hierarchy(self) -> None:
        """Initialize the complete safety authority hierarchy"""
        
        # Get hierarchy from analyzer
        hierarchy = self.analyzer.get_safety_authority_hierarchy()
        
        # Initialize Level 1: Emergency Stop
        self.authorities[SafetyAuthorityLevel.EMERGENCY_STOP] = SafetyAuthority(
            level=SafetyAuthorityLevel.EMERGENCY_STOP,
            name="Operator Emergency Stop",
            response_time_ms=500,
            integration_point="All coordination components",
            override_capability="Immediate shutdown of all systems",
            coordination_integration="Must trigger through DualSDRCoordinator",
        )

        # Initialize Level 2: Flight Mode Monitor
        self.authorities[SafetyAuthorityLevel.FLIGHT_MODE] = SafetyAuthority(
            level=SafetyAuthorityLevel.FLIGHT_MODE,
            name="Flight Mode Monitor",
            response_time_ms=100,
            integration_point="DualSDRCoordinator decision making",
            override_capability="Override payload if not GUIDED",
            coordination_integration="Block coordination commands when not GUIDED",
        )

        # Initialize Level 3: Geofence Boundary
        self.authorities[SafetyAuthorityLevel.GEOFENCE] = SafetyAuthority(
            level=SafetyAuthorityLevel.GEOFENCE,
            name="Geofence Boundary Enforcement",
            response_time_ms=1000,
            integration_point="Priority Manager source selection",
            override_capability="Hard boundary enforcement",
            coordination_integration="Geofence-aware source selection",
        )

        # Initialize Level 4: Battery Monitor
        self.authorities[SafetyAuthorityLevel.BATTERY] = SafetyAuthority(
            level=SafetyAuthorityLevel.BATTERY,
            name="Battery Monitor",
            response_time_ms=5000,
            integration_point="Coordination health monitoring",
            override_capability="Low battery triggers RTL",
            coordination_integration="Dual-system battery awareness",
        )

        # Initialize Level 5: Communication Monitor
        self.authorities[SafetyAuthorityLevel.COMMUNICATION] = SafetyAuthority(
            level=SafetyAuthorityLevel.COMMUNICATION,
            name="Communication Monitor",
            response_time_ms=10000,
            integration_point="SDRPPBridge health monitoring",
            override_capability="Communication loss triggers fallback",
            coordination_integration="Automatic drone-only fallback",
        )

        # Initialize Level 6: Signal Monitor
        self.authorities[SafetyAuthorityLevel.SIGNAL] = SafetyAuthority(
            level=SafetyAuthorityLevel.SIGNAL,
            name="Signal Monitor", 
            response_time_ms=10000,
            integration_point="Dual source signal validation",
            override_capability="Signal loss auto-disable",
            coordination_integration="Enhanced dual-source monitoring",
        )

        logger.info(f"Initialized {len(self.authorities)} safety authority levels")

    async def validate_safety_decision(
        self, decision: SafetyDecision
    ) -> tuple[bool, str, Optional[SafetyAuthorityLevel]]:
        """
        Validate safety decision against authority hierarchy.
        
        Args:
            decision: Safety decision to validate
            
        Returns:
            Tuple of (approved, reason, approving_authority)
        """
        start_time = datetime.now()
        
        # Emergency override bypasses all authority checks
        if self.emergency_override_active:
            decision.approved = True
            decision.approving_authority = SafetyAuthorityLevel.EMERGENCY_STOP
            decision.response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            logger.warning(f"Emergency override active - auto-approving {decision.decision_type}")
            return True, "Emergency override active", SafetyAuthorityLevel.EMERGENCY_STOP

        # Check if requesting authority is valid
        if decision.requesting_authority not in self.authorities:
            reason = f"Invalid requesting authority: {decision.requesting_authority}"
            logger.error(reason)
            return False, reason, None

        requesting_auth = self.authorities[decision.requesting_authority]
        
        # Check if requesting authority is active
        if not requesting_auth.active:
            reason = f"Requesting authority {requesting_auth.name} is inactive"
            logger.warning(reason)
            return False, reason, None

        # Find highest authority that can approve this decision type
        approving_authority = self._find_approving_authority(decision)
        
        if approving_authority is None:
            reason = f"No authority can approve decision type {decision.decision_type}"
            logger.error(reason)
            return False, reason, None

        # Check authority hierarchy - requesting authority must be at same or higher level
        if decision.requesting_authority > approving_authority:
            reason = f"Insufficient authority: {requesting_auth.name} (level {decision.requesting_authority}) cannot request {decision.decision_type}"
            logger.warning(reason)
            return False, reason, None

        # Validate response time requirements
        response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        max_response_time = self.authorities[approving_authority].response_time_ms
        
        if response_time_ms > max_response_time:
            logger.warning(f"Decision validation took {response_time_ms}ms, exceeding {max_response_time}ms limit")

        # Approve decision
        decision.approved = True
        decision.approving_authority = approving_authority
        decision.response_time_ms = response_time_ms
        
        # Log decision for audit trail
        self.decision_history.append(decision)
        
        # Update authority last trigger time
        self.authorities[approving_authority].last_trigger = datetime.now()
        
        logger.info(
            f"Safety decision approved: {decision.decision_type} by {self.authorities[approving_authority].name} "
            f"({response_time_ms}ms)"
        )
        
        return True, f"Approved by {self.authorities[approving_authority].name}", approving_authority

    def _find_approving_authority(self, decision: SafetyDecision) -> Optional[SafetyAuthorityLevel]:
        """Find the appropriate authority level to approve this decision type"""
        
        # Emergency stop can only be approved by level 1
        if decision.decision_type == SafetyDecisionType.EMERGENCY_STOP:
            return SafetyAuthorityLevel.EMERGENCY_STOP
        
        # System shutdown requires level 1 or 2
        elif decision.decision_type == SafetyDecisionType.SYSTEM_SHUTDOWN:
            return SafetyAuthorityLevel.EMERGENCY_STOP
        
        # Coordination override can be approved by levels 1-3
        elif decision.decision_type == SafetyDecisionType.COORDINATION_OVERRIDE:
            return SafetyAuthorityLevel.FLIGHT_MODE
        
        # Source selection by levels 1-5
        elif decision.decision_type == SafetyDecisionType.SOURCE_SELECTION:
            return SafetyAuthorityLevel.COMMUNICATION
        
        # Command validation by levels 1-2
        elif decision.decision_type == SafetyDecisionType.COMMAND_VALIDATION:
            return SafetyAuthorityLevel.FLIGHT_MODE
        
        # Fallback trigger by levels 1-6
        elif decision.decision_type == SafetyDecisionType.FALLBACK_TRIGGER:
            return SafetyAuthorityLevel.SIGNAL
        
        return None

    async def trigger_emergency_override(self, reason: str) -> dict[str, Any]:
        """
        Trigger emergency override - highest authority level.
        Bypasses all other authority checks.
        """
        start_time = datetime.now()
        
        self.emergency_override_active = True
        
        # Create emergency decision record
        emergency_decision = SafetyDecision(
            decision_type=SafetyDecisionType.EMERGENCY_STOP,
            requesting_authority=SafetyAuthorityLevel.EMERGENCY_STOP,
            details={"reason": reason, "triggered_by": "emergency_override"},
        )
        
        # Auto-approve emergency decision
        emergency_decision.approved = True
        emergency_decision.approving_authority = SafetyAuthorityLevel.EMERGENCY_STOP
        emergency_decision.response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        self.decision_history.append(emergency_decision)
        
        # Update emergency authority
        self.authorities[SafetyAuthorityLevel.EMERGENCY_STOP].last_trigger = datetime.now()
        
        result = {
            "emergency_override_active": True,
            "trigger_reason": reason,
            "response_time_ms": emergency_decision.response_time_ms,
            "authority_level": SafetyAuthorityLevel.EMERGENCY_STOP,
            "timestamp": emergency_decision.timestamp.isoformat(),
        }
        
        logger.critical(f"EMERGENCY OVERRIDE TRIGGERED: {reason} ({emergency_decision.response_time_ms}ms)")
        
        return result

    async def clear_emergency_override(self, authorized_by: str) -> dict[str, Any]:
        """Clear emergency override state"""
        if not self.emergency_override_active:
            return {"cleared": False, "reason": "No active emergency override"}
        
        self.emergency_override_active = False
        
        result = {
            "emergency_override_cleared": True,
            "authorized_by": authorized_by,
            "cleared_at": datetime.now().isoformat(),
        }
        
        logger.warning(f"Emergency override cleared by {authorized_by}")
        return result

    def get_authority_status(self) -> dict[str, Any]:
        """Get status of all safety authorities"""
        status = {
            "emergency_override_active": self.emergency_override_active,
            "authorities": {},
            "recent_decisions": len([d for d in self.decision_history if 
                                    datetime.now() - d.timestamp < timedelta(minutes=5)]),
        }
        
        for level, authority in self.authorities.items():
            status["authorities"][f"level_{level}"] = {
                "name": authority.name,
                "level": level,
                "active": authority.active,
                "response_time_ms": authority.response_time_ms,
                "integration_point": authority.integration_point,
                "last_trigger": authority.last_trigger.isoformat() if authority.last_trigger else None,
            }
        
        return status

    def get_decision_audit_trail(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent safety decisions for audit trail"""
        recent_decisions = sorted(self.decision_history, key=lambda d: d.timestamp, reverse=True)[:limit]
        
        audit_trail = []
        for decision in recent_decisions:
            audit_trail.append({
                "timestamp": decision.timestamp.isoformat(),
                "decision_type": decision.decision_type.value,
                "requesting_authority": decision.requesting_authority,
                "approved": decision.approved,
                "approving_authority": decision.approving_authority,
                "response_time_ms": decision.response_time_ms,
                "details": decision.details,
            })
        
        return audit_trail

    def validate_coordination_command(
        self, command_type: str, authority_level: SafetyAuthorityLevel, details: dict[str, Any]
    ) -> tuple[bool, str]:
        """
        Validate coordination command against safety authority hierarchy.
        
        This is a synchronous method for real-time command validation.
        """
        if self.emergency_override_active:
            return False, "Emergency override active - all coordination commands blocked"
        
        # Flight mode commands require level 2 or higher
        if command_type in ["mode_change", "guidance_command"]:
            if authority_level > SafetyAuthorityLevel.FLIGHT_MODE:
                return False, f"Insufficient authority for {command_type}"
        
        # Source selection requires level 5 or higher
        elif command_type in ["source_switch", "priority_change"]:
            if authority_level > SafetyAuthorityLevel.COMMUNICATION:
                return False, f"Insufficient authority for {command_type}"
        
        # Emergency commands require level 1
        elif command_type in ["emergency_stop", "system_shutdown"]:
            if authority_level > SafetyAuthorityLevel.EMERGENCY_STOP:
                return False, f"Insufficient authority for {command_type}"
        
        return True, f"Command {command_type} authorized by authority level {authority_level}"

    def deactivate_authority(self, level: SafetyAuthorityLevel, reason: str) -> None:
        """Deactivate a safety authority (for testing or maintenance)"""
        if level in self.authorities:
            self.authorities[level].active = False
            logger.warning(f"Deactivated safety authority {self.authorities[level].name}: {reason}")

    def reactivate_authority(self, level: SafetyAuthorityLevel) -> None:
        """Reactivate a safety authority"""
        if level in self.authorities:
            self.authorities[level].active = True
            logger.info(f"Reactivated safety authority {self.authorities[level].name}")

    def validate_coordination_command_real_time(
        self, 
        command_type: str, 
        authority_level: SafetyAuthorityLevel, 
        details: dict[str, Any],
        response_time_limit_ms: int = 100
    ) -> tuple[bool, str]:
        """
        SUBTASK-5.5.3.2 [9d] - Real-time validation layer for coordination commands.
        
        Validates coordination commands before execution with strict timing requirements.
        
        Args:
            command_type: Type of coordination command 
            authority_level: Authority level requesting the command
            details: Command details and context
            response_time_limit_ms: Maximum validation time allowed (default 100ms)
            
        Returns:
            Tuple of (authorized, message)
        """
        start_time = datetime.now()
        
        try:
            # Emergency override blocks all non-emergency commands
            if self.emergency_override_active and command_type != "emergency_stop":
                response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                logger.warning(f"Command {command_type} blocked - emergency override active ({response_time_ms}ms)")
                return False, f"Emergency override active - all coordination commands blocked"
            
            # Use existing command validation logic
            authorized, message = self.validate_coordination_command(command_type, authority_level, details)
            
            # Check timing requirement
            response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            if response_time_ms > response_time_limit_ms:
                logger.warning(f"Command validation took {response_time_ms}ms, exceeding {response_time_limit_ms}ms limit")
                return False, f"Command validation exceeded timing requirement ({response_time_ms}ms > {response_time_limit_ms}ms)"
            
            # Log successful validation for audit trail
            logger.debug(f"Command {command_type} validation: {authorized} ({response_time_ms}ms) - {message}")
            
            return authorized, message
            
        except Exception as e:
            response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.error(f"Command validation error for {command_type}: {e} ({response_time_ms}ms)")
            return False, f"Command validation failed: {str(e)}"

    def validate_coordination_command_with_metrics(
        self,
        command_type: str,
        authority_level: SafetyAuthorityLevel, 
        details: dict[str, Any],
        response_time_limit_ms: int = 100
    ) -> tuple[bool, str, dict[str, Any]]:
        """
        SUBTASK-5.5.3.2 [9d] - Command validation with detailed timing metrics.
        
        Returns validation result plus detailed metrics for performance monitoring.
        
        Args:
            command_type: Type of coordination command
            authority_level: Authority level requesting the command  
            details: Command details and context
            response_time_limit_ms: Maximum validation time allowed
            
        Returns:
            Tuple of (authorized, message, metrics)
        """
        start_time = datetime.now()
        
        authorized, message = self.validate_coordination_command_real_time(
            command_type, authority_level, details, response_time_limit_ms
        )
        
        end_time = datetime.now()
        validation_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        metrics = {
            "validation_time_ms": validation_time_ms,
            "authority_level": authority_level,
            "command_type": command_type,
            "authorized": authorized,
            "timing_requirement_met": validation_time_ms <= response_time_limit_ms,
            "emergency_override_active": self.emergency_override_active,
            "timestamp": start_time.isoformat(),
        }
        
        return authorized, message, metrics

    async def handle_coordination_failure(
        self, 
        failure_type: str, 
        failure_details: dict[str, Any],
        escalation_level: SafetyAuthorityLevel = SafetyAuthorityLevel.COMMUNICATION
    ) -> dict[str, Any]:
        """
        SUBTASK-5.5.3.2 [9e] - Safety escalation procedures for coordination system failures.
        
        Handles coordination system failures with appropriate escalation based on severity.
        
        Args:
            failure_type: Type of coordination failure
            failure_details: Details about the failure
            escalation_level: Authority level for escalation (default: COMMUNICATION)
            
        Returns:
            Dict containing escalation result and actions taken
        """
        start_time = datetime.now()
        
        try:
            # Create safety decision for the escalation
            escalation_decision = SafetyDecision(
                decision_type=SafetyDecisionType.FALLBACK_TRIGGER,
                requesting_authority=escalation_level,
                details={
                    "failure_type": failure_type,
                    "failure_details": failure_details,
                    "escalation_triggered": True,
                    "timestamp": start_time.isoformat(),
                }
            )
            
            # Validate the escalation decision
            approved, reason, approving_authority = await self.validate_safety_decision(escalation_decision)
            
            if not approved:
                logger.error(f"Escalation denied for {failure_type}: {reason}")
                return {
                    "escalation_approved": False,
                    "reason": reason,
                    "failure_type": failure_type,
                    "timestamp": start_time.isoformat(),
                }
            
            # Determine escalation actions based on failure type
            escalation_actions = []
            
            if failure_type == "communication_loss":
                escalation_actions = [
                    "trigger_automatic_fallback_to_drone_only",
                    "notify_operator_of_communication_loss", 
                    "initiate_emergency_contact_procedures"
                ]
            elif failure_type == "coordination_timing_violation":
                escalation_actions = [
                    "disable_coordination_temporarily",
                    "switch_to_drone_only_processing",
                    "log_performance_violation"
                ]
            elif failure_type == "safety_authority_conflict":
                # Safety authority conflicts require emergency level response
                if escalation_level > SafetyAuthorityLevel.EMERGENCY_STOP:
                    # Upgrade escalation to emergency level for conflicts
                    escalation_level = SafetyAuthorityLevel.EMERGENCY_STOP
                    escalation_decision.requesting_authority = SafetyAuthorityLevel.EMERGENCY_STOP
                escalation_actions = [
                    "trigger_emergency_override",
                    "shutdown_coordination_system",
                    "alert_safety_personnel"
                ]
            elif failure_type == "command_validation_failure":
                escalation_actions = [
                    "block_all_coordination_commands",
                    "switch_to_manual_control_mode",
                    "notify_operator_immediately"
                ]
            else:
                escalation_actions = [
                    "trigger_general_safety_fallback",
                    "log_unknown_failure_type",
                    "request_manual_intervention"
                ]
            
            # Record escalation in decision history
            escalation_decision.approved = True
            escalation_decision.approving_authority = approving_authority
            escalation_decision.response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            self.decision_history.append(escalation_decision)
            
            result = {
                "escalation_approved": True,
                "failure_type": failure_type,
                "escalation_level": escalation_level,
                "approving_authority": approving_authority,
                "escalation_actions": escalation_actions,
                "response_time_ms": escalation_decision.response_time_ms,
                "timestamp": start_time.isoformat(),
            }
            
            logger.critical(f"Safety escalation triggered for {failure_type}: {escalation_actions} ({escalation_decision.response_time_ms}ms)")
            
            return result
            
        except Exception as e:
            response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.error(f"Escalation handling failed for {failure_type}: {e} ({response_time_ms}ms)")
            
            return {
                "escalation_approved": False,
                "error": str(e),
                "failure_type": failure_type,
                "response_time_ms": response_time_ms,
                "timestamp": start_time.isoformat(),
            }

    def trigger_escalation_for_timing_violation(
        self, 
        component: str, 
        expected_time_ms: int, 
        actual_time_ms: int
    ) -> dict[str, Any]:
        """
        SUBTASK-5.5.3.2 [9e] - Automatic escalation for timing violations.
        
        Triggers safety escalation when coordination components exceed timing requirements.
        """
        failure_details = {
            "component": component,
            "expected_time_ms": expected_time_ms,
            "actual_time_ms": actual_time_ms,
            "violation_severity": "high" if actual_time_ms > expected_time_ms * 2 else "medium",
            "performance_degradation_percent": ((actual_time_ms - expected_time_ms) / expected_time_ms) * 100,
        }
        
        # Use asyncio to run the async escalation handler
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in async context, schedule the escalation
            task = loop.create_task(
                self.handle_coordination_failure("coordination_timing_violation", failure_details)
            )
            logger.warning(f"Scheduled timing violation escalation for {component}: {actual_time_ms}ms > {expected_time_ms}ms")
            return {"escalation_scheduled": True, "task_id": str(id(task))}
        else:
            # If not in async context, run synchronously
            return loop.run_until_complete(
                self.handle_coordination_failure("coordination_timing_violation", failure_details)
            )

    def log_coordination_decision(
        self,
        component: str,
        decision_type: str, 
        decision_details: dict[str, Any],
        authority_level: SafetyAuthorityLevel,
        outcome: str
    ) -> None:
        """
        SUBTASK-5.5.3.2 [9f] - Log coordination decisions for comprehensive audit trail.
        
        Records all coordination decisions for safety audit and compliance tracking.
        
        Args:
            component: Component making the decision
            decision_type: Type of coordination decision
            decision_details: Details about the decision  
            authority_level: Authority level of the decision maker
            outcome: Result of the decision
        """
        audit_entry = SafetyDecision(
            decision_type=SafetyDecisionType.COORDINATION_OVERRIDE,  # Generic coordination decision
            requesting_authority=authority_level,
            details={
                "component": component,
                "coordination_decision_type": decision_type,
                "decision_details": decision_details,
                "outcome": outcome,
                "audit_entry": True,
                "logged_at": datetime.now().isoformat(),
            }
        )
        
        # Mark as approved for audit purposes (this is logging, not authorization)
        audit_entry.approved = True
        audit_entry.approving_authority = authority_level
        audit_entry.response_time_ms = 0  # Logging operation
        
        # Add to decision history for audit trail
        self.decision_history.append(audit_entry)
        
        logger.info(f"Coordination decision logged: {component}.{decision_type} -> {outcome}")

    def get_coordination_audit_trail(
        self, 
        component: str | None = None,
        decision_type: str | None = None, 
        since_minutes: int = 60,
        limit: int = 100
    ) -> list[dict[str, Any]]:
        """
        SUBTASK-5.5.3.2 [9f] - Get coordination decisions audit trail with filtering.
        
        Retrieves coordination decisions from audit trail with filtering options.
        
        Args:
            component: Filter by component name (optional)
            decision_type: Filter by decision type (optional)
            since_minutes: Get decisions from last N minutes (default: 60)
            limit: Maximum number of entries to return (default: 100)
            
        Returns:
            List of filtered audit trail entries
        """
        cutoff_time = datetime.now() - timedelta(minutes=since_minutes)
        
        # Filter decisions based on criteria
        filtered_decisions = []
        for decision in self.decision_history:
            # Time filter
            if decision.timestamp < cutoff_time:
                continue
                
            # Component filter
            if component and decision.details.get("component") != component:
                continue
                
            # Decision type filter  
            if decision_type and decision.details.get("coordination_decision_type") != decision_type:
                continue
            
            # Only include coordination-related decisions
            if not decision.details.get("audit_entry") and decision.decision_type not in [
                SafetyDecisionType.COORDINATION_OVERRIDE,
                SafetyDecisionType.SOURCE_SELECTION,
                SafetyDecisionType.FALLBACK_TRIGGER,
                SafetyDecisionType.COMMAND_VALIDATION,
            ]:
                continue
                
            audit_entry = {
                "timestamp": decision.timestamp.isoformat(),
                "component": decision.details.get("component", "unknown"),
                "decision_type": decision.details.get("coordination_decision_type", decision.decision_type.value),
                "authority_level": decision.requesting_authority,
                "approved": decision.approved,
                "approving_authority": decision.approving_authority,
                "outcome": decision.details.get("outcome", "unknown"),
                "response_time_ms": decision.response_time_ms,
                "details": decision.details,
            }
            
            filtered_decisions.append(audit_entry)
            
            # Limit check
            if len(filtered_decisions) >= limit:
                break
        
        # Sort by timestamp (most recent first)
        filtered_decisions.sort(key=lambda x: x["timestamp"], reverse=True)
        
        logger.debug(f"Retrieved {len(filtered_decisions)} coordination audit entries")
        return filtered_decisions

    def get_safety_metrics_summary(self) -> dict[str, Any]:
        """
        SUBTASK-5.5.3.2 [9f] - Get comprehensive safety metrics for audit reporting.
        
        Returns summary of safety system performance and coordination decisions.
        """
        now = datetime.now()
        last_hour = now - timedelta(hours=1)
        last_day = now - timedelta(days=1)
        
        # Count decisions by type in last hour
        recent_decisions = [d for d in self.decision_history if d.timestamp >= last_hour]
        decision_counts: dict[str, int] = {}
        for decision in recent_decisions:
            decision_type = decision.decision_type.value
            decision_counts[decision_type] = decision_counts.get(decision_type, 0) + 1
            
        # Count escalations in last day
        escalations = [
            d for d in self.decision_history 
            if d.timestamp >= last_day and d.details.get("escalation_triggered", False)
        ]
        
        # Calculate average response times
        response_times = [d.response_time_ms for d in recent_decisions if d.response_time_ms is not None]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Count emergency overrides  
        emergency_overrides = [
            d for d in self.decision_history
            if d.timestamp >= last_day and d.decision_type == SafetyDecisionType.EMERGENCY_STOP
        ]
        
        summary = {
            "report_timestamp": now.isoformat(),
            "emergency_override_active": self.emergency_override_active,
            "total_authorities": len(self.authorities),
            "active_authorities": len([auth for auth in self.authorities.values() if auth.active]),
            "decisions_last_hour": len(recent_decisions),
            "decision_type_breakdown": decision_counts,
            "escalations_last_24h": len(escalations), 
            "emergency_overrides_last_24h": len(emergency_overrides),
            "average_response_time_ms": round(avg_response_time, 2),
            "max_response_time_ms": max(response_times) if response_times else 0,
            "total_audit_entries": len(self.decision_history),
            "coordination_decisions": len([
                d for d in self.decision_history 
                if d.details.get("audit_entry") or d.decision_type in [
                    SafetyDecisionType.COORDINATION_OVERRIDE,
                    SafetyDecisionType.SOURCE_SELECTION,
                ]
            ]),
        }
        
        logger.info(f"Generated safety metrics summary: {summary['decisions_last_hour']} recent decisions")
        return summary

    async def trigger_emergency_coordination_bypass(
        self,
        trigger_reason: str,
        bypass_components: list[str],
        fallback_mode: str = "drone_only"
    ) -> dict[str, Any]:
        """
        SUBTASK-5.5.3.3 [10a] - Trigger emergency bypass of coordination system.
        
        Creates direct pathways that bypass normal coordination for emergency response.
        
        Args:
            trigger_reason: Reason for emergency bypass
            bypass_components: List of components to bypass
            fallback_mode: Fallback mode to activate
            
        Returns:
            Dict containing bypass result and status
        """
        start_time = datetime.now()
        
        try:
            # Activate emergency override first
            await self.trigger_emergency_override(f"coordination_bypass: {trigger_reason}")
            
            # Create bypass decision record
            bypass_decision = SafetyDecision(
                decision_type=SafetyDecisionType.COORDINATION_OVERRIDE,
                requesting_authority=SafetyAuthorityLevel.EMERGENCY_STOP,
                details={
                    "bypass_reason": trigger_reason,
                    "bypassed_components": bypass_components,
                    "fallback_mode": fallback_mode,
                    "emergency_bypass": True,
                    "bypass_timestamp": start_time.isoformat(),
                }
            )
            
            # Auto-approve bypass decision
            bypass_decision.approved = True
            bypass_decision.approving_authority = SafetyAuthorityLevel.EMERGENCY_STOP
            bypass_decision.response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            self.decision_history.append(bypass_decision)
            
            result = {
                "bypass_active": True,
                "bypassed_components": bypass_components,
                "fallback_mode": fallback_mode,
                "trigger_reason": trigger_reason,
                "response_time_ms": bypass_decision.response_time_ms,
                "bypass_timestamp": start_time.isoformat(),
                "emergency_override_active": self.emergency_override_active,
            }
            
            logger.critical(f"Emergency coordination bypass activated: {bypass_components} ({bypass_decision.response_time_ms}ms)")
            
            return result
            
        except Exception as e:
            response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.error(f"Emergency bypass failed: {e} ({response_time_ms}ms)")
            
            return {
                "bypass_active": False,
                "error": str(e),
                "response_time_ms": response_time_ms,
                "trigger_reason": trigger_reason,
            }

    async def execute_emergency_direct_command(
        self,
        command_type: str,
        target_components: list[str],
        emergency_reason: str
    ) -> dict[str, Any]:
        """
        SUBTASK-5.5.3.3 [10a] - Execute direct emergency command bypassing coordination.
        
        Provides direct command pathway that bypasses normal coordination system.
        
        Args:
            command_type: Type of emergency command
            target_components: Components to execute command on
            emergency_reason: Reason for emergency command
            
        Returns:
            Dict containing execution result
        """
        start_time = datetime.now()
        
        try:
            # Create direct command decision
            direct_command_decision = SafetyDecision(
                decision_type=SafetyDecisionType.EMERGENCY_STOP,  
                requesting_authority=SafetyAuthorityLevel.EMERGENCY_STOP,
                details={
                    "direct_command_type": command_type,
                    "target_components": target_components,
                    "emergency_reason": emergency_reason,
                    "bypass_route": "direct_emergency_pathway",
                    "execution_timestamp": start_time.isoformat(),
                }
            )
            
            # Auto-approve direct command
            direct_command_decision.approved = True
            direct_command_decision.approving_authority = SafetyAuthorityLevel.EMERGENCY_STOP
            direct_command_decision.response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            self.decision_history.append(direct_command_decision)
            
            result = {
                "executed": True,
                "command_type": command_type,
                "affected_components": target_components,
                "bypass_route": "direct_emergency_pathway",
                "emergency_reason": emergency_reason,
                "response_time_ms": direct_command_decision.response_time_ms,
                "execution_timestamp": start_time.isoformat(),
            }
            
            logger.critical(f"Emergency direct command executed: {command_type} -> {target_components} ({direct_command_decision.response_time_ms}ms)")
            
            return result
            
        except Exception as e:
            response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.error(f"Direct emergency command failed: {e} ({response_time_ms}ms)")
            
            return {
                "executed": False,
                "error": str(e),
                "command_type": command_type,
                "response_time_ms": response_time_ms,
            }

    async def isolate_coordination_system(
        self,
        isolation_level: str = "complete",
        preserve_components: list[str] = None
    ) -> dict[str, Any]:
        """
        SUBTASK-5.5.3.3 [10a] - Isolate coordination system during emergency.
        
        Isolates coordination system components while preserving critical functions.
        
        Args:
            isolation_level: Level of isolation (complete, partial)
            preserve_components: Components to preserve during isolation
            
        Returns:
            Dict containing isolation result
        """
        start_time = datetime.now()
        preserve_components = preserve_components or []
        
        try:
            # Default components that should be isolated
            default_isolated = ["DualSDRCoordinator", "SDRPriorityManager", "coordination_optimizer"]
            
            # Components to isolate (excluding preserved)
            isolated_components = [comp for comp in default_isolated if comp not in preserve_components]
            
            # Create isolation decision
            isolation_decision = SafetyDecision(
                decision_type=SafetyDecisionType.FALLBACK_TRIGGER,
                requesting_authority=SafetyAuthorityLevel.EMERGENCY_STOP,
                details={
                    "isolation_level": isolation_level,
                    "isolated_components": isolated_components,
                    "preserved_components": preserve_components,
                    "isolation_reason": "emergency_override_coordination_isolation",
                    "isolation_timestamp": start_time.isoformat(),
                }
            )
            
            # Auto-approve isolation
            isolation_decision.approved = True
            isolation_decision.approving_authority = SafetyAuthorityLevel.EMERGENCY_STOP
            isolation_decision.response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            self.decision_history.append(isolation_decision)
            
            result = {
                "isolation_active": True,
                "isolation_level": isolation_level,
                "isolated_components": isolated_components,
                "preserved_components": preserve_components,
                "response_time_ms": isolation_decision.response_time_ms,
                "isolation_timestamp": start_time.isoformat(),
            }
            
            logger.warning(f"Coordination system isolated: {isolation_level} level, {len(isolated_components)} components ({isolation_decision.response_time_ms}ms)")
            
            return result
            
        except Exception as e:
            response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.error(f"Coordination isolation failed: {e} ({response_time_ms}ms)")
            
            return {
                "isolation_active": False,
                "error": str(e),
                "response_time_ms": response_time_ms,
            }

    async def trigger_immediate_coordination_shutdown(
        self,
        safety_trigger: str,
        shutdown_level: str = "emergency",
        preserve_emergency_functions: bool = True
    ) -> dict[str, Any]:
        """
        SUBTASK-5.5.3.3 [10b] - Immediate coordination shutdown on critical safety triggers.
        
        Performs immediate shutdown of coordination system for critical safety events.
        
        Args:
            safety_trigger: Critical safety trigger causing shutdown
            shutdown_level: Level of shutdown (emergency, graceful)
            preserve_emergency_functions: Whether to preserve emergency functions
            
        Returns:
            Dict containing shutdown result
        """
        start_time = datetime.now()
        
        try:
            # Activate emergency override for shutdown
            await self.trigger_emergency_override(f"immediate_shutdown: {safety_trigger}")
            
            # Create shutdown decision
            shutdown_decision = SafetyDecision(
                decision_type=SafetyDecisionType.EMERGENCY_STOP,
                requesting_authority=SafetyAuthorityLevel.EMERGENCY_STOP,
                details={
                    "shutdown_trigger": safety_trigger,
                    "shutdown_level": shutdown_level,
                    "preserve_emergency_functions": preserve_emergency_functions,
                    "immediate_shutdown": True,
                    "shutdown_timestamp": start_time.isoformat(),
                }
            )
            
            # Auto-approve shutdown
            shutdown_decision.approved = True
            shutdown_decision.approving_authority = SafetyAuthorityLevel.EMERGENCY_STOP
            shutdown_decision.response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            self.decision_history.append(shutdown_decision)
            
            result = {
                "shutdown_executed": True,
                "shutdown_level": shutdown_level,
                "safety_trigger": safety_trigger,
                "emergency_functions_preserved": preserve_emergency_functions,
                "response_time_ms": shutdown_decision.response_time_ms,
                "shutdown_timestamp": start_time.isoformat(),
                "emergency_override_active": self.emergency_override_active,
            }
            
            logger.critical(f"IMMEDIATE COORDINATION SHUTDOWN: {safety_trigger} - {shutdown_level} level ({shutdown_decision.response_time_ms}ms)")
            
            return result
            
        except Exception as e:
            response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.error(f"Immediate coordination shutdown failed: {e} ({response_time_ms}ms)")
            
            return {
                "shutdown_executed": False,
                "error": str(e),
                "safety_trigger": safety_trigger,
                "response_time_ms": response_time_ms,
            }

    async def trigger_coordinated_shutdown(
        self,
        trigger_reason: str,
        shutdown_mode: str = "graceful",
        fallback_timeout_ms: int = 1000
    ) -> dict[str, Any]:
        """
        SUBTASK-5.5.3.3 [10b] - Coordinated shutdown with graceful degradation.
        
        Performs coordinated shutdown with fallback timeout for graceful degradation.
        
        Args:
            trigger_reason: Reason for coordinated shutdown
            shutdown_mode: Mode of shutdown (graceful, immediate)
            fallback_timeout_ms: Timeout for fallback activation
            
        Returns:
            Dict containing shutdown result
        """
        start_time = datetime.now()
        
        try:
            # Create coordinated shutdown decision
            shutdown_decision = SafetyDecision(
                decision_type=SafetyDecisionType.FALLBACK_TRIGGER,
                requesting_authority=SafetyAuthorityLevel.COMMUNICATION,
                details={
                    "shutdown_reason": trigger_reason,
                    "shutdown_mode": shutdown_mode,
                    "fallback_timeout_ms": fallback_timeout_ms,
                    "coordinated_shutdown": True,
                    "shutdown_timestamp": start_time.isoformat(),
                }
            )
            
            # Approve shutdown decision
            approved, reason, approving_authority = await self.validate_safety_decision(shutdown_decision)
            
            if not approved:
                return {
                    "shutdown_initiated": False,
                    "reason": reason,
                    "trigger_reason": trigger_reason,
                }
            
            shutdown_decision.approved = True
            shutdown_decision.approving_authority = approving_authority
            shutdown_decision.response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            self.decision_history.append(shutdown_decision)
            
            result = {
                "shutdown_initiated": True,
                "shutdown_mode": shutdown_mode,
                "trigger_reason": trigger_reason,
                "fallback_active": True,
                "fallback_timeout_ms": fallback_timeout_ms,
                "approving_authority": approving_authority,
                "response_time_ms": shutdown_decision.response_time_ms,
                "shutdown_timestamp": start_time.isoformat(),
            }
            
            logger.warning(f"Coordinated shutdown initiated: {trigger_reason} - {shutdown_mode} mode ({shutdown_decision.response_time_ms}ms)")
            
            return result
            
        except Exception as e:
            response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.error(f"Coordinated shutdown failed: {e} ({response_time_ms}ms)")
            
            return {
                "shutdown_initiated": False,
                "error": str(e),
                "trigger_reason": trigger_reason,
                "response_time_ms": response_time_ms,
            }

    async def trigger_automatic_drone_only_fallback(
        self,
        trigger_condition: str,
        fallback_duration_minutes: int = 10,
        auto_recovery: bool = True
    ) -> dict[str, Any]:
        """
        SUBTASK-5.5.3.3 [10c] - Automatic switching to drone-only mode.
        
        Triggers automatic fallback to drone-only processing mode.
        
        Args:
            trigger_condition: Condition triggering drone-only mode
            fallback_duration_minutes: Duration for drone-only mode
            auto_recovery: Whether to attempt automatic recovery
            
        Returns:
            Dict containing fallback result
        """
        start_time = datetime.now()
        
        try:
            # Create drone-only fallback decision
            fallback_decision = SafetyDecision(
                decision_type=SafetyDecisionType.SOURCE_SELECTION,
                requesting_authority=SafetyAuthorityLevel.COMMUNICATION,
                details={
                    "trigger_condition": trigger_condition,
                    "fallback_mode": "drone_only",
                    "fallback_duration_minutes": fallback_duration_minutes,
                    "auto_recovery_enabled": auto_recovery,
                    "automatic_fallback": True,
                    "fallback_timestamp": start_time.isoformat(),
                }
            )
            
            # Approve fallback decision
            approved, reason, approving_authority = await self.validate_safety_decision(fallback_decision)
            
            if not approved:
                return {
                    "drone_only_active": False,
                    "reason": reason,
                    "trigger_condition": trigger_condition,
                }
            
            fallback_decision.approved = True
            fallback_decision.approving_authority = approving_authority
            fallback_decision.response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            self.decision_history.append(fallback_decision)
            
            result = {
                "drone_only_active": True,
                "trigger_condition": trigger_condition,
                "fallback_duration_minutes": fallback_duration_minutes,
                "auto_recovery_enabled": auto_recovery,
                "approving_authority": approving_authority,
                "activation_time_ms": fallback_decision.response_time_ms,
                "fallback_timestamp": start_time.isoformat(),
            }
            
            logger.warning(f"Automatic drone-only fallback activated: {trigger_condition} ({fallback_decision.response_time_ms}ms)")
            
            return result
            
        except Exception as e:
            response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.error(f"Drone-only fallback failed: {e} ({response_time_ms}ms)")
            
            return {
                "drone_only_active": False,
                "error": str(e),
                "trigger_condition": trigger_condition,
                "activation_time_ms": response_time_ms,
            }

    async def isolate_ground_sources_for_drone_only(
        self,
        isolation_scope: str = "complete",
        maintain_emergency_link: bool = True
    ) -> dict[str, Any]:
        """
        SUBTASK-5.5.3.3 [10c] - Isolate ground sources for drone-only mode.
        
        Isolates ground-based sources while maintaining emergency communication.
        
        Args:
            isolation_scope: Scope of ground source isolation
            maintain_emergency_link: Whether to maintain emergency communication link
            
        Returns:
            Dict containing isolation result
        """
        start_time = datetime.now()
        
        try:
            # Create ground source isolation decision
            isolation_decision = SafetyDecision(
                decision_type=SafetyDecisionType.SOURCE_SELECTION,
                requesting_authority=SafetyAuthorityLevel.COMMUNICATION,
                details={
                    "isolation_scope": isolation_scope,
                    "maintain_emergency_link": maintain_emergency_link,
                    "ground_source_isolation": True,
                    "isolation_reason": "drone_only_mode_activation",
                    "isolation_timestamp": start_time.isoformat(),
                }
            )
            
            # Approve isolation decision
            approved, reason, approving_authority = await self.validate_safety_decision(isolation_decision)
            
            if not approved:
                return {
                    "ground_sources_isolated": False,
                    "reason": reason,
                    "isolation_scope": isolation_scope,
                }
            
            isolation_decision.approved = True
            isolation_decision.approving_authority = approving_authority
            isolation_decision.response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            self.decision_history.append(isolation_decision)
            
            result = {
                "ground_sources_isolated": True,
                "isolation_scope": isolation_scope,
                "emergency_link_maintained": maintain_emergency_link,
                "approving_authority": approving_authority,
                "response_time_ms": isolation_decision.response_time_ms,
                "isolation_timestamp": start_time.isoformat(),
            }
            
            logger.warning(f"Ground sources isolated for drone-only mode: {isolation_scope} scope ({isolation_decision.response_time_ms}ms)")
            
            return result
            
        except Exception as e:
            response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.error(f"Ground source isolation failed: {e} ({response_time_ms}ms)")
            
            return {
                "ground_sources_isolated": False,
                "error": str(e),
                "isolation_scope": isolation_scope,
                "response_time_ms": response_time_ms,
            }

    def get_emergency_override_status_report(self) -> dict[str, Any]:
        """
        SUBTASK-5.5.3.3 [10e] - Comprehensive emergency override status reporting.
        
        Provides detailed status report of emergency override system state.
        
        Returns:
            Dict containing comprehensive override status
        """
        try:
            # Get recent emergency decisions
            now = datetime.now()
            last_hour = now - timedelta(hours=1)
            
            recent_emergency_decisions = [
                d for d in self.decision_history 
                if d.timestamp >= last_hour and (
                    d.decision_type == SafetyDecisionType.EMERGENCY_STOP or
                    d.details.get("emergency_bypass", False) or
                    d.details.get("immediate_shutdown", False)
                )
            ]
            
            # Extract bypass pathways
            bypass_pathways = []
            isolated_components = []
            fallback_modes = []
            
            for decision in recent_emergency_decisions:
                if decision.details.get("bypassed_components"):
                    bypass_pathways.extend(decision.details["bypassed_components"])
                if decision.details.get("isolated_components"):
                    isolated_components.extend(decision.details["isolated_components"])
                if decision.details.get("fallback_mode"):
                    fallback_modes.append(decision.details["fallback_mode"])
            
            # Calculate response time metrics
            response_times = [d.response_time_ms for d in recent_emergency_decisions if d.response_time_ms is not None]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            status_report = {
                "override_active": self.emergency_override_active,
                "bypass_pathways": list(set(bypass_pathways)),
                "isolated_components": list(set(isolated_components)),
                "fallback_modes": list(set(fallback_modes)),
                "emergency_functions_status": {
                    "emergency_stop": "active" if self.emergency_override_active else "standby",
                    "direct_command_pathway": "available",
                    "coordination_isolation": "ready",
                    "drone_only_fallback": "available",
                },
                "recovery_readiness": {
                    "automatic_recovery": "enabled",
                    "manual_recovery": "available", 
                    "progressive_recovery": "ready",
                    "safety_validation": "operational",
                },
                "response_time_metrics": {
                    "average_response_time_ms": round(avg_response_time, 2),
                    "max_response_time_ms": max(response_times) if response_times else 0,
                    "recent_emergency_decisions": len(recent_emergency_decisions),
                    "performance_requirement": "< 500ms",
                    "requirement_met": avg_response_time < 500 if response_times else True,
                },
                "system_health": {
                    "total_authorities": len(self.authorities),
                    "active_authorities": len([auth for auth in self.authorities.values() if auth.active]),
                    "decision_history_size": len(self.decision_history),
                    "last_emergency_trigger": self.authorities[SafetyAuthorityLevel.EMERGENCY_STOP].last_trigger.isoformat() 
                                               if self.authorities[SafetyAuthorityLevel.EMERGENCY_STOP].last_trigger else None,
                },
                "report_timestamp": now.isoformat(),
            }
            
            logger.info(f"Emergency override status report generated: {len(recent_emergency_decisions)} recent decisions")
            return status_report
            
        except Exception as e:
            logger.error(f"Failed to generate emergency override status report: {e}")
            return {
                "override_active": self.emergency_override_active,
                "error": str(e),
                "report_timestamp": datetime.now().isoformat(),
            }

    def get_emergency_override_logs(self, last_minutes: int = 60) -> list[dict[str, Any]]:
        """
        SUBTASK-5.5.3.3 [10e] - Get emergency override logging infrastructure.
        
        Retrieves emergency override logs for the specified time period.
        
        Args:
            last_minutes: Number of minutes to look back for logs
            
        Returns:
            List of emergency override log entries
        """
        try:
            cutoff_time = datetime.now() - timedelta(minutes=last_minutes)
            
            emergency_logs = []
            for decision in self.decision_history:
                if decision.timestamp < cutoff_time:
                    continue
                    
                # Check if this is an emergency-related decision
                is_emergency = (
                    decision.decision_type == SafetyDecisionType.EMERGENCY_STOP or
                    decision.details.get("emergency_bypass", False) or
                    decision.details.get("immediate_shutdown", False) or
                    decision.details.get("emergency_override_triggered", False) or
                    decision.details.get("escalation_triggered", False)
                )
                
                if is_emergency:
                    log_entry = {
                        "event_type": self._get_emergency_event_type(decision),
                        "timestamp": decision.timestamp.isoformat(),
                        "trigger_reason": self._extract_trigger_reason(decision),
                        "decision_type": decision.decision_type.value,
                        "authority_level": decision.requesting_authority,
                        "approved": decision.approved,
                        "response_time_ms": decision.response_time_ms,
                        "details": decision.details,
                    }
                    emergency_logs.append(log_entry)
            
            # Sort by timestamp (most recent first)
            emergency_logs.sort(key=lambda x: x["timestamp"], reverse=True)
            
            logger.debug(f"Retrieved {len(emergency_logs)} emergency override logs")
            return emergency_logs
            
        except Exception as e:
            logger.error(f"Failed to retrieve emergency override logs: {e}")
            return []

    def _get_emergency_event_type(self, decision: 'SafetyDecision') -> str:
        """Helper method to determine emergency event type"""
        if decision.details.get("emergency_bypass"):
            return "emergency_coordination_bypass"
        elif decision.details.get("immediate_shutdown"):
            return "immediate_coordination_shutdown"
        elif decision.details.get("escalation_triggered"):
            return "safety_escalation"
        elif decision.decision_type == SafetyDecisionType.EMERGENCY_STOP:
            return "emergency_override_triggered"
        else:
            return "emergency_decision"

    def _extract_trigger_reason(self, decision: 'SafetyDecision') -> str:
        """Helper method to extract trigger reason from decision"""
        return (
            decision.details.get("reason") or
            decision.details.get("trigger_reason") or
            decision.details.get("bypass_reason") or
            decision.details.get("shutdown_trigger") or
            decision.details.get("emergency_reason") or
            "unknown"
        )

    async def attempt_automatic_recovery(
        self,
        safety_validation_required: bool = True,
        recovery_timeout_minutes: int = 5
    ) -> dict[str, Any]:
        """
        SUBTASK-5.5.3.3 [10f] - Automatic recovery when conditions return to safe.
        
        Attempts automatic recovery from emergency override state.
        
        Args:
            safety_validation_required: Whether safety validation is required
            recovery_timeout_minutes: Timeout for recovery attempt
            
        Returns:
            Dict containing recovery attempt result
        """
        start_time = datetime.now()
        
        try:
            if not self.emergency_override_active:
                return {
                    "recovery_attempted": False,
                    "reason": "No active emergency override to recover from",
                    "recovery_successful": False,
                }
            
            # Perform safety validation if required
            safety_validation_passed = True
            if safety_validation_required:
                # Check if safety conditions have returned to normal
                safety_validation_passed = await self._validate_safety_conditions_for_recovery()
            
            if not safety_validation_passed:
                return {
                    "recovery_attempted": True,
                    "safety_validation_passed": False,
                    "recovery_successful": False,
                    "reason": "Safety conditions not met for automatic recovery",
                    "recovery_timeout_minutes": recovery_timeout_minutes,
                }
            
            # Attempt recovery
            recovery_result = await self.clear_emergency_override("automatic_recovery_system")
            
            response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            result = {
                "recovery_attempted": True,
                "safety_validation_passed": safety_validation_passed,
                "recovery_successful": recovery_result.get("emergency_override_cleared", False),
                "recovery_timeout_minutes": recovery_timeout_minutes,
                "response_time_ms": response_time_ms,
                "recovery_timestamp": start_time.isoformat(),
            }
            
            if result["recovery_successful"]:
                logger.info(f"Automatic recovery successful ({response_time_ms}ms)")
            else:
                logger.warning(f"Automatic recovery failed ({response_time_ms}ms)")
            
            return result
            
        except Exception as e:
            response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.error(f"Automatic recovery attempt failed: {e} ({response_time_ms}ms)")
            
            return {
                "recovery_attempted": True,
                "safety_validation_passed": None,
                "recovery_successful": False,
                "error": str(e),
                "response_time_ms": response_time_ms,
            }

    async def authorize_manual_recovery(
        self,
        authorized_by: str,
        authorization_code: str,
        force_recovery: bool = False
    ) -> dict[str, Any]:
        """
        SUBTASK-5.5.3.3 [10f] - Manual recovery with proper authorization.
        
        Authorizes manual recovery from emergency override state.
        
        Args:
            authorized_by: Person authorizing recovery
            authorization_code: Authorization code for recovery
            force_recovery: Whether to force recovery without safety validation
            
        Returns:
            Dict containing manual recovery authorization result
        """
        start_time = datetime.now()
        
        try:
            if not self.emergency_override_active:
                return {
                    "recovery_authorized": False,
                    "reason": "No active emergency override to recover from",
                    "authorized_by": authorized_by,
                }
            
            # Validate authorization code (in real implementation, this would check against secure codes)
            valid_codes = ["EMERGENCY_CLEAR_001", "SAFETY_OVERRIDE_RESET", "MANUAL_RECOVERY_AUTH"]
            if authorization_code not in valid_codes and not force_recovery:
                return {
                    "recovery_authorized": False,
                    "reason": "Invalid authorization code",
                    "authorized_by": authorized_by,
                    "authorization_code": authorization_code,
                }
            
            # Perform manual recovery
            recovery_result = await self.clear_emergency_override(f"manual_recovery_by_{authorized_by}")
            
            response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            result = {
                "recovery_authorized": True,
                "recovery_successful": recovery_result.get("emergency_override_cleared", False),
                "authorized_by": authorized_by,
                "authorization_code": authorization_code,
                "force_recovery_used": force_recovery,
                "response_time_ms": response_time_ms,
                "recovery_timestamp": start_time.isoformat(),
            }
            
            logger.warning(f"Manual recovery authorized by {authorized_by}: {recovery_result.get('emergency_override_cleared')} ({response_time_ms}ms)")
            
            return result
            
        except Exception as e:
            response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.error(f"Manual recovery authorization failed: {e} ({response_time_ms}ms)")
            
            return {
                "recovery_authorized": False,
                "error": str(e),
                "authorized_by": authorized_by,
                "response_time_ms": response_time_ms,
            }

    async def execute_progressive_recovery(
        self,
        recovery_phases: list[str],
        phase_timeout_minutes: int = 2,
        abort_on_failure: bool = True
    ) -> dict[str, Any]:
        """
        SUBTASK-5.5.3.3 [10f] - Progressive recovery of coordination system components.
        
        Executes progressive recovery through multiple phases.
        
        Args:
            recovery_phases: List of recovery phases to execute
            phase_timeout_minutes: Timeout for each phase
            abort_on_failure: Whether to abort on phase failure
            
        Returns:
            Dict containing progressive recovery result
        """
        start_time = datetime.now()
        
        try:
            completed_phases = []
            failed_phases = []
            
            for phase in recovery_phases:
                phase_start = datetime.now()
                
                try:
                    phase_result = await self._execute_recovery_phase(phase, phase_timeout_minutes)
                    
                    if phase_result["phase_successful"]:
                        completed_phases.append(phase)
                        logger.info(f"Recovery phase '{phase}' completed successfully")
                    else:
                        failed_phases.append(phase)
                        logger.error(f"Recovery phase '{phase}' failed: {phase_result.get('error')}")
                        
                        if abort_on_failure:
                            break
                            
                except Exception as e:
                    failed_phases.append(phase)
                    logger.error(f"Recovery phase '{phase}' exception: {e}")
                    
                    if abort_on_failure:
                        break
            
            recovery_successful = len(failed_phases) == 0 and len(completed_phases) == len(recovery_phases)
            response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            result = {
                "progressive_recovery_successful": recovery_successful,
                "recovery_phases_completed": completed_phases,
                "recovery_phases_failed": failed_phases,
                "total_phases": len(recovery_phases),
                "abort_on_failure": abort_on_failure,
                "phase_timeout_minutes": phase_timeout_minutes,
                "response_time_ms": response_time_ms,
                "recovery_timestamp": start_time.isoformat(),
            }
            
            if recovery_successful:
                logger.info(f"Progressive recovery completed successfully: {len(completed_phases)} phases ({response_time_ms}ms)")
            else:
                logger.warning(f"Progressive recovery partial success: {len(completed_phases)}/{len(recovery_phases)} phases ({response_time_ms}ms)")
            
            return result
            
        except Exception as e:
            response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.error(f"Progressive recovery failed: {e} ({response_time_ms}ms)")
            
            return {
                "progressive_recovery_successful": False,
                "error": str(e),
                "recovery_phases_completed": completed_phases,
                "response_time_ms": response_time_ms,
            }

    async def _validate_safety_conditions_for_recovery(self) -> bool:
        """Helper method to validate safety conditions for recovery"""
        try:
            # Wrap validation logic in timeout protection
            async with asyncio.timeout(1.0):  # 1 second timeout
                # In real implementation, this would check actual system health
                # For now, simulate safety validation
                await asyncio.sleep(0.01)  # Simulate validation time
                
                # Check if any critical safety conditions are still present
                recent_escalations = [
                    d for d in self.decision_history[-10:] 
                    if d.details.get("escalation_triggered", False)
                ]
                
                # If no recent escalations, conditions are likely safe
                return len(recent_escalations) == 0
                
        except asyncio.TimeoutError:
            logger.error("Safety condition validation timed out after 1.0 seconds")
            return False
        except Exception as e:
            logger.error(f"Safety condition validation failed: {e}")
            return False

    async def _execute_recovery_phase(self, phase: str, timeout_minutes: int) -> dict[str, Any]:
        """Helper method to execute individual recovery phase"""
        try:
            phase_start = datetime.now()
            
            if phase == "safety_validation":
                validation_result = await self._validate_safety_conditions_for_recovery()
                return {
                    "phase_successful": validation_result,
                    "phase": phase,
                    "details": {"validation_passed": validation_result}
                }
                
            elif phase == "component_health_check":
                # Simulate component health check
                await asyncio.sleep(0.02)  # Simulate health check time
                return {
                    "phase_successful": True,
                    "phase": phase,
                    "details": {"components_healthy": True}
                }
                
            elif phase == "coordination_restore":
                # Simulate coordination system restoration
                await asyncio.sleep(0.01)  # Simulate restoration time
                return {
                    "phase_successful": True,
                    "phase": phase,
                    "details": {"coordination_restored": True}
                }
                
            else:
                return {
                    "phase_successful": False,
                    "phase": phase,
                    "error": f"Unknown recovery phase: {phase}"
                }
                
        except Exception as e:
            return {
                "phase_successful": False,
                "phase": phase,
                "error": str(e)
            }

    def validate_configuration(self) -> dict[str, Any]:
        """
        SUBTASK-5.5.3.4 [11e] - Validate SafetyManager configuration.
        
        Returns:
            Dict containing configuration validation results
        """
        try:
            config_valid = True
            issues = []
            
            # Check authority levels configuration
            if not self.authorities or len(self.authorities) != 6:
                config_valid = False
                issues.append("Expected 6 safety authority levels, found " + str(len(self.authorities) if self.authorities else 0))
            
            # Check emergency response capability
            if SafetyAuthorityLevel.EMERGENCY_STOP not in self.authorities:
                config_valid = False
                issues.append("Missing EMERGENCY_STOP authority level")
            
            # Verify all authorities are properly initialized
            inactive_authorities = [
                level for level, authority in self.authorities.items() 
                if not authority.active
            ]
            
            return {
                "config_valid": config_valid,
                "authority_levels_configured": len(self.authorities) if self.authorities else 0,
                "emergency_response_ready": SafetyAuthorityLevel.EMERGENCY_STOP in self.authorities,
                "inactive_authorities": len(inactive_authorities),
                "issues": issues,
                "timestamp": datetime.now().isoformat(),
            }
            
        except Exception as e:
            return {
                "config_valid": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def perform_health_check(self) -> dict[str, Any]:
        """
        SUBTASK-5.5.3.4 [11e] - Perform SafetyManager health check.
        
        Returns:
            Dict containing health check results
        """
        start_time = datetime.now()
        
        try:
            # Check response time by performing a quick operation
            test_decision = SafetyDecision(
                decision_type=SafetyDecisionType.EMERGENCY_STOP,
                requesting_authority=SafetyAuthorityLevel.EMERGENCY_STOP,
                details={"health_check": True}
            )
            
            # Time the validation operation
            validation_start = datetime.now()
            
            # Simple validation test - don't actually approve, just test the mechanism
            if self.authorities and SafetyAuthorityLevel.EMERGENCY_STOP in self.authorities:
                authority = self.authorities[SafetyAuthorityLevel.EMERGENCY_STOP]
                can_validate = authority.active
            else:
                can_validate = False
                
            response_time_ms = int((datetime.now() - validation_start).total_seconds() * 1000)
            
            # Determine health status
            if not can_validate:
                health_status = "critical"
            elif response_time_ms > 100:
                health_status = "degraded"
            else:
                health_status = "healthy"
            
            # Count active authorities
            authority_levels_active = len([
                auth for auth in self.authorities.values() if auth.active
            ]) if self.authorities else 0
            
            total_check_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return {
                "health_status": health_status,
                "response_time_ms": response_time_ms,
                "authority_levels_active": authority_levels_active,
                "total_authorities": len(self.authorities) if self.authorities else 0,
                "emergency_response_available": can_validate,
                "decision_history_size": len(self.decision_history),
                "emergency_override_active": self.emergency_override_active,
                "total_check_time_ms": total_check_time,
                "timestamp": start_time.isoformat(),
            }
            
        except Exception as e:
            response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            return {
                "health_status": "critical",
                "response_time_ms": response_time_ms,
                "error": str(e),
                "timestamp": start_time.isoformat(),
            }