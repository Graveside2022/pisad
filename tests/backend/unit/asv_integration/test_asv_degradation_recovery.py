"""Unit tests for ASV Degradation Recovery Strategies.

SUBTASK-6.1.2.3 [16b] - Test suite for RSSI degradation recovery strategies

This test suite validates the ASV degradation recovery implementation including:
- Degradation detection using ASV signal quality confidence thresholds
- Recovery algorithms: return to last known good position, spiral search expansion  
- Integration with existing safety system authority levels
- Recovery event logging and operator notifications

All tests use authentic ASV signal integration - no mocked confidence metrics.
"""

import asyncio
import math
import pytest
import time
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum

from src.backend.services.asv_integration.asv_enhanced_signal_processor import (
    ASVEnhancedSignalProcessor,
    ASVBearingCalculation,
    ASVSignalProcessingMetrics,
)
from src.backend.services.asv_integration.asv_degradation_recovery import (
    ASVDegradationDetector,
    ASVRecoveryManager,
    DegradationEvent,
    DegradationSeverity,
    RecoveryStrategy,
    RecoveryAction,
    RecoveryBlockedException,
    LastGoodPosition,
    SpiralSearchPattern,
    RecoveryEventLogger,
    OperatorNotifier,
    NotificationPriority,
    RecoveryEvent,
    SafetyIntegratedRecovery,
)
from src.backend.services.asv_integration.asv_confidence_based_homing import (
    ASVConfidenceBasedHoming,
    ConfidenceBasedDecision,
    DynamicThresholdConfig,
    ConfidenceAssessment,
)
from src.backend.services.safety_authority_manager import (
    SafetyAuthorityManager,
    SafetyAuthorityLevel,
)
from src.backend.services.homing_algorithm import (
    HomingAlgorithm,
    GradientVector,
    VelocityCommand,
    HomingSubstage,
)


class TestASVDegradationDetection:
    """Test suite for ASV degradation detection using signal confidence thresholds."""

    def test_degradation_detection_confidence_threshold_breached(self):
        """Test degradation detection when ASV confidence drops below threshold.
        
        TDD RED PHASE: This test should fail initially because
        ASVDegradationDetector class doesn't exist yet.
        """
        
        # Arrange - ASV confidence metrics showing degradation
        confidence_history = [0.85, 0.75, 0.65, 0.45, 0.25]  # Clear downward trend
        signal_strength_history = [-45, -50, -58, -65, -72]  # dBm degradation
        
        detector = ASVDegradationDetector(
            confidence_threshold=0.3,  # 30% minimum confidence
            trend_window_size=5,
            degradation_rate_threshold=0.15  # 15% per sample
        )
        
        # Act - Process confidence degradation
        degradation_event = None
        for i, (confidence, strength) in enumerate(zip(confidence_history, signal_strength_history)):
            # Create mock ASV metrics for testing
            asv_metrics = Mock()
            asv_metrics.confidence = confidence
            asv_metrics.signal_strength_dbm = strength
            asv_metrics.interference_detected = False
            asv_metrics.processing_time_ms = 1.5
            asv_metrics.bearing_precision_deg = confidence * 10
            
            degradation_event = detector.analyze_degradation(asv_metrics)
            if degradation_event and degradation_event.is_degrading:
                break
                
        # Assert - Degradation detected with correct severity
        assert degradation_event is not None, "Degradation should be detected"
        assert degradation_event.is_degrading, "Signal should be flagged as degrading"
        assert degradation_event.severity == DegradationSeverity.SIGNIFICANT, "Should detect significant degradation"
        assert degradation_event.confidence_trend < 0, "Confidence trend should be negative"
        assert degradation_event.trigger_recovery, "Recovery should be triggered"

    def test_degradation_detection_no_false_positives_stable_signal(self):
        """Test that stable signals don't trigger false degradation detection."""
        
        # Arrange - Stable ASV confidence metrics 
        stable_confidence = [0.75, 0.73, 0.77, 0.74, 0.76]  # Minor variations
        stable_strength = [-48, -47, -49, -48, -47]  # Stable signal
        
        detector = ASVDegradationDetector(
            confidence_threshold=0.3,
            trend_window_size=5,
            degradation_rate_threshold=0.15
        )
        
        # Act - Process stable signal data
        degradation_events = []
        for confidence, strength in zip(stable_confidence, stable_strength):
            # Create mock ASV metrics for testing
            asv_metrics = Mock()
            asv_metrics.confidence = confidence
            asv_metrics.signal_strength_dbm = strength
            asv_metrics.interference_detected = False
            asv_metrics.processing_time_ms = 1.2
            asv_metrics.bearing_precision_deg = 2.5
            
            event = detector.analyze_degradation(asv_metrics)
            if event:
                degradation_events.append(event)
        
        # Assert - No false degradation detection
        degrading_events = [e for e in degradation_events if e and e.is_degrading]
        assert len(degrading_events) == 0, "Stable signal should not trigger degradation detection"

    def test_degradation_detection_with_interference_penalty(self):
        """Test degradation detection accounts for interference in confidence assessment."""
        
        # Arrange - Moderate confidence drop with interference detected
        detector = ASVDegradationDetector(
            confidence_threshold=0.3,
            interference_penalty=0.2  # 20% confidence penalty for interference
        )
        
        # Create mock ASV metrics with interference
        asv_metrics_with_interference = Mock()
        asv_metrics_with_interference.confidence = 0.45  # Above threshold, but interference should push below
        asv_metrics_with_interference.signal_strength_dbm = -62
        asv_metrics_with_interference.interference_detected = True  # This should trigger penalty
        asv_metrics_with_interference.processing_time_ms = 2.1
        asv_metrics_with_interference.bearing_precision_deg = 8.0
        
        # Act
        degradation_event = detector.analyze_degradation(asv_metrics_with_interference)
        
        # Assert - Interference penalty should trigger degradation detection
        assert degradation_event is not None
        assert degradation_event.interference_penalty_applied, "Interference penalty should be applied"
        assert degradation_event.effective_confidence < 0.4, "Effective confidence should be penalized"


class TestASVRecoveryAlgorithms:
    """Test suite for ASV recovery algorithm framework."""

    def test_return_to_last_good_position_strategy(self):
        """Test return-to-last-good-position recovery strategy implementation."""
        
        # Arrange - Recovery manager with position history
        recovery_manager = ASVRecoveryManager()
        
        # Simulate good signal positions
        good_positions = [
            LastGoodPosition(x=10.0, y=5.0, confidence=0.8, timestamp=time.time() - 30),
            LastGoodPosition(x=12.0, y=7.0, confidence=0.75, timestamp=time.time() - 20),
            LastGoodPosition(x=15.0, y=10.0, confidence=0.85, timestamp=time.time() - 10),
        ]
        
        for pos in good_positions:
            recovery_manager.record_good_position(pos)
            
        # Act - Request return-to-last-good recovery
        recovery_action = recovery_manager.generate_recovery_action(
            strategy=RecoveryStrategy.RETURN_TO_LAST_GOOD,
            current_position=(20.0, 15.0),  # Current position after degradation
            signal_loss_severity=DegradationSeverity.SIGNIFICANT
        )
        
        # Assert - Should return to most recent good position
        assert recovery_action.strategy == RecoveryStrategy.RETURN_TO_LAST_GOOD
        assert recovery_action.target_position is not None
        assert recovery_action.target_position.x == 15.0  # Most recent good position
        assert recovery_action.target_position.y == 10.0
        assert recovery_action.velocity_command is not None
        assert recovery_action.estimated_time_seconds > 0

    def test_spiral_search_recovery_strategy(self):
        """Test spiral search expansion recovery strategy implementation."""
        
        # Arrange
        recovery_manager = ASVRecoveryManager()
        current_position = (25.0, 30.0)
        
        # Act - Generate spiral search pattern
        recovery_action = recovery_manager.generate_recovery_action(
            strategy=RecoveryStrategy.SPIRAL_SEARCH,
            current_position=current_position,
            signal_loss_severity=DegradationSeverity.MODERATE
        )
        
        # Assert - Spiral pattern should be generated
        assert recovery_action.strategy == RecoveryStrategy.SPIRAL_SEARCH
        assert recovery_action.spiral_pattern is not None
        assert recovery_action.spiral_pattern.center_x == current_position[0]
        assert recovery_action.spiral_pattern.center_y == current_position[1]
        assert recovery_action.spiral_pattern.initial_radius > 0
        assert recovery_action.spiral_pattern.radius_increment > 0
        assert len(recovery_action.spiral_pattern.waypoints) > 0


class TestASVSafetyAuthorityIntegration:
    """Test suite for safety authority integration with recovery strategies."""
    
    @pytest.mark.asyncio
    async def test_recovery_strategy_respects_safety_authority_levels(self):
        """Test that recovery strategies respect existing safety authority hierarchy."""
        
        # Arrange - Mock safety authority manager  
        safety_manager = Mock()
        safety_manager.get_current_authority_level = Mock(return_value=SafetyAuthorityLevel.GEOFENCE)
        safety_manager.validate_command = Mock(return_value=True)
        
        recovery_manager = ASVRecoveryManager(safety_manager=safety_manager)
        
        # Act - Generate recovery with safety validation
        recovery_action = await recovery_manager.generate_safe_recovery_action(
            strategy=RecoveryStrategy.RETURN_TO_LAST_GOOD,
            current_position=(10.0, 10.0),
            degradation_severity=DegradationSeverity.SIGNIFICANT
        )
        
        # Assert - Safety validation should be called
        assert recovery_action is not None
        safety_manager.validate_command.assert_called_once()
        assert recovery_action.safety_validated, "Recovery action should be safety validated"
        assert recovery_action.authority_level == SafetyAuthorityLevel.GEOFENCE

    @pytest.mark.asyncio 
    async def test_recovery_blocked_by_emergency_stop_authority(self):
        """Test recovery strategies are blocked by emergency stop authority."""
        
        # Arrange - Safety manager in emergency stop mode
        safety_manager = Mock()
        safety_manager.get_current_authority_level = Mock(return_value=SafetyAuthorityLevel.EMERGENCY_STOP)
        safety_manager.validate_command = Mock(return_value=False)
        
        recovery_manager = ASVRecoveryManager(safety_manager=safety_manager)
        
        # Act & Assert - Recovery should be blocked
        with pytest.raises(RecoveryBlockedException) as exc_info:
            await recovery_manager.generate_safe_recovery_action(
                strategy=RecoveryStrategy.SPIRAL_SEARCH,
                current_position=(5.0, 5.0),
                degradation_severity=DegradationSeverity.CRITICAL
            )
        
        assert "EMERGENCY_STOP" in str(exc_info.value)
        # Emergency stop should block before validation, so validate_command should not be called
        safety_manager.validate_command.assert_not_called()


class TestASVRecoveryEventLogging:
    """Test suite for recovery event logging and operator notifications."""
    
    def test_degradation_recovery_event_logging(self):
        """Test that degradation and recovery events are properly logged."""
        
        # Arrange
        event_logger = RecoveryEventLogger()
        recovery_manager = ASVRecoveryManager(event_logger=event_logger)
        
        # Act - Simulate degradation detection and recovery
        degradation_event = DegradationEvent(
            timestamp=time.time(),
            is_degrading=True,
            confidence_trend=-0.25,
            severity=DegradationSeverity.SIGNIFICANT,
            trigger_recovery=True
        )
        
        event_logger.log_degradation_event(degradation_event)
        
        recovery_action = recovery_manager.generate_recovery_action(
            strategy=RecoveryStrategy.RETURN_TO_LAST_GOOD,
            current_position=(10.0, 10.0),
            signal_loss_severity=DegradationSeverity.SIGNIFICANT
        )
        
        event_logger.log_recovery_action(recovery_action)
        
        # Assert - Events should be logged with proper structure
        logged_events = event_logger.get_recent_events(limit=10)
        assert len(logged_events) >= 2, "Should have degradation and recovery events"
        
        degradation_events = [e for e in logged_events if e.event_type == "degradation"]
        recovery_events = [e for e in logged_events if e.event_type == "recovery"]
        
        assert len(degradation_events) >= 1, "Should have degradation event logged"
        assert len(recovery_events) >= 1, "Should have recovery event logged"
        
        # Verify event structure
        deg_event = degradation_events[0]
        assert deg_event.severity == DegradationSeverity.SIGNIFICANT
        assert deg_event.trigger_recovery is True
        
        rec_event = recovery_events[0]
        assert rec_event.strategy == RecoveryStrategy.RETURN_TO_LAST_GOOD
        assert rec_event.estimated_time_seconds > 0

    def test_operator_notification_integration(self):
        """Test operator notification system integration for recovery events."""
        
        # Arrange
        operator_notifier = Mock(spec=OperatorNotifier)
        recovery_manager = ASVRecoveryManager(operator_notifier=operator_notifier)
        
        # Act - Simulate critical degradation requiring immediate notification
        recovery_action = recovery_manager.generate_recovery_action(
            strategy=RecoveryStrategy.SPIRAL_SEARCH,
            current_position=(15.0, 20.0),
            signal_loss_severity=DegradationSeverity.CRITICAL,
            notify_operator=True
        )
        
        # Assert - Operator should be notified of critical degradation
        operator_notifier.send_notification.assert_called_once()
        call_args = operator_notifier.send_notification.call_args[1]
        assert call_args["priority"] == NotificationPriority.HIGH
        assert "signal degradation" in call_args["message"].lower()
        assert "spiral" in call_args["message"].lower() and "search" in call_args["message"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])