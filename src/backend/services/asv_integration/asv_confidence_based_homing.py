"""ASV Confidence-Based Homing Integration.

SUBTASK-6.1.2.2 [15c] - Add ASV confidence-based signal quality assessment for course correction decisions

This module implements confidence-based signal quality assessment for homing course correction
decisions, integrating ASV analyzer confidence metrics into the existing decision tree with
dynamic threshold adjustment and fallback strategies.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from enum import Enum

from src.backend.services.asv_integration.asv_enhanced_signal_processor import (
    ASVBearingCalculation,
    ASVEnhancedSignalProcessor,
)
from src.backend.services.asv_integration.asv_enhanced_homing_integration import (
    ASVEnhancedGradient,
    ASVEnhancedHomingIntegration,
)
from src.backend.services.homing_algorithm import (
    HomingAlgorithm,
    GradientVector,
    VelocityCommand,
)

logger = logging.getLogger(__name__)


class ConfidenceAssessment(str, Enum):
    """Confidence assessment levels for ASV signal quality."""

    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    CRITICAL = "CRITICAL"


@dataclass
class DynamicThresholdConfig:
    """Configuration for dynamic threshold adjustment based on signal quality."""

    high_quality_threshold: float = 0.3  # Lower threshold for high quality signals
    moderate_quality_threshold: float = 0.6  # Standard threshold
    low_quality_threshold: float = 0.8  # Higher threshold for low quality signals
    signal_quality_boundaries: Optional[list[float]] = None  # [high/moderate, moderate/low] boundaries
    interference_penalty_factor: float = 0.2  # Penalty for interference detection

    def __post_init__(self) -> None:
        if self.signal_quality_boundaries is None:
            self.signal_quality_boundaries = [0.7, 0.4]  # High/Moderate/Low boundaries


@dataclass
class ConfidenceBasedDecision:
    """Decision result from confidence-based assessment."""

    proceed_with_homing: bool
    dynamic_threshold: float
    confidence_assessment: str
    fallback_strategy: str
    signal_degradation_detected: bool = False
    safety_override_reason: Optional[str] = None
    decision_time_ms: float = 0.0
    asv_confidence: float = 0.0
    signal_quality: float = 0.0


class ASVConfidenceBasedHoming:
    """
    ASV Confidence-Based Homing Decision System.

    Implements confidence-based signal quality assessment for course correction decisions,
    replacing static thresholds with dynamic assessment based on ASV analyzer outputs.

    Key Features:
    - Dynamic threshold adjustment based on signal quality
    - Confidence-based fallback strategies
    - Integration with existing homing algorithm decision tree
    - Preservation of all safety authority mechanisms
    """

    def __init__(
        self,
        asv_processor: Optional[ASVEnhancedSignalProcessor] = None,
        homing_algorithm: Optional[HomingAlgorithm] = None,
        threshold_config: Optional[DynamicThresholdConfig] = None,
    ):
        """Initialize confidence-based homing system.

        Args:
            asv_processor: ASV enhanced signal processor
            homing_algorithm: Existing homing algorithm instance
            threshold_config: Dynamic threshold configuration
        """
        self._asv_processor = asv_processor
        self._homing_algorithm = homing_algorithm
        self._threshold_config = threshold_config or DynamicThresholdConfig()

        # Safety override mechanism
        self._safety_override_active = False
        self._safety_override_reason: Optional[str] = None

        # Performance tracking
        self._decisions_made = 0
        self._high_confidence_decisions = 0
        self._fallback_activations = 0
        self._average_decision_time_ms = 0.0

        logger.info("ASV Confidence-Based Homing initialized")

    def evaluate_confidence_based_decision(
        self,
        asv_bearing: ASVBearingCalculation,
        current_position: Optional[tuple[float, float]] = None,
    ) -> ConfidenceBasedDecision:
        """Evaluate confidence-based decision for homing course correction.

        Args:
            asv_bearing: ASV bearing calculation with confidence metrics
            current_position: Current position (x, y) for context

        Returns:
            ConfidenceBasedDecision with recommendation and strategy
        """
        decision_start = time.perf_counter()

        # Check safety override first
        if self._safety_override_active:
            decision = ConfidenceBasedDecision(
                proceed_with_homing=False,
                dynamic_threshold=1.0,  # Maximum threshold when overridden
                confidence_assessment=ConfidenceAssessment.CRITICAL.value,
                fallback_strategy="SAFETY_OVERRIDE",
                safety_override_reason=self._safety_override_reason,
            )
            self._update_decision_metrics(decision, decision_start)
            return decision

        # [15c-1] Calculate dynamic threshold based on ASV signal quality
        dynamic_threshold = self._calculate_dynamic_threshold(asv_bearing)

        # [15c-2] Assess confidence level and signal degradation
        confidence_assessment = self._assess_confidence_level(asv_bearing)
        signal_degradation = self._detect_signal_degradation(asv_bearing)

        # [15c-3] Determine if homing should proceed
        proceed_with_homing = asv_bearing.confidence >= dynamic_threshold

        # [15c-4] Select fallback strategy if needed
        fallback_strategy = (
            self._select_fallback_strategy(asv_bearing, confidence_assessment, signal_degradation)
            if not proceed_with_homing
            else "NONE"
        )

        decision = ConfidenceBasedDecision(
            proceed_with_homing=proceed_with_homing,
            dynamic_threshold=dynamic_threshold,
            confidence_assessment=confidence_assessment,
            fallback_strategy=fallback_strategy,
            signal_degradation_detected=signal_degradation,
            asv_confidence=asv_bearing.confidence,
            signal_quality=asv_bearing.signal_quality,
        )

        self._update_decision_metrics(decision, decision_start)

        logger.debug(
            f"Confidence decision: proceed={proceed_with_homing}, "
            f"threshold={dynamic_threshold:.2f}, assessment={confidence_assessment}, "
            f"fallback={fallback_strategy}"
        )

        return decision

    def _calculate_dynamic_threshold(self, asv_bearing: ASVBearingCalculation) -> float:
        """Calculate dynamic confidence threshold based on ASV signal quality.

        [15c-1] Implementation of dynamic threshold adjustment
        """
        signal_quality = asv_bearing.signal_quality
        
        # Ensure boundaries are available
        boundaries = self._threshold_config.signal_quality_boundaries
        if boundaries is None:
            boundaries = [0.7, 0.4]  # Default boundaries

        # Determine base threshold based on signal quality ranges
        if signal_quality >= boundaries[0]:
            # High quality signal - lower threshold requirements
            base_threshold = self._threshold_config.high_quality_threshold
        elif signal_quality >= boundaries[1]:
            # Moderate quality signal - standard threshold
            base_threshold = self._threshold_config.moderate_quality_threshold
        else:
            # Low quality signal - higher threshold requirements
            base_threshold = self._threshold_config.low_quality_threshold

        # Apply interference penalty
        interference_adjustment = 0.0
        if asv_bearing.interference_detected:
            interference_adjustment = self._threshold_config.interference_penalty_factor

        # Apply signal strength adjustment
        strength_adjustment = self._calculate_strength_adjustment(asv_bearing.signal_strength_dbm)

        dynamic_threshold = base_threshold + interference_adjustment + strength_adjustment

        # Clamp to reasonable range
        return max(0.1, min(0.95, dynamic_threshold))

    def _calculate_strength_adjustment(self, signal_strength_dbm: float) -> float:
        """Calculate threshold adjustment based on signal strength."""
        # Very weak signals (-120 to -100 dBm) require higher confidence
        if signal_strength_dbm < -110:
            return 0.15  # Increase threshold
        # Strong signals (-80 to -60 dBm) can accept lower confidence
        elif signal_strength_dbm > -80:
            return -0.1  # Decrease threshold
        # Moderate signals (-110 to -80 dBm) use base threshold
        return 0.0

    def _assess_confidence_level(self, asv_bearing: ASVBearingCalculation) -> str:
        """Assess confidence level based on ASV metrics.

        [15c-2] Implementation of confidence level assessment
        """
        confidence = asv_bearing.confidence
        signal_quality = asv_bearing.signal_quality

        # Combined assessment using both confidence and signal quality
        combined_score = (confidence * 0.7) + (signal_quality * 0.3)

        if combined_score >= 0.75:
            return ConfidenceAssessment.HIGH.value
        elif combined_score >= 0.45:
            return ConfidenceAssessment.MODERATE.value
        elif combined_score >= 0.2:
            return ConfidenceAssessment.LOW.value
        else:
            return ConfidenceAssessment.CRITICAL.value

    def _detect_signal_degradation(self, asv_bearing: ASVBearingCalculation) -> bool:
        """Detect signal degradation based on ASV analysis."""
        # Check for interference
        if asv_bearing.interference_detected:
            return True

        # Check for poor signal quality combined with low confidence
        if asv_bearing.signal_quality < 0.3 and asv_bearing.confidence < 0.4:
            return True

        # Check for poor precision (high uncertainty)
        if asv_bearing.precision_deg > 20.0:  # Much worse than ASV target of ±2°
            return True

        # Check signal strength degradation
        if asv_bearing.signal_strength_dbm < -115.0:  # Very weak signal
            return True

        return False

    def _select_fallback_strategy(
        self,
        asv_bearing: ASVBearingCalculation,
        confidence_assessment: str,
        signal_degradation: bool,
    ) -> str:
        """Select appropriate fallback strategy based on confidence assessment.

        [15c-3] Implementation of confidence-based fallback strategies
        """
        if confidence_assessment == ConfidenceAssessment.CRITICAL.value:
            return "RETURN_TO_LAST_KNOWN"

        if signal_degradation:
            if asv_bearing.interference_detected:
                return "SPIRAL_SEARCH"  # Move away from interference
            else:
                return "SAMPLING"  # Try to improve signal via S-turns

        if confidence_assessment == ConfidenceAssessment.LOW.value:
            # For low confidence, try sampling maneuvers
            return "SAMPLING"

        if confidence_assessment == ConfidenceAssessment.MODERATE.value:
            # For moderate confidence, use spiral search to improve positioning
            return "SPIRAL_SEARCH"

        return "NONE"

    def enhance_gradient_with_asv_confidence(
        self, original_gradient: GradientVector, asv_bearing: ASVBearingCalculation
    ) -> ASVEnhancedGradient:
        """Enhance existing gradient with ASV confidence metrics.

        [15c-4] Integration with existing homing algorithm decision tree
        """
        # Convert ASV confidence (0-1) to gradient confidence (0-100)
        enhanced_confidence = asv_bearing.confidence * 100.0

        # Use ASV bearing for enhanced direction
        enhanced_direction = asv_bearing.bearing_deg

        # Scale magnitude based on ASV precision (better precision = higher magnitude)
        enhanced_magnitude = max(0.1, 2.0 / max(0.1, asv_bearing.precision_deg))

        return ASVEnhancedGradient(
            magnitude=enhanced_magnitude,
            direction=enhanced_direction,
            confidence=enhanced_confidence,
            asv_bearing_deg=asv_bearing.bearing_deg,
            asv_confidence=asv_bearing.confidence,
            asv_precision_deg=asv_bearing.precision_deg,
            signal_strength_dbm=asv_bearing.signal_strength_dbm,
            interference_detected=asv_bearing.interference_detected,
            processing_method="asv_professional",
            calculation_time_ms=0.0,  # Will be set by caller
        )

    def integrate_with_homing_algorithm(
        self, asv_bearing: ASVBearingCalculation, current_heading: float, current_time: float
    ) -> Optional[VelocityCommand]:
        """Integrate confidence-based decisions with existing homing algorithm.

        This method bridges ASV confidence assessment with the existing homing
        algorithm's velocity command generation.
        """
        if not self._homing_algorithm:
            logger.warning("No homing algorithm available for integration")
            return None

        # Make confidence-based decision
        decision = self.evaluate_confidence_based_decision(asv_bearing)

        if not decision.proceed_with_homing:
            logger.info(f"Homing not recommended: {decision.fallback_strategy}")
            return self._generate_fallback_command(decision, current_heading, current_time)

        # Enhance gradient with ASV confidence
        # Create base gradient from ASV bearing for integration
        base_gradient = GradientVector(
            magnitude=1.0,  # Will be enhanced
            direction=asv_bearing.bearing_deg,
            confidence=asv_bearing.confidence * 100.0,
        )

        enhanced_gradient = self.enhance_gradient_with_asv_confidence(base_gradient, asv_bearing)

        # Convert back to standard gradient for homing algorithm
        standard_gradient = enhanced_gradient.to_gradient_vector()

        # Use existing homing algorithm with enhanced gradient
        return self._homing_algorithm.generate_velocity_command(
            standard_gradient, current_heading, current_time
        )

    def _generate_fallback_command(
        self, decision: ConfidenceBasedDecision, current_heading: float, current_time: float
    ) -> VelocityCommand:
        """Generate fallback velocity command based on confidence decision."""
        if decision.fallback_strategy == "RETURN_TO_LAST_KNOWN":
            # Stop and hold position
            return VelocityCommand(forward_velocity=0.0, yaw_rate=0.0)

        elif decision.fallback_strategy == "SPIRAL_SEARCH":
            # Slow spiral search pattern
            return VelocityCommand(forward_velocity=1.5, yaw_rate=0.3)

        elif decision.fallback_strategy == "SAMPLING":
            # S-turn sampling maneuver
            phase = (current_time % 10.0) / 10.0 * 2.0 * 3.14159
            yaw_rate = 0.4 * math.sin(phase)  # S-turn pattern
            return VelocityCommand(forward_velocity=2.0, yaw_rate=yaw_rate)

        elif decision.fallback_strategy == "SAFETY_OVERRIDE":
            # Complete stop for safety
            return VelocityCommand(forward_velocity=0.0, yaw_rate=0.0)

        else:
            # Default fallback
            return VelocityCommand(forward_velocity=1.0, yaw_rate=0.0)

    def _update_decision_metrics(self, decision: ConfidenceBasedDecision, start_time: float) -> None:
        """Update decision performance metrics."""
        decision_time_ms = (time.perf_counter() - start_time) * 1000
        decision.decision_time_ms = decision_time_ms

        self._decisions_made += 1

        if decision.confidence_assessment == ConfidenceAssessment.HIGH.value:
            self._high_confidence_decisions += 1

        if decision.fallback_strategy != "NONE":
            self._fallback_activations += 1

        # Update average decision time (exponential moving average)
        alpha = 0.1
        if self._average_decision_time_ms == 0.0:
            self._average_decision_time_ms = decision_time_ms
        else:
            self._average_decision_time_ms = (
                alpha * decision_time_ms + (1 - alpha) * self._average_decision_time_ms
            )

    def set_safety_override(self, active: bool, reason: Optional[str] = None) -> None:
        """Set safety override to prevent homing regardless of confidence."""
        self._safety_override_active = active
        self._safety_override_reason = reason

        if active:
            logger.warning(f"Safety override activated: {reason}")
        else:
            logger.info("Safety override deactivated")

    def configure_threshold_parameters(self, config: DynamicThresholdConfig) -> None:
        """Configure dynamic threshold parameters."""
        self._threshold_config = config
        logger.info("Dynamic threshold configuration updated")

    def get_confidence_metrics(self) -> Dict[str, Any]:
        """Get confidence-based decision metrics."""
        return {
            "decisions_made": self._decisions_made,
            "high_confidence_rate": (
                self._high_confidence_decisions / max(1, self._decisions_made)
            ),
            "fallback_activation_rate": (self._fallback_activations / max(1, self._decisions_made)),
            "average_decision_time_ms": self._average_decision_time_ms,
            "safety_override_active": self._safety_override_active,
            "current_threshold_config": {
                "high_quality": self._threshold_config.high_quality_threshold,
                "moderate_quality": self._threshold_config.moderate_quality_threshold,
                "low_quality": self._threshold_config.low_quality_threshold,
                "interference_penalty": self._threshold_config.interference_penalty_factor,
            },
        }

    def reset_metrics(self) -> None:
        """Reset all decision metrics."""
        self._decisions_made = 0
        self._high_confidence_decisions = 0
        self._fallback_activations = 0
        self._average_decision_time_ms = 0.0
        logger.info("Confidence decision metrics reset")


# Import math for fallback command generation
import math
