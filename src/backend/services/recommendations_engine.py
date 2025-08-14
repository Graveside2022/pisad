"""Recommendations engine for system improvement analysis."""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
from pydantic import BaseModel

from src.backend.services.performance_analytics import MissionPerformanceMetrics
from src.backend.utils.logging import get_logger

logger = get_logger(__name__)


class RecommendationType(str, Enum):
    """Types of recommendations."""

    PARAMETER_TUNING = "parameter_tuning"
    HARDWARE_UPGRADE = "hardware_upgrade"
    SEARCH_PATTERN = "search_pattern"
    OPERATIONAL = "operational"
    FEATURE_REQUEST = "feature_request"


class RecommendationPriority(str, Enum):
    """Recommendation priority levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Recommendation:
    """Individual recommendation."""

    id: str
    type: RecommendationType
    priority: RecommendationPriority
    title: str
    description: str
    expected_improvement: str
    implementation_effort: str  # "low", "medium", "high"
    affected_metrics: list[str] = field(default_factory=list)
    specific_parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformancePattern:
    """Identified performance pattern."""

    pattern_type: str
    frequency: int
    impact_score: float
    missions_affected: list[UUID]
    description: str


class SystemRecommendations(BaseModel):
    """System-wide recommendations."""

    total_missions_analyzed: int
    common_patterns: list[dict[str, Any]]
    parameter_recommendations: list[dict[str, Any]]
    hardware_recommendations: list[dict[str, Any]]
    search_pattern_recommendations: list[dict[str, Any]]
    v2_feature_recommendations: list[dict[str, Any]]
    critical_issues: list[dict[str, Any]]


class RecommendationsEngine:
    """Engine for generating system improvement recommendations."""

    def __init__(self) -> None:
        """Initialize the recommendations engine."""
        self.patterns_database = self._load_patterns_database()
        self.parameter_thresholds = self._load_parameter_thresholds()

    def _load_patterns_database(self) -> dict[str, Any]:
        """Load known performance patterns database."""
        return {
            "low_detection_rate": {
                "threshold": 5.0,  # detections per hour
                "recommendations": [
                    "Increase SDR gain settings",
                    "Adjust detection threshold",
                    "Improve antenna positioning",
                ],
            },
            "poor_approach_efficiency": {
                "threshold": 60.0,  # percentage
                "recommendations": [
                    "Tune approach algorithm parameters",
                    "Implement predictive path planning",
                    "Adjust approach velocity limits",
                ],
            },
            "low_coverage": {
                "threshold": 70.0,  # percentage
                "recommendations": [
                    "Increase search altitude",
                    "Optimize search pattern spacing",
                    "Extend mission duration",
                ],
            },
            "high_false_positives": {
                "threshold": 0.2,  # ratio
                "recommendations": [
                    "Implement better signal validation",
                    "Add frequency filtering",
                    "Improve SNR thresholds",
                ],
            },
        }

    def _load_parameter_thresholds(self) -> dict[str, Any]:
        """Load optimal parameter thresholds."""
        return {
            "sdr_gain": {"min": 20, "max": 40, "optimal": 30},
            "detection_threshold_dbm": {"min": -90, "max": -60, "optimal": -75},
            "search_altitude_m": {"min": 50, "max": 150, "optimal": 100},
            "search_speed_ms": {"min": 3, "max": 10, "optimal": 5},
            "approach_speed_ms": {"min": 1, "max": 5, "optimal": 2},
            "hover_radius_m": {"min": 5, "max": 20, "optimal": 10},
            "signal_confidence_threshold": {"min": 60, "max": 90, "optimal": 75},
        }

    def analyze_performance_data(
        self, metrics_list: list[MissionPerformanceMetrics]
    ) -> list[PerformancePattern]:
        """
        Analyze multiple missions to identify patterns.

        Args:
            metrics_list: List of mission performance metrics

        Returns:
            List of identified performance patterns
        """
        patterns = []

        # Detection rate pattern
        low_detection_missions = []
        for metrics in metrics_list:
            if (
                metrics.detection_metrics.get("detections_per_hour", 0)
                < self.patterns_database["low_detection_rate"]["threshold"]
            ):
                low_detection_missions.append(metrics.mission_id)

        if len(low_detection_missions) > len(metrics_list) * 0.3:
            patterns.append(
                PerformancePattern(
                    pattern_type="low_detection_rate",
                    frequency=len(low_detection_missions),
                    impact_score=0.8,
                    missions_affected=low_detection_missions,
                    description="Consistent low detection rates across missions",
                )
            )

        # Approach efficiency pattern
        poor_approach_missions = []
        for metrics in metrics_list:
            if (
                metrics.approach_metrics.get("approach_efficiency", 100)
                < self.patterns_database["poor_approach_efficiency"]["threshold"]
            ):
                poor_approach_missions.append(metrics.mission_id)

        if len(poor_approach_missions) > len(metrics_list) * 0.3:
            patterns.append(
                PerformancePattern(
                    pattern_type="poor_approach_efficiency",
                    frequency=len(poor_approach_missions),
                    impact_score=0.7,
                    missions_affected=poor_approach_missions,
                    description="Inefficient approach paths in multiple missions",
                )
            )

        # Coverage pattern
        low_coverage_missions = []
        for metrics in metrics_list:
            if (
                metrics.search_metrics.get("coverage_percentage", 100)
                < self.patterns_database["low_coverage"]["threshold"]
            ):
                low_coverage_missions.append(metrics.mission_id)

        if len(low_coverage_missions) > len(metrics_list) * 0.4:
            patterns.append(
                PerformancePattern(
                    pattern_type="low_coverage",
                    frequency=len(low_coverage_missions),
                    impact_score=0.6,
                    missions_affected=low_coverage_missions,
                    description="Insufficient search area coverage",
                )
            )

        return patterns

    def generate_parameter_recommendations(
        self, metrics_list: list[MissionPerformanceMetrics]
    ) -> list[Recommendation]:
        """
        Generate parameter tuning recommendations.

        Args:
            metrics_list: List of mission performance metrics

        Returns:
            List of parameter tuning recommendations
        """
        recommendations = []

        # Analyze average metrics
        avg_detection_rate = np.mean(
            [m.detection_metrics.get("detections_per_hour", 0) for m in metrics_list]
        )
        avg_confidence = np.mean(
            [m.detection_metrics.get("mean_detection_confidence", 0) for m in metrics_list]
        )
        # Removed unused avg_false_positive_rate calculation

        # SDR Gain recommendation
        if avg_detection_rate < 5.0:
            recommendations.append(
                Recommendation(
                    id="param_sdr_gain",
                    type=RecommendationType.PARAMETER_TUNING,
                    priority=RecommendationPriority.HIGH,
                    title="Increase SDR Gain",
                    description="Current detection rate is below optimal. Increasing SDR gain may improve signal detection.",
                    expected_improvement="30-50% increase in detection rate",
                    implementation_effort="low",
                    affected_metrics=["detection_rate", "signal_strength"],
                    specific_parameters={
                        "sdr_gain": {
                            "current": 25,
                            "recommended": 35,
                            "range": [30, 40],
                        }
                    },
                )
            )

        # Detection threshold recommendation
        if avg_confidence < 70:
            recommendations.append(
                Recommendation(
                    id="param_detection_threshold",
                    type=RecommendationType.PARAMETER_TUNING,
                    priority=RecommendationPriority.MEDIUM,
                    title="Adjust Detection Threshold",
                    description="Detection confidence is low. Fine-tuning the detection threshold may improve accuracy.",
                    expected_improvement="15-25% improvement in confidence",
                    implementation_effort="low",
                    affected_metrics=["detection_confidence", "false_positives"],
                    specific_parameters={
                        "detection_threshold_dbm": {
                            "current": -70,
                            "recommended": -75,
                            "range": [-80, -70],
                        }
                    },
                )
            )

        # Search parameters
        avg_coverage = np.mean(
            [m.search_metrics.get("coverage_percentage", 0) for m in metrics_list]
        )
        if avg_coverage < 75:
            recommendations.append(
                Recommendation(
                    id="param_search_pattern",
                    type=RecommendationType.PARAMETER_TUNING,
                    priority=RecommendationPriority.HIGH,
                    title="Optimize Search Pattern Parameters",
                    description="Search coverage is suboptimal. Adjusting pattern parameters can improve area coverage.",
                    expected_improvement="20-30% better coverage",
                    implementation_effort="medium",
                    affected_metrics=["coverage_percentage", "search_efficiency"],
                    specific_parameters={
                        "search_altitude_m": {
                            "current": 80,
                            "recommended": 100,
                        },
                        "search_line_spacing_m": {
                            "current": 50,
                            "recommended": 40,
                        },
                    },
                )
            )

        return recommendations

    def suggest_hardware_upgrades(
        self, metrics_list: list[MissionPerformanceMetrics]
    ) -> list[Recommendation]:
        """
        Suggest hardware upgrades based on limitations.

        Args:
            metrics_list: List of mission performance metrics

        Returns:
            List of hardware upgrade recommendations
        """
        recommendations = []

        # Analyze hardware limitations
        avg_rssi = np.mean([m.approach_metrics.get("final_rssi_dbm", -100) for m in metrics_list])
        avg_snr = np.mean(
            [m.environmental_correlation.get("rf_noise_correlation", 0) for m in metrics_list]
        )

        # Antenna upgrade
        if avg_rssi < -60:
            recommendations.append(
                Recommendation(
                    id="hw_antenna",
                    type=RecommendationType.HARDWARE_UPGRADE,
                    priority=RecommendationPriority.MEDIUM,
                    title="Upgrade to High-Gain Directional Antenna",
                    description="Signal reception is weak. A better antenna can significantly improve detection range.",
                    expected_improvement="10-15 dB signal improvement",
                    implementation_effort="medium",
                    affected_metrics=["rssi", "detection_range", "false_negatives"],
                    specific_parameters={
                        "antenna_type": "Yagi directional",
                        "gain": "9 dBi",
                        "frequency_range": "118-137 MHz",
                    },
                )
            )

        # SDR upgrade
        if avg_snr < 0.5:
            recommendations.append(
                Recommendation(
                    id="hw_sdr",
                    type=RecommendationType.HARDWARE_UPGRADE,
                    priority=RecommendationPriority.LOW,
                    title="Upgrade to Higher-Performance SDR",
                    description="Current SDR may have sensitivity limitations. Consider upgrading to a better model.",
                    expected_improvement="3-5 dB sensitivity improvement",
                    implementation_effort="high",
                    affected_metrics=["snr", "detection_sensitivity"],
                    specific_parameters={
                        "recommended_model": "HackRF One or USRP B200",
                        "sensitivity": "-120 dBm",
                        "dynamic_range": "70 dB",
                    },
                )
            )

        # Processing hardware
        cpu_usage = np.mean(
            [m.search_metrics.get("average_speed_kmh", 0) / 10 for m in metrics_list]
        )
        if cpu_usage < 0.7:  # Proxy for processing limitations
            recommendations.append(
                Recommendation(
                    id="hw_processor",
                    type=RecommendationType.HARDWARE_UPGRADE,
                    priority=RecommendationPriority.LOW,
                    title="Upgrade Processing Hardware",
                    description="Processing limitations may be affecting real-time performance.",
                    expected_improvement="Faster signal processing and decision making",
                    implementation_effort="high",
                    affected_metrics=["processing_latency", "real_time_performance"],
                    specific_parameters={
                        "recommended": "Raspberry Pi 5 8GB or Jetson Nano",
                        "current": "Raspberry Pi 4 4GB",
                    },
                )
            )

        return recommendations

    def identify_optimal_search_patterns(
        self,
        metrics_list: list[MissionPerformanceMetrics],
        terrain_data: dict[str, Any] | None = None,
    ) -> list[Recommendation]:
        """
        Identify optimal search patterns for different scenarios.

        Args:
            metrics_list: List of mission performance metrics
            terrain_data: Optional terrain information

        Returns:
            List of search pattern recommendations
        """
        recommendations = []

        # Analyze search efficiency by pattern type
        patterns_efficiency: dict[str, list[tuple[float, float]]] = {}
        for metrics in metrics_list:
            # Infer pattern type from metrics (simplified)
            efficiency = metrics.search_metrics.get("search_pattern_efficiency", 0)
            coverage = metrics.search_metrics.get("coverage_percentage", 0)

            if efficiency > 70 and coverage > 80:
                pattern_type = "grid"
            elif efficiency > 60:
                pattern_type = "spiral"
            else:
                pattern_type = "random"

            if pattern_type not in patterns_efficiency:
                patterns_efficiency[pattern_type] = []
            patterns_efficiency[pattern_type].append((efficiency, coverage))

        # Generate recommendations based on analysis
        best_pattern = max(
            patterns_efficiency.items(),
            key=lambda x: np.mean([e[0] for e in x[1]]) if x[1] else 0,
        )

        if best_pattern[0] == "grid":
            recommendations.append(
                Recommendation(
                    id="pattern_grid",
                    type=RecommendationType.SEARCH_PATTERN,
                    priority=RecommendationPriority.MEDIUM,
                    title="Use Grid Search Pattern",
                    description="Grid patterns show best performance for systematic area coverage.",
                    expected_improvement="Consistent 80%+ coverage",
                    implementation_effort="low",
                    affected_metrics=["coverage", "search_time"],
                    specific_parameters={
                        "pattern": "grid",
                        "line_spacing": "40m",
                        "altitude": "100m",
                        "speed": "5 m/s",
                    },
                )
            )
        elif best_pattern[0] == "spiral":
            recommendations.append(
                Recommendation(
                    id="pattern_spiral",
                    type=RecommendationType.SEARCH_PATTERN,
                    priority=RecommendationPriority.MEDIUM,
                    title="Use Expanding Spiral Pattern",
                    description="Spiral patterns are efficient for point-source searches.",
                    expected_improvement="Faster initial detection",
                    implementation_effort="low",
                    affected_metrics=["time_to_detection", "efficiency"],
                    specific_parameters={
                        "pattern": "spiral",
                        "initial_radius": "20m",
                        "expansion_rate": "10m per revolution",
                        "altitude": "80m",
                    },
                )
            )

        # Terrain-specific recommendations
        if terrain_data:
            if terrain_data.get("type") == "mountainous":
                recommendations.append(
                    Recommendation(
                        id="pattern_terrain_following",
                        type=RecommendationType.SEARCH_PATTERN,
                        priority=RecommendationPriority.HIGH,
                        title="Implement Terrain-Following Search",
                        description="Mountainous terrain requires adaptive altitude patterns.",
                        expected_improvement="Better signal detection in valleys",
                        implementation_effort="high",
                        affected_metrics=["detection_rate", "safety"],
                        specific_parameters={
                            "pattern": "terrain_following",
                            "altitude_agl": "100m",
                            "terrain_buffer": "50m",
                        },
                    )
                )

        return recommendations

    def create_v2_feature_recommendations(
        self, metrics_list: list[MissionPerformanceMetrics], field_feedback: list[str] | None = None
    ) -> list[Recommendation]:
        """
        Create recommendations for v2.0 features.

        Args:
            metrics_list: List of mission performance metrics
            field_feedback: Optional field operator feedback

        Returns:
            List of v2.0 feature recommendations
        """
        recommendations = []

        # Analyze common issues
        avg_false_positives = np.mean(
            [m.false_positive_analysis.get("false_positives", 0) for m in metrics_list]
        )
        avg_approach_time = np.mean(
            [m.approach_metrics.get("approach_time_s", 0) for m in metrics_list]
        )

        # Machine learning features
        if avg_false_positives > 2:
            recommendations.append(
                Recommendation(
                    id="v2_ml_detection",
                    type=RecommendationType.FEATURE_REQUEST,
                    priority=RecommendationPriority.HIGH,
                    title="Implement ML-Based Signal Classification",
                    description="Machine learning can significantly reduce false positives by learning signal patterns.",
                    expected_improvement="80% reduction in false positives",
                    implementation_effort="high",
                    affected_metrics=["false_positives", "detection_accuracy"],
                    specific_parameters={
                        "algorithm": "CNN for signal classification",
                        "training_data": "1000+ labeled signals",
                        "update_frequency": "weekly",
                    },
                )
            )

        # Autonomous features
        if avg_approach_time > 300:  # 5 minutes
            recommendations.append(
                Recommendation(
                    id="v2_auto_approach",
                    type=RecommendationType.FEATURE_REQUEST,
                    priority=RecommendationPriority.MEDIUM,
                    title="Add Fully Autonomous Approach Mode",
                    description="Implement AI-driven approach that requires no operator intervention.",
                    expected_improvement="50% reduction in approach time",
                    implementation_effort="high",
                    affected_metrics=["approach_time", "operator_workload"],
                    specific_parameters={
                        "features": [
                            "Predictive path planning",
                            "Obstacle avoidance",
                            "Auto-hover on signal lock",
                        ]
                    },
                )
            )

        # Multi-drone coordination
        total_area = sum([m.search_metrics.get("total_area_km2", 0) for m in metrics_list])
        if total_area > 10:  # Large search areas
            recommendations.append(
                Recommendation(
                    id="v2_multi_drone",
                    type=RecommendationType.FEATURE_REQUEST,
                    priority=RecommendationPriority.LOW,
                    title="Multi-Drone Coordination System",
                    description="Enable multiple drones to search cooperatively for faster coverage.",
                    expected_improvement="3x faster area coverage",
                    implementation_effort="high",
                    affected_metrics=["search_time", "coverage"],
                    specific_parameters={
                        "max_drones": 3,
                        "coordination": "distributed search sectors",
                        "communication": "mesh network",
                    },
                )
            )

        # Field feedback features
        if field_feedback:
            for feedback in field_feedback:
                if "weather" in feedback.lower():
                    recommendations.append(
                        Recommendation(
                            id="v2_weather_adaptation",
                            type=RecommendationType.FEATURE_REQUEST,
                            priority=RecommendationPriority.MEDIUM,
                            title="Weather-Adaptive Search Parameters",
                            description="Automatically adjust search parameters based on weather conditions.",
                            expected_improvement="Maintained performance in adverse weather",
                            implementation_effort="medium",
                            affected_metrics=["reliability", "weather_resilience"],
                            specific_parameters={
                                "sensors": "onboard weather station",
                                "adaptations": ["wind compensation", "rain mode"],
                            },
                        )
                    )
                    break

        return recommendations

    def generate_system_recommendations(
        self,
        metrics_list: list[MissionPerformanceMetrics],
        terrain_data: dict[str, Any] | None = None,
        field_feedback: list[str] | None = None,
    ) -> SystemRecommendations:
        """
        Generate comprehensive system recommendations.

        Args:
            metrics_list: List of mission performance metrics
            terrain_data: Optional terrain information
            field_feedback: Optional field operator feedback

        Returns:
            System-wide recommendations
        """
        # Identify patterns
        patterns = self.analyze_performance_data(metrics_list)

        # Generate all recommendation types
        param_recs = self.generate_parameter_recommendations(metrics_list)
        hardware_recs = self.suggest_hardware_upgrades(metrics_list)
        pattern_recs = self.identify_optimal_search_patterns(metrics_list, terrain_data)
        v2_recs = self.create_v2_feature_recommendations(metrics_list, field_feedback)

        # Identify critical issues
        critical_issues = []
        for rec in param_recs + hardware_recs + pattern_recs:
            if rec.priority == RecommendationPriority.CRITICAL:
                critical_issues.append(rec.__dict__)

        return SystemRecommendations(
            total_missions_analyzed=len(metrics_list),
            common_patterns=[p.__dict__ for p in patterns],
            parameter_recommendations=[r.__dict__ for r in param_recs],
            hardware_recommendations=[r.__dict__ for r in hardware_recs],
            search_pattern_recommendations=[r.__dict__ for r in pattern_recs],
            v2_feature_recommendations=[r.__dict__ for r in v2_recs],
            critical_issues=critical_issues,
        )

    def export_recommendations(
        self, recommendations: SystemRecommendations, output_path: Path
    ) -> bool:
        """
        Export recommendations to file.

        Args:
            recommendations: System recommendations
            output_path: Output file path

        Returns:
            True if export successful
        """
        try:
            with open(output_path, "w") as f:
                json.dump(recommendations.model_dump(), f, indent=2, default=str)
            logger.info(f"Exported recommendations to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export recommendations: {e}")
            return False
