"""Unit tests for recommendations engine."""

import json
from uuid import uuid4

import pytest

from src.backend.services.performance_analytics import MissionPerformanceMetrics
from src.backend.services.recommendations_engine import (
    RecommendationPriority,
    RecommendationsEngine,
    RecommendationType,
)


@pytest.fixture
def recommendations_engine():
    """Create a recommendations engine instance."""
    return RecommendationsEngine()


@pytest.fixture
def sample_metrics_good():
    """Create sample metrics with good performance."""
    return MissionPerformanceMetrics(
        mission_id=uuid4(),
        detection_metrics={
            "detections_per_hour": 10.0,
            "mean_detection_confidence": 85.0,
        },
        approach_metrics={
            "approach_efficiency": 80.0,
            "final_rssi_dbm": -50.0,
        },
        search_metrics={
            "coverage_percentage": 85.0,
            "search_pattern_efficiency": 75.0,
            "total_area_km2": 2.0,
        },
        false_positive_analysis={
            "false_positives": 1,
            "precision": 0.9,
        },
        environmental_correlation={
            "rf_noise_correlation": 0.7,
        },
        baseline_comparison={},
        overall_score=80.0,
        recommendations=[],
    )


@pytest.fixture
def sample_metrics_poor():
    """Create sample metrics with poor performance."""
    return MissionPerformanceMetrics(
        mission_id=uuid4(),
        detection_metrics={
            "detections_per_hour": 3.0,
            "mean_detection_confidence": 60.0,
        },
        approach_metrics={
            "approach_efficiency": 50.0,
            "final_rssi_dbm": -75.0,
            "approach_time_s": 400.0,
        },
        search_metrics={
            "coverage_percentage": 60.0,
            "search_pattern_efficiency": 55.0,
            "total_area_km2": 5.0,
            "average_speed_kmh": 4.0,
        },
        false_positive_analysis={
            "false_positives": 5,
            "precision": 0.6,
        },
        environmental_correlation={
            "rf_noise_correlation": 0.3,
        },
        baseline_comparison={},
        overall_score=45.0,
        recommendations=[],
    )


def test_analyze_performance_data(recommendations_engine, sample_metrics_good, sample_metrics_poor):
    """Test performance pattern analysis."""
    metrics_list = [sample_metrics_poor] * 4 + [sample_metrics_good] * 2

    patterns = recommendations_engine.analyze_performance_data(metrics_list)

    assert len(patterns) > 0
    # Should identify low detection rate pattern (4 out of 6 missions)
    detection_pattern = next((p for p in patterns if p.pattern_type == "low_detection_rate"), None)
    assert detection_pattern is not None
    assert detection_pattern.frequency == 4
    assert detection_pattern.impact_score > 0

    # Should identify poor approach efficiency
    approach_pattern = next(
        (p for p in patterns if p.pattern_type == "poor_approach_efficiency"), None
    )
    assert approach_pattern is not None


def test_generate_parameter_recommendations_low_detection(
    recommendations_engine, sample_metrics_poor
):
    """Test parameter recommendations for low detection rate."""
    metrics_list = [sample_metrics_poor] * 3

    recommendations = recommendations_engine.generate_parameter_recommendations(metrics_list)

    assert len(recommendations) > 0

    # Should recommend SDR gain increase
    sdr_rec = next((r for r in recommendations if r.id == "param_sdr_gain"), None)
    assert sdr_rec is not None
    assert sdr_rec.type == RecommendationType.PARAMETER_TUNING
    assert sdr_rec.priority == RecommendationPriority.HIGH
    assert "sdr_gain" in sdr_rec.specific_parameters


def test_generate_parameter_recommendations_low_confidence(
    recommendations_engine, sample_metrics_poor
):
    """Test parameter recommendations for low confidence."""
    metrics_list = [sample_metrics_poor] * 3

    recommendations = recommendations_engine.generate_parameter_recommendations(metrics_list)

    # Should recommend detection threshold adjustment
    threshold_rec = next((r for r in recommendations if r.id == "param_detection_threshold"), None)
    assert threshold_rec is not None
    assert threshold_rec.priority == RecommendationPriority.MEDIUM
    assert "detection_threshold_dbm" in threshold_rec.specific_parameters


def test_generate_parameter_recommendations_low_coverage(
    recommendations_engine, sample_metrics_poor
):
    """Test parameter recommendations for low coverage."""
    metrics_list = [sample_metrics_poor] * 3

    recommendations = recommendations_engine.generate_parameter_recommendations(metrics_list)

    # Should recommend search pattern optimization
    pattern_rec = next((r for r in recommendations if r.id == "param_search_pattern"), None)
    assert pattern_rec is not None
    assert pattern_rec.priority == RecommendationPriority.HIGH
    assert "search_altitude_m" in pattern_rec.specific_parameters


def test_suggest_hardware_upgrades_weak_signal(recommendations_engine, sample_metrics_poor):
    """Test hardware upgrade suggestions for weak signals."""
    metrics_list = [sample_metrics_poor] * 3

    recommendations = recommendations_engine.suggest_hardware_upgrades(metrics_list)

    assert len(recommendations) > 0

    # Should recommend antenna upgrade
    antenna_rec = next((r for r in recommendations if r.id == "hw_antenna"), None)
    assert antenna_rec is not None
    assert antenna_rec.type == RecommendationType.HARDWARE_UPGRADE
    assert "antenna_type" in antenna_rec.specific_parameters


def test_suggest_hardware_upgrades_poor_snr(recommendations_engine):
    """Test hardware upgrade suggestions for poor SNR."""
    metrics = MissionPerformanceMetrics(
        mission_id=uuid4(),
        detection_metrics={},
        approach_metrics={"final_rssi_dbm": -80},
        search_metrics={"average_speed_kmh": 3.0},
        false_positive_analysis={},
        environmental_correlation={"rf_noise_correlation": 0.2},
        baseline_comparison={},
        overall_score=40.0,
        recommendations=[],
    )
    metrics_list = [metrics] * 3

    recommendations = recommendations_engine.suggest_hardware_upgrades(metrics_list)

    # Should recommend SDR upgrade
    sdr_rec = next((r for r in recommendations if r.id == "hw_sdr"), None)
    assert sdr_rec is not None
    assert sdr_rec.priority == RecommendationPriority.LOW


def test_identify_optimal_search_patterns_grid(recommendations_engine, sample_metrics_good):
    """Test search pattern identification for grid pattern."""
    metrics_list = [sample_metrics_good] * 5

    recommendations = recommendations_engine.identify_optimal_search_patterns(metrics_list)

    assert len(recommendations) > 0
    grid_rec = next((r for r in recommendations if "grid" in r.id.lower()), None)
    assert grid_rec is not None
    assert grid_rec.type == RecommendationType.SEARCH_PATTERN


def test_identify_optimal_search_patterns_terrain(recommendations_engine):
    """Test search pattern recommendations with terrain data."""
    metrics_list = [
        MissionPerformanceMetrics(
            mission_id=uuid4(),
            detection_metrics={},
            approach_metrics={},
            search_metrics={
                "search_pattern_efficiency": 65,
                "coverage_percentage": 70,
            },
            false_positive_analysis={},
            environmental_correlation={},
            baseline_comparison={},
            overall_score=60.0,
            recommendations=[],
        )
    ]

    terrain_data = {"type": "mountainous"}

    recommendations = recommendations_engine.identify_optimal_search_patterns(
        metrics_list, terrain_data
    )

    # Should recommend terrain-following pattern
    terrain_rec = next((r for r in recommendations if "terrain" in r.id.lower()), None)
    assert terrain_rec is not None
    assert terrain_rec.priority == RecommendationPriority.HIGH


def test_create_v2_feature_recommendations_ml(recommendations_engine):
    """Test v2.0 feature recommendations for ML features."""
    metrics = MissionPerformanceMetrics(
        mission_id=uuid4(),
        detection_metrics={},
        approach_metrics={"approach_time_s": 180},
        search_metrics={},
        false_positive_analysis={"false_positives": 5},
        environmental_correlation={},
        baseline_comparison={},
        overall_score=50.0,
        recommendations=[],
    )
    metrics_list = [metrics] * 5

    recommendations = recommendations_engine.create_v2_feature_recommendations(metrics_list)

    assert len(recommendations) > 0
    ml_rec = next((r for r in recommendations if "ml" in r.id.lower()), None)
    assert ml_rec is not None
    assert ml_rec.type == RecommendationType.FEATURE_REQUEST
    assert ml_rec.priority == RecommendationPriority.HIGH


def test_create_v2_feature_recommendations_autonomous(recommendations_engine):
    """Test v2.0 feature recommendations for autonomous features."""
    metrics = MissionPerformanceMetrics(
        mission_id=uuid4(),
        detection_metrics={},
        approach_metrics={"approach_time_s": 360},  # 6 minutes
        search_metrics={"total_area_km2": 2},
        false_positive_analysis={"false_positives": 1},
        environmental_correlation={},
        baseline_comparison={},
        overall_score=60.0,
        recommendations=[],
    )
    metrics_list = [metrics] * 3

    recommendations = recommendations_engine.create_v2_feature_recommendations(metrics_list)

    auto_rec = next((r for r in recommendations if "auto" in r.id.lower()), None)
    assert auto_rec is not None
    assert auto_rec.priority == RecommendationPriority.MEDIUM


def test_create_v2_feature_recommendations_multi_drone(recommendations_engine):
    """Test v2.0 feature recommendations for multi-drone coordination."""
    metrics = MissionPerformanceMetrics(
        mission_id=uuid4(),
        detection_metrics={},
        approach_metrics={"approach_time_s": 180},
        search_metrics={"total_area_km2": 15},  # Large area
        false_positive_analysis={"false_positives": 1},
        environmental_correlation={},
        baseline_comparison={},
        overall_score=70.0,
        recommendations=[],
    )
    metrics_list = [metrics] * 2

    recommendations = recommendations_engine.create_v2_feature_recommendations(metrics_list)

    drone_rec = next((r for r in recommendations if "drone" in r.id.lower()), None)
    assert drone_rec is not None
    assert drone_rec.priority == RecommendationPriority.LOW


def test_create_v2_feature_recommendations_with_feedback(recommendations_engine):
    """Test v2.0 feature recommendations with field feedback."""
    metrics_list = [
        MissionPerformanceMetrics(
            mission_id=uuid4(),
            detection_metrics={},
            approach_metrics={"approach_time_s": 180},
            search_metrics={},
            false_positive_analysis={"false_positives": 1},
            environmental_correlation={},
            baseline_comparison={},
            overall_score=70.0,
            recommendations=[],
        )
    ]

    field_feedback = ["Strong winds affected search", "Weather was challenging"]

    recommendations = recommendations_engine.create_v2_feature_recommendations(
        metrics_list, field_feedback
    )

    weather_rec = next((r for r in recommendations if "weather" in r.id.lower()), None)
    assert weather_rec is not None
    assert weather_rec.type == RecommendationType.FEATURE_REQUEST


def test_generate_system_recommendations(
    recommendations_engine, sample_metrics_good, sample_metrics_poor
):
    """Test comprehensive system recommendations generation."""
    metrics_list = [sample_metrics_poor] * 3 + [sample_metrics_good] * 2

    system_recs = recommendations_engine.generate_system_recommendations(metrics_list)

    assert system_recs.total_missions_analyzed == 5
    assert len(system_recs.common_patterns) > 0
    assert len(system_recs.parameter_recommendations) > 0
    assert len(system_recs.hardware_recommendations) > 0
    assert len(system_recs.search_pattern_recommendations) > 0
    assert len(system_recs.v2_feature_recommendations) > 0


def test_export_recommendations(recommendations_engine, tmp_path):
    """Test exporting recommendations to file."""
    metrics_list = [
        MissionPerformanceMetrics(
            mission_id=uuid4(),
            detection_metrics={"detections_per_hour": 5},
            approach_metrics={},
            search_metrics={},
            false_positive_analysis={},
            environmental_correlation={},
            baseline_comparison={},
            overall_score=60.0,
            recommendations=[],
        )
    ]

    system_recs = recommendations_engine.generate_system_recommendations(metrics_list)
    output_path = tmp_path / "recommendations.json"

    success = recommendations_engine.export_recommendations(system_recs, output_path)

    assert success is True
    assert output_path.exists()

    # Verify content
    with open(output_path) as f:
        data = json.load(f)
    assert data["total_missions_analyzed"] == 1


def test_patterns_database_loaded(recommendations_engine):
    """Test that patterns database is properly loaded."""
    assert recommendations_engine.patterns_database is not None
    assert "low_detection_rate" in recommendations_engine.patterns_database
    assert "poor_approach_efficiency" in recommendations_engine.patterns_database
    assert "low_coverage" in recommendations_engine.patterns_database
    assert "high_false_positives" in recommendations_engine.patterns_database


def test_parameter_thresholds_loaded(recommendations_engine):
    """Test that parameter thresholds are properly loaded."""
    assert recommendations_engine.parameter_thresholds is not None
    assert "sdr_gain" in recommendations_engine.parameter_thresholds
    assert "detection_threshold_dbm" in recommendations_engine.parameter_thresholds
    assert "search_altitude_m" in recommendations_engine.parameter_thresholds

    # Check structure
    sdr_gain = recommendations_engine.parameter_thresholds["sdr_gain"]
    assert "min" in sdr_gain
    assert "max" in sdr_gain
    assert "optimal" in sdr_gain


def test_recommendation_priority_ordering(recommendations_engine):
    """Test that recommendations are properly prioritized."""
    metrics_list = [
        MissionPerformanceMetrics(
            mission_id=uuid4(),
            detection_metrics={"detections_per_hour": 2, "mean_detection_confidence": 50},
            approach_metrics={"approach_efficiency": 40, "final_rssi_dbm": -85},
            search_metrics={"coverage_percentage": 50},
            false_positive_analysis={"false_positives": 10, "precision": 0.4},
            environmental_correlation={"rf_noise_correlation": 0.1},
            baseline_comparison={},
            overall_score=30.0,
            recommendations=[],
        )
    ] * 5

    param_recs = recommendations_engine.generate_parameter_recommendations(metrics_list)

    # Check that high priority recommendations exist
    high_priority = [r for r in param_recs if r.priority == RecommendationPriority.HIGH]
    assert len(high_priority) > 0

    # Check that recommendations have required fields
    for rec in param_recs:
        assert rec.id
        assert rec.type
        assert rec.priority
        assert rec.title
        assert rec.description
        assert rec.expected_improvement
        assert rec.implementation_effort in ["low", "medium", "high"]
