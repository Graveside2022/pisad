"""Unit tests for performance analytics service."""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from src.backend.services.performance_analytics import (
    ApproachMetrics,
    BaselineComparison,
    DetectionMetrics,
    EnvironmentalCorrelation,
    FalsePositiveNegativeAnalysis,
    PerformanceAnalytics,
    SearchMetrics,
)


@pytest.fixture
def analytics_service():
    """Create a performance analytics service instance."""
    return PerformanceAnalytics()


@pytest.fixture
def sample_telemetry_data():
    """Create sample telemetry data."""
    base_time = datetime.now()
    data = []
    for i in range(100):
        data.append(
            {
                "timestamp": (base_time + timedelta(seconds=i)).isoformat(),
                "latitude": 47.6062 + i * 0.0001,
                "longitude": -122.3321 + i * 0.00015,
                "altitude": 100.0 + i * 0.5,
                "groundspeed": 5.0,
                "airspeed": 5.5,
                "rssi_dbm": -80 + i * 0.2,
                "snr_db": 10.0,
                "beacon_detected": i > 50,
                "system_state": "APPROACHING" if i > 50 else "SEARCHING",
            }
        )
    return data


@pytest.fixture
def sample_detection_events():
    """Create sample detection events."""
    base_time = datetime.now()
    events = []
    for i in range(10, 60, 10):
        events.append(
            {
                "timestamp": (base_time + timedelta(seconds=i)).isoformat(),
                "frequency": 121500000,
                "rssi": -70 + i * 0.3,
                "snr": 12.0,
                "confidence": 75.0 + i * 0.3,
                "location": {
                    "lat": 47.6062 + i * 0.0001,
                    "lon": -122.3321 + i * 0.00015,
                },
                "state": "SEARCHING",
            }
        )
    return events


@pytest.fixture
def ground_truth_beacons():
    """Create ground truth beacon locations."""
    return [
        {"latitude": 47.612, "longitude": -122.320},
        {"latitude": 47.608, "longitude": -122.325},
    ]


def test_calculate_detection_metrics(
    analytics_service, sample_telemetry_data, sample_detection_events
):
    """Test detection metrics calculation."""
    metrics = analytics_service.calculate_detection_metrics(
        sample_telemetry_data, sample_detection_events, search_area_km2=2.0
    )

    assert isinstance(metrics, DetectionMetrics)
    assert metrics.total_detections == 5
    assert metrics.detections_per_hour > 0
    assert metrics.detections_per_km2 == 2.5
    assert metrics.first_detection_time is not None
    assert metrics.mean_detection_confidence > 0
    assert 0 <= metrics.detection_coverage <= 100


def test_calculate_detection_metrics_empty_data(analytics_service):
    """Test detection metrics with empty data."""
    metrics = analytics_service.calculate_detection_metrics([], [], 1.0)

    assert metrics.total_detections == 0
    assert metrics.detections_per_hour == 0
    assert metrics.detections_per_km2 == 0
    assert metrics.first_detection_time is None


def test_compute_approach_accuracy(analytics_service, sample_telemetry_data):
    """Test approach accuracy computation."""
    beacon_location = (47.615, -122.318)
    metrics = analytics_service.compute_approach_accuracy(sample_telemetry_data, beacon_location)

    assert isinstance(metrics, ApproachMetrics)
    assert metrics.final_distance_m is not None
    assert metrics.final_distance_m > 0
    assert metrics.approach_time_s is not None
    assert metrics.approach_efficiency > 0
    assert metrics.final_rssi_dbm is not None
    assert metrics.rssi_improvement_db > 0


def test_compute_approach_accuracy_no_beacon_detected(analytics_service):
    """Test approach accuracy when no beacon detected."""
    telemetry = [
        {
            "timestamp": datetime.now().isoformat(),
            "latitude": 47.6062,
            "longitude": -122.3321,
            "beacon_detected": False,
        }
    ]
    metrics = analytics_service.compute_approach_accuracy(telemetry, (47.615, -122.318))

    assert metrics.final_distance_m is None
    assert metrics.approach_time_s is None


def test_measure_search_efficiency(analytics_service, sample_telemetry_data):
    """Test search efficiency measurement."""
    metrics = analytics_service.measure_search_efficiency(sample_telemetry_data, 2.0)

    assert isinstance(metrics, SearchMetrics)
    assert metrics.total_area_km2 == 2.0
    assert metrics.area_covered_km2 > 0
    assert 0 <= metrics.coverage_percentage <= 100
    assert metrics.total_distance_km > 0
    assert metrics.search_time_minutes > 0
    assert metrics.average_speed_kmh > 0
    assert metrics.search_pattern_efficiency > 0


def test_measure_search_efficiency_empty_data(analytics_service):
    """Test search efficiency with empty data."""
    metrics = analytics_service.measure_search_efficiency([], 1.0)

    assert metrics.total_area_km2 == 1.0
    assert metrics.area_covered_km2 == 0
    assert metrics.coverage_percentage == 0
    assert metrics.total_distance_km == 0


def test_analyze_false_positives(analytics_service, sample_detection_events, ground_truth_beacons):
    """Test false positive analysis."""
    analysis = analytics_service.analyze_false_positives(
        sample_detection_events, ground_truth_beacons
    )

    assert isinstance(analysis, FalsePositiveNegativeAnalysis)
    assert analysis.true_positives >= 0
    assert analysis.false_positives >= 0
    assert analysis.false_negatives >= 0
    assert 0 <= analysis.precision <= 1
    assert 0 <= analysis.recall <= 1
    assert 0 <= analysis.f1_score <= 1


def test_correlate_environmental_factors(
    analytics_service, sample_telemetry_data, sample_detection_events
):
    """Test environmental correlation analysis."""
    weather_data = {
        "wind_speed_ms": 5.0,
        "precipitation_mm": 2.0,
    }
    correlation = analytics_service.correlate_environmental_factors(
        sample_telemetry_data, sample_detection_events, weather_data
    )

    assert isinstance(correlation, EnvironmentalCorrelation)
    assert -1 <= correlation.rf_noise_correlation <= 1
    assert 0 <= correlation.weather_impact_score <= 100
    assert 0 <= correlation.terrain_impact_score <= 100
    assert 0 <= correlation.time_of_day_impact <= 100
    assert correlation.altitude_correlation >= 0


def test_correlate_environmental_factors_no_weather(
    analytics_service, sample_telemetry_data, sample_detection_events
):
    """Test environmental correlation without weather data."""
    correlation = analytics_service.correlate_environmental_factors(
        sample_telemetry_data, sample_detection_events
    )

    assert correlation.weather_impact_score == 0
    # RF noise correlation will be 0 if there's no variance in SNR values
    assert correlation.rf_noise_correlation == 0.0


def test_compare_to_baseline(analytics_service):
    """Test baseline comparison."""
    mission_metrics = {
        "search_time_minutes": 60,
        "area_covered_km2": 3.0,
        "final_distance_m": 25,
    }
    comparison = analytics_service.compare_to_baseline(mission_metrics)

    assert isinstance(comparison, BaselineComparison)
    assert comparison.time_improvement_percent == 50.0
    assert comparison.area_reduction_percent == 40.0
    assert comparison.accuracy_improvement_percent == 50.0
    assert comparison.cost_reduction_percent > 0
    assert comparison.operator_workload_reduction > 0


def test_generate_performance_report(
    analytics_service,
    sample_telemetry_data,
    sample_detection_events,
    ground_truth_beacons,
):
    """Test complete performance report generation."""
    mission_id = uuid4()
    beacon_location = (47.615, -122.318)
    weather_data = {"wind_speed_ms": 3.0, "precipitation_mm": 0.5}

    report = analytics_service.generate_performance_report(
        mission_id=mission_id,
        telemetry_data=sample_telemetry_data,
        detection_events=sample_detection_events,
        search_area_km2=2.0,
        beacon_location=beacon_location,
        ground_truth_beacons=ground_truth_beacons,
        weather_data=weather_data,
    )

    assert report.mission_id == mission_id
    assert "total_detections" in report.detection_metrics
    assert "final_distance_m" in report.approach_metrics
    assert "total_area_km2" in report.search_metrics
    assert "precision" in report.false_positive_analysis
    assert "rf_noise_correlation" in report.environmental_correlation
    assert "time_improvement_percent" in report.baseline_comparison
    assert 0 <= report.overall_score <= 100
    assert isinstance(report.recommendations, list)
    assert len(report.recommendations) > 0


def test_calculate_distance(analytics_service):
    """Test GPS distance calculation."""
    # Test known distance between two points
    point1 = (47.6062, -122.3321)  # Seattle
    point2 = (47.6088, -122.3350)  # ~370m away
    distance = analytics_service._calculate_distance(point1, point2)

    assert 350 < distance < 400  # Should be roughly 370m


def test_calculate_path_distance(analytics_service):
    """Test path distance calculation."""
    telemetry = [
        {"latitude": 47.6062, "longitude": -122.3321},
        {"latitude": 47.6063, "longitude": -122.3322},
        {"latitude": 47.6064, "longitude": -122.3323},
    ]
    distance = analytics_service._calculate_path_distance(telemetry)

    assert distance > 0
    assert distance < 1000  # Should be less than 1km for these small movements


def test_calculate_covered_area(analytics_service):
    """Test covered area calculation."""
    telemetry = [
        {"latitude": 47.6062, "longitude": -122.3321},
        {"latitude": 47.6162, "longitude": -122.3321},  # ~1.1km north
        {"latitude": 47.6162, "longitude": -122.3421},  # ~0.9km west
        {"latitude": 47.6062, "longitude": -122.3421},  # ~1.1km south
    ]
    area = analytics_service._calculate_covered_area(telemetry)

    assert area > 0.5  # Should be roughly 1 km²
    assert area < 2.0


def test_calculate_overall_score(analytics_service):
    """Test overall score calculation."""
    detection = DetectionMetrics(mean_detection_confidence=80)
    approach = ApproachMetrics(approach_efficiency=75)
    search = SearchMetrics(search_pattern_efficiency=70)
    false_positive = FalsePositiveNegativeAnalysis(f1_score=0.85)
    baseline = BaselineComparison(time_improvement_percent=30, accuracy_improvement_percent=40)

    score = analytics_service._calculate_overall_score(
        detection, approach, search, false_positive, baseline
    )

    assert 0 <= score <= 100
    assert score > 50  # Should be decent score with these metrics


def test_generate_recommendations(analytics_service):
    """Test recommendation generation."""
    detection = DetectionMetrics(mean_detection_confidence=65, first_detection_time=400)
    approach = ApproachMetrics(approach_efficiency=60, rssi_improvement_db=5)
    search = SearchMetrics(coverage_percentage=70, search_pattern_efficiency=65)
    false_positive = FalsePositiveNegativeAnalysis(precision=0.7, recall=0.75)
    environmental = EnvironmentalCorrelation(rf_noise_correlation=0.3, weather_impact_score=60)

    recommendations = analytics_service._generate_recommendations(
        detection, approach, search, false_positive, environmental
    )

    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
    assert any("SDR gain" in r for r in recommendations)
    assert any("approach" in r.lower() for r in recommendations)


def test_baseline_data_loaded(analytics_service):
    """Test that baseline data is properly loaded."""
    assert analytics_service.baseline_data is not None
    assert "average_search_time_minutes" in analytics_service.baseline_data
    assert analytics_service.baseline_data["average_search_time_minutes"] > 0
    assert "average_area_covered_km2" in analytics_service.baseline_data
    assert "average_final_distance_m" in analytics_service.baseline_data
