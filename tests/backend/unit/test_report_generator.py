"""Unit tests for report generator service."""

import json
from pathlib import Path
from uuid import uuid4

import pytest

from src.backend.services.performance_analytics import MissionPerformanceMetrics
from src.backend.services.report_generator import (
    EmailConfig,
    ReportConfig,
    ReportGenerator,
)


@pytest.fixture
def report_generator():
    """Create a report generator instance."""
    return ReportGenerator()


@pytest.fixture
def sample_metrics():
    """Create sample mission performance metrics."""
    return MissionPerformanceMetrics(
        mission_id=uuid4(),
        detection_metrics={
            "total_detections": 15,
            "detections_per_hour": 8.5,
            "detections_per_km2": 7.5,
            "first_detection_time": 120.0,
            "mean_detection_confidence": 82.5,
            "detection_coverage": 78.0,
        },
        approach_metrics={
            "final_distance_m": 12.5,
            "approach_time_s": 180.0,
            "approach_efficiency": 85.0,
            "final_rssi_dbm": -45.0,
            "rssi_improvement_db": 25.0,
            "approach_velocity_ms": 2.5,
        },
        search_metrics={
            "total_area_km2": 2.0,
            "area_covered_km2": 1.6,
            "coverage_percentage": 80.0,
            "total_distance_km": 5.2,
            "search_time_minutes": 45.0,
            "average_speed_kmh": 7.0,
            "search_pattern_efficiency": 75.0,
        },
        false_positive_analysis={
            "false_positives": 2,
            "false_negatives": 1,
            "true_positives": 13,
            "true_negatives": 50,
            "precision": 0.867,
            "recall": 0.929,
            "f1_score": 0.897,
        },
        environmental_correlation={
            "rf_noise_correlation": 0.65,
            "weather_impact_score": 85.0,
            "terrain_impact_score": 70.0,
            "time_of_day_impact": 90.0,
            "altitude_correlation": 125.0,
        },
        baseline_comparison={
            "time_improvement_percent": 62.5,
            "area_reduction_percent": 40.0,
            "accuracy_improvement_percent": 75.0,
            "cost_reduction_percent": 55.0,
            "operator_workload_reduction": 60.0,
        },
        overall_score=78.5,
        recommendations=[
            "Optimize search pattern for better coverage",
            "Adjust SDR gain settings for improved detection",
            "Consider higher altitude for initial search phase",
        ],
    )


def test_generate_mission_summary(report_generator, sample_metrics):
    """Test mission summary generation."""
    summary = report_generator.generate_mission_summary(sample_metrics)

    assert "mission_id" in summary
    assert summary["overall_score"] == 78.5
    assert "key_metrics" in summary
    assert summary["key_metrics"]["total_detections"] == 15
    assert summary["key_metrics"]["detections_per_hour"] == 8.5
    assert summary["key_metrics"]["final_distance_m"] == 12.5
    assert summary["key_metrics"]["search_time_minutes"] == 45.0
    assert summary["key_metrics"]["area_covered_km2"] == 1.6
    assert summary["key_metrics"]["coverage_percentage"] == 80.0

    assert "performance_vs_baseline" in summary
    assert summary["performance_vs_baseline"]["time_improvement"] == "62.5%"
    assert summary["performance_vs_baseline"]["accuracy_improvement"] == "75.0%"
    assert summary["performance_vs_baseline"]["cost_reduction"] == "55.0%"

    assert "top_recommendations" in summary
    assert len(summary["top_recommendations"]) == 3


def test_create_performance_visualizations(report_generator, sample_metrics, tmp_path):
    """Test chart generation."""
    output_dir = tmp_path / "charts"
    chart_files = report_generator.create_performance_visualizations(sample_metrics, output_dir)

    assert len(chart_files) > 0
    for chart_file in chart_files:
        assert chart_file.exists()
        assert chart_file.suffix == ".png"

    # Check specific charts were created
    expected_charts = [
        "detection_metrics.png",
        "search_efficiency.png",
        "baseline_comparison.png",
        "performance_radar.png",
    ]
    created_files = [f.name for f in chart_files]
    for expected in expected_charts:
        assert expected in created_files


def test_generate_json_report(report_generator, sample_metrics, tmp_path):
    """Test JSON report generation."""
    output_path = tmp_path / "report.json"
    success = report_generator.generate_json_report(sample_metrics, output_path)

    assert success is True
    assert output_path.exists()

    # Verify JSON content
    with open(output_path) as f:
        report_data = json.load(f)

    assert "report_metadata" in report_data
    assert "generated_at" in report_data["report_metadata"]
    assert "report_version" in report_data["report_metadata"]

    assert "mission_summary" in report_data
    assert "detailed_metrics" in report_data

    # Check mission summary content
    summary = report_data["mission_summary"]
    assert summary["overall_score"] == 78.5
    assert summary["key_metrics"]["total_detections"] == 15


def test_generate_pdf_report(report_generator, sample_metrics, tmp_path):
    """Test PDF report generation."""
    output_path = tmp_path / "report.pdf"
    config = ReportConfig(include_charts=True, include_recommendations=True)

    success = report_generator.generate_pdf_report(sample_metrics, output_path, config)

    assert success is True
    assert output_path.exists()
    assert output_path.stat().st_size > 0  # PDF should have content


def test_generate_pdf_report_no_charts(report_generator, sample_metrics, tmp_path):
    """Test PDF report generation without charts."""
    output_path = tmp_path / "report_no_charts.pdf"
    config = ReportConfig(include_charts=False, include_recommendations=True)

    success = report_generator.generate_pdf_report(sample_metrics, output_path, config)

    assert success is True
    assert output_path.exists()


def test_generate_pdf_report_no_recommendations(report_generator, sample_metrics, tmp_path):
    """Test PDF report generation without recommendations."""
    output_path = tmp_path / "report_no_recs.pdf"
    config = ReportConfig(include_charts=False, include_recommendations=False)

    success = report_generator.generate_pdf_report(sample_metrics, output_path, config)

    assert success is True
    assert output_path.exists()


def test_email_config_validation():
    """Test email configuration validation."""
    config = EmailConfig(
        recipient="test@example.com",
        subject="Test Report",
        body="Test body",
        smtp_server="smtp.example.com",
        smtp_port=587,
        sender_email="sender@example.com",
    )

    assert config.recipient == "test@example.com"
    assert config.subject == "Test Report"
    assert config.smtp_port == 587


def test_report_config_defaults():
    """Test report configuration defaults."""
    config = ReportConfig()

    assert config.include_charts is True
    assert config.include_recommendations is True
    assert config.include_raw_data is False
    assert config.chart_dpi == 100
    assert config.page_size == "letter"


def test_custom_styles_setup(report_generator):
    """Test that custom styles are properly set up."""
    assert "CustomTitle" in report_generator.styles
    assert "CustomHeading" in report_generator.styles
    assert "CustomBody" in report_generator.styles


def test_create_detection_chart(report_generator, sample_metrics, tmp_path):
    """Test detection metrics chart creation."""
    output_dir = tmp_path / "charts"
    output_dir.mkdir()

    chart_path = report_generator._create_detection_chart(sample_metrics, output_dir)

    assert chart_path is not None
    assert chart_path.exists()
    assert chart_path.name == "detection_metrics.png"


def test_create_efficiency_chart(report_generator, sample_metrics, tmp_path):
    """Test search efficiency chart creation."""
    output_dir = tmp_path / "charts"
    output_dir.mkdir()

    chart_path = report_generator._create_efficiency_chart(sample_metrics, output_dir)

    assert chart_path is not None
    assert chart_path.exists()
    assert chart_path.name == "search_efficiency.png"


def test_create_comparison_chart(report_generator, sample_metrics, tmp_path):
    """Test baseline comparison chart creation."""
    output_dir = tmp_path / "charts"
    output_dir.mkdir()

    chart_path = report_generator._create_comparison_chart(sample_metrics, output_dir)

    assert chart_path is not None
    assert chart_path.exists()
    assert chart_path.name == "baseline_comparison.png"


def test_create_radar_chart(report_generator, sample_metrics, tmp_path):
    """Test radar chart creation."""
    output_dir = tmp_path / "charts"
    output_dir.mkdir()

    chart_path = report_generator._create_radar_chart(sample_metrics, output_dir)

    assert chart_path is not None
    assert chart_path.exists()
    assert chart_path.name == "performance_radar.png"


def test_generate_summary_with_missing_recommendations(report_generator):
    """Test summary generation when recommendations are missing."""
    metrics = MissionPerformanceMetrics(
        mission_id=uuid4(),
        detection_metrics={},
        approach_metrics={},
        search_metrics={},
        false_positive_analysis={},
        environmental_correlation={},
        baseline_comparison={},
        overall_score=50.0,
        recommendations=[],  # Empty recommendations
    )

    summary = report_generator.generate_mission_summary(metrics)
    assert summary["top_recommendations"] == []


def test_generate_pdf_with_invalid_path(report_generator, sample_metrics):
    """Test PDF generation with invalid output path."""
    invalid_path = Path("/invalid/path/report.pdf")
    success = report_generator.generate_pdf_report(sample_metrics, invalid_path)

    assert success is False


def test_generate_json_with_invalid_path(report_generator, sample_metrics):
    """Test JSON generation with invalid output path."""
    invalid_path = Path("/invalid/path/report.json")
    success = report_generator.generate_json_report(sample_metrics, invalid_path)

    assert success is False
