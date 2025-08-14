"""Comprehensive tests for recommendations engine."""

import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from uuid import UUID, uuid4

import numpy as np
import pytest

from src.backend.services.performance_analytics import MissionPerformanceMetrics
from src.backend.services.recommendations_engine import (
    PerformancePattern,
    Recommendation,
    RecommendationPriority,
    RecommendationType,
    RecommendationsEngine,
    SystemRecommendations,
)


class TestRecommendationEnums:
    """Test recommendation enumerations."""
    
    def test_recommendation_type_enum(self):
        """Test RecommendationType enum values."""
        assert RecommendationType.PARAMETER_TUNING.value == "parameter_tuning"
        assert RecommendationType.HARDWARE_UPGRADE.value == "hardware_upgrade"
        assert RecommendationType.SEARCH_PATTERN.value == "search_pattern"
        assert RecommendationType.OPERATIONAL.value == "operational"
        assert RecommendationType.FEATURE_REQUEST.value == "feature_request"
    
    def test_recommendation_priority_enum(self):
        """Test RecommendationPriority enum values."""
        assert RecommendationPriority.CRITICAL.value == "critical"
        assert RecommendationPriority.HIGH.value == "high"
        assert RecommendationPriority.MEDIUM.value == "medium"
        assert RecommendationPriority.LOW.value == "low"


class TestRecommendation:
    """Test Recommendation dataclass."""
    
    def test_recommendation_creation(self):
        """Test creating a recommendation."""
        rec = Recommendation(
            id="rec-001",
            type=RecommendationType.PARAMETER_TUNING,
            priority=RecommendationPriority.HIGH,
            title="Increase SDR Gain",
            description="Current gain settings are too low for reliable detection",
            expected_improvement="30% increase in detection range",
            implementation_effort="low",
            affected_metrics=["detection_rate", "signal_quality"],
            specific_parameters={"sdr_gain": 40, "current_gain": 30}
        )
        
        assert rec.id == "rec-001"
        assert rec.type == RecommendationType.PARAMETER_TUNING
        assert rec.priority == RecommendationPriority.HIGH
        assert rec.title == "Increase SDR Gain"
        assert len(rec.affected_metrics) == 2
        assert rec.specific_parameters["sdr_gain"] == 40
    
    def test_recommendation_defaults(self):
        """Test recommendation with default values."""
        rec = Recommendation(
            id="rec-002",
            type=RecommendationType.OPERATIONAL,
            priority=RecommendationPriority.MEDIUM,
            title="Test",
            description="Test description",
            expected_improvement="None",
            implementation_effort="medium"
        )
        
        assert rec.affected_metrics == []
        assert rec.specific_parameters == {}


class TestPerformancePattern:
    """Test PerformancePattern dataclass."""
    
    def test_pattern_creation(self):
        """Test creating a performance pattern."""
        mission_ids = [uuid4() for _ in range(3)]
        pattern = PerformancePattern(
            pattern_type="low_detection_rate",
            frequency=5,
            impact_score=8.5,
            missions_affected=mission_ids,
            description="Consistently low beacon detection rate"
        )
        
        assert pattern.pattern_type == "low_detection_rate"
        assert pattern.frequency == 5
        assert pattern.impact_score == 8.5
        assert len(pattern.missions_affected) == 3
        assert all(isinstance(m, UUID) for m in pattern.missions_affected)


class TestSystemRecommendations:
    """Test SystemRecommendations model."""
    
    def test_system_recommendations_creation(self):
        """Test creating system recommendations."""
        recs = SystemRecommendations(
            total_missions_analyzed=10,
            common_patterns=[
                {"pattern": "low_detection", "count": 3}
            ],
            parameter_recommendations=[
                {"param": "gain", "value": 40}
            ],
            hardware_recommendations=[
                {"component": "antenna", "upgrade": "directional"}
            ],
            search_pattern_recommendations=[
                {"pattern": "spiral", "reason": "better coverage"}
            ],
            v2_feature_recommendations=[
                {"feature": "auto_gain", "benefit": "adaptive detection"}
            ],
            critical_issues=[
                {"issue": "battery_drain", "impact": "mission_abort"}
            ]
        )
        
        assert recs.total_missions_analyzed == 10
        assert len(recs.common_patterns) == 1
        assert len(recs.parameter_recommendations) == 1
        assert len(recs.hardware_recommendations) == 1
        assert len(recs.search_pattern_recommendations) == 1
        assert len(recs.v2_feature_recommendations) == 1
        assert len(recs.critical_issues) == 1
    
    def test_system_recommendations_json_serialization(self):
        """Test JSON serialization of system recommendations."""
        recs = SystemRecommendations(
            total_missions_analyzed=5,
            common_patterns=[],
            parameter_recommendations=[],
            hardware_recommendations=[],
            search_pattern_recommendations=[],
            v2_feature_recommendations=[],
            critical_issues=[]
        )
        
        json_str = recs.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["total_missions_analyzed"] == 5


class TestRecommendationsEngine:
    """Test RecommendationsEngine class."""
    
    @pytest.fixture
    def engine(self):
        """Create recommendations engine instance."""
        return RecommendationsEngine()
    
    def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert engine.patterns_database is not None
        assert engine.parameter_thresholds is not None
        assert "low_detection_rate" in engine.patterns_database
        assert "poor_approach_efficiency" in engine.patterns_database
    
    def test_load_patterns_database(self, engine):
        """Test loading patterns database."""
        patterns = engine._load_patterns_database()
        
        assert "low_detection_rate" in patterns
        assert patterns["low_detection_rate"]["threshold"] == 5.0
        assert len(patterns["low_detection_rate"]["recommendations"]) > 0
        
        assert "poor_approach_efficiency" in patterns
        assert patterns["poor_approach_efficiency"]["threshold"] == 60.0
    
    def test_load_parameter_thresholds(self, engine):
        """Test loading parameter thresholds."""
        thresholds = engine._load_parameter_thresholds()
        
        assert "sdr_gain" in thresholds
        assert thresholds["sdr_gain"]["min"] == 20
        assert thresholds["sdr_gain"]["max"] == 60
        assert thresholds["sdr_gain"]["optimal"] == 40
        
        assert "detection_threshold" in thresholds
        assert "approach_velocity" in thresholds
    
    def test_analyze_mission_metrics(self, engine):
        """Test analyzing individual mission metrics."""
        metrics = MissionPerformanceMetrics(
            mission_id=uuid4(),
            detection_rate=3.0,  # Below threshold
            approach_efficiency=50.0,  # Below threshold
            signal_quality_consistency=0.6,
            search_pattern_coverage=0.7,
            false_positive_rate=0.2,
            response_time_avg=5.0,
            environmental_factors={}
        )
        
        recommendations = engine.analyze_mission_metrics(metrics)
        
        assert len(recommendations) > 0
        # Should have recommendations for low detection rate
        detection_recs = [r for r in recommendations if "detection" in r.title.lower()]
        assert len(detection_recs) > 0
        
        # Should have recommendations for poor approach efficiency
        approach_recs = [r for r in recommendations if "approach" in r.title.lower()]
        assert len(approach_recs) > 0
    
    def test_analyze_mission_with_good_metrics(self, engine):
        """Test analyzing mission with good metrics."""
        metrics = MissionPerformanceMetrics(
            mission_id=uuid4(),
            detection_rate=10.0,  # Good
            approach_efficiency=85.0,  # Good
            signal_quality_consistency=0.9,
            search_pattern_coverage=0.95,
            false_positive_rate=0.05,
            response_time_avg=2.0,
            environmental_factors={}
        )
        
        recommendations = engine.analyze_mission_metrics(metrics)
        
        # Should have fewer or no critical recommendations
        critical_recs = [r for r in recommendations if r.priority == RecommendationPriority.CRITICAL]
        assert len(critical_recs) == 0
    
    def test_identify_patterns(self, engine):
        """Test identifying patterns across missions."""
        metrics_list = []
        for _ in range(5):
            metrics = MissionPerformanceMetrics(
                mission_id=uuid4(),
                detection_rate=2.0,  # Consistently low
                approach_efficiency=75.0,
                signal_quality_consistency=0.7,
                search_pattern_coverage=0.8,
                false_positive_rate=0.1,
                response_time_avg=3.0,
                environmental_factors={}
            )
            metrics_list.append(metrics)
        
        patterns = engine.identify_patterns(metrics_list)
        
        assert len(patterns) > 0
        # Should identify low detection rate pattern
        low_detection_patterns = [p for p in patterns if p.pattern_type == "low_detection_rate"]
        assert len(low_detection_patterns) > 0
        assert low_detection_patterns[0].frequency == 5
    
    def test_identify_mixed_patterns(self, engine):
        """Test identifying patterns with mixed performance."""
        metrics_list = []
        
        # Some with low detection
        for _ in range(3):
            metrics = MissionPerformanceMetrics(
                mission_id=uuid4(),
                detection_rate=2.0,
                approach_efficiency=80.0,
                signal_quality_consistency=0.7,
                search_pattern_coverage=0.8,
                false_positive_rate=0.1,
                response_time_avg=3.0,
                environmental_factors={}
            )
            metrics_list.append(metrics)
        
        # Some with poor approach
        for _ in range(2):
            metrics = MissionPerformanceMetrics(
                mission_id=uuid4(),
                detection_rate=8.0,
                approach_efficiency=40.0,
                signal_quality_consistency=0.7,
                search_pattern_coverage=0.8,
                false_positive_rate=0.1,
                response_time_avg=3.0,
                environmental_factors={}
            )
            metrics_list.append(metrics)
        
        patterns = engine.identify_patterns(metrics_list)
        
        assert len(patterns) >= 2
        pattern_types = [p.pattern_type for p in patterns]
        assert "low_detection_rate" in pattern_types
        assert "poor_approach_efficiency" in pattern_types
    
    def test_generate_parameter_recommendations(self, engine):
        """Test generating parameter tuning recommendations."""
        patterns = [
            PerformancePattern(
                pattern_type="low_detection_rate",
                frequency=5,
                impact_score=8.0,
                missions_affected=[uuid4() for _ in range(5)],
                description="Low detection"
            )
        ]
        
        recommendations = engine.generate_parameter_recommendations(patterns)
        
        assert len(recommendations) > 0
        # Should recommend gain adjustment
        gain_recs = [r for r in recommendations if "gain" in r.title.lower()]
        assert len(gain_recs) > 0
        assert gain_recs[0].type == RecommendationType.PARAMETER_TUNING
    
    def test_generate_hardware_recommendations(self, engine):
        """Test generating hardware upgrade recommendations."""
        patterns = [
            PerformancePattern(
                pattern_type="poor_signal_quality",
                frequency=8,
                impact_score=9.0,
                missions_affected=[uuid4() for _ in range(8)],
                description="Consistently poor signal quality"
            )
        ]
        
        recommendations = engine.generate_hardware_recommendations(patterns)
        
        assert len(recommendations) > 0
        # Should recommend hardware upgrades
        assert any(r.type == RecommendationType.HARDWARE_UPGRADE for r in recommendations)
    
    def test_generate_search_pattern_recommendations(self, engine):
        """Test generating search pattern recommendations."""
        metrics_list = []
        for _ in range(3):
            metrics = MissionPerformanceMetrics(
                mission_id=uuid4(),
                detection_rate=7.0,
                approach_efficiency=75.0,
                signal_quality_consistency=0.7,
                search_pattern_coverage=0.4,  # Poor coverage
                false_positive_rate=0.1,
                response_time_avg=3.0,
                environmental_factors={}
            )
            metrics_list.append(metrics)
        
        recommendations = engine.generate_search_pattern_recommendations(metrics_list)
        
        assert len(recommendations) > 0
        assert any(r.type == RecommendationType.SEARCH_PATTERN for r in recommendations)
    
    def test_generate_v2_features(self, engine):
        """Test generating v2 feature recommendations."""
        patterns = [
            PerformancePattern(
                pattern_type="low_detection_rate",
                frequency=10,
                impact_score=9.5,
                missions_affected=[uuid4() for _ in range(10)],
                description="Persistent detection issues"
            )
        ]
        
        features = engine.generate_v2_features(patterns)
        
        assert len(features) > 0
        assert any(r.type == RecommendationType.FEATURE_REQUEST for r in features)
    
    def test_generate_system_recommendations(self, engine):
        """Test generating complete system recommendations."""
        metrics_list = []
        for i in range(10):
            metrics = MissionPerformanceMetrics(
                mission_id=uuid4(),
                detection_rate=3.0 if i < 5 else 8.0,
                approach_efficiency=50.0 if i < 3 else 80.0,
                signal_quality_consistency=0.6,
                search_pattern_coverage=0.7,
                false_positive_rate=0.15,
                response_time_avg=4.0,
                environmental_factors={"wind_speed": 10.0}
            )
            metrics_list.append(metrics)
        
        system_recs = engine.generate_system_recommendations(metrics_list)
        
        assert system_recs.total_missions_analyzed == 10
        assert len(system_recs.common_patterns) > 0
        assert len(system_recs.parameter_recommendations) > 0
        assert len(system_recs.critical_issues) >= 0
    
    def test_generate_recommendations_empty_list(self, engine):
        """Test generating recommendations with no missions."""
        system_recs = engine.generate_system_recommendations([])
        
        assert system_recs.total_missions_analyzed == 0
        assert len(system_recs.common_patterns) == 0
        assert len(system_recs.parameter_recommendations) == 0
    
    def test_prioritize_recommendations(self, engine):
        """Test recommendation prioritization."""
        recommendations = [
            Recommendation(
                id="1",
                type=RecommendationType.PARAMETER_TUNING,
                priority=RecommendationPriority.LOW,
                title="Minor tuning",
                description="Small improvement",
                expected_improvement="5%",
                implementation_effort="low"
            ),
            Recommendation(
                id="2",
                type=RecommendationType.HARDWARE_UPGRADE,
                priority=RecommendationPriority.CRITICAL,
                title="Critical upgrade",
                description="Essential for operation",
                expected_improvement="50%",
                implementation_effort="high"
            ),
            Recommendation(
                id="3",
                type=RecommendationType.OPERATIONAL,
                priority=RecommendationPriority.HIGH,
                title="Important change",
                description="Significant improvement",
                expected_improvement="20%",
                implementation_effort="medium"
            )
        ]
        
        sorted_recs = engine.prioritize_recommendations(recommendations)
        
        assert sorted_recs[0].priority == RecommendationPriority.CRITICAL
        assert sorted_recs[1].priority == RecommendationPriority.HIGH
        assert sorted_recs[2].priority == RecommendationPriority.LOW
    
    def test_export_recommendations(self, engine, tmp_path):
        """Test exporting recommendations to file."""
        system_recs = SystemRecommendations(
            total_missions_analyzed=5,
            common_patterns=[{"pattern": "test"}],
            parameter_recommendations=[{"param": "gain"}],
            hardware_recommendations=[],
            search_pattern_recommendations=[],
            v2_feature_recommendations=[],
            critical_issues=[]
        )
        
        output_file = tmp_path / "recommendations.json"
        engine.export_recommendations(system_recs, output_file)
        
        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)
        assert data["total_missions_analyzed"] == 5
    
    def test_calculate_impact_score(self, engine):
        """Test impact score calculation."""
        pattern = PerformancePattern(
            pattern_type="low_detection_rate",
            frequency=5,
            impact_score=0.0,  # Will be calculated
            missions_affected=[uuid4() for _ in range(5)],
            description="Test pattern"
        )
        
        score = engine.calculate_impact_score(pattern)
        
        assert score > 0
        assert score <= 10
    
    def test_recommendation_deduplication(self, engine):
        """Test that duplicate recommendations are removed."""
        metrics = MissionPerformanceMetrics(
            mission_id=uuid4(),
            detection_rate=2.0,  # Will trigger multiple similar recommendations
            approach_efficiency=50.0,
            signal_quality_consistency=0.5,
            search_pattern_coverage=0.5,
            false_positive_rate=0.3,
            response_time_avg=6.0,
            environmental_factors={}
        )
        
        recommendations = engine.analyze_mission_metrics(metrics)
        
        # Check for duplicates by title
        titles = [r.title for r in recommendations]
        assert len(titles) == len(set(titles))  # No duplicates
    
    def test_environmental_factor_consideration(self, engine):
        """Test that environmental factors affect recommendations."""
        metrics = MissionPerformanceMetrics(
            mission_id=uuid4(),
            detection_rate=4.0,
            approach_efficiency=60.0,
            signal_quality_consistency=0.6,
            search_pattern_coverage=0.7,
            false_positive_rate=0.2,
            response_time_avg=4.0,
            environmental_factors={
                "wind_speed": 25.0,  # High wind
                "temperature": 45.0,  # High temperature
                "humidity": 90.0  # High humidity
            }
        )
        
        recommendations = engine.analyze_mission_metrics(metrics)
        
        # Should have environmental-related recommendations
        env_recs = [r for r in recommendations if "environmental" in r.description.lower() 
                    or "weather" in r.description.lower()
                    or "conditions" in r.description.lower()]
        assert len(env_recs) > 0