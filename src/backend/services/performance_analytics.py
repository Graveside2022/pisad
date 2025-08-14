"""Performance analytics service for mission data analysis."""

import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

import numpy as np
from pydantic import BaseModel, Field

from src.backend.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DetectionMetrics:
    """Detection rate metrics."""

    total_detections: int = 0
    detections_per_hour: float = 0.0
    detections_per_km2: float = 0.0
    first_detection_time: float | None = None  # Seconds from start
    mean_detection_confidence: float = 0.0
    detection_coverage: float = 0.0  # Percentage of search area


@dataclass
class ApproachMetrics:
    """Approach accuracy metrics."""

    final_distance_m: float | None = None
    approach_time_s: float | None = None
    approach_efficiency: float = 0.0  # Direct path vs actual path
    final_rssi_dbm: float | None = None
    rssi_improvement_db: float = 0.0
    approach_velocity_ms: float = 0.0


@dataclass
class SearchMetrics:
    """Search efficiency metrics."""

    total_area_km2: float = 0.0
    area_covered_km2: float = 0.0
    coverage_percentage: float = 0.0
    total_distance_km: float = 0.0
    search_time_minutes: float = 0.0
    average_speed_kmh: float = 0.0
    search_pattern_efficiency: float = 0.0


@dataclass
class FalsePositiveNegativeAnalysis:
    """False positive/negative detection analysis."""

    false_positives: int = 0
    false_negatives: int = 0
    true_positives: int = 0
    true_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0


@dataclass
class EnvironmentalCorrelation:
    """Environmental correlation analysis."""

    rf_noise_correlation: float = 0.0
    weather_impact_score: float = 0.0
    terrain_impact_score: float = 0.0
    time_of_day_impact: float = 0.0
    altitude_correlation: float = 0.0


@dataclass
class BaselineComparison:
    """Comparison metrics against baseline methods."""

    time_improvement_percent: float = 0.0
    area_reduction_percent: float = 0.0
    accuracy_improvement_percent: float = 0.0
    cost_reduction_percent: float = 0.0
    operator_workload_reduction: float = 0.0


class MissionPerformanceMetrics(BaseModel):
    """Complete mission performance metrics."""

    mission_id: UUID
    detection_metrics: dict[str, Any]
    approach_metrics: dict[str, Any]
    search_metrics: dict[str, Any]
    false_positive_analysis: dict[str, Any]
    environmental_correlation: dict[str, Any]
    baseline_comparison: dict[str, Any]
    overall_score: float = Field(ge=0, le=100)
    recommendations: list[str] = Field(default_factory=list)


class PerformanceAnalytics:
    """Service for analyzing mission performance data."""

    def __init__(self) -> None:
        """Initialize the performance analytics service."""
        self.baseline_data = self._load_baseline_data()

    def _load_baseline_data(self) -> dict[str, float]:
        """Load baseline manual search method data."""
        return {
            "average_search_time_minutes": 120.0,
            "average_area_covered_km2": 5.0,
            "average_final_distance_m": 50.0,
            "average_operator_hours": 2.0,
            "average_fuel_cost_usd": 50.0,
        }

    def calculate_detection_metrics(
        self,
        telemetry_data: list[dict[str, Any]],
        detection_events: list[dict[str, Any]],
        search_area_km2: float,
    ) -> DetectionMetrics:
        """
        Calculate detection rate metrics.

        Args:
            telemetry_data: List of telemetry frames
            detection_events: List of signal detection events
            search_area_km2: Total search area in km²

        Returns:
            Detection metrics
        """
        metrics = DetectionMetrics()

        if not telemetry_data:
            return metrics

        # Calculate time range
        start_time = datetime.fromisoformat(telemetry_data[0]["timestamp"])
        end_time = datetime.fromisoformat(telemetry_data[-1]["timestamp"])
        duration_hours = (end_time - start_time).total_seconds() / 3600

        # Detection counts and rates
        metrics.total_detections = len(detection_events)
        if duration_hours > 0:
            metrics.detections_per_hour = metrics.total_detections / duration_hours

        if search_area_km2 > 0:
            metrics.detections_per_km2 = metrics.total_detections / search_area_km2

        # First detection time
        if detection_events:
            first_detection_time = datetime.fromisoformat(detection_events[0]["timestamp"])
            metrics.first_detection_time = (first_detection_time - start_time).total_seconds()

            # Mean confidence
            confidences = [d.get("confidence", 0) for d in detection_events]
            metrics.mean_detection_confidence = statistics.mean(confidences) if confidences else 0

        # Detection coverage (approximate based on telemetry positions)
        covered_area = self._calculate_covered_area(telemetry_data)
        if search_area_km2 > 0:
            metrics.detection_coverage = min(100, (covered_area / search_area_km2) * 100)

        return metrics

    def compute_approach_accuracy(
        self,
        telemetry_data: list[dict[str, Any]],
        beacon_location: tuple[float, float] | None = None,
    ) -> ApproachMetrics:
        """
        Compute approach accuracy statistics.

        Args:
            telemetry_data: List of telemetry frames
            beacon_location: Known beacon location (lat, lon)

        Returns:
            Approach metrics
        """
        metrics = ApproachMetrics()

        if not telemetry_data or not beacon_location:
            return metrics

        # Find approach phase (when beacon was detected)
        approach_start_idx = None
        for i, frame in enumerate(telemetry_data):
            if frame.get("beacon_detected", False):
                approach_start_idx = i
                break

        if approach_start_idx is None:
            return metrics

        # Get approach phase data
        approach_data = telemetry_data[approach_start_idx:]
        if not approach_data:
            return metrics

        # Calculate final distance
        final_frame = approach_data[-1]
        final_lat = final_frame.get("latitude", 0)
        final_lon = final_frame.get("longitude", 0)
        metrics.final_distance_m = self._calculate_distance((final_lat, final_lon), beacon_location)

        # Approach time
        start_time = datetime.fromisoformat(approach_data[0]["timestamp"])
        end_time = datetime.fromisoformat(approach_data[-1]["timestamp"])
        metrics.approach_time_s = (end_time - start_time).total_seconds()

        # Approach efficiency (direct path vs actual path)
        initial_distance = self._calculate_distance(
            (approach_data[0]["latitude"], approach_data[0]["longitude"]),
            beacon_location,
        )
        actual_path_distance = self._calculate_path_distance(approach_data)
        if actual_path_distance > 0:
            metrics.approach_efficiency = min(100, (initial_distance / actual_path_distance) * 100)

        # RSSI metrics
        initial_rssi = approach_data[0].get("rssi_dbm", -100)
        metrics.final_rssi_dbm = final_frame.get("rssi_dbm", -100)
        metrics.rssi_improvement_db = metrics.final_rssi_dbm - initial_rssi

        # Average approach velocity
        if metrics.approach_time_s > 0:
            metrics.approach_velocity_ms = actual_path_distance / metrics.approach_time_s

        return metrics

    def measure_search_efficiency(
        self, telemetry_data: list[dict[str, Any]], search_area_km2: float
    ) -> SearchMetrics:
        """
        Measure search efficiency metrics.

        Args:
            telemetry_data: List of telemetry frames
            search_area_km2: Total search area in km²

        Returns:
            Search efficiency metrics
        """
        metrics = SearchMetrics()
        metrics.total_area_km2 = search_area_km2

        if not telemetry_data:
            return metrics

        # Calculate covered area
        metrics.area_covered_km2 = self._calculate_covered_area(telemetry_data)
        if search_area_km2 > 0:
            metrics.coverage_percentage = min(
                100, (metrics.area_covered_km2 / search_area_km2) * 100
            )

        # Total distance traveled
        metrics.total_distance_km = self._calculate_path_distance(telemetry_data) / 1000

        # Search time
        start_time = datetime.fromisoformat(telemetry_data[0]["timestamp"])
        end_time = datetime.fromisoformat(telemetry_data[-1]["timestamp"])
        metrics.search_time_minutes = (end_time - start_time).total_seconds() / 60

        # Average speed
        if metrics.search_time_minutes > 0:
            metrics.average_speed_kmh = metrics.total_distance_km / (
                metrics.search_time_minutes / 60
            )

        # Search pattern efficiency (ideal grid vs actual path)
        ideal_grid_distance = self._calculate_ideal_grid_distance(search_area_km2)
        if ideal_grid_distance > 0:
            metrics.search_pattern_efficiency = min(
                100, (ideal_grid_distance / metrics.total_distance_km) * 100
            )

        return metrics

    def analyze_false_positives(
        self,
        detection_events: list[dict[str, Any]],
        ground_truth_beacons: list[dict[str, Any]],
    ) -> FalsePositiveNegativeAnalysis:
        """
        Analyze false positive and negative detections.

        Args:
            detection_events: List of signal detection events
            ground_truth_beacons: List of actual beacon locations

        Returns:
            False positive/negative analysis
        """
        analysis = FalsePositiveNegativeAnalysis()

        # Match detections to ground truth
        for detection in detection_events:
            matched = False
            for beacon in ground_truth_beacons:
                distance = self._calculate_distance(
                    (detection["location"]["lat"], detection["location"]["lon"]),
                    (beacon["latitude"], beacon["longitude"]),
                )
                # Consider detection valid if within 100m of actual beacon
                if distance < 100:
                    analysis.true_positives += 1
                    matched = True
                    break
            if not matched:
                analysis.false_positives += 1

        # Check for missed beacons
        for beacon in ground_truth_beacons:
            matched = False
            for detection in detection_events:
                distance = self._calculate_distance(
                    (detection["location"]["lat"], detection["location"]["lon"]),
                    (beacon["latitude"], beacon["longitude"]),
                )
                if distance < 100:
                    matched = True
                    break
            if not matched:
                analysis.false_negatives += 1

        # Calculate precision, recall, F1
        if analysis.true_positives + analysis.false_positives > 0:
            analysis.precision = analysis.true_positives / (
                analysis.true_positives + analysis.false_positives
            )

        if analysis.true_positives + analysis.false_negatives > 0:
            analysis.recall = analysis.true_positives / (
                analysis.true_positives + analysis.false_negatives
            )

        if analysis.precision + analysis.recall > 0:
            analysis.f1_score = (
                2 * (analysis.precision * analysis.recall) / (analysis.precision + analysis.recall)
            )

        return analysis

    def correlate_environmental_factors(
        self,
        telemetry_data: list[dict[str, Any]],
        detection_events: list[dict[str, Any]],
        weather_data: dict[str, Any] | None = None,
    ) -> EnvironmentalCorrelation:
        """
        Correlate environmental factors with detection performance.

        Args:
            telemetry_data: List of telemetry frames
            detection_events: List of signal detection events
            weather_data: Optional weather conditions

        Returns:
            Environmental correlation analysis
        """
        correlation = EnvironmentalCorrelation()

        if not telemetry_data or not detection_events:
            return correlation

        # RF noise correlation
        detection_rssi = [d.get("rssi", -100) for d in detection_events]
        detection_snr = [d.get("snr", 0) for d in detection_events]
        if detection_rssi and detection_snr and len(detection_rssi) > 1:
            # Higher SNR should correlate with better detection
            # Check for variance to avoid NaN from corrcoef
            if np.std(detection_rssi) > 0 and np.std(detection_snr) > 0:
                correlation.rf_noise_correlation = np.corrcoef(detection_rssi, detection_snr)[0, 1]
            else:
                correlation.rf_noise_correlation = 0.0  # No variance means no correlation

        # Weather impact (if data available)
        if weather_data:
            wind_speed = weather_data.get("wind_speed_ms", 0)
            precipitation = weather_data.get("precipitation_mm", 0)
            # Higher wind/rain = lower performance
            correlation.weather_impact_score = max(0, 100 - (wind_speed * 5 + precipitation * 10))

        # Terrain impact (based on altitude variance)
        altitudes = [frame.get("altitude", 0) for frame in telemetry_data]
        if altitudes:
            altitude_variance = statistics.variance(altitudes) if len(altitudes) > 1 else 0
            # Higher variance = more challenging terrain
            correlation.terrain_impact_score = max(0, 100 - altitude_variance / 10)

        # Time of day impact
        timestamps = [datetime.fromisoformat(frame["timestamp"]) for frame in telemetry_data]
        if timestamps:
            hours = [t.hour for t in timestamps]
            # Prefer daytime operations (6 AM - 6 PM)
            daytime_ratio = sum(1 for h in hours if 6 <= h <= 18) / len(hours)
            correlation.time_of_day_impact = daytime_ratio * 100

        # Altitude correlation with detection performance
        if detection_events and altitudes:
            detection_times = [datetime.fromisoformat(d["timestamp"]) for d in detection_events]
            detection_altitudes = []
            for det_time in detection_times:
                # Find closest telemetry frame
                for frame in telemetry_data:
                    frame_time = datetime.fromisoformat(frame["timestamp"])
                    if abs((frame_time - det_time).total_seconds()) < 1:
                        detection_altitudes.append(frame.get("altitude", 0))
                        break
            if detection_altitudes:
                # Higher altitude might mean better signal propagation
                correlation.altitude_correlation = statistics.mean(detection_altitudes)

        return correlation

    def compare_to_baseline(self, mission_metrics: dict[str, Any]) -> BaselineComparison:
        """
        Compare mission performance to baseline manual methods.

        Args:
            mission_metrics: Current mission performance data

        Returns:
            Baseline comparison metrics
        """
        comparison = BaselineComparison()

        # Time improvement
        mission_time = mission_metrics.get("search_time_minutes", 0)
        if mission_time > 0:
            comparison.time_improvement_percent = (
                (self.baseline_data["average_search_time_minutes"] - mission_time)
                / self.baseline_data["average_search_time_minutes"]
                * 100
            )

        # Area reduction
        mission_area = mission_metrics.get("area_covered_km2", 0)
        if mission_area > 0:
            comparison.area_reduction_percent = (
                (self.baseline_data["average_area_covered_km2"] - mission_area)
                / self.baseline_data["average_area_covered_km2"]
                * 100
            )

        # Accuracy improvement
        mission_distance = mission_metrics.get("final_distance_m", 0)
        if mission_distance > 0:
            comparison.accuracy_improvement_percent = (
                (self.baseline_data["average_final_distance_m"] - mission_distance)
                / self.baseline_data["average_final_distance_m"]
                * 100
            )

        # Cost reduction (fuel based on time)
        if mission_time > 0:
            mission_fuel_cost = (
                mission_time
                / self.baseline_data["average_search_time_minutes"]
                * self.baseline_data["average_fuel_cost_usd"]
            )
            comparison.cost_reduction_percent = (
                (self.baseline_data["average_fuel_cost_usd"] - mission_fuel_cost)
                / self.baseline_data["average_fuel_cost_usd"]
                * 100
            )

        # Operator workload reduction
        mission_operator_hours = mission_time / 60
        comparison.operator_workload_reduction = (
            (self.baseline_data["average_operator_hours"] - mission_operator_hours)
            / self.baseline_data["average_operator_hours"]
            * 100
        )

        return comparison

    def generate_performance_report(
        self,
        mission_id: UUID,
        telemetry_data: list[dict[str, Any]],
        detection_events: list[dict[str, Any]],
        search_area_km2: float,
        beacon_location: tuple[float, float] | None = None,
        ground_truth_beacons: list[dict[str, Any]] | None = None,
        weather_data: dict[str, Any] | None = None,
    ) -> MissionPerformanceMetrics:
        """
        Generate comprehensive performance report for a mission.

        Args:
            mission_id: Mission identifier
            telemetry_data: List of telemetry frames
            detection_events: List of signal detection events
            search_area_km2: Total search area
            beacon_location: Known beacon location for accuracy
            ground_truth_beacons: Ground truth for false positive analysis
            weather_data: Weather conditions during mission

        Returns:
            Complete mission performance metrics
        """
        # Calculate all metrics
        detection_metrics = self.calculate_detection_metrics(
            telemetry_data, detection_events, search_area_km2
        )
        approach_metrics = self.compute_approach_accuracy(telemetry_data, beacon_location)
        search_metrics = self.measure_search_efficiency(telemetry_data, search_area_km2)

        # False positive analysis
        false_positive_analysis = FalsePositiveNegativeAnalysis()
        if ground_truth_beacons:
            false_positive_analysis = self.analyze_false_positives(
                detection_events, ground_truth_beacons
            )

        # Environmental correlation
        environmental_correlation = self.correlate_environmental_factors(
            telemetry_data, detection_events, weather_data
        )

        # Baseline comparison
        mission_data = {
            "search_time_minutes": search_metrics.search_time_minutes,
            "area_covered_km2": search_metrics.area_covered_km2,
            "final_distance_m": approach_metrics.final_distance_m,
        }
        baseline_comparison = self.compare_to_baseline(mission_data)

        # Calculate overall score
        overall_score = self._calculate_overall_score(
            detection_metrics,
            approach_metrics,
            search_metrics,
            false_positive_analysis,
            baseline_comparison,
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            detection_metrics,
            approach_metrics,
            search_metrics,
            false_positive_analysis,
            environmental_correlation,
        )

        return MissionPerformanceMetrics(
            mission_id=mission_id,
            detection_metrics=detection_metrics.__dict__,
            approach_metrics=approach_metrics.__dict__,
            search_metrics=search_metrics.__dict__,
            false_positive_analysis=false_positive_analysis.__dict__,
            environmental_correlation=environmental_correlation.__dict__,
            baseline_comparison=baseline_comparison.__dict__,
            overall_score=overall_score,
            recommendations=recommendations,
        )

    def _calculate_distance(
        self, point1: tuple[float, float], point2: tuple[float, float]
    ) -> float:
        """Calculate distance between two GPS points in meters."""
        from math import asin, cos, radians, sin, sqrt

        lat1, lon1 = point1
        lat2, lon2 = point2

        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371000  # Earth radius in meters
        return c * r

    def _calculate_path_distance(self, telemetry_data: list[dict[str, Any]]) -> float:
        """Calculate total path distance in meters."""
        total_distance = 0.0
        for i in range(1, len(telemetry_data)):
            point1 = (
                telemetry_data[i - 1]["latitude"],
                telemetry_data[i - 1]["longitude"],
            )
            point2 = (telemetry_data[i]["latitude"], telemetry_data[i]["longitude"])
            total_distance += self._calculate_distance(point1, point2)
        return total_distance

    def _calculate_covered_area(self, telemetry_data: list[dict[str, Any]]) -> float:
        """Estimate covered area in km² using convex hull."""
        if len(telemetry_data) < 3:
            return 0

        points = [(frame["latitude"], frame["longitude"]) for frame in telemetry_data]

        # Simple rectangular approximation
        from math import cos, radians

        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        lat_range = max(lats) - min(lats)
        lon_range = max(lons) - min(lons)

        # Convert degrees to km (approximate)
        lat_km = lat_range * 111
        lon_km = lon_range * 111 * cos(radians(statistics.mean(lats)))

        return float(lat_km * lon_km)

    def _calculate_ideal_grid_distance(self, area_km2: float) -> float:
        """Calculate ideal grid search pattern distance."""
        from math import sqrt

        # Assume square search area with 100m spacing
        side_length_km = sqrt(area_km2)
        num_lines = int(side_length_km * 10)  # 100m spacing
        return side_length_km * num_lines

    def _calculate_overall_score(
        self,
        detection: DetectionMetrics,
        approach: ApproachMetrics,
        search: SearchMetrics,
        false_positive: FalsePositiveNegativeAnalysis,
        baseline: BaselineComparison,
    ) -> float:
        """Calculate overall performance score (0-100)."""
        scores = []

        # Detection score (weighted 25%)
        detection_score = min(100, detection.mean_detection_confidence)
        scores.append(detection_score * 0.25)

        # Approach score (weighted 25%)
        approach_score = approach.approach_efficiency
        scores.append(approach_score * 0.25)

        # Search score (weighted 20%)
        search_score = search.search_pattern_efficiency
        scores.append(search_score * 0.20)

        # Accuracy score (weighted 20%)
        accuracy_score = false_positive.f1_score * 100
        scores.append(accuracy_score * 0.20)

        # Improvement score (weighted 10%)
        improvement_score = max(
            0,
            min(
                100,
                (baseline.time_improvement_percent + baseline.accuracy_improvement_percent) / 2,
            ),
        )
        scores.append(improvement_score * 0.10)

        return sum(scores)

    def _generate_recommendations(
        self,
        detection: DetectionMetrics,
        approach: ApproachMetrics,
        search: SearchMetrics,
        false_positive: FalsePositiveNegativeAnalysis,
        environmental: EnvironmentalCorrelation,
    ) -> list[str]:
        """Generate recommendations based on performance analysis."""
        recommendations = []

        # Detection recommendations
        if detection.mean_detection_confidence < 70:
            recommendations.append(
                "Consider adjusting SDR gain settings for better signal detection"
            )
        if detection.first_detection_time and detection.first_detection_time > 300:
            recommendations.append("Optimize search pattern to achieve faster initial detection")

        # Approach recommendations
        if approach.approach_efficiency < 70:
            recommendations.append("Improve approach algorithm to follow more direct paths")
        if approach.rssi_improvement_db < 10:
            recommendations.append("Tune approach parameters for better signal tracking")

        # Search recommendations
        if search.coverage_percentage < 80:
            recommendations.append("Increase search altitude or adjust pattern for better coverage")
        if search.search_pattern_efficiency < 70:
            recommendations.append("Consider using adaptive search patterns based on terrain")

        # Accuracy recommendations
        if false_positive.precision < 0.8:
            recommendations.append("Implement better signal validation to reduce false positives")
        if false_positive.recall < 0.8:
            recommendations.append("Adjust detection threshold to reduce false negatives")

        # Environmental recommendations
        if environmental.rf_noise_correlation < 0.5:
            recommendations.append("Consider implementing noise filtering algorithms")
        if environmental.weather_impact_score < 70:
            recommendations.append("Plan missions during favorable weather conditions")

        return recommendations
