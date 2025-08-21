"""
Frequency Conflict Detection Utility

TASK-6.3.1 [28d2] - Frequency conflict detection with existing radio systems
Detects potential interference with existing radio services and provides
recommendations for frequency selection optimization.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RadioService:
    """Definition of a radio service with frequency range and characteristics."""

    name: str
    start_freq_hz: int
    end_freq_hz: int
    service_type: str
    protection_level: str  # "critical", "high", "medium", "low"
    typical_bandwidth_hz: int
    notes: str = ""


class FrequencyConflictDetector:
    """
    Detects frequency conflicts with existing radio services.

    Analyzes potential interference between proposed frequency usage
    and existing radio services including:
    - Commercial broadcast services
    - Aviation navigation and communication
    - Maritime communication and emergency
    - Public safety and government
    - Amateur radio bands
    - ISM and unlicensed bands
    """

    def __init__(self):
        """Initialize conflict detector with radio service database."""
        self._radio_services = self._load_radio_services()
        self._interference_thresholds = self._load_interference_thresholds()

        logger.info("Frequency conflict detector initialized")

    def detect_conflicts(self, frequency_hz: int, bandwidth_hz: int) -> Dict[str, Any]:
        """
        Detect conflicts with existing radio services.

        Args:
            frequency_hz: Center frequency to check in Hz
            bandwidth_hz: Bandwidth of the signal in Hz

        Returns:
            Dict containing conflict analysis results
        """
        # Calculate frequency range
        freq_start = frequency_hz - (bandwidth_hz // 2)
        freq_end = frequency_hz + (bandwidth_hz // 2)

        conflicts = []
        potential_conflicts = []

        for service in self._radio_services:
            conflict_result = self._check_service_conflict(
                freq_start, freq_end, service
            )

            if conflict_result["conflict_type"] == "direct":
                conflicts.append(conflict_result)
            elif conflict_result["conflict_type"] == "adjacent":
                potential_conflicts.append(conflict_result)

        # Calculate overall conflict assessment
        has_conflicts = len(conflicts) > 0
        conflict_severity = self._assess_conflict_severity(conflicts)

        return {
            "conflicts": has_conflicts,
            "frequency_hz": frequency_hz,
            "frequency_mhz": frequency_hz / 1e6,
            "bandwidth_khz": bandwidth_hz / 1e3,
            "conflicting_services": [c["service_name"] for c in conflicts],
            "potential_conflicts": [c["service_name"] for c in potential_conflicts],
            "severity": conflict_severity,
            "direct_conflicts": conflicts,
            "adjacent_conflicts": potential_conflicts,
            "recommendations": self._generate_conflict_recommendations(
                frequency_hz, bandwidth_hz, conflicts, potential_conflicts
            ),
        }

    def find_clear_frequencies(
        self,
        bandwidth_hz: int,
        freq_range: Tuple[int, int] = (24_000_000, 1_750_000_000),
    ) -> List[Dict[str, Any]]:
        """
        Find frequencies with minimal conflicts in specified range.

        Args:
            bandwidth_hz: Required bandwidth in Hz
            freq_range: Frequency range to search (start_hz, end_hz)

        Returns:
            List of clear frequency recommendations
        """
        start_freq, end_freq = freq_range
        step_size = bandwidth_hz * 2  # Search every 2x bandwidth
        clear_frequencies = []

        current_freq = start_freq
        while current_freq + bandwidth_hz <= end_freq:
            conflict_result = self.detect_conflicts(current_freq, bandwidth_hz)

            if not conflict_result["conflicts"]:
                # Calculate quality score based on potential conflicts
                quality_score = self._calculate_frequency_quality(
                    current_freq, bandwidth_hz, conflict_result
                )

                clear_frequencies.append(
                    {
                        "frequency_hz": current_freq,
                        "frequency_mhz": current_freq / 1e6,
                        "quality_score": quality_score,
                        "potential_conflicts": len(
                            conflict_result["potential_conflicts"]
                        ),
                        "notes": f"Clear frequency with {len(conflict_result['potential_conflicts'])} nearby services",
                    }
                )

            current_freq += step_size

        # Sort by quality score (higher is better)
        return sorted(clear_frequencies, key=lambda x: x["quality_score"], reverse=True)

    def _load_radio_services(self) -> List[RadioService]:
        """Load database of radio services and their frequency allocations."""
        return [
            # FM Broadcast
            RadioService(
                name="FM broadcast",
                start_freq_hz=88_000_000,
                end_freq_hz=108_000_000,
                service_type="broadcast",
                protection_level="critical",
                typical_bandwidth_hz=200_000,
                notes="Commercial FM radio stations",
            ),
            # VOR Navigation
            RadioService(
                name="VOR navigation",
                start_freq_hz=108_000_000,
                end_freq_hz=118_000_000,
                service_type="aviation_navigation",
                protection_level="critical",
                typical_bandwidth_hz=25_000,
                notes="Aviation VOR navigation beacons",
            ),
            # Air Traffic Control
            RadioService(
                name="air_traffic_control",
                start_freq_hz=118_000_000,
                end_freq_hz=137_000_000,
                service_type="aviation_communication",
                protection_level="critical",
                typical_bandwidth_hz=25_000,
                notes="Aviation communication and control",
            ),
            # Public Safety (VHF)
            RadioService(
                name="public_safety_vhf",
                start_freq_hz=150_000_000,
                end_freq_hz=174_000_000,
                service_type="public_safety",
                protection_level="high",
                typical_bandwidth_hz=25_000,
                notes="Police, fire, emergency services",
            ),
            # Amateur Radio 2m
            RadioService(
                name="amateur_2m",
                start_freq_hz=144_000_000,
                end_freq_hz=148_000_000,
                service_type="amateur",
                protection_level="medium",
                typical_bandwidth_hz=25_000,
                notes="Amateur radio 2 meter band",
            ),
            # Maritime VHF
            RadioService(
                name="maritime_vhf",
                start_freq_hz=156_000_000,
                end_freq_hz=162_500_000,
                service_type="maritime",
                protection_level="high",
                typical_bandwidth_hz=25_000,
                notes="Maritime communication and emergency",
            ),
            # Weather Radio
            RadioService(
                name="weather_radio",
                start_freq_hz=162_400_000,
                end_freq_hz=162_550_000,
                service_type="emergency",
                protection_level="high",
                typical_bandwidth_hz=25_000,
                notes="NOAA Weather Radio",
            ),
            # Emergency Beacons (406 MHz)
            RadioService(
                name="emergency_beacons",
                start_freq_hz=406_000_000,
                end_freq_hz=406_100_000,
                service_type="emergency",
                protection_level="critical",
                typical_bandwidth_hz=50_000,
                notes="COSPAS-SARSAT emergency beacons",
            ),
            # UHF Military
            RadioService(
                name="uhf_military",
                start_freq_hz=225_000_000,
                end_freq_hz=400_000_000,
                service_type="military",
                protection_level="critical",
                typical_bandwidth_hz=25_000,
                notes="Military communication systems",
            ),
            # Cell Phone Bands (LTE Band 13)
            RadioService(
                name="lte_band_13",
                start_freq_hz=746_000_000,
                end_freq_hz=756_000_000,
                service_type="cellular",
                protection_level="high",
                typical_bandwidth_hz=10_000_000,
                notes="LTE cellular uplink",
            ),
            # GPS L1
            RadioService(
                name="gps_l1",
                start_freq_hz=1_575_000_000,
                end_freq_hz=1_576_000_000,
                service_type="navigation",
                protection_level="critical",
                typical_bandwidth_hz=2_000_000,
                notes="GPS L1 C/A signals",
            ),
            # ISM 2.4 GHz
            RadioService(
                name="ism_2400",
                start_freq_hz=2_400_000_000,
                end_freq_hz=2_500_000_000,
                service_type="unlicensed",
                protection_level="low",
                typical_bandwidth_hz=20_000_000,
                notes="WiFi, Bluetooth, industrial",
            ),
        ]

    def _load_interference_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Load interference thresholds for different service types."""
        return {
            "critical": {
                "adjacent_band_separation_hz": 200_000,  # 200 kHz minimum
                "harmonic_protection_db": 60.0,
                "intermod_protection_db": 40.0,
            },
            "high": {
                "adjacent_band_separation_hz": 100_000,  # 100 kHz minimum
                "harmonic_protection_db": 50.0,
                "intermod_protection_db": 30.0,
            },
            "medium": {
                "adjacent_band_separation_hz": 50_000,  # 50 kHz minimum
                "harmonic_protection_db": 40.0,
                "intermod_protection_db": 25.0,
            },
            "low": {
                "adjacent_band_separation_hz": 25_000,  # 25 kHz minimum
                "harmonic_protection_db": 30.0,
                "intermod_protection_db": 20.0,
            },
        }

    def _check_service_conflict(
        self, freq_start: int, freq_end: int, service: RadioService
    ) -> Dict[str, Any]:
        """Check for conflicts between frequency range and radio service."""

        # Check for direct overlap
        if freq_start < service.end_freq_hz and freq_end > service.start_freq_hz:
            overlap_start = max(freq_start, service.start_freq_hz)
            overlap_end = min(freq_end, service.end_freq_hz)
            overlap_bandwidth = overlap_end - overlap_start

            return {
                "conflict_type": "direct",
                "service_name": service.name,
                "service_type": service.service_type,
                "protection_level": service.protection_level,
                "overlap_hz": overlap_bandwidth,
                "overlap_percent": (overlap_bandwidth / (freq_end - freq_start)) * 100,
                "severity": self._get_conflict_severity(
                    service.protection_level, "direct"
                ),
                "notes": f"Direct overlap with {service.name}",
            }

        # Check for adjacent band conflicts
        threshold = self._interference_thresholds[service.protection_level]
        min_separation = threshold["adjacent_band_separation_hz"]

        # Check separation distances
        separation_low = freq_start - service.end_freq_hz
        separation_high = service.start_freq_hz - freq_end
        min_actual_separation = max(separation_low, separation_high)

        if 0 < min_actual_separation < min_separation:
            return {
                "conflict_type": "adjacent",
                "service_name": service.name,
                "service_type": service.service_type,
                "protection_level": service.protection_level,
                "separation_hz": min_actual_separation,
                "required_separation_hz": min_separation,
                "severity": self._get_conflict_severity(
                    service.protection_level, "adjacent"
                ),
                "notes": f"Adjacent band interference risk with {service.name}",
            }

        # No conflict
        return {
            "conflict_type": "none",
            "service_name": service.name,
            "separation_hz": (
                min_actual_separation if min_actual_separation > 0 else None
            ),
        }

    def _get_conflict_severity(self, protection_level: str, conflict_type: str) -> str:
        """Determine conflict severity based on protection level and type."""
        severity_matrix = {
            ("critical", "direct"): "critical",
            ("critical", "adjacent"): "high",
            ("high", "direct"): "high",
            ("high", "adjacent"): "medium",
            ("medium", "direct"): "medium",
            ("medium", "adjacent"): "low",
            ("low", "direct"): "low",
            ("low", "adjacent"): "low",
        }

        return severity_matrix.get((protection_level, conflict_type), "low")

    def _assess_conflict_severity(self, conflicts: List[Dict[str, Any]]) -> str:
        """Assess overall conflict severity from list of conflicts."""
        if not conflicts:
            return "none"

        severities = [c["severity"] for c in conflicts]

        if "critical" in severities:
            return "critical"
        elif "high" in severities:
            return "high"
        elif "medium" in severities:
            return "medium"
        else:
            return "low"

    def _calculate_frequency_quality(
        self, frequency_hz: int, bandwidth_hz: int, conflict_result: Dict[str, Any]
    ) -> float:
        """Calculate quality score for a frequency (0.0 to 1.0, higher is better)."""
        base_score = 1.0

        # Deduct for potential conflicts
        potential_conflicts = len(conflict_result["potential_conflicts"])
        base_score -= potential_conflicts * 0.1

        # Prefer frequencies in emergency/SAR bands
        if 160_000_000 <= frequency_hz <= 165_000_000:  # Maritime emergency
            base_score += 0.2
        elif 406_000_000 <= frequency_hz <= 406_100_000:  # Emergency beacons
            base_score += 0.2
        elif 121_000_000 <= frequency_hz <= 122_000_000:  # Aviation emergency
            base_score += 0.2

        # Prefer frequencies in HackRF sweet spot
        if 100_000_000 <= frequency_hz <= 1_000_000_000:
            base_score += 0.1

        return max(0.0, min(1.0, base_score))

    def _generate_conflict_recommendations(
        self,
        frequency_hz: int,
        bandwidth_hz: int,
        conflicts: List[Dict[str, Any]],
        potential_conflicts: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate recommendations for resolving conflicts."""
        recommendations = []

        if not conflicts and not potential_conflicts:
            recommendations.append("Frequency appears clear of major conflicts")
            return recommendations

        if conflicts:
            critical_conflicts = [
                c for c in conflicts if c["severity"] in ["critical", "high"]
            ]
            if critical_conflicts:
                recommendations.append(
                    "CRITICAL: Avoid this frequency due to protected service conflicts"
                )
                for conflict in critical_conflicts:
                    recommendations.append(
                        f"  - Conflicts with {conflict['service_name']} ({conflict['service_type']})"
                    )

        if potential_conflicts:
            recommendations.append(
                "Consider alternative frequencies to avoid adjacent band interference"
            )

        # Suggest alternative frequency ranges
        if conflicts or len(potential_conflicts) > 2:
            recommendations.append("Recommended alternative bands:")
            recommendations.append("  - 162.025 MHz (Maritime emergency)")
            recommendations.append("  - 406.000 MHz (Emergency beacons)")
            recommendations.append("  - 915 MHz (ISM band)")
            recommendations.append("  - 2.4 GHz (ISM band)")

        return recommendations
