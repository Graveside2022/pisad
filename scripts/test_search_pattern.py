#!/usr/bin/env python3
"""Test script for search pattern generation functionality."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backend.services.search_pattern_generator import (
    CenterRadiusBoundary,
    CornerBoundary,
    PatternType,
    SearchPatternGenerator,
)
from src.backend.services.waypoint_exporter import WaypointExporter


def test_search_patterns():
    """Test all search pattern types."""
    generator = SearchPatternGenerator(default_altitude=50.0)
    exporter = WaypointExporter()

    # Test center-radius boundary with expanding square
    print("Testing Expanding Square Pattern...")
    boundary1 = CenterRadiusBoundary(37.7749, -122.4194, 300)
    pattern1 = generator.generate_pattern(
        PatternType.EXPANDING_SQUARE, spacing=75.0, velocity=7.0, boundary=boundary1
    )
    print(f"  Generated {pattern1.total_waypoints} waypoints")
    print(f"  Estimated time: {pattern1.estimated_time_remaining:.0f} seconds")

    # Test spiral pattern
    print("\nTesting Spiral Pattern...")
    pattern2 = generator.generate_pattern(
        PatternType.SPIRAL, spacing=80.0, velocity=6.0, boundary=boundary1
    )
    print(f"  Generated {pattern2.total_waypoints} waypoints")

    # Test corner boundary with lawnmower
    print("\nTesting Lawnmower Pattern...")
    boundary2 = CornerBoundary(
        [(37.770, -122.425), (37.780, -122.425), (37.780, -122.415), (37.770, -122.415)]
    )
    pattern3 = generator.generate_pattern(
        PatternType.LAWNMOWER, spacing=100.0, velocity=10.0, boundary=boundary2
    )
    print(f"  Generated {pattern3.total_waypoints} waypoints")

    # Test waypoint export
    print("\nTesting Waypoint Export...")
    qgc_export = exporter.export_qgc_wpl(pattern1.waypoints, 37.7749, -122.4194)
    print(f"  QGC export: {len(qgc_export.splitlines())} lines")

    mp_export = exporter.export_mission_planner(pattern1.waypoints)
    print(f"  Mission Planner export: {len(mp_export.splitlines())} lines")

    kml_export = exporter.export_kml(pattern1.waypoints, "Test Pattern")
    print(f"  KML export: {len(kml_export.splitlines())} lines")

    # Test import
    print("\nTesting Waypoint Import...")
    imported = exporter.import_qgc_wpl(qgc_export)
    print(f"  Imported {len(imported)} waypoints from QGC format")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_search_patterns()
