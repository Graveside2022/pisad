#!/usr/bin/env python3
"""Coverage Baseline Measurement Script.

This script measures code coverage baseline and tracks improvements
as we add hardware tests.
"""

import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


def run_coverage_test(markers: str = "", verbose: bool = True) -> Dict[str, Any]:
    """Run pytest with coverage and return results.
    
    Args:
        markers: Pytest markers to filter tests (e.g., "not hardware")
        verbose: Show detailed output
        
    Returns:
        Dictionary with coverage results
    """
    cmd = ["pytest"]
    
    if markers:
        cmd.extend(["-m", markers])
    
    cmd.extend([
        "--cov=src",
        "--cov-report=json",
        "--cov-report=term",
        "-q" if not verbose else "-v",
    ])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse coverage.json
    coverage_file = Path("coverage.json")
    if coverage_file.exists():
        with open(coverage_file) as f:
            coverage_data = json.load(f)
            
        # Extract summary
        summary = coverage_data.get("totals", {})
        return {
            "percent_covered": summary.get("percent_covered", 0),
            "num_statements": summary.get("num_statements", 0),
            "num_missing": summary.get("num_missing", 0),
            "num_branches": summary.get("num_branches", 0),
            "num_partial_branches": summary.get("num_partial_branches", 0),
            "covered_lines": summary.get("covered_lines", 0),
            "missing_lines": summary.get("missing_lines", 0),
            "files": coverage_data.get("files", {}),
        }
    
    return {"error": "Coverage data not found"}


def get_uncovered_modules(coverage_data: Dict[str, Any]) -> Dict[str, int]:
    """Get modules with low coverage.
    
    Args:
        coverage_data: Coverage results
        
    Returns:
        Dictionary of module -> missing lines
    """
    uncovered = {}
    
    for file_path, file_data in coverage_data.get("files", {}).items():
        if file_data["summary"]["percent_covered"] < 80:
            uncovered[file_path] = file_data["summary"]["missing_lines"]
    
    return uncovered


def compare_coverage(baseline: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two coverage results.
    
    Args:
        baseline: Baseline coverage data
        current: Current coverage data
        
    Returns:
        Comparison results
    """
    return {
        "percent_change": current["percent_covered"] - baseline["percent_covered"],
        "lines_added": current["covered_lines"] - baseline["covered_lines"],
        "lines_removed_from_missing": baseline["num_missing"] - current["num_missing"],
        "baseline_percent": baseline["percent_covered"],
        "current_percent": current["percent_covered"],
    }


def save_baseline(coverage_data: Dict[str, Any], filename: str = "coverage_baseline.json") -> None:
    """Save coverage baseline to file.
    
    Args:
        coverage_data: Coverage results to save
        filename: Output filename
    """
    baseline = {
        "timestamp": datetime.now().isoformat(),
        "coverage": coverage_data,
    }
    
    with open(filename, "w") as f:
        json.dump(baseline, f, indent=2)
    
    print(f"Baseline saved to {filename}")


def load_baseline(filename: str = "coverage_baseline.json") -> Dict[str, Any]:
    """Load coverage baseline from file.
    
    Args:
        filename: Baseline filename
        
    Returns:
        Baseline coverage data
    """
    baseline_file = Path(filename)
    if baseline_file.exists():
        with open(baseline_file) as f:
            data = json.load(f)
        return data["coverage"]
    
    return None


def print_coverage_report(coverage_data: Dict[str, Any]) -> None:
    """Print formatted coverage report.
    
    Args:
        coverage_data: Coverage results
    """
    print("\n" + "=" * 60)
    print("COVERAGE REPORT")
    print("=" * 60)
    
    print(f"Total Coverage: {coverage_data['percent_covered']:.2f}%")
    print(f"Statements: {coverage_data['covered_lines']}/{coverage_data['num_statements']}")
    print(f"Missing: {coverage_data['num_missing']} lines")
    
    if coverage_data['num_branches'] > 0:
        branch_coverage = (1 - coverage_data['num_partial_branches'] / coverage_data['num_branches']) * 100
        print(f"Branch Coverage: {branch_coverage:.2f}%")
    
    print("\nModules with < 80% coverage:")
    print("-" * 40)
    
    uncovered = get_uncovered_modules(coverage_data)
    for module, missing in sorted(uncovered.items(), key=lambda x: x[1], reverse=True)[:10]:
        module_name = module.replace("src/", "")
        print(f"  {module_name}: {missing} lines missing")


def print_comparison(comparison: Dict[str, Any]) -> None:
    """Print coverage comparison.
    
    Args:
        comparison: Comparison results
    """
    print("\n" + "=" * 60)
    print("COVERAGE COMPARISON")
    print("=" * 60)
    
    print(f"Baseline: {comparison['baseline_percent']:.2f}%")
    print(f"Current:  {comparison['current_percent']:.2f}%")
    
    change = comparison['percent_change']
    symbol = "📈" if change > 0 else "📉" if change < 0 else "➡️"
    
    print(f"Change:   {change:+.2f}% {symbol}")
    print(f"Lines covered: {comparison['lines_added']:+d}")
    print(f"Missing reduced: {comparison['lines_removed_from_missing']}")
    
    # Grade the improvement
    if change >= 15:
        print("\n✅ EXCELLENT: Major coverage improvement!")
    elif change >= 10:
        print("\n✅ GREAT: Significant coverage improvement!")
    elif change >= 5:
        print("\n✅ GOOD: Notable coverage improvement")
    elif change >= 0:
        print("\n➡️ OK: Coverage maintained")
    else:
        print("\n⚠️ WARNING: Coverage decreased!")


def main():
    """Main function to measure and track coverage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Measure test coverage")
    parser.add_argument("--baseline", action="store_true", help="Save as baseline")
    parser.add_argument("--compare", action="store_true", help="Compare with baseline")
    parser.add_argument("--mock-only", action="store_true", help="Run mock tests only")
    parser.add_argument("--hardware-only", action="store_true", help="Run hardware tests only")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    # Determine test markers
    markers = ""
    if args.mock_only:
        markers = "mock_hardware and not hardware"
        print("Running mock hardware tests only...")
    elif args.hardware_only:
        markers = "hardware"
        print("Running real hardware tests only...")
    elif not args.all:
        markers = "not hardware"
        print("Running tests without hardware...")
    else:
        print("Running all tests...")
    
    # Run coverage
    coverage_data = run_coverage_test(markers)
    
    if "error" in coverage_data:
        print(f"Error: {coverage_data['error']}")
        sys.exit(1)
    
    # Print report
    print_coverage_report(coverage_data)
    
    # Handle baseline operations
    if args.baseline:
        save_baseline(coverage_data)
    
    if args.compare:
        baseline = load_baseline()
        if baseline:
            comparison = compare_coverage(baseline, coverage_data)
            print_comparison(comparison)
        else:
            print("\n⚠️ No baseline found. Run with --baseline first.")
    
    # Check coverage target
    target = 85.0
    if coverage_data['percent_covered'] >= target:
        print(f"\n🎉 SUCCESS: Coverage target of {target}% achieved!")
        sys.exit(0)
    else:
        gap = target - coverage_data['percent_covered']
        print(f"\n📊 Coverage gap: {gap:.2f}% to reach {target}% target")
        sys.exit(0 if not args.compare else 1)


if __name__ == "__main__":
    main()