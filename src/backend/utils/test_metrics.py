"""
Test Metrics and Quality Monitoring System

This module provides comprehensive test quality metrics including:
- Test Value Ratio: Percentage of tests that trace to user stories or hazards
- Hazard Coverage: Percentage of HARA hazards with test coverage
- Performance Metrics: Test execution time and resource usage
- Quality Dashboard: Real-time test quality visualization
"""

import ast
import asyncio
import json
import re
import typing
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import psutil


@dataclass
class TestMetadata:
    """Metadata extracted from test docstrings"""

    file_path: str
    test_name: str
    user_story: str | None = None
    hazard_id: str | None = None
    user_action: str | None = None
    expected_result: str | None = None
    failure_impact: str | None = None
    test_value: str | None = None
    execution_time: float = 0.0
    memory_usage: float = 0.0


@dataclass
class TestValueMetrics:
    """Test value ratio and traceability metrics"""

    total_tests: int = 0
    tests_with_trace: int = 0
    tests_with_user_story: int = 0
    tests_with_hazard: int = 0
    tests_with_value: int = 0
    untraceable_tests: list[str] = field(default_factory=list)

    @property
    def value_ratio(self) -> float:
        """Calculate test value ratio (percentage with traceability)"""
        if self.total_tests == 0:
            return 0.0
        return (self.tests_with_trace / self.total_tests) * 100

    @property
    def story_coverage(self) -> float:
        """Percentage of tests with user story trace"""
        if self.total_tests == 0:
            return 0.0
        return (self.tests_with_user_story / self.total_tests) * 100

    @property
    def hazard_coverage(self) -> float:
        """Percentage of tests with hazard trace"""
        if self.total_tests == 0:
            return 0.0
        return (self.tests_with_hazard / self.total_tests) * 100


@dataclass
class HazardCoverageMetrics:
    """HARA hazard coverage tracking"""

    defined_hazards: set[str] = field(default_factory=set)
    covered_hazards: set[str] = field(default_factory=set)
    hazard_test_map: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    uncovered_hazards: set[str] = field(default_factory=set)

    @property
    def coverage_percentage(self) -> float:
        """Calculate hazard coverage percentage"""
        if not self.defined_hazards:
            return 0.0
        return (len(self.covered_hazards) / len(self.defined_hazards)) * 100

    def update_coverage(self) -> None:
        """Update uncovered hazards set"""
        self.uncovered_hazards = self.defined_hazards - self.covered_hazards


@dataclass
class TestPerformanceMetrics:
    """Test execution performance metrics"""

    total_runtime: float = 0.0
    unit_test_time: float = 0.0
    integration_test_time: float = 0.0
    e2e_test_time: float = 0.0
    sitl_test_time: float = 0.0
    slowest_tests: list[tuple[str, float]] = field(default_factory=list)
    peak_memory_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    parallel_efficiency: float = 0.0


class TestMetricsAnalyzer:
    """Analyze test suite for quality metrics"""

    # HARA hazards from PRD and safety documentation
    DEFINED_HAZARDS: typing.ClassVar[set[str]] = {
        "HARA-PWR-001",  # Low battery leading to crash
        "HARA-GPS-001",  # GPS loss during operation
        "HARA-CMD-001",  # Unauthorized command execution
        "HARA-SIG-001",  # False signal detection
        "HARA-VEL-001",  # Excessive velocity commands
    }

    def __init__(self, test_dir: Path = Path("tests")):
        self.test_dir = test_dir
        self.test_metadata: list[TestMetadata] = []
        self.value_metrics = TestValueMetrics()
        self.hazard_metrics = HazardCoverageMetrics(defined_hazards=self.DEFINED_HAZARDS.copy())
        self.performance_metrics = TestPerformanceMetrics()

    def extract_test_metadata(self, file_path: Path) -> list[TestMetadata]:
        """Extract metadata from test file docstrings"""
        metadata_list = []

        try:
            with open(file_path) as f:
                content = f.read()

            # Parse AST to find test functions and their docstrings
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    metadata = TestMetadata(file_path=str(file_path), test_name=node.name)

                    # Extract docstring
                    docstring = ast.get_docstring(node) or ""

                    # Parse backwards analysis format
                    if "BACKWARDS ANALYSIS:" in docstring:
                        lines = docstring.split("\n")
                        for line in lines:
                            if "User Action:" in line:
                                metadata.user_action = line.split("User Action:")[-1].strip()
                            elif "Expected Result:" in line:
                                metadata.expected_result = line.split("Expected Result:")[
                                    -1
                                ].strip()
                            elif "Failure Impact:" in line:
                                metadata.failure_impact = line.split("Failure Impact:")[-1].strip()

                    # Parse requirement trace
                    if "User Story:" in docstring:
                        match = re.search(r"User Story:\s*#?(\S+)", docstring)
                        if match:
                            metadata.user_story = match.group(1)

                    if "Hazard:" in docstring or "HARA-" in docstring:
                        match = re.search(r"(HARA-\S+)", docstring)
                        if match:
                            metadata.hazard_id = match.group(1)

                    if "TEST VALUE:" in docstring:
                        match = re.search(r"TEST VALUE:\s*(.+?)(?:\n|$)", docstring, re.DOTALL)
                        if match:
                            metadata.test_value = match.group(1).strip()

                    metadata_list.append(metadata)

        except Exception as e:
            print(f"Error parsing {file_path}: {e}")

        return metadata_list

    def analyze_test_suite(self) -> None:
        """Analyze entire test suite for metrics"""
        # Reset metrics
        self.test_metadata.clear()
        self.value_metrics = TestValueMetrics()
        self.hazard_metrics = HazardCoverageMetrics(defined_hazards=self.DEFINED_HAZARDS.copy())

        # Find all test files
        test_files = list(self.test_dir.rglob("test_*.py"))

        for test_file in test_files:
            file_metadata = self.extract_test_metadata(test_file)
            self.test_metadata.extend(file_metadata)

        # Calculate metrics
        for metadata in self.test_metadata:
            self.value_metrics.total_tests += 1

            # Check traceability
            has_trace = bool(metadata.user_story or metadata.hazard_id)
            if has_trace:
                self.value_metrics.tests_with_trace += 1
            else:
                self.value_metrics.untraceable_tests.append(
                    f"{metadata.file_path}::{metadata.test_name}"
                )

            if metadata.user_story:
                self.value_metrics.tests_with_user_story += 1

            if metadata.hazard_id:
                self.value_metrics.tests_with_hazard += 1
                self.hazard_metrics.covered_hazards.add(metadata.hazard_id)
                self.hazard_metrics.hazard_test_map[metadata.hazard_id].append(
                    f"{metadata.file_path}::{metadata.test_name}"
                )

            if metadata.test_value:
                self.value_metrics.tests_with_value += 1

        # Update hazard coverage
        self.hazard_metrics.update_coverage()

    def measure_test_performance(self, pytest_json_report: Path | None = None) -> None:
        """Measure test execution performance from pytest JSON report"""
        if pytest_json_report and pytest_json_report.exists():
            with open(pytest_json_report) as f:
                report = json.load(f)

            # Extract timing data
            if "duration" in report:
                self.performance_metrics.total_runtime = report["duration"]

            # Analyze test categories
            if "tests" in report:
                for test in report["tests"]:
                    duration = test.get("duration", 0.0)
                    nodeid = test.get("nodeid", "")

                    if "unit" in nodeid:
                        self.performance_metrics.unit_test_time += duration
                    elif "integration" in nodeid:
                        self.performance_metrics.integration_test_time += duration
                    elif "e2e" in nodeid:
                        self.performance_metrics.e2e_test_time += duration
                    elif "sitl" in nodeid:
                        self.performance_metrics.sitl_test_time += duration

                    # Track slowest tests
                    self.performance_metrics.slowest_tests.append((nodeid, duration))

                # Sort and keep top 10 slowest
                self.performance_metrics.slowest_tests.sort(key=lambda x: x[1], reverse=True)
                self.performance_metrics.slowest_tests = self.performance_metrics.slowest_tests[:10]

        # Measure current resource usage
        process = psutil.Process()
        self.performance_metrics.peak_memory_mb = process.memory_info().rss / 1024 / 1024
        self.performance_metrics.cpu_usage_percent = process.cpu_percent(interval=1)

    def generate_quality_dashboard(self) -> dict[str, typing.Any]:
        """Generate test quality dashboard data"""
        dashboard = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": self.value_metrics.total_tests,
                "test_value_ratio": f"{self.value_metrics.value_ratio:.1f}%",
                "hazard_coverage": f"{self.hazard_metrics.coverage_percentage:.1f}%",
                "execution_time": f"{self.performance_metrics.total_runtime:.2f}s",
            },
            "traceability": {
                "tests_with_trace": self.value_metrics.tests_with_trace,
                "tests_with_user_story": self.value_metrics.tests_with_user_story,
                "tests_with_hazard": self.value_metrics.tests_with_hazard,
                "tests_with_value": self.value_metrics.tests_with_value,
                "value_ratio": self.value_metrics.value_ratio,
                "story_coverage": self.value_metrics.story_coverage,
                "hazard_test_coverage": self.value_metrics.hazard_coverage,
            },
            "hazards": {
                "defined": list(self.hazard_metrics.defined_hazards),
                "covered": list(self.hazard_metrics.covered_hazards),
                "uncovered": list(self.hazard_metrics.uncovered_hazards),
                "coverage_percentage": self.hazard_metrics.coverage_percentage,
                "test_mapping": dict(self.hazard_metrics.hazard_test_map),
            },
            "performance": {
                "total_runtime": self.performance_metrics.total_runtime,
                "unit_test_time": self.performance_metrics.unit_test_time,
                "integration_test_time": self.performance_metrics.integration_test_time,
                "e2e_test_time": self.performance_metrics.e2e_test_time,
                "sitl_test_time": self.performance_metrics.sitl_test_time,
                "slowest_tests": self.performance_metrics.slowest_tests[:5],
                "peak_memory_mb": self.performance_metrics.peak_memory_mb,
                "cpu_usage_percent": self.performance_metrics.cpu_usage_percent,
            },
            "quality_issues": {
                "untraceable_tests": self.value_metrics.untraceable_tests[:10],
                "uncovered_hazards": list(self.hazard_metrics.uncovered_hazards),
                "slow_tests": [
                    test
                    for test, time in self.performance_metrics.slowest_tests
                    if time > 1.0  # Tests taking more than 1 second
                ],
            },
        }

        return dashboard

    def save_dashboard(self, output_path: Path = Path("test_quality_dashboard.json")) -> None:
        """Save dashboard to JSON file"""
        dashboard = self.generate_quality_dashboard()
        with open(output_path, "w") as f:
            json.dump(dashboard, f, indent=2)
        print(f"Dashboard saved to {output_path}")

    def print_summary(self) -> None:
        """Print test quality summary to console"""
        print("\n" + "=" * 60)
        print("TEST QUALITY METRICS SUMMARY")
        print("=" * 60)

        print("\n📊 Test Value Metrics:")
        print(f"  Total Tests: {self.value_metrics.total_tests}")
        print(f"  Test Value Ratio: {self.value_metrics.value_ratio:.1f}%")
        print(f"  Story Coverage: {self.value_metrics.story_coverage:.1f}%")
        print(f"  Hazard Test Coverage: {self.value_metrics.hazard_coverage:.1f}%")

        print("\n🎯 Hazard Coverage:")
        print(f"  Defined Hazards: {len(self.hazard_metrics.defined_hazards)}")
        print(f"  Covered Hazards: {len(self.hazard_metrics.covered_hazards)}")
        print(f"  Coverage: {self.hazard_metrics.coverage_percentage:.1f}%")
        if self.hazard_metrics.uncovered_hazards:
            print(f"  ⚠️  Uncovered: {', '.join(self.hazard_metrics.uncovered_hazards)}")

        print("\n⚡ Performance Metrics:")
        print(f"  Total Runtime: {self.performance_metrics.total_runtime:.2f}s")
        print(f"  Peak Memory: {self.performance_metrics.peak_memory_mb:.1f} MB")
        print(f"  CPU Usage: {self.performance_metrics.cpu_usage_percent:.1f}%")

        if self.value_metrics.untraceable_tests:
            print("\n⚠️  Top Untraceable Tests:")
            for test in self.value_metrics.untraceable_tests[:5]:
                print(f"    - {test}")

        print("\n" + "=" * 60)


async def continuous_monitoring(interval: int = 60) -> None:
    """Continuous test quality monitoring"""
    analyzer = TestMetricsAnalyzer(Path("tests"))

    while True:
        print(f"\n[{datetime.now().isoformat()}] Running test quality analysis...")

        # Analyze test suite
        analyzer.analyze_test_suite()

        # Check for pytest JSON report
        json_report = Path(".pytest_cache/report.json")
        if json_report.exists():
            analyzer.measure_test_performance(json_report)

        # Generate and save dashboard
        analyzer.save_dashboard()
        analyzer.print_summary()

        # Check quality gates
        if analyzer.value_metrics.value_ratio < 80:
            print("⚠️  WARNING: Test value ratio below 80% threshold!")

        if analyzer.hazard_metrics.coverage_percentage < 100:
            print("⚠️  WARNING: Hazard coverage below 100% threshold!")

        if analyzer.performance_metrics.total_runtime > 240:  # 4 minutes
            print("⚠️  WARNING: Test runtime exceeds 4 minute target!")

        await asyncio.sleep(interval)


if __name__ == "__main__":
    # Run one-time analysis
    analyzer = TestMetricsAnalyzer(Path("tests"))
    analyzer.analyze_test_suite()
    analyzer.measure_test_performance()
    analyzer.save_dashboard()
    analyzer.print_summary()
