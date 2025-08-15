"""
Pytest plugin for test performance monitoring and metrics collection

This plugin automatically collects:
- Test execution times
- Memory usage per test
- Test categorization (unit/integration/e2e/sitl)
- Failure patterns
- Resource usage
"""

import json
import time
from pathlib import Path

import psutil
import pytest


class TestPerformanceMonitor:
    """Monitor test performance metrics"""

    def __init__(self):
        self.test_metrics: list[dict] = []
        self.start_time: float = 0
        self.end_time: float = 0
        self.initial_memory: float = 0
        self.peak_memory: float = 0
        self.test_categories = {
            "unit": [],
            "integration": [],
            "e2e": [],
            "sitl": [],
            "property": [],
            "contract": [],
        }

    def start_monitoring(self):
        """Start monitoring session"""
        self.start_time = time.time()
        process = psutil.Process()
        self.initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    def end_monitoring(self):
        """End monitoring session"""
        self.end_time = time.time()
        process = psutil.Process()
        self.peak_memory = max(self.peak_memory, process.memory_info().rss / 1024 / 1024)

    def record_test(
        self, nodeid: str, duration: float, outcome: str, memory_mb: float, markers: list[str]
    ):
        """Record test execution metrics"""
        # Categorize test
        category = "unit"  # default
        if "integration" in nodeid or "integration" in markers:
            category = "integration"
        elif "e2e" in nodeid or "e2e" in markers:
            category = "e2e"
        elif "sitl" in nodeid or "sitl" in markers:
            category = "sitl"
        elif "property" in nodeid:
            category = "property"
        elif "contract" in nodeid:
            category = "contract"

        test_data = {
            "nodeid": nodeid,
            "duration": duration,
            "outcome": outcome,
            "memory_mb": memory_mb,
            "category": category,
            "markers": markers,
        }

        self.test_metrics.append(test_data)
        self.test_categories[category].append(test_data)

    def generate_report(self) -> dict:
        """Generate performance report"""
        total_runtime = self.end_time - self.start_time if self.end_time else 0

        # Calculate category times
        category_times = {}
        for category, tests in self.test_categories.items():
            category_times[category] = sum(t["duration"] for t in tests)

        # Find slowest tests
        slowest_tests = sorted(self.test_metrics, key=lambda x: x["duration"], reverse=True)[:10]

        # Calculate pass rate
        passed = sum(1 for t in self.test_metrics if t["outcome"] == "passed")
        total = len(self.test_metrics)
        pass_rate = (passed / total * 100) if total else 0

        report = {
            "summary": {
                "total_runtime": total_runtime,
                "total_tests": total,
                "passed": passed,
                "failed": total - passed,
                "pass_rate": pass_rate,
                "initial_memory_mb": self.initial_memory,
                "peak_memory_mb": self.peak_memory,
                "memory_growth_mb": self.peak_memory - self.initial_memory,
            },
            "categories": {
                category: {
                    "count": len(tests),
                    "total_time": category_times[category],
                    "avg_time": category_times[category] / len(tests) if tests else 0,
                }
                for category, tests in self.test_categories.items()
            },
            "slowest_tests": [
                {
                    "nodeid": t["nodeid"],
                    "duration": t["duration"],
                    "category": t["category"],
                }
                for t in slowest_tests
            ],
            "performance_targets": {
                "total_runtime_target": 240,  # 4 minutes
                "unit_test_target": 30,  # 30 seconds
                "integration_test_target": 120,  # 2 minutes
                "sitl_test_target": 150,  # 2.5 minutes
                "target_met": total_runtime <= 240,
            },
        }

        return report


# Global monitor instance
monitor = TestPerformanceMonitor()


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Configure pytest with performance monitoring"""
    config.addinivalue_line("markers", "performance: mark test for performance monitoring")
    config.addinivalue_line("markers", "slow: mark test as known to be slow")


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    """Start monitoring at session start"""
    monitor.start_monitoring()


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """End monitoring and generate report at session end"""
    monitor.end_monitoring()

    # Generate and save report
    report = monitor.generate_report()

    # Save JSON report
    report_path = Path(".pytest_cache/performance_report.json")
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("TEST PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Total Runtime: {report['summary']['total_runtime']:.2f}s")
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Pass Rate: {report['summary']['pass_rate']:.1f}%")
    print(f"Memory Growth: {report['summary']['memory_growth_mb']:.1f} MB")

    if report["performance_targets"]["target_met"]:
        print("✅ Performance target MET (<4 minutes)")
    else:
        print("❌ Performance target FAILED (>4 minutes)")

    print("\nCategory Breakdown:")
    for category, stats in report["categories"].items():
        if stats["count"] > 0:
            print(
                f"  {category}: {stats['count']} tests, {stats['total_time']:.2f}s total, {stats['avg_time']:.3f}s avg"
            )

    print("\nSlowest Tests:")
    for test in report["slowest_tests"][:5]:
        print(f"  {test['duration']:.2f}s - {test['nodeid'][:60]}...")
    print("=" * 60)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    """Monitor individual test execution"""
    # Get test markers
    markers = [m.name for m in item.iter_markers()]

    # Start test monitoring
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024
    start_time = time.time()

    # Run test
    outcome = yield

    # End test monitoring
    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024
    duration = end_time - start_time

    # Update peak memory
    monitor.peak_memory = max(monitor.peak_memory, end_memory)

    # Determine outcome
    test_outcome = "passed"
    if hasattr(outcome, "excinfo") and outcome.excinfo:
        test_outcome = "failed"
    elif hasattr(item, "_skipped"):
        test_outcome = "skipped"

    # Record metrics
    monitor.record_test(
        nodeid=item.nodeid,
        duration=duration,
        outcome=test_outcome,
        memory_mb=end_memory - start_memory,
        markers=markers,
    )


def pytest_addoption(parser):
    """Add performance monitoring options"""
    parser.addoption(
        "--perf-threshold",
        action="store",
        default=240,
        type=int,
        help="Performance threshold in seconds (default: 240s = 4 minutes)",
    )
    parser.addoption(
        "--perf-report", action="store_true", help="Generate detailed performance report"
    )


def pytest_collection_modifyitems(config, items):
    """Mark slow tests based on historical data"""
    # Load historical performance data if available
    perf_history = Path(".pytest_cache/performance_history.json")
    slow_tests = set()

    if perf_history.exists():
        with open(perf_history) as f:
            history = json.load(f)
            # Mark tests that historically take >1 second as slow
            slow_tests = {
                test["nodeid"]
                for test in history.get("tests", [])
                if test.get("avg_duration", 0) > 1.0
            }

    # Apply slow marker to historically slow tests
    for item in items:
        if item.nodeid in slow_tests:
            item.add_marker(pytest.mark.slow)
