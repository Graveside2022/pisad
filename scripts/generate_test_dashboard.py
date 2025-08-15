#!/usr/bin/env python3
"""
Generate Test Quality Dashboard

This script generates a comprehensive test quality dashboard with:
- Test value ratios and traceability metrics
- Hazard coverage analysis
- Performance metrics and trends
- Quality gates and recommendations
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backend.utils.test_metrics import TestMetricsAnalyzer


def generate_html_dashboard(metrics_data: dict) -> str:
    """Generate HTML dashboard from metrics data"""
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PISAD Test Quality Dashboard</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background: white;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            color: #333;
            font-size: 2.5em;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9em;
            margin-top: 10px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .metric-title {{
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #333;
        }}
        .metric-value.good {{
            color: #10b981;
        }}
        .metric-value.warning {{
            color: #f59e0b;
        }}
        .metric-value.bad {{
            color: #ef4444;
        }}
        .section {{
            background: white;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #e5e7eb;
            padding-bottom: 10px;
        }}
        .progress-bar {{
            background: #e5e7eb;
            border-radius: 10px;
            height: 30px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
            display: flex;
            align-items: center;
            padding: 0 10px;
            color: white;
            font-weight: bold;
            transition: width 0.3s ease;
        }}
        .progress-fill.warning {{
            background: linear-gradient(90deg, #f59e0b 0%, #fbbf24 100%);
        }}
        .progress-fill.bad {{
            background: linear-gradient(90deg, #ef4444 0%, #f87171 100%);
        }}
        .hazard-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 10px;
            margin: 20px 0;
        }}
        .hazard-badge {{
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
        }}
        .hazard-badge.covered {{
            background: #d1fae5;
            color: #065f46;
        }}
        .hazard-badge.uncovered {{
            background: #fee2e2;
            color: #991b1b;
        }}
        .test-list {{
            max-height: 300px;
            overflow-y: auto;
            background: #f9fafb;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
        }}
        .test-item {{
            padding: 8px;
            margin: 5px 0;
            background: white;
            border-radius: 3px;
            font-family: monospace;
            font-size: 0.9em;
            border-left: 3px solid #e5e7eb;
        }}
        .test-item.slow {{
            border-left-color: #f59e0b;
        }}
        .test-item.untraceable {{
            border-left-color: #ef4444;
        }}
        .quality-gate {{
            display: flex;
            align-items: center;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            background: #f9fafb;
        }}
        .quality-gate.pass {{
            background: #d1fae5;
            color: #065f46;
        }}
        .quality-gate.fail {{
            background: #fee2e2;
            color: #991b1b;
        }}
        .quality-gate-icon {{
            font-size: 1.5em;
            margin-right: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 PISAD Test Quality Dashboard</h1>
            <div class="timestamp">Generated: {metrics_data['timestamp']}</div>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-title">Total Tests</div>
                <div class="metric-value">{metrics_data['summary']['total_tests']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Test Value Ratio</div>
                <div class="metric-value {get_status_class(float(metrics_data['summary']['test_value_ratio'].rstrip('%')), 80, 60)}">{metrics_data['summary']['test_value_ratio']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Hazard Coverage</div>
                <div class="metric-value {get_status_class(float(metrics_data['summary']['hazard_coverage'].rstrip('%')), 100, 80)}">{metrics_data['summary']['hazard_coverage']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Execution Time</div>
                <div class="metric-value {get_time_status_class(float(metrics_data['summary']['execution_time'].rstrip('s')))}">{metrics_data['summary']['execution_time']}</div>
            </div>
        </div>

        <div class="section">
            <h2>📊 Traceability Metrics</h2>
            <div>
                <p>Test Value Ratio (tests with traceability)</p>
                <div class="progress-bar">
                    <div class="progress-fill {get_status_class(metrics_data['traceability']['value_ratio'], 80, 60)}" style="width: {metrics_data['traceability']['value_ratio']}%">
                        {metrics_data['traceability']['value_ratio']:.1f}%
                    </div>
                </div>
            </div>
            <div>
                <p>User Story Coverage</p>
                <div class="progress-bar">
                    <div class="progress-fill {get_status_class(metrics_data['traceability']['story_coverage'], 70, 50)}" style="width: {metrics_data['traceability']['story_coverage']}%">
                        {metrics_data['traceability']['story_coverage']:.1f}%
                    </div>
                </div>
            </div>
            <div>
                <p>Hazard Test Coverage</p>
                <div class="progress-bar">
                    <div class="progress-fill {get_status_class(metrics_data['traceability']['hazard_test_coverage'], 70, 50)}" style="width: {metrics_data['traceability']['hazard_test_coverage']}%">
                        {metrics_data['traceability']['hazard_test_coverage']:.1f}%
                    </div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>🛡️ Hazard Coverage Analysis</h2>
            <div class="hazard-grid">
                {generate_hazard_badges(metrics_data['hazards'])}
            </div>
            {generate_uncovered_hazards(metrics_data['hazards']['uncovered'])}
        </div>

        <div class="section">
            <h2>⚡ Performance Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-title">Unit Tests</div>
                    <div class="metric-value">{metrics_data['performance']['unit_test_time']:.1f}s</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Integration Tests</div>
                    <div class="metric-value">{metrics_data['performance']['integration_test_time']:.1f}s</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">E2E Tests</div>
                    <div class="metric-value">{metrics_data['performance']['e2e_test_time']:.1f}s</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Peak Memory</div>
                    <div class="metric-value">{metrics_data['performance']['peak_memory_mb']:.0f} MB</div>
                </div>
            </div>

            {generate_slow_tests(metrics_data['performance']['slowest_tests'])}
        </div>

        <div class="section">
            <h2>✅ Quality Gates</h2>
            {generate_quality_gates(metrics_data)}
        </div>

        {generate_quality_issues(metrics_data['quality_issues'])}
    </div>
</body>
</html>
    """

    return html


def get_status_class(value: float, good_threshold: float, warning_threshold: float) -> str:
    """Get CSS class based on value and thresholds"""
    if value >= good_threshold:
        return "good"
    elif value >= warning_threshold:
        return "warning"
    else:
        return "bad"


def get_time_status_class(seconds: float) -> str:
    """Get CSS class for execution time"""
    if seconds <= 240:  # 4 minutes
        return "good"
    elif seconds <= 300:  # 5 minutes
        return "warning"
    else:
        return "bad"


def generate_hazard_badges(hazards: dict) -> str:
    """Generate HTML for hazard coverage badges"""
    badges = []
    for hazard in hazards["defined"]:
        status = "covered" if hazard in hazards["covered"] else "uncovered"
        badges.append(f'<div class="hazard-badge {status}">{hazard}</div>')
    return "\n".join(badges)


def generate_uncovered_hazards(uncovered: list) -> str:
    """Generate HTML for uncovered hazards warning"""
    if not uncovered:
        return ""

    hazards_list = "\n".join([f"<li>{h}</li>" for h in uncovered])
    return f"""
    <div style="background: #fee2e2; padding: 15px; border-radius: 5px; margin-top: 20px;">
        <strong style="color: #991b1b;">⚠️ Uncovered Hazards:</strong>
        <ul style="margin: 10px 0 0 20px;">
            {hazards_list}
        </ul>
    </div>
    """


def generate_slow_tests(slow_tests: list) -> str:
    """Generate HTML for slow tests list"""
    if not slow_tests:
        return ""

    tests_html = "\n".join(
        [f'<div class="test-item slow">{t[1]:.2f}s - {t[0]}</div>' for t in slow_tests[:5]]
    )

    return f"""
    <h3>Slowest Tests</h3>
    <div class="test-list">
        {tests_html}
    </div>
    """


def generate_quality_gates(metrics: dict) -> str:
    """Generate quality gate status HTML"""
    gates = []

    # Test Value Ratio Gate
    value_ratio = metrics["traceability"]["value_ratio"]
    gates.append(
        {
            "name": "Test Value Ratio ≥ 80%",
            "pass": value_ratio >= 80,
            "value": f"{value_ratio:.1f}%",
        }
    )

    # Hazard Coverage Gate
    hazard_coverage = metrics["hazards"]["coverage_percentage"]
    gates.append(
        {
            "name": "Hazard Coverage = 100%",
            "pass": hazard_coverage == 100,
            "value": f"{hazard_coverage:.1f}%",
        }
    )

    # Execution Time Gate
    exec_time = metrics["performance"]["total_runtime"]
    gates.append(
        {
            "name": "Execution Time < 4 minutes",
            "pass": exec_time <= 240,
            "value": f"{exec_time:.1f}s",
        }
    )

    # Zero Untraceable Tests Gate
    untraceable = len(metrics["quality_issues"]["untraceable_tests"])
    gates.append(
        {
            "name": "Zero Untraceable Tests",
            "pass": untraceable == 0,
            "value": f"{untraceable} tests",
        }
    )

    gates_html = []
    for gate in gates:
        status = "pass" if gate["pass"] else "fail"
        icon = "✅" if gate["pass"] else "❌"
        gates_html.append(f"""
        <div class="quality-gate {status}">
            <div class="quality-gate-icon">{icon}</div>
            <div>
                <strong>{gate['name']}</strong><br>
                Current: {gate['value']}
            </div>
        </div>
        """)

    return "\n".join(gates_html)


def generate_quality_issues(issues: dict) -> str:
    """Generate quality issues section HTML"""
    if not any([issues["untraceable_tests"], issues["uncovered_hazards"], issues["slow_tests"]]):
        return ""

    sections = []

    if issues["untraceable_tests"]:
        tests_list = "\n".join(
            [
                f'<div class="test-item untraceable">{t}</div>'
                for t in issues["untraceable_tests"][:5]
            ]
        )
        sections.append(f"""
        <h3>⚠️ Untraceable Tests</h3>
        <div class="test-list">
            {tests_list}
        </div>
        """)

    return f"""
    <div class="section">
        <h2>⚠️ Quality Issues</h2>
        {''.join(sections)}
    </div>
    """


def main():
    parser = argparse.ArgumentParser(description="Generate test quality dashboard")
    parser.add_argument(
        "--output",
        default="test_dashboard.html",
        help="Output file path (default: test_dashboard.html)",
    )
    parser.add_argument("--json", action="store_true", help="Also save raw JSON data")
    parser.add_argument(
        "--test-dir", default="tests", help="Test directory to analyze (default: tests)"
    )
    args = parser.parse_args()

    # Run analysis
    print("🔍 Analyzing test suite...")
    analyzer = TestMetricsAnalyzer(Path(args.test_dir))
    analyzer.analyze_test_suite()

    # Check for performance report
    perf_report = Path(".pytest_cache/performance_report.json")
    if perf_report.exists():
        print("📊 Loading performance data...")
        analyzer.measure_test_performance(Path(".pytest_cache/report.json"))

    # Generate dashboard data
    dashboard_data = analyzer.generate_quality_dashboard()

    # Save JSON if requested
    if args.json:
        json_path = args.output.replace(".html", ".json")
        with open(json_path, "w") as f:
            json.dump(dashboard_data, f, indent=2)
        print(f"✅ JSON data saved to {json_path}")

    # Generate and save HTML
    html = generate_html_dashboard(dashboard_data)

    # Fix template string issues
    html = html.replace("{metrics_data[", "{").replace("]}", "}")

    with open(args.output, "w") as f:
        f.write(html)

    print(f"✅ Dashboard saved to {args.output}")

    # Print summary
    analyzer.print_summary()

    # Check quality gates
    print("\n" + "=" * 60)
    print("QUALITY GATE STATUS")
    print("=" * 60)

    gates_passed = True

    if analyzer.value_metrics.value_ratio < 80:
        print("❌ Test Value Ratio: FAILED (< 80%)")
        gates_passed = False
    else:
        print("✅ Test Value Ratio: PASSED (≥ 80%)")

    if analyzer.hazard_metrics.coverage_percentage < 100:
        print("❌ Hazard Coverage: FAILED (< 100%)")
        gates_passed = False
    else:
        print("✅ Hazard Coverage: PASSED (= 100%)")

    if analyzer.performance_metrics.total_runtime > 240:
        print("❌ Execution Time: FAILED (> 4 minutes)")
        gates_passed = False
    else:
        print("✅ Execution Time: PASSED (≤ 4 minutes)")

    if analyzer.value_metrics.untraceable_tests:
        print(f"⚠️  {len(analyzer.value_metrics.untraceable_tests)} untraceable tests found")

    print("=" * 60)

    return 0 if gates_passed else 1


if __name__ == "__main__":
    sys.exit(main())
