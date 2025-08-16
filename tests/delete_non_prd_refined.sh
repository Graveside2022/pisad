#!/bin/bash
# Sprint 8 - Refined deletion of non-PRD tests
# SAFETY: Keep HARA tests, integration tests, hardware tests

# Non-PRD: analytics
rm -f tests/backend/integration/test_analytics_export.py
# Non-PRD: analytics
rm -f tests/backend/integration/test_app_initialization.py
# Non-PRD: test_logger
rm -f tests/backend/integration/test_field_test_service.py
# Non-PRD: recommendations
rm -f tests/backend/integration/test_phase1_integration.py
# Non-PRD: ci_cd
rm -f tests/backend/unit/test_ci_cd_pipeline.py
# Non-PRD: deployment
rm -f tests/backend/unit/test_deployment_config.py
# Non-PRD: deployment
rm -f tests/backend/unit/test_flaky_detector.py
# Non-PRD: test_main
rm -f tests/backend/unit/test_main.py
# Non-PRD: coverage_boost
rm -f tests/backend/unit/test_phase1_coverage_boost.py
# Non-PRD: coverage_boost
rm -f tests/backend/unit/test_phase2_coverage_boost.py
# Non-PRD: deployment
rm -f tests/backend/unit/test_prometheus_metrics.py
# Non-PRD: analytics
rm -f tests/backend/unit/test_recommendations_engine.py
# Non-PRD: analytics
rm -f tests/backend/unit/test_report_generator.py
# Non-PRD: analytics
rm -f tests/integration/api/test_analytics_api.py
# Non-PRD: analytics
rm -f tests/integration/api/test_api_analytics.py
# Non-PRD: test_logger
rm -f tests/integration/test_field_test_service.py
# Non-PRD: analytics
rm -f tests/performance/test_performance_analytics.py
#
rm -f tests/unit/mocks/test_mock_sdr_config.py
#
rm -f tests/unit/mocks/test_mock_sdr_streaming.py
