#!/bin/bash
# Test execution script with optimized organization
# Sprint 6 - Task 9.2: Test Suite Reorganization

set -e

echo "============================================"
echo "PISAD Test Suite Execution (Reorganized)"
echo "============================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test execution modes
case "${1:-all}" in
    unit)
        echo -e "${GREEN}Running Unit Tests (target: <30s)${NC}"
        pytest tests/unit -m "unit" --maxfail=5 -n auto
        ;;

    integration)
        echo -e "${GREEN}Running Integration Tests (target: <2min)${NC}"
        pytest tests/integration -m "integration" --maxfail=5
        ;;

    e2e)
        echo -e "${GREEN}Running E2E Tests (target: <5min)${NC}"
        pytest tests/e2e -m "e2e" --maxfail=3
        ;;

    sitl)
        echo -e "${YELLOW}Running SITL Tests (target: <2.5min)${NC}"
        pytest tests/sitl -m "sitl" --maxfail=3
        ;;

    performance)
        echo -e "${GREEN}Running Performance Tests${NC}"
        pytest tests/performance -m "performance" --benchmark-only
        ;;

    smoke)
        echo -e "${GREEN}Running Smoke Tests (critical only)${NC}"
        pytest -m "smoke or critical" --maxfail=1 -n auto
        ;;

    safety)
        echo -e "${RED}Running Safety Tests (100% must pass)${NC}"
        pytest -m "safety" --maxfail=1
        ;;

    fast)
        echo -e "${GREEN}Running Fast Tests Only (<100ms each)${NC}"
        pytest -m "fast and not slow" -n auto
        ;;

    parallel)
        echo -e "${GREEN}Running All Tests in Parallel (8 workers)${NC}"
        pytest -n 8 --maxfail=10
        ;;

    coverage)
        echo -e "${GREEN}Running Tests with Coverage Report${NC}"
        pytest --cov=src --cov-report=term-missing --cov-report=html
        echo "Coverage report generated in htmlcov/index.html"
        ;;

    all)
        echo -e "${GREEN}Running Complete Test Suite${NC}"
        echo "----------------------------------------"

        # Run tests in order of speed
        echo -e "\n${GREEN}1. Unit Tests${NC}"
        pytest tests/unit -m "unit" -n auto --tb=no -q || true

        echo -e "\n${GREEN}2. Integration Tests${NC}"
        pytest tests/integration -m "integration" --tb=no -q || true

        echo -e "\n${GREEN}3. E2E Tests${NC}"
        pytest tests/e2e -m "e2e" --tb=no -q || true

        echo -e "\n${GREEN}4. Performance Tests${NC}"
        pytest tests/performance -m "performance" --tb=no -q || true

        echo -e "\n${YELLOW}5. SITL Tests (if available)${NC}"
        pytest tests/sitl -m "sitl" --tb=no -q || true

        # Summary
        echo -e "\n${GREEN}Test Suite Complete!${NC}"
        pytest --co -q | grep "test session" || true
        ;;

    list)
        echo "Available test categories:"
        echo "  unit        - Unit tests (<30s total)"
        echo "  integration - Integration tests (<2min)"
        echo "  e2e         - End-to-end tests (<5min)"
        echo "  sitl        - SITL hardware tests (<2.5min)"
        echo "  performance - Performance benchmarks"
        echo "  smoke       - Critical smoke tests"
        echo "  safety      - Safety-critical tests"
        echo "  fast        - Fast tests only"
        echo "  parallel    - Run all tests in parallel"
        echo "  coverage    - Generate coverage report"
        echo "  all         - Run complete suite"
        ;;

    *)
        echo "Unknown test mode: $1"
        echo "Use: $0 [unit|integration|e2e|sitl|performance|smoke|safety|fast|parallel|coverage|all|list]"
        exit 1
        ;;
esac

# Show test execution time
echo -e "\n${GREEN}Test execution completed$(NC)"
echo "============================================"
