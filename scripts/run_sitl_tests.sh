#!/bin/bash
# Run SITL Integration Tests for PISAD
# Story 4.7 - Sprint 5: SITL Integration

set -e

echo "================================================"
echo "PISAD SITL Integration Test Runner"
echo "Story 4.7 - Sprint 5: SITL Integration"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if ArduPilot is installed
ARDUPILOT_PATH="$HOME/ardupilot"
if [ ! -d "$ARDUPILOT_PATH" ]; then
    echo -e "${YELLOW}ArduPilot not found at $ARDUPILOT_PATH${NC}"
    echo "Installing ArduPilot..."
    python3 scripts/sitl_setup.py install

    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install ArduPilot${NC}"
        exit 1
    fi
fi

# Check if SITL is built
if [ ! -f "$ARDUPILOT_PATH/build/sitl/bin/arducopter" ]; then
    echo -e "${YELLOW}SITL not built, building now...${NC}"
    python3 scripts/sitl_setup.py build --vehicle copter

    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to build SITL${NC}"
        exit 1
    fi
fi

# Function to cleanup SITL on exit
cleanup() {
    echo -e "\n${YELLOW}Cleaning up SITL processes...${NC}"
    pkill -f sim_vehicle.py 2>/dev/null || true
    pkill -f arducopter 2>/dev/null || true
    pkill -f mavproxy.py 2>/dev/null || true
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Start SITL in background
echo -e "\n${GREEN}Starting ArduPilot SITL...${NC}"
python3 scripts/sitl_setup.py start --vehicle copter --no-console &
SITL_PID=$!

# Wait for SITL to initialize
echo "Waiting for SITL to initialize..."
sleep 10

# Check if SITL is running
if ! kill -0 $SITL_PID 2>/dev/null; then
    echo -e "${RED}SITL failed to start${NC}"
    exit 1
fi

echo -e "${GREEN}SITL started successfully${NC}"

# Run the integration tests
echo -e "\n${GREEN}Running SITL Integration Tests...${NC}"
echo "================================================"

# Set environment variable to enable SITL tests
export RUN_SITL_TESTS=1

# Run tests with pytest
if command -v uv &> /dev/null; then
    # Use uv if available
    uv run pytest tests/backend/integration/test_sitl_integration.py -v --tb=short
else
    # Fallback to standard pytest
    pytest tests/backend/integration/test_sitl_integration.py -v --tb=short
fi

TEST_RESULT=$?

# Display results
echo "================================================"
if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ All SITL integration tests passed!${NC}"
else
    echo -e "${RED}✗ Some tests failed${NC}"
fi

# Stop SITL
echo -e "\n${YELLOW}Stopping SITL...${NC}"
python3 scripts/sitl_setup.py stop

exit $TEST_RESULT
