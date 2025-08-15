#!/bin/bash
# Production build script for PISAD
# Optimized for Raspberry Pi 5 deployment

set -e  # Exit on error

echo "🚀 Starting PISAD production build..."
echo "Target: Raspberry Pi 5 (ARM64)"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: Must run from project root directory${NC}"
    exit 1
fi

# Step 1: Clean previous builds
echo "📦 Step 1: Cleaning previous builds..."
rm -rf src/frontend/dist
rm -rf build
rm -rf dist
echo -e "${GREEN}✓ Clean complete${NC}"
echo ""

# Step 2: Install/update dependencies
echo "📦 Step 2: Installing dependencies..."
echo "  - Python dependencies with uv..."
uv sync --all-extras

echo "  - Frontend dependencies..."
cd src/frontend
npm ci
cd ../..
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Step 3: Run quality checks
echo "🔍 Step 3: Running quality checks..."
echo "  - Running trunk checks..."
npx trunk check --all --no-fix || {
    echo -e "${YELLOW}⚠ Some trunk checks failed. Review and fix before deployment.${NC}"
}

echo "  - Running Python type checks..."
uv run mypy src/backend --strict --ignore-missing-imports || {
    echo -e "${YELLOW}⚠ Type checking issues found. Review and fix before deployment.${NC}"
}

echo "  - Running TypeScript type checks..."
cd src/frontend
npm run tsc -- --noEmit || {
    echo -e "${YELLOW}⚠ TypeScript issues found. Review and fix before deployment.${NC}"
}
cd ../..
echo -e "${GREEN}✓ Quality checks complete${NC}"
echo ""

# Step 4: Run tests
echo "🧪 Step 4: Running tests..."
echo "  - Backend tests with coverage..."
uv run pytest tests/backend/unit -v --cov=src/backend --cov-fail-under=65 || {
    echo -e "${RED}✗ Backend tests failed or coverage below 65%${NC}"
    exit 1
}

echo "  - Frontend tests..."
cd src/frontend
npm test || {
    echo -e "${RED}✗ Frontend tests failed${NC}"
    exit 1
}
cd ../..
echo -e "${GREEN}✓ All tests passed${NC}"
echo ""

# Step 5: Build frontend
echo "🏗️ Step 5: Building frontend for production..."
cd src/frontend
NODE_ENV=production npm run build

# Check bundle size
BUNDLE_SIZE=$(du -sh dist | cut -f1)
echo "  Bundle size: ${BUNDLE_SIZE}"

# Check if bundle is under 5MB
BUNDLE_SIZE_MB=$(du -sm dist | cut -f1)
if [ "$BUNDLE_SIZE_MB" -gt 5 ]; then
    echo -e "${YELLOW}⚠ Warning: Bundle size (${BUNDLE_SIZE}) exceeds 5MB target${NC}"
else
    echo -e "${GREEN}✓ Bundle size optimal for Pi 5 deployment${NC}"
fi
cd ../..
echo ""

# Step 6: Tag version
echo "🏷️ Step 6: Version tagging..."
VERSION=$(grep version pyproject.toml | head -1 | cut -d'"' -f2)
GIT_HASH=$(git rev-parse --short HEAD)
BUILD_TIME=$(date -u +"%Y%m%d-%H%M%S")
BUILD_TAG="v${VERSION}-${GIT_HASH}-${BUILD_TIME}"

echo "  Version: ${VERSION}"
echo "  Git Hash: ${GIT_HASH}"
echo "  Build Tag: ${BUILD_TAG}"

# Create version file
cat > version.json <<EOF
{
  "version": "${VERSION}",
  "gitHash": "${GIT_HASH}",
  "buildTime": "${BUILD_TIME}",
  "buildTag": "${BUILD_TAG}"
}
EOF
echo -e "${GREEN}✓ Version tagged: ${BUILD_TAG}${NC}"
echo ""

# Step 7: Create deployment package
echo "📦 Step 7: Creating deployment package..."
mkdir -p dist

# Copy necessary files
cp -r src dist/
cp -r config dist/
cp -r deployment dist/
cp pyproject.toml dist/
cp uv.lock dist/
cp README.md dist/
cp version.json dist/

# Create deployment archive
tar -czf "pisad-${BUILD_TAG}.tar.gz" dist/

echo -e "${GREEN}✓ Deployment package created: pisad-${BUILD_TAG}.tar.gz${NC}"
echo ""

# Final summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✅ PRODUCTION BUILD COMPLETE${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Build Details:"
echo "  - Version: ${VERSION}"
echo "  - Build Tag: ${BUILD_TAG}"
echo "  - Frontend Bundle: ${BUNDLE_SIZE}"
echo "  - Package: pisad-${BUILD_TAG}.tar.gz"
echo ""
echo "Next Steps:"
echo "  1. Copy package to Pi 5: scp pisad-${BUILD_TAG}.tar.gz pisad@<pi-ip>:/home/pisad/"
echo "  2. Extract on Pi 5: tar -xzf pisad-${BUILD_TAG}.tar.gz"
echo "  3. Install service: sudo systemctl restart pisad"
echo "  4. Verify deployment: curl http://<pi-ip>:8080/api/health"
echo ""
echo "Build completed at $(date)"
