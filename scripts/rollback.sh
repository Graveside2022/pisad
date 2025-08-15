#!/bin/bash
# Rollback script for PISAD deployment
# Allows quick rollback to previous version using git tags

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "🔄 PISAD Rollback Script"
echo "========================"
echo ""

# Check we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: Must run from project root directory${NC}"
    exit 1
fi

# Check git is available
if ! command -v git &> /dev/null; then
    echo -e "${RED}Error: git is not installed${NC}"
    exit 1
fi

# Function to get current version
get_current_version() {
    if [ -f "version.json" ]; then
        grep '"version"' version.json | cut -d'"' -f4
    else
        grep version pyproject.toml | head -1 | cut -d'"' -f2
    fi
}

# Function to list available versions
list_versions() {
    echo "Available versions (tags):"
    git tag -l "v*" --sort=-version:refname | head -10
}

# Function to perform rollback
rollback_to_version() {
    local target_version=$1

    echo -e "${YELLOW}⚠ WARNING: This will rollback to version ${target_version}${NC}"
    echo "Current version: $(get_current_version)"
    echo ""
    read -p "Are you sure you want to continue? (y/N): " -n 1 -r
    echo ""

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Rollback cancelled"
        exit 0
    fi

    # Create backup tag of current state
    local backup_tag="backup-$(date +%Y%m%d-%H%M%S)"
    echo "Creating backup tag: ${backup_tag}"
    git tag "${backup_tag}"

    # Stash any local changes
    echo "Stashing local changes..."
    git stash push -m "Rollback stash $(date)"

    # Checkout target version
    echo "Checking out version ${target_version}..."
    git checkout "${target_version}"

    # Install dependencies for this version
    echo "Installing dependencies..."
    uv sync --all-extras
    cd src/frontend && npm ci && cd ../..

    # Rebuild frontend
    echo "Rebuilding frontend..."
    cd src/frontend && npm run build && cd ../..

    # Restart service
    echo "Restarting service..."
    sudo systemctl restart pisad

    echo -e "${GREEN}✅ Rollback complete!${NC}"
    echo "Rolled back to version: ${target_version}"
    echo "Backup created with tag: ${backup_tag}"
    echo ""
    echo "To restore the backup, run:"
    echo "  git checkout ${backup_tag}"
}

# Function to create version tag
create_version_tag() {
    local version=$(get_current_version)
    local tag_name="v${version}"

    echo "Creating version tag: ${tag_name}"

    # Check if tag already exists
    if git rev-parse "${tag_name}" >/dev/null 2>&1; then
        echo -e "${YELLOW}Tag ${tag_name} already exists${NC}"
        read -p "Create timestamped tag instead? (y/N): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            tag_name="${tag_name}-$(date +%Y%m%d-%H%M%S)"
            echo "Creating tag: ${tag_name}"
        else
            echo "Tag creation cancelled"
            return
        fi
    fi

    # Create annotated tag
    git tag -a "${tag_name}" -m "Release version ${version}"
    echo -e "${GREEN}✅ Tag created: ${tag_name}${NC}"

    read -p "Push tag to remote? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git push origin "${tag_name}"
        echo -e "${GREEN}✅ Tag pushed to remote${NC}"
    fi
}

# Function to show version history
show_history() {
    echo "Version History (last 10 commits):"
    echo "=================================="
    git log --oneline --decorate --graph -10
}

# Function to verify deployment
verify_deployment() {
    echo "Verifying deployment..."
    echo ""

    # Check service status
    echo "Service status:"
    systemctl status pisad --no-pager | head -10
    echo ""

    # Check API health
    echo "API health check:"
    curl -s http://localhost:8080/api/health | jq . || echo "API not responding"
    echo ""

    # Check current version
    echo "Current version: $(get_current_version)"
    echo "Git branch: $(git branch --show-current)"
    echo "Git commit: $(git rev-parse --short HEAD)"
}

# Main menu
show_menu() {
    echo "Select an option:"
    echo "  1) List available versions"
    echo "  2) Rollback to specific version"
    echo "  3) Create version tag"
    echo "  4) Show version history"
    echo "  5) Verify current deployment"
    echo "  6) Emergency rollback to last tag"
    echo "  0) Exit"
    echo ""
}

# Emergency rollback function
emergency_rollback() {
    echo -e "${RED}🚨 EMERGENCY ROLLBACK${NC}"
    echo "Rolling back to last stable version..."

    # Get last tag
    local last_tag=$(git describe --tags --abbrev=0)

    if [ -z "$last_tag" ]; then
        echo -e "${RED}Error: No tags found for rollback${NC}"
        exit 1
    fi

    echo "Last stable version: ${last_tag}"
    rollback_to_version "${last_tag}"
}

# Parse command line arguments
if [ "$1" == "--emergency" ]; then
    emergency_rollback
    exit 0
elif [ "$1" == "--version" ] && [ -n "$2" ]; then
    rollback_to_version "$2"
    exit 0
elif [ "$1" == "--tag" ]; then
    create_version_tag
    exit 0
elif [ "$1" == "--verify" ]; then
    verify_deployment
    exit 0
fi

# Interactive mode
while true; do
    show_menu
    read -p "Enter choice: " choice

    case $choice in
        1)
            list_versions
            echo ""
            ;;
        2)
            list_versions
            echo ""
            read -p "Enter version to rollback to (e.g., v1.0.0): " version
            if [ -n "$version" ]; then
                rollback_to_version "$version"
            fi
            ;;
        3)
            create_version_tag
            echo ""
            ;;
        4)
            show_history
            echo ""
            ;;
        5)
            verify_deployment
            echo ""
            ;;
        6)
            emergency_rollback
            ;;
        0)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option${NC}"
            echo ""
            ;;
    esac

    read -p "Press Enter to continue..."
    clear
done
