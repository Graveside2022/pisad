#!/bin/bash

# Script to add Serena MCP server to Claude Code configuration

echo "Setting up Serena MCP server for Claude Code..."

# Check if the .claude-code directory exists
if [ ! -d "$HOME/.claude-code" ]; then
    echo "Creating ~/.claude-code directory..."
    mkdir -p "$HOME/.claude-code"
fi

# Create or update the settings.json file
SETTINGS_FILE="$HOME/.claude-code/settings.json"

# Check if settings.json exists
if [ -f "$SETTINGS_FILE" ]; then
    echo "Backing up existing settings.json to settings.json.backup"
    cp "$SETTINGS_FILE" "$SETTINGS_FILE.backup"

    # Check if mcpServers already exists in the file
    if grep -q '"mcpServers"' "$SETTINGS_FILE"; then
        echo "mcpServers configuration already exists. Please manually add Serena server."
        echo "Add this to your mcpServers section:"
        cat << 'EOF'
    "serena": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/oraios/serena",
        "serena",
        "start-mcp-server",
        "--context",
        "ide-assistant",
        "--project",
        "/home/pisad/projects/pisad"
      ]
    }
EOF
        exit 1
    fi
fi

# Create the settings.json with Serena MCP configuration
cat > "$SETTINGS_FILE" << 'EOF'
{
  "mcpServers": {
    "serena": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/oraios/serena",
        "serena",
        "start-mcp-server",
        "--context",
        "ide-assistant",
        "--project",
        "/home/pisad/projects/pisad"
      ]
    }
  }
}
EOF

echo "Serena MCP server configuration added to $SETTINGS_FILE"
echo ""
echo "Please restart Claude Code for the changes to take effect."
echo "You can restart by running: claude-code --restart"
