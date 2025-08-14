# CLAUDE.md - AI Assistant Context and Capabilities

## Project Overview
This is the PISAD (Passive Inverse Synthetic Aperture Detection) project - a system for drone signal detection and homing using SDR technology and MAVLink integration.

## Available MCP Servers and Tools

### When to Use Each Tool - Quick Reference

| Task | Use This Tool/Server | Specific Commands |
|------|---------------------|-------------------|
| Need library documentation | upstash-context-7-mcp | resolve-library-id → get-library-docs |
| Debug frontend issues | diulela-browser-tools-mcp | getConsoleErrors, runDebuggerMode |
| Find code examples | octocode | githubSearchCode, packageSearch |
| File operations | omniscience-labs-desktopcommandermcp | read_file, write_file, edit_block |
| Browser testing | playwright | browser_snapshot, browser_click |
| Fast code search (local) | ripgrep (rg) | rg "pattern", rg -t py "import" |
| Check PyPI packages | pypi-mcp-server | get_package_info, search_packages |
| Expose local API | fastapi-mcp | Create FastAPI server from tools |

---

## 1. upstash-context-7-mcp - Library Documentation

### When to Use:
- **Installing new packages**: Get docs before adding dependencies
- **Learning new APIs**: Understand library methods and patterns
- **Troubleshooting library issues**: Check official docs for solutions
- **Migration/upgrades**: Review breaking changes and migration guides

### Tools and Context:

#### resolve-library-id
**When**: ALWAYS call first when searching for any library documentation
**Context**: User asks about React hooks, Vue composition API, Express middleware, etc.
```
Example: User: "How do I use React hooks?"
Action: resolve-library-id(libraryName="react") → get-library-docs(topic="hooks")
```

#### get-library-docs
**When**: After resolving library ID, fetch actual documentation
**Context**: Need specific implementation details, API references, examples
**Parameters**:
- `tokens`: Use 10000 for overview, 20000+ for deep dive
- `topic`: Be specific - "useState", "routing", "middleware", etc.

---

## 2. diulela-browser-tools-mcp - Browser Debugging

### When to Use:
- **Frontend not working**: Console errors breaking functionality
- **Performance issues**: Page loading slowly
- **SEO problems**: Meta tags, structured data issues
- **Accessibility failures**: WCAG compliance checking
- **Network failures**: API calls failing, CORS issues

### Tools and Context:

#### Debugging Workflow:
```
1. getConsoleErrors - When: JS errors visible, features broken
2. getNetworkErrors - When: API calls failing, resources not loading
3. runDebuggerMode - When: Need comprehensive debugging session
4. takeScreenshot - When: Visual bugs, before/after comparisons
```

#### Audit Workflow:
```
1. runAuditMode - When: General optimization needed
2. runAccessibilityAudit - When: WCAG compliance required
3. runPerformanceAudit - When: Slow page loads, poor metrics
4. runSEOAudit - When: Search visibility issues
5. runNextJSAudit - When: Next.js specific optimizations
```

#### Best Context Examples:
- **User says "page is slow"** → runPerformanceAudit
- **User says "button doesn't work"** → getConsoleErrors → runDebuggerMode
- **User says "API failing"** → getNetworkErrors → getNetworkLogs
- **Before deployment** → runAuditMode for comprehensive check

---

## 3. octocode - GitHub Code Search

### When to Use:
- **Need implementation examples**: How others solved similar problems
- **Research best practices**: Find high-quality implementations
- **Debug obscure errors**: Search for similar error messages
- **Evaluate libraries**: Check real-world usage before adopting
- **Security research**: Find vulnerability patterns

### Tools and Context:

#### githubSearchCode
**When**: Looking for specific code patterns, functions, or implementations
**Context Strategy**:
```
Exploratory: queryTerms: ["signal"], ["beacon"], ["homing"]
Specific: queryTerms: ["calculateBearing", "signalStrength"]
Debug: queryTerms: ["Error: Cannot read property", "CORS"]
```

#### githubSearchRepositories
**When**: Finding projects using similar tech stack
**Context Examples**:
- Find SDR projects: topic=["software-defined-radio", "rtl-sdr"]
- Find drone projects: topic=["mavlink", "ardupilot", "px4"]
- Quality filter: stars=">100" for proven solutions

#### githubGetFileContent
**When**: Need to examine specific implementation in detail
**Context Usage**:
```
After search finds good example:
1. githubViewRepoStructure - Understand project layout
2. githubGetFileContent - Get specific file with matchString for context
```

#### packageSearch
**When**: Researching npm/Python packages before installation
**Context Examples**:
- User wants HTTP client: packageSearch(npmPackages=[{name: "axios"}, {name: "fetch"}])
- Compare options: Get download stats, dependencies, repo links

### Search Patterns for PISAD Project:
```
SDR: "rtl-sdr", "hackrf", "signal processing"
MAVLink: "pymavlink", "MAVSDK", "ardupilot"
Homing: "bearing calculation", "signal direction", "RSSI"
FastAPI: "websocket", "real-time", "async"
```

---

## 4. omniscience-labs-desktopcommandermcp - File Operations

### When to Use:
- **ANY file reading**: Instead of `cat`, use read_file
- **ANY file writing**: Instead of echo/redirect, use write_file
- **Code searching**: Instead of grep, use search_code
- **File listing**: Instead of ls, use list_directory
- **Code refactoring**: Use edit_block for surgical changes

### Tools and Context:

#### File Reading Context:
```
read_file:
- View entire file: No parameters
- Check logs: offset=-50 (last 50 lines)
- Read header: offset=0, length=20
- Large files: Use offset/length for pagination
```

#### File Writing Context:
```
write_file - ALWAYS CHUNK TO 25-30 LINES:
1. New file: mode='rewrite' for first chunk
2. Continue: mode='append' for subsequent chunks
3. User says "continue": Read file first, then append remaining
```

#### Edit Block Context:
```
edit_block - Surgical replacements:
- Changing function: Include 1-3 lines context
- Multiple changes: Separate edit_block calls
- Refactoring: Small, focused changes
```

#### Search Context:
```
search_files: Finding by name
- Config files: pattern="*.yaml" or "*.json"
- Python files: pattern="*.py"
- Test files: pattern="test_*.py"

search_code: Finding by content
- Find imports: pattern="^import.*mavlink"
- Find functions: pattern="def.*calculate"
- Find TODOs: pattern="TODO|FIXME"
```

### PISAD-Specific Paths:
```
Source code: /home/pisad/projects/pisad/src/
Tests: /home/pisad/projects/pisad/tests/
Config: /home/pisad/projects/pisad/config/
Documentation: /home/pisad/projects/pisad/docs/
```

---

## 5. playwright - Browser Automation

### When to Use:
- **E2E testing**: Test user workflows in browser
- **Web scraping**: Extract data from websites
- **Visual regression**: Screenshot comparisons
- **Form testing**: Automated form submission
- **Cross-browser testing**: Verify compatibility

### Tools and Context:

#### Testing Workflow:
```
1. browser_navigate - Go to test URL
2. browser_snapshot - Get page structure (BETTER than screenshot for actions)
3. browser_click/type - Interact with elements
4. browser_evaluate - Run JS assertions
5. browser_take_screenshot - Visual verification
```

#### Key Context Patterns:
```
Always use browser_snapshot before actions - provides element refs
Always provide element description for user permission
Use browser_evaluate for complex DOM queries
Handle async with browser_wait_for
```

### PISAD Testing Context:
```
Test dashboard: browser_navigate("http://localhost:3000")
Test WebSocket: browser_console_messages() after connection
Test real-time updates: browser_wait_for(text="Signal detected")
Test map interaction: browser_click on map markers
```

---

## 6. ripgrep (rg) - Ultra-Fast Code Search

### When to Use:
- **Code searching**: Faster than grep, git grep, ag, ack
- **Large codebases**: Searches gigabytes in milliseconds
- **Refactoring**: Find all occurrences before changes
- **Debugging**: Search for error messages, stack traces
- **Code review**: Find patterns, anti-patterns, TODOs
- **Security audit**: Search for hardcoded secrets, vulnerabilities

### Core Features:
```bash
# Basic search
rg "pattern"                      # Search in current directory
rg "pattern" /path/to/search      # Search in specific path
rg -i "pattern"                   # Case-insensitive search
rg -w "word"                      # Match whole words only
rg -F "literal string"            # Fixed string (no regex)
```

### Advanced Search Patterns:

#### File Type Filtering:
```bash
rg -t py "import numpy"           # Search only Python files
rg -t js "console.log"            # Search only JavaScript
rg -t yaml "apiVersion"           # Search only YAML files
rg -T md "TODO"                   # Exclude markdown files
rg -g "*.tsx" "useState"          # Glob pattern for file names
rg -g "!node_modules" "express"   # Exclude directories
```

#### Context and Output Control:
```bash
rg -A 3 "error"                   # Show 3 lines after match
rg -B 3 "error"                   # Show 3 lines before match
rg -C 3 "error"                   # Show 3 lines before and after
rg -l "pattern"                   # List only file names
rg -c "pattern"                   # Count matches per file
rg --stats "pattern"              # Show search statistics
```

#### Regex Power Features:
```bash
rg "def \w+\(.*\):"              # Find Python function definitions
rg "TODO|FIXME|XXX"              # Multiple patterns
rg "^import .* from"             # Lines starting with import
rg "console\.(log|error|warn)"  # Grouped alternatives
rg "\berror\b"                   # Word boundaries
```

#### Replacement (dry-run):
```bash
rg "oldFunc" --replace "newFunc" # Preview replacements
rg "v(\d+)" -r 'version$1'       # Capture groups
```

### PISAD Project-Specific Patterns:

#### Signal Processing:
```bash
rg -t py "signal|beacon|rssi" src/backend/services/
rg "calculateBearing|calculateDistance" --type-add 'pisad:*.py'
rg -A 5 "class.*Processor" src/
```

#### MAVLink Integration:
```bash
rg "mavlink|MAVLink" -g "!*.lock"
rg "connect_mavlink|send_command" src/backend/services/
rg -t py "pymavlink" --files-with-matches
```

#### Configuration:
```bash
rg -t yaml "sdr_" config/
rg "BEACON_FREQUENCY|SAMPLE_RATE" -t py
```

#### Error Tracking:
```bash
rg -i "error|exception" -g "*.log"
rg "logger\.(error|critical)" src/
rg "try:" -A 10 src/  # Find try blocks with context
```

### Performance Tips:

#### Parallel Search:
```bash
rg --threads 4 "pattern"         # Use 4 threads (auto-detects by default)
rg -j 1 "pattern"                # Single thread for ordered output
```

#### Binary and Hidden Files:
```bash
rg --hidden "pattern"            # Search hidden files
rg --no-ignore "pattern"         # Don't respect .gitignore
rg -a "pattern"                  # Search binary files as text
```

#### Smart Case:
```bash
rg -S "pattern"                  # Smart case (case-insensitive unless uppercase in pattern)
```

### Integration with Other Tools:

#### With fzf (fuzzy finder):
```bash
rg --files | fzf                # Interactive file selection
rg -l "pattern" | xargs -I {} code {}  # Open matches in VS Code
```

#### Git Integration:
```bash
rg "pattern" $(git diff --name-only)  # Search only changed files
rg "TODO" $(git ls-files)            # Search tracked files only
```

#### Pipe Chains:
```bash
rg -l "import numpy" | xargs rg "signal_process"  # Search in files that import numpy
rg "def.*test" -t py | wc -l                      # Count test functions
```

### Common Use Cases for PISAD:

#### 1. Find All API Endpoints:
```bash
rg "@app\.(get|post|put|delete)" src/backend/api/
rg "router\." -A 2 src/backend/api/routes/
```

#### 2. Locate Configuration Usage:
```bash
rg "config\." -t py src/
rg "os\.environ" src/  # Environment variables
```

#### 3. Find Untested Code:
```bash
rg "def \w+" src/ -l | xargs -I {} sh -c 'echo {}; rg -c "test_" tests/ | grep {}'
```

#### 4. Security Audit:
```bash
rg "(api_key|secret|password|token)" -i
rg "eval\(|exec\(" -t py  # Dangerous functions
rg "TODO.*security" -i
```

#### 5. WebSocket Messages:
```bash
rg "ws\.send|websocket\.send" src/
rg '"type":\s*"[^"]*"' -o  # Extract message types
```

### Ripgrep Config File (.ripgreprc):
```bash
# Create project-specific config
echo '--smart-case
--hidden
--glob=!.git
--glob=!node_modules
--glob=!.venv
--glob=!*.pyc
--glob=!*.pyo
--glob=!__pycache__
--glob=!*.log
--glob=!*.sqlite
--type-add=pisad:*.{py,tsx,ts,jsx,js}
--colors=match:fg:yellow
--colors=line:fg:green' > ~/.ripgreprc

export RIPGREP_CONFIG_PATH=~/.ripgreprc
```

### Performance Comparison:
```
Task: Search "signal" in PISAD project
grep -r: ~2.3s
git grep: ~0.8s
ag: ~0.4s
rg: ~0.05s  (45x faster than grep!)
```

### When NOT to Use ripgrep:
- Binary file analysis (use hexdump, strings)
- Log file tailing (use tail -f)
- Simple file reading (use cat, less)
- Database queries (use SQL)

---

## 7. pypi-mcp-server - PyPI Package Information

### When to Use:
- **Before installing packages**: Get package details, dependencies, version history
- **Comparing alternatives**: Search for similar packages, compare downloads/stats
- **Dependency analysis**: Check package dependencies and requirements
- **Version selection**: Review release history and compatibility
- **Package research**: Find packages by functionality, check maintenance status

### Tools and Context:

#### get_package_info
**When**: Need detailed information about a specific PyPI package
**Context**: Check before installation, verify compatibility, review dependencies
**Parameters**:
- `package_name`: Required - exact PyPI package name
- `version`: Optional - specific version (defaults to latest)
```
Example: User: "What dependencies does pytest require?"
Action: get_package_info(package_name="pytest")
```

#### search_packages
**When**: Looking for packages by functionality or keyword
**Context**: Finding alternatives, discovering new tools, exploring ecosystem
```
Example: User: "Find Python packages for SDR signal processing"
Action: search_packages(query="SDR signal processing")
```

#### get_package_releases
**When**: Need version history and release information
**Context**: Choosing stable versions, checking update frequency, migration planning
```
Example: User: "What versions of numpy are available?"
Action: get_package_releases(package_name="numpy")
```

#### get_package_stats
**When**: Evaluating package popularity and usage
**Context**: Assessing package health, comparing alternatives, trust evaluation
```
Example: User: "How popular is fastapi compared to flask?"
Action: get_package_stats(package_name="fastapi") + get_package_stats(package_name="flask")
```

### PISAD-Specific Use Cases:

#### SDR Package Research:
```
# Find SDR libraries
search_packages("rtl-sdr")
search_packages("hackrf")
search_packages("signal processing")

# Check before installation
get_package_info("pyrtlsdr")
get_package_info("scipy")
```

#### MAVLink/Drone Packages:
```
# MAVLink ecosystem
get_package_info("pymavlink")
get_package_info("mavsdk")
search_packages("ardupilot")

# Check compatibility
get_package_releases("pymavlink")  # Version history
```

#### Web Framework Dependencies:
```
# FastAPI ecosystem
get_package_info("fastapi")
get_package_info("uvicorn")
get_package_info("websockets")

# Check dependencies before upgrade
get_package_info("fastapi", version="0.100.0")
```

### Integration with Other Tools:

#### Package Discovery Workflow:
```
1. search_packages("keyword") - Find options
2. get_package_stats() - Compare popularity
3. get_package_info() - Check dependencies
4. githubSearchRepositories() - Find usage examples
5. get_package_releases() - Select version
```

#### Dependency Audit:
```
1. Read requirements.txt/pyproject.toml
2. For each package:
   - get_package_info() for current version
   - get_package_releases() for updates
   - get_package_stats() for health check
```

### Best Practices:
- Always check package info before adding to requirements
- Compare multiple packages using stats before choosing
- Review dependencies to avoid conflicts
- Check release frequency for maintenance status
- Verify Python version compatibility

---

## 8. fastapi-mcp - FastAPI Server Creation

### When to Use:
- **Expose tools as API**: Convert any MCP tool to REST endpoint
- **Create mock servers**: Test API integrations
- **Build microservices**: Quick API prototypes
- **WebSocket servers**: Real-time communication endpoints

### Installation:
```bash
uv add fastapi-mcp  # Already installed in this project
```

### Usage Context:
```python
from fastapi_mcp import FastAPIMCP

# Convert MCP tools to FastAPI endpoints
app = FastAPIMCP(
    mcp_server="your-mcp-server",
    tools=["tool1", "tool2"]
)

# Automatically generates:
# POST /tools/tool1
# POST /tools/tool2
# WebSocket /ws
```

### PISAD Integration:
- Signal processing endpoints
- Telemetry data streaming
- Mission control API
- Real-time beacon updates

---

## 9. Context7 Variants (community/http/sse)

### When to Use:
All three provide same functionality, choose based on connection:
- **context7-community**: Default, community-maintained
- **context7-http**: When SSE connections fail
- **context7-sse**: For streaming documentation updates

Same tools as upstash-context-7-mcp, use as fallback options.

---

## Integration Patterns for PISAD

### 1. Adding New SDR Feature
```
1. packageSearch for SDR libraries (numpy, scipy, rtl-sdr)
2. resolve-library-id → get-library-docs for implementation
3. githubSearchCode for "signal processing" examples
4. edit_block to integrate into signal_processor.py
5. playwright to test dashboard updates
```

### 2. Debugging MAVLink Issues
```
1. getConsoleErrors for frontend errors
2. getNetworkErrors for WebSocket issues
3. githubSearchCode for "pymavlink" + error message
4. search_code locally for MAVLink service
5. edit_block to fix issues
```

### 3. Optimizing Performance
```
1. runPerformanceAudit on dashboard
2. githubSearchCode for "react optimization"
3. search_code for inefficient patterns
4. edit_block to implement fixes
5. runPerformanceAudit to verify improvements
```

### 4. Field Test Preparation
```
1. read_file on test procedures
2. githubSearchRepositories for drone testing tools
3. write_file to create test scripts
4. browser_snapshot to verify dashboard ready
```

---

## Command Patterns to Remember

### Always Prefer:
```
read_file > cat
write_file > echo/redirect
rg > grep (ripgrep is 45x faster!)
search_code (MCP) > grep (for project search)
rg > git grep > ag > ack > grep (speed hierarchy)
list_directory > ls
edit_block > sed/awk
search_files > find
```

### Ripgrep vs Other Tools:
```
rg "pattern" > grep -r "pattern"           # 45x faster
rg -t py "import" > find . -name "*.py" | xargs grep "import"  # Simpler & faster
rg -l "TODO" > git grep -l "TODO"          # Works outside git repos
rg -C 3 "error" > grep -A 3 -B 3 "error"   # Cleaner context syntax
```

### File Operations:
```
ALWAYS use absolute paths: /home/pisad/projects/pisad/...
ALWAYS chunk writes to 25-30 lines
ALWAYS read before editing
```

### Search Strategy:
```
Start broad: Single word queries
Get specific: Exact function names
Use bulk: Multiple queries at once
```

### Testing Flow:
```
Snapshot first → Get refs → Interact → Verify
```

---

## Project-Specific Commands

### Running Tests:
```bash
pytest tests/backend/unit/
pytest tests/backend/integration/
pytest tests/e2e/
```

### Starting Services:
```bash
# Backend
uvicorn src.backend.main:app --reload

# Frontend
npm run dev

# Both
npm run dev:all
```

### Code Quality:
```bash
# Python
ruff check src/
mypy src/

# JavaScript/TypeScript
npm run lint
npm run typecheck
```

---

## Important Notes

1. **MCP servers provide specialized capabilities** - use the right tool for each task
2. **Batch operations when possible** - many tools support bulk queries
3. **Context is key** - provide specific topics/patterns for better results
4. **Always verify paths** before file operations
5. **Chunk file writes** to avoid performance issues
6. **Use browser_snapshot** not screenshot for automation
7. **Resolve library IDs** before fetching documentation
8. **Search progressively** - start broad, narrow based on results

This document should be referenced for all development tasks to ensure optimal tool usage and efficient problem-solving.