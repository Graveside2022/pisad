<!-- # CLAUDE.md - AI Assistant Tool Reference

## CRITICAL: Code Quality Enforcement

### Before ANY Code Submission
1. **Run ALL quality checks in parallel:**
   ```bash
   npx trunk check --all  # Meta-linter for all languages
   tsc --noEmit          # TypeScript type safety
   mypy src/             # Python type checking
   black --check src/    # Python formatting
   ruff check src/       # Python linting
   ```

2. **Choose the BEST tool for exceptional quality:**
   - **Speed critical**: rg > grep, fd > find, uv > pip
   - **Type safety**: tsc for TS, mypy for Python - NEVER skip
   - **Formatting**: black for Python, prettier via trunk for JS/TS
   - **Linting**: trunk.io orchestrates all linters - USE IT
   - **Testing**: pytest with coverage - aim for >80%

3. **Quality Gates - MUST PASS:**
   - Zero type errors (tsc/mypy)
   - All trunk checks pass
   - Code formatted (black/prettier)
   - Tests pass with coverage
   - No security issues (trunk includes security scanners)

### Automated Code Review Checklist
Before ANY code changes:
1. **Static Analysis**: `npx trunk check --all`
2. **Type Safety**: `tsc --noEmit && mypy src/`
3. **Formatting**: `black src/ && npx prettier --write .`
4. **Linting**: `ruff check src/ --fix`
5. **Security**: `npx trunk check --filter=security`
6. **Tests**: `pytest tests/ --cov=src --cov-fail-under=80`
7. **Performance**: Check with `pv` for data pipelines, `btop` for resource usage
8. **Dependencies**: Verify with `uv pip list` and `npm ls`

## MCP Servers and Their Tools

### 1. upstash-context-7-mcp
- **resolve-library-id**: Find library IDs for documentation lookup
- **get-library-docs**: Fetch library documentation with topic filtering

### 2. diulela-browser-tools-mcp
- **getConsoleLogs**: View browser console output
- **getConsoleErrors**: View browser console errors
- **getNetworkErrors**: View network error logs
- **getNetworkLogs**: View all network logs
- **takeScreenshot**: Capture browser screenshot
- **getSelectedElement**: Get currently selected DOM element
- **wipeLogs**: Clear browser logs
- **runAccessibilityAudit**: Check WCAG compliance
- **runPerformanceAudit**: Analyze page performance
- **runSEOAudit**: Check SEO optimization
- **runNextJSAudit**: Next.js specific audits
- **runDebuggerMode**: Comprehensive debugging session
- **runAuditMode**: Complete optimization audit
- **runBestPracticesAudit**: Check web best practices

### 3. octocode
- **githubSearchCode**: Search code across GitHub repositories
- **githubSearchRepositories**: Find GitHub repositories
- **githubGetFileContent**: Fetch file contents from GitHub
- **githubViewRepoStructure**: Explore repository structure
- **githubSearchCommits**: Search commit history
- **githubSearchPullRequests**: Search and analyze PRs
- **packageSearch**: Search NPM/PyPI packages

### 4. omniscience-labs-desktopcommandermcp
- **get_config**: Get server configuration
- **set_config_value**: Update configuration
- **read_file**: Read file contents with offset/length
- **read_multiple_files**: Read multiple files simultaneously
- **write_file**: Write/append to files (chunk to 25-30 lines)
- **create_directory**: Create directories
- **list_directory**: List directory contents
- **move_file**: Move/rename files
- **search_files**: Find files by name pattern
- **search_code**: Search code content using ripgrep
- **get_file_info**: Get file metadata
- **edit_block**: Surgical text replacements
- **execute_command**: Run terminal commands
- **read_output**: Read command output
- **force_terminate**: Kill terminal session
- **list_sessions**: List active sessions
- **list_processes**: List running processes
- **kill_process**: Terminate process by PID

### 5. playwright
- **browser_close**: Close browser page
- **browser_resize**: Resize browser window
- **browser_console_messages**: Get console messages
- **browser_handle_dialog**: Handle browser dialogs
- **browser_evaluate**: Execute JavaScript
- **browser_file_upload**: Upload files
- **browser_install**: Install browser
- **browser_press_key**: Press keyboard key
- **browser_type**: Type text into element
- **browser_navigate**: Navigate to URL
- **browser_navigate_back**: Go back
- **browser_navigate_forward**: Go forward
- **browser_network_requests**: Get network requests
- **browser_take_screenshot**: Take screenshot
- **browser_snapshot**: Get accessibility snapshot (prefer over screenshot)
- **browser_click**: Click element
- **browser_drag**: Drag and drop
- **browser_hover**: Hover over element
- **browser_select_option**: Select dropdown option
- **browser_tab_list**: List tabs
- **browser_tab_new**: Open new tab
- **browser_tab_select**: Select tab
- **browser_tab_close**: Close tab
- **browser_wait_for**: Wait for condition

### 6. pypi-mcp-server
- **get_package_info**: Get PyPI package details
- **search_packages**: Search PyPI packages
- **get_package_releases**: Get version history
- **get_package_stats**: Get download statistics

### 7. fastapi-mcp
- Create FastAPI servers from MCP tools

### 8. context7-http / context7-sse / context7-community
- **resolve-library-id**: Alternative library ID resolution
- **get-library-docs**: Alternative documentation fetching
- **resolve-library-uri**: Community version URI resolution
- **search-library-docs**: Community version doc search

### 9. filesystem
- **read_file**: Read file contents (deprecated, use read_text_file)
- **read_text_file**: Read text files with head/tail
- **read_media_file**: Read images/audio as base64
- **read_multiple_files**: Batch file reading
- **write_file**: Create/overwrite files
- **edit_file**: Line-based edits with diff
- **create_directory**: Create directories
- **list_directory**: List directory contents
- **list_directory_with_sizes**: List with file sizes
- **directory_tree**: Get recursive tree structure
- **move_file**: Move/rename files
- **search_files**: Find files by pattern
- **get_file_info**: Get file metadata
- **list_allowed_directories**: Get accessible directories

## System Tools

### File Operations
- **fd**: Find files and directories (faster than find, respects .gitignore)
- **find**: Traditional file search (use fd when possible)
- **fzf**: Fuzzy finder for interactive selection
- **ripgrep (rg)**: Ultra-fast text search (45x faster than grep)
- **grep**: Traditional text search (use rg when possible)
- **awk**: Text processing and data extraction
- **sed**: Stream editor for text transformation
- **jq**: JSON processor and query tool
- **rsync**: Efficient file synchronization

### System Monitoring
- **btop**: Interactive process viewer (better than htop/top)
- **duf**: Disk usage with better formatting (better than df)
- **pv**: Pipe viewer for monitoring data flow in pipelines
- **strace**: System call tracer for debugging

### Directory Navigation
- **zoxide**: Smart directory jumper (better than cd)
  - `z <partial-path>`: Jump to frecent directories
  - `zi`: Interactive directory selection

### Package Management
- **uv**: Fast Python package installer (10-100x faster than pip)
- **uvx**: Run Python tools in isolated environments without installation

### Development Tools
- **tsc**: TypeScript compiler and type checker (MUST run before commits)
- **trunk.io (npx trunk)**: Meta-linter orchestrating all code quality tools
- **make**: Build automation tool
- **bats**: Bash Automated Testing System for shell script testing

## Python Tools (uv-installed)

### Testing Frameworks
- **pytest**: Python testing framework
- **pytest-asyncio**: Async test support
- **pytest-benchmark**: Performance benchmarking
- **pytest-cov**: Coverage reporting
- **pytest-mock**: Mock/patch helpers
- **pytest-timeout**: Test timeout management
- **pytest-xdist**: Parallel test execution
- **hypothesis**: Property-based testing
- **responses**: Mock HTTP responses
- **factory-boy**: Test fixture factory
- **faker**: Fake data generation

### Code Quality (MUST USE)
- **black**: Python code formatter (run before every commit)
- **ruff**: Fast Python linter (replaces flake8/pylint, 10-100x faster)
- **flake8**: Style guide enforcement (use ruff instead when possible)
- **mypy**: Static type checker (MANDATORY for type safety)
- **coverage**: Code coverage measurement (maintain >80%)
- **ipdb**: Interactive Python debugger

### Development Libraries
- **pre-commit**: Git hook framework for code quality checks (installed via uv)
- **fastapi**: Modern web API framework
- **uvicorn**: ASGI server
- **httpx**: HTTP client with async support
- **pydantic**: Data validation using Python type annotations
- **pydantic-settings**: Settings management with environment variables
- **sqlalchemy**: SQL toolkit and ORM
- **websockets**: WebSocket implementation
- **starlette**: ASGI framework (FastAPI foundation)
- **sse-starlette**: Server-sent events for Starlette/FastAPI
- **httptools**: High-performance HTTP parser
- **uvloop**: Ultra-fast asyncio event loop
- **watchfiles**: Fast file watching for hot reload
- **python-multipart**: Form data parsing
- **mcp**: Model Context Protocol implementation
- **rich**: Terminal formatting and display
- **typer**: CLI application builder
- **python-dotenv**: Environment variable management
- **aiofiles**: Async file operations
- **fastapi-mcp**: FastAPI MCP integration

### Scientific/Data
- **numpy**: Numerical computing
- **scipy**: Scientific computing
- **matplotlib**: Plotting library
- **pandas**: Data analysis (if installed)

### Project-Specific
- **pymavlink**: MAVLink protocol implementation
- **pyserial**: Serial port communication (required for pymavlink)
- **fastcrc**: Fast CRC calculations
- **psutil**: System and process utilities
- **lxml**: XML/HTML processing
- **reportlab**: PDF generation

## Tool Selection Guidelines

### Use Instead Of
- **fd** > find (respects .gitignore, faster, better UX)
- **rg** > grep (45x faster, better defaults)
- **zoxide** > cd (frecency-based navigation)
- **uv** > pip (10-100x faster)
- **ruff** > flake8/pylint (faster, more comprehensive)
- **btop** > htop > top (better visualization)
- **duf** > df (cleaner output)
- **fzf** with any command for interactive selection
- **pv** for any pipe operation needing progress
- **jq** > python/awk for JSON (purpose-built)

### Performance Hierarchy
- Text search: rg > ag > ack > git grep > grep
- File finding: fd > find
- Python packages: uv > pip
- Python linting: ruff > flake8 > pylint
- Process monitoring: btop > htop > top

## Command Reminders

### Always Use
- Absolute paths: `/home/pisad/projects/pisad/...`
- Chunk file writes to 25-30 lines
- Read before editing files
- browser_snapshot before browser interactions
- resolve-library-id before get-library-docs

### Code Quality Workflow (MANDATORY)
1. **Before writing code**: Check existing patterns with `rg`
2. **While coding**: Run `mypy` and `tsc` frequently
3. **Before commits**:
   ```bash
   # Python
   black src/
   ruff check src/ --fix
   mypy src/
   pytest tests/ --cov=src --cov-report=term-missing

   # TypeScript/JavaScript
   npx trunk check --all
   tsc --noEmit
   npm run lint
   npm test
   ```
4. **Final check**: `npx trunk check --all --no-fix`

### Tool Selection for Excellence
- **Finding code**: `rg` for speed, `fd` for files
- **Package management**: Always `uv`, never `pip`
- **Running tools**: `uvx` for isolated execution
- **Type checking**: Never skip `mypy` (Python) or `tsc` (TypeScript)
- **Formatting**: `black` (Python), `prettier` via trunk (JS/TS)
- **Linting**: `trunk check` orchestrates everything
- **Testing**: `pytest` with coverage reports
- **JSON processing**: `jq` for queries and transformations
- **Pipeline monitoring**: `pv` to see progress
- **Process debugging**: `strace` for system calls
- **Shell testing**: `bats` for test suites

## Project Paths
- Source: `/home/pisad/projects/pisad/src/`
- Tests: `/home/pisad/projects/pisad/tests/`
- Config: `/home/pisad/projects/pisad/config/`
- Docs: `/home/pisad/projects/pisad/docs/` -->
