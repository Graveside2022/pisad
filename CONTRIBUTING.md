# Contributing to PISAD

Thank you for your interest in contributing to PISAD! This document provides guidelines and workflows for contributing to the project.

## Development Setup

### Prerequisites

- Python 3.11, 3.12, or 3.13
- Node.js 20.x LTS
- Git
- uv (Python package manager)

### Initial Setup

1. **Fork and clone the repository**
```bash
git clone https://github.com/yourusername/pisad.git
cd pisad
```

2. **Install uv (ultra-fast Python package manager)**
```bash
# If not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.bashrc
```

3. **Set up Python environment with uv**
```bash
# Create virtual environment and install all dependencies
uv sync --all-extras --dev

# This single command:
# - Creates a .venv if it doesn't exist
# - Installs all dependencies from pyproject.toml
# - Installs dev dependencies
# - Syncs with uv.lock for reproducible builds
```

4. **Install pre-commit hooks**
```bash
# Install pre-commit hooks for automatic code quality checks
uv run pre-commit install

# Run hooks manually on all files
uv run pre-commit run --all-files
```

5. **Set up frontend**
```bash
cd src/frontend
npm ci
npm run build
cd ../..
```

## Development Workflow

### Package Management with uv

**Why uv?**
- 10-100x faster than pip
- Built-in virtual environment management
- Reproducible builds with uv.lock
- Automatic dependency resolution

**Common uv commands:**
```bash
# Install all dependencies
uv sync

# Add a new dependency
uv add package-name

# Add a dev dependency
uv add --dev package-name

# Update dependencies
uv lock --upgrade

# Run a command in the environment
uv run python script.py
uv run pytest

# List installed packages
uv pip list
```

### Code Quality Standards

All code must pass quality checks before merging:

1. **Type checking** (mandatory)
```bash
# Python
uv run mypy src/backend --strict

# TypeScript
cd src/frontend && npm run tsc -- --noEmit
```

2. **Linting and formatting**
```bash
# Run all checks with trunk
npx trunk check --all

# Python specific
uv run ruff check src/
uv run black src/

# Auto-fix issues
uv run ruff check src/ --fix
uv run black src/
```

3. **Testing**
```bash
# Backend tests with coverage
uv run pytest tests/backend/unit --cov=src/backend --cov-report=term-missing

# Frontend tests
cd src/frontend && npm test

# Integration tests
uv run pytest tests/backend/integration
```

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`. They ensure:
- Code is formatted (black, prettier)
- Code passes linting (ruff, eslint)
- No syntax errors
- No large files accidentally committed
- Tests pass

**Managing pre-commit:**
```bash
# Install hooks (one-time setup)
uv run pre-commit install

# Run manually
uv run pre-commit run --all-files

# Skip hooks temporarily (use sparingly)
git commit --no-verify -m "message"

# Update hooks
uv run pre-commit autoupdate
```

### Branch Strategy

1. **Main branches:**
   - `master` or `main`: Production-ready code
   - `develop`: Integration branch for features

2. **Feature branches:**
   - Branch from: `develop`
   - Naming: `feature/description` or `story-X.Y-description`
   - Example: `feature/add-beacon-filtering` or `story-4.4-cicd-pipeline`

3. **Bugfix branches:**
   - Branch from: `develop` or `master` (for hotfixes)
   - Naming: `bugfix/description` or `hotfix/description`

### Commit Messages

Follow conventional commits format:
```text
type(scope): subject

body (optional)

footer (optional)
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Test additions or fixes
- `chore`: Build process or auxiliary tool changes
- `perf`: Performance improvements

**Examples:**
```bash
git commit -m "feat(backend): add RSSI signal filtering"
git commit -m "fix(frontend): resolve dashboard memory leak"
git commit -m "docs: update deployment instructions"
```

### Pull Request Process

1. **Before creating a PR:**
```bash
# Update your branch
git checkout develop
git pull origin develop
git checkout your-branch
git rebase develop

# Run all quality checks
npx trunk check --all
uv run mypy src/backend --strict
uv run pytest tests/backend/unit --cov=src/backend
cd src/frontend && npm test
```

2. **PR checklist:**
- [ ] Code passes all quality checks
- [ ] Tests added/updated for new functionality
- [ ] Documentation updated if needed
- [ ] Coverage maintained or improved (backend ≥65%, frontend ≥50%)
- [ ] No console.log or print statements
- [ ] PR description explains changes

3. **PR review process:**
- At least one approval required
- CI pipeline must pass
- Address all review comments
- Squash commits before merging

## Testing Guidelines

### Test Coverage Requirements
- **Backend**: ≥65% coverage (67% typical due to hardware abstraction)
- **Frontend**: ≥50% coverage
- **New features**: Must include tests

### Running Tests
```bash
# Full test suite
uv run pytest

# Specific test file
uv run pytest tests/backend/unit/test_sdr_service.py

# With coverage
uv run pytest --cov=src/backend --cov-report=html

# Parallel execution
uv run pytest -n auto

# Watch mode for TDD
uv run pytest-watch
```

### Writing Tests
```python
# Use pytest fixtures for setup
@pytest.fixture
def mock_sdr():
    return Mock(spec=SDRService)

# Use clear test names
def test_rssi_processing_filters_noise():
    # Arrange
    processor = SignalProcessor()
    noisy_signal = generate_noisy_signal()

    # Act
    filtered = processor.filter(noisy_signal)

    # Assert
    assert filtered.snr > 10
```

## Performance Considerations

### Target Metrics
- MAVLink latency: <1% packet loss
- RSSI processing: <100ms response time
- Memory usage: <2GB steady state
- CPU usage: <50% during active operations

### Profiling
```bash
# Python profiling
uv run python -m cProfile -o profile.stats src/backend/main.py

# Memory profiling
uv run python -m memory_profiler src/backend/main.py

# Frontend profiling
cd src/frontend && npm run build -- --analyze
```

## Documentation

### Code Documentation
- Use docstrings for all public functions/classes
- Include type hints for all function parameters
- Add inline comments for complex logic

### API Documentation
- Update OpenAPI/Swagger specs for API changes
- Document request/response schemas
- Include example payloads

## Troubleshooting Development Issues

### uv sync fails
```bash
# Clear cache and retry
rm -rf .venv
uv cache clean
uv sync --all-extras --dev
```

### Pre-commit hooks fail
```bash
# Update hooks
uv run pre-commit autoupdate

# Run specific hook
uv run pre-commit run black --all-files
```

### Type checking errors
```bash
# Install type stubs
uv add --dev types-package-name

# Ignore specific error
# type: ignore[error-code]
```

## Getting Help

- GitHub Issues: Report bugs or request features
- Discussions: Ask questions or propose ideas
- Documentation: Check docs/ folder
- CLAUDE.md: AI assistant reference for tools and workflows

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Follow the project's technical standards

Thank you for contributing to PISAD!
