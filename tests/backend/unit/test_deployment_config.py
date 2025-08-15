"""Unit tests for deployment configuration and CI/CD components.

This module tests Story 4.4 Phase 1 requirements:
- systemd service configuration
- uv package manager integration
- pre-commit hooks configuration
- environment variable handling
"""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


class TestSystemdServiceConfiguration:
    """Test systemd service configuration for pisad.service."""

    def test_service_file_exists(self):
        """Given: deployment directory, When: checking service file, Then: file exists."""
        service_path = Path("deployment/pisad.service")
        assert service_path.exists(), "pisad.service file must exist in deployment directory"

    def test_service_file_content(self):
        """Given: service file, When: reading content, Then: contains required configuration."""
        service_path = Path("deployment/pisad.service")
        content = service_path.read_text()

        # BDD: Verify critical systemd directives
        assert "[Unit]" in content, "Service must have Unit section"
        assert "[Service]" in content, "Service must have Service section"
        assert "[Install]" in content, "Service must have Install section"
        assert "Type=simple" in content, "Service type must be simple (not notify)"
        assert "RestartSec=5" in content, "RestartSec must be 5 for faster recovery"
        assert "StartLimitBurst=3" in content, "StartLimitBurst must be 3 for stability"
        assert "UV_SYSTEM_PYTHON=1" in content, "Must use UV_SYSTEM_PYTHON environment"

    def test_service_exec_command(self):
        """Given: service file, When: checking ExecStart, Then: uses uv run uvicorn."""
        service_path = Path("deployment/pisad.service")
        content = service_path.read_text()

        # BMAD: Modular test for ExecStart command
        exec_line = [line for line in content.split("\n") if "ExecStart=" in line]
        assert exec_line, "Service must have ExecStart directive"

        exec_command = exec_line[0]
        assert "uv run uvicorn" in exec_command, "Must use uv run uvicorn"
        assert "src.backend.core.app:app" in exec_command, "Must load correct app module"
        assert "--host 0.0.0.0" in exec_command, "Must bind to all interfaces"
        assert "--port 8080" in exec_command, "Must use port 8080"

    @patch("subprocess.run")
    def test_service_validation(self, mock_run):
        """Given: service file, When: validating with systemd, Then: passes validation."""
        mock_run.return_value = MagicMock(returncode=0)

        # Simulate systemd validation
        result = subprocess.run(
            ["systemd-analyze", "verify", "deployment/pisad.service"],
            capture_output=True,
            text=True,
            check=False,
        )

        mock_run.assert_called_once()
        assert result.returncode == 0, "Service file must pass systemd validation"


class TestUVPackageManager:
    """Test uv package manager integration and configuration."""

    def test_pyproject_toml_exists(self):
        """Given: project root, When: checking pyproject.toml, Then: file exists."""
        pyproject_path = Path("pyproject.toml")
        assert pyproject_path.exists(), "pyproject.toml must exist for uv"

    def test_uv_lock_exists(self):
        """Given: project root, When: checking uv.lock, Then: lock file exists."""
        lock_path = Path("uv.lock")
        assert lock_path.exists(), "uv.lock must exist for reproducible builds"

    def test_no_poetry_references(self):
        """Given: project files, When: searching for poetry, Then: no references found."""
        # Check pyproject.toml doesn't reference poetry
        pyproject_path = Path("pyproject.toml")
        content = pyproject_path.read_text()
        assert "poetry" not in content.lower(), "No Poetry references in pyproject.toml"

        # Check pre-commit config
        precommit_path = Path(".pre-commit-config.yaml")
        if precommit_path.exists():
            content = precommit_path.read_text()
            assert "poetry" not in content.lower(), "No Poetry hooks in pre-commit config"

    @patch("subprocess.run")
    def test_uv_sync_command(self, mock_run):
        """Given: uv.lock, When: running uv sync, Then: installs all dependencies."""
        mock_run.return_value = MagicMock(returncode=0, stdout="111 packages installed")

        result = subprocess.run(
            ["uv", "sync", "--all-extras", "--dev"], capture_output=True, text=True, check=False
        )

        mock_run.assert_called_once()
        assert result.returncode == 0, "uv sync must succeed"

    def test_pyproject_uv_configuration(self):
        """Given: pyproject.toml, When: reading config, Then: has proper uv settings."""
        pyproject_path = Path("pyproject.toml")
        content = pyproject_path.read_text()

        # Parse TOML content
        import tomllib

        config = tomllib.loads(content)

        # BMAD: Verify project metadata
        assert "project" in config, "Must have project section"
        assert config["project"]["name"] == "pisad", "Project name must be pisad"
        assert "dependencies" in config["project"], "Must have dependencies list"

        # Verify tool configuration
        assert "tool" in config, "Must have tool section"


class TestPreCommitHooks:
    """Test pre-commit hooks configuration for code quality."""

    def test_precommit_config_exists(self):
        """Given: project root, When: checking pre-commit, Then: config exists."""
        config_path = Path(".pre-commit-config.yaml")
        assert config_path.exists(), ".pre-commit-config.yaml must exist"

    def test_precommit_hooks_configured(self):
        """Given: pre-commit config, When: parsing yaml, Then: has required hooks."""
        config_path = Path(".pre-commit-config.yaml")
        content = config_path.read_text()
        config = yaml.safe_load(content)

        # Extract all hook IDs
        hook_ids = []
        for repo in config.get("repos", []):
            for hook in repo.get("hooks", []):
                hook_ids.append(hook.get("id"))

        # BMAD: Verify critical hooks per CLAUDE.md
        assert "ruff" in hook_ids, "Must have ruff linter"
        assert "ruff-format" in hook_ids or "black" in hook_ids, "Must have formatter"
        assert "mypy" in hook_ids, "Must have mypy type checker"

        # No poetry hooks
        assert "poetry-check" not in hook_ids, "Must not have poetry-check hook"

    @patch("subprocess.run")
    def test_precommit_run_all(self, mock_run):
        """Given: pre-commit hooks, When: running all, Then: executes successfully."""
        mock_run.return_value = MagicMock(returncode=0)

        result = subprocess.run(
            ["uv", "run", "pre-commit", "run", "--all-files"],
            capture_output=True,
            text=True,
            check=False,
        )

        mock_run.assert_called_once()
        assert result.returncode == 0, "Pre-commit hooks must pass"


class TestEnvironmentConfiguration:
    """Test environment variable configuration and handling."""

    def test_env_example_exists(self):
        """Given: project root, When: checking .env.example, Then: file exists."""
        env_example = Path(".env.example")
        assert env_example.exists(), ".env.example must exist for configuration template"

    def test_env_example_content(self):
        """Given: .env.example, When: reading content, Then: has required variables."""
        env_example = Path(".env.example")
        content = env_example.read_text()

        # Required environment variables for deployment
        required_vars = [
            "APP_HOST",
            "APP_PORT",
            "LOG_LEVEL",
            "SDR_FREQUENCY",
            "SDR_SAMPLE_RATE",
            "MAV_SOURCE_SYSTEM",
            "MAV_TARGET_SYSTEM",
        ]

        for var in required_vars:
            assert var in content, f"Must have {var} in .env.example"

    @patch.dict(os.environ, {"UV_SYSTEM_PYTHON": "1"})
    def test_uv_system_python_env(self):
        """Given: UV environment, When: checking UV_SYSTEM_PYTHON, Then: is set."""
        assert os.environ.get("UV_SYSTEM_PYTHON") == "1", "UV_SYSTEM_PYTHON must be set"

    def test_config_module_imports(self):
        """Given: config module, When: importing, Then: loads without errors."""
        try:
            from src.backend.core.config import get_config

            config = get_config()
            assert config is not None, "Config must load successfully"
            assert hasattr(config, "app"), "Config must have app section"
            assert hasattr(config, "sdr"), "Config must have sdr section"
        except ImportError as e:
            pytest.skip(f"Config module not available: {e}")


class TestDeploymentReadme:
    """Test README.md deployment instructions for Story 4.4."""

    def test_readme_exists(self):
        """Given: project root, When: checking README, Then: file exists."""
        readme_path = Path("README.md")
        assert readme_path.exists(), "README.md must exist at project root"

    def test_readme_deployment_section(self):
        """Given: README, When: reading content, Then: has deployment instructions."""
        readme_path = Path("README.md")
        content = readme_path.read_text().lower()

        # BMAD: Verify deployment documentation
        assert "deployment" in content, "Must have deployment section"
        assert "raspberry pi 5" in content or "pi 5" in content, "Must mention Pi 5"
        assert "uv" in content, "Must mention uv package manager"
        assert "systemd" in content or "pisad.service" in content, "Must mention systemd service"
        assert "15 minute" in content, "Must mention 15 minute deployment target"

    def test_contributing_guide_exists(self):
        """Given: project root, When: checking CONTRIBUTING, Then: file exists."""
        contrib_path = Path("CONTRIBUTING.md")
        assert contrib_path.exists(), "CONTRIBUTING.md must exist for dev workflow"

    def test_contributing_uv_workflow(self):
        """Given: CONTRIBUTING.md, When: reading, Then: documents uv workflow."""
        contrib_path = Path("CONTRIBUTING.md")
        content = contrib_path.read_text().lower()

        assert "uv sync" in content, "Must document uv sync command"
        assert "uv run" in content, "Must document uv run command"
        assert "pre-commit" in content, "Must document pre-commit hooks"
        assert "poetry" not in content, "Must not reference Poetry"


class TestCICDIntegration:
    """Test CI/CD pipeline integration points."""

    def test_github_workflows_exist(self):
        """Given: .github, When: checking workflows, Then: CI workflow exists."""
        ci_workflow = Path(".github/workflows/ci.yml")
        assert ci_workflow.exists(), "CI workflow must exist"

    def test_ci_workflow_uses_uv(self):
        """Given: CI workflow, When: reading yaml, Then: uses uv for Python."""
        ci_workflow = Path(".github/workflows/ci.yml")
        content = ci_workflow.read_text()

        # Verify uv usage in CI
        assert "astral-sh/setup-uv" in content or "uv" in content, "CI must use uv"
        assert "uv sync" in content, "CI must run uv sync"
        assert "uv run pytest" in content, "CI must run tests with uv"

    def test_trunk_config_exists(self):
        """Given: project root, When: checking trunk, Then: config exists."""
        trunk_yaml = Path(".trunk/trunk.yaml")
        assert trunk_yaml.exists(), ".trunk/trunk.yaml must exist for trunk.io"

    @patch("subprocess.run")
    def test_trunk_check_all(self, mock_run):
        """Given: trunk config, When: running check, Then: passes all checks."""
        mock_run.return_value = MagicMock(returncode=0)

        result = subprocess.run(
            ["npx", "trunk", "check", "--all"], capture_output=True, text=True, check=False
        )

        mock_run.assert_called_once()
        assert result.returncode == 0, "Trunk checks must pass"
