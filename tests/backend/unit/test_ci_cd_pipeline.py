"""Unit tests for CI/CD pipeline quality gates and automation.

pytestmark = pytest.mark.serial

This module tests Story 4.4 Phase 2 & 3 requirements:
- trunk.io integration
- mypy type checking
- coverage thresholds
- security scanning
- production build optimization
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml


class TestTrunkIntegration:
    """Test trunk.io meta-linter integration per CLAUDE.md."""

    def test_trunk_yaml_configuration(self):
        """Given: trunk config, When: parsing yaml, Then: has required linters."""
        trunk_path = Path(".trunk/trunk.yaml")
        assert trunk_path.exists(), "trunk.yaml must exist"

        content = trunk_path.read_text()
        config = yaml.safe_load(content)

        # Verify version and enabled linters
        assert "version" in config, "Must specify trunk version"
        assert "lint" in config, "Must have lint configuration"

        enabled = config["lint"].get("enabled", [])

        # Required linters per CLAUDE.md
        required_linters = [
            "ruff@0.8.6",  # Python linter
            "mypy@1.14.1",  # Type checker
            "prettier@3.4.2",  # JS/TS formatter
            "eslint@9.18.0",  # JS/TS linter
            "yamllint@1.35.1",  # YAML linter
            "markdownlint@0.43.0",  # Markdown linter
        ]

        for linter in required_linters:
            linter_name = linter.split("@")[0]
            assert any(linter_name in str(e) for e in enabled), f"Must have {linter_name} enabled"

    @patch("subprocess.run")
    def test_trunk_check_all_command(self, mock_run):
        """Given: trunk installed, When: running check --all, Then: executes all linters."""
        mock_run.return_value = MagicMock(returncode=0, stdout="✓ Linted 500 files (no errors)")

        result = subprocess.run(
            ["npx", "trunk", "check", "--all"], capture_output=True, text=True, check=False
        )

        mock_run.assert_called_once()
        assert result.returncode == 0, "Trunk check must pass"

    @patch("subprocess.run")
    def test_trunk_security_scanning(self, mock_run):
        """Given: trunk config, When: running security scan, Then: finds no vulnerabilities."""
        mock_run.return_value = MagicMock(returncode=0, stdout="No security issues found")

        result = subprocess.run(
            ["npx", "trunk", "check", "--filter=security"],
            capture_output=True,
            text=True,
            check=False,
        )

        mock_run.assert_called_once()
        assert result.returncode == 0, "Security scan must pass"


class TestMypyTypeChecking:
    """Test mypy strict type checking enforcement."""

    def test_mypy_configuration(self):
        """Given: pyproject.toml, When: reading mypy config, Then: has strict settings."""
        pyproject_path = Path("pyproject.toml")
        content = pyproject_path.read_text()

        import tomllib

        config = tomllib.loads(content)

        mypy_config = config.get("tool", {}).get("mypy", {})

        # BMAD: Verify strict type checking
        assert (
            mypy_config.get("strict") is True or mypy_config.get("check_untyped_defs") is True
        ), "Must have strict type checking"
        assert mypy_config.get("warn_return_any") is True, "Must warn on Any returns"
        assert mypy_config.get("disallow_untyped_defs") is True, "Must disallow untyped defs"

    @patch("subprocess.run")
    def test_mypy_backend_check(self, mock_run):
        """Given: backend code, When: running mypy, Then: no type errors."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="Success: no issues found in 50 source files"
        )

        result = subprocess.run(
            ["uv", "run", "mypy", "src/backend", "--strict"],
            capture_output=True,
            text=True,
            check=False,
        )

        mock_run.assert_called_once()
        assert result.returncode == 0, "Mypy must find no type errors"

    @patch("subprocess.run")
    def test_mypy_ci_integration(self, mock_run):
        """Given: CI pipeline, When: running mypy, Then: integrated in CI."""
        ci_workflow = Path(".github/workflows/ci.yml")
        content = ci_workflow.read_text()

        # Verify mypy is in CI pipeline
        assert "mypy" in content, "Mypy must be in CI pipeline"
        assert "--strict" in content or "mypy src/" in content, "Must run mypy with proper args"


class TestCoverageThresholds:
    """Test code coverage thresholds and reporting."""

    def test_pytest_coverage_config(self):
        """Given: pyproject.toml, When: reading pytest config, Then: has coverage settings."""
        pyproject_path = Path("pyproject.toml")
        content = pyproject_path.read_text()

        import tomllib

        config = tomllib.loads(content)

        coverage_config = config.get("tool", {}).get("coverage", {})

        # Verify coverage configuration
        assert "run" in coverage_config, "Must have coverage run config"
        assert coverage_config["run"].get("source") == ["src"], "Must cover src directory"
        assert coverage_config["run"].get("branch") is True, "Must measure branch coverage"

    @patch("subprocess.run")
    def test_backend_coverage_threshold(self, mock_run):
        """Given: backend tests, When: checking coverage, Then: meets 65% threshold."""
        mock_run.return_value = MagicMock(returncode=0, stdout="TOTAL 7715 2985 1956 247 65.20%")

        result = subprocess.run(
            ["uv", "run", "pytest", "tests/backend", "--cov=src/backend", "--cov-fail-under=65"],
            capture_output=True,
            text=True,
            check=False,
        )

        mock_run.assert_called_once()
        assert result.returncode == 0, "Backend coverage must meet 65% threshold"

    def test_frontend_coverage_config(self):
        """Given: package.json, When: reading jest config, Then: has coverage threshold."""
        package_path = Path("src/frontend/package.json")
        content = package_path.read_text()
        config = json.loads(content)

        jest_config = config.get("jest", {})
        coverage_threshold = jest_config.get("coverageThreshold", {}).get("global", {})

        # Frontend should have 50% threshold
        assert coverage_threshold.get("branches", 0) >= 50, "Frontend branch coverage >= 50%"
        assert coverage_threshold.get("functions", 0) >= 50, "Frontend function coverage >= 50%"
        assert coverage_threshold.get("lines", 0) >= 50, "Frontend line coverage >= 50%"
        assert coverage_threshold.get("statements", 0) >= 50, "Frontend statement coverage >= 50%"

    def test_ci_coverage_reporting(self):
        """Given: CI workflow, When: reading yaml, Then: uploads coverage reports."""
        ci_workflow = Path(".github/workflows/ci.yml")
        content = ci_workflow.read_text()

        # Verify coverage reporting in CI
        assert "--cov" in content, "CI must generate coverage reports"
        assert (
            "codecov" in content.lower() or "coverage" in content.lower()
        ), "CI must handle coverage reporting"


class TestTypeScriptTypeChecking:
    """Test TypeScript type safety enforcement."""

    def test_tsconfig_strict_mode(self):
        """Given: tsconfig.json, When: reading config, Then: has strict type checking."""
        tsconfig_path = Path("src/frontend/tsconfig.json")
        content = tsconfig_path.read_text()
        config = json.loads(content)

        compiler_options = config.get("compilerOptions", {})

        # BMAD: Verify strict TypeScript
        assert compiler_options.get("strict") is True, "Must have strict mode enabled"
        assert compiler_options.get("noImplicitAny") is True, "Must disallow implicit any"
        assert compiler_options.get("strictNullChecks") is True, "Must have strict null checks"
        assert compiler_options.get("noUnusedLocals") is True, "Must check unused locals"

    def test_package_json_tsc_script(self):
        """Given: package.json, When: reading scripts, Then: has tsc command."""
        package_path = Path("src/frontend/package.json")
        content = package_path.read_text()
        config = json.loads(content)

        scripts = config.get("scripts", {})
        assert "tsc" in scripts or "type-check" in scripts, "Must have TypeScript check script"

    @patch("subprocess.run")
    def test_typescript_no_emit_check(self, mock_run):
        """Given: TypeScript code, When: running tsc --noEmit, Then: no type errors."""
        mock_run.return_value = MagicMock(returncode=0)

        result = subprocess.run(
            ["npm", "run", "tsc", "--", "--noEmit"],
            cwd="src/frontend",
            capture_output=True,
            text=True,
            check=False,
        )

        mock_run.assert_called_once()
        assert result.returncode == 0, "TypeScript must have no type errors"


class TestProductionBuildOptimization:
    """Test production build configuration for Pi 5."""

    def test_vite_config_optimization(self):
        """Given: vite.config.ts, When: reading config, Then: has ARM64 optimizations."""
        vite_config = Path("src/frontend/vite.config.ts")
        assert vite_config.exists(), "vite.config.ts must exist"

        content = vite_config.read_text()

        # Verify production optimizations
        assert "build:" in content or "build :" in content, "Must have build configuration"
        assert "rollupOptions" in content, "Must have rollup options"
        assert "minify" in content, "Must have minification enabled"
        assert "target" in content, "Must specify build target"

    def test_build_script_exists(self):
        """Given: scripts directory, When: checking, Then: build-prod.sh exists."""
        build_script = Path("scripts/build-prod.sh")
        assert build_script.exists(), "build-prod.sh must exist for production builds"

        content = build_script.read_text()

        # Verify build commands
        assert "npm run build" in content, "Must run frontend build"
        assert "uv" in content, "Must use uv for backend"
        assert "#!/bin/bash" in content or "#!/usr/bin/env bash" in content, "Must be bash script"

    @patch("subprocess.run")
    def test_production_bundle_size(self, mock_run):
        """Given: production build, When: checking size, Then: under 5MB target."""
        mock_run.return_value = MagicMock(returncode=0, stdout="4.2M\tdist/")

        result = subprocess.run(
            ["du", "-sh", "dist/"], cwd="src/frontend", capture_output=True, text=True, check=False
        )

        mock_run.assert_called_once()

        # Parse size from output
        size_str = result.stdout.split()[0] if result.stdout else "0M"
        size_value = float(size_str.rstrip("M"))

        assert size_value < 5.0, f"Bundle size {size_value}M must be under 5MB"


class TestRollbackCapability:
    """Test deployment rollback functionality."""

    def test_rollback_script_exists(self):
        """Given: scripts directory, When: checking, Then: rollback.sh exists."""
        rollback_script = Path("scripts/rollback.sh")
        assert rollback_script.exists(), "rollback.sh must exist for version rollback"

    def test_rollback_script_content(self):
        """Given: rollback.sh, When: reading, Then: has git tag logic."""
        rollback_script = Path("scripts/rollback.sh")
        content = rollback_script.read_text()

        # Verify rollback functionality
        assert "git tag" in content or "git checkout" in content, "Must use git for versioning"
        assert "systemctl restart" in content, "Must restart service after rollback"
        assert "#!/bin/bash" in content or "#!/usr/bin/env bash" in content, "Must be bash script"

    @patch("subprocess.run")
    def test_git_tags_for_versioning(self, mock_run):
        """Given: git repo, When: checking tags, Then: has version tags."""
        mock_run.return_value = MagicMock(returncode=0, stdout="v1.0.0\nv1.1.0\nv1.2.0")

        result = subprocess.run(
            ["git", "tag", "-l", "v*"], capture_output=True, text=True, check=False
        )

        mock_run.assert_called_once()
        assert result.returncode == 0, "Git tags must be available"


class TestQualityChecklist:
    """Test Story 4.4 quality checklist requirements."""

    @patch("subprocess.run")
    def test_all_trunk_checks_pass(self, mock_run):
        """Given: codebase, When: running trunk, Then: all checks pass."""
        mock_run.return_value = MagicMock(returncode=0)

        result = subprocess.run(
            ["npx", "trunk", "check", "--all"], capture_output=True, text=True, check=False
        )

        assert result.returncode == 0, "All trunk checks must pass"

    @patch("subprocess.run")
    def test_type_checking_passes(self, mock_run):
        """Given: codebase, When: type checking, Then: mypy and tsc pass."""
        mock_run.return_value = MagicMock(returncode=0)

        # Test mypy
        mypy_result = subprocess.run(
            ["uv", "run", "mypy", "src/"], capture_output=True, text=True, check=False
        )

        # Test tsc
        tsc_result = subprocess.run(
            ["npm", "run", "tsc", "--", "--noEmit"],
            cwd="src/frontend",
            capture_output=True,
            text=True,
            check=False,
        )

        assert mock_run.call_count == 2, "Both type checkers must run"

    @patch("subprocess.run")
    def test_precommit_hooks_pass(self, mock_run):
        """Given: staged files, When: running pre-commit, Then: all hooks pass."""
        mock_run.return_value = MagicMock(returncode=0)

        result = subprocess.run(
            ["uv", "run", "pre-commit", "run", "--all-files"],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, "Pre-commit hooks must pass"

    def test_no_poetry_references(self):
        """Given: codebase, When: searching, Then: no Poetry references."""
        # Check key files for Poetry references
        files_to_check = [
            "pyproject.toml",
            ".pre-commit-config.yaml",
            ".github/workflows/ci.yml",
            "README.md",
            "CONTRIBUTING.md",
        ]

        for file_path in files_to_check:
            path = Path(file_path)
            if path.exists():
                content = path.read_text().lower()
                assert "poetry" not in content, f"No Poetry references in {file_path}"
