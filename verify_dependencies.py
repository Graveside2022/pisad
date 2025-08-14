#!/usr/bin/env python3
"""
Comprehensive dependency verification for PISAD project.
Checks all installed packages, versions, and potential conflicts.
"""

import importlib.metadata
import subprocess
import sys

import pkg_resources


def run_command(cmd: str) -> tuple[bool, str]:
    """Run a shell command and return success status and output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)


def check_python_package(name: str) -> dict:
    """Check if a Python package is installed and get its version."""
    try:
        version = importlib.metadata.version(name)
        return {"installed": True, "version": version, "name": name}
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False, "version": None, "name": name}


def check_system_package(name: str) -> dict:
    """Check if a system package is installed."""
    success, output = run_command(f"dpkg -l | grep -E '^ii.*{name}'")
    if success and output:
        lines = output.strip().split("\n")
        if lines:
            parts = lines[0].split()
            if len(parts) >= 3:
                return {"installed": True, "version": parts[2], "name": name}
    return {"installed": False, "version": None, "name": name}


def check_hackrf() -> dict:
    """Check HackRF installation."""
    success, output = run_command("hackrf_info --version")
    if success:
        lines = output.strip().split("\n")
        version = "unknown"
        for line in lines:
            if "version:" in line:
                version = line.split("version:")[-1].strip()
                break
        return {"installed": True, "version": version, "name": "HackRF"}
    return {"installed": False, "version": None, "name": "HackRF"}


def check_dependency_conflicts():
    """Check for known dependency conflicts."""
    conflicts = []

    # Check for conflicting numpy versions
    try:
        dist = pkg_resources.get_distribution("numpy")
        numpy_deps = []
        for req in pkg_resources.working_set:
            if "numpy" in str(req.requires()):
                numpy_deps.append(f"{req.project_name}: {req.requires()}")
        if numpy_deps:
            conflicts.append(
                {
                    "type": "info",
                    "message": f"Packages depending on numpy: {', '.join(numpy_deps[:5])}",
                }
            )
    except:
        pass

    # Check for Python version compatibility
    python_version = sys.version_info
    if python_version >= (3, 13):
        # DroneKit has issues with Python 3.13
        if check_python_package("dronekit")["installed"]:
            conflicts.append(
                {
                    "type": "warning",
                    "message": "DroneKit may have compatibility issues with Python 3.13+ (collections.MutableMapping deprecated)",
                }
            )

    return conflicts


def main():
    print("=" * 80)
    print("PISAD Dependency Verification Report")
    print("=" * 80)
    print()

    # Python version
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print()

    # Define dependencies to check
    python_packages = [
        # MAVLink/Drone
        ("mavproxy", "1.8.74"),
        ("dronekit", "2.9.2"),
        ("pymavlink", "2.4.49"),
        ("pyserial", "3.5"),
        ("future", "1.0.0"),
        # SDR
        ("pyrtlsdr", "0.3.0"),
        # Computer Vision
        ("opencv-python-headless", "4.12.0"),
        # Scientific Computing
        ("numpy", "2.2.1"),
        ("scipy", "1.15.1"),
        ("matplotlib", "3.10.5"),
        # Web Framework
        ("fastapi", "0.116.1"),
        ("uvicorn", "0.35.0"),
        ("websockets", "15.0.1"),
        ("pydantic", "2.10.5"),
        # Database
        ("sqlalchemy", "2.0.43"),
        ("aiosqlite", "0.20.0"),
        # Testing
        ("pytest", "8.4.1"),
        ("pytest-asyncio", "1.1.0"),
        ("pytest-cov", "6.0.0"),
        ("hypothesis", "6.138.0"),
        # Development
        ("black", "24.10.0"),
        ("ruff", None),
        ("mypy", None),
        ("ipdb", None),
    ]

    system_packages = [
        "python3-wxgtk4.0",
        "python3-opencv",
        "python3-matplotlib",
        "cmake",
        "g++",
        "libusb-1.0-0-dev",
        "libfftw3-dev",
        "pkg-config",
    ]

    # Check Python packages
    print("### Python Packages ###")
    print("-" * 40)
    missing_packages = []
    version_mismatches = []

    for package_name, expected_version in python_packages:
        result = check_python_package(package_name)
        status_symbol = "✓" if result["installed"] else "✗"

        if result["installed"]:
            version_str = result["version"]
            if expected_version and result["version"] != expected_version:
                version_str += f" (expected: {expected_version})"
                version_mismatches.append((package_name, result["version"], expected_version))
            print(f"{status_symbol} {package_name:30} {version_str}")
        else:
            print(f"{status_symbol} {package_name:30} NOT INSTALLED")
            missing_packages.append(package_name)

    print()

    # Check system packages
    print("### System Packages ###")
    print("-" * 40)
    missing_system = []

    for package_name in system_packages:
        result = check_system_package(package_name)
        status_symbol = "✓" if result["installed"] else "✗"

        if result["installed"]:
            print(f"{status_symbol} {package_name:30} {result['version']}")
        else:
            print(f"{status_symbol} {package_name:30} NOT INSTALLED")
            missing_system.append(package_name)

    print()

    # Check HackRF
    print("### SDR Hardware ###")
    print("-" * 40)
    hackrf = check_hackrf()
    status_symbol = "✓" if hackrf["installed"] else "✗"
    if hackrf["installed"]:
        print(f"{status_symbol} {hackrf['name']:30} {hackrf['version']}")
    else:
        print(f"{status_symbol} {hackrf['name']:30} NOT INSTALLED")

    print()

    # Check for conflicts
    print("### Dependency Analysis ###")
    print("-" * 40)

    conflicts = check_dependency_conflicts()

    if conflicts:
        for conflict in conflicts:
            if conflict["type"] == "warning":
                print(f"⚠️  WARNING: {conflict['message']}")
            else:
                print(f"ℹ️  INFO: {conflict['message']}")
    else:
        print("✓ No known conflicts detected")

    print()

    # Run uv pip check
    print("### UV Package Compatibility Check ###")
    print("-" * 40)
    success, output = run_command("uv pip check")
    if success:
        if "All installed packages are compatible" in output:
            print("✓ All installed packages are compatible")
        else:
            print(output)
    else:
        print("✗ Could not run uv pip check")

    print()

    # Summary
    print("### Summary ###")
    print("-" * 40)

    total_issues = len(missing_packages) + len(missing_system) + len(version_mismatches)

    if total_issues == 0:
        print("✅ All dependencies are properly installed and compatible!")
    else:
        if missing_packages:
            print(f"⚠️  Missing Python packages: {', '.join(missing_packages)}")
        if missing_system:
            print(f"⚠️  Missing system packages: {', '.join(missing_system)}")
        if version_mismatches:
            print("⚠️  Version mismatches:")
            for name, actual, expected in version_mismatches:
                print(f"    - {name}: {actual} (expected {expected})")

    if conflicts:
        warnings = [c for c in conflicts if c["type"] == "warning"]
        if warnings:
            print(f"\n⚠️  {len(warnings)} compatibility warning(s) found")

    print()
    print("=" * 80)

    return total_issues == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
