#!/usr/bin/env python3
"""
Test runner for RKLLM OpenAI API tests.

This script provides a convenient way to run tests with different configurations
and handle dependencies that might not be available.
"""

import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are available."""
    missing_deps = []

    try:
        import pytest
    except ImportError:
        missing_deps.append("pytest")

    try:
        import requests
    except ImportError:
        missing_deps.append("requests")

    try:
        import openai
    except ImportError:
        missing_deps.append("openai")

    try:
        import pydantic
    except ImportError:
        missing_deps.append("pydantic")

    # Flask is optional for some tests
    flask_available = True
    try:
        import flask
    except ImportError:
        flask_available = False

    return missing_deps, flask_available


def install_dependencies():
    """Install missing dependencies."""
    print("Installing test dependencies...")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "pytest",
            "requests",
            "openai",
            "pydantic",
            "flask",
            "flask-cors",
        ]
    )


def run_tests(test_type="all", verbose=False, install_deps=False):
    """Run tests with specified configuration."""

    if install_deps:
        install_dependencies()

    missing_deps, flask_available = check_dependencies()

    if missing_deps:
        print(f"Missing dependencies: {', '.join(missing_deps)}")
        print("Run with --install-deps to install them automatically")
        return 1

    # Build pytest command
    cmd = [sys.executable, "-m", "pytest"]

    if verbose:
        cmd.append("-v")
    else:
        cmd.extend(["-v", "--tb=short"])

    # Add test selection based on type
    if test_type == "unit":
        cmd.extend(["-m", "not integration and not requires_server"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
        if not flask_available:
            print(
                "Warning: Flask not available - some integration tests will be skipped"
            )
    elif test_type == "api":
        cmd.extend(["-m", "requires_server"])
        if not flask_available:
            print("Flask not available - API tests cannot run")
            return 1
    elif test_type == "fast":
        cmd.extend(["-m", "not slow"])
    elif test_type == "all":
        if not flask_available:
            print("Warning: Flask not available - some tests will be skipped")

    # Add test directory
    test_dir = Path(__file__).parent
    cmd.append(str(test_dir))

    print(f"Running command: {' '.join(cmd)}")

    # Set environment
    import os

    env = os.environ.copy()
    project_root = Path(__file__).parent.parent
    env["PYTHONPATH"] = str(project_root)

    # Run tests
    result = subprocess.run(cmd, env=env)
    return result.returncode


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run RKLLM OpenAI API tests")
    parser.add_argument(
        "--type",
        choices=["all", "unit", "integration", "api", "fast"],
        default="all",
        help="Type of tests to run",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install missing dependencies automatically",
    )
    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="List available tests without running them",
    )

    args = parser.parse_args()

    if args.list_tests:
        # List tests
        cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q"]
        test_dir = Path(__file__).parent
        cmd.append(str(test_dir))
        subprocess.run(cmd)
        return 0

    return run_tests(
        test_type=args.type, verbose=args.verbose, install_deps=args.install_deps
    )


if __name__ == "__main__":
    sys.exit(main())
