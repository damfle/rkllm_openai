#!/usr/bin/env python3
"""
Simple test runner for RKLLM OpenAI API tests.
"""

import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Check if pytest is available."""
    try:
        import pytest

        return True
    except ImportError:
        print("❌ pytest not found. Install with: pip install pytest")
        return False


def run_tests():
    """Run the test suite."""
    if not check_dependencies():
        return 1

    test_dir = Path(__file__).parent

    # Run pytest with basic options
    cmd = [sys.executable, "-m", "pytest", str(test_dir), "-v", "--tb=short"]

    print(f"Running tests in {test_dir}")
    print("=" * 50)

    try:
        result = subprocess.run(cmd)
        return result.returncode
    except KeyboardInterrupt:
        print("\n❌ Tests interrupted")
        return 1


if __name__ == "__main__":
    exit_code = run_tests()
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    sys.exit(exit_code)
