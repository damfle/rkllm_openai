"""
Pytest configuration and shared fixtures for RKLLM OpenAI API tests.
"""

import os
import subprocess
import sys
import time
from typing import Generator, Optional

import pytest
import requests


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "server_dependent: marks tests that require a running server"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


@pytest.fixture(scope="session")
def test_config():
    """Test configuration object."""

    class TestConfig:
        BASE_URL = os.getenv("RKLLM_TEST_BASE_URL", "http://localhost:8080")
        API_KEY = os.getenv("RKLLM_TEST_API_KEY", "dummy-key")
        MODEL_NAME = os.getenv("RKLLM_TEST_MODEL", "test-model")
        TIMEOUT = int(os.getenv("RKLLM_TEST_TIMEOUT", "30"))
        SKIP_SERVER_START = (
            os.getenv("RKLLM_SKIP_SERVER_START", "false").lower() == "true"
        )

    return TestConfig


def wait_for_server(base_url: str, timeout: int = 30) -> bool:
    """Wait for the server to be ready."""
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)

    return False


@pytest.fixture(scope="session")
def server_process(test_config) -> Generator[Optional[subprocess.Popen], None, None]:
    """Start the RKLLM server for testing if not skipped."""
    if test_config.SKIP_SERVER_START:
        # Assume server is already running
        yield None
        return

    # Start the server
    cmd = [
        sys.executable,
        "-m",
        "rkllm_openai.server",
        "--host",
        "localhost",
        "--port",
        "8080",
        "--model",
        test_config.MODEL_NAME,
    ]

    process = None
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Wait for server to be ready
        if not wait_for_server(test_config.BASE_URL, test_config.TIMEOUT):
            if process.poll() is None:
                process.terminate()
                process.wait()
            raise RuntimeError("Server failed to start within timeout")

        yield process

    finally:
        if process and process.poll() is None:
            process.terminate()
            process.wait()


@pytest.fixture
def openai_client(test_config):
    """Create OpenAI client configured for our test server."""
    try:
        from openai import OpenAI

        return OpenAI(
            api_key=test_config.API_KEY, base_url=f"{test_config.BASE_URL}/v1"
        )
    except ImportError:
        pytest.skip("OpenAI SDK not installed. Run: pip install openai")
