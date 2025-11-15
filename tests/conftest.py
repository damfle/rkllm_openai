"""
Pytest configuration and shared fixtures for RKLLM OpenAI API tests.
"""

import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Generator, Optional

import pytest
import requests
from openai import OpenAI


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Get the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def temp_model_file() -> Generator[str, None, None]:
    """Create a temporary model file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".rkllm", delete=False) as temp_file:
        # Write some dummy content to make it a valid file
        temp_file.write(b"dummy model content for testing")
        temp_file_path = temp_file.name

    yield temp_file_path

    # Cleanup
    if os.path.exists(temp_file_path):
        os.unlink(temp_file_path)


@pytest.fixture
def temp_chat_template() -> Generator[str, None, None]:
    """Create a temporary chat template file for testing."""
    template_content = """
{%- if messages[0]['role'] == 'system' -%}
    {%- set system_message = messages[0]['content'] -%}
    {%- set messages = messages[1:] -%}
{%- else -%}
    {%- set system_message = "You are a helpful assistant." -%}
{%- endif -%}

<|im_start|>system
{{ system_message }}<|im_end|>
{%- for message in messages %}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{%- endfor %}
<|im_start|>assistant
""".strip()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jinja2", delete=False
    ) as temp_file:
        temp_file.write(template_content)
        temp_file_path = temp_file.name

    yield temp_file_path

    # Cleanup
    if os.path.exists(temp_file_path):
        os.unlink(temp_file_path)


@pytest.fixture
def mock_rkllm_lib() -> Generator[str, None, None]:
    """Create a mock RKLLM library file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".so", delete=False) as temp_file:
        # Write some dummy content to make it a valid file
        temp_file.write(b"dummy library content for testing")
        temp_file_path = temp_file.name

    yield temp_file_path

    # Cleanup
    if os.path.exists(temp_file_path):
        os.unlink(temp_file_path)


class MockRKLLMServer:
    """Mock RKLLM server for testing without actual model loading."""

    def __init__(
        self, model_path: str, lib_path: str, host: str = "127.0.0.1", port: int = 0
    ):
        self.model_path = model_path
        self.lib_path = lib_path
        self.host = host
        self.port = port
        self.process = None
        self.actual_port = None

    def start(self) -> str:
        """Start the mock server and return the base URL."""
        # For now, we'll create a simple mock server
        # In a real implementation, you might want to patch the RKLLM class
        # to avoid actual model loading

        # Find an available port if port is 0
        if self.port == 0:
            import socket

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                s.listen(1)
                self.actual_port = s.getsockname()[1]
        else:
            self.actual_port = self.port

        # Start the server in a separate process
        cmd = [
            sys.executable,
            "-m",
            "rkllm_openai.server",
            "--model-path",
            self.model_path,
            "--lib-path",
            self.lib_path,
            "--host",
            self.host,
            "--port",
            str(self.actual_port),
        ]

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(os.environ, PYTHONPATH=str(Path(__file__).parent.parent)),
        )

        # Wait for server to start
        base_url = f"http://{self.host}:{self.actual_port}"
        max_retries = 30
        for _ in range(max_retries):
            try:
                response = requests.get(f"{base_url}/health", timeout=1)
                if response.status_code == 200:
                    return base_url
            except requests.exceptions.RequestException:
                pass
            time.sleep(0.1)

        # If we get here, server didn't start
        if self.process:
            stdout, stderr = self.process.communicate(timeout=5)
            raise RuntimeError(
                f"Server failed to start. "
                f"Stdout: {stdout.decode()}, "
                f"Stderr: {stderr.decode()}"
            )

        raise RuntimeError("Server failed to start")

    def stop(self):
        """Stop the mock server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


@pytest.fixture
def mock_server(
    temp_model_file: str, mock_rkllm_lib: str
) -> Generator[MockRKLLMServer, None, None]:
    """Create and start a mock RKLLM server for testing."""
    server = MockRKLLMServer(temp_model_file, mock_rkllm_lib)

    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def running_server(mock_server: MockRKLLMServer) -> Generator[str, None, None]:
    """Start the mock server and return the base URL."""
    base_url = mock_server.start()
    yield base_url
    mock_server.stop()


@pytest.fixture
def openai_client(running_server: str) -> OpenAI:
    """Create an OpenAI client configured for the test server."""
    return OpenAI(
        base_url=f"{running_server}/v1",
        api_key="test-key",  # Server doesn't validate this
    )


@pytest.fixture
def model_name(temp_model_file: str) -> str:
    """Get the model name derived from the temp model file."""
    return Path(temp_model_file).stem


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Add project root to Python path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid or "server" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Mark tests that require actual model files
        if "model" in item.nodeid and "mock" not in item.nodeid:
            item.add_marker(pytest.mark.requires_model)

        # Mark tests that require a running server
        if any(keyword in item.nodeid for keyword in ["server", "api", "client"]):
            item.add_marker(pytest.mark.requires_server)


# Helper functions for tests
def wait_for_condition(
    condition_func, timeout: float = 10.0, interval: float = 0.1
) -> bool:
    """Wait for a condition to become true."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if condition_func():
            return True
        time.sleep(interval)
    return False


def is_server_healthy(base_url: str) -> bool:
    """Check if the server is healthy and responding."""
    try:
        response = requests.get(f"{base_url}/health", timeout=1)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False
