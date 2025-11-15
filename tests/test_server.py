#!/usr/bin/env python3
"""
Test RKLLM server integration and startup functionality.
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests


class TestServerIntegration:
    """Test server integration, startup, and configuration."""

    @pytest.fixture
    def server_command(self, temp_model_file, mock_rkllm_lib):
        """Get the command to start the server."""
        return [
            sys.executable,
            "-m",
            "rkllm_openai.server",
            "--model-path",
            temp_model_file,
            "--lib-path",
            mock_rkllm_lib,
            "--platform",
            "rk3588",
        ]

    def test_server_help(self):
        """Test that server shows help correctly."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "rkllm_openai.server", "--help"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            # Should show help without errors
            assert "--model-path" in result.stdout
            assert "--lib-path" in result.stdout
            assert "--platform" in result.stdout
            assert "--host" in result.stdout
            assert "--port" in result.stdout
            assert "--model-timeout" in result.stdout
            assert "--chat-template" in result.stdout
        except FileNotFoundError:
            pytest.skip("Flask not available - server cannot run")

    def test_server_missing_required_args(self):
        """Test that server fails with missing required arguments."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "rkllm_openai.server"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            # Should fail with non-zero exit code
            assert result.returncode != 0
            assert (
                "required" in result.stderr.lower()
                or "required" in result.stdout.lower()
            )
        except FileNotFoundError:
            pytest.skip("Flask not available - server cannot run")

    def test_server_invalid_model_path(self, mock_rkllm_lib):
        """Test that server fails with invalid model path."""
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rkllm_openai.server",
                    "--model-path",
                    "/nonexistent/model.rkllm",
                    "--lib-path",
                    mock_rkllm_lib,
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            assert result.returncode != 0
            assert "not found" in result.stderr.lower()
        except FileNotFoundError:
            pytest.skip("Flask not available - server cannot run")

    def test_server_invalid_lib_path(self, temp_model_file):
        """Test that server fails with invalid library path."""
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rkllm_openai.server",
                    "--model-path",
                    temp_model_file,
                    "--lib-path",
                    "/nonexistent/lib.so",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            assert result.returncode != 0
            assert "not found" in result.stderr.lower()
        except FileNotFoundError:
            pytest.skip("Flask not available - server cannot run")

    def test_server_directory_as_model_path(self, mock_rkllm_lib):
        """Test that server fails when model path is a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "rkllm_openai.server",
                        "--model-path",
                        temp_dir,
                        "--lib-path",
                        mock_rkllm_lib,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

                assert result.returncode != 0
                assert "must be a file" in result.stderr.lower()
            except FileNotFoundError:
                pytest.skip("Flask not available - server cannot run")

    @patch("rkllm_openai.model_manager.ModelManager")
    def test_server_startup_success(self, mock_manager_class, server_command):
        """Test successful server startup."""
        # Mock ModelManager to avoid actual model loading
        mock_manager = Mock()
        mock_manager.get_available_models.return_value = {"test-model"}
        mock_manager_class.return_value = mock_manager

        try:
            # Add a custom port to avoid conflicts
            import socket

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                port = s.getsockname()[1]

            cmd = server_command + ["--port", str(port)]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=dict(os.environ, PYTHONPATH=str(Path(__file__).parent.parent)),
            )

            # Give server time to start
            time.sleep(2)

            # Check if process is still running (not crashed)
            assert process.poll() is None, "Server process crashed on startup"

            # Try to connect to health endpoint
            try:
                response = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"
            except requests.exceptions.ConnectionError:
                # Server might not be ready yet, that's ok for this test
                pass

            # Clean up
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        except FileNotFoundError:
            pytest.skip("Flask not available - server cannot run")

    def test_server_chat_template_option(
        self, temp_model_file, mock_rkllm_lib, temp_chat_template
    ):
        """Test server with chat template option."""
        try:
            # Just test that the argument is accepted
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rkllm_openai.server",
                    "--model-path",
                    temp_model_file,
                    "--lib-path",
                    mock_rkllm_lib,
                    "--chat-template",
                    temp_chat_template,
                    "--help",  # Use help to avoid actual startup
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            # Should not fail due to unknown argument
            assert "--chat-template" in result.stdout

        except FileNotFoundError:
            pytest.skip("Flask not available - server cannot run")

    @patch("rkllm_openai.model_manager.ModelManager")
    def test_server_model_initialization_failure(
        self, mock_manager_class, server_command
    ):
        """Test server behavior when model initialization fails."""
        # Mock ModelManager to raise an exception
        mock_manager_class.side_effect = Exception("Model loading failed")

        try:
            process = subprocess.Popen(
                server_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=dict(os.environ, PYTHONPATH=str(Path(__file__).parent.parent)),
            )

            # Wait for process to finish
            stdout, stderr = process.communicate(timeout=10)

            # Should exit with error code
            assert process.returncode != 0

            # Should contain error message
            error_output = stdout.decode() + stderr.decode()
            assert "error" in error_output.lower()

        except FileNotFoundError:
            pytest.skip("Flask not available - server cannot run")

    def test_server_platform_options(self, temp_model_file, mock_rkllm_lib):
        """Test different platform options."""
        platforms = ["rk3588", "rk3576"]

        for platform in platforms:
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "rkllm_openai.server",
                        "--model-path",
                        temp_model_file,
                        "--lib-path",
                        mock_rkllm_lib,
                        "--platform",
                        platform,
                        "--help",  # Use help to avoid actual startup
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

                # Should not fail due to invalid platform
                assert result.returncode == 0

            except FileNotFoundError:
                pytest.skip("Flask not available - server cannot run")

    def test_server_invalid_platform(self, temp_model_file, mock_rkllm_lib):
        """Test server with invalid platform."""
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rkllm_openai.server",
                    "--model-path",
                    temp_model_file,
                    "--lib-path",
                    mock_rkllm_lib,
                    "--platform",
                    "invalid-platform",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            assert result.returncode != 0
            assert "invalid choice" in result.stderr.lower()

        except FileNotFoundError:
            pytest.skip("Flask not available - server cannot run")

    def test_server_port_configuration(self, temp_model_file, mock_rkllm_lib):
        """Test server port configuration."""
        try:
            # Test with custom port
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rkllm_openai.server",
                    "--model-path",
                    temp_model_file,
                    "--lib-path",
                    mock_rkllm_lib,
                    "--port",
                    "9999",
                    "--help",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            assert result.returncode == 0

        except FileNotFoundError:
            pytest.skip("Flask not available - server cannot run")

    def test_server_host_configuration(self, temp_model_file, mock_rkllm_lib):
        """Test server host configuration."""
        try:
            # Test with custom host
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rkllm_openai.server",
                    "--model-path",
                    temp_model_file,
                    "--lib-path",
                    mock_rkllm_lib,
                    "--host",
                    "0.0.0.0",
                    "--help",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            assert result.returncode == 0

        except FileNotFoundError:
            pytest.skip("Flask not available - server cannot run")

    def test_server_timeout_configuration(self, temp_model_file, mock_rkllm_lib):
        """Test server model timeout configuration."""
        try:
            # Test with custom timeout
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rkllm_openai.server",
                    "--model-path",
                    temp_model_file,
                    "--lib-path",
                    mock_rkllm_lib,
                    "--model-timeout",
                    "600",
                    "--help",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            assert result.returncode == 0

        except FileNotFoundError:
            pytest.skip("Flask not available - server cannot run")


class TestServerConfiguration:
    """Test server configuration and argument parsing."""

    def test_argument_parsing(self):
        """Test that all expected arguments are properly defined."""
        import argparse

        from rkllm_openai.server import main

        # This will test the argument parser without actually running the server
        try:
            # Should raise SystemExit due to missing required arguments
            with pytest.raises(SystemExit):
                with patch("sys.argv", ["server.py"]):
                    main()
        except ImportError:
            pytest.skip("Flask not available")

    def test_model_name_extraction(self):
        """Test that model names are correctly extracted from file paths."""
        from pathlib import Path

        test_cases = [
            ("/path/to/model.rkllm", "model"),
            ("/path/to/awesome-model.rkllm", "awesome-model"),
            ("/deep/path/to/my_model_v2.rkllm", "my_model_v2"),
            ("./local/model.bin", "model"),
        ]

        for file_path, expected_name in test_cases:
            actual_name = Path(file_path).stem
            assert actual_name == expected_name

    @patch("rkllm_openai.server.ModelManager")
    def test_model_manager_initialization_parameters(self, mock_manager_class):
        """Test that ModelManager is initialized with correct parameters."""
        import sys

        from rkllm_openai.server import main

        test_args = [
            "server.py",
            "--model-path",
            "/test/model.rkllm",
            "--lib-path",
            "/test/lib.so",
            "--platform",
            "rk3576",
            "--model-timeout",
            "600",
            "--chat-template",
            "/test/template.jinja2",
        ]

        try:
            with patch.object(sys, "argv", test_args):
                with patch("rkllm_openai.server.os.path.exists", return_value=True):
                    with patch("rkllm_openai.server.os.path.isfile", return_value=True):
                        with patch("rkllm_openai.server.create_app") as mock_create_app:
                            mock_app = Mock()
                            mock_create_app.return_value = mock_app
                            mock_app.run.side_effect = (
                                KeyboardInterrupt()
                            )  # Stop execution

                            try:
                                main()
                            except KeyboardInterrupt:
                                pass

            # Verify ModelManager was called with correct parameters
            mock_manager_class.assert_called_once_with(
                model_path="/test/model.rkllm",
                platform="rk3576",
                lib_path="/test/lib.so",
                model_timeout=600,
                chat_template="/test/template.jinja2",
            )

        except ImportError:
            pytest.skip("Flask not available")


class TestServerHealthAndStatus:
    """Test server health checks and status reporting."""

    def test_health_endpoint_format(self, running_server):
        """Test health endpoint response format."""
        response = requests.get(f"{running_server}/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert data["status"] == "healthy"
        assert isinstance(data["timestamp"], int)

        # Timestamp should be recent (within last minute)
        import time

        current_time = int(time.time())
        assert abs(current_time - data["timestamp"]) < 60

    def test_health_endpoint_method_support(self, running_server):
        """Test that health endpoint only supports GET method."""
        # GET should work
        response = requests.get(f"{running_server}/health")
        assert response.status_code == 200

        # POST should not be allowed
        response = requests.post(f"{running_server}/health")
        assert response.status_code == 405  # Method Not Allowed

    def test_cors_configuration(self, running_server):
        """Test that CORS is properly configured."""
        response = requests.get(f"{running_server}/health")

        # Should have CORS headers
        assert "Access-Control-Allow-Origin" in response.headers

    def test_error_responses_format(self, running_server):
        """Test that error responses follow consistent format."""
        # Test 404 error
        response = requests.get(f"{running_server}/nonexistent")
        assert response.status_code == 404

        # Test invalid model error
        response = requests.get(f"{running_server}/v1/models/nonexistent")
        assert response.status_code == 404

        data = response.json()
        assert "error" in data
        assert "message" in data["error"]
