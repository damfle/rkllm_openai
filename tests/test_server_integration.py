#!/usr/bin/env python3
"""
Integration tests for RKLLM OpenAI server endpoints.

Tests the actual HTTP endpoints using requests library.
"""

import json
import time
from typing import Any, Dict

import pytest
import requests


class TestServerIntegration:
    """Integration tests for server endpoints."""

    @pytest.fixture(autouse=True)
    def setup(self, test_config):
        """Setup test configuration."""
        self.base_url = test_config.BASE_URL
        self.api_key = test_config.API_KEY
        self.model_name = test_config.MODEL_NAME
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def test_health_endpoint(self):
        """Test health endpoint."""
        response = requests.get(f"{self.base_url}/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data or "message" in data

    def test_list_models_endpoint(self):
        """Test models listing endpoint."""
        response = requests.get(f"{self.base_url}/v1/models", headers=self.headers)
        assert response.status_code == 200

        data = response.json()
        assert "data" in data
        assert isinstance(data["data"], list)
        assert data["object"] == "list"

    def test_get_model_endpoint(self):
        """Test get specific model endpoint."""
        response = requests.get(
            f"{self.base_url}/v1/models/{self.model_name}", headers=self.headers
        )

        if response.status_code == 200:
            data = response.json()
            assert data["id"] == self.model_name
            assert data["object"] == "model"
        elif response.status_code == 404:
            # Model not available, which is acceptable for testing
            pytest.skip(f"Model {self.model_name} not available")

    def test_chat_completion_basic(self):
        """Test basic chat completion."""
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
        }

        response = requests.post(
            f"{self.base_url}/v1/chat/completions", headers=self.headers, json=payload
        )

        if response.status_code == 400 and "not found" in response.text:
            pytest.skip(f"Model {self.model_name} not available")

        assert response.status_code == 200
        data = response.json()

        assert "choices" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "content" in data["choices"][0]["message"]

    def test_chat_completion_streaming(self):
        """Test streaming chat completion."""
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": "Count to 3"}],
            "max_tokens": 20,
            "stream": True,
        }

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=self.headers,
            json=payload,
            stream=True,
        )

        if response.status_code == 400 and "not found" in response.text:
            pytest.skip(f"Model {self.model_name} not available")

        assert response.status_code == 200

        chunks_received = 0
        for line in response.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_part = line_str[6:]
                    if data_part == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(data_part)
                        chunks_received += 1
                        assert "choices" in chunk_data
                    except json.JSONDecodeError:
                        pass

        assert chunks_received > 0

    def test_completion_basic(self):
        """Test basic text completion."""
        payload = {
            "model": self.model_name,
            "prompt": "The capital of France is",
            "max_tokens": 5,
        }

        response = requests.post(
            f"{self.base_url}/v1/completions", headers=self.headers, json=payload
        )

        if response.status_code == 400 and "not found" in response.text:
            pytest.skip(f"Model {self.model_name} not available")

        if response.status_code == 404:
            pytest.skip("Completions endpoint not implemented")

        assert response.status_code == 200
        data = response.json()

        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "text" in data["choices"][0]

    def test_embeddings_basic(self):
        """Test basic embeddings."""
        payload = {"model": self.model_name, "input": "Hello world"}

        response = requests.post(
            f"{self.base_url}/v1/embeddings", headers=self.headers, json=payload
        )

        if response.status_code == 400 and "not found" in response.text:
            pytest.skip(f"Model {self.model_name} not available")

        if response.status_code == 404:
            pytest.skip("Embeddings endpoint not implemented")

        assert response.status_code == 200
        data = response.json()

        assert "data" in data
        assert len(data["data"]) > 0
        assert "embedding" in data["data"][0]
        assert isinstance(data["data"][0]["embedding"], list)

    def test_error_handling_invalid_model(self):
        """Test error handling with invalid model."""
        payload = {
            "model": "nonexistent-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
        }

        response = requests.post(
            f"{self.base_url}/v1/chat/completions", headers=self.headers, json=payload
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_error_handling_empty_messages(self):
        """Test error handling with empty messages."""
        payload = {"model": self.model_name, "messages": [], "max_tokens": 10}

        response = requests.post(
            f"{self.base_url}/v1/chat/completions", headers=self.headers, json=payload
        )

        # Should return an error for empty messages
        assert response.status_code == 400

    def test_model_switching(self):
        """Test switching between models if multiple are available."""
        # Get list of available models
        response = requests.get(f"{self.base_url}/v1/models", headers=self.headers)
        assert response.status_code == 200

        models = response.json()["data"]
        if len(models) < 2:
            pytest.skip("Need at least 2 models to test switching")

        # Test requests to different models
        for model in models[:2]:
            payload = {
                "model": model["id"],
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
            }

            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json=payload,
            )

            if response.status_code == 200:
                data = response.json()
                assert data["model"] == model["id"]

    def test_concurrent_requests(self):
        """Test handling concurrent requests."""
        import queue
        import threading

        results = queue.Queue()

        def make_request():
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
            }

            try:
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=30,
                )
                results.put(response.status_code)
            except Exception as e:
                results.put(f"Error: {e}")

        # Start multiple concurrent requests
        threads = []
        for _ in range(3):
            t = threading.Thread(target=make_request)
            threads.append(t)
            t.start()

        # Wait for all to complete
        for t in threads:
            t.join(timeout=60)

        # Check results
        success_count = 0
        while not results.empty():
            result = results.get()
            if result == 200:
                success_count += 1
            elif isinstance(result, str) and "not found" in result:
                pytest.skip(f"Model {self.model_name} not available")

        # At least one request should succeed
        assert success_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
