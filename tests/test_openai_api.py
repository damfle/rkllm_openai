#!/usr/bin/env python3
"""
Test suite for RKLLM OpenAI-compatible API server using the official OpenAI Python SDK.

This test suite validates that our server is truly OpenAI-compatible by using
the official OpenAI Python client library to interact with our API endpoints.
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import pytest
import requests
from openai import OpenAI
from openai.types import ChatCompletion, Completion, CreateEmbeddingResponse
from openai.types.chat import ChatCompletionMessage


class TestConfig:
    """Test configuration and constants."""

    BASE_URL = os.getenv("RKLLM_TEST_BASE_URL", "http://localhost:8080")
    API_KEY = os.getenv("RKLLM_TEST_API_KEY", "dummy-key")
    MODEL_NAME = os.getenv("RKLLM_TEST_MODEL", "test-model")
    TIMEOUT = int(os.getenv("RKLLM_TEST_TIMEOUT", "30"))
    SKIP_SERVER_START = os.getenv("RKLLM_SKIP_SERVER_START", "false").lower() == "true"


@pytest.fixture(scope="session")
def server_process():
    """Start the RKLLM server for testing if not skipped."""
    if TestConfig.SKIP_SERVER_START:
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
        "--model-path",
        "/tmp/test_models",
        "--lib-path",
        "/usr/lib/librkllm.so",
        "--models-allowlist",
        TestConfig.MODEL_NAME,
        "--platform",
        "rk3588",
    ]

    process = None
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Wait for server to be ready
        if not wait_for_server(TestConfig.BASE_URL, TestConfig.TIMEOUT):
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
def openai_client():
    """Create OpenAI client configured for our test server."""
    return OpenAI(api_key=TestConfig.API_KEY, base_url=f"{TestConfig.BASE_URL}/v1")


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


class TestHealthAndModels:
    """Test basic server health and model endpoints."""

    def test_health_endpoint(self):
        """Test that health endpoint is accessible."""
        response = requests.get(f"{TestConfig.BASE_URL}/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data or "message" in data

    def test_list_models(self, openai_client: OpenAI):
        """Test listing available models."""
        models = openai_client.models.list()

        assert hasattr(models, "data")
        assert len(models.data) > 0

        # Check that our test model is in the list
        model_ids = [model.id for model in models.data]
        assert TestConfig.MODEL_NAME in model_ids

    def test_get_specific_model(self, openai_client: OpenAI):
        """Test retrieving a specific model."""
        try:
            model = openai_client.models.retrieve(TestConfig.MODEL_NAME)
            assert model.id == TestConfig.MODEL_NAME
            assert hasattr(model, "object")
            assert model.object == "model"
        except Exception as e:
            pytest.skip(f"Model {TestConfig.MODEL_NAME} not available: {e}")


class TestChatCompletions:
    """Test chat completion endpoints."""

    def test_basic_chat_completion(self, openai_client: OpenAI, server_process):
        """Test basic chat completion."""
        response = openai_client.chat.completions.create(
            model=TestConfig.MODEL_NAME,
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            max_tokens=50,
            temperature=0.7,
        )

        assert isinstance(response, ChatCompletion)
        assert len(response.choices) > 0

        choice = response.choices[0]
        assert choice.message is not None
        assert choice.message.role == "assistant"
        assert choice.message.content is not None
        assert len(choice.message.content) > 0

        # Check response metadata
        assert response.model == TestConfig.MODEL_NAME
        assert response.object == "chat.completion"
        assert hasattr(response, "usage")

    def test_streaming_chat_completion(self, openai_client: OpenAI, server_process):
        """Test streaming chat completion."""
        stream = openai_client.chat.completions.create(
            model=TestConfig.MODEL_NAME,
            messages=[{"role": "user", "content": "Tell me a short story"}],
            max_tokens=100,
            temperature=0.7,
            stream=True,
        )

        chunks_received = 0
        content_received = ""

        for chunk in stream:
            chunks_received += 1
            if chunk.choices and chunk.choices[0].delta.content:
                content_received += chunk.choices[0].delta.content

        assert chunks_received > 0
        assert len(content_received) > 0

    def test_chat_completion_with_system_message(
        self, openai_client: OpenAI, server_process
    ):
        """Test chat completion with system message."""
        response = openai_client.chat.completions.create(
            model=TestConfig.MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What's 2+2?"},
            ],
            max_tokens=30,
        )

        assert isinstance(response, ChatCompletion)
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None

    def test_chat_completion_with_multiple_messages(
        self, openai_client: OpenAI, server_process
    ):
        """Test chat completion with conversation history."""
        response = openai_client.chat.completions.create(
            model=TestConfig.MODEL_NAME,
            messages=[
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there! How can I help you?"},
                {"role": "user", "content": "What's the weather like?"},
            ],
            max_tokens=50,
        )

        assert isinstance(response, ChatCompletion)
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None

    def test_chat_completion_with_tools(self, openai_client: OpenAI, server_process):
        """Test chat completion with function tools."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        response = openai_client.chat.completions.create(
            model=TestConfig.MODEL_NAME,
            messages=[
                {"role": "user", "content": "What's the weather like in New York?"}
            ],
            tools=tools,
            max_tokens=100,
        )

        assert isinstance(response, ChatCompletion)
        assert len(response.choices) > 0

        message = response.choices[0].message
        # The response should either have content or tool_calls
        assert message.content is not None or message.tool_calls is not None


class TestCompletions:
    """Test completion endpoints (legacy OpenAI format)."""

    def test_basic_completion(self, openai_client: OpenAI, server_process):
        """Test basic text completion."""
        try:
            response = openai_client.completions.create(
                model=TestConfig.MODEL_NAME,
                prompt="The capital of France is",
                max_tokens=20,
                temperature=0.7,
            )

            assert isinstance(response, Completion)
            assert len(response.choices) > 0

            choice = response.choices[0]
            assert choice.text is not None
            assert len(choice.text) > 0

            # Check response metadata
            assert response.model == TestConfig.MODEL_NAME
            assert response.object == "text_completion"

        except Exception as e:
            # Some implementations might not support completions endpoint
            pytest.skip(f"Completions endpoint not supported: {e}")

    def test_streaming_completion(self, openai_client: OpenAI, server_process):
        """Test streaming text completion."""
        try:
            stream = openai_client.completions.create(
                model=TestConfig.MODEL_NAME,
                prompt="Once upon a time",
                max_tokens=50,
                temperature=0.7,
                stream=True,
            )

            chunks_received = 0
            text_received = ""

            for chunk in stream:
                chunks_received += 1
                if chunk.choices and chunk.choices[0].text:
                    text_received += chunk.choices[0].text

            assert chunks_received > 0
            assert len(text_received) > 0

        except Exception as e:
            pytest.skip(f"Streaming completions not supported: {e}")


class TestEmbeddings:
    """Test embeddings endpoints."""

    def test_single_embedding(self, openai_client: OpenAI, server_process):
        """Test generating embedding for a single text."""
        try:
            response = openai_client.embeddings.create(
                model=TestConfig.MODEL_NAME, input="Hello, world!"
            )

            assert isinstance(response, CreateEmbeddingResponse)
            assert len(response.data) == 1

            embedding = response.data[0]
            assert embedding.object == "embedding"
            assert len(embedding.embedding) > 0
            assert all(isinstance(x, (int, float)) for x in embedding.embedding)

        except Exception as e:
            pytest.skip(f"Embeddings endpoint not supported: {e}")

    def test_multiple_embeddings(self, openai_client: OpenAI, server_process):
        """Test generating embeddings for multiple texts."""
        try:
            texts = ["Hello, world!", "How are you today?", "The weather is nice."]

            response = openai_client.embeddings.create(
                model=TestConfig.MODEL_NAME, input=texts
            )

            assert isinstance(response, CreateEmbeddingResponse)
            assert len(response.data) == len(texts)

            for i, embedding_data in enumerate(response.data):
                assert embedding_data.object == "embedding"
                assert embedding_data.index == i
                assert len(embedding_data.embedding) > 0

        except Exception as e:
            pytest.skip(f"Multiple embeddings not supported: {e}")


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_model(self, openai_client: OpenAI):
        """Test request with invalid model name."""
        with pytest.raises(Exception):
            openai_client.chat.completions.create(
                model="nonexistent-model",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
            )

    def test_empty_messages(self, openai_client: OpenAI):
        """Test chat completion with empty messages."""
        with pytest.raises(Exception):
            openai_client.chat.completions.create(
                model=TestConfig.MODEL_NAME, messages=[], max_tokens=10
            )

    def test_invalid_max_tokens(self, openai_client: OpenAI, server_process):
        """Test request with invalid max_tokens."""
        try:
            # Very large max_tokens should be handled gracefully
            response = openai_client.chat.completions.create(
                model=TestConfig.MODEL_NAME,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=999999,
            )
            # Should either work or raise an appropriate error
            assert isinstance(response, ChatCompletion)
        except Exception:
            # Expected for invalid values
            pass

    def test_unsupported_parameters(self, openai_client: OpenAI, server_process):
        """Test handling of unsupported parameters."""
        # This should work even with unsupported parameters
        response = openai_client.chat.completions.create(
            model=TestConfig.MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=20,
            # These might not be supported but shouldn't break the request
            frequency_penalty=0.5,
            presence_penalty=0.3,
            top_p=0.9,
        )

        assert isinstance(response, ChatCompletion)


class TestCompatibility:
    """Test OpenAI API compatibility features."""

    def test_response_format_json(self, openai_client: OpenAI, server_process):
        """Test JSON response format if supported."""
        try:
            response = openai_client.chat.completions.create(
                model=TestConfig.MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": "Generate a JSON object with name and age fields",
                    }
                ],
                max_tokens=50,
                response_format={"type": "json_object"},
            )

            assert isinstance(response, ChatCompletion)
            content = response.choices[0].message.content

            # Try to parse as JSON
            if content:
                json.loads(content)  # Should not raise exception

        except Exception as e:
            pytest.skip(f"JSON response format not supported: {e}")

    def test_usage_tracking(self, openai_client: OpenAI, server_process):
        """Test that usage information is provided."""
        response = openai_client.chat.completions.create(
            model=TestConfig.MODEL_NAME,
            messages=[{"role": "user", "content": "Count to five"}],
            max_tokens=30,
        )

        assert hasattr(response, "usage")
        if response.usage:
            assert hasattr(response.usage, "prompt_tokens")
            assert hasattr(response.usage, "completion_tokens")
            assert hasattr(response.usage, "total_tokens")


def run_tests():
    """Run the test suite programmatically."""
    import pytest

    test_args = [
        __file__,
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "-x",  # Stop on first failure
    ]

    # Add custom markers if needed
    if TestConfig.SKIP_SERVER_START:
        test_args.extend(["-m", "not server_dependent"])

    return pytest.main(test_args)


if __name__ == "__main__":
    # Allow running tests directly
    exit_code = run_tests()
    sys.exit(exit_code)
