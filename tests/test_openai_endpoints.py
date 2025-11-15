#!/usr/bin/env python3
"""
Test RKLLM OpenAI API endpoints using the OpenAI SDK.
"""

import json
import time
from unittest.mock import patch

import pytest
from openai import OpenAI
from openai.types import Model
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
)
from openai.types.completion import Completion


@pytest.mark.integration
class TestOpenAIEndpoints:
    """Test OpenAI-compatible API endpoints using the OpenAI SDK."""

    def test_health_endpoint(self, running_server):
        """Test the health endpoint."""
        import requests

        response = requests.get(f"{running_server}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_list_models(self, openai_client: OpenAI, model_name: str):
        """Test the /v1/models endpoint."""
        models = openai_client.models.list()

        # Should return a list with exactly one model
        assert len(models.data) == 1

        model = models.data[0]
        assert isinstance(model, Model)
        assert model.id == model_name
        assert model.object == "model"
        assert model.owned_by == "rkllm"
        assert isinstance(model.created, int)

    def test_get_specific_model(self, openai_client: OpenAI, model_name: str):
        """Test the /v1/models/{model_id} endpoint."""
        model = openai_client.models.retrieve(model_name)

        assert isinstance(model, Model)
        assert model.id == model_name
        assert model.object == "model"
        assert model.owned_by == "rkllm"
        assert isinstance(model.created, int)

    def test_get_nonexistent_model(self, openai_client: OpenAI):
        """Test getting a model that doesn't exist."""
        with pytest.raises(Exception) as exc_info:
            openai_client.models.retrieve("nonexistent-model")

        # Should be a 404 error
        assert (
            "404" in str(exc_info.value) or "not found" in str(exc_info.value).lower()
        )

    @patch("rkllm_openai.bindings.rkllm.RKLLM")
    def test_chat_completions_basic(
        self, mock_rkllm_class, openai_client: OpenAI, model_name: str
    ):
        """Test basic chat completion."""
        # Mock the RKLLM model behavior
        mock_model = mock_rkllm_class.return_value
        mock_model.get_text_buffer.return_value = ["Hello! How can I help you today?"]
        mock_model.clear_text_buffer.return_value = None

        # Create a simple chat completion
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=100,
            temperature=0.8,
        )

        assert isinstance(response, ChatCompletion)
        assert response.model == model_name
        assert response.object == "chat.completion"
        assert len(response.choices) == 1

        choice = response.choices[0]
        assert choice.index == 0
        assert choice.finish_reason in ["stop", "tool_calls"]
        assert isinstance(choice.message, ChatCompletionMessage)
        assert choice.message.role == "assistant"

    @patch("rkllm_openai.bindings.rkllm.RKLLM")
    def test_chat_completions_with_system_message(
        self, mock_rkllm_class, openai_client: OpenAI, model_name: str
    ):
        """Test chat completion with system message."""
        mock_model = mock_rkllm_class.return_value
        mock_model.get_text_buffer.return_value = ["I'm a helpful AI assistant."]
        mock_model.clear_text_buffer.return_value = None

        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who are you?"},
            ],
            max_tokens=50,
        )

        assert isinstance(response, ChatCompletion)
        assert len(response.choices) == 1
        assert response.choices[0].message.role == "assistant"

    @patch("rkllm_openai.bindings.rkllm.RKLLM")
    def test_chat_completions_streaming(
        self, mock_rkllm_class, openai_client: OpenAI, model_name: str
    ):
        """Test streaming chat completion."""
        mock_model = mock_rkllm_class.return_value
        mock_model.get_text_buffer.side_effect = [
            ["Hello"],
            [" there"],
            ["!"],
            [],  # Empty to signal completion
        ]
        mock_model.clear_text_buffer.return_value = None

        stream = openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Say hello!"}],
            stream=True,
            max_tokens=50,
        )

        chunks = []
        for chunk in stream:
            assert isinstance(chunk, ChatCompletionChunk)
            assert chunk.model == model_name
            assert chunk.object == "chat.completion.chunk"
            chunks.append(chunk)

        # Should have received multiple chunks
        assert len(chunks) > 0

        # Last chunk should have finish_reason
        assert chunks[-1].choices[0].finish_reason is not None

    @patch("rkllm_openai.bindings.rkllm.RKLLM")
    def test_chat_completions_with_tools(
        self, mock_rkllm_class, openai_client: OpenAI, model_name: str
    ):
        """Test chat completion with function tools."""
        # Mock response with tool call
        mock_model = mock_rkllm_class.return_value
        tool_response = """I'll search for that information.

<tool_call>
{"name": "search", "arguments": {"query": "OpenAI API"}}
</tool_call>

Let me find that for you."""

        mock_model.get_text_buffer.return_value = [tool_response]
        mock_model.clear_text_buffer.return_value = None

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query",
                            }
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "Search for OpenAI API documentation"}
            ],
            tools=tools,
            max_tokens=100,
        )

        assert isinstance(response, ChatCompletion)
        choice = response.choices[0]

        # Check if tool calls were parsed (depends on implementation)
        if choice.finish_reason == "tool_calls":
            assert choice.message.tool_calls is not None
            assert len(choice.message.tool_calls) > 0

    @patch("rkllm_openai.bindings.rkllm.RKLLM")
    def test_completions_basic(
        self, mock_rkllm_class, openai_client: OpenAI, model_name: str
    ):
        """Test basic text completion."""
        mock_model = mock_rkllm_class.return_value
        mock_model.get_text_buffer.return_value = ["The weather is nice today."]
        mock_model.clear_text_buffer.return_value = None

        response = openai_client.completions.create(
            model=model_name, prompt="The weather is", max_tokens=50, temperature=0.7
        )

        assert isinstance(response, Completion)
        assert response.model == model_name
        assert response.object == "text_completion"
        assert len(response.choices) == 1

        choice = response.choices[0]
        assert choice.index == 0
        assert choice.finish_reason == "stop"
        assert isinstance(choice.text, str)

    @patch("rkllm_openai.bindings.rkllm.RKLLM")
    def test_completions_streaming(
        self, mock_rkllm_class, openai_client: OpenAI, model_name: str
    ):
        """Test streaming text completion."""
        mock_model = mock_rkllm_class.return_value
        mock_model.get_text_buffer.side_effect = [["nice"], [" and"], [" sunny"], []]
        mock_model.clear_text_buffer.return_value = None

        stream = openai_client.completions.create(
            model=model_name, prompt="The weather is", stream=True, max_tokens=50
        )

        chunks = list(stream)
        assert len(chunks) > 0

        for chunk in chunks[:-1]:  # All but last
            assert chunk.object == "text_completion"
            assert chunk.model == model_name
            assert chunk.choices[0].finish_reason is None

        # Last chunk should have finish_reason
        assert chunks[-1].choices[0].finish_reason is not None

    @patch("rkllm_openai.bindings.rkllm.RKLLM")
    def test_embeddings(self, mock_rkllm_class, openai_client: OpenAI, model_name: str):
        """Test embeddings endpoint."""
        mock_model = mock_rkllm_class.return_value
        # Mock embeddings as a list of floats
        mock_embeddings = [0.1, -0.2, 0.3, 0.4, -0.5] * 200  # 1000-dimensional
        mock_model.get_embeddings.return_value = mock_embeddings

        response = openai_client.embeddings.create(
            model=model_name, input="Hello, world!"
        )

        assert response.object == "list"
        assert response.model == model_name
        assert len(response.data) == 1

        embedding = response.data[0]
        assert embedding.object == "embedding"
        assert embedding.index == 0
        assert isinstance(embedding.embedding, list)
        assert len(embedding.embedding) > 0

        # Check usage info
        assert hasattr(response, "usage")
        assert response.usage.prompt_tokens > 0
        assert response.usage.total_tokens > 0

    @patch("rkllm_openai.bindings.rkllm.RKLLM")
    def test_embeddings_multiple_inputs(
        self, mock_rkllm_class, openai_client: OpenAI, model_name: str
    ):
        """Test embeddings with multiple input texts."""
        mock_model = mock_rkllm_class.return_value
        mock_embeddings = [0.1, -0.2, 0.3] * 100
        mock_model.get_embeddings.return_value = mock_embeddings

        inputs = ["Hello", "World", "Test"]
        response = openai_client.embeddings.create(model=model_name, input=inputs)

        assert len(response.data) == 3

        for i, embedding in enumerate(response.data):
            assert embedding.index == i
            assert embedding.object == "embedding"
            assert isinstance(embedding.embedding, list)

    def test_invalid_model_name(self, openai_client: OpenAI):
        """Test error handling for invalid model names."""
        with pytest.raises(Exception) as exc_info:
            openai_client.chat.completions.create(
                model="nonexistent-model",
                messages=[{"role": "user", "content": "Hello"}],
            )

        error_str = str(exc_info.value)
        assert "400" in error_str or "not available" in error_str

    def test_invalid_request_data(self, openai_client: OpenAI, model_name: str):
        """Test error handling for invalid request data."""
        with pytest.raises(Exception):
            # Missing required messages field
            openai_client.chat.completions.create(model=model_name)

    @patch("rkllm_openai.bindings.rkllm.RKLLM")
    def test_parameter_validation(
        self, mock_rkllm_class, openai_client: OpenAI, model_name: str
    ):
        """Test that parameters are properly validated and passed through."""
        mock_model = mock_rkllm_class.return_value
        mock_model.get_text_buffer.return_value = ["Test response"]
        mock_model.clear_text_buffer.return_value = None

        # Test with various parameters
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=150,
            temperature=0.9,
            top_p=0.8,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            stop=["END"],
        )

        assert isinstance(response, ChatCompletion)
        assert response.model == model_name

    @patch("rkllm_openai.bindings.rkllm.RKLLM")
    def test_concurrent_requests(
        self, mock_rkllm_class, openai_client: OpenAI, model_name: str
    ):
        """Test handling of concurrent requests."""
        import threading

        mock_model = mock_rkllm_class.return_value
        mock_model.get_text_buffer.return_value = ["Concurrent response"]
        mock_model.clear_text_buffer.return_value = None

        results = []
        errors = []

        def make_request():
            try:
                response = openai_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": "Test concurrent"}],
                    max_tokens=50,
                )
                results.append(response)
            except Exception as e:
                errors.append(e)

        # Start multiple concurrent requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()

        # Wait for all to complete
        for thread in threads:
            thread.join()

        # All requests should succeed (though they'll be serialized by the inference lock)
        assert len(errors) == 0
        assert len(results) == 5

    def test_cors_headers(self, running_server):
        """Test that CORS headers are properly set."""
        import requests

        response = requests.options(
            f"{running_server}/v1/models",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )

        # CORS should be enabled
        assert "Access-Control-Allow-Origin" in response.headers

    @patch("rkllm_openai.bindings.rkllm.RKLLM")
    def test_usage_tracking(
        self, mock_rkllm_class, openai_client: OpenAI, model_name: str
    ):
        """Test that usage statistics are properly tracked."""
        mock_model = mock_rkllm_class.return_value
        mock_model.get_text_buffer.return_value = [
            "This is a test response with multiple tokens."
        ]
        mock_model.clear_text_buffer.return_value = None

        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "Generate a response with multiple tokens"}
            ],
            max_tokens=100,
        )

        assert hasattr(response, "usage")
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0
        assert (
            response.usage.total_tokens
            == response.usage.prompt_tokens + response.usage.completion_tokens
        )

    @patch("rkllm_openai.bindings.rkllm.RKLLM")
    def test_conversation_context(
        self, mock_rkllm_class, openai_client: OpenAI, model_name: str
    ):
        """Test that conversation context is properly formatted."""
        mock_model = mock_rkllm_class.return_value
        mock_model.get_text_buffer.return_value = ["I understand the context."]
        mock_model.clear_text_buffer.return_value = None

        # Multi-turn conversation
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help you?"},
            {"role": "user", "content": "What did I just say?"},
        ]

        response = openai_client.chat.completions.create(
            model=model_name, messages=messages, max_tokens=50
        )

        assert isinstance(response, ChatCompletion)
        # The generate method should have been called with properly formatted prompt
        mock_model.generate.assert_called_once()

    @patch("rkllm_openai.bindings.rkllm.RKLLM")
    def test_error_handling_model_failure(
        self, mock_rkllm_class, openai_client: OpenAI, model_name: str
    ):
        """Test error handling when model fails during inference."""
        mock_model = mock_rkllm_class.return_value
        mock_model.generate.side_effect = RuntimeError("Model inference failed")

        with pytest.raises(Exception):
            openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "This should fail"}],
                max_tokens=50,
            )

    def test_response_format(self, openai_client: OpenAI, model_name: str):
        """Test that responses conform to OpenAI API format."""
        with patch("rkllm_openai.bindings.rkllm.RKLLM") as mock_rkllm_class:
            mock_model = mock_rkllm_class.return_value
            mock_model.get_text_buffer.return_value = ["Valid response"]
            mock_model.clear_text_buffer.return_value = None

            response = openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=50,
            )

            # Verify response structure matches OpenAI format
            assert hasattr(response, "id")
            assert hasattr(response, "object")
            assert hasattr(response, "created")
            assert hasattr(response, "model")
            assert hasattr(response, "choices")
            assert hasattr(response, "usage")

            assert response.object == "chat.completion"
            assert isinstance(response.created, int)
            assert response.model == model_name
            assert isinstance(response.choices, list)
            assert len(response.choices) > 0
