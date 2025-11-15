#!/usr/bin/env python3
"""
Test basic imports and module structure for RKLLM OpenAI API.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestImports:
    """Test that all modules can be imported correctly."""

    def test_main_package_import(self):
        """Test that the main package can be imported."""
        import rkllm_openai

        assert hasattr(rkllm_openai, "__version__")
        assert hasattr(rkllm_openai, "create_app")
        assert hasattr(rkllm_openai, "main")

    def test_server_import(self):
        """Test that the server module can be imported."""
        from rkllm_openai import server

        assert hasattr(server, "create_app")
        assert hasattr(server, "main")
        assert callable(server.create_app)
        assert callable(server.main)

    def test_model_manager_import(self):
        """Test that the model manager can be imported."""
        from rkllm_openai.model_manager import ModelManager

        assert ModelManager is not None
        # Check that the class has the expected methods
        expected_methods = [
            "get_model",
            "get_available_models",
            "get_loaded_models",
            "is_model_loaded",
            "unload_all_models",
            "shutdown",
        ]
        for method in expected_methods:
            assert hasattr(ModelManager, method)

    def test_commons_import(self):
        """Test that commons modules can be imported."""
        # Test that all models are proper Pydantic BaseModel classes
        import pydantic

        from rkllm_openai.commons import (
            ChatCompletionRequest,
            ChatMessage,
            CompletionRequest,
            EmbeddingRequest,
            Tool,
            ToolCall,
            ToolFunction,
            clean_content_for_tools,
            parse_tool_calls,
        )

        model_classes = [
            ChatCompletionRequest,
            ChatMessage,
            CompletionRequest,
            EmbeddingRequest,
            Tool,
            ToolCall,
            ToolFunction,
        ]
        for model_class in model_classes:
            assert issubclass(model_class, pydantic.BaseModel)

    def test_response_generators_import(self):
        """Test that response generators can be imported."""
        try:
            from rkllm_openai.commons import (
                generate_chat_completion,
                generate_completion,
                stream_chat_completion,
                stream_completion,
            )

            # If Flask is available, these should be callable
            assert callable(generate_chat_completion)
            assert callable(generate_completion)
            assert callable(stream_chat_completion)
            assert callable(stream_completion)
        except ImportError:
            # If Flask is not available, this is expected
            pytest.skip("Flask not available - response generators cannot be imported")

    def test_bindings_import(self):
        """Test that bindings can be imported."""
        from rkllm_openai.bindings import (
            RKLLM,
            RKLLM_INPUT_PROMPT,
            RKLLM_RUN_FINISH,
            RKLLM_RUN_NORMAL,
            RKLLMInput,
            RKLLMParam,
        )

        # Test that RKLLM class exists and has expected methods
        expected_methods = [
            "generate",
            "get_state",
            "is_finished",
            "has_error",
            "get_text_buffer",
            "clear_text_buffer",
            "release",
        ]
        for method in expected_methods:
            assert hasattr(RKLLM, method)

    def test_tool_utilities_import(self):
        """Test that tool utilities can be imported and work."""
        from rkllm_openai.commons.tool_utils import (
            clean_content_for_tools,
            parse_tool_calls,
        )

        # Test parse_tool_calls with sample content
        content_with_tool = """
        I'll help you with that task.

        <tool_call>
        {"name": "search", "arguments": {"query": "test"}}
        </tool_call>

        Let me search for that information.
        """

        tool_calls = parse_tool_calls(content_with_tool)
        assert isinstance(tool_calls, list)
        if tool_calls:  # If parsing found tool calls
            assert "id" in tool_calls[0]
            assert "type" in tool_calls[0]
            assert "function" in tool_calls[0]

        # Test clean_content_for_tools
        cleaned = clean_content_for_tools(content_with_tool)
        assert "<tool_call>" not in cleaned
        assert "I'll help you with that task." in cleaned

    def test_models_validation(self):
        """Test that Pydantic models work correctly."""
        from rkllm_openai.commons import ChatCompletionRequest, ChatMessage

        # Test ChatMessage creation
        message = ChatMessage(role="user", content="Hello, world!")
        assert message.role == "user"
        assert message.content == "Hello, world!"
        assert message.tool_calls is None

        # Test ChatCompletionRequest creation
        request = ChatCompletionRequest(
            model="test-model", messages=[message], max_tokens=100, temperature=0.8
        )
        assert request.model == "test-model"
        assert len(request.messages) == 1
        assert request.max_tokens == 100
        assert request.temperature == 0.8

    def test_constants_values(self):
        """Test that constants have expected values."""
        from rkllm_openai.bindings import (
            RKLLM_INPUT_PROMPT,
            RKLLM_INPUT_TOKEN,
            RKLLM_RUN_ERROR,
            RKLLM_RUN_FINISH,
            RKLLM_RUN_NORMAL,
            RKLLM_RUN_WAITING,
        )

        # Test that constants have the expected values from the C library
        assert RKLLM_RUN_NORMAL == 0
        assert RKLLM_RUN_WAITING == 1
        assert RKLLM_RUN_FINISH == 2
        assert RKLLM_RUN_ERROR == 3
        assert RKLLM_INPUT_PROMPT == 0
        assert RKLLM_INPUT_TOKEN == 1

    def test_package_structure(self):
        """Test that the package structure is correct."""
        import rkllm_openai

        # Test that __all__ contains expected exports
        expected_exports = ["create_app", "main", "bindings"]
        for export in expected_exports:
            assert export in rkllm_openai.__all__
            assert hasattr(rkllm_openai, export)

    def test_lazy_imports(self):
        """Test that lazy imports work correctly."""
        # Test that create_app and main can be called without importing Flask initially
        import rkllm_openai

        # These should be callable functions
        assert callable(rkllm_openai.create_app)
        assert callable(rkllm_openai.main)

        # They should be wrappers that do lazy imports
        import inspect

        create_app_source = inspect.getsource(rkllm_openai.create_app)
        main_source = inspect.getsource(rkllm_openai.main)

        # Should contain lazy import patterns
        assert "from .server import" in create_app_source
        assert "from .server import" in main_source
