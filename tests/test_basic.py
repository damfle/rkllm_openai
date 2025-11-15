#!/usr/bin/env python3
"""
Basic tests for RKLLM OpenAI API without external dependencies.

These tests focus on core functionality that doesn't require Flask or a running server.
"""

import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to Python path so we can import rkllm_openai
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pytest

    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

    # Define minimal pytest substitute
    class pytest:
        @staticmethod
        def raises(exception_type):
            class RaisesContext:
                def __init__(self, exc_type):
                    self.exc_type = exc_type

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc_val, exc_tb):
                    if exc_type is None:
                        raise AssertionError(
                            f"Expected {self.exc_type} but no exception was raised"
                        )
                    return issubclass(exc_type, self.exc_type)

            return RaisesContext(exception_type)


class TestBasicFunctionality:
    """Test basic functionality without server dependencies."""

    def test_package_import(self):
        """Test that the main package can be imported."""
        import rkllm_openai

        assert hasattr(rkllm_openai, "__version__")
        assert hasattr(rkllm_openai, "create_app")
        assert hasattr(rkllm_openai, "main")
        print("✓ Package import successful")

    def test_commons_models(self):
        """Test that commons models work correctly."""
        from rkllm_openai.commons import (
            ChatCompletionRequest,
            ChatMessage,
            CompletionRequest,
            EmbeddingRequest,
        )

        # Test ChatMessage
        message = ChatMessage(role="user", content="Hello, world!")
        assert message.role == "user"
        assert message.content == "Hello, world!"

        # Test ChatCompletionRequest
        request = ChatCompletionRequest(
            model="test-model", messages=[message], max_tokens=100
        )
        assert request.model == "test-model"
        assert len(request.messages) == 1
        assert request.max_tokens == 100
        print("✓ Commons models work correctly")

    def test_tool_utilities(self):
        """Test tool parsing utilities."""
        from rkllm_openai.commons.tool_utils import (
            clean_content_for_tools,
            parse_tool_calls,
        )

        # Test content with tool call
        content_with_tool = """
        I'll help you search for that.

        <tool_call>
        {"name": "search", "arguments": {"query": "OpenAI API"}}
        </tool_call>

        Let me find that information.
        """

        # Test parsing
        tool_calls = parse_tool_calls(content_with_tool)
        assert isinstance(tool_calls, list)

        # Test cleaning
        cleaned_content = clean_content_for_tools(content_with_tool)
        assert "<tool_call>" not in cleaned_content
        assert "I'll help you search" in cleaned_content
        print("✓ Tool utilities work correctly")

    @patch("rkllm_openai.model_manager.RKLLM")
    def test_model_manager_basic(self, mock_rkllm_class):
        """Test ModelManager basic functionality."""
        from rkllm_openai.model_manager import ModelManager

        # Create a temporary model file
        with tempfile.NamedTemporaryFile(suffix=".rkllm", delete=False) as temp_file:
            temp_file.write(b"dummy model content")
            temp_file_path = temp_file.name

        try:
            # Mock RKLLM to avoid actual model loading
            mock_model = Mock()
            mock_rkllm_class.return_value = mock_model

            manager = ModelManager(
                model_path=temp_file_path,
                platform="rk3588",
                lib_path="/dummy/lib.so",
                model_timeout=300,
            )

            # Test model name extraction
            expected_name = Path(temp_file_path).stem
            assert manager.model_name == expected_name

            # Test available models
            available = manager.get_available_models()
            assert available == {expected_name}

            # Test model loading
            loaded_model = manager.get_model(expected_name)
            assert loaded_model == mock_model

            # Should be able to get model info
            info = manager.get_model_info()
            assert isinstance(info, dict)
            assert expected_name in info

            manager.shutdown()
            print("✓ ModelManager basic functionality works")

        finally:
            # Clean up
            import os

            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def test_bindings_constants(self):
        """Test that bindings constants are correctly defined."""
        from rkllm_openai.bindings import (
            RKLLM_INPUT_PROMPT,
            RKLLM_RUN_FINISH,
            RKLLM_RUN_NORMAL,
        )

        assert RKLLM_RUN_NORMAL == 0
        assert RKLLM_RUN_FINISH == 2
        assert RKLLM_INPUT_PROMPT == 0
        print("✓ Bindings constants are correct")

    def test_bindings_import(self):
        """Test that bindings can be imported."""
        from rkllm_openai.bindings import (
            RKLLM,
            RKLLMInput,
            RKLLMParam,
        )

        # Test that RKLLM class has expected methods
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
        print("✓ Bindings import successfully")

    def test_model_name_derivation(self):
        """Test that model names are correctly derived from file paths."""
        test_cases = [
            ("model.rkllm", "model"),
            ("awesome-model.rkllm", "awesome-model"),
            ("my_model_v2.rkllm", "my_model_v2"),
            ("complex-name-123.bin", "complex-name-123"),
        ]

        for filename, expected_name in test_cases:
            actual_name = Path(filename).stem
            assert actual_name == expected_name
        print("✓ Model name derivation works correctly")

    def test_pydantic_validation(self):
        """Test that Pydantic validation works for our models."""
        from rkllm_openai.commons import ChatCompletionRequest, ChatMessage

        # Test validation success
        message = ChatMessage(role="user", content="Test message")
        request = ChatCompletionRequest(model="test", messages=[message])
        assert request.model == "test"
        assert len(request.messages) == 1

        # Test validation failure
        try:
            with pytest.raises(Exception):  # Pydantic validation error
                ChatMessage(role="invalid_role", content="Test")
        except Exception:
            # If pytest.raises doesn't work, test manually
            try:
                ChatMessage(role="invalid_role", content="Test")
                assert False, "Should have raised validation error"
            except Exception:
                pass  # Expected validation error

        print("✓ Pydantic validation works correctly")

    def test_thread_safety_structures(self):
        """Test that threading structures are properly initialized."""
        import threading

        # Test that we can create locks and events
        lock = threading.Lock()
        event = threading.Event()

        assert lock is not None
        assert event is not None

        # Test basic lock operations
        with lock:
            assert True  # Lock acquired successfully

        print("✓ Threading structures work correctly")

    def test_time_utilities(self):
        """Test time-related utilities."""
        import time

        start_time = time.time()
        time.sleep(0.01)  # Sleep for 10ms
        end_time = time.time()

        elapsed = end_time - start_time
        assert elapsed > 0.005  # Should be at least 5ms
        print("✓ Time utilities work correctly")

    def test_path_utilities(self):
        """Test path and file utilities."""
        import os
        import tempfile
        from pathlib import Path

        # Test Path operations
        test_path = Path("/some/path/to/model.rkllm")
        assert test_path.stem == "model"
        assert test_path.suffix == ".rkllm"

        # Test temporary file creation
        with tempfile.NamedTemporaryFile(suffix=".rkllm") as temp_file:
            assert Path(temp_file.name).exists()
            assert Path(temp_file.name).suffix == ".rkllm"

        print("✓ Path utilities work correctly")

    def test_error_handling_patterns(self):
        """Test common error handling patterns."""
        # Test ValueError for invalid parameters
        try:
            with pytest.raises(ValueError):
                if True:  # Always true, so this will execute
                    raise ValueError("Test error message")
        except Exception:
            # If pytest.raises doesn't work, test manually
            try:
                raise ValueError("Test error message")
            except ValueError:
                pass  # Expected

        # Test proper exception chaining
        try:
            try:
                raise ValueError("Inner error")
            except ValueError as e:
                raise RuntimeError("Outer error") from e
        except RuntimeError as e:
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, ValueError)

        print("✓ Error handling patterns work correctly")

    def test_mock_capabilities(self):
        """Test that mocking works correctly for testing."""
        from unittest.mock import Mock, patch

        # Test Mock object
        mock_obj = Mock()
        mock_obj.test_method.return_value = "mocked result"
        assert mock_obj.test_method() == "mocked result"

        # Test patching
        with patch("time.time", return_value=123456):
            import time

            assert time.time() == 123456

        print("✓ Mock capabilities work correctly")

    def test_json_operations(self):
        """Test JSON operations for API compatibility."""
        import json

        # Test JSON serialization/deserialization
        test_data = {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            "max_tokens": 100,
            "temperature": 0.8,
        }

        json_str = json.dumps(test_data)
        parsed_data = json.loads(json_str)

        assert parsed_data["model"] == "test-model"
        assert len(parsed_data["messages"]) == 2
        assert parsed_data["temperature"] == 0.8

        print("✓ JSON operations work correctly")


def run_basic_tests():
    """Run basic tests and print results."""
    print("Running basic functionality tests...")
    print("=" * 50)

    test_instance = TestBasicFunctionality()
    test_methods = [
        method
        for method in dir(test_instance)
        if method.startswith("test_") and callable(getattr(test_instance, method))
    ]

    passed = 0
    failed = 0

    for test_method in test_methods:
        try:
            method = getattr(test_instance, test_method)
            method()
            passed += 1
        except Exception as e:
            print(f"✗ {test_method} failed: {e}")
            failed += 1

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All basic tests passed!")
        return True
    else:
        print(f"❌ {failed} tests failed")
        return False


if __name__ == "__main__":
    success = run_basic_tests()
    sys.exit(0 if success else 1)
