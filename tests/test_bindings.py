#!/usr/bin/env python3
"""
Test script for RKLLM bindings.

This script tests the bindings package to ensure the refactoring worked correctly.
"""

import pytest


class TestBindings:
    """Test RKLLM bindings imports and functionality."""

    def test_imports(self):
        """Test that all bindings can be imported correctly."""
        from rkllm_openai.bindings import (
            RKLLM,
            RKLLM_INPUT_PROMPT,
            RKLLM_RUN_FINISH,
            RKLLM_RUN_NORMAL,
            RKLLMInput,
            RKLLMParam,
        )

    def test_constants(self):
        """Test that constants are properly defined."""
        from rkllm_openai.bindings import (
            RKLLM_INPUT_EMBED,
            RKLLM_INPUT_MULTIMODAL,
            RKLLM_INPUT_PROMPT,
            RKLLM_INPUT_TOKEN,
            RKLLM_RUN_ERROR,
            RKLLM_RUN_FINISH,
            RKLLM_RUN_NORMAL,
            RKLLM_RUN_WAITING,
        )

        # Verify constants have expected values
        assert RKLLM_RUN_NORMAL == 0
        assert RKLLM_RUN_WAITING == 1
        assert RKLLM_RUN_FINISH == 2
        assert RKLLM_RUN_ERROR == 3
        assert RKLLM_INPUT_PROMPT == 0
        assert RKLLM_INPUT_TOKEN == 1
        assert RKLLM_INPUT_EMBED == 2
        assert RKLLM_INPUT_MULTIMODAL == 3

    def test_structures(self):
        """Test that structures can be instantiated."""
        from rkllm_openai.bindings import (
            RKLLMExtendParam,
            RKLLMInput,
            RKLLMParam,
            RKLLMResult,
        )

        # Try to create structures
        param = RKLLMParam()
        input_struct = RKLLMInput()
        extend_param = RKLLMExtendParam()
        result = RKLLMResult()

    def test_rkllm_class(self):
        """Test RKLLM class can be imported and basic methods exist."""
        from rkllm_openai.bindings import RKLLM

        # Check that class has expected methods
        expected_methods = [
            "generate",
            "get_embeddings",
            "get_text_buffer",
            "clear_text_buffer",
            "get_state",
            "is_finished",
            "has_error",
            "release",
        ]

        for method in expected_methods:
            assert hasattr(RKLLM, method), f"RKLLM missing method: {method}"

    def test_server_import(self):
        """Test that server can import bindings."""
        from rkllm_openai.server import create_app, main

    def test_package_structure(self):
        """Test the overall package structure."""
        import rkllm_openai
        import rkllm_openai.bindings
        import rkllm_openai.bindings.constants
        import rkllm_openai.bindings.rkllm
        import rkllm_openai.bindings.structures


if __name__ == "__main__":
    pytest.main([__file__])
