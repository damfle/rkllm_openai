#!/usr/bin/env python3
"""
Simple import tests to verify all modules can be imported correctly.
"""

import pytest


class TestImports:
    """Test that all modules can be imported without errors."""

    def test_import_bindings(self):
        """Test importing bindings module."""
        try:
            from rkllm_openai import bindings
            from rkllm_openai.bindings import RKLLM, RKLLMInput, RKLLMParam

            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import bindings: {e}")

    def test_import_commons(self):
        """Test importing commons module."""
        try:
            from rkllm_openai import commons
            from rkllm_openai.commons import ChatCompletionRequest, CompletionRequest

            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import commons: {e}")

    def test_import_model_manager(self):
        """Test importing model manager."""
        try:
            from rkllm_openai.model_manager import ModelManager

            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import model manager: {e}")

    def test_import_server(self):
        """Test importing server module."""
        try:
            from rkllm_openai.server import create_app, main

            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import server: {e}")

    def test_package_structure(self):
        """Test overall package structure."""
        try:
            import rkllm_openai
            import rkllm_openai.bindings
            import rkllm_openai.commons

            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import package structure: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
