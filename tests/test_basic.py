#!/usr/bin/env python3
"""
Basic tests for RKLLM OpenAI API without external dependencies.
"""

import sys
import traceback
from pathlib import Path

# Add parent directory to Python path so we can import rkllm_openai
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_import_bindings():
    """Test importing bindings module."""
    try:
        from rkllm_openai import bindings
        from rkllm_openai.bindings import (
            RKLLM,
            RKLLM_INPUT_PROMPT,
            RKLLM_RUN_FINISH,
            RKLLM_RUN_NORMAL,
            RKLLMInput,
            RKLLMParam,
        )

        print("✓ Bindings import successful")
        return True
    except Exception as e:
        print(f"✗ Bindings import failed: {e}")
        traceback.print_exc()
        return False


def test_import_commons():
    """Test importing commons module."""
    try:
        from rkllm_openai import commons
        from rkllm_openai.commons import (
            ChatCompletionRequest,
            CompletionRequest,
            EmbeddingRequest,
        )

        print("✓ Commons import successful")
        return True
    except Exception as e:
        print(f"✗ Commons import failed: {e}")
        traceback.print_exc()
        return False


def test_import_model_manager():
    """Test importing model manager."""
    try:
        from rkllm_openai.model_manager import ModelManager

        print("✓ ModelManager import successful")
        return True
    except Exception as e:
        print(f"✗ ModelManager import failed: {e}")
        traceback.print_exc()
        return False


def test_bindings_constants():
    """Test that bindings constants have correct values."""
    try:
        from rkllm_openai.bindings import (
            RKLLM_INPUT_PROMPT,
            RKLLM_RUN_FINISH,
            RKLLM_RUN_NORMAL,
        )

        assert RKLLM_RUN_NORMAL == 0
        assert RKLLM_RUN_FINISH == 2
        assert RKLLM_INPUT_PROMPT == 0

        print("✓ Bindings constants are correct")
        return True
    except Exception as e:
        print(f"✗ Bindings constants test failed: {e}")
        traceback.print_exc()
        return False


def test_bindings_structures():
    """Test that bindings structures can be instantiated."""
    try:
        from rkllm_openai.bindings import (
            RKLLMInput,
            RKLLMParam,
        )

        param = RKLLMParam()
        input_struct = RKLLMInput()

        print("✓ Bindings structures instantiation successful")
        return True
    except Exception as e:
        print(f"✗ Bindings structures test failed: {e}")
        traceback.print_exc()
        return False


def test_rkllm_class():
    """Test RKLLM class has expected methods."""
    try:
        from rkllm_openai.bindings import RKLLM

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

        print("✓ RKLLM class has all expected methods")
        return True
    except Exception as e:
        print(f"✗ RKLLM class test failed: {e}")
        traceback.print_exc()
        return False


def test_model_manager_basic():
    """Test ModelManager basic functionality without actual models."""
    try:
        import os
        import tempfile

        from rkllm_openai.model_manager import ModelManager

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create empty model directory
            allowed_models = {"test_model"}

            # This should not crash even if no models exist
            try:
                manager = ModelManager(
                    model_dir=tmpdir,
                    platform="rk3588",
                    lib_path="/tmp/dummy.so",
                    allowed_models=allowed_models,
                    model_timeout=60,
                )

                # Should be able to get available models (empty)
                available = manager.get_available_models()
                assert isinstance(available, set)

                # Should be able to get model info
                info = manager.get_model_info()
                assert isinstance(info, dict)

                manager.shutdown()

            except Exception as inner_e:
                # This is expected since we don't have real models
                if "not found in model directory" in str(inner_e):
                    pass  # Expected error
                else:
                    raise inner_e

        print("✓ ModelManager basic functionality works")
        return True
    except Exception as e:
        print(f"✗ ModelManager basic test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    tests = [
        ("Bindings Import", test_import_bindings),
        ("Commons Import", test_import_commons),
        ("ModelManager Import", test_import_model_manager),
        ("Bindings Constants", test_bindings_constants),
        ("Bindings Structures", test_bindings_structures),
        ("RKLLM Class", test_rkllm_class),
        ("ModelManager Basic", test_model_manager_basic),
    ]

    print("🧪 Running RKLLM OpenAI API basic tests")
    print("=" * 50)

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All basic tests passed!")
        return 0
    else:
        print(f"❌ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
