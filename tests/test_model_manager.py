#!/usr/bin/env python3
"""
Test ModelManager with current single-model implementation.
"""

import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rkllm_openai.model_manager import ModelManager


class TestModelManager:
    """Test the ModelManager class with single model support."""

    @pytest.fixture
    def temp_model_file(self):
        """Create a temporary model file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".rkllm", delete=False) as temp_file:
            temp_file.write(b"dummy model content")
            temp_file_path = temp_file.name

        yield temp_file_path

        # Cleanup
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

    @pytest.fixture
    def temp_chat_template(self):
        """Create a temporary chat template file."""
        template_content = """
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

        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

    @patch("rkllm_openai.model_manager.RKLLM")
    def test_init_with_valid_file(self, mock_rkllm_class, temp_model_file):
        """Test ModelManager initialization with a valid model file."""
        manager = ModelManager(
            model_path=temp_model_file,
            platform="rk3588",
            lib_path="/dummy/lib.so",
            model_timeout=300,
        )

        # Test that model name is derived from filename
        expected_name = Path(temp_model_file).stem
        assert manager.model_name == expected_name

        # Test that model path is stored correctly
        assert str(manager.model_path) == temp_model_file

        # Test available models
        available = manager.get_available_models()
        assert available == {expected_name}

        manager.shutdown()

    def test_init_with_invalid_file(self):
        """Test ModelManager initialization with invalid file."""
        with pytest.raises(ValueError, match="Model file does not exist"):
            ModelManager(
                model_path="/nonexistent/model.rkllm",
                platform="rk3588",
                lib_path="/dummy/lib.so",
            )

    @patch("rkllm_openai.model_manager.RKLLM")
    def test_init_with_chat_template(
        self, mock_rkllm_class, temp_model_file, temp_chat_template
    ):
        """Test ModelManager initialization with chat template."""
        manager = ModelManager(
            model_path=temp_model_file,
            platform="rk3588",
            lib_path="/dummy/lib.so",
            chat_template=temp_chat_template,
        )

        assert manager.chat_template == temp_chat_template
        manager.shutdown()

    @patch("rkllm_openai.model_manager.RKLLM")
    def test_get_model_success(self, mock_rkllm_class, temp_model_file):
        """Test successful model loading."""
        mock_model = MagicMock()
        mock_rkllm_class.return_value = mock_model

        manager = ModelManager(
            model_path=temp_model_file, platform="rk3588", lib_path="/dummy/lib.so"
        )

        model_name = manager.model_name
        loaded_model = manager.get_model(model_name)

        # Verify the model was loaded correctly
        assert loaded_model == mock_model
        mock_rkllm_class.assert_called_once_with(
            temp_model_file, "rk3588", "/dummy/lib.so"
        )

        # Verify model is tracked as loaded
        assert manager.is_model_loaded(model_name)
        assert model_name in manager.get_loaded_models()

        manager.shutdown()

    @patch("rkllm_openai.model_manager.RKLLM")
    def test_get_model_with_chat_template(
        self, mock_rkllm_class, temp_model_file, temp_chat_template
    ):
        """Test model loading with chat template application."""
        mock_model = MagicMock()
        mock_rkllm_class.return_value = mock_model

        manager = ModelManager(
            model_path=temp_model_file,
            platform="rk3588",
            lib_path="/dummy/lib.so",
            chat_template=temp_chat_template,
        )

        model_name = manager.model_name
        loaded_model = manager.get_model(model_name)

        # Verify chat template was applied
        mock_model.load_chat_template_from_file.assert_called_once_with(
            temp_chat_template
        )

        manager.shutdown()

    @patch("rkllm_openai.model_manager.RKLLM")
    def test_get_model_wrong_name(self, mock_rkllm_class, temp_model_file):
        """Test error when requesting wrong model name."""
        manager = ModelManager(
            model_path=temp_model_file, platform="rk3588", lib_path="/dummy/lib.so"
        )

        with pytest.raises(ValueError, match="not available"):
            manager.get_model("wrong-model-name")

        manager.shutdown()

    @patch("rkllm_openai.model_manager.RKLLM")
    def test_model_caching(self, mock_rkllm_class, temp_model_file):
        """Test that models are cached and not reloaded."""
        mock_model = MagicMock()
        mock_rkllm_class.return_value = mock_model

        manager = ModelManager(
            model_path=temp_model_file, platform="rk3588", lib_path="/dummy/lib.so"
        )

        model_name = manager.model_name

        # Load model twice
        model1 = manager.get_model(model_name)
        model2 = manager.get_model(model_name)

        # Should be the same instance (cached)
        assert model1 is model2

        # RKLLM should only be called once
        mock_rkllm_class.assert_called_once()

        manager.shutdown()

    @patch("rkllm_openai.model_manager.RKLLM")
    def test_model_loading_failure(self, mock_rkllm_class, temp_model_file):
        """Test handling of model loading failure."""
        mock_rkllm_class.side_effect = Exception("Failed to load model")

        manager = ModelManager(
            model_path=temp_model_file, platform="rk3588", lib_path="/dummy/lib.so"
        )

        model_name = manager.model_name

        with pytest.raises(Exception, match="Failed to load model"):
            manager.get_model(model_name)

        # Model should not be tracked as loaded
        assert not manager.is_model_loaded(model_name)
        assert len(manager.get_loaded_models()) == 0

        manager.shutdown()

    @patch("rkllm_openai.model_manager.RKLLM")
    def test_preload_model(self, mock_rkllm_class, temp_model_file):
        """Test preloading a model."""
        mock_model = MagicMock()
        mock_rkllm_class.return_value = mock_model

        manager = ModelManager(
            model_path=temp_model_file, platform="rk3588", lib_path="/dummy/lib.so"
        )

        model_name = manager.model_name

        # Preload model
        manager.preload_model(model_name)

        # Model should be loaded
        assert manager.is_model_loaded(model_name)
        mock_rkllm_class.assert_called_once()

        manager.shutdown()

    @patch("rkllm_openai.model_manager.RKLLM")
    def test_unload_all_models(self, mock_rkllm_class, temp_model_file):
        """Test unloading all models."""
        mock_model = MagicMock()
        mock_rkllm_class.return_value = mock_model

        manager = ModelManager(
            model_path=temp_model_file, platform="rk3588", lib_path="/dummy/lib.so"
        )

        model_name = manager.model_name

        # Load model
        manager.get_model(model_name)
        assert manager.is_model_loaded(model_name)

        # Unload all models
        manager.unload_all_models()

        # Model should no longer be loaded
        assert not manager.is_model_loaded(model_name)
        assert len(manager.get_loaded_models()) == 0

        # Release should have been called
        mock_model.release.assert_called_once()

        manager.shutdown()

    @patch("rkllm_openai.model_manager.RKLLM")
    def test_model_timeout_cleanup(self, mock_rkllm_class, temp_model_file):
        """Test automatic model cleanup after timeout."""
        mock_model = MagicMock()
        mock_rkllm_class.return_value = mock_model

        # Use very short timeout for testing
        manager = ModelManager(
            model_path=temp_model_file,
            platform="rk3588",
            lib_path="/dummy/lib.so",
            model_timeout=1,  # 1 second
        )

        model_name = manager.model_name

        # Load model
        manager.get_model(model_name)
        assert manager.is_model_loaded(model_name)

        # Wait for timeout
        time.sleep(2)

        # Give cleanup thread time to run
        time.sleep(1)

        # Model should be unloaded
        assert not manager.is_model_loaded(model_name)

        manager.shutdown()

    @patch("rkllm_openai.model_manager.RKLLM")
    def test_extend_model_timeout(self, mock_rkllm_class, temp_model_file):
        """Test extending model timeout."""
        mock_model = MagicMock()
        mock_rkllm_class.return_value = mock_model

        manager = ModelManager(
            model_path=temp_model_file, platform="rk3588", lib_path="/dummy/lib.so"
        )

        model_name = manager.model_name

        # Load model and get initial timestamp
        manager.get_model(model_name)
        original_time = manager._last_used[model_name]

        time.sleep(0.1)  # Small delay

        # Extend timeout
        manager.extend_model_timeout(model_name)
        new_time = manager._last_used[model_name]

        assert new_time > original_time

        manager.shutdown()

    @patch("rkllm_openai.model_manager.RKLLM")
    def test_extend_timeout_wrong_model(self, mock_rkllm_class, temp_model_file):
        """Test error when extending timeout for wrong model."""
        manager = ModelManager(
            model_path=temp_model_file, platform="rk3588", lib_path="/dummy/lib.so"
        )

        with pytest.raises(ValueError, match="not available"):
            manager.extend_model_timeout("wrong-model")

        manager.shutdown()

    @patch("rkllm_openai.model_manager.RKLLM")
    def test_get_model_info(self, mock_rkllm_class, temp_model_file):
        """Test getting model information."""
        mock_model = MagicMock()
        mock_rkllm_class.return_value = mock_model

        manager = ModelManager(
            model_path=temp_model_file, platform="rk3588", lib_path="/dummy/lib.so"
        )

        model_name = manager.model_name

        # Get info before loading
        info = manager.get_model_info()
        assert model_name in info
        assert info[model_name]["available"] is True
        assert info[model_name]["loaded"] is False
        assert info[model_name]["path"] == temp_model_file

        # Load model and check info again
        manager.get_model(model_name)
        info = manager.get_model_info()
        assert info[model_name]["loaded"] is True
        assert info[model_name]["last_used"] is not None

        manager.shutdown()

    @patch("rkllm_openai.model_manager.RKLLM")
    def test_load_chat_template_loaded_model(
        self, mock_rkllm_class, temp_model_file, temp_chat_template
    ):
        """Test loading chat template for an already loaded model."""
        mock_model = MagicMock()
        mock_rkllm_class.return_value = mock_model

        manager = ModelManager(
            model_path=temp_model_file, platform="rk3588", lib_path="/dummy/lib.so"
        )

        model_name = manager.model_name

        # Load model first
        manager.get_model(model_name)

        # Load chat template
        manager.load_chat_template(model_name, temp_chat_template)

        # Verify template was loaded
        mock_model.load_chat_template_from_file.assert_called_with(temp_chat_template)

        manager.shutdown()

    @patch("rkllm_openai.model_manager.RKLLM")
    def test_load_chat_template_wrong_model(
        self, mock_rkllm_class, temp_model_file, temp_chat_template
    ):
        """Test error when loading chat template for wrong model."""
        manager = ModelManager(
            model_path=temp_model_file, platform="rk3588", lib_path="/dummy/lib.so"
        )

        with pytest.raises(ValueError, match="not available"):
            manager.load_chat_template("wrong-model", temp_chat_template)

        manager.shutdown()

    @patch("rkllm_openai.model_manager.RKLLM")
    def test_cleanup_thread_lifecycle(self, mock_rkllm_class, temp_model_file):
        """Test that cleanup thread starts and stops properly."""
        manager = ModelManager(
            model_path=temp_model_file, platform="rk3588", lib_path="/dummy/lib.so"
        )

        # Cleanup thread should be started
        assert manager._cleanup_thread is not None
        assert manager._cleanup_thread.is_alive()

        # Shutdown should stop the thread
        manager.shutdown()

        # Give thread time to stop
        time.sleep(0.5)
        assert not manager._cleanup_thread.is_alive()

    @patch("rkllm_openai.model_manager.RKLLM")
    def test_thread_safety(self, mock_rkllm_class, temp_model_file):
        """Test that ModelManager is thread-safe."""
        mock_model = MagicMock()
        mock_rkllm_class.return_value = mock_model

        manager = ModelManager(
            model_path=temp_model_file, platform="rk3588", lib_path="/dummy/lib.so"
        )

        model_name = manager.model_name
        results = []
        errors = []

        def load_model():
            try:
                model = manager.get_model(model_name)
                results.append(model)
            except Exception as e:
                errors.append(e)

        # Start multiple threads loading the same model
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=load_model)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should have no errors and all results should be the same instance
        assert len(errors) == 0
        assert len(results) == 10
        assert all(result is results[0] for result in results)

        # RKLLM should only be called once (model cached)
        mock_rkllm_class.assert_called_once()

        manager.shutdown()

    def test_model_name_extraction(self, temp_model_file):
        """Test that model names are correctly extracted from file paths."""
        # Test with different file extensions
        test_cases = [
            ("/path/to/model.rkllm", "model"),
            ("/path/to/my-awesome-model.rkllm", "my-awesome-model"),
            ("/path/to/model_v2.bin", "model_v2"),
        ]

        for file_path, expected_name in test_cases:
            # Create temporary file with specific name
            temp_dir = Path(temp_model_file).parent
            test_file = temp_dir / Path(file_path).name
            test_file.write_bytes(b"dummy content")

            try:
                manager = ModelManager(
                    model_path=str(test_file),
                    platform="rk3588",
                    lib_path="/dummy/lib.so",
                )

                assert manager.model_name == expected_name
                manager.shutdown()
            finally:
                if test_file.exists():
                    test_file.unlink()
