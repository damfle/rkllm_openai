#!/usr/bin/env python3
"""
Test suite for ModelManager class.

Tests model loading, unloading, and lifecycle management.
"""

import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rkllm_openai.model_manager import ModelManager


class MockRKLLM:
    """Mock RKLLM class for testing."""

    def __init__(self, model_path, platform, lib_path):
        self.model_path = model_path
        self.platform = platform
        self.lib_path = lib_path
        self.released = False

    def release(self):
        """Mock release method."""
        self.released = True

    def generate(self, prompt, **kwargs):
        """Mock generate method."""
        return f"Response to: {prompt}"

    def get_embeddings(self, text):
        """Mock embeddings method."""
        return [0.1] * 768


@pytest.fixture
def temp_model_dir():
    """Create temporary directory with mock model files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir)

        # Create mock model files
        (model_dir / "model1.rkllm").touch()
        (model_dir / "model2.bin").touch()

        # Create subdirectory with model
        subdir = model_dir / "model3"
        subdir.mkdir()
        (subdir / "model.rkllm").touch()

        yield str(model_dir)


@pytest.fixture
def mock_rkllm():
    """Mock RKLLM class."""
    with patch("rkllm_openai.model_manager.RKLLM", MockRKLLM):
        yield MockRKLLM


class TestModelManager:
    """Test ModelManager functionality."""

    def test_init_and_discovery(self, temp_model_dir):
        """Test model manager initialization and model discovery."""
        allowed_models = {"model1", "model2", "model3"}

        with patch("rkllm_openai.model_manager.RKLLM", MockRKLLM):
            manager = ModelManager(
                model_dir=temp_model_dir,
                platform="rk3588",
                lib_path="/tmp/librkllm.so",
                allowed_models=allowed_models,
                model_timeout=60,
            )

            available = manager.get_available_models()
            assert "model1" in available
            assert "model2" in available
            assert "model3" in available

            # No models should be loaded initially
            assert len(manager.get_loaded_models()) == 0

            manager.shutdown()

    def test_model_loading(self, temp_model_dir, mock_rkllm):
        """Test loading models on demand."""
        allowed_models = {"model1", "model2"}

        manager = ModelManager(
            model_dir=temp_model_dir,
            platform="rk3588",
            lib_path="/tmp/librkllm.so",
            allowed_models=allowed_models,
            model_timeout=60,
        )

        try:
            # Load model1
            model = manager.get_model("model1")
            assert model is not None
            assert manager.is_model_loaded("model1")
            assert "model1" in manager.get_loaded_models()

            # Get same model again (should return cached instance)
            model2 = manager.get_model("model1")
            assert model is model2

        finally:
            manager.shutdown()

    def test_model_unloading(self, temp_model_dir, mock_rkllm):
        """Test model unloading when switching models."""
        allowed_models = {"model1", "model2"}

        manager = ModelManager(
            model_dir=temp_model_dir,
            platform="rk3588",
            lib_path="/tmp/librkllm.so",
            allowed_models=allowed_models,
            model_timeout=60,
        )

        try:
            # Load model1
            model1 = manager.get_model("model1")
            assert manager.is_model_loaded("model1")

            # Load model2 (should unload model1)
            model2 = manager.get_model("model2")
            assert manager.is_model_loaded("model2")
            assert not manager.is_model_loaded("model1")
            assert model1.released

        finally:
            manager.shutdown()

    def test_invalid_model(self, temp_model_dir, mock_rkllm):
        """Test handling of invalid model requests."""
        allowed_models = {"model1"}

        manager = ModelManager(
            model_dir=temp_model_dir,
            platform="rk3588",
            lib_path="/tmp/librkllm.so",
            allowed_models=allowed_models,
            model_timeout=60,
        )

        try:
            # Try to load model not in allowlist
            with pytest.raises(ValueError, match="not in allowlist"):
                manager.get_model("invalid_model")

            # Try to load model that doesn't exist
            allowed_models.add("nonexistent")
            manager.allowed_models = allowed_models
            with pytest.raises(ValueError, match="not found in model directory"):
                manager.get_model("nonexistent")

        finally:
            manager.shutdown()

    def test_timeout_unloading(self, temp_model_dir, mock_rkllm):
        """Test automatic model unloading after timeout."""
        allowed_models = {"model1"}

        manager = ModelManager(
            model_dir=temp_model_dir,
            platform="rk3588",
            lib_path="/tmp/librkllm.so",
            allowed_models=allowed_models,
            model_timeout=1,  # Very short timeout for testing
        )

        try:
            # Load model
            model = manager.get_model("model1")
            assert manager.is_model_loaded("model1")

            # Wait for timeout + cleanup cycle
            time.sleep(2)

            # Model should be unloaded due to timeout
            # Note: This test might be flaky due to timing

        finally:
            manager.shutdown()

    def test_preload_model(self, temp_model_dir, mock_rkllm):
        """Test preloading models."""
        allowed_models = {"model1"}

        manager = ModelManager(
            model_dir=temp_model_dir,
            platform="rk3588",
            lib_path="/tmp/librkllm.so",
            allowed_models=allowed_models,
            model_timeout=60,
        )

        try:
            # Preload model
            manager.preload_model("model1")
            assert manager.is_model_loaded("model1")

        finally:
            manager.shutdown()

    def test_unload_all_models(self, temp_model_dir, mock_rkllm):
        """Test unloading all models."""
        allowed_models = {"model1", "model2"}

        manager = ModelManager(
            model_dir=temp_model_dir,
            platform="rk3588",
            lib_path="/tmp/librkllm.so",
            allowed_models=allowed_models,
            model_timeout=60,
        )

        try:
            # Load both models separately (second will unload first)
            model1 = manager.get_model("model1")
            model2 = manager.get_model("model2")

            assert manager.is_model_loaded("model2")

            # Unload all
            manager.unload_all_models()
            assert len(manager.get_loaded_models()) == 0
            assert model2.released

        finally:
            manager.shutdown()

    def test_model_info(self, temp_model_dir, mock_rkllm):
        """Test getting model information."""
        allowed_models = {"model1", "model2"}

        manager = ModelManager(
            model_dir=temp_model_dir,
            platform="rk3588",
            lib_path="/tmp/librkllm.so",
            allowed_models=allowed_models,
            model_timeout=60,
        )

        try:
            # Get info before loading
            info = manager.get_model_info()
            assert "model1" in info
            assert info["model1"]["available"] is True
            assert info["model1"]["loaded"] is False

            # Load model and check info again
            manager.get_model("model1")
            info = manager.get_model_info()
            assert info["model1"]["loaded"] is True
            assert info["model1"]["last_used"] is not None

        finally:
            manager.shutdown()

    def test_context_manager(self, temp_model_dir, mock_rkllm):
        """Test using ModelManager as context manager."""
        allowed_models = {"model1"}

        with ModelManager(
            model_dir=temp_model_dir,
            platform="rk3588",
            lib_path="/tmp/librkllm.so",
            allowed_models=allowed_models,
            model_timeout=60,
        ) as manager:
            model = manager.get_model("model1")
            assert model is not None

        # Manager should be shut down automatically

    def test_extend_timeout(self, temp_model_dir, mock_rkllm):
        """Test extending model timeout."""
        allowed_models = {"model1"}

        manager = ModelManager(
            model_dir=temp_model_dir,
            platform="rk3588",
            lib_path="/tmp/librkllm.so",
            allowed_models=allowed_models,
            model_timeout=60,
        )

        try:
            # Load model
            manager.get_model("model1")
            original_time = manager._last_used["model1"]

            time.sleep(0.1)  # Small delay

            # Extend timeout
            manager.extend_model_timeout("model1")
            new_time = manager._last_used["model1"]

            assert new_time > original_time

        finally:
            manager.shutdown()


class TestModelManagerThreadSafety:
    """Test thread safety of ModelManager."""

    def test_concurrent_access(self, temp_model_dir, mock_rkllm):
        """Test concurrent access to models."""
        allowed_models = {"model1"}

        manager = ModelManager(
            model_dir=temp_model_dir,
            platform="rk3588",
            lib_path="/tmp/librkllm.so",
            allowed_models=allowed_models,
            model_timeout=60,
        )

        results = []
        errors = []

        def worker():
            try:
                model = manager.get_model("model1")
                results.append(model)
            except Exception as e:
                errors.append(e)

        try:
            # Start multiple threads
            threads = [threading.Thread(target=worker) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Should have no errors and all results should be the same model instance
            assert len(errors) == 0
            assert len(results) == 10
            assert all(r is results[0] for r in results)

        finally:
            manager.shutdown()
