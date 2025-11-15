#!/usr/bin/env python3
"""
Model Manager for RKLLM OpenAI server.

Handles lazy loading, unloading, and lifecycle management of multiple models.
"""

import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Set

from .bindings import RKLLM


class ModelManager:
    """Manages multiple RKLLM models with lazy loading and automatic unloading."""

    def __init__(
        self,
        model_dir: str,
        platform: str,
        lib_path: str,
        allowed_models: Set[str],
        model_timeout: int = 300,  # 5 minutes default
    ):
        """
        Initialize the model manager.

        Args:
            model_dir: Directory containing model files
            platform: Platform identifier (rk3588, rk3576)
            lib_path: Path to RKLLM library
            allowed_models: Set of allowed model names
            model_timeout: Time in seconds before unloading inactive models
        """
        self.model_dir = Path(model_dir)
        self.platform = platform
        self.lib_path = lib_path
        self.allowed_models = allowed_models
        self.model_timeout = model_timeout

        # Model storage and metadata
        self._models: Dict[str, RKLLM] = {}
        self._model_paths: Dict[str, str] = {}
        self._last_used: Dict[str, float] = {}
        self._lock = threading.RLock()

        # Cleanup thread
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()

        # Discover available models
        self._discover_models()

        # Start cleanup thread
        self._start_cleanup_thread()

    def _discover_models(self):
        """Discover model files in the model directory."""
        if not self.model_dir.exists():
            raise ValueError(f"Model directory does not exist: {self.model_dir}")

        for model_name in self.allowed_models:
            # Look for model file with common extensions
            for ext in [".rkllm", ".bin", ""]:
                model_path = self.model_dir / f"{model_name}{ext}"
                if model_path.exists():
                    self._model_paths[model_name] = str(model_path)
                    break
            else:
                # Check if it's a directory with model files
                model_subdir = self.model_dir / model_name
                if model_subdir.is_dir():
                    # Look for model files in subdirectory
                    for ext in [".rkllm", ".bin"]:
                        for model_file in model_subdir.glob(f"*{ext}"):
                            self._model_paths[model_name] = str(model_file)
                            break
                        if model_name in self._model_paths:
                            break

        print(f"Discovered models: {list(self._model_paths.keys())}")

    def _start_cleanup_thread(self):
        """Start the background cleanup thread."""
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_worker, daemon=True
        )
        self._cleanup_thread.start()

    def _cleanup_worker(self):
        """Background worker to unload inactive models."""
        while not self._stop_cleanup.wait(30):  # Check every 30 seconds
            with self._lock:
                current_time = time.time()
                models_to_unload = []

                for model_name, last_used in self._last_used.items():
                    if current_time - last_used > self.model_timeout:
                        models_to_unload.append(model_name)

                for model_name in models_to_unload:
                    self._unload_model(model_name)

    def _load_model(self, model_name: str) -> RKLLM:
        """Load a model if not already loaded."""
        if model_name in self._models:
            return self._models[model_name]

        if model_name not in self._model_paths:
            raise ValueError(f"Model {model_name} not found in model directory")

        print(f"Loading model: {model_name}")
        try:
            model = RKLLM(self._model_paths[model_name], self.platform, self.lib_path)
            self._models[model_name] = model
            print(f"Successfully loaded model: {model_name}")
            return model
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            raise

    def _unload_model(self, model_name: str):
        """Unload a model from memory."""
        if model_name in self._models:
            print(f"Unloading model: {model_name}")
            try:
                self._models[model_name].release()
                del self._models[model_name]
                if model_name in self._last_used:
                    del self._last_used[model_name]
                print(f"Successfully unloaded model: {model_name}")
            except Exception as e:
                print(f"Error unloading model {model_name}: {e}")

    def get_model(self, model_name: str) -> RKLLM:
        """
        Get a model, loading it if necessary.

        Args:
            model_name: Name of the model to get

        Returns:
            RKLLM instance

        Raises:
            ValueError: If model is not in allowlist or not found
        """
        if model_name not in self.allowed_models:
            raise ValueError(f"Model {model_name} not in allowlist")

        with self._lock:
            # Check if we need to unload other models first
            if model_name not in self._models and self._models:
                # Unload all other models to free memory
                other_models = list(self._models.keys())
                for other_model in other_models:
                    if other_model != model_name:
                        self._unload_model(other_model)

            # Load the requested model
            model = self._load_model(model_name)
            self._last_used[model_name] = time.time()
            return model

    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is currently loaded."""
        with self._lock:
            return model_name in self._models

    def get_loaded_models(self) -> Set[str]:
        """Get the set of currently loaded model names."""
        with self._lock:
            return set(self._models.keys())

    def get_available_models(self) -> Set[str]:
        """Get the set of available model names."""
        return set(self._model_paths.keys())

    def preload_model(self, model_name: str):
        """Preload a model without waiting for a request."""
        with self._lock:
            self._load_model(model_name)
            self._last_used[model_name] = time.time()

    def unload_all_models(self):
        """Unload all currently loaded models."""
        with self._lock:
            models_to_unload = list(self._models.keys())
            for model_name in models_to_unload:
                self._unload_model(model_name)

    def shutdown(self):
        """Shutdown the model manager and cleanup resources."""
        print("Shutting down model manager...")

        # Stop cleanup thread
        if self._cleanup_thread:
            self._stop_cleanup.set()
            self._cleanup_thread.join(timeout=5)

        # Unload all models
        self.unload_all_models()
        print("Model manager shutdown complete")

    def get_model_info(self) -> Dict[str, Dict]:
        """Get information about all models."""
        with self._lock:
            info = {}
            for model_name in self.allowed_models:
                info[model_name] = {
                    "available": model_name in self._model_paths,
                    "loaded": model_name in self._models,
                    "path": self._model_paths.get(model_name),
                    "last_used": self._last_used.get(model_name),
                }
            return info

    def extend_model_timeout(self, model_name: str):
        """Extend the timeout for a specific model."""
        with self._lock:
            if model_name in self._models:
                self._last_used[model_name] = time.time()

    def set_model_timeout(self, timeout: int):
        """Set the global model timeout."""
        self.model_timeout = max(60, timeout)  # Minimum 1 minute

    def load_chat_template(self, model_name: str, template_path: str):
        """Load a chat template for a specific model."""
        with self._lock:
            if model_name in self._models:
                model = self._models[model_name]
                if os.path.exists(template_path):
                    model.load_chat_template_from_file(template_path)
                    print(f"Loaded chat template for {model_name} from {template_path}")
                else:
                    print(f"Warning: Chat template file not found: {template_path}")
            else:
                print(f"Model {model_name} not loaded, cannot set chat template")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
