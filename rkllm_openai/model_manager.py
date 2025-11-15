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
        model_path: str,
        platform: str,
        lib_path: str,
        model_timeout: int = 300,  # 5 minutes default
        chat_template: Optional[str] = None,
    ):
        """
        Initialize the model manager.

        Args:
            model_path: Path to the RKLLM model file
            platform: Platform identifier (rk3588, rk3576)
            lib_path: Path to RKLLM library
            model_timeout: Time in seconds before unloading inactive models
            chat_template: Optional path to chat template file
        """
        self.model_path = Path(model_path)
        self.model_name = self.model_path.stem  # Extract filename without extension
        self.platform = platform
        self.lib_path = lib_path
        self.model_timeout = model_timeout
        self.chat_template = chat_template

        # Model storage and metadata
        self._models: Dict[str, RKLLM] = {}
        self._last_used: Dict[str, float] = {}
        self._lock = threading.RLock()

        # Cleanup thread
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()

        # Validate model file exists
        if not self.model_path.exists():
            raise ValueError(f"Model file does not exist: {self.model_path}")

        print(f"Model configured: {self.model_name} -> {self.model_path}")

        # Start cleanup thread
        self._start_cleanup_thread()

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

        if model_name != self.model_name:
            raise ValueError(
                f"Model {model_name} not available. Only {self.model_name} is configured"
            )

        print(f"Loading model: {model_name}")
        try:
            model = RKLLM(str(self.model_path), self.platform, self.lib_path)
            self._models[model_name] = model

            # Apply chat template if configured
            if self.chat_template and os.path.exists(self.chat_template):
                model.load_chat_template_from_file(self.chat_template)
                print(f"Applied chat template to model: {model_name}")

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
            ValueError: If model is not the configured model
        """
        if model_name != self.model_name:
            raise ValueError(
                f"Model {model_name} not available. Only {self.model_name} is configured"
            )

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
        return {self.model_name}

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
            info = {
                self.model_name: {
                    "available": True,
                    "loaded": self.model_name in self._models,
                    "path": str(self.model_path),
                    "last_used": self._last_used.get(self.model_name),
                }
            }
            return info

    def extend_model_timeout(self, model_name: str):
        """Extend the timeout for a specific model."""
        if model_name != self.model_name:
            raise ValueError(
                f"Model {model_name} not available. Only {self.model_name} is configured"
            )

        with self._lock:
            if model_name in self._models:
                self._last_used[model_name] = time.time()

    def load_chat_template(self, model_name: str, template_path: str):
        """Load a chat template for a specific model."""
        if model_name != self.model_name:
            raise ValueError(
                f"Model {model_name} not available. Only {self.model_name} is configured"
            )

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
