#!/usr/bin/env python3
"""
Model Manager for RKLLM OpenAI server.

Simple wrapper around a single RKLLM model.
"""

import os
from pathlib import Path
from typing import Optional, Set

from .bindings import RKLLM


class ModelManager:
    """Simple model manager for a single RKLLM model."""

    def __init__(
        self,
        model_path: str,
        platform: str,
        lib_path: Optional[str] = None,
        model_timeout: int = 300,  # Unused but kept for compatibility
        chat_template: Optional[str] = None,
    ):
        """
        Initialize the model manager.

        Args:
            model_path: Path to the RKLLM model file
            platform: Platform identifier (rk3588, rk3576)
            lib_path: Path to RKLLM library (optional, will search common paths)
            model_timeout: Unused but kept for compatibility
            chat_template: Optional path to chat template file
        """
        self.model_path = Path(model_path)
        self.model_name = self.model_path.stem
        self.platform = platform
        self.lib_path = lib_path
        self.chat_template = chat_template
        self._model: Optional[RKLLM] = None

        # Validate model file exists
        if not self.model_path.exists():
            raise ValueError(f"Model file does not exist: {self.model_path}")

        print(f"Model configured: {self.model_name} -> {self.model_path}")

    def _load_model(self) -> RKLLM:
        """Load the model if not already loaded."""
        if self._model is not None:
            return self._model

        print(f"Loading model: {self.model_name}")
        try:
            self._model = RKLLM(str(self.model_path), self.platform, self.lib_path)

            # Apply chat template if configured
            if self.chat_template and os.path.exists(self.chat_template):
                self._model.load_chat_template_from_file(self.chat_template)
                print(f"Applied chat template to model: {self.model_name}")

            print(f"Successfully loaded model: {self.model_name}")
            return self._model
        except Exception as e:
            print(f"Failed to load model {self.model_name}: {e}")
            raise

    def get_model(self, model_name: str) -> RKLLM:
        """
        Get the model, loading it if necessary.

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

        return self._load_model()

    def is_model_loaded(self, model_name: str) -> bool:
        """Check if the model is currently loaded."""
        return model_name == self.model_name and self._model is not None

    def get_loaded_models(self) -> Set[str]:
        """Get the set of currently loaded model names."""
        return {self.model_name} if self._model is not None else set()

    def get_available_models(self) -> Set[str]:
        """Get the set of available model names."""
        return {self.model_name}

    def preload_model(self, model_name: str):
        """Preload the model without waiting for a request."""
        if model_name != self.model_name:
            raise ValueError(
                f"Model {model_name} not available. Only {self.model_name} is configured"
            )
        self._load_model()

    def shutdown(self):
        """Shutdown the model manager and cleanup resources."""
        print("Shutting down model manager...")
        if self._model is not None:
            print(f"Unloading model: {self.model_name}")
            try:
                self._model.release()
                self._model = None
                print(f"Successfully unloaded model: {self.model_name}")
            except Exception as e:
                print(f"Error unloading model {self.model_name}: {e}")
        print("Model manager shutdown complete")
