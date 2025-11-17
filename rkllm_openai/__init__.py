"""
RKLLM OpenAI-compatible API server.

A Flask-based server that provides OpenAI-compatible endpoints for RKLLM models.
"""

__version__ = "0.1.0"
__author__ = "RKLLM OpenAI"
__email__ = "rkllm@example.com"

from . import bindings
from .server import create_app, main

__all__ = ["create_app", "main", "bindings"]
