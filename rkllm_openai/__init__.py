"""
RKLLM OpenAI-compatible API server.

A Flask-based server that provides OpenAI-compatible endpoints for RKLLM models.
"""

__version__ = "0.1.0"
__author__ = "RKLLM OpenAI"
__email__ = "rkllm@example.com"

from . import bindings


def create_app(*args, **kwargs):
    """Lazy import of create_app to avoid Flask dependency issues."""
    from .server import create_app as _create_app

    return _create_app(*args, **kwargs)


def main():
    """Lazy import of main to avoid Flask dependency issues."""
    from .server import main as _main

    return _main()


__all__ = ["create_app", "main", "bindings"]
