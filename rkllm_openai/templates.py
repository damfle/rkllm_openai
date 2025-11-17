"""
Template utilities for rkllm_openai package.

This module provides utilities for locating and loading chat templates
from various possible locations (development, package, docker, etc.).
"""

import os
from pathlib import Path
from typing import Dict, List, Optional


def get_templates_directory() -> Optional[str]:
    """
    Get the templates directory, trying multiple possible locations.

    Returns:
        Path to templates directory if found, None otherwise
    """
    # Try multiple possible locations for templates
    possible_dirs = [
        # Development environment
        Path(__file__).parent.parent / "assets",
        # Package installation
        Path(__file__).parent / "assets",
        # Docker/runtime environment
        Path.cwd() / "assets",
        # Alternative package location
        Path(__file__).parent.parent.parent / "assets",
    ]

    for assets_dir in possible_dirs:
        if assets_dir.exists() and assets_dir.is_dir():
            # Check if it contains template files
            if any(assets_dir.glob("*.jinja2")):
                return str(assets_dir)

    return None


def get_template_path(template_name: str) -> Optional[str]:
    """
    Get the full path to a specific template file.

    Args:
        template_name: Name of the template file (with or without .jinja2 extension)

    Returns:
        Full path to template file if found, None otherwise
    """
    if not template_name.endswith(".jinja2"):
        template_name += ".jinja2"

    templates_dir = get_templates_directory()
    if not templates_dir:
        return None

    template_path = Path(templates_dir) / template_name
    if template_path.exists():
        return str(template_path)

    return None


def list_available_templates() -> List[str]:
    """
    List all available template files.

    Returns:
        List of template filenames (without .jinja2 extension)
    """
    templates_dir = get_templates_directory()
    if not templates_dir:
        return []

    templates = []
    for template_file in Path(templates_dir).glob("*.jinja2"):
        # Remove .jinja2 extension for display
        templates.append(template_file.stem)

    return sorted(templates)


def get_default_template_path() -> Optional[str]:
    """
    Get the path to the default chat template.

    Returns:
        Path to default template if found, None otherwise
    """
    return get_template_path("chat_template")


def get_template_info() -> Dict[str, str]:
    """
    Get information about available templates.

    Returns:
        Dictionary mapping template names to their descriptions
    """
    templates_info = {
        "chat_template": "Generic ChatML template (default)",
        "gemma-3-it": "Optimized for Gemma 3 instruction-tuned models",
        "llama-3.2-instruct": "Optimized for Llama 3.2 instruct models",
        "phi-3.5-mini-instruct": "Optimized for Phi 3.5 mini instruct models",
        "qwen3-thinking": "Optimized for Qwen models with thinking support",
    }

    available_templates = list_available_templates()
    return {
        name: desc
        for name, desc in templates_info.items()
        if name in available_templates
    }


def load_template_content(template_name: str) -> Optional[str]:
    """
    Load the content of a template file.

    Args:
        template_name: Name of the template file (with or without .jinja2 extension)

    Returns:
        Template content as string if found, None otherwise
    """
    template_path = get_template_path(template_name)
    if not template_path:
        return None

    try:
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    except (IOError, UnicodeDecodeError):
        return None
