"""
Prompt utilities for handling system prompts and formatting.
"""

from typing import List

from .models import ChatMessage


def extract_system_prompt(messages: List[ChatMessage]) -> str:
    """Extract system prompt from messages."""
    for message in messages:
        if message.role == "system":
            return message.content or "You are a helpful assistant."
    return "You are a helpful assistant."
