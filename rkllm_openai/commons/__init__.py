"""
Commons package for shared utilities and models.
"""

# Always import models and utilities that don't depend on Flask
from .models import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    EmbeddingRequest,
    Tool,
    ToolCall,
    ToolFunction,
)
from .tool_utils import clean_content_for_tools, parse_tool_calls

# Conditionally import response generators to avoid Flask dependency issues
try:
    from .response_generators import (
        generate_chat_completion,
        generate_completion,
        stream_chat_completion,
        stream_completion,
    )

    _response_generators_available = True
except ImportError:
    _response_generators_available = False

__all__ = [
    # Models
    "ChatCompletionRequest",
    "ChatMessage",
    "CompletionRequest",
    "EmbeddingRequest",
    "Tool",
    "ToolCall",
    "ToolFunction",
    # Tool utilities
    "parse_tool_calls",
    "clean_content_for_tools",
]

# Add response generators to __all__ if available
if _response_generators_available:
    __all__.extend(
        [
            "generate_chat_completion",
            "stream_chat_completion",
            "generate_completion",
            "stream_completion",
        ]
    )
