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
from .response_generators import (
    generate_chat_completion,
    generate_completion,
    stream_chat_completion,
    stream_completion,
)
from .tool_utils import (
    clean_content_for_tools,
    convert_openai_tools_to_rkllm_format,
    format_tools_for_prompt,
    get_forced_tool_name,
    get_system_prompt_with_tools,
    parse_tool_calls,
    should_force_tool_use,
)

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
    "convert_openai_tools_to_rkllm_format",
    "format_tools_for_prompt",
    "get_system_prompt_with_tools",
    "should_force_tool_use",
    "get_forced_tool_name",
]

__all__.extend(
    [
        "generate_chat_completion",
        "stream_chat_completion",
        "generate_completion",
        "stream_completion",
    ]
)
