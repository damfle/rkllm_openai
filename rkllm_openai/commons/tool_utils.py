"""
Tool utilities for parsing and handling function calls.
"""

import json
import re
import uuid
from typing import Dict, List, Optional, Union

from pydantic import BaseModel


def _to_dict(obj: Union[dict, BaseModel]) -> dict:
    """Convert Pydantic model or dict to dict."""
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    return obj


def parse_tool_calls(content: str) -> List[dict]:
    """Parse tool calls from model response."""
    tool_calls = []

    # Look for <tool_call>...</tool_call> patterns
    pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    matches = re.findall(pattern, content, re.DOTALL)

    for i, match in enumerate(matches):
        try:
            tool_data = json.loads(match)
            tool_call = {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": tool_data["name"],
                    "arguments": json.dumps(tool_data.get("arguments", {})),
                },
            }
            tool_calls.append(tool_call)
        except json.JSONDecodeError:
            continue

    return tool_calls


def clean_content_for_tools(content: str) -> str:
    """Remove tool call markers from content."""
    # Remove <tool_call>...</tool_call> patterns
    cleaned = re.sub(
        r"<tool_call>\s*\{.*?\}\s*</tool_call>", "", content, flags=re.DOTALL
    )
    return cleaned.strip()


def convert_openai_tools_to_rkllm_format(tools: List[Union[dict, BaseModel]]) -> str:
    """Convert OpenAI tool format to RKLLM-compatible JSON string."""
    if not tools:
        return "[]"

    rkllm_tools = []
    for tool in tools:
        tool_dict = _to_dict(tool)
        if tool_dict.get("type") == "function":
            function = tool_dict.get("function", {})
            function_dict = _to_dict(function)
            # Only include non-empty values
            rkllm_tool = {}
            if function_dict.get("name"):
                rkllm_tool["name"] = function_dict["name"]
            if function_dict.get("description"):
                rkllm_tool["description"] = function_dict["description"]
            if function_dict.get("parameters"):
                rkllm_tool["parameters"] = function_dict["parameters"]

            if rkllm_tool:  # Only add if we have actual content
                rkllm_tools.append(rkllm_tool)

    # Use compact JSON format without spaces
    return json.dumps(rkllm_tools, separators=(",", ":"), ensure_ascii=True)


def format_tools_for_prompt(tools: List[Union[dict, BaseModel]]) -> str:
    """Format tools for inclusion in prompt."""
    if not tools:
        return ""

    tool_descriptions = []
    for tool in tools:
        tool_dict = _to_dict(tool)
        if tool_dict.get("type") == "function":
            function = tool_dict.get("function", {})
            function_dict = _to_dict(function)
            name = function_dict.get("name", "")
            description = function_dict.get("description", "")
            parameters = function_dict.get("parameters", {})

            tool_desc = f"Function: {name}\nDescription: {description}"
            if parameters.get("properties"):
                params = []
                for param_name, param_info in parameters["properties"].items():
                    param_type = param_info.get("type", "string")
                    param_desc = param_info.get("description", "")
                    params.append(f"- {param_name} ({param_type}): {param_desc}")
                tool_desc += f"\nParameters:\n" + "\n".join(params)

            if parameters.get("required"):
                tool_desc += (
                    f"\nRequired parameters: {', '.join(parameters['required'])}"
                )

            tool_descriptions.append(tool_desc)

    return (
        "\n\nAvailable functions:\n"
        + "\n\n".join(tool_descriptions)
        + '\n\nTo call a function, use this format:\n<tool_call>\n{"name": "function_name", "arguments": {"param": "value"}}\n</tool_call>'
    )


def get_system_prompt_with_tools(
    original_system_prompt: str, tools: List[Union[dict, BaseModel]]
) -> str:
    """Combine system prompt with tool instructions."""
    if not tools:
        return original_system_prompt

    tool_instructions = format_tools_for_prompt(tools)

    if original_system_prompt:
        return f"{original_system_prompt}\n\n{tool_instructions}"
    else:
        return f"You are a helpful assistant.{tool_instructions}"


def should_force_tool_use(tool_choice: Optional[Union[str, dict]]) -> bool:
    """Check if tool use should be forced based on tool_choice."""
    if tool_choice is None or tool_choice == "auto":
        return False
    elif tool_choice == "none":
        return False
    elif tool_choice == "required":
        return True
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        return True
    return False


def get_forced_tool_name(tool_choice: Optional[Union[str, dict]]) -> Optional[str]:
    """Get the specific tool name if tool use is forced to a specific function."""
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        function = tool_choice.get("function", {})
        return function.get("name")
    return None
