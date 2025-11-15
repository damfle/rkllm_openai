"""
Tool utilities for parsing and handling function calls.
"""

import json
import re
import uuid
from typing import List


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
