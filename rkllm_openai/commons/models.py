"""
Pydantic models for API validation.
"""

from typing import List, Optional, Union

from pydantic import BaseModel


class ToolFunction(BaseModel):
    name: str
    description: str
    parameters: dict


class Tool(BaseModel):
    type: str = "function"
    function: ToolFunction


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: dict


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 4096
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.9
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, dict]] = None


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: Optional[int] = 4096
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.9
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    encoding_format: Optional[str] = "float"
