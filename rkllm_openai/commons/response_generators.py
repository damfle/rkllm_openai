"""
Response generators for handling completion and streaming responses.
"""

import json
import threading
import time
import uuid
from typing import TYPE_CHECKING

from flask import jsonify

from .models import ChatCompletionRequest, CompletionRequest
from .tool_utils import (
    clean_content_for_tools,
    get_forced_tool_name,
    parse_tool_calls,
    should_force_tool_use,
)

if TYPE_CHECKING:
    from ..bindings import RKLLM


def _should_flush_token_buffer(char: str) -> bool:
    """
    Check if a character should trigger flushing the token buffer.

    Args:
        char: Character to check

    Returns:
        True if this character should trigger a flush (whitespace, newline, etc.)
    """
    return char.isspace()


def _process_token_buffer(
    text_buffer: list[str], accumulated_buffer: list[str]
) -> list[str]:
    """
    Process token buffer and return complete words ready to be sent.

    Args:
        text_buffer: New tokens from the model
        accumulated_buffer: Buffer of accumulated tokens waiting for word boundary

    Returns:
        List of complete words ready to be sent
    """
    words_to_send = []

    for token in text_buffer:
        accumulated_buffer.append(token)

        # Check if the token ends with a character that indicates word boundary
        if token and _should_flush_token_buffer(token[-1]):
            # Send the complete accumulated text as one chunk
            complete_text = "".join(accumulated_buffer)
            if complete_text:
                words_to_send.append(complete_text)
            accumulated_buffer.clear()

    return words_to_send


def generate_chat_completion(
    prompt: str, req: ChatCompletionRequest, current_model: "RKLLM"
):
    """Generate non-streaming chat completion."""
    # Clear the model's text buffer
    current_model.clear_text_buffer()

    # Run inference in a separate thread
    def run_inference():
        current_model.generate(prompt)

    thread = threading.Thread(target=run_inference)
    thread.start()

    # Wait for completion
    output = ""
    while thread.is_alive() or len(current_model.get_text_buffer()) > 0:
        text_buffer = current_model.get_text_buffer()
        if text_buffer:
            output += "".join(text_buffer)
            current_model.clear_text_buffer()
        thread.join(timeout=0.01)

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"

    # Parse tool calls from response
    tool_calls = parse_tool_calls(output)
    content = clean_content_for_tools(output).strip() if tool_calls else output.strip()

    # Handle tool choice validation
    if req.tools and should_force_tool_use(req.tool_choice):
        forced_tool = get_forced_tool_name(req.tool_choice)
        if forced_tool and not tool_calls:
            # Model didn't call required tool - this is an error condition
            # We'll still return the response but note the issue
            pass
        elif forced_tool and tool_calls:
            # Verify the correct tool was called
            called_tools = [tc["function"]["name"] for tc in tool_calls]
            if forced_tool not in called_tools:
                # Wrong tool called - this is also an error condition
                pass

    # Build response message
    response_message = {"role": "assistant"}
    if content:
        response_message["content"] = content
    else:
        response_message["content"] = None
    if tool_calls:
        response_message["tool_calls"] = tool_calls

    # Determine finish reason
    finish_reason = "tool_calls" if tool_calls else "stop"
    if req.tool_choice == "none" and tool_calls:
        # Model called tools when explicitly told not to
        finish_reason = "stop"  # Treat as regular completion

    return jsonify(
        {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": response_message,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(output.split()),
                "total_tokens": len(prompt.split()) + len(output.split()),
            },
        }
    )


def stream_chat_completion(
    prompt: str, req: ChatCompletionRequest, current_model: "RKLLM"
):
    """Generate streaming chat completion."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"

    # Clear the model's text buffer
    current_model.clear_text_buffer()

    # Run inference in a separate thread
    def run_inference():
        current_model.generate(prompt)

    thread = threading.Thread(target=run_inference)
    thread.start()

    # Stream responses
    accumulated_content = ""
    tool_calls_sent = False
    token_buffer = []  # Buffer for accumulating tokens until word boundaries

    while thread.is_alive() or len(current_model.get_text_buffer()) > 0:
        text_buffer = current_model.get_text_buffer()

        if text_buffer:
            # Process tokens and get complete words ready to send
            words_to_send = _process_token_buffer(text_buffer, token_buffer)

            for text_chunk in words_to_send:
                accumulated_content += text_chunk

                # Check if we have a complete tool call
                tool_calls = parse_tool_calls(accumulated_content)
                if tool_calls and not tool_calls_sent:
                    # Send tool calls chunk (only once)
                    tool_calls_sent = True
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": req.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"tool_calls": tool_calls},
                                "logprobs": None,
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                else:
                    # Send regular content chunk
                    clean_chunk = clean_content_for_tools(text_chunk)
                    if clean_chunk:
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": req.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "content": clean_chunk,
                                    },
                                    "logprobs": None,
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

            current_model.clear_text_buffer()

        thread.join(timeout=0.01)

    # Send any remaining tokens in the buffer
    if token_buffer:
        remaining_text = "".join(token_buffer)
        if remaining_text:
            accumulated_content += remaining_text
            clean_chunk = clean_content_for_tools(remaining_text)
            if clean_chunk:
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": clean_chunk},
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

    # Send final chunk with finish reason
    final_tool_calls = parse_tool_calls(accumulated_content)
    finish_reason = "tool_calls" if final_tool_calls else "stop"

    # Handle tool choice validation for streaming
    if req.tool_choice == "none" and final_tool_calls:
        finish_reason = "stop"

    final_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {"index": 0, "delta": {}, "logprobs": None, "finish_reason": finish_reason}
        ],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


def generate_completion(prompt: str, req: CompletionRequest, current_model: "RKLLM"):
    """Generate non-streaming completion."""
    # Clear the model's text buffer
    current_model.clear_text_buffer()

    # Run inference in a separate thread
    def run_inference():
        current_model.generate(prompt)

    thread = threading.Thread(target=run_inference)
    thread.start()

    # Wait for completion
    output = ""
    while thread.is_alive() or len(current_model.get_text_buffer()) > 0:
        text_buffer = current_model.get_text_buffer()
        if text_buffer:
            output += "".join(text_buffer)
            current_model.clear_text_buffer()
        thread.join(timeout=0.01)

    completion_id = f"cmpl-{uuid.uuid4().hex}"

    return jsonify(
        {
            "id": completion_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "text": output.strip(),
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(output.split()),
                "total_tokens": len(prompt.split()) + len(output.split()),
            },
        }
    )


def stream_completion(prompt: str, req: CompletionRequest, current_model: "RKLLM"):
    """Generate streaming completion."""
    completion_id = f"cmpl-{uuid.uuid4().hex}"

    # Clear the model's text buffer
    current_model.clear_text_buffer()

    # Run inference in a separate thread
    def run_inference():
        current_model.generate(prompt)

    thread = threading.Thread(target=run_inference)
    thread.start()

    # Stream responses
    token_buffer = []  # Buffer for accumulating tokens until word boundaries

    while thread.is_alive() or len(current_model.get_text_buffer()) > 0:
        text_buffer = current_model.get_text_buffer()

        if text_buffer:
            # Process tokens and get complete words ready to send
            words_to_send = _process_token_buffer(text_buffer, token_buffer)

            for text_chunk in words_to_send:
                chunk = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "text": text_chunk,
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            current_model.clear_text_buffer()

        thread.join(timeout=0.01)

    # Send any remaining tokens in the buffer
    if token_buffer:
        remaining_text = "".join(token_buffer)
        if remaining_text:
            chunk = {
                "id": completion_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [
                    {
                        "index": 0,
                        "text": remaining_text,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

    # Send final chunk
    final_chunk = {
        "id": completion_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {"index": 0, "text": "", "logprobs": None, "finish_reason": "stop"}
        ],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"
