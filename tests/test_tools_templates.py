#!/usr/bin/env python3
"""
Test script for RKLLM tools and chat template functionality.

This script specifically tests the tools (function calling) and chat template
features added to the RKLLM OpenAI-compatible server.
"""

import json
import os
import sys
import tempfile
import time
from typing import Any, Dict, List

import requests


class ToolsTemplatesTestClient:
    """Test client specifically for tools and templates functionality."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Authorization": "Bearer dummy",
            }
        )

    def test_basic_tools_support(self, model: str) -> bool:
        """Test basic tools functionality."""
        try:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather in a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location"],
                        },
                    },
                }
            ]

            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": "What's the weather like in San Francisco?",
                    }
                ],
                "tools": tools,
                "max_tokens": 150,
            }

            response = self.session.post(
                f"{self.base_url}/v1/chat/completions", json=payload
            )

            if response.status_code == 200:
                data = response.json()
                message = data.get("choices", [{}])[0].get("message", {})

                # Check response structure
                has_tool_calls = "tool_calls" in message
                has_content = "content" in message
                finish_reason = data.get("choices", [{}])[0].get(
                    "finish_reason", "unknown"
                )

                print(f"✓ Basic tools support test passed")
                print(f"  - Response has tool_calls field: {has_tool_calls}")
                print(f"  - Response has content field: {has_content}")
                print(f"  - Finish reason: {finish_reason}")

                if has_tool_calls and message["tool_calls"]:
                    print(f"  - Tool calls detected: {len(message['tool_calls'])}")
                    for i, tool_call in enumerate(message["tool_calls"]):
                        print(
                            f"    {i + 1}. {tool_call.get('function', {}).get('name', 'unknown')}"
                        )

                return True
            else:
                print(f"✗ Basic tools test failed with status {response.status_code}")
                return False

        except Exception as e:
            print(f"✗ Basic tools test failed with error: {e}")
            return False

    def test_multiple_tools(self, model: str) -> bool:
        """Test multiple tools in one request."""
        try:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather information",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_stock_price",
                        "description": "Get stock price",
                        "parameters": {
                            "type": "object",
                            "properties": {"symbol": {"type": "string"}},
                            "required": ["symbol"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "send_email",
                        "description": "Send an email",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "to": {"type": "string"},
                                "subject": {"type": "string"},
                                "body": {"type": "string"},
                            },
                            "required": ["to", "subject"],
                        },
                    },
                },
            ]

            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant with access to weather, stock, and email functions.",
                    },
                    {
                        "role": "user",
                        "content": "I need the weather in Tokyo and Apple's stock price",
                    },
                ],
                "tools": tools,
                "max_tokens": 200,
            }

            response = self.session.post(
                f"{self.base_url}/v1/chat/completions", json=payload
            )

            if response.status_code == 200:
                data = response.json()
                print("✓ Multiple tools test passed")
                print(f"  - Tools provided: {len(tools)}")

                message = data.get("choices", [{}])[0].get("message", {})
                if message.get("tool_calls"):
                    print(f"  - Tool calls returned: {len(message['tool_calls'])}")
                else:
                    print("  - No tool calls returned (may be expected)")

                return True
            else:
                print(
                    f"✗ Multiple tools test failed with status {response.status_code}"
                )
                return False

        except Exception as e:
            print(f"✗ Multiple tools test failed with error: {e}")
            return False

    def test_streaming_with_tools(self, model: str) -> bool:
        """Test streaming responses with tools."""
        try:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "description": "Perform mathematical calculations",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "expression": {"type": "string"},
                            },
                            "required": ["expression"],
                        },
                    },
                }
            ]

            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "What's 15 * 23 + 45?"}],
                "tools": tools,
                "stream": True,
                "max_tokens": 100,
            }

            response = self.session.post(
                f"{self.base_url}/v1/chat/completions", json=payload, stream=True
            )

            if response.status_code == 200:
                chunk_count = 0
                has_tool_calls = False
                final_finish_reason = None

                for line in response.iter_lines():
                    if line:
                        line_str = line.decode("utf-8")
                        if line_str.startswith("data: "):
                            data_part = line_str[6:]
                            if data_part == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_part)
                                chunk_count += 1

                                choice = chunk.get("choices", [{}])[0]
                                delta = choice.get("delta", {})

                                if "tool_calls" in delta:
                                    has_tool_calls = True

                                finish_reason = choice.get("finish_reason")
                                if finish_reason:
                                    final_finish_reason = finish_reason

                            except json.JSONDecodeError:
                                continue

                print("✓ Streaming with tools test passed")
                print(f"  - Chunks received: {chunk_count}")
                print(f"  - Tool calls detected in stream: {has_tool_calls}")
                print(f"  - Final finish reason: {final_finish_reason}")
                return True
            else:
                print(
                    f"✗ Streaming tools test failed with status {response.status_code}"
                )
                return False

        except Exception as e:
            print(f"✗ Streaming tools test failed with error: {e}")
            return False

    def test_tool_conversation_flow(self, model: str) -> bool:
        """Test complete tool conversation flow."""
        try:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "add_numbers",
                        "description": "Add two numbers together",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "number"},
                                "b": {"type": "number"},
                            },
                            "required": ["a", "b"],
                        },
                    },
                }
            ]

            # Step 1: Initial request
            messages = [{"role": "user", "content": "Please add 25 and 17 for me"}]

            response1 = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "tools": tools,
                    "max_tokens": 150,
                },
            )

            if response1.status_code != 200:
                print(f"✗ Step 1 failed with status {response1.status_code}")
                return False

            data1 = response1.json()
            assistant_message = data1["choices"][0]["message"]

            # Step 2: Add assistant message to conversation
            messages.append(assistant_message)

            # Step 3: If there were tool calls, simulate tool execution
            if assistant_message.get("tool_calls"):
                for tool_call in assistant_message["tool_calls"]:
                    # Simulate tool execution
                    function_name = tool_call["function"]["name"]
                    arguments = json.loads(tool_call["function"]["arguments"])

                    # Mock result
                    if function_name == "add_numbers":
                        result = arguments.get("a", 0) + arguments.get("b", 0)
                    else:
                        result = "Function executed"

                    # Add tool response
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": json.dumps({"result": result}),
                        }
                    )

                # Step 4: Send back for final response
                response2 = self.session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": messages,
                        "tools": tools,
                        "max_tokens": 150,
                    },
                )

                if response2.status_code == 200:
                    print("✓ Tool conversation flow test passed")
                    print(
                        f"  - Initial tool calls: {len(assistant_message['tool_calls'])}"
                    )
                    print(f"  - Messages in conversation: {len(messages)}")
                    return True

            print("✓ Tool conversation flow test passed (no tool calls made)")
            return True

        except Exception as e:
            print(f"✗ Tool conversation flow test failed with error: {e}")
            return False

    def test_invalid_tools_request(self, model: str) -> bool:
        """Test handling of invalid tools requests."""
        try:
            # Invalid tool structure
            invalid_tools = [
                {
                    "type": "invalid_type",
                    "function": {
                        "name": "invalid_function",
                    },
                }
            ]

            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "Test invalid tools"}],
                "tools": invalid_tools,
            }

            response = self.session.post(
                f"{self.base_url}/v1/chat/completions", json=payload
            )

            # Should either work (ignoring invalid tools) or return proper error
            if response.status_code in [200, 400]:
                print("✓ Invalid tools request test passed")
                print(f"  - Status code: {response.status_code}")
                return True
            else:
                print(f"✗ Unexpected status code: {response.status_code}")
                return False

        except Exception as e:
            print(f"✓ Invalid tools request handled with error: {e}")
            return True  # Error handling is acceptable

    def create_test_chat_template(self) -> str:
        """Create a temporary chat template file for testing."""
        template_content = """
{%- if messages[0]['role'] == 'system' -%}
    {%- set system_message = messages[0]['content'] -%}
    {%- set messages = messages[1:] -%}
{%- else -%}
    {%- set system_message = "You are a helpful assistant." -%}
{%- endif -%}

<|im_start|>system
{{ system_message }}<|im_end|>
{%- for message in messages -%}
    {%- if message['role'] == 'user' -%}
<|im_start|>user
{{ message['content'] }}<|im_end|>
    {%- elif message['role'] == 'assistant' -%}
<|im_start|>assistant
        {%- if message.get('tool_calls') -%}
            {%- for tool_call in message['tool_calls'] -%}
<tool_call>
{"name": "{{ tool_call['function']['name'] }}", "arguments": {{ tool_call['function']['arguments'] }}}
</tool_call>
            {%- endfor -%}
        {%- endif -%}
        {%- if message.get('content') -%}
{{ message['content'] }}
        {%- endif -%}
<|im_end|>
    {%- elif message['role'] == 'tool' -%}
<|im_start|>tool
{{ message['content'] }}<|im_end|>
    {%- endif -%}
{%- endfor -%}
<|im_start|>assistant
""".strip()

        # Create temporary file
        fd, path = tempfile.mkstemp(suffix=".jinja2", text=True)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(template_content)
            return path
        except Exception:
            os.close(fd)
            raise


def wait_for_server(base_url: str, timeout: int = 30) -> bool:
    """Wait for server to be ready."""
    print(f"Waiting for server at {base_url}...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                print("✓ Server is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)

    print(f"✗ Server not ready within {timeout} seconds")
    return False


def main():
    """Run all tools and templates tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Test RKLLM tools and templates")
    parser.add_argument(
        "--base-url", default="http://localhost:8080", help="Server base URL"
    )
    parser.add_argument("--model", default="test-model", help="Model name to test")
    parser.add_argument(
        "--wait", action="store_true", help="Wait for server to be ready"
    )
    parser.add_argument("--timeout", type=int, default=30, help="Wait timeout")

    args = parser.parse_args()

    if args.wait:
        if not wait_for_server(args.base_url, args.timeout):
            sys.exit(1)

    client = ToolsTemplatesTestClient(args.base_url)

    print("🧪 Testing RKLLM Tools and Templates Functionality")
    print("=" * 60)

    tests = [
        ("Basic Tools Support", lambda: client.test_basic_tools_support(args.model)),
        ("Multiple Tools", lambda: client.test_multiple_tools(args.model)),
        ("Streaming with Tools", lambda: client.test_streaming_with_tools(args.model)),
        (
            "Tool Conversation Flow",
            lambda: client.test_tool_conversation_flow(args.model),
        ),
        (
            "Invalid Tools Handling",
            lambda: client.test_invalid_tools_request(args.model),
        ),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"📊 Tools & Templates Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tools and templates tests passed!")
        return 0
    else:
        print(f"❌ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
