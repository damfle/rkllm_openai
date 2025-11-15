#!/usr/bin/env python3
"""
Test script for RKLLM OpenAI-compatible API server.

This script tests all the endpoints to ensure they work correctly.
"""

import json
import sys
import time
from typing import Any, Dict

import requests


class TestClient:
    """Test client for the RKLLM OpenAI server."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Authorization": "Bearer dummy",  # Not required but for compatibility
            }
        )

    def test_health(self) -> bool:
        """Test the health endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Health check passed: {data.get('message', 'OK')}")
                return True
            else:
                print(f"✗ Health check failed with status {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Health check failed with error: {e}")
            return False

    def test_list_models(self) -> bool:
        """Test the list models endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/v1/models")
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                print(f"✓ List models passed: Found {len(models)} models")
                for model in models:
                    print(f"  - {model.get('id', 'unknown')}")
                return True
            else:
                print(f"✗ List models failed with status {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ List models failed with error: {e}")
            return False

    def test_get_model(self, model_id: str) -> bool:
        """Test getting a specific model."""
        try:
            response = self.session.get(f"{self.base_url}/v1/models/{model_id}")
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Get model '{model_id}' passed: {data.get('id')}")
                return True
            elif response.status_code == 404:
                print(f"✗ Model '{model_id}' not found (expected if not in allowlist)")
                return False
            else:
                print(f"✗ Get model failed with status {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Get model failed with error: {e}")
            return False

    def test_chat_completion(self, model: str, stream: bool = False) -> bool:
        """Test chat completion endpoint."""
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
                "max_tokens": 100,
                "temperature": 0.7,
                "stream": stream,
            }

            if stream:
                response = self.session.post(
                    f"{self.base_url}/v1/chat/completions", json=payload, stream=True
                )
                if response.status_code == 200:
                    print(f"✓ Streaming chat completion started successfully")
                    chunk_count = 0
                    for line in response.iter_lines():
                        if line:
                            line_str = line.decode("utf-8")
                            if line_str.startswith("data: "):
                                data_part = line_str[6:]
                                if data_part == "[DONE]":
                                    break
                                try:
                                    json.loads(data_part)
                                    chunk_count += 1
                                except json.JSONDecodeError:
                                    pass
                    print(f"✓ Received {chunk_count} chunks")
                    return True
                else:
                    print(
                        f"✗ Streaming chat completion failed with status {response.status_code}"
                    )
                    return False
            else:
                response = self.session.post(
                    f"{self.base_url}/v1/chat/completions", json=payload
                )
                if response.status_code == 200:
                    data = response.json()
                    content = (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    print(f"✓ Chat completion passed: '{content[:50]}...'")
                    return True
                else:
                    print(
                        f"✗ Chat completion failed with status {response.status_code}"
                    )
                    if response.text:
                        print(f"  Error: {response.text}")
                    return False
        except Exception as e:
            print(f"✗ Chat completion failed with error: {e}")
            return False

    def test_chat_completion_with_tools(self, model: str) -> bool:
        """Test chat completion with tools."""
        try:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather in a location",
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

                # Check if we got tool calls or regular content
                has_tool_calls = "tool_calls" in message and message["tool_calls"]
                has_content = "content" in message and message["content"]

                print(f"✓ Chat completion with tools passed")
                if has_tool_calls:
                    print(f"  - Tool calls detected: {len(message['tool_calls'])}")
                if has_content:
                    print(f"  - Content: {message['content'][:50]}...")

                return True
            else:
                print(
                    f"✗ Chat completion with tools failed with status {response.status_code}"
                )
                if response.text:
                    print(f"  Error: {response.text}")
                return False
        except Exception as e:
            print(f"✗ Chat completion with tools failed with error: {e}")
            return False

    def test_completion(self, model: str, stream: bool = False) -> bool:
        """Test completion endpoint."""
        try:
            payload = {
                "model": model,
                "prompt": "The capital of France is",
                "max_tokens": 50,
                "temperature": 0.7,
                "stream": stream,
            }

            if stream:
                response = self.session.post(
                    f"{self.base_url}/v1/completions", json=payload, stream=True
                )
                if response.status_code == 200:
                    print(f"✓ Streaming completion started successfully")
                    chunk_count = 0
                    for line in response.iter_lines():
                        if line:
                            line_str = line.decode("utf-8")
                            if line_str.startswith("data: "):
                                data_part = line_str[6:]
                                if data_part == "[DONE]":
                                    break
                                try:
                                    json.loads(data_part)
                                    chunk_count += 1
                                except json.JSONDecodeError:
                                    pass
                    print(f"✓ Received {chunk_count} chunks")
                    return True
                else:
                    print(
                        f"✗ Streaming completion failed with status {response.status_code}"
                    )
                    return False
            else:
                response = self.session.post(
                    f"{self.base_url}/v1/completions", json=payload
                )
                if response.status_code == 200:
                    data = response.json()
                    text = data.get("choices", [{}])[0].get("text", "")
                    print(f"✓ Completion passed: '{text[:50]}...'")
                    return True
                else:
                    print(f"✗ Completion failed with status {response.status_code}")
                    if response.text:
                        print(f"  Error: {response.text}")
                    return False
        except Exception as e:
            print(f"✗ Completion failed with error: {e}")
            return False

    def test_embeddings(self, model: str) -> bool:
        """Test embeddings endpoint."""
        try:
            payload = {"model": model, "input": ["Hello world", "How are you?"]}

            response = self.session.post(f"{self.base_url}/v1/embeddings", json=payload)
            if response.status_code == 200:
                data = response.json()
                embeddings = data.get("data", [])
                print(f"✓ Embeddings passed: Generated {len(embeddings)} embeddings")
                if embeddings:
                    embedding_size = len(embeddings[0].get("embedding", []))
                    print(f"  Embedding dimension: {embedding_size}")
                return True
            else:
                print(f"✗ Embeddings failed with status {response.status_code}")
                if response.text:
                    print(f"  Error: {response.text}")
                return False
        except Exception as e:
            print(f"✗ Embeddings failed with error: {e}")
            return False


def wait_for_server(base_url: str, timeout: int = 30) -> bool:
    """Wait for server to be ready."""
    print(f"Waiting for server at {base_url} to be ready...")
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

    print(f"✗ Server did not become ready within {timeout} seconds")
    return False


def main():
    """Run all tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Test RKLLM OpenAI-compatible server")
    parser.add_argument(
        "--base-url", default="http://localhost:8080", help="Server base URL"
    )
    parser.add_argument("--model", default="test-model", help="Model name to test with")
    parser.add_argument(
        "--wait", action="store_true", help="Wait for server to be ready"
    )
    parser.add_argument("--timeout", type=int, default=30, help="Timeout for waiting")

    args = parser.parse_args()

    if args.wait:
        if not wait_for_server(args.base_url, args.timeout):
            sys.exit(1)

    client = TestClient(args.base_url)

    print(f"\n🧪 Testing RKLLM OpenAI server at {args.base_url}")
    print("=" * 50)

    tests = [
        ("Health Check", lambda: client.test_health()),
        ("List Models", lambda: client.test_list_models()),
        ("Get Model", lambda: client.test_get_model(args.model)),
        (
            "Chat Completion",
            lambda: client.test_chat_completion(args.model, stream=False),
        ),
        (
            "Chat Completion with Tools",
            lambda: client.test_chat_completion_with_tools(args.model),
        ),
        (
            "Streaming Chat Completion",
            lambda: client.test_chat_completion(args.model, stream=True),
        ),
        ("Completion", lambda: client.test_completion(args.model, stream=False)),
        (
            "Streaming Completion",
            lambda: client.test_completion(args.model, stream=True),
        ),
        ("Embeddings", lambda: client.test_embeddings(args.model)),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"❌ {failed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
