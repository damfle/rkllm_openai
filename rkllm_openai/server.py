#!/usr/bin/env python3
"""
OpenAI-compatible API server for RKLLM models.

This server provides OpenAI-compatible endpoints for RKLLM models including:
- /v1/chat/completions
- /v1/completions
- /v1/models
- /v1/models/{model}
- /v1/embeddings
- /health (async health check)
"""

import argparse
import logging
import os
import sys
import threading
import time
from typing import Optional

from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from .commons import (
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
    convert_openai_tools_to_rkllm_format,
    get_system_prompt_with_tools,
    should_force_tool_use,
)
from .commons.response_generators import (
    generate_chat_completion,
    generate_completion,
    stream_chat_completion,
    stream_completion,
)
from .model_manager import ModelManager

# Global variables
model_manager: Optional[ModelManager] = None
inference_lock = threading.Lock()


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    CORS(app)

    # Custom logging filter to exclude health endpoint
    class HealthFilter(logging.Filter):
        def filter(self, record):
            return "/health" not in record.getMessage()

    # Apply filter to werkzeug logger
    logging.getLogger("werkzeug").addFilter(HealthFilter())

    @app.route("/health", methods=["GET"])
    def health():
        """Health check endpoint."""
        return jsonify({"status": "healthy", "timestamp": int(time.time())})

    @app.route("/v1/models", methods=["GET"])
    def list_models():
        """List available models."""
        if model_manager is None:
            return jsonify({"error": {"message": "Model manager not initialized"}}), 500
        available_models = model_manager.get_available_models()
        return jsonify(
            {
                "object": "list",
                "data": [
                    {
                        "id": model,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "rkllm",
                    }
                    for model in available_models
                ],
            }
        )

    @app.route("/v1/models/<model_id>", methods=["GET"])
    def get_model(model_id):
        """Get specific model information."""
        if model_manager is None:
            return jsonify({"error": {"message": "Model manager not initialized"}}), 500
        available_models = model_manager.get_available_models()
        if model_id not in available_models:
            return (
                jsonify(
                    {
                        "error": {
                            "message": f"Model {model_id} not found",
                            "type": "invalid_request_error",
                            "code": "model_not_found",
                        }
                    }
                ),
                404,
            )

        return jsonify(
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "rkllm",
            }
        )

    @app.route("/v1/chat/completions", methods=["POST"])
    def chat_completions():
        """Handle chat completion requests."""
        with inference_lock:
            try:
                req = ChatCompletionRequest(**request.json)

                # Get the model for this request
                if model_manager is None:
                    return jsonify(
                        {"error": {"message": "Model manager not initialized"}}
                    ), 500
                try:
                    current_model = model_manager.get_model(req.model)
                except ValueError as e:
                    return (
                        jsonify(
                            {
                                "error": {
                                    "message": str(e),
                                    "type": "invalid_request_error",
                                    "code": "model_not_found",
                                }
                            }
                        ),
                        400,
                    )

                # Handle tools if provided
                if req.tools:
                    # Convert OpenAI tools to RKLLM format
                    tools_json = convert_openai_tools_to_rkllm_format(req.tools)

                    # Extract actual system prompt from messages if provided
                    user_system_prompt = "You are a helpful assistant."
                    for message in req.messages:
                        if message.role == "system":
                            user_system_prompt = message.content
                            break

                    # Enhance with tool instructions
                    enhanced_system_prompt = get_system_prompt_with_tools(
                        user_system_prompt, req.tools
                    )

                    # Configure model with tools
                    current_model.set_function_tools(
                        system_prompt=enhanced_system_prompt,
                        tools=tools_json,
                        tool_response_str="tool_response",
                    )
                else:
                    # Clear any previously set tools if no tools in this request
                    current_model.clear_tools()

                # Convert messages to prompt
                prompt_parts = []
                system_message_found = False

                for message in req.messages:
                    if message.role == "system":
                        system_message_found = True
                        # Use the original system message in the prompt
                        prompt_parts.append(f"System: {message.content}")
                    elif message.role == "user":
                        prompt_parts.append(f"User: {message.content}")
                    elif message.role == "assistant":
                        if message.content:
                            prompt_parts.append(f"Assistant: {message.content}")
                    elif message.role == "tool":
                        prompt_parts.append(f"Tool response: {message.content}")

                # If no system message, use default
                if not system_message_found:
                    prompt_parts.insert(0, f"System: You are a helpful assistant.")

                prompt = "\n".join(prompt_parts) + "\nAssistant:"

                if req.stream:
                    return Response(
                        stream_chat_completion(prompt, req, current_model),
                        mimetype="text/plain",
                    )
                else:
                    return generate_chat_completion(prompt, req, current_model)

            except Exception as e:
                return (
                    jsonify(
                        {
                            "error": {
                                "message": str(e),
                                "type": "invalid_request_error",
                            }
                        }
                    ),
                    400,
                )

    @app.route("/v1/completions", methods=["POST"])
    def completions():
        """Handle completion requests."""
        with inference_lock:
            try:
                req = CompletionRequest(**request.json)

                # Get the model for this request
                if model_manager is None:
                    return jsonify(
                        {"error": {"message": "Model manager not initialized"}}
                    ), 500
                try:
                    current_model = model_manager.get_model(req.model)
                except ValueError as e:
                    return (
                        jsonify(
                            {
                                "error": {
                                    "message": str(e),
                                    "type": "invalid_request_error",
                                    "code": "model_not_found",
                                }
                            }
                        ),
                        400,
                    )

                if req.stream:
                    return Response(
                        stream_completion(req.prompt, req, current_model),
                        mimetype="text/plain",
                    )
                else:
                    return generate_completion(req.prompt, req, current_model)

            except Exception as e:
                return (
                    jsonify(
                        {
                            "error": {
                                "message": str(e),
                                "type": "invalid_request_error",
                            }
                        }
                    ),
                    400,
                )

    @app.route("/v1/embeddings", methods=["POST"])
    def embeddings():
        """Handle embedding requests."""
        try:
            req = EmbeddingRequest(**request.json)

            # Get the model for this request
            if model_manager is None:
                return jsonify(
                    {"error": {"message": "Model manager not initialized"}}
                ), 500
            try:
                current_model = model_manager.get_model(req.model)
            except ValueError as e:
                return (
                    jsonify(
                        {
                            "error": {
                                "message": str(e),
                                "type": "invalid_request_error",
                                "code": "model_not_found",
                            }
                        }
                    ),
                    400,
                )

            # Handle single string or list of strings
            inputs = req.input if isinstance(req.input, list) else [req.input]

            embeddings_data = []
            for i, text in enumerate(inputs):
                embedding = current_model.get_embeddings(text)
                embeddings_data.append(
                    {
                        "object": "embedding",
                        "embedding": embedding,
                        "index": i,
                    }
                )

            return jsonify(
                {
                    "object": "list",
                    "data": embeddings_data,
                    "model": req.model,
                    "usage": {
                        "prompt_tokens": sum(len(text.split()) for text in inputs),
                        "total_tokens": sum(len(text.split()) for text in inputs),
                    },
                }
            )

        except Exception as e:
            return (
                jsonify(
                    {
                        "error": {
                            "message": str(e),
                            "type": "invalid_request_error",
                        }
                    }
                ),
                400,
            )

    return app


def main():
    """Main entry point for the server."""
    parser = argparse.ArgumentParser(description="RKLLM OpenAI-compatible server")
    parser.add_argument(
        "--model-path", required=True, help="Path to RKLLM model file (.rkllm)"
    )
    parser.add_argument(
        "--platform", choices=["rk3588", "rk3576"], default="rk3588", help="Platform"
    )
    parser.add_argument(
        "--lib-path",
        help="Path to RKLLM library (optional, will search common paths if not specified)",
    )

    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--model-timeout",
        type=int,
        default=300,
        help="Model unload timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--chat-template", help="Path to chat template file (jinja2 format)"
    )

    args = parser.parse_args()

    # Validate model path (single file)
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)

    if not os.path.isfile(args.model_path):
        print(f"Error: Model path must be a file: {args.model_path}")
        sys.exit(1)

    # Validate library path if provided
    if args.lib_path and not os.path.exists(args.lib_path):
        print(f"Error: Library file not found: {args.lib_path}")
        sys.exit(1)

    # Initialize global model manager
    global model_manager
    try:
        model_manager = ModelManager(
            model_path=args.model_path,
            platform=args.platform,
            lib_path=args.lib_path,
            model_timeout=args.model_timeout,
            chat_template=args.chat_template,
        )
        print(
            f"Model manager initialized with model: {list(model_manager.get_available_models())[0]}"
        )

        # Log chat template status
        if args.chat_template:
            if os.path.exists(args.chat_template):
                print(f"Chat template configured: {args.chat_template}")
            else:
                print(f"Warning: Chat template file not found: {args.chat_template}")

    except Exception as e:
        print(f"Error initializing model manager: {e}")
        sys.exit(1)

    # Create and run the app
    app = create_app()
    print(f"Starting server on {args.host}:{args.port}")
    print(f"Model timeout: {args.model_timeout} seconds")

    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    finally:
        # Clean up model manager on shutdown
        if model_manager:
            model_manager.shutdown()


if __name__ == "__main__":
    main()
