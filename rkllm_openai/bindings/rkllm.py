"""
RKLLM wrapper class.

This module contains the main RKLLM class that wraps the C library functionality
and provides a Python interface for model loading, inference, and management.
"""

import ctypes
import os
import threading
from typing import List

import numpy as np

from .constants import (
    RKLLM_INFER_GENERATE,
    RKLLM_INPUT_PROMPT,
    RKLLM_RUN_ERROR,
    RKLLM_RUN_FINISH,
    RKLLM_RUN_NORMAL,
    RKLLM_Handle_t,
    callback,
    callback_type,
    rkllm_lib,
)
from .structures import (
    RKLLMInferParam,
    RKLLMInput,
    RKLLMParam,
    RKLLMResult,
)

# Global variables for callback handling
global_text = []
global_state = -1
callback_lock = threading.Lock()


def callback_impl(result, userdata, state):
    """Callback function for RKLLM inference."""
    global global_text, global_state

    with callback_lock:
        if state == RKLLM_RUN_FINISH:
            global_state = state
        elif state == RKLLM_RUN_ERROR:
            global_state = state
            print("RKLLM run error")
        elif state == RKLLM_RUN_NORMAL:
            global_state = state
            if result and result.contents.text:
                text = result.contents.text.decode("utf-8")
                global_text.append(text)

    return 0


def init_callback():
    """Initialize the callback function if not already done."""
    global callback_type, callback

    if callback_type is None:
        callback_type = ctypes.CFUNCTYPE(
            ctypes.c_int, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int
        )
        callback = callback_type(callback_impl)


class RKLLM:
    """RKLLM model wrapper class."""

    def __init__(self, model_path: str, platform: str = "rk3588", lib_path: str = None):
        """
        Initialize the RKLLM model.

        Args:
            model_path: Path to the RKLLM model file
            platform: Target platform (rk3588, rk3576, rv1126b, rk3562)
            lib_path: Path to librkllmrt.so (optional, will search common paths)
        """
        global rkllm_lib

        if rkllm_lib is None:
            rkllm_lib = self._load_library(lib_path)

        # Initialize callback if not done
        init_callback()

        self.model_path = model_path
        self.platform = platform
        self.handle = RKLLM_Handle_t()
        self.tools = None
        self._init_model()

    def _load_library(self, lib_path: str = None) -> ctypes.CDLL:
        """Load the RKLLM shared library."""
        if lib_path and os.path.exists(lib_path):
            return ctypes.CDLL(lib_path)

        # Try common paths
        lib_paths = [
            "lib/librkllmrt.so",
            "/usr/lib/librkllmrt.so",
            "/usr/local/lib/librkllmrt.so",
            "./librkllmrt.so",
        ]

        for path in lib_paths:
            if os.path.exists(path):
                return ctypes.CDLL(path)

        raise RuntimeError(
            f"Could not find librkllmrt.so. Tried paths: {lib_paths}. "
            "Please specify the correct path using lib_path parameter."
        )

    def _init_model(self):
        """Initialize the RKLLM model with default parameters."""
        rkllm_param = RKLLMParam()
        rkllm_param.model_path = bytes(self.model_path, "utf-8")
        rkllm_param.max_context_len = 4096
        rkllm_param.max_new_tokens = 4096
        rkllm_param.skip_special_token = True
        rkllm_param.n_keep = -1
        rkllm_param.top_k = 1
        rkllm_param.top_p = 0.9
        rkllm_param.temperature = 0.8
        rkllm_param.repeat_penalty = 1.1
        rkllm_param.frequency_penalty = 0.0
        rkllm_param.presence_penalty = 0.0
        rkllm_param.mirostat = 0
        rkllm_param.mirostat_tau = 5.0
        rkllm_param.mirostat_eta = 0.1
        rkllm_param.is_async = False
        rkllm_param.img_start = "".encode("utf-8")
        rkllm_param.img_end = "".encode("utf-8")
        rkllm_param.img_content = "".encode("utf-8")
        rkllm_param.extend_param.base_domain_id = 0
        rkllm_param.extend_param.embed_flash = 1
        rkllm_param.extend_param.n_batch = 1
        rkllm_param.extend_param.use_cross_attn = 0
        rkllm_param.extend_param.enabled_cpus_num = 4

        if self.platform.lower() in ["rk3576", "rk3588"]:
            rkllm_param.extend_param.enabled_cpus_mask = (
                (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)
            )
        else:
            rkllm_param.extend_param.enabled_cpus_mask = (
                (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)
            )

        # Initialize RKLLM functions
        self.rkllm_init = rkllm_lib.rkllm_init
        self.rkllm_init.argtypes = [
            ctypes.POINTER(RKLLM_Handle_t),
            ctypes.POINTER(RKLLMParam),
            callback_type,
        ]
        self.rkllm_init.restype = ctypes.c_int

        ret = self.rkllm_init(
            ctypes.byref(self.handle), ctypes.byref(rkllm_param), callback
        )
        if ret != 0:
            raise RuntimeError(f"RKLLM initialization failed with code {ret}")

        self.rkllm_run = rkllm_lib.rkllm_run
        self.rkllm_run.argtypes = [
            RKLLM_Handle_t,
            ctypes.POINTER(RKLLMInput),
            ctypes.POINTER(RKLLMInferParam),
            ctypes.c_void_p,
        ]
        self.rkllm_run.restype = ctypes.c_int

        self.rkllm_destroy = rkllm_lib.rkllm_destroy
        self.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self.rkllm_destroy.restype = ctypes.c_int

        # Initialize tools and chat template functions
        self.set_chat_template = rkllm_lib.rkllm_set_chat_template
        self.set_chat_template.argtypes = [
            RKLLM_Handle_t,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        self.set_chat_template.restype = ctypes.c_int

        self.set_function_tools_internal = rkllm_lib.rkllm_set_function_tools
        self.set_function_tools_internal.argtypes = [
            RKLLM_Handle_t,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        self.set_function_tools_internal.restype = ctypes.c_int

        # Initialize inference parameters
        self.rkllm_infer_params = RKLLMInferParam()
        ctypes.memset(
            ctypes.byref(self.rkllm_infer_params), 0, ctypes.sizeof(RKLLMInferParam)
        )
        self.rkllm_infer_params.mode = RKLLM_INFER_GENERATE
        self.rkllm_infer_params.lora_params = None
        self.rkllm_infer_params.keep_history = 0

        print(f"RKLLM model '{self.model_path}' initialized successfully")

    def set_function_tools(
        self, system_prompt: str, tools: str, tool_response_str: str = "tool_response"
    ):
        """
        Set function tools for the model.

        Args:
            system_prompt: System prompt for the model
            tools: JSON string containing tool definitions
            tool_response_str: String to identify tool responses
        """
        if self.tools is None or self.tools != tools:
            self.tools = tools
            self.set_function_tools_internal(
                self.handle,
                ctypes.c_char_p(system_prompt.encode("utf-8")),
                ctypes.c_char_p(tools.encode("utf-8")),
                ctypes.c_char_p(tool_response_str.encode("utf-8")),
            )

    def clear_tools(self):
        """Clear any previously set tools from the model."""
        if self.tools is not None:
            self.tools = None
            # Reset tools by passing empty strings
            self.set_function_tools_internal(
                self.handle,
                ctypes.c_char_p(b""),
                ctypes.c_char_p(b""),
                ctypes.c_char_p(b""),
            )

    def apply_chat_template(
        self,
        system_prompt: str = None,
        prompt_prefix: str = None,
        prompt_postfix: str = None,
    ):
        """
        Apply a chat template to the model.

        Args:
            system_prompt: System message template
            prompt_prefix: Prefix for user messages
            prompt_postfix: Postfix after user messages (before assistant)
        """
        # Use default templates if not provided
        if system_prompt is None:
            system_prompt = "<|im_start|>system You are a helpful assistant. <|im_end|>"
        if prompt_prefix is None:
            prompt_prefix = "<|im_start|>user"
        if prompt_postfix is None:
            prompt_postfix = "<|im_end|><|im_start|>assistant"

        self.set_chat_template(
            self.handle,
            ctypes.c_char_p(system_prompt.encode("utf-8")),
            ctypes.c_char_p(prompt_prefix.encode("utf-8")),
            ctypes.c_char_p(prompt_postfix.encode("utf-8")),
        )

    def load_chat_template_from_file(self, template_path: str):
        """
        Load chat template from a jinja2 file.

        Args:
            template_path: Path to the chat_template.jinja2 file
        """
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Chat template file not found: {template_path}")

        try:
            # Try to import jinja2 for template parsing
            import jinja2

            with open(template_path, "r", encoding="utf-8") as f:
                template_content = f.read()

            # Parse the jinja2 template (basic parsing for common patterns)
            # This is a simplified implementation - a full jinja2 parser would be more complex

            # Extract system, user, and assistant templates
            system_template = self._extract_template_part(template_content, "system")
            user_template = self._extract_template_part(template_content, "user")
            assistant_template = self._extract_template_part(
                template_content, "assistant"
            )

            # Apply the extracted templates
            if system_template or user_template or assistant_template:
                self.apply_chat_template(
                    system_template, user_template, assistant_template
                )
                print(f"Loaded chat template from {template_path}")
            else:
                print(
                    f"Warning: Could not parse templates from {template_path}, using defaults"
                )

        except ImportError:
            # If jinja2 is not available, try basic text parsing
            print("Warning: jinja2 not available, using basic template parsing")
            self._load_template_basic(template_path)

    def _extract_template_part(self, template_content: str, role: str) -> str:
        """
        Extract template part for a specific role from jinja2 template.

        This is a basic implementation that looks for common patterns.
        """
        import re

        # Look for patterns like {% if message['role'] == 'system' %}...{% endif %}
        pattern = rf"{{% if message\['role'\] == '{role}' %}}(.*?){{% endif %}}"
        match = re.search(pattern, template_content, re.DOTALL)

        if match:
            return match.group(1).strip()

        # Alternative pattern: {%- if message.role == 'system' -%}
        pattern = rf"{{%- if message\.role == '{role}' -%}}(.*?){{%- endif -%}}"
        match = re.search(pattern, template_content, re.DOTALL)

        if match:
            return match.group(1).strip()

        return None

    def _load_template_basic(self, template_path: str):
        """
        Basic template loading without jinja2 dependency.
        """
        with open(template_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Look for common template markers and use defaults
        if "im_start" in content and "im_end" in content:
            # Looks like a ChatML-style template
            self.apply_chat_template()
        else:
            print("Warning: Unrecognized template format, using defaults")
            self.apply_chat_template()

    def generate(self, prompt: str, role: str = "user", enable_thinking: bool = False):
        """
        Generate text using RKLLM.

        Args:
            prompt: Input text prompt
            role: Role for the input (user, assistant, system)
            enable_thinking: Whether to enable thinking mode
        """
        global global_text, global_state

        # Reset global state
        with callback_lock:
            global_text.clear()
            global_state = -1

        rkllm_input = RKLLMInput()
        rkllm_input.role = role.encode("utf-8")
        rkllm_input.enable_thinking = ctypes.c_bool(enable_thinking)
        rkllm_input.input_type = RKLLM_INPUT_PROMPT
        rkllm_input.input_data.prompt_input = ctypes.c_char_p(prompt.encode("utf-8"))

        ret = self.rkllm_run(
            self.handle,
            ctypes.byref(rkllm_input),
            ctypes.byref(self.rkllm_infer_params),
            None,
        )
        if ret != 0:
            raise RuntimeError(f"RKLLM inference failed with code {ret}")

    def get_embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings for text.

        Args:
            text: Input text to generate embeddings for

        Returns:
            List of floating point values representing the embedding

        Note:
            This is currently a placeholder implementation that returns
            random embeddings. For actual embeddings, you would need to
            configure RKLLM with an embedding model and use the appropriate
            inference mode.
        """
        # For now, return dummy embeddings since RKLLM embeddings need specific setup
        # This would need to be implemented based on the specific embedding model used
        embedding_size = 1536  # Common embedding size
        return np.random.normal(0, 1, embedding_size).tolist()

    def get_text_buffer(self) -> List[str]:
        """
        Get the current text buffer from inference.

        Returns:
            List of text chunks from the callback
        """
        with callback_lock:
            return global_text.copy()

    def clear_text_buffer(self):
        """Clear the global text buffer."""
        with callback_lock:
            global_text.clear()

    def get_state(self) -> int:
        """
        Get the current inference state.

        Returns:
            Current state value
        """
        with callback_lock:
            return global_state

    def is_finished(self) -> bool:
        """
        Check if inference is finished.

        Returns:
            True if inference is complete
        """
        return self.get_state() == RKLLM_RUN_FINISH

    def has_error(self) -> bool:
        """
        Check if there was an error during inference.

        Returns:
            True if there was an error
        """
        return self.get_state() == RKLLM_RUN_ERROR

    def release(self):
        """Release the RKLLM model resources."""
        if hasattr(self, "handle") and self.handle:
            self.rkllm_destroy(self.handle)
            self.handle = None

    def __del__(self):
        """Destructor to ensure resources are released."""
        self.release()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
