"""
RKLLM C bindings package.

This package contains all the ctypes bindings for the RKLLM C library,
including structures, enums, and the main RKLLM wrapper class.

New features:
- Tools/function calling support
- Chat template loading from jinja2 files
- Enhanced callback handling for streaming
"""

from .constants import *
from .rkllm import RKLLM
from .structures import *

__all__ = [
    # Constants
    "RKLLM_RUN_NORMAL",
    "RKLLM_RUN_WAITING",
    "RKLLM_RUN_FINISH",
    "RKLLM_RUN_ERROR",
    "RKLLM_INPUT_PROMPT",
    "RKLLM_INPUT_TOKEN",
    "RKLLM_INPUT_EMBED",
    "RKLLM_INPUT_MULTIMODAL",
    "RKLLM_INFER_GENERATE",
    "RKLLM_INFER_GET_LAST_HIDDEN_LAYER",
    "RKLLM_INFER_GET_LOGITS",
    # Ctypes
    "LLMCallState",
    "RKLLMInputType",
    "RKLLMInferMode",
    "RKLLM_Handle_t",
    # Structures
    "RKLLMExtendParam",
    "RKLLMParam",
    "RKLLMLoraAdapter",
    "RKLLMEmbedInput",
    "RKLLMTokenInput",
    "RKLLMMultiModelInput",
    "RKLLMInputUnion",
    "RKLLMInput",
    "RKLLMLoraParam",
    "RKLLMPromptCacheParam",
    "RKLLMInferParam",
    "RKLLMResultLastHiddenLayer",
    "RKLLMResultLogits",
    "RKLLMPerfStat",
    "RKLLMResult",
    # Main class with tools and template support
    "RKLLM",
]
