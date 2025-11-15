"""
RKLLM constants and enums.

This module contains all the constants used by the RKLLM C library.
"""

import ctypes

# RKLLM library handles and types
rkllm_lib = None
RKLLM_Handle_t = ctypes.c_void_p
userdata = ctypes.c_void_p(None)

# LLM Call State Constants
RKLLM_RUN_NORMAL = 0
RKLLM_RUN_WAITING = 1
RKLLM_RUN_FINISH = 2
RKLLM_RUN_ERROR = 3

# RKLLM Input Type Constants
RKLLM_INPUT_PROMPT = 0
RKLLM_INPUT_TOKEN = 1
RKLLM_INPUT_EMBED = 2
RKLLM_INPUT_MULTIMODAL = 3

# RKLLM Inference Mode Constants
RKLLM_INFER_GENERATE = 0
RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1
RKLLM_INFER_GET_LOGITS = 2

# Ctypes for enums
LLMCallState = ctypes.c_int
RKLLMInputType = ctypes.c_int
RKLLMInferMode = ctypes.c_int

# Callback function type (will be defined when needed)
callback_type = None
callback = None
