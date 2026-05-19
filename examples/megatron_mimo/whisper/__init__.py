# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from .whisper_layer_specs import (
    get_whisper_layer_local_spec,
    get_whisper_layer_with_transformer_engine_spec,
)
from .whisper_model import WhisperEncoder


__all__ = [
    "WhisperEncoder",
    "get_whisper_layer_with_transformer_engine_spec",
    "get_whisper_layer_local_spec",
]
