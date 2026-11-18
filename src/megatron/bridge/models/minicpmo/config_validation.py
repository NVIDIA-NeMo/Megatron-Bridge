# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint identity and understanding-stage configuration validation."""

from copy import deepcopy
from typing import Any


def validate_minicpmo_config(config: dict[str, Any]) -> dict[str, Any]:
    """Copy and validate a MiniCPM-o 4.5 checkpoint without changing its caller.

    Args:
        config: Serialized Hugging Face configuration.

    Returns:
        An independent configuration for non-streaming understanding.

    Raises:
        ValueError: If the checkpoint identifies another architecture/version or
            omits a required enabled perception configuration.
    """
    if config.get("model_type") != "minicpmo" or str(config.get("version")) != "4.5":
        raise ValueError("This bridge supports MiniCPM-o 4.5 only, not MiniCPM-o 2.6.")
    if config.get("architectures") != ["MiniCPMO"]:
        raise ValueError("Expected the MiniCPMO checkpoint architecture.")
    for enabled, field_name in (("init_vision", "vision_config"), ("init_audio", "audio_config")):
        if enabled not in config or not isinstance(config[enabled], bool):
            raise ValueError(f"MiniCPM-o must explicitly declare {enabled}.")
        if config[enabled] and not isinstance(config.get(field_name), dict):
            raise ValueError(f"Enabled MiniCPM-o perception requires {field_name}.")
    result = deepcopy(config)
    result.update(init_tts=False, stream_input=False, use_cache=False)
    return result
