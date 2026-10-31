# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o 4.5 registration using Qwen3's existing conversion rules."""

import json
from pathlib import Path
from typing import Any

from transformers import PretrainedConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping, QKVMapping, ReplicatedMapping
from megatron.bridge.models.minicpmo.config_validation import validate_minicpmo_config
from megatron.bridge.models.minicpmo.minicpmo_config import MiniCPMOModelConfig
from megatron.bridge.models.minicpmo.modeling_minicpmo import MiniCPMOModel
from megatron.bridge.models.qwen.qwen3_bridge import Qwen3Bridge


@MegatronModelBridge.register_bridge(source="MiniCPMO", target=MiniCPMOModel, model_type="minicpmo")
class MiniCPMOBridge(Qwen3Bridge):
    """Convert the understanding stage while preserving MiniCPM-o identity.

    The language weights are Qwen3 weights under ``llm``. Perception weights
    are replicated and retain their Hugging Face names under ``perception``.
    TTS is deliberately excluded from this understanding-stage model.
    """

    MODEL_CONFIG_CLASS = MiniCPMOModelConfig
    USE_MODEL_CONFIG_FOR_CONVERSION = True
    HF_EXPORT_IGNORED_SOURCE_PREFIXES = ("tts.",)
    # The checkpoint's remote tokenizer/processor target its native HF runtime.
    # Re-saving with a newer Transformers can change their metadata schema.
    ADDITIONAL_FILE_PATTERNS = [
        "tokenizer*.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "processor_config.json",
    ]

    def postprocess_hf_export_artifacts(self, path: Path) -> None:
        """Keep component metadata compatible with the native processor loader.

        The native processor receives its components as positional arguments.
        Newer Transformers also serializes them inside processor_config.json,
        which supplies the same arguments twice when loaded by native HF 4.51.
        Component settings remain in the preserved source artifact files.
        """
        config_path = path / "config.json"
        if config_path.is_file():
            model_config = json.loads(config_path.read_text())
            # HF 5 moves these fields; the checkpoint's native HF 4 loader
            # otherwise silently falls back to a different rotary base/dtype.
            rotary = model_config.get("rope_parameters", {})
            if "rope_theta" in rotary:
                model_config["rope_theta"] = rotary["rope_theta"]
            if "dtype" in model_config:
                model_config["torch_dtype"] = model_config["dtype"]
            config_path.write_text(json.dumps(model_config, indent=2) + "\n")
        processor_path = path / "processor_config.json"
        if not processor_path.is_file():
            return
        config = json.loads(processor_path.read_text())
        for component in ("image_processor", "audio_processor", "tokenizer"):
            config.pop(component, None)
        processor_path.write_text(json.dumps(config, indent=2) + "\n")

    def hf_config_to_model_config_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Use Qwen3 language conversion and retain the perception configuration."""
        original = validate_minicpmo_config(hf_config.to_dict())
        kwargs = super().hf_config_to_model_config_kwargs(hf_config)
        kwargs["minicpmo_config"] = original
        return kwargs

    @classmethod
    def megatron_to_hf_config(cls, provider: MiniCPMOModelConfig) -> dict[str, Any]:
        """Preserve perception configuration when exporting understanding weights."""
        result = validate_minicpmo_config(provider.minicpmo_config)
        result.update(super().megatron_to_hf_config(provider))
        return result

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Namespace Qwen3 mappings and add replicated perception parameters."""
        mappings = []
        for mapping in super().mapping_registry().mappings:
            if mapping.megatron_param.startswith("mtp."):
                continue
            name = "language_model." + mapping.megatron_param
            if isinstance(mapping, QKVMapping):
                mappings.append(QKVMapping(name, **{k: "llm." + v for k, v in mapping.hf_param.items()}))
            elif isinstance(mapping, GatedMLPMapping):
                mappings.append(GatedMLPMapping(name, **{k: "llm." + v for k, v in mapping.hf_param.items()}))
            elif isinstance(mapping, AutoMapping):
                mappings.append(AutoMapping(name, "llm." + mapping.hf_param, mapping.permute_dims))
            else:
                raise TypeError(f"Unsupported inherited Qwen3 mapping: {type(mapping).__name__}")
        for module in ("vpm", "resampler", "apm", "audio_projection_layer"):
            mappings.append(ReplicatedMapping(f"perception.{module}.**", f"{module}.**"))
        return MegatronMappingRegistry(*mappings)
