# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Construct MiniCPM-o through Bridge's existing GPT builder."""

from megatron.core.process_groups_config import ProcessGroupCollection

from megatron.bridge.models.gpt.gpt_builder import GPTModelBuilder
from megatron.bridge.models.minicpmo.minicpmo_config import MiniCPMOModelConfig
from megatron.bridge.models.minicpmo.modeling_minicpmo import MiniCPMOModel, MiniCPMOPerception


class MiniCPMOModelBuilder(GPTModelBuilder):
    """Compose native perception and the shared Qwen3-compatible decoder."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> MiniCPMOModel:
        """Build the initial dense understanding model, rejecting unsupported modes."""
        config = self._model_config
        if not isinstance(config, MiniCPMOModelConfig):
            raise TypeError("MiniCPMOModelBuilder requires MiniCPMOModelConfig.")
        transformer = config.transformer
        if transformer.pipeline_model_parallel_size != 1 or transformer.context_parallel_size != 1:
            raise ValueError("MiniCPM-o currently requires PP=CP=1.")
        if transformer.sequence_parallel or getattr(transformer, "mtp_num_layers", None):
            raise ValueError("MiniCPM-o sequence parallelism and MTP are not supported yet.")
        if pre_process is False or post_process is False or vp_stage is not None:
            raise ValueError("MiniCPM-o requires a complete, non-interleaved decoder stage.")
        if transformer.fp8 is not None or getattr(transformer, "fp4", None) is not None:
            raise ValueError("MiniCPM-o quantized construction is not supported yet.")
        if transformer.num_moe_experts is not None:
            raise ValueError("MiniCPM-o 4.5 uses the dense Qwen3 decoder.")
        metadata = config.extra_checkpoint_metadata or {}
        source = metadata.get("hf_model_id")
        if not source:
            raise ValueError("Use AutoBridge.from_hf_pretrained to preserve the perception source path.")
        # Check trust before constructing a language model or importing checkpoint code.
        if not config.trust_remote_code:
            raise ValueError("MiniCPM-o model construction requires explicit trust_remote_code=True.")
        perception = MiniCPMOPerception(config, source=str(source), revision=metadata.get("hf_model_revision"))
        language = super().build_model(pg_collection, True, True, None)
        embedding_weight = language.embedding.word_embeddings.weight
        perception.to(device=embedding_weight.device, dtype=embedding_weight.dtype)
        language.requires_grad_(not config.freeze_language_model)
        return MiniCPMOModel(language, perception)
