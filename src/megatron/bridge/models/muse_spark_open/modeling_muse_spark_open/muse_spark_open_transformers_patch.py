# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""HF export-only causal LM for standalone Muse Spark Open text checkpoints.

Transformers ships the Muse Spark Open language model (`MuseSparkTextModel`) but no
text-only causal LM. This class pairs it with the VL checkpoint's output head,
including its output multiplier and tanh soft-cap, under the same parameter names
(`model.*`, `lm_head.*`). This file is copied into text exports and depends only on
Torch and Transformers, not on Megatron-Bridge.
"""

# This module is copied into exported HF checkpoints and loaded through their
# AutoModelForCausalLM auto_map. It is never used by the Megatron forward pass.

import torch
from torch import nn
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers.models.muse_spark.configuration_muse_spark import MuseSparkTextConfig
from transformers.models.muse_spark.modeling_muse_spark import MuseSparkPreTrainedModel, MuseSparkTextModel


class MuseSparkForCausalLM(MuseSparkPreTrainedModel, GenerationMixin):
    """HF language-only entrypoint with the complete Muse Spark Open output head."""

    config_class = MuseSparkTextConfig
    config: MuseSparkTextConfig
    _no_split_modules = ["MuseSparkTextDecoderLayer"]

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        # HF inspects the defining file for its experts decorator. The decoder
        # comes from Transformers, so report its capability instead of silently
        # changing the default experts backend to eager.
        return MuseSparkPreTrainedModel._can_set_experts_implementation()

    def __init__(self, config: MuseSparkTextConfig) -> None:
        super().__init__(config)
        self.model = MuseSparkTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> MoeCausalLMOutputWithPast:
        """Match the VL model's head: T * tanh(lm_head(h) * output_multiplier / T)."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        logits = logits * self.config.output_multiplier
        logits = logits / self.config.final_logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
