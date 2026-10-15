# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""HF export-only patch preserving the VL checkpoint's output-head semantics.

The Transformers text-only CausalLM class omits its configured output
multiplier and soft-cap. Reuse its decoder and forward/generation/loss machinery,
adding those operations at the head without changing parameter names or values.
This file is copied into text exports and depends only on Torch and Transformers,
not on Megatron-Bridge.
"""

# This module is copied into exported HF checkpoints and loaded through their
# AutoModelForCausalLM auto_map. It is never used by the Megatron forward pass.

import torch
from torch import nn
from transformers.models.muse_spark.configuration_muse_spark import MuseSparkTextConfig
from transformers.models.muse_spark.modeling_muse_spark import MuseSparkForCausalLM as HFMuseSparkForCausalLM


class MuseSparkTextLMHead(nn.Linear):
    """Reuse the existing head parameters and apply the VL model's logit transforms."""

    def __init__(self, linear: nn.Linear, config: MuseSparkTextConfig) -> None:
        nn.Module.__init__(self)
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias
        self.output_multiplier = config.output_multiplier
        self.final_logit_softcapping = config.final_logit_softcapping

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Match the VL head's operation order, including BF16 rounding."""
        logits = super().forward(hidden_states)
        logits = logits * self.output_multiplier
        logits = logits / self.final_logit_softcapping
        logits = torch.tanh(logits)
        return logits * self.final_logit_softcapping


class MuseSparkForCausalLM(HFMuseSparkForCausalLM):
    """HF language-only entrypoint with the complete Muse Spark Open output head."""

    config_class = MuseSparkTextConfig

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        # HF inspects the defining file for its experts decorator. This small
        # subclass delegates the decoder, so retain the base class's capability
        # instead of silently changing the default backend to eager.
        return HFMuseSparkForCausalLM._can_set_experts_implementation()

    def __init__(self, config: MuseSparkTextConfig) -> None:
        super().__init__(config)
        self.lm_head = MuseSparkTextLMHead(self.lm_head, config)
