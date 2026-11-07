# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Checkpoint-compatible joint schema head for Cloudflare CLEF."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from megatron.bridge.decision.clef_head import (
    EncodedQuestion,
    EncodedRecord,
)
from megatron.bridge.decision.clef_head import (
    JointSchemaHead as _JointSchemaHead,
)


@dataclass(frozen=True)
class ClefHeadConfig:
    """Declarative configuration matching joint_head_config.json.

    Args:
        hidden_size: Backbone hidden-state and output-embedding width.
        width: Internal decision-head width.
        routing_layers: Number of option-to-evidence cross-attention blocks.
        layers: Number of joint field decoder blocks.
        heads: Attention head count.
        feedforward: Feed-forward width in attention blocks.
        dropout: Dropout probability used while post-training the head.
    """

    hidden_size: int
    width: int
    routing_layers: int
    layers: int
    heads: int
    feedforward: int
    dropout: float = 0.0

    def __post_init__(self) -> None:
        dimensions = (
            self.hidden_size,
            self.width,
            self.routing_layers,
            self.layers,
            self.heads,
            self.feedforward,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("CLEF head dimensions and layer counts must be positive")
        if self.width % self.heads:
            raise ValueError("CLEF head width must be divisible by heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def build(self) -> JointSchemaHead:
        """Build a checkpoint-compatible joint schema head.

        Returns:
            Newly initialized CLEF decision head.
        """
        return JointSchemaHead(self)


__all__ = ["ClefHeadConfig", "EncodedQuestion", "EncodedRecord", "JointSchemaHead"]


class JointSchemaHead(_JointSchemaHead):
    """Declarative configuration interface to the shared CLEF head implementation.

    Args:
        config: Dimensions and routing settings matching the released checkpoint.
    """

    def __init__(self, config: ClefHeadConfig) -> None:
        super().__init__(
            hidden_size=config.hidden_size,
            width=config.width,
            routing_layers=config.routing_layers,
            layers=config.layers,
            heads=config.heads,
            feedforward=config.feedforward,
            dropout=config.dropout,
        )
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        records: list[EncodedRecord],
        output_embedding_weight: torch.Tensor,
    ) -> list[list[torch.Tensor]]:
        """Return one option-logit vector per question in each encoded record.

        Args:
            hidden_states: Tensor [batch, sequence, hidden].
            input_ids: Tensor [batch, sequence] of vocabulary IDs.
            attention_mask: Tensor [batch, sequence] with one for real tokens.
            records: Batch-aligned question and option spans.
            output_embedding_weight: Frozen output rows [vocabulary, hidden].

        Returns:
            Nested batch/question list of option logits.
        """
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
        if input_ids.shape != attention_mask.shape or input_ids.shape != hidden_states.shape[:2]:
            raise ValueError("input_ids, attention_mask, and hidden_states must share batch/sequence dimensions")
        if len(records) != hidden_states.shape[0]:
            raise ValueError("records must contain one item per batch row")
        if output_embedding_weight.ndim != 2 or output_embedding_weight.shape[1] != self.config.hidden_size:
            raise ValueError("output_embedding_weight must have shape [vocab, hidden]")
        return super().forward(hidden_states, input_ids, attention_mask, records, output_embedding_weight)
