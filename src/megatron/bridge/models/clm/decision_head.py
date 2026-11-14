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

"""Projection heads for Contrastive-LM decision-model post-training.

The released ``Contrastive-LM/CLM-v0.1-8B`` checkpoint keeps its Qwen3-8B
encoder frozen and trains two projection heads over last-token-pooled encoder
embeddings. This module reproduces the published head/checkpoint contract while
leaving backbone construction and distributed placement to Megatron Bridge.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import nn
from torch.nn import functional


@dataclass(frozen=True)
class CLMHeadConfig:
    """Declarative configuration for a CLM state/action head pair.

    Args:
        hidden_size: Width of each frozen backbone embedding.
        width: Hidden width of each projection MLP.
        depth: Number of linear layers in each projection MLP; at least two.
        projection_dim: Width of the normalized decision embedding.
        activation: One of ``"gelu"``, ``"relu"``, or ``"silu"``.
        layer_norm: Apply LayerNorm before intermediate hidden layers.
        residual: Add residual connections around intermediate hidden layers.
    """

    hidden_size: int
    width: int
    depth: int
    projection_dim: int
    activation: str
    layer_norm: bool
    residual: bool

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.width <= 0 or self.projection_dim <= 0:
            raise ValueError("CLM head dimensions must be positive")
        if self.depth < 2:
            raise ValueError("CLM head depth must be at least two")
        if self.activation not in {"gelu", "relu", "silu"}:
            raise ValueError(f"Unsupported CLM activation: {self.activation!r}")

    def build(self, *, initial_temperature: float = 0.07) -> CLMHeadPair:
        """Build the configured head pair.

        Args:
            initial_temperature: Positive InfoNCE softmax temperature.

        Returns:
            Trainable CLM state/action projection heads.
        """
        return CLMHeadPair(self, initial_temperature=initial_temperature)


class _ProjectionHead(nn.Module):
    def __init__(self, config: CLMHeadConfig) -> None:
        super().__init__()
        activations: dict[str, type[nn.Module]] = {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
        }
        # Attribute names intentionally match Contrastive-LM's published checkpoint.
        self.inp = nn.Linear(config.hidden_size, config.width)
        self.hidden = nn.ModuleList(nn.Linear(config.width, config.width) for _ in range(config.depth - 2))
        self.norms = nn.ModuleList(
            nn.LayerNorm(config.width) if config.layer_norm else nn.Identity() for _ in range(config.depth - 2)
        )
        self.out = nn.Linear(config.width, config.projection_dim)
        self.act = activations[config.activation]()
        self.residual = config.residual

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project frozen encoder embeddings.

        Args:
            embeddings: Tensor of shape [..., hidden].

        Returns:
            Tensor of shape [..., projection].
        """
        projected = self.act(self.inp(embeddings))
        for linear, norm in zip(self.hidden, self.norms):
            update = self.act(norm(linear(projected)))
            projected = projected + update if self.residual else update
        return self.out(projected)


class CLMHeadPair(nn.Module):
    """Published CLM state and action projections plus learned temperature."""

    def __init__(self, config: CLMHeadConfig, *, initial_temperature: float) -> None:
        super().__init__()
        if initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive")
        self.config = config
        self.state_head = _ProjectionHead(config)
        self.action_head = _ProjectionHead(config)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / initial_temperature), dtype=torch.float32))

    def project_states(self, state_embeddings: torch.Tensor) -> torch.Tensor:
        """Project and normalize state embeddings.

        Args:
            state_embeddings: Tensor of shape [batch, hidden].

        Returns:
            Tensor of shape [batch, projection].
        """
        self._validate_embeddings(state_embeddings, name="state_embeddings")
        return functional.normalize(self.state_head(state_embeddings), dim=-1)

    def project_actions(self, action_embeddings: torch.Tensor) -> torch.Tensor:
        """Project and normalize action embeddings.

        Args:
            action_embeddings: Tensor of shape [candidates, hidden].

        Returns:
            Tensor of shape [candidates, projection].
        """
        self._validate_embeddings(action_embeddings, name="action_embeddings")
        return functional.normalize(self.action_head(action_embeddings), dim=-1)

    def forward(self, state_embeddings: torch.Tensor, action_embeddings: torch.Tensor) -> torch.Tensor:
        """Score every state against every candidate action.

        Args:
            state_embeddings: Tensor of shape [batch, hidden].
            action_embeddings: Tensor of shape [candidates, hidden].

        Returns:
            Tensor of shape [batch, candidates] containing scaled cosine logits.
        """
        states = self.project_states(state_embeddings)
        actions = self.project_actions(action_embeddings)
        scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
        return scale * states @ actions.transpose(0, 1)

    @classmethod
    def from_upstream_checkpoint(cls, path: str | Path) -> CLMHeadPair:
        """Load the official Contrastive-LM ``.pt`` head checkpoint.

        Args:
            path: Local ``CLM_v0.1-8B.pt``-compatible checkpoint path.

        Returns:
            Head pair with the published state, action, and temperature tensors.
        """
        raw = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(raw, Mapping):
            raise ValueError("CLM checkpoint must contain a mapping")
        raw_cfg = raw.get("cfg")
        if not isinstance(raw_cfg, Mapping):
            raise ValueError("CLM checkpoint is missing its cfg mapping")
        cfg = cast(Mapping[str, object], raw_cfg)
        config = CLMHeadConfig(
            hidden_size=int(cfg.get("hidden_size", 4096)),
            width=int(cfg["width"]),
            depth=int(cfg["depth"]),
            projection_dim=int(cfg.get("projection_dim", raw.get("projection_dim", 512))),
            activation=str(cfg.get("activation", "gelu")),
            layer_norm=bool(cfg.get("layernorm", False)),
            residual=bool(cfg.get("residual", False)),
        )
        model = cls(config, initial_temperature=0.07)
        model.state_head.load_state_dict(cast(Mapping[str, torch.Tensor], raw["state_head"]))
        model.action_head.load_state_dict(cast(Mapping[str, torch.Tensor], raw["action_head"]))
        with torch.no_grad():
            model.logit_scale.copy_(torch.as_tensor(raw["logit_scale"], dtype=torch.float32))
        return model

    def _validate_embeddings(self, embeddings: torch.Tensor, *, name: str) -> None:
        if embeddings.ndim != 2 or embeddings.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"{name} must have shape [rows, {self.config.hidden_size}], got {tuple(embeddings.shape)}"
            )
