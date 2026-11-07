# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Contrastive Language Model (CLM) projection heads.

CLM (https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) is a frozen Qwen3-8B encoder (last-token pooled, L2-normalised
hidden state) followed by two small MLP projection heads.  A (state, candidate) pair is scored by
``exp(logit_scale) * cos(state_head(e_state), action_head(e_candidate))`` and a question's answer distribution is the
softmax of those scores over its options.

The modules here reproduce the released architecture parameter-for-parameter so that checkpoints round-trip with the
``contrastive-lm`` package (``clm-serve`` / ``clm.Engine``): the checkpoint is a ``torch.save`` dict with
``state_head``, ``action_head``, ``logit_scale`` and ``cfg``.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from megatron.bridge.models.clm.decision_head import CLMHeadConfig, CLMHeadPair

from .clm_prompting import clm_pairs


CLM_HIDDEN = 4096
CLM_PROJ_DIM = 512
CLM_MAX_LOGIT_SCALE = math.log(100.0)
CLM_HF_REPO = "Contrastive-LM/CLM-v0.1-8B"
CLM_HF_REVISION = "e939398d4556fcd9400c76fa8c5a513202f42b0a"
CLM_ENCODER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
CLM_HF_FILE = "CLM_v0.1-8B.pt"
CLM_ENCODER = "Qwen/Qwen3-8B"
CLM_MAX_TOKENS = 2048


class CLMHeads(CLMHeadPair):
    """Recipe interface for the shared state/action projection implementation.

    Args:
        cfg: Released checkpoint configuration dictionary.
        logit_scale: Initial logarithm of the inverse softmax temperature.
    """

    def __init__(self, cfg: dict[str, Any], logit_scale: float = math.log(1 / 0.07)) -> None:
        self.cfg = dict(cfg)
        config = CLMHeadConfig(
            hidden_size=cfg.get("hidden_size", CLM_HIDDEN),
            width=cfg["width"],
            depth=cfg["depth"],
            projection_dim=cfg.get("projection_dim", CLM_PROJ_DIM),
            activation=cfg.get("activation", "gelu"),
            layer_norm=cfg.get("layernorm", False),
            residual=cfg.get("residual", False),
        )
        super().__init__(config, initial_temperature=math.exp(-logit_scale))

    @property
    def scale(self) -> torch.Tensor:
        """Scalar tensor ``exp(logit_scale)`` clamped to at most 100, as in CLM training and serving."""
        return self.logit_scale.clamp(max=CLM_MAX_LOGIT_SCALE).exp()

    def project_states(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalised state projections.

        Args:
            x: Tensor of shape [states, hidden] with encoder embeddings of state texts.

        Returns:
            Tensor of shape [states, proj] in the heads' dtype.
        """
        return F.normalize(self.state_head(x.to(self.logit_scale.dtype)), dim=-1)

    def project_actions(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalised candidate projections.

        Args:
            x: Tensor of shape [candidates, hidden] with encoder embeddings of option texts.

        Returns:
            Tensor of shape [candidates, proj] in the heads' dtype.
        """
        return F.normalize(self.action_head(x.to(self.logit_scale.dtype)), dim=-1)

    def question_logits(
        self, state_emb: torch.Tensor, option_emb: torch.Tensor, option_counts: Sequence[int]
    ) -> list[torch.Tensor]:
        """Per-question option logits.

        Args:
            state_emb: Tensor of shape [questions, hidden] with the encoder embedding of each question's state text.
            option_emb: Tensor of shape [candidates, hidden] with the encoder embedding of every option text, flattened
                question-major so that ``candidates == sum(option_counts)``.
            option_counts: Number of options of each question.

        Returns:
            List of ``questions`` tensors; the i-th has shape [option_counts[i]] and holds ``scale * cosine`` logits.
        """
        zs = self.project_states(state_emb)
        za = self.project_actions(option_emb)
        out, start = [], 0
        for i, n in enumerate(option_counts):
            out.append(self.scale * (za[start : start + n] @ zs[i]))
            start += n
        return out


def load_clm_heads(
    path: str | os.PathLike | None = None, map_location: str = "cpu", *, revision: str | None = None
) -> CLMHeads:
    """Load a CLM head checkpoint (defaults to the released CLM-v0.1-8B heads from the Hugging Face Hub)."""
    if path is None or str(path).startswith("hf://"):
        from huggingface_hub import hf_hub_download

        repo, filename = CLM_HF_REPO, CLM_HF_FILE
        if path is not None:
            repo, _, filename = str(path)[len("hf://") :].rpartition("/")
        path = hf_hub_download(repo, filename, revision=revision or (CLM_HF_REVISION if repo == CLM_HF_REPO else None))
    ck = torch.load(path, map_location=map_location, weights_only=True)
    cfg = dict(ck["cfg"])
    if "projection_dim" in ck:
        cfg["projection_dim"] = ck["projection_dim"]
    heads = CLMHeads(cfg, logit_scale=float(torch.as_tensor(ck["logit_scale"]).float()))
    heads.state_head.load_state_dict(ck["state_head"])
    heads.action_head.load_state_dict(ck["action_head"])
    return heads


def save_clm_heads(heads: CLMHeads, path: str | os.PathLike, extra: dict[str, Any] | None = None) -> None:
    """Save heads in the ``contrastive-lm`` checkpoint format (loadable by ``clm-serve`` and :func:`load_clm_heads`)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ck = {
        "state_head": {k: v.detach().float().cpu() for k, v in heads.state_head.state_dict().items()},
        "action_head": {k: v.detach().float().cpu() for k, v in heads.action_head.state_dict().items()},
        "logit_scale": heads.logit_scale.detach().float().cpu(),
        "cfg": dict(heads.cfg),
        "projection_dim": heads.cfg.get("projection_dim", CLM_PROJ_DIM),
    }
    ck.update(extra or {})
    torch.save(ck, path)


def collect_clm_texts(records: Sequence[dict[str, Any]]) -> list[str]:
    """Every unique text the encoder must embed for these records (state texts first, then candidates)."""
    states, candidates = {}, {}
    for record in records:
        for state, _, texts in clm_pairs(record).values():
            states.setdefault(state, None)
            for text in texts:
                candidates.setdefault(text, None)
    return list(states) + [t for t in candidates if t not in states]


class EmbeddingTable:
    """Text -> row lookup into a table of L2-normalised encoder embeddings."""

    def __init__(self, texts: Sequence[str], embeddings: torch.Tensor):
        """Create the table.

        Args:
            texts: Unique texts, one per row of ``embeddings``.
            embeddings: Tensor of shape [texts, hidden].
        """
        if len(texts) != embeddings.shape[0]:
            raise ValueError(f"{len(texts)} texts but {embeddings.shape[0]} embeddings")
        self.index = {t: i for i, t in enumerate(texts)}
        self.embeddings = embeddings

    def lookup(self, texts: Sequence[str], device: torch.device | str | None = None) -> torch.Tensor:
        """Return a tensor of shape [len(texts), hidden] (a copy, optionally moved to ``device``)."""
        rows = torch.tensor([self.index[t] for t in texts], dtype=torch.long)
        out = self.embeddings[rows]
        return out.to(device) if device is not None else out

    def save(self, path: str | os.PathLike) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"texts": list(self.index), "embeddings": self.embeddings.cpu()}, path)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "EmbeddingTable":
        ck = torch.load(path, map_location="cpu", weights_only=True)
        return cls(ck["texts"], ck["embeddings"])

    def missing(self, texts: Sequence[str]) -> list[str]:
        return [t for t in texts if t not in self.index]


def clm_batch(records: Sequence[dict[str, Any]], table: EmbeddingTable, device: torch.device | str | None = None):
    """Gather encoder embeddings for a batch of records.

    Returns:
        ``(state_emb, option_emb, option_counts, option_orders)``: ``state_emb`` has shape [questions, hidden] (one row
        per question of every record), ``option_emb`` has shape [candidates, hidden] (question-major), and
        ``option_orders[i]`` is the ``{question_id: option ids}`` mapping of record ``i`` in CLM order.
    """
    states, options, counts, orders = [], [], [], []
    for record in records:
        order = {}
        for qid, (state, keys, texts) in clm_pairs(record).items():
            states.append(state)
            options.extend(texts)
            counts.append(len(texts))
            order[qid] = keys
        orders.append(order)
    return table.lookup(states, device), table.lookup(options, device), counts, orders
