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

"""DSpark draft heads: the sequential (Markov / RNN) head and the confidence head.

DSpark (arXiv:2607.05147) draft training. The parallel backbone predicts every
position of a block in one pass; these small heads add the pieces the paper needs
on top of that:

* the **sequential head** injects intra-block token dependency onto the parallel
  logits, as a factorized ``P(x | x0) = prod_k p_k(x_k | x0, x_1..x_{k-1})`` bias.
  ``VanillaMarkov`` is a memoryless low-rank token-to-token additive logit bias;
  ``GatedMarkovHead`` gates the previous-token embedding by the backbone hidden
  state; ``RNNHead`` carries a recurrent state so position k sees the full prefix.
* the **confidence head** predicts a per-position acceptance logit.

Only the training-time (teacher-forced, parallel) ``apply_block_logits`` path is
provided here; the serial autoregressive sampling used at inference is a separate
concern. These modules depend only on ``torch``.
"""

from __future__ import annotations

import torch
from torch import nn


class VanillaMarkov(nn.Module):
    """Memoryless low-rank token-to-token additive logit bias.

    The step bias is ``markov_w2(markov_w1[prev_token])``: a rank-``markov_rank``
    factorization of a ``[vocab, vocab]`` transition-bias matrix.
    """

    def __init__(self, *, vocab_size: int, markov_rank: int):
        super().__init__()
        if markov_rank <= 0:
            raise ValueError(f"VanillaMarkov requires markov_rank > 0, got {markov_rank}.")
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        self.markov_head_type = "vanilla"
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up ``markov_w1`` for ``token_ids`` (any shape) -> ``[..., markov_rank]``."""
        return self.markov_w1(token_ids.long())

    def project_bias(self, latent_states: torch.Tensor) -> torch.Tensor:
        """Project ``[..., markov_rank]`` latent states to a ``[..., vocab]`` logit bias."""
        return self.markov_w2(latent_states)

    def compute_step_bias(self, token_ids: torch.Tensor, hidden_states: torch.Tensor | None) -> torch.Tensor:
        """Logit bias for the previous tokens ``token_ids`` -> ``[..., vocab]`` (hidden unused)."""
        del hidden_states
        return self.project_bias(self.get_prev_embeddings(token_ids))

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor | None,
    ) -> torch.Tensor:
        """Add the (teacher-forced) sequential bias to the whole block's logits.

        Args:
            base_logits: Backbone logits ``[batch, num_blocks, block_size, vocab]``.
            token_ids: Teacher-forced previous token per position
                ``[batch, num_blocks, block_size]``.
            hidden_states: Backbone hidden ``[batch, num_blocks, block_size, hidden]``
                (used by the gated / RNN heads, ignored here).

        Returns:
            Corrected logits ``[batch, num_blocks, block_size, vocab]``.
        """
        if base_logits.size(-2) == 0:
            return base_logits
        return base_logits + self.compute_step_bias(token_ids, hidden_states)


class GatedMarkovHead(VanillaMarkov):
    """``VanillaMarkov`` whose previous-token embedding is gated by the backbone hidden."""

    def __init__(self, *, vocab_size: int, markov_rank: int, hidden_size: int):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.markov_head_type = "gated"
        self.gate_proj = nn.Linear(hidden_size + markov_rank, markov_rank)

    def compute_step_bias(self, token_ids: torch.Tensor, hidden_states: torch.Tensor | None) -> torch.Tensor:
        """Gate ``markov_w1[prev]`` by ``sigmoid(gate_proj([hidden; prev]))`` before ``markov_w2``.

        Args:
            token_ids: Previous token ids ``[..., ]``.
            hidden_states: Backbone hidden ``[..., hidden]`` (required).

        Returns:
            Logit bias ``[..., vocab]``.
        """
        if hidden_states is None:
            raise ValueError("GatedMarkovHead requires hidden_states.")
        prev_embeddings = self.get_prev_embeddings(token_ids)
        gate_inputs = torch.cat([hidden_states, prev_embeddings], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_inputs)).to(dtype=prev_embeddings.dtype)
        return self.project_bias(gate * prev_embeddings)


class RNNHead(VanillaMarkov):
    """GRU-like head carrying recurrent state across a block so position k sees ``x_1..x_{k-1}``."""

    def __init__(self, *, vocab_size: int, markov_rank: int, hidden_size: int):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.markov_head_type = "rnn"
        self.hidden_size = int(hidden_size)
        # Joint projection of [state; W1[prev]; hidden] -> [gate; candidate; output].
        self.joint_proj = nn.Linear(2 * markov_rank + hidden_size, 3 * markov_rank)

    def _rnn_step(
        self, state: torch.Tensor, prev_embeddings: torch.Tensor, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One recurrent step.

        Args:
            state: Previous recurrent state ``[..., markov_rank]``.
            prev_embeddings: ``markov_w1[prev]`` ``[..., markov_rank]``.
            hidden_states: Backbone hidden at this position ``[..., hidden]``.

        Returns:
            ``(new_state [..., markov_rank], logit_bias [..., vocab])``.
        """
        z = torch.cat([state, prev_embeddings, hidden_states], dim=-1)
        gate_raw, candidate_raw, output_raw = self.joint_proj(z).chunk(3, dim=-1)
        gate = torch.sigmoid(gate_raw)
        candidate = torch.tanh(candidate_raw)
        new_state = gate * state + (1.0 - gate) * candidate
        return new_state, self.project_bias(torch.tanh(output_raw))

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor | None,
    ) -> torch.Tensor:
        """Unroll the recurrence over the block (teacher-forced). Layout matches the base class."""
        if hidden_states is None:
            raise ValueError("RNNHead requires hidden_states.")
        block_size = base_logits.size(-2)
        if block_size == 0:
            return base_logits
        leading_shape = base_logits.shape[:-2]  # [batch, num_blocks]
        state = torch.zeros(*leading_shape, self.markov_rank, device=base_logits.device, dtype=hidden_states.dtype)
        output_logits = []
        for k in range(block_size):
            prev_emb = self.get_prev_embeddings(token_ids[..., k])
            state, bias = self._rnn_step(state, prev_emb, hidden_states[..., k, :])
            output_logits.append(base_logits[..., k, :] + bias)
        return torch.stack(output_logits, dim=-2)


def build_markov_head(
    *, markov_rank: int, markov_head_type: str, vocab_size: int, hidden_size: int
) -> nn.Module | None:
    """Construct the sequential head, or ``None`` when ``markov_rank == 0`` (disabled).

    Args:
        markov_rank: Low-rank size; ``0`` disables the head.
        markov_head_type: One of ``"vanilla"``, ``"gated"``, ``"rnn"``.
        vocab_size: Draft vocabulary size.
        hidden_size: Backbone hidden size (used by ``gated`` / ``rnn``).

    Returns:
        The head module, or ``None``.
    """
    if markov_rank < 0:
        raise ValueError(f"markov_rank must be >= 0, got {markov_rank}.")
    if markov_rank == 0:
        return None
    head_type = str(markov_head_type).lower()
    if head_type == "vanilla":
        return VanillaMarkov(vocab_size=vocab_size, markov_rank=markov_rank)
    if head_type == "gated":
        return GatedMarkovHead(vocab_size=vocab_size, markov_rank=markov_rank, hidden_size=hidden_size)
    if head_type == "rnn":
        return RNNHead(vocab_size=vocab_size, markov_rank=markov_rank, hidden_size=hidden_size)
    raise ValueError(f"Unsupported markov_head_type: {markov_head_type!r}.")


class ConfidenceHead(nn.Module):
    """Per-position acceptance-logit predictor (a single linear projection to a scalar)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(int(input_dim), 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Project ``[..., input_dim]`` features to a ``[...]`` acceptance logit."""
        return self.proj(features).squeeze(-1)
