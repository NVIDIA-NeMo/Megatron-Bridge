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

"""Megatron-Core glue for decision models: final hidden states, vocab-parallel row gathers, adapter I/O.

Decision heads read the backbone's final (post-norm) hidden states instead of next-token logits.  Megatron's
``GPTModel`` and Qwen3.5 ``Qwen3VLModel`` both accept an ``output_processor`` that replaces the output layer, so a head
can be attached without changing the model.  The helpers here:

* :func:`final_hidden_states_processor` - an ``output_processor`` returning batch-first hidden states, gathered over
  the sequence-parallel region when sequence parallelism is on.
* :func:`gather_vocab_parallel_rows` - read a handful of rows of a vocab-parallel output-layer weight (Clef's joint
  head uses the LM-head rows of option tokens as lexical anchors).
* :func:`last_token_pool` - CLM-style last-token pooling for right-padded batches.
* :func:`save_trainable_state` / :func:`load_trainable_state` - LoRA weights by parameter name, one file per
  tensor-parallel rank, so post-training stages (SFT -> RL) can be chained without a full checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.distributed as dist


_WRAPPER_PREFIXES = ("module.", "_checkpoint_wrapped_module.", "_orig_mod.")


def canonical_name(name: str) -> str:
    """Parameter name without DDP / Float16Module / checkpoint wrapper prefixes."""
    changed = True
    while changed:
        changed = False
        for prefix in _WRAPPER_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                changed = True
    return name.replace("._checkpoint_wrapped_module", "")


def final_hidden_states_processor(*, tp_group: Any | None) -> Callable[..., torch.Tensor]:
    """Build an ``output_processor`` that returns the decoder's final hidden states.

    Args:
        tp_group: Tensor-parallel process group, used to undo the sequence-parallel scatter.

    Returns:
        A callable for ``GPTModel.forward(..., output_processor=...)``.  It receives ``hidden_states`` of shape
        [sequence / tp, batch, hidden] when sequence parallelism is on (else [sequence, batch, hidden]) and returns a
        contiguous tensor of shape [batch, sequence, hidden], identical on every tensor-parallel rank.
    """

    def processor(*, hidden_states: torch.Tensor, config: Any = None, **_: Any) -> torch.Tensor:
        """Gather and transpose ``hidden_states`` (see :func:`final_hidden_states_processor`)."""
        if config is not None and getattr(config, "sequence_parallel", False) and tp_group is not None:
            from megatron.core.tensor_parallel import gather_from_sequence_parallel_region

            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, tensor_parallel_output_grad=False, group=tp_group
            )
        return hidden_states.transpose(0, 1).contiguous()

    return processor


@torch.no_grad()
def gather_vocab_parallel_rows(
    weight_shard: torch.Tensor, token_ids: Sequence[int], *, tp_group: Any | None
) -> torch.Tensor:
    """Gather full rows of a vocab-parallel weight.

    Args:
        weight_shard: Tensor of shape [vocab / tp, hidden]; this rank's contiguous slice of the (padded) vocabulary,
            starting at ``tp_rank * (vocab / tp)``.
        token_ids: Token ids whose rows are needed.
        tp_group: Tensor-parallel process group (None when tp == 1).

    Returns:
        Tensor of shape [len(token_ids), hidden] with the full rows, identical on every tensor-parallel rank.
    """
    shard = weight_shard.shape[0]
    rank = dist.get_rank(tp_group) if tp_group is not None and dist.is_initialized() else 0
    ids = torch.tensor(list(token_ids), dtype=torch.long, device=weight_shard.device)
    local = ids - rank * shard
    owned = (local >= 0) & (local < shard)
    rows = torch.zeros((len(ids), weight_shard.shape[1]), dtype=weight_shard.dtype, device=weight_shard.device)
    rows[owned] = weight_shard[local[owned]]
    if tp_group is not None and dist.is_initialized() and dist.get_world_size(tp_group) > 1:
        dist.all_reduce(rows, group=tp_group)
    return rows


def last_token_pool(hidden: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Hidden state of the last real token of each right-padded sequence.

    Args:
        hidden: Tensor of shape [batch, sequence, hidden].
        lengths: Long tensor of shape [batch] with the number of real tokens per row (>= 1).

    Returns:
        Tensor of shape [batch, hidden].
    """
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths.to(hidden.device) - 1]


def trainable_named_parameters(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    """``(canonical name, parameter)`` for every parameter with ``requires_grad``."""
    return [(canonical_name(n), p) for n, p in model.named_parameters() if p.requires_grad]


def save_trainable_state(model: torch.nn.Module, directory: str | Path, *, tp_rank: int, write: bool) -> None:
    """Save this tensor-parallel rank's trainable tensors to ``directory/adapter_tp{tp_rank}.safetensors``.

    Args:
        model: Unwrapped or wrapped Megatron module.
        directory: Output directory.
        tp_rank: Tensor-parallel rank (selects the file name).
        write: Whether this rank writes (normally data-parallel rank 0 of each tensor-parallel rank).
    """
    if not write:
        return
    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {n: p.detach().float().cpu().contiguous() for n, p in trainable_named_parameters(model)}
    save_file(tensors, str(directory / f"adapter_tp{tp_rank}.safetensors"))


@torch.no_grad()
def load_trainable_state(model: torch.nn.Module, directory: str | Path, *, tp_rank: int) -> int:
    """Load tensors written by :func:`save_trainable_state` (same tensor-parallel size) into ``model``.

    Returns:
        Number of tensors loaded.

    Raises:
        KeyError: If a trainable parameter has no saved tensor.
    """
    from safetensors.torch import load_file

    state = load_file(str(Path(directory) / f"adapter_tp{tp_rank}.safetensors"))
    params = dict(trainable_named_parameters(model))
    missing = sorted(set(params) - set(state))
    if missing:
        raise KeyError(f"{directory} is missing {len(missing)} trainable tensors, e.g. {missing[:3]}")
    for name, p in params.items():
        p.copy_(state[name].to(device=p.device, dtype=p.dtype))
    return len(params)


def all_reduce_mean_grads(module: torch.nn.Module, *, group: Any | None) -> None:
    """Average the gradients of a replicated module over ``group`` (e.g. the data-parallel group)."""
    if group is None or not dist.is_initialized() or dist.get_world_size(group) == 1:
        return
    world = dist.get_world_size(group)
    for p in module.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad, group=group)
        p.grad.div_(world)
