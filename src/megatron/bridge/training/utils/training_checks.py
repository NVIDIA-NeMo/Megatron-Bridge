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

"""Training checks that record parameter updates and expert load during a run.

``ParameterUpdateCheck`` records, for every parameter tensor, whether it trained (held by the optimizer, received a
gradient at least once, FP32 master weights changed) or stayed frozen (outside the optimizer, values unchanged).
``ExpertLoadMonitor`` counts the tokens each expert of every MoE layer receives in each training step. Both write a
JSON report on rank 0 for a regression check to compare with its limits; neither changes training.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from megatron.bridge.training.callbacks import Callback, CallbackContext
from megatron.bridge.utils.common_utils import get_rank_safe, print_rank_0


logger = logging.getLogger(__name__)

PARAMETER_UPDATES_REPORT = "parameter_updates.json"
EXPERT_LOAD_REPORT = "expert_load.json"

# Values reduced at once by ``tensor_fingerprint``; bounds its float64 working copy to 32 MiB.
_FINGERPRINT_CHUNK_NUMEL = 1 << 22
_WRAPPER_PREFIX = re.compile(r"^(module\.)+")

MasterWeights = Callable[[], dict[str, torch.Tensor]]


def _distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


@torch.no_grad()
def tensor_fingerprint(tensor: torch.Tensor, chunk_numel: int = _FINGERPRINT_CHUNK_NUMEL) -> torch.Tensor:
    """Return the float64 sum and sum of squares of a tensor's values.

    The fingerprints of two tensors differ whenever any value differs, except for changes that preserve both sums
    exactly. Values are reduced in chunks so the float64 working copy stays small.

    Args:
        tensor: Tensor to fingerprint. Transformer Engine quantized tensors are dequantized first.
        chunk_numel: Number of values reduced at once.

    Returns:
        A float64 tensor of shape ``[2]`` on the tensor's device.
    """
    if type(tensor).__module__.startswith("transformer_engine") and hasattr(tensor, "dequantize"):
        tensor = tensor.dequantize()
    flat = tensor.detach().reshape(-1)
    fingerprint = torch.zeros(2, dtype=torch.float64, device=flat.device)
    for chunk in flat.split(chunk_numel):
        values = chunk.to(torch.float64)
        fingerprint[0] += values.sum()
        fingerprint[1] += torch.dot(values, values)
    return fingerprint


def _parallel_layout(model: Sequence[torch.nn.Module]) -> tuple[int, int, int]:
    """Return the pipeline rank, pipeline size and expert-parallel rank of this process."""
    if not _distributed():
        return 0, 1, 0
    from megatron.bridge.training.utils.pg_utils import get_pg_collection

    pg_collection = get_pg_collection(list(model))

    def rank_and_size(name: str) -> tuple[int, int]:
        group = getattr(pg_collection, name, None)
        if group is None:
            return 0, 1
        return torch.distributed.get_rank(group), torch.distributed.get_world_size(group)

    pp_rank, pp_size = rank_and_size("pp")
    ep_rank, _ = rank_and_size("ep")
    return pp_rank, pp_size, ep_rank


def _chunk_prefix(chunk_index: int, num_chunks: int, pp_rank: int, pp_size: int) -> str:
    prefix = f"pp{pp_rank}." if pp_size > 1 else ""
    if num_chunks > 1:
        prefix += f"vp{chunk_index}."
    return prefix


def parameter_keys(
    model: Sequence[torch.nn.Module], *, pp_rank: int = 0, pp_size: int = 1, ep_rank: int = 0
) -> list[tuple[str, torch.nn.Parameter]]:
    """Name every parameter tensor this rank holds with a key that is unique across ranks.

    Keys are the parameter names without wrapper prefixes. Pipeline stages and virtual chunks add ``pp<rank>.`` and
    ``vp<chunk>.``, because their layer indices are local; expert parameters add ``ep<rank>.``, because each
    expert-parallel rank holds different experts under the same local names. Shards of one tensor on tensor-,
    data- or context-parallel ranks share a key.

    Args:
        model: Model chunks on this rank.
        pp_rank: Pipeline-parallel rank.
        pp_size: Pipeline-parallel size.
        ep_rank: Expert-parallel rank.

    Returns:
        ``(key, parameter)`` pairs, each parameter object listed once.
    """
    entries = []
    seen: set[int] = set()
    for chunk_index, chunk in enumerate(model):
        prefix = _chunk_prefix(chunk_index, len(model), pp_rank, pp_size)
        for name, param in chunk.named_parameters():
            if id(param) in seen:
                continue
            seen.add(id(param))
            expert = getattr(param, "allreduce", True) is False
            key = prefix + (f"ep{ep_rank}." if expert else "") + _WRAPPER_PREFIX.sub("", name)
            entries.append((key, param))
    return entries


def _plain_optimizer_state(optimizer: torch.optim.Optimizer, master: torch.Tensor) -> dict[str, torch.Tensor]:
    tensors = {"param": master}
    for name, value in optimizer.state.get(master, {}).items():
        if isinstance(value, torch.Tensor):
            tensors[name] = value
    return tensors


def master_weight_accessors(optimizer: Any) -> dict[int, MasterWeights]:
    """Map each model parameter with master weights on this rank to a function returning them and their state.

    Supports Megatron-Core's distributed optimizer (including the precision-aware variant and chained optimizers),
    its non-distributed mixed-precision optimizer, and plain optimizers whose parameters are their own master weights.

    Args:
        optimizer: The training optimizer.

    Returns:
        A mapping from ``id(model_param)`` to a function that returns ``{"param": master weights, <state>: tensor}``
        for the part of the parameter this rank holds.
    """
    accessors: dict[int, MasterWeights] = {}
    for part in getattr(optimizer, "chained_optimizers", None) or [optimizer]:
        index_map = getattr(part, "model_param_group_index_map", None)
        get_states = getattr(part, "_get_main_param_and_optimizer_states", None)
        if index_map is not None and get_states is not None:
            for model_param in index_map:
                accessors[id(model_param)] = functools.partial(get_states, model_param)
            continue
        inner = getattr(part, "optimizer", part)
        float16_groups = getattr(part, "float16_groups", None)
        master_groups = getattr(part, "fp32_from_float16_groups", None)
        if float16_groups is not None and master_groups is not None:
            for model_group, master_group in zip(float16_groups, master_groups):
                for model_param, master in zip(model_group, master_group):
                    accessors[id(model_param)] = functools.partial(_plain_optimizer_state, inner, master)
            for group in getattr(part, "fp32_from_fp32_groups", None) or []:
                for param in group:
                    accessors[id(param)] = functools.partial(_plain_optimizer_state, inner, param)
            continue
        for group in getattr(inner, "param_groups", []):
            for param in group["params"]:
                accessors.setdefault(id(param), functools.partial(_plain_optimizer_state, inner, param))
    return accessors


def _gather_on_rank_zero(record: Any) -> list[Any] | None:
    """Gather one picklable object per rank on rank 0; other ranks get ``None``."""
    if not _distributed():
        return [record]
    rank = torch.distributed.get_rank()
    gathered = [None] * torch.distributed.get_world_size() if rank == 0 else None
    torch.distributed.gather_object(record, gathered, dst=0)
    return gathered


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1) + "\n")
    os.replace(temporary, path)


def merge_parameter_records(records: Sequence[dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Merge per-rank parameter records into one record per tensor.

    A tensor trained if any rank holding part of it saw that part train, so ``optimizer``, ``gradient`` and
    ``changed`` combine with OR. ``None`` means a rank had no information (it holds none of the tensor's master
    weights) and is ignored.

    Args:
        records: One ``{key: record}`` mapping per rank.

    Returns:
        One merged record per key.
    """

    def combine(first: bool | None, second: bool | None) -> bool | None:
        if first is None:
            return second
        if second is None:
            return first
        return first or second

    merged: dict[str, dict[str, Any]] = {}
    for rank_records in records:
        for key, record in rank_records.items():
            current = merged.get(key)
            if current is None:
                merged[key] = dict(record)
                continue
            current["trainable"] = current["trainable"] or record["trainable"]
            for field in ("optimizer", "gradient", "changed"):
                current[field] = combine(current.get(field), record.get(field))
    for record in merged.values():
        record["optimizer"] = bool(record.get("optimizer"))
    return dict(sorted(merged.items()))


class ParameterUpdateCheck(Callback):
    """Record whether every parameter tensor trained or stayed frozen, and write the result as JSON.

    A trainable tensor (``requires_grad``) passes when the optimizer holds it, its Adam second moment is nonzero
    (it received a gradient at least once) and its FP32 master weights changed after the first step. Master weights
    are compared rather than the BF16 model weights, because at small learning rates a value near 1.0 can move less
    than one BF16 step over a short run. A frozen tensor passes when the optimizer does not hold it and its values
    are bitwise unchanged since the start of training. This callback only records; the CI evaluator applies the
    case's declared exceptions and frozen set.

    Args:
        report_path: Where rank 0 writes the JSON report.
    """

    def __init__(self, report_path: str | os.PathLike[str]) -> None:
        self.report_path = Path(report_path)
        self._entries: list[tuple[str, torch.nn.Parameter]] = []
        self._accessors: dict[int, MasterWeights] = {}
        self._frozen_start: dict[str, torch.Tensor] = {}
        self._master_start: dict[str, torch.Tensor] = {}
        self._pending: list[tuple[str, MasterWeights]] = []

    def on_train_start(self, context: CallbackContext) -> None:
        """Fingerprint the frozen tensors before the first step."""
        pp_rank, pp_size, ep_rank = _parallel_layout(context.model)
        self._entries = parameter_keys(context.model, pp_rank=pp_rank, pp_size=pp_size, ep_rank=ep_rank)
        self._accessors = master_weight_accessors(context.optimizer) if context.optimizer is not None else {}
        self._frozen_start = {
            key: tensor_fingerprint(param) for key, param in self._entries if not param.requires_grad
        }
        self._master_start = {}
        self._pending = [
            (key, self._accessors[id(param)])
            for key, param in self._entries
            if param.requires_grad and id(param) in self._accessors
        ]

    def on_train_step_end(self, context: CallbackContext) -> None:
        """Fingerprint the master weights after the first step, once the optimizer state exists."""
        pending = []
        for key, accessor in self._pending:
            try:
                self._master_start[key] = tensor_fingerprint(accessor()["param"])
            except KeyError:
                # The precision-aware optimizer creates its master weights at its first update.
                pending.append((key, accessor))
        self._pending = pending

    def on_train_end(self, context: CallbackContext) -> None:
        """Compare with the recorded fingerprints and write the report on rank 0."""
        if self._pending:
            logger.warning(
                "No master weights were recorded for %d tensors; reporting them as unknown.", len(self._pending)
            )
        master_start = self._master_start
        records: dict[str, dict[str, Any]] = {}
        for key, param in self._entries:
            accessor = self._accessors.get(id(param))
            record: dict[str, Any] = {"trainable": param.requires_grad, "numel": param.numel()}
            if not param.requires_grad:
                start = self._frozen_start[key]
                record.update(
                    optimizer=accessor is not None,
                    gradient=None,
                    changed=not torch.equal(start, tensor_fingerprint(param)),
                )
            elif accessor is None:
                record.update(optimizer=False, gradient=None, changed=None)
            else:
                states = accessor()
                second_moment = states.get("exp_avg_sq")
                start = master_start.get(key)
                record.update(
                    optimizer=True,
                    gradient=None if second_moment is None else bool(torch.count_nonzero(second_moment).item()),
                    changed=None if start is None else not torch.equal(start, tensor_fingerprint(states["param"])),
                )
            records[key] = record

        gathered = _gather_on_rank_zero(records)
        if gathered is None:
            return
        tensors = merge_parameter_records(gathered)
        trainable = [record for record in tensors.values() if record["trainable"]]
        frozen = [record for record in tensors.values() if not record["trainable"]]
        summary = {
            "trainable_tensors": len(trainable),
            "frozen_tensors": len(frozen),
            "trainable_not_in_optimizer": sum(not record["optimizer"] for record in trainable),
            "trainable_without_gradient": sum(record["gradient"] is False for record in trainable),
            "trainable_unchanged": sum(record["changed"] is False for record in trainable),
            "frozen_changed": sum(bool(record["changed"]) for record in frozen),
            "frozen_in_optimizer": sum(record["optimizer"] for record in frozen),
        }
        _write_json(self.report_path, {"version": 1, "summary": summary, "tensors": tensors})
        print_rank_0(
            "parameter updates | "
            + " | ".join(f"{name.replace('_', ' ')}: {value}" for name, value in summary.items())
            + f" | report: {self.report_path}"
        )


def expert_load_statistics(counts: torch.Tensor) -> dict[str, list[Any]]:
    """Summarize one step's tokens per expert for each MoE layer.

    Args:
        counts: ``[layers, experts]`` token counts. Counting every token the same number of times (for example
            twice under activation recompute) does not change the ratios.

    Returns:
        Per-layer ``tokens`` (total assignments), ``max_load`` and ``min_load`` (most and least loaded expert as a
        multiple of the mean; ``None`` for a layer that received no tokens) and ``idle`` (experts with no tokens).
    """
    values = counts.detach().to(torch.float64).cpu()
    totals = values.sum(dim=1)
    mean = totals / values.shape[1]
    has_tokens = totals > 0
    max_load = torch.where(has_tokens, values.max(dim=1).values / mean.clamp_min(1e-30), torch.nan)
    min_load = torch.where(has_tokens, values.min(dim=1).values / mean.clamp_min(1e-30), torch.nan)

    def as_list(tensor: torch.Tensor) -> list[float | None]:
        return [None if torch.isnan(value) else float(value) for value in tensor]

    return {
        "tokens": [int(value) for value in totals],
        "max_load": as_list(max_load),
        "min_load": as_list(min_load),
        "idle": [int(value) for value in (values == 0).sum(dim=1)],
    }


def moe_layers(model: Sequence[torch.nn.Module]) -> list[tuple[int, str, torch.nn.Module]]:
    """Find the MoE layers in each model chunk.

    A module counts as an MoE layer when it has an ``experts`` submodule, the global ``local_expert_indices`` of the
    experts this rank holds, and a config with ``num_moe_experts`` (Megatron-Core's ``BaseMoELayer``).

    Args:
        model: Model chunks on this rank.

    Returns:
        ``(chunk index, module name without wrapper prefixes, module)`` for each MoE layer.
    """
    layers = []
    for chunk_index, chunk in enumerate(model):
        for name, module in chunk.named_modules():
            experts = getattr(module, "experts", None)
            if not isinstance(experts, torch.nn.Module) or getattr(module, "local_expert_indices", None) is None:
                continue
            if not getattr(getattr(module, "config", None), "num_moe_experts", None):
                continue
            layers.append((chunk_index, _WRAPPER_PREFIX.sub("", name), module))
    return layers


class ExpertLoadMonitor(Callback):
    """Count the tokens each expert receives per training step, log a summary line and write a JSON report.

    A forward pre-hook on each MoE layer's experts reads the tokens per local expert after dispatch, so the counts
    are what the experts actually processed, and the hook keeps working when the router runs inside a CUDA graph.
    After each step one all-reduce sums the ``[layers, experts]`` counts over all ranks; MTP and hash-routed MoE
    layers are included, and evaluation during training is not counted. This callback only records; the CI
    evaluator applies the case's limits.

    Args:
        report_path: Where rank 0 writes the JSON report.
    """

    def __init__(self, report_path: str | os.PathLike[str]) -> None:
        self.report_path = Path(report_path)
        self._layer_keys: list[str] = []
        self._counts: torch.Tensor | None = None
        # Dispatchers that already moved their counts to the host add them here, avoiding a copy per microbatch.
        self._host_counts: torch.Tensor | None = None
        self._run_counts: torch.Tensor | None = None
        self._steps: list[dict[str, Any]] = []
        self._handles: list[Any] = []
        self._num_experts = 0
        self._paused = False

    def on_train_start(self, context: CallbackContext) -> None:
        """Register one counting hook per local MoE layer."""
        pp_rank, pp_size, _ = _parallel_layout(context.model)
        local = [
            (_chunk_prefix(chunk, len(context.model), pp_rank, pp_size) + name, module)
            for chunk, name, module in moe_layers(context.model)
        ]
        descriptions = _all_gather([(key, module.config.num_moe_experts) for key, module in local])
        experts_per_layer = dict(item for rank_items in descriptions for item in rank_items)
        if not experts_per_layer:
            return
        if len(set(experts_per_layer.values())) != 1:
            raise ValueError(f"MoE layers have different expert counts: {experts_per_layer}")
        self._layer_keys = sorted(experts_per_layer)
        self._num_experts = next(iter(experts_per_layer.values()))
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else None
        self._counts = torch.zeros(len(self._layer_keys), self._num_experts, dtype=torch.int64, device=device)
        self._host_counts = torch.zeros(len(self._layer_keys), self._num_experts, dtype=torch.int64)
        self._run_counts = torch.zeros_like(self._counts)
        rows = {key: row for row, key in enumerate(self._layer_keys)}
        for key, module in local:
            indices = [int(index) for index in module.local_expert_indices]
            hook = functools.partial(
                self._count,
                rows[key],
                {
                    self._counts.device: torch.tensor(indices, device=self._counts.device),
                    self._host_counts.device: torch.tensor(indices),
                },
            )
            self._handles.append(module.experts.register_forward_pre_hook(hook, with_kwargs=True))

    def _count(
        self,
        row: int,
        indices: dict[torch.device, torch.Tensor],
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        tokens = args[1] if len(args) > 1 else kwargs.get("tokens_per_expert")
        if tokens is None or self._paused:
            return
        tokens = torch.as_tensor(tokens).reshape(-1).to(torch.int64)
        target = self._host_counts if tokens.device.type == "cpu" else self._counts
        local_indices = indices[target.device]
        if tokens.numel() != local_indices.numel():
            raise ValueError(
                f"An MoE layer's experts received counts for {tokens.numel()} experts but hold {local_indices.numel()}."
            )
        target[row].index_add_(0, local_indices, tokens.to(target.device))

    def on_eval_start(self, context: CallbackContext) -> None:
        """Stop counting while the model evaluates; validation tokens are not a training step's load."""
        self._paused = True

    def on_eval_end(self, context: CallbackContext) -> None:
        """Count again once evaluation has finished."""
        self._paused = False

    def on_train_step_end(self, context: CallbackContext) -> None:
        """Sum this step's counts over all ranks and record them."""
        if self._counts is None:
            return
        if self._host_counts.any():
            self._counts += self._host_counts.to(self._counts.device)
            self._host_counts.zero_()
        if _distributed():
            torch.distributed.all_reduce(self._counts)
        self._run_counts += self._counts
        statistics = expert_load_statistics(self._counts)
        self._counts.zero_()
        if get_rank_safe() != 0:
            return
        iteration = context.state.train_state.step + 1
        self._steps.append({"iteration": iteration, **statistics})
        loads = [(value, key) for value, key in zip(statistics["max_load"], self._layer_keys) if value is not None]
        lowest = [value for value in statistics["min_load"] if value is not None]
        worst = max(loads) if loads else (float("nan"), "none")
        print_rank_0(
            f"expert load | iteration {iteration:8d} | worst max/mean load: {worst[0]:.3f} ({worst[1]}) | "
            f"lowest min/mean load: {min(lowest, default=float('nan')):.3f} | "
            f"experts without tokens: {sum(statistics['idle'])}"
        )

    def on_train_end(self, context: CallbackContext) -> None:
        """Remove the hooks and write the report on rank 0."""
        for handle in self._handles:
            handle.remove()
        self._handles = []
        if self._run_counts is None or get_rank_safe() != 0:
            return
        totals = self._run_counts.cpu()
        report = {
            "version": 1,
            "num_experts": self._num_experts,
            "layers": self._layer_keys,
            "steps": self._steps,
            "total_tokens": totals.tolist(),
            "experts_without_tokens_in_run": int((totals == 0).sum()),
        }
        _write_json(self.report_path, report)
        print_rank_0(f"expert load | report: {self.report_path}")


def _all_gather(items: Any) -> list[Any]:
    """Gather one picklable object from every rank on every rank."""
    if not _distributed():
        return [items]
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, items)
    return gathered
