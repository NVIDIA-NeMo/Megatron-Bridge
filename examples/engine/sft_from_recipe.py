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
"""SFT demo: SFT through ``TrainingEngine`` on offline-packed data, logged to W&B for the baseline comparison.

Each iteration reads ``--packs-per-step`` packed rows (in file order), expands them to Datums, runs
``forward_backward`` in ``--calls-per-step`` chunks, and calls ``optim_step`` with the learning rate from
the recipe's warmup + cosine schedule computed here (the caller owns the schedule).

``--save-at N --save-dir ROOT`` saves an engine checkpoint after step N (``--async-save`` for the
asynchronous path, waited for at the end); ``--resume-from ROOT`` restores it into a fresh engine and
continues from the next step on the next data batch. The data cursor travels as ``user_state``.
With a LoRA engine, ``--check-resume-guards ROOT`` adds two resume checks: right after the resume,
``load_state`` must refuse the checkpoint under other LoRA settings before reading anything; and after
the run, a save to ``ROOT`` must be what a following ``load_state(ROOT)`` restores, although an earlier
save to ``ROOT`` was already loaded in the same process.

``--check-accumulation`` instead builds two engines from the same checkpoint and applies one update to
each from the same packs, as one call and as four calls, then compares the accumulated gradient buffers
before the step, the gradient norms, and the resulting parameters (the strict accumulation gate). The
buffer comparison is the decisive one: a first Adam step is sign-only, so parameters alone would not
tell gradients of equal norm and sign apart. The strict comparison uses ``PackReplayEngine``,
whose ``preprocess`` keeps each original parquet pack as one microbatch, so both paths run identical
microbatches and only the accumulation mechanics differ; the default re-packing engine is compared
too, as an indication of bf16 noise from different packing.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections.abc import Iterator
from typing import Any

import pyarrow.parquet as pq
import torch

import megatron.bridge.engine.recipes as recipes
from megatron.bridge.engine import AdamParams, Datum, LoraConfig, PreparedMicrobatch, TrainingEngine, check_datums
from megatron.bridge.engine.batching import assign_to_dp_ranks, filler_microbatch, pack_sequences, shard_for_cp
from megatron.bridge.engine.datum import LORA_EXPERT_ADAPTER_LAYOUTS
from megatron.bridge.utils.common_utils import print_rank_0


logger = logging.getLogger(__name__)


class PackReplayEngine(TrainingEngine):
    """Custom preprocessing: one microbatch per original parquet pack (``Datum.sample_inputs['pack']``).

    Shows the override seam: group Datums your own way, then reuse the engine's public blocks for
    materialization, DP distribution and CP sharding. The engine still validates the result.
    """

    def preprocess(self, data, *, training: bool) -> list[PreparedMicrobatch]:
        """Group by original pack, keep file order, deal to DP replicas, shard for CP."""
        datums = check_datums(data)
        by_pack: dict[int, list[int]] = {}
        for index, datum in enumerate(datums):
            by_pack.setdefault(int(datum.sample_inputs["pack"]), []).append(index)
        groups = [by_pack[key] for key in sorted(by_pack)]
        dp, cp = self.process_groups.dp, self.process_groups.cp
        device = torch.device("cuda", torch.cuda.current_device())
        microbatches = []
        seq_length = self.recipe.model.seq_length
        pad_to_max_length = self.recipe.pad_to_max_length
        filler_width = seq_length if pad_to_max_length else None
        for group in assign_to_dp_ranks(groups, dp.size())[dp.rank()]:
            if group is None:
                mb = filler_microbatch(
                    pad_multiple=self.pad_multiple,
                    device=device,
                    layout="thd",
                    width=filler_width,
                    schema_from=datums[0],
                )
            else:
                mb = pack_sequences(
                    datums,
                    group,
                    seq_length=seq_length,
                    pad_multiple=self.pad_multiple,
                    device=device,
                    pad_to_max_length=pad_to_max_length,
                )
            microbatches.append(shard_for_cp(mb, cp))
        return microbatches


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--recipe-kwargs", default="{}")
    parser.add_argument("--overrides", default="{}", help="Recipe fields by dotted path, as JSON")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parquet", required=True, help="Offline-packed training parquet")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--packs-per-step", type=int, default=32)
    parser.add_argument("--calls-per-step", type=int, default=1)
    parser.add_argument("--max-lr", type=float, default=2e-5)
    parser.add_argument("--min-lr", type=float, default=2e-6)
    parser.add_argument("--warmup-iters", type=int, default=200)
    parser.add_argument("--decay-iters", type=int, default=300000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--check-accumulation", action="store_true")
    parser.add_argument("--save-at", type=int, default=0, help="Save an engine checkpoint after this step (0: never)")
    parser.add_argument("--save-dir", default=None, help="Checkpoint root for --save-at")
    parser.add_argument("--async-save", action="store_true", help="Use the asynchronous save for --save-at")
    parser.add_argument("--resume-from", default=None, help="Engine checkpoint root to resume from")
    parser.add_argument("--export-dir", default=None, help="After training, export HF weights here and verify")
    parser.add_argument(
        "--base-model-name-or-path",
        default=None,
        help="HF ID or directory of the model --checkpoint holds, recorded as the adapters' base in the adapter "
        "export (needed by --export-dir with --lora-rank)",
    )
    parser.add_argument("--lora-rank", type=int, default=0, help="Train LoRA adapters of this rank (0: full weights)")
    parser.add_argument("--lora-alpha", type=float, default=64.0, help="LoRA scaling numerator (alpha / rank)")
    parser.add_argument(
        "--lora-expert-adapters",
        choices=sorted(LORA_EXPERT_ADAPTER_LAYOUTS),
        default="shared",
        help="LoRA layout on grouped MoE expert linears (LoraConfig.expert_adapters)",
    )
    parser.add_argument(
        "--check-resume-guards",
        default=None,
        metavar="ROOT",
        help="With --resume-from and --lora-rank: check the LoRA-settings refusal and a reload after a newer save",
    )
    parser.add_argument(
        "--check-disable-adapters",
        action="store_true",
        help="LoRA gate: after --iters updates, adapters-off scores must equal a frozen engine on the same base",
    )
    return parser.parse_args()


def lora_config(args: argparse.Namespace) -> LoraConfig | None:
    """``--lora-rank`` / ``--lora-alpha`` / ``--lora-expert-adapters`` as the engine's LoRA settings, or ``None``
    for full-weight training."""
    if not args.lora_rank:
        return None
    return LoraConfig(rank=args.lora_rank, alpha=args.lora_alpha, expert_adapters=args.lora_expert_adapters)


def check_lora_settings_guard(engine: TrainingEngine, path: str) -> None:
    """LoRA resume gate: ``load_state`` refuses a checkpoint saved with other LoRA settings, before reading it.

    The engine's recorded alpha is changed for the call only. Adapters keep their own scale, so the model is
    untouched, and the refused load must leave the update count where it was.
    """
    peft, updates = engine._peft, engine.step
    alpha = peft.alpha
    peft.alpha = alpha * 2
    try:
        engine.load_state(path)
    except ValueError as error:
        assert "was saved with lora=" in str(error), error
    else:
        raise AssertionError("load_state accepted a checkpoint saved with another LoRA alpha")
    finally:
        peft.alpha = alpha
    assert engine.step == updates, f"the refused load moved the update count to {engine.step}"
    print_rank_0("[guard] load_state refused other LoRA settings before reading: PASSED")


def copy_trainable_weights(source: TrainingEngine, target: TrainingEngine) -> None:
    """Give ``target`` the trainable weights of ``source`` and refresh its optimizer's fp32 masters from them.

    The two-engine gates need identical starting states. Base weights come from the checkpoint, but LoRA's
    lora_A is drawn from the RNG at build time, and the second engine built in a process draws other values.
    """
    with torch.no_grad():
        for chunk_source, chunk_target in zip(source.model, target.model):
            pairs = zip(chunk_source.named_parameters(), chunk_target.named_parameters())
            for (name_source, p_source), (name_target, p_target) in pairs:
                assert name_source == name_target, f"parameter order differs: {name_source} vs {name_target}"
                if p_source.requires_grad:
                    p_target.copy_(p_source)
    target.optimizer.reload_model_params()


def check_adapter_export(engine: TrainingEngine, adapter_dir: str, base_model_name_or_path: str) -> None:
    """Adapter-only export: a PEFT directory, then the adapter stream, rewritten into the same PEFT layout,
    compared tensor by tensor against the file (rank 0). Streamed names must be distinct: the PEFT
    conversion keeps one tensor per name, so a repeated name would lose data on both sides alike."""
    import os
    from collections import Counter

    from safetensors import safe_open

    from megatron.bridge.models.conversion.model_bridge import HFWeightTuple
    from megatron.bridge.models.conversion.peft_bridge import convert_adapter_weights_to_peft_state

    start = time.time()
    engine.export(path=adapter_dir, adapter_only=True, base_model_name_or_path=base_model_name_or_path)
    torch.distributed.barrier()
    print_rank_0(f"[export] adapter directory {adapter_dir} written in {time.time() - start:.1f}s")
    rank0 = torch.distributed.get_rank() == 0
    streamed, families = [], {}
    for name, tensor in engine.export(streaming=True, adapter_only=True):
        if not rank0:
            continue
        streamed.append(HFWeightTuple(name, tensor.detach().cpu().clone()))
        stem = name.split(".lora_")[0]
        family = ".".join(part for part in stem.split(".")[-2:] if not part.isdigit())
        families[family] = families.get(family, 0) + 1
    if rank0:
        counts = Counter(item.param_name for item in streamed)
        repeated = sorted(name for name, count in counts.items() if count > 1)
        assert not repeated, f"the adapter stream repeats {len(repeated)} names, e.g. {repeated[0]}"
        state, _, _ = convert_adapter_weights_to_peft_state(streamed)
        saved = safe_open(os.path.join(adapter_dir, "adapter_model.safetensors"), framework="pt")
        file_keys = set(saved.keys())
        missing = sorted(set(state) - file_keys)
        extra = sorted(file_keys - set(state))
        max_diff = max(
            (
                (state[key].float() - saved.get_tensor(key).float()).abs().max().item()
                for key in set(state) & file_keys
            ),
            default=0.0,
        )
        with open(os.path.join(adapter_dir, "adapter_config.json")) as handle:
            config = json.load(handle)
        print_rank_0(
            f"[export] adapters: {len(streamed)} streamed tensors -> {len(state)} PEFT tensors, {len(file_keys)} in "
            f"the file; per module: {families}; r={config.get('r')} lora_alpha={config.get('lora_alpha')} "
            f"targets={config.get('target_modules')}; max |stream - file|={max_diff:.3e}; "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
        assert streamed, "the adapter stream is empty"
        assert not missing and not extra, f"stream and file disagree: {len(missing)} missing, {len(extra)} extra"
        assert max_diff == 0.0, f"streamed and written adapters differ by {max_diff}"
    print_rank_0("ADAPTER EXPORT CHECK PASSED")


def _max_abs_diff(a: list[torch.Tensor], b: list[torch.Tensor]) -> tuple[float, float]:
    """Max and mean |a - b| over every token of every Datum (scores are identical on every rank)."""
    deltas = torch.cat([(x.float() - y.float()).abs().flatten() for x, y in zip(a, b)])
    return deltas.max().item(), deltas.mean().item()


def check_disable_adapters(args: argparse.Namespace, recipe) -> None:
    """LoRA gate: ``forward_only(disable_adapters=True)`` scores exactly what a frozen engine on the base scores.

    Before any update lora_B is zero, so adapters on and off must already agree exactly. After ``--iters``
    updates at the constant rate ``--max-lr`` they must differ with the adapters on, and the adapters-off
    scores must still equal a separately built frozen engine on the same checkpoint, token for token.
    """
    if not args.lora_rank:
        raise SystemExit("--check-disable-adapters needs --lora-rank")
    batches = iter_pack_batches(args.parquet, args.packs_per_step)
    scoring = next(batches)
    policy = TrainingEngine.from_recipe(
        recipe, checkpoint_path=args.checkpoint, trainable=True, lora=lora_config(args)
    )
    on = policy.forward_only(scoring).logprobs
    off = policy.forward_only(scoring, disable_adapters=True).logprobs
    initial_max, _ = _max_abs_diff(on, off)
    print_rank_0(f"[adapters] before any update: max |on - off|={initial_max:.3e}")
    assert initial_max == 0.0, "zero-initialized adapters changed the scores"
    for step in range(1, args.iters + 1):
        outs = [policy.forward_backward(chunk) for chunk in _chunks(next(batches), args.calls_per_step)]
        result = policy.optim_step(AdamParams(learning_rate=args.max_lr, weight_decay=args.weight_decay))
        loss = sum(o.loss_sum for o in outs) / max(sum(o.num_loss_tokens for o in outs), 1)
        print_rank_0(f"[adapters] update {step}: lm loss {loss:.4f} grad_norm {result['grad_norm']:.4f}")
        assert result["update_successful"], f"update {step} was skipped"
    on = policy.forward_only(scoring).logprobs
    off = policy.forward_only(scoring, disable_adapters=True).logprobs
    frozen = TrainingEngine.from_recipe(recipe, checkpoint_path=args.checkpoint, trainable=False)
    base = frozen.forward_only(scoring).logprobs
    frozen.close()
    off_max, off_mean = _max_abs_diff(off, base)
    on_max, on_mean = _max_abs_diff(on, base)
    print_rank_0(
        f"[adapters] after {args.iters} updates: adapters off vs frozen engine max={off_max:.3e} mean={off_mean:.3e}; "
        f"adapters on vs frozen engine max={on_max:.3e} mean={on_mean:.3e}"
    )
    assert off_max == 0.0, f"adapters-off scores differ from the frozen engine by up to {off_max}"
    assert on_max > 0.0, "trained adapters did not change any score"
    policy.close()
    print_rank_0("DISABLE ADAPTERS GATE PASSED")


def check_export(engine: TrainingEngine, export_dir: str) -> None:
    """Directory export, then streaming export compared tensor by tensor against the written files (rank 0)."""
    import json
    import os

    from safetensors import safe_open

    start = time.time()
    engine.export(path=export_dir)
    torch.distributed.barrier()
    print_rank_0(f"[export] directory {export_dir} written in {time.time() - start:.1f}s")
    rank0 = torch.distributed.get_rank() == 0
    weight_map: dict[str, str] = {}
    if rank0:
        index = os.path.join(export_dir, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index) as handle:
                weight_map = json.load(handle)["weight_map"]
        else:
            weight_map = {}
    start = time.time()
    count, max_diff, missing = 0, 0.0, []
    files: dict[str, Any] = {}
    for name, tensor in engine.export(streaming=True):
        count += 1
        if not rank0:
            continue
        file_name = weight_map.get(name, "model.safetensors")
        if file_name not in files:
            files[file_name] = safe_open(os.path.join(export_dir, file_name), framework="pt")
        if name not in files[file_name].keys():
            missing.append(name)
            continue
        saved = files[file_name].get_tensor(name)
        max_diff = max(max_diff, (tensor.detach().cpu().float() - saved.float()).abs().max().item())
    print_rank_0(
        f"[export] streamed {count} tensors in {time.time() - start:.1f}s; directory holds "
        f"{len(weight_map) if weight_map else 'n/a'}; max |stream - file|={max_diff:.3e}; missing={missing[:5]}"
    )
    if rank0:
        assert not missing, f"{len(missing)} streamed tensors are not in the directory export"
        assert max_diff == 0.0, f"streamed and written weights differ by {max_diff}"
        assert not weight_map or count == len(weight_map), "streamed and written tensor counts differ"
    print_rank_0("EXPORT CHECK PASSED")


def iter_pack_batches(path: str, packs_per_step: int, *, skip_steps: int = 0) -> Iterator[list[Datum]]:
    """Yield ``packs_per_step`` packed rows at a time as Datums, in file order (same on every rank).

    ``skip_steps`` drops that many leading batches, so a resumed run continues on the data its
    checkpointed predecessor had not consumed.
    """
    pack_index = 0
    for batch_index, batch in enumerate(pq.ParquetFile(path).iter_batches(batch_size=packs_per_step)):
        if batch_index < skip_steps:
            pack_index += batch.num_rows
            continue
        datums: list[Datum] = []
        for row in batch.to_pylist():
            ids = torch.tensor(row["input_ids"], dtype=torch.int64)
            mask = torch.tensor(row["loss_mask"], dtype=torch.float32)
            boundaries = list(row["seq_start_id"]) + [len(ids)]
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                if end - start >= 2:
                    datums.append(
                        Datum(
                            input_ids=ids[start : end - 1],
                            target_ids=ids[start + 1 : end],
                            loss_mask=mask[start : end - 1],
                            sample_inputs={"pack": pack_index},
                        )
                    )
            pack_index += 1
        yield datums


def learning_rate(step: int, args: argparse.Namespace) -> float:
    """Megatron's linear warmup then cosine decay, evaluated for 1-based ``step``."""
    if step <= args.warmup_iters:
        return args.max_lr * step / args.warmup_iters
    if step > args.decay_iters:
        return args.min_lr
    ratio = (step - args.warmup_iters) / (args.decay_iters - args.warmup_iters)
    return args.min_lr + 0.5 * (math.cos(math.pi * ratio) + 1.0) * (args.max_lr - args.min_lr)


def _chunks(datums: list[Datum], n: int) -> list[list[Datum]]:
    """Split into ``n`` chunks along original pack boundaries (never splits a pack across calls)."""
    packs = sorted({int(d.sample_inputs["pack"]) for d in datums})
    size = -(-len(packs) // n)
    chunks = []
    for i in range(0, len(packs), size):
        members = set(packs[i : i + size])
        chunks.append([d for d in datums if int(d.sample_inputs["pack"]) in members])
    return chunks


def _master_params(engine: TrainingEngine) -> list[torch.Tensor]:
    """FP32 master shards of the distributed optimizer (fall back to model parameters)."""
    optimizer = engine.optimizer
    inner = getattr(optimizer, "chained_optimizers", [optimizer])
    shards = [t for opt in inner for group in getattr(opt, "shard_fp32_from_float16_groups", []) for t in group]
    return shards or [p.detach() for p in engine.model[0].parameters()]


def _param_delta(a: TrainingEngine, b: TrainingEngine, *, lr: float) -> tuple[float, float]:
    """Over all FP32 master parameters, reduced over ranks: max |a - b| and the fraction of elements whose
    update differs materially (by more than 10% of an Adam step, i.e. a different update direction)."""
    diff = torch.zeros((), device="cuda")
    differing = torch.zeros((), device="cuda", dtype=torch.float64)
    total = torch.zeros((), device="cuda", dtype=torch.float64)
    for pa, pb in zip(_master_params(a), _master_params(b)):
        delta = (pa.detach().float() - pb.detach().float()).abs()
        diff = torch.maximum(diff, delta.max())
        differing += (delta > 0.1 * lr).sum()
        total += delta.numel()
    torch.distributed.all_reduce(diff, op=torch.distributed.ReduceOp.MAX)
    torch.distributed.all_reduce(differing)
    torch.distributed.all_reduce(total)
    return diff.item(), (differing / total).item()


def _grad_delta(a: TrainingEngine, b: TrainingEngine) -> tuple[float, float]:
    """Over the accumulated FP32 gradient buffers (``main_grad`` of every parameter, dense and expert):
    max |a - b| after summing each parameter's difference over the group DDP reduces it over, and that
    difference relative to the largest reduced gradient entry.

    Summing the difference over the data-parallel group (DP x CP for dense parameters, expert-DP for
    expert parameters) makes the comparison independent of which DP replica held which microbatch,
    so it is valid for the pack-replay engine and the re-packing engine alike."""
    groups = a.process_groups
    diff = torch.zeros((), device="cuda")
    scale = torch.zeros((), device="cuda")
    for chunk_a, chunk_b in zip(a.model, b.model):
        for pa, pb in zip(chunk_a.parameters(), chunk_b.parameters()):
            ga, gb = getattr(pa, "main_grad", None), getattr(pb, "main_grad", None)
            if ga is None or gb is None:
                assert ga is None and gb is None, "gradient buffers differ in layout"
                continue
            group = groups.dp_cp if getattr(pa, "allreduce", True) else groups.expt_dp
            delta = ga.float() - gb.float()
            total = ga.float().clone()
            if group is not None and group.size() > 1:
                torch.distributed.all_reduce(delta, group=group)
                torch.distributed.all_reduce(total, group=group)
            diff = torch.maximum(diff, delta.abs().max())
            scale = torch.maximum(scale, total.abs().max())
    torch.distributed.all_reduce(diff, op=torch.distributed.ReduceOp.MAX)
    torch.distributed.all_reduce(scale, op=torch.distributed.ReduceOp.MAX)
    return diff.item(), (diff / scale.clamp(min=1e-30)).item()


def _compare_one_vs_four(
    args: argparse.Namespace, recipe, engine_cls: type[TrainingEngine], datums: list[Datum], label: str
) -> tuple[float, float, float, float]:
    """Apply one update from identical state as 1 call and as 4 calls.

    Returns (grad rel diff, grad-norm rel diff, max param diff, fraction of params with a different update)."""
    adam = AdamParams(learning_rate=args.max_lr, weight_decay=args.weight_decay)
    lora = lora_config(args)
    one = engine_cls.from_recipe(recipe, checkpoint_path=args.checkpoint, trainable=True, lora=lora)
    four = engine_cls.from_recipe(recipe, checkpoint_path=args.checkpoint, trainable=True, lora=lora)
    if lora is not None:
        copy_trainable_weights(one, four)
    diff0, _ = _param_delta(one, four, lr=adam.learning_rate)
    assert diff0 == 0.0, f"engines differ before training: {diff0}"

    out1 = one.forward_backward(datums)
    outs4 = [four.forward_backward(chunk) for chunk in _chunks(datums, 4)]
    grad_diff, grad_rel = _grad_delta(one, four)
    print_rank_0(f"[accum:{label}] DP-reduced grad sums: max |diff|={grad_diff:.3e} (relative {grad_rel:.3e})")
    step1 = one.optim_step(adam)
    step4 = four.optim_step(adam)
    tokens4 = sum(o.num_loss_tokens for o in outs4)
    loss4 = sum(o.loss_sum for o in outs4)
    print_rank_0(
        f"[accum:{label}] 1 call : loss_sum={out1.loss_sum:.4f} tokens={out1.num_loss_tokens} "
        f"grad_norm={step1['grad_norm']:.6f}"
    )
    print_rank_0(f"[accum:{label}] 4 calls: loss_sum={loss4:.4f} tokens={tokens4} grad_norm={step4['grad_norm']:.6f}")
    assert out1.num_loss_tokens == tokens4, "token counts differ between 1-call and 4-call accumulation"
    assert step1["update_successful"] and step4["update_successful"]
    rel = abs(step1["grad_norm"] - step4["grad_norm"]) / max(step1["grad_norm"], 1e-12)
    diff, fraction = _param_delta(one, four, lr=adam.learning_rate)
    print_rank_0(
        f"[accum:{label}] grad_norm rel diff={rel:.3e}; max |master param diff|={diff:.3e} "
        f"(2*lr={2 * adam.learning_rate:.1e}); fraction with a different update={fraction:.3e}"
    )
    one.close()
    four.close()
    return grad_rel, rel, diff, fraction


def check_accumulation(args: argparse.Namespace, recipe) -> None:
    """Strict gate on identical microbatches (pack replay), plus the re-packing comparison for information."""
    datums = next(iter_pack_batches(args.parquet, args.packs_per_step))
    grad_rel, rel, diff, fraction = _compare_one_vs_four(args, recipe, PackReplayEngine, datums, "same-packs")
    assert grad_rel < 1e-4, f"reduced gradients differ by {grad_rel:.3e} (relative) on identical microbatches"
    assert rel < 1e-4, f"grad norm differs by {rel:.3e} on identical microbatches"
    assert fraction < 1e-5, f"{fraction:.3e} of elements took a different update on identical microbatches"
    _compare_one_vs_four(args, recipe, TrainingEngine, datums, "repacked")
    print_rank_0("ACCUMULATION GATE PASSED")


def train(args: argparse.Namespace, recipe) -> None:
    """The engine-driven reproduction of the baseline schedule."""
    adapter_export = bool(args.export_dir and args.lora_rank and args.lora_expert_adapters != "outer_shared")
    if adapter_export and not args.base_model_name_or_path:
        raise SystemExit("--export-dir with --lora-rank needs --base-model-name-or-path for the adapter export")
    engine = TrainingEngine.from_recipe(
        recipe, checkpoint_path=args.checkpoint, trainable=True, lora=lora_config(args)
    )
    print_rank_0(f"[engine] built; parallel_info={engine.parallel_info}")
    run = None
    if args.wandb_project and torch.distributed.get_rank() == 0:
        import wandb

        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name, config=vars(args))

    first_step = 1
    if args.resume_from:
        # The data cursor and the schedule step come from the saved user state. engine.step counts
        # successful updates only, so it falls behind after a skipped update and is not a data cursor.
        user_state = engine.load_state(args.resume_from)
        assert user_state is not None and "consumed_steps" in user_state, f"no data cursor in {user_state!r}"
        first_step = user_state["consumed_steps"] + 1
        print_rank_0(f"[ckpt] resumed from {args.resume_from}: updates {engine.step}, user_state {user_state}")
    if args.check_resume_guards:
        if not (args.resume_from and args.lora_rank):
            raise SystemExit("--check-resume-guards needs --resume-from and --lora-rank")
        check_lora_settings_guard(engine, args.resume_from)
        # A first save and load of the guard root, so a cached read of its tracker would be stale later.
        engine.save_state(args.check_resume_guards, user_state={"consumed_steps": first_step - 1})
        engine.load_state(args.check_resume_guards)
    batches = iter_pack_batches(args.parquet, args.packs_per_step, skip_steps=first_step - 1)
    handle = None
    for step in range(first_step, args.iters + 1):
        datums = next(batches)
        start = time.time()
        outs = [engine.forward_backward(chunk) for chunk in _chunks(datums, args.calls_per_step)]
        lr = learning_rate(step, args)
        result = engine.optim_step(AdamParams(learning_rate=lr, weight_decay=args.weight_decay))
        torch.cuda.synchronize()
        elapsed = time.time() - start
        loss_sum = sum(o.loss_sum for o in outs)
        tokens = sum(o.num_loss_tokens for o in outs)
        metrics = {
            "lm loss": loss_sum / max(tokens, 1),
            "loss_sum": loss_sum,
            "num_loss_tokens": tokens,
            "num_datums": len(datums),
            "grad_norm": result["grad_norm"],
            "learning_rate": lr,
            "iteration_time": elapsed,
            "update_successful": result["update_successful"],
        }
        print_rank_0(
            f"[train] step {step}: lm loss {metrics['lm loss']:.4f} tokens {tokens} "
            f"grad_norm {result['grad_norm']:.4f} lr {lr:.3e} time {elapsed:.1f}s"
        )
        if run is not None:
            run.log(metrics, step=step)
        if args.save_at and step == args.save_at:
            start = time.time()
            handle = engine.save_state(args.save_dir, user_state={"consumed_steps": step}, async_save=args.async_save)
            print_rank_0(f"[ckpt] save_state({args.save_dir}) at step {step} issued in {time.time() - start:.1f}s")
    if handle is not None:
        start = time.time()
        handle.wait()
        print_rank_0(f"[ckpt] checkpoint {handle.path} durable; wait took {time.time() - start:.1f}s")
    if args.check_resume_guards:
        updates = engine.step
        engine.save_state(args.check_resume_guards, user_state={"consumed_steps": args.iters})
        engine.load_state(args.check_resume_guards)
        assert engine.step == updates, f"a load after a newer save restored update {engine.step}, not {updates}"
        print_rank_0(f"[guard] load_state after a newer save to the same root restored update {updates}: PASSED")
    if args.export_dir:
        check_export(engine, args.export_dir)
        if adapter_export:
            check_adapter_export(engine, args.export_dir.rstrip("/") + "-adapter", args.base_model_name_or_path)
        elif args.lora_rank:
            print_rank_0("[export] no adapter-only export: not implemented for outer_shared expert adapters yet")
    if run is not None:
        run.finish()
    engine.close()
    print_rank_0("TRAINING RUN FINISHED")


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    recipe = getattr(recipes, args.recipe)(**json.loads(args.recipe_kwargs))
    recipes.apply_recipe_overrides(recipe, json.loads(args.overrides))
    if args.check_accumulation:
        check_accumulation(args, recipe)
    elif args.check_disable_adapters:
        check_disable_adapters(args, recipe)
    else:
        train(args, recipe)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
