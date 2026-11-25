#!/usr/bin/env python3
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

"""Record two native ERNIE optimizer updates for the eight-document 256K fixture."""

import argparse
import functools
import hashlib
import importlib
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from megatron.core import parallel_state as ps


SHAPES = {"lm_head": (201216, 4096), "embedding": (201216, 4096), "last_shared_fc2": (4096, 2048)}
NAMES = {
    "embedding.word_embeddings.weight": "embedding",
    "output_layer.weight": "lm_head",
    "decoder.layers.42.mlp.shared_experts.linear_fc2.weight": "last_shared_fc2",
    "decoder.layers.85.inner_layer.mlp.shared_experts.linear_fc2.weight": "last_shared_fc2",
}


def write_new(path, fields):
    """Create a rank record exclusively to catch reused capture paths."""
    with path.open("x") as file:
        json.dump(fields, file)


def indices(name, start, end):
    """Select absolute parameter coordinates within an optimizer shard."""
    selected = np.arange(start + (-start) % 1021, end, 1021, dtype=np.int64)
    if name == "embedding":
        bos = np.arange(max(4096, start), min(8192, end), dtype=np.int64)
        selected = np.union1d(selected, bos)
    return selected


def track_batches(role, batches):
    """Map actual contiguous or per-document zigzag CP rows back to physical positions."""

    def observe(tokens, labels, positions, packed):
        def cpu(x):
            return x.detach().reshape(-1).cpu().numpy().astype(np.int32)

        bounds = getattr(packed, "cu_seqlens_q_padded", None)
        bounds = cpu(packed.cu_seqlens_q if bounds is None else bounds)
        if not np.array_equal(bounds, np.arange(0, 262145, 32768)) or len(batches) >= 8:
            raise ValueError("Expected eight equal documents and eight microbatches per update.")
        cp = ps.get_context_parallel_rank()
        if role == "swift":
            physical = np.arange(cp * 16384, (cp + 1) * 16384)
        else:
            physical = np.concatenate(
                [
                    np.arange(start + part * 1024, start + (part + 1) * 1024)
                    for start in bounds[:-1]
                    for part in (cp, 31 - cp)
                ]
            )
        if not np.array_equal(cpu(positions), physical % 32768):
            raise ValueError("Observed CP positions differ from the eight-document fixture.")
        batches.append((physical.astype(np.int32), cpu(tokens), cpu(labels)))

    if role == "swift":
        from mcore_bridge.model.gpts.ernie5_5 import Ernie5_5GPTModel

        original = Ernie5_5GPTModel.forward
        signature = inspect.signature(original)

        @functools.wraps(original)
        def forward(*args, **kwargs):
            fields = signature.bind(*args, **kwargs).arguments
            observe(fields["input_ids"], fields["labels"], fields["position_ids"], fields["packed_seq_params"])
            return original(*args, **kwargs)

        Ernie5_5GPTModel.forward = forward
        return Ernie5_5GPTModel, "forward", original
    else:
        from megatron.bridge.models.ernie import ernie55_step

        original = ernie55_step.get_batch

        @functools.wraps(original)
        def get_batch(*args, **kwargs):
            batch = original(*args, **kwargs)
            observe(batch.tokens, batch.labels, batch.position_ids, batch.packed_seq_params)
            return batch

        ernie55_step.get_batch = get_batch
        return ernie55_step, "get_batch", original


def optimizer_nodes(optimizer):
    """Walk the native chained optimizer and its component optimizers."""
    yield optimizer
    for child in getattr(optimizer, "chained_optimizers", ()):
        yield from optimizer_nodes(child)


def master_shards(model_parts, optimizer):
    """Resolve each observed FP32 master to global row-major parameter coordinates."""
    named = {}
    for part in model_parts:
        for name, param in part.named_parameters(remove_duplicate=False):
            while name.startswith("module."):
                name = name.removeprefix("module.")
            if name in NAMES:
                key = NAMES[name]
                if key in named and param is not named[key]:
                    raise ValueError(f"Ambiguous parameter {key}")
                named[key] = param
    if {k: tuple(p.shape) for k, p in named.items()} != SHAPES:
        raise ValueError("This probe requires the released Base checkpoint and all 43 blocks.")
    if named["lm_head"].data_ptr() == named["embedding"].data_ptr():
        raise ValueError("Expected independent embedding and LM weights.")
    logical = {id(p): k for k, p in named.items()}
    result = {}
    fields = [
        ("float16_groups", "fp32_from_float16_groups"),
        ("model_float16_groups", "shard_fp32_from_float16_groups"),
        ("model_fp32_groups", "shard_fp32_groups"),
        ("model_float16_groups", "shard_float16_groups"),
    ]
    for node in optimizer_nodes(optimizer):
        if getattr(node, "chained_optimizers", ()):
            continue
        groups = {id(m): group for group in node.param_groups for m in group["params"]}
        mapped = [(p, p) for p in named.values()]
        for field, master_field in fields:
            for model_group, master_group in zip(
                getattr(node, field, ()) or (), getattr(node, master_field, ()) or ()
            ):
                mapped.extend(zip(model_group, master_group))
        for param, master in mapped:
            if id(param) not in logical or id(master) not in groups or id(master) in result:
                continue
            name = logical[id(param)]
            start, end = 0, param.numel()
            if param in getattr(node, "model_param_gbuf_map", {}):
                span = node._get_model_param_range_map(param)["param"]
                start, end = span.start, span.end
            if master.numel() != end - start or master.dtype != torch.float32:
                raise ValueError(f"Expected an FP32 master covering {name}[{start}:{end}].")
            result[id(master)] = (name, master, start, end, groups[id(master)], node.optimizer)
    return list(result.values())


def shard_owners(shards):
    """Require complete logical coverage and choose one copy of each replicated range."""
    local = {name: (start, end) for name, _, start, end, _, _ in shards}
    if len(local) != len(shards):
        raise ValueError("Unsupported multiple local shards of the same parameter.")
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    primary = {}
    for name, shape in SHAPES.items():
        ranks = {}
        for rank, record in enumerate(gathered):
            if name in record:
                ranks.setdefault(record[name], []).append(rank)
        cursor = 0
        for (start, end), owners in sorted(ranks.items()):
            if cursor != start or start >= end:
                raise ValueError(f"Missing or overlapping {name} shards at {start}.")
            cursor = end
            if dist.get_rank() in owners:
                primary[name] = owners[0] == dist.get_rank()
        if cursor != int(np.prod(shape)):
            raise ValueError(f"Incomplete master coverage for {name}.")
    return primary


def sample_tensor(value, coordinate):
    """Read an FP32 master, reduced gradient or optimizer state at shard-local indices."""
    return value.detach().reshape(-1).index_select(0, coordinate).float().cpu().numpy()


def state_signature(optimizer, master, coordinate):
    """Fingerprint the optimizer moments that persist from the first step into the second."""
    result = {}
    for name, value in getattr(optimizer, "state", {}).get(master, {}).items():
        if torch.is_tensor(value):
            if value.numel() == 1:
                result[str(name)] = float(value.item())
            elif value.numel() == master.numel():
                sample = sample_tensor(value, coordinate.to(value.device))
                result[str(name)] = hashlib.sha256(sample.tobytes()).hexdigest()
        elif isinstance(value, (int, float)):
            result[str(name)] = value
    return result


class Recorder:
    """Observe one native optimizer and its two uninterrupted GA8 steps."""

    def __init__(self, role, output, models, optimizer):
        self.role, self.output = role, output
        self.step = 0
        self.batches, self.observed, self.values = [], {}, {}
        self.shards = master_shards(models, optimizer)
        self.primary = shard_owners(self.shards)
        self.global_indices = {name: indices(name, start, end) for name, _, start, end, _, _ in self.shards}
        self.local_indices = {
            name: torch.as_tensor(self.global_indices[name] - start, device=master.device)
            for name, master, start, _, _, _ in self.shards
        }
        self.rank = dist.get_rank()
        if dist.get_world_size() != 64 or ps.get_context_parallel_world_size() != 16:
            raise ValueError("Expected 64 GPUs and CP16.")
        self.output.mkdir(exist_ok=True, parents=True)
        write_new(self.output / f"initial-{self.rank:02}.json", {"rank": self.rank, "role": role})
        self.hooks = [track_batches(role, self.batches)]
        # Observe the same FP32 gradient tensors the trainer sends to its native norm calculation.
        for node in optimizer_nodes(optimizer):
            original = node.get_grads_for_grad_norm

            @functools.wraps(original)
            def observe(*args, _original=original, **kwargs):
                grads = _original(*args, **kwargs)
                self.capture(grads)
                return grads

            self.hooks.append((node, "get_grads_for_grad_norm", original))
            node.get_grads_for_grad_norm = observe
        original_step = optimizer.step

        @functools.wraps(original_step)
        def step(*args, **kwargs):
            if len(self.batches) != 8 or self.step >= 2:
                raise ValueError("The probe requires exactly two sequential GA8 optimizer updates.")
            self.step += 1
            pre_state = self.states()
            rates = {name: {"lr": float(group["lr"])} for name, _, _, _, group, _ in self.shards}
            result = original_step(*args, **kwargs)
            if not result[0] or self.observed.keys() != self.primary.keys():
                raise RuntimeError("The optimizer skipped its update or a preclip gradient was missing.")
            folder = self.output / f"step{self.step}"
            folder.mkdir(exist_ok=True)
            for name, master, *_ in self.shards:
                if self.primary[name]:
                    values = self.values[name]
                    values["post"] = sample_tensor(master, self.local_indices[name])
                    with (folder / f"{self.rank:02}-{name}.npz").open("xb") as stream:
                        np.savez_compressed(stream, **values)
            with (folder / f"{self.rank:02}-batches.npz").open("xb") as stream:
                np.savez_compressed(
                    stream,
                    **{
                        f"{field}{i}": values[j]
                        for i, values in enumerate(self.batches)
                        for j, field in enumerate(("pos", "token", "label"))
                    },
                )
            write_new(
                folder / f"complete-{self.rank:02}.json",
                {
                    "rank": self.rank,
                    "role": role,
                    "dp": ps.get_data_parallel_rank(with_context_parallel=False),
                    "cp": ps.get_context_parallel_rank(),
                    "update_success": True,
                    "mb": 8,
                    "optimizer_step_count": self.step,
                    "logical": {k: {"shape": v} for k, v in SHAPES.items()},
                    "grad_norm": float(result[1]),
                    "loss_scale": float(optimizer.get_loss_scale().item()),
                    "param_group": rates,
                    "pre_state": pre_state,
                    "post_state": self.states(),
                    "observed": self.observed,
                },
            )
            self.batches.clear()
            self.values.clear()
            self.observed.clear()
            return result

        self.hooks.append((optimizer, "step", original_step))
        optimizer.step = step

    def close(self):
        """Restore training methods before running the next microbatch with updated weights."""
        for owner, name, previous in reversed(self.hooks):
            setattr(owner, name, previous)
        self.hooks.clear()

    def states(self):
        return {
            name: state_signature(opt, master, self.local_indices[name]) for name, master, _, _, _, opt in self.shards
        }

    def capture(self, gradients):
        included = {(getattr(g, "_local_tensor", g).data_ptr(), g.numel()) for g in gradients}
        for name, master, start, end, _, _ in self.shards:
            if name in self.observed:
                continue
            for field in ("grad", "decoupled_grad"):
                grad = getattr(master, field, None)
                grad = getattr(grad, "_local_tensor", grad)
                if grad is None or (grad.data_ptr(), grad.numel()) not in included:
                    continue
                if grad.dtype != torch.float32:
                    raise TypeError(f"{name} gradient is {grad.dtype}. Expected FP32.")
                ix = self.local_indices[name]
                self.values[name] = dict(
                    index=self.global_indices[name], pre=sample_tensor(master, ix), grad=sample_tensor(grad, ix)
                )
                self.observed[name] = dict(start=start, end=end, primary=self.primary[name])
                break


def install(role, output, next_heads=False):
    """Attach to the native training step and retain its optimizer across both updates."""
    if role == "swift":
        from swift.megatron.trainers.base import BaseMegatronTrainer

        owner = BaseMegatronTrainer
    else:
        owner = importlib.import_module("megatron.bridge.training.train")
    original = owner.train_step
    signature = inspect.signature(original)
    recorded_optimizer = None
    audit = None
    save_heads = None

    @functools.wraps(original)
    def train_step(*args, **kwargs):
        nonlocal recorded_optimizer, audit, save_heads
        if save_heads is not None:
            return original(*args, **kwargs)
        if role == "swift":
            trainer = args[0]
            optimizer, models = trainer.optimizer, trainer.wrapped_models
        else:
            bound = signature.bind(*args, **kwargs).arguments
            optimizer, models = bound["optimizer"], bound["model"]
        if audit is None:
            recorded_optimizer = optimizer
            audit = Recorder(role, output, models, optimizer)
        elif optimizer is not recorded_optimizer:
            raise ValueError("Optimizer identity changed between updates.")
        count = audit.step
        result = original(*args, **kwargs)
        if audit.step != count + 1:
            raise RuntimeError("Native optimizer did not take exactly one update.")
        if audit.step == 2:
            if next_heads:
                audit.close()
                if role == "swift":
                    from swift_head_parity_256k import HeadCapture

                    capture = HeadCapture(output / "heads")
                    capture.install()
                    save_heads = capture.save
                else:
                    from head_parity_256k import install as record_heads

                    save_heads = record_heads(output / "heads")
            elif role == "swift":
                raise RecordedSwiftUpdates
        return result

    owner.train_step = train_step

    def finish():
        if next_heads:
            if save_heads is None:
                raise RuntimeError("The trainer ended before its second optimizer update.")
            save_heads()

    return finish


class RecordedSwiftUpdates(BaseException):
    """Stop the bounded Swift capture after its second native optimizer and scheduler call."""


def main():
    """Run a bounded original Swift or public Bridge optimizer capture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--next-heads",
        action="store_true",
        help="Capture both heads on the first microbatch after the two optimizer updates. Set train_iters to 3.",
    )
    parser.add_argument("role", choices=("bridge", "swift"))
    parser.add_argument("output", type=Path)
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    tail = args.training_args[1:] if args.training_args[:1] == ["--"] else args.training_args
    if int(os.environ.get("WORLD_SIZE", 0)) != 64:
        parser.error("Use torchrun on all 64 GPUs.")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if args.role == "bridge":
        finish_capture = install(args.role, args.output, args.next_heads)
        source = Path(__file__).resolve().parents[3] / "scripts/training/run_recipe.py"
        sys.path.insert(0, str(source.parent))
        from run_recipe import main as run_recipe

        run_recipe(
            [
                "--recipe",
                "ernie55_pretrain_256k_64gpu_gb300_bf16_config",
                "--mode",
                "pretrain",
                "--step_func",
                "ernie55_step",
                *tail,
            ],
            bootstrap_entrypoint=(Path(__file__).resolve(), sys.argv[1:]),
        )
        finish_capture()
    else:
        from megatron.core.distributed import DistributedDataParallel
        from swift.megatron import megatron_sft_main
        from swift_head_parity_256k import CaptureComplete

        # Load the original pipeline and its argument definitions before importing trainer adapters.
        finish_capture = install(args.role, args.output, args.next_heads)

        torch.use_deterministic_algorithms(True)
        # Release allocator cache once before the original synchronization on this 288 GB setup.
        original = DistributedDataParallel.finish_grad_sync
        first = True

        @functools.wraps(original)
        def finish(module, *args, **kwargs):
            nonlocal first
            if first:
                first = False
                torch.cuda.empty_cache()
            return original(module, *args, **kwargs)

        DistributedDataParallel.finish_grad_sync = finish
        try:
            sys.argv[1:] = tail
            megatron_sft_main()
            raise RuntimeError("Original Swift returned before both optimizer updates were recorded.")
        except (RecordedSwiftUpdates, CaptureComplete):
            finish_capture()
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
