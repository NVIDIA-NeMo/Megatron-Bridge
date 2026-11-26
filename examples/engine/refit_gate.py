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
"""Refit gate: a TrainingEngine on M GPUs pushes its weights to N rollout ranks on other GPUs.

The trainer perturbs every weight (so nothing matches the checkpoint by accident), refits a few times,
then streams ``export()`` as the reference. Each rollout rank checks every tensor it received, bit for
bit, against that reference (per-expert tensors against slices of the packed HF expert tensors), and
rollout rank 0 checks that the plan covers every exported tensor. With ``--vllm`` each rollout rank is a
vLLM engine (TP 1) that loads the refit through ``reload_weights``; its parameters must then equal what the
same engine holds after loading the reference export. The rollout side never imports Megatron: it loads
``refit.py`` by path, and checks ``sys.modules`` at the end. Run both sides at once, e.g.::

    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc-per-node 2 refit_gate.py trainer \\
        --hf-path /path/to/qwen35_moe_text --port 29700 --rollout-world 2 --tp 2 --ep 2 --out /tmp/gate &
    for r in 0 1; do CUDA_VISIBLE_DEVICES=$((r + 2)) python refit_gate.py rollout --rank $r \\
        --port 29700 --rollout-world 2 --out /tmp/gate & done; wait
"""

from __future__ import annotations

import argparse
import faulthandler
import importlib.util
import logging
import os
import re
import sys
import time
import zlib
from pathlib import Path
from types import ModuleType

import torch


logger = logging.getLogger("refit_gate")
_PACKED_EXPERTS = re.compile(r"(.*\.experts)\.(gate_up_proj|down_proj)$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("role", choices=("trainer", "rollout"))
    parser.add_argument("--hf-path", help="Qwen3.5-MoE text checkpoint (trainer)")
    parser.add_argument("--address", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--rank", type=int, default=0, help="rollout rank")
    parser.add_argument("--rollout-world", type=int, required=True)
    parser.add_argument("--rollout-tp", type=int, default=1)
    parser.add_argument("--rollout-ep", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1, help="trainer tensor parallelism")
    parser.add_argument("--ep", type=int, default=1, help="trainer expert parallelism")
    parser.add_argument("--refits", type=int, default=3)
    parser.add_argument("--bucket-mb", type=int, default=64)
    parser.add_argument("--out", required=True, help="directory for the reference export")
    parser.add_argument("--vllm", action="store_true", help="rollout ranks are vLLM engines on --hf-path")
    parser.add_argument("--pp", type=int, default=1, help="trainer pipeline parallelism")
    parser.add_argument("--lora", action="store_true", help="trainer is a LoRA engine (merged refit)")
    parser.add_argument("--fp8", action="store_true", help="trainer keeps FP8 primary weights (TE current scaling)")
    parser.add_argument("--timeout", type=float, default=300, help="refit timeout in seconds")
    parser.add_argument(
        "--absent-rank", type=int, default=-1, help="this rollout rank joins, then never receives: expect timeouts"
    )
    return parser.parse_args()


def _standalone_refit() -> ModuleType:
    """Load refit.py by path, the way a rollout process without the megatron.bridge package does."""
    path = Path(__file__).resolve().parents[2] / "src/megatron/bridge/engine/refit.py"
    spec = importlib.util.spec_from_file_location("bridge_refit", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up while the class body runs
    spec.loader.exec_module(module)
    return module


def _layout(refit: ModuleType, args: argparse.Namespace):
    return refit.RolloutLayout(args.rollout_world, args.rollout_tp, args.rollout_ep)


def _expand(reference: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Add the per-expert HF tensors the refit sends for each packed HF expert tensor."""
    out = dict(reference)
    for name, tensor in reference.items():
        match = _PACKED_EXPERTS.fullmatch(name)
        if match is None:
            continue
        prefix, kind = match.groups()
        for e, expert in enumerate(tensor):
            if kind == "down_proj":
                out[f"{prefix}.{e}.down_proj.weight"] = expert
            else:
                gate, up = expert.chunk(2, dim=0)
                out[f"{prefix}.{e}.gate_proj.weight"], out[f"{prefix}.{e}.up_proj.weight"] = gate, up
    return out


def run_trainer(args: argparse.Namespace) -> None:
    """Build the engine, perturb it, refit, and publish the reference export."""
    from transformers import AutoConfig

    import megatron.bridge.engine.refit as refit
    from megatron.bridge.engine import LoraConfig, TrainingEngine
    from megatron.bridge.engine.recipes.qwen35 import _rl_recipe
    from megatron.bridge.models.conversion.auto_bridge import AutoBridge
    from megatron.bridge.training.mixed_precision import bf16_with_fp8_current_scaling_mixed
    from megatron.bridge.utils.common_utils import print_rank_0

    model = AutoBridge.from_hf_config(AutoConfig.from_pretrained(args.hf_path)).get_model_config()
    model.tensor_model_parallel_size = args.tp
    model.sequence_parallel = args.tp > 1
    model.expert_model_parallel_size = args.ep
    model.expert_tensor_parallel_size = 1
    model.pipeline_model_parallel_size = args.pp
    model.seq_length = 512
    model.pipeline_dtype = torch.bfloat16
    recipe = _rl_recipe(model, env_vars={}, pad_to_max_length=False, hf_path=args.hf_path, hf_revision=None)
    if args.fp8:
        recipe.mixed_precision = bf16_with_fp8_current_scaling_mixed()
        recipe.mixed_precision.fp8_param_gather = True
    lora = LoraConfig(rank=8, alpha=16) if args.lora else None
    engine = TrainingEngine.from_recipe(recipe, checkpoint_path=args.hf_path, lora=lora, spawn_async_save_worker=False)

    with torch.no_grad():
        for chunk in engine.model:
            for name, param in chunk.named_parameters():
                if type(param.data) is not torch.Tensor:  # FP8 storage keeps the checkpoint values
                    continue
                generator = torch.Generator(device=param.device).manual_seed(zlib.crc32(name.encode()))
                noise = torch.randn(param.shape, generator=generator, device=param.device, dtype=torch.float32)
                param.add_((0.01 * noise).to(param.dtype))

    start = time.perf_counter()
    plan = engine.connect_rollout(args.address, args.port, _layout(refit, args), bucket_bytes=args.bucket_mb << 20)
    routes = {True: [0, 0], False: [0, 0]}
    for entry in plan.entries:
        routes[entry.reshard][0] += 1
        routes[entry.reshard][1] += torch.Size(entry.shape).numel() * entry.dtype.itemsize
    print_rank_0(
        f"[gate] connected in {time.perf_counter() - start:.1f} s: {plan.num_buckets} buckets, "
        f"reshard {routes[True][0]} tensors {routes[True][1] / 2**20:.1f} MiB, "
        f"collective {routes[False][0]} tensors {routes[False][1] / 2**20:.1f} MiB"
    )
    for i in range(args.refits):
        start = time.perf_counter()
        try:
            enqueued = engine.refit(timeout_s=args.timeout)
            issue = time.perf_counter() - start
            enqueued.synchronize()
        except TimeoutError:
            if args.absent_rank < 0:
                raise
            print_rank_0(f"[gate] trainer: TimeoutError after {time.perf_counter() - start:.1f} s, as expected")
            os._exit(0)  # the channel is aborted; nothing more to check
        if args.absent_rank >= 0:
            raise AssertionError("a refit with an absent rollout rank completed")
        torch.distributed.barrier()
        print_rank_0(
            f"[gate] refit {i}: issued in {issue * 1e3:.1f} ms, done in {(time.perf_counter() - start) * 1e3:.1f} ms"
        )

    # Baseline: the colocated path's cost on the trainer alone, the full gather of every tensor.
    for i in range(2):
        start = time.perf_counter()
        for _ in engine.export(streaming=True):
            pass
        torch.cuda.synchronize()
        torch.distributed.barrier()
        print_rank_0(f"[gate] export(streaming=True) {i}: {(time.perf_counter() - start) * 1e3:.1f} ms")

    is_writer = torch.distributed.get_rank() == 0
    reference = {name: tensor.cpu() for name, tensor in engine.export(streaming=True) if is_writer}
    if is_writer:
        torch.save(reference, f"{args.out}/reference.pt.tmp")
        os.replace(f"{args.out}/reference.pt.tmp", f"{args.out}/reference.pt")
    torch.distributed.barrier()
    engine.close()
    print_rank_0("[gate] trainer done")


def _vllm_runner(hf_path: str):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # keep the engine core, and its model, in this process
    from vllm import LLM

    llm = LLM(
        model=hf_path,
        load_format="dummy",
        skip_tokenizer_init=True,
        enforce_eager=True,
        gpu_memory_utilization=0.3,
        max_model_len=512,
        dtype="bfloat16",
    )
    core = llm.llm_engine.engine_core.engine_core
    return llm, core.model_executor.driver_worker.worker.model_runner


def _snapshot(runner) -> dict[str, torch.Tensor]:
    return {name: param.detach().cpu().clone() for name, param in runner.model.named_parameters()}


def _check_plan_covers(plan, exported: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]) -> None:
    planned = {entry.name for entry in plan.entries}
    stray = planned - set(reference)
    missing = [
        name
        for name in exported
        if name not in planned
        and not (_PACKED_EXPERTS.fullmatch(name) and set(_expand({name: exported[name]})) - {name} <= planned)
    ]
    if missing or stray:
        raise AssertionError(f"plan misses {missing[:5]}, has unknown {sorted(stray)[:5]}")


def run_rollout(args: argparse.Namespace) -> None:
    """Receive the refits and check the last one against the reference export."""
    torch.cuda.set_device(0)
    refit = _standalone_refit()
    layout = _layout(refit, args)
    llm = runner = None
    if args.vllm:
        if layout != refit.RolloutLayout(args.rollout_world):
            raise ValueError("--vllm runs one TP-1 engine per rank")
        llm, runner = _vllm_runner(args.hf_path)  # the LLM must stay alive: its shutdown drops the model
    receiver = refit.RefitReceiver(args.address, args.port, args.rank, layout)
    if args.rank == args.absent_rank:
        logger.info(f"[gate] rollout rank {args.rank}: joined, now leaving without receiving")
        os._exit(0)
    for _ in range(args.refits):
        received: dict[str, torch.Tensor] = {}  # free the previous refit first, as an engine loading in place would
        start = time.perf_counter()
        try:
            staged = receiver.receive(timeout_s=args.timeout)
            if runner is not None:
                runner.reload_weights(weights_iterator=((e.name, t) for e, t in staged), is_checkpoint_format=True)
            else:
                received = {entry.name: tensor for entry, tensor in staged}
            receiver.synchronize()
        except TimeoutError:
            if args.absent_rank < 0:
                raise
            logger.info(
                f"[gate] rollout rank {args.rank}: TimeoutError after {time.perf_counter() - start:.1f} s, as expected"
            )
            os._exit(0)
        if args.absent_rank >= 0:
            raise AssertionError("a refit with an absent rollout rank completed")
        torch.cuda.synchronize()
        logger.info(f"[gate] rollout rank {args.rank}: refit received in {(time.perf_counter() - start) * 1e3:.1f} ms")

    while not os.path.exists(f"{args.out}/reference.pt"):
        time.sleep(1)
    exported = torch.load(f"{args.out}/reference.pt", mmap=True)
    reference = _expand(exported)
    if args.rank == 0:
        _check_plan_covers(receiver.plan, exported, reference)

    if runner is not None:
        # The colocated path loads Bridge's export (packed experts) through the same reload_weights.
        refitted = _snapshot(runner)
        runner.reload_weights(weights_iterator=((n, t.cuda()) for n, t in exported.items()), is_checkpoint_format=True)
        expected = _snapshot(runner)
        mismatched = [name for name in expected if not torch.equal(refitted[name], expected[name])]
        checked = f"{len(expected)} vLLM parameters"
    else:
        tp_rank = args.rank % layout.tensor_parallel_size
        mismatched = []
        for entry, shape in receiver.local_entries():
            want = reference[entry.name]
            if shape != entry.shape:
                want = want.chunk(layout.tensor_parallel_size, dim=entry.tp_dim)[tp_rank]
            if not torch.equal(received[entry.name].cpu(), want):
                mismatched.append(entry.name)
        checked = f"{len(received)} tensors"
    if mismatched:
        raise AssertionError(f"rollout rank {args.rank}: {len(mismatched)} differ, e.g. {mismatched[:5]}")
    leaked = sorted(name for name in sys.modules if name == "megatron" or name.startswith("megatron."))
    if leaked:
        raise AssertionError(f"the rollout imported Megatron: {leaked[:5]}")
    logger.info(f"[gate] rollout rank {args.rank}: {checked} match the reference; no megatron module imported")
    receiver.close()


def main() -> None:
    """Run one side of the gate."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    args = _parse_args()
    if args.absent_rank >= 0:
        # A C-level timer: prints every thread's stack even when the main thread holds the GIL in a C call.
        faulthandler.dump_traceback_later(args.timeout + 60, repeat=True, exit=False)
    if args.role == "trainer":
        run_trainer(args)
    else:
        run_rollout(args)


if __name__ == "__main__":
    main()
