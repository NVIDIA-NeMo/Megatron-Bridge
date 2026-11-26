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
"""vllm on its own GPUs (not colocated): the trainer's ``RemoteVllmRollout`` and the rollout ranks' server.

Run this file once per rollout GPU (``--rank r``); each process is one TP-1 vllm engine. The trainer
side is ``RemoteVllmRollout``, a drop-in for ``VllmRollout`` (``vllm_rollout.py``) in the RL examples' round:

- ``refresh_from(policy)`` refits the rollout with the refit API (``TrainingEngine.refit``; the rollout
  ranks stream ``RefitReceiver.receive`` into vllm's ``reload_weights``) and returns once the transfer
  is enqueued; ``optim_step`` orders itself after it.
- ``generate(...)`` deals the prompt groups round-robin to the rollout ranks and returns the samples in
  prompt order on every trainer rank.
- ``sleep()`` and ``wake_kv_cache()`` do nothing: vllm keeps its GPUs.

Commands and samples go through a TCP store on trainer rank 0 (``port + 1``).
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import time
from datetime import timedelta

import torch
from dapo_math import Sample

from megatron.bridge.engine.refit import RefitReceiver, RolloutLayout
from megatron.bridge.utils.common_utils import print_rank_0


logger = logging.getLogger("vllm_remote_rollout")


def _open_store(address: str, port: int, timeout_s: float, *, is_master: bool) -> torch.distributed.TCPStore:
    return torch.distributed.TCPStore(
        address, port + 1, is_master=is_master, wait_for_workers=False, timeout=timedelta(seconds=timeout_s)
    )


def _post(store: torch.distributed.TCPStore, key: str, value: object) -> None:
    store.set(key, pickle.dumps(value))


def _get(store: torch.distributed.TCPStore, key: str) -> object:
    return pickle.loads(store.get(key))


class RemoteVllmRollout:
    """The trainer's handle on the rollout ranks; construct it on every trainer rank (collective).

    Construction connects the policy to the rollout (``TrainingEngine.connect_rollout``), which waits
    until every rollout rank has built its vllm engine and joined.
    """

    colocated = False

    def __init__(
        self,
        policy,
        *,
        address: str,
        port: int,
        world_size: int,
        bucket_bytes: int,
        refit_timeout_s: float,
        store_timeout_s: float,
    ):
        self.world_size = world_size
        self.refit_timeout_s = refit_timeout_s
        is_leader = torch.distributed.get_rank() == 0
        self.store = _open_store(address, port, store_timeout_s, is_master=True) if is_leader else None
        self.step = 0
        self.posted: list[str] = []
        start = time.perf_counter()
        plan = policy.connect_rollout(address, port, RolloutLayout(world_size), bucket_bytes=bucket_bytes)
        print_rank_0(
            f"[remote] {world_size} rollout ranks connected in {time.perf_counter() - start:.1f} s: "
            f"{len(plan.entries)} tensors in {plan.num_buckets} buckets"
        )

    def _command(self, command: dict | None) -> int:
        """Post the next command for every rollout rank (from rank 0); returns its step."""
        step, self.step = self.step, self.step + 1
        if self.store is not None:
            key = f"command/{step}"
            _post(self.store, key, command)
            self.posted.append(key)
        return step

    def sleep(self) -> None:
        pass

    def wake_kv_cache(self) -> None:
        pass

    def refresh_from(self, policy) -> None:
        """Refit the rollout from the policy; returns once the transfer is enqueued."""
        self._command({"op": "refit"})
        policy.refit(timeout_s=self.refit_timeout_s)

    def generate(
        self, prompts: list, *, group_size: int, max_tokens: int, temperature: float = 1.0, top_p: float = 1.0
    ):
        """``group_size`` samples per prompt from the rollout ranks, in prompt order, on every trainer rank."""
        start = time.perf_counter()
        step = self._command(
            {
                "op": "generate",
                "prompts": [(p.index, p.prompt_ids) for p in prompts],
                "n": group_size,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
        records = [None]
        if self.store is not None:
            keys = [f"result/{step}/{rank}" for rank in range(self.world_size)]
            records = [[record for key in keys for record in _get(self.store, key)]]
            # Every rollout rank has read all posted commands once it answered this one.
            for key in self.posted + keys:
                self.store.delete_key(key)
            self.posted = []
        torch.distributed.broadcast_object_list(records, src=0)

        by_index = {p.index: [] for p in prompts}
        for record in records[0]:
            by_index[record["index"]].append(record)
        samples = [
            Sample(
                prompt=prompt,
                response_ids=record["response_ids"],
                rollout_logprobs=record["logprobs"],
                text=record["text"],
                extras={"finish_reason": record["finish_reason"]},
            )
            for prompt in prompts
            for record in by_index[prompt.index]
        ]
        tokens = sum(len(s.response_ids) for s in samples)
        elapsed = time.perf_counter() - start
        print_rank_0(f"[remote] generated {len(samples)} samples ({tokens} response tokens) in {elapsed:.1f} s")
        return samples

    def close(self) -> None:
        """Tell the rollout ranks to leave; rank 0 hosts the store, so it waits until they all have."""
        self._command(None)
        if self.store is not None:
            self.store.wait([f"left/{rank}" for rank in range(self.world_size)])


# ---- rollout rank ---------------------------------------------------------------------------------------


def _generate(llm, command: dict) -> list[dict]:
    """``n`` completions per ``(index, prompt_ids)``, as plain records the trainer turns into Samples."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    prompts = command["prompts"]
    if not prompts:
        return []
    params = SamplingParams(
        n=command["n"],
        temperature=command["temperature"],
        top_p=command["top_p"],
        max_tokens=command["max_tokens"],
        logprobs=0,
    )
    outputs = llm.generate([TokensPrompt(prompt_token_ids=ids) for _, ids in prompts], params, use_tqdm=False)
    return [
        {
            "index": index,
            "response_ids": list(completion.token_ids),
            "logprobs": [entry[token].logprob for token, entry in zip(completion.token_ids, completion.logprobs)],
            "text": completion.text,
            "finish_reason": completion.finish_reason,
        }
        for (index, _), output in zip(prompts, outputs)
        for completion in output.outputs
    ]


def serve() -> None:
    """One rollout rank: a TP-1 vllm engine that runs the trainer's commands until told to leave."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rank", type=int, required=True, help="This rank among the rollout ranks")
    parser.add_argument("--rollout-world", type=int, required=True, help="Rollout ranks")
    parser.add_argument("--address", default="127.0.0.1", help="Host of trainer rank 0")
    parser.add_argument("--port", type=int, required=True, help="The trainer's --rollout-port")
    parser.add_argument(
        "--hf-path", required=True, help="HF directory or ID of the policy's model (config, tokenizer)"
    )
    parser.add_argument("--max-model-len", type=int, required=True, help="At least max prompt + max response tokens")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--vllm-cuda-graphs", action="store_true", help="Let vllm compile and capture CUDA graphs")
    parser.add_argument("--refit-timeout", type=float, default=600.0, help="Seconds one refit may take")
    parser.add_argument("--store-timeout", type=float, default=4 * 3600.0, help="Seconds to wait for a command")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # keep the engine core, and its model, in this process
    from vllm import LLM

    torch.cuda.set_device(0)
    # Dummy weights: the trainer refits before its first generate.
    llm = LLM(
        model=args.hf_path,
        load_format="dummy",
        enforce_eager=not args.vllm_cuda_graphs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_prefix_caching=False,
        seed=args.seed + args.rank,
        dtype="bfloat16",
    )
    runner = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner

    trainer_addr, trainer_port = args.address, args.port
    trainer_store = _open_store(trainer_addr, trainer_port, args.store_timeout, is_master=False)
    receiver = RefitReceiver(trainer_addr, trainer_port, args.rank, RolloutLayout(args.rollout_world))
    logger.info(f"[rollout {args.rank}] connected")

    step = 0
    while (command := _get(trainer_store, f"command/{step}")) is not None:
        start = time.perf_counter()
        if command["op"] == "refit":
            staged = receiver.receive(timeout_s=args.refit_timeout)
            # vllm repacks expert weights after loading, so checkpoint-format tensors go through the layerwise reload.
            runner.reload_weights(
                weights_iterator=((entry.name, tensor) for entry, tensor in staged), is_checkpoint_format=True
            )
            receiver.synchronize()
        else:
            mine = dict(command, prompts=command["prompts"][args.rank :: args.rollout_world])
            _post(trainer_store, f"result/{step}/{args.rank}", _generate(llm, mine))
        logger.info(f"[rollout {args.rank}] {command['op']} {step} in {time.perf_counter() - start:.2f} s")
        step += 1
    trainer_store.set(f"left/{args.rank}", b"")
    receiver.close()


if __name__ == "__main__":
    serve()
