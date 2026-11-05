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
"""vllm colocated in the training process: one vllm instance per rank (tensor parallel 1), sharing the GPUs
with the engines through sleep mode.

Residency protocol, per round (the caller orders the calls):
  1. policy on the GPU, vllm asleep: ``refresh_from(policy)`` wakes vllm's weights and streams the
     engine's HF export into ``model.load_weights`` (no disk, no IPC).
  2. engines offloaded with ``cpu()``: ``wake_kv_cache()`` then ``generate(...)``.
  3. ``sleep()`` (level 2 drops weights and KV cache; the next refresh rewrites the weights).
Every rank generates a disjoint shard of prompt groups and the samples are all-gathered, so every rank
ends up with the same global sample list, which is what the engines expect.
"""

from __future__ import annotations

import time

import torch
from dapo_math import MathPrompt, Sample

from megatron.bridge.engine import TrainingEngine
from megatron.bridge.utils.common_utils import print_rank_0


_MODEL_PATHS = (
    "llm_engine.model_executor.driver_worker.worker.model_runner.model",
    "llm_engine.model_executor.driver_worker.model_runner.model",
    "llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model",
    "llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner.model",
)


def _in_process_model(llm):
    for path in _MODEL_PATHS:
        obj = llm
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        return obj
    raise AttributeError("could not find vllm's in-process model; vllm's object layout changed")


class VllmRollout:
    """vllm 0.29 ``LLM`` built over the already-initialized torch.distributed world (external_launcher)."""

    def __init__(
        self,
        hf_path: str,
        *,
        gpu_memory_utilization: float,
        max_model_len: int,
        seed: int = 0,
        enforce_eager: bool = True,
    ):
        from vllm import LLM

        self.rank = torch.distributed.get_rank()
        self.world = torch.distributed.get_world_size()
        start = time.perf_counter()
        self.llm = LLM(
            model=hf_path,
            tensor_parallel_size=1,
            # One replica per rank. Without the data-parallel declaration vllm's setup stalls when the
            # torch.distributed world is larger than TP x PP.
            data_parallel_size=self.world,
            distributed_executor_backend="external_launcher",
            enable_sleep_mode=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_prefix_caching=False,
            # Async scheduling (step_with_batch_queue) issues the DP padding all-reduce in a different
            # order than the plain step loop idle replicas run, and the two collide on the gloo group.
            async_scheduling=False,
            seed=seed + self.rank,
            dtype="bfloat16",
            # Eager by default: the torch.compile / CUDA-graph path stalled for many minutes on this
            # image while the eager path ran every probe cleanly. Opt in with enforce_eager=False.
            enforce_eager=enforce_eager,
        )
        self.model = _in_process_model(self.llm)
        self.runner = self.llm.llm_engine.model_executor.driver_worker.worker.model_runner
        self.awake = True
        print_rank_0(f"[vllm] built in {time.perf_counter() - start:.1f} s (one instance per rank, TP 1)")

    def sleep(self) -> None:
        if self.awake:
            self.llm.sleep(level=2)
            self.awake = False

    def refresh_from(self, policy: TrainingEngine) -> None:
        """Wake the weights and overwrite them from the policy's streaming HF export. Policy must be on the GPU."""
        assert policy.device == "cuda", "refresh_from needs the policy resident (export gathers from the GPU)"
        start = time.perf_counter()
        self.llm.wake_up(tags=["weights"])
        # vllm repacks expert weights into a kernel layout after loading, so checkpoint-format tensors
        # must go through the runner's layerwise reload rather than a bare model.load_weights.
        self.runner.reload_weights(weights_iterator=policy.export(streaming=True), is_checkpoint_format=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print_rank_0(f"[vllm] weights refreshed from the policy in {elapsed:.1f} s")

    def wake_kv_cache(self) -> None:
        self.llm.wake_up(tags=["kv_cache"])
        self.awake = True

    def generate(
        self,
        prompts: list[MathPrompt],
        *,
        group_size: int,
        max_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> list[Sample]:
        """``group_size`` samples per prompt, prompt groups dealt round-robin to ranks, gathered in prompt order."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        params = SamplingParams(n=group_size, temperature=temperature, top_p=top_p, max_tokens=max_tokens, logprobs=0)
        mine = prompts[self.rank :: self.world]
        start = time.perf_counter()
        print(f"[vllm] rank {self.rank}: {len(mine)} prompts x {group_size} samples", flush=True)
        if mine:
            requests = [TokensPrompt(prompt_token_ids=p.prompt_ids) for p in mine]
            outputs = self.llm.generate(requests, params, use_tqdm=False)
        else:
            # Data-parallel replicas all-reduce "has unfinished requests" every engine step, so a rank
            # with no prompts must still drive its engine until every replica is done: one throwaway
            # request keeps it in the loop (vllm keeps idle replicas stepping while peers have work).
            idle = SamplingParams(n=1, temperature=0.0, max_tokens=1)
            self.llm.generate([TokensPrompt(prompt_token_ids=prompts[0].prompt_ids)], idle, use_tqdm=False)
            outputs = []
        local: list[Sample] = []
        for prompt, output in zip(mine, outputs):
            for completion in output.outputs:
                logprobs = [entry[token].logprob for token, entry in zip(completion.token_ids, completion.logprobs)]
                local.append(
                    Sample(
                        prompt=prompt,
                        response_ids=list(completion.token_ids),
                        rollout_logprobs=logprobs,
                        text=completion.text,
                        extras={"finish_reason": completion.finish_reason},
                    )
                )
        gathered: list[list[Sample] | None] = [None] * self.world
        torch.distributed.all_gather_object(gathered, local)
        by_prompt = {s.prompt.index: [] for s in (x for shard in gathered for x in shard)}
        for shard in gathered:
            for sample in shard:
                by_prompt[sample.prompt.index].append(sample)
        samples = [s for p in prompts for s in by_prompt.get(p.index, [])]
        tokens = sum(len(s.response_ids) for s in samples)
        elapsed = time.perf_counter() - start
        print_rank_0(f"[vllm] generated {len(samples)} samples ({tokens} response tokens) in {elapsed:.1f} s")
        return samples
