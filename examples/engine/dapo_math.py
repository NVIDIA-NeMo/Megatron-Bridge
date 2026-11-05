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
"""DAPO-Math-17k prompts, the rule-based reward, and the sample-to-Datum assembly shared by the RL examples.

The parquet follows verl's layout: ``prompt`` is a list of chat messages, ``reward_model.ground_truth``
the integer answer as text. Rewards are +1 / -1 as in DAPO (``dapo_math_reward.compute_score``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pyarrow.parquet as pq
import torch
from dapo_math_reward import compute_score

from megatron.bridge.engine import Datum


@dataclass
class MathPrompt:
    """One prompt: its chat-templated token ids and the answer used for the reward."""

    index: int
    prompt_ids: list[int]
    ground_truth: str


@dataclass
class Sample:
    """One generated response for a prompt, with the generation backend's per-token logprobs."""

    prompt: MathPrompt
    response_ids: list[int]
    rollout_logprobs: list[float]
    text: str
    reward: float = 0.0
    advantage: float = 0.0
    extras: dict = field(default_factory=dict)


def load_prompts(
    path: str, tokenizer, *, limit: int, offset: int = 0, max_prompt_tokens: int = 2048
) -> list[MathPrompt]:
    """Read ``limit`` prompts starting at ``offset`` (file order, identical on every rank)."""
    prompts: list[MathPrompt] = []
    index = -1
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
        for row in batch.to_pylist():
            index += 1
            if index < offset:
                continue
            messages = row["prompt"]
            # Render to text first: tokenize=True returns a dict-like encoding in recent transformers.
            text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) > max_prompt_tokens:
                continue
            truth = str(row["reward_model"]["ground_truth"])
            prompts.append(MathPrompt(index=index, prompt_ids=list(ids), ground_truth=truth))
            if len(prompts) == limit:
                return prompts
    return prompts


def score(sample: Sample) -> float:
    """DAPO reward: +1 for a correct final answer, -1 otherwise."""
    result = compute_score(sample.text, sample.prompt.ground_truth)
    sample.extras["pred"] = result["pred"]
    return float(result["score"])


def group_advantages(samples: list[Sample], *, group_size: int, eps: float = 1e-6) -> None:
    """GRPO: reward minus the group mean, divided by the group standard deviation; groups are consecutive."""
    for start in range(0, len(samples), group_size):
        group = samples[start : start + group_size]
        rewards = torch.tensor([s.reward for s in group])
        std = rewards.std(unbiased=False) if len(group) > 1 else torch.tensor(0.0)
        for s, r in zip(group, rewards):
            s.advantage = float((r - rewards.mean()) / (std + eps))


def to_datum(sample: Sample, **per_token: torch.Tensor | None) -> Datum:
    """Prompt + response as one Datum: targets are the response tokens, every prompt position is masked out.

    Per-token fields (``advantages`` broadcast from the scalar, ``rollout_logprobs`` from the sampler, and
    any ``per_token`` override) are ``[L]`` with ``L = len(prompt) + len(response) - 1``.
    """
    full = torch.tensor(sample.prompt.prompt_ids + sample.response_ids, dtype=torch.int64)
    length = len(full) - 1
    response = len(sample.response_ids)
    loss_mask = torch.zeros(length, dtype=torch.float32)
    loss_mask[length - response :] = 1.0
    advantages = torch.zeros(length, dtype=torch.float32)
    advantages[length - response :] = sample.advantage
    rollout = torch.zeros(length, dtype=torch.float32)
    rollout[length - response :] = torch.tensor(sample.rollout_logprobs, dtype=torch.float32)
    fields = {"advantages": advantages, "rollout_logprobs": rollout}
    fields.update({k: v for k, v in per_token.items() if v is not None})
    return Datum(
        input_ids=full[:-1],
        target_ids=full[1:],
        loss_mask=loss_mask,
        sample_inputs={"prompt_index": sample.prompt.index, "reward": sample.reward},
        **fields,
    )


def response_slice(datum: Datum, values: torch.Tensor) -> torch.Tensor:
    """The response positions of a ``[L]`` per-target tensor returned by ``forward_only`` for ``datum``."""
    return values[datum.loss_mask.bool()]


def gae(
    rewards: torch.Tensor, values: torch.Tensor, *, gamma: float = 1.0, lam: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized advantage estimation over one response (reward on the last token only).

    Returns ``(advantages, returns)`` with ``returns = advantages + values``."""
    advantages = torch.zeros_like(values)
    last = 0.0
    for t in reversed(range(len(values))):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        last = delta + gamma * lam * last
        advantages[t] = last
    return advantages, advantages + values
