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

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


_SCRIPT = Path(__file__).parents[3] / "examples" / "rl" / "rlhf_with_bridge.py"


def _main_function() -> ast.FunctionDef:
    tree = ast.parse(_SCRIPT.read_text(), filename=str(_SCRIPT))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")


def _training_loop() -> ast.For:
    return next(node for node in _main_function().body if isinstance(node, ast.For))


def _execute_rollout_batch_prefix(namespace: dict) -> dict:
    loop = copy.deepcopy(_training_loop())
    microbatch_index = next(
        index
        for index, statement in enumerate(loop.body)
        if isinstance(statement, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "mb_iter" for target in statement.targets)
    )
    loop.body = loop.body[: microbatch_index + 1]
    module = ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[]))
    exec(compile(module, str(_SCRIPT), "exec"), namespace)
    return next(namespace["mb_iter"])


def _load_rl_loss_fn(rewards: torch.Tensor):
    loss_fn = next(
        node for node in _training_loop().body if isinstance(node, ast.FunctionDef) and node.name == "rl_loss_fn"
    )
    module = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(loss_fn)], type_ignores=[]))
    namespace = {"F": F, "rewards_t": rewards, "torch": torch}
    exec(compile(module, str(_SCRIPT), "exec"), namespace)
    return namespace["rl_loss_fn"]


class _Batch(dict):
    def to(self, device: torch.device):
        return _Batch({key: value.to(device) for key, value in self.items()})


class _Tokenizer:
    pad_token_id = 99
    eos_token_id = 99

    def __init__(self) -> None:
        self.sampled_ids = torch.tensor([[11, 12, 31, self.eos_token_id]])
        self._calls = 0

    def __call__(self, _texts, **_kwargs):
        self._calls += 1
        if self._calls == 1:
            return _Batch(
                {
                    "input_ids": torch.tensor([[11, 12]]),
                    "attention_mask": torch.ones((1, 2), dtype=torch.long),
                }
            )

        # Re-tokenizing decoded text loses the sampled terminal EOS token.
        return _Batch(
            {
                "input_ids": torch.tensor([[11, 12, 31]]),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
            }
        )

    def batch_decode(self, _token_ids, **_kwargs) -> list[str]:
        return ["decoded text"]


class _PolicyModel:
    def __init__(self, sampled_ids: torch.Tensor) -> None:
        self.sampled_ids = sampled_ids

    def generate(self, **_kwargs) -> torch.Tensor:
        return self.sampled_ids


@pytest.mark.unit
def test_rlhf_rollout_batch_preserves_sampled_ids_and_completion_boundary() -> None:
    tokenizer = _Tokenizer()
    namespace = {
        "args": SimpleNamespace(train_iters=1, prompts=["prompt"], seq_length=4, max_new_tokens=2),
        "device": torch.device("cpu"),
        "gen_tokenizer": tokenizer,
        "hf_gen_model": _PolicyModel(tokenizer.sampled_ids),
        "make_microbatch_iterator": lambda batch, num_microbatches: iter([batch] * num_microbatches),
        "reward_pipe": lambda _text: [{"label": "POSITIVE", "score": 1.0}],
        "torch": torch,
    }

    batch = _execute_rollout_batch_prefix(namespace)

    torch.testing.assert_close(batch["input_ids"], tokenizer.sampled_ids)
    torch.testing.assert_close(batch["loss_mask"], torch.tensor([[0, 0, 1, 1]]))


@pytest.mark.unit
def test_rlhf_rollout_batch_masks_padding_after_eos() -> None:
    tokenizer = _Tokenizer()
    tokenizer.sampled_ids = torch.tensor([[11, 12, 31, tokenizer.eos_token_id, tokenizer.pad_token_id]])
    namespace = {
        "args": SimpleNamespace(train_iters=1, prompts=["prompt"], seq_length=5, max_new_tokens=3),
        "device": torch.device("cpu"),
        "gen_tokenizer": tokenizer,
        "hf_gen_model": _PolicyModel(tokenizer.sampled_ids),
        "make_microbatch_iterator": lambda batch, num_microbatches: iter([batch] * num_microbatches),
        "reward_pipe": lambda _text: [{"label": "POSITIVE", "score": 1.0}],
        "torch": torch,
    }

    batch = _execute_rollout_batch_prefix(namespace)

    torch.testing.assert_close(batch["attention_mask"], torch.tensor([[1, 1, 1, 1, 0]]))
    torch.testing.assert_close(batch["loss_mask"], torch.tensor([[0, 0, 1, 1, 0]]))


@pytest.mark.unit
def test_rlhf_loss_has_no_prompt_token_gradient() -> None:
    input_ids = torch.tensor([[0, 1, 2, 3, 4]])
    batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "loss_mask": torch.tensor([[0, 0, 0, 1, 1]]),
    }
    logits = torch.zeros((1, 5, 5), requires_grad=True)

    loss, _ = _load_rl_loss_fn(torch.tensor([1.0]))(logits, batch)
    loss.backward()

    assert torch.count_nonzero(logits.grad[:, :2]) == 0
    assert torch.count_nonzero(logits.grad[:, 2:4]) > 0
