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

import pytest
import torch

from megatron.bridge.models.qwen_vl.qwen3_vl_step import forward_step, get_batch_from_iterator


pytestmark = pytest.mark.unit


def test_get_batch_from_iterator_rejects_collate_time_packing_metadata():
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "position_ids": torch.tensor([[0, 1, 2]]),
        "visual_inputs": None,
        "cu_seqlens_q": torch.tensor([0, 3], dtype=torch.int32),
    }

    with pytest.raises(ValueError, match="does not support collate-time in-batch packing"):
        get_batch_from_iterator(
            iter([batch]),
            is_first_pp_stage=True,
            is_last_pp_stage=True,
        )


def test_get_batch_from_iterator_allows_deferred_none_packing_metadata(monkeypatch):
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "position_ids": torch.tensor([[0, 1, 2]]),
        "labels": torch.tensor([[2, 3, -100]]),
        "loss_mask": torch.tensor([[1.0, 1.0, 0.0]]),
        "visual_inputs": None,
        "cu_seqlens_q": None,
        "cu_seqlens": None,
    }
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **kwargs: self)

    result = get_batch_from_iterator(
        iter([batch]),
        is_first_pp_stage=True,
        is_last_pp_stage=True,
    )

    assert torch.equal(result["input_ids"], batch["input_ids"])


def test_forward_step_uses_compacted_loss_mask_from_model(monkeypatch):
    class _ProcessGroup:
        def rank(self):
            return 0

        def size(self):
            return 1

    class _Timer:
        def __call__(self, *args, **kwargs):  # noqa: ARG002
            return self

        def start(self):
            return None

        def stop(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *exc):  # noqa: ARG002
            return False

    process_group = _ProcessGroup()
    pg_collection = type(
        "PGCollection",
        (),
        {"pp": process_group, "tp": process_group, "cp": process_group, "ep": process_group},
    )()
    config = type(
        "ModelConfig",
        (),
        {"mtp_num_layers": 0, "seq_length": 4, "overlap_moe_expert_parallel_comm": False},
    )()
    compacted_loss_mask = torch.tensor([[1.0, 0.0]])

    class _Model:
        def __call__(self, **kwargs):  # noqa: ARG002
            return torch.tensor(0.0), compacted_loss_mask

    state = type(
        "State",
        (),
        {
            "timers": _Timer(),
            "straggler_timer": _Timer(),
            "cfg": type(
                "Config",
                (),
                {
                    "dataset": type("Dataset", (), {"enable_in_batch_packing": True})(),
                    "rerun_state_machine": type(
                        "Rerun", (), {"check_for_nan_in_loss": False, "check_for_spiky_loss": False}
                    )(),
                },
            )(),
        },
    )()
    tokens = torch.tensor([[1, 2, 3, 0]])
    labels = torch.tensor([[2, 3, -100, -100]])
    original_loss_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    position_ids = torch.arange(4).unsqueeze(0)
    attention_mask = torch.ones_like(tokens, dtype=torch.bool)

    monkeypatch.setattr("megatron.bridge.models.qwen_vl.qwen3_vl_step.get_pg_collection", lambda _: pg_collection)
    monkeypatch.setattr("megatron.bridge.models.qwen_vl.qwen3_vl_step.is_pp_first_stage", lambda _: True)
    monkeypatch.setattr("megatron.bridge.models.qwen_vl.qwen3_vl_step.is_pp_last_stage", lambda _: True)
    monkeypatch.setattr("megatron.bridge.models.qwen_vl.qwen3_vl_step.get_model_config", lambda _: config)
    monkeypatch.setattr(
        "megatron.bridge.models.qwen_vl.qwen3_vl_step.get_batch",
        lambda *args, **kwargs: (tokens, labels, original_loss_mask, attention_mask, position_ids, {}),
    )
    monkeypatch.setattr(
        "megatron.bridge.models.qwen_vl.qwen3_vl_step._pad_and_pack_qwen3_vl_step",
        lambda *args, **kwargs: (*args[:5], object()),
    )
    monkeypatch.setattr(
        "megatron.bridge.models.qwen_vl.qwen3_vl_step.get_batch_on_this_cp_rank",
        lambda forward_args, **kwargs: forward_args,
    )
    captured_loss_masks = []
    monkeypatch.setattr(
        "megatron.bridge.models.qwen_vl.qwen3_vl_step._create_loss_function",
        lambda loss_mask, *args: captured_loss_masks.append(loss_mask) or object(),
    )

    output, _ = forward_step(state, iter(()), _Model())

    assert output.item() == 0.0
    assert captured_loss_masks == [compacted_loss_mask]
