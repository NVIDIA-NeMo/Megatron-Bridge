# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for EMA callback checkpoint integration and helper utilities."""

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.training.callbacks import CheckpointCallbackContext
from megatron.bridge.training.checkpointing import CheckpointType
from megatron.bridge.training.ema import EMACallback
from megatron.bridge.training.ema_checkpoint import (
    EMA_DIRNAME,
    has_ema_state,
    load_ema_user_state,
    save_ema_user_state,
)


class TestEMACheckpointHelpers:
    def test_has_ema_state_returns_true_for_valid_payload(self):
        user_state = {
            "ema_state": {"chunk0.weight": torch.ones(2)},
            "ema_updates": 3,
            "ema_skipped_iters": 0,
        }
        assert has_ema_state(user_state) is True

    def test_has_ema_state_returns_false_for_missing_payload(self):
        assert has_ema_state({}) is False
        assert has_ema_state(None) is False
        assert has_ema_state({"ema_state": {}}) is False

    def test_save_and_load_ema_user_state_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "megatron.bridge.training.ema_checkpoint.get_rank_safe",
            lambda: 0,
        )

        checkpoint_name = str(tmp_path / "iter_0000001")
        user_state = {
            "ema_state": {
                "chunk0.weight": torch.tensor([1.0, 2.0], dtype=torch.float32),
            },
            "ema_updates": 5,
            "ema_skipped_iters": 1,
        }

        saved = save_ema_user_state(checkpoint_name, user_state)
        assert saved is True

        ema_file = tmp_path / "iter_0000001" / EMA_DIRNAME / "rank_00000.pt"
        assert ema_file.exists()

        restored_state = {}
        loaded = load_ema_user_state(checkpoint_name, restored_state)

        assert loaded is True
        assert restored_state["ema_updates"] == 5
        assert restored_state["ema_skipped_iters"] == 1
        assert "chunk0.weight" in restored_state["ema_state"]
        assert torch.equal(
            restored_state["ema_state"]["chunk0.weight"],
            torch.tensor([1.0, 2.0], dtype=torch.float32),
        )

    def test_load_ema_user_state_returns_false_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "megatron.bridge.training.ema_checkpoint.get_rank_safe",
            lambda: 0,
        )

        restored_state = {}
        loaded = load_ema_user_state(str(tmp_path / "iter_0000001"), restored_state)

        assert loaded is False
        assert restored_state == {}


class TestEMACallbackCheckpointHooks:
    def test_on_checkpoint_save_persists_sidecar(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "megatron.bridge.training.ema_checkpoint.get_rank_safe",
            lambda: 0,
        )

        callback = EMACallback(decay=0.95, store_on_cpu=True)

        state = SimpleNamespace(
            cfg=SimpleNamespace(
                checkpoint=SimpleNamespace(async_save=False),
            )
        )

        context = CheckpointCallbackContext(
            state=state,
            checkpoint_name=str(tmp_path / "iter_0000002"),
            checkpoint_type=CheckpointType.GLOBAL,
            user_state={
                "ema_state": {"chunk0.weight": torch.ones(3)},
                "ema_updates": 2,
                "ema_skipped_iters": 0,
            },
        )

        callback.on_checkpoint_save(context)

        ema_file = tmp_path / "iter_0000002" / EMA_DIRNAME / "rank_00000.pt"
        assert ema_file.exists()

    def test_on_checkpoint_save_raises_for_async_save(self, tmp_path):
        callback = EMACallback()

        state = SimpleNamespace(
            cfg=SimpleNamespace(
                checkpoint=SimpleNamespace(async_save=True),
            )
        )

        context = CheckpointCallbackContext(
            state=state,
            checkpoint_name=str(tmp_path / "iter_0000003"),
            checkpoint_type=CheckpointType.GLOBAL,
            user_state={
                "ema_state": {"chunk0.weight": torch.ones(1)},
                "ema_updates": 1,
                "ema_skipped_iters": 0,
            },
        )

        with pytest.raises(NotImplementedError, match="async_save=True"):
            callback.on_checkpoint_save(context)

    def test_on_checkpoint_load_restores_sidecar(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "megatron.bridge.training.ema_checkpoint.get_rank_safe",
            lambda: 0,
        )

        checkpoint_name = str(tmp_path / "iter_0000004")
        save_ema_user_state(
            checkpoint_name,
            {
                "ema_state": {"chunk0.weight": torch.tensor([3.0])},
                "ema_updates": 4,
                "ema_skipped_iters": 1,
            },
        )

        callback = EMACallback()

        state = SimpleNamespace(
            train_state=SimpleNamespace(step=4),
            cfg=SimpleNamespace(
                checkpoint=SimpleNamespace(finetune=False),
            ),
        )

        user_state = {}
        context = CheckpointCallbackContext(
            state=state,
            checkpoint_name=checkpoint_name,
            checkpoint_type=CheckpointType.GLOBAL,
            user_state=user_state,
        )

        callback.on_checkpoint_load(context)

        assert user_state["ema_updates"] == 4
        assert user_state["ema_skipped_iters"] == 1
        assert "chunk0.weight" in user_state["ema_state"]

    def test_on_checkpoint_load_skips_for_finetune(self, tmp_path):
        callback = EMACallback()

        user_state = {}
        state = SimpleNamespace(
            train_state=SimpleNamespace(step=4),
            cfg=SimpleNamespace(
                checkpoint=SimpleNamespace(finetune=True),
            ),
        )

        context = CheckpointCallbackContext(
            state=state,
            checkpoint_name=str(tmp_path / "iter_0000005"),
            checkpoint_type=CheckpointType.GLOBAL,
            user_state=user_state,
        )

        callback.on_checkpoint_load(context)

        assert user_state == {}