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

"""Unit tests for get_batch_on_this_tp_rank in batch_utils.py.

The ``TestGetBatchOnThisTpRank`` class mocks all distributed operations so the
tests run without GPUs.

The ``TestGetBatchDistributed`` class spawns two processes (TP=2) with real NCCL
groups so that ``torch.distributed.broadcast`` and ``broadcast_object_list``
exercise genuine rank-0 -> rank-1 communication.
"""

from __future__ import annotations

import datetime
import os
import socket
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.bridge.training.utils.batch_utils import get_batch_on_this_tp_rank


def _make_cfg(pp_size: int = 1) -> MagicMock:
    """Create a minimal mock ConfigContainer."""
    cfg = MagicMock()
    cfg.model.pipeline_model_parallel_size = pp_size
    cfg.model.seq_length = 16
    cfg.train.micro_batch_size = 1
    cfg.dataset.create_attention_mask = False
    return cfg


def _make_pg() -> MagicMock:
    """Create a minimal mock pg_collection with TP/PP groups."""
    pg = MagicMock()
    pg.tp = MagicMock()
    pg.pp = MagicMock()
    return pg


def _make_batch(extra_keys: dict | None = None) -> dict[str, torch.Tensor]:
    """Create a synthetic dataloader batch (CPU tensors)."""
    batch: dict[str, torch.Tensor] = {
        "tokens": torch.randint(0, 100, (1, 16), dtype=torch.int64),
        "labels": torch.randint(0, 100, (1, 16), dtype=torch.int64),
        "loss_mask": torch.ones(1, 16, dtype=torch.float32),
        "position_ids": torch.arange(16, dtype=torch.int64).unsqueeze(0),
    }
    if extra_keys:
        batch.update(extra_keys)
    return batch


# All tests mock distributed so they run on CPU without a process group.
_DIST_PATCHES = {
    "torch.distributed.get_process_group_ranks": lambda *a, **kw: [0, 1, 2, 3, 4, 5, 6, 7],
    "torch.distributed.get_rank": lambda *a, **kw: 0,
    "torch.distributed.broadcast": lambda *a, **kw: None,
}


class TestGetBatchOnThisTpRank:
    """Tests for get_batch_on_this_tp_rank with mocked distributed ops."""

    @patch("torch.distributed.broadcast_object_list")
    @patch("torch.distributed.broadcast")
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_process_group_ranks", return_value=[0, 1, 2, 3, 4, 5, 6, 7])
    def test_standard_keys_returned(self, _ranks, _rank, _bcast, _obj_bcast):
        """TP rank 0 loads data and the result contains all standard keys."""
        _obj_bcast.side_effect = lambda obj_list, **kw: None

        result = get_batch_on_this_tp_rank(
            iter([_make_batch()]), _make_cfg(), pg_collection=_make_pg()
        )

        for key in ("tokens", "labels", "loss_mask", "position_ids"):
            assert key in result, f"missing standard key: {key}"

    @patch("torch.distributed.broadcast_object_list")
    @patch("torch.distributed.broadcast")
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_process_group_ranks", return_value=[0, 1, 2, 3, 4, 5, 6, 7])
    def test_extra_keys_broadcast(self, _ranks, _rank, _bcast, mock_obj_bcast):
        """Extra keys (e.g. packed-seq metadata) reach the output via broadcast_object_list."""
        extra = {
            "cu_seqlens": torch.tensor([0, 5, 10], dtype=torch.int32),
            "max_seqlen": torch.tensor(5, dtype=torch.int32),
        }
        # First call broadcasts has_extra flag; second call broadcasts the dict.
        # On TP rank 0 broadcast_object_list is a no-op (data stays in-place).
        mock_obj_bcast.side_effect = lambda obj_list, **kw: None

        result = get_batch_on_this_tp_rank(
            iter([_make_batch(extra_keys=extra)]), _make_cfg(), pg_collection=_make_pg()
        )

        assert "cu_seqlens" in result, "extra key cu_seqlens not in result"
        assert "max_seqlen" in result, "extra key max_seqlen not in result"
        assert torch.equal(result["cu_seqlens"].cpu(), extra["cu_seqlens"])

    @patch("torch.distributed.broadcast_object_list")
    @patch("torch.distributed.broadcast")
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_process_group_ranks", return_value=[0, 1, 2, 3, 4, 5, 6, 7])
    def test_no_extra_keys_skips_heavy_broadcast(self, _ranks, _rank, _bcast, mock_obj_bcast):
        """When the batch has only standard keys the heavy broadcast_object_list is skipped."""
        call_payloads: list = []

        def _capture(obj_list, **kw):
            call_payloads.append(obj_list[0])

        mock_obj_bcast.side_effect = _capture

        get_batch_on_this_tp_rank(
            iter([_make_batch()]), _make_cfg(), pg_collection=_make_pg()
        )

        # First (and only) call should be the has_extra flag = False.
        # The heavy second broadcast_object_list for the actual extra dict
        # should NOT be called.
        assert len(call_payloads) == 1, (
            f"Expected 1 broadcast_object_list call (flag only), got {len(call_payloads)}"
        )
        assert call_payloads[0] is False, (
            f"has_extra flag should be False, got {call_payloads[0]}"
        )

    @pytest.mark.parametrize(
        "is_first, is_last, broadcast_all, expected_count",
        [
            (True, False, False, 2),   # first stage: tokens, position_ids
            (False, True, False, 2),   # last stage: labels, loss_mask
            (False, False, False, 0),  # mid stage: nothing
            (True, False, True, 4),    # broadcast_all overrides PP filtering
        ],
        ids=["pp_first", "pp_last", "pp_mid", "broadcast_all"],
    )
    @patch("torch.distributed.broadcast_object_list")
    @patch("torch.distributed.broadcast")
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_process_group_ranks", return_value=[0, 1, 2, 3, 4, 5, 6, 7])
    def test_pp_stage_broadcast_filtering(
        self, _ranks, _rank, mock_bcast, _obj_bcast,
        is_first, is_last, broadcast_all, expected_count,
    ):
        """Only the keys relevant to the PP stage are broadcast."""
        _obj_bcast.side_effect = lambda obj_list, **kw: None
        with (
            patch("megatron.bridge.training.utils.batch_utils.is_pp_first_stage", return_value=is_first),
            patch("megatron.bridge.training.utils.batch_utils.is_pp_last_stage", return_value=is_last),
        ):
            get_batch_on_this_tp_rank(
                iter([_make_batch()]),
                _make_cfg(pp_size=2),
                pg_collection=_make_pg(),
                broadcast_all_keys=broadcast_all,
            )

        assert mock_bcast.call_count == expected_count


# ---------------------------------------------------------------------------
# Tests that exercise *real* distributed communication (TP=2, world_size=2)
# ---------------------------------------------------------------------------

def _gpu_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _dist_worker(rank: int, world_size: int, port: int, test_case: str) -> None:
    """Worker spawned by each distributed test (runs once per rank).

    Both ranks build the *same* deterministic source batch (via a fixed seed).
    Only rank 0 feeds it to ``get_batch_on_this_tp_rank``; rank 1 receives the
    data through real NCCL broadcasts.  Both ranks then verify the result
    against the known source.
    """
    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection

    os.environ.update({
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
        "WORLD_SIZE": str(world_size),
    })
    torch.cuda.set_device(rank % torch.cuda.device_count())

    dist.init_process_group(
        backend="nccl", world_size=world_size, rank=rank,
        timeout=datetime.timedelta(minutes=2),
    )
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
    )
    pg_collection = ProcessGroupCollection.use_mpu_process_groups(
        required_pgs=["tp", "pp"],
    )

    try:
        # Deterministic batch – identical on every rank so we can verify.
        torch.manual_seed(42)
        src_batch: dict = {
            "tokens": torch.randint(0, 100, (2, 16), dtype=torch.int64),
            "labels": torch.randint(0, 100, (2, 16), dtype=torch.int64),
            "loss_mask": torch.ones(2, 16, dtype=torch.float32),
            "position_ids": torch.arange(16, dtype=torch.int64).unsqueeze(0).expand(2, -1).contiguous(),
        }

        if test_case == "extra_tensor":
            src_batch["cu_seqlens"] = torch.tensor([0, 8, 16], dtype=torch.int32)
            src_batch["max_seqlen"] = torch.tensor(8, dtype=torch.int32)
        elif test_case == "non_tensor_extra":
            src_batch["metadata"] = {"source": "test", "version": 2}

        cfg = MagicMock()
        cfg.model.pipeline_model_parallel_size = 1
        cfg.model.seq_length = 16
        cfg.train.micro_batch_size = 2
        cfg.dataset.create_attention_mask = False

        # Only TP-rank 0 supplies data; rank 1 receives via broadcast.
        data_iter = iter([src_batch]) if rank == 0 else None

        result = get_batch_on_this_tp_rank(
            data_iter, cfg, pg_collection=pg_collection,
        )

        # -- assertions (run on every rank) ---------------------------------
        for key in ("tokens", "labels", "loss_mask", "position_ids"):
            assert key in result, f"rank {rank}: missing key {key}"
            assert result[key].is_cuda, f"rank {rank}: {key} not on CUDA"
            assert torch.equal(result[key].cpu(), src_batch[key]), (
                f"rank {rank}: value mismatch for {key}"
            )

        if test_case == "extra_tensor":
            for key in ("cu_seqlens", "max_seqlen"):
                assert key in result, f"rank {rank}: missing extra key {key}"
                assert result[key].is_cuda, f"rank {rank}: {key} not on CUDA"
                assert torch.equal(result[key].cpu(), src_batch[key]), (
                    f"rank {rank}: value mismatch for extra key {key}"
                )
        elif test_case == "non_tensor_extra":
            assert result["metadata"] == {"source": "test", "version": 2}, (
                f"rank {rank}: metadata mismatch"
            )

    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.skipif(not _gpu_available(), reason="requires at least 1 GPU")
class TestGetBatchDistributed:
    """Spawn two NCCL ranks (TP=2) and verify real broadcast communication.

    Rank 0 loads data and broadcasts to rank 1.  Both ranks assert that the
    received batch matches the known source tensors.
    """

    def test_standard_keys_roundtrip(self):
        """Standard batch keys survive a real rank-0 -> rank-1 broadcast."""
        mp.spawn(_dist_worker, nprocs=2, args=(2, _find_free_port(), "standard"))

    def test_extra_tensor_keys_roundtrip(self):
        """Extra tensor keys are broadcast via broadcast_object_list to rank 1."""
        mp.spawn(_dist_worker, nprocs=2, args=(2, _find_free_port(), "extra_tensor"))

    def test_non_tensor_extra_key(self):
        """Non-tensor extra values survive the object-list broadcast to rank 1."""
        mp.spawn(_dist_worker, nprocs=2, args=(2, _find_free_port(), "non_tensor_extra"))
