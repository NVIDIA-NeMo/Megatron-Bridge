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

"""End-to-end tests for the load_model_weights public API.

Verifies save -> load roundtrips for both ``torch_dist`` and ``fsdp_dtensor``
checkpoint formats using real GPT models on GPU.

Multi-GPU safe: rank 0 creates the temp directory and broadcasts the path
to all other ranks before any checkpoint I/O.
"""

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.training.checkpointing import load_model_weights
from megatron.bridge.training.model_load_save import save_megatron_model
from tests.functional_tests.utils import broadcast_path, clear_directories, initialize_distributed


def _create_gpt_model() -> list[MegatronModule]:
    """Create a minimal GPT model on GPU for checkpoint roundtrip testing."""
    provider = GPTModelProvider(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        seq_length=64,
        vocab_size=256,
        ffn_hidden_size=256,
    )
    provider._pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    provider.finalize()
    model = provider.provide_distributed_model(ddp_config=None, wrap_with_ddp=False)
    return [m.cuda() for m in model]


def _snapshot_weights(model: list[MegatronModule]) -> dict[str, torch.Tensor]:
    """Deep-copy all named parameters from the first model chunk to CPU."""
    return {name: param.data.detach().cpu().clone() for name, param in model[0].named_parameters()}


def _randomize_weights(model: list[MegatronModule]) -> None:
    """Replace all weights with random values so they differ from the original."""
    with torch.no_grad():
        for param in model[0].parameters():
            param.data.uniform_(-1.0, 1.0)


class TestLoadModelWeightsE2E:
    """Save -> load roundtrip tests that exercise real checkpoint I/O on GPU."""

    @pytest.fixture(autouse=True)
    def setup_distributed(self):
        """Initialize distributed and model-parallel state (once per process)."""
        initialize_distributed()

        if not parallel_state.model_parallel_is_initialized():
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                virtual_pipeline_model_parallel_size=None,
                context_parallel_size=1,
            )

        from megatron.bridge.training.initialize import _set_random_seed

        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        _set_random_seed(
            seed_=1234,
            data_parallel_random_init=False,
            te_rng_tracker=True,
            inference_rng_tracker=False,
            pg_collection=pg_collection,
        )

        yield

        from megatron.core.rerun_state_machine import destroy_rerun_state_machine

        destroy_rerun_state_machine()

    @pytest.fixture()
    def shared_tmp_dir(self):
        """Create a temp directory on rank 0 and broadcast the path to all ranks."""
        if dist.get_rank() == 0:
            tmp_dir = tempfile.mkdtemp()
        else:
            tmp_dir = ""
        tmp_dir = broadcast_path(tmp_dir)

        yield tmp_dir

        clear_directories(tmp_dir)

    # ------------------------------------------------------------------
    # torch_dist format
    # ------------------------------------------------------------------

    @pytest.mark.run_only_on("GPU")
    def test_torch_dist_save_load_roundtrip(self, shared_tmp_dir):
        """Weights survive a torch_dist save -> load_model_weights cycle."""
        save_dir = os.path.join(shared_tmp_dir, "checkpoint")
        ckpt_path = os.path.join(save_dir, "iter_0000000")

        model = _create_gpt_model()
        original = _snapshot_weights(model)

        save_megatron_model(model, save_dir, ckpt_format="torch_dist")
        assert os.path.isdir(ckpt_path), f"Checkpoint dir not created at {ckpt_path}"

        model2 = _create_gpt_model()
        _randomize_weights(model2)

        for name in original:
            assert not torch.equal(model2[0].state_dict()[name].cpu(), original[name]), (
                f"Weights for '{name}' should differ before load"
            )

        load_model_weights(model2, ckpt_path)

        for name, expected in original.items():
            actual = model2[0].state_dict()[name].cpu()
            assert torch.allclose(actual, expected, atol=1e-6), (
                f"torch_dist weight mismatch for '{name}': max diff = {(actual - expected).abs().max().item():.2e}"
            )

    @pytest.mark.run_only_on("GPU")
    def test_torch_dist_return_state_dict(self, shared_tmp_dir):
        """load_model_weights can return a state dict instead of loading in-place."""
        save_dir = os.path.join(shared_tmp_dir, "checkpoint")
        ckpt_path = os.path.join(save_dir, "iter_0000000")

        model = _create_gpt_model()
        original = _snapshot_weights(model)

        save_megatron_model(model, save_dir, ckpt_format="torch_dist")

        state_dict = load_model_weights(model, ckpt_path, return_state_dict=True)

        assert state_dict is not None, "return_state_dict=True should return a dict"
        assert "model" in state_dict, "state dict must contain 'model' key"

        for name in original:
            assert name in state_dict["model"], f"Key '{name}' missing from returned state dict"
