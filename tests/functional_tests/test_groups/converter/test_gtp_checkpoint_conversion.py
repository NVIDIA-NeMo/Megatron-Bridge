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

"""Two-GPU HF conversion and model/RNG checkpoint coverage for GTP."""

import gc
from pathlib import Path

import pytest
import torch
from megatron.core.num_microbatches_calculator import init_num_microbatches_calculator
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.gtp_api import HAVE_GTP
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, get_gtp_remat_rng_tracker_name
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.gpt.vanilla_gpt import vanilla_gpt_pretrain_config
from megatron.bridge.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.bridge.training.initialize import destroy_global_state, init_rerun_state
from megatron.bridge.training.model_load_save import load_megatron_model
from megatron.bridge.training.state import GlobalState
from tests.functional_tests.utils import broadcast_path, initialize_distributed


@pytest.fixture(scope="session", autouse=True)
def ensure_test_data():
    """This module generates its tiny HF checkpoints and needs no downloaded data."""
    yield


def _save_toy_hf_model(path: Path, *, moe: bool, builder: bool) -> None:
    torch.manual_seed(1234)
    config_cls = Qwen3MoeConfig if moe else Qwen3Config
    model_cls = Qwen3MoeForCausalLM if moe else Qwen3ForCausalLM
    if builder:
        config_cls, model_cls = LlamaConfig, LlamaForCausalLM
    kwargs = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        attention_dropout=0.0,
    )
    if moe:
        kwargs.update(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=64)
    model_cls(config_cls(**kwargs)).bfloat16().save_pretrained(path)


def _assert_export_matches(bridge, models) -> None:
    source = bridge.hf_pretrained.state
    seen = set()
    for name, tensor in bridge.export_hf_weights(models, show_progress=False):
        assert torch.equal(tensor.cpu(), source[name].cpu()), name
        seen.add(name)
    assert seen == set(source.keys())


@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize(
    "topology,model_api",
    [
        ("dense_gtp", "provider"),
        ("expert_gtp", "provider"),
        ("tp", "provider"),
        ("dense_gtp", "builder"),
        ("tp", "builder"),
    ],
)
@pytest.mark.parametrize("fully_parallel", [False, True])
def test_gtp_hf_roundtrip_and_checkpoint(tmp_path, topology, fully_parallel, model_api):
    if topology != "tp" and not HAVE_GTP:
        pytest.skip("GTP requires TransformerEngine >= 2.19")
    initialize_distributed()
    if torch.distributed.get_world_size() != 2:
        pytest.skip("This test requires exactly two distributed ranks")
    root = Path(broadcast_path(tmp_path))
    hf_path = root / "hf"
    if torch.distributed.get_rank() == 0:
        _save_toy_hf_model(hf_path, moe=topology == "expert_gtp", builder=model_api == "builder")
    torch.distributed.barrier()

    models = None
    try:
        bridge = AutoBridge.from_hf_pretrained(str(hf_path), torch_dtype=torch.bfloat16)
        provider = (
            bridge.to_megatron_provider(load_weights=True) if model_api == "provider" else bridge.get_model_config()
        )
        provider.tensor_model_parallel_size = 2 if topology == "tp" else 1
        provider.tensor_parallel_num_weight_shards = 2 if topology in ("tp", "dense_gtp") else 1
        provider.expert_tensor_parallel_size = 1
        provider.expert_tensor_parallel_num_weight_shards = 2 if topology == "expert_gtp" else 1
        provider.pipeline_dtype = torch.bfloat16
        provider.params_dtype = torch.bfloat16
        provider.sequence_parallel = topology == "tp"
        provider.seq_length = 32
        provider.finalize()
        if model_api == "provider":
            provider.initialize_model_parallel(seed=1234)
            models = provider.provide_distributed_model(wrap_with_ddp=False)
        else:
            models = bridge.get_model(provider, wrap_with_ddp=False)
        pg = ProcessGroupCollection.use_mpu_process_groups()
        gtp_params = [p for m in models for p in m.parameters() if getattr(p, "is_gtp_weight_remat", False)]
        assert bool(gtp_params) == (topology != "tp")
        _assert_export_matches(bridge, models)

        cfg = vanilla_gpt_pretrain_config()
        cfg.model = provider
        cfg.optimizer.use_distributed_optimizer = False
        cfg.checkpoint.save = str(root / "checkpoint")
        cfg.checkpoint.load = cfg.checkpoint.save
        cfg.checkpoint.ckpt_format = "torch_dist"
        cfg.checkpoint.async_save = False
        cfg.checkpoint.fully_parallel_save = fully_parallel
        cfg.checkpoint.fully_parallel_load = fully_parallel
        cfg.checkpoint.save_optim = False
        cfg.checkpoint.load_optim = False
        cfg.checkpoint.save_rng = True
        cfg.checkpoint.load_rng = True
        cfg.rng.data_parallel_random_init = True
        init_num_microbatches_calculator(
            rank=torch.distributed.get_rank(),
            global_batch_size=cfg.train.global_batch_size,
            micro_batch_size=cfg.train.micro_batch_size,
            data_parallel_size=pg.dp_cp_gtp_remat.size(),
        )
        init_rerun_state(cfg.rerun_state_machine)
        state = GlobalState()
        state.cfg = cfg
        state.train_state.step = 3

        # Distinct per-rank streams expose accidental replica restoration.
        torch.manual_seed(5000 + torch.distributed.get_rank())
        torch.cuda.manual_seed(6000 + torch.distributed.get_rank())
        rng_tracker = get_cuda_rng_tracker()
        rng_names = tuple(sorted(rng_tracker.get_states()))
        assert rng_names
        if topology != "tp":
            assert get_gtp_remat_rng_tracker_name(is_expert=topology == "expert_gtp") in rng_names
        for name in rng_names:
            with rng_tracker.fork(name):
                torch.rand(11 * (torch.distributed.get_rank() + 1), device="cuda")
        save_checkpoint(state, models, None, None, 0, pg_collection=pg)
        expected_cpu = torch.rand(8)
        expected_cuda = torch.rand(8, device="cuda")
        expected_named_cuda = {}
        for name in rng_names:
            with rng_tracker.fork(name):
                expected_named_cuda[name] = torch.rand(8, device="cuda")
                torch.rand(17, device="cuda")
        with torch.no_grad():
            for model in models:
                for param in model.parameters():
                    param.add_(1)
        torch.manual_seed(9999)
        torch.cuda.manual_seed(9999)
        iteration, _ = load_checkpoint(state, models, None, None, pg_collection=pg)
        assert iteration == 3
        assert torch.equal(torch.rand(8), expected_cpu)
        assert torch.equal(torch.rand(8, device="cuda"), expected_cuda)
        assert set(rng_tracker.get_states()) == set(rng_names)
        for name in rng_names:
            with rng_tracker.fork(name):
                assert torch.equal(torch.rand(8, device="cuda"), expected_named_cuda[name]), name
        _assert_export_matches(bridge, models)

        if topology == "dense_gtp" and model_api == "provider":
            # The training entrypoint must reject unsafe GTP resharding before
            # generating a loading scaffold or writing any model parameters.
            saved_shards = provider.tensor_parallel_num_weight_shards
            saved_remat = provider.gtp_weight_remat_size
            saved_finetune = cfg.checkpoint.finetune
            try:
                provider.tensor_parallel_num_weight_shards = provider.tensor_model_parallel_size
                provider.gtp_weight_remat_size = 1
                for finetune in (False, True):
                    cfg.checkpoint.finetune = finetune
                    with pytest.raises(ValueError, match="Resharding a GTP checkpoint"):
                        load_checkpoint(state, models, None, None, pg_collection=pg)
            finally:
                provider.tensor_parallel_num_weight_shards = saved_shards
                provider.gtp_weight_remat_size = saved_remat
                cfg.checkpoint.finetune = saved_finetune
            _assert_export_matches(bridge, models)

        # Preserve the saved GTP topology when loading through the model-only export API.
        del models
        models = None
        destroy_global_state()
        checkpoint_path = str(root / "checkpoint" / "iter_0000003")
        overrides = None
        if topology != "tp":
            with pytest.raises(ValueError, match="Resharding a GTP checkpoint"):
                load_megatron_model(checkpoint_path)
            overrides = {
                "tensor_model_parallel_size": provider.tensor_model_parallel_size,
                "tensor_parallel_num_weight_shards": provider.tensor_parallel_num_weight_shards,
                "expert_tensor_parallel_size": provider.expert_tensor_parallel_size,
                "expert_tensor_parallel_num_weight_shards": provider.expert_tensor_parallel_num_weight_shards,
            }
        models = load_megatron_model(checkpoint_path, mp_overrides=overrides)
        _assert_export_matches(bridge, models)
    finally:
        del models
        gc.collect()
        destroy_global_state()
        torch.cuda.empty_cache()
