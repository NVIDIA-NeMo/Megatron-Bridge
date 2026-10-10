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

"""The entry points an RL framework drives: model config, model build, container, overrides."""

import os
import sys

import pytest
import torch


def _tiny_checkpoint(tmp_path):
    """Save the tiny HF checkpoint used by the functional fixtures."""
    root = os.path.join(os.path.dirname(__file__), "..", "..", "..", "functional_tests", "test_groups", "models")
    sys.path.insert(0, os.path.abspath(root))
    from shensi import make_tiny_hf_model

    path = tmp_path / "shensi_rl_tiny"
    make_tiny_hf_model().save_pretrained(path, safe_serialization=False)
    return path


@pytest.fixture()
def distributed(tmp_path_factory):
    """Single-rank process group and Megatron-Core model-parallel state."""
    if not torch.cuda.is_available():
        pytest.skip("the Shensi RL-integration test needs a GPU")
    import torch.distributed as dist
    from megatron.core import parallel_state

    created_group = False
    if dist.is_initialized():
        if dist.get_world_size() != 1:
            pytest.skip("runs on a single rank")
    else:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        store = tmp_path_factory.mktemp("shensi_rl_dist") / "store"
        dist.init_process_group(backend="nccl", init_method=f"file://{store}", rank=0, world_size=1)
        created_group = True
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
    )
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(0)
    yield
    parallel_state.destroy_model_parallel()
    if created_group:
        dist.destroy_process_group()


def test_model_config_and_model_build(distributed, tmp_path):
    """`get_model_config()` / `get_model()` are the builder path RL frameworks use."""
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.shensi.shensi_builder import ShensiModelConfig

    bridge = AutoBridge.from_hf_pretrained(_tiny_checkpoint(tmp_path))
    model_config = bridge.get_model_config()
    assert isinstance(model_config, ShensiModelConfig)
    assert model_config.vocab_size and model_config.seq_length
    assert model_config.transformer.num_layers

    models = bridge.get_model(wrap_with_ddp=False)
    assert len(models) == 1
    assert type(models[0]).__name__ == "ShensiModel"


def test_provider_overrides_and_config_container(distributed, tmp_path):
    """`apply_overrides_and_finalize` + `ConfigContainer.validate` are the provider path."""
    from megatron.bridge import AutoBridge
    from megatron.bridge.training.config import (
        CheckpointConfig,
        ConfigContainer,
        DistributedDataParallelConfig,
        LoggerConfig,
        OptimizerConfig,
        RNGConfig,
        SchedulerConfig,
        TokenizerConfig,
        TrainingConfig,
    )

    bridge = AutoBridge.from_hf_pretrained(_tiny_checkpoint(tmp_path))
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.apply_overrides_and_finalize(
        dtype=torch.bfloat16,
        overrides={
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "recompute_granularity": "full",
            "recompute_method": "block",
            "recompute_num_layers": 1,
        },
    )
    assert provider.params_dtype == torch.bfloat16
    assert provider.bf16 and not provider.fp16
    assert provider.recompute_granularity == "full"

    cfg = ConfigContainer(
        model=provider,
        train=TrainingConfig(micro_batch_size=1, global_batch_size=2, train_iters=1),
        optimizer=OptimizerConfig(),
        scheduler=SchedulerConfig(),
        ddp=DistributedDataParallelConfig(),
        checkpoint=CheckpointConfig(),
        logger=LoggerConfig(),
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=provider.vocab_size),
        dataset=None,
        rng=RNGConfig(),
    )
    cfg.validate()


def test_forward_accepts_keyword_inputs_only(distributed, tmp_path):
    """RL logprob calls pass `input_ids` and may omit `position_ids`."""
    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(_tiny_checkpoint(tmp_path))
    model = bridge.get_model(wrap_with_ddp=False)[0]
    tokens = torch.randint(1, 32, (1, 8), device="cuda")
    logits = model(input_ids=tokens)
    assert logits.shape[:2] == (1, 8)  # [batch, seq, vocab]


def test_packed_sequences_are_rejected(distributed, tmp_path):
    """A packed batch must fail loudly instead of silently ranking across segments."""
    from megatron.core.packed_seq_params import PackedSeqParams

    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(_tiny_checkpoint(tmp_path))
    model = bridge.get_model(wrap_with_ddp=False)[0]
    tokens = torch.randint(1, 32, (1, 8), device="cuda")
    packed = PackedSeqParams(
        cu_seqlens_q=torch.tensor([0, 8], dtype=torch.int32, device="cuda"),
        cu_seqlens_kv=torch.tensor([0, 8], dtype=torch.int32, device="cuda"),
        qkv_format="thd",
    )
    with pytest.raises(NotImplementedError, match="packed sequences"):
        model(input_ids=tokens, packed_seq_params=packed)
