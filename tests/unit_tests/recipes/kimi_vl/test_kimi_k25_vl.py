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

#
# Test purpose:
# - Cover the previously untested kimi_k25_vl SFT recipe (issue #3177).
# - Monkeypatch AutoBridge to avoid HF Hub I/O.
# - Exercise the pipeline-layout helper (valid PP/VP combinations + the
#   error path for an unknown combination).
# - Sanity-check parallelism, MoE / Muon DDP wiring, mixed precision,
#   tokenizer wiring (vocab_size pulled from model), and the rope-fusion
#   gating of the experimental dist flag.
#

import importlib

import pytest
import torch

from megatron.bridge.recipes.kimi_vl.kimi_k25_vl import (
    _get_kimi_k25_vl_pipeline_layout,
    kimi_k25_vl_sft_config,
)
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


class _FakeKimiK25VLProvider:
    """Fake provider returned by AutoBridge.to_megatron_provider.

    Mirrors the attribute surface the recipe touches so that mutation,
    reads (vocab_size, apply_rope_fusion), and finalize() all succeed.
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.vocab_size = 163840
        self.apply_rope_fusion = False

    def finalize(self):
        return None


class _FakeAutoBridge:
    """AutoBridge stub that bypasses HF Hub network access."""

    @classmethod
    def from_hf_pretrained(cls, *args, **kwargs):
        return cls()

    def to_megatron_provider(self, *args, **kwargs):
        return _FakeKimiK25VLProvider()


@pytest.fixture(autouse=True)
def _patch_autobridge(monkeypatch):
    """Monkeypatch AutoBridge in the kimi_k25_vl recipe module to avoid HF I/O."""
    mod = importlib.import_module("megatron.bridge.recipes.kimi_vl.kimi_k25_vl")
    monkeypatch.setattr(mod, "AutoBridge", _FakeAutoBridge)


class TestKimiK25VLPipelineLayout:
    """Test cases for the _get_kimi_k25_vl_pipeline_layout helper."""

    def test_pipeline_layout_pp1_vp1(self):
        """Single-stage pipeline returns no layout."""
        assert _get_kimi_k25_vl_pipeline_layout(1, 1) is None

    def test_pipeline_layout_pp16_vp1_default(self):
        """PP=16/VP=1 (the SFT default) produces the expected stage breakdown."""
        layout = _get_kimi_k25_vl_pipeline_layout(16, 1)
        expected = [["embedding"] + ["decoder"] * 4] + [["decoder"] * 4] * 14 + [["decoder", "loss"]]
        assert layout == expected

    @pytest.mark.parametrize(
        "pp,vp",
        [(4, 1), (8, 1), (4, 2), (16, 1), (8, 2), (4, 4), (2, 8)],
    )
    def test_known_pp_vp_combinations_return_a_list(self, pp: int, vp: int):
        """All documented PP/VP combinations return a non-empty list-of-lists."""
        layout = _get_kimi_k25_vl_pipeline_layout(pp, vp)
        assert isinstance(layout, list)
        assert layout, "layout should not be empty for known combinations"
        for stage in layout:
            assert isinstance(stage, list)

    def test_pipeline_layout_vp_none_treated_as_one(self):
        """Passing vp_size=None is normalized to 1 and resolves like PP=16, VP=1."""
        assert _get_kimi_k25_vl_pipeline_layout(16, None) == _get_kimi_k25_vl_pipeline_layout(16, 1)

    def test_pipeline_layout_invalid_combination_raises(self):
        """An unknown PP/VP combination raises a clear ValueError."""
        with pytest.raises(ValueError, match="Invalid PP and VP size"):
            _get_kimi_k25_vl_pipeline_layout(3, 1)

    def test_pipeline_layout_returns_fresh_list(self):
        """The helper returns a deep-ish copy so mutation does not leak across calls."""
        layout_a = _get_kimi_k25_vl_pipeline_layout(16, 1)
        layout_b = _get_kimi_k25_vl_pipeline_layout(16, 1)
        # Different list objects so callers can mutate safely.
        assert layout_a is not layout_b
        for stage_a, stage_b in zip(layout_a, layout_b):
            assert stage_a is not stage_b


class TestKimiK25VLSftConfig:
    """Test cases for kimi_k25_vl_sft_config."""

    def test_sft_config_basic_structure(self):
        """SFT config is a valid ConfigContainer with all required components."""
        cfg = kimi_k25_vl_sft_config()

        assert isinstance(cfg, ConfigContainer)
        assert cfg.model is not None
        assert cfg.train is not None
        assert cfg.optimizer is not None
        assert cfg.scheduler is not None
        assert cfg.dataset is not None
        assert cfg.tokenizer is not None
        assert cfg.checkpoint is not None
        assert cfg.comm_overlap is not None

    def test_sft_config_default_training_settings(self):
        """Default training settings match the recipe contract."""
        cfg = kimi_k25_vl_sft_config()

        assert cfg.train.train_iters == 1_000_000
        assert cfg.train.global_batch_size == 4096
        assert cfg.train.micro_batch_size == 1
        assert cfg.train.manual_gc is True
        assert cfg.train.manual_gc_interval == 5
        assert cfg.train.manual_gc_eval == 5
        assert cfg.validation.eval_interval == 2000

    def test_sft_config_parallelism(self):
        """Default parallelism is TP=2, PP=16, EP=32, with sequence parallel on."""
        cfg = kimi_k25_vl_sft_config()

        assert cfg.model.tensor_model_parallel_size == 2
        assert cfg.model.pipeline_model_parallel_size == 16
        assert cfg.model.pipeline_dtype == torch.bfloat16
        assert cfg.model.virtual_pipeline_model_parallel_size is None
        assert cfg.model.context_parallel_size == 1
        assert cfg.model.expert_model_parallel_size == 32
        assert cfg.model.sequence_parallel is True
        assert cfg.model.expert_tensor_parallel_size == 1

    def test_sft_config_full_recompute(self):
        """Full activation recompute with uniform method (1 layer)."""
        cfg = kimi_k25_vl_sft_config()

        assert cfg.model.recompute_granularity == "full"
        assert cfg.model.recompute_modules is None
        assert cfg.model.recompute_method == "uniform"
        assert cfg.model.recompute_num_layers == 1
        assert cfg.model.fine_grained_activation_offloading is False
        assert cfg.model.offload_modules is None

    def test_sft_config_pipeline_split_settings(self):
        """Asymmetric pipeline split flags are off by default."""
        cfg = kimi_k25_vl_sft_config()

        assert cfg.model.account_for_embedding_in_pipeline_split is False
        assert cfg.model.account_for_loss_in_pipeline_split is False
        assert cfg.model.num_layers_in_first_pipeline_stage is None
        assert cfg.model.num_layers_in_last_pipeline_stage is None

    def test_sft_config_pipeline_layout_matches_helper(self):
        """The configured pipeline layout matches the helper's PP=16/VP=1 output."""
        cfg = kimi_k25_vl_sft_config()

        assert cfg.model.pipeline_model_parallel_layout == _get_kimi_k25_vl_pipeline_layout(16, 1)

    def test_sft_config_ddp_settings_for_muon(self):
        """DDP settings respect Muon's constraints (no dist optimizer, no param overlap)."""
        cfg = kimi_k25_vl_sft_config()

        assert cfg.ddp.use_distributed_optimizer is False
        assert cfg.ddp.overlap_param_gather is False
        assert cfg.ddp.overlap_grad_reduce is True
        assert cfg.ddp.grad_reduce_in_fp32 is True
        assert cfg.ddp.check_for_nan_in_grad is True
        assert cfg.ddp.use_megatron_fsdp is False
        assert cfg.ddp.average_in_collective is True
        assert cfg.ddp.data_parallel_sharding_strategy == "no_shard"

    def test_sft_config_dataset_configuration(self):
        """Dataset uses sequence length 4096 and the model's HF processor path."""
        cfg = kimi_k25_vl_sft_config()

        assert cfg.dataset.sequence_length == 4096
        assert cfg.dataset.num_workers == 8
        assert cfg.dataset.pack_sequences_in_batch is False
        assert cfg.dataset.hf_processor_path == "moonshotai/Kimi-K2.5"
        assert cfg.dataset.blend is None

    def test_sft_config_tokenizer_pulls_vocab_size_from_model(self):
        """Tokenizer wiring: NullTokenizer with vocab_size pulled from the provider."""
        cfg = kimi_k25_vl_sft_config()

        assert cfg.tokenizer.tokenizer_type == "NullTokenizer"
        assert cfg.tokenizer.tokenizer_model is None
        # The fake provider exposes vocab_size=163840.
        assert cfg.tokenizer.vocab_size == cfg.model.vocab_size == 163840

    def test_sft_config_mixed_precision(self):
        """Mixed precision is bf16 throughout, no autocast, fp32 grad reduce."""
        cfg = kimi_k25_vl_sft_config()

        assert isinstance(cfg.mixed_precision, MixedPrecisionConfig)
        assert cfg.mixed_precision.bf16 is True
        assert cfg.mixed_precision.params_dtype == torch.bfloat16
        assert cfg.mixed_precision.pipeline_dtype == torch.bfloat16
        assert cfg.mixed_precision.autocast_enabled is False
        assert cfg.mixed_precision.grad_reduce_in_fp32 is True

    def test_sft_config_optimizer_precision(self):
        """Optimizer precision is fp32 across grads, params, and moments."""
        cfg = kimi_k25_vl_sft_config()

        assert cfg.optimizer.use_precision_aware_optimizer is False
        assert cfg.optimizer.main_grads_dtype == torch.float32
        assert cfg.optimizer.main_params_dtype == torch.float32
        assert cfg.optimizer.exp_avg_dtype == torch.float32
        assert cfg.optimizer.exp_avg_sq_dtype == torch.float32

    def test_sft_config_uses_muon_optimizer(self):
        """The SFT config selects the distributed Muon optimizer."""
        cfg = kimi_k25_vl_sft_config()

        # distributed_muon_with_cosine_annealing sets optimizer="dist_muon".
        assert cfg.optimizer.optimizer == "dist_muon"

    def test_sft_config_moe_settings(self):
        """MoE wiring: alltoall dispatcher, deepep flex backend, grouped GEMM on."""
        cfg = kimi_k25_vl_sft_config()

        assert cfg.model.moe_token_dispatcher_type == "alltoall"
        assert cfg.model.moe_flex_dispatcher_backend == "deepep"
        assert cfg.model.moe_hybridep_num_sms == 16
        assert cfg.model.moe_router_fusion is False
        assert cfg.model.moe_permute_fusion is True
        assert cfg.model.moe_grouped_gemm is True
        assert cfg.model.moe_router_padding_for_fp8 is False
        assert cfg.model.moe_shared_expert_overlap is True
        assert cfg.model.moe_router_force_load_balancing is False

    def test_sft_config_transformer_engine_and_cuda_graph(self):
        """TE backend with CUDA graphs disabled by default (warmup steps still set)."""
        cfg = kimi_k25_vl_sft_config()

        assert cfg.model.transformer_impl == "transformer_engine"
        assert cfg.model.cuda_graph_impl == "none"
        assert cfg.model.cuda_graph_scope == "full"
        assert cfg.model.cuda_graph_warmup_steps == 3

    def test_sft_config_kernel_selections(self):
        """Default attention backend is None; cross-entropy fusion uses TE."""
        cfg = kimi_k25_vl_sft_config()

        assert cfg.model.attention_backend is None
        assert cfg.model.cross_entropy_loss_fusion is True
        assert cfg.model.cross_entropy_fusion_impl == "te"

    def test_sft_config_comm_overlap(self):
        """Comm overlap is off (TP overlap, wgrad delay, MoE EP overlap)."""
        cfg = kimi_k25_vl_sft_config()

        assert cfg.comm_overlap.tp_comm_overlap is False
        assert cfg.comm_overlap.delay_wgrad_compute is False
        assert cfg.comm_overlap.overlap_moe_expert_parallel_comm is False

    def test_sft_config_checkpoint(self):
        """Checkpoint cadence and async-save default."""
        cfg = kimi_k25_vl_sft_config()

        assert cfg.checkpoint.save_interval == 2000
        assert cfg.checkpoint.async_save is False

    def test_sft_config_rope_fusion_off_keeps_experimental_default(self):
        """When provider.apply_rope_fusion is False, experimental dist stays False (default)."""
        cfg = kimi_k25_vl_sft_config()

        assert cfg.dist.enable_megatron_core_experimental is False

    def test_sft_config_rope_fusion_on_enables_experimental(self, monkeypatch):
        """When provider.apply_rope_fusion is True, the experimental dist flag is enabled."""

        # Override the provider stub to flip apply_rope_fusion on.
        class _FakeProviderRopeOn(_FakeKimiK25VLProvider):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.apply_rope_fusion = True

        class _FakeBridgeRopeOn(_FakeAutoBridge):
            def to_megatron_provider(self, *args, **kwargs):
                return _FakeProviderRopeOn()

        mod = importlib.import_module("megatron.bridge.recipes.kimi_vl.kimi_k25_vl")
        monkeypatch.setattr(mod, "AutoBridge", _FakeBridgeRopeOn)

        cfg = kimi_k25_vl_sft_config()

        assert cfg.dist.enable_megatron_core_experimental is True
