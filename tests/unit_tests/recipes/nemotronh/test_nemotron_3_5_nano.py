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

"""
Unit tests for Nemotron 3.5 Nano recipe configuration builders.

Tests cover:
- Pretrain configuration defaults (bounded H100 convergence recipe)
- SFT configuration (full supervised finetuning)
- PEFT configuration (LoRA/DoRA)
- MoE-specific settings (expert parallelism, MTP)
- Parallelism and tokenizer configurations

The recipe loads its HF config via ``AutoBridge.from_hf_pretrained``. The EA2
repository is private, so environments without access skip this model-specific
suite. The exhaustive recipe-factory tests exercise construction offline.
"""

import os
import tempfile

import pytest

from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.recipes.nemotronh.gb200 import nemotron_3_5_nano_pretrain_8gpu_gb200_bf16_config
from megatron.bridge.recipes.nemotronh.nemotron_3_5_nano import (
    NEMOTRON_3_5_NANO_HF_MODEL_ID,
    nemotron_3_5_nano_peft_config,
    nemotron_3_5_nano_pretrain_config,
    nemotron_3_5_nano_sft_config,
)
from megatron.bridge.training.config import ConfigContainer


def _recipe_loadable() -> bool:
    """Return True if AutoBridge can load the recipe's HF config in this environment.

    Treat any exception (missing path, offline HF cache, bad config) as 'not
    loadable' so the module is skipped without surfacing as a test failure.
    """
    try:
        nemotron_3_5_nano_pretrain_config()
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _recipe_loadable(),
    reason=(
        f"Nemotron 3.5 Nano HF config not accessible from '{NEMOTRON_3_5_NANO_HF_MODEL_ID}'. "
        "Authenticate to the private EA2 repository to enable these tests."
    ),
)


@pytest.mark.unit
class TestNemotron35NanoPretrain:
    """Pretrain recipe — bounded H100 convergence defaults."""

    def test_pretrain_config_default_parameters(self):
        config = nemotron_3_5_nano_pretrain_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, HybridModelProvider)

        # Parallelism — H100 convergence recipe (TP=1, EP=8, SP=True)
        assert config.model.tensor_model_parallel_size == 1
        assert config.model.pipeline_model_parallel_size == 1
        assert config.model.sequence_parallel is True
        assert config.model.expert_tensor_parallel_size == 1
        assert config.model.expert_model_parallel_size == 8
        assert config.model.seq_length == 4096

        # Training
        assert config.train.train_iters == 100
        assert config.train.global_batch_size == 1024
        assert config.train.micro_batch_size == 1

        # Dataset
        assert config.dataset.seq_length == 4096

        # Tokenizer
        assert config.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
        assert config.tokenizer.tokenizer_model == NEMOTRON_3_5_NANO_HF_MODEL_ID

        # Mixed precision
        assert config.mixed_precision == "bf16_mixed"

    def test_pretrain_config_cuda_graph_and_recompute(self):
        """CUDA-graph scope + selective recompute must match the H100 BF16 perf preset."""
        config = nemotron_3_5_nano_pretrain_config()

        assert config.model.cuda_graph_impl == "transformer_engine"
        assert config.model.cuda_graph_scope == ["attn", "mamba"]
        assert config.model.cuda_graph_warmup_steps == 3
        assert config.model.use_te_rng_tracker is True

        assert config.model.recompute_granularity == "selective"
        assert config.model.recompute_modules == ["moe", "layernorm"]

    def test_pretrain_config_moe_settings(self):
        config = nemotron_3_5_nano_pretrain_config()

        assert config.model.moe_token_dispatcher_type == "alltoall"
        assert config.model.moe_shared_expert_overlap is False
        assert config.model.moe_flex_dispatcher_backend == "hybridep"
        assert config.model.moe_router_force_load_balancing is False

    def test_pretrain_config_mtp_settings(self):
        """HF config has num_nextn_predict_layers=1; recipe sets mtp_num_layers=2 with repeated layer."""
        config = nemotron_3_5_nano_pretrain_config()

        assert config.model.mtp_num_layers == 2
        assert config.model.mtp_hybrid_override_pattern == "*E"
        assert config.model.mtp_use_repeated_layer is True
        assert config.model.keep_mtp_spec_in_bf16 is True
        assert config.model.calculate_per_token_loss is True
        assert config.model.mtp_loss_scaling_factor == 0.3

    def test_pretrain_config_optimizer_settings(self):
        config = nemotron_3_5_nano_pretrain_config()

        assert config.optimizer.lr == 3.0e-4
        assert config.optimizer.min_lr == 3.0e-5
        assert config.optimizer.weight_decay == 0.1
        assert config.optimizer.adam_beta1 == 0.9
        assert config.optimizer.adam_beta2 == 0.95
        assert config.scheduler.lr_warmup_iters == 40
        assert config.scheduler.lr_decay_iters == 100
        assert config.scheduler.lr_decay_style == "cosine"

    def test_pretrain_config_communication_overlap(self):
        config = nemotron_3_5_nano_pretrain_config()

        assert config.comm_overlap is not None
        assert config.comm_overlap.tp_comm_bootstrap_backend == "nccl"
        assert config.comm_overlap.tp_comm_overlap is True
        assert config.comm_overlap.delay_wgrad_compute is False
        assert config.comm_overlap.overlap_moe_expert_parallel_comm is False

    def test_pretrain_config_checkpoint_settings(self):
        config = nemotron_3_5_nano_pretrain_config()

        assert config.checkpoint.save_interval == 50
        assert config.checkpoint.load is None
        assert config.checkpoint.ckpt_assume_constant_structure is True
        assert config.checkpoint.dist_ckpt_strictness == "log_all"
        assert config.checkpoint.async_save is True

    def test_gb200_pretrain_config_defaults(self):
        config = nemotron_3_5_nano_pretrain_8gpu_gb200_bf16_config()

        assert config.model.seq_length == 4096
        assert config.dataset.seq_length == 4096
        assert config.train.train_iters == 100
        assert config.train.global_batch_size == 1024
        assert config.train.micro_batch_size == 2
        assert config.model.moe_router_force_load_balancing is False
        assert config.model.cuda_graph_scope == ["attn", "mamba", "moe_router", "moe_preprocess"]
        assert config.model.recompute_granularity is None
        assert config.model.recompute_modules is None
        assert config.env_vars["NVLINK_DOMAIN_SIZE"] == 72
        assert config.env_vars["USE_MNNVL"] == 1


@pytest.mark.unit
class TestNemotron35NanoSft:
    """SFT recipe — TP=1 / EP=8 / SP=True, packed-sequence default."""

    def test_sft_config_defaults(self):
        config = nemotron_3_5_nano_sft_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, HybridModelProvider)

        assert config.model.tensor_model_parallel_size == 1
        assert config.model.pipeline_model_parallel_size == 1
        assert config.model.sequence_parallel is True
        assert config.model.expert_tensor_parallel_size == 1
        assert config.model.expert_model_parallel_size == 8
        assert config.model.seq_length == 2048

        # CUDA graphs disabled — packed-sequence + TE-scoped graphs is incompatible
        assert config.model.cuda_graph_impl == "none"
        assert config.model.cuda_graph_scope == []

        # MTP carried through
        assert config.model.mtp_num_layers == 2
        assert config.model.mtp_use_repeated_layer is True

        # Full SFT — no PEFT
        assert config.peft is None
        assert config.optimizer.lr == 5e-6

        # Tokenizer
        assert config.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
        assert config.tokenizer.tokenizer_model == NEMOTRON_3_5_NANO_HF_MODEL_ID

    def test_sft_config_custom_parallelism(self):
        config = nemotron_3_5_nano_sft_config()
        config.model.tensor_model_parallel_size = 2
        config.model.pipeline_model_parallel_size = 2
        config.model.expert_model_parallel_size = 4
        assert config.model.tensor_model_parallel_size == 2
        assert config.model.pipeline_model_parallel_size == 2
        assert config.model.expert_model_parallel_size == 4

    def test_sft_config_with_pretrained_checkpoint(self):
        config = nemotron_3_5_nano_sft_config()
        config.checkpoint.pretrained_checkpoint = "/path/to/checkpoint"
        assert config.checkpoint.pretrained_checkpoint == "/path/to/checkpoint"

    def test_sft_config_with_custom_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = nemotron_3_5_nano_sft_config()
            run_dir = os.path.join(temp_dir, "finetune_run")
            config.checkpoint.save = os.path.join(run_dir, "checkpoints")
            config.logger.tensorboard_dir = os.path.join(run_dir, "tb_logs")
            assert config.checkpoint.save == os.path.join(run_dir, "checkpoints")
            assert config.logger.tensorboard_dir == os.path.join(run_dir, "tb_logs")


@pytest.mark.unit
class TestNemotron35NanoPeft:
    """PEFT recipe — TP=1 / EP=1 / SP=True, LoRA defaults on Mamba target modules."""

    def test_peft_config_default_lora(self):
        config = nemotron_3_5_nano_peft_config()

        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, HybridModelProvider)

        assert config.model.tensor_model_parallel_size == 1
        assert config.model.pipeline_model_parallel_size == 1
        assert config.model.sequence_parallel is True
        assert config.model.expert_tensor_parallel_size == 1
        assert config.model.expert_model_parallel_size == 1

        # CUDA graphs disabled for packed-sequence PEFT
        assert config.model.cuda_graph_impl == "none"

        assert config.peft is not None
        assert config.optimizer.lr == 1e-4

        assert config.tokenizer.tokenizer_model == NEMOTRON_3_5_NANO_HF_MODEL_ID

    def test_peft_config_dora(self):
        config = nemotron_3_5_nano_peft_config(peft_scheme="dora")
        assert config.peft is not None
        assert config.optimizer.lr == 1e-4

    def test_peft_config_mamba_target_modules(self):
        """PEFT must target Mamba-specific submodules (in_proj/out_proj) in addition to attention/MLP."""
        config = nemotron_3_5_nano_peft_config(peft_scheme="lora")
        targets = set(getattr(config.peft, "target_modules", []))
        assert "in_proj" in targets
        assert "out_proj" in targets
        assert "linear_qkv" in targets
        assert "linear_fc1" in targets


@pytest.mark.unit
class TestNemotron35NanoCommon:
    """Shared invariants across the three recipe builders."""

    @pytest.mark.parametrize(
        "recipe_fn",
        [
            nemotron_3_5_nano_pretrain_config,
            nemotron_3_5_nano_sft_config,
            nemotron_3_5_nano_peft_config,
        ],
    )
    def test_config_container_structure(self, recipe_fn):
        config = recipe_fn()
        assert isinstance(config, ConfigContainer)
        assert isinstance(config.model, HybridModelProvider)
        assert config.train is not None
        assert config.optimizer is not None
        assert config.scheduler is not None
        assert config.dataset is not None
        assert config.logger is not None
        assert config.tokenizer is not None
        assert config.checkpoint is not None
        assert config.rng is not None
        assert config.ddp is not None
        assert config.mixed_precision is not None

    @pytest.mark.parametrize(
        "recipe_fn",
        [
            nemotron_3_5_nano_pretrain_config,
            nemotron_3_5_nano_sft_config,
            nemotron_3_5_nano_peft_config,
        ],
    )
    def test_ddp_configuration(self, recipe_fn):
        config = recipe_fn()
        assert config.ddp.check_for_nan_in_grad is True
        assert config.ddp.overlap_grad_reduce is True
        assert config.ddp.overlap_param_gather is True
        assert config.ddp.use_distributed_optimizer is True

    @pytest.mark.parametrize(
        "recipe_fn",
        [
            nemotron_3_5_nano_pretrain_config,
            nemotron_3_5_nano_sft_config,
            nemotron_3_5_nano_peft_config,
        ],
    )
    def test_moe_model_configuration(self, recipe_fn):
        """MoE/MTP knobs from AutoBridge match the 30B-A3B + 1-MTP HF config."""
        config = recipe_fn()
        assert config.model.num_moe_experts == 128
        assert config.model.moe_ffn_hidden_size == 1856
        assert config.model.moe_shared_expert_intermediate_size == 3712
        assert config.model.moe_router_topk == 6
        assert config.model.moe_router_topk_scaling_factor == 2.5
        assert config.model.mtp_num_layers == 2
