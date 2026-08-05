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

"""Unit tests for the Megatron Bridge 0.5 Nemotron 3.5 Lightning recipes."""

from unittest.mock import patch

import pytest

from megatron.bridge.models.mamba.mamba_provider import MambaModelProvider
from megatron.bridge.recipes.nemotronh.nemotron_3_5_lightning import (
    NEMOTRON_3_5_LIGHTNING_HF_MODEL_ID,
    NEMOTRON_3_5_LIGHTNING_HF_MODEL_REVISION,
    OPENMATHINSTRUCT2_REVISION,
    nemotron_3_5_lightning_peft_config,
    nemotron_3_5_lightning_pretrain_8k_config,
    nemotron_3_5_lightning_pretrain_8k_fsdp_config,
    nemotron_3_5_lightning_pretrain_config,
    nemotron_3_5_lightning_sft_config,
    nemotron_3_5_lightning_sft_openmathinstruct2_packed_config,
    nemotron_3_5_lightning_sft_openmathinstruct2_packed_tp1_config,
)
from megatron.bridge.training.config import ConfigContainer, runtime_config_update
from megatron.bridge.utils.cuda_graph import cuda_graph_module_names


@pytest.mark.unit
class TestNemotron35LightningRecipes:
    """Verify the public checkpoint and MTP contract across recipe families."""

    @pytest.mark.parametrize(
        "recipe_factory",
        [
            nemotron_3_5_lightning_pretrain_config,
            nemotron_3_5_lightning_sft_config,
            nemotron_3_5_lightning_peft_config,
            nemotron_3_5_lightning_sft_openmathinstruct2_packed_config,
        ],
    )
    def test_common_checkpoint_contract(self, recipe_factory):
        cfg = recipe_factory()

        assert isinstance(cfg, ConfigContainer)
        assert isinstance(cfg.model, MambaModelProvider)
        assert cfg.model.mtp_num_layers == 2
        assert cfg.model.mtp_hybrid_override_pattern == "*E"
        assert cfg.model.mtp_use_repeated_layer is True
        assert cfg.model.keep_mtp_spec_in_bf16 is True
        assert cfg.model.mtp_loss_scaling_factor == 0.3
        assert cfg.model.hf_model_id == NEMOTRON_3_5_LIGHTNING_HF_MODEL_ID
        assert cfg.model.hf_model_revision == NEMOTRON_3_5_LIGHTNING_HF_MODEL_REVISION
        assert cfg.tokenizer.tokenizer_model == NEMOTRON_3_5_LIGHTNING_HF_MODEL_ID
        assert cfg.tokenizer.hf_tokenizer_kwargs == {"revision": NEMOTRON_3_5_LIGHTNING_HF_MODEL_REVISION}

    @pytest.mark.parametrize(
        "recipe_factory",
        [
            nemotron_3_5_lightning_pretrain_config,
            nemotron_3_5_lightning_sft_config,
            nemotron_3_5_lightning_peft_config,
            nemotron_3_5_lightning_sft_openmathinstruct2_packed_config,
            nemotron_3_5_lightning_pretrain_8k_config,
            nemotron_3_5_lightning_pretrain_8k_fsdp_config,
            nemotron_3_5_lightning_sft_openmathinstruct2_packed_tp1_config,
        ],
    )
    def test_recipe_finalizes(self, recipe_factory, tmp_path):
        cfg = recipe_factory()
        if cfg.peft is not None:
            if not hasattr(cfg.checkpoint, "pretrained_checkpoint"):
                raise ValueError("CheckpointConfig has no pretrained_checkpoint field")
            cfg.checkpoint.pretrained_checkpoint = str(tmp_path)
        with patch("megatron.bridge.training.config.get_world_size_safe", return_value=16):
            runtime_config_update(cfg)

    def test_h100_pretrain_contract(self):
        cfg = nemotron_3_5_lightning_pretrain_config()

        assert cfg.model.tensor_model_parallel_size == 1
        assert cfg.model.pipeline_model_parallel_size == 1
        assert cfg.model.context_parallel_size == 2
        assert cfg.model.cp_comm_type == "p2p"
        assert cfg.model.sequence_parallel is False
        assert cfg.model.expert_model_parallel_size == 8
        assert cfg.model.seq_length == 8192
        assert cfg.train.global_batch_size == 512
        assert cfg.train.micro_batch_size == 1
        assert cfg.model.moe_token_dispatcher_type == "flex"
        assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
        assert cfg.model.recompute_modules == ["moe", "layernorm", "core_attn"]
        assert cuda_graph_module_names(cfg.model) == ["mamba"]
        assert cfg.checkpoint.async_save is False
        assert cfg.ddp.average_in_collective is True

    def test_sft_and_peft_contract(self):
        sft_cfg = nemotron_3_5_lightning_sft_config()
        peft_cfg = nemotron_3_5_lightning_peft_config()

        assert sft_cfg.peft is None
        assert sft_cfg.optimizer.lr == 5e-6
        assert peft_cfg.peft is not None
        assert peft_cfg.optimizer.lr == 1e-4

    def test_packed_openmath_contract(self):
        cfg = nemotron_3_5_lightning_sft_openmathinstruct2_packed_config()

        assert cfg.model.seq_length == 4096
        assert cfg.model.tensor_model_parallel_size == 2
        assert cfg.model.sequence_parallel is True
        assert cfg.model.recompute_modules == ["moe", "layernorm", "core_attn", "mlp"]
        assert cfg.dataset.dataset_name == "nvidia/OpenMathInstruct-2"
        assert cfg.dataset.hf_kwargs == {"revision": OPENMATHINSTRUCT2_REVISION}
        assert cfg.dataset.packed_sequence_specs is not None
        assert cfg.dataset.packed_sequence_specs.packed_sequence_size == 4096
        assert cfg.dataset.packed_sequence_specs.pad_seq_to_mult == 2
        assert cfg.dataset.packed_sequence_specs.tokenizer_model_name == NEMOTRON_3_5_LIGHTNING_HF_MODEL_ID
        assert cfg.train.train_iters == 100
        assert cfg.train.global_batch_size == 128
        assert cfg.train.micro_batch_size == 1
        assert cfg.train.empty_unused_memory_level == 2
        assert cfg.validation.eval_iters == 0
        assert cfg.checkpoint.save_optim is False
        assert cfg.checkpoint.save_rng is False

    def test_gb200_execution_variants(self):
        pretrain_cfg = nemotron_3_5_lightning_pretrain_8k_config()
        fsdp_cfg = nemotron_3_5_lightning_pretrain_8k_fsdp_config()
        sft_cfg = nemotron_3_5_lightning_sft_openmathinstruct2_packed_tp1_config()

        assert pretrain_cfg.train.micro_batch_size == 2
        assert pretrain_cfg.model.context_parallel_size == 1
        assert pretrain_cfg.model.cuda_graph_impl == "none"
        assert pretrain_cfg.model.recompute_modules is None
        assert fsdp_cfg.dist.use_megatron_fsdp is True
        assert fsdp_cfg.ddp.use_megatron_fsdp is True
        assert fsdp_cfg.checkpoint.ckpt_format == "fsdp_dtensor"
        assert sft_cfg.model.tensor_model_parallel_size == 1
        assert sft_cfg.model.sequence_parallel is False
        assert sft_cfg.model.moe_flex_dispatcher_backend == "hybridep"
        assert sft_cfg.model.moe_hybridep_num_sms == 32
