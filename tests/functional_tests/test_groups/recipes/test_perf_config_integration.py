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

"""Functional tests for flat performance recipe integration."""

import importlib
import inspect
import sys
from pathlib import Path

import pytest


SCRIPTS_PERF_PATH = Path(__file__).parents[4] / "scripts" / "performance"
sys.path.insert(0, str(SCRIPTS_PERF_PATH))


class _OfflineKimiProvider:
    """Minimal Kimi provider for env-only tests that must run with HF offline."""

    vocab_size = 163840
    apply_rope_fusion = False

    def finalize(self):
        return None


class _OfflineKimiAutoBridge:
    @classmethod
    def from_hf_pretrained(cls, *args, **kwargs):
        return cls()

    def to_megatron_provider(self, *args, **kwargs):
        return _OfflineKimiProvider()


class TestPerfConfigIntegration:
    """Test performance recipe integration with flat performance and model recipes."""

    def test_llama3_8b_flat_perf_config_instantiation(self):
        """Test that a Llama3 8B flat perf recipe can be instantiated."""
        from utils.utils import get_perf_recipe_by_name

        cfg = get_perf_recipe_by_name(
            model_recipe_name="llama3_8b",
            task="pretrain",
            num_gpus=8,
            gpu="h100",
            precision="bf16",
            config_variant=None,
        )

        assert cfg.model is not None
        assert cfg.mixed_precision is not None
        assert cfg.train is not None
        assert cfg.dataset is not None

    def test_deepseek_v3_flat_perf_config_instantiation(self):
        """Test that a DeepSeek-V3 flat perf recipe can be instantiated."""
        from utils.utils import get_perf_recipe_by_name

        cfg = get_perf_recipe_by_name(
            model_recipe_name="deepseek_v3",
            task="pretrain",
            num_gpus=1024,
            gpu="h100",
            precision="bf16",
            config_variant=None,
        )

        assert cfg.model is not None
        assert hasattr(cfg.model, "moe_flex_dispatcher_backend")

    def test_qwen3_30b_flat_perf_config_instantiation(self):
        """Test that a Qwen3 MoE flat perf recipe can be instantiated."""
        from utils.utils import get_perf_recipe_by_name

        cfg = get_perf_recipe_by_name(
            model_recipe_name="qwen3_30b_a3b",
            task="pretrain",
            num_gpus=16,
            gpu="h100",
            precision="bf16",
            config_variant=None,
        )

        assert cfg.model is not None
        assert cfg.comm_overlap is not None

    def test_precision_config_variations(self):
        """Test that different flat perf precision recipes load."""
        from utils.utils import get_perf_recipe_by_name

        cfg_bf16 = get_perf_recipe_by_name(
            model_recipe_name="llama3_8b",
            task="pretrain",
            num_gpus=8,
            gpu="h100",
            precision="bf16",
            config_variant=None,
        )
        cfg_fp8 = get_perf_recipe_by_name(
            model_recipe_name="llama3_8b",
            task="pretrain",
            num_gpus=8,
            gpu="h100",
            precision="fp8_cs",
            config_variant=None,
        )

        assert cfg_bf16.mixed_precision is not None
        assert cfg_fp8.mixed_precision is not None

    def test_workload_base_config_uses_default_flat_recipe(self):
        """Test that workload defaults use the default flat recipe."""
        from utils.utils import get_workload_base_config

        cfg = get_workload_base_config(
            model_family_name="llama",
            model_recipe_name="llama3_8b",
            gpu="h100",
            compute_dtype="bf16",
            task="pretrain",
            config_variant=None,
        )

        assert cfg.num_gpus == 8
        assert cfg.gbs_scaling_factor == cfg.global_batch_size / 8

    def test_workload_base_config_default_is_not_nearest_gpu_count(self):
        """Test that default selection keeps explicit flat-recipe defaults, not nearest GPU count."""
        from utils.utils import get_workload_base_config

        cfg = get_workload_base_config(
            model_family_name="llama",
            model_recipe_name="llama31_405b",
            gpu="h100",
            compute_dtype="bf16",
            task="pretrain",
            config_variant=None,
        )

        assert cfg.num_gpus == 1024

    def test_workload_base_config_derives_from_flat_recipe(self):
        """Test that workload defaults use flat perf recipes as the source of truth."""
        from utils.utils import get_perf_recipe_by_name, get_workload_base_config

        cfg = get_workload_base_config(
            model_family_name="deepseek",
            model_recipe_name="deepseek_v3",
            gpu="h100",
            compute_dtype="bf16",
            task="pretrain",
            config_variant=None,
        )
        recipe = get_perf_recipe_by_name(
            model_recipe_name="deepseek_v3",
            task="pretrain",
            num_gpus=1024,
            gpu="h100",
            precision="bf16",
            config_variant=None,
        )

        assert cfg.num_gpus == 1024
        assert cfg.global_batch_size == recipe.train.global_batch_size
        assert cfg.tensor_model_parallel_size == recipe.model.tensor_model_parallel_size
        assert cfg.env_vars == recipe.env_vars
        assert cfg.env_vars is not recipe.env_vars

    @pytest.mark.parametrize(
        ("family", "builder", "expected_env"),
        [
            (
                "deepseek",
                (
                    "megatron.bridge.perf_recipes.deepseek.gb200.deepseek_v3:"
                    "deepseek_v3_pretrain_256gpu_gb200_bf16_config"
                ),
                {
                    "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
                    "NVLINK_DOMAIN_SIZE": 72,
                    "USE_MNNVL": 1,
                    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
                },
            ),
            (
                "gpt_oss",
                ("megatron.bridge.perf_recipes.gpt_oss.gb200.gpt_oss:gpt_oss_120b_pretrain_64gpu_gb200_fp8mx_config"),
                {"NVTE_FWD_LAYERNORM_SM_MARGIN": 20, "NVLINK_DOMAIN_SIZE": 72, "USE_MNNVL": 1},
            ),
            (
                "kimi",
                "megatron.bridge.perf_recipes.kimi.gb300.kimi_k2:kimi_k2_pretrain_256gpu_gb300_fp8mx_config",
                {
                    "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
                    "NVLINK_DOMAIN_SIZE": 72,
                    "NVTE_NORM_FWD_USE_CUDNN": 1,
                },
            ),
            (
                "llama",
                "megatron.bridge.perf_recipes.llama.h100.llama3:llama3_8b_pretrain_8gpu_h100_fp8cs_config",
                {"NVTE_FWD_LAYERNORM_SM_MARGIN": 20, "NCCL_CTA_POLICY": 1},
            ),
            (
                "nemotronh",
                (
                    "megatron.bridge.perf_recipes.nemotronh.gb200.nemotronh:"
                    "nemotron_3_super_pretrain_64gpu_gb200_bf16_config"
                ),
                {"NVTE_FWD_LAYERNORM_SM_MARGIN": 20, "NVLINK_DOMAIN_SIZE": 72, "USE_MNNVL": 1},
            ),
            (
                "qwen",
                ("megatron.bridge.perf_recipes.qwen.gb200.qwen3_moe:qwen3_30b_a3b_pretrain_8gpu_gb200_fp8mx_config"),
                {
                    "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
                    "NVLINK_DOMAIN_SIZE": 72,
                    "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
                },
            ),
            (
                "qwen_vl",
                (
                    "megatron.bridge.perf_recipes.qwen_vl.gb200.qwen3_vl:"
                    "qwen3_vl_30b_a3b_pretrain_8gpu_gb200_bf16_config"
                ),
                {"NVTE_FWD_LAYERNORM_SM_MARGIN": 20, "NVLINK_DOMAIN_SIZE": 72, "USE_MNNVL": 1},
            ),
            (
                "wan",
                "megatron.bridge.perf_recipes.wan.h100.wan:wan_14b_pretrain_32gpu_h100_bf16_config",
                {"NVTE_FWD_LAYERNORM_SM_MARGIN": 20, "CUDA_DEVICE_MAX_CONNECTIONS": 1},
            ),
        ],
    )
    def test_nemo_ci_perf_family_builders_embed_environment_settings(self, family, builder, expected_env, monkeypatch):
        """Direct builders for every active nemo-ci family should carry process settings."""
        if family == "kimi":
            kimi_recipe_module = importlib.import_module("megatron.bridge.recipes.kimi.h100.kimi_k2")
            monkeypatch.setattr(kimi_recipe_module, "AutoBridge", _OfflineKimiAutoBridge)

        module_name, function_name = builder.split(":", 1)
        recipe = getattr(importlib.import_module(module_name), function_name)()

        assert recipe.env_vars.items() >= expected_env.items(), family

    def test_generated_workload_metadata_is_not_required(self):
        """Test that removed perf configs do not leave a generated metadata mirror."""
        assert not (SCRIPTS_PERF_PATH / "utils" / "workload_metadata.py").exists()

    def test_unsupported_config_variant_errors(self):
        """Test that unknown workload variants are not silently collapsed to default."""
        from utils.utils import get_workload_base_config

        with pytest.raises(ValueError, match="unknown"):
            get_workload_base_config(
                model_family_name="llama",
                model_recipe_name="llama3_8b",
                gpu="h100",
                compute_dtype="bf16",
                task="pretrain",
                config_variant="unknown",
            )

    def test_list_available_config_variants_accepts_named_parameters(self):
        """Test that config variant discovery accepts named parameters."""
        from utils.utils import list_available_config_variants

        variants = list_available_config_variants(
            model_family_name="llama",
            model_recipe_name="llama3_8b",
            gpu="h100",
            compute_dtype="bf16",
            task="pretrain",
        )

        assert variants == [None]

    def test_list_available_config_variants_keeps_suffixless_first(self):
        """Test that interactive selection prefers the suffix-less recipe."""
        from utils.utils import list_available_config_variants

        variants = list_available_config_variants(
            model_family_name="deepseek",
            model_recipe_name="deepseek_v3",
            gpu="h100",
            compute_dtype="fp8_sc",
            task="pretrain",
        )

        assert variants == [None, "large_scale"]

    def test_build_recipe_config_llama_sets_paths(self):
        """Test that the recipe config helper sets expected /nemo_run paths."""
        from utils.utils import build_recipe_config

        cfg = build_recipe_config(
            model_family_name="llama",
            model_recipe_name="llama3_8b",
            train_task="pretrain",
            wandb_experiment_name="test_experiment",
        )

        assert cfg.checkpoint.save == "/nemo_run/test_experiment/checkpoints"
        assert cfg.checkpoint.load == "/nemo_run/test_experiment/checkpoints"
        assert cfg.logger.tensorboard_dir == "/nemo_run/test_experiment/tb_logs"
        assert cfg.logger.wandb_exp_name == "test_experiment"
        assert cfg.logger.wandb_save_dir == "/nemo_run/test_experiment/wandb"

    def test_build_recipe_config_deepseek_sets_paths(self):
        """Test that build_recipe_config works with DeepSeek recipes."""
        from utils.utils import build_recipe_config

        cfg = build_recipe_config(
            model_family_name="deepseek",
            model_recipe_name="deepseek_v3",
            train_task="pretrain",
            wandb_experiment_name="deepseek_test",
        )

        assert cfg.logger.wandb_exp_name == "deepseek_test"
        assert cfg.checkpoint.save == "/nemo_run/deepseek_test/checkpoints"

    def test_get_perf_optimized_recipe_uses_requested_gpu_count(self):
        """Test that flat perf recipe selection uses the requested GPU count."""
        from utils.utils import get_perf_optimized_recipe

        cfg = get_perf_optimized_recipe(
            model_family_name="deepseek",
            model_recipe_name="deepseek_v3",
            train_task="pretrain",
            gpu="h100",
            compute_dtype="bf16",
            config_variant=None,
            num_gpus=1024,
        )

        assert cfg.train.global_batch_size == 16384

    def test_kimi_flat_perf_recipes_are_parameterless(self):
        """Test that Kimi flat recipes expose fixed recipe entry points."""
        from megatron.bridge.perf_recipes.kimi.h100.kimi_k2 import kimi_k2_pretrain_1024gpu_h100_bf16_config

        assert not inspect.signature(kimi_k2_pretrain_1024gpu_h100_bf16_config).parameters

    def test_config_overrides_after_precision(self):
        """Test that config properties can be overridden after precision is applied."""
        from utils.utils import get_perf_recipe_by_name

        cfg = get_perf_recipe_by_name(
            model_recipe_name="llama3_8b",
            task="pretrain",
            num_gpus=8,
            gpu="h100",
            precision="bf16",
            config_variant=None,
        )

        cfg.train.train_iters = 100
        cfg.train.global_batch_size = 16

        assert cfg.train.train_iters == 100
        assert cfg.train.global_batch_size == 16
