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

import inspect
import sys
from pathlib import Path

import pytest


SCRIPTS_PERF_PATH = Path(__file__).parents[4] / "scripts" / "performance"
sys.path.insert(0, str(SCRIPTS_PERF_PATH))


class TestPerfConfigIntegration:
    """Test performance recipe integration with flat perf recipes and library recipes."""

    def test_llama3_8b_flat_perf_config_instantiation(self):
        """Test that a Llama3 8B flat perf recipe can be instantiated."""
        from utils.utils import get_perf_recipe_by_name

        cfg = get_perf_recipe_by_name(
            model_recipe_name="llama3_8b",
            task="pretrain",
            num_gpus=8,
            gpu="h100",
            precision="bf16",
            config_variant="v2",
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
            config_variant="v2",
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
            config_variant="v2",
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
            config_variant="v2",
        )
        cfg_fp8 = get_perf_recipe_by_name(
            model_recipe_name="llama3_8b",
            task="pretrain",
            num_gpus=8,
            gpu="h100",
            precision="fp8_cs",
            config_variant="v2",
        )

        assert cfg_bf16.mixed_precision is not None
        assert cfg_fp8.mixed_precision is not None

    def test_workload_base_config_falls_back_for_gpu_scaling(self):
        """Test that non-exact GPU counts still use a flat recipe base for scaling."""
        from utils.utils import get_workload_base_config

        cfg = get_workload_base_config(
            model_family_name="llama",
            model_recipe_name="llama3_8b",
            gpu="h100",
            compute_dtype="bf16",
            task="pretrain",
            config_variant="v2",
            num_gpus=16,
        )

        assert cfg.num_gpus == 8
        assert cfg.gbs_scaling_factor == cfg.global_batch_size / 8

    def test_workload_base_config_fallback_uses_legacy_default_gpu_count(self):
        """Test that fallback selection prefers legacy defaults over the smallest recipe."""
        from utils.utils import get_workload_base_config

        cfg = get_workload_base_config(
            model_family_name="llama",
            model_recipe_name="llama31_405b",
            gpu="h100",
            compute_dtype="bf16",
            task="pretrain",
            config_variant="v2",
            num_gpus=256,
        )

        assert cfg.num_gpus == 1024

    def test_workload_base_config_uses_lightweight_metadata(self, monkeypatch):
        """Test that launcher-side workload lookup does not import full Bridge recipes."""
        import utils.utils as perf_utils

        def _raise_if_recipe_imported(*args, **kwargs):
            raise AssertionError("get_workload_base_config should use lightweight metadata")

        monkeypatch.setattr(perf_utils, "get_perf_recipe_by_name", _raise_if_recipe_imported)

        cfg = perf_utils.get_workload_base_config(
            model_family_name="deepseek",
            model_recipe_name="deepseek_v3",
            gpu="h100",
            compute_dtype="bf16",
            task="pretrain",
            config_variant="v2",
            num_gpus=1024,
        )

        assert cfg.num_gpus == 1024
        assert cfg.global_batch_size == 16384

    def test_workload_base_config_requires_lightweight_metadata(self, monkeypatch):
        """Test that launcher-side workload lookup never falls through to recipe imports."""
        import utils.utils as perf_utils

        recipe_name = "llama3_8b_pretrain_8gpu_h100_bf16_config"

        def _raise_if_recipe_imported(*args, **kwargs):
            raise AssertionError("get_workload_base_config should not import full recipes")

        monkeypatch.delitem(perf_utils.WORKLOAD_BASE_CONFIGS, recipe_name)
        monkeypatch.setattr(perf_utils, "get_perf_recipe_by_name", _raise_if_recipe_imported)

        with pytest.raises(ValueError, match="Missing lightweight metadata"):
            perf_utils.get_workload_base_config(
                model_family_name="llama",
                model_recipe_name="llama3_8b",
                gpu="h100",
                compute_dtype="bf16",
                task="pretrain",
                config_variant="v2",
                num_gpus=8,
            )

    def test_unsupported_legacy_config_variant_errors(self):
        """Test that removed v1 workload variants are not silently collapsed to v2."""
        from utils.utils import get_workload_base_config

        with pytest.raises(ValueError, match="v1"):
            get_workload_base_config(
                model_family_name="llama",
                model_recipe_name="llama3_8b",
                gpu="h100",
                compute_dtype="bf16",
                task="pretrain",
                config_variant="v1",
            )

    def test_list_available_config_variants_accepts_legacy_signature(self):
        """Test that callers can still pass the unused model family argument."""
        from utils.utils import list_available_config_variants

        variants = list_available_config_variants("llama", "llama3_8b", "h100", "bf16", "pretrain")

        assert variants == ["v2"]

    def test_get_library_recipe_llama_sets_paths(self):
        """Test that the legacy library recipe helper sets expected /nemo_run paths."""
        from utils.utils import get_library_recipe

        cfg = get_library_recipe(
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

    def test_get_library_recipe_deepseek_sets_paths(self):
        """Test that get_library_recipe works with DeepSeek recipes."""
        from utils.utils import get_library_recipe

        cfg = get_library_recipe(
            model_family_name="deepseek",
            model_recipe_name="deepseek_v3",
            train_task="pretrain",
            wandb_experiment_name="deepseek_test",
        )

        assert cfg.logger.wandb_exp_name == "deepseek_test"
        assert cfg.checkpoint.save == "/nemo_run/deepseek_test/checkpoints"

    def test_get_perf_optimized_recipe_uses_legacy_default_gpu_count(self):
        """Test that omitted GPU count keeps the old DeepSeek H100 v2 default."""
        from utils.utils import get_perf_optimized_recipe

        cfg = get_perf_optimized_recipe(
            model_family_name="deepseek",
            model_recipe_name="deepseek_v3",
            train_task="pretrain",
            gpu="h100",
            compute_dtype="bf16",
            config_variant="v2",
        )

        assert cfg.train.global_batch_size == 16384

    def test_get_perf_optimized_recipe_kimi_adam_optimizer(self):
        """Test that the Kimi Adam override path applies without import errors."""
        from utils.utils import get_perf_optimized_recipe

        cfg = get_perf_optimized_recipe(
            model_family_name="kimi",
            model_recipe_name="kimi_k2",
            train_task="pretrain",
            gpu="h100",
            compute_dtype="bf16",
            config_variant="v2",
            optimizer_type="adam",
        )

        assert cfg.ddp.use_distributed_optimizer is True
        assert cfg.ddp.overlap_param_gather is True

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
            config_variant="v2",
        )

        cfg.train.train_iters = 100
        cfg.train.global_batch_size = 16

        assert cfg.train.train_iters == 100
        assert cfg.train.global_batch_size == 16
