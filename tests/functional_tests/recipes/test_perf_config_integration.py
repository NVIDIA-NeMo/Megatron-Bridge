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

"""
Functional tests for performance config integration with library recipes.

These tests verify that:
1. Performance configs can correctly instantiate library recipes (which use parameterless API)
2. The apply_precision_config helper works correctly
3. The get_library_recipe function works with the new parameterless recipes
"""

import sys
from pathlib import Path


# Add the performance scripts to the path for testing
SCRIPTS_PERF_PATH = Path(__file__).parents[3] / "scripts" / "performance"
sys.path.insert(0, str(SCRIPTS_PERF_PATH))


class TestPerfConfigIntegration:
    """Test class for performance config integration with library recipes."""

    def test_llama3_8b_perf_config_instantiation(self):
        """Test that Llama3 8B perf configs can be instantiated correctly."""
        from configs.llama.llama3_llm_pretrain import llama3_8b_pretrain_config_h100

        # Should not raise any errors
        cfg = llama3_8b_pretrain_config_h100(precision="bf16", mock=True)

        # Verify the config has expected structure
        assert cfg is not None
        assert cfg.model is not None
        assert cfg.mixed_precision is not None
        assert cfg.train is not None
        assert cfg.dataset is not None

    def test_llama3_70b_perf_config_instantiation(self):
        """Test that Llama3 70B perf configs can be instantiated correctly."""
        from configs.llama.llama3_llm_pretrain import llama3_70b_pretrain_config_h100

        cfg = llama3_70b_pretrain_config_h100(precision="bf16", mock=True)

        assert cfg is not None
        assert cfg.model is not None
        assert cfg.mixed_precision is not None

    def test_direct_precision_override(self):
        """Test that precision can be set directly on ConfigContainer."""
        from megatron.bridge.recipes.llama import llama3_8b_pretrain_config
        from megatron.bridge.training.mixed_precision import bf16_mixed

        # Get a config without precision set
        cfg = llama3_8b_pretrain_config()

        # Apply a specific precision config directly
        precision_config = bf16_mixed()
        cfg.mixed_precision = precision_config

        # Verify the precision was applied
        assert cfg.mixed_precision == precision_config

    def test_deepseek_v3_perf_config_instantiation(self):
        """Test that DeepSeek-V3 perf configs can be instantiated correctly."""
        from configs.deepseek.deepseek_llm_pretrain import deepseek_v3_pretrain_config_h100

        cfg = deepseek_v3_pretrain_config_h100(precision="bf16", mock=True)

        assert cfg is not None
        assert cfg.model is not None
        # DeepSeek configs should have MoE-related settings
        assert hasattr(cfg.model, "moe_flex_dispatcher_backend")

    def test_qwen3_30b_perf_config_instantiation(self):
        """Test that Qwen3 30B A3B perf configs can be instantiated correctly."""
        from configs.qwen.qwen3_llm_pretrain import qwen3_30b_a3b_pretrain_config_h100

        cfg = qwen3_30b_a3b_pretrain_config_h100(precision="bf16", mock=True)

        assert cfg is not None
        assert cfg.model is not None
        assert cfg.comm_overlap is not None

    def test_nemotronh_56b_perf_config_instantiation(self):
        """Test that NemotronH 56B perf configs can be instantiated correctly."""
        from configs.nemotronh.nemotronh_llm_pretrain import nemotronh_56b_pretrain_config_h100

        cfg = nemotronh_56b_pretrain_config_h100(precision="bf16", mock=True)

        assert cfg is not None
        assert cfg.model is not None
        assert cfg.mixed_precision is not None

    def test_gpt_oss_120b_perf_config_instantiation(self):
        """Test that GPT-OSS 120B perf configs can be instantiated correctly."""
        from configs.gpt_oss.gpt_oss_llm_pretrain import gpt_oss_120b_pretrain_config_h100

        cfg = gpt_oss_120b_pretrain_config_h100(precision="bf16", mock=True)

        assert cfg is not None
        assert cfg.model is not None
        assert cfg.mixed_precision is not None

    def test_llama31_405b_perf_config_instantiation(self):
        """Test that Llama 3.1 405B perf configs can be instantiated correctly."""
        from configs.llama.llama31_llm_pretrain import llama31_405b_pretrain_config_h100

        cfg = llama31_405b_pretrain_config_h100(precision="bf16", mock=True)

        assert cfg is not None
        assert cfg.model is not None
        assert cfg.comm_overlap is not None

    def test_get_library_recipe_llama(self):
        """Test that get_library_recipe works with Llama recipes and sets all paths."""
        from utils.utils import get_library_recipe

        cfg = get_library_recipe(
            model_family_name="llama",
            model_recipe_name="llama3_8b",
            train_task="pretrain",
            wandb_experiment_name="test_experiment",
        )

        assert cfg is not None
        # Verify all paths are set correctly based on dir="/nemo_run/" and name="test_experiment"
        assert cfg.checkpoint.save == "/nemo_run/test_experiment/checkpoints"
        assert cfg.checkpoint.load == "/nemo_run/test_experiment/checkpoints"
        assert cfg.logger.tensorboard_dir == "/nemo_run/test_experiment/tb_logs"
        assert cfg.logger.wandb_exp_name == "test_experiment"
        assert cfg.logger.wandb_save_dir == "/nemo_run/test_experiment/wandb"

    def test_get_library_recipe_deepseek(self):
        """Test that get_library_recipe works with DeepSeek recipes."""
        from utils.utils import get_library_recipe

        cfg = get_library_recipe(
            model_family_name="deepseek",
            model_recipe_name="deepseek_v3",
            train_task="pretrain",
            wandb_experiment_name="deepseek_test",
        )

        assert cfg is not None
        assert cfg.logger.wandb_exp_name == "deepseek_test"
        assert cfg.checkpoint.save == "/nemo_run/deepseek_test/checkpoints"

    def test_precision_config_variations(self):
        """Test that different precision configs work correctly."""
        from configs.llama.llama3_llm_pretrain import llama3_8b_pretrain_config_h100

        # Test BF16
        cfg_bf16 = llama3_8b_pretrain_config_h100(precision="bf16", mock=True)
        assert cfg_bf16.mixed_precision is not None

        # Test FP8 CS
        cfg_fp8 = llama3_8b_pretrain_config_h100(precision="fp8_cs", mock=True)
        assert cfg_fp8.mixed_precision is not None

    def test_config_overrides_after_precision(self):
        """Test that config properties can be overridden after precision is applied."""
        from configs.llama.llama3_llm_pretrain import llama3_8b_pretrain_config_h100

        cfg = llama3_8b_pretrain_config_h100(precision="bf16", mock=True)

        # Should be able to override properties after precision config is applied
        cfg.train.train_iters = 100
        cfg.train.global_batch_size = 16

        assert cfg.train.train_iters == 100
        assert cfg.train.global_batch_size == 16
