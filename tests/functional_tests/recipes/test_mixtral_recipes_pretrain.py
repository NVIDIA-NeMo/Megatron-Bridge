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

"""Functional smoke tests for Mixtral recipe configurations."""

import pytest

from megatron.bridge.recipes.mixtral import (
    mixtral_8x7b_pretrain_config,
    mixtral_8x22b_pretrain_config,
)
from tests.functional_tests.recipes.utils import run_pretrain_recipe_test


MIXTRAL_PRETRAIN_RECIPES = [
    # (config_func, name, recipe_kwargs, parallelism_overrides, model_overrides)
    (
        mixtral_8x7b_pretrain_config,
        "mixtral_8x7b",
        {"use_null_tokenizer": True},  # Avoid downloading tokenizer
        {"expert_model_parallel_size": 2},  # Reduce EP for test
        {"num_layers": 2},  # Use minimal layers for testing
    ),
    # Note: Mixtral 8x22B test skipped due to memory constraints
    # The 8x22B model has 8 experts with hidden_size=6144 and ffn_hidden_size=16384
    # Even with 2 layers and EP=2, this exceeds available GPU memory (2x A100 80GB)
    # Unit tests in tests/unit_tests/models/mixtral/ validate both 8x7B and 8x22B configs
]


class TestMixtralRecipes:
    """Test class for Mixtral recipe functional tests."""

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("config_func,recipe_name,recipe_kwargs,parallelism_overrides,model_overrides", MIXTRAL_PRETRAIN_RECIPES)
    def test_mixtral_pretrain_recipes(self, config_func, recipe_name, recipe_kwargs, parallelism_overrides, model_overrides, tmp_path):
        """Functional test for Mixtral recipes with appropriate parallelism configurations."""
        # Merge recipe_kwargs with standard test config
        from tests.functional_tests.utils import broadcast_path, initialize_distributed
        
        initialize_distributed()
        shared_base_dir = broadcast_path(tmp_path)
        
        # Create config with recipe_kwargs
        config = config_func(
            dir=str(shared_base_dir), 
            name=f"{recipe_name}_functional_test",
            mock=True,
            **recipe_kwargs
        )
        
        # Apply test overrides
        config.train.train_iters = 10
        config.train.eval_interval = 5
        config.train.eval_iters = 2
        config.train.micro_batch_size = 1
        config.train.global_batch_size = 8
        config.scheduler.lr_warmup_iters = 2
        config.model.seq_length = 512
        config.dataset.sequence_length = 512
        
        # Apply parallelism overrides
        for key, value in parallelism_overrides.items():
            if hasattr(config.model, key):
                setattr(config.model, key, value)
        
        # Apply model overrides
        for key, value in model_overrides.items():
            setattr(config.model, key, value)
        
        # Run training
        from megatron.bridge.training.gpt_step import forward_step
        from megatron.bridge.training.pretrain import pretrain
        from tests.functional_tests.utils import clear_directories, verify_checkpoint_files
        
        try:
            pretrain(config, forward_step)
            verify_checkpoint_files(config.checkpoint.save, 10)
        finally:
            clear_directories(tmp_path)

