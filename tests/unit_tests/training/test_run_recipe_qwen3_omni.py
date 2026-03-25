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

"""Unit tests for Qwen3-Omni training entry wiring in run_recipe.py."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock


def _load_run_recipe_module():
    """Load run_recipe.py with lightweight stub modules for local unit testing."""

    script_path = (
        Path(__file__).resolve().parents[3] / "scripts" / "training" / "run_recipe.py"
    )
    module_name = "test_run_recipe_qwen3_omni_module"

    recipe_config = object()
    recipes_module = types.ModuleType("megatron.bridge.recipes")
    recipes_module.qwen3_omni_30b_a3b_sft_config = lambda **_: recipe_config

    qwen3_omni_step = types.ModuleType("megatron.bridge.models.qwen_omni.qwen3_omni_step")
    qwen3_omni_step.forward_step = object()

    qwen3_vl_step = types.ModuleType("megatron.bridge.models.qwen_vl.qwen3_vl_step")
    qwen3_vl_step.forward_step = object()

    gpt_step = types.ModuleType("megatron.bridge.training.gpt_step")
    gpt_step.forward_step = object()

    vlm_step = types.ModuleType("megatron.bridge.training.vlm_step")
    vlm_step.forward_step = object()

    llava_step = types.ModuleType("megatron.bridge.training.llava_step")
    llava_step.forward_step = object()

    finetune_module = types.ModuleType("megatron.bridge.training.finetune")
    finetune_module.finetune = Mock(name="finetune")

    pretrain_module = types.ModuleType("megatron.bridge.training.pretrain")
    pretrain_module.pretrain = Mock(name="pretrain")

    config_module = types.ModuleType("megatron.bridge.training.config")
    config_module.ConfigContainer = object

    omegaconf_module = types.ModuleType("megatron.bridge.training.utils.omegaconf_utils")
    omegaconf_module.process_config_with_overrides = lambda config, cli_overrides=None: config

    stub_modules = {
        "megatron.bridge.recipes": recipes_module,
        "megatron.bridge.models.qwen_omni.qwen3_omni_step": qwen3_omni_step,
        "megatron.bridge.models.qwen_vl.qwen3_vl_step": qwen3_vl_step,
        "megatron.bridge.training.gpt_step": gpt_step,
        "megatron.bridge.training.vlm_step": vlm_step,
        "megatron.bridge.training.llava_step": llava_step,
        "megatron.bridge.training.finetune": finetune_module,
        "megatron.bridge.training.pretrain": pretrain_module,
        "megatron.bridge.training.config": config_module,
        "megatron.bridge.training.utils.omegaconf_utils": omegaconf_module,
    }

    previous_modules = {name: sys.modules.get(name) for name in stub_modules}
    sys.modules.update(stub_modules)

    try:
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    test_handles = {
        "omni_forward_step": qwen3_omni_step.forward_step,
        "finetune": finetune_module.finetune,
        "pretrain": pretrain_module.pretrain,
        "recipe_config": recipe_config,
    }
    return module, test_handles


class TestRunRecipeQwen3Omni:
    """Tests for wiring Qwen3-Omni into the generic training entrypoint."""

    def test_load_forward_step_returns_qwen3_omni_handler(self):
        """The run_recipe registry should expose qwen3_omni_step."""

        module, handles = _load_run_recipe_module()

        assert module.load_forward_step("qwen3_omni_step") is handles["omni_forward_step"]

    def test_qwen3_omni_step_is_exposed_in_cli_choices(self):
        """The CLI parser should advertise qwen3_omni_step as a valid step function."""

        module, _ = _load_run_recipe_module()

        original_argv = sys.argv
        sys.argv = ["run_recipe.py", "--recipe", "qwen3_omni_30b_a3b_sft_config"]
        try:
            args, _ = module.parse_args()
        finally:
            sys.argv = original_argv

        assert "qwen3_omni_step" in module.STEP_FUNCTIONS
        assert args.step_func == "gpt_step"

    def test_main_routes_qwen3_omni_step_to_finetune(self):
        """The generic training entry should pass the Omni step function into finetune."""

        module, handles = _load_run_recipe_module()

        original_argv = sys.argv
        sys.argv = [
            "run_recipe.py",
            "--mode",
            "finetune",
            "--recipe",
            "qwen3_omni_30b_a3b_sft_config",
            "--step_func",
            "qwen3_omni_step",
        ]
        try:
            module.main()
        finally:
            sys.argv = original_argv

        handles["finetune"].assert_called_once_with(
            config=handles["recipe_config"],
            forward_step_func=handles["omni_forward_step"],
        )
        handles["pretrain"].assert_not_called()
