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

"""Unit tests for Qwen3-Omni training entry wiring in recipe_runner.py."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _load_recipe_runner_module():
    """Load recipe_runner.py with lightweight stub modules for local unit testing."""

    script_path = Path(__file__).resolve().parents[3] / "scripts" / "training" / "recipe_runner.py"
    module_name = "test_recipe_runner_qwen3_omni_module"

    megatron_module = _package("megatron")
    bridge_module = _package("megatron.bridge")
    models_module = _package("megatron.bridge.models")
    qwen_omni_models_module = _package("megatron.bridge.models.qwen_omni")
    qwen_vl_models_module = _package("megatron.bridge.models.qwen_vl")
    stepfun_models_module = _package("megatron.bridge.models.stepfun")
    diffusion_module = _package("megatron.bridge.diffusion")
    diffusion_models_module = _package("megatron.bridge.diffusion.models")
    flux_models_module = _package("megatron.bridge.diffusion.models.flux")
    wan_models_module = _package("megatron.bridge.diffusion.models.wan")
    recipes_module = _package("megatron.bridge.recipes")
    recipes_utils_module = _package("megatron.bridge.recipes.utils")
    training_module = _package("megatron.bridge.training")
    training_utils_module = _package("megatron.bridge.training.utils")
    utils_module = _package("megatron.bridge.utils")

    qwen3_omni_step = types.ModuleType("megatron.bridge.models.qwen_omni.qwen3_omni_step")
    qwen3_omni_step.forward_step = Mock(name="qwen3_omni_forward_step")

    qwen3_vl_step = types.ModuleType("megatron.bridge.models.qwen_vl.qwen3_vl_step")
    qwen3_vl_step.forward_step = object()

    step37_flickr8k_step = types.ModuleType("megatron.bridge.models.stepfun.step37_flickr8k_step")
    step37_flickr8k_step.forward_step = object()

    gpt_step = types.ModuleType("megatron.bridge.training.gpt_step")
    gpt_step.forward_step = object()

    vlm_step = types.ModuleType("megatron.bridge.training.vlm_step")
    vlm_step.forward_step = object()

    llava_step = types.ModuleType("megatron.bridge.training.llava_step")
    llava_step.forward_step = object()

    nemotron_omni_step = types.ModuleType("megatron.bridge.training.nemotron_omni_step")
    nemotron_omni_step.forward_step = object()

    audio_lm_step = types.ModuleType("megatron.bridge.training.audio_lm_step")
    audio_lm_step.forward_step = object()

    flux_step = types.ModuleType("megatron.bridge.diffusion.models.flux.flux_step")

    class FluxForwardStep:
        pass

    flux_step.FluxForwardStep = FluxForwardStep

    wan_step = types.ModuleType("megatron.bridge.diffusion.models.wan.wan_step")

    class WanForwardStep:
        def __init__(self, mode=None):
            self.mode = mode

    wan_step.WanForwardStep = WanForwardStep

    determinism_utils_module = types.ModuleType("megatron.bridge.recipes.utils.determinism_utils")
    determinism_utils_module.apply_determinism_overrides = Mock(name="apply_determinism_overrides")

    finetune_module = types.ModuleType("megatron.bridge.training.finetune")
    finetune_module.finetune = Mock(name="finetune")

    pretrain_module = types.ModuleType("megatron.bridge.training.pretrain")
    pretrain_module.pretrain = Mock(name="pretrain")

    class TokenizerConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    config_module = types.ModuleType("megatron.bridge.training.config")
    config_module.ConfigContainer = object
    config_module.TokenizerConfig = TokenizerConfig
    config_module.apply_environment_variables = Mock(name="apply_environment_variables")

    omegaconf_module = types.ModuleType("megatron.bridge.training.utils.omegaconf_utils")
    omegaconf_module.process_config_with_overrides = lambda config, cli_overrides=None: config

    common_utils_module = types.ModuleType("megatron.bridge.utils.common_utils")
    common_utils_module.get_rank_safe = lambda: 1

    stub_modules = {
        "megatron": megatron_module,
        "megatron.bridge": bridge_module,
        "megatron.bridge.models": models_module,
        "megatron.bridge.models.qwen_omni": qwen_omni_models_module,
        "megatron.bridge.models.qwen_vl": qwen_vl_models_module,
        "megatron.bridge.models.stepfun": stepfun_models_module,
        "megatron.bridge.diffusion": diffusion_module,
        "megatron.bridge.diffusion.models": diffusion_models_module,
        "megatron.bridge.diffusion.models.flux": flux_models_module,
        "megatron.bridge.diffusion.models.wan": wan_models_module,
        "megatron.bridge.recipes": recipes_module,
        "megatron.bridge.recipes.utils": recipes_utils_module,
        "megatron.bridge.recipes.utils.determinism_utils": determinism_utils_module,
        "megatron.bridge.training": training_module,
        "megatron.bridge.training.utils": training_utils_module,
        "megatron.bridge.utils": utils_module,
        "megatron.bridge.utils.common_utils": common_utils_module,
        "megatron.bridge.diffusion.models.flux.flux_step": flux_step,
        "megatron.bridge.diffusion.models.wan.wan_step": wan_step,
        "megatron.bridge.models.qwen_omni.qwen3_omni_step": qwen3_omni_step,
        "megatron.bridge.models.qwen_vl.qwen3_vl_step": qwen3_vl_step,
        "megatron.bridge.models.stepfun.step37_flickr8k_step": step37_flickr8k_step,
        "megatron.bridge.training.audio_lm_step": audio_lm_step,
        "megatron.bridge.training.gpt_step": gpt_step,
        "megatron.bridge.training.vlm_step": vlm_step,
        "megatron.bridge.training.llava_step": llava_step,
        "megatron.bridge.training.nemotron_omni_step": nemotron_omni_step,
        "megatron.bridge.training.finetune": finetune_module,
        "megatron.bridge.training.pretrain": pretrain_module,
        "megatron.bridge.training.config": config_module,
        "megatron.bridge.training.utils.omegaconf_utils": omegaconf_module,
    }

    previous_modules = {name: sys.modules.get(name) for name in stub_modules}
    sys.modules.update(stub_modules)
    script_dir = str(script_path.parent)
    inserted_script_dir = script_dir not in sys.path
    if inserted_script_dir:
        sys.path.insert(0, script_dir)

    try:
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module._destroy_process_group = Mock(name="destroy_process_group")
        module._get_rank_safe = lambda: 1
    finally:
        if inserted_script_dir:
            sys.path.remove(script_dir)
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    test_handles = {
        "apply_environment_variables": config_module.apply_environment_variables,
        "finetune": finetune_module.finetune,
        "omni_forward_step": qwen3_omni_step.forward_step,
        "pretrain": pretrain_module.pretrain,
        "wan_forward_step": WanForwardStep,
    }
    return module, test_handles


class TestRecipeRunnerQwen3Omni:
    """Tests for wiring Qwen3-Omni into the shared recipe runner."""

    def test_apply_runtime_environment_applies_recipe_defaults(self):
        """The shared runner should export recipe-owned environment defaults."""
        module, handles = _load_recipe_runner_module()
        config = SimpleNamespace(ddp=SimpleNamespace(nccl_ub=False))

        assert module.apply_runtime_environment(config) is config

        handles["apply_environment_variables"].assert_called_once_with(config)

    def test_llm_step_alias_loads_gpt_forward_step(self):
        """The public LLM step should resolve lazily to the GPT forward step."""
        module, _ = _load_recipe_runner_module()
        assert module.STEP_FUNCTIONS["llm_step"] == module.STEP_FUNCTIONS["gpt_step"]

        forward_step = Mock()
        module.STEP_FUNCTIONS["llm_step"] = forward_step
        module.STEP_FUNCTIONS["gpt_step"] = forward_step

        assert module.load_forward_step("llm_step") is module.load_forward_step("gpt_step")

    def test_load_forward_step_returns_qwen3_omni_handler(self):
        """The shared registry should expose qwen3_omni_step."""

        module, handles = _load_recipe_runner_module()

        module.STEP_FUNCTIONS["qwen3_omni_step"] = handles["omni_forward_step"]
        assert module.load_forward_step("qwen3_omni_step") is handles["omni_forward_step"]

    def test_class_based_forward_step_receives_mode(self):
        """Class-based diffusion steps should still receive the selected train mode."""

        module, handles = _load_recipe_runner_module()
        module.STEP_FUNCTIONS["wan_step"] = handles["wan_forward_step"]

        forward_step = module.load_forward_step("wan_step", mode="finetune")

        assert forward_step.mode == "finetune"

    def test_run_config_routes_qwen3_omni_step_to_finetune(self):
        """The shared runner should pass the Omni step function into finetune."""

        module, handles = _load_recipe_runner_module()
        config = object()
        module.TRAIN_FUNCTIONS["finetune"] = handles["finetune"]

        module.run_config(config=config, mode="finetune", step_func=handles["omni_forward_step"])

        handles["finetune"].assert_called_once_with(config=config, forward_step_func=handles["omni_forward_step"])
        handles["pretrain"].assert_not_called()

    def test_run_config_dumps_environment_before_training(self):
        """Environment diagnostics should run before training."""
        module, handles = _load_recipe_runner_module()
        events = []
        module.dump_env_rank0 = Mock(side_effect=lambda: events.append("dump"))
        handles["finetune"].side_effect = lambda **kwargs: events.append("training")
        module.TRAIN_FUNCTIONS["finetune"] = handles["finetune"]

        module.run_config(config=object(), mode="finetune", step_func=object(), dump_environment=True)

        assert events == ["dump", "training"]

    def test_sync_model_dataset_sequence_length_accepts_sequence_length_alias(self):
        """Hydra overrides using dataset.seq_length should still align the model."""
        module, _ = _load_recipe_runner_module()
        config = SimpleNamespace(
            dataset=SimpleNamespace(sequence_length=256),
            model=SimpleNamespace(seq_length=1024),
        )

        module.sync_model_dataset_sequence_length(config)

        assert config.model.seq_length == 256
