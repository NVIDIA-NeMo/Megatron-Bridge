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

import pytest


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

    naming_module = types.ModuleType("megatron.bridge.recipes.utils.naming")
    naming_module.recipe_variant_suffix = lambda config_variant: (
        "" if config_variant is None or config_variant in {"v1", "v2", "v3"} else f"_{config_variant}"
    )
    naming_module.recipe_function_name = (
        lambda *, model_recipe_name, task, num_gpus, gpu, precision, config_variant=None: (
            f"{model_recipe_name}_{task}_{num_gpus}gpu_{gpu}_{precision}"
            f"{naming_module.recipe_variant_suffix(config_variant)}_config"
        )
    )

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
    config_module.runtime_config_update = Mock(name="runtime_config_update")

    omegaconf_module = types.ModuleType("megatron.bridge.training.utils.omegaconf_utils")
    omegaconf_module.process_config_with_overrides = lambda config, cli_overrides=None: config

    common_utils_module = types.ModuleType("megatron.bridge.utils.common_utils")
    common_utils_module.get_rank_safe = lambda: 1

    torch_module = types.ModuleType("torch")
    torch_module.distributed = types.SimpleNamespace(
        barrier=Mock(name="barrier"),
        destroy_process_group=Mock(name="destroy_process_group"),
        is_initialized=lambda: False,
    )

    stub_modules = {
        "torch": torch_module,
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
        "megatron.bridge.recipes.utils.naming": naming_module,
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
        "apply_environment_variables": config_module.apply_environment_variables,
        "finetune": finetune_module.finetune,
        "omni_forward_step": qwen3_omni_step.forward_step,
        "pretrain": pretrain_module.pretrain,
        "wan_forward_step": WanForwardStep,
    }
    return module, test_handles


def _load_run_recipe_module():
    """Load run_recipe.py with its external modules stubbed."""
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "training" / "run_recipe.py"
    module_name = "test_unified_run_recipe_module"

    recipe_runner = types.ModuleType("recipe_runner")
    recipe_runner.PRECISION_ALIASES = {
        "bf16": "bf16",
        "fp8_cs": "fp8cs",
        "fp8cs": "fp8cs",
        "fp8_mx": "fp8mx",
        "fp8mx": "fp8mx",
        "fp8_sc": "fp8sc",
        "fp8sc": "fp8sc",
        "nvfp4": "nvfp4",
    }
    recipe_runner.RecipeSource = str
    for name in (
        "apply_cli_overrides",
        "apply_determinism",
        "apply_launcher_overrides",
        "infer_train_mode",
        "load_forward_step",
        "load_library_recipe_by_family",
        "load_perf_recipe_by_name",
        "load_recipe",
        "resolve_recipe_source",
        "run_config",
        "sync_model_dataset_sequence_length",
    ):
        setattr(recipe_runner, name, Mock(name=name))

    recipe_runner.apply_cli_overrides.side_effect = lambda cfg, _: cfg
    recipe_runner.apply_determinism.side_effect = lambda cfg, **_: cfg
    recipe_runner.apply_launcher_overrides.side_effect = lambda cfg, _, **__: cfg
    recipe_runner.sync_model_dataset_sequence_length.side_effect = lambda cfg: cfg
    recipe_runner.infer_train_mode.return_value = "pretrain"
    recipe_runner.load_forward_step.return_value = object()
    recipe_runner.resolve_recipe_source.side_effect = (
        lambda _, source="auto": "recipes" if source == "auto" else source
    )

    megatron_module = _package("megatron")
    bridge_module = _package("megatron.bridge")
    recipes_module = _package("megatron.bridge.recipes")
    recipes_utils_module = _package("megatron.bridge.recipes.utils")
    dataset_utils_module = types.ModuleType("megatron.bridge.recipes.utils.dataset_utils")
    dataset_utils_module.DATASET_TYPES = [
        "llm-pretrain",
        "llm-pretrain-mock",
        "llm-finetune",
        "llm-finetune-preloaded",
        "vlm-energon",
        "vlm-hf",
        "vlm-preloaded",
    ]
    dataset_utils_module.apply_dataset_override = Mock(name="apply_dataset_override")
    dataset_utils_module.apply_dataset_override.side_effect = lambda cfg, **_: cfg
    dataset_utils_module.infer_mode_from_dataset = Mock(return_value="finetune")

    stub_modules = {
        "recipe_runner": recipe_runner,
        "megatron": megatron_module,
        "megatron.bridge": bridge_module,
        "megatron.bridge.recipes": recipes_module,
        "megatron.bridge.recipes.utils": recipes_utils_module,
        "megatron.bridge.recipes.utils.dataset_utils": dataset_utils_module,
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

    handles = {
        "apply_dataset_override": dataset_utils_module.apply_dataset_override,
        "infer_mode_from_dataset": dataset_utils_module.infer_mode_from_dataset,
        "recipe_runner": recipe_runner,
    }
    return module, handles


class TestRecipeRunnerQwen3Omni:
    """Tests for wiring Qwen3-Omni into the shared recipe runner."""

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

        module.run_config(config=config, mode="finetune", step_func=handles["omni_forward_step"])

        handles["finetune"].assert_called_once_with(config=config, forward_step_func=handles["omni_forward_step"])
        handles["pretrain"].assert_not_called()

    def test_run_config_applies_environment_before_dump_and_training(self):
        """Recipe environment defaults should be visible to diagnostics and training."""
        module, handles = _load_recipe_runner_module()
        events = []
        handles["apply_environment_variables"].side_effect = lambda config: events.append("environment")
        module.dump_env_rank0 = Mock(side_effect=lambda: events.append("dump"))
        handles["finetune"].side_effect = lambda **kwargs: events.append("training")

        module.run_config(config=object(), mode="finetune", step_func=object(), dump_environment=True)

        assert events == ["environment", "dump", "training"]

    def test_dry_run_uses_target_gpu_count_over_slurm_allocation(self, monkeypatch, tmp_path):
        """Dry-run topology validation should use the requested target GPU count."""
        module, _ = _load_recipe_runner_module()
        config = SimpleNamespace(to_yaml=Mock(), print_yaml=Mock())
        monkeypatch.setenv("WORLD_SIZE", "1")
        monkeypatch.setenv("RANK", "7")
        monkeypatch.setenv("SLURM_NTASKS", "1")
        monkeypatch.setenv("SLURM_PROCID", "7")

        module.run_config(
            config=config,
            mode="pretrain",
            step_func=object(),
            dryrun=True,
            save_config_filepath=str(tmp_path / "config.yaml"),
            dryrun_num_gpus=8,
        )

        assert module.os.environ["WORLD_SIZE"] == "8"
        assert module.os.environ["RANK"] == "0"
        module.runtime_config_update.assert_called_once_with(config)

    def test_load_recipe_auto_prefers_library_recipe(self):
        """Auto source resolution should check library recipes before flat perf recipes."""

        module, _ = _load_recipe_runner_module()
        library_config = object()
        perf_config = object()
        module.find_library_recipe = Mock(return_value=lambda: library_config)
        module.find_perf_recipe = Mock(return_value=lambda: perf_config)

        cfg = module.load_recipe("shared_name_config", source="auto")

        assert cfg is library_config
        module.find_perf_recipe.assert_not_called()

    def test_library_lora_selector_uses_peft_recipe_name(self):
        """Library LoRA selectors should resolve the existing PEFT recipe namespace."""
        module, _ = _load_recipe_runner_module()
        config = object()
        module.importlib.import_module = Mock(return_value=SimpleNamespace())
        module.recipe_function_name = Mock(return_value="gpt_oss_20b_peft_1gpu_h100_fp8cs_config")
        module.find_library_recipe = Mock(return_value=lambda: config)

        result = module.load_library_recipe_by_family(
            model_family_name="gpt_oss",
            model_recipe_name="gpt_oss_20b",
            train_task="lora",
            num_gpus=1,
            gpu="h100",
            precision="fp8_cs",
            config_variant="v2",
            wandb_experiment_name=None,
        )

        assert result is config
        module.recipe_function_name.assert_called_once_with(
            model_recipe_name="gpt_oss_20b",
            task="peft",
            num_gpus=1,
            gpu="h100",
            precision="fp8_cs",
            config_variant="v2",
        )

    def test_seq_length_shortcut_updates_model_and_dataset(self):
        """The easy seq-length flag should keep model and dataset lengths aligned."""
        module, _ = _load_recipe_runner_module()
        config = SimpleNamespace(
            train=SimpleNamespace(),
            validation=SimpleNamespace(),
            dist=SimpleNamespace(),
            optimizer=SimpleNamespace(),
            scheduler=SimpleNamespace(),
            checkpoint=SimpleNamespace(),
            dataset=SimpleNamespace(seq_length=1024, sequence_length=1024),
            model=SimpleNamespace(seq_length=1024),
            logger=SimpleNamespace(),
            ddp=SimpleNamespace(nccl_ub=False),
        )
        args = SimpleNamespace(seq_length=512, tokenizer_type=None, tokenizer_model=None, vocab_size=32000)

        module.apply_launcher_overrides(config, args, recipe_source="recipes")

        assert config.dataset.seq_length == 512
        assert config.dataset.sequence_length == 512
        assert config.model.seq_length == 512

    def test_sync_model_dataset_sequence_length_accepts_sequence_length_alias(self):
        """Hydra overrides using dataset.sequence_length should still align the model."""
        module, _ = _load_recipe_runner_module()
        config = SimpleNamespace(
            dataset=SimpleNamespace(sequence_length=256),
            model=SimpleNamespace(seq_length=1024),
        )

        module.sync_model_dataset_sequence_length(config)

        assert config.model.seq_length == 256


class TestUnifiedRunRecipeRouting:
    """Tests for the independent training recipe launcher."""

    def test_parse_args_collects_set_and_trailing_overrides(self):
        module, _ = _load_run_recipe_module()

        args, overrides = module.parse_args(
            [
                "--recipe",
                "vanilla_gpt_pretrain_config",
                "--set",
                "optimizer.lr=0.0003",
                "train.train_iters=5",
            ]
        )

        assert args.recipe == "vanilla_gpt_pretrain_config"
        assert overrides == ["optimizer.lr=0.0003", "train.train_iters=5"]

    def test_ambiguous_full_recipe_requires_explicit_mode_or_task(self):
        module, handles = _load_run_recipe_module()
        handles["recipe_runner"].infer_train_mode.side_effect = ValueError("ambiguous")
        args = SimpleNamespace(mode=None, recipe="custom_config", task=None)

        with pytest.raises(ValueError, match="ambiguous"):
            module._infer_mode(args, dataset=None)

    def test_full_recipe_auto_prefers_library_recipe(self):
        module, handles = _load_run_recipe_module()
        recipe = object()
        events = []
        handles["recipe_runner"].load_recipe.return_value = recipe
        handles["recipe_runner"].apply_cli_overrides.side_effect = lambda cfg, _: events.append("cli") or cfg
        handles["recipe_runner"].sync_model_dataset_sequence_length.side_effect = (
            lambda cfg: events.append("sync") or cfg
        )

        module.main(["--recipe", "vanilla_gpt_pretrain_config", "--max-steps", "2"])

        handles["recipe_runner"].load_recipe.assert_called_once_with(
            "vanilla_gpt_pretrain_config",
            None,
            False,
            None,
            None,
            source="recipes",
        )
        handles["recipe_runner"].apply_launcher_overrides.assert_called_once()
        assert handles["recipe_runner"].apply_launcher_overrides.call_args.kwargs["recipe_source"] == "recipes"
        handles["recipe_runner"].run_config.assert_called_once()
        assert events == ["cli", "sync"]

    def test_perf_selector_loads_flat_perf_recipe(self):
        module, handles = _load_run_recipe_module()
        recipe = object()
        handles["recipe_runner"].load_perf_recipe_by_name.return_value = recipe

        module.main(
            [
                "--source",
                "perf_recipes",
                "--model",
                "qwen3_moe",
                "--task",
                "pretrain",
                "--gpus",
                "8",
                "--gpu",
                "h100",
                "--dtype",
                "fp8_cs",
            ]
        )

        handles["recipe_runner"].load_perf_recipe_by_name.assert_called_once_with(
            model_recipe_name="qwen3_moe",
            task="pretrain",
            num_gpus=8,
            gpu="h100",
            precision="fp8_cs",
            config_variant=None,
        )
        assert handles["recipe_runner"].apply_launcher_overrides.call_args.kwargs["recipe_source"] == "perf_recipes"
        assert handles["recipe_runner"].run_config.call_args.kwargs["barrier_before_destroy"] is True

    def test_dataset_alias_and_preset_apply_before_config_overrides(self):
        module, handles = _load_run_recipe_module()
        recipe = object()
        handles["recipe_runner"].load_recipe.return_value = recipe

        module.main(
            [
                "--recipe",
                "llama3_8b_sft_config",
                "--data",
                "squad_packed",
                "--dataset-preset",
                "squad",
                "--seq-length",
                "128",
                "optimizer.lr=0.0001",
            ]
        )

        handles["apply_dataset_override"].assert_called_once_with(
            recipe,
            dataset_type="llm-finetune",
            packed_sequence=True,
            seq_length=128,
            cli_overrides=["optimizer.lr=0.0001", "dataset.hf_dataset.dataset_name=squad"],
        )
        handles["infer_mode_from_dataset"].assert_called_once_with("llm-finetune")

    def test_full_perf_recipe_dry_run_uses_requested_gpu_count(self):
        module, handles = _load_run_recipe_module()
        recipe = object()
        handles["recipe_runner"].load_recipe.return_value = recipe

        module.main(
            [
                "--recipe",
                "llama3_8b_pretrain_8gpu_h100_bf16_config",
                "--source",
                "perf_recipes",
                "--gpus",
                "8",
                "--dry-run",
            ]
        )

        handles["recipe_runner"].load_recipe.assert_called_once_with(
            "llama3_8b_pretrain_8gpu_h100_bf16_config",
            None,
            False,
            None,
            None,
            source="perf_recipes",
        )
        assert handles["recipe_runner"].run_config.call_args.kwargs["dryrun"] is True
        assert handles["recipe_runner"].run_config.call_args.kwargs["dryrun_num_gpus"] == 8
