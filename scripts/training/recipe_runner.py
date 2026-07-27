#!/usr/bin/env python3
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

"""Shared helpers for recipe-based training entry points."""

from __future__ import annotations

import functools
import importlib
import inspect
import logging
import math
import os
import pkgutil
import re
import sys
from collections.abc import Callable
from types import ModuleType
from typing import cast

from recipe_metadata import (
    BENCHMARK_RECIPE_PRECEDENCE_COLLISIONS,
    LIBRARY_RECIPE_PRECEDENCE_COLLISIONS,
    benchmark_recipe_family,
    resolved_benchmark_recipe_metadata,
)

from megatron.bridge.training.config import ConfigContainer, apply_environment_variables


logger = logging.getLogger(__name__)

SENSITIVE_ENV_VAR_PATTERN = re.compile(
    r"(^|_)(TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|ACCESS_KEY|SECRET_KEY|PRIVATE_KEY|AUTHORIZATION)(_|$)",
    re.IGNORECASE,
)

StepFunctionEntry = Callable | tuple[str, str]
TrainFunctionEntry = Callable | tuple[str, str]

STEP_FUNCTIONS: dict[str, StepFunctionEntry] = {
    "audio_lm_step": ("megatron.bridge.training.audio_lm_step", "forward_step"),
    "gpt_step": ("megatron.bridge.training.gpt_step", "forward_step"),
    "llm_step": ("megatron.bridge.training.gpt_step", "forward_step"),
    "vlm_step": ("megatron.bridge.training.vlm_step", "forward_step"),
    "qwen3_omni_step": ("megatron.bridge.models.qwen_omni.qwen3_omni_step", "forward_step"),
    "qwen3_vl_step": ("megatron.bridge.models.qwen_vl.qwen3_vl_step", "forward_step"),
    "step37_flickr8k_step": ("megatron.bridge.models.stepfun.step37_flickr8k_step", "forward_step"),
    "llava_step": ("megatron.bridge.training.llava_step", "forward_step"),
    "nemotron_omni_step": ("megatron.bridge.training.nemotron_omni_step", "forward_step"),
    "flux_step": ("megatron.bridge.diffusion.models.flux.flux_step", "FluxForwardStep"),
    "wan_step": ("megatron.bridge.diffusion.models.wan.wan_step", "WanForwardStep"),
}

STEP_MODALITIES = {
    "audio_lm_step": "audio",
    "gpt_step": "text",
    "llm_step": "text",
    "vlm_step": "vlm",
    "qwen3_omni_step": "omni",
    "qwen3_vl_step": "vlm",
    "step37_flickr8k_step": "vlm",
    "llava_step": "vlm",
    "nemotron_omni_step": "vlm",
    "flux_step": "diffusion",
    "wan_step": "diffusion",
}

TRAIN_FUNCTIONS: dict[str, TrainFunctionEntry] = {
    "pretrain": ("megatron.bridge.training.pretrain", "pretrain"),
    "finetune": ("megatron.bridge.training.finetune", "finetune"),
}

ERR_UNKNOWN_STEP = "Unknown step type: {step_type}. Choose from: {choices}"
RECIPE_ENV_BOOTSTRAP_MARKER = "_MB_TRAINING_RECIPE_ENV_BOOTSTRAPPED"


def dump_env_rank0() -> None:
    """Write a redacted compute-node environment dump on Slurm rank zero."""
    if os.environ.get("SLURM_JOB_ID") is None or int(os.environ.get("SLURM_PROCID", "-1")) != 0:
        return

    env_path = f"/nemo_run/env_{os.environ['SLURM_JOB_ID']}.log"
    try:
        fd = os.open(env_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as env_file:
            for key, value in sorted(os.environ.items()):
                if SENSITIVE_ENV_VAR_PATTERN.search(key):
                    env_file.write(f"{key}=[REDACTED]\n")
                else:
                    safe_value = value.replace("\r", "\\r").replace("\n", "\\n")
                    env_file.write(f"{key}={safe_value}\n")
        logger.info("Environment dump written to %s (mode 600)", env_path)
    except OSError as error:
        logger.warning("Failed to write environment dump to %s: %s", env_path, error)


def _recipe_kwargs_for_signature(
    config_builder: Callable,
    *,
    peft_scheme: str | None,
    seq_length: int | None = None,
) -> dict[str, object]:
    """Build kwargs accepted by a recipe function."""
    try:
        sig = inspect.signature(config_builder)
        params = sig.parameters
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

        accepts_peft = "peft" in params or has_var_keyword
        accepts_peft_scheme = "peft_scheme" in params
        accepts_seq_length = "seq_length" in params or has_var_keyword
    except (ValueError, TypeError):
        accepts_peft = True
        accepts_peft_scheme = False
        accepts_seq_length = False

    kwargs: dict[str, object] = {}
    if peft_scheme is not None:
        if accepts_peft_scheme:
            kwargs["peft_scheme"] = peft_scheme
        elif accepts_peft:
            kwargs["peft"] = peft_scheme
        elif peft_scheme.lower() != "lora":
            builder_name = getattr(config_builder, "__name__", repr(config_builder))
            raise ValueError(
                f"Recipe '{builder_name}' does not accept a configurable PEFT scheme; "
                f"--mode {peft_scheme} is unsupported."
            )
    if accepts_seq_length and seq_length is not None:
        kwargs["seq_length"] = seq_length
    return kwargs


def _load_with_optional_kwargs(
    config_builder: Callable,
    *,
    peft_scheme: str | None,
    seq_length: int | None = None,
) -> ConfigContainer:
    """Call a recipe function with only the optional kwargs it accepts."""
    kwargs = _recipe_kwargs_for_signature(
        config_builder,
        peft_scheme=peft_scheme,
        seq_length=seq_length,
    )
    return config_builder(**kwargs)


@functools.lru_cache(maxsize=1)
def library_recipe_package() -> ModuleType:
    """Import the library recipe package only when that source is selected."""
    import megatron.bridge.recipes as recipes

    return recipes


@functools.lru_cache(maxsize=1)
def library_h100_modules() -> tuple[str, ...]:
    """Return import paths for H100 library recipe alias packages."""
    recipes = library_recipe_package()
    module_names = []
    for module_info in pkgutil.iter_modules(recipes.__path__):
        if not module_info.ispkg or module_info.name.startswith("_") or module_info.name == "utils":
            continue
        h100_module_name = f"{recipes.__name__}.{module_info.name}.h100"
        try:
            importlib.import_module(h100_module_name)
        except ModuleNotFoundError as exc:
            if exc.name != h100_module_name:
                raise
            continue
        module_names.append(h100_module_name)
    return tuple(sorted(module_names))


def find_library_recipe(recipe_name: str, *, model_family_name: str | None = None) -> Callable | None:
    """Find a library recipe function by legacy or H100-style exported function name."""
    recipes = library_recipe_package()
    if hasattr(recipes, recipe_name):
        recipe_fn = getattr(recipes, recipe_name)
        if callable(recipe_fn):
            return cast(Callable, recipe_fn)

    if model_family_name is not None:
        module_names = (f"{recipes.__name__}.{model_family_name}.h100",)
    else:
        module_names = library_h100_modules()

    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name:
                raise
            continue
        recipe_fn = getattr(module, recipe_name, None)
        if callable(recipe_fn):
            return cast(Callable, recipe_fn)
    return None


def find_benchmark_recipe(recipe_name: str) -> Callable | None:
    """Find one flat benchmark recipe by its exported function name."""
    try:
        family = benchmark_recipe_family(recipe_name)
    except ValueError:
        return None
    module_name = f"megatron.bridge.perf_recipes.{family}"
    recipe_fn = getattr(importlib.import_module(module_name), recipe_name, None)
    return cast(Callable, recipe_fn) if callable(recipe_fn) else None


def load_recipe(
    recipe_name: str,
    peft_scheme: str | None = None,
    seq_length: int | None = None,
) -> ConfigContainer:
    """Load an exact benchmark recipe when exported, otherwise a library recipe."""
    if resolved_benchmark_recipe_metadata(recipe_name) is not None:
        config_builder = find_benchmark_recipe(recipe_name)
        package_name = "megatron.bridge.perf_recipes"
        recipe_kind = "benchmark"
        if recipe_name in BENCHMARK_RECIPE_PRECEDENCE_COLLISIONS:
            logger.warning(
                "Recipe '%s' is exported by both packages; selecting the benchmark definition. "
                "Use the corresponding generic alias from megatron.bridge.recipes.",
                recipe_name,
            )
    else:
        config_builder = find_library_recipe(recipe_name)
        package_name = "megatron.bridge.recipes"
        recipe_kind = "library"
        if recipe_name in LIBRARY_RECIPE_PRECEDENCE_COLLISIONS:
            logger.warning(
                "Recipe '%s' is exported by both packages; selecting the library definition until unified "
                "benchmark finetuning is supported.",
                recipe_name,
            )

    if config_builder is None:
        if recipe_kind == "library":
            raise AttributeError(
                f"Recipe '{recipe_name}' not found in megatron.bridge.recipes or megatron.bridge.perf_recipes."
            )
        raise AttributeError(f"Recipe '{recipe_name}' is indexed but not callable in {package_name}.")
    logger.info("Loading %s recipe '%s' from %s", recipe_kind, recipe_name, package_name)
    return _load_with_optional_kwargs(
        config_builder,
        peft_scheme=peft_scheme,
        seq_length=seq_length,
    )


def load_forward_step(step_type: str, mode: str | None = None) -> Callable:
    """Load a forward-step callable by name."""
    step_key = step_type.lower()
    if step_key not in STEP_FUNCTIONS:
        raise ValueError(ERR_UNKNOWN_STEP.format(step_type=step_type, choices=", ".join(STEP_FUNCTIONS)))
    step = _load_step_function(step_key)
    if inspect.isclass(step):
        if "mode" in inspect.signature(step.__init__).parameters:
            return step(mode=mode)
        return step()
    return step


@functools.lru_cache(maxsize=None)
def _load_step_function(step_key: str) -> Callable:
    """Import a forward-step callable from the lazy step registry."""
    entry = STEP_FUNCTIONS[step_key]
    if callable(entry):
        return entry

    module_name, attribute_name = entry
    return cast(Callable, getattr(importlib.import_module(module_name), attribute_name))


def apply_cli_overrides(config: ConfigContainer, cli_overrides: list[str] | None) -> ConfigContainer:
    """Apply Hydra-style CLI overrides to a ConfigContainer."""
    from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides

    return process_config_with_overrides(config, cli_overrides=cli_overrides or None)


def _set_if_present(container: object, field_name: str, value: object | None) -> None:
    """Set a dataclass-style field only when both object and value are present."""
    if value is not None and container is not None and hasattr(container, field_name):
        setattr(container, field_name, value)


def apply_runtime_environment(config: ConfigContainer) -> ConfigContainer:
    """Apply environment settings derived from the resolved ConfigContainer."""
    apply_environment_variables(config)
    if getattr(getattr(config, "ddp", None), "nccl_ub", False):
        os.environ.setdefault("NCCL_NVLS_ENABLE", "1")
        os.environ.setdefault("NCCL_CTA_POLICY", "1")
    return config


def apply_determinism(config: ConfigContainer, *, deterministic: bool) -> ConfigContainer:
    """Apply deterministic-training config overrides when requested."""
    if deterministic:
        from megatron.bridge.recipes.utils.determinism_utils import apply_determinism_overrides

        apply_determinism_overrides(config)
    return config


def bootstrap_recipe_environment(
    config: ConfigContainer,
    *,
    script_path: str,
    argv: list[str],
) -> ConfigContainer:
    """Apply recipe environment defaults and re-exec before training imports.

    The PID-bound marker prevents inherited or stale values from skipping the
    bootstrap in a new process. ``execvpe`` preserves the current PID, so the
    second interpreter returns normally after re-applying idempotent defaults.
    """
    config = apply_runtime_environment(config)
    if os.environ.get(RECIPE_ENV_BOOTSTRAP_MARKER) == str(os.getpid()):
        return config

    environment = dict(os.environ)
    environment[RECIPE_ENV_BOOTSTRAP_MARKER] = str(os.getpid())
    os.execvpe(
        sys.executable,
        [sys.executable, script_path, *argv],
        environment,
    )
    raise RuntimeError("os.execvpe returned unexpectedly")


def sync_finetuning_cp_invariants(config: ConfigContainer, *, mode: str) -> ConfigContainer:
    """Apply the loss-reduction settings required by SFT with context parallelism."""
    if mode != "finetune":
        return config

    dataset = getattr(config, "dataset", None)
    is_finetuning_dataset = hasattr(dataset, "enable_offline_packing") or hasattr(dataset, "enable_in_batch_packing")
    if not is_finetuning_dataset:
        return config

    model = getattr(config, "model", None)
    dist = getattr(config, "dist", None)
    context_parallel_sizes = {getattr(model, "context_parallel_size", 1)}
    eval_context_parallel_size = getattr(dist, "eval_context_parallel_size", None)
    if eval_context_parallel_size is not None:
        context_parallel_sizes.add(eval_context_parallel_size)

    if any(size > 1 for size in context_parallel_sizes):
        _set_if_present(model, "calculate_per_token_loss", True)
        _set_if_present(getattr(config, "ddp", None), "average_in_collective", False)
    return config


def sync_model_pipeline_layout(
    config: ConfigContainer,
    *,
    cli_overrides: list[str],
) -> ConfigContainer:
    """Rebuild a recipe-owned pipeline layout after PP or VP overrides."""
    override_fields = {override.lstrip("+~").split("=", 1)[0] for override in cli_overrides}
    topology_fields = {
        "model.pipeline_model_parallel_size",
        "model.virtual_pipeline_model_parallel_size",
    }
    if not override_fields.intersection(topology_fields):
        return config
    if "model.pipeline_model_parallel_layout" in override_fields:
        return config

    model = getattr(config, "model", None)
    layout_builder = getattr(model, "_pipeline_model_parallel_layout_builder", None)
    if layout_builder is None:
        return config

    model.pipeline_model_parallel_layout = layout_builder(
        model.pipeline_model_parallel_size,
        model.virtual_pipeline_model_parallel_size,
    )
    return config


def sync_offline_packing_alignment(config: ConfigContainer) -> ConfigContainer:
    """Align offline-packed samples to the resolved length and parallel topology."""
    dataset = getattr(config, "dataset", None)
    if not getattr(dataset, "enable_offline_packing", False):
        return config

    packing_specs = getattr(dataset, "offline_packing_specs", None)
    if packing_specs is None:
        from megatron.bridge.data.packing import PackedSequenceSpecs

        packing_specs = PackedSequenceSpecs()
        dataset.offline_packing_specs = packing_specs
    elif isinstance(packing_specs, dict):
        from megatron.bridge.data.packing import PackedSequenceSpecs

        packing_specs = PackedSequenceSpecs(**packing_specs)
        dataset.offline_packing_specs = packing_specs

    seq_length = getattr(dataset, "seq_length", None)
    if seq_length is not None:
        packing_specs.packed_sequence_size = seq_length

    model = getattr(config, "model", None)
    dist = getattr(config, "dist", None)
    context_parallel_size = getattr(model, "context_parallel_size", 1)
    eval_context_parallel_size = getattr(dist, "eval_context_parallel_size", None)
    context_parallel_sizes = {context_parallel_size}
    if eval_context_parallel_size is not None:
        context_parallel_sizes.add(eval_context_parallel_size)

    tensor_parallel_size = getattr(model, "tensor_model_parallel_size", 1)
    sequence_parallel = bool(getattr(model, "sequence_parallel", False))
    cp_multiples = [2 * size if size > 1 else 1 for size in context_parallel_sizes]
    sp_multiples = [
        size * tensor_parallel_size if sequence_parallel and tensor_parallel_size > 1 else 1
        for size in context_parallel_sizes
    ]
    packing_specs.pad_seq_to_mult = math.lcm(packing_specs.pad_seq_to_mult or 1, *cp_multiples, *sp_multiples)
    return config


def sync_model_dataset_sequence_length(config: ConfigContainer) -> ConfigContainer:
    """Keep model sequence length aligned with dataset sequence length when both exist."""
    if not (
        hasattr(config, "model")
        and config.model is not None
        and hasattr(config.model, "seq_length")
        and hasattr(config, "dataset")
        and config.dataset is not None
    ):
        return config

    dataset_seq_length = getattr(config.dataset, "seq_length", None)
    if dataset_seq_length is None:
        dataset_seq_length = getattr(config.dataset, "sequence_length", None)
    if dataset_seq_length is not None and config.model.seq_length != dataset_seq_length:
        config.model.seq_length = dataset_seq_length
    return config


def save_config(config: ConfigContainer, save_path: str) -> None:
    """Save and log a ConfigContainer YAML file."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    config.to_yaml(save_path)
    logger.info("ConfigContainer saved to: %s", os.path.abspath(save_path))
    config.print_yaml()


def run_config(
    *,
    config: ConfigContainer,
    mode: str,
    step_func: Callable,
    dryrun: bool = False,
    save_config_filepath: str | None = None,
    dryrun_world_size: int | None = None,
    dump_environment: bool = False,
) -> bool:
    """Run or dry-run a ConfigContainer with the selected training function.

    Returns:
        True when the caller should return without launching training.
    """
    if dump_environment:
        dump_env_rank0()

    if dryrun:
        if dryrun_world_size is not None:
            from megatron.bridge.training.config import runtime_config_update

            temporary_environment = {"WORLD_SIZE": str(dryrun_world_size), "RANK": "0"}
            previous_environment = {name: os.environ.get(name) for name in temporary_environment}
            try:
                os.environ.update(temporary_environment)
                runtime_config_update(config)
            finally:
                for name, value in previous_environment.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
        logger_config = getattr(config, "logger", None)
        configured_path = getattr(logger_config, "save_config_filepath", None)
        save_config(config, save_config_filepath or configured_path or "ConfigContainer.yaml")
        return True

    if _get_rank_safe() == 0:
        logger.info("Final configuration:")
        config.print_yaml()

    train_func = _load_train_function(mode)
    train_func(config=config, forward_step_func=step_func)

    _destroy_process_group()

    return False


def _load_train_function(mode: str) -> Callable:
    """Import the selected training loop only after environment bootstrap."""
    entry = TRAIN_FUNCTIONS[mode]
    if callable(entry):
        return entry
    module_name, attribute_name = entry
    return cast(Callable, getattr(importlib.import_module(module_name), attribute_name))


def _destroy_process_group() -> None:
    """Destroy an initialized process group without importing Torch during bootstrap."""
    import torch

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _get_rank_safe() -> int:
    """Read the distributed rank without importing runtime helpers during bootstrap."""
    from megatron.bridge.utils.common_utils import get_rank_safe

    return get_rank_safe()
