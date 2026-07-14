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

import functools
import importlib
import inspect
import logging
import math
import os
import pkgutil
import re
from collections.abc import Callable
from typing import cast

import torch

import megatron.bridge.recipes as recipes
from megatron.bridge.recipes.utils.determinism_utils import apply_determinism_overrides
from megatron.bridge.training.config import ConfigContainer, TokenizerConfig
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides
from megatron.bridge.utils.common_utils import get_rank_safe


logger = logging.getLogger(__name__)

SENSITIVE_ENV_VAR_PATTERN = re.compile(
    r"(^|_)(TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|ACCESS_KEY|SECRET_KEY|PRIVATE_KEY|AUTHORIZATION)(_|$)",
    re.IGNORECASE,
)

StepFunctionEntry = Callable | tuple[str, str]

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

TRAIN_FUNCTIONS = {
    "pretrain": pretrain,
    "finetune": finetune,
}

ERR_UNKNOWN_STEP = "Unknown step type: {step_type}. Choose from: {choices}"


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
def library_h100_modules() -> tuple[str, ...]:
    """Return import paths for H100 library recipe alias packages."""
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


def load_recipe(
    recipe_name: str,
    peft_scheme: str | None = None,
    seq_length: int | None = None,
) -> ConfigContainer:
    """Load a recipe from the library recipe package."""
    config_builder = find_library_recipe(recipe_name)
    if config_builder is None:
        raise AttributeError(f"Recipe '{recipe_name}' not found in megatron.bridge.recipes.")
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
    return process_config_with_overrides(config, cli_overrides=cli_overrides or None)


def apply_tokenizer_override(
    config: ConfigContainer,
    *,
    tokenizer_type: str | None,
    tokenizer_model: str | None,
    vocab_size: int,
) -> ConfigContainer:
    """Apply tokenizer-related CLI overrides."""
    if tokenizer_type == "NullTokenizer":
        config.tokenizer = TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=vocab_size)
    elif tokenizer_type == "HuggingFaceTokenizer":
        if not tokenizer_model:
            raise ValueError("--tokenizer-model is required when using HuggingFaceTokenizer")
        config.tokenizer = TokenizerConfig(tokenizer_type="HuggingFaceTokenizer", tokenizer_model=tokenizer_model)
    elif tokenizer_type == "SentencePieceTokenizer":
        if not tokenizer_model:
            raise ValueError("--tokenizer-model is required for SentencePieceTokenizer")
        config.tokenizer = TokenizerConfig(tokenizer_type="SentencePieceTokenizer", tokenizer_model=tokenizer_model)
    return config


def _set_if_present(container: object, field_name: str, value: object | None) -> None:
    """Set a dataclass-style field only when both object and value are present."""
    if value is not None and container is not None and hasattr(container, field_name):
        setattr(container, field_name, value)


def _get_arg(args: object, name: str, default: object | None = None) -> object | None:
    """Read an argparse Namespace field without requiring every caller to define it."""
    return getattr(args, name, default)


def apply_launcher_overrides(config: ConfigContainer, args: object) -> ConfigContainer:
    """Apply easy launcher flags before Hydra-style overrides.

    The generic ``key=value`` overrides remain the most expressive path. These
    flags cover the common cases users expect from a simple training launcher.
    """
    train = getattr(config, "train", None)
    validation = getattr(config, "validation", None)
    dist = getattr(config, "dist", None)
    optimizer = getattr(config, "optimizer", None)
    scheduler = getattr(config, "scheduler", None)
    checkpoint = getattr(config, "checkpoint", None)
    dataset = getattr(config, "dataset", None)
    model = getattr(config, "model", None)
    logger_config = getattr(config, "logger", None)
    ddp = getattr(config, "ddp", None)

    _set_if_present(train, "train_iters", _get_arg(args, "max_steps"))
    _set_if_present(train, "global_batch_size", _get_arg(args, "global_batch_size"))
    _set_if_present(train, "micro_batch_size", _get_arg(args, "micro_batch_size"))
    _set_if_present(train, "eval_interval", _get_arg(args, "eval_interval"))
    _set_if_present(train, "eval_iters", _get_arg(args, "eval_iters"))
    _set_if_present(validation, "eval_interval", _get_arg(args, "eval_interval"))
    _set_if_present(validation, "eval_iters", _get_arg(args, "eval_iters"))

    _set_if_present(dist, "distributed_timeout_minutes", _get_arg(args, "distributed_timeout_minutes"))

    _set_if_present(optimizer, "lr", _get_arg(args, "lr"))
    _set_if_present(optimizer, "min_lr", _get_arg(args, "min_lr"))
    _set_if_present(scheduler, "lr_warmup_iters", _get_arg(args, "warmup_iters"))
    _set_if_present(scheduler, "lr_decay_iters", _get_arg(args, "lr_decay_iters"))

    _set_if_present(checkpoint, "pretrained_checkpoint", _get_arg(args, "pretrained_checkpoint"))
    _set_if_present(checkpoint, "save", _get_arg(args, "save_dir"))
    _set_if_present(checkpoint, "load", _get_arg(args, "load_dir"))
    _set_if_present(checkpoint, "save_interval", _get_arg(args, "save_interval"))
    _set_if_present(checkpoint, "most_recent_k", _get_arg(args, "most_recent_k"))

    seq_length = _get_arg(args, "seq_length")
    _set_if_present(dataset, "seq_length", seq_length)
    _set_if_present(dataset, "sequence_length", seq_length)
    _set_if_present(model, "seq_length", seq_length)
    _set_if_present(model, "tensor_model_parallel_size", _get_arg(args, "tensor_model_parallel_size"))
    if (
        _get_arg(args, "tensor_model_parallel_size") is not None
        and model is not None
        and hasattr(model, "sequence_parallel")
    ):
        config.model.sequence_parallel = bool(config.model.tensor_model_parallel_size > 1)
    _set_if_present(model, "pipeline_model_parallel_size", _get_arg(args, "pipeline_model_parallel_size"))
    _set_if_present(model, "context_parallel_size", _get_arg(args, "context_parallel_size"))
    vp_size = _get_arg(args, "virtual_pipeline_model_parallel_size", -1)
    if vp_size != -1 and model is not None and hasattr(model, "virtual_pipeline_model_parallel_size"):
        model.virtual_pipeline_model_parallel_size = vp_size
    _set_if_present(model, "expert_model_parallel_size", _get_arg(args, "expert_model_parallel_size"))
    _set_if_present(model, "expert_tensor_parallel_size", _get_arg(args, "expert_tensor_parallel_size"))

    save_config_filepath = _get_arg(args, "save_config_filepath")
    if save_config_filepath is not None and logger_config is not None:
        config.logger.save_config_filepath = cast(str, save_config_filepath)
        os.makedirs(os.path.dirname(os.path.abspath(config.logger.save_config_filepath)), exist_ok=True)
    _set_if_present(logger_config, "log_interval", _get_arg(args, "log_interval"))
    _set_if_present(logger_config, "wandb_project", _get_arg(args, "wandb_project"))
    _set_if_present(logger_config, "wandb_entity", _get_arg(args, "wandb_entity"))
    _set_if_present(logger_config, "wandb_exp_name", _get_arg(args, "wandb_name"))
    _set_if_present(logger_config, "wandb_save_dir", _get_arg(args, "wandb_dir"))

    config = apply_tokenizer_override(
        config,
        tokenizer_type=cast(str | None, _get_arg(args, "tokenizer_type")),
        tokenizer_model=cast(str | None, _get_arg(args, "tokenizer_model")),
        vocab_size=cast(int, _get_arg(args, "vocab_size", 32000)),
    )

    if getattr(ddp, "nccl_ub", False):
        os.environ["NCCL_NVLS_ENABLE"] = "1"
        os.environ["NCCL_CTA_POLICY"] = "1"

    return config


def apply_determinism(config: ConfigContainer, *, deterministic: bool) -> ConfigContainer:
    """Apply deterministic-training config overrides when requested."""
    if deterministic:
        os.environ.setdefault("NCCL_ALGO", "Ring")
        os.environ.setdefault("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        apply_determinism_overrides(config)
    return config


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


def sync_offline_packing_alignment(config: ConfigContainer) -> ConfigContainer:
    """Align offline-packed samples to the resolved length and parallel topology."""
    dataset = getattr(config, "dataset", None)
    if not getattr(dataset, "enable_offline_packing", False):
        return config

    packing_specs = getattr(dataset, "offline_packing_specs", None)
    if packing_specs is None:
        raise ValueError("offline_packing_specs must be set when enable_offline_packing=True.")

    sequence_length = getattr(dataset, "seq_length", None)
    if sequence_length is None:
        sequence_length = getattr(dataset, "sequence_length", None)
    if sequence_length is not None:
        packing_specs.packed_sequence_size = sequence_length

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
    dump_environment: bool = False,
) -> bool:
    """Run or dry-run a ConfigContainer with the selected training function.

    Returns:
        True when the caller should return without launching training.
    """
    if dump_environment:
        dump_env_rank0()

    if dryrun:
        save_config(config, save_config_filepath or "ConfigContainer.yaml")
        return True

    if get_rank_safe() == 0:
        logger.info("Final configuration:")
        config.print_yaml()

    train_func = TRAIN_FUNCTIONS[mode]
    train_func(config=config, forward_step_func=step_func)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    return False
