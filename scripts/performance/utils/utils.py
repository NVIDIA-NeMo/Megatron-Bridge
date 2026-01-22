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

import importlib
import logging
import select
import sys
from dataclasses import dataclass, fields


logger = logging.getLogger(__name__)

# Default timeout for interactive config variant selection (in seconds)
CONFIG_VARIANT_SELECTION_TIMEOUT = 15


@dataclass
class WorkloadBaseConfig:
    """Container for workload base configs. This object exists because we cannot import MBridge on the headnode but need a place to store recipe overrides."""

    # NOTE: `num_gpus` is for representation purposes only. It is only meant to
    # communicate number of GPUs to be used for a specific workload in the file-
    # "scripts/performance/configs/<model_family_name>/workload_base_configs.py".

    # NOTE: You can specify number of GPUs to use for a SLURM job from command
    # line like `-ng/--num_gpus <num_gpus>` ("scripts/performance/README.md")
    # or update your sbatch script.
    num_gpus: int = 1

    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: int | None = None
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int | None = None

    global_batch_size: int = 1
    micro_batch_size: int = 1

    use_megatron_fsdp: bool | None = None
    nccl_ub: bool | None = None
    cuda_graph_impl: str | None = None
    cuda_graph_scope: str | None = None
    cpu_offloading_num_layers: int | None = None
    recompute_num_layers: int | None = None
    recompute_modules: list[str] | None = None

    # MoE configuration
    moe_flex_dispatcher_backend: str | None = None
    moe_a2a_overlap: bool | None = False
    peft: str | None = None

    # Pipeline parallelism layout
    pp_layout: Optional[str] = None

    @property
    def sequence_parallel(self) -> bool:
        """Get the sequence parallel flag."""
        return bool(self.tensor_model_parallel_size > 1)

    @property
    def gbs_scaling_factor(self) -> float:
        """Get the global batch size scaling factor."""
        return self.global_batch_size / self.num_gpus


def get_workload_base_config(
    model_family_name: str,
    model_recipe_name: str,
    gpu: str,
    compute_dtype: str,
    task: str,
    config_variant: str = "v1",
) -> dict[str, int]:
    """Get the workload base config for a given model, size, GPU, compute dtype, and FP8 recipe."""
    module_name = f"configs.{model_family_name}"
    try:
        module = importlib.import_module(module_name)
        logger.info(f"Imported module '{module_name}'.")
    except ModuleNotFoundError as exc:
        raise ValueError(f"Failed to import module '{module_name}'") from exc

    # Try versioned config name first (e.g., LLAMA3_70B_PRETRAIN_CONFIG_GB300_BF16_V1)
    versioned_config_name = f"{model_recipe_name}_{task}_config_{gpu}_{compute_dtype}_{config_variant}".upper()

    # Try versioned name first
    workload_base_config = getattr(module, versioned_config_name, None)
    if workload_base_config is not None:
        logger.info(f"Loaded config: {versioned_config_name}")
        logger.info(f"{workload_base_config}")
        return workload_base_config

    # If default v2 is unavailable, fall back to v1 when present
    if config_variant.lower() == "v2":
        fallback_versioned_config_name = f"{model_recipe_name}_{task}_config_{gpu}_{compute_dtype}_v1".upper()
        workload_base_config = getattr(module, fallback_versioned_config_name, None)
        if workload_base_config is not None:
            logger.warning(
                "Config variant '%s' not found; falling back to '%s'.",
                versioned_config_name,
                fallback_versioned_config_name,
            )
            logger.info(f"{workload_base_config}")
            return workload_base_config

    # Fallback to non-versioned config name for backward compatibility
    # (e.g., DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_BF16)
    base_config_name = f"{model_recipe_name}_{task}_config_{gpu}_{compute_dtype}".upper()

    # Fall back to non-versioned name (backward compatibility)
    workload_base_config = getattr(module, base_config_name, None)
    if workload_base_config is not None:
        logger.info(f"Loaded non-versioned config (fallback): {base_config_name}")
        return workload_base_config

    # Neither found - show helpful error with available variants
    available_variants = list_available_config_variants(model_family_name, model_recipe_name, gpu, compute_dtype, task)
    raise ValueError(
        f"Failed to get config from {module_name=}. "
        f"Tried: {versioned_config_name}, {base_config_name}. "
        f"Available variants: {available_variants}"
    )


def get_exp_name_config(
    args,
    model_family_name: str,
    model_recipe_name: str,
    gpu: str,
    compute_dtype: str,
    task: str,
    config_variant: str = "v1",
) -> str:
    """Get the experiment name from the base config and user overrides."""
    base_config = get_workload_base_config(
        model_family_name, model_recipe_name, gpu, compute_dtype, task, config_variant
    )
    num_gpus = args.num_gpus if args.num_gpus is not None else base_config.num_gpus
    tp_size = (
        args.tensor_model_parallel_size
        if args.tensor_model_parallel_size is not None
        else base_config.tensor_model_parallel_size
    )
    pp_size = (
        args.pipeline_model_parallel_size
        if args.pipeline_model_parallel_size is not None
        else base_config.pipeline_model_parallel_size
    )
    cp_size = (
        args.context_parallel_size if args.context_parallel_size is not None else base_config.context_parallel_size
    )
    vp_size = (
        args.virtual_pipeline_model_parallel_size
        if args.virtual_pipeline_model_parallel_size != -1
        else base_config.virtual_pipeline_model_parallel_size
    )
    ep_size = (
        args.expert_model_parallel_size
        if args.expert_model_parallel_size is not None
        else base_config.expert_model_parallel_size
    )
    etp_size = (
        args.expert_tensor_parallel_size
        if args.expert_tensor_parallel_size is not None
        else base_config.expert_tensor_parallel_size
    )
    mbs_size = args.micro_batch_size if args.micro_batch_size is not None else base_config.micro_batch_size
    gbs_size = args.global_batch_size if args.global_batch_size is not None else base_config.global_batch_size
    exp_config = f"gpus{num_gpus}_tp{tp_size}_pp{pp_size}_cp{cp_size}_vp{vp_size}_ep{ep_size}_etp{etp_size}_mbs{mbs_size}_gbs{gbs_size}"
    return exp_config


def list_available_config_variants(
    model_family_name: str,
    model_recipe_name: str,
    gpu: str,
    compute_dtype: str,
    task: str,
) -> list[str]:
    """List all available config variants for a given model/task/gpu/dtype combination.

    Returns:
        List of available variant names (e.g., ['v1', 'v2']) or ['(default)'] for non-versioned configs.
    """
    base_name = f"{model_recipe_name}_{task}_config_{gpu}_{compute_dtype}".upper()

    module_name = f"configs.{model_family_name}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ValueError(f"Failed to import module '{module_name}'") from exc

    variants = []

    # Check for versioned configs (e.g., *_V1, *_V2)
    for name in dir(module):
        if name.startswith(base_name + "_"):
            obj = getattr(module, name)
            if isinstance(obj, WorkloadBaseConfig):
                # Extract variant suffix (e.g., "LLAMA3_70B_PRETRAIN_CONFIG_GB300_BF16_V1" -> "v1")
                suffix = name[len(base_name) + 1 :]  # +1 for the underscore
                variants.append(suffix.lower())

    # Check for non-versioned config (backward compatibility)
    if hasattr(module, base_name):
        obj = getattr(module, base_name)
        if isinstance(obj, WorkloadBaseConfig):
            variants.append("(default)")

    return sorted(variants)


def get_perf_optimized_recipe(
    model_family_name: str,
    model_recipe_name: str,
    train_task: str,
    gpu: str,
    compute_dtype: str,
    mock: bool = True,
    config_variant: str = "v1",
):
    """Get the performance optimized recipe."""
    module_name = f"configs.{model_family_name}"
    try:
        module = importlib.import_module(module_name)
        logger.debug("Imported configuration module '%s'.", module_name)
    except ModuleNotFoundError as exc:
        raise ValueError(f"Failed to import configuration module '{module_name}'") from exc

    recipe_name = f"{model_recipe_name}_{train_task}_config_{gpu}"
    try:
        recipe_builder = getattr(module, recipe_name)
    except AttributeError as err:
        raise ValueError(f"Failed to get recipe builder '{recipe_name}' from module '{module_name}'") from err

    if train_task == "pretrain":
        return recipe_builder(precision=compute_dtype, mock=mock, config_variant=config_variant)
    else:
        return recipe_builder(precision=compute_dtype, config_variant=config_variant)


def get_library_recipe(model_family_name: str, model_recipe_name: str, train_task: str, wandb_experiment_name: str):
    """Get the library recipe.

    Note: Library pretrain recipes no longer accept kwargs. This function calls the recipe
    without arguments and then configures the output directories on the returned config.

    The old API was: recipe_builder(dir="/nemo_run/", name=wandb_experiment_name)
    This set:
        - run_output_dir = "/nemo_run/{name}"
        - checkpoint_dir = "/nemo_run/{name}/checkpoints"
        - tensorboard_dir = "/nemo_run/{name}/tb_logs"
    """
    import os

    family_pkg_path = f"megatron.bridge.recipes.{model_family_name}"
    family_pkg = importlib.import_module(family_pkg_path)

    if model_recipe_name == "deepseek_v3_32nodes" and train_task == "pretrain":
        model_recipe_name = "deepseek_v3_pretrain_config_32nodes"
    elif train_task == "pretrain":
        model_recipe_name = f"{model_recipe_name}_{train_task}_config"
    else:
        model_recipe_name = f"{model_recipe_name}_finetune_config"

    recipe_builder = getattr(family_pkg, model_recipe_name)

    # Library pretrain recipes no longer accept kwargs - call without args
    # and configure the returned ConfigContainer
    cfg = recipe_builder()

    # Set output directories that were previously configured via dir="/nemo_run/" and name=wandb_experiment_name
    base_output_dir = "/nemo_run"
    run_output_dir = os.path.join(base_output_dir, wandb_experiment_name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    # Checkpoint paths
    cfg.checkpoint.save = checkpoint_dir
    cfg.checkpoint.load = checkpoint_dir

    # Logger paths
    cfg.logger.tensorboard_dir = tensorboard_dir
    cfg.logger.wandb_exp_name = wandb_experiment_name
    cfg.logger.wandb_save_dir = os.path.join(run_output_dir, "wandb")

    return cfg


class _Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"
    WHITE = "\033[37m"


def _display_config_variants(
    model_family_name: str,
    model_recipe_name: str,
    gpu: str,
    compute_dtype: str,
    task: str,
    variants: list[str],
    timeout: int,
) -> None:
    """Display available config variants with their configurations.

    Args:
        variants: List of available variant names
        timeout: Timeout in seconds for user input
    """
    c = _Colors

    print(f"\n{c.DIM}{'=' * 80}{c.RESET}")
    print(
        f"{c.BOLD}{c.WHITE}Available config variants for {c.CYAN}{model_recipe_name}{c.WHITE}/{c.MAGENTA}{task}{c.WHITE}/{c.YELLOW}{gpu}{c.WHITE}/{c.GREEN}{compute_dtype}{c.WHITE}:{c.RESET}"
    )
    print(f"{c.DIM}{'=' * 80}{c.RESET}")

    # Fields to highlight in cyan (important config differences)
    highlight_fields = {"num_gpus", "global_batch_size"}

    for i, variant in enumerate(variants, 1):
        default_marker = f" {c.GREEN}(default){c.RESET}" if i == 1 else ""
        config_name = f"{model_recipe_name}_{task}_config_{gpu}_{compute_dtype}_{variant}".upper()
        print(
            f"\n  {c.BOLD}{c.CYAN}[{i}]{c.RESET} {c.BOLD}{c.WHITE}{variant}{c.RESET} {c.DIM}-{c.RESET} {c.YELLOW}{config_name}{c.RESET}{default_marker}"
        )
        print(f"  {c.DIM}{'-' * 76}{c.RESET}")

        # Fetch and display the WorkloadBaseConfig for this variant
        try:
            config = get_workload_base_config(model_family_name, model_recipe_name, gpu, compute_dtype, task, variant)
            for field in fields(config):
                value = getattr(config, field.name)
                if value is not None:
                    if field.name in highlight_fields:
                        print(f"      {c.CYAN}{field.name}: {value}{c.RESET}")
                    else:
                        print(f"      {field.name}: {value}")
        except ValueError:
            print(f"      {c.DIM}(config not found){c.RESET}")

    print(f"\n{c.DIM}{'=' * 80}{c.RESET}")
    print(f"\nSelect [1-{len(variants)}] (default: 1, timeout: {timeout}s): ", end="", flush=True)


def _get_user_selection_with_timeout(num_variants: int, timeout: int) -> int:
    """Get user selection with timeout, returning 1-based choice index.

    Args:
        num_variants: Number of available variants to choose from
        timeout: Timeout in seconds for user input

    Returns:
        1-based index of the selected variant (defaults to 1 on timeout/invalid input)
    """
    try:
        ready, _, _ = select.select([sys.stdin], [], [], float(timeout))
        if ready:
            user_input = sys.stdin.readline().strip()
            if user_input == "":
                return 1
            try:
                choice = int(user_input)
                if choice < 1 or choice > num_variants:
                    print("Invalid choice. Using default (1).")
                    return 1
                return choice
            except ValueError:
                print("Invalid input. Using default (1).")
                return 1
        else:
            print("\n⏱ Timeout - proceeding with default (1)")
            return 1
    except (OSError, AttributeError):
        # select.select doesn't work on Windows, fall back to default
        logger.warning("Interactive selection not available on this platform. Using default variant.")
        return 1


def select_config_variant_interactive(
    model_family_name: str,
    model_recipe_name: str,
    gpu: str,
    compute_dtype: str,
    task: str,
    timeout: int = CONFIG_VARIANT_SELECTION_TIMEOUT,
) -> str:
    """Interactively select a config variant with timeout.

    Args:
        timeout: Timeout in seconds for user input (default: 15)

    Returns:
        Selected config variant name (e.g., 'v1', 'v2')
    """
    try:
        variants = list_available_config_variants(model_family_name, model_recipe_name, gpu, compute_dtype, task)
    except ValueError as e:
        logger.error(f"Failed to list config variants: {e}")
        sys.exit(1)

    if not variants:
        logger.error(
            f"No config variants found for {model_recipe_name}/{task}/{gpu}/{compute_dtype}. "
            f"Please add configs with naming pattern: {model_recipe_name.upper()}_{task.upper()}_CONFIG_{gpu.upper()}_{compute_dtype.upper()}_V1"
        )
        sys.exit(1)

    # Display available variants
    _display_config_variants(model_family_name, model_recipe_name, gpu, compute_dtype, task, variants, timeout)

    # Get user selection
    choice = _get_user_selection_with_timeout(len(variants), timeout)

    selected_variant = variants[choice - 1]
    print(f"\nUsing config variant: {selected_variant}")
    return selected_variant
