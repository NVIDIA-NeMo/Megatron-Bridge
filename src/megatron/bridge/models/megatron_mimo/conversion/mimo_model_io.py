# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""MegatronMIMO model save/load helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from megatron.bridge.training.checkpointing import load_checkpoint, maybe_finalize_async_save, save_checkpoint
from megatron.bridge.training.config import CheckpointConfig, ConfigContainer, LoggerConfig, OptimizerConfig
from megatron.bridge.training.megatron_mimo_parallel_utils import get_active_module_pg
from megatron.bridge.training.state import GlobalState


if TYPE_CHECKING:
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.models.mimo import MimoModel

    from megatron.bridge.models.megatron_mimo.megatron_mimo_config import MegatronMIMOParallelismConfig
    from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import (
        MegatronMIMOInfra,
        MegatronMIMOProvider,
    )


logger = logging.getLogger(__name__)


def save_megatron_mimo_model(
    model: "MimoModel",
    infra: "MegatronMIMOInfra",
    provider: "MegatronMIMOProvider",
    path: Union[str, Path],
    *,
    hf_tokenizer_path: Optional[Union[str, Path]] = None,
    hf_tokenizer_kwargs: Optional[dict] = None,
    ckpt_format: str = "torch_dist",
) -> None:
    """Save a MegatronMIMO model in Megatron distributed-checkpoint format.

    Args:
        model: Constructed ``MimoModel``.
        infra: ``MegatronMIMOInfra`` from model construction.
        provider: Provider used to reconstruct the model on load.
        path: Directory to save the dist-checkpoint into.
        hf_tokenizer_path: Optional HF model ID or path for tokenizer assets.
        hf_tokenizer_kwargs: Optional kwargs for ``AutoTokenizer.from_pretrained``.
        ckpt_format: Checkpoint format. Default ``"torch_dist"``.
    """
    tokenizer_config = None
    if hf_tokenizer_path is not None:
        from megatron.bridge.training.tokenizers.config import TokenizerConfig

        tokenizer_config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=str(hf_tokenizer_path),
            hf_tokenizer_kwargs=hf_tokenizer_kwargs or {},
        )

    active_module_name, local_pg_collection = get_active_module_pg(infra)

    state = GlobalState()
    state.cfg = ConfigContainer(
        model=provider,
        train=None,
        optimizer=OptimizerConfig(use_distributed_optimizer=False),
        ddp=None,
        scheduler=None,
        dataset=None,
        logger=LoggerConfig(),
        tokenizer=tokenizer_config,
        checkpoint=CheckpointConfig(
            # Async NVRx + block below: the sync mcore path deadlocks on
            # disjoint per-component state dicts at multi-billion-param scale.
            async_save=True,
            async_strategy="nvrx",
            use_persistent_ckpt_worker=False,
            save=str(path),
            save_optim=False,
            save_rng=False,
            ckpt_format=ckpt_format,
            dist_ckpt_optim_fully_reshardable=True,
            # fully_parallel_save=True triggers world-wide access-integrity
            # allgathers over disjoint per-component state dicts.
            fully_parallel_save=False,
        ),
        dist=None,
    )

    logger.info(
        "save_megatron_mimo_model: saving to %s (module=%r, ckpt_format=%s)",
        path,
        active_module_name,
        ckpt_format,
    )

    # Derived ModuleSpec fields don't round-trip through yaml — snapshot,
    # clear, then restore around the save.
    _saved_derived = _snapshot_derived_spec_fields(provider)
    try:
        _clear_derived_spec_fields(provider)
        state.initialize_async_checkpoint_worker()
        save_checkpoint(
            state=state,
            model=[model],
            optimizer=None,
            opt_param_scheduler=None,
            num_floating_point_operations_so_far=0,
            callback_manager=None,
            pg_collection=local_pg_collection,
            module_name=active_module_name,
        )
        maybe_finalize_async_save(state, state.cfg.checkpoint, blocking=True, terminate=True)
    finally:
        _restore_derived_spec_fields(provider, _saved_derived)

    if tokenizer_config is not None:
        from megatron.bridge.training.checkpointing import get_checkpoint_name, save_tokenizer_assets
        from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer

        tokenizer = build_tokenizer(tokenizer_config)
        checkpoint_name = get_checkpoint_name(str(path), 0, release=False)
        save_tokenizer_assets(tokenizer, tokenizer_config, checkpoint_name)


def load_megatron_mimo_model(
    path: Union[str, Path],
    *,
    parallelism_config: Optional["MegatronMIMOParallelismConfig"] = None,
    ddp_config: Optional["DistributedDataParallelConfig"] = None,
    fp16: bool = False,
    bf16: bool = True,
    wrap_with_ddp: bool = False,
    data_parallel_random_init: bool = False,
) -> tuple["MimoModel", "MegatronMIMOInfra", "MegatronMIMOProvider"]:
    """Load a MegatronMIMO model from a Megatron distributed-checkpoint.

    Args:
        path: Checkpoint parent directory or an ``iter_*`` directory.
        parallelism_config: Optional per-component parallelism override.
        ddp_config: DDP config forwarded to ``build_megatron_mimo_model``.
        fp16 / bf16: Precision flags forwarded to model construction.
        wrap_with_ddp: Whether to DDP-wrap.
        data_parallel_random_init: Forwarded to ``build_megatron_mimo_model``.

    Returns:
        ``(mimo_model, infra, provider)``.
    """
    from megatron.bridge.training.model_load_save import load_model_config

    iter_path = _resolve_iter_folder(Path(path))

    provider, _mlm_args = load_model_config(str(iter_path))

    if parallelism_config is not None:
        provider.megatron_mimo_parallelism_config = parallelism_config

    logger.info(
        "load_megatron_mimo_model: loading from %s (resolved iter=%s)",
        path,
        iter_path.name,
    )

    from megatron.bridge.models.megatron_mimo import build_megatron_mimo_model

    mimo_model, infra = build_megatron_mimo_model(
        provider,
        ddp_config=ddp_config,
        fp16=fp16,
        bf16=bf16,
        wrap_with_ddp=wrap_with_ddp,
        data_parallel_random_init=data_parallel_random_init,
    )

    active_module_name, local_pg_collection = get_active_module_pg(infra)

    state = GlobalState()
    state.cfg = ConfigContainer(
        model=provider,
        train=None,
        optimizer=OptimizerConfig(use_distributed_optimizer=False),
        ddp=None,
        scheduler=None,
        dataset=None,
        logger=LoggerConfig(),
        tokenizer=None,
        checkpoint=CheckpointConfig(
            async_save=False,
            load=str(path),
            load_optim=False,
            load_rng=False,
            ckpt_format="torch_dist",
            # Must match conversion save. MIMO conversion checkpoints use
            # disjoint per-component state dicts that do not pass global
            # fully-parallel access validation.
            fully_parallel_save=False,
        ),
        dist=None,
    )

    # ``load_checkpoint`` expects the global microbatch calculator to exist.
    from megatron.core import num_microbatches_calculator as nmc

    if nmc._GLOBAL_NUM_MICROBATCHES_CALCULATOR is None:
        import torch.distributed as _dist

        nmc.init_num_microbatches_calculator(
            _dist.get_rank() if _dist.is_initialized() else 0,
            None,  # rampup_batch_size
            1,  # global_batch_size
            1,  # micro_batch_size
            1,  # data_parallel_size
            False,  # decrease_batch_size_if_needed
        )

    load_checkpoint(
        state=state,
        model=[mimo_model],
        optimizer=None,
        opt_param_scheduler=None,
        pg_collection=local_pg_collection,
        module_name=active_module_name,
    )

    return mimo_model, infra, provider


def _snapshot_derived_spec_fields(provider: "MegatronMIMOProvider") -> dict:
    """Capture the provider's runtime-derived spec fields for restoration."""
    return {
        "language_model_spec": provider.language_model_spec,
        "modality_submodules_spec": provider.modality_submodules_spec,
        "special_token_ids": provider.special_token_ids,
        "_grids": getattr(provider, "_grids", None),
    }


def _clear_derived_spec_fields(provider: "MegatronMIMOProvider") -> None:
    """Reset derived spec fields so yaml serialisation captures only inputs."""
    provider.language_model_spec = None
    provider.modality_submodules_spec = {}
    provider.special_token_ids = {}
    provider._grids = None


def _restore_derived_spec_fields(provider: "MegatronMIMOProvider", saved: dict) -> None:
    """Restore derived spec fields after a save, leaving the provider usable."""
    provider.language_model_spec = saved["language_model_spec"]
    provider.modality_submodules_spec = saved["modality_submodules_spec"]
    provider.special_token_ids = saved["special_token_ids"]
    provider._grids = saved["_grids"]


def _resolve_iter_folder(path: Path) -> Path:
    """Resolve ``path`` to the checkpoint iteration selected for loading."""
    from megatron.bridge.training.checkpointing import (
        _DIRECT_ITERATION_DIR_SENTINEL,
        _resolve_checkpoint_iteration,
    )
    from megatron.bridge.training.utils.checkpoint_utils import get_checkpoint_name

    iteration, release = _resolve_checkpoint_iteration(str(path), None)
    if iteration == _DIRECT_ITERATION_DIR_SENTINEL:
        return path

    if iteration >= 0 or release:
        return Path(get_checkpoint_name(str(path), iteration, release))

    raise FileNotFoundError(
        f"No checkpoint tracker found under {path}; expected either a MIMO "
        f"dist-checkpoint parent directory or an iter folder directly."
    )
