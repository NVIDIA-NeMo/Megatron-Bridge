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

"""DPO training entry point: run-config guardrails, artifact validation, loop hand-off."""

import logging
from typing import Any, Literal

from megatron.bridge.data.builders.dpo import DPODatasetConfig
from megatron.bridge.data.datasets.preference import ScoringFingerprint, validate_scoring_metadata
from megatron.bridge.training.callbacks import Callback, CallbackManager
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.forward_step_func_types import ForwardStepCallable


logger = logging.getLogger(__name__)


def dpo_train(
    config: ConfigContainer,
    forward_step_func: ForwardStepCallable,
    callbacks: list[Callback] | CallbackManager | None = None,
) -> None:
    """Run DPO training over a scored preference dataset."""
    validate_dpo_run_config(config)
    finetune(config, forward_step_func, callbacks=callbacks)


def validate_dpo_run_config(config: ConfigContainer) -> None:
    """Fail fast, listing every violation, if the run would train incorrectly."""
    problems: list[str] = []
    dataset_is_dpo = isinstance(config.dataset, DPODatasetConfig)
    if config.dpo is None:
        problems.append("config.dpo must be a DPOLossConfig; the DPO forward step binds it into the loss.")
    if not dataset_is_dpo:
        problems.append(
            f"config.dataset must be a DPODatasetConfig, got {type(config.dataset).__name__}: DPO needs "
            "the pair-granular dataset and loader."
        )
    elif not config.dataset.ref_artifact:
        problems.append(
            "dataset.ref_artifact must be set: training needs the offline scorer's artifact "
            "(score_reference_logprobs.py --output) to anchor the margins."
        )
    if not config.model.calculate_per_token_loss:
        problems.append(
            "model.calculate_per_token_loss must be True: the DPO loss returns a raw (loss_sum, "
            "num_live_pairs) that finalize_model_grads normalizes by the global live-pair count."
        )
    if config.ddp.average_in_collective:
        problems.append(
            "ddp.average_in_collective must be False: gradients are normalized by the global live-pair "
            "count, not averaged over DP ranks."
        )
    if config.model.context_parallel_size != 1:
        problems.append("context_parallel_size > 1 is not supported for DPO.")
    if config.model.mtp_num_layers:
        problems.append(
            "model.mtp_num_layers must be None for DPO: an MTP head would backpropagate its own "
            "next-token CE on top of the DPO loss."
        )
    if config.train.micro_batch_size % 2 != 0 or config.train.global_batch_size % 2 != 0:
        problems.append(
            f"Batch sizes are row-denominated and must be even (one pair == two rows); got "
            f"micro_batch_size={config.train.micro_batch_size}, "
            f"global_batch_size={config.train.global_batch_size}."
        )

    validation_enabled = (config.validation.eval_iters or 0) > 0
    if validation_enabled:
        if not (dataset_is_dpo and config.dataset.has_validation_split):
            problems.append(
                "validation.eval_iters > 0 needs a scored validation split: set dataset.validation_source "
                "plus dataset.validation_ref_artifact."
            )
        if not config.validation.eval_interval:
            problems.append(
                "validation.eval_interval must be set when eval_iters > 0: the loop only evaluates "
                "every eval_interval steps."
            )
        if config.dist.eval_context_parallel_size is not None:
            problems.append("dist.eval_context_parallel_size is not supported for DPO.")
        for name in ("eval_global_batch_size", "eval_micro_batch_size"):
            value = getattr(config.validation, name)
            if value is not None and value % 2 != 0:
                problems.append(
                    f"validation.{name} is row-denominated and must be even (one pair == two rows); got {value}."
                )
    if problems:
        raise ValueError("DPO run config is invalid:\n- " + "\n- ".join(problems))

    _validate_artifact(config, config.dataset.ref_artifact, "train")
    if validation_enabled:
        _validate_artifact(config, config.dataset.validation_ref_artifact, "validation")


def _validate_artifact(config: ConfigContainer, artifact_dir: str, split: Literal["train", "validation"]) -> None:
    metadata = validate_scoring_metadata(artifact_dir, expected_scoring_metadata(config, split))
    _warn_on_expert_layout_mismatch(config, artifact_dir, metadata)


def _warn_on_expert_layout_mismatch(config: ConfigContainer, artifact_dir: str, metadata: dict[str, Any]) -> None:
    """Warn, never fail, when the trainer's expert layout (EP, ETP) differs from the scoring run's."""
    scorer_tp = metadata.get("tensor_model_parallel_size", 1)
    scored = (
        metadata.get("expert_model_parallel_size", 1),
        metadata.get("expert_tensor_parallel_size", scorer_tp),
    )
    trainer = (
        config.model.expert_model_parallel_size,
        config.model.expert_tensor_parallel_size or config.model.tensor_model_parallel_size,
    )
    if scored != trainer:
        logger.warning(
            "Expert layout differs from the scoring run: %s was scored at (EP, ETP)=%s, this run "
            "trains at %s. Expect a small step-0 shift, not margin corruption.",
            artifact_dir,
            scored,
            trainer,
        )


def expected_scoring_metadata(
    config: ConfigContainer, split: Literal["train", "validation"] = "train"
) -> ScoringFingerprint:
    field = "validation_source" if split == "validation" else "source"
    source = config.dataset.resolve_source(getattr(config.dataset, field), field=field)
    dataset_key, source_split = config.dataset.source_identity(source)
    return ScoringFingerprint(
        dataset=dataset_key,
        split=source_split,
        tokenizer=config.dataset.tokenizer_name,
        max_seq_length=config.dataset.seq_length,
        prompt_key=config.dataset.prompt_key,
        tensor_model_parallel_size=config.model.tensor_model_parallel_size,
        sequence_parallel=bool(config.model.sequence_parallel),
    )
