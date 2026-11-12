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
    """Fail fast if the run has no usable reference artifact for each split it trains or evaluates on.

    The loss contract itself (per-token loss, no collective averaging, CP=1, no MTP, even
    row counts) is enforced by ``ConfigContainer.validate`` for any DPO dataset.
    """
    if not isinstance(config.dataset, DPODatasetConfig):
        raise ValueError(f"DPO requires a DPODatasetConfig dataset, got {type(config.dataset).__name__}.")
    if not config.dataset.ref_artifact:
        raise ValueError("DPO requires dataset.ref_artifact, the output of score_reference_logprobs.py.")

    validation_enabled = (config.validation.eval_iters or 0) > 0
    if validation_enabled:
        if not config.dataset.has_validation_split:
            raise ValueError(
                "validation.eval_iters > 0 requires a scored validation split: "
                "dataset.validation_source and dataset.validation_ref_artifact."
            )
        if not config.validation.eval_interval:
            raise ValueError("validation.eval_iters > 0 requires validation.eval_interval.")

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
    """The scoring fingerprint this run's config implies for ``split``, compared against the artifact's."""
    field = "validation_source" if split == "validation" else "source"
    source = config.dataset.resolve_source(getattr(config.dataset, field), field=field)
    dataset_key, source_split = config.dataset.source_identity(source)
    return ScoringFingerprint(
        dataset=dataset_key,
        split=source_split,
        tokenizer=config.tokenizer.tokenizer_model,
        max_seq_length=config.dataset.seq_length,
        prompt_key=config.dataset.prompt_key,
        tools_key=config.dataset.tools_key,
        tensor_model_parallel_size=config.model.tensor_model_parallel_size,
        sequence_parallel=bool(config.model.sequence_parallel),
        chat_template_kwargs=dict(config.dataset.chat_template_kwargs)
        if config.dataset.chat_template_kwargs
        else None,
    )
