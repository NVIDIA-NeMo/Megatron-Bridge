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

import logging
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Optional

import modelopt.torch.distill as mtd
import modelopt.torch.distill.plugins.megatron as mtd_mcore
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.utils import unwrap_model

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.models.transformer_config import TransformerConfig


if TYPE_CHECKING:
    from megatron.bridge.training.post_training.distillation import ModelOptDistillConfig


logger = logging.getLogger(__name__)


@dataclass
class DistillationProvider(TransformerConfig):
    """Provider for Bridge language models in distillation mode.

    Please use `convert_to_distillation_provider()` to create an instance of this class.

    Args:
        teacher: The teacher model provider.
        kd_config: Knowledge-distillation configuration.
        distill_submodule: If set, distill only this submodule of the built model (e.g.
            ``"language_model"`` for VLMs); the rest of the model is exported unchanged (only the
            submodule is trained).
            ``None`` distills the whole model.
    """

    teacher: Optional[GPTModelProvider | HybridModelProvider] = None
    kd_config: Optional["ModelOptDistillConfig"] = None
    distill_submodule: Optional[str] = None

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Use `convert_to_distillation_provider()` to create an instance of this class.")

    def __post_init__(self):
        assert getattr(self, "teacher", None) is not None, "Teacher model must be provided."

        shared_attrs = [
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
            "seq_length",
            "pipeline_dtype",
        ]
        for attr in shared_attrs:
            if getattr(self, attr) != getattr(self.teacher, attr):
                raise ValueError(f"Student and teacher providers must have the same {attr}.")

        # Logits are overwritten in-place when TE cross-entropy loss is enabled, so switch it back to native version.
        self.cross_entropy_fusion_impl = "native"

        # Hack to dynamically subclass other providers and still use their methods
        self._super_class = self.__class__.__bases__[0]

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> LanguageModule:
        """Build the (un-converted) student model.

        The knowledge-distillation conversion is deferred to ``_convert_hook`` (a pre-wrap hook
        registered by ``convert_to_distillation_provider``) so it runs *after* the student's weights
        are loaded. This lets a caller restore a quantized student (QAD) via an earlier pre-wrap hook,
        and lets a weight-loaded submodule (VLMs) be extracted before it is wrapped with the teacher.

        Args:
            pre_process: Whether to include pre-processing in the model, defaults to first pipeline stage
            post_process: Whether to include post-processing in the model, defaults to last pipeline stage
            vp_stage: Virtual pipeline stage

        Returns:
            The un-converted student model (converted later by ``_convert_hook``).
        """
        if vp_stage is not None:
            raise ValueError("ModelOpt KD currently does not support virtual-pipeline parallel.")
        return self._super_class.provide(self, pre_process, post_process, vp_stage)

    def _convert_hook(self, model_chunks: list) -> list:
        """Pre-wrap hook that applies the KD conversion after the student is weight-loaded.

        With ``distill_submodule`` set (e.g. VLMs), only that submodule is distilled and returned as
        the model; the full model is retained on ``full_model`` so the distilled submodule can be
        exported back within it. Registered after the bridge's weight-load hook, so weights are present.
        """
        assert len(model_chunks) == 1, "ModelOpt KD does not support virtual pipeline (>1 model chunk)."
        student_model = unwrap_model(model_chunks[0])
        # Hack to get teacher's pre-wrap hooks called to potentially load HF weights
        teacher_model = unwrap_model(
            self.teacher.provide_distributed_model(wrap_with_ddp=False, mixed_precision_wrapper=None)[0]
        )
        if self.distill_submodule is not None:
            #: The full built model, so callers can export the (in-place) distilled submodule within it.
            self.full_model = student_model
            student_model = getattr(student_model, self.distill_submodule)
            teacher_model = getattr(teacher_model, self.distill_submodule)

        kd_cfg = mtd_mcore.setup_distillation_config(self.kd_config, student_model.config, teacher_model.config)
        modelopt_cfg = {
            "teacher_model": teacher_model,
            "criterion": kd_cfg.criterion,
            "loss_balancer": kd_cfg.loss_balancer,
        }
        # ``mtd.convert`` mutates in place, so for the submodule case ``full_model`` already holds the
        # distilled submodule for export.
        kd_model = mtd.convert(student_model, mode=[("kd_loss", modelopt_cfg)])
        if self.distill_submodule is not None:
            # Export reads the distilled submodule from ``full_model``; enforce the in-place contract.
            assert getattr(self.full_model, self.distill_submodule) is kd_model
        mtd_mcore.adjust_distillation_model_for_mcore(kd_model, kd_cfg)
        return [kd_model]

    def to_cfg_dict(self) -> dict[str, Any]:
        """Custom method to save equivalent to the original provider class.

        Used by `_ConfigContainerBase` to serialize the main `ConfigContainer` to YAML.
        There is no need to restore a `DistillationProvider` from the run config file, as
        it can always be re-converted using the original student provider.

        Returns:
            Dictionary representation of this provider class
        """
        from megatron.bridge.training.utils.config_utils import _ConfigContainerBase

        result = {"_target_": f"{self._super_class.__module__}.{self._super_class.__qualname__}"}
        # Use fields from the actual student provider class, not DistillationProvider.
        # DistillationProvider's __dataclass_fields__ only includes TransformerConfig fields
        # (set at class definition time), missing GPTModelProvider-level fields like
        # vocab_size, share_embeddings_and_output_weights, etc.
        for field in fields(self._super_class):
            if field.name.startswith("_"):
                continue
            result[field.name] = _ConfigContainerBase._convert_value_to_dict(getattr(self, field.name))
        return result

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        # Mirror to teacher if it has that attribute
        if hasattr(self.teacher, name):
            setattr(self.teacher, name, value)


def convert_to_distillation_provider(
    student_provider: GPTModelProvider | HybridModelProvider,
    teacher_provider: GPTModelProvider | HybridModelProvider,
    kd_config: Optional["ModelOptDistillConfig"] = None,
    *,
    distill_submodule: Optional[str] = None,
) -> "DistillationProvider":
    """Convert a given model provider to a DistillationProvider.

    The KD conversion runs in a pre-wrap hook (after the student's weights are loaded), not in
    ``provide()``. To initialize the student from a checkpoint before conversion (e.g. QAD), register
    your own pre-wrap hook with ``student_provider.register_pre_wrap_hook(fn, prepend=True)`` so it runs
    before the KD-conversion hook.

    Args:
        student_provider: The student model provider (also the base class of the returned provider).
        teacher_provider: The teacher model provider.
        kd_config: Knowledge-distillation configuration.
        distill_submodule: If set, distill only this submodule of the built model (e.g.
            ``"language_model"`` for VLMs); the rest of the model is exported unchanged (only the
            submodule is trained).
    """

    assert isinstance(student_provider, (GPTModelProvider, HybridModelProvider)), (
        "Student provider must be a subclass of GPTModelProvider or HybridModelProvider."
    )
    assert isinstance(teacher_provider, (GPTModelProvider, HybridModelProvider)), (
        "Teacher provider must be a subclass of GPTModelProvider or HybridModelProvider."
    )

    DistillationProvider.__bases__ = (type(student_provider),)
    student_provider.__class__ = DistillationProvider

    student_provider.teacher = teacher_provider
    student_provider.kd_config = kd_config
    student_provider.distill_submodule = distill_submodule
    student_provider.__post_init__()

    # Convert after the bridge's weight-load hook (appended => runs last), so the student is fully
    # weight-loaded before it is wrapped with the teacher. Set _pre_wrap_hooks via object.__setattr__
    # (not register_pre_wrap_hook) to bypass __setattr__'s teacher-mirroring: when the student starts
    # with no hooks (e.g. QAD builds it with load_weights=False), the mirror would share the hook list
    # with the teacher, so building the teacher inside _convert_hook would recurse into _convert_hook.
    hooks = [*getattr(student_provider, "_pre_wrap_hooks", []), student_provider._convert_hook]
    object.__setattr__(student_provider, "_pre_wrap_hooks", hooks)

    return student_provider
