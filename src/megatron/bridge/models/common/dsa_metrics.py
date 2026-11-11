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

"""DSA metric setup for models that use fixed tracker storage."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.training.models.base import ModelConfig


if TYPE_CHECKING:
    from megatron.bridge.models.model_provider import ModelProviderMixin


@dataclass
class _DSAMetricTrackerHook:
    pg_collection: ProcessGroupCollection

    def __call__(self, model: list[MegatronModule]) -> list[MegatronModule]:
        try:
            from megatron.core.transformer.experimental_attention_variant.dsa_logging import (
                initialize_dsa_metric_tracker,
            )
        except ImportError:
            return model  # Older MCore revisions do not expose explicit tracker initialization.
        # All owning PP ranks participate, including those without a local DSA writer.
        initialize_dsa_metric_tracker(model, self.pg_collection)
        return model


def register_dsa_metric_hook(
    model_config: "ModelConfig | ModelProviderMixin", pg_collection: ProcessGroupCollection
) -> None:
    """Register one post-wrap tracker hook, replacing any previous build's groups."""
    if isinstance(model_config, ModelConfig):
        hooks = model_config.post_wrap_hooks
    else:
        hooks = getattr(model_config, "_post_wrap_hooks", [])
    hooks[:] = [hook for hook in hooks if not isinstance(hook, _DSAMetricTrackerHook)]
    if not (getattr(model_config, "dsa_indexer_loss_coeff", None) or 0.0) > 0:
        return
    hook = _DSAMetricTrackerHook(pg_collection)
    if isinstance(model_config, ModelConfig):
        hooks.append(hook)
    else:
        model_config.register_post_wrap_hook(hook)


def reset_dsa_metric_tracker() -> None:
    """Release fixed tracker storage before destroying its process groups."""
    try:
        from megatron.core.transformer.experimental_attention_variant import dsa_logging
    except ImportError:
        return
    if hasattr(dsa_logging, "initialize_dsa_metric_tracker"):
        dsa_logging.DSAIndexerLossLoggingHelper.tracker.clear()
