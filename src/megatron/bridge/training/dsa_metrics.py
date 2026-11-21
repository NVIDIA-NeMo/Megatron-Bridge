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

"""Clear fixed-storage DSA metrics between training attempts without rebinding tensors."""

from __future__ import annotations

from importlib import import_module
from numbers import Real
from types import ModuleType
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from megatron.bridge.models.transformer_config import TransformerConfig


_DSA_LOGGING_MODULE = "megatron.core.transformer.experimental_attention_variant.dsa_logging"


def _get_dsa_logging(model_config: TransformerConfig) -> ModuleType | None:
    coefficient = getattr(model_config, "dsa_indexer_loss_coeff", None)
    if not isinstance(coefficient, Real) or coefficient <= 0:
        return None
    try:
        dsa_logging = import_module(_DSA_LOGGING_MODULE)
    except ModuleNotFoundError as error:
        if error.name != _DSA_LOGGING_MODULE:
            raise
        return None
    # Older MCore versions, including the measured private DSv4 pin, own their
    # tracker in dsa.py and do not expose the fixed-storage lifecycle API.
    if not hasattr(dsa_logging, "initialize_dsa_metric_tracker"):
        return None
    return dsa_logging


def reset_dsa_metrics(model_config: TransformerConfig) -> None:
    """Clear values in place before a training attempt, retaining captured storage and groups."""
    dsa_logging = _get_dsa_logging(model_config)
    if dsa_logging is not None:
        dsa_logging.DSAIndexerLossLoggingHelper.clean_loss_in_tracker(preserve_groups=True)
