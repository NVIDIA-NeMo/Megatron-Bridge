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

"""Compatibility helpers for loading PEFT adapters in the 26.06.01 image."""

from __future__ import annotations

import inspect
import logging
from typing import Any


LOGGER = logging.getLogger(__name__)


def apply_peft_weight_converter_compatibility() -> bool:
    """Make the 26.06.01 Transformers ``WeightConverter`` accept PEFT metadata.

    PEFT 0.19.1 preserves ``distributed_operation`` and
    ``quantization_operation`` while deriving adapter weight conversions. The
    Transformers 5.8.1 constructor bundled in 26.06.01 stores those fields on
    the base class but does not accept them as constructor arguments. Newer
    compatible constructors need no modification.

    Returns:
        ``True`` when the compatibility wrapper was installed, otherwise
        ``False``.
    """
    from transformers.core_model_loading import WeightConverter

    if getattr(WeightConverter, "_megatron_bridge_peft_compat", False):
        return False

    original_init = WeightConverter.__init__
    original_parameters = inspect.signature(original_init).parameters
    compatibility_parameters = {"distributed_operation", "quantization_operation"}
    if compatibility_parameters.issubset(original_parameters):
        return False

    def compatible_init(
        self: Any,
        source_patterns: str | list[str],
        target_patterns: str | list[str],
        operations: list[Any],
        distributed_operation: Any = None,
        quantization_operation: Any = None,
    ) -> None:
        init_kwargs: dict[str, Any] = {
            "source_patterns": source_patterns,
            "target_patterns": target_patterns,
            "operations": operations,
        }
        if "distributed_operation" in original_parameters:
            init_kwargs["distributed_operation"] = distributed_operation
        if "quantization_operation" in original_parameters:
            init_kwargs["quantization_operation"] = quantization_operation
        original_init(self, **init_kwargs)
        self.distributed_operation = distributed_operation
        self.quantization_operation = quantization_operation

    setattr(WeightConverter, "__init__", compatible_init)
    setattr(WeightConverter, "_megatron_bridge_peft_compat", True)
    LOGGER.info("Enabled the PEFT/Transformers WeightConverter compatibility wrapper.")
    return True
