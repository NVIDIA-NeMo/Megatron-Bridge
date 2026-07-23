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

"""Shared recipe naming helpers."""

from __future__ import annotations


# Precision shorthand used in flat recipe names:
# cs = FP8 current scaling, mx = MXFP8, sc = FP8 subchannel.
PRECISION_NAME_MAP = {
    "bf16": "bf16",
    "fp8_cs": "fp8cs",
    "fp8_mx": "fp8mx",
    "fp8_sc": "fp8sc",
    "nvfp4": "nvfp4",
}


def normalize_precision_name(precision: str) -> str:
    """Return the precision token used in flat recipe function names."""
    return PRECISION_NAME_MAP.get(precision.lower(), precision.lower().replace("_", ""))


def recipe_variant_suffix(config_variant: str | None) -> str:
    """Return the function-name suffix used for non-canonical recipe variants."""
    return f"_{config_variant}" if config_variant and config_variant not in {"v1", "v2", "v3"} else ""


def recipe_function_name(
    *,
    model_recipe_name: str,
    task: str,
    num_gpus: int,
    gpu: str,
    precision: str,
    config_variant: str | None = None,
) -> str:
    """Build a flat recipe function name from CLI dimensions."""
    return (
        f"{model_recipe_name}_{task}_{num_gpus}gpu_{gpu}_{normalize_precision_name(precision)}"
        f"{recipe_variant_suffix(config_variant)}_config"
    )
