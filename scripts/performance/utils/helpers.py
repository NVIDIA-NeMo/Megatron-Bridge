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

from megatron.bridge.training.comm_overlap import *
from megatron.bridge.training.mixed_precision import (
    bf16_mixed,
    bf16_with_fp8_current_scaling_mixed,
    bf16_with_fp8_mixed,
    bf16_with_fp8_subchannel_scaling_mixed,
    bf16_with_mxfp8_mixed,
)


COMM_OVERLAP_CONFIG_MAP = {
    "llama3_70b": {
        "h100": {
            "bf16": userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
            "fp8": userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192,
        },
        "b200": {
            "bf16": userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192,
            "fp8": userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192,
        },
        "gb200": {
            "bf16": userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192,
            "fp8": userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192,
        },
    },
    "llama31_405b": {
        "h100": {
            "bf16": userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192,
            "fp8": userbuffers_fp8_h100_h16384_tp8_cp2_mbs1_seqlen8192,
        },
        "b200": {
            "bf16": userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192,
            "fp8": userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192,
        },
        "gb200": {
            "bf16": userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192,
            "fp8": userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192,
        },
    },
}


def get_precision_config(compute_dtype: str, fp8_recipe: str):
    """Get the precision configs for the given compute dtype and FP8 recipe."""
    if compute_dtype == "fp8":
        if fp8_recipe == "ds":
            return bf16_with_fp8_mixed()
        elif fp8_recipe == "cs":
            return bf16_with_fp8_current_scaling_mixed()
        elif fp8_recipe == "mx":
            return bf16_with_mxfp8_mixed()
        elif fp8_recipe == "ss":
            return bf16_with_fp8_subchannel_scaling_mixed()
        else:
            raise ValueError(f"Invalid FP8 recipe: {fp8_recipe}")
    elif compute_dtype == "bf16":
        return bf16_mixed()
    else:
        raise ValueError(f"Invalid compute dtype: {compute_dtype}")
