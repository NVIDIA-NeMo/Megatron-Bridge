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

from typing import Any


def get_perf_matrix_overrides(yaml_root: Any, args: Any) -> Any:
    """Get the performance matrix overrides from the YAML file."""
    perf = yaml_root.get("perf_matrix") if hasattr(yaml_root, "get") else None
    if not perf:
        return
    if args.gpu not in perf:
        return
    num_gpus_value = args.num_gpus or args.gpus_per_node
    num_gpus_yaml_key = f"num_gpus_{num_gpus_value}"
    gpu_block = perf.get(args.gpu) or {}
    preset = gpu_block.get(num_gpus_yaml_key) or {}

    # weak scaling for deepseek
    if preset == {} and args.model_name in ["deepseek"]:
        default_num_gpus = 1024 if args.gpu.lower() in ["h100"] else 256
        num_gpus_yaml_key = f"num_gpus_{default_num_gpus}"
        preset = gpu_block.get(num_gpus_yaml_key)
        preset["common"]["gbs"] = args.num_gpus * 8

    elif preset == {} and args.model_name in ["llama3", "llama31"]:
        gpu_defaults = {
            "gb300": {
                "405b": 128,
                "70b":  64,
                "8b":   8,
            },
            "gb200": {
                "405b": 128,
                "70b":  64,
                "8b":   8,
            },
            "b200": {
                "405b": 128,
                "70b":  64,
                "8b":   8,
            },
            "h100": {
                "405b": 1024,
                "70b":  64,
                "8b":   8,
            },
        }
        default_num_gpus = gpu_defaults[args.gpu][args.model_size]
        num_gpus_yaml_key = f"num_gpus_{default_num_gpus}"
        preset = gpu_block.get(num_gpus_yaml_key)
        scaling_factor = preset["common"]["gbs"] / default_num_gpus
        preset["common"]["gbs"] = int(args.num_gpus * scaling_factor)

    return preset
