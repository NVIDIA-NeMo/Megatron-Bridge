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

    if preset == {} and args.model_name in ["deepseek", "llama3", "llama31"]:
        defaults = yaml_root.get("defaults")
        yaml_gpu_defaults = defaults.get("gpu_defaults")
        default_num_gpus = yaml_gpu_defaults.get(args.gpu) or yaml_gpu_defaults.get(args.gpu.lower())
        num_gpus_yaml_key = f"num_gpus_{default_num_gpus}"
        preset = gpu_block.get(num_gpus_yaml_key)
        scaling_factor = defaults.get("gbs_scaling_factor_default")
        preset["common"]["gbs"] = int(args.num_gpus * scaling_factor)

    if args.tensor_parallel_size:
        preset["common"]["tp"] = args.tensor_parallel_size
    if args.pipeline_parallel_size:
        preset["common"]["pp"] = args.pipeline_parallel_size
    if args.context_parallel_size:
        preset["common"]["cp"] = args.context_parallel_size
    if args.virtual_pipeline_parallel_size:
        preset["common"]["vp"] = args.virtual_pipeline_parallel_size
    if args.expert_parallel_size:
        preset["common"]["ep"] = args.expert_parallel_size
    if args.expert_tensor_parallel_size:
        preset["common"]["etp"] = args.expert_tensor_parallel_size
    if args.micro_batch_size:
        preset["common"]["mbs"] = args.micro_batch_size
    if args.global_batch_size:
        preset["common"]["gbs"] = args.global_batch_size

    return preset
