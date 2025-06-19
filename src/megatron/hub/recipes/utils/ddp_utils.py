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

from megatron.core.distributed import DistributedDataParallelConfig


def ddp_config(
    check_for_nan_in_grad: bool = True,
    grad_reduce_in_fp32: bool = True,
    overlap_grad_reduce: bool = True,
    overlap_param_gather: bool = True,
    average_in_collective: bool = True,
    use_distributed_optimizer: bool = True,
) -> DistributedDataParallelConfig:
    """
    Creates Distributed Data Parallel config.
    """
    return  DistributedDataParallelConfig(
        check_for_nan_in_grad=check_for_nan_in_grad,
        grad_reduce_in_fp32=grad_reduce_in_fp32,
        overlap_grad_reduce=overlap_grad_reduce,
        overlap_param_gather=overlap_param_gather,
        average_in_collective=average_in_collective,
        use_distributed_optimizer=use_distributed_optimizer,
    )