#!/bin/bash
# GPU_COUNT=x4
# CI_TIMEOUT=30
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -xeuo pipefail

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_DEVICE_MAX_CONNECTIONS=32
export CUDNNFE_CLUSTER_OVERLAP_MARGIN=8
export NCCL_GRAPH_REGISTER=0
export NCCL_NET_GDR_C2C=1
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_NVLS_ENABLE=0
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=8
export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128
export NVLINK_DOMAIN_SIZE=72
export NVTE_BWD_LAYERNORM_SM_MARGIN=20
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=20
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=False
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_HIGH_PRIORITY=1
export USE_MNNVL=1

uv run python -m torch.distributed.run --nproc_per_node=4 --nnodes=1 -m coverage run \
  --data-file=/opt/Megatron-Bridge/.coverage --source=/opt/Megatron-Bridge/ --parallel-mode \
  -m pytest -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA \
  tests/functional_tests/test_groups/recipes/test_qwen3_moe_perf_proxy.py::TestQwen3MoePerfProxy::test_gb200_fp8mx

coverage combine -q
