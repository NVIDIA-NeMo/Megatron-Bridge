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
from dataclasses import asdict, dataclass, fields

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from megatron.core.utils import get_te_version, is_te_min_version, is_torch_min_version

from megatron.bridge.models import GPTModelProvider, T5ModelProvider


try:
    import transformer_engine  # type: ignore  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False


@dataclass
class TPOverlapCfg:
    """Dataclass for linear layer TP overlap config."""

    pass


@dataclass
class PipelineOverlapCfg(TPOverlapCfg):
    """Dataclass for pipeline TP overlap config."""

    num_sm: int
    cga_size: int
    num_splits: int
    set_sm_margin: bool
    fp8_buf: bool = (False,)
    atomic_gemm: bool = False
    method: str = "pipeline"


@dataclass
class RingExchangeOverlapCfg(TPOverlapCfg):
    """Dataclass for ring exchange TP overlap config."""

    aggregate: bool = False
    method: str = "ring_exchange"
    num_sm: int = 1
    cga_size: int = 1
    set_sm_margin: bool = False
    fp8_buf: bool = False
    atomic_gemm: bool = False


@dataclass
class BulkOverlapCfg(TPOverlapCfg):
    """Dataclass for bulk TP overlap config."""

    num_sm: int
    cga_size: int
    set_sm_margin: bool
    method: str = "bulk"


@dataclass
class TransformerLayerTPOverlapCfg:
    """Dataclass for transformer layer TP overlap config."""

    qkv_dgrad: TPOverlapCfg
    qkv_wgrad: TPOverlapCfg
    fc1_dgrad: TPOverlapCfg
    fc1_wgrad: TPOverlapCfg
    qkv_fprop: TPOverlapCfg
    proj_dgrad: TPOverlapCfg
    fc1_fprop: TPOverlapCfg
    fc2_dgrad: TPOverlapCfg
    proj_fprop: TPOverlapCfg
    fc2_fprop: TPOverlapCfg


# TODO: Add more configs and create a getter function for expose a single api
# Model configs: H100/70B/TP8/MBS1/SeqLen8K
userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=24, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=PipelineOverlapCfg(num_sm=24, cga_size=2, num_splits=4, set_sm_margin=True),
    fc2_fprop=PipelineOverlapCfg(num_sm=16, cga_size=2, num_splits=4, set_sm_margin=True),
)

userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=24, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
    fc2_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
)

userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=16, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=24, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=PipelineOverlapCfg(num_sm=32, cga_size=2, num_splits=4, set_sm_margin=True),
    fc2_fprop=PipelineOverlapCfg(num_sm=16, cga_size=2, num_splits=4, set_sm_margin=True),
)

userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=24, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
    fc2_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
)

# llama3.1 405b
userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=24, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=PipelineOverlapCfg(num_sm=24, cga_size=2, num_splits=4, set_sm_margin=True),
    fc2_fprop=PipelineOverlapCfg(num_sm=8, cga_size=2, num_splits=4, set_sm_margin=True),
)

userbuffers_fp8_h100_h16384_tp8_cp2_mbs1_seqlen8192 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=24, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
    fc2_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
)

userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=8, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=24, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=PipelineOverlapCfg(num_sm=32, cga_size=2, num_splits=4, set_sm_margin=True),
    fc2_fprop=PipelineOverlapCfg(num_sm=8, cga_size=2, num_splits=4, set_sm_margin=True),
)

userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=8, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=32, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=8, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
    fc2_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
)

# llama3 70b LoRA
userbuffers_fp8_h100_h8192_tp2_mbs1_seqlen4096_lora = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    qkv_wgrad=None,
    fc1_dgrad=PipelineOverlapCfg(num_sm=16, cga_size=2, num_splits=4, set_sm_margin=True, fp8_buf=True),
    fc1_wgrad=None,
    qkv_fprop=RingExchangeOverlapCfg(set_sm_margin=True),
    proj_dgrad=RingExchangeOverlapCfg(set_sm_margin=True),
    fc1_fprop=RingExchangeOverlapCfg(set_sm_margin=True),
    fc2_dgrad=RingExchangeOverlapCfg(set_sm_margin=True),
    proj_fprop=RingExchangeOverlapCfg(cga_size=2, set_sm_margin=True, fp8_buf=True),
    fc2_fprop=RingExchangeOverlapCfg(cga_size=2, set_sm_margin=True, fp8_buf=True),
)

# llama3.1 405b LoRA
userbuffers_fp8_h100_h16384_tp4_mbs1_seqlen2048_lora = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    qkv_wgrad=None,
    fc1_dgrad=PipelineOverlapCfg(num_sm=16, cga_size=2, num_splits=4, set_sm_margin=True, fp8_buf=True),
    fc1_wgrad=None,
    qkv_fprop=RingExchangeOverlapCfg(aggregate=True),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=True),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=True),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=True),
    proj_fprop=PipelineOverlapCfg(num_sm=32, cga_size=2, num_splits=4, set_sm_margin=True, fp8_buf=True),
    fc2_fprop=PipelineOverlapCfg(num_sm=16, cga_size=2, num_splits=4, set_sm_margin=True, fp8_buf=True),
)

# GPT3 20b
userbuffers_bf16_h100_h6144_tp2_mbs2_seqlen2048 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=8, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=16, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=16, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=PipelineOverlapCfg(num_sm=16, cga_size=2, num_splits=4, set_sm_margin=True),
    fc2_fprop=PipelineOverlapCfg(num_sm=24, cga_size=2, num_splits=4, set_sm_margin=True),
)

userbuffers_fp8_h100_h6144_tp2_mbs2_seqlen2048 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=8, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=16, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=16, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=PipelineOverlapCfg(num_sm=16, cga_size=2, num_splits=4, set_sm_margin=True, fp8_buf=True),
    fc2_fprop=PipelineOverlapCfg(num_sm=24, cga_size=2, num_splits=4, set_sm_margin=True, fp8_buf=True),
)

# GPT3 175b
userbuffers_bf16_h100_h12288_tp4_mbs1_seqlen2048 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=PipelineOverlapCfg(num_sm=24, cga_size=2, num_splits=4, set_sm_margin=True),
    fc2_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
)

userbuffers_fp8_h100_h12288_tp4_mbs1_seqlen2048 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
    fc2_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
)

userbuffers_bf16_b200_h12288_tp4_mbs1_seqlen2048 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=16, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=8, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
    fc2_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
)

userbuffers_fp8_b200_h12288_tp4_mbs1_seqlen2048 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=16, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=8, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
    fc2_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
)

# Nemotron 15B
userbuffers_bf16_b200_h6144_tp2_mbs1_seqlen4096 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=8, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=32, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=8, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
    fc2_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
)

# Nemotron 340B
userbuffers_bf16_b200_h18432_tp8_mbs1_seqlen4096 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=8, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=32, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=8, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
    fc2_fprop=PipelineOverlapCfg(num_sm=24, cga_size=2, num_splits=4, set_sm_margin=True),
)

userbuffers_fp8_b200_h18432_tp8_mbs1_seqlen4096 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=8, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=32, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=8, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
    fc2_fprop=RingExchangeOverlapCfg(num_sm=1, set_sm_margin=True),
)


@dataclass
class _CommOverlapConfig:
    # Tensor parallel communication overlap (experimental)
    tp_comm_overlap: bool = None
    tp_comm_overlap_cfg: dict = None
    tp_comm_bootstrap_backend: str = None
    # Pipeline parallel communication overlap
    overlap_p2p_comm: bool = None
    batch_p2p_comm: bool = None
    # Data parallel communication overlap
    overlap_grad_reduce: bool = None
    overlap_param_gather: bool = None
    overlap_param_gather_with_optimizer_step: bool = None
    align_param_gather: bool = None
    bucket_size: int = None
    # Pipeline bubble overlap
    defer_embedding_wgrad_compute: bool = None
    wgrad_deferral_limit: int = None
    # MOE expert parallel comm
    overlap_moe_expert_parallel_comm: bool = None
    delay_wgrad_compute: bool = None


@dataclass(kw_only=True)
class CommOverlapConfig:
    """Configuration for communication overlap optimizations in distributed training.

    This class manages tensor parallel, pipeline parallel, and data parallel
    communication overlap settings to improve training performance.
    """

    tp_comm_overlap: bool
    tp_comm_overlap_cfg: TransformerLayerTPOverlapCfg | None = None
    tp_comm_bootstrap_backend: str | None = None
    overlap_p2p_comm: bool | None = None
    batch_p2p_comm: bool | None = None
    overlap_grad_reduce: bool | None = None
    overlap_param_gather: bool | None = None
    overlap_param_gather_with_optimizer_step: bool | None = None
    align_param_gather: bool | None = None
    bucket_size: int | None = None
    defer_embedding_wgrad_compute: bool | None = None
    wgrad_deferral_limit: int | None = None
    data_parallel_size: int | None = None
    overlap_moe_expert_parallel_comm: bool | None = None
    delay_wgrad_compute: bool | None = None

    def finalize(self):
        # Don't recreate the user_comm_overlap_cfg if the post init is re-run
        if hasattr(self, "user_comm_overlap_cfg") and self.user_comm_overlap_cfg is not None:
            return

        self.user_comm_overlap_cfg = _CommOverlapConfig(
            tp_comm_overlap=self.tp_comm_overlap,
            tp_comm_overlap_cfg=self.tp_comm_overlap_cfg,
            tp_comm_bootstrap_backend=self.tp_comm_bootstrap_backend,
            overlap_p2p_comm=self.overlap_p2p_comm,
            batch_p2p_comm=self.batch_p2p_comm,
            overlap_grad_reduce=self.overlap_grad_reduce,
            overlap_param_gather=self.overlap_param_gather,
            overlap_param_gather_with_optimizer_step=self.overlap_param_gather_with_optimizer_step,
            align_param_gather=self.align_param_gather,
            bucket_size=self.bucket_size,
            defer_embedding_wgrad_compute=self.defer_embedding_wgrad_compute,
            wgrad_deferral_limit=self.wgrad_deferral_limit,
            overlap_moe_expert_parallel_comm=self.overlap_moe_expert_parallel_comm,
            delay_wgrad_compute=self.delay_wgrad_compute,
        )
        self.tp_comm_overlap_cfg = None
        self.tp_comm_bootstrap_backend = None

    def _get_model_comm_overlap_cfgs(
        self,
        model_cfg: GPTModelProvider | T5ModelProvider,
        ddp_config: DistributedDataParallelConfig,
    ) -> _CommOverlapConfig:
        comm_overlap_cfg = _CommOverlapConfig()

        vp_size = model_cfg.virtual_pipeline_model_parallel_size
        if vp_size is None:
            vp_size = 1

        # Optimizations disabled by default, can be overriden by user
        comm_overlap_cfg.tp_comm_overlap = False
        comm_overlap_cfg.tp_comm_overlap_cfg = None
        comm_overlap_cfg.tp_comm_bootstrap_backend = None
        comm_overlap_cfg.defer_embedding_wgrad_compute = False
        comm_overlap_cfg.wgrad_deferral_limit = -1
        comm_overlap_cfg.overlap_moe_expert_parallel_comm = False
        comm_overlap_cfg.delay_wgrad_compute = False

        # Check if TP overlap can be safely enabled
        if self.user_comm_overlap_cfg.tp_comm_overlap is True:
            if model_cfg.tensor_model_parallel_size < 2:
                logging.warning("Disabling tensor parallel communication overlap due to TP size < 2.")
                self.user_comm_overlap_cfg.tp_comm_overlap = False
            elif not model_cfg.sequence_parallel:
                logging.warning("Disabling tensor parallel communication overlap due to sequence_parallel=False.")
                self.user_comm_overlap_cfg.tp_comm_overlap = False
            elif not HAVE_TE:
                logging.warning("Disabling tensor parallel communication overlap due to TE not detected.")
                self.user_comm_overlap_cfg.tp_comm_overlap = False

        # PP overlap
        if model_cfg.pipeline_model_parallel_size > 1:
            if vp_size > 1:
                comm_overlap_cfg.overlap_p2p_comm = True
                comm_overlap_cfg.batch_p2p_comm = False
            else:
                comm_overlap_cfg.overlap_p2p_comm = False
                comm_overlap_cfg.batch_p2p_comm = True
        else:
            comm_overlap_cfg.overlap_p2p_comm = False
            comm_overlap_cfg.batch_p2p_comm = False

        # MOE expert parallel comm overlap
        assert hasattr(model_cfg, "overlap_moe_expert_parallel_comm"), (
            f"model_cfg: {model_cfg} does not have overlap_moe_expert_parallel_comm"
        )

        if self.user_comm_overlap_cfg.overlap_moe_expert_parallel_comm is True:
            assert model_cfg.expert_model_parallel_size > 1, (
                "overlap_moe_expert_parallel_comm is only supported when expert_model_parallel_size > 1"
            )
            assert model_cfg.num_moe_experts > 1, (
                f"overlap_moe_expert_parallel_comm is only supported when num_moe_experts > 1, \
                    but got {model_cfg.num_moe_experts}"
            )
            assert model_cfg.moe_token_dispatcher_type in ["alltoall", "flex"], (
                f"overlap_moe_expert_parallel_comm is only supported when moe_token_dispatcher_type == 'alltoall' or 'flex',\
                      but got {model_cfg.moe_token_dispatcher_type}"
            )
            assert model_cfg.bf16 or model_cfg.fp16, (
                "overlap_moe_expert_parallel_comm is only supported when using bf16 or fp16 models"
            )
            assert is_torch_min_version("2.6.0"), "A2A Overlap encounters hang issue with torch version < 2.6.0"
            if model_cfg.pipeline_model_parallel_size > 1:
                assert model_cfg.virtual_pipeline_model_parallel_size is not None, (
                    "If enabling EP A2A overlap, virtual_pipeline_model_parallel_size "
                    "must be specified when pipeline_model_parallel_size > 1"
                )
            assert model_cfg.recompute_granularity != "full", (
                "disable full recomputation when enabling overlap_moe_expert_parallel_comm"
            )
            assert model_cfg.recompute_method is None, (
                "disable recomputation method when enabling overlap_moe_expert_parallel_comm"
            )
            assert model_cfg.recompute_num_layers is None, (
                "recompute_num_layers must be None when enabling overlap_moe_expert_parallel_comm"
            )
            assert not model_cfg.moe_shared_expert_overlap, (
                "disable moe_shared_expert_overlap when enabling overlap_moe_expert_parallel_comm"
            )
            assert model_cfg.mtp_num_layers is None or model_cfg.mtp_num_layers == 1, (
                "MTP layernum only supports 1 when enabling overlap_moe_expert_parallel_comm."
            )

        if self.user_comm_overlap_cfg.delay_wgrad_compute is True:
            if ddp_config.overlap_grad_reduce or self.user_comm_overlap_cfg.overlap_grad_reduce:
                assert is_te_min_version("2.7.0"), (
                    f"TE version >= 2.7.0 is required for overlap_grad_reduce when using"
                    f"delay_wgrad_compute. Current TE version: {get_te_version()}"
                )
            if model_cfg.gradient_accumulation_fusion is True:
                assert is_te_min_version("2.7.0"), (
                    f"TE version >= 2.7.0 is required for gradient_accumulation_fusion when using"
                    f"delay_wgrad_compute. Current TE version: {get_te_version()}"
                )

            assert (
                model_cfg.overlap_moe_expert_parallel_comm
                or self.user_comm_overlap_cfg.overlap_moe_expert_parallel_comm
            ), "overlap_moe_expert_parallel_comm is required for delay_wgrad_compute"
            assert not model_cfg.moe_use_legacy_grouped_gemm, (
                "delay_wgrad_compute is not supported with legacy groupedgemm implementation"
            )

        comm_overlap_cfg = self._override_user_cfgs(comm_overlap_cfg)
        return comm_overlap_cfg

    def _get_optimizer_overlap_cfgs(self, model_cfg: GPTModelProvider | T5ModelProvider) -> _CommOverlapConfig:
        vp_size = model_cfg.virtual_pipeline_model_parallel_size
        if vp_size is None:
            vp_size = 1

        comm_overlap_cfg = _CommOverlapConfig()
        comm_overlap_cfg.bucket_size = None
        comm_overlap_cfg.overlap_grad_reduce = False
        comm_overlap_cfg.overlap_param_gather = False
        comm_overlap_cfg.overlap_param_gather_with_optimizer_step = False
        comm_overlap_cfg.align_param_gather = False

        if self.data_parallel_size > 1:
            comm_overlap_cfg.bucket_size = 128 * 1024 * 1024
            comm_overlap_cfg.overlap_grad_reduce = True
            comm_overlap_cfg.overlap_param_gather = True
            if model_cfg.pipeline_model_parallel_size > 1 and vp_size > 1:
                # Currently disabled due to an issue with checkpointing
                # comm_overlap_cfg.overlap_param_gather_with_optimizer_step = True
                comm_overlap_cfg.align_param_gather = True

        comm_overlap_cfg = self._override_user_cfgs(comm_overlap_cfg)
        return comm_overlap_cfg

    def _apply_cfgs(self, src_cfg, dest_cfg):
        # apply optimizations into dest_cfg
        for field in fields(src_cfg):
            if hasattr(dest_cfg, field.name):
                setattr(dest_cfg, field.name, getattr(src_cfg, field.name))

    def _override_user_cfgs(self, comm_overlap_cfg):
        # override default configs with any user provided configs
        if isinstance(self.user_comm_overlap_cfg, _CommOverlapConfig):
            for field in fields(self.user_comm_overlap_cfg):
                user_value = getattr(self.user_comm_overlap_cfg, field.name)
                if user_value is not None:
                    setattr(comm_overlap_cfg, field.name, user_value)

        return comm_overlap_cfg

    def setup(
        self,
        model_config: GPTModelProvider | T5ModelProvider,
        optimizer_config: OptimizerConfig,
        ddp_config: DistributedDataParallelConfig,
    ) -> None:
        """Set up communication overlap configurations for the model, optimizer, and DDP.

        Args:
            model_config: Model configuration containing parallelism settings
            optimizer_config: Optimizer configuration for gradient overlap settings
            ddp_config: Distributed data parallel configuration
        """
        comm_overlap_cfg = self._get_model_comm_overlap_cfgs(model_config, ddp_config)
        self._apply_cfgs(comm_overlap_cfg, model_config)
        if model_config.tp_comm_overlap:
            if comm_overlap_cfg.tp_comm_overlap_cfg is None:
                logging.warning(
                    "Tensor parallel overlap: No overlap config provided. "
                    "Initializing TP comm overlap with the default config."
                )
                model_config.tp_comm_overlap_cfg = None
            else:
                # ub_cfgs is a dataclass, however TE needs a dict, so convert here
                model_config.tp_comm_overlap_cfg = asdict(comm_overlap_cfg.tp_comm_overlap_cfg)
                # remove keys with None values from dictionary to match TE's expectations
                model_config.tp_comm_overlap_cfg = {
                    key: value for key, value in model_config.tp_comm_overlap_cfg.items() if value is not None
                }
            model_config.tp_comm_bootstrap_backend = comm_overlap_cfg.tp_comm_bootstrap_backend

        # Data parallel overlap is only available with the Megatron DDP and Distributed optimizer
        if (
            isinstance(optimizer_config, OptimizerConfig)
            and isinstance(ddp_config, DistributedDataParallelConfig)
            and ddp_config.use_distributed_optimizer
        ):
            comm_overlap_cfg = self._get_optimizer_overlap_cfgs(model_config)
            self._apply_cfgs(comm_overlap_cfg, optimizer_config)
            self._apply_cfgs(comm_overlap_cfg, ddp_config)
