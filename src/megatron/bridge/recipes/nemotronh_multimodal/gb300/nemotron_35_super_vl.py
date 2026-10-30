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

"""Nemotron 3.5 Super VL GB300 training recipes."""

from megatron.bridge.recipes.nemotronh_multimodal.gb200.nemotron_35_super_vl import (
    nemotron_35_super_vl_pretrain_64gpu_gb200_bf16_config,
)
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_with_mxfp8_mixed


_NCCL_EP_WIRE_FIELDS = ("moe_dispatch_fwd_dtype", "moe_combine_bwd_dtype")


def _require_nccl_ep_wire_fields(model_cfg: object) -> None:
    """Raise unless the model provider declares the NCCL EP wire-format fields.

    Assigning a field that Megatron-Core does not declare would not raise: the recipe would look tuned while the
    BF16 wire stays in effect.

    Args:
        model_cfg: The model provider the recipe is about to configure.

    Raises:
        ValueError: If a wire-format field is missing from the provider.
    """
    missing = [
        name for name in _NCCL_EP_WIRE_FIELDS if not (hasattr(type(model_cfg), name) or hasattr(model_cfg, name))
    ]
    if missing:
        raise ValueError(
            "nemotron_35_super_vl_pretrain_64gpu_gb300_fp8mx_config needs a Megatron-Core that declares "
            f"{', '.join(missing)} (the NCCL EP wire formats); the installed one does not."
        )


def nemotron_35_super_vl_pretrain_64gpu_gb300_fp8mx_config() -> ConfigContainer:
    """Return the 64-GB300 MXFP8 pretraining config for Super VL with NCCL EP.

    Starts from :func:`nemotron_35_super_vl_pretrain_64gpu_gb200_bf16_config` and keeps its data, objective, optimizer,
    schedule, batch shape (sequence length 8192, micro batch 1, global batch 512), natural routing, scoped CUDA graphs,
    full-precision optimizer state and FP32 gradient reduction. It changes the execution and the arithmetic precision
    (MXFP8 compute and wire) to the fastest of the eight configurations measured on 16 GB300 nodes (64 GPUs in one
    NVL72 NVLink domain, expert parallelism fixed at 64):

    * Expert parallelism only: tensor parallelism 1 and sequence parallelism off, so the 64 GPUs are data parallel for
      the dense layers and each holds 8 of the 512 experts. The language-model activations of a whole 8192-token micro
      batch then stay on one GPU. In the HybridEP runs of the screen, BF16 at this layout and a micro batch of 2 ran
      out of memory.
    * MXFP8 for the language model (``bf16_with_mxfp8_mixed``: parameters gathered in FP8, gradient buffer reused for
      the parameter all-gather). The RADIO vision tower is BF16 by provider configuration and the vision projector
      follows the language model's FP8 setting; neither ran in the measurement (text-only rows).
    * Transformer Engine's op fuser with the CuTe DSL fused grouped MLP, which needs compute capability 10.x and the
      CuTe DSL and cuDNN frontend packages of the container; without them Transformer Engine falls back silently.
      It also reads the gate ``NVTE_CUTEDSL_FUSED_GROUPED_MLP`` once, when it is imported.
      ``scripts/training/run_recipe.py``, which ``train.sh`` runs, sets it in time by re-executing itself; with any
      other entry point, export it before Python starts, or the unfused grouped GEMM runs.
    * NCCL EP instead of HybridEP, in eager mode (dropless, no expert-rank capacity factor), with MXFP8 on the wire for
      the two quantized exchanges (dispatch forward and combine backward). With NCCL EP the library pads every expert
      to its own alignment, so ``moe_router_padding_for_quantization`` only satisfies a configuration check.
      ``NCCL_EP_HT_EM_PULL_PUSH`` is required: by the library's staging-buffer sizing (confirmed at small scale, not
      run at expert parallelism 64), eager NCCL EP would reserve about 70 GiB of staging buffers plus about 1 GiB of
      routing state per MoE layer on every rank and not fit. Pull/push mode needs all 64 ranks inside one NVL72
      NVLink domain, and Transformer Engine's grouped quantization accepts at most 64 local experts, so keep the
      expert-parallel size at 8 or more. The wire formats need a Megatron-Core that declares
      ``moe_dispatch_fwd_dtype`` and ``moe_combine_bwd_dtype``; the recipe raises a ValueError otherwise.

    Measurements (16 GB300 nodes; text-only mock rows from ``scripts/training/run_recipe.py --dataset mock
    model.image_token_index=131000 --max_steps 60 scheduler.lr_decay_iters=60 checkpoint.save=null
    checkpoint.async_save=false``; natural routing from a randomly initialized router; median of steps 21-60, during
    which the step time still fell by about 5%):

    * One allocation ran eight configurations back to back. Expert parallelism only with MXFP8 and HybridEP took
      6,945 ms per step; adding the op fuser took 6,491 ms, NCCL EP with the BF16 wire 5,922 ms, and the MXFP8 wire
      5,868 ms (87.2 samples/s, 946.2 TFLOP/s per GPU). The wire's last step of about 1% is about twice the 0.6%
      difference between two repeats of the control but below the 2% bar set before the study, so its benefit is not
      established; leaving both wire fields at BF16 gives the 5,922 ms variant (937.6 TFLOP/s per GPU).
    * The GB200 BF16 recipe (tensor parallelism 2 with sequence parallelism) took 9,088 ms per step (611.0 TFLOP/s per
      GPU) on the same mock data in an earlier allocation; the expert-parallel-only MXFP8 layout re-measured in the
      later allocation was 1.1% slower than in the earlier one. The step from that baseline was not separated into its
      tensor-parallel and precision causes.
    * TFLOP/s per GPU is the logged model-FLOPs rate: 5,553 TFLOP-seconds divided by the step time at this shape.
    * Peak device memory on the busiest rank was 259 GiB of 276.6 GiB (NVML samples every 10 s), about 6% of
      headroom, with text-only rows and a random router.

    Not measured: image-bearing or padded real data (routing balance and memory depend on them), a trained router,
    checkpoint saving and loading with MXFP8 parameters (the recipe keeps the GB200 checkpoint settings, so set
    ``checkpoint.save`` explicitly), runs longer than 60 steps, the vision tower, and the TE CUDA-graph scope on this
    stack (inherited from the GB200 recipe, not ablated). The loss of the mock rows is trivially learnable, so it says
    nothing about convergence; MXFP8 compute and the quantized wire change the numerics, and this recipe has no
    convergence evidence on real data. The measurement ran on a build older than main with FlashAttention 2.

    The mock rows are sequential token ids and hit the ``<image>`` placeholder id of the checkpoint, hence
    ``model.image_token_index=131000`` in the command above; do not set it for real image data. For Energon data pass
    ``dataset.path=...``. On Slurm keep the 16 nodes inside one NVL72 block (for example ``segment=16``).

    ``NVTE_FLASH_ATTN_V4`` is 0 because the first backward pass of an image-bearing run failed with the
    FlashAttention-4 beta (4.0.0b24) while compiling its backward preprocessing kernel; the error pattern matches the
    vision tower's head dimension of 80, which was not isolated by an experiment. The measured text-only run therefore
    used FlashAttention 2 for the language-model attention layers. This makes FlashAttention 2 the only attention
    backend for the whole process, so the image needs a Blackwell-capable flash-attn 2.x; a launcher that exports
    ``NVTE_FLASH_ATTN_V4=1`` overrides the recipe. Remove it once the attention backend supports that head dimension.

    The recipe is an NCCL EP stack: do not switch single settings (dispatcher, tensor parallelism, op fuser) by
    command-line override without also setting the fields coupled to them.

    Returns:
        The Super-VL GB300 MXFP8 pretraining configuration.

    Raises:
        ValueError: If the installed Megatron-Core does not declare the NCCL EP wire-format fields.
    """
    cfg = nemotron_35_super_vl_pretrain_64gpu_gb200_bf16_config()
    _require_nccl_ep_wire_fields(cfg.model)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.sequence_parallel = False

    cfg.mixed_precision = bf16_with_mxfp8_mixed()
    cfg.model.moe_router_padding_for_quantization = True

    cfg.model.use_transformer_engine_op_fuser = True
    cfg.model.moe_use_grouped_tensor = True

    cfg.model.moe_flex_dispatcher_backend = "ncclep"
    cfg.model.moe_dispatch_fwd_dtype = "mxfp8"
    cfg.model.moe_combine_bwd_dtype = "mxfp8"

    # The complete environment, declared here rather than inherited from the GB200 recipe: the four HybridEP names
    # (NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN, NUM_OF_TOKENS_PER_CHUNK_COMBINE_API, NVLINK_DOMAIN_SIZE, USE_MNNVL)
    # are not read by the NCCL EP backend.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FLASH_ATTN_V4": 0,
        "NCCL_EP_HT_EM_PULL_PUSH": 1,
    }
    return cfg


__all__ = ["nemotron_35_super_vl_pretrain_64gpu_gb300_fp8mx_config"]
