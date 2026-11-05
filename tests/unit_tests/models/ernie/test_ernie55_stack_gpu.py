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

"""GPU tests of a whole toy ERNIE 5.5 model: fp32 streams, multimax, SD-MTP, recompute and checkpoints.

The model is :class:`Ernie55HybridModel` with the pattern ``"H-DE/DE"`` (an H layer, a dense layer,
a D layer, an MoE layer and one MTP depth whose ``DE`` body shares the decoder's last ``DE`` pair),
hyper-connections with four streams, Transformer Engine layers, bf16 parameters, a packed THD batch of
four documents and per-row loss. Every test needs one CUDA device, Transformer Engine,
``fast_hadamard_transform`` (the D indexer) and a Megatron-Core build with the DSv4
``_post_o_group_proj`` hook (dev f78809f6d or later). Cases with ``backend="cudnn"`` also need the fused
kernels of the D and H layers (FlashMLA and the cuDNN DSA kernels); ``backend="none"`` runs the
PyTorch reference attention, which for packed THD H layers exists on Megatron-Core dev only. Without
them the tests skip and name what is missing; with ``ERNIE55_GPU_STRICT=1`` they fail instead.

What is covered:

* ``test_toy_model_trains``: 5 iterations through ``get_forward_backward_func`` under
  ``Float16Module``, DDP (overlapped grad reduce) and the distributed optimizer, with hooks that
  assert fp32 decoder streams, bf16 layer inputs and bf16 MTP streams on every iteration. The LM and
  MTP losses are finite and logged, every parameter gets a gradient (except the documented zero-init
  cases on the first step), the multimax parameters carry gradients and form their own norm group
  (the logged grad norm excludes them), and the LM loss falls. After training, the MTP body still
  computes with the decoder's Transformer Engine tensors: for every shared decoder parameter
  (attention projections and norms, the indexer, the sink, VHA and output gate, both hyper-connection
  modules, the router, the shared expert, and each routed-expert projection across its experts),
  perturbing it changes the MTP stack's output for a fixed single-document input, and restoring it
  restores the output bitwise.
* ``test_full_recompute_training_matches_no_recompute``: two iterations with and without full
  recompute, both under DDP's overlapped grad reduce. With recompute, every layer's backward runs in
  its own autograd call, so each parameter the MTP body shares with the decoder gets one gradient
  accumulation per use and DDP's grad hook fires twice per micro-batch. The losses, every parameter's
  main grad (a relative bound, plus an absolute floor for scalars such as the mHC gates, whose gradients
  are rounding noise), the shared parameters' main grads taken together and the gradient norms must
  match the run without recompute. The MTP loss weight is 1 so that the MTP body's share of the shared
  gradients stands out, and the second iteration starts both runs from the same weights.
* ``test_full_recompute_replays_mappings_and_routing_bitwise``: with full recompute, the first
  forward runs under ``torch.no_grad()`` and the backward replays it with gradients. Hooks record the
  ERNIE mHC mappings of every hyper-connection module and the routing of every router in both passes;
  they must be bitwise equal. ``test_megatron_core_mhc_recompute_replay`` measures the same for
  Megatron-Core's own (compiled) mHC arithmetic and reports an expected failure when it differs,
  which is why ERNIE 5.5 runs the eager arithmetic.
* ``test_torch_dist_checkpoint_round_trip``: a ``torch_dist`` checkpoint of the model and the
  distributed optimizer (both sharding formats Bridge writes) restores every parameter, buffer and
  optimizer state bitwise, keeps the MTP body aliased to the decoder, writes each shared tensor once
  (no MTP-body parameter keys), and the restored model reproduces the per-token losses bitwise.

Run on one GPU; ``-rA`` lists skip reasons and the ``log_cli`` options print the measured metrics::

    ERNIE55_GPU_STRICT=1 uv run python -m torch.distributed.run --nproc-per-node 1 -m pytest -rA \\
        -o log_cli=true -o log_cli_level=INFO tests/unit_tests/models/ernie/test_ernie55_stack_gpu.py
"""

from __future__ import annotations

import functools
import importlib
import itertools
import logging
import math
import os
import re

import pytest
import torch


logger = logging.getLogger(__name__)

# ERNIE55_GPU_STRICT=1 turns every skip of this file into a failure (GPU acceptance runs).
_STRICT = os.environ.get("ERNIE55_GPU_STRICT") == "1"

pytestmark = [
    pytest.mark.unit,
    pytest.mark.run_only_on("GPU"),
    pytest.mark.skipif(
        not torch.cuda.is_available() and not _STRICT, reason="the ERNIE 5.5 stack GPU tests need a CUDA device"
    ),
]

_PATTERN = "H-DE/DE"
_MAIN_LAYERS = 4
_SHARED_DECODER_LAYERS = ("decoder.layers.2.", "decoder.layers.3.")
_DOC_LENS = (200, 150, 100, 62)  # 512 tokens; the two long documents give the H compressor blocks
_VOCAB = 512
_ITERATIONS = 5
_BUCKET_SIZE = 1_000_000
_STREAMS = 4
# The SD-MTP binding probe feeds one document of all the tokens: D-layer queries past topk + window
# candidates (positions >= 192 here) then depend on the indexer's selection, so its parameters count too.
_PROBE_DOC_LENS = (512,)
# Main-grad bound between full recompute and no recompute. Every parameter must satisfy
#     ||g_recompute - g_plain|| <= RTOL * max(||g_recompute||, ||g_plain||),
# a scalar parameter with ATOL * ||g_plain||_all added on the right (the norm over all main grads). RTOL:
# without recompute the two uses of a shared parameter are summed in bf16 before DDP adds them to the fp32
# main grad; with recompute each use is added separately, so the two differ by bf16 rounding of that sum (at
# most 2^-9 relative), plus the backward's atomics-order noise, which two identical runs without recompute
# show as well (up to 1.3e-2, in the H layer and the embedding on the PyTorch reference path). ATOL: a
# scalar's gradient is a single sum over every token, and the mHC gates' sums cancel down to rounding noise.
# Layer 0's streams are copies of the embedding, so its aggregate is a multiple of one stream, which the
# scale-invariant input norm undoes (alpha_pre gets almost no gradient), and dL/dH_res is constant over the
# input stream, which the Sinkhorn's column normalization cancels (alpha_res gets none in exact arithmetic).
# Two identical runs differ in the gates by up to 1e-5 of the all-parameter norm, in either direction.
_RECOMPUTE_GRAD_RTOL = 2e-2
_RECOMPUTE_SCALAR_GRAD_ATOL = 1e-4
# MTP loss weight in that test (Megatron-Core's default is 0.1). At 0.1 the MTP body's use of a shared tensor
# carries only about 1% of its main grad, so the bound would miss that use being lost or counted twice
# everywhere but in the D-layer indexer, whose loss is not scaled. At 1 it carries 5% to 17%.
_RECOMPUTE_MTP_LOSS_WEIGHT = 1.0
# A fused-path-compatible toy ERNIE 5.5: FlashMLA fixes the H width (d_qk = d_v = 512, 64 heads) and
# the D width (d_qk = 576 = 512 + 64, 64 heads); cuDNN fixes the indexer head width.
_GEOMETRY = dict(
    hidden_size=512,
    num_attention_heads=64,
    q_lora_rank=256,
    v_head_dim=512,
    qk_pos_emb_head_dim=64,
    output_projection_groups=8,
    output_projection_lora_rank=256,
    csa_window_size=128,
    csa_compress_rotary_base=160000.0,
    ernie_vha_rank=4,
    ernie_d_num_attention_heads=64,
    ernie_d_q_lora_rank=256,
    ernie_d_kv_lora_rank=512,
    ernie_d_qk_head_dim=64,
    ernie_d_qk_pos_emb_head_dim=64,
    ernie_d_v_head_dim=32,
    ernie_d_rotary_base=10000.0,
    dsa_indexer_n_heads=32,
    dsa_indexer_head_dim=128,
    dsa_indexer_topk=64,
    ffn_hidden_size=1024,
    num_moe_experts=8,
    moe_router_topk=2,
    moe_ffn_hidden_size=256,
    moe_shared_expert_intermediate_size=256,
)
_COMMON = dict(
    normalization="RMSNorm",
    layernorm_epsilon=1e-6,
    attention_dropout=0.0,
    hidden_dropout=0.0,
    add_bias_linear=False,
    experimental_attention_variant="dsv4_hybrid",
    apply_rope_fusion=False,
    bias_activation_fusion=False,
    gradient_accumulation_fusion=False,
    bf16=True,
    params_dtype=torch.bfloat16,
    init_method_std=0.02,
    dsa_indexer_loss_coeff=0.01,
    dsa_indexer_use_sparse_loss=True,
    dsa_indexer_k_norm_fp32=True,
    gated_linear_unit=True,
    moe_grouped_gemm=True,
    moe_router_score_function="sigmoid",
    moe_router_enable_expert_bias=True,
    moe_router_bias_update_rate=1e-4,
    moe_router_load_balancing_type="none",
    moe_aux_loss_coeff=0.0,
    moe_router_dtype="fp32",
    moe_token_dispatcher_type="alltoall",
    enable_mhc_connections=True,
    mhc_num_residual_streams=_STREAMS,
    mhc_sinkhorn_iterations=20,
    mtp_num_layers=1,
    ernie_mtp_shared_last_layer=True,
)
# Parameters whose gradient is exactly zero on the first step: VHA's U while V is at its zero init, and
# the multimax ranges while the scales are at their zero init (d/dr is proportional to t).
_ZERO_GRAD_AT_INIT = ("vha_postmix_u", "multimax_ranges")
_CUDNN_DSA_WRAPPERS = (
    "indexer_forward_wrapper",
    "indexer_top_k_wrapper",
    "indexer_backward_wrapper",
    "sparse_attn_score_recompute_wrapper",
    "sparse_attention_backward_wrapper",
)


# ---------------------------------------------------------------------------------------------
# Environment probes and fixtures
# ---------------------------------------------------------------------------------------------


def _device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _skip(reason: str) -> None:
    """Skip the test, or fail it under ``ERNIE55_GPU_STRICT=1``, where a skip would hide a missing kernel."""
    if _STRICT:
        pytest.fail(f"{reason} (ERNIE55_GPU_STRICT=1 turns this skip into a failure)", pytrace=False)
    pytest.skip(reason)


@functools.lru_cache(maxsize=None)
def _missing_reference_dependencies() -> tuple[str, ...]:
    missing = []
    if not torch.cuda.is_available():
        missing.append("a CUDA device")
    for module_name, label in (
        ("transformer_engine.pytorch", "Transformer Engine"),
        ("fast_hadamard_transform", "fast_hadamard_transform"),
    ):
        try:
            importlib.import_module(module_name)
        except Exception as error:
            missing.append(f"{label} ({type(error).__name__}: {error})")
    return tuple(missing)


def _flash_mla_launch_error(d_qk: int, d_v: int, topk: int) -> str | None:
    """Launch FlashMLA once in a layer's format: an import can succeed on a GPU the cubins were not built for."""
    from megatron.core.transformer.experimental_attention_variant.csa_utils import fused_sparse_attention

    generator = torch.Generator(device=_device()).manual_seed(0)
    query = torch.randn(4, 64, d_qk, device=_device(), generator=generator).to(torch.bfloat16)
    kv = torch.randn(8, d_qk, device=_device(), generator=generator).to(torch.bfloat16)
    indices = torch.full((4, topk), -1, dtype=torch.int32, device=_device())
    indices[:, 0] = 0
    try:
        fused_sparse_attention._csa_fwd_flash_mla(
            query, kv, indices, d_qk**-0.5, d_v=d_v, attn_sink=torch.zeros(64, device=_device())
        )
        torch.cuda.synchronize()
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None


@functools.lru_cache(maxsize=None)
def _missing_fused_dependencies() -> tuple[str, ...]:
    missing = list(_missing_reference_dependencies())
    if missing:
        return tuple(missing)
    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        missing.append(f"an SM90+ GPU (found sm_{major}{minor})")
    try:
        from flash_mla import flash_mla_sparse_fwd  # noqa: F401
    except Exception as error:
        missing.append(f"FlashMLA flash_mla_sparse_fwd ({type(error).__name__}: {error})")
    try:
        from cudnn import DSA
    except Exception as error:
        missing.append(f"cudnn-frontend DSA ({type(error).__name__}: {error})")
    else:
        absent = [name for name in _CUDNN_DSA_WRAPPERS if not hasattr(DSA, name)]
        if absent:
            missing.append(f"cudnn.DSA wrappers {', '.join(absent)}")
    try:
        from megatron.core.transformer.experimental_attention_variant.csa_utils import thd_layout_kernels
    except ImportError:
        pass
    else:
        if not getattr(thd_layout_kernels, "_CUTE_AVAILABLE", True):
            missing.append("CuTe DSL (nvidia-cutlass-dsl) for the THD index kernels")
    if not missing:
        for label, d_qk, d_v, topk in (("H", 512, 512, 128), ("D", 576, 512, 512)):
            error = _flash_mla_launch_error(d_qk, d_v, topk)
            if error is not None:
                missing.append(f"a FlashMLA kernel for the {label} layer on this GPU ({error})")
    return tuple(missing)


@pytest.fixture(scope="module")
def distributed(tmp_path_factory):
    """Single-rank process group and Megatron-Core model-parallel state (TP = PP = CP = EP = 1)."""
    missing = _missing_reference_dependencies()
    if missing:
        _skip("the ERNIE 5.5 stack GPU tests need " + "; ".join(missing))
    from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
        DSv4HybridAttention,
    )

    if not callable(getattr(DSv4HybridAttention, "_post_o_group_proj", None)):
        _skip("ERNIE 5.5 H layers need Megatron-Core's DSv4 _post_o_group_proj hook (dev f78809f6d+)")
    import torch.distributed as dist
    from megatron.core import parallel_state

    created_group = False
    if dist.is_initialized():
        if dist.get_world_size() != 1:
            _skip("the ERNIE 5.5 stack GPU tests run on a single rank")
    else:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        store = tmp_path_factory.mktemp("ernie55_stack_gpu_dist") / "store"
        dist.init_process_group(backend="nccl", init_method=f"file://{store}", rank=0, world_size=1)
        created_group = True
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
    )
    yield
    parallel_state.destroy_model_parallel()
    if created_group:
        dist.destroy_process_group()


def _require_backend(backend: str) -> None:
    if backend == "cudnn":
        missing = _missing_fused_dependencies()
        if missing:
            _skip("the fused ERNIE 5.5 attention needs " + "; ".join(missing))
    else:
        from megatron.core.transformer.experimental_attention_variant import csa

        # Megatron-Core main runs packed THD H layers only on the fused path; dev also has a PyTorch reference.
        if hasattr(csa.CompressedSparseAttention, "_forward_packed"):
            _skip("this Megatron-Core has no PyTorch reference path for packed THD H layers")


@pytest.fixture(autouse=True)
def _fresh_trackers():
    """Empty the process-wide DSA, MTP and MoE loss trackers around every test."""
    from megatron.core.transformer.experimental_attention_variant import dsa
    from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

    def reset() -> None:
        dsa.DSAIndexerLossLoggingHelper.tracker.clear()
        dsa.DSAIndexerLossAutoScaler.main_loss_backward_scale = None
        MTPLossLoggingHelper.tracker.clear()

    reset()
    yield
    reset()


# ---------------------------------------------------------------------------------------------
# Builders and inputs
# ---------------------------------------------------------------------------------------------


def _config(backend: str, **overrides):
    import torch.nn.functional as F

    from megatron.bridge.models.ernie.ernie55_config import Ernie55TransformerConfig

    main_pattern, mtp_pattern = _PATTERN.split("/")
    symbols = main_pattern + mtp_pattern
    values = dict(
        num_layers=len(main_pattern),
        csa_compress_ratios=[128 if symbol == "H" else 0 for symbol in symbols],
        dsa_kernel_backend=backend,
        activation_func=F.silu,
        **_COMMON,
        **_GEOMETRY,
    )
    values.update(overrides)
    config = Ernie55TransformerConfig(**values)
    config.finalize()
    return config


def _build_model(config, *, seed: int):
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    from megatron.bridge.models.ernie.modeling_ernie55 import Ernie55HybridModel, ernie55_hybrid_stack_spec

    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    model = Ernie55HybridModel(
        config=config,
        hybrid_stack_spec=ernie55_hybrid_stack_spec(config),
        vocab_size=_VOCAB,
        max_sequence_length=sum(_DOC_LENS),
        hybrid_layer_pattern=_PATTERN,
        position_embedding_type="none",
    )
    return model.to(_device())


def _packed_seq_params(doc_lens: tuple[int, ...] = _DOC_LENS):
    from megatron.core.packed_seq_params import PackedSeqParams

    cu_seqlens = torch.tensor([0, *itertools.accumulate(doc_lens)], dtype=torch.int32, device=_device())
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max(doc_lens),
        max_seqlen_kv=max(doc_lens),
    )


def _batch(*, seed: int = 0) -> dict:
    """A packed THD micro-batch: next-token targets inside each document, none across documents."""
    generator = torch.Generator().manual_seed(seed)
    num_tokens = sum(_DOC_LENS)
    tokens = torch.randint(0, _VOCAB, (1, num_tokens), generator=generator)
    cu_seqlens = torch.tensor([0, *itertools.accumulate(_DOC_LENS)])
    loss_mask = torch.ones(1, num_tokens)
    loss_mask[0, cu_seqlens[1:] - 1] = 0.0
    position_ids = torch.cat([torch.arange(length) for length in _DOC_LENS]).unsqueeze(0)
    batch = dict(tokens=tokens, labels=tokens.roll(-1, dims=1), loss_mask=loss_mask, position_ids=position_ids)
    batch = {name: tensor.to(_device()) for name, tensor in batch.items()}
    batch["packed_seq_params"] = _packed_seq_params()
    return batch


def _forward(model, batch):
    return model(
        input_ids=batch["tokens"],
        position_ids=batch["position_ids"],
        attention_mask=None,
        labels=batch["labels"],
        loss_mask=batch["loss_mask"],
        packed_seq_params=batch["packed_seq_params"],
    )


def _forward_step(data_iterator, model):
    batch = next(data_iterator)
    losses = _forward(model, batch)
    loss_mask = batch["loss_mask"]

    def loss_func(output_tensor):
        loss = (output_tensor.float() * loss_mask).sum() / loss_mask.sum()
        return loss, {"lm loss": loss.detach()}

    return losses, loss_func


def _unwrap(model):
    """DDP -> Float16Module -> Ernie55HybridModel."""
    return model.module.module


def _training_setup(config, *, seed: int, lr: float = 1e-3):
    """The toy model under Float16Module and DDP (overlapped grad reduce), with the distributed optimizer."""
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig, finalize_model_grads
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.transformer.module import Float16Module

    model = Float16Module(config, _build_model(config, seed=seed))
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=True,
        use_distributed_optimizer=True,
        bucket_size=_BUCKET_SIZE,
    )
    model = DistributedDataParallel(config, ddp_config, model)
    optimizer_config = OptimizerConfig(
        optimizer="adam",
        lr=lr,
        min_lr=lr,
        weight_decay=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=True,
        clip_grad=1.0,
    )
    optimizer = get_megatron_optimizer(optimizer_config, [model])
    config.no_sync_func = model.no_sync
    config.finalize_model_grads_func = finalize_model_grads
    config.grad_scale_func = optimizer.scale_loss
    return model, optimizer


def _run_schedule(model, batches) -> float:
    from megatron.core.pipeline_parallel import get_forward_backward_func

    model.zero_grad_buffer()
    outputs = get_forward_backward_func()(
        forward_step_func=_forward_step,
        data_iterator=iter(batches),
        model=[model],
        num_microbatches=len(batches),
        seq_length=batches[0]["tokens"].size(1),
        micro_batch_size=1,
        forward_only=False,
    )
    return torch.stack([item["lm loss"] for item in outputs]).mean().item()


def _pop_mtp_loss() -> torch.Tensor:
    """MTP loss logged since the last call (summed over micro-batches), one value per depth."""
    from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

    tracker = MTPLossLoggingHelper.tracker
    # Megatron-Core dev accumulates "loss_sums" (save_loss_to_tracker); main "loss_values".
    values = tracker.get("loss_sums", tracker.get("loss_values"))
    loss = torch.zeros(0) if values is None else values.detach().float().cpu().clone()
    MTPLossLoggingHelper.tracker.clear()
    return loss


def _report(label: str, **metrics) -> None:
    """Log one line of measured metrics (shown with ``-o log_cli=true -o log_cli_level=INFO``)."""
    text = ", ".join(
        f"{name}={value:.6g}" if isinstance(value, float) else f"{name}={value}" for name, value in metrics.items()
    )
    logger.info("[ernie55-stack-gpu] %s: %s", label, text)


class _StreamDtypes:
    """Hooks asserting the stream contract on every forward: fp32 decoder streams, bf16 layers and MTP streams."""

    def __init__(self, model) -> None:
        from megatron.core.models.hybrid.layers.hybrid_hyper_connection import HyperConnectionHybridLayer
        from megatron.core.transformer.moe.moe_layer import MoELayer

        from megatron.bridge.models.ernie.modeling_ernie55 import is_downcasting_norm

        self.violations: list[str] = []
        self.calls = 0
        for name, module in model.named_modules():
            if isinstance(module, MoELayer):
                # The router, routed experts and shared expert see the bf16 pre-norm output.
                for part in ("router", "experts", "shared_experts"):
                    getattr(module, part).register_forward_pre_hook(
                        self._input_check(f"{name}.{part}", torch.bfloat16), with_kwargs=True
                    )
            if isinstance(module, HyperConnectionHybridLayer):
                stream_dtype = torch.float32 if name.startswith("decoder.") else torch.bfloat16
                module.register_forward_pre_hook(self._input_check(name, stream_dtype), with_kwargs=True)
                module.register_forward_hook(self._output_check(name, stream_dtype))
            elif is_downcasting_norm(module):
                # The pre-norm sees the fp32 layer input (or the fp32 decoder streams) and hands bf16 on.
                module.register_forward_pre_hook(
                    self._input_check(name, torch.float32), with_kwargs=True, prepend=True
                )
                module.register_forward_hook(self._output_check(name, torch.bfloat16))

    def _input_check(self, name, dtype):
        def hook(module, args, kwargs):
            hidden = args[0] if args else kwargs.get("hidden_states", next(iter(kwargs.values())))
            self.calls += 1
            if hidden.dtype != dtype:
                self.violations.append(f"{name} input {hidden.dtype}, expected {dtype}")

        return hook

    def _output_check(self, name, dtype):
        def hook(module, args, output):
            output = output[0] if isinstance(output, tuple) else output
            if output.dtype != dtype:
                self.violations.append(f"{name} output {output.dtype}, expected {dtype}")

        return hook


def _binding_probes(model, aliases: dict[str, str]) -> list[tuple[str, list[str]]]:
    """``(label, decoder parameter names)``: one probe per decoder parameter the MTP body shares.

    The routed experts are the exception: each projection is probed as a whole, across the experts
    (``local_experts.*`` modules or Transformer Engine's grouped ``weight<k>``), so an expert that
    receives no tokens for the probe input cannot fail it.
    """
    probes: dict[str, list[str]] = {}
    for name in sorted(set(aliases.values())):
        owner = model.get_submodule(name.rpartition(".")[0])
        key = re.sub(r"local_experts\.\d+", "local_experts.*", name)
        if ".experts." in key:
            key = re.sub(r"(weight|bias)\d+$", r"\1*", key)
        probes.setdefault(f"{key} ({type(owner).__name__})", []).append(name)
    return list(probes.items())


def _check_mtp_body_binding(model, aliases: dict[str, str], label: str) -> None:
    """Every shared decoder parameter drives the MTP body: perturb it, compare, restore it bitwise."""
    body = model.mtp.layers[0].mtp_model_layer
    generator = torch.Generator(device=_device()).manual_seed(3)
    fixed_input = torch.randn(
        sum(_PROBE_DOC_LENS), 1, _STREAMS * _GEOMETRY["hidden_size"], device=_device(), generator=generator
    ).to(torch.bfloat16)
    packed_seq_params = _packed_seq_params(_PROBE_DOC_LENS)

    def body_output() -> torch.Tensor:
        with torch.no_grad():
            return body(hidden_states=fixed_input, attention_mask=None, packed_seq_params=packed_seq_params).clone()

    was_training = model.training
    model.eval()
    try:
        before = body_output()
        probes = _binding_probes(model, aliases)
        unbound, not_restored, changes = [], [], []
        for probe, names in probes:
            params = [model.get_parameter(name) for name in names]
            saved = [param.detach().clone() for param in params]
            with torch.no_grad():
                for param in params:
                    noise = torch.randn(param.shape, device=param.device, generator=generator)
                    param.add_(0.05 * noise.to(param.dtype))
            perturbed = body_output()
            with torch.no_grad():
                for param, value in zip(params, saved):
                    param.copy_(value)
            restored = body_output()
            changes.append(float((perturbed.float() - before.float()).abs().max()))
            if torch.equal(perturbed, before):
                unbound.append(probe)
            if not torch.equal(restored, before):
                not_restored.append(probe)
    finally:
        model.train(was_training)
    weakest = min(range(len(changes)), key=changes.__getitem__)
    _report(
        f"sd_mtp_binding[{label}]", probes=len(probes), min_max_change=changes[weakest], weakest=probes[weakest][0]
    )
    for probe, _ in probes:
        logger.info("[ernie55-stack-gpu] SD-MTP binding probe: %s", probe)
    covered = " ".join(probe for probe, _ in probes)
    for part in (".indexer.", ".router.", ".experts.", ".shared_experts.", ".hyper_connection.", "layernorm"):
        assert part in covered, f"no binding probe covers {part!r}: {covered}"
    assert not unbound, f"the MTP body did not see these decoder parameters change: {unbound}"
    assert not not_restored, f"restoring these decoder parameters did not restore the MTP body: {not_restored}"


# ---------------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["none", "cudnn"])
def test_toy_model_trains(distributed, backend):
    """Five iterations: fp32 streams, MTP loss, multimax in its own norm group, SD-MTP intact after TE builds."""
    from megatron.bridge.models.ernie.modeling_ernie55 import (
        MULTIMAX_GRAD_NORM_GROUP,
        MULTIMAX_PARAMETER_NAMES,
        check_ernie55_mtp_aliases,
    )

    _require_backend(backend)
    batches = [_batch(seed=0), _batch(seed=1)]
    model, optimizer = _training_setup(_config(backend), seed=7)
    inner = _unwrap(model)
    streams = _StreamDtypes(inner)
    aliases = inner.ernie55_mtp_aliases
    assert aliases and all(source.startswith(_SHARED_DECODER_LAYERS) for source in aliases.values())
    # Float16Module and DDP kept every alias the decoder's own Parameter.
    assert check_ernie55_mtp_aliases(model) == len(aliases)
    names = [name for name, _ in inner.named_parameters()]
    assert not [name for name in names if "mtp_model_layer" in name]
    multimax_names = [f"output_layer.{name}" for name in MULTIMAX_PARAMETER_NAMES]

    lm_losses, mtp_losses = [], []
    for iteration in range(_ITERATIONS):
        optimizer.zero_grad()
        lm_loss = _run_schedule(model, batches)
        mtp_loss = _pop_mtp_loss()
        grads = {name: param.main_grad.detach().float().clone() for name, param in inner.named_parameters()}
        assert not streams.violations, streams.violations[:10]
        assert math.isfinite(lm_loss), f"iteration {iteration}: LM loss {lm_loss}"
        assert mtp_loss.numel() == 1 and bool(torch.isfinite(mtp_loss).all()) and mtp_loss.item() > 0, mtp_loss
        assert all(bool(torch.isfinite(grad).all()) for grad in grads.values()), f"iteration {iteration}"
        without_grad = sorted(name for name, grad in grads.items() if not torch.any(grad != 0))
        allowed = [name for name in without_grad if name.endswith(_ZERO_GRAD_AT_INIT)] if iteration == 0 else []
        assert without_grad == allowed, f"iteration {iteration}: no gradient for {without_grad}"
        assert torch.any(grads["output_layer.multimax_ts"] != 0)

        # The logged grad norm leaves the multimax group out (and, as Megatron-Core does, shared copies).
        main_norm = math.sqrt(
            sum(
                float(grad.double().square().sum())
                for n, grad in grads.items()
                if n not in multimax_names and not getattr(inner.get_parameter(n), "shared", False)
            )
        )
        multimax_norm = math.sqrt(sum(float(grads[n].double().square().sum()) for n in multimax_names))
        update_successful, grad_norm, _ = optimizer.step()
        assert update_successful, f"iteration {iteration}: the optimizer step was skipped"
        # Megatron-Core returns the norm as a 1-element CUDA tensor when Transformer Engine's
        # multi_tensor_scale_tensor is available (get_grad_norm_fp32), and as a float otherwise.
        grad_norm = float(grad_norm)
        assert grad_norm == pytest.approx(main_norm, rel=1e-3), (grad_norm, main_norm, multimax_norm)
        group_norms = getattr(optimizer, "grad_norms_by_group", {})
        assert float(group_norms[MULTIMAX_GRAD_NORM_GROUP]) == pytest.approx(multimax_norm, rel=1e-3, abs=1e-12)
        lm_losses.append(lm_loss)
        mtp_losses.append(mtp_loss.item())
        _report(
            f"toy_training[{backend}] iteration {iteration}",
            lm=lm_loss,
            mtp=mtp_loss.item(),
            grad_norm=float(grad_norm),
            multimax_norm=multimax_norm,
        )
    assert streams.calls > 0
    assert lm_losses[-1] < lm_losses[0], lm_losses
    for name in MULTIMAX_PARAMETER_NAMES:
        assert torch.any(inner.output_layer.get_parameter(name) != 0), f"{name} did not train"

    # SD-MTP survives Transformer Engine's modules and training: the MTP body computes with the decoder's tensors.
    assert check_ernie55_mtp_aliases(model) == len(aliases)
    _check_mtp_body_binding(inner, aliases, backend)


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    scale = max(float(actual.double().norm()), float(expected.double().norm()))
    return 0.0 if scale == 0.0 else float((actual.double() - expected.double()).norm()) / scale


def _main_grad_bound_ratios(actual: dict, expected: dict) -> dict[str, float]:
    """Per parameter, ``||actual - expected||`` over its bound (see ``_RECOMPUTE_GRAD_RTOL``); at most 1 passes."""
    total = math.sqrt(sum(float(grad.double().square().sum()) for grad in expected.values()))
    ratios = {}
    for name, grad in expected.items():
        grad, other = grad.double(), actual[name].double()
        bound = _RECOMPUTE_GRAD_RTOL * max(float(other.norm()), float(grad.norm()))
        if grad.numel() == 1:
            bound += _RECOMPUTE_SCALAR_GRAD_ATOL * total
        error = float((other - grad).norm())
        ratios[name] = error / bound if bound else (0.0 if error == 0.0 else math.inf)
    return ratios


def _copy_weights(source, target) -> None:
    """Give ``target`` the parameters and buffers (the routers' expert biases among them) of ``source``."""
    with torch.no_grad():
        for name, param in source.named_parameters():
            target.get_parameter(name).copy_(param)
        for name, buffer in source.named_buffers():
            target.get_buffer(name).copy_(buffer)


def _recompute_config(backend: str, **overrides):
    return _config(
        backend, recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1, **overrides
    )


@pytest.mark.parametrize("backend", ["none", "cudnn"])
def test_full_recompute_training_matches_no_recompute(distributed, backend):
    """Two iterations with and without full recompute: losses, main grads and gradient norms agree.

    Both runs use Float16Module, DDP with the overlapped grad reduce and the distributed optimizer, and
    start from the same weights. Megatron-Core's DDP checks its grad-ready counts from the second
    iteration on, so the recompute run's double grad-hook firing for shared parameters is exercised
    against those checks too. Every parameter's main grad must agree within the bound described at
    ``_RECOMPUTE_GRAD_RTOL``, and the main grads of the parameters the MTP body shares, taken together,
    within ``_RECOMPUTE_GRAD_RTOL``; ``_RECOMPUTE_MTP_LOSS_WEIGHT`` makes a lost or doubled MTP use
    exceed both. Before the second iteration the recompute run takes the plain run's weights and
    buffers: Adam's first step moves every element whose gradient sign is rounding noise by about lr
    either way, so even two runs without recompute drift apart.
    """
    from megatron.bridge.models.ernie.modeling_ernie55 import MULTIMAX_GRAD_NORM_GROUP, check_ernie55_mtp_aliases

    _require_backend(backend)
    batches = [_batch(seed=0), _batch(seed=1)]
    weight = dict(mtp_loss_scaling_factor=_RECOMPUTE_MTP_LOSS_WEIGHT)
    runs = {
        "plain": _training_setup(_config(backend, **weight), seed=7),
        "recompute": _training_setup(_recompute_config(backend, **weight), seed=7),
    }
    plain = _unwrap(runs["plain"][0])
    recompute = _unwrap(runs["recompute"][0])
    assert recompute.config.recompute_granularity == "full" and plain.config.recompute_granularity is None
    for name, param in plain.named_parameters():
        assert torch.equal(recompute.get_parameter(name), param), f"{name}: the two runs start from different weights"
    shared = sorted(set(plain.ernie55_mtp_aliases.values()))
    assert shared and check_ernie55_mtp_aliases(recompute) == len(recompute.ernie55_mtp_aliases)

    for iteration in range(2):
        if iteration:
            _copy_weights(plain, recompute)
        results = {}
        for label, (model, optimizer) in runs.items():
            optimizer.zero_grad()
            lm_loss = _run_schedule(model, batches)
            mtp_loss = _pop_mtp_loss()
            grads = {name: param.main_grad.detach().clone() for name, param in _unwrap(model).named_parameters()}
            update_successful, grad_norm, _ = optimizer.step()
            assert update_successful, f"{label} iteration {iteration}: the optimizer step was skipped"
            multimax_norm = float(getattr(optimizer, "grad_norms_by_group", {})[MULTIMAX_GRAD_NORM_GROUP])
            results[label] = (lm_loss, mtp_loss, grads, float(grad_norm), multimax_norm)
        lm, mtp, grads, grad_norm, multimax_norm = results["plain"]
        lm_r, mtp_r, grads_r, grad_norm_r, multimax_norm_r = results["recompute"]
        errors = {name: _relative_l2(grads_r[name], grad) for name, grad in grads.items()}
        ratios = _main_grad_bound_ratios(grads_r, grads)
        worst = max(ratios, key=ratios.get)
        shared_error = _relative_l2(
            torch.cat([grads_r[name].flatten() for name in shared]),
            torch.cat([grads[name].flatten() for name in shared]),
        )
        _report(
            f"recompute_training[{backend}] iteration {iteration}",
            lm=lm,
            lm_recompute=lm_r,
            mtp=mtp.item(),
            mtp_recompute=mtp_r.item(),
            grad_norm=grad_norm,
            grad_norm_recompute=grad_norm_r,
            max_rel_l2_shared=max(errors[name] for name in shared),
            max_rel_l2_other=max(error for name, error in errors.items() if name not in shared),
            rel_l2_shared_together=shared_error,
            max_bound_ratio=ratios[worst],
            worst=worst,
        )
        assert lm_r == pytest.approx(lm, rel=1e-4), (iteration, lm, lm_r)
        assert mtp_r.numel() == 1 and mtp_r.item() == pytest.approx(mtp.item(), rel=1e-4), (iteration, mtp, mtp_r)
        assert ratios[worst] <= 1.0, (
            f"iteration {iteration}: {worst} main grad differs by {errors[worst]:.4g} (relative L2), "
            f"{ratios[worst]:.3g} times its bound"
        )
        assert shared_error <= _RECOMPUTE_GRAD_RTOL, (iteration, shared_error)
        assert grad_norm_r == pytest.approx(grad_norm, rel=_RECOMPUTE_GRAD_RTOL), (iteration, grad_norm, grad_norm_r)
        assert multimax_norm_r == pytest.approx(multimax_norm, rel=_RECOMPUTE_GRAD_RTOL, abs=1e-12)
    assert check_ernie55_mtp_aliases(recompute) == len(recompute.ernie55_mtp_aliases)


# ---------------------------------------------------------------------------------------------
# Full recompute: the replay must reproduce the first pass
# ---------------------------------------------------------------------------------------------


def _record_replay(monkeypatch, model, batch, *, mhc_class) -> dict:
    """Run one forward and backward; record every mHC mapping and routing, keyed by module and pass."""
    from megatron.core.transformer.moe.router import TopKRouter

    records: dict[tuple[str, int], list[tuple[bool, list[torch.Tensor]]]] = {}
    original = mhc_class.compute_mappings

    def compute_mappings(self, x):
        mappings = original(self, x)
        records.setdefault(("mhc", id(self)), []).append(
            (torch.is_grad_enabled(), [t.detach().clone() for t in mappings])
        )
        return mappings

    monkeypatch.setattr(mhc_class, "compute_mappings", compute_mappings)

    def router_hook(module, args, output):
        tensors = [
            t.detach().clone() for t in (output if isinstance(output, tuple) else (output,)) if torch.is_tensor(t)
        ]
        records.setdefault(("router", id(module)), []).append((torch.is_grad_enabled(), tensors))

    routers = [module for module in model.modules() if isinstance(module, TopKRouter)]
    handles = [router.register_forward_hook(router_hook) for router in routers]
    try:
        losses = _forward(model, batch)
        loss_mask = batch["loss_mask"]
        ((losses.float() * loss_mask).sum() / loss_mask.sum()).backward()
    finally:
        for handle in handles:
            handle.remove()
    return records


def _replay_mismatches(records: dict) -> tuple[int, list[str]]:
    """``(number of recorded modules, descriptions of modules whose replay differs from the first pass)``."""
    mismatches = []
    for (kind, module_id), entries in records.items():
        first = [tensors for grad_enabled, tensors in entries if not grad_enabled]
        replay = [tensors for grad_enabled, tensors in entries if grad_enabled]
        assert len(first) == 1 and len(replay) == 1, (kind, module_id, [grad for grad, _ in entries])
        for index, (a, b) in enumerate(zip(first[0], replay[0], strict=True)):
            if not torch.equal(a, b):
                differing = (a != b).sum().item()
                mismatches.append(f"{kind}@{module_id} output {index}: {differing} of {a.numel()} elements differ")
    return len(records), mismatches


@pytest.mark.parametrize("backend", ["none", "cudnn"])
def test_full_recompute_replays_mappings_and_routing_bitwise(distributed, monkeypatch, backend):
    from megatron.bridge.models.ernie.modeling_ernie55 import Ernie55HyperConnectionModule

    _require_backend(backend)
    model = _build_model(_recompute_config(backend), seed=21).train()
    records = _record_replay(monkeypatch, model, _batch(), mhc_class=Ernie55HyperConnectionModule)
    kinds = [kind for kind, _ in records]
    # Six hyper-connection modules (four decoder layers, two MTP layers) and two routers (decoder and MTP).
    assert kinds.count("mhc") == 6 and kinds.count("router") == 2, kinds
    count, mismatches = _replay_mismatches(records)
    _report(f"recompute_replay[{backend}]", modules=count, mismatches=len(mismatches))
    assert not mismatches, mismatches


@pytest.mark.parametrize("backend", ["none"])
def test_megatron_core_mhc_recompute_replay(distributed, monkeypatch, backend):
    """Measurement: does Megatron-Core's own (compiled) mHC arithmetic replay bitwise under full recompute?

    The model is the same toy model with its hyper-connection modules left as Megatron-Core built them.
    A difference is reported as an expected failure: it is the reason ERNIE 5.5 runs the eager arithmetic.
    """
    from megatron.core.transformer.hyper_connection import HyperConnectionModule

    from megatron.bridge.models.ernie.modeling_ernie55 import hybrid_stack
    from megatron.bridge.models.ernie.modeling_ernie55 import model as ernie_model

    _require_backend(backend)
    for module in (hybrid_stack, ernie_model):
        monkeypatch.setattr(module, "use_ernie55_hyper_connections", lambda root: 0)
    model = _build_model(_recompute_config(backend), seed=21).train()
    assert all(type(m) is HyperConnectionModule for m in model.modules() if isinstance(m, HyperConnectionModule))
    records = _record_replay(monkeypatch, model, _batch(), mhc_class=HyperConnectionModule)
    count, mismatches = _replay_mismatches(records)
    _report(f"megatron_core_recompute_replay[{backend}]", modules=count, mismatches=len(mismatches))
    for line in mismatches:
        logger.info("[ernie55-stack-gpu] Megatron-Core replay mismatch: %s", line)
    if mismatches:
        pytest.xfail(f"Megatron-Core's compiled mHC replay differs from its no-grad forward: {mismatches[:3]}")


# ---------------------------------------------------------------------------------------------
# torch_dist checkpoints
# ---------------------------------------------------------------------------------------------

# The distributed-optimizer formats Bridge writes into torch_dist checkpoints (training/checkpointing.py).
_OPTIMIZER_SHARDING_TYPES = ("dp_reshardable", "fully_reshardable")


def _optimizer_state_tensors(model, optimizer) -> dict[str, torch.Tensor]:
    """Every optimizer state tensor keyed by its parameter (the ``fully_reshardable`` model-space view)."""
    from megatron.core.dist_checkpointing.dict_utils import nested_values
    from megatron.core.dist_checkpointing.mapping import ShardedTensor

    state = optimizer.sharded_state_dict(
        _unwrap(model).sharded_state_dict(), metadata={"distrib_optim_sharding_type": "fully_reshardable"}
    )
    return {item.key: item.data.detach().clone() for item in nested_values(state) if isinstance(item, ShardedTensor)}


def _eval_losses(model, batch) -> torch.Tensor:
    """Per-token LM losses in eval mode (they depend on every parameter, multimax included)."""
    inner = _unwrap(model)
    was_training = inner.training
    inner.eval()
    try:
        with torch.no_grad():
            return _forward(inner, batch).clone()
    finally:
        inner.train(was_training)


@pytest.mark.parametrize("sharding_type", _OPTIMIZER_SHARDING_TYPES)
def test_torch_dist_checkpoint_round_trip(distributed, tmp_path, sharding_type):
    from megatron.core import dist_checkpointing
    from megatron.core.dist_checkpointing.serialization import load_tensors_metadata

    from megatron.bridge.models.ernie.modeling_ernie55 import check_ernie55_mtp_aliases

    _require_backend("none")
    batches = [_batch(seed=0)]
    model, optimizer = _training_setup(_config("none"), seed=17)
    for _ in range(2):
        optimizer.zero_grad()
        _run_schedule(model, batches)
        assert optimizer.step()[0], "the optimizer step was skipped"
    _pop_mtp_loss()
    source = _unwrap(model)
    expected_losses = _eval_losses(model, batches[0])

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    metadata = {"distrib_optim_sharding_type": sharding_type}
    model_state = source.sharded_state_dict()
    dist_checkpointing.save(
        {"model": model_state, "optimizer": optimizer.sharded_state_dict(model_state, metadata=metadata)},
        str(checkpoint_dir),
    )
    # Each shared tensor is written once, under its decoder key; the MTP body keeps only its buffers.
    keys = set(load_tensors_metadata(str(checkpoint_dir)))
    body_keys = sorted(key for key in keys if "mtp_model_layer" in key)
    assert body_keys and all(key.endswith("router.expert_bias") for key in body_keys), body_keys

    restored_model, restored_optimizer = _training_setup(_config("none"), seed=18)
    template_model = _unwrap(restored_model).sharded_state_dict()
    template = {
        "model": template_model,
        "optimizer": restored_optimizer.sharded_state_dict(template_model, is_loading=True, metadata=metadata),
    }
    loaded = dist_checkpointing.load(template, str(checkpoint_dir))
    _unwrap(restored_model).load_state_dict(loaded["model"])
    restored_optimizer.load_state_dict(loaded["optimizer"])

    restored = _unwrap(restored_model)
    assert sorted(name for name, _ in restored.named_parameters()) == sorted(
        name for name, _ in source.named_parameters()
    )
    for name, param in source.named_parameters():
        assert restored.get_parameter(name).dtype == param.dtype and torch.equal(
            restored.get_parameter(name), param
        ), name
    for (name, buffer), (other_name, other) in zip(source.named_buffers(), restored.named_buffers(), strict=True):
        if name.endswith("expert_bias"):
            assert other_name == name and torch.equal(other, buffer), name
    for alias, decoder_name in restored.ernie55_mtp_aliases.items():
        assert restored.get_parameter(alias) is restored.get_parameter(decoder_name), alias
    assert check_ernie55_mtp_aliases(restored_model) == len(restored.ernie55_mtp_aliases) > 0
    saved_states = _optimizer_state_tensors(model, optimizer)
    restored_states = _optimizer_state_tensors(restored_model, restored_optimizer)
    assert sorted(saved_states) == sorted(restored_states)
    assert not [key for key in saved_states if "mtp_model_layer" in key]
    for key, value in saved_states.items():
        assert torch.equal(restored_states[key], value), key
    assert torch.equal(_eval_losses(restored_model, batches[0]), expected_losses)
