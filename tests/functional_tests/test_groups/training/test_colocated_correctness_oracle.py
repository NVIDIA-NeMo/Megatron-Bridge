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
"""Colocated MegatronMIMO numerical correctness oracle (equal-DP, 2 GPUs).

Builds two MIMO models on every rank: a distributed model via the per-
module colocated path (``MegatronMIMOProvider`` with both modules at
``rank_offset=0``, DP=2 each, full ``setup_megatron_mimo`` setup) and a
reference model with the same ``parallelism_config`` but no DDP wrapping
or schedule machinery. After copying reference weights into the dist
model, both sides consume the same per-rank DP slice of one deterministic
global batch (asymmetric loss_mask) and run one forward+backward.

Dist drives ``forward_backward_no_pipelining`` with the language
``pg_collection`` — the same path ``train_step_megatron_mimo`` dispatches
in production. Optimizer step is deliberately skipped: a gradient-level
oracle isolates the schedule + DDP + loss-reduction path from optimizer
drift.

Reference runs the same forward on the same per-rank DP slice (no DDP),
producing local ``param.grad``. The test then manually all-reduces ref
grads (``SUM``) across the DP group and divides by the global token count,
matching dist's "DDP-allreduce-sum + finalize-divide-by-global-tokens"
arithmetic exactly. This mirrors per-rank computations bit-for-bit; only
the cross-rank reduction step differs (manual vs DDP+finalize), and both
do the same SUM operation.

Final assertion: ``dist_param.main_grad`` ≈ ``ref_param.grad``
element-wise.

Run:
    torchrun --nproc_per_node=2 -m pytest -v -s -x \\
        tests/functional_tests/test_groups/training/test_colocated_correctness_oracle.py
"""

from __future__ import annotations

# Determinism env vars must be set before any model construction. ``setdefault``
# preserves user overrides (e.g., a CI runner that already pinned them).
import os


os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")

from typing import Iterator

import pytest
import torch
import torch.distributed as dist
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_local_spec
from megatron.core.pipeline_parallel.schedules import forward_backward_no_pipelining
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)
from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import MegatronMIMOProvider
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    LoggerConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    megatron_mimo_runtime_config_update,
)
from megatron.bridge.training.megatron_mimo_parallel_utils import (
    unwrap_megatron_mimo_model,
    zero_grad_buffer_for_multimodule,
)
from megatron.bridge.training.megatron_mimo_step import forward_step as megatron_mimo_forward_step
from megatron.bridge.training.setup_megatron_mimo import setup_megatron_mimo
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.utils.train_utils import prepare_forward_step_func
from tests.functional_tests.utils import initialize_distributed


# ── Constants (small dims for fast oracle) ──────────────────────────────────

_HIDDEN_SIZE = 64
_FFN_HIDDEN_SIZE = 256
_NUM_HEADS = 4
_NUM_LAYERS = 2
_VOCAB_SIZE = 1000
_SEQ_LENGTH = 32
_IMG_SIZE = 32
_PATCH_DIM = 16
# CLIP ViT produces (img_h/patch_dim)^2 + 1 (class token) tokens.
_ENCODER_SEQ_LEN = (_IMG_SIZE // _PATCH_DIM) ** 2 + 1  # 5
# Place the special token at the top of the vocabulary so random text
# samples (drawn from [1, _SPECIAL_TOKEN_ID)) cannot collide with it.
# A collision would manifest as MIMO seeing extra vision-placeholder tokens
# in the text portion and tripping ``align_embeddings_by_token_positions``.
_SPECIAL_TOKEN_ID = 999
_GLOBAL_BATCH_SIZE = 2
_MICRO_BATCH_SIZE = 2  # = GBS since num_microbatches=1; sampler dp_size=1


# ── Model configs (no dropout, fp32, no fused/flash) ────────────────────────


def _make_vision_config() -> TransformerConfig:
    cfg = TransformerConfig(
        num_layers=_NUM_LAYERS,
        hidden_size=_HIDDEN_SIZE,
        ffn_hidden_size=_FFN_HIDDEN_SIZE,
        num_attention_heads=_NUM_HEADS,
        pipeline_dtype=torch.float32,
        bf16=False,
        fp16=False,
        variable_seq_lengths=True,
        moe_token_dispatcher_type="alltoall",
        # per-token-loss + pure-SUM DDP so per-rank grads scale to global mean.
        calculate_per_token_loss=True,
    )
    cfg.add_bias_linear = True
    cfg.add_qkv_bias = True
    cfg.hidden_dropout = 0.0
    cfg.attention_dropout = 0.0
    cfg.gated_linear_unit = False
    cfg.layernorm_zero_centered_gamma = False
    cfg.apply_query_key_layer_scaling = False
    cfg.bias_activation_fusion = False
    cfg.bias_dropout_fusion = False
    cfg.attention_softmax_in_fp32 = True
    cfg.normalization = "LayerNorm"
    cfg.apply_rope_fusion = False
    return cfg


def _make_language_config() -> TransformerConfig:
    cfg = TransformerConfig(
        num_layers=_NUM_LAYERS,
        hidden_size=_HIDDEN_SIZE,
        ffn_hidden_size=_FFN_HIDDEN_SIZE,
        num_attention_heads=_NUM_HEADS,
        pipeline_dtype=torch.float32,
        bf16=False,
        fp16=False,
        variable_seq_lengths=True,
        moe_token_dispatcher_type="alltoall",
        calculate_per_token_loss=True,
    )
    cfg.hidden_dropout = 0.0
    cfg.attention_dropout = 0.0
    cfg.bias_activation_fusion = False
    cfg.bias_dropout_fusion = False
    cfg.attention_softmax_in_fp32 = True
    cfg.apply_rope_fusion = False
    return cfg


def _build_model_specs():
    """Specs using local layer specs (no TE), small dims, fp32."""
    vision_encoder = ModuleSpec(
        module=CLIPViTModel,
        params={
            "transformer_config": _make_vision_config(),
            "transformer_layer_spec": get_vit_layer_with_local_spec(),
            "patch_dim": _PATCH_DIM,
            "img_h": _IMG_SIZE,
            "img_w": _IMG_SIZE,
        },
    )
    vision_submodule_spec = ModuleSpec(
        module=VisionModalitySubmodules,
        params={},
        submodules={"encoders": {"clip": vision_encoder}},
    )
    language_model_spec = ModuleSpec(
        module=GPTModel,
        params={
            "config": _make_language_config(),
            "transformer_layer_spec": get_gpt_layer_local_spec(),
            "vocab_size": _VOCAB_SIZE,
            "max_sequence_length": _SEQ_LENGTH,
        },
    )
    return language_model_spec, {"vision": vision_submodule_spec}, {"vision": _SPECIAL_TOKEN_ID}


def _build_dist_provider(par_cfg: MegatronMIMOParallelismConfig) -> MegatronMIMOProvider:
    language_model_spec, modality_submodules_spec, special_token_ids = _build_model_specs()
    provider = MegatronMIMOProvider(
        language_model_spec=language_model_spec,
        modality_submodules_spec=modality_submodules_spec,
        special_token_ids=special_token_ids,
        megatron_mimo_parallelism_config=par_cfg,
        topology={"vision": ["language"], "language": []},
        bf16=False,  # oracle runs fp32 for determinism; provider default is bf16=True
    )
    if not hasattr(provider, "num_moe_experts"):
        provider.num_moe_experts = None
    return provider


def _build_ref_model(par_cfg: MegatronMIMOParallelismConfig) -> torch.nn.Module:
    """Non-DDP reference: bare ``MimoModel`` per rank, same per-module grids
    as dist but no DDP wrapping or setup machinery.

    Why not the legacy ``parallelism_config=None`` path: it requires
    ``initialize_model_parallel`` to have populated every global field
    (``_INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP`` etc.), which MegatronMIMO
    intentionally skips. Setting `parallelism_config=par_cfg` and calling
    only `provide()` gives us a model that uses per-module pg_collections
    explicitly (no global-state lookups), but is bare — no DDP, no
    optimizer, no schedule hooks.

    At equal DP, each rank holds full param tensors (DP just replicates),
    sees the full global batch, and computes the gradient locally.
    """
    language_model_spec, modality_submodules_spec, special_token_ids = _build_model_specs()
    provider = MegatronMIMOProvider(
        language_model_spec=language_model_spec,
        modality_submodules_spec=modality_submodules_spec,
        special_token_ids=special_token_ids,
        megatron_mimo_parallelism_config=par_cfg,
        topology={"vision": ["language"], "language": []},
        bf16=False,
    )
    if not hasattr(provider, "num_moe_experts"):
        provider.num_moe_experts = None
    # provide() returns a CPU model; move to current device for the run.
    ref_model = provider.provide()
    ref_model = ref_model.cuda(torch.cuda.current_device())
    ref_model.train()
    return ref_model


def _build_dist_config(par_cfg: MegatronMIMOParallelismConfig) -> ConfigContainer:
    train_cfg = TrainingConfig(
        micro_batch_size=_MICRO_BATCH_SIZE,
        global_batch_size=_GLOBAL_BATCH_SIZE,
        train_iters=1,
    )
    train_cfg.num_microbatches = 1

    opt_config = OptimizerConfig(
        bf16=False,
        fp16=False,
        use_distributed_optimizer=True,
        lr=1e-4,
        min_lr=0.0,
    )

    return ConfigContainer(
        train=train_cfg,
        model=_build_dist_provider(par_cfg),
        optimizer=opt_config,
        scheduler=SchedulerConfig(start_weight_decay=0.0, end_weight_decay=0.0),
        # Dataset isn't used: the build_data_iterators_fn we pass to
        # setup_megatron_mimo synthesises a single-batch iterator directly.
        # Leave a placeholder mock provider so config validation passes.
        dataset=_PlaceholderDataProvider(),
        logger=LoggerConfig(),
        tokenizer=TokenizerConfig(),
        checkpoint=CheckpointConfig(),
    )


class _PlaceholderDataProvider:
    """No-op stand-in for ``cfg.dataset`` — unused because the test supplies
    its own ``build_data_iterators_fn``."""


# ── Deterministic batch generation ───────────────────────────────────────────


def _generate_global_batch(seed: int = 12345) -> dict:
    """Build one deterministic global batch and broadcast from rank 0.

    Asymmetric loss_mask: sample 0 has more valid tokens than sample 1 to
    catch local-mean vs global-mean denominator bugs (a uniform mask would
    let those bugs pass).

    The dataclass shape mirrors what ``slice_batch_for_megatron_mimo``
    expects in the forward step:

    * ``input_ids[B, S]``: positions ``[0, _ENCODER_SEQ_LEN)`` are the
      special token ID (vision placeholders), the rest are random text.
    * ``labels[B, S]``: clone of input_ids with image placeholders → -100.
    * ``loss_mask[B, S]``: 1.0 where loss applies, 0.0 elsewhere. Asymmetric
      across samples.
    * ``position_ids[B, S]``: ``arange(S)`` per sample.
    * ``modality_inputs["vision"]["clip"]["x"][B, 3, H, W]``: float images.
    * ``attention_mask = None``.
    """
    if dist.get_rank() == 0:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        text_len = _SEQ_LENGTH - _ENCODER_SEQ_LEN
        image_tokens = torch.full((_GLOBAL_BATCH_SIZE, _ENCODER_SEQ_LEN), _SPECIAL_TOKEN_ID, dtype=torch.long)
        # Cap random text below _SPECIAL_TOKEN_ID so the placeholder positions
        # MIMO counts in input_ids[..., :_ENCODER_SEQ_LEN] are the only ones
        # equal to _SPECIAL_TOKEN_ID.
        text_tokens = torch.randint(1, _SPECIAL_TOKEN_ID, (_GLOBAL_BATCH_SIZE, text_len), generator=gen)
        input_ids = torch.cat([image_tokens, text_tokens], dim=1)

        labels = input_ids.clone()
        labels[input_ids == _SPECIAL_TOKEN_ID] = -100

        loss_mask = torch.zeros(_GLOBAL_BATCH_SIZE, _SEQ_LENGTH, dtype=torch.float32)
        # Sample 0: all text positions valid (more tokens).
        loss_mask[0, _ENCODER_SEQ_LEN:] = 1.0
        # Sample 1: only the second half of text positions valid (fewer tokens).
        half = _ENCODER_SEQ_LEN + (text_len // 2)
        loss_mask[1, half:] = 1.0

        position_ids = torch.arange(_SEQ_LENGTH).unsqueeze(0).expand(_GLOBAL_BATCH_SIZE, -1).contiguous()

        # Float images (no rescaling): the encoder runs in fp32.
        pixel_values = torch.randn(_GLOBAL_BATCH_SIZE, 3, _IMG_SIZE, _IMG_SIZE, generator=gen, dtype=torch.float32)
    else:
        input_ids = torch.empty(_GLOBAL_BATCH_SIZE, _SEQ_LENGTH, dtype=torch.long)
        labels = torch.empty(_GLOBAL_BATCH_SIZE, _SEQ_LENGTH, dtype=torch.long)
        loss_mask = torch.empty(_GLOBAL_BATCH_SIZE, _SEQ_LENGTH, dtype=torch.float32)
        position_ids = torch.empty(_GLOBAL_BATCH_SIZE, _SEQ_LENGTH, dtype=torch.long)
        pixel_values = torch.empty(_GLOBAL_BATCH_SIZE, 3, _IMG_SIZE, _IMG_SIZE, dtype=torch.float32)

    # Broadcast from rank 0 so every rank has the same global batch.
    for t in (input_ids, labels, loss_mask, position_ids, pixel_values):
        t_cuda = t.cuda(non_blocking=False)
        dist.broadcast(t_cuda, src=0)
        t.copy_(t_cuda.cpu())

    device = torch.cuda.current_device()
    return {
        "input_ids": input_ids.to(device, non_blocking=False),
        "labels": labels.to(device, non_blocking=False),
        "loss_mask": loss_mask.to(device, non_blocking=False),
        "position_ids": position_ids.to(device, non_blocking=False),
        "modality_inputs": {"vision": {"clip": {"x": pixel_values.to(device, non_blocking=False)}}},
        "attention_mask": None,
    }


def _make_data_iterator(batch: dict) -> Iterator[dict]:
    """Yield the same global batch indefinitely (one step in this test, but
    forward_step may inspect the iterator more than once)."""
    while True:
        # Yield a fresh shallow copy each time so any in-place mutation by
        # slice_batch_for_megatron_mimo doesn't accumulate.
        yield {**batch, "modality_inputs": {**batch["modality_inputs"]}}


# ── Setup pipeline ───────────────────────────────────────────────────────────


def _build_data_iterators_fn(_cfg, _infra, *, train_state=None):
    """build_data_iterators_fn signature for setup_megatron_mimo.

    The actual batch is generated at the test level and threaded through
    via a closure-bound iterator constructed below.
    """
    # The real batch + iterator are wired in the test body; this function
    # returns a placeholder that setup_megatron_mimo only inspects to fill
    # ``setup_output.train_data_iterator``. We replace it before running the
    # schedule.
    return iter([]), None


# ── Reference-side execution ─────────────────────────────────────────────────


def _slice_global_batch_for_rank(global_batch: dict, dp_rank: int, dp_size: int) -> dict:
    """Slice the global batch by DP rank (matches what dist applies via
    ``slice_batch_for_megatron_mimo`` inside ``megatron_mimo_step.forward_step``).

    Equal-DP geometry: encoder DP and language DP both = dp_size, so a single
    contiguous slice is correct for all keys (text + modality).
    """
    sliced: dict = {}
    for key, value in global_batch.items():
        if isinstance(value, torch.Tensor):
            slice_size = value.size(0) // dp_size
            start = dp_rank * slice_size
            sliced[key] = value[start : start + slice_size]
        elif isinstance(value, dict):
            sliced[key] = _slice_global_batch_for_rank(value, dp_rank, dp_size)
        else:
            sliced[key] = value
    return sliced


def _ref_forward_backward_local(ref_model: torch.nn.Module, local_batch: dict) -> torch.Tensor:
    """Run one forward+backward on the reference model with the per-rank
    DP slice. Loss is the LOCAL sum of masked per-token losses (no division)
    — the test's outer all-reduce + divide-by-global-tokens mirrors dist's
    DDP-allreduce-sum + finalize-divide-by-global-tokens.

    Mirroring dist's per-rank computation bit-for-bit (same input, same
    forward, same local backward) keeps fp32 GEMM accumulation order
    identical. Only the cross-rank reduction step differs (manual here vs
    DDP+finalize on the dist side); both perform the same SUM op.
    """
    output_tensor = ref_model(
        input_ids=local_batch["input_ids"],
        labels=local_batch["labels"],
        loss_mask=local_batch["loss_mask"],
        position_ids=local_batch["position_ids"],
        modality_inputs=local_batch["modality_inputs"],
        attention_mask=None,
    )
    if isinstance(output_tensor, tuple):
        per_token_losses, _ = output_tensor
    else:
        per_token_losses = output_tensor

    losses = per_token_losses.float()
    loss_mask = local_batch["loss_mask"].contiguous().view(-1).float()
    total_loss = torch.sum(losses.view(-1) * loss_mask)  # local sum, no div
    total_loss.backward()
    return total_loss.detach()


def _allreduce_and_scale_ref_grads(ref_model: torch.nn.Module, dp_group, global_total_tokens: int) -> None:
    """Mirror dist's "DDP-allreduce-sum + finalize-divide-by-global-tokens"
    on the reference's local grads.

    Each rank's ref param.grad is the local-batch grad. Sum across DP gives
    the full-batch grad. Dividing by the global token count gives the per-
    token-mean grad — the same scalar applied to dist's main_grad.
    """
    for p in ref_model.parameters():
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=dp_group)
        p.grad.mul_(1.0 / max(global_total_tokens, 1))


# ── Weight copying and gradient comparison ──────────────────────────────────


def _named_params_unwrapped(model: torch.nn.Module) -> dict:
    """Return ``{normalized_name: param}`` for a model, with DDP/Float16Module
    wrapping stripped from parameter names.

    Both DDP and Float16Module nest the wrapped module under ``.module``, so a
    DDP-wrapped ``language_model`` shows up as ``language_model.module.…`` in
    ``named_parameters()``. The bare reference has no such wrapping, so we
    normalize the dist names by collapsing every ``.module.`` segment to ``.``
    before pairing. This matches the semantic identity of the parameter
    independent of how it was wrapped.
    """
    inner = unwrap_megatron_mimo_model(model)
    out: dict[str, torch.nn.Parameter] = {}
    for name, p in inner.named_parameters():
        # Collapse ".module." → "." (handles DDP and Float16Module nesting).
        # Iterate until stable to handle nested wrappers.
        normalized = name
        while ".module." in normalized:
            normalized = normalized.replace(".module.", ".")
        out[normalized] = p
    return out


def _copy_ref_weights_to_dist(ref_model: torch.nn.Module, dist_model: torch.nn.Module) -> None:
    """Pair params by normalized name; copy ref → dist tensor data in place.

    Equal-DP: every dist rank holds the full param, so a direct copy works.
    Mismatched names abort the test — we expect 1:1 correspondence at this
    geometry.
    """
    ref_params = _named_params_unwrapped(ref_model)
    dist_params = _named_params_unwrapped(dist_model)
    missing_in_dist = sorted(set(ref_params) - set(dist_params))
    missing_in_ref = sorted(set(dist_params) - set(ref_params))
    assert not missing_in_dist and not missing_in_ref, (
        f"Param name mismatch between ref and dist (cannot copy weights):\n"
        f"  in ref but not dist: {missing_in_dist[:5]}\n"
        f"  in dist but not ref: {missing_in_ref[:5]}"
    )
    with torch.no_grad():
        for name, ref_p in ref_params.items():
            dist_p = dist_params[name]
            assert dist_p.shape == ref_p.shape, (
                f"Shape mismatch for {name}: ref={tuple(ref_p.shape)}, dist={tuple(dist_p.shape)}"
            )
            dist_p.data.copy_(ref_p.data.to(dist_p.device, dist_p.dtype))


def _compare_grads(
    ref_model: torch.nn.Module,
    dist_model: torch.nn.Module,
    *,
    rtol: float = 1e-4,
    atol: float = 5e-6,
) -> None:
    """Assert dist_param.main_grad ≈ ref_param.grad for every paired param.

    DDP buffers reduced grads in ``param.main_grad``; the bare reference
    has grads in ``param.grad``. Both-None is fine (frozen / unused param);
    one-None is a fail.
    """
    ref_params = _named_params_unwrapped(ref_model)
    dist_params = _named_params_unwrapped(dist_model)

    mismatches: list[str] = []
    one_sided: list[str] = []
    compared = 0

    for name, ref_p in ref_params.items():
        dist_p = dist_params[name]
        ref_grad = ref_p.grad
        # DDP-managed grad lives in main_grad; fall back to .grad for params
        # that aren't bucket-managed (e.g., embeddings if treated specially).
        dist_grad = getattr(dist_p, "main_grad", None)
        if dist_grad is None:
            dist_grad = dist_p.grad

        if ref_grad is None and dist_grad is None:
            continue
        if (ref_grad is None) != (dist_grad is None):
            one_sided.append(
                f"{name}: ref_grad={'None' if ref_grad is None else 'present'}, "
                f"dist_grad={'None' if dist_grad is None else 'present'}"
            )
            continue

        compared += 1
        if not torch.allclose(
            dist_grad.detach().float(),
            ref_grad.detach().float().to(dist_grad.device),
            rtol=rtol,
            atol=atol,
        ):
            diff = (dist_grad.detach().float() - ref_grad.detach().float().to(dist_grad.device)).abs()
            mismatches.append(
                f"{name}: max_abs_diff={diff.max().item():.3e}, "
                f"mean_abs_diff={diff.mean().item():.3e}, "
                f"shape={tuple(dist_grad.shape)}"
            )

    assert not one_sided, "Gradient one-sided None (ref vs dist):\n  " + "\n  ".join(one_sided[:10])
    assert not mismatches, (
        f"Gradient mismatches in {len(mismatches)} of {compared} compared params:\n  " + "\n  ".join(mismatches[:10])
    )


# ── The test ────────────────────────────────────────────────────────────────


class TestColocatedCorrectnessOracle:
    """Numerical correctness oracle for colocated equal-DP training.

    Requires 2 GPUs.
    """

    @pytest.mark.run_only_on("GPU")
    def test_equal_dp_grads_match_full_replica_reference(self):
        """Dist (colocated, enc(tp1,dp2) x llm(tp1,dp2)) matches a non-
        distributed reference on element-wise gradient comparison after one
        forward+backward step with an asymmetric loss_mask.
        """
        initialize_distributed()

        world_size = dist.get_world_size()
        if world_size != 2:
            pytest.skip(f"Oracle requires exactly 2 GPUs, got {world_size}")

        # Determinism pinning. Env vars at module top; here we pin algos and
        # the framework-level deterministic flag.
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(12345)
        torch.cuda.manual_seed_all(12345)

        # report_theoretical_memory chokes on MegatronMIMOProvider (no
        # kv_channels); same monkey-patch as test_pretrain_megatron_mimo.
        import megatron.bridge.training.utils.train_utils as _tu

        _tu.report_theoretical_memory = lambda *a, **kw: None

        # mcore's tensor_parallel layers (used by local layer specs) read from
        # parallel_state._GLOBAL_MEMORY_BUFFER. initialize_model_parallel
        # normally sets it; MegatronMIMO skips that, so initialise it here.
        # The setter asserts current value is None, so guard against double-init.
        from megatron.core import parallel_state as _mpu

        if _mpu._GLOBAL_MEMORY_BUFFER is None:
            _mpu._set_global_memory_buffer()

        # Build the colocated parallelism config: both modules share rank
        # range [0, 2), TP=1, DP=2 each.
        par_cfg = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                    data_parallel_size=2,
                    rank_offset=0,
                ),
                "vision": ModuleParallelismConfig(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                    data_parallel_size=2,
                    rank_offset=0,
                ),
            },
        )

        # ── Distributed side: full setup_megatron_mimo path. ──
        dist_cfg = _build_dist_config(par_cfg)
        megatron_mimo_runtime_config_update(dist_cfg)
        global_state = GlobalState()
        global_state.cfg = dist_cfg
        setup_output = setup_megatron_mimo(state=global_state, build_data_iterators_fn=_build_data_iterators_fn)

        # ── Reference side: same per-module grids, bare MimoModel (no DDP). ──
        # Built AFTER dist setup so the second provider re-uses the same
        # already-created HyperCommGrid / ProcessGroupCollection cache;
        # construction is idempotent in our equal-DP geometry.
        ref_model = _build_ref_model(par_cfg)

        # ── Init parity: copy ref weights into dist. ──
        _copy_ref_weights_to_dist(ref_model, setup_output.model)

        # ── Generate one deterministic global batch with asymmetric mask. ──
        batch = _generate_global_batch(seed=67890)

        # ── Distributed forward+backward via the colocated schedule. ──
        zero_grad_buffer_for_multimodule(setup_output.module_to_grid_tuple)
        wrapped_forward_step = prepare_forward_step_func(megatron_mimo_forward_step, global_state)
        infra = setup_output.megatron_mimo_infra
        forward_backward_no_pipelining(
            forward_step_func=wrapped_forward_step,
            data_iterator=_make_data_iterator(batch),
            model=[setup_output.model],
            num_microbatches=1,
            seq_length=_SEQ_LENGTH,
            micro_batch_size=_MICRO_BATCH_SIZE,
            forward_only=False,
            pg_collection=infra.pg_collections["language"],
            # Force a synchronous DDP all-reduce inside finalize_model_grads.
            # Without this, finish_grad_sync only waits on in-flight async
            # syncs — and our oracle config doesn't enable overlap_grad_reduce,
            # so nothing would be in flight and main_grad would stay at each
            # rank's local value.
            force_all_reduce=True,
        )

        # ── Reference forward+backward on the per-rank DP slice. ──
        # Same per-rank computation as dist (identical fp32 GEMM accumulation
        # order); cross-rank reduction is done manually below to match dist's
        # DDP-allreduce-sum + finalize-divide-by-global-tokens.
        dp_size = world_size  # equal-DP, both modules have dp=2 = world_size
        dp_rank = dist.get_rank()
        local_batch = _slice_global_batch_for_rank(batch, dp_rank, dp_size)
        ref_model.zero_grad(set_to_none=True)
        _ref_forward_backward_local(ref_model, local_batch)

        # Mirror dist's grad-finalize: all-reduce-sum across DP, divide by
        # global token count. Use the language DP group — at equal-DP, all
        # modules' DP groups span the same ranks, so a single all-reduce
        # group is sufficient for the test's parameter set.
        global_total_tokens = int(batch["loss_mask"].sum().item())
        _allreduce_and_scale_ref_grads(ref_model, infra.pg_collections["language"].dp, global_total_tokens)

        # ── Compare finalized dist grads to reference grads. ──
        _compare_grads(ref_model, setup_output.model)
