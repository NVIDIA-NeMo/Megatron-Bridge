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

"""Tests for the opt-in static vision path (``model.vision_static_shapes``) of Qwen3.5-VL.

Covers the CPU batch converter (MRoPE ids, position-embedding and rotary indices, scatter indices, FLOP statistics),
``Qwen3VLVisionModel.forward_static`` against the eager per-image vision path, the sync-free scatter of the vision
embeddings, CUDA graph capture and replay of the static vision path, one microbatch through ``Qwen3VLModel``, the
over-cap handling, and the collate / Energon task-encoder hooks (these need the Qwen3.5-VL processor in the local
Hugging Face cache and are skipped without it). The GPU checks run on one GPU.
"""

import datetime
import json
import logging
import math
import os
import random
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from megatron.bridge.models.qwen_vl import static_vision as sv
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope import get_rope_index
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import Qwen3VLVisionRotaryEmbedding
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.vision_model import Qwen3VLVisionModel
from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs


IMG, VID, VSTART, VEND = 248056, 248057, 248053, 248054
PATCH_DIM = 3 * 2 * 16 * 16


logger = logging.getLogger(__name__)

pytestmark = pytest.mark.unit

# Processor whose image settings the collate checks use; loaded from the local Hugging Face cache only.
HF_PROCESSOR = "Qwen/Qwen3.5-397B-A17B"


def _record(name: str, **metrics) -> None:
    """Log the measurements of one check; pytest shows them with a failing test."""
    logger.info("%s %s", name, json.dumps(metrics, default=str))


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().double()
    b = b.detach().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


# ---------------------------------------------------------------------------------------------------------------
# Synthetic collate-style batches
# ---------------------------------------------------------------------------------------------------------------


def _datacomp_grids(n: int, seed: int) -> list:
    """(h, w) patch grids with patch counts typical of DataComp images at max_pixels 200704 (720-980 patches)."""
    counts = [784, 864, 884, 720, 900, 936, 980, 812, 840]
    rng = random.Random(seed)
    grids = []
    for _ in range(n):
        c = rng.choice(counts)
        q = c // 4
        pairs = [(a, q // a) for a in range(1, q + 1) if q % a == 0]
        a, b = rng.choice(pairs[max(0, len(pairs) // 2 - 1) : len(pairs) // 2 + 1])  # near square
        grids.append((2 * a, 2 * b))
    return grids


def make_batch(grids_per_sample, seq_len: int, seed: int, *, img=IMG, vstart=VSTART, vend=VEND, vocab=1000):
    """A batch in the form ``qwen2_5_collate_fn`` returns (padded to ``seq_len`` with pad id 0)."""
    gen = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    B = len(grids_per_sample)
    input_ids = torch.zeros(B, seq_len, dtype=torch.long)
    grid_rows = []
    for b, grids in enumerate(grids_per_sample):
        toks = [rng.randrange(1, vocab) for _ in range(rng.randrange(3, 12))]
        for h, w in grids:
            toks += [vstart] + [img] * (h * w // 4) + [vend]
            toks += [rng.randrange(1, vocab) for _ in range(rng.randrange(2, 8))]
            grid_rows.append([1, h, w])
        toks += [rng.randrange(1, vocab) for _ in range(rng.randrange(2, 20))]
        assert len(toks) <= seq_len, (len(toks), seq_len)
        input_ids[b, : len(toks)] = torch.tensor(toks)
    grid = torch.tensor(grid_rows, dtype=torch.long) if grid_rows else None
    n_real = sum(h * w for g in grids_per_sample for h, w in g)
    pv = torch.randn(n_real, PATCH_DIM, generator=gen) if n_real else None
    attn = (input_ids != 0).long()
    labels = torch.cat([input_ids[:, 1:], torch.full((B, 1), -100)], dim=1)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": attn.float(),
        "position_ids": torch.arange(seq_len).expand(B, -1).clone(),
        "attention_mask": attn,
        "visual_inputs": GenericVisualInputs(pixel_values=pv, image_grid_thw=grid),
    }


def _copy_batch(batch):
    out = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    vi = batch["visual_inputs"]
    out["visual_inputs"] = GenericVisualInputs(
        pixel_values=None if vi.pixel_values is None else vi.pixel_values.clone(),
        image_grid_thw=None if vi.image_grid_thw is None else vi.image_grid_thw.clone(),
    )
    return out


MROPE_CASES = [
    ("mock_28x28", [[(28, 28)]], 4096, 1024, 1),
    ("text_only", [[]], 512, 1024, 1),
    ("mbs2_text_only", [[], []], 512, 1024, 1),
    ("datacomp_like", [[g] for g in _datacomp_grids(1, 0)], 4096, 1024, 1),
    ("extreme_wide", [[(2, 300)]], 2048, 1024, 1),
    ("extreme_tall", [[(300, 2)]], 2048, 1024, 1),
    ("mbs4_mixed", [[(28, 28)], [], [(16, 40), (8, 8)], [(4, 60), (10, 12), (6, 6)]], 4096, 4096, 6),
    ("mbs3_three_images", [[(20, 20), (12, 30), (8, 8)], [(14, 14)], [(28, 28)]], 4096, 4096, 5),
]


def _spec(P, N, S, **kw):
    return sv.QwenVLStaticVisionSpec(patch_cap=P, max_images=N, seq_length=S, **kw)


# ---------------------------------------------------------------------------------------------------------------
# Process group / parallel state for the GPU checks
# ---------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mpu():
    from megatron.core.tensor_parallel import random as tp_random
    from transformer_engine.pytorch import distributed as te_distributed

    created = False
    added_env = []
    if not dist.is_initialized():
        for name, value in (("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", "29517")):
            if name not in os.environ:
                os.environ[name] = value
                added_env.append(name)
        torch.cuda.set_device(0)
        dist.init_process_group(backend="nccl", world_size=1, rank=0, timeout=datetime.timedelta(minutes=30))
        created = True
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    # The capture checks need Transformer Engine's graph-safe RNG tracker, the one full-iteration CUDA graphs run with.
    # Without force_reset_rng, a tracker that an earlier test in this process created stays in place: Megatron Core's
    # default tracker holds tensor states, which CUDAGraph.register_generator_state rejects. That tracker and
    # Transformer Engine's active RNG states are restored afterwards, so later tests see them unchanged.
    saved_rng = (
        tp_random._CUDA_RNG_STATE_TRACKER,
        tp_random._CUDA_RNG_STATE_TRACKER_INITIALIZED,
        te_distributed.get_all_rng_states(),
    )
    model_parallel_cuda_manual_seed(1234, te_rng_tracker=True, force_reset_rng=True)
    yield
    tp_random._CUDA_RNG_STATE_TRACKER, tp_random._CUDA_RNG_STATE_TRACKER_INITIALIZED, te_states = saved_rng
    te_distributed.set_all_rng_states(te_states)
    parallel_state.destroy_model_parallel()
    if created and dist.is_initialized():
        dist.destroy_process_group()
    for name in added_env:
        os.environ.pop(name, None)


@pytest.fixture
def te_deterministic(request):
    """Deterministic TE attention backward (NVTE_ALLOW_NONDETERMINISTIC_ALGO=0, read when TE builds the module).

    With the default (nondeterministic) setting the vision-attention backward accumulates in a varying order, so even
    two eager runs differ at about 2e-3 relative L2 in the bf16 gradients; comparisons are then against that floor.
    """
    mode = getattr(request, "param", "0")
    old = os.environ.get("NVTE_ALLOW_NONDETERMINISTIC_ALGO")
    os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = mode
    yield mode
    if old is None:
        os.environ.pop("NVTE_ALLOW_NONDETERMINISTIC_ALGO", None)
    else:
        os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = old


def _vision_shell(pos_dtype, n_pos=2304, hidden=1152, heads=16, merge=2):
    m = Qwen3VLVisionModel.__new__(Qwen3VLVisionModel)
    nn.Module.__init__(m)
    m.config = SimpleNamespace(spatial_merge_size=merge)
    m.spatial_merge_size = merge
    m.num_grid_per_side = int(n_pos**0.5)
    torch.manual_seed(7)
    m.pos_embed = nn.Embedding(n_pos, hidden).cuda().to(pos_dtype)
    m.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(hidden // heads // 2)
    return m


def _qwen35_vision_hf_config(depth: int, out_hidden: int):
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeVisionConfig

    return Qwen3_5MoeVisionConfig(
        depth=depth,
        hidden_size=1152,
        num_heads=16,
        intermediate_size=4304,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        in_channels=3,
        out_hidden_size=out_hidden,
        num_position_embeddings=2304,
    )


def _language_config(hidden=1024, recompute=False, **kw):
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import Qwen3VLTransformerConfig

    base = dict(
        num_layers=2,
        hidden_size=hidden,
        num_attention_heads=8,
        num_query_groups=8,
        kv_channels=128,
        ffn_hidden_size=2 * hidden,
        vocab_size=1024,
        language_max_sequence_length=4096,
        patch_size=16,
        temporal_patch_size=2,
        in_channels=3,
        spatial_merge_size=2,
        out_hidden_size=hidden,
        rotary_base=10000,
        rotary_percent=1.0,
        mrope_section=[24, 20, 20],
        image_token_id=1000,
        video_token_id=1001,
        vision_start_token_id=1002,
        normalization="RMSNorm",
        activation_func=F.silu,
        gated_linear_unit=True,
        add_bias_linear=False,
        add_qkv_bias=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    if recompute:
        base.update(recompute_granularity="selective")
    base.update(kw)
    return Qwen3VLTransformerConfig(**base)


def _build_vision_model(depth=2, out_hidden=4096, recompute=False, patch_cap=None):
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TENorm, TERowParallelLinear
    from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
    from megatron.core.process_groups_config import ProcessGroupCollection

    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import get_vision_model_config
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import PatchMergerSubmodules

    hf = _qwen35_vision_hf_config(depth, out_hidden)
    vcfg = get_vision_model_config(hf, megatron_config=_language_config(hidden=out_hidden, recompute=recompute))
    spec = get_vit_layer_with_transformer_engine_spec()
    spec.submodules.self_attention.module = Qwen3VLSelfAttention
    merger = PatchMergerSubmodules(
        patch_norm=TENorm, linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear
    )
    torch.manual_seed(11)
    model_parallel_cuda_manual_seed(1234, te_rng_tracker=True)
    vm = Qwen3VLVisionModel(
        vcfg,
        spec,
        merger,
        pre_process=True,
        post_process=True,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )
    vm = vm.cuda().to(torch.bfloat16)
    vm.train()
    vm.static_patch_cap = patch_cap
    return vm


def _static_inputs(out: dict, device="cuda"):
    return [
        out[k].to(device)
        for k in (
            "vision_pixel_values",
            "vision_cu_seqlens",
            "vision_patch_is_real",
            "vision_pos_embed_idx",
            "vision_pos_embed_w",
            "vision_rope_pos_ids",
        )
    ]


# ---------------------------------------------------------------------------------------------------------------
# MRoPE ids
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize("name,grids,S,P,N", MROPE_CASES)
def test_mrope_ids_equal_device_get_rope_index(name, grids, S, P, N):
    batch = make_batch(grids, S, seed=hash(name) % 1000)
    ids = batch["input_ids"].clone()
    grid = batch["visual_inputs"].image_grid_thw
    out = sv.build_static_vision_batch(_copy_batch(batch), _spec(P, N, S))
    ref, _ = get_rope_index(
        2,
        IMG,
        VID,
        VSTART,
        ids.cuda(),
        image_grid_thw=None if grid is None else grid.cuda(),
        video_grid_thw=None,
        attention_mask=None,
        packed_seq_params=None,
    )
    ok = torch.equal(out["mrope_position_ids"], ref.cpu())
    _record("mrope", case=name, B=len(grids), images=sum(len(g) for g in grids), equal=ok)
    assert ok


# ---------------------------------------------------------------------------------------------------------------
# Position embedding and rotary (bitwise against the model's own functions)
# ---------------------------------------------------------------------------------------------------------------

POS_GRIDS = [
    [(28, 28)],
    [(2, 300)],
    [(300, 2)],
    [(16, 40), (8, 8), (4, 60)],
    [(48, 48)],
    [(96, 10)],
    _datacomp_grids(3, 1),
]


@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize("pos_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("grids", POS_GRIDS)
def test_pos_embed_gather_bitwise_equals_interpolation(mpu, pos_dtype, grids):
    m = _vision_shell(pos_dtype)
    grid = torch.tensor([[1, h, w] for h, w in grids], dtype=torch.long)
    n = int(grid.prod(-1).sum())
    P = 4 * math.ceil((n + 8) / 4)
    ref = m.fast_pos_embed_interpolate(grid.cuda())
    idx, w = sv.pos_embed_interp_static(grid, 48, 2)
    idx_p = torch.zeros(4, P, dtype=torch.long)
    w_p = torch.zeros(4, P)
    idx_p[:, :n] = idx
    w_p[:, :n] = w
    pe = m.pos_embed(idx_p.cuda()) * w_p.cuda().to(pos_dtype)[:, :, None]
    st = pe[0] + pe[1] + pe[2] + pe[3]
    ok = torch.equal(st[:n], ref) and bool((st[n:] == 0).all())
    _record(
        "pos_embed",
        grids=grids,
        dtype=str(pos_dtype),
        bitwise=torch.equal(st[:n], ref),
        pad_zero=bool((st[n:] == 0).all()),
        max_abs=float((st[:n].float() - ref.float()).abs().max()),
    )
    assert ok


@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize("grids", POS_GRIDS)
def test_rotary_gather_bitwise_equals_rot_pos_emb(mpu, grids):
    m = _vision_shell(torch.bfloat16)
    grid = torch.tensor([[1, h, w] for h, w in grids], dtype=torch.long)
    n = int(grid.prod(-1).sum())
    P = 4 * math.ceil((n + 8) / 4)
    ref = m.rot_pos_emb(grid.cuda())
    ref = ref.reshape(n, 1, 1, -1).repeat(1, 1, 1, 2)
    ids = torch.zeros(P, 2, dtype=torch.long)
    ids[:n] = sv.rot_pos_ids_static(grid, 2)
    st = m.rotary_pos_emb(P)[ids.cuda()].flatten(1).reshape(P, 1, 1, -1).repeat(1, 1, 1, 2)
    ok = torch.equal(st[:n], ref)
    _record("rotary", grids=grids, bitwise=ok)
    assert ok


# ---------------------------------------------------------------------------------------------------------------
# Static vs eager per-image vision path (real Qwen3.5 vision dims, depth 2, bf16)
# ---------------------------------------------------------------------------------------------------------------

VISION_TOWER_CASES = [
    ("mock_784", [(28, 28)], 1024, 1),
    ("datacomp_like", _datacomp_grids(1, 2), 1024, 1),
    ("three_images", [(16, 40), (8, 8), (28, 28)], 2048, 4),
    ("extreme", [(2, 300)], 1024, 1),
]


@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize("te_deterministic", ["0", "1"], indirect=True)
@pytest.mark.parametrize("name,grids,P,N", VISION_TOWER_CASES)
def test_forward_static_matches_eager_per_image_vision(mpu, te_deterministic, name, grids, P, N):
    vm = _build_vision_model(depth=2, out_hidden=4096, patch_cap=P)
    S = 4096
    batch = make_batch([grids], S, seed=3)
    grid = batch["visual_inputs"].image_grid_thw
    pv = batch["visual_inputs"].pixel_values
    n = int(grid.prod(-1).sum())
    m_real = n // 4
    out = sv.build_static_vision_batch(_copy_batch(batch), _spec(P, N, S))
    torch.manual_seed(5)
    cot = torch.randn(m_real, 4096, device="cuda")

    def run(static: bool):
        vm.zero_grad(set_to_none=True)
        if static:
            y, _ = vm.forward_static(*_static_inputs(out))
            y_real = y[:m_real]
            loss = (y_real.float() * cot).sum()
        else:
            y, _ = vm(hidden_states=pv.cuda(), grid_thw=grid.cuda())
            y_real = y
            loss = (y_real.float() * cot).sum()
        loss.backward()
        grads = {k: p.grad.detach().float().clone() for k, p in vm.named_parameters() if p.grad is not None}
        return y.detach(), y_real.detach(), grads

    y_e, yr_e, g_e = run(False)
    _, _, g_e2 = run(False)
    y_s, yr_s, g_s = run(True)
    fwd = _rel_l2(yr_s, yr_e)
    per_param = {k: _rel_l2(g_s[k], g_e[k]) for k in g_e}
    # pos_embed.weight: the embedding backward reduces duplicate indices in an order that depends on the index count
    # (PyTorch switches kernels at 3072 indices); the static path always gathers 4 * P indices, so for small images
    # its bf16 gradient can differ from the eager path by rounding. Reported separately.
    gmax = max(v for k, v in per_param.items() if k != "pos_embed.weight")
    g_pos = per_param.get("pos_embed.weight", 0.0)
    floor = max(_rel_l2(g_e2[k], g_e[k]) for k in g_e)
    finite = bool(torch.isfinite(y_s).all()) and all(bool(torch.isfinite(g).all()) for g in g_s.values())
    _record(
        "static_vs_eager",
        case=name,
        te_nondeterministic=te_deterministic,
        patches=n,
        P=P,
        fwd_rel_l2=fwd,
        fwd_bitwise=torch.equal(yr_s, yr_e),
        grad_rel_l2_max_excl_pos_embed=gmax,
        grad_rel_l2_pos_embed=g_pos,
        grads_bitwise=all(torch.equal(g_s[k], g_e[k]) for k in g_e),
        nonbitwise_params=sorted(k for k in g_e if not torch.equal(g_s[k], g_e[k]))[:6],
        eager_vs_eager_grad_rel_l2_max=floor,
        finite=finite,
    )
    assert fwd <= 1e-3 and finite
    # Deterministic TE attention ("0"): strict. Nondeterministic ("1"): bf16 gradients vary run to run (~2e-3).
    assert gmax <= (1e-3 if te_deterministic == "0" else 5e-3)
    assert g_pos <= 1e-2


# ---------------------------------------------------------------------------------------------------------------
# Merged-row order and scatter
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize("name,grids,S,P,N", [c for c in MROPE_CASES if c[0] != "text_only"])
def test_scatter_matches_boolean_index_put(name, grids, S, P, N):
    batch = make_batch(grids, S, seed=17)
    ids = batch["input_ids"].clone()
    out = sv.build_static_vision_batch(_copy_batch(batch), _spec(P, N, S))
    B = ids.shape[0]
    H = 64
    M = P // 4
    torch.manual_seed(1)
    base = torch.randn(S, B, H, device="cuda", dtype=torch.bfloat16)
    rows = torch.arange(M, device="cuda", dtype=torch.float32)[:, None].expand(M, H).to(torch.bfloat16).contiguous()
    mask = ids.cuda() == IMG
    n_tok = int(mask.sum())

    # As in Qwen3VLModel.forward: the text embeddings are a non-leaf clone of the embedding output.
    a_text = base.clone().requires_grad_(True)
    a_vis = rows.clone().requires_grad_(True)
    y_a = sv.scatter_vision_embeds_static(a_text.clone(), a_vis, out["vision_dst_index"].cuda())

    b_text = base.clone().requires_grad_(True)
    b_vis = rows.clone().requires_grad_(True)
    tmp = b_text.clone().transpose(0, 1).contiguous()
    tmp[mask] = b_vis[:n_tok]
    y_b = tmp.transpose(0, 1).contiguous()

    torch.manual_seed(2)
    g = torch.randn_like(y_a)
    y_a.backward(g)
    y_b.backward(g)
    fwd = torch.equal(y_a, y_b)
    gt = torch.equal(a_text.grad, b_text.grad)
    gv = torch.equal(a_vis.grad[:n_tok], b_vis.grad[:n_tok]) and bool((a_vis.grad[n_tok:] == 0).all())
    _record("scatter", case=name, B=B, image_tokens=n_tok, fwd_equal=fwd, grad_text_equal=gt, grad_vis_equal=gv)
    assert fwd and gt and gv


# ---------------------------------------------------------------------------------------------------------------
# Capture safety
# ---------------------------------------------------------------------------------------------------------------


def _capture(fn):
    from megatron.core.tensor_parallel.random import (
        cudagraph_needs_generator_registration,
        get_all_rng_states,
        prime_cuda_rng_states_for_graph_capture,
    )

    graph = torch.cuda.CUDAGraph()
    if cudagraph_needs_generator_registration():
        for _, state in get_all_rng_states().items():
            graph.register_generator_state(state)
    # A process-wide autograd flag, as Megatron Core sets it for its captures; restored so later tests keep theirs.
    set_override = getattr(torch.autograd.graph, "set_override_stale_capture_stream", None)
    prior_override = None
    if set_override is not None:
        prior_override = torch._C._override_stale_capture_stream()
        set_override(True)
    try:
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
            prime_cuda_rng_states_for_graph_capture()
            result = fn()
        torch.cuda.synchronize()
    finally:
        if set_override is not None:
            set_override(prior_override)
    return graph, result


@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize("te_deterministic", ["0", "1"], indirect=True)
@pytest.mark.parametrize("recompute", [False, True])
def test_static_vision_capture_and_replay_without_host_sync(mpu, te_deterministic, recompute):
    P, N, S = 1024, 1, 4096
    vm = _build_vision_model(depth=2, out_hidden=1024, recompute=recompute, patch_cap=P)
    params = [p for p in vm.parameters() if p.requires_grad]
    b1 = sv.build_static_vision_batch(_copy_batch(make_batch([[(28, 28)]], S, seed=21)), _spec(P, N, S))
    b2 = sv.build_static_vision_batch(_copy_batch(make_batch([_datacomp_grids(1, 9)], S, seed=22)), _spec(P, N, S))
    static_in = [t.clone() for t in _static_inputs(b1)]
    dst = b1["vision_dst_index"].cuda().clone()
    ids = b1["input_ids"].cuda().clone()
    text = torch.randn(S, 1, 1024, device="cuda", dtype=torch.bfloat16)
    cot = torch.randn(S, 1, 1024, device="cuda", dtype=torch.float32)

    def step():
        y, _ = vm.forward_static(*static_in)
        comb = sv.scatter_vision_embeds_static(text, y, dst)
        loss = (comb.float() * cot).sum()
        loss.backward()
        return loss.detach()

    def load(b):
        for t, s in zip(static_in, _static_inputs(b)):
            t.copy_(s)
        dst.copy_(b["vision_dst_index"].cuda())
        ids.copy_(b["input_ids"].cuda())

    def grads():
        return [p.grad.detach().clone() for p in params]

    # (a) warm-up on a side stream (cuDNN/cuBLAS/TE workspaces, lazy inv_freq), then (b) one eager pass with the
    # PyTorch sync debugger armed.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            for p in params:
                p.grad = None
            step()
    torch.cuda.current_stream().wait_stream(side)
    for p in params:
        p.grad = None
    sync_error = None
    torch.cuda.set_sync_debug_mode("error")
    try:
        step()
    except RuntimeError as err:
        sync_error = str(err).splitlines()[0][:300]
    finally:
        torch.cuda.set_sync_debug_mode("default")

    # Eager references on b2 (twice: the eager-vs-eager determinism floor).
    load(b2)
    for p in params:
        p.grad = None
    loss_e1 = step()
    g_e1 = grads()
    for p in params:
        p.grad = None
    loss_e2 = step()
    g_e2 = grads()

    # (c) capture on b1, then replay on b2.
    load(b1)
    for p in params:
        p.grad = None
    capture_error = None
    try:
        graph, loss_static = _capture(step)
    except Exception as err:  # noqa: BLE001 - reported below
        capture_error = f"{type(err).__name__}: {str(err).splitlines()[0][:300]}"
        graph = None
    replay = {}
    if graph is not None:
        load(b2)
        graph.replay()
        torch.cuda.synchronize()
        g_r = [p.grad.detach().clone() for p in params]
        replay = dict(
            loss_bitwise=torch.equal(loss_static, loss_e1),
            grads_bitwise=all(torch.equal(a, b) for a, b in zip(g_r, g_e1)),
            loss_rel=abs(float(loss_static) - float(loss_e1)) / max(abs(float(loss_e1)), 1e-30),
            grad_rel_l2_max=max(_rel_l2(a, b) for a, b in zip(g_r, g_e1)),
            eager_vs_eager_bitwise=torch.equal(loss_e1, loss_e2)
            and all(torch.equal(a, b) for a, b in zip(g_e1, g_e2)),
        )
        # Replay again on the same inputs: identical (no stale state carried between replays).
        loss_first = loss_static.clone()
        graph.replay()
        torch.cuda.synchronize()
        replay["replay_repeat_bitwise"] = torch.equal(loss_static, loss_first) and all(
            torch.equal(p.grad, g) for p, g in zip(params, g_r)
        )
    _record(
        "capture",
        recompute=recompute,
        te_nondeterministic=te_deterministic,
        sync_debug_error=sync_error,
        capture_error=capture_error,
        **replay,
    )
    assert sync_error is None, sync_error
    assert capture_error is None, capture_error
    if te_deterministic == "0":
        assert replay["loss_bitwise"] and replay["grads_bitwise"] and replay["replay_repeat_bitwise"]
    else:
        assert replay["loss_rel"] <= 1e-3 and replay["grad_rel_l2_max"] <= 1e-2


# ---------------------------------------------------------------------------------------------------------------
# One-microbatch loss through Qwen3VLModel (2 dense LLM layers, real Qwen3.5 vision dims at depth 2)
# ---------------------------------------------------------------------------------------------------------------


def _build_tiny_qwen3vl(P: int, model_cls=None):
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.process_groups_config import ProcessGroupCollection

    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel

    model_cls = model_cls or Qwen3VLModel
    lang = _language_config(hidden=1024)
    lang.vision_static_shapes = True
    lang.vision_patch_cap = P
    hf_vision = _qwen35_vision_hf_config(depth=2, out_hidden=1024)
    spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=None, moe_grouped_gemm=False, qk_layernorm=False, fp8=False
    )
    torch.manual_seed(13)
    model_parallel_cuda_manual_seed(1234, te_rng_tracker=True)
    model = model_cls(
        language_transformer_config=lang,
        language_transformer_layer_spec=spec,
        vision_transformer_config=hf_vision,
        parallel_output=True,
        pre_process=True,
        post_process=True,
        add_encoder=True,
        add_decoder=True,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )
    model = model.cuda().to(torch.bfloat16)
    model.train()
    return model


def _tiny_batch(S=1024, P=1024):
    batch = make_batch([[(28, 28)]], S, seed=31, img=1000, vstart=1002, vend=1003, vocab=900)
    spec = _spec(P, 1, S, image_token_id=1000, video_token_id=1001, vision_start_token_id=1002)
    static = sv.build_static_vision_batch(_copy_batch(batch), spec)
    return batch, static


def _loss_and_grads(model, kwargs, loss_mask):
    model.zero_grad(set_to_none=True)
    out = model(**kwargs)
    loss = (out.float() * loss_mask).sum() / loss_mask.sum()
    loss.backward()
    grads = {k: p.grad.detach().float().clone() for k, p in model.named_parameters() if p.grad is not None}
    return loss.detach(), grads


@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize("te_deterministic", ["0", "1"], indirect=True)
def test_one_microbatch_loss_static_vs_dynamic(mpu, te_deterministic):
    P, S = 1024, 1024
    model = _build_tiny_qwen3vl(P)
    batch, static = _tiny_batch(S, P)
    ids = batch["input_ids"].cuda()
    labels = batch["labels"].cuda()
    loss_mask = batch["loss_mask"].cuda()
    dyn_kwargs = dict(
        input_ids=ids,
        position_ids=None,
        attention_mask=None,
        labels=labels,
        loss_mask=loss_mask,
        pixel_values=batch["visual_inputs"].pixel_values.cuda(),
        image_grid_thw=batch["visual_inputs"].image_grid_thw.cuda(),
    )
    st_kwargs = dict(
        input_ids=ids,
        position_ids=static["mrope_position_ids"].cuda(),
        attention_mask=None,
        labels=labels,
        loss_mask=loss_mask,
        static_vision_inputs={k: static[k].cuda() for k in sv.STATIC_VISION_KEYS[1:]},
    )
    l_d, g_d = _loss_and_grads(model, dyn_kwargs, loss_mask)
    l_d2, g_d2 = _loss_and_grads(model, dyn_kwargs, loss_mask)
    l_s1, g_s1 = _loss_and_grads(model, st_kwargs, loss_mask)
    l_s2, g_s2 = _loss_and_grads(model, st_kwargs, loss_mask)
    rel = abs(float(l_s1) - float(l_d)) / abs(float(l_d))
    finite = all(bool(torch.isfinite(g).all()) for g in g_s1.values()) and set(g_s1) == set(g_d)
    gmax = max(_rel_l2(g_s1[k], g_d[k]) for k in g_d)
    floor = max(_rel_l2(g_d2[k], g_d[k]) for k in g_d)
    twice = torch.equal(l_s1, l_s2) and all(torch.equal(g_s1[k], g_s2[k]) for k in g_s1)
    _record(
        "one_microbatch_loss",
        te_nondeterministic=te_deterministic,
        loss_dynamic=float(l_d),
        loss_static=float(l_s1),
        loss_rel_diff=rel,
        loss_bitwise=torch.equal(l_s1, l_d),
        grad_rel_l2_max=gmax,
        grads_bitwise=all(torch.equal(g_s1[k], g_d[k]) for k in g_d),
        eager_vs_eager_grad_rel_l2_max=floor,
        grads_finite=finite,
        static_twice_bitwise=twice,
        params_with_grad=len(g_s1),
    )
    assert rel <= 1e-3 and finite
    # Deterministic TE attention ("0"): strict. Nondeterministic ("1"): bf16 gradients vary run to run (~2e-3).
    assert gmax <= (1e-3 if te_deterministic == "0" else 5e-3)
    if te_deterministic == "0":
        assert twice


# ---------------------------------------------------------------------------------------------------------------
# Batch-contract invariants, the mock collate, over-cap handling
# ---------------------------------------------------------------------------------------------------------------


def _random_layout(rng, N, P, S):
    B = rng.randint(1, 4)
    budget = (P - 4 * (N + 1)) // max(1, N)
    grids = [[] for _ in range(B)]
    n_img = rng.randint(0, N)
    for _ in range(n_img):
        while True:
            h, w = 2 * rng.randint(1, 40), 2 * rng.randint(1, 40)
            if h * w <= budget:
                break
        grids[rng.randrange(B)].append((h, w))
    return grids


@pytest.mark.parametrize("P,N", [(1024, 1), (4096, 3)])
def test_static_batch_contract_invariants(P, N):
    from megatron.bridge.training.utils.flop_utils import vision_patch_stats_from_grid_thw

    S = 4096
    rng = random.Random(P + N)
    shapes = None
    n_batches = 500
    for i in range(n_batches):
        grids = _random_layout(rng, N, P, S)
        B = len(grids)
        batch = make_batch(grids, S, seed=i)
        ids = batch["input_ids"].clone()
        grid = batch["visual_inputs"].image_grid_thw
        out = sv.build_static_vision_batch(_copy_batch(batch), _spec(P, N, S))
        sig = {k: (tuple(out[k].shape), out[k].dtype) for k in sv.STATIC_VISION_KEYS[1:]}
        if shapes is None:
            shapes = sig
        assert sig == shapes, (sig, shapes)
        assert out["mrope_position_ids"].shape == (3, B, S)
        assert out["visual_inputs"] is None
        cu = out["vision_cu_seqlens"]
        assert cu.dtype == torch.int32 and int(cu[0]) == 0 and int(cu[-1]) == P
        assert bool((cu[1:] > cu[:-1]).all()) and int(cu[-1] - cu[-2]) >= 4
        n_real = 0 if grid is None else int(grid.prod(-1).sum())
        assert int(out["vision_patch_is_real"].sum()) == n_real
        pos = torch.nonzero((ids == IMG).reshape(-1)).squeeze(1)
        dst = out["vision_dst_index"]
        assert torch.equal(dst[: pos.numel()], pos)
        dummy = dst[pos.numel() :]
        assert bool((dummy >= B * S).all()) and dummy.unique().numel() == dummy.numel()
        stats = (0, 0, 0) if grid is None else vision_patch_stats_from_grid_thw(grid, spatial_merge_size=2)
        assert (
            out["flops_vision_patch_sum"],
            out["flops_vision_patch_sq_sum"],
            out["flops_vision_merged_token_sum"],
        ) == tuple(stats)
    _record("contract_invariants", P=P, N=N, batches=n_batches, ok=True)

    # Over the cap: the converter raises (the graph cannot fall back).
    with pytest.raises(sv.StaticVisionOverCapError):
        sv.build_static_vision_batch(_copy_batch(make_batch([[(32, 32)]], S, seed=1)), _spec(1024, 1, S))
    # Videos are rejected.
    vb = make_batch([[]], S, seed=2)
    vb["input_ids"][0, 5] = VID
    with pytest.raises(ValueError):
        sv.build_static_vision_batch(vb, _spec(P, N, S))
    # Truncated image span (token count != patches / 4) is rejected.
    tb = make_batch([[(28, 28)]], S, seed=3)
    tb["input_ids"][0, (tb["input_ids"][0] == IMG).nonzero()[-1]] = 7
    with pytest.raises(ValueError):
        sv.build_static_vision_batch(tb, _spec(1024, 1, S))


def test_over_cap_policies():
    from PIL import Image

    proc = SimpleNamespace(
        image_processor=SimpleNamespace(patch_size=16, merge_size=2, min_pixels=200704, max_pixels=1003520, size={})
    )
    big = Image.new("RGB", (2000, 1500))
    small = Image.new("RGB", (256, 256))
    spec_d = _spec(1024, 1, 4096, over_cap_policy="downscale")
    spec_e = _spec(1024, 1, 4096, over_cap_policy="error")
    out = sv.fit_example_images_to_static_budget([[big]], [[]], proc, spec_d, min_pixels=None, max_pixels=None)
    w, h = out[0][0].size
    n = sv.predict_image_patches(h, w, patch_size=16, merge_size=2, min_pixels=200704, max_pixels=1003520)
    assert n <= spec_d.per_image_patch_budget
    same = sv.fit_example_images_to_static_budget([[small]], [[]], proc, spec_d, min_pixels=None, max_pixels=None)
    assert same[0][0] is small
    with pytest.raises(sv.StaticVisionOverCapError):
        sv.fit_example_images_to_static_budget([[big]], [[]], proc, spec_e, min_pixels=None, max_pixels=None)
    with pytest.raises(ValueError):
        sv.fit_example_images_to_static_budget([[small]], [["video"]], proc, spec_d, min_pixels=None, max_pixels=None)
    with pytest.raises(sv.StaticVisionOverCapError):
        sv.fit_example_images_to_static_budget([[small, small]], [[]], proc, spec_d, min_pixels=None, max_pixels=None)
    _record(
        "over_cap",
        downscaled_to=(w, h),
        downscaled_patches=n,
        budget=spec_d.per_image_patch_budget,
        stats=sv.over_cap_stats(),
    )


def _load_processor():
    from transformers import AutoProcessor

    try:
        return AutoProcessor.from_pretrained(HF_PROCESSOR, local_files_only=True)
    except (OSError, ValueError) as err:
        pytest.skip(f"{HF_PROCESSOR} processor not in the local Hugging Face cache: {err}")


def test_mock_collate_static_vision(tmp_path):
    from megatron.bridge.data.builders.mock_vlm_sft import MockVLMSFTDatasetConfig, make_mock_vlm_examples
    from megatron.bridge.models.qwen_vl.data.collate_fn import qwen2_5_collate_fn

    proc = _load_processor()
    cfg = MockVLMSFTDatasetConfig(
        seq_length=4096, hf_processor_path=HF_PROCESSOR, num_base_examples=16, pad_to_max_length=True
    )
    examples = make_mock_vlm_examples(cfg)
    spec = _spec(1024, 1, 4096)
    results = []
    for i in range(8):
        ex = [examples[i]]
        dyn = qwen2_5_collate_fn(ex, proc, sequence_length=4096, pad_to_max_length=True, pad_to_multiple_of=128)
        st = qwen2_5_collate_fn(
            [examples[i]],
            proc,
            sequence_length=4096,
            pad_to_max_length=True,
            pad_to_multiple_of=128,
            static_vision=spec,
        )
        grid = dyn["visual_inputs"].image_grid_thw.reshape(-1, 3)
        n_real = int(grid.prod(-1).sum())
        for k in ("input_ids", "labels", "loss_mask", "attention_mask", "position_ids"):
            assert torch.equal(dyn[k], st[k]), k
        ref, _ = get_rope_index(
            2,
            spec.image_token_id,
            spec.video_token_id,
            spec.vision_start_token_id,
            dyn["input_ids"],
            image_grid_thw=grid,
            video_grid_thw=None,
            attention_mask=None,
            packed_seq_params=None,
        )
        assert torch.equal(st["mrope_position_ids"], ref)
        assert torch.equal(
            st["vision_pixel_values"][:n_real], dyn["visual_inputs"].pixel_values.reshape(n_real, -1).float()
        )
        results.append(
            dict(
                grid=grid.tolist(),
                n_real=n_real,
                image_tokens=int((st["input_ids"] == spec.image_token_id).sum()),
                cu=st["vision_cu_seqlens"].tolist(),
                S=int(st["input_ids"].shape[1]),
                keys=sorted(k for k in st if not k.startswith("flops_")),
            )
        )
    _record("mock_collate", samples=len(results), first=results[0], all_784=all(r["n_real"] == 784 for r in results))
    assert all(r["S"] == 4096 for r in results)


# ---------------------------------------------------------------------------------------------------------------
# FLOP statistics
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.run_only_on("GPU")
def test_static_flops_iterator_matches_dynamic_accounting():
    from megatron.bridge.models.qwen_vl.qwen3_vl_step import _pad_and_pack_qwen3_vl_step
    from megatron.bridge.training.utils.flop_utils import (
        _accumulator_to_int,
        accumulate_flops_metadata,
        vision_patch_stats_from_grid_thw,
    )

    names = (
        "_flops_seqlen_sum",
        "_flops_seqlen_sq_sum",
        "_flops_vision_patch_sum",
        "_flops_vision_patch_sq_sum",
        "_flops_vision_merged_token_sum",
    )

    def fresh():
        st = SimpleNamespace(**{n: 0 for n in names})
        st._flops_vision_patches = 0
        st._flops_requires_global_reduce = False
        return st

    pg = SimpleNamespace(tp=SimpleNamespace(size=lambda: 1), cp=SimpleNamespace(size=lambda: 1))
    S = 4096
    layouts = [[[(28, 28)]], [[]], [[(16, 40)], [(8, 8), (4, 60)]], [_datacomp_grids(1, 4)]]
    dyn, stat = fresh(), fresh()
    static_batches = []
    for i, grids in enumerate(layouts):
        batch = make_batch(grids, S, seed=40 + i)
        grid = batch["visual_inputs"].image_grid_thw
        tokens = batch["input_ids"].cuda()
        out = _pad_and_pack_qwen3_vl_step(
            tokens,
            batch["labels"].cuda(),
            batch["loss_mask"].cuda(),
            None,
            batch["position_ids"].cuda(),
            pg,
            use_fp8_padding=True,
            force_to_pad_to_seq_len=True,
            seq_length=S,
            padding_mask=None,
        )
        packed = out[5]
        stats = (0, 0, 0)
        if grid is not None and grid.numel() > 0:
            stats = vision_patch_stats_from_grid_thw(grid.cuda(), spatial_merge_size=2)
        accumulate_flops_metadata(dyn, out[0], vp_stage=None, cu_seqlens=packed.cu_seqlens_q, vision_patch_stats=stats)
        static_batches.append(sv.build_static_vision_batch(_copy_batch(batch), _spec(4096, 3, S)))
    for _ in sv.StaticVisionFlopsIterator(iter(static_batches), stat):
        pass
    a = {n: _accumulator_to_int(getattr(dyn, n)) for n in names}
    b = {n: _accumulator_to_int(getattr(stat, n)) for n in names}
    _record(
        "flops",
        dynamic=a,
        static=b,
        equal=a == b,
        global_reduce=(dyn._flops_requires_global_reduce, stat._flops_requires_global_reduce),
    )
    assert a == b and stat._flops_requires_global_reduce


def test_energon_hf_task_encoder_static_vision():
    """The HF Energon task encoder pre-checks images in the collate and converts the batch once in encode_batch."""
    from megatron.bridge.data.builders.mock_vlm_sft import MockVLMSFTDatasetConfig, make_mock_vlm_examples
    from megatron.bridge.data.energon.hf_task_encoder import HFEnergonSample, HFTaskEncoder

    proc = _load_processor()
    cfg = MockVLMSFTDatasetConfig(
        seq_length=4096, hf_processor_path=HF_PROCESSOR, num_base_examples=4, pad_to_max_length=True
    )
    examples = make_mock_vlm_examples(cfg)
    spec = _spec(1024, 1, 4096)
    kw = dict(processor=proc, seq_length=4096, pad_to_max_length=True, return_padding_mask=True, max_pixels=200704)
    te_static = HFTaskEncoder(**kw, static_vision=spec)
    te_dyn = HFTaskEncoder(**kw)
    samples = [HFEnergonSample(__key__="k0", __subflavors__={}, example=examples[0])]
    out = te_static.encode_batch(te_static.batch(samples))
    ref = te_dyn.encode_batch(te_dyn.batch(samples))
    assert out["visual_inputs"] is None and all(k in out for k in sv.STATIC_VISION_KEYS)
    for k in ("input_ids", "labels", "loss_mask", "attention_mask", "padding_mask", "position_ids"):
        assert torch.equal(out[k], ref[k]), k
    grid = ref["visual_inputs"].image_grid_thw.reshape(-1, 3)
    mrope, _ = get_rope_index(
        2,
        IMG,
        VID,
        VSTART,
        ref["input_ids"],
        image_grid_thw=grid,
        video_grid_thw=None,
        attention_mask=None,
        packed_seq_params=None,
    )
    assert torch.equal(out["mrope_position_ids"], mrope)
    _record(
        "energon_task_encoder",
        keys=sorted(k for k in out if not k.startswith("flops_")),
        padding_mask=tuple(out["padding_mask"].shape),
        grid=grid.tolist(),
        cu=out["vision_cu_seqlens"].tolist(),
    )


# ---------------------------------------------------------------------------------------------------------------
# Config wiring (ConfigContainer.validate -> configure_static_vision)
# ---------------------------------------------------------------------------------------------------------------


# The model configs the validation runs with: a plain namespace, the Qwen3.5-VL provider, and the builder-backed
# Qwen35VLModelConfig that the Qwen3.5-VL recipes build with AutoBridge.
MODEL_KINDS = ("namespace", "provider", "builder")


def _qwen35_vl_model_config(model_kind: str):
    from megatron.bridge.models.qwen_vl.qwen35_vl_provider import (
        _TRANSFORMERS_HAS_QWEN3_5_MOE,
        Qwen35VLMoEModelProvider,
    )

    if not _TRANSFORMERS_HAS_QWEN3_5_MOE:
        pytest.skip("transformers does not have qwen3_5_moe support")
    if model_kind == "provider":
        return Qwen35VLMoEModelProvider(num_layers=60, hidden_size=4096, num_attention_heads=32)

    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig

    from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import Qwen35VLMoEBridge

    return Qwen35VLMoEBridge().hf_config_to_model_config(Qwen3_5MoeConfig())


def _static_model(model_kind="namespace", **overrides):
    values = dict(
        vision_static_shapes=True,
        vision_patch_cap=1024,
        vision_max_images=1,
        vision_over_cap_policy="downscale",
        vision_cuda_graph_impl="none",
        cuda_graph_impl="full_iteration",
        context_parallel_size=1,
        sequence_parallel=False,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        seq_length=4096,
        image_token_id=IMG,
        video_token_id=VID,
        vision_start_token_id=VSTART,
    )
    values.update(overrides)
    if model_kind == "namespace":
        values.setdefault(
            "vision_config",
            SimpleNamespace(spatial_merge_size=2, patch_size=16, temporal_patch_size=2, num_position_embeddings=2304),
        )
        return SimpleNamespace(**values)

    model = _qwen35_vl_model_config(model_kind)
    values.setdefault("vision_config", _qwen35_vision_hf_config(depth=2, out_hidden=64))
    for name, value in values.items():
        # Set only declared fields, as a recipe or a CLI override can.
        if not hasattr(model, name):
            raise ValueError(f"{type(model).__name__} has no field {name!r}")
        setattr(model, name, value)
    return model


def _static_dataset(kind: str, **overrides):
    from megatron.bridge.data.builders.energon import (
        EnergonDatasetConfig,
        HFEnergonTaskEncoderConfig,
        QwenVLEnergonTaskEncoderConfig,
    )
    from megatron.bridge.data.builders.mock_vlm_sft import MockVLMSFTDatasetConfig

    if kind == "mock":
        values = dict(seq_length=4096, hf_processor_path=HF_PROCESSOR, pad_to_max_length=True)
        values.update(overrides)
        return MockVLMSFTDatasetConfig(**values)
    encoder_cls = HFEnergonTaskEncoderConfig if kind == "energon_hf" else QwenVLEnergonTaskEncoderConfig
    values = dict(
        path="/data/shards",
        seq_length=4096,
        micro_batch_size=1,
        task_encoder=encoder_cls(hf_processor_path=HF_PROCESSOR),
        pad_to_max_length=True,
    )
    values.update(overrides)
    return EnergonDatasetConfig(**values)


def _static_cfg(model=None, dataset=None, rerun_mode="disabled"):
    return SimpleNamespace(
        model=model if model is not None else _static_model(),
        dataset=dataset if dataset is not None else _static_dataset("mock", prefetch_factor=16),
        rerun_state_machine=SimpleNamespace(rerun_mode=rerun_mode),
    )


@pytest.mark.parametrize("model_kind", MODEL_KINDS)
@pytest.mark.parametrize("kind", ["mock", "energon_hf"])
def test_configure_static_vision_hands_the_spec_to_the_data_pipeline(kind, model_kind):
    dataset = _static_dataset(kind, prefetch_factor=16)
    spec = sv.configure_static_vision(_static_cfg(model=_static_model(model_kind), dataset=dataset))

    target = dataset if kind == "mock" else dataset.task_encoder
    assert target.static_vision is spec
    assert (spec.patch_cap, spec.max_images, spec.seq_length, spec.num_grid_per_side) == (1024, 1, 4096, 48)
    assert (spec.image_token_id, spec.video_token_id, spec.vision_start_token_id) == (IMG, VID, VSTART)
    assert spec.over_cap_policy == "downscale" and spec.per_image_patch_budget == 1020


@pytest.mark.parametrize(
    ("model_overrides", "dataset_overrides", "rerun_mode", "match"),
    [
        ({"vision_cuda_graph_impl": "transformer_engine"}, {}, "disabled", "vision_cuda_graph_impl"),
        ({"vision_patch_cap": 1022}, {}, "disabled", "vision_patch_cap"),
        ({"vision_max_images": 0}, {}, "disabled", "vision_max_images"),
        ({"sequence_parallel": True}, {}, "disabled", "sequence_parallel"),
        ({"context_parallel_size": 2}, {}, "disabled", "context_parallel_size"),
        ({"pipeline_model_parallel_size": 2}, {}, "disabled", "pipeline_model_parallel_size"),
        ({"sequence_packing_scheduler": "default"}, {}, "disabled", "global batch packing"),
        ({"vision_over_cap_policy": "drop"}, {}, "disabled", "vision_over_cap_policy"),
        ({}, {"pad_to_max_length": False}, "disabled", "pad_to_max_length"),
        ({}, {"enable_in_batch_packing": True}, "disabled", "enable_in_batch_packing"),
        ({}, {"seq_length": 2048}, "disabled", "seq_length"),
        ({}, {}, "validate_results", "rerun_mode"),
    ],
)
@pytest.mark.parametrize("model_kind", MODEL_KINDS)
def test_configure_static_vision_rejects_unsupported_runs(
    model_overrides, dataset_overrides, rerun_mode, match, model_kind
):
    cfg = _static_cfg(
        model=_static_model(model_kind, **model_overrides),
        dataset=_static_dataset("mock", **dataset_overrides),
        rerun_mode=rerun_mode,
    )

    with pytest.raises(ValueError, match=match):
        sv.configure_static_vision(cfg)


def test_configure_static_vision_rejects_unhooked_task_encoders():
    with pytest.raises(ValueError, match="does not support static vision"):
        sv.configure_static_vision(_static_cfg(dataset=_static_dataset("energon_qwen")))


@pytest.mark.parametrize("model_kind", MODEL_KINDS)
def test_configure_static_vision_warns_without_prefetch_under_full_iteration(caplog, model_kind):
    with caplog.at_level(logging.WARNING, logger=sv.__name__):
        sv.configure_static_vision(_static_cfg(model=_static_model(model_kind), dataset=_static_dataset("mock")))
    assert "prefetch_factor" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=sv.__name__):
        sv.configure_static_vision(
            _static_cfg(model=_static_model(model_kind, cuda_graph_impl="none"), dataset=_static_dataset("mock"))
        )
    assert "prefetch_factor" not in caplog.text
