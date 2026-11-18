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

"""CPU test of the whole ERNIE 5.5 model under context parallelism: CP = 2 and 4 against CP = 1.

``test_ernie55_cp.py`` checks one D attention module at CP > 1. This file checks the model around it:
:class:`~megatron.bridge.models.ernie.modeling_ernie55.Ernie55HybridModel` with fp32 hyper-connection
streams and downcasting norms, dense and MoE layers and the multimax head, fed by
:func:`~megatron.bridge.models.ernie.ernie55_context_parallel.build_ernie55_thd_cp_batch`. Its D layers
convert their rows from the stack's contiguous partition to zigzag and back. On ``CP`` gloo ranks
against the same model on one rank:

* the per-token LM losses and the final-norm rows of every rank equal the CP = 1 rows;
* the parameter gradients equal the CP = 1 gradients under the training emulation of
  ``test_ernie55_cp.py``: the LM gradient and the indexer-loss scale are multiplied by ``CP`` (the
  schedule's per-row scaling) and the gradients are averaged over the CP group (DDP);
* the ranks' indexer losses add up to the CP = 1 loss;
* the same holds with full recompute of every layer (Megatron-Core's checkpoint).

The pattern is D-only (``"D-DE"``): H layers at CP > 1 run Megatron-Core DSv4's contiguous CP path,
which builds its layout with CuTe DSL kernels (``test_ernie55_cp_gpu.py`` covers them). MTP at CP > 1 is
covered by ``test_ernie55_cp_mtp.py``. Parameters are fp32, so CP changes only the order of cross-rank sums. Every
gradient is compared relative to the largest gradient of the same parameter over all layers: at the
first layer the hyper-connection streams are identical copies of the embedding, so the gradients of
``alpha_pre`` (through a scale-invariant input norm) and ``alpha_res`` (through a doubly stochastic
``H_res``) are almost exactly zero, and their own magnitude is no scale for an error.
"""

from __future__ import annotations

import contextlib
import logging
import re
from dataclasses import dataclass

import pytest
import torch
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.experimental_attention_variant import dsa

from megatron.bridge.models.ernie.ernie55_context_parallel import IGNORE_LABEL, build_ernie55_thd_cp_batch
from megatron.bridge.models.ernie.modeling_ernie55 import context_parallel
from tests.unit_tests.models.ernie.ernie55_cp_harness import run_cases
from tests.unit_tests.models.ernie.ernie55_test_utils import cpu_current_device
from tests.unit_tests.models.ernie.test_ernie55_cp import _cpu_pos_emb_on_this_cp_rank, _rows_of_rank
from tests.unit_tests.models.ernie.test_ernie55_mhc_mtp import (
    VOCAB,
    _build_model,
    _hadamard,
    _LocalLinear,
    _StackNorm,
)


logger = logging.getLogger(__name__)

pytestmark = pytest.mark.unit

# D, dense, D, MoE: two D layers whose CP conversions meet hyper-connection streams.
PATTERN = "D-DE"
# Unaligned samples: padded to 2 * CP, their boundaries cut the ranks' contiguous chunks at CP = 2 and 4.
SAMPLE_LENS = (37, 21, 50)
PAD_TOKEN = 0
# At least every remote candidate of the longest padded document: selection is not ranking.
TOPK = 64
# fp32 CP > 1 against fp32 CP = 1 differs only in summation order. Gradients, sums over every row of
# the pack, get a wider bar (the worst measured on CPU was about 1e-5).
ROW_TOL = 2e-5
GRAD_TOL = 1e-4


@dataclass
class _Rank:
    rank: int
    world_size: int


def setup_rank(world_size: int) -> _Rank:
    """The CPU stand-ins the ERNIE 5.5 model needs on a spawned rank (those of ``test_ernie55_mhc_mtp.py``)."""
    from megatron.core.models.common.embeddings import rotary_pos_embedding, yarn_rotary_pos_embedding
    from megatron.core.models.hybrid import hybrid_block
    from megatron.core.tensor_parallel import random as tensor_parallel_random
    from megatron.core.transformer import multi_latent_attention
    from megatron.core.transformer import multi_token_prediction as mtp_module
    from megatron.core.transformer.experimental_attention_variant import absorbed_mla
    from megatron.core.transformer.experimental_attention_variant import deepseek_v4_hybrid_attention as dsv4

    # Router buffers, RoPE tables and the loss trackers are allocated on torch.cuda.current_device();
    # the router's expert-bias update is torch.compile'd, whose device probes need the real device.
    torch.cuda.current_device = cpu_current_device
    dsv4.TELinear = _LocalLinear
    dsv4._q_rms_norm = lambda q, eps: q * torch.rsqrt(q.square().mean(-1, keepdim=True) + eps)
    dsa.rotate_activation = _hadamard
    for module in (absorbed_mla, multi_latent_attention):
        module.TELinear = _LocalLinear
    # HybridStack builds its final norm as TENorm directly, not from the spec.
    hybrid_block.TENorm = _StackNorm
    mtp_module.is_inference_column_parallel_linear = lambda module: False
    # Megatron-Core's hyper-connection forward opens NVTX ranges, which need a CUDA build.
    torch.cuda.nvtx.range = lambda *args, **kwargs: contextlib.nullcontext()
    # The unpacked CP slice of a RoPE table copies through pinned CUDA memory; YaRN builds one at construction.
    for module in (rotary_pos_embedding, yarn_rotary_pos_embedding):
        if hasattr(module, "get_pos_emb_on_this_cp_rank"):
            module.get_pos_emb_on_this_cp_rank = _cpu_pos_emb_on_this_cp_rank
    # Megatron-Core's checkpoint keeps the CUDA RNG state for the recompute; CPU has none.
    tensor_parallel_random._get_cuda_rng_state = lambda *args, **kwargs: torch.zeros(1, dtype=torch.uint8)
    tensor_parallel_random._set_cuda_rng_state = lambda *args, **kwargs: None
    return _Rank(torch.distributed.get_rank(), world_size)


def _samples(seed: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """``(input_ids, labels)`` of unaligned samples; each sample's last token has no target."""
    generator = torch.Generator().manual_seed(seed)
    samples = []
    for length in SAMPLE_LENS:
        tokens = torch.randint(1, VOCAB, (length,), generator=generator)
        labels = tokens.roll(-1)
        labels[-1] = IGNORE_LABEL
        samples.append((tokens, labels))
    return samples


def _padded(
    samples: list[tuple[torch.Tensor, torch.Tensor]], multiple: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """The samples padded to a multiple of ``multiple`` the way ``build_ernie55_thd_cp_batch`` pads them."""
    padded = []
    for tokens, labels in samples:
        pad = -tokens.numel() % multiple
        padded.append(
            (
                torch.cat([tokens, tokens.new_full((pad,), PAD_TOKEN)]),
                torch.cat([labels, labels.new_full((pad,), IGNORE_LABEL)]),
            )
        )
    return padded


def _model_parallel(cp_size: int) -> ProcessGroupCollection:
    """Model-parallel state with ``CP = cp_size`` (the rest of the ranks is data parallelism)."""
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=cp_size
    )
    return ProcessGroupCollection.use_mpu_process_groups()


@dataclass
class _Run:
    losses: torch.Tensor
    final_norm: torch.Tensor
    grads: dict[str, torch.Tensor]
    indexer_loss: torch.Tensor


def _run(model, batch, *, scale: float) -> _Run:
    """Forward and backward of ``sum(losses * mask) * scale`` with the indexer-loss scale ``scale``."""
    tracker = dsa.DSAIndexerLossLoggingHelper.tracker
    tracker.clear()
    dsa.DSAIndexerLossAutoScaler.set_loss_scale(torch.tensor(scale))
    model.zero_grad(set_to_none=True)
    final_norm = []
    handle = model.decoder.final_norm.register_forward_hook(
        lambda module, args, output: final_norm.append(output.detach().clone())
    )
    try:
        losses = model(
            input_ids=batch.tokens,
            position_ids=batch.position_ids,
            attention_mask=None,
            labels=batch.labels,
            packed_seq_params=batch.packed_seq_params,
        )
    finally:
        handle.remove()
    (losses * batch.loss_mask * scale).sum().backward()
    grads = {name: param.grad.detach().clone() for name, param in model.named_parameters() if param.grad is not None}
    return _Run(losses.detach(), final_norm[-1], grads, tracker["values"].detach().clone())


def _error(actual: torch.Tensor, expected: torch.Tensor, scale: float | None = None) -> float:
    """``max|a - e|`` over ``scale`` (default ``max|e|``)."""
    expected = expected.double()
    scale = expected.abs().max().item() if scale is None else scale
    return (actual.double() - expected).abs().max().item() / max(scale, 1e-30)


def _all_reduce_sum(tensor: torch.Tensor, group) -> torch.Tensor:
    tensor = tensor.detach().clone()
    torch.distributed.all_reduce(tensor, group=group)
    return tensor


def _compare_with_cp1(rank: _Rank, *, seed: int, **overrides) -> None:
    """Run the model at CP = world size and at CP = 1 (every rank alone) and compare everything."""
    cp_size = rank.world_size
    samples = _samples(seed)
    values = dict(params_dtype=torch.float32, dsa_indexer_topk=TOPK, cp_comm_type="all_gather", **overrides)

    # CP = 1 on every rank: the other ranks are data parallel, and nothing communicates in the forward.
    reference = _build_model(_model_parallel(1), pattern=PATTERN, seed=seed, context_parallel_size=1, **values)
    reference_batch = build_ernie55_thd_cp_batch(_padded(samples, 2 * cp_size), pad_token_id=PAD_TOKEN, cp_group=None)
    expected = _run(reference, reference_batch, scale=1.0)
    parallel_state.destroy_model_parallel()

    pg_collection = _model_parallel(cp_size)
    cp_group = pg_collection.cp
    model = _build_model(
        pg_collection,
        pattern=PATTERN,
        seed=seed,
        context_parallel_size=cp_size,
        cp_partition_mode="contiguous",
        **values,
    )
    try:
        batch = build_ernie55_thd_cp_batch(
            samples, pad_token_id=PAD_TOKEN, cp_group=cp_group, cp_partition_mode="contiguous"
        )
        lengths = [int(n) for n in batch.packed_seq_params.thd_cp_host_cu_seqlens_q]
        rows = _rows_of_rank(
            dict(doc_lens=[b - a for a, b in zip(lengths, lengths[1:])]), cp_size, rank.rank, "contiguous"
        )
        assert torch.equal(batch.tokens[0], reference_batch.tokens[0, rows])
        assert torch.equal(batch.labels[0], reference_batch.labels[0, rows])
        assert torch.equal(batch.position_ids[0], reference_batch.position_ids[0, rows])
        actual = _run(model, batch, scale=float(cp_size))
        # Every D layer restored the stack partition after its forward and its recompute.
        assert batch.packed_seq_params.cp_partition_mode == "contiguous"

        errors = {}
        mask = batch.loss_mask[0].bool()
        errors["loss rows"] = _error(actual.losses[0][mask], expected.losses[0, rows][mask])
        errors["final-norm rows"] = _error(actual.final_norm, expected.final_norm[rows])
        errors["indexer loss summed over CP"] = _error(
            _all_reduce_sum(actual.indexer_loss, cp_group), expected.indexer_loss
        )
        assert actual.grads.keys() == expected.grads.keys(), sorted(actual.grads.keys() ^ expected.grads.keys())
        kind_scale = {}
        for name, grad in expected.grads.items():
            kind = re.sub(r"\.layers\.\d+\.", ".layers.*.", name)
            kind_scale[kind] = max(kind_scale.get(kind, 0.0), grad.double().abs().max().item())
        grad_errors = {
            name: _error(
                _all_reduce_sum(grad, cp_group) / cp_size,
                expected.grads[name],
                kind_scale[re.sub(r"\.layers\.\d+\.", ".layers.*.", name)],
            )
            for name, grad in actual.grads.items()
        }
    finally:
        parallel_state.destroy_model_parallel()

    worst_grad = max(grad_errors, key=grad_errors.get)
    logger.info(
        "[ernie55-cp-model rank %d CP%d %s] %s; worst gradient %s %.2e over %d parameters",
        rank.rank,
        cp_size,
        overrides,
        {key: f"{value:.2e}" for key, value in errors.items()},
        worst_grad,
        grad_errors[worst_grad],
        len(grad_errors),
    )
    failures = [f"{what}: {error:.3e} > {ROW_TOL:.0e}" for what, error in errors.items() if error > ROW_TOL]
    failures += [
        f"gradient of {name}: {error:.3e} > {GRAD_TOL:.0e}" for name, error in grad_errors.items() if error > GRAD_TOL
    ]
    assert not failures, "\n".join(failures)


def case_model(rank: _Rank) -> None:
    """The ERNIE 5.5 model (hyper-connections, multimax, MoE) at CP > 1 equals CP = 1."""
    _compare_with_cp1(rank, seed=3)


def case_model_full_recompute(rank: _Rank) -> None:
    """Full recompute of every layer: the no-grad forward and the replay convert from the restored partition."""
    _compare_with_cp1(rank, seed=5, recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1)


CASES = ["case_model", "case_model_full_recompute"]


@pytest.fixture(scope="module", params=[2, 4], ids=["cp2", "cp4"])
def cp_results(request, tmp_path_factory):
    """Run every case once on ``CP`` spawned gloo ranks."""
    if context_parallel.load_cp_layout_api() is None:
        pytest.skip("Megatron-Core main has no CP partition converter; ERNIE 5.5 rejects CP > 1 there")
    return run_cases(
        world_size=request.param,
        module_name=__name__,
        setup_name="setup_rank",
        case_names=CASES,
        workdir=tmp_path_factory.mktemp(f"ernie55_cp_model{request.param}"),
    )


@pytest.mark.parametrize("case", CASES, ids=[case.removeprefix("case_") for case in CASES])
def test_context_parallel_model(cp_results, case):
    failures = {rank: result["error"] for rank, result in cp_results[case].items() if not result["ok"]}
    assert len(cp_results[case]) > 1, "no rank reported"
    assert not failures, "\n".join(f"--- rank {rank} ---\n{error}" for rank, error in failures.items())
