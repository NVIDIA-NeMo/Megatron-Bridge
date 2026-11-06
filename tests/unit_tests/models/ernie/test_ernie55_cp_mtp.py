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

"""CPU tests of the ERNIE 5.5 per-row MTP loss: CP = 2 and 4 against CP = 1, and CP = 1 against Megatron-Core.

The per-row loss normalizes every MTP depth by the row's supervised tokens counted over the context-parallel
group (``CP * S / N``), like the per-row LM loss. On ``CP`` gloo ranks against the same model on one rank, with the
training emulation of a micro-batch (per-row LM loss, MTP loss scale 1, indexer-loss scale ``CP``, gradients averaged
over the CP group as DDP does):

* every parameter gradient equals the CP = 1 gradient, the SD-MTP body (shared with the last decoder layers)
  included, also under full recompute;
* the logged MTP loss, averaged over the CP group as the tracker does, equals the CP = 1 loss, which equals an
  independent fp64 recomputation from the MTP logits;
* Megatron-Core's own MTP loss (each rank's token mean) does not: the samples mask their prompts, so the ranks hold
  different numbers of supervised tokens. This is the error the per-row loss exists to remove.

At CP = 1 the per-row loss equals Megatron-Core's ``process_mtp_loss`` (both divide by the row's tokens), which the
single-process test checks for the gradients and every depth's logged loss, also with detached MTP heads.

The patterns are D-only with one or two shared-body MTP depths (``"D-DE/DE"``, ``"D-DE/DE/DE"``): H layers at
CP > 1 run Megatron-Core DSv4's CuTe DSL kernels (``test_ernie55_cp_gpu.py``). Two depths check that each depth is
scaled by ``1 / depths`` and reads its own hidden states and labels rolled ``depth`` times. Parameters are fp32.

A Megatron-Core build that runs ERNIE 5.5 at CP > 1 (it has the CP partition converter) must also have the grouped
MTP roll, which the CP64 recipe trains with: on such a build these tests fail instead of skipping. Only builds
without either (Megatron-Core main) skip them.
"""

from __future__ import annotations

import logging
import re

import pytest
import torch
from megatron.core import parallel_state
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer.experimental_attention_variant import dsa
from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler, MTPLossLoggingHelper

from megatron.bridge.models.ernie.ernie55_context_parallel import (
    IGNORE_LABEL,
    build_ernie55_thd_cp_batch,
    ernie55_per_row_lm_loss,
)
from megatron.bridge.models.ernie.modeling_ernie55 import PER_ROW_MTP_LOSS_SUPPORTED, context_parallel
from tests.unit_tests.models.ernie.ernie55_cp_harness import run_cases
from tests.unit_tests.models.ernie.ernie55_test_utils import mtp_row_loss_oracle
from tests.unit_tests.models.ernie.test_ernie55_cp import _rows_of_rank
from tests.unit_tests.models.ernie.test_ernie55_cp_model import (  # noqa: F401 - setup_rank runs on spawned ranks
    PAD_TOKEN,
    _all_reduce_sum,
    _error,
    _model_parallel,
    _padded,
    _Rank,
    setup_rank,
)
from tests.unit_tests.models.ernie.test_ernie55_mhc_mtp import (  # noqa: F401 - pytest fixtures
    VOCAB,
    _build_model,
    _cpu_nvtx,
    _fresh_trackers,
    pg_collection,
)


logger = logging.getLogger(__name__)

# ERNIE 5.5 runs at CP > 1 on this build: it must then have the per-row MTP loss too (fail, do not skip).
_CP_CAPABLE = context_parallel.load_cp_layout_api() is not None

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        not PER_ROW_MTP_LOSS_SUPPORTED and not _CP_CAPABLE,
        reason="needs Megatron-Core dev's grouped MTP roll (the per-row MTP loss); this build has no CP either",
    ),
]

PATTERN = "D-DE/DE"
# Two MTP depths: each depth's scale, hidden states and twice-rolled labels.
PATTERN_TWO_DEPTHS = "D-DE/DE/DE"
# Unaligned samples; boundaries cut the ranks' contiguous chunks at CP = 2 and 4.
SAMPLE_LENS = (37, 21, 50)
# Prompt tokens without a loss at the start of every sample: the ranks hold different numbers of supervised tokens.
PROMPT_LENS = (11, 3, 29)
TOPK = 64
ROW_TOL = 2e-5
GRAD_TOL = 1e-4


def _samples(seed: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    samples = []
    for length, prompt in zip(SAMPLE_LENS, PROMPT_LENS):
        tokens = torch.randint(1, VOCAB, (length,), generator=generator)
        labels = tokens.roll(-1)
        labels[-1] = IGNORE_LABEL
        labels[:prompt] = IGNORE_LABEL
        samples.append((tokens, labels))
    return samples


def _mtp_depths(pattern: str) -> int:
    return pattern.count("/")


def _run(model, batch, *, cp_group, cp_size: int, mcore_mtp_loss: bool = False) -> dict:
    """One per-row training micro-batch: forward, per-row LM loss, backward; returns the gradients and the logs.

    ``mtp_logged`` holds every depth's tracked loss and ``mtp_logits`` every depth's logits, in depth order.
    """
    MTPLossLoggingHelper.tracker.clear()
    dsa.DSAIndexerLossLoggingHelper.tracker.clear()
    # The schedule's scales for one micro-batch with calculate_per_token_loss=False.
    MTPLossAutoScaler.set_loss_scale(torch.tensor(1.0))
    dsa.DSAIndexerLossAutoScaler.set_loss_scale(torch.tensor(float(cp_size)))
    model.zero_grad(set_to_none=True)
    mtp_logits = []
    # Clone: Megatron-Core's vocab-parallel cross entropy exponentiates the logits in place.
    handle = model.output_layer.register_forward_hook(
        lambda module, args, output: mtp_logits.append(output[0].detach().clone())
    )
    forward = HybridModel.forward if mcore_mtp_loss else type(model).forward
    try:
        losses = forward(
            model,
            input_ids=batch.tokens,
            position_ids=batch.position_ids,
            attention_mask=None,
            labels=batch.labels.clamp(min=0),
            loss_mask=batch.loss_mask,
            packed_seq_params=batch.packed_seq_params,
        )
    finally:
        handle.remove()
    loss, num_tokens, _ = ernie55_per_row_lm_loss(batch.loss_mask, losses, cp_group=cp_group)
    assert int(num_tokens) == 1
    loss.backward()
    grads = {name: param.grad.detach().clone() for name, param in model.named_parameters() if param.grad is not None}
    depths = model.config.mtp_num_layers
    # The output layer runs for every MTP depth in order first, then for the main head.
    assert len(mtp_logits) == depths + 1
    logged = MTPLossLoggingHelper.tracker["loss_sums"].detach().clone()
    assert logged.shape == (depths,)
    return dict(grads=grads, mtp_logged=logged, mtp_logits=mtp_logits[:depths])


def _grad_errors(actual: dict, expected: dict, cp_group, cp_size: int) -> dict[str, float]:
    """Per-parameter error of ``actual`` (averaged over ``cp_group`` unless it is ``None``) against ``expected``.

    Every error is relative to the largest gradient of the same parameter over all layers.
    """
    kind_scale = {}
    for name, grad in expected.items():
        kind = re.sub(r"\.layers\.\d+\.", ".layers.*.", name)
        kind_scale[kind] = max(kind_scale.get(kind, 0.0), grad.double().abs().max().item())
    return {
        name: _error(
            grad if cp_group is None else _all_reduce_sum(grad, cp_group) / cp_size,
            expected[name],
            kind_scale[re.sub(r"\.layers\.\d+\.", ".layers.*.", name)],
        )
        for name, grad in actual.items()
    }


def _compare_with_cp1(rank: _Rank, *, seed: int, pattern: str = PATTERN, **overrides) -> None:
    cp_size = rank.world_size
    samples = _samples(seed)
    values = dict(
        params_dtype=torch.float32,
        dsa_indexer_topk=TOPK,
        cp_comm_type="all_gather",
        ernie_per_row_loss=True,
        calculate_per_token_loss=False,
        **overrides,
    )

    reference = _build_model(_model_parallel(1), pattern=pattern, seed=seed, context_parallel_size=1, **values)
    reference_batch = build_ernie55_thd_cp_batch(_padded(samples, 2 * cp_size), pad_token_id=PAD_TOKEN, cp_group=None)
    expected = _run(reference, reference_batch, cp_group=None, cp_size=1)
    # CP = 1 run-to-run noise: the same pack with its documents reversed changes only the summation order.
    reordered = build_ernie55_thd_cp_batch(_padded(samples[::-1], 2 * cp_size), pad_token_id=PAD_TOKEN, cp_group=None)
    noise = _grad_errors(_run(reference, reordered, cp_group=None, cp_size=1)["grads"], expected["grads"], None, 1)
    cu = [int(n) for n in reference_batch.packed_seq_params.cu_seqlens_q]
    oracles = [
        mtp_row_loss_oracle(logits, reference_batch.labels, reference_batch.loss_mask, cu, depth=depth)
        for depth, logits in enumerate(expected["mtp_logits"], start=1)
    ]
    parallel_state.destroy_model_parallel()

    pg_collection = _model_parallel(cp_size)
    cp_group = pg_collection.cp
    model = _build_model(
        pg_collection,
        pattern=pattern,
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
        assert torch.equal(batch.labels[0], reference_batch.labels[0, rows])
        counts = [torch.zeros(1) for _ in range(cp_size)]
        torch.distributed.all_gather(counts, batch.loss_mask.sum().reshape(1), group=cp_group)
        assert len({float(count) for count in counts}) > 1, "the ranks must hold different supervised-token counts"

        actual = _run(model, batch, cp_group=cp_group, cp_size=cp_size)
        mcore = _run(model, batch, cp_group=cp_group, cp_size=cp_size, mcore_mtp_loss=True)
        assert batch.packed_seq_params.cp_partition_mode == "contiguous"

        logged = _all_reduce_sum(actual["mtp_logged"], cp_group) / cp_size
        errors = {
            f"depth {depth} logged MTP loss vs CP = 1": _error(logged[depth - 1], expected["mtp_logged"][depth - 1])
            for depth in range(1, _mtp_depths(pattern) + 1)
        }
        errors.update(
            {
                f"depth {depth} CP = 1 logged MTP loss vs fp64 oracle": _error(
                    expected["mtp_logged"][depth - 1], torch.tensor(oracle)
                )
                for depth, oracle in enumerate(oracles, start=1)
            }
        )
        grad_errors = _grad_errors(actual["grads"], expected["grads"], cp_group, cp_size)
        mcore_grad_errors = _grad_errors(mcore["grads"], expected["grads"], cp_group, cp_size)
    finally:
        parallel_state.destroy_model_parallel()

    worst = max(grad_errors, key=grad_errors.get)
    worst_mcore = max(mcore_grad_errors, key=mcore_grad_errors.get)
    logger.info(
        "[ernie55-cp-mtp rank %d CP%d %s] %s; worst gradient %s %.2e; Megatron-Core MTP loss: worst gradient %s %.2e",
        rank.rank,
        cp_size,
        overrides,
        {key: f"{value:.2e}" for key, value in errors.items()},
        worst,
        grad_errors[worst],
        worst_mcore,
        mcore_grad_errors[worst_mcore],
    )
    failures = [f"{what}: {error:.3e} > {ROW_TOL:.0e}" for what, error in errors.items() if error > ROW_TOL]
    # A gradient whose own CP = 1 noise exceeds the bar (a near-zero scalar sum) is held to 3x that noise.
    failures += [
        f"gradient of {name}: {error:.3e} > max({GRAD_TOL:.0e}, 3 x CP = 1 noise {noise[name]:.3e})"
        for name, error in grad_errors.items()
        if error > max(GRAD_TOL, 3 * noise[name])
    ]
    # Megatron-Core's per-rank token mean must be measurably wrong here, or this test proves nothing.
    if mcore_grad_errors[worst_mcore] < 100 * GRAD_TOL:
        failures.append(f"Megatron-Core's MTP loss matched CP = 1 too ({mcore_grad_errors[worst_mcore]:.3e})")
    assert not failures, "\n".join(failures)


def case_per_row_mtp(rank: _Rank) -> None:
    """Per-row MTP loss at CP > 1 equals CP = 1."""
    _compare_with_cp1(rank, seed=7)


def case_per_row_mtp_full_recompute(rank: _Rank) -> None:
    """The same with full recompute of every layer."""
    _compare_with_cp1(rank, seed=9, recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1)


def case_per_row_mtp_two_depths(rank: _Rank) -> None:
    """Two MTP depths: the second depth's labels roll twice across the ranks' boundaries."""
    _compare_with_cp1(rank, seed=13, pattern=PATTERN_TWO_DEPTHS)


CASES = ["case_per_row_mtp", "case_per_row_mtp_full_recompute", "case_per_row_mtp_two_depths"]


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
        workdir=tmp_path_factory.mktemp(f"ernie55_cp_mtp{request.param}"),
    )


@pytest.mark.parametrize("case", CASES, ids=[case.removeprefix("case_") for case in CASES])
def test_per_row_mtp_loss_under_context_parallelism(cp_results, case):
    failures = {rank: result["error"] for rank, result in cp_results[case].items() if not result["ok"]}
    assert len(cp_results[case]) > 1, "no rank reported"
    assert not failures, "\n".join(f"--- rank {rank} ---\n{error}" for rank, error in failures.items())


def test_cp_capable_builds_have_the_per_row_mtp_loss():
    """The CP64 recipe trains MTP at CP > 1 with the per-row loss: a build that runs CP must have its MTP roll."""
    assert PER_ROW_MTP_LOSS_SUPPORTED, (
        "This Megatron-Core build runs ERNIE 5.5 at context_parallel_size > 1 (it has the CP partition converter) but "
        "lacks dev's grouped MTP roll, so per-row MTP at CP > 1 is rejected and the CP64 recipe cannot train."
    )


@pytest.mark.parametrize(
    "pattern, overrides",
    [
        (PATTERN, {}),
        (PATTERN_TWO_DEPTHS, {}),
        # SD-MTP rejects detached heads (the MTP body is the decoder's): an MTP body of its own.
        (PATTERN_TWO_DEPTHS, dict(mtp_detach_heads=True, ernie_mtp_shared_last_layer=False)),
    ],
    ids=["one-depth", "two-depths", "two-depths-detached-heads"],
)
def test_per_row_mtp_loss_equals_megatron_core_at_cp1(pg_collection, pattern, overrides):
    """At CP = 1 both divide each depth by the row's supervised tokens: same gradients and logged losses.

    With two depths this pins each depth's ``1 / depths`` scale and its own hidden states and labels; with
    ``mtp_detach_heads`` the MTP losses must not reach the output layer's weight, as in Megatron-Core.
    """
    batch = build_ernie55_thd_cp_batch(_samples(11), pad_token_id=PAD_TOKEN, cp_group=None)
    values = dict(params_dtype=torch.float32, dsa_indexer_topk=TOPK, ernie_per_row_loss=True, **overrides)
    model = _build_model(pg_collection, pattern=pattern, seed=11, **values)
    ernie = _run(model, batch, cp_group=None, cp_size=1)
    mcore = _run(model, batch, cp_group=None, cp_size=1, mcore_mtp_loss=True)
    depths = _mtp_depths(pattern)
    assert ernie["mtp_logged"].shape == mcore["mtp_logged"].shape == (depths,)
    for depth in range(depths):
        assert _error(ernie["mtp_logged"][depth], mcore["mtp_logged"][depth]) <= ROW_TOL, depth
    cu = [int(n) for n in batch.packed_seq_params.cu_seqlens_q]
    for depth, logits in enumerate(ernie["mtp_logits"], start=1):
        oracle = mtp_row_loss_oracle(logits, batch.labels, batch.loss_mask, cu, depth=depth)
        assert _error(ernie["mtp_logged"][depth - 1], torch.tensor(oracle)) <= ROW_TOL, depth
    assert ernie["grads"].keys() == mcore["grads"].keys()
    errors = {name: _error(grad, mcore["grads"][name]) for name, grad in ernie["grads"].items()}
    worst = max(errors, key=errors.get)
    assert errors[worst] <= GRAD_TOL, f"{worst}: {errors[worst]:.3e}"


def test_the_override_is_scoped_to_the_forward():
    """Megatron-Core's process_mtp_loss is restored after the block, also when the forward raises."""
    from megatron.core.models.hybrid import hybrid_model

    from megatron.bridge.models.ernie.modeling_ernie55.mtp_loss import ernie55_process_mtp_loss, per_row_mtp_loss_in

    original = hybrid_model.process_mtp_loss
    with pytest.raises(RuntimeError, match="boom"):
        with per_row_mtp_loss_in(hybrid_model):
            assert hybrid_model.process_mtp_loss is ernie55_process_mtp_loss
            raise RuntimeError("boom")
    assert hybrid_model.process_mtp_loss is original


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(mtp_loss_early_backward=True), "mtp_loss_early_backward"),
        (dict(mtp_loss_type="e2e_tv"), "mtp_loss_type='e2e_tv'"),
    ],
    ids=["early-backward", "loss-type"],
)
def test_per_row_loss_rejects_other_mtp_objectives_at_config_time(overrides, message):
    from tests.unit_tests.models.ernie.test_ernie55_mhc_mtp import _config

    with pytest.raises(ValueError, match=message):
        _config(PATTERN, ernie_per_row_loss=True, **overrides)
    # Megatron-Core's own MTP loss (per-token mode) keeps them.
    _config(PATTERN, ernie_per_row_loss=False, **overrides)


def _plain_mtp_config(**overrides):
    from types import SimpleNamespace

    return SimpleNamespace(**{"mtp_num_layers": 1, "mtp_loss_type": "cross_entropy", **overrides})


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(labels=None), "labels=None"),
        (dict(mtp_input_mask=torch.ones(1, 4)), "mtp_input_mask"),
        (dict(scale_logits_fn=lambda logits: logits), "muP"),
        (dict(config=_plain_mtp_config(mtp_loss_type="e2e_tv")), "mtp_loss_type other than 'cross_entropy'"),
        (
            dict(config=_plain_mtp_config(cross_entropy_loss_fusion=True, cross_entropy_fusion_impl="linear")),
            "fused linear cross entropy",
        ),
        (dict(config=_plain_mtp_config(mtp_loss_early_backward=True)), "mtp_loss_early_backward"),
    ],
    ids=["no-labels", "input-mask", "mup", "loss-type", "fused-linear-ce", "early-backward"],
)
def test_unsupported_mtp_objectives_fail_loudly(overrides, message):
    from megatron.bridge.models.ernie.modeling_ernie55.mtp_loss import ernie55_process_mtp_loss

    arguments = dict(
        hidden_states=torch.zeros(8, 1, 2),
        labels=torch.zeros(1, 4, dtype=torch.long),
        loss_mask=torch.ones(1, 4),
        output_layer=None,
        output_weight=None,
        is_training=True,
        compute_language_model_loss=None,
        config=_plain_mtp_config(),
    )
    arguments.update(overrides)
    with pytest.raises(ValueError, match=message):
        ernie55_process_mtp_loss(**arguments)
