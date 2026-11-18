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

"""CPU test of the ERNIE 5.5 training step under context parallelism: CP = 2 and 4 against CP = 1.

:func:`~megatron.bridge.models.ernie.ernie55_step.forward_step` is the training step of the CP64 recipe
(``ernie55_sft_64gpu_gb300_bf16_config``, run with ``--step_func ernie55_step``). On ``CP`` gloo ranks every rank reads
the same collated micro-batch (three unaligned documents whose prompts carry no loss, so the ranks hold different
numbers of supervised tokens); the step pads the documents to ``2 * CP``, builds the rank's contiguous rows with
prebuilt routes, runs the model with the per-row MTP loss and returns the per-row LM loss function. Against the same
model and documents (padded alike) through the step on one rank:

* the reported LM loss summed over the CP group is the CP = 1 row mean;
* every MTP depth's tracked loss, averaged over the CP group as the tracker does, equals the CP = 1 loss;
* every parameter gradient, averaged over the CP group as DDP does, equals the CP = 1 gradient.

A rank-local token mean (the per-row loss without the CP group) must miss the CP = 1 row mean on these documents,
or the test would prove nothing. The schedule's scales for one micro-batch with ``calculate_per_token_loss=False``
are emulated: MTP loss scale 1 and indexer-loss scale ``CP``. The pattern is D-only with one shared-body MTP depth
(``"D-DE/DE"``): H layers at CP > 1 run Megatron-Core DSv4's CuTe DSL kernels. Parameters are fp32.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tests.unit_tests.models.ernie.ernie55_test_utils import allow_conversion_imports_without_te


# The step's lazy imports (FLOP accounting, process groups) pull in Bridge's conversion and training stacks. The
# conversion stack imports its LoRA layers, which need Transformer Engine; without TE, import it with a placeholder
# for them. The training stack imports ``datasets``; without it the module is skipped. The conversion stack must be
# imported first: importing the training stack first runs into a circular import of the HF model classes.
allow_conversion_imports_without_te()
pytest.importorskip("megatron.bridge.models.conversion.model_bridge", reason="needs Bridge's conversion stack")
pytest.importorskip("megatron.bridge.training.utils.train_utils", reason="needs Bridge's training stack")

from megatron.core import parallel_state
from megatron.core.transformer.experimental_attention_variant import dsa
from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler, MTPLossLoggingHelper

from megatron.bridge.models.ernie.ernie55_context_parallel import IGNORE_LABEL, ernie55_per_row_lm_loss
from megatron.bridge.models.ernie.ernie55_data import collate_ernie55_rows
from megatron.bridge.models.ernie.ernie55_step import forward_step
from megatron.bridge.models.ernie.modeling_ernie55 import context_parallel
from megatron.bridge.training.utils import flop_utils
from tests.unit_tests.models.ernie.ernie55_cp_harness import run_cases
from tests.unit_tests.models.ernie.ernie55_test_utils import mtp_tracked_losses
from tests.unit_tests.models.ernie.test_ernie55_cp_model import (  # noqa: F401 - setup_rank runs on spawned ranks
    PAD_TOKEN,
    _all_reduce_sum,
    _error,
    _model_parallel,
    _Rank,
    setup_rank,
)
from tests.unit_tests.models.ernie.test_ernie55_cp_mtp import GRAD_TOL, PATTERN, ROW_TOL, TOPK, _grad_errors, _samples
from tests.unit_tests.models.ernie.test_ernie55_mhc_mtp import _build_model


logger = logging.getLogger(__name__)

pytestmark = pytest.mark.unit


class _Timer:
    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None


class _StragglerTimer:
    def __call__(self, **kwargs) -> "_StragglerTimer":
        return self

    def __enter__(self) -> "_StragglerTimer":
        return self

    def __exit__(self, *args) -> bool:
        return False


def _state() -> SimpleNamespace:
    """The parts of Bridge's ``GlobalState`` the step reads."""
    return SimpleNamespace(
        timers=lambda *args, **kwargs: _Timer(),
        straggler_timer=_StragglerTimer(),
        cfg=SimpleNamespace(
            rerun_state_machine=SimpleNamespace(check_for_nan_in_loss=True, check_for_spiky_loss=False)
        ),
    )


def _collated(documents: list[tuple[torch.Tensor, torch.Tensor]]) -> dict:
    """The data loader's micro-batch of ``documents`` (one packed row, ``cu_seqlens`` and the pad token)."""
    row = [(tokens.tolist(), labels.tolist()) for tokens, labels in documents]
    return collate_ernie55_rows([row], pad_token_id=PAD_TOKEN)


def _padded_documents(documents, multiple: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """``documents`` padded to ``multiple`` the way the step pads them at CP > 1 (pad token, label ``-100``)."""
    padded = []
    for tokens, labels in documents:
        pad = -tokens.numel() % multiple
        padded.append(
            (
                torch.cat([tokens, tokens.new_full((pad,), PAD_TOKEN)]),
                torch.cat([labels, labels.new_full((pad,), IGNORE_LABEL)]),
            )
        )
    return padded


def _step(model, collated: dict, *, cp_size: int) -> dict:
    """One training micro-batch through ``forward_step``, its loss function and the backward."""
    MTPLossLoggingHelper.tracker.clear()
    dsa.DSAIndexerLossLoggingHelper.tracker.clear()
    # The schedule's scales for one micro-batch with calculate_per_token_loss=False.
    MTPLossAutoScaler.set_loss_scale(torch.tensor(1.0))
    dsa.DSAIndexerLossAutoScaler.set_loss_scale(torch.tensor(float(cp_size)))
    model.zero_grad(set_to_none=True)
    flops = []
    with mock.patch.object(
        flop_utils, "accumulate_flops_metadata", lambda state, tokens, **kwargs: flops.append(kwargs)
    ):
        output, loss_function = forward_step(_state(), iter([collated]), model)
    assert len(flops) == 1
    loss, num_tokens, report = loss_function(output)
    assert int(num_tokens) == 1
    loss_mask = loss_function.args[0]
    # A rank-local token mean of the same losses: what the step would report without the CP group.
    rank_local = ernie55_per_row_lm_loss(loss_mask, output.detach(), cp_group=None)[2]["lm loss"]
    loss.backward()
    return dict(
        report=report["lm loss"].detach().clone(),
        rank_local=rank_local.detach().clone(),
        supervised=loss_mask.sum().reshape(1),
        mtp=mtp_tracked_losses().detach().clone(),
        flops=flops[0],
        grads={
            name: param.grad.detach().clone() for name, param in model.named_parameters() if param.grad is not None
        },
    )


def case_step_matches_cp1(rank: _Rank) -> None:
    """``forward_step`` at CP > 1 equals the step at CP = 1: LM row mean, MTP loss and gradients."""
    cp_size = rank.world_size
    seed = 13
    documents = _samples(seed)
    values = dict(
        params_dtype=torch.float32,
        dsa_indexer_topk=TOPK,
        cp_comm_type="all_gather",
        ernie_per_row_loss=True,
        calculate_per_token_loss=False,
    )

    reference = _build_model(_model_parallel(1), pattern=PATTERN, seed=seed, context_parallel_size=1, **values)
    expected = _step(reference, _collated(_padded_documents(documents, 2 * cp_size)), cp_size=1)
    # CP = 1 run-to-run noise: the same documents in reverse order change only the summation order.
    reordered = _step(reference, _collated(_padded_documents(documents[::-1], 2 * cp_size)), cp_size=1)
    noise = _grad_errors(reordered["grads"], expected["grads"], None, 1)
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
        actual = _step(model, _collated(documents), cp_size=cp_size)
        counts = [torch.zeros(1) for _ in range(cp_size)]
        torch.distributed.all_gather(counts, actual["supervised"].float(), group=cp_group)
        assert len({float(count) for count in counts}) > 1, "the ranks must hold different supervised-token counts"
        report = _all_reduce_sum(actual["report"], cp_group)
        rank_local = _all_reduce_sum(actual["rank_local"], cp_group)
        mtp = _all_reduce_sum(actual["mtp"], cp_group) / cp_size
        grad_errors = _grad_errors(actual["grads"], expected["grads"], cp_group, cp_size)
    finally:
        parallel_state.destroy_model_parallel()

    # The FLOP accounting sees the whole padded pack and the CP size, not this rank's rows.
    assert actual["flops"]["context_parallel_size"] == cp_size
    assert actual["flops"]["config_seq_len"] == expected["flops"]["config_seq_len"]
    assert torch.equal(actual["flops"]["cu_seqlens"].cpu(), expected["flops"]["cu_seqlens"].cpu())

    row_mean = expected["report"][0] / expected["report"][1]
    errors = {
        "reported LM row mean vs CP = 1": _error(report[0] / report[1], row_mean),
        **{
            f"depth {depth + 1} tracked MTP loss vs CP = 1": _error(mtp[depth], expected["mtp"][depth])
            for depth in range(expected["mtp"].numel())
        },
    }
    worst = max(grad_errors, key=grad_errors.get)
    rank_local_error = _error(rank_local[0] / cp_size, row_mean)
    logger.info(
        "[ernie55-step-cp rank %d CP%d] %s; worst gradient %s %.2e; rank-local token mean off by %.2e",
        rank.rank,
        cp_size,
        {key: f"{value:.2e}" for key, value in errors.items()},
        worst,
        grad_errors[worst],
        rank_local_error,
    )
    failures = [f"{what}: {error:.3e} > {ROW_TOL:.0e}" for what, error in errors.items() if error > ROW_TOL]
    # A gradient whose own CP = 1 noise exceeds the bar (a near-zero scalar sum) is held to 3x that noise.
    failures += [
        f"gradient of {name}: {error:.3e} > max({GRAD_TOL:.0e}, 3 x CP = 1 noise {noise[name]:.3e})"
        for name, error in grad_errors.items()
        if error > max(GRAD_TOL, 3 * noise[name])
    ]
    if rank_local_error < 100 * ROW_TOL:
        failures.append(f"a rank-local token mean matched the CP = 1 row mean too ({rank_local_error:.3e})")
    assert not failures, "\n".join(failures)


CASES = ["case_step_matches_cp1"]


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
        workdir=tmp_path_factory.mktemp(f"ernie55_step_cp{request.param}"),
    )


@pytest.mark.parametrize("case", CASES, ids=[case.removeprefix("case_") for case in CASES])
def test_forward_step_under_context_parallelism(cp_results, case):
    failures = {rank: result["error"] for rank, result in cp_results[case].items() if not result["ok"]}
    assert len(cp_results[case]) > 1, "no rank reported"
    assert not failures, "\n".join(f"--- rank {rank} ---\n{error}" for rank, error in failures.items())
