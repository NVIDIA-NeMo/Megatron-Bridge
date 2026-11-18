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

"""CPU tests of the ERNIE 5.5 D-layer Transformer Engine CUDA graphs (``ernie_graph_d_layers``,
``ernie_graph_mtp_layers``; ``modeling_ernie55.cuda_graph``).

Nothing here captures a CUDA graph (``test_ernie55_d_cuda_graph_gpu.py`` does). A "simulated capture" runs a wrapper's
capture function on CPU with :func:`~megatron.bridge.models.ernie.modeling_ernie55.ops.stream_is_capturing`
answering ``True``, so every host-planned fallback that a real capture forbids raises instead of running:

* D1, the configuration envelope of the two switches, and that both are off by default;
* D2, which wrappers of a CPU-built ``H-DEHEDE/DE`` model become D-graph wrappers (decoder, MTP depth), with the same
  module tree, parameters and checkpoint keys, and that nothing changes with the switches off;
* D3, the discovery filter: Megatron-Core's discovery of that model keeps the D wrappers (and the MTP depth's, marked
  as an MTP callable, right after the decoder's last D layer, whose parameters it shares), and parameters shared by
  graphed layers that are not neighbors, or by three of them, are rejected;
* D4, static inputs and host plan: fp32 decoder streams and activation-dtype MTP streams, the plan of one document
  over the whole row, attached to the metadata Megatron-Core rebuilds for a capture, released with the graphs; the
  indexer-loss scale created on the device before the capture; a tracker without fixed capacity rejected;
* D5, a simulated capture of every D wrapper (decoder and MTP depth) at CP = 1 gives the eager forward's output and
  gradients bitwise, also with the batched VHA head mix (``ernie_vha_batched_mix``), which the graphs then capture;
* D6, the capture guards: the D attention under a capture without its wrapper's plan, ``pack_end`` and the zigzag
  geometry without their host notes, and an inverse key permutation asked for but not prebuilt, raise; the batch
  builder notes the boundaries (CP = 1) and prebuilds the zigzag geometry (CP > 1), with the inverse key
  permutation on request (``ernie_cp_key_gather_inverse_grad``), which the D layers then only look up;
* D7, replay through a whole model's forward: a D graph gets exactly its captured inputs for a micro-batch of the
  captured composition; in a decoder of D layers alone, another composition, another ``max_seqlen`` or metadata
  without host boundaries run the D layers eagerly, bitwise like a model without graphs, with one warning per reason;
  with H layers, the H graphs reject another document count or ``max_seqlen`` first, and only the metadata without
  host boundaries reaches the D layers' fallback;
* D8, the SD-MTP shared gradients: every graphed wrapper (H, D and the MTP depth's D) replaced by a CPU emulation of a
  Transformer Engine graph (its own autograd graph, parameter gradients returned by the graph's backward, as
  Transformer Engine's ``Graphed`` function does), the whole model's loss and every parameter gradient, the
  parameters the MTP depth shares with the decoder's last D layer included, equal the eager ones bitwise;
* D9, on ``CP`` = 2 and 4 spawned gloo ranks (``ernie55_cp_harness``): the plan's route and geometry equal the batch
  builder's, a simulated capture of every D wrapper gives the eager output and gradients bitwise, also with
  ``ernie_cp_key_gather_inverse_grad`` (the plan pins the inverse key permutation, and both key gathers take the
  inverse backward in the eager and the captured run) and with it and ``ernie_vha_batched_mix`` together, and
  replay and fallback work as at CP = 1;
* D10, Megatron-Core DDP's gradient-ready counts with ``overlap_grad_reduce`` (its real bucket-group bookkeeping, the
  collective replaced by a recorder): a step whose shared parameter's hook fires fewer times than in the first batch
  never reduces, and ``Ernie55TECudaGraphHelper.create_cudagraphs`` makes the first graphed batch learn the counts
  again, after which every graphed step reduces.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from megatron.core.models.hybrid.hybrid_block import HyperConnectionHybridLayer
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.experimental_attention_variant import dsa

from megatron.bridge.models.ernie.ernie55_config import Ernie55TransformerConfig
from megatron.bridge.models.ernie.ernie55_context_parallel import IGNORE_LABEL, build_ernie55_thd_cp_batch
from megatron.bridge.models.ernie.modeling_ernie55 import (
    Ernie55DHyperConnectionHybridLayer,
    Ernie55HCAHyperConnectionHybridLayer,
    Ernie55LatentMQASelfAttention,
    Ernie55MTPDHyperConnectionHybridLayer,
    Ernie55TECudaGraphHelper,
    context_parallel,
    cuda_graph,
    ops,
    use_ernie55_d_layer_cuda_graphs,
    validate_ernie55_cuda_graph_config,
)
from tests.unit_tests.models.ernie.ernie55_cp_harness import run_cases
from tests.unit_tests.models.ernie.ernie55_test_utils import record_batched_vha_mixes
from tests.unit_tests.models.ernie.test_ernie55_cuda_graph import (
    _TE_ATTENTION_GRAPHS,
    _graph_config,
    _h_wrapper,
    _helper_with_discovery,
)
from tests.unit_tests.models.ernie.test_ernie55_mhc_mtp import HIDDEN as MODEL_HIDDEN
from tests.unit_tests.models.ernie.test_ernie55_mhc_mtp import (
    SEQ,
    STREAMS,
    VOCAB,
    _build_model,
    _config,
    needs_hook,
    pg_collection,  # noqa: F401 - the CPU model builder's fixture, requested by name
)


pytestmark = pytest.mark.unit

_LOGGER = "megatron.bridge.models.ernie.modeling_ernie55.cuda_graph"
_PACKED_FIELDS = ("cu_seqlens_q", "cu_seqlens_kv", "cu_seqlens_q_padded", "cu_seqlens_kv_padded")
# The H-layer graphs of test_ernie55_cuda_graph.py (CP = 1: max_seqlen_per_dp_cp_rank = SEQ) plus the D layers'.
_D_GRAPHS = {**_TE_ATTENTION_GRAPHS, "ernie_graph_d_layers": True}
_D_AND_MTP_GRAPHS = {**_D_GRAPHS, "ernie_graph_mtp_layers": True}
# H at 1, D at 3 and 7, MoE at 4, 6 and 8; the MTP depth's DE shares the decoder's last DE (layers 7 and 8).
_PATTERN = "H-DEHEDE/DE"
# Without H layers (whose graphs reject rows of another document count first), so other compositions reach the D
# layers.
_D_PATTERN = "D-DE/DE"


@pytest.fixture(autouse=True)
def _cpu_state(monkeypatch):
    """CPU stand-ins for NVTX, fresh DSA trackers and loss scale, and no logged fallbacks, around every test."""
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(cuda_graph, "_WARNED_FALLBACKS", set())
    dsa.DSAIndexerLossLoggingHelper.tracker.clear()
    dsa.DSAIndexerLossAutoScaler.main_loss_backward_scale = None
    yield
    dsa.DSAIndexerLossLoggingHelper.tracker.clear()
    dsa.DSAIndexerLossAutoScaler.main_loss_backward_scale = None


@contextlib.contextmanager
def _simulated_capture():
    """Make ``ops.stream_is_capturing`` answer ``True``: the host-planned fallbacks then raise, as in a capture."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ops, "stream_is_capturing", lambda: True)
        yield


def _fixed_tracker(model, pg_collection) -> None:
    """The fixed-capacity DSA tracker Bridge's setup initializes (a captured D layer writes into it in place)."""
    from megatron.core.transformer.experimental_attention_variant.dsa_logging import initialize_dsa_metric_tracker

    dsa.DSAIndexerLossLoggingHelper.tracker.clear()
    initialize_dsa_metric_tracker([model], pg_collection)


def _publish_indexer_loss_scale(device, *, cp_size: int, num_microbatches: int) -> None:
    """Publish the indexer-loss scale with Megatron-Core's own schedule code, as training does after every forward.

    ``forward_step_calc_loss`` of a DSv4 hybrid stack (no ``grad_scale_func``, per-row loss) passes
    ``set_loss_scale`` a size-1 ``_compute_loss_scale() * cp_size / num_microbatches``; the GPU suite uses this too.
    """
    from megatron.core.pipeline_parallel import schedules

    config = SimpleNamespace(
        timers=None,
        num_moe_experts=None,
        mtp_num_layers=None,
        calculate_per_token_loss=False,
        grad_scale_func=None,
        experimental_attention_variant="dsv4_hybrid",
        experimental_attention_variant_loss_scale_func=None,
    )
    schedules.forward_step_calc_loss(
        None,
        torch.zeros((), device=device),
        lambda output: (output, {}),
        config,
        None,
        False,
        num_microbatches,
        [],
        cp_group_size=cp_size,
        is_last_stage=True,
    )


def _one_document_batch(seed: int = 7, lengths: tuple[int, ...] = (SEQ,), **kwargs):
    """A packed row of documents of ``lengths`` tokens (one document of SEQ by default) at CP = 1."""
    generator = torch.Generator().manual_seed(seed)
    samples = []
    for length in lengths:
        tokens = torch.randint(1, VOCAB, (length,), generator=generator)
        labels = tokens.roll(-1)
        labels[-1] = IGNORE_LABEL
        samples.append((tokens, labels))
    return build_ernie55_thd_cp_batch(samples, pad_token_id=0, cp_group=None, **kwargs)


def _graphed_wrappers(model) -> list:
    """The decoder's and MTP depths' wrappers of the graphed classes, in forward order."""
    layers = list(model.decoder.layers)
    if getattr(model, "mtp", None) is not None:
        layers += [layer for depth in model.mtp.layers for layer in depth.mtp_model_layer.layers]
    graphed = (Ernie55HCAHyperConnectionHybridLayer, Ernie55DHyperConnectionHybridLayer)
    return [layer for layer in layers if isinstance(layer, graphed)]


def _model_forward(model, batch):
    return model(
        input_ids=batch.tokens,
        position_ids=batch.position_ids,
        attention_mask=None,
        labels=batch.labels.clamp(min=0),
        loss_mask=batch.loss_mask,
        packed_seq_params=batch.packed_seq_params,
    )


# ---------------------------------------------------------------------------------------------
# D1: configuration envelope of the switches
# ---------------------------------------------------------------------------------------------


class TestEnvelope:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"ernie_graph_d_layers": True},
            {"ernie_graph_d_layers": True, "context_parallel_size": 64},
            # D graphs give a decoder without H layers something to graph.
            {"ernie_graph_d_layers": True, "csa_compress_ratios": [0, 0]},
            {
                "ernie_graph_d_layers": True,
                "ernie_graph_mtp_layers": True,
                "mtp_num_layers": 1,
                "ernie_mtp_shared_last_layer": True,
            },
            # Without SD-MTP the MTP depths share nothing with the decoder: any number of them.
            {
                "ernie_graph_d_layers": True,
                "ernie_graph_mtp_layers": True,
                "mtp_num_layers": 2,
                "ernie_mtp_shared_last_layer": False,
            },
            # Off switches with graphs off are the default.
            {"cuda_graph_impl": "none", "ernie_graph_d_layers": False, "ernie_graph_mtp_layers": False},
        ],
        ids=lambda overrides: str(overrides),
    )
    def test_accepts(self, overrides):
        for layer in ("H", "D"):
            validate_ernie55_cuda_graph_config(_graph_config(**overrides), layer=layer)

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"cuda_graph_impl": "none", "ernie_graph_d_layers": True}, "are off .*but ernie_graph_d_layers"),
            ({"cuda_graph_impl": "none", "ernie_graph_mtp_layers": True}, "are off .*ernie_graph_mtp_layers"),
            ({"ernie_graph_mtp_layers": True, "mtp_num_layers": 1}, "need ernie_graph_d_layers=True"),
            ({"ernie_graph_d_layers": True, "thd_max_packed_sequences": 2}, "need thd_max_packed_sequences=1, got 2"),
            (
                {"ernie_graph_d_layers": True, "ernie_graph_mtp_layers": True, "mtp_num_layers": None},
                "need an MTP",
            ),
            (
                {
                    "ernie_graph_d_layers": True,
                    "ernie_graph_mtp_layers": True,
                    "mtp_num_layers": 2,
                    "ernie_mtp_shared_last_layer": True,
                },
                "support one SD-MTP depth",
            ),
            # The rest of the envelope applies as it does to the H graphs.
            ({"ernie_graph_d_layers": True, "use_te_rng_tracker": False}, "use_te_rng_tracker=True"),
            ({"ernie_graph_d_layers": True, "experimental_attention_variant": "dsa"}, "dsv4_hybrid"),
            ({"ernie_graph_d_layers": True, "cuda_graph_modules": []}, r"need cuda_graph_modules=\['attn'\]"),
        ],
        ids=lambda value: str(value) if isinstance(value, dict) else None,
    )
    def test_rejects(self, overrides, message):
        for layer in ("H", "D"):
            with pytest.raises(ValueError, match=f"^ERNIE 5.5 {layer} layer: CUDA graphs .*{message}"):
                validate_ernie55_cuda_graph_config(_graph_config(**overrides), layer=layer)

    def test_without_the_switch_a_decoder_needs_an_h_layer(self):
        """The H-graph rule is unchanged with the switches off."""
        with pytest.raises(ValueError, match="need an H layer in the decoder"):
            validate_ernie55_cuda_graph_config(_graph_config(csa_compress_ratios=[0, 0]), layer="D")

    def test_switches_default_off(self):
        import dataclasses

        defaults = {field.name: field.default for field in dataclasses.fields(Ernie55TransformerConfig)}
        assert defaults["ernie_graph_d_layers"] is False and defaults["ernie_graph_mtp_layers"] is False

    @pytest.mark.parametrize("name", ["ernie_graph_d_layers", "ernie_graph_mtp_layers"])
    @pytest.mark.parametrize("value", ["true", 1, None])
    def test_finalize_requires_bools(self, name, value):
        with pytest.raises(ValueError, match=f"{name} must be a bool"):
            _config("H-DE", ernie_mtp_shared_last_layer=False, **{name: value})

    def test_finalize_accepts_a_decoder_of_d_layers_with_the_switch(self):
        config = _config("D-DE", ernie_mtp_shared_last_layer=False, **_D_GRAPHS)
        assert config.ernie_graph_d_layers is True
        with pytest.raises(ValueError, match="need an H layer in the decoder"):
            _config("D-DE", ernie_mtp_shared_last_layer=False, **_TE_ATTENTION_GRAPHS)


# ---------------------------------------------------------------------------------------------
# D2: which wrappers become D-graph wrappers
# ---------------------------------------------------------------------------------------------


@needs_hook
class TestConversion:
    @staticmethod
    def _wrappers(model) -> tuple[list, list]:
        decoder = list(model.decoder.layers)
        mtp = [layer for depth in model.mtp.layers for layer in depth.mtp_model_layer.layers]
        return decoder, mtp

    @pytest.mark.parametrize("mtp_graphs", [False, True], ids=["decoder", "decoder-and-mtp"])
    def test_converts_exactly_the_switched_d_wrappers(self, pg_collection, mtp_graphs):
        plain = _build_model(pg_collection, pattern=_PATTERN)
        graphed = _build_model(pg_collection, pattern=_PATTERN, **(_D_AND_MTP_GRAPHS if mtp_graphs else _D_GRAPHS))
        decoder, mtp = self._wrappers(graphed)
        expected = {"H": Ernie55HCAHyperConnectionHybridLayer, "D": Ernie55DHyperConnectionHybridLayer}
        for index, layer in enumerate(decoder):
            assert type(layer) is expected.get(_PATTERN[index], HyperConnectionHybridLayer), index
        mtp_d = Ernie55MTPDHyperConnectionHybridLayer if mtp_graphs else HyperConnectionHybridLayer
        assert [type(layer) for layer in mtp] == [mtp_d, HyperConnectionHybridLayer]
        # Same module tree, parameter order, values and checkpoint keys as without graphs.
        assert [name for name, _ in graphed.named_modules()] == [name for name, _ in plain.named_modules()]
        plain_params = dict(plain.named_parameters())
        assert list(dict(graphed.named_parameters())) == list(plain_params)
        for name, param in graphed.named_parameters():
            assert torch.equal(param, plain_params[name]), name
        assert list(graphed.state_dict()) == list(plain.state_dict())
        # The MTP depth's D layer still computes with the decoder's last D layer's parameters.
        assert mtp[0].inner_layer.self_attention.k_b_proj is decoder[6].inner_layer.self_attention.k_b_proj
        # Idempotent; nested stacks are not visited.
        assert use_ernie55_d_layer_cuda_graphs(graphed.decoder) == 2
        assert use_ernie55_d_layer_cuda_graphs(graphed.mtp.layers[0].mtp_model_layer, mtp=mtp_graphs) == 1

    def test_d_graphs_have_no_offload_boundary(self, pg_collection, monkeypatch):
        """AbsorbedMLA offloads nothing, so a D graph has no offload boundary, even with the H layers' scopes."""
        from megatron.core.transformer import transformer_layer

        # Megatron-Core asks for Transformer Engine 2.14+ when a layer advertises an offload boundary in a graph.
        monkeypatch.setattr(transformer_layer, "is_te_min_version", lambda *args, **kwargs: True)
        model = _build_model(
            pg_collection,
            pattern=_PATTERN,
            fine_grained_activation_offloading=True,
            offload_modules=["core_attn", "attn_proj"],
            **_D_AND_MTP_GRAPHS,
        )
        h, d = model.decoder.layers[0], model.decoder.layers[2]
        mtp_d = model.mtp.layers[0].mtp_model_layer.layers[0]
        # Megatron-Core's config-level answer claims one for the D layer too; the DSv4 H attention does offload.
        assert d.inner_layer.offload_scope_in_cuda_graph(require_concrete_modules=True) is True
        assert h.offload_module_in_cuda_graph is True
        assert d.offload_module_in_cuda_graph is False and mtp_d.offload_module_in_cuda_graph is False

    def test_without_the_switch_d_wrappers_keep_their_class(self, pg_collection):
        model = _build_model(pg_collection, pattern=_PATTERN, **_TE_ATTENTION_GRAPHS)
        decoder, mtp = self._wrappers(model)
        assert [type(layer).__name__ for layer in decoder if cuda_graph._is_d_wrapper(layer)] == [
            "HyperConnectionHybridLayer"
        ] * 2
        assert type(mtp[0]) is HyperConnectionHybridLayer

    def test_conversion_rejects_unknown_subclasses_and_the_other_kind(self, pg_collection):
        model = _build_model(pg_collection, pattern=_PATTERN, **_D_GRAPHS)
        with pytest.raises(TypeError, match="converts Megatron-Core's HyperConnectionHybridLayer only"):
            use_ernie55_d_layer_cuda_graphs(model.decoder, mtp=True)

        class OtherWrapper(HyperConnectionHybridLayer):
            pass

        model.decoder.layers[2].__class__ = OtherWrapper
        with pytest.raises(TypeError, match="cannot give the D-layer wrapper OtherWrapper"):
            use_ernie55_d_layer_cuda_graphs(model.decoder)


# ---------------------------------------------------------------------------------------------
# D3: discovery filter
# ---------------------------------------------------------------------------------------------


def _d_graph_wrapper(layer_number: int, *, mtp: bool = False, anchor: torch.nn.Parameter | None = None):
    """A converted D wrapper without the layer: an ERNIE D attention (only its type is checked), one parameter."""
    cls = Ernie55MTPDHyperConnectionHybridLayer if mtp else Ernie55DHyperConnectionHybridLayer
    wrapper = cls.__new__(cls)
    torch.nn.Module.__init__(wrapper)
    wrapper.inner_layer = SimpleNamespace(
        self_attention=Ernie55LatentMQASelfAttention.__new__(Ernie55LatentMQASelfAttention)
    )
    wrapper.layer_number = layer_number
    wrapper.anchor = anchor if anchor is not None else torch.nn.Parameter(torch.zeros(2))
    return wrapper


class TestDiscovery:
    @staticmethod
    def _discover(monkeypatch, items, **config):
        mtp_params = [layer.anchor for layer, is_mtp in items if is_mtp]
        chunk = SimpleNamespace(mtp=SimpleNamespace(parameters=lambda: iter(mtp_params)))
        helper = _helper_with_discovery(monkeypatch, [(chunk, items)])
        helper.config = SimpleNamespace(pipeline_model_parallel_size=1, **config)
        helper._discover_layers()
        return helper

    def test_keeps_the_switched_wrappers_in_forward_order(self, monkeypatch):
        h1, d3, h5 = _h_wrapper(layer_number=1), _d_graph_wrapper(3), _h_wrapper(layer_number=5)
        d7 = _d_graph_wrapper(7)
        mtp_d = _d_graph_wrapper(9, mtp=True, anchor=d7.anchor)
        items = [(h1, False), (d3, False), (h5, False), (d7, False), (mtp_d, True)]
        helper = self._discover(monkeypatch, items, ernie_graph_d_layers=True, ernie_graph_mtp_layers=True)
        assert helper.callables_per_chunk == [[h1, d3, h5, d7, mtp_d]]
        assert helper.callables_per_chunk_is_mtp == [[False, False, False, False, True]]
        assert helper.flattened_callables_is_mtp == [False, False, False, False, True]
        assert helper.num_layers_per_chunk == [5]
        # The MTP depth's D layer stays eager without its switch; the decoder's D layers do not need it.
        helper = self._discover(monkeypatch, items, ernie_graph_d_layers=True)
        assert helper.callables_per_chunk == [[h1, d3, h5, d7]]
        assert helper.callables_per_chunk_is_mtp == [[False] * 4]

    def test_shared_parameters_need_two_neighboring_graphs(self, monkeypatch):
        shared = torch.nn.Parameter(torch.zeros(2))
        d3, h5 = _d_graph_wrapper(3, anchor=shared), _h_wrapper(layer_number=5)
        mtp_d = _d_graph_wrapper(9, mtp=True, anchor=shared)
        with pytest.raises(RuntimeError, match=r"shared by the graphed layers \[3, 9\] \(positions \[0, 2\]"):
            self._discover(
                monkeypatch,
                [(d3, False), (h5, False), (mtp_d, True)],
                ernie_graph_d_layers=True,
                ernie_graph_mtp_layers=True,
            )
        three = [(_d_graph_wrapper(3, anchor=shared), False), (_d_graph_wrapper(7, anchor=shared), False)]
        with pytest.raises(RuntimeError, match="exactly two adjacent graphs"):
            self._discover(
                monkeypatch, three + [(mtp_d, True)], ernie_graph_d_layers=True, ernie_graph_mtp_layers=True
            )

    def test_an_unconverted_d_wrapper_is_rejected_with_its_switch(self, monkeypatch):
        from tests.unit_tests.models.ernie.test_ernie55_cuda_graph import _d_wrapper

        with pytest.raises(TypeError, match="Ernie55HybridStack converts it to Ernie55DHyperConnectionHybridLayer"):
            self._discover(monkeypatch, [(_d_wrapper(3), False)], ernie_graph_d_layers=True)
        mtp_d = _d_wrapper(9)
        with pytest.raises(TypeError, match="Ernie55HybridModel converts it to Ernie55MTPDHyperConnectionHybridLayer"):
            self._discover(
                monkeypatch,
                [(_d_graph_wrapper(3), False), (mtp_d, True)],
                ernie_graph_d_layers=True,
                ernie_graph_mtp_layers=True,
            )

    def test_nothing_to_graph(self, monkeypatch):
        mtp_d = _d_graph_wrapper(9, mtp=True)
        with pytest.raises(RuntimeError, match="no H or D attention layer to graph"):
            self._discover(monkeypatch, [(mtp_d, True)], ernie_graph_d_layers=True)

    @needs_hook
    @pytest.mark.parametrize("mtp_graphs", [False, True], ids=["decoder", "decoder-and-mtp"])
    def test_megatron_core_discovery_of_a_model(self, pg_collection, mtp_graphs):
        """Megatron-Core's own discovery of a CPU ``H-DEHEDE/DE`` model (SD-MTP), then the filter."""
        from megatron.core.transformer import cuda_graphs as mcore_cuda_graphs

        model = _build_model(pg_collection, pattern=_PATTERN, **(_D_AND_MTP_GRAPHS if mtp_graphs else _D_GRAPHS))
        helper = Ernie55TECudaGraphHelper.__new__(Ernie55TECudaGraphHelper)
        helper.model, helper.config = [model], model.config
        helper.tp_group, helper.dp_cp_group = pg_collection.tp, pg_collection.dp_cp
        mcore_cuda_graphs.TECudaGraphHelper._discover_layers(helper)
        helper._discover_layers()
        decoder = model.decoder.layers
        expected = [decoder[0], decoder[2], decoder[4], decoder[6]]
        if mtp_graphs:
            expected.append(model.mtp.layers[0].mtp_model_layer.layers[0])
        assert helper.callables_per_chunk == [expected]
        assert helper.callables_per_chunk_is_mtp == [[False] * 4 + [True] * mtp_graphs]
        assert helper.flattened_callables == expected
        assert helper.chunks_with_decoder[0] is model


# ---------------------------------------------------------------------------------------------
# D4: static inputs and host plan
# ---------------------------------------------------------------------------------------------


@needs_hook
class TestStaticInputsAndPlan:
    def test_static_inputs_and_plan(self, pg_collection):
        model = _build_model(pg_collection, pattern=_PATTERN, **_D_AND_MTP_GRAPHS)
        _fixed_tracker(model, pg_collection)
        decoder_d = model.decoder.layers[2]
        mtp_d = model.mtp.layers[0].mtp_model_layer.layers[0]
        width = STREAMS * MODEL_HIDDEN
        for wrapper, dtype in ((decoder_d, torch.float32), (mtp_d, torch.bfloat16)):
            inputs = wrapper.get_layer_static_inputs(SEQ, 1)
            assert list(inputs) == ["hidden_states", *_PACKED_FIELDS]
            hidden = inputs["hidden_states"]
            assert hidden.dtype == dtype and tuple(hidden.shape) == (SEQ, 1, width) and hidden.requires_grad
            for name in _PACKED_FIELDS:
                assert inputs[name].dtype == torch.int32 and inputs[name].tolist() == [0, SEQ], name
            plan = wrapper.__dict__[cuda_graph._D_GRAPH_PLAN]
            assert (plan.boundaries, plan.max_seqlen, plan.num_boundaries, plan.cp_size) == ((0, SEQ), SEQ, 2, 1)
            assert ops.noted_pack_boundaries(plan.template) == (0, SEQ)
            # Built once per device: a second request reuses it.
            wrapper.get_layer_static_inputs(SEQ, 1)
            assert wrapper.__dict__[cuda_graph._D_GRAPH_PLAN] is plan

            # The metadata Megatron-Core rebuilds for a capture carries the plan.
            kwargs = {name: inputs[name] for name in _PACKED_FIELDS}
            wrapper._reconstruct_packed_seq_params_from_kwargs(kwargs)
            params = kwargs["packed_seq_params"]
            assert getattr(params, cuda_graph.D_GRAPH_PLAN_ATTR) is plan
            assert ops.noted_pack_boundaries(params) == (0, SEQ) and ops.pack_end(params) == SEQ
            assert (params.max_seqlen_q, params.max_seqlen_kv) == (SEQ, SEQ)

            # Megatron-Core's graph deletion releases the plan.
            wrapper.clear_te_cuda_graph_static_hidden_inputs()
            assert cuda_graph._D_GRAPH_PLAN not in wrapper.__dict__

    def test_capture_without_the_plan_raises(self, pg_collection):
        model = _build_model(pg_collection, pattern=_PATTERN, **_D_GRAPHS)
        wrapper = model.decoder.layers[2]
        cu_seqlens = torch.tensor([0, SEQ], dtype=torch.int32)
        kwargs = {name: cu_seqlens.clone() for name in _PACKED_FIELDS}
        with pytest.raises(RuntimeError, match="captured without its host plan"):
            wrapper._reconstruct_packed_seq_params_from_kwargs(kwargs)

    def test_static_inputs_need_one_document(self, pg_collection):
        model = _build_model(pg_collection, pattern=_PATTERN, **_D_GRAPHS)
        _fixed_tracker(model, pg_collection)
        wrapper = model.decoder.layers[2]
        wrapper.config.thd_max_packed_sequences = 2
        try:
            with pytest.raises(ValueError, match="needs thd_max_packed_sequences=1, got 2"):
                wrapper.get_layer_static_inputs(SEQ, 1)
        finally:
            wrapper.config.thd_max_packed_sequences = 1

    def test_indexer_loss_bookkeeping_is_made_capturable(self, pg_collection):
        model = _build_model(pg_collection, pattern=_PATTERN, **_D_GRAPHS)
        wrapper = model.decoder.layers[2]
        # A lazily grown tracker could be rebound after the capture.
        with pytest.raises(RuntimeError, match="tracker has no fixed capacity"):
            wrapper.get_layer_static_inputs(SEQ, 1)
        _fixed_tracker(model, pg_collection)
        scaler = dsa.DSAIndexerLossAutoScaler
        # A capture before the first step (cuda_graph_warmup_steps=0): the schedule has not set the scale yet.
        assert scaler.main_loss_backward_scale is None
        wrapper.get_layer_static_inputs(SEQ, 1)
        scale = scaler.main_loss_backward_scale
        assert isinstance(scale, torch.Tensor) and scale.device == next(wrapper.parameters()).device
        assert scale.dtype == torch.float32 and tuple(scale.shape) == (1,) and scale.item() == 1.0
        # Megatron-Core's schedule then publishes its size-1 scale (loss scale x CP / micro-batches) after the first
        # forward by copying it into that tensor, which a captured backward reads in place.
        _publish_indexer_loss_scale(scale.device, cp_size=2, num_microbatches=4)
        assert scaler.main_loss_backward_scale is scale and tuple(scale.shape) == (1,) and scale.item() == 0.5
        # An existing scale is kept.
        wrapper.get_layer_static_inputs(SEQ, 1)
        assert scaler.main_loss_backward_scale is scale


# ---------------------------------------------------------------------------------------------
# D5: a simulated capture of every D wrapper equals its eager forward (CP = 1)
# ---------------------------------------------------------------------------------------------


def _run_wrapper(call, hidden: torch.Tensor, grad_output: torch.Tensor, wrapper) -> dict[str, torch.Tensor]:
    """Forward and backward of ``call(hidden)``; the output, the input gradient and every parameter gradient."""
    wrapper.zero_grad(set_to_none=True)
    inputs = hidden.detach().clone().requires_grad_(True)
    output = call(inputs)
    output.backward(grad_output)
    tensors = {"output": output.detach().clone(), "input": inputs.grad.detach().clone()}
    tensors.update(
        {name: param.grad.detach().clone() for name, param in wrapper.named_parameters() if param.grad is not None}
    )
    return tensors


def _assert_bitwise(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor], label: str) -> None:
    assert actual.keys() == expected.keys(), f"{label}: {sorted(actual.keys() ^ expected.keys())}"
    different = [name for name in expected if not torch.equal(actual[name], expected[name])]
    assert not different, f"{label}: not bitwise equal: {different}"


def _capture_matches_eager(wrapper, batch, *, seed: int, label: str) -> None:
    """A simulated capture call on the wrapper's static inputs against its eager forward on the micro-batch."""
    static_inputs = wrapper.get_layer_static_inputs(wrapper.config.max_seqlen_per_dp_cp_rank * _cp_size(wrapper), 1)
    static_hidden = static_inputs.pop("hidden_states")
    generator = torch.Generator().manual_seed(seed)
    hidden = (0.02 * torch.randn(static_hidden.shape, generator=generator)).to(static_hidden.dtype)
    grad_output = torch.randn(static_hidden.shape, generator=generator).to(static_hidden.dtype)

    def eager(inputs):
        output, context = wrapper(hidden_states=inputs, attention_mask=None, packed_seq_params=batch.packed_seq_params)
        assert context is None
        return output

    def captured(inputs):
        with _simulated_capture():
            outputs = wrapper._te_cuda_graph_capture(inputs, **static_inputs)
        assert isinstance(outputs, tuple) and len(outputs) == 1
        return outputs[0]

    expected = _run_wrapper(eager, hidden, grad_output, wrapper)
    actual = _run_wrapper(captured, hidden, grad_output, wrapper)
    _assert_bitwise(actual, expected, label)
    assert actual["output"].dtype == static_hidden.dtype


def _cp_size(wrapper) -> int:
    return wrapper.config.context_parallel_size


@needs_hook
def test_simulated_capture_matches_eager(pg_collection):
    """D5: every D wrapper (the decoder's two, the MTP depth's) at CP = 1, from the plan, not the batch's notes."""
    model = _build_model(pg_collection, pattern=_PATTERN, **_D_AND_MTP_GRAPHS)
    _fixed_tracker(model, pg_collection)
    batch = _one_document_batch()
    wrappers = [model.decoder.layers[2], model.decoder.layers[6], model.mtp.layers[0].mtp_model_layer.layers[0]]
    for index, wrapper in enumerate(wrappers):
        _capture_matches_eager(wrapper, batch, seed=11 + index, label=f"{type(wrapper).__name__} {index}")


@needs_hook
def test_simulated_capture_matches_eager_with_the_batched_vha_mix(pg_collection, monkeypatch):
    """D5 with ``ernie_vha_batched_mix``: every D wrapper captures the batched VHA head mix, bitwise like eager."""
    model = _build_model(pg_collection, pattern=_PATTERN, ernie_vha_batched_mix=True, **_D_AND_MTP_GRAPHS)
    _fixed_tracker(model, pg_collection)
    batch = _one_document_batch()
    calls = record_batched_vha_mixes(monkeypatch)
    wrappers = [model.decoder.layers[2], model.decoder.layers[6], model.mtp.layers[0].mtp_model_layer.layers[0]]
    for index, wrapper in enumerate(wrappers):
        calls.clear()
        _capture_matches_eager(wrapper, batch, seed=41 + index, label=f"batched VHA {type(wrapper).__name__} {index}")
        # The D core's mix ran batched in the eager forward and in the captured one.
        assert len(calls) == 2, calls


# ---------------------------------------------------------------------------------------------
# D6: capture guards and the batch builder's host notes
# ---------------------------------------------------------------------------------------------


@needs_hook
def test_d_attention_refuses_a_capture_without_its_wrappers_plan(pg_collection):
    model = _build_model(pg_collection, pattern=_PATTERN, **_D_GRAPHS)
    attention = model.decoder.layers[2].inner_layer.self_attention
    batch = _one_document_batch()
    with _simulated_capture(), pytest.raises(RuntimeError, match="captured only through their graph wrapper"):
        attention.forward(hidden_states=None, attention_mask=None, packed_seq_params=batch.packed_seq_params)
    # The MTP depth's D attention follows its own switch.
    mtp_attention = model.mtp.layers[0].mtp_model_layer.layers[0].inner_layer.self_attention
    with _simulated_capture(), pytest.raises(RuntimeError, match="set ernie_graph_mtp_layers=True"):
        mtp_attention.forward(hidden_states=None, attention_mask=None, packed_seq_params=batch.packed_seq_params)


class TestHostNotes:
    def test_pack_end_does_not_read_the_device_under_a_capture(self):
        cu = torch.tensor([0, 3, 9], dtype=torch.int32)
        params = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=6, max_seqlen_kv=6)
        with _simulated_capture(), pytest.raises(RuntimeError, match="not allowed inside a CUDA graph capture"):
            ops.pack_end(params)
        ops.record_pack_boundaries(params, [0, 3, 9])
        with _simulated_capture():
            assert ops.pack_end(params) == 9
        assert ops.noted_pack_boundaries(params) == (0, 3, 9)
        # Another boundary tensor is not covered by the note.
        params.cu_seqlens_q = params.cu_seqlens_kv = torch.tensor([0, 9], dtype=torch.int32)
        assert ops.noted_pack_boundaries(params) is None
        with pytest.raises(ValueError, match="at least one boundary"):
            ops.record_pack_boundaries(params, [])

    def test_batch_builder_notes_the_boundaries_at_cp1(self):
        batch = _one_document_batch(lengths=(180, 120))
        assert ops.noted_pack_boundaries(batch.packed_seq_params) == (0, 180, 300)
        assert ops.pack_end(batch.packed_seq_params) == 300

    @staticmethod
    def _group(rank: int, size: int = 2) -> SimpleNamespace:
        return SimpleNamespace(size=lambda: size, rank=lambda: rank)

    def test_batch_builder_prebuilds_the_zigzag_geometry(self, pg_collection):
        if context_parallel.load_cp_layout_api() is None:
            pytest.skip("Megatron-Core main has no zigzag CP helpers")
        generator = torch.Generator().manual_seed(0)
        samples = [(torch.randint(1, VOCAB, (length,), generator=generator),) * 2 for length in (24, 40)]
        for rank in range(2):
            batch = build_ernie55_thd_cp_batch(
                samples, pad_token_id=0, cp_group=self._group(rank), cp_partition_mode="contiguous"
            )
            params = batch.packed_seq_params
            key, geometry = getattr(params, context_parallel._GEOMETRY_CACHE_ATTR)
            assert key == (2, rank, 32, "cpu", (0, 24, 64))
            # The D layer, in the zigzag partition, only looks it up: no device work, also under a capture.
            params.cp_partition_mode = "zigzag"
            with _simulated_capture():
                found = context_parallel.zigzag_cp_geometry(
                    params, cp_group=self._group(rank), local_rows=32, device=torch.device("cpu"), what="core"
                )
            assert found is geometry

    def test_batch_builder_prebuilds_the_inverse_key_permutation_on_request(self, pg_collection):
        if context_parallel.load_cp_layout_api() is None:
            pytest.skip("Megatron-Core main has no zigzag CP helpers")
        generator = torch.Generator().manual_seed(0)
        samples = [(torch.randint(1, VOCAB, (length,), generator=generator),) * 2 for length in (24, 40)]
        for rank in range(2):
            # Off by default: the geometry has no inverse, as before.
            plain = build_ernie55_thd_cp_batch(
                samples, pad_token_id=0, cp_group=self._group(rank), cp_partition_mode="contiguous"
            )
            _, plain_geometry = getattr(plain.packed_seq_params, context_parallel._GEOMETRY_CACHE_ATTR)
            assert plain_geometry.key_reorder_inverse is None
            batch = build_ernie55_thd_cp_batch(
                samples,
                pad_token_id=0,
                cp_group=self._group(rank),
                cp_partition_mode="contiguous",
                with_key_reorder_inverse=True,
            )
            params = batch.packed_seq_params
            _, geometry = getattr(params, context_parallel._GEOMETRY_CACHE_ATTR)
            assert torch.equal(geometry.key_reorder, plain_geometry.key_reorder)
            order = geometry.key_reorder
            assert torch.equal(geometry.key_reorder_inverse[order], torch.arange(order.numel()))
            # The D layer with ernie_cp_key_gather_inverse_grad only looks it up, also under a capture.
            params.cp_partition_mode = "zigzag"
            with _simulated_capture():
                found = context_parallel.zigzag_cp_geometry(
                    params,
                    cp_group=self._group(rank),
                    local_rows=32,
                    device=torch.device("cpu"),
                    what="core",
                    with_inverse=True,
                )
            assert found is geometry

    def test_a_missing_inverse_is_not_added_under_a_capture(self):
        if context_parallel.load_cp_layout_api() is None:
            pytest.skip("Megatron-Core main has no zigzag CP helpers")
        cu = [0, 24, 64]
        metadata = SimpleNamespace(
            qkv_format="thd",
            cp_partition_mode="zigzag",
            thd_cp_host_cu_seqlens_q=cu,
            thd_cp_host_cu_seqlens_kv=list(cu),
            max_seqlen_q=40,
            max_seqlen_kv=40,
        )
        geometry = context_parallel.prebuild_zigzag_cp_geometry(
            metadata, cp_group=self._group(1), local_rows=32, device="cpu"
        )
        assert geometry.key_reorder_inverse is None
        kwargs = dict(cp_group=self._group(1), local_rows=32, device=torch.device("cpu"), what="core")
        with _simulated_capture(), pytest.raises(RuntimeError, match="has no inverse key permutation"):
            context_parallel.zigzag_cp_geometry(metadata, with_inverse=True, **kwargs)
        # Nothing was cached; outside a capture the inverse is added to the cached geometry.
        assert getattr(metadata, context_parallel._GEOMETRY_CACHE_ATTR)[1] is geometry
        added = context_parallel.zigzag_cp_geometry(metadata, with_inverse=True, **kwargs)
        assert added.key_reorder is geometry.key_reorder and added.query_positions is geometry.query_positions
        assert torch.equal(added.key_reorder_inverse[added.key_reorder], torch.arange(added.key_reorder.numel()))
        with _simulated_capture():
            assert context_parallel.zigzag_cp_geometry(metadata, with_inverse=True, **kwargs) is added

    def test_a_missing_geometry_is_not_built_under_a_capture(self):
        if context_parallel.load_cp_layout_api() is None:
            pytest.skip("Megatron-Core main has no zigzag CP helpers")
        cu = [0, 24, 64]
        metadata = SimpleNamespace(
            qkv_format="thd",
            cp_partition_mode="zigzag",
            thd_cp_host_cu_seqlens_q=cu,
            thd_cp_host_cu_seqlens_kv=list(cu),
            max_seqlen_q=40,
            max_seqlen_kv=40,
        )
        with _simulated_capture(), pytest.raises(RuntimeError, match="copies host lists to the device"):
            context_parallel.zigzag_cp_geometry(
                metadata, cp_group=self._group(1), local_rows=32, device=torch.device("cpu"), what="core"
            )

    def test_share_zigzag_cp_geometry(self):
        if context_parallel.load_cp_layout_api() is None:
            pytest.skip("Megatron-Core main has no zigzag CP helpers")

        def metadata(cu):
            return SimpleNamespace(
                qkv_format="thd",
                cp_partition_mode="contiguous",
                thd_cp_host_cu_seqlens_q=cu,
                thd_cp_host_cu_seqlens_kv=list(cu),
                max_seqlen_q=64,
                max_seqlen_kv=64,
            )

        source = metadata([0, 64])
        geometry = context_parallel.prebuild_zigzag_cp_geometry(
            source, cp_group=self._group(0), local_rows=32, device="cpu"
        )
        target = metadata([0, 64])
        assert context_parallel.share_zigzag_cp_geometry(source, target) is geometry
        target.cp_partition_mode = "zigzag"
        with _simulated_capture():
            found = context_parallel.zigzag_cp_geometry(
                target, cp_group=self._group(0), local_rows=32, device=torch.device("cpu"), what="core"
            )
        assert found is geometry
        with pytest.raises(ValueError, match="cannot serve a pack with the boundaries"):
            context_parallel.share_zigzag_cp_geometry(source, metadata([0, 24, 64]))
        with pytest.raises(ValueError, match="no zigzag CP geometry"):
            context_parallel.share_zigzag_cp_geometry(metadata([0, 64]), target)


# ---------------------------------------------------------------------------------------------
# D7: replay through a whole model's forward, and the eager fallback
# ---------------------------------------------------------------------------------------------


@dataclass
class _Recorded:
    calls: list
    """``(wrapper layer number, args, kwargs)`` of every stand-in graph call."""


def _install_stand_in_graphs(model) -> _Recorded:
    """A recording stand-in graph for every graphed wrapper (it returns its input streams), after the static inputs and
    host plan a capture builds."""
    recorded = _Recorded(calls=[])
    for wrapper in _graphed_wrappers(model):
        config = wrapper.config
        wrapper.get_layer_static_inputs(config.max_seqlen_per_dp_cp_rank * config.context_parallel_size, 1)

        def graph(*args, _layer=wrapper.layer_number, **kwargs):
            recorded.calls.append((_layer, args, kwargs))
            return (args[0].clone(),)

        wrapper.cuda_graphs = [graph]
    return recorded


@needs_hook
class TestReplayAndFallback:
    @pytest.fixture
    def model(self, pg_collection):
        model = _build_model(pg_collection, pattern=_D_PATTERN, **_D_AND_MTP_GRAPHS)
        _fixed_tracker(model, pg_collection)
        return model

    def test_the_graphs_get_exactly_their_captured_inputs(self, model):
        recorded = _install_stand_in_graphs(model)
        batch = _one_document_batch()
        losses = _model_forward(model, batch)
        (losses * batch.loss_mask).sum().backward()
        assert [layer for layer, _, _ in recorded.calls] == [1, 3, 1]  # decoder D1, D3, the MTP depth's D
        width = STREAMS * MODEL_HIDDEN
        for (_, args, kwargs), dtype in zip(recorded.calls, (torch.float32, torch.float32, torch.bfloat16)):
            (hidden,) = args
            assert hidden.dtype == dtype and tuple(hidden.shape) == (SEQ, 1, width)
            for name in _PACKED_FIELDS:
                assert torch.equal(kwargs[name], batch.packed_seq_params.cu_seqlens_q_padded), name
            others = {
                name for name, value in kwargs.items() if not name.startswith("cu_seqlens") and value is not None
            }
            assert others == {"is_first_microbatch"}, sorted(others)

    @pytest.mark.parametrize(
        ("case", "reason"),
        [
            ("two-documents", r"document boundaries \[0, 180, 300\] differ from the captured \[0, 300\]"),
            ("max-seqlen", r"\(max_seqlen_q, max_seqlen_kv\) = \(302, 302\) differ from the captured \(300, 300\)"),
            ("no-host-notes", "were not noted on the host"),
        ],
    )
    def test_other_compositions_run_eagerly(self, model, caplog, case, reason):
        if case == "two-documents":
            batch = _one_document_batch(lengths=(180, 120))
        elif case == "max-seqlen":
            batch = _one_document_batch(max_seqlen=SEQ + 2)
        else:
            batch = _one_document_batch()
            params = batch.packed_seq_params
            # Metadata built elsewhere: the same boundaries in a tensor of its own, nothing noted on the host.
            cu = params.cu_seqlens_q.clone()
            batch.packed_seq_params = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu,
                cu_seqlens_kv=cu,
                cu_seqlens_q_padded=cu,
                cu_seqlens_kv_padded=cu,
                max_seqlen_q=params.max_seqlen_q,
                max_seqlen_kv=params.max_seqlen_kv,
            )
        expected = _model_forward(model, batch)
        (expected * batch.loss_mask).sum().backward()
        expected_grads = {
            name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None
        }
        model.zero_grad(set_to_none=True)

        recorded = _install_stand_in_graphs(model)
        with caplog.at_level("WARNING", logger=_LOGGER):
            actual = _model_forward(model, batch)
            (actual * batch.loss_mask).sum().backward()
        assert recorded.calls == [], "a graph replayed a micro-batch of another composition"
        assert torch.equal(actual, expected)
        grads = {name: param.grad for name, param in model.named_parameters() if param.grad is not None}
        _assert_bitwise(grads, expected_grads, case)
        warnings = [record.getMessage() for record in caplog.records if "runs eagerly" in record.getMessage()]
        assert len(warnings) == 1, warnings  # once per reason and process, not per layer
        assert re.search(reason, warnings[0]), warnings[0]

    @pytest.mark.parametrize(
        ("case", "error"),
        [
            ("two-documents", r"H-layer CUDA graph \(layer 1\) replays rows of exactly thd_max_packed_sequences=1"),
            ("max-seqlen", r"H-layer CUDA graph \(layer 1\) was captured with max_seqlen_q=300"),
            ("no-host-notes", None),
        ],
    )
    def test_with_h_layers_only_the_captured_composition_runs(self, pg_collection, caplog, case, error):
        """With H layers in the decoder, the H graphs reject another composition before the D layers can fall back.

        Only a micro-batch the H graphs accept whose host plan still differs (here metadata without the batch
        builder's host notes) reaches the D layers' eager fallback.
        """
        model = _build_model(pg_collection, pattern=_PATTERN, **_D_AND_MTP_GRAPHS)
        _fixed_tracker(model, pg_collection)
        if case == "two-documents":
            batch = _one_document_batch(lengths=(180, 120))
        elif case == "max-seqlen":
            batch = _one_document_batch(max_seqlen=SEQ + 2)
        else:
            batch = _one_document_batch()
            params = batch.packed_seq_params
            cu = params.cu_seqlens_q.clone()
            batch.packed_seq_params = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu,
                cu_seqlens_kv=cu,
                cu_seqlens_q_padded=cu,
                cu_seqlens_kv_padded=cu,
                max_seqlen_q=params.max_seqlen_q,
                max_seqlen_kv=params.max_seqlen_kv,
            )
        recorded = _install_stand_in_graphs(model)
        with caplog.at_level("WARNING", logger=_LOGGER):
            if error is not None:
                with pytest.raises(ValueError, match=error):
                    _model_forward(model, batch)
            else:
                _model_forward(model, batch)
        warnings = [record.getMessage() for record in caplog.records if "runs eagerly" in record.getMessage()]
        if error is not None:
            assert recorded.calls == [] and warnings == [], (recorded.calls, warnings)
        else:
            # The H graphs (layers 1 and 5) replay; the D layers (3, 7 and the MTP depth's) run eagerly.
            assert [layer for layer, _, _ in recorded.calls] == [1, 5]
            assert len(warnings) == 1 and "were not noted on the host" in warnings[0], warnings

    @pytest.mark.parametrize("offloading", [False, True])
    def test_forward_replay_waits_for_the_offloading_graphs(self, model, monkeypatch, offloading):
        """With fine-grained activation offloading, a D graph's forward replay waits for the offload graph stream.

        The H graphs replay on the offload manager's graph stream, and the current stream resumes before their tails
        have copied the offloaded activations, whose memory a later-captured D graph may hold. So the current stream
        must wait for that stream right before the D graph replays; without offloading nothing waits. Stand-in streams
        record the order on CPU.
        """
        from megatron.core.pipeline_parallel import fine_grained_activation_offload as offload

        events = []
        graph_stream = SimpleNamespace(name="offload manager's graph stream")

        class _CurrentStream:
            def wait_stream(self, stream):
                events.append(("wait", stream))

        monkeypatch.setattr(torch.cuda, "current_stream", lambda *args, **kwargs: _CurrentStream())
        monkeypatch.setattr(
            offload.FineGrainedActivationOffloadingInterface, "cuda_graph_stream", staticmethod(lambda: graph_stream)
        )
        wrapper = model.decoder.layers[0]
        assert isinstance(wrapper, Ernie55DHyperConnectionHybridLayer) and not wrapper.offload_module_in_cuda_graph
        monkeypatch.setattr(wrapper.config, "fine_grained_activation_offloading", offloading)
        config = wrapper.config
        wrapper.get_layer_static_inputs(config.max_seqlen_per_dp_cp_rank * config.context_parallel_size, 1)

        def graph(*args, **kwargs):
            events.append(("replay", wrapper.layer_number))
            return (args[0].clone(),)

        wrapper.cuda_graphs = [graph]
        params = _one_document_batch().packed_seq_params
        hidden = torch.zeros(SEQ, 1, STREAMS * MODEL_HIDDEN)
        wrapper(hidden_states=hidden, packed_seq_params=params)
        wrapper(hidden_states=hidden, packed_seq_params=params)
        replay = ("replay", wrapper.layer_number)
        if offloading:
            assert events == [("wait", graph_stream), replay] * 2, events
        else:
            assert events == [replay] * 2, events

    def test_the_static_input_contract_still_raises(self, model):
        """Inputs TE would convert or drop are errors, not fallbacks (as for the H graphs)."""
        recorded = _install_stand_in_graphs(model)
        wrapper = model.decoder.layers[0]
        params = _one_document_batch().packed_seq_params
        width = STREAMS * MODEL_HIDDEN
        with pytest.raises(ValueError, match="replays fp32 streams"):
            wrapper(hidden_states=torch.zeros(SEQ, 1, width, dtype=torch.bfloat16), packed_seq_params=params)
        with pytest.raises(ValueError, match=r"has no input \['attention_mask'\]"):
            wrapper(
                hidden_states=torch.zeros(SEQ, 1, width),
                attention_mask=torch.ones(1, 1, SEQ, SEQ, dtype=torch.bool),
                packed_seq_params=params,
            )
        mtp_wrapper = model.mtp.layers[0].mtp_model_layer.layers[0]
        with pytest.raises(ValueError, match="MTP D-layer CUDA graph .* replays bf16 streams"):
            mtp_wrapper(hidden_states=torch.zeros(SEQ, 1, width), packed_seq_params=params)
        assert recorded.calls == []


# ---------------------------------------------------------------------------------------------
# D8: SD-MTP shared gradients with emulated Transformer Engine graphs
# ---------------------------------------------------------------------------------------------


def _emulated_te_graph(wrapper, calls: list):
    """A CPU stand-in for Transformer Engine's graphed callable of ``wrapper``; every call appends to ``calls``.

    Like TE's ``Graphed`` function: the call runs the wrapper's capture function on detached copies of its inputs in
    an autograd graph of its own, returns its outputs detached, and its backward returns the gradients of the inputs
    and of every parameter of the wrapper (summed within the callable), which autograd then accumulates into the
    parameters, adding the contributions of every callable that uses a shared parameter.
    """
    params = tuple(wrapper.parameters())

    class Graphed(torch.autograd.Function):
        @staticmethod
        def forward(ctx, hidden, *tensors):
            cu_seqlens, _ = tensors[: len(_PACKED_FIELDS)], tensors[len(_PACKED_FIELDS) :]
            inputs = hidden.detach().requires_grad_(True)
            with torch.enable_grad():
                outputs = wrapper._te_cuda_graph_capture(inputs, **dict(zip(_PACKED_FIELDS, cu_seqlens)))
            ctx.graph = (inputs, outputs)
            return tuple(output.detach() for output in outputs)

        @staticmethod
        def backward(ctx, *grad_outputs):
            inputs, outputs = ctx.graph
            grads = torch.autograd.grad(outputs, (inputs, *params), grad_outputs, allow_unused=True)
            return (grads[0], *([None] * len(_PACKED_FIELDS)), *grads[1:])

    def graph(hidden, **kwargs):
        calls.append(wrapper.layer_number)
        return Graphed.apply(hidden, *(kwargs[name] for name in _PACKED_FIELDS), *params)

    return graph


@needs_hook
@pytest.mark.parametrize("mtp_graphs", [True, False], ids=["mtp-graphed", "mtp-eager"])
def test_shared_mtp_parameters_get_the_eager_summed_gradient(pg_collection, mtp_graphs):
    """D8: with every graphed wrapper emulated, the loss and every gradient equal the eager ones bitwise.

    The MTP depth's D layer shares every parameter with the decoder's last D layer (layer 7). With both graphed, each
    graph returns its own contribution and autograd adds the two, as it adds the two uses in eager execution; with the
    MTP depth eager, the decoder graph's contribution is added to the eager one.
    """
    model = _build_model(pg_collection, pattern=_PATTERN, **(_D_AND_MTP_GRAPHS if mtp_graphs else _D_GRAPHS))
    _fixed_tracker(model, pg_collection)
    batch = _one_document_batch()
    wrappers = _graphed_wrappers(model)
    assert len(wrappers) == 4 + mtp_graphs
    shared = {id(param) for param in model.mtp.layers[0].mtp_model_layer.layers[0].parameters()}
    assert shared and shared <= {id(param) for param in model.decoder.layers[6].parameters()}

    def run() -> dict[str, torch.Tensor]:
        model.zero_grad(set_to_none=True)
        dsa.DSAIndexerLossLoggingHelper.clean_loss_in_tracker(preserve_groups=True)
        losses = _model_forward(model, batch)
        (losses * batch.loss_mask).sum().backward()
        tensors = {"losses": losses.detach().clone()}
        tensors.update(
            {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}
        )
        return tensors

    expected = run()
    calls = []
    for wrapper in wrappers:
        wrapper.get_layer_static_inputs(SEQ, 1)
        wrapper.cuda_graphs = [_emulated_te_graph(wrapper, calls)]
    actual = run()
    assert calls == [wrapper.layer_number for wrapper in wrappers], "a wrapper did not replay its emulated graph"
    _assert_bitwise(actual, expected, "emulated graphs against eager")
    shared_names = [name for name, param in model.named_parameters() if id(param) in shared]
    assert shared_names and all(name in actual for name in shared_names)


# ---------------------------------------------------------------------------------------------
# D10: DDP gradient-ready counts re-learned after the capture (overlap_grad_reduce)
# ---------------------------------------------------------------------------------------------


class _Bucket:
    """The attributes of Megatron-Core's ``_ParamAndGradBucket`` that a bucket group's bookkeeping reads."""

    def __init__(self, params):
        self.params_list = list(params)
        self.params = set(self.params_list)
        self.params_with_extra_main_grads = []


def _overlapping_bucket_group(params, *, overlap_grad_reduce=True):
    """Megatron-Core's real ``_ParamAndGradBucketGroup`` with its collective replaced by a recorder.

    ``started`` lists, per reduce issued, whether ``register_grad_ready`` (the backward's last hook) or
    ``finish_grad_sync`` (after the backward) issued it.
    """
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup

    config = DistributedDataParallelConfig(overlap_grad_reduce=overlap_grad_reduce, use_distributed_optimizer=False)
    group = _ParamAndGradBucketGroup([_Bucket(params)], config, SimpleNamespace(rank=lambda: 0), 1)
    group.started = []
    group.in_finish = False

    def start_grad_sync(force_all_reduce=False):
        group.started.append("finish_grad_sync" if group.in_finish else "register_grad_ready")
        group.grad_reduce_handle = SimpleNamespace(wait=lambda: None)

    group.start_grad_sync = start_grad_sync
    return group


def _ddp_step(group, hook_calls):
    """One training step's DDP bookkeeping: ``zero_grad_buffer`` (reset), the backward's hooks, then
    ``finalize_model_grads`` (finish_grad_sync)."""
    group.reset()
    for param, calls in hook_calls:
        for _ in range(calls):
            group.register_grad_ready(param)
    group.in_finish = True
    try:
        group.finish_grad_sync()
    finally:
        group.in_finish = False


class TestDDPGradReadyCounts:
    """D10: graph replays fire a shared parameter's DDP hook fewer times than eager layers; the counts are re-learned.

    Eagerly, the input norm that the decoder's last D layer shares with the MTP depth's D layer is recomputed
    (``layernorm`` recompute) in each of its two uses, and each recompute runs its own backward: two hook calls per
    step. Graphed, both graphs return their contributions to autograd, which adds them: one call (seen on GPU at the
    first graphed iteration with ``overlap_grad_reduce``). Megatron-Core's bucket group waits for the count of the first
    batch, so without the re-learn the step never reduces.
    """

    EAGER = 2  # hook calls of the shared norm weight per eager step
    GRAPHED = 1  # with both D layers graphed

    def _params(self):
        return torch.nn.Parameter(torch.zeros(4)), torch.nn.Parameter(torch.zeros(4))

    def test_without_relearning_the_graphed_step_never_reduces(self):
        shared, other = self._params()
        group = _overlapping_bucket_group([shared, other])
        for _ in range(3):  # eager steps: the first records nothing, the second records the golden counts
            _ddp_step(group, [(shared, self.EAGER), (other, 1)])
        assert group.started == ["finish_grad_sync", "register_grad_ready", "register_grad_ready"]
        with pytest.raises(AssertionError, match=r"Communication call has not been issued .*2/2 params"):
            _ddp_step(group, [(shared, self.GRAPHED), (other, 1)])

    def test_relearned_counts_reduce_every_graphed_step(self):
        shared, other = self._params()
        group = _overlapping_bucket_group([shared, other])
        for _ in range(3):
            _ddp_step(group, [(shared, self.EAGER), (other, 1)])
        chunk = SimpleNamespace(bucket_groups=[group], expert_parallel_bucket_groups=[])
        assert cuda_graph._relearn_ddp_grad_ready_counts([chunk]) == 1
        group.started.clear()
        for _ in range(3):  # graphed steps: the first reduces after its backward, the next ones at their last hook
            _ddp_step(group, [(shared, self.GRAPHED), (other, 1)])
        assert group.started == ["finish_grad_sync", "register_grad_ready", "register_grad_ready"]
        assert group.golden_per_param_grad_ready_counts == {shared: self.GRAPHED, other: 1}

    def test_unchanged_counts_still_reduce_after_relearning(self):
        """H-layer graphs (no shared parameter) keep the eager counts: the re-learn costs one reduce after the
        backward, then nothing changes."""
        shared, other = self._params()
        group = _overlapping_bucket_group([shared, other])
        for _ in range(3):
            _ddp_step(group, [(shared, 1), (other, 1)])
        cuda_graph._relearn_ddp_grad_ready_counts([SimpleNamespace(bucket_groups=[group])])
        group.started.clear()
        for _ in range(2):
            _ddp_step(group, [(shared, 1), (other, 1)])
        assert group.started == ["finish_grad_sync", "register_grad_ready"]

    def test_only_overlapping_ddp_bucket_groups_are_reset(self):
        overlapping = _overlapping_bucket_group(list(self._params()))
        expert = _overlapping_bucket_group(list(self._params()))
        synchronous = _overlapping_bucket_group(list(self._params()), overlap_grad_reduce=False)
        for group in (overlapping, expert):
            group.is_first_batch = False
            group.golden_per_param_grad_ready_counts = {param: 1 for param in group.params}
            group.per_param_grad_ready_counts = dict(group.golden_per_param_grad_ready_counts)
        synchronous.is_first_batch = False
        chunks = [
            SimpleNamespace(bucket_groups=[overlapping, synchronous], expert_parallel_bucket_groups=[expert]),
            torch.nn.Linear(2, 2),  # a chunk that Megatron-Core's DDP does not wrap
        ]
        assert cuda_graph._relearn_ddp_grad_ready_counts(chunks) == 2
        for group in (overlapping, expert):
            assert group.is_first_batch
            assert group.golden_per_param_grad_ready_counts == {} and group.per_param_grad_ready_counts == {}
        assert not synchronous.is_first_batch

    def test_a_bucket_group_without_the_ready_count_state_raises(self):
        """A Megatron-Core DDP that renames the private state fails loudly instead of never re-learning."""
        group = _overlapping_bucket_group(list(self._params()))
        del group.golden_per_param_grad_ready_counts
        with pytest.raises(RuntimeError, match="golden_per_param_grad_ready_counts"):
            cuda_graph._relearn_ddp_grad_ready_counts([SimpleNamespace(bucket_groups=[group])])

    @pytest.mark.parametrize("graphed", [True, False], ids=["layers-graphed", "nothing-graphed"])
    def test_create_cudagraphs_relearns_after_the_capture(self, monkeypatch, graphed):
        captured = []
        monkeypatch.setattr(
            cuda_graph.TECudaGraphHelper, "create_cudagraphs", lambda self: captured.append(self.model)
        )
        group = _overlapping_bucket_group(list(self._params()))
        group.is_first_batch = False
        helper = Ernie55TECudaGraphHelper.__new__(Ernie55TECudaGraphHelper)
        helper.model = [SimpleNamespace(bucket_groups=[group], expert_parallel_bucket_groups=[])]
        helper.flattened_callables = [object()] if graphed else []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        helper.create_cudagraphs()
        assert captured == [helper.model]
        assert group.is_first_batch == graphed


# ---------------------------------------------------------------------------------------------
# D9: CP = 2 and 4 on spawned gloo ranks
# ---------------------------------------------------------------------------------------------

_CP_LOCAL_ROWS = 32
_CP_PATTERN = "D-DE"


def setup_rank(world_size: int):
    """The CPU stand-ins of the ERNIE 5.5 model on a spawned rank (``test_ernie55_cp_model.py``'s)."""
    from tests.unit_tests.models.ernie.test_ernie55_cp_model import setup_rank as model_setup_rank

    rank = model_setup_rank(world_size)
    cuda_graph._WARNED_FALLBACKS.clear()
    return rank


def _cp_model(cp_size: int, *, seed: int = 3, **extra):
    from tests.unit_tests.models.ernie.test_ernie55_cp_model import _model_parallel

    pg_collection = _model_parallel(cp_size)
    overrides = dict(
        _D_GRAPHS,
        max_seqlen_per_dp_cp_rank=_CP_LOCAL_ROWS,
        context_parallel_size=cp_size,
        cp_partition_mode="contiguous",
        cp_comm_type="all_gather",
        params_dtype=torch.float32,
        dsa_indexer_topk=64,
        **extra,
    )
    model = _build_model(pg_collection, pattern=_CP_PATTERN, seed=seed, **overrides)
    _fixed_tracker(model, pg_collection)
    return model, pg_collection


def _cp_batch(pg_collection, lengths: tuple[int, ...], *, seed: int = 5, **kwargs):
    generator = torch.Generator().manual_seed(seed)
    samples = []
    for length in lengths:
        tokens = torch.randint(1, VOCAB, (length,), generator=generator)
        labels = tokens.roll(-1)
        labels[-1] = IGNORE_LABEL
        samples.append((tokens, labels))
    return build_ernie55_thd_cp_batch(
        samples, pad_token_id=0, cp_group=pg_collection.cp, cp_partition_mode="contiguous", **kwargs
    )


def case_plan_matches_the_batch_builder(rank) -> None:
    """The plan's route and geometry are the batch builder's for a one-document row."""
    from megatron.core import parallel_state

    cp_size = rank.world_size
    total = _CP_LOCAL_ROWS * cp_size
    model, pg_collection = _cp_model(cp_size)
    try:
        batch = _cp_batch(pg_collection, (total,))
        for wrapper in (model.decoder.layers[0], model.decoder.layers[2]):
            assert type(wrapper) is Ernie55DHyperConnectionHybridLayer
            wrapper.get_layer_static_inputs(total, 1)
            plan = wrapper.__dict__[cuda_graph._D_GRAPH_PLAN]
            params = batch.packed_seq_params
            assert (plan.boundaries, plan.max_seqlen, plan.cp_size) == ((0, total), total, cp_size)
            assert list(plan.boundaries) == [int(value) for value in params.thd_cp_host_cu_seqlens_q]
            route, expected_route = plan.template.cp_partition_route, params.cp_partition_route
            assert route.zigzag_split_sizes == expected_route.zigzag_split_sizes
            assert route.contiguous_split_sizes == expected_route.contiguous_split_sizes
            for name in ("zigzag_index", "contiguous_index"):
                mine, theirs = getattr(route, name), getattr(expected_route, name)
                assert (mine is None and theirs is None) or torch.equal(mine, theirs), name
            _, geometry = getattr(plan.template, context_parallel._GEOMETRY_CACHE_ATTR)
            _, expected_geometry = getattr(params, context_parallel._GEOMETRY_CACHE_ATTR)
            assert torch.equal(geometry.query_positions, expected_geometry.query_positions)
            assert torch.equal(geometry.key_reorder, expected_geometry.key_reorder)
    finally:
        parallel_state.destroy_model_parallel()


def case_simulated_capture_matches_eager(rank) -> None:
    """A simulated capture of each D wrapper gives its eager output and gradients bitwise at CP > 1."""
    from megatron.core import parallel_state

    cp_size = rank.world_size
    model, pg_collection = _cp_model(cp_size)
    try:
        batch = _cp_batch(pg_collection, (_CP_LOCAL_ROWS * cp_size,))
        for index, wrapper in enumerate((model.decoder.layers[0], model.decoder.layers[2])):
            _capture_matches_eager(wrapper, batch, seed=21 + index, label=f"CP{cp_size} rank {rank.rank} D{index}")
        # The forward restored the stack partition.
        assert batch.packed_seq_params.cp_partition_mode == "contiguous"
    finally:
        parallel_state.destroy_model_parallel()


def case_simulated_capture_with_the_inverse_key_gradient(rank) -> None:
    """With ``ernie_cp_key_gather_inverse_grad`` the plan pins the inverse key permutation; capture matches eager."""
    from megatron.core import parallel_state

    cp_size = rank.world_size
    model, pg_collection = _cp_model(cp_size, ernie_cp_key_gather_inverse_grad=True)
    applied = []
    permute_rows = context_parallel._PermuteRows.apply

    def counted(*args):
        applied.append(args[0].shape)
        return permute_rows(*args)

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(context_parallel._PermuteRows, "apply", counted)
            batch = _cp_batch(pg_collection, (_CP_LOCAL_ROWS * cp_size,), with_key_reorder_inverse=True)
            _, batch_geometry = getattr(batch.packed_seq_params, context_parallel._GEOMETRY_CACHE_ATTR)
            for index, wrapper in enumerate((model.decoder.layers[0], model.decoder.layers[2])):
                applied.clear()
                label = f"CP{cp_size} rank {rank.rank} D{index} inverse key gradient"
                _capture_matches_eager(wrapper, batch, seed=31 + index, label=label)
                # Two key gathers (latent key, indexer key) in the eager run and two in the capture.
                assert len(applied) == 4, applied
                _, geometry = getattr(
                    wrapper.__dict__[cuda_graph._D_GRAPH_PLAN].template, context_parallel._GEOMETRY_CACHE_ATTR
                )
                assert geometry.key_reorder_inverse is not None
                assert torch.equal(geometry.key_reorder_inverse, batch_geometry.key_reorder_inverse)
    finally:
        parallel_state.destroy_model_parallel()


def case_simulated_capture_with_both_d_layer_switches(rank) -> None:
    """The inverse key gradient and the batched VHA mix together inside the D graphs: capture matches eager."""
    from megatron.core import parallel_state

    from megatron.bridge.models.ernie.modeling_ernie55 import ops as ernie_ops

    cp_size = rank.world_size
    model, pg_collection = _cp_model(cp_size, ernie_cp_key_gather_inverse_grad=True, ernie_vha_batched_mix=True)
    applied = {"permute": 0, "mix": 0}
    permute_rows, head_mix = context_parallel._PermuteRows.apply, ernie_ops._TokenBatchedHeadMix.apply

    def counted_permute(*args):
        applied["permute"] += 1
        return permute_rows(*args)

    def counted_mix(*args):
        applied["mix"] += 1
        return head_mix(*args)

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(context_parallel._PermuteRows, "apply", counted_permute)
            patch.setattr(ernie_ops._TokenBatchedHeadMix, "apply", counted_mix)
            batch = _cp_batch(pg_collection, (_CP_LOCAL_ROWS * cp_size,), with_key_reorder_inverse=True)
            for index, wrapper in enumerate((model.decoder.layers[0], model.decoder.layers[2])):
                applied.update(permute=0, mix=0)
                label = f"CP{cp_size} rank {rank.rank} D{index} both switches"
                _capture_matches_eager(wrapper, batch, seed=51 + index, label=label)
                # Eager and captured run: two key gathers and one head mix each.
                assert applied == {"permute": 4, "mix": 2}, applied
    finally:
        parallel_state.destroy_model_parallel()


def case_replay_and_fallback(rank) -> None:
    """Stand-in graphs replay a one-document row; a two-document row runs the D layers eagerly, bitwise."""
    from megatron.core import parallel_state

    cp_size = rank.world_size
    total = _CP_LOCAL_ROWS * cp_size
    model, pg_collection = _cp_model(cp_size)
    try:
        recorded = _install_stand_in_graphs(model)
        one = _cp_batch(pg_collection, (total,))
        _model_forward(model, one)
        assert [layer for layer, _, _ in recorded.calls] == [1, 3], recorded.calls

        two = _cp_batch(pg_collection, (2 * cp_size * 3, total - 2 * cp_size * 3))
        graphs = {id(wrapper): wrapper.cuda_graphs for wrapper in _graphed_wrappers(model)}
        for wrapper in _graphed_wrappers(model):
            wrapper.cuda_graphs = []
        expected = _model_forward(model, two)
        for wrapper in _graphed_wrappers(model):
            wrapper.cuda_graphs = graphs[id(wrapper)]
        recorded.calls.clear()
        actual = _model_forward(model, two)
        assert recorded.calls == [], "a graph replayed a micro-batch of another composition"
        assert torch.equal(actual, expected)
    finally:
        parallel_state.destroy_model_parallel()


CASES = [
    "case_plan_matches_the_batch_builder",
    "case_simulated_capture_matches_eager",
    "case_simulated_capture_with_the_inverse_key_gradient",
    "case_simulated_capture_with_both_d_layer_switches",
    "case_replay_and_fallback",
]


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
        workdir=tmp_path_factory.mktemp(f"ernie55_d_cuda_graph_cp{request.param}"),
    )


@pytest.mark.parametrize("case", CASES, ids=[case.removeprefix("case_") for case in CASES])
def test_context_parallel(cp_results, case):
    failures = {rank: result["error"] for rank, result in cp_results[case].items() if not result["ok"]}
    assert len(cp_results[case]) > 1, "no rank reported"
    assert not failures, "\n".join(f"--- rank {rank} ---\n{error}" for rank, error in failures.items())
