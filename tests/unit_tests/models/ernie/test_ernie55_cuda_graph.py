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

"""CPU tests of the ERNIE 5.5 H-layer Transformer Engine CUDA graphs (``modeling_ernie55.cuda_graph``).

Nothing here needs a GPU or captures a graph (``test_ernie55_cuda_graph_gpu.py`` does):

* the configuration envelope of :func:`validate_ernie55_cuda_graph_config`;
* the static inputs and the replay checks of :class:`Ernie55HCAHyperConnectionHybridLayer`, on a stand-in wrapper;
* which wrappers of a CPU-built ``H-DEHEDE/DE`` model become H-graph wrappers, and that nothing else changes;
* the discovery filter of :class:`Ernie55TECudaGraphHelper`, on stubbed Megatron-Core discovery results, including a
  pipeline stage without H layers;
* the training loop's choice of the helper class, the helper's ``NCCL_GRAPH_REGISTER`` check, its
  ``graph_capture_record_stream_reuse`` warning and its post-capture memory log;
* the D attention's refusal of a graph capture;
* that an H wrapper's capture keeps the RoPE tables its graph reads alive through any churn of Megatron-Core's shared
  RoPE cache, until the graphs are deleted.
"""

from __future__ import annotations

import contextlib
import dataclasses
import gc
import inspect
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.hybrid.hybrid_block import HyperConnectionHybridLayer
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper
from megatron.core.transformer.enums import CudaGraphModule

from megatron.bridge.models.ernie.ernie55_config import Ernie55TransformerConfig
from megatron.bridge.models.ernie.modeling_ernie55 import (
    Ernie55HCAHyperConnectionHybridLayer,
    Ernie55HCASelfAttention,
    Ernie55LatentMQASelfAttention,
    Ernie55TECudaGraphHelper,
    use_ernie55_attention_cuda_graphs,
    validate_ernie55_config,
    validate_ernie55_cuda_graph_config,
    validate_ernie55_hca_config,
)
from tests.unit_tests.models.ernie.test_ernie55_mhc_mtp import HIDDEN as MODEL_HIDDEN
from tests.unit_tests.models.ernie.test_ernie55_mhc_mtp import (
    SEQ,
    _build_model,
    _config,
    needs_hook,
    pg_collection,  # noqa: F401 - the CPU model builder's fixture, requested by name
)
from tests.unit_tests.models.ernie.test_ernie55_mhc_mtp import STREAMS as MODEL_STREAMS
from tests.unit_tests.models.ernie.test_ernie55_mhc_mtp import VOCAB as MODEL_VOCAB


pytestmark = pytest.mark.unit

HIDDEN = 8
STREAMS = 4
LOCAL_ROWS = 16
WIDTH = HIDDEN * STREAMS
# Overrides of the CPU model config that turn on the H-layer graphs (CP = 1: max_seqlen_per_dp_cp_rank = SEQ).
_TE_ATTENTION_GRAPHS = dict(
    cuda_graph_impl="transformer_engine",
    cuda_graph_modules=["attn"],
    use_te_rng_tracker=True,
    max_seqlen_per_dp_cp_rank=SEQ,
    thd_max_packed_sequences=1,
)


# ---------------------------------------------------------------------------------------------
# T1: configuration envelope
# ---------------------------------------------------------------------------------------------


def _graph_config(**overrides) -> SimpleNamespace:
    """A config with every field the envelope reads, inside the envelope unless ``overrides`` say otherwise."""
    values = dict(
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=[CudaGraphModule.attn],
        cuda_graph_scope=None,
        enable_cuda_graph=False,
        experimental_attention_variant="dsv4_hybrid",
        num_layers=2,
        csa_compress_ratios=[128, 0],
        enable_mhc_connections=True,
        mhc_recompute_attn_cuda_graph_split=False,
        use_te_rng_tracker=True,
        pipeline_model_parallel_size=1,
        cuda_graph_warmup_steps=3,
        delay_wgrad_compute=False,
        overlap_moe_expert_parallel_comm=False,
        sequence_packing_scheduler=None,
        max_seqlen_per_dp_cp_rank=4096,
        thd_max_packed_sequences=1,
        context_parallel_size=1,
        cp_partition_mode="contiguous",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class TestEnvelope:
    @pytest.mark.parametrize(
        "overrides",
        [
            {},
            {"cuda_graph_modules": ["attn"]},
            {"context_parallel_size": 64, "max_seqlen_per_dp_cp_rank": 4096},
            {"thd_max_packed_sequences": 5},
            # The decoder's H layer may come anywhere; without csa_compress_ratios the H layers are not counted.
            {"num_layers": 4, "csa_compress_ratios": [0, 0, 128, 0, 0, 0]},
            {"csa_compress_ratios": None},
            # Capture before the first step is fine without pipeline parallelism.
            {"cuda_graph_warmup_steps": 0},
            {"pipeline_model_parallel_size": 2, "cuda_graph_warmup_steps": 1},
            # Without CUDA graphs nothing else is read.
            {"cuda_graph_impl": "none", "cuda_graph_modules": [], "use_te_rng_tracker": False},
            {"cuda_graph_impl": None, "max_seqlen_per_dp_cp_rank": None, "thd_max_packed_sequences": None},
        ],
        ids=lambda overrides: str(overrides),
    )
    def test_accepts(self, overrides):
        for layer in ("H", "D"):
            validate_ernie55_cuda_graph_config(_graph_config(**overrides), layer=layer)

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"cuda_graph_impl": "local"}, "are not supported with cuda_graph_impl='local'"),
            ({"cuda_graph_impl": "full_iteration", "cuda_graph_modules": []}, "cuda_graph_impl='full_iteration'"),
            ({"cuda_graph_impl": "none", "enable_cuda_graph": True}, "are not supported through enable_cuda_graph"),
            ({"cuda_graph_modules": []}, r"need cuda_graph_modules=\['attn'\], got \[\]"),
            ({"cuda_graph_modules": [CudaGraphModule.attn, CudaGraphModule.moe_router]}, "need cuda_graph_modules"),
            ({"cuda_graph_modules": [CudaGraphModule.moe]}, "need cuda_graph_modules"),
            ({"cuda_graph_modules": [CudaGraphModule.mlp]}, "need cuda_graph_modules"),
            ({"cuda_graph_modules": None, "cuda_graph_scope": ["moe_router"]}, "need cuda_graph_modules"),
            # Nothing to graph: a D-only stack, no H layer in the decoder (MTP depths never graph), no mHC wrappers.
            (
                {"experimental_attention_variant": "dsa"},
                "need experimental_attention_variant='dsv4_hybrid', got 'dsa'",
            ),
            ({"csa_compress_ratios": [0, 0]}, "need an H layer in the decoder"),
            ({"csa_compress_ratios": [0, 0, 128, 0]}, "need an H layer in the decoder"),
            ({"enable_mhc_connections": False}, "need enable_mhc_connections=True"),
            ({"mhc_recompute_attn_cuda_graph_split": True}, "mhc_recompute_attn_cuda_graph_split=False"),
            ({"use_te_rng_tracker": False}, "use_te_rng_tracker=True"),
            ({"pipeline_model_parallel_size": 2, "cuda_graph_warmup_steps": 0}, "need cuda_graph_warmup_steps >= 1"),
            ({"delay_wgrad_compute": True}, "delay_wgrad_compute=False"),
            ({"overlap_moe_expert_parallel_comm": True}, "overlap_moe_expert_parallel_comm=False"),
            ({"sequence_packing_scheduler": "dp_balanced"}, "sequence_packing_scheduler=None"),
            ({"max_seqlen_per_dp_cp_rank": None}, "max_seqlen_per_dp_cp_rank"),
            ({"max_seqlen_per_dp_cp_rank": 0}, "max_seqlen_per_dp_cp_rank"),
            ({"thd_max_packed_sequences": None}, "thd_max_packed_sequences"),
            ({"thd_max_packed_sequences": True}, "thd_max_packed_sequences"),
            ({"context_parallel_size": 2, "cp_partition_mode": "zigzag"}, "cp_partition_mode='contiguous'"),
            # The H layers' opt-in zigzag partition is not validated with graphs yet.
            (
                {"context_parallel_size": 2, "cp_partition_mode": "zigzag", "dsv4_cp_zigzag": True},
                r"are not validated with the H layers' zigzag partition \(dsv4_cp_zigzag=True\)",
            ),
        ],
        ids=lambda value: str(value) if isinstance(value, dict) else None,
    )
    def test_rejects(self, overrides, message):
        for layer in ("H", "D"):
            with pytest.raises(ValueError, match=f"^ERNIE 5.5 {layer} layer: CUDA graphs .*{message}"):
                validate_ernie55_cuda_graph_config(_graph_config(**overrides), layer=layer)

    def test_layer_validators_use_the_envelope(self):
        """Both layer validators raise the envelope's message with their own prefix (checked on a real config)."""
        config = _config("H-DE", ernie_mtp_shared_last_layer=False, **_TE_ATTENTION_GRAPHS)
        validate_ernie55_hca_config(config)
        validate_ernie55_config(config)
        config.use_te_rng_tracker = False
        with pytest.raises(ValueError, match="^ERNIE 5.5 H layer: CUDA graphs need use_te_rng_tracker"):
            validate_ernie55_hca_config(config)
        with pytest.raises(ValueError, match="^ERNIE 5.5 D layer: CUDA graphs need use_te_rng_tracker"):
            validate_ernie55_config(config)

    def test_finalize_checks_the_envelope(self):
        overrides = {**_TE_ATTENTION_GRAPHS, "thd_max_packed_sequences": None}
        with pytest.raises(ValueError, match="CUDA graphs need thd_max_packed_sequences"):
            _config("H-DE", ernie_mtp_shared_last_layer=False, **overrides)

    @pytest.mark.parametrize(
        ("pattern", "overrides", "message"),
        [
            # Without hyper-connections Ernie55HybridStack converts no H layer, so the helper would find nothing.
            ("H-DE", {"enable_mhc_connections": False}, "CUDA graphs need enable_mhc_connections=True"),
            # A stack whose decoder has only D layers.
            ("D-DE", {}, "CUDA graphs need an H layer in the decoder"),
        ],
        ids=["no-mhc", "no-decoder-h-layer"],
    )
    def test_finalize_rejects_stacks_with_nothing_to_graph(self, pattern, overrides, message):
        """Such configs used to finalize and fail only at the start of training, in the helper's discovery."""
        with pytest.raises(ValueError, match=message):
            _config(pattern, ernie_mtp_shared_last_layer=False, **_TE_ATTENTION_GRAPHS, **overrides)


# ---------------------------------------------------------------------------------------------
# Stand-in H wrapper (T2, T4, T5)
# ---------------------------------------------------------------------------------------------


def _h_wrapper(
    *, layer_number: int = 1, cp: int = 1, documents: int = 1, local_rows: int | None = LOCAL_ROWS
) -> Ernie55HCAHyperConnectionHybridLayer:
    """An H-graph wrapper without the layer: the config fields the graph methods read and one parameter."""
    wrapper = Ernie55HCAHyperConnectionHybridLayer.__new__(Ernie55HCAHyperConnectionHybridLayer)
    torch.nn.Module.__init__(wrapper)
    wrapper.config = SimpleNamespace(
        hidden_size=HIDDEN,
        num_residual_streams=STREAMS,
        max_seqlen_per_dp_cp_rank=local_rows,
        thd_max_packed_sequences=documents,
        context_parallel_size=cp,
        cp_partition_mode="contiguous",
    )
    wrapper.layer_number = layer_number
    wrapper.anchor = torch.nn.Parameter(torch.zeros(2))
    return wrapper


def _d_wrapper(layer_number: int = 2) -> HyperConnectionHybridLayer:
    """A Megatron-Core wrapper whose inner layer has an ERNIE D attention (only ``isinstance`` is checked)."""
    wrapper = HyperConnectionHybridLayer.__new__(HyperConnectionHybridLayer)
    torch.nn.Module.__init__(wrapper)
    wrapper.inner_layer = SimpleNamespace(
        self_attention=Ernie55LatentMQASelfAttention.__new__(Ernie55LatentMQASelfAttention)
    )
    wrapper.layer_number = layer_number
    wrapper.anchor = torch.nn.Parameter(torch.zeros(2))
    return wrapper


# ---------------------------------------------------------------------------------------------
# T2: static inputs
# ---------------------------------------------------------------------------------------------


class TestStaticInputs:
    @pytest.mark.parametrize(("cp", "documents"), [(1, 1), (4, 1), (4, 3)])
    def test_fp32_streams_and_one_packed_row(self, cp, documents):
        total = LOCAL_ROWS * cp
        inputs = _h_wrapper(cp=cp, documents=documents).get_layer_static_inputs(total, 1)
        assert list(inputs) == [
            "hidden_states",
            "cu_seqlens_q",
            "cu_seqlens_kv",
            "cu_seqlens_q_padded",
            "cu_seqlens_kv_padded",
        ]
        hidden = inputs["hidden_states"]
        assert hidden.dtype == torch.float32 and tuple(hidden.shape) == (LOCAL_ROWS, 1, WIDTH)
        assert hidden.requires_grad and hidden.is_leaf and bool((hidden == 1).all())
        boundaries = [0] + [total] * documents
        pointers = set()
        for name in list(inputs)[1:]:
            cu_seqlens = inputs[name]
            assert cu_seqlens.dtype == torch.int32 and tuple(cu_seqlens.shape) == (documents + 1,), name
            assert cu_seqlens.tolist() == boundaries, name
            assert not cu_seqlens.requires_grad, name
            pointers.add(cu_seqlens.data_ptr())
        assert len(pointers) == 4, "the four cu_seqlens inputs must be separate tensors"

    def test_inputs_follow_the_parameters_device(self):
        wrapper = _h_wrapper()
        for tensor in wrapper.get_layer_static_inputs(LOCAL_ROWS, 1).values():
            assert tensor.device == wrapper.anchor.device

    @pytest.mark.parametrize(
        ("seq_length", "micro_batch_size"), [(LOCAL_ROWS * 2 + 1, 1), (LOCAL_ROWS, 1), (LOCAL_ROWS * 2, 2)]
    )
    def test_rejects_other_shapes(self, seq_length, micro_batch_size):
        with pytest.raises(ValueError, match="capture packed rows of"):
            _h_wrapper(cp=2).get_layer_static_inputs(seq_length, micro_batch_size)

    def test_needs_the_capacities(self):
        with pytest.raises(ValueError, match="max_seqlen_per_dp_cp_rank and thd_max_packed_sequences"):
            _h_wrapper(local_rows=None).get_layer_static_inputs(LOCAL_ROWS, 1)


# ---------------------------------------------------------------------------------------------
# T3: which wrappers of a model become H-graph wrappers
# ---------------------------------------------------------------------------------------------


@needs_hook
class TestSwapScope:
    PATTERN = "H-DEHEDE/DE"

    @staticmethod
    def _wrappers(model) -> tuple[list, list]:
        decoder = list(model.decoder.layers)
        mtp = [layer for depth in model.mtp.layers for layer in depth.mtp_model_layer.layers]
        return decoder, mtp

    def test_without_cuda_graphs_nothing_changes_class(self, pg_collection):
        model = _build_model(pg_collection, pattern=self.PATTERN)
        decoder, mtp = self._wrappers(model)
        assert len(decoder) == 8 and len(mtp) == 2
        assert all(type(layer) is HyperConnectionHybridLayer for layer in decoder + mtp)

    def test_transformer_engine_graphs_convert_exactly_the_decoder_h_wrappers(self, pg_collection):
        plain = _build_model(pg_collection, pattern=self.PATTERN)
        graphed = _build_model(pg_collection, pattern=self.PATTERN, **_TE_ATTENTION_GRAPHS)
        decoder, mtp = self._wrappers(graphed)
        for index, layer in enumerate(decoder):
            is_h = isinstance(getattr(layer.inner_layer, "self_attention", None), Ernie55HCASelfAttention)
            assert is_h == (self.PATTERN[index] == "H"), index
            expected = Ernie55HCAHyperConnectionHybridLayer if is_h else HyperConnectionHybridLayer
            assert type(layer) is expected, (index, type(layer).__name__)
        assert all(type(layer) is HyperConnectionHybridLayer for layer in mtp)
        # Same module tree, parameter order, values and checkpoint keys as without graphs.
        assert [name for name, _ in graphed.named_modules()] == [name for name, _ in plain.named_modules()]
        plain_params = dict(plain.named_parameters())
        assert list(dict(graphed.named_parameters())) == list(plain_params)
        for name, param in graphed.named_parameters():
            assert torch.equal(param, plain_params[name]), name
        assert list(graphed.state_dict()) == list(plain.state_dict())
        # Idempotent; nested stacks are not visited.
        assert use_ernie55_attention_cuda_graphs(graphed.decoder) == 2
        assert use_ernie55_attention_cuda_graphs(graphed.mtp.layers[0].mtp_model_layer) == 0


def test_conversion_rejects_unknown_wrapper_subclasses():
    class OtherWrapper(HyperConnectionHybridLayer):
        pass

    wrapper = OtherWrapper.__new__(OtherWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper.inner_layer = SimpleNamespace(self_attention=Ernie55HCASelfAttention.__new__(Ernie55HCASelfAttention))
    with pytest.raises(TypeError, match="converts Megatron-Core's HyperConnectionHybridLayer only"):
        use_ernie55_attention_cuda_graphs(SimpleNamespace(layers=[wrapper]))


# ---------------------------------------------------------------------------------------------
# T4: discovery filter
# ---------------------------------------------------------------------------------------------


def _helper_with_discovery(monkeypatch, chunks) -> Ernie55TECudaGraphHelper:
    """A helper whose Megatron-Core discovery returns ``chunks`` = ``[(chunk, [(callable, is_mtp), ...]), ...]``."""

    def discover(self):
        self.chunks_with_decoder = [chunk for chunk, _ in chunks]
        self.callables_per_chunk = [[layer for layer, _ in items] for _, items in chunks]
        self.callables_per_chunk_is_mtp = [[is_mtp for _, is_mtp in items] for _, items in chunks]
        self.num_layers_per_chunk = [len(items) for _, items in chunks]
        self.flattened_callables = [layer for _, items in chunks for layer, _ in items]
        self.flattened_callables_is_mtp = [is_mtp for _, items in chunks for _, is_mtp in items]

    monkeypatch.setattr(TECudaGraphHelper, "_discover_layers", discover)
    return Ernie55TECudaGraphHelper.__new__(Ernie55TECudaGraphHelper)


class TestDiscoveryFilter:
    def test_keeps_only_decoder_h_wrappers(self, monkeypatch):
        """Chunk 0: decoder H, D, H, D and MTP D, H; chunk 1: a D; chunk 2: no decoder; chunk 3: an H."""
        h1, h5, mtp_h, h12 = (_h_wrapper(layer_number=number) for number in (1, 5, 10, 12))
        d3, d7, mtp_d = _d_wrapper(3), _d_wrapper(7), _d_wrapper(9)
        chunk0 = SimpleNamespace(mtp=SimpleNamespace(parameters=lambda: iter([mtp_d.anchor, mtp_h.anchor])))
        chunk1, chunk3 = SimpleNamespace(), SimpleNamespace()
        helper = _helper_with_discovery(
            monkeypatch,
            [
                (chunk0, [(h1, False), (d3, False), (h5, False), (d7, False), (mtp_d, True), (mtp_h, True)]),
                (chunk1, [(_d_wrapper(4), False)]),
                (None, []),
                (chunk3, [(h12, False)]),
            ],
        )
        helper._discover_layers()
        assert helper.callables_per_chunk == [[h1, h5], [], [], [h12]]
        assert helper.callables_per_chunk_is_mtp == [[False, False], [], [], [False]]
        assert helper.num_layers_per_chunk == [2, 0, 0, 1]
        assert helper.flattened_callables == [h1, h5, h12]
        assert helper.flattened_callables_is_mtp == [False, False, False]
        expected_chunks = [chunk0, None, None, chunk3]
        assert len(helper.chunks_with_decoder) == len(expected_chunks)
        assert all(chunk is expected for chunk, expected in zip(helper.chunks_with_decoder, expected_chunks))

    def test_raises_when_nothing_is_kept(self, monkeypatch):
        helper = _helper_with_discovery(
            monkeypatch, [(SimpleNamespace(), [(_d_wrapper(), False), (_h_wrapper(), True)])]
        )
        helper.config = SimpleNamespace(pipeline_model_parallel_size=1)
        with pytest.raises(RuntimeError, match="no H attention layer to graph"):
            helper._discover_layers()

    def test_a_pipeline_stage_without_h_layers_captures_nothing(self, monkeypatch, caplog):
        """A D-only stage (e.g. a light last stage with the MTP depth) keeps nothing, like Megatron-Core's helper."""
        chunks = [(SimpleNamespace(), [(_d_wrapper(), False), (_h_wrapper(), True)]), (None, [])]
        helper = _helper_with_discovery(monkeypatch, chunks)
        helper.config = SimpleNamespace(pipeline_model_parallel_size=2)
        with caplog.at_level("WARNING", logger="megatron.bridge.models.ernie.modeling_ernie55.cuda_graph"):
            helper._discover_layers()
        assert "has no H attention layer and captures nothing" in caplog.text
        assert helper.chunks_with_decoder == [None, None]
        assert helper.callables_per_chunk == [[], []]
        assert helper.callables_per_chunk_is_mtp == [[], []]
        assert helper.num_layers_per_chunk == [0, 0]
        assert helper.flattened_callables == [] and helper.flattened_callables_is_mtp == []

    @pytest.mark.parametrize("foreign", ["unconverted-h", "linear"])
    def test_raises_on_a_foreign_callable(self, monkeypatch, foreign):
        if foreign == "linear":
            layer = torch.nn.Linear(2, 2)
        else:
            layer = HyperConnectionHybridLayer.__new__(HyperConnectionHybridLayer)
            torch.nn.Module.__init__(layer)
            layer.inner_layer = SimpleNamespace(
                self_attention=Ernie55HCASelfAttention.__new__(Ernie55HCASelfAttention)
            )
        helper = _helper_with_discovery(monkeypatch, [(SimpleNamespace(), [(_h_wrapper(), False), (layer, False)])])
        with pytest.raises(TypeError, match="neither an H nor a D attention wrapper"):
            helper._discover_layers()

    def test_raises_on_a_parameter_shared_with_the_mtp_block(self, monkeypatch):
        h = _h_wrapper()
        chunk = SimpleNamespace(mtp=SimpleNamespace(parameters=lambda: iter([h.anchor])))
        helper = _helper_with_discovery(monkeypatch, [(chunk, [(h, False)])])
        with pytest.raises(RuntimeError, match="shares parameters with the MTP block"):
            helper._discover_layers()

    def test_raises_on_a_group_tail(self, monkeypatch):
        h = _h_wrapper()
        object.__setattr__(h, "_te_cuda_graph_group_tail", _d_wrapper())
        helper = _helper_with_discovery(monkeypatch, [(SimpleNamespace(), [(h, False)])])
        with pytest.raises(RuntimeError, match="grouped H layer 1 with the next layer"):
            helper._discover_layers()

    @needs_hook
    def test_megatron_core_discovery_of_a_model_keeps_its_decoder_h_layers(self, pg_collection):
        """Megatron-Core's own discovery on a CPU ``H-DEHEDE/DE`` model (SD-MTP), then the filter."""
        from megatron.core.transformer import cuda_graphs as mcore_cuda_graphs

        model = _build_model(pg_collection, pattern=TestSwapScope.PATTERN, **_TE_ATTENTION_GRAPHS)
        helper = Ernie55TECudaGraphHelper.__new__(Ernie55TECudaGraphHelper)
        helper.model, helper.config = [model], model.config
        helper.tp_group, helper.dp_cp_group = pg_collection.tp, pg_collection.dp_cp
        # What Megatron-Core alone would graph: every attention wrapper, the MTP depth's D included.
        mcore_cuda_graphs.TECudaGraphHelper._discover_layers(helper)
        attention = [type(layer.inner_layer.self_attention).__name__ for layer in helper.flattened_callables]
        assert (
            attention.count("Ernie55HCASelfAttention") == 2 and attention.count("Ernie55LatentMQASelfAttention") == 3
        )
        assert helper.flattened_callables_is_mtp.count(True) == 1
        helper._discover_layers()
        decoder = model.decoder.layers
        assert helper.callables_per_chunk == [[decoder[0], decoder[4]]]
        assert helper.callables_per_chunk_is_mtp == [[False, False]]
        assert helper.num_layers_per_chunk == [2]
        assert helper.flattened_callables == [decoder[0], decoder[4]]
        assert helper.chunks_with_decoder[0] is model


# ---------------------------------------------------------------------------------------------
# T5: replay checks
# ---------------------------------------------------------------------------------------------


def _row_params(*, total: int, documents: int = 1, cp_partition_mode: str = "contiguous", **overrides):
    """Packed metadata of one row of ``total`` rows split into ``documents`` documents (on CPU)."""
    bounds = [0] + [total * (index + 1) // documents for index in range(documents)]
    values = dict(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor(bounds, dtype=torch.int32),
        cu_seqlens_kv=torch.tensor(bounds, dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor(bounds, dtype=torch.int32),
        cu_seqlens_kv_padded=torch.tensor(bounds, dtype=torch.int32),
        max_seqlen_q=total,
        max_seqlen_kv=total,
        cp_partition_mode=cp_partition_mode,
    )
    values.update(overrides)
    return PackedSeqParams(**values)


@pytest.fixture
def replays(monkeypatch):
    """Record Megatron-Core's replay (``HyperConnectionHybridLayer._te_cuda_graph_replay``) instead of running it."""
    calls = []

    def fake_replay(self, *args, **kwargs):
        calls.append((args, kwargs))
        return "replayed", None

    monkeypatch.setattr(HyperConnectionHybridLayer, "_te_cuda_graph_replay", fake_replay)
    return calls


class TestReplayChecks:
    @pytest.mark.parametrize(("cp", "documents"), [(1, 1), (2, 1), (2, 2)])
    def test_valid_rows_reach_megatron_core(self, replays, cp, documents):
        wrapper = _h_wrapper(cp=cp, documents=documents)
        params = _row_params(total=LOCAL_ROWS * cp, documents=documents)
        hidden = torch.zeros(LOCAL_ROWS, 1, WIDTH)
        result = wrapper._te_cuda_graph_replay(
            hidden_states=hidden, attention_mask=None, rotary_pos_emb=None, padding_mask=None, packed_seq_params=params
        )
        assert result == ("replayed", None)
        ((args, kwargs),) = replays
        assert args == () and kwargs["hidden_states"] is hidden and kwargs["packed_seq_params"] is params
        # A shorter longest document, and the hidden states passed positionally, replay as well.
        shorter = _row_params(total=LOCAL_ROWS * cp, documents=documents, max_seqlen_q=3)
        wrapper._te_cuda_graph_replay(hidden, packed_seq_params=shorter)
        assert len(replays) == 2

    @pytest.mark.parametrize(
        ("case", "message"),
        [
            ("bf16-streams", "replays fp32 streams"),
            ("fewer-rows", "replays fp32 streams"),
            ("one-stream", "replays fp32 streams"),
            ("hidden-twice", "only positional input"),
            ("attention-mask", r"has no input \['attention_mask'\]"),
            ("padding-mask", r"has no input \['padding_mask'\]"),
            ("no-packed-params", "packed THD micro-batches only"),
            ("sbhd", "packed THD micro-batches only"),
            ("zigzag", "contiguous CP partition only"),
            ("long-document", "max_seqlen_q"),
            ("device-max-seqlen", "max_seqlen_kv"),
            ("two-documents", "exactly thd_max_packed_sequences=1 documents"),
            ("missing-boundaries", "cu_seqlens_kv_padded must be int32"),
            ("int64-boundaries", "cu_seqlens_q must be int32"),
        ],
    )
    def test_rejects_rows_the_graph_cannot_replay(self, replays, case, message):
        cp = 2
        wrapper = _h_wrapper(cp=cp)
        total = LOCAL_ROWS * cp
        hidden = torch.zeros(LOCAL_ROWS, 1, WIDTH)
        args = ()
        kwargs = dict(hidden_states=hidden, attention_mask=None, packed_seq_params=_row_params(total=total))
        if case == "bf16-streams":
            kwargs["hidden_states"] = hidden.bfloat16()
        elif case == "fewer-rows":
            kwargs["hidden_states"] = hidden[:-1]
        elif case == "one-stream":
            kwargs["hidden_states"] = hidden[..., :HIDDEN]
        elif case == "hidden-twice":
            args = (hidden,)
        elif case == "attention-mask":
            kwargs["attention_mask"] = torch.ones(1, 1, LOCAL_ROWS, total, dtype=torch.bool)
        elif case == "padding-mask":
            kwargs["padding_mask"] = torch.zeros(1, LOCAL_ROWS, dtype=torch.bool)
        elif case == "no-packed-params":
            kwargs["packed_seq_params"] = None
        elif case == "sbhd":
            kwargs["packed_seq_params"] = _row_params(total=total, qkv_format="sbhd")
        elif case == "zigzag":
            kwargs["packed_seq_params"] = _row_params(total=total, cp_partition_mode="zigzag")
        elif case == "long-document":
            kwargs["packed_seq_params"] = _row_params(total=total, max_seqlen_q=total + 1)
        elif case == "device-max-seqlen":
            kwargs["packed_seq_params"] = _row_params(total=total, max_seqlen_kv=torch.tensor(total))
        elif case == "two-documents":
            kwargs["packed_seq_params"] = _row_params(total=total, documents=2)
        elif case == "missing-boundaries":
            kwargs["packed_seq_params"] = _row_params(total=total, cu_seqlens_kv_padded=None)
        elif case == "int64-boundaries":
            kwargs["packed_seq_params"] = _row_params(total=total, cu_seqlens_q=torch.tensor([0, total]))
        with pytest.raises(ValueError, match=message):
            wrapper._te_cuda_graph_replay(*args, **kwargs)
        assert replays == [], "Megatron-Core's replay ran after a failed check"


# ---------------------------------------------------------------------------------------------
# T6: the training loop's helper class
# ---------------------------------------------------------------------------------------------


class TestTrainingLoopHelperClass:
    @pytest.fixture
    def train_module(self):
        return pytest.importorskip("megatron.bridge.training.train")

    def test_ernie_config_class_names_the_ernie_helper(self, train_module):
        # A class attribute, not a config field: YAML, overrides and serialization never see it.
        assert "te_cuda_graph_helper" not in {field.name for field in dataclasses.fields(Ernie55TransformerConfig)}
        config = Ernie55TransformerConfig.__new__(Ernie55TransformerConfig)
        assert train_module._te_cuda_graph_helper_class(config) is Ernie55TECudaGraphHelper

    @pytest.mark.parametrize(
        "make_config", [lambda: SimpleNamespace(cuda_graph_impl="transformer_engine"), Mock, MagicMock]
    )
    def test_other_configs_get_megatron_cores_helper(self, train_module, monkeypatch, make_config):
        assert train_module._te_cuda_graph_helper_class(make_config()) is TECudaGraphHelper
        # The module-level name is looked up at call time, so tests can replace Megatron-Core's helper.
        replacement = type("ReplacementHelper", (), {})
        monkeypatch.setattr(train_module, "TECudaGraphHelper", replacement)
        assert train_module._te_cuda_graph_helper_class(make_config()) is replacement

    def test_a_target_that_is_not_a_class_raises(self, train_module):
        class Config:
            te_cuda_graph_helper = (
                "megatron.bridge.models.ernie.modeling_ernie55.cuda_graph.use_ernie55_attention_cuda_graphs"
            )

        with pytest.raises(TypeError, match="is not a class"):
            train_module._te_cuda_graph_helper_class(Config())

    def test_helper_signature_takes_the_process_groups(self):
        """The training loop passes pg_collection only to helpers whose constructor names it."""
        assert "pg_collection" in inspect.signature(Ernie55TECudaGraphHelper.__init__).parameters


# ---------------------------------------------------------------------------------------------
# T7: NCCL environment check
# ---------------------------------------------------------------------------------------------


class TestNcclGraphRegister:
    @pytest.fixture
    def base_init(self, monkeypatch):
        """Record Megatron-Core's constructor instead of running it."""
        calls = []
        monkeypatch.setattr(TECudaGraphHelper, "__init__", lambda self, *args, **kwargs: calls.append((args, kwargs)))
        for name in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF", "NCCL_GRAPH_REGISTER"):
            monkeypatch.delenv(name, raising=False)
        return calls

    @pytest.mark.parametrize("variable", ["PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"])
    @pytest.mark.parametrize("register", [None, "1"])
    def test_expandable_segments_need_nccl_graph_register_off(self, monkeypatch, base_init, variable, register):
        monkeypatch.setenv(variable, "max_split_size_mb:512,expandable_segments:True")
        if register is not None:
            monkeypatch.setenv("NCCL_GRAPH_REGISTER", register)
        with pytest.raises(RuntimeError, match=f"{variable} enables expandable segments.*NCCL_GRAPH_REGISTER=0"):
            Ernie55TECudaGraphHelper([], None, 8, 1)
        assert base_init == []

    @pytest.mark.parametrize(
        "environment",
        [
            {},
            {"PYTORCH_ALLOC_CONF": "expandable_segments:True", "NCCL_GRAPH_REGISTER": "0"},
            {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "NCCL_GRAPH_REGISTER": "0"},
            {"PYTORCH_ALLOC_CONF": "expandable_segments:False"},
        ],
        ids=lambda environment: ",".join(f"{key}={value}" for key, value in environment.items()) or "unset",
    )
    def test_other_environments_reach_megatron_core(self, monkeypatch, base_init, environment):
        for name, value in environment.items():
            monkeypatch.setenv(name, value)
        model, config, groups = [object()], object(), object()
        Ernie55TECudaGraphHelper(model, config, 8, 1, optimizers=None, pg_collection=groups)
        ((args, kwargs),) = base_init
        assert args == (model, config, 8, 1)
        assert kwargs == {"optimizers": [], "pg_collection": groups}


_LOGGER = "megatron.bridge.models.ernie.modeling_ernie55.cuda_graph"
_REUSE = "expandable_segments:True,graph_capture_record_stream_reuse:True"


class TestRecordStreamReuseWarning:
    """Offloading inside the graphs without graph_capture_record_stream_reuse costs memory: the helper warns."""

    @pytest.fixture
    def warnings_for(self, monkeypatch, caplog):
        """Build the helper (Megatron-Core's constructor stubbed) in an environment; return the reuse warnings."""
        monkeypatch.setattr(TECudaGraphHelper, "__init__", lambda self, *args, **kwargs: None)
        for name in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("NCCL_GRAPH_REGISTER", "0")

        def run(environment: dict[str, str], *, offloading: bool = True) -> list[str]:
            for name, value in environment.items():
                monkeypatch.setenv(name, value)
            caplog.clear()
            config = SimpleNamespace(fine_grained_activation_offloading=offloading)
            with caplog.at_level("WARNING", logger=_LOGGER):
                Ernie55TECudaGraphHelper([], config, 8, 1)
            return [record.getMessage() for record in caplog.records if "record_stream_reuse" in record.getMessage()]

        return run

    @pytest.mark.parametrize(
        "environment",
        [
            {},
            {"PYTORCH_ALLOC_CONF": "expandable_segments:True"},
            {"PYTORCH_ALLOC_CONF": _REUSE, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            {"PYTORCH_CUDA_ALLOC_CONF": _REUSE, "PYTORCH_ALLOC_CONF": "expandable_segments:True"},
        ],
        ids=["unset", "expandable-only", "cuda-variable-without", "new-variable-without"],
    )
    def test_warns_unless_every_set_variable_enables_it(self, warnings_for, environment):
        (message,) = warnings_for(environment)
        assert "graph_capture_record_stream_reuse:True in every allocator variable" in message

    @pytest.mark.parametrize(
        "environment",
        [
            {"PYTORCH_ALLOC_CONF": _REUSE},
            {"PYTORCH_ALLOC_CONF": _REUSE, "PYTORCH_CUDA_ALLOC_CONF": _REUSE},
        ],
        ids=["new-variable", "both"],
    )
    def test_quiet_when_enabled(self, warnings_for, environment):
        assert warnings_for(environment) == []

    def test_quiet_without_offloading(self, warnings_for):
        assert warnings_for({}, offloading=False) == []


class TestCaptureMemoryLog:
    """After capture the helper logs the rank's allocated and reserved memory (the graph pool is only reserved)."""

    @pytest.fixture
    def capture(self, monkeypatch, caplog):
        calls = []
        monkeypatch.setattr(TECudaGraphHelper, "create_cudagraphs", lambda self: calls.append(self))
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *args, **kwargs: 3 * 2**30)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *args, **kwargs: 5 * 2**30)

        def run(callables: list) -> str:
            helper = Ernie55TECudaGraphHelper.__new__(Ernie55TECudaGraphHelper)
            helper.flattened_callables = callables
            helper.model = []  # Megatron-Core's constructor sets it; no DDP chunk here (D10 tests the re-learn)
            with caplog.at_level("INFO", logger=_LOGGER):
                helper.create_cudagraphs()
            assert calls == [helper], "Megatron-Core's capture did not run exactly once"
            calls.clear()
            return caplog.text

        return run

    def test_logs_allocated_and_reserved_memory_after_capture(self, capture):
        text = capture([object()])
        assert "after capture 3.00 GiB allocated, 5.00 GiB reserved; the 2.00 GiB reserved but not allocated" in text

    def test_a_stage_that_captured_nothing_logs_nothing(self, capture):
        assert "after capture" not in capture([])


# ---------------------------------------------------------------------------------------------
# T8: replay inside a whole model's forward (the training call path)
# ---------------------------------------------------------------------------------------------


@needs_hook
class TestReplayThroughTheModel:
    """A CPU ``H-DE`` model with a recording stand-in for the H layer's graph, fed a packed THD row as in training.

    The H wrapper replays through Megatron-Core's ``__call__`` with the keyword arguments ``HybridStack`` passes; the
    stand-in graph records what reaches it and returns its input streams. The D and MoE layers run eagerly.
    """

    DOCUMENTS = (180, 120)  # SEQ rows at CP = 1

    @pytest.fixture
    def run_model(self, pg_collection, monkeypatch):
        from megatron.core.transformer.experimental_attention_variant import dsa

        from megatron.bridge.models.ernie.ernie55_context_parallel import IGNORE_LABEL, build_ernie55_thd_cp_batch

        # Megatron-Core's hyper-connection forward opens NVTX ranges, which need a CUDA build.
        monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *args, **kwargs: contextlib.nullcontext())
        dsa.DSAIndexerLossLoggingHelper.tracker.clear()

        def run(documents: int):
            overrides = {**_TE_ATTENTION_GRAPHS, "thd_max_packed_sequences": documents}
            model = _build_model(pg_collection, pattern="H-DE", ernie_mtp_shared_last_layer=False, **overrides)
            generator = torch.Generator().manual_seed(7)
            samples = []
            for length in self.DOCUMENTS:
                tokens = torch.randint(1, MODEL_VOCAB, (length,), generator=generator)
                labels = tokens.roll(-1)
                labels[-1] = IGNORE_LABEL
                samples.append((tokens, labels))
            batch = build_ernie55_thd_cp_batch(samples, pad_token_id=0, cp_group=None)
            calls = []

            def graph(*args, **kwargs):
                calls.append((args, kwargs))
                return (args[0].clone(),)

            wrapper = model.decoder.layers[0]
            assert type(wrapper) is Ernie55HCAHyperConnectionHybridLayer
            wrapper.cuda_graphs = [graph]
            losses = model(
                input_ids=batch.tokens,
                position_ids=batch.position_ids,
                attention_mask=None,
                labels=batch.labels.clamp(min=0),
                loss_mask=batch.loss_mask,
                packed_seq_params=batch.packed_seq_params,
            )
            (losses * batch.loss_mask).sum().backward()
            return losses, batch, calls

        yield run
        dsa.DSAIndexerLossLoggingHelper.tracker.clear()
        dsa.DSAIndexerLossAutoScaler.main_loss_backward_scale = None

    def test_the_graph_gets_exactly_its_captured_inputs(self, run_model):
        losses, batch, calls = run_model(documents=len(self.DOCUMENTS))
        assert bool(torch.isfinite(losses).all())
        ((args, kwargs),) = calls
        (hidden,) = args
        assert hidden.dtype == torch.float32 and tuple(hidden.shape) == (SEQ, 1, MODEL_STREAMS * MODEL_HIDDEN)
        boundaries = batch.packed_seq_params.cu_seqlens_q_padded
        for name in ("cu_seqlens_q", "cu_seqlens_kv", "cu_seqlens_q_padded", "cu_seqlens_kv_padded"):
            assert kwargs[name].dtype == torch.int32 and torch.equal(kwargs[name], boundaries), name
        assert "packed_seq_params" not in kwargs and "hidden_states" not in kwargs
        # Everything else Megatron-Core hands over is None (TE drops what it did not capture), except its flag.
        others = {name for name, value in kwargs.items() if not name.startswith("cu_seqlens") and value is not None}
        assert others == {"is_first_microbatch"}, sorted(others)

    def test_a_row_with_another_document_count_raises_before_the_graph(self, run_model):
        with pytest.raises(ValueError, match="exactly thd_max_packed_sequences=1 documents"):
            run_model(documents=1)


# ---------------------------------------------------------------------------------------------
# T9: the D attention refuses a graph capture
# ---------------------------------------------------------------------------------------------


@needs_hook
@pytest.mark.parametrize("cuda_graph_impl", ["none", "transformer_engine"])
def test_d_attention_refuses_a_graph_capture(pg_collection, monkeypatch, cuda_graph_impl):
    """With TE graphs a D attention raises under a capture before reading its inputs; without graphs it never asks.

    A captured D layer would replay the host CP routes and split sizes of the capture micro-batch for every later row,
    so this guard is the D layers' safety net against any capture path other than Ernie55TECudaGraphHelper.
    """
    overrides = _TE_ATTENTION_GRAPHS if cuda_graph_impl == "transformer_engine" else {}
    model = _build_model(pg_collection, pattern="H-DE", ernie_mtp_shared_last_layer=False, **overrides)
    attention = model.decoder.layers[2].inner_layer.self_attention
    assert isinstance(attention, Ernie55LatentMQASelfAttention)
    queries = []

    def capturing() -> bool:
        queries.append(True)
        return True

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", capturing)
    if cuda_graph_impl == "transformer_engine":
        with pytest.raises(RuntimeError, match="D layers run eagerly"):
            attention.forward(hidden_states=None, attention_mask=None)
        assert queries == [True]
    else:
        # Past the (absent) guard the forward fails on the missing input, not on the capture.
        with pytest.raises(Exception) as error:
            attention.forward(hidden_states=None, attention_mask=None)
        assert "D layers run eagerly" not in str(error.value)
        assert queries == [], "without CUDA graphs the D attention must not query the capture state"


# ---------------------------------------------------------------------------------------------
# T10: the RoPE tables an H graph reads live as long as its graphs
# ---------------------------------------------------------------------------------------------


@needs_hook
class TestRopeTablesLiveWithTheGraphs:
    """An H graph reads its RoPE table in place, so Megatron-Core's shared RoPE cache must not be all that holds it.

    Megatron-Core caches ``RotaryEmbedding.forward`` in one 32-entry ``lru_cache`` shared by every rotary embedding.
    After capture the H layers stop calling theirs (their graphs replay), so the eager D and MTP layers' lookups
    evict the H tables within one forward at the full model depth, and ``cache_clear`` (eval CP rebinding, YaRN's
    constructor) drops them at once. Here each H wrapper runs its capture call on CPU on its static inputs (TE calls it
    for warmup and capture), then the cache is cleared and refilled by 40 other rotary embeddings.
    """

    PATTERN = "H-DEHEDE/DE"

    @pytest.fixture(autouse=True)
    def _cpu_nvtx(self, monkeypatch):
        # Megatron-Core's hyper-connection forward opens NVTX ranges, which need a CUDA build.
        monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *args, **kwargs: contextlib.nullcontext())

    @staticmethod
    def _capture(wrapper, capture) -> list[weakref.ref]:
        """Run ``capture(wrapper, ...)`` on the wrapper's static inputs; return weak references to the tables it read."""
        from megatron.bridge.models.ernie.modeling_ernie55.hca_attention import rotary_embedding_references

        read = []
        embeddings = {id(embedding): embedding for _, embedding in rotary_embedding_references(wrapper)}
        handles = [
            embedding.register_forward_hook(lambda module, args, output: read.append(weakref.ref(output)))
            for embedding in embeddings.values()
        ]
        static_inputs = wrapper.get_layer_static_inputs(SEQ, 1)
        hidden_states = static_inputs.pop("hidden_states")
        try:
            outputs = capture(wrapper, hidden_states, **static_inputs)
        finally:
            for handle in handles:
                handle.remove()
        assert isinstance(outputs, tuple) and outputs[0].dtype == torch.float32
        return read

    @staticmethod
    def _churn_the_shared_cache() -> None:
        """What the eager layers, eval CP rebinding and YaRN's constructor do to the shared cache after capture."""
        RotaryEmbedding.forward.cache_clear()
        for index in range(40):
            RotaryEmbedding(8, rotary_percent=1.0, rotary_base=1000.0 + index)(SEQ, packed_seq=True)
        gc.collect()

    def test_capture_keeps_the_tables_until_the_graphs_are_deleted(self, pg_collection):
        model = _build_model(pg_collection, pattern=self.PATTERN, **_TE_ATTENTION_GRAPHS)
        wrappers = [layer for layer in model.decoder.layers if type(layer) is Ernie55HCAHyperConnectionHybridLayer]
        assert len(wrappers) == 2

        # Control: after Megatron-Core's own capture the shared cache is the only holder of the tables.
        unkept = [
            ref
            for wrapper in wrappers
            for ref in self._capture(wrapper, HyperConnectionHybridLayer._te_cuda_graph_capture)
        ]
        assert unkept, "the H forward read no RoPE table"
        self._churn_the_shared_cache()
        assert all(ref() is None for ref in unkept), "something else holds the tables; the control no longer applies"

        # The ERNIE H wrapper's capture keeps them through any churn of the shared cache...
        kept = [
            ref
            for wrapper in wrappers
            for ref in self._capture(wrapper, Ernie55HCAHyperConnectionHybridLayer._te_cuda_graph_capture)
        ]
        assert len(kept) == len(unkept)
        self._churn_the_shared_cache()
        assert all(ref() is not None for ref in kept)
        # ...until Megatron-Core's graph deletion releases them with the graphs.
        for wrapper in wrappers:
            wrapper.clear_te_cuda_graph_static_hidden_inputs()
        self._churn_the_shared_cache()
        assert all(ref() is None for ref in kept)
