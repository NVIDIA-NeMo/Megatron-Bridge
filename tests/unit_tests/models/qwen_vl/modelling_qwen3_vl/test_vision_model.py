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

"""Unit tests for Qwen3-VL vision model (CUDA graph sequence padding and helpers)."""

import datetime
import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import Qwen3VLTransformerConfig
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import PatchMergerSubmodules, Qwen3VLVisionPatchMerger
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.vision_model import (
    Qwen3VLVisionModel,
    _maybe_pad_vision_sequence_for_cuda_graph,
    _vision_forward_packed_attention_setup,
)


def _minimal_vision_config(**kwargs):
    base = dict(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        num_query_groups=4,
        kv_channels=16,
        ffn_hidden_size=128,
        vocab_size=100,
        language_max_sequence_length=128,
        num_position_embeddings=64,
        spatial_merge_size=2,
        patch_size=16,
        out_hidden_size=64,
    )
    base.update(kwargs)
    return Qwen3VLTransformerConfig(**base)


def _vision_model_shell(config: Qwen3VLTransformerConfig) -> Qwen3VLVisionModel:
    """Build a :class:`Qwen3VLVisionModel` instance without running ``__init__`` (for helper tests)."""
    m = Qwen3VLVisionModel.__new__(Qwen3VLVisionModel)
    nn.Module.__init__(m)
    m.config = config
    return m


class TestMaybePadVisionSequenceForCudaGraph:
    """Tests for ``_maybe_pad_vision_sequence_for_cuda_graph`` (vision CUDA graph fixed length)."""

    def test_no_pad_when_seq_equals_max(self):
        seq_len, hidden, rot_d = 8, 32, 16
        hidden_states = torch.randn(seq_len, hidden)
        rotary_pos_emb = torch.randn(seq_len, 1, 1, rot_d)
        out_h, out_r, out_len = _maybe_pad_vision_sequence_for_cuda_graph(
            hidden_states, rotary_pos_emb, seq_len, max_seq_len=8
        )
        assert out_len == 8
        assert out_h.shape == (8, hidden)
        assert out_r.shape == (8, 1, 1, rot_d)
        assert torch.equal(out_h, hidden_states)
        assert torch.equal(out_r, rotary_pos_emb)

    def test_pad_to_max_appends_zeros(self):
        seq_len, max_seq_len, hidden, rot_d = 3, 8, 4, 6
        hidden_states = torch.arange(seq_len * hidden, dtype=torch.float32).reshape(seq_len, hidden)
        rotary_pos_emb = torch.ones(seq_len, 1, 1, rot_d)
        out_h, out_r, out_len = _maybe_pad_vision_sequence_for_cuda_graph(
            hidden_states, rotary_pos_emb, seq_len, max_seq_len=max_seq_len
        )
        assert out_len == max_seq_len
        assert out_h.shape == (max_seq_len, hidden)
        assert out_r.shape == (max_seq_len, 1, 1, rot_d)
        assert torch.equal(out_h[:seq_len], hidden_states)
        assert torch.equal(out_h[seq_len:], torch.zeros(max_seq_len - seq_len, hidden))
        assert torch.equal(out_r[:seq_len], rotary_pos_emb)
        assert torch.equal(out_r[seq_len:], torch.zeros(max_seq_len - seq_len, 1, 1, rot_d))

    def test_raises_when_seq_exceeds_max(self):
        seq_len, max_seq_len = 10, 4
        hidden_states = torch.randn(seq_len, 8)
        rotary_pos_emb = torch.randn(seq_len, 1, 1, 4)
        with pytest.raises(ValueError, match="exceeds max_vision_cuda_graph_seq_length"):
            _maybe_pad_vision_sequence_for_cuda_graph(hidden_states, rotary_pos_emb, seq_len, max_seq_len)


class TestVisionForwardPackedAttentionSetup:
    """Tests for ``_vision_forward_packed_attention_setup``."""

    def test_non_cuda_graph_path_calls_builder_and_no_mask(self):
        grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
        hidden = torch.zeros(5, 1, 8, dtype=torch.float32)
        sentinel = object()

        def _builder(_g):
            assert torch.equal(_g, grid)
            return sentinel

        packed, mask = _vision_forward_packed_attention_setup(
            use_cuda_graph_padding=False,
            hidden_states=hidden,
            original_seq_len=5,
            seq_len=5,
            grid_thw=grid,
            build_packed_seq_params=_builder,
        )
        assert packed is sentinel
        assert mask is None

    def test_cuda_graph_no_padding_no_mask(self):
        grid = torch.tensor([[1, 2, 2]], dtype=torch.long)
        hidden = torch.randn(8, 1, 16, dtype=torch.float32)
        packed, mask = _vision_forward_packed_attention_setup(
            use_cuda_graph_padding=True,
            hidden_states=hidden,
            original_seq_len=8,
            seq_len=8,
            grid_thw=grid,
            build_packed_seq_params=lambda _: pytest.fail("should not call build_packed_seq_params"),
        )
        assert packed is None
        assert mask is None

    def test_cuda_graph_with_padding_additive_mask(self):
        original_seq_len, seq_len, hidden = 2, 5, 4
        hidden_states = torch.randn(seq_len, 1, hidden, dtype=torch.float32)
        grid = torch.tensor([[1, 2, 2]], dtype=torch.long)

        packed, mask = _vision_forward_packed_attention_setup(
            use_cuda_graph_padding=True,
            hidden_states=hidden_states,
            original_seq_len=original_seq_len,
            seq_len=seq_len,
            grid_thw=grid,
            build_packed_seq_params=lambda _: pytest.fail("should not call build_packed_seq_params"),
        )
        assert packed is None
        assert mask is not None
        assert mask.shape == (1, 1, seq_len, seq_len)
        assert mask.dtype == hidden_states.dtype
        assert mask.device == hidden_states.device

        min_val = torch.finfo(hidden_states.dtype).min
        # Real token block: no additive masking
        assert torch.all(mask[:, :, :original_seq_len, :original_seq_len] == 0)
        # Padding keys: columns >= original_seq_len
        assert torch.all(mask[:, :, :original_seq_len, original_seq_len:] == min_val)
        # Padding queries: rows >= original_seq_len
        assert torch.all(mask[:, :, original_seq_len:, :] == min_val)


class TestQwen3VLVisionModelCudaGraphHelpers:
    """Tests for ``_get_max_vision_seq_length`` and ``_uses_vision_cuda_graph``."""

    def test_get_max_vision_seq_length_uses_config_override(self):
        cfg = _minimal_vision_config(max_vision_cuda_graph_seq_length=512)
        m = _vision_model_shell(cfg)
        assert m._get_max_vision_seq_length() == 512

    def test_get_max_vision_seq_length_default_from_num_position_embeddings(self):
        cfg = _minimal_vision_config(
            num_position_embeddings=64,
            spatial_merge_size=2,
            max_vision_cuda_graph_seq_length=None,
        )
        m = _vision_model_shell(cfg)
        assert m._get_max_vision_seq_length() == 64 // (2**2)

    def test_get_max_vision_seq_length_zero_override_falls_back_to_default(self):
        """``0`` is falsy in ``if max_vision_cuda_graph_seq_length``; use derived default."""
        cfg = _minimal_vision_config(
            num_position_embeddings=64,
            spatial_merge_size=2,
            max_vision_cuda_graph_seq_length=0,
        )
        m = _vision_model_shell(cfg)
        assert m._get_max_vision_seq_length() == 16

    def test_uses_vision_cuda_graph_true_in_training_with_te_impl(self):
        cfg = _minimal_vision_config(cuda_graph_impl="transformer_engine")
        m = _vision_model_shell(cfg)
        m.train()
        assert m._uses_vision_cuda_graph() is True

    def test_uses_vision_cuda_graph_false_in_eval(self):
        cfg = _minimal_vision_config(cuda_graph_impl="transformer_engine")
        m = _vision_model_shell(cfg)
        m.eval()
        assert m._uses_vision_cuda_graph() is False

    def test_uses_vision_cuda_graph_false_when_impl_not_te(self):
        cfg = _minimal_vision_config(cuda_graph_impl="none")
        m = _vision_model_shell(cfg)
        m.train()
        assert m._uses_vision_cuda_graph() is False


class _NormForTest(nn.LayerNorm):
    """``torch.nn.LayerNorm`` with the ``patch_norm`` build signature."""

    def __init__(self, config, hidden_size, eps):
        super().__init__(hidden_size, eps=eps)


class _LinearForTest(nn.Linear):
    """``torch.nn.Linear`` with the Megatron parallel-linear build signature, returning ``(output, bias)``."""

    def __init__(self, input_size, output_size, *, bias, **kwargs):
        super().__init__(input_size, output_size, bias=bias)

    def forward(self, x):
        return super().forward(x), None


class _RowwiseDecoderForTest(nn.Module):
    """Stand-in for the vision transformer block that transforms every row on its own, so the test isolates the merger.

    Like the real block, it turns the zero padding into non-zero finite rows, so padded rows leaking into the merged
    tokens would show.
    """

    def __init__(self, hidden_size, with_deepstack=False):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.with_deepstack = with_deepstack
        self.rows_seen = []

    def forward(self, hidden_states, attention_mask=None, inference_params=None, rotary_pos_emb=None, **kwargs):
        self.rows_seen.append(hidden_states.shape[0])
        out = hidden_states + torch.tanh(self.proj(hidden_states))
        deepstack = []
        if self.with_deepstack:  # like a deepstack merger: one feature per group of 4 rows
            deepstack = [2.0 * out.reshape(-1, 4 * out.shape[-1])[:, : out.shape[-1]]]
        return out, deepstack


_IN_DIM = 12
_OUT_DIM = 24
# Three images of 24, 8 and 12 patches: 44 patches, every image a multiple of spatial_merge_size**2.
_GRID = torch.tensor([[1, 4, 6], [1, 2, 4], [1, 6, 2]], dtype=torch.long)
_NUM_PATCHES = 44


def _vision_model_for_forward(max_seq_len, with_deepstack=False):
    """A float64 :class:`Qwen3VLVisionModel` that runs the real ``forward`` and the real patch merger on CPU, with
    stand-ins for the patch embedding, position embeddings and transformer block."""
    torch.manual_seed(0)
    config = _minimal_vision_config(
        cuda_graph_impl="transformer_engine",
        max_vision_cuda_graph_seq_length=max_seq_len,
        out_hidden_size=_OUT_DIM,
    )
    m = _vision_model_shell(config)
    m.spatial_merge_size = config.spatial_merge_size
    m.input_tensor = None
    m.patch_embed = nn.Linear(_IN_DIM, config.hidden_size)
    m.decoder = _RowwiseDecoderForTest(config.hidden_size, with_deepstack=with_deepstack)
    m.merger = Qwen3VLVisionPatchMerger(
        config,
        PatchMergerSubmodules(patch_norm=_NormForTest, linear_fc1=_LinearForTest, linear_fc2=_LinearForTest),
    )
    m.to(torch.float64)

    def pos_embeds(grid_thw):
        n = int(grid_thw.prod(-1).sum())
        return torch.linspace(-1.0, 1.0, n * config.hidden_size, dtype=torch.float64).view(n, config.hidden_size)

    m.fast_pos_embed_interpolate = pos_embeds
    m.rot_pos_emb = lambda grid_thw: torch.zeros(int(grid_thw.prod(-1).sum()), 4, dtype=torch.float64)
    m.merger_rows_seen = []
    m.merger.register_forward_pre_hook(lambda module, args: m.merger_rows_seen.append(args[0].shape[0]))
    return m


def _forward_backward(m, x, cotangent, pad):
    """Run forward + backward with (``pad=True``) or without the vision CUDA graph padding path."""
    m.config.cuda_graph_impl = "transformer_engine" if pad else "none"
    m.zero_grad(set_to_none=True)
    x = x.detach().clone().requires_grad_(True)
    out, deepstack = m(x, _GRID)
    (out * cotangent).sum().backward()
    grads = {name: p.grad.detach().clone() for name, p in m.named_parameters()}
    return out.detach(), [f.detach() for f in deepstack], x.grad.detach(), grads


class TestQwen3VLVisionModelMergerOnPaddedRows:
    """With the vision CUDA graph padding, the merger runs on the padded rows and the padded merged tokens are dropped
    afterwards, so its row count does not change with the patch count (Transformer Engine would otherwise build a new
    patch_norm plan per new count). The result must equal merging the real rows only."""

    # 44 = no padding needed; 56 = padded; 57 = a cap that is not a multiple of spatial_merge_size**2
    @pytest.mark.parametrize("max_seq_len", [_NUM_PATCHES, 56, 57])
    def test_padded_path_matches_unpadded_path(self, max_seq_len):
        m = _vision_model_for_forward(max_seq_len)
        x = torch.randn(_NUM_PATCHES, _IN_DIM, dtype=torch.float64)
        cotangent = torch.randn(_NUM_PATCHES // 4, _OUT_DIM, dtype=torch.float64)
        out_p, _, dx_p, grads_p = _forward_backward(m, x, cotangent, pad=True)
        out_u, _, dx_u, grads_u = _forward_backward(m, x, cotangent, pad=False)

        assert m.decoder.rows_seen == [max_seq_len, _NUM_PATCHES]
        assert m.merger_rows_seen == [max_seq_len - max_seq_len % 4, _NUM_PATCHES]
        assert out_p.shape == (_NUM_PATCHES // 4, _OUT_DIM)
        torch.testing.assert_close(out_p, out_u)
        torch.testing.assert_close(dx_p, dx_u)
        assert grads_p.keys() == grads_u.keys()
        assert any(name.startswith("merger.") for name in grads_u)
        for name in grads_u:
            torch.testing.assert_close(grads_p[name], grads_u[name], msg=name)

    def test_merger_row_count_does_not_depend_on_patch_count(self):
        m = _vision_model_for_forward(64)
        for grid in ([[1, 4, 6]], [[1, 2, 4], [1, 6, 2]], [[1, 8, 6], [1, 2, 2]]):  # 24, 20 and 52 patches
            grid = torch.tensor(grid, dtype=torch.long)
            n = int(grid.prod(-1).sum())
            out, _ = m(torch.randn(n, _IN_DIM, dtype=torch.float64), grid)
            assert out.shape == (n // 4, _OUT_DIM)
        assert m.merger_rows_seen == [64, 64, 64]

    def test_padded_rows_get_zero_gradient(self):
        m = _vision_model_for_forward(56)
        seen = {}

        def keep_output_and_grad(module, args, output):
            seen["out"] = output[0].detach().clone()
            output[0].register_hook(lambda grad: seen.setdefault("grad", grad.detach().clone()))

        m.decoder.register_forward_hook(keep_output_and_grad)
        x = torch.randn(_NUM_PATCHES, _IN_DIM, dtype=torch.float64)
        _forward_backward(m, x, torch.randn(_NUM_PATCHES // 4, _OUT_DIM, dtype=torch.float64), pad=True)

        assert seen["out"].shape[0] == 56
        assert torch.isfinite(seen["out"]).all()
        assert seen["out"][_NUM_PATCHES:].abs().sum() > 0  # the padded rows the merger sees are not zero ...
        assert torch.equal(seen["grad"][_NUM_PATCHES:], torch.zeros_like(seen["grad"][_NUM_PATCHES:]))  # ... no grad
        assert seen["grad"][:_NUM_PATCHES].abs().sum() > 0

    def test_deepstack_features_are_cut_to_the_real_merged_tokens(self):
        m = _vision_model_for_forward(56, with_deepstack=True)
        x = torch.randn(_NUM_PATCHES, _IN_DIM, dtype=torch.float64)
        cotangent = torch.randn(_NUM_PATCHES // 4, _OUT_DIM, dtype=torch.float64)
        _, deepstack_p, _, _ = _forward_backward(m, x, cotangent, pad=True)
        _, deepstack_u, _, _ = _forward_backward(m, x, cotangent, pad=False)
        assert len(deepstack_p) == len(deepstack_u) == 1
        assert deepstack_p[0].shape[0] == _NUM_PATCHES // 4
        torch.testing.assert_close(deepstack_p[0], deepstack_u[0])

    # Without the CUDA graph padding (vision graphs off, or eval mode) the merger runs on the decoder output as is.
    @pytest.mark.parametrize(("cuda_graph_impl", "training"), [("none", True), ("transformer_engine", False)])
    def test_unpadded_path_merges_the_decoder_output_unchanged(self, cuda_graph_impl, training):
        m = _vision_model_for_forward(56)
        m.config.cuda_graph_impl = cuda_graph_impl
        m.train(training)
        x = torch.randn(_NUM_PATCHES, _IN_DIM, dtype=torch.float64)
        out, deepstack = m(x, _GRID)
        assert m.decoder.rows_seen == [_NUM_PATCHES]
        assert deepstack == []

        hidden = m.patch_embed(x) + m.fast_pos_embed_interpolate(_GRID)
        expected = m.merger(m.decoder(hidden[:, None])[0])
        assert m.merger_rows_seen == [_NUM_PATCHES, _NUM_PATCHES]
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.run_only_on("GPU")
class TestQwen3VLVisionPatchMergerPaddedRowsTE:
    """The Transformer Engine patch merger gives the real merged tokens and gradients unchanged when it also merges
    finite padded rows that are cut off afterwards."""

    @classmethod
    def setup_class(cls):
        cls.created_process_group = False
        if torch.cuda.is_available() and not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29500")
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
            dist.init_process_group(
                backend="nccl",
                world_size=int(os.environ.get("WORLD_SIZE", "1")),
                rank=int(os.environ.get("RANK", "0")),
                timeout=datetime.timedelta(minutes=30),
            )
            cls.created_process_group = True

    @classmethod
    def teardown_class(cls):
        if cls.created_process_group and dist.is_initialized():
            dist.destroy_process_group()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Transformer Engine merger needs CUDA")
    def test_te_merger_on_padded_rows_matches_real_rows(self):
        from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TENorm, TERowParallelLinear

        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        model_parallel_cuda_manual_seed(123)
        try:
            config = Qwen3VLTransformerConfig(
                num_layers=1,
                hidden_size=128,
                num_attention_heads=2,
                spatial_merge_size=2,
                out_hidden_size=256,
                params_dtype=torch.bfloat16,
                bf16=True,
            )
            merger = Qwen3VLVisionPatchMerger(
                config,
                PatchMergerSubmodules(
                    patch_norm=TENorm, linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear
                ),
            ).cuda()
            real, padded = 200, 256
            x = torch.randn(real, 1, 128, device="cuda", dtype=torch.bfloat16)
            pad_rows = 3.0 * torch.randn(padded - real, 1, 128, device="cuda", dtype=torch.bfloat16)
            cotangent = torch.randn(real // 4, 256, device="cuda", dtype=torch.float32)

            def run(rows):
                merger.zero_grad(set_to_none=True)
                rows = rows.detach().clone().requires_grad_(True)
                out = merger(rows)[: real // 4]
                (out.float() * cotangent).sum().backward()
                grads = {name: p.grad.detach().clone() for name, p in merger.named_parameters()}
                return out.detach(), rows.grad.detach(), grads

            out_u, dx_u, grads_u = run(x)
            out_p, dx_p, grads_p = run(torch.cat([x, pad_rows]))
            torch.testing.assert_close(out_p, out_u)
            torch.testing.assert_close(dx_p[:real], dx_u)
            assert torch.equal(dx_p[real:], torch.zeros_like(dx_p[real:]))
            for name in grads_u:
                torch.testing.assert_close(grads_p[name], grads_u[name], msg=name)
        finally:
            parallel_state.destroy_model_parallel()
