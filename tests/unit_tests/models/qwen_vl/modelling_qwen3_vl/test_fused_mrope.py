# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Parity and dispatch tests for the Bridge-owned Qwen-VL fused mRoPE path."""

from types import SimpleNamespace

import pytest
import torch
from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl import fused_mrope
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl import rope as qwen_rope


class FakeCPGroup:
    """Minimal process-group contract used by the rotary dispatcher."""

    def __init__(self, size: int = 1, rank: int = 0) -> None:
        self._size = size
        self._rank = rank

    def size(self) -> int:
        """Return the fake context-parallel size."""
        return self._size

    def rank(self) -> int:
        """Return the fake context-parallel rank."""
        return self._rank


@pytest.fixture(autouse=True)
def clear_fallback_warnings():
    """Keep warn-once state isolated between tests."""
    qwen_rope._ROPE_FUSION_FALLBACK_WARNINGS.clear()
    yield
    qwen_rope._ROPE_FUSION_FALLBACK_WARNINGS.clear()


def _config(
    *,
    apply_rope_fusion: bool,
    mrope_section: list[int],
    mrope_interleaved: bool,
    context_parallel_size: int = 1,
    apply_rotary_pos_emb_in_fp32: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        apply_rope_fusion=apply_rope_fusion,
        mrope_section=mrope_section,
        mrope_interleaved=mrope_interleaved,
        rotary_interleaved=False,
        context_parallel_size=context_parallel_size,
        apply_rotary_pos_emb_in_fp32=apply_rotary_pos_emb_in_fp32,
    )


def _tolerances(dtype: torch.dtype) -> dict[str, float]:
    if dtype == torch.bfloat16:
        return {"rtol": 2.0e-2, "atol": 5.0e-2}
    if dtype == torch.float16:
        return {"rtol": 3.0e-3, "atol": 1.0e-2}
    return {"rtol": 1.0e-6, "atol": 1.0e-6}


def _cp_indices(cu_seqlens: list[int], cp_size: int, cp_rank: int) -> list[int]:
    indices: list[int] = []
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        local_length = (end - start) // cp_size
        first_length = (local_length + 1) // 2
        second_length = local_length // 2
        indices.extend(range(start + cp_rank * first_length, start + (cp_rank + 1) * first_length))
        indices.extend(range(end - (cp_rank + 1) * second_length, end - cp_rank * second_length))
    return indices


@pytest.mark.parametrize(
    "section,interleaved",
    [
        ([11, 11, 10], True),
        ([10, 11, 11], False),
        ([24, 20, 20], True),
        ([0, 16, 16], False),
    ],
)
def test_materialize_raw_mrope_matches_manual_axis_selection(section, interleaved):
    """Raw T/H/W frequencies reproduce the established materialized layout."""
    half_rotary_dim = sum(section)
    freqs = torch.stack(
        [torch.full((1, 7, half_rotary_dim), float(axis + 1), dtype=torch.float32) for axis in range(3)]
    )

    actual = qwen_rope.materialize_mrope_freqs(
        freqs,
        section,
        interleaved_mrope=interleaved,
    )

    if interleaved:
        selected = freqs[0].clone()
        selected[..., 1 : section[1] * 3 : 3] = freqs[1, ..., 1 : section[1] * 3 : 3]
        selected[..., 2 : section[2] * 3 : 3] = freqs[2, ..., 2 : section[2] * 3 : 3]
    else:
        selected = torch.cat(
            (
                freqs[0, ..., : section[0]],
                freqs[1, ..., section[0] : section[0] + section[1]],
                freqs[2, ..., section[0] + section[1] :],
            ),
            dim=-1,
        )
    expected = torch.cat((selected, selected), dim=-1)[..., None, :].transpose(0, 1).contiguous()
    torch.testing.assert_close(actual, expected)


def test_triton_unavailable_falls_back_without_calling_fused(monkeypatch):
    """A no-op or mocked fused API can never silently supply the output."""
    generator = torch.Generator().manual_seed(1234)
    t = torch.randn(9, 2, 3, 80, dtype=torch.float32, generator=generator)
    freqs = torch.randn(3, 2, 9, 32, dtype=torch.float32, generator=generator)
    section = [11, 11, 10]
    config = _config(apply_rope_fusion=True, mrope_section=section, mrope_interleaved=True)

    monkeypatch.setattr(qwen_rope, "get_fused_mrope_unavailable_reason", lambda *args, **kwargs: "no Triton")

    def unexpected_fused_call(*args, **kwargs):
        raise AssertionError("fused mRoPE must not be called after an unavailable result")

    monkeypatch.setattr(qwen_rope, "fused_apply_mrope", unexpected_fused_call)
    with pytest.warns(UserWarning, match="no Triton.*Using the unfused implementation"):
        actual = qwen_rope.apply_rotary_pos_emb_absolute(t, freqs, config)

    materialized = qwen_rope.materialize_mrope_freqs(freqs, section, interleaved_mrope=True)
    expected = _apply_rotary_pos_emb_bshd(t, materialized, rotary_interleaved=False)
    torch.testing.assert_close(actual, expected)


def test_unsupported_interleaved_section_falls_back_before_availability_check(monkeypatch):
    """Legacy Qwen3-VL sections remain correct when fusion is requested."""
    generator = torch.Generator().manual_seed(2234)
    t = torch.randn(6, 1, 2, 128, dtype=torch.float32, generator=generator)
    freqs = torch.randn(3, 1, 6, 64, dtype=torch.float32, generator=generator)
    section = [24, 20, 20]
    config = _config(apply_rope_fusion=True, mrope_section=section, mrope_interleaved=True)

    def unexpected_call(*args, **kwargs):
        raise AssertionError("unsupported section must not reach fused availability or execution")

    monkeypatch.setattr(qwen_rope, "get_fused_mrope_unavailable_reason", unexpected_call)
    monkeypatch.setattr(qwen_rope, "fused_apply_mrope", unexpected_call)
    with pytest.warns(UserWarning, match="stride-three interleaved mRoPE.*unfused"):
        actual = qwen_rope.apply_rotary_pos_emb_absolute(t, freqs, config)

    materialized = qwen_rope.materialize_mrope_freqs(freqs, section, interleaved_mrope=True)
    expected = _apply_rotary_pos_emb_bshd(t, materialized, rotary_interleaved=False)
    torch.testing.assert_close(actual, expected)


def test_bshd_dispatch_selects_fused_raw_path(monkeypatch):
    """Bridge dispatch passes raw frequencies and FP32 intent to Triton."""
    t = torch.zeros(5, 1, 2, 64, dtype=torch.bfloat16)
    freqs = torch.zeros(3, 1, 5, 32, dtype=torch.float32)
    section = [11, 11, 10]
    config = _config(
        apply_rope_fusion=True,
        mrope_section=section,
        mrope_interleaved=True,
        apply_rotary_pos_emb_in_fp32=True,
    )
    calls: dict[str, object] = {}

    monkeypatch.setattr(qwen_rope, "get_fused_mrope_unavailable_reason", lambda *args, **kwargs: None)

    def fake_fused(tensor, raw_freqs, raw_section, **kwargs):
        calls.update(
            tensor=tensor,
            freqs=raw_freqs,
            section=raw_section,
            kwargs=kwargs,
        )
        return tensor + 1

    monkeypatch.setattr(qwen_rope, "fused_apply_mrope", fake_fused)
    actual = qwen_rope.apply_rotary_pos_emb_absolute(t, freqs, config, max_seqlen=5)

    torch.testing.assert_close(actual, torch.ones_like(t))
    assert calls["tensor"] is t
    assert calls["freqs"] is freqs
    assert calls["section"] == section
    assert calls["kwargs"] == {"interleaved_mrope": True, "fp32_compute": True}


def test_materialized_sequence_length_three_is_not_misclassified_as_raw(monkeypatch):
    """Legacy ``[S=3, B, 1, D]`` remains unambiguous against the raw axis dimension."""
    t = torch.randn(3, 1, 2, 80, dtype=torch.float32)
    section = [11, 11, 10]
    freqs = torch.randn(3, 1, 3, 32, dtype=torch.float32)
    materialized = qwen_rope.materialize_mrope_freqs(freqs, section, interleaved_mrope=True)
    config = _config(apply_rope_fusion=True, mrope_section=section, mrope_interleaved=True)

    def unexpected_fused_call(*args, **kwargs):
        raise AssertionError("materialized frequencies must stay on the established path")

    monkeypatch.setattr(qwen_rope, "fused_apply_mrope", unexpected_fused_call)
    actual = qwen_rope.apply_rotary_pos_emb_absolute(t, materialized, config)
    expected = _apply_rotary_pos_emb_bshd(t, materialized, rotary_interleaved=False)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("freqs_are_local", [False, True])
def test_thd_dispatch_propagates_cp_mapping_and_local_cu_seqlens(monkeypatch, freqs_are_local):
    """Packed dispatch distinguishes global fusion input from pre-sharded input."""
    cp_group = FakeCPGroup(size=2, rank=1)
    t = torch.zeros(12, 2, 64, dtype=torch.bfloat16)
    global_freqs = torch.zeros(3, 1, 24, 32, dtype=torch.float32)
    freqs = global_freqs[:, :, _cp_indices([0, 10, 24], 2, 1), :].contiguous() if freqs_are_local else global_freqs
    cu_seqlens = torch.tensor([0, 10, 24], dtype=torch.int32)
    section = [11, 11, 10]
    config = _config(
        apply_rope_fusion=True,
        mrope_section=section,
        mrope_interleaved=True,
        context_parallel_size=2,
    )
    calls: dict[str, object] = {}

    monkeypatch.setattr(qwen_rope, "get_fused_mrope_thd_unavailable_reason", lambda *args, **kwargs: None)

    def fake_fused(tensor, launch_cu_seqlens, raw_freqs, raw_section, **kwargs):
        calls.update(
            tensor=tensor,
            cu_seqlens=launch_cu_seqlens.clone(),
            freqs=raw_freqs,
            section=raw_section,
            kwargs=kwargs,
        )
        return tensor + 1

    monkeypatch.setattr(qwen_rope, "fused_apply_mrope_thd", fake_fused)
    actual = qwen_rope.apply_rotary_pos_emb_absolute(
        t,
        freqs,
        config,
        cu_seqlens,
        cp_group=cp_group,
        max_seqlen=14,
    )

    torch.testing.assert_close(actual, torch.ones_like(t))
    expected_cu = torch.tensor([0, 5, 12], dtype=torch.int32) if freqs_are_local else cu_seqlens
    torch.testing.assert_close(calls["cu_seqlens"], expected_cu)
    assert calls["freqs"] is freqs
    assert calls["section"] == section
    assert calls["kwargs"]["cp_size"] == (1 if freqs_are_local else 2)
    assert calls["kwargs"]["cp_rank"] == (0 if freqs_are_local else 1)


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_thd_unfused_fallback_supports_odd_local_sequence_lengths(monkeypatch, cp_rank):
    """Fallback uses ceil/floor CP segments for odd local packed lengths."""
    generator = torch.Generator().manual_seed(3234)
    cp_group = FakeCPGroup(size=2, rank=cp_rank)
    t = torch.randn(12, 2, 80, dtype=torch.float32, generator=generator)
    freqs = torch.randn(3, 1, 24, 32, dtype=torch.float32, generator=generator)
    cu_seqlens = torch.tensor([0, 10, 24], dtype=torch.int32)
    section = [11, 11, 10]
    config = _config(
        apply_rope_fusion=True,
        mrope_section=section,
        mrope_interleaved=True,
        context_parallel_size=2,
    )

    monkeypatch.setattr(
        qwen_rope,
        "get_fused_mrope_thd_unavailable_reason",
        lambda *args, **kwargs: "unsupported test layout",
    )
    with pytest.warns(UserWarning, match="unsupported test layout.*unfused"):
        actual = qwen_rope.apply_rotary_pos_emb_absolute(
            t,
            freqs,
            config,
            cu_seqlens,
            cp_group=cp_group,
            max_seqlen=14,
        )

    materialized = qwen_rope.materialize_mrope_freqs(freqs, section, interleaved_mrope=True)
    indices = torch.tensor(_cp_indices(cu_seqlens.tolist(), 2, cp_rank), dtype=torch.long)
    expected = _apply_rotary_pos_emb_bshd(
        t[:, None],
        materialized.index_select(0, indices),
        rotary_interleaved=False,
    ).squeeze(1)
    torch.testing.assert_close(actual, expected)


def test_real_unavailable_reason_on_cpu_never_calls_fused(monkeypatch):
    """The vendored availability API produces a safe CPU fallback."""
    t = torch.randn(4, 1, 2, 16)
    freqs = torch.randn(3, 1, 4, 8)
    section = [3, 3, 2]
    config = _config(apply_rope_fusion=True, mrope_section=section, mrope_interleaved=True)

    assert fused_mrope.get_fused_mrope_unavailable_reason(t, freqs) is not None

    def unexpected_fused_call(*args, **kwargs):
        raise AssertionError("CPU fallback must not call the fused API")

    monkeypatch.setattr(qwen_rope, "fused_apply_mrope", unexpected_fused_call)
    with pytest.warns(UserWarning, match="(CUDA tensors|Triton is not available).*unfused"):
        actual = qwen_rope.apply_rotary_pos_emb_absolute(t, freqs, config)

    materialized = qwen_rope.materialize_mrope_freqs(freqs, section, interleaved_mrope=True)
    expected = _apply_rotary_pos_emb_bshd(t, materialized, rotary_interleaved=False)
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_qwen35_rotary_embedding_returns_raw_without_materializing():
    """The language embedding keeps global raw frequencies for packed CP."""
    section = [11, 11, 10]
    position_ids = torch.stack(
        (
            torch.arange(24, device="cuda"),
            torch.arange(24, device="cuda") * 2,
            torch.arange(24, device="cuda") * 3,
        )
    )[:, None, :]
    rotary = qwen_rope.Qwen3VLMultimodalRotaryEmbedding(
        kv_channels=256,
        rotary_percent=0.25,
        rotary_base=10_000_000,
        cp_group=FakeCPGroup(size=2),
        return_raw_freqs=True,
    )
    rotary.is_thd_format = True

    raw_freqs = rotary(position_ids, section)
    rotary.return_raw_freqs = False
    materialized = rotary(position_ids, section)

    assert raw_freqs.shape == (3, 1, 24, 32)
    assert materialized.shape == (24, 1, 1, 64)
    converted = qwen_rope.materialize_mrope_freqs(raw_freqs, section, interleaved_mrope=True)
    torch.testing.assert_close(converted, materialized)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not fused_mrope.is_fused_mrope_available(), reason="Triton fused mRoPE unavailable")
@pytest.mark.parametrize("unsupported", ["input_dtype", "freq_dtype", "head_stride"])
def test_unsupported_cuda_inputs_fall_back_without_calling_fused(monkeypatch, unsupported):
    """Unsupported CUDA dtype and layout cases retain the unfused result."""
    generator = torch.Generator(device="cuda").manual_seed(3734)
    section = [11, 11, 10]
    if unsupported == "input_dtype":
        t = torch.randn(8, 1, 2, 80, dtype=torch.float64, device="cuda", generator=generator)
    elif unsupported == "head_stride":
        t = torch.randn(8, 1, 2, 160, dtype=torch.float32, device="cuda", generator=generator)[..., ::2]
    else:
        t = torch.randn(8, 1, 2, 80, dtype=torch.float32, device="cuda", generator=generator)
    freqs = torch.randn(3, 1, 8, 32, dtype=torch.float32, device="cuda", generator=generator)
    if unsupported == "freq_dtype":
        freqs = freqs.half()
    config = _config(apply_rope_fusion=True, mrope_section=section, mrope_interleaved=True)

    def unexpected_fused_call(*args, **kwargs):
        raise AssertionError("unsupported inputs must not call the fused kernel")

    monkeypatch.setattr(qwen_rope, "fused_apply_mrope", unexpected_fused_call)
    with pytest.warns(UserWarning, match="unavailable.*unfused"):
        actual = qwen_rope.apply_rotary_pos_emb_absolute(t, freqs, config)

    materialized = qwen_rope.materialize_mrope_freqs(freqs, section, interleaved_mrope=True)
    expected = _apply_rotary_pos_emb_bshd(t, materialized, rotary_interleaved=False)
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not fused_mrope.is_fused_mrope_available(), reason="Triton fused mRoPE unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["bshd", "thd"])
def test_fused_dispatch_matches_unfused_forward_backward_cuda(dtype, layout, monkeypatch):
    """Fused and unfused Qwen3.5-VL math match for both supported layouts."""
    generator = torch.Generator(device="cuda").manual_seed(4234)
    section = [11, 11, 10]
    config = _config(apply_rope_fusion=True, mrope_section=section, mrope_interleaved=True)
    if layout == "bshd":
        t_ref = torch.randn(
            32,
            1,
            3,
            256,
            dtype=dtype,
            device="cuda",
            generator=generator,
            requires_grad=True,
        )
        freqs = torch.randn(3, 1, 32, 32, dtype=torch.float32, device="cuda", generator=generator)
        cu_seqlens = None
    else:
        t_ref = torch.randn(
            32,
            3,
            256,
            dtype=dtype,
            device="cuda",
            generator=generator,
            requires_grad=True,
        )
        freqs = torch.randn(3, 1, 32, 32, dtype=torch.float32, device="cuda", generator=generator)
        cu_seqlens = torch.tensor([0, 12, 32], dtype=torch.int32, device="cuda")
    t_fused = t_ref.detach().clone().requires_grad_(True)

    materialized = qwen_rope.materialize_mrope_freqs(freqs, section, interleaved_mrope=True)
    if layout == "bshd":
        expected = _apply_rotary_pos_emb_bshd(t_ref, materialized, rotary_interleaved=False)
        fused_name = "fused_apply_mrope"
    else:
        expected = _apply_rotary_pos_emb_bshd(
            t_ref[:, None],
            materialized,
            rotary_interleaved=False,
        ).squeeze(1)
        fused_name = "fused_apply_mrope_thd"

    fused_calls = 0
    original_fused = getattr(qwen_rope, fused_name)

    def wrapped_fused(*args, **kwargs):
        nonlocal fused_calls
        fused_calls += 1
        return original_fused(*args, **kwargs)

    monkeypatch.setattr(qwen_rope, fused_name, wrapped_fused)
    actual = qwen_rope.apply_rotary_pos_emb_absolute(
        t_fused,
        freqs,
        config,
        cu_seqlens,
        cp_group=FakeCPGroup(),
        max_seqlen=20 if layout == "thd" else 32,
    )

    assert fused_calls == 1
    tolerances = _tolerances(dtype)
    torch.testing.assert_close(actual.float(), expected.float(), **tolerances)
    grad = torch.randn_like(expected)
    expected.backward(grad)
    actual.backward(grad)
    torch.testing.assert_close(t_fused.grad.float(), t_ref.grad.float(), **tolerances)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not fused_mrope.is_fused_mrope_available(), reason="Triton fused mRoPE unavailable")
@pytest.mark.parametrize("cp_rank", [0, 1])
def test_fused_thd_cp_odd_local_lengths_match_manual_reference_cuda(cp_rank, monkeypatch):
    """The Triton THD kernel maps odd local CP halves without gaps or overlap."""
    generator = torch.Generator(device="cuda").manual_seed(5234)
    section = [11, 11, 10]
    cu_seqlens = torch.tensor([0, 10, 24], dtype=torch.int32, device="cuda")
    t_ref = torch.randn(
        12,
        2,
        256,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
        requires_grad=True,
    )
    t_fused = t_ref.detach().clone().requires_grad_(True)
    freqs = torch.randn(3, 1, 24, 32, dtype=torch.float32, device="cuda", generator=generator)
    config = _config(
        apply_rope_fusion=True,
        mrope_section=section,
        mrope_interleaved=True,
        context_parallel_size=2,
    )
    materialized = qwen_rope.materialize_mrope_freqs(freqs, section, interleaved_mrope=True)
    indices = torch.tensor(_cp_indices([0, 10, 24], 2, cp_rank), dtype=torch.long, device="cuda")
    expected = _apply_rotary_pos_emb_bshd(
        t_ref[:, None],
        materialized.index_select(0, indices),
        rotary_interleaved=False,
    ).squeeze(1)
    fused_calls = 0
    original_fused = qwen_rope.fused_apply_mrope_thd

    def counted_fused(*args, **kwargs):
        nonlocal fused_calls
        fused_calls += 1
        return original_fused(*args, **kwargs)

    monkeypatch.setattr(qwen_rope, "fused_apply_mrope_thd", counted_fused)
    actual = qwen_rope.apply_rotary_pos_emb_absolute(
        t_fused,
        freqs,
        config,
        cu_seqlens,
        cp_group=FakeCPGroup(size=2, rank=cp_rank),
        max_seqlen=14,
    )

    assert fused_calls == 1
    tolerances = _tolerances(torch.bfloat16)
    torch.testing.assert_close(actual.float(), expected.float(), **tolerances)
    grad = torch.randn_like(expected)
    expected.backward(grad)
    actual.backward(grad)
    torch.testing.assert_close(t_fused.grad.float(), t_ref.grad.float(), **tolerances)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not fused_mrope.is_fused_mrope_available(), reason="Triton fused mRoPE unavailable")
def test_vision_fp32_fused_thd_matches_explicit_float_compute_cuda(monkeypatch):
    """Vision's BF16 inputs retain the established FP32 rotary math."""
    generator = torch.Generator(device="cuda").manual_seed(6234)
    tokens, heads, head_dim = 24, 2, 72
    section = [0, 16, 16]
    cu_seqlens = torch.tensor([0, 8, 24], dtype=torch.int32, device="cuda")
    t_ref = torch.randn(
        tokens,
        heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
        requires_grad=True,
    )
    t_fused = t_ref.detach().clone().requires_grad_(True)
    freqs = torch.randn(3, 1, tokens, 32, dtype=torch.float32, device="cuda", generator=generator)
    config = _config(
        apply_rope_fusion=True,
        mrope_section=section,
        mrope_interleaved=False,
        apply_rotary_pos_emb_in_fp32=True,
    )

    materialized = qwen_rope.materialize_mrope_freqs(freqs, section, interleaved_mrope=False)
    expected = (
        _apply_rotary_pos_emb_bshd(
            t_ref.float()[:, None],
            materialized,
            rotary_interleaved=False,
        )
        .squeeze(1)
        .to(t_ref.dtype)
    )
    fused_calls = 0
    original_fused = qwen_rope.fused_apply_mrope_thd

    def counted_fused(*args, **kwargs):
        nonlocal fused_calls
        fused_calls += 1
        return original_fused(*args, **kwargs)

    monkeypatch.setattr(qwen_rope, "fused_apply_mrope_thd", counted_fused)
    actual = qwen_rope.apply_rotary_pos_emb_absolute(
        t_fused,
        freqs,
        config,
        cu_seqlens,
        cp_group=FakeCPGroup(),
        max_seqlen=16,
    )

    assert fused_calls == 1
    tolerances = _tolerances(torch.bfloat16)
    torch.testing.assert_close(actual.float(), expected.float(), **tolerances)
    grad = torch.randn_like(expected)
    expected.backward(grad)
    actual.backward(grad)
    torch.testing.assert_close(t_fused.grad.float(), t_ref.grad.float(), **tolerances)
