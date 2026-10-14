"""Shape-only QSA shortcut contracts and optional native CUDA attention parity."""

import os
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType

from megatron.bridge.models.qwen.modeling_qwen4_exp import qsa


pytestmark = pytest.mark.unit


def _owner(*, sequence_parallel=False, force_sparse=False):
    module = qsa.QwenSparseSelfAttention.__new__(qsa.QwenSparseSelfAttention)
    torch.nn.Module.__init__(module)
    module.config = SimpleNamespace(
        sequence_parallel=sequence_parallel, qsa_force_sparse=force_sparse, qwen4_mrope=True
    )
    module.pg_collection = SimpleNamespace(tp=object())
    module.indexer = Mock(block_topk=2, compress_ratio=4)
    module.core_attention = Mock()
    return module


@pytest.mark.parametrize("length,expected", [(1, True), (8, True), (11, True), (12, False), (0, False)])
def test_dense_guard_includes_partial_block_only(length, expected):
    module = _owner()
    assert module._can_skip_indexer(torch.empty(length, 1, 8)) is expected


@pytest.mark.parametrize("local_length,expected", [(1, True), (2, False)])
def test_dense_guard_uses_gathered_sequence_length(monkeypatch, local_length, expected):
    module = _owner(sequence_parallel=True)
    monkeypatch.setattr(qsa, "get_pg_size", lambda group: 8)
    assert module._can_skip_indexer(torch.empty(local_length, 1, 8)) is expected


@pytest.mark.parametrize("packed", [False, True])
def test_production_forward_skips_indexer_and_preserves_attention_arguments(monkeypatch, packed):
    module = _owner()
    hidden = torch.randn(8, 1, 8, requires_grad=True)
    rotary = torch.randn(8, 1, 1, 4)
    mask = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    params = PackedSeqParams(qkv_format="thd") if packed else None
    result = (hidden, None)
    attention = Mock(return_value=result)
    monkeypatch.setattr(qsa.Qwen3VLSelfAttention, "forward", attention)
    assert module(hidden, mask, rotary_pos_emb=rotary, packed_seq_params=params) is result
    module.indexer.assert_not_called()
    module.core_attention.set_selection.assert_called_once_with(qsa._DENSE_QSA_SELECTION)
    assert attention.call_args.args == (module, hidden, mask)
    assert attention.call_args.kwargs["key_value_states"] is None
    assert attention.call_args.kwargs["rotary_pos_emb"] is rotary
    assert attention.call_args.kwargs["packed_seq_params"] is params


@pytest.mark.parametrize("force_sparse,length", [(True, 8), (False, 12)])
def test_force_sparse_and_long_packed_sequence_keep_original_indexer(monkeypatch, force_sparse, length):
    module = _owner(force_sparse=force_sparse)
    hidden = torch.randn(length, 1, 8)
    rotary = torch.randn(length, 1, 1, 4)
    params = PackedSeqParams(qkv_format="thd")
    monkeypatch.setattr(qsa.Qwen3VLSelfAttention, "forward", Mock(return_value=(hidden, None)))
    module(hidden, None, rotary_pos_emb=rotary, packed_seq_params=params)
    module.indexer.assert_called_once_with(hidden, rotary, params)
    module.core_attention.set_selection.assert_called_once_with(module.indexer.return_value)


def test_dense_shortcut_preserves_invalid_context_rejections():
    module = _owner()
    hidden = torch.empty(8, 1, 8)
    with pytest.raises(NotImplementedError, match="inference"):
        module(hidden, None, inference_context=object())
    with pytest.raises(AssertionError, match="rotary"):
        module(hidden, None)
    with pytest.raises(AssertionError, match="single row"):
        module(torch.empty(4, 2, 8), None, rotary_pos_emb=hidden, packed_seq_params=PackedSeqParams(qkv_format="thd"))


def test_dense_marker_is_explicit_and_cannot_enable_forced_sparse():
    core = qsa.QSACoreAttention.__new__(qsa.QSACoreAttention)
    torch.nn.Module.__init__(core)
    core.config = SimpleNamespace(qsa_force_sparse=False)
    core._selection = None
    core.dense_core_attention = Mock(return_value=object())
    query = torch.randn(3, 1, 1, 4)
    with pytest.raises(RuntimeError, match="without a block selection"):
        core(query, query, query, None)
    core.set_selection(qsa._DENSE_QSA_SELECTION)
    assert core(query, query, query, None) is core.dense_core_attention.return_value
    core.config.qsa_force_sparse = True
    with pytest.raises(RuntimeError, match="forced sparse"):
        core(query, query, query, None)
    core.dense_core_attention.assert_called_once()


@pytest.fixture
def native_qsa_cuda(tmp_path):
    """Opt-in one-GPU actual Core/TE constructor, never a synthetic backend."""
    if os.environ.get("RUN_QSA_CUDA_TESTS") != "1":
        pytest.skip("Set RUN_QSA_CUDA_TESTS=1 for the native CUDA attention comparison")
    assert torch.cuda.is_available(), "Explicit native QSA qualification requires CUDA"
    from megatron.core import parallel_state
    from megatron.core.models.backends import get_backend_from_config
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.spec_utils import ModuleSpec

    from megatron.bridge.models.qwen.modeling_qwen4_exp.model_config import Qwen4ExpTransformerConfig

    assert not torch.distributed.is_initialized(), "Native fixture owns its single-rank process groups"
    torch.cuda.set_device(0)
    torch.distributed.init_process_group("nccl", init_method=(tmp_path / "rendezvous").as_uri(), rank=0, world_size=1)
    try:
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
        model_parallel_cuda_manual_seed(173)
        config = Qwen4ExpTransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=2,
            num_query_groups=1,
            kv_channels=16,
            ffn_hidden_size=64,
            params_dtype=torch.bfloat16,
            bf16=True,
            normalization="RMSNorm",
            qk_layernorm=True,
            add_bias_linear=False,
            attention_output_gate=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            gradient_accumulation_fusion=False,
            qwen4_mrope=True,
            qsa_indexer_n_heads=2,
            qsa_indexer_head_dim=16,
            qsa_indexer_budget=8,
            qsa_indexer_compress_ratio=4,
            apply_rope_fusion=False,
        )
        backend = get_backend_from_config(config)
        norm = backend.layer_norm(rms_norm=True, for_qk=True)
        modules = qsa.QwenSparseSelfAttentionSubmodules(
            linear_qkv=backend.column_parallel_linear(),
            linear_proj=backend.row_parallel_linear(),
            core_attention=partial(qsa.QSACoreAttention, dense_core_attention=backend.core_attention()),
            q_layernorm=norm,
            k_layernorm=norm,
            indexer=ModuleSpec(
                module=qsa.QSAIndexer,
                submodules=qsa.QSAIndexerSubmodules(linear_qk=backend.linear(), q_layernorm=norm, k_layernorm=norm),
            ),
        )
        module = (
            qsa.QwenSparseSelfAttention(
                config,
                modules,
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
                pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
            )
            .cuda()
            .train()
        )
        yield module
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("recompute", [False, True])
def test_native_attention_outputs_and_all_vjps_two_forwards_reverse_backward(native_qsa_cuda, packed, recompute):
    """Compare the old indexer path with bypass through the actual full attention.

    Two live forward graphs use different inputs/lengths; reverse backward
    exercises the persistent selection under selective core-attention replay.
    This is a kernel-level qualification, not full-model or TP8 acceptance.
    """
    module = native_qsa_cuda
    module.checkpoint_core_attention = recompute
    lengths = [8, 11] if not packed else [8, 10]
    values = [torch.randn(s, 1, 32, device="cuda", dtype=torch.bfloat16) for s in lengths]
    cotangents = [torch.randn_like(x) for x in values]
    frequencies = [torch.randn(s, 1, 1, 16, device="cuda") for s in lengths]
    state_keys = tuple(module.state_dict())

    def run(skip):
        module.zero_grad(set_to_none=True)
        inputs = [x.detach().clone().requires_grad_() for x in values]
        outputs = []
        with patch.object(module, "_can_skip_indexer", return_value=skip):
            for x, freq in zip(inputs, frequencies):
                params = None
                if packed:
                    cu = torch.tensor([0, 4, x.shape[0]], dtype=torch.int32, device="cuda")
                    params = PackedSeqParams(
                        qkv_format="thd",
                        cu_seqlens_q=cu,
                        cu_seqlens_kv=cu,
                        max_seqlen_q=x.shape[0] - 4,
                        max_seqlen_kv=x.shape[0] - 4,
                    )
                out, bias = module(x, None, rotary_pos_emb=freq, packed_seq_params=params)
                assert bias is None
                outputs.append(out)
            for out, grad in reversed(list(zip(outputs, cotangents))):
                out.backward(grad)
        gradients = {n: None if p.grad is None else p.grad.detach().clone() for n, p in module.named_parameters()}
        return [y.detach().clone() for y in outputs], [x.grad.detach().clone() for x in inputs], gradients

    expected, actual = run(False), run(True)
    for old, new in zip(expected[:2], actual[:2]):
        for lhs, rhs in zip(old, new):
            torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)
    assert expected[2].keys() == actual[2].keys()
    assert any("linear_qkv.weight" in n and g is not None for n, g in actual[2].items())
    assert any("linear_proj.weight" in n and g is not None for n, g in actual[2].items())
    for name, gradient in expected[2].items():
        if gradient is None:
            assert actual[2][name] is None, name
        else:
            torch.testing.assert_close(gradient, actual[2][name], rtol=0, atol=0, msg=name)
    assert all(g is None for n, g in actual[2].items() if n.startswith("indexer."))
    assert tuple(module.state_dict()) == state_keys
