# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the GLM-5.3-Flash VLM wrapper: pipeline stages, packed layouts and provider config."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.packed_seq_params import PackedSeqParams

from megatron.bridge.models.glm5next import glm53flash_bridge
from megatron.bridge.models.glm5next.glm53flash_bridge import (
    GLM53FlashBridge,
    GLM53FlashModel,
    GLM53FlashModelProvider,
)


pytestmark = pytest.mark.unit

HIDDEN = 3
LABEL_OFFSET = 1000


class _Group:
    def __init__(self, size: int, rank: int = 0):
        self._size = size
        self._rank = rank

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank


class _StubLanguageModel:
    """Records the language-model call; embeds a token ID as a constant hidden vector."""

    def __init__(self):
        self.calls = []
        self.embedding_calls = 0
        self.input_tensor = None
        self.shared_embedding_or_output_weight = None

    def embedding(self, input_ids, position_ids=None):
        self.embedding_calls += 1
        # [batch, seq] -> [seq, batch, hidden]
        return input_ids.float().unsqueeze(-1).expand(*input_ids.shape, HIDDEN).transpose(0, 1).contiguous()

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return torch.zeros(1)


def _config(*, tp=1, cp=1, cp_rank=0, fp8=None, sequence_parallel=False, language_model=None):
    pg_collection = SimpleNamespace(tp=_Group(tp), cp=_Group(cp, cp_rank), pp=_Group(1))
    return SimpleNamespace(
        fp8=fp8,
        sequence_parallel=sequence_parallel,
        _pg_collection=pg_collection,
        share_embeddings_and_output_weights=False,
        vision_config=SimpleNamespace(spatial_merge_size=2),
        provide_language_model=lambda **_: language_model or _StubLanguageModel(),
    )


def _model(*, pre_process=True, post_process=True, **config_kwargs) -> GLM53FlashModel:
    config = _config(language_model=_StubLanguageModel(), **config_kwargs)
    # Construct without the HF vision tower, then mark the stage that embeds tokens.
    model = GLM53FlashModel(config, pre_process=False, post_process=post_process)
    model.pre_process = pre_process
    return model


def _dense_batch():
    """Two right-padded rows with valid lengths 2 and 3; labels and loss mask derive from IDs."""
    input_ids = torch.tensor([[11, 12, 0, 0], [21, 22, 23, 0]])
    attention_mask = input_ids != 0
    labels = torch.where(attention_mask, input_ids + LABEL_OFFSET, -100)
    loss_mask = attention_mask.float()
    return input_ids, attention_mask, labels, loss_mask


def _assert_aligned_supervision(call, loss_mask):
    """Packed labels, loss mask and embeddings must follow the packed token order exactly."""
    packed_ids = call["input_ids"]
    valid = packed_ids != 0
    torch.testing.assert_close(call["labels"], torch.where(valid, packed_ids + LABEL_OFFSET, 0))
    torch.testing.assert_close(loss_mask, valid.float())
    torch.testing.assert_close(call["loss_mask"], loss_mask)
    # Embeddings are [tokens, 1, hidden] and encode each packed token ID.
    torch.testing.assert_close(call["decoder_input"][:, 0, 0], packed_ids[0].float())


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def test_non_preprocess_stage_constructs_and_forwards_without_vision_or_embedding():
    model = _model(pre_process=False, post_process=True)
    assert not hasattr(model, "visual")
    assert not hasattr(model, "get_image_features")

    activations = torch.randn(5, 1, HIDDEN)
    model.set_input_tensor(activations)
    input_ids, attention_mask, labels, loss_mask = _dense_batch()
    output, packed_loss_mask = model(
        input_ids=input_ids, labels=labels, loss_mask=loss_mask, attention_mask=attention_mask
    )

    language_model = model.language_model
    assert language_model.input_tensor is activations
    assert language_model.embedding_calls == 0
    assert len(language_model.calls) == 1
    call = language_model.calls[0]
    assert call["decoder_input"] is None
    torch.testing.assert_close(call["input_ids"], torch.tensor([[11, 12, 21, 22, 23]]))
    torch.testing.assert_close(call["packed_seq_params"].cu_seqlens_q, torch.tensor([0, 2, 5], dtype=torch.int32))
    torch.testing.assert_close(call["labels"], torch.tensor([[1011, 1012, 1021, 1022, 1023]]))
    torch.testing.assert_close(packed_loss_mask, torch.ones(1, 5))


@pytest.mark.parametrize(
    ("pp_rank", "vp_stage", "vp_size", "expected"),
    [
        (0, None, None, (True, False)),
        (1, None, None, (False, True)),
        (0, 0, 2, (True, False)),
        (0, 1, 2, (False, False)),
        (1, 0, 2, (False, False)),
        (1, 1, 2, (False, True)),
    ],
)
def test_provider_resolves_omitted_stage_flags(pp_rank, vp_stage, vp_size, expected):
    provider = SimpleNamespace(
        virtual_pipeline_model_parallel_size=vp_size,
        _pg_collection=SimpleNamespace(pp=_Group(2, pp_rank)),
        freeze_language_model=False,
        freeze_vision_model=False,
        freeze_vision_projection=False,
    )
    with (
        patch("megatron.core.pipeline_parallel.utils.get_pg_rank", side_effect=lambda group: group.rank()),
        patch("megatron.core.pipeline_parallel.utils.get_pg_size", side_effect=lambda group: group.size()),
        patch.object(glm53flash_bridge, "GLM53FlashModel") as model_cls,
    ):
        GLM53FlashModelProvider.provide(provider, vp_stage=vp_stage)

    kwargs = model_cls.call_args.kwargs
    assert (kwargs["pre_process"], kwargs["post_process"]) == expected
    assert kwargs["vp_stage"] == vp_stage


# ---------------------------------------------------------------------------
# Dense rows without packed metadata (Bridge BSHD step)
# ---------------------------------------------------------------------------


def test_text_forward_without_attention_mask_matches_all_ones_mask():
    input_ids = torch.tensor([[5, 6, 7], [8, 9, 10]])
    calls = []
    for attention_mask in (None, torch.ones_like(input_ids, dtype=torch.bool)):
        model = _model()
        model(input_ids=input_ids, attention_mask=attention_mask)
        calls.append(model.language_model.calls[0])

    omitted, explicit = calls
    torch.testing.assert_close(omitted["input_ids"], torch.tensor([[5, 6, 7, 8, 9, 10]]))
    torch.testing.assert_close(omitted["input_ids"], explicit["input_ids"])
    torch.testing.assert_close(omitted["decoder_input"], explicit["decoder_input"])
    for field in ("cu_seqlens_q", "cu_seqlens_kv", "cu_seqlens_q_padded"):
        torch.testing.assert_close(
            getattr(omitted["packed_seq_params"], field), getattr(explicit["packed_seq_params"], field)
        )


@pytest.mark.parametrize(
    ("config_kwargs", "expected_cu_seqlens", "expected_width"),
    [
        ({}, [0, 2, 5], 5),
        ({"tp": 2}, [0, 2, 6], 6),
        ({"fp8": "e4m3"}, [0, 16, 128], 128),
        ({"cp": 2, "cp_rank": 0}, [0, 4, 8], 4),
        ({"cp": 2, "cp_rank": 1}, [0, 4, 8], 4),
    ],
)
def test_dense_supervision_uses_the_packed_layout(config_kwargs, expected_cu_seqlens, expected_width):
    model = _model(**config_kwargs)
    input_ids, attention_mask, labels, loss_mask = _dense_batch()
    output, packed_loss_mask = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels, loss_mask=loss_mask
    )

    call = model.language_model.calls[0]
    assert call["input_ids"].shape == (1, expected_width)
    assert call["labels"].shape == (1, expected_width)
    assert packed_loss_mask.shape == (1, expected_width)
    torch.testing.assert_close(
        call["packed_seq_params"].cu_seqlens_q, torch.tensor(expected_cu_seqlens, dtype=torch.int32)
    )
    _assert_aligned_supervision(call, packed_loss_mask)
    # Every valid token appears exactly once across the layout (or this CP rank's share of it).
    if config_kwargs.get("cp", 1) == 1:
        assert sorted(call["input_ids"][call["input_ids"] != 0].tolist()) == [11, 12, 21, 22, 23]


def test_left_padded_integer_mask_keeps_valid_token_order():
    model = _model()
    input_ids = torch.tensor([[0, 0, 11, 12], [0, 21, 22, 23]])
    attention_mask = (input_ids != 0).long()
    labels = torch.where(input_ids != 0, input_ids + LABEL_OFFSET, -100)
    _, packed_loss_mask = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels, loss_mask=attention_mask.float()
    )
    call = model.language_model.calls[0]
    torch.testing.assert_close(call["input_ids"], torch.tensor([[11, 12, 21, 22, 23]]))
    _assert_aligned_supervision(call, packed_loss_mask)


# ---------------------------------------------------------------------------
# Dense rows with caller-built metadata (verl)
# ---------------------------------------------------------------------------


def test_caller_metadata_and_thd_supervision_pass_through():
    model = _model()
    input_ids, attention_mask, _, _ = _dense_batch()
    caller_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 2, 5], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 2, 5], dtype=torch.int32),
        max_seqlen_q=3,
        max_seqlen_kv=3,
    )
    thd_labels = torch.arange(5).unsqueeze(0)
    thd_loss_mask = torch.ones(1, 5)

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        packed_seq_params=caller_params,
        labels=thd_labels,
        loss_mask=thd_loss_mask,
    )

    assert not isinstance(output, tuple)
    call = model.language_model.calls[0]
    assert call["packed_seq_params"] is caller_params
    assert call["labels"] is thd_labels
    assert call["loss_mask"] is thd_loss_mask
    torch.testing.assert_close(call["input_ids"], torch.tensor([[11, 12, 21, 22, 23]]))


def test_caller_metadata_must_describe_one_sequence_per_row():
    model = _model()
    input_ids, attention_mask, _, _ = _dense_batch()
    one_sequence = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 5], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 5], dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="each input row must be one sequence"):
        model(input_ids=input_ids, attention_mask=attention_mask, packed_seq_params=one_sequence)


# ---------------------------------------------------------------------------
# Already-packed THD stream (Bridge collate-time packing)
# ---------------------------------------------------------------------------


def _packed_batch():
    input_ids = torch.tensor([[11, 12, 21, 22, 23, 0]])
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)
    cu_seqlens_padded = torch.tensor([0, 2, 6], dtype=torch.int32)
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=4,
        max_seqlen_kv=4,
    )
    labels = torch.where(input_ids != 0, input_ids + LABEL_OFFSET, 0)
    loss_mask = (input_ids != 0).float()
    return input_ids, params, labels, loss_mask


def test_existing_thd_stream_keeps_its_boundaries():
    model = _model()
    input_ids, params, labels, loss_mask = _packed_batch()
    output = model(input_ids=input_ids, packed_seq_params=params, labels=labels, loss_mask=loss_mask)

    assert not isinstance(output, tuple)
    call = model.language_model.calls[0]
    assert call["packed_seq_params"] is params
    assert call["input_ids"] is input_ids
    assert call["labels"] is labels
    assert call["loss_mask"] is loss_mask
    torch.testing.assert_close(call["decoder_input"][:, 0, 0], input_ids[0].float())


def test_existing_thd_stream_selects_this_cp_rank_once():
    model = _model(cp=2, cp_rank=1)
    input_ids, params, labels, loss_mask = _packed_batch()
    cp_index = torch.tensor([1, 2, 5])
    with patch.object(glm53flash_bridge, "get_packed_seq_cp_partition_indices", return_value=cp_index) as indices:
        _, local_loss_mask = model(input_ids=input_ids, packed_seq_params=params, labels=labels, loss_mask=loss_mask)

    indices.assert_called_once()
    assert indices.call_args.kwargs["total_tokens"] == 6
    assert indices.call_args.kwargs["cp_rank"] == 1
    call = model.language_model.calls[0]
    assert call["packed_seq_params"] is params
    torch.testing.assert_close(call["input_ids"], torch.tensor([[12, 21, 0]]))
    _assert_aligned_supervision(call, local_loss_mask)


def test_packed_stream_requires_a_single_row():
    model = _model()
    input_ids, params, _, _ = _packed_batch()
    with pytest.raises(ValueError, match=r"\[1, total_tokens\]"):
        model(input_ids=input_ids.repeat(2, 1), packed_seq_params=params)


def test_sequence_parallel_scatters_after_packing():
    model = _model(tp=2, sequence_parallel=True)
    input_ids, attention_mask, _, _ = _dense_batch()
    with patch.object(
        glm53flash_bridge, "scatter_to_sequence_parallel_region", side_effect=lambda tensor, group: tensor[:3]
    ) as scatter:
        model(input_ids=input_ids, attention_mask=attention_mask)

    packed = scatter.call_args.args[0]
    assert packed.shape == (6, 1, HIDDEN)
    torch.testing.assert_close(packed[:, 0, 0], torch.tensor([11.0, 12.0, 21.0, 22.0, 23.0, 0.0]))
    assert model.language_model.calls[0]["decoder_input"].shape == (3, 1, HIDDEN)


# ---------------------------------------------------------------------------
# Provider: MTP depth
# ---------------------------------------------------------------------------


def _glm_config(num_nextn_predict_layers: int):
    config_module = pytest.importorskip("transformers.models.glm5_next.configuration_glm5_next")
    return config_module.Glm5NextConfig(
        text_config={
            "scoring_func": "sigmoid",
            "num_nextn_predict_layers": num_nextn_predict_layers,
            "linear_attn_config": {"num_heads": 64, "head_dim": 128, "short_conv_kernel_size": 4},
        }
    )


def test_zero_depth_source_stays_mtp_free_after_repeated_finalize():
    config = _glm_config(0)
    provider = GLM53FlashBridge().provider_bridge(SimpleNamespace(config=config))
    for _ in range(2):
        provider.finalize()
        assert provider.mtp_num_layers == 0
        assert provider.mtp_hybrid_override_pattern is None
        assert Symbols.MTP_SEPARATOR not in provider.hybrid_layer_pattern

    bridge = GLM53FlashBridge()
    bridge.hf_config = config
    assert not any(".mtp." in mapping.megatron_param for mapping in bridge.mapping_registry().mappings)


@pytest.mark.parametrize("num_mtp", [1, 2])
def test_positive_depth_source_keeps_its_mtp_depth(num_mtp):
    config = _glm_config(num_mtp)
    provider = GLM53FlashBridge().provider_bridge(SimpleNamespace(config=config))
    assert provider.mtp_use_repeated_layer
    assert provider.mtp_loss_scaling_factor == 0.1
    for _ in range(2):
        provider.finalize()
        assert provider.mtp_num_layers == num_mtp
        assert provider.hybrid_layer_pattern.count(Symbols.MTP_SEPARATOR) == num_mtp
        assert provider.hybrid_layer_pattern.endswith(Symbols.MTP_SEPARATOR + "DE")

    bridge = GLM53FlashBridge()
    bridge.hf_config = config
    mtp_params = {
        mapping.megatron_param for mapping in bridge.mapping_registry().mappings if ".mtp." in mapping.megatron_param
    }
    assert f"language_model.mtp.layers.{num_mtp - 1}.e_proj.weight" in mtp_params
