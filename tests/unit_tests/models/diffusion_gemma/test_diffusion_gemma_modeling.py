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

"""CPU contracts for DiffusionGemma model composition without process groups."""

from types import SimpleNamespace

import pytest
import torch
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.transformer.enums import AttnMaskType
from torch import nn
from torch.nn import functional as F

from megatron.bridge.models.diffusion_gemma.modeling_diffusion_gemma import (
    DiffusionGemmaEncoderOutput,
    DiffusionGemmaLanguageModelEmbedding,
    DiffusionGemmaModel,
    DiffusionGemmaRMSNorm,
    DiffusionGemmaSelfConditioning,
    _attention_qkv,
    _replace_attention_args,
)
from megatron.bridge.models.gemma.gemma3_provider import Gemma3LanguageModelEmbedding
from megatron.bridge.models.gemma_vl.modeling_gemma4_vl import Gemma4VLModel


pytestmark = pytest.mark.unit


class _Core(nn.Module):
    def __init__(self, *, window=None, image_mask=None):
        super().__init__()
        self.calls = []
        if window is not None:
            self._image_attention_window = window
        if image_mask is not None:
            self._image_bidirectional_attention = image_mask

    def forward(self, query, key, value, attention_mask=None, attn_mask_type=None):
        self.calls.append(
            {
                "query": query,
                "key": key,
                "value": value,
                "attention_mask": attention_mask,
                "attn_mask_type": attn_mask_type,
                "image_mask": getattr(self, "_image_bidirectional_attention", None),
            }
        )
        return query


class _Decoder(nn.Module):
    def __init__(self, cores=()):
        super().__init__()
        self.layers = [SimpleNamespace(self_attention=SimpleNamespace(core_attention=core)) for core in cores]
        self.calls = []

    def forward(self, hidden_states, attention_mask, rotary_pos_emb):
        self.calls.append((hidden_states, attention_mask, rotary_pos_emb))
        return hidden_states + 1


class _LanguageModel(nn.Module):
    def __init__(self, *, cores=(), weight=None):
        super().__init__()
        self.post_process = True
        self.decoder = _Decoder(cores)
        self.weight = nn.Parameter(weight if weight is not None else torch.eye(3))
        self.rotary_calls = []
        self.output_calls = []

    def rotary_pos_emb(self, length, offset):
        self.rotary_calls.append((length, offset))
        return torch.tensor([length, offset])

    def shared_embedding_or_output_weight(self):
        return self.weight

    def output_layer(self, hidden, *, weight, runtime_gather_output):
        self.output_calls.append((hidden.shape, runtime_gather_output))
        return torch.matmul(hidden, weight.T), None


def _bare_model(*, language_model=None, **config):
    model = DiffusionGemmaModel.__new__(DiffusionGemmaModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(sequence_parallel=False, hidden_size=4, **config)
    model.language_model = language_model or _LanguageModel()
    model._encoder_padding_mask = None
    return model


def test_embedding_scales_outputs_and_freezes_only_padding_lookups(monkeypatch):
    def base_forward(self, input_ids, position_ids, tokentype_ids=None):
        return self.table[input_ids].transpose(0, 1)

    monkeypatch.setattr(LanguageModelEmbedding, "forward", base_forward)
    embedding = DiffusionGemmaLanguageModelEmbedding.__new__(DiffusionGemmaLanguageModelEmbedding)
    nn.Module.__init__(embedding)
    embedding.config = SimpleNamespace(hidden_size=4)
    embedding.table = nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4))
    embedding.padding_idx = 0
    input_ids = torch.tensor([[0, 1, 1], [2, 0, 1]])

    output = embedding(input_ids, position_ids=None)
    assert torch.equal(output, embedding.table[input_ids].transpose(0, 1) * 2)
    output.sum().backward()
    assert torch.equal(embedding.table.grad[0], torch.zeros(4))
    assert torch.equal(embedding.table.grad[1], torch.full((4,), 6.0))
    assert torch.equal(embedding.table.grad[2], torch.full((4,), 2.0))

    embedding.table.grad = None
    embedding.padding_idx = None
    embedding(input_ids, position_ids=None).sum().backward()
    assert torch.equal(embedding.table.grad[0], torch.full((4,), 4.0))


def test_rmsnorm_runs_in_fp32_and_scale_is_optional():
    hidden = torch.tensor([[3.0, 4.0], [1.0, -1.0]], dtype=torch.bfloat16)
    scaled = DiffusionGemmaRMSNorm(2, eps=0.0)
    with torch.no_grad():
        scaled.weight.copy_(torch.tensor([2.0, 0.5]))
    unscaled = DiffusionGemmaRMSNorm(2, eps=0.0, with_scale=False)

    expected = hidden.float() * torch.rsqrt(hidden.float().pow(2).mean(-1, keepdim=True))
    assert scaled(hidden).dtype == torch.bfloat16
    assert torch.allclose(scaled(hidden).float(), (expected * torch.tensor([2.0, 0.5])).bfloat16().float())
    assert torch.allclose(unscaled(hidden).float(), expected.bfloat16().float())
    assert not hasattr(unscaled, "weight")


def test_self_conditioning_adds_gated_signal_before_unscaled_post_norm():
    torch.manual_seed(0)
    block = DiffusionGemmaSelfConditioning(hidden_size=4, intermediate_size=6, eps=1e-6)
    inputs = torch.randn(2, 3, 4, requires_grad=True)
    signal = torch.randn(2, 3, 4, requires_grad=True)

    output = block(inputs, signal)
    normed = block.pre_norm(signal)
    expected = block.post_norm(
        inputs + block.down_proj(F.gelu(block.gate_proj(normed), approximate="tanh") * block.up_proj(normed))
    )
    assert torch.allclose(output, expected)
    assert torch.allclose(output.pow(2).mean(-1), torch.ones(2, 3), atol=1e-5)
    output.sum().backward()
    assert inputs.grad is not None and signal.grad is not None


def test_init_converts_embedding_forces_eager_vision_and_builds_typed_self_conditioning(monkeypatch):
    embedding = Gemma3LanguageModelEmbedding.__new__(Gemma3LanguageModelEmbedding)
    nn.Module.__init__(embedding)
    vision_child = nn.Linear(2, 2)
    vision_child.config = SimpleNamespace(_attn_implementation="sdpa")

    def parent_init(self, *, config, pre_process, post_process, vp_stage):
        nn.Module.__init__(self)
        self.config = config
        self.language_model = SimpleNamespace(embedding=embedding)
        self.vision_tower = nn.Sequential(vision_child, nn.ReLU())

    monkeypatch.setattr(Gemma4VLModel, "__init__", parent_init)
    config = SimpleNamespace(
        text_config=SimpleNamespace(pad_token_id=7, hidden_size=4, intermediate_size=6, rms_norm_eps=1e-6),
        params_dtype=torch.bfloat16,
    )
    model = DiffusionGemmaModel(config)

    assert type(embedding) is DiffusionGemmaLanguageModelEmbedding
    assert embedding.padding_idx == 7
    assert vision_child.config._attn_implementation == "eager"
    assert model.self_conditioning.gate_proj.weight.dtype == torch.bfloat16
    assert model._encoder_padding_mask is None

    model = DiffusionGemmaModel(config, pre_process=False)
    assert not hasattr(model, "self_conditioning")


def test_output_head_is_restored_and_tp_group_is_optional():
    model = _bare_model()
    with pytest.raises(RuntimeError, match="boom"):
        with model._without_output_head():
            assert not model.language_model.post_process
            raise RuntimeError("boom")
    assert model.language_model.post_process
    assert model._tp_group() is None
    model.config._pg_collection = SimpleNamespace(tp="tp-group")
    assert model._tp_group() == "tp-group"


def test_encoder_padding_blocks_pad_keys_but_keeps_pad_queries_finite(monkeypatch):
    base_mask = torch.zeros(1, 1, 3, 3, dtype=torch.bool)
    monkeypatch.setattr(
        Gemma4VLModel, "_compute_attention_mask", lambda self, input_ids, mm_token_type_ids=None: base_mask
    )
    model = _bare_model()
    input_ids = torch.ones(1, 3, dtype=torch.long)

    assert model._compute_attention_mask(input_ids) is base_mask
    model._encoder_padding_mask = torch.ones_like(input_ids)
    assert model._compute_attention_mask(input_ids) is base_mask
    model._encoder_padding_mask = torch.ones(1, 2)
    with pytest.raises(ValueError, match="attention_mask shape"):
        model._compute_attention_mask(input_ids)

    model._encoder_padding_mask = torch.tensor([[0, 1, 1]])
    mask = model._compute_attention_mask(input_ids)[0, 0]
    assert mask.tolist() == [[False, False, False], [True, False, False], [True, False, False]]

    monkeypatch.setattr(Gemma4VLModel, "_compute_attention_mask", lambda self, input_ids, mm_token_type_ids=None: None)
    assert model._compute_attention_mask(input_ids) is None


def test_encode_captures_each_layer_kv_and_removes_temporary_state(monkeypatch):
    cores = [_Core(), _Core()]
    model = _bare_model(language_model=_LanguageModel(cores=cores))
    mask = torch.tensor([[1, 1, 0]])
    seen = {}

    def parent_forward(self, **kwargs):
        seen["post_process"] = self.language_model.post_process
        seen["padding"] = self._encoder_padding_mask
        for index, core in enumerate(cores):
            q = torch.full((3, 1, 1, 2), float(index))
            k = q + 10
            v = q + 20
            if index:
                core(query=q, key=k, value=v)
            else:
                core(q, k, v)
        return torch.arange(12, dtype=torch.float32).reshape(3, 1, 4)

    monkeypatch.setattr(Gemma4VLModel, "forward", parent_forward)
    output = model.encode(input_ids=torch.ones(1, 3), attention_mask=mask, return_key_values=True)

    assert seen == {"post_process": False, "padding": mask}
    assert output.last_hidden_state.shape == (1, 3, 4)
    assert [value[0][0, 0, 0, 0].item() for value in output.key_values] == [10.0, 11.0]
    assert [value[1][0, 0, 0, 0].item() for value in output.key_values] == [20.0, 21.0]
    assert model.language_model.post_process
    assert model._encoder_padding_mask is None
    assert all(not core._forward_pre_hooks for core in cores)


def test_encode_rejects_lm_targets_and_missing_layer_capture(monkeypatch):
    model = _bare_model(language_model=_LanguageModel(cores=[_Core(), _Core()]))
    with pytest.raises(ValueError, match="does not accept labels"):
        model.encode(labels=torch.ones(1))

    monkeypatch.setattr(Gemma4VLModel, "forward", lambda self, **kwargs: torch.zeros(2, 1, 4))
    with pytest.raises(RuntimeError, match="Failed to capture"):
        model.encode(return_key_values=True)
    assert all(
        not layer.self_attention.core_attention._forward_pre_hooks for layer in model.language_model.decoder.layers
    )


def test_decoder_embeddings_project_probabilities_and_respect_selection_mask():
    model = _bare_model()
    model.config.hidden_size = 4
    word_embeddings = nn.Embedding(3, 4)
    with torch.no_grad():
        word_embeddings.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4))
    calls = []

    class _Embedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.word_embeddings = word_embeddings

        def forward(self, input_ids, position_ids):
            assert position_ids is None
            return torch.ones(input_ids.shape[1], input_ids.shape[0], 4)

    class _SelfConditioning(nn.Module):
        def forward(self, inputs, signal):
            calls.append(signal.clone())
            return inputs + signal

    model.language_model.embedding = _Embedding()
    model.self_conditioning = _SelfConditioning()
    decoder_ids = torch.zeros(2, 2, dtype=torch.long)

    output = model._decoder_embeddings(decoder_ids, None, None)
    assert output.shape == (2, 2, 4)
    assert torch.equal(calls[-1], torch.zeros(2, 2, 4))

    logits = torch.tensor([[[0.0, 0.0, 0.0, 99.0], [10.0, 0.0, 0.0, 0.0]], [[0.0, 10.0, 0.0, 0.0]] * 2])
    model._decoder_embeddings(decoder_ids, logits, torch.tensor([True, False]))
    expected = torch.matmul(logits[..., :3].softmax(-1), word_embeddings.weight) * 2
    expected[1] = 0
    assert torch.allclose(calls[-1], expected)


def test_decoder_attention_prepends_encoder_kv_with_window_and_restores_masks():
    local = _Core(window=3, image_mask=True)
    global_core = _Core()
    model = _bare_model(language_model=_LanguageModel(cores=[local, global_core]))
    encoder_key = torch.arange(16, dtype=torch.float32).reshape(4, 2, 1, 2)
    key_values = [(encoder_key, -encoder_key), (encoder_key + 100, -encoder_key - 100)]
    encoder_mask = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 0]])
    q = torch.zeros(2, 2, 1, 2)
    k = torch.ones(2, 2, 1, 2)
    v = k + 1

    with model._decoder_attention(key_values, encoder_mask, canvas_length=2):
        local(q, k, v)
        global_core(query=q, key=k, value=v)

    local_call = local.calls[-1]
    assert local_call["image_mask"] is False
    assert torch.equal(local_call["key"][:2], encoder_key[-2:])
    assert local_call["attention_mask"].shape == (2, 1, 2, 4)
    assert local_call["attention_mask"][1, 0, 0].tolist() == [False, True, False, False]
    assert local_call["attn_mask_type"] == AttnMaskType.arbitrary
    global_call = global_core.calls[-1]
    assert torch.equal(global_call["key"][:4], encoder_key + 100)
    assert global_call["attention_mask"][0, 0, 0].tolist() == [True, False, False, False, False, False]
    assert local._image_bidirectional_attention is True
    assert not local._forward_pre_hooks and not global_core._forward_pre_hooks


def test_decoder_attention_defaults_missing_prefix_mask_to_visible():
    core = _Core()
    model = _bare_model(language_model=_LanguageModel(cores=[core]))
    key = torch.ones(2, 1, 1, 2)
    with model._decoder_attention([(key, key)], None, canvas_length=1):
        core(torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2), None, AttnMaskType.causal)
    call = core.calls[-1]
    assert not call["attention_mask"].any()
    assert call["attn_mask_type"] == AttnMaskType.arbitrary


def test_decode_uses_encoder_offset_and_returns_batch_first_float_logits():
    language_model = _LanguageModel(weight=torch.eye(3))
    model = _bare_model(language_model=language_model)
    model._decoder_embeddings = lambda ids, logits, mask: torch.ones(ids.shape[1], ids.shape[0], 3)
    logits = model.decode(torch.zeros(2, 4, dtype=torch.long), [], encoder_length=7)
    assert language_model.rotary_calls == [(4, 7)]
    assert logits.shape == (2, 4, 3)
    assert logits.dtype == torch.float32
    assert torch.equal(logits, torch.full((2, 4, 3), 2.0))


def test_forward_projects_only_target_positions_for_encoder_loss():
    vocab = 3
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [5.0, 5.0]])
    language_model = _LanguageModel(weight=weight)
    model = _bare_model(
        language_model=language_model,
        tensor_model_parallel_size=1,
        text_config=SimpleNamespace(vocab_size=vocab),
    )
    prompt = torch.tensor([[10, 11, 12]])
    target = torch.tensor([[1, 2]])
    target_mask = torch.tensor([[1, 0]])
    prompt_mask = torch.tensor([[0, 1, 1]])
    mm = torch.tensor([[0, 1, 0]])
    clean_hidden = torch.arange(10, dtype=torch.float32).reshape(1, 5, 2)
    encode_calls = []
    decode_calls = []

    def encode(**kwargs):
        encode_calls.append(kwargs)
        if kwargs.get("return_key_values"):
            return DiffusionGemmaEncoderOutput(torch.zeros(1, 3, 2), key_values=["kv"])
        return DiffusionGemmaEncoderOutput(clean_hidden)

    def decode(*args, **kwargs):
        decode_calls.append((args, kwargs))
        return torch.zeros(1, 2, vocab)

    model.encode = encode
    model.decode = decode
    output = model(
        input_ids=prompt,
        decoder_input_ids=torch.zeros(1, 2, dtype=torch.long),
        mm_token_type_ids=mm,
        encoder_attention_mask=prompt_mask,
        encoder_target_ids=target,
        encoder_target_mask=target_mask,
    )

    assert torch.equal(encode_calls[0]["position_ids"], torch.tensor([[0, 1, 2]]))
    assert decode_calls[0][1]["encoder_length"] == 3
    assert torch.equal(encode_calls[1]["input_ids"], torch.tensor([[10, 11, 12, 1, 2]]))
    assert torch.equal(encode_calls[1]["attention_mask"], torch.tensor([[0, 1, 1, 1, 0]]))
    assert torch.equal(encode_calls[1]["mm_token_type_ids"], torch.tensor([[0, 1, 0, 0, 0]]))
    selected = clean_hidden[:, 2:4]
    logits = torch.matmul(selected, weight.T)[..., :vocab]
    expected = F.cross_entropy(logits.reshape(-1, vocab), torch.tensor([1, -100]), ignore_index=-100)
    assert torch.allclose(output.encoder_loss, expected)
    assert language_model.output_calls[-1][0] == (2, 1, 2)


def test_forward_requires_encoder_target_mask():
    model = _bare_model()
    model.encode = lambda **kwargs: DiffusionGemmaEncoderOutput(torch.zeros(1, 1, 1), key_values=[])
    model.decode = lambda *args, **kwargs: torch.zeros(1, 1, 1)
    with pytest.raises(ValueError, match="encoder_target_mask is required"):
        model(torch.ones(1, 1), torch.ones(1, 1), encoder_target_ids=torch.ones(1, 1))


def test_freeze_delegates_parent_modules_and_optionally_freezes_self_conditioning(monkeypatch):
    calls = []
    monkeypatch.setattr(Gemma4VLModel, "freeze", lambda self, **kwargs: calls.append(kwargs))
    model = _bare_model()
    model.self_conditioning = nn.Linear(2, 2)

    model.freeze(True, False, True, freeze_self_conditioning=False)
    assert all(param.requires_grad for param in model.self_conditioning.parameters())
    model.freeze(False, True, False, freeze_audio_model=True, freeze_self_conditioning=True)

    assert calls[-1] == {
        "freeze_language_model": False,
        "freeze_vision_model": True,
        "freeze_vision_projection": False,
        "freeze_audio_model": True,
        "freeze_audio_projection": False,
    }
    assert not any(param.requires_grad for param in model.self_conditioning.parameters())


def test_attention_argument_helpers_support_positional_and_keyword_calls():
    q, k, v, mask = (torch.tensor(i) for i in range(4))
    assert _attention_qkv((q, k, v), {}) == (q, k, v)
    assert _attention_qkv((), {"query": q, "key": k, "value": v}) == (q, k, v)

    args, kwargs = _replace_attention_args((1, 2, 3, 4, 5, 6), {"x": 1}, q, k, v, mask)
    assert args[:5] == (q, k, v, mask, AttnMaskType.arbitrary)
    assert args[5] == 6 and kwargs == {"x": 1}
    args, kwargs = _replace_attention_args((), {}, q, k, v, mask)
    assert args == ()
    assert kwargs == {
        "query": q,
        "key": k,
        "value": v,
        "attention_mask": mask,
        "attn_mask_type": AttnMaskType.arbitrary,
    }
