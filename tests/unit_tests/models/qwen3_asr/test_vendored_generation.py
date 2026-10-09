"""Numerical checks for the vendored ASR classes, independent of AutoModel registration."""

import copy
import logging

import pytest
import torch

from megatron.bridge.models.qwen3_asr.hf_qwen3_asr import Qwen3ASRConfig, Qwen3ASRForConditionalGeneration
from megatron.bridge.models.qwen3_asr.hf_qwen3_asr.modeling_qwen3_asr import Qwen3ASRThinkerTextRotaryEmbedding


pytestmark = pytest.mark.unit
logger = logging.getLogger(__name__)


def tiny_config(*, rope_scaling=None, pad_token_id=None):
    text = dict(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rope_theta=10000.0,
        rope_scaling=rope_scaling,
    )
    thinker = dict(
        audio_config=dict(
            num_mel_bins=128,
            encoder_layers=1,
            encoder_attention_heads=2,
            encoder_ffn_dim=64,
            d_model=32,
            output_dim=64,
            max_source_positions=1500,
            n_window=100,
            n_window_infer=400,
            conv_chunksize=500,
            downsample_hidden_size=8,
        ),
        text_config=text,
        audio_token_id=62,
        audio_start_token_id=63,
    )
    if pad_token_id is not None:
        text["pad_token_id"] = pad_token_id
        thinker["pad_token_id"] = pad_token_id
    return Qwen3ASRConfig(thinker_config=thinker)


def make_model(*, attention="eager", rope_scaling=None, pad_token_id=None):
    torch.manual_seed(42)
    config = tiny_config(rope_scaling=rope_scaling, pad_token_id=pad_token_id)
    config._attn_implementation = attention
    return Qwen3ASRForConditionalGeneration(config).float().eval()


def greedy_reference(thinker, ids, mask, steps):
    for _ in range(steps):
        logits = thinker(input_ids=ids, attention_mask=mask, use_cache=False).logits
        assert torch.isfinite(logits).all()
        ids = torch.cat((ids, logits[:, -1].argmax(-1, keepdim=True)), dim=-1)
        mask = torch.cat((mask, torch.ones_like(mask[:, :1])), dim=-1)
    return ids


@pytest.mark.parametrize("padding", [None, 0])
@torch.no_grad()
def test_construction_forward_and_reload(tmp_path, padding):
    model = make_model(pad_token_id=padding)
    assert model.thinker.model.embed_tokens.padding_idx == padding
    assert model.thinker.lm_head.weight is not model.thinker.model.embed_tokens.weight
    ids = torch.tensor([[3, 7, 11, 5]])
    logits = model.thinker(input_ids=ids, use_cache=False).logits
    assert logits.shape == (1, 4, 64)
    assert torch.isfinite(logits).all()
    assert logits.std() > 0.01
    assert not torch.equal(logits[:, 0], logits[:, -1])
    model.save_pretrained(tmp_path)
    loaded = Qwen3ASRForConditionalGeneration.from_pretrained(tmp_path, attn_implementation="eager").eval()
    torch.testing.assert_close(loaded.thinker(input_ids=ids, use_cache=False).logits, logits, rtol=0, atol=0)
    logger.info("logits shape=%s std=%g finite=%s reload exact", tuple(logits.shape), logits.std(), True)


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("padded", [False, True])
@torch.no_grad()
def test_greedy_generation_matches_full_prefix(attention, padded):
    model = make_model(attention=attention, rope_scaling={"rope_type": "default", "mrope_section": [4, 2, 2]})
    ids = torch.tensor([[0, 0, 3, 7, 11], [2, 4, 6, 8, 10]]) if padded else torch.tensor([[3, 7, 11]])
    mask = (ids != 0).long()
    expected = greedy_reference(model.thinker, ids.clone(), mask.clone(), 3)
    arguments = dict(attention_mask=mask, max_new_tokens=3, do_sample=False, eos_token_id=None, pad_token_id=0)
    cached = model.thinker.generate(ids, use_cache=True, **arguments)
    uncached = model.thinker.generate(ids, use_cache=False, **arguments)
    repeated = model.thinker.generate(ids, use_cache=True, **arguments)
    top_level = model.generate(input_ids=ids, use_cache=True, **arguments).sequences
    assert cached.shape == (ids.shape[0], ids.shape[1] + 3)
    assert ((cached >= 0) & (cached < 64)).all()
    for result in (cached, uncached, repeated, top_level):
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
    if padded:
        for index in range(ids.shape[0]):
            unpadded = ids[index][mask[index].bool()].unsqueeze(0)
            single = model.thinker.generate(
                unpadded,
                attention_mask=torch.ones_like(unpadded),
                max_new_tokens=3,
                do_sample=False,
                eos_token_id=None,
                pad_token_id=0,
            )
            torch.testing.assert_close(single[:, -3:], cached[index : index + 1, -3:], rtol=0, atol=0)
    logger.info(
        "%s padded=%s generation=%s matches full-prefix, uncached and repeat", attention, padded, cached.tolist()
    )


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@torch.no_grad()
def test_cached_logits_and_causal_mask(attention):
    model = make_model(attention=attention).thinker
    ids = torch.tensor([[0, 3, 7, 11, 5]])
    mask = (ids != 0).long()
    full = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    prefix = model(input_ids=ids[:, :3], attention_mask=mask[:, :3], use_cache=True)
    for stop in (4, 5):
        step = model(
            input_ids=ids[:, stop - 1 : stop],
            attention_mask=mask[:, :stop],
            past_key_values=prefix.past_key_values,
            use_cache=True,
        )
        torch.testing.assert_close(step.logits[:, -1], full[:, stop - 1], rtol=1e-4, atol=1e-5)
        prefix = step
    changed = ids.clone()
    changed[:, -1] = 17
    altered = model(input_ids=changed, attention_mask=mask, use_cache=False).logits
    real_prefix = mask[:, :-1].bool()
    torch.testing.assert_close(altered[:, :-1][real_prefix], full[:, :-1][real_prefix], rtol=1e-4, atol=1e-5)
    positions = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)
    embedded = model(
        inputs_embeds=model.get_input_embeddings()(ids), attention_mask=mask, position_ids=positions, use_cache=False
    ).logits
    torch.testing.assert_close(embedded, full, rtol=1e-4, atol=1e-5)
    logger.info(
        "%s cached/full-prefix max final error=%g; causal and embeddings checks pass",
        attention,
        (step.logits[:, -1] - full[:, -1]).abs().max(),
    )


@pytest.mark.parametrize("kind", ["implicit", "default", "linear"])
@torch.no_grad()
def test_rotary_matches_formula(kind):
    scaling = None if kind == "implicit" else {"rope_type": kind, "mrope_section": [4, 2, 2]}
    if kind == "linear":
        scaling["factor"] = 2.0
    config = tiny_config(rope_scaling=scaling).get_text_config()
    rotary = Qwen3ASRThinkerTextRotaryEmbedding(config)
    x = torch.zeros(2, 4, 64)
    position_ids = torch.tensor([[0, 1, 3, 9], [2, 4, 6, 10]])
    cosine, sine = rotary(x, position_ids)
    inv = 10000.0 ** (-torch.arange(0, 16, 2).float() / 16)
    if kind == "linear":
        inv /= 2
    phase = position_ids[..., None] * inv
    phase = torch.cat((phase, phase), -1)
    torch.testing.assert_close(cosine, phase.cos())
    torch.testing.assert_close(sine, phase.sin())
    # Construction/post_init must preserve the same frequencies, including nondefault scaling.
    model = make_model(rope_scaling=copy.deepcopy(scaling))
    torch.testing.assert_close(model.thinker.model.rotary_emb.inv_freq, inv)


@torch.no_grad()
def test_audio_cached_generation_matches_full_prefix():
    model = make_model().thinker
    ids = torch.tensor([[3, 62, 62, 62, 62, 7]])
    mask = torch.ones_like(ids)
    features = torch.randn(1, 128, 32)
    feature_mask = torch.ones(1, 32, dtype=torch.long)
    kwargs = dict(input_features=features, feature_attention_mask=feature_mask)
    expected = ids.clone()
    expected_mask = mask.clone()
    for _ in range(3):
        out = model(input_ids=expected, attention_mask=expected_mask, use_cache=False, **kwargs)
        assert torch.isfinite(out.logits).all()
        expected = torch.cat((expected, out.logits[:, -1].argmax(-1, keepdim=True)), -1)
        expected_mask = torch.cat((expected_mask, torch.ones_like(expected_mask[:, :1])), -1)
    for use_cache in (True, False):
        result = model.generate(
            ids,
            attention_mask=mask,
            max_new_tokens=3,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=0,
            use_cache=use_cache,
            **kwargs,
        )
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
    logger.info("real audio + text generation=%s matches full-prefix", expected.tolist())


@torch.no_grad()
def test_text_logits_match_native_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    model = make_model().thinker
    config_dict = model.config.text_config.to_dict()
    config_dict.pop("model_type")
    reference_config = Qwen3Config(**config_dict)
    reference_config._attn_implementation = "eager"
    reference = Qwen3ForCausalLM(reference_config).eval()
    text_weights = {k: v for k, v in model.state_dict().items() if not k.startswith("audio_tower.")}
    reference.load_state_dict(text_weights, strict=True)
    ids = torch.tensor([[0, 0, 3, 7, 11], [2, 4, 6, 8, 10]])
    mask = (ids != 0).long()
    positions = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)
    actual = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False).logits
    expected = reference(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False).logits
    torch.testing.assert_close(actual[mask.bool()], expected[mask.bool()], rtol=1e-4, atol=1e-5)
    logger.info(
        "native Qwen3 shared-weight reference max logit error=%g",
        (actual[mask.bool()] - expected[mask.bool()]).abs().max(),
    )


@torch.no_grad()
def test_cached_input_embeddings_and_explicit_positions():
    model = make_model().thinker
    ids = torch.tensor([[0, 3, 7, 11]])
    mask = (ids != 0).long()
    full = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    prefix = model(input_ids=ids[:, :3], attention_mask=mask[:, :3], use_cache=True)
    step = model(
        inputs_embeds=model.get_input_embeddings()(ids[:, -1:]),
        attention_mask=mask,
        past_key_values=prefix.past_key_values,
        use_cache=True,
    )
    torch.testing.assert_close(step.logits[:, -1], full[:, -1], rtol=1e-4, atol=1e-5)
    positions = torch.tensor([[0, 2, 4, 9]])
    prepared = model.prepare_inputs_for_generation(
        ids,
        position_ids=positions,
        next_sequence_length=1,
        past_key_values=prefix.past_key_values,
        attention_mask=mask,
        is_first_iteration=False,
    )
    torch.testing.assert_close(prepared["position_ids"], positions[:, -1:])


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("nested_object", [False, True])
@torch.no_grad()
def test_weight_sharing_survives_reload(tmp_path, tied, nested_object):
    if nested_object:
        config = tiny_config()
        # Config objects may be edited between recipe creation and model construction.
        config.thinker_config.text_config.tie_word_embeddings = tied
        config.thinker_config.tie_word_embeddings = not tied
    else:
        config_dict = tiny_config().to_dict()
        config_dict["thinker_config"]["text_config"]["tie_word_embeddings"] = tied
        config_dict["thinker_config"]["tie_word_embeddings"] = not tied
        config = Qwen3ASRConfig(**config_dict)
    config._attn_implementation = "eager"
    model = Qwen3ASRForConditionalGeneration(config).eval()
    assert (model.thinker.lm_head.weight is model.thinker.model.embed_tokens.weight) == tied
    model.save_pretrained(tmp_path)
    loaded = Qwen3ASRForConditionalGeneration.from_pretrained(tmp_path, attn_implementation="eager").eval()
    assert (loaded.thinker.lm_head.weight is loaded.thinker.model.embed_tokens.weight) == tied
    ids = torch.tensor([[3, 7, 11]])
    torch.testing.assert_close(loaded.thinker(input_ids=ids).logits, model.thinker(input_ids=ids).logits)


@torch.no_grad()
def test_cached_positions_ignore_previous_request():
    model = make_model().thinker
    old_ids = torch.tensor([[0, 0, 3, 5], [0, 2, 4, 6]])
    model(input_ids=old_ids, attention_mask=(old_ids != 0).long(), use_cache=False)
    ids = torch.tensor([[3, 7, 11, 5]])
    mask = torch.ones_like(ids)
    prefix = model(
        input_ids=ids[:, :3],
        attention_mask=mask[:, :3],
        position_ids=torch.arange(3).unsqueeze(0),
        use_cache=True,
    )
    step = model(
        input_ids=ids[:, -1:],
        attention_mask=mask,
        past_key_values=prefix.past_key_values,
        use_cache=True,
    )
    expected = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, -1:]
    assert step.logits.shape == (1, 1, 64)
    torch.testing.assert_close(step.logits, expected, rtol=1e-4, atol=1e-5)


@torch.no_grad()
def test_single_real_token_padded_cached_logits():
    model = make_model().thinker
    ids = torch.tensor([[0, 0, 3, 7]])
    mask = (ids != 0).long()
    prefix = model(input_ids=ids[:, :3], attention_mask=mask[:, :3], use_cache=True)
    torch.testing.assert_close(prefix.rope_deltas, torch.tensor([[-2]]))
    step = model(input_ids=ids[:, -1:], attention_mask=mask, past_key_values=prefix.past_key_values, use_cache=True)
    full = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, -1:]
    torch.testing.assert_close(step.logits, full, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_audio_checkpoint_reload(tmp_path, dtype):
    model = make_model().to(dtype=dtype)
    ids = torch.tensor([[3, 62, 62, 62, 62, 7]])
    mask = torch.ones_like(ids)
    features = torch.randn(1, 128, 32).to(dtype=dtype)
    feature_mask = torch.ones(1, 32, dtype=torch.long)
    kwargs = dict(input_features=features, feature_attention_mask=feature_mask)
    expected = model.thinker(input_ids=ids, attention_mask=mask, use_cache=False, **kwargs).logits
    model.save_pretrained(tmp_path)
    loaded = Qwen3ASRForConditionalGeneration.from_pretrained(
        tmp_path, dtype=dtype, attn_implementation="eager"
    ).eval()
    # Match the original whole-model cast, including reconstructed nonpersistent buffers.
    loaded = loaded.to(dtype=dtype)
    torch.testing.assert_close(
        loaded.thinker.get_audio_features(**kwargs), model.thinker.get_audio_features(**kwargs), rtol=0, atol=0
    )
    original_buffer = model.thinker.audio_tower.positional_embedding.positional_embedding
    loaded_buffer = loaded.thinker.audio_tower.positional_embedding.positional_embedding
    torch.testing.assert_close(loaded_buffer, original_buffer, rtol=0, atol=0)
    actual = loaded.thinker(input_ids=ids, attention_mask=mask, use_cache=False, **kwargs).logits
    assert torch.isfinite(actual).all() and actual.float().std() > 0.01
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    options = dict(attention_mask=mask, max_new_tokens=3, do_sample=False, eos_token_id=None, pad_token_id=0)
    reference = model.generate(input_ids=ids, use_cache=False, **options, **kwargs).sequences
    generated = loaded.generate(input_ids=ids, use_cache=True, **options, **kwargs).sequences
    torch.testing.assert_close(generated, reference, rtol=0, atol=0)
    logger.info("audio reload %s buffer/logits exact; generated tokens=%s", dtype, generated.tolist())
