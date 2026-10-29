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

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.minimax_m3.modeling_minimax_m3_vl import (
    MiniMaxM3LightningIndexerState,
    MiniMaxM3ProjectorMLP,
    MiniMaxM3VisionModel,
    MiniMaxM3VLModel,
    _apply_vision_rope,
)


def test_lightning_indexer_state_has_exact_frozen_weight_shapes():
    config = SimpleNamespace(
        hidden_size=16,
        index_n_heads=2,
        index_head_dim=4,
        params_dtype=torch.bfloat16,
    )

    indexer = MiniMaxM3LightningIndexerState(config)

    assert indexer.q_proj.weight.shape == (8, 16)
    assert indexer.k_proj.weight.shape == (4, 16)
    assert indexer.q_norm.weight.shape == (4,)
    assert indexer.k_norm.weight.shape == (4,)
    assert all(parameter.dtype == torch.bfloat16 for parameter in indexer.parameters())
    assert all(not parameter.requires_grad for parameter in indexer.parameters())


def test_wrapper_keeps_only_indexers_owned_by_local_pipeline_layers():
    class LocalLayer(torch.nn.Module):
        def __init__(self, layer_number: int) -> None:
            super().__init__()
            self.layer_number = layer_number

    class LanguageModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = torch.nn.Module()
            self.decoder.layers = torch.nn.ModuleList([LocalLayer(1), LocalLayer(4)])
            self.shared_embedding_or_output_weight = None

    config = SimpleNamespace(
        hidden_size=16,
        index_n_heads=2,
        index_head_dim=4,
        params_dtype=torch.bfloat16,
        lightning_indexer_layers=[1, 3],
        provide_language_model=lambda **_kwargs: LanguageModel(),
        share_embeddings_and_output_weights=False,
    )

    model = MiniMaxM3VLModel(config, pre_process=False)

    assert list(model.lightning_indexers) == ["3"]
    assert set(model.state_dict()) == {
        "lightning_indexers.3.q_proj.weight",
        "lightning_indexers.3.k_proj.weight",
        "lightning_indexers.3.q_norm.weight",
        "lightning_indexers.3.k_norm.weight",
    }


def test_lightning_indexer_distributed_checkpoint_round_trip(tmp_path):
    from megatron.core import parallel_state
    from megatron.core.dist_checkpointing import load, save

    class LocalLayer(torch.nn.Module):
        def __init__(self, layer_number: int) -> None:
            super().__init__()
            self.layer_number = layer_number

    class LanguageModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = torch.nn.Module()
            self.decoder.layers = torch.nn.ModuleList([LocalLayer(2)])
            self.shared_embedding_or_output_weight = None

    def make_model() -> MiniMaxM3VLModel:
        config = SimpleNamespace(
            hidden_size=16,
            index_n_heads=2,
            index_head_dim=4,
            params_dtype=torch.bfloat16,
            lightning_indexer_layers=[1],
            provide_language_model=lambda **_kwargs: LanguageModel(),
            share_embeddings_and_output_weights=False,
        )
        return MiniMaxM3VLModel(config, pre_process=False)

    rendezvous = tmp_path / "dist_init"
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
    )
    parallel_state.initialize_model_parallel()

    try:
        source = make_model()
        with torch.no_grad():
            for parameter_idx, parameter in enumerate(source.lightning_indexers.parameters()):
                values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
                parameter.copy_(values.add(parameter_idx).to(parameter.dtype))
        expected_state = {name: tensor.clone() for name, tensor in source.state_dict().items()}
        save(source.sharded_state_dict(), checkpoint_dir, async_sharded_save=False)

        destination = make_model()
        loaded_state = load(destination.sharded_state_dict(), checkpoint_dir)
        load_result = destination.load_state_dict(loaded_state, strict=True)

        assert not load_result.missing_keys
        assert not load_result.unexpected_keys
        for name, expected in expected_state.items():
            torch.testing.assert_close(destination.state_dict()[name], expected, rtol=0, atol=0)
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def _vision_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=3,
        num_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        attention_dropout=0.0,
        layer_norm_eps=1e-5,
        rope_theta=10000.0,
    )


def test_vision_model_preserves_legacy_checkpoint_namespace_and_shapes():
    model = MiniMaxM3VisionModel(_vision_config(), spatial_merge_size=2)
    pixel_values = torch.randn(4, 3 * 2 * 2 * 2)
    grid_thw = torch.tensor([[1, 2, 2]])

    output = model(pixel_values, grid_thw)

    assert output.shape == (1, 4, 24)
    state = model.state_dict()
    assert "embeddings.patch_embedding.weight" in state
    assert "pre_layrnorm.weight" in state
    assert "encoder.layers.0.self_attn.q_proj.weight" in state
    assert "encoder.layers.0.mlp.fc2.bias" in state


def test_vision_model_matches_native_transformers_forward():
    configuration = pytest.importorskip("transformers.models.minimax_m3_vl.configuration_minimax_m3_vl")
    modeling = pytest.importorskip("transformers.models.minimax_m3_vl.modeling_minimax_m3_vl")
    hf_config = configuration.MiniMaxM3VLVisionConfig(
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=3,
        num_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        attention_dropout=0.0,
        layer_norm_eps=1e-5,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    native_model = modeling.MiniMaxM3VLVisionModel(hf_config).eval()
    bridge_model = MiniMaxM3VisionModel(_vision_config(), spatial_merge_size=2).eval()

    native_state = native_model.state_dict()
    renamed_state = {}
    for name in bridge_model.state_dict():
        native_name = name.replace("embeddings.patch_embedding", "embeddings.proj")
        native_name = native_name.replace("encoder.layers", "layers")
        renamed_state[name] = native_state[native_name]
    bridge_model.load_state_dict(renamed_state, strict=True)

    pixel_values = torch.randn(24, 3 * 2 * 2 * 2)
    grid_thw = torch.tensor([[1, 2, 4], [2, 4, 2]])

    with torch.no_grad():
        expected = native_model(pixel_values, grid_thw).last_hidden_state
        actual = bridge_model(pixel_values, grid_thw)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_patch_embedding_remains_float32():
    model = MiniMaxM3VisionModel(_vision_config(), spatial_merge_size=2)
    patch_embedding = model.embeddings.patch_embedding

    assert patch_embedding.weight.dtype == torch.float32
    assert patch_embedding._keep_in_float32_parameter_names == ("weight",)


def test_vision_rope_leaves_unrotated_tail_unchanged():
    query = torch.randn(1, 2, 1, 8)
    key = torch.randn(1, 2, 1, 8)
    cosine = torch.zeros(2, 6)
    sine = torch.ones(2, 6)

    rotated_query, rotated_key = _apply_vision_rope(query, key, cosine, sine)

    torch.testing.assert_close(rotated_query[..., 6:], query[..., 6:])
    torch.testing.assert_close(rotated_key[..., 6:], key[..., 6:])
    assert not torch.equal(rotated_query[..., :6], query[..., :6])
    assert not torch.equal(rotated_key[..., :6], key[..., :6])


def test_two_projectors_match_minimax_merge_shapes():
    patch_projector = MiniMaxM3ProjectorMLP(24, 32, 16, bias=True)
    merge_projector = MiniMaxM3ProjectorMLP(16 * 4, 32, 16, bias=True)
    patch_features = patch_projector(torch.randn(4, 24))

    output = merge_projector(patch_features.reshape(1, -1))

    assert patch_features.shape == (4, 16)
    assert output.shape == (1, 16)


def test_scatter_features_handles_image_and_count_mismatch():
    inputs_embeds = torch.zeros(1, 3, 4)
    input_ids = torch.tensor([[7, 1, 7]])
    features = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    output = MiniMaxM3VLModel._scatter_features(
        inputs_embeds,
        input_ids,
        token_id=7,
        features=features,
        modality="image",
    )

    torch.testing.assert_close(output[0, 0], features[0])
    torch.testing.assert_close(output[0, 2], features[1])

    with pytest.raises(ValueError, match="does not match placeholder"):
        MiniMaxM3VLModel._scatter_features(
            inputs_embeds,
            input_ids,
            token_id=7,
            features=features[:1],
            modality="image",
        )


def test_wrapper_exposes_decoder_and_forwards_inference_arguments(monkeypatch):
    captured = {}

    class LanguageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = object()

        def forward(self, **kwargs):
            captured.update(kwargs)
            return torch.ones(1)

    model = MiniMaxM3VLModel.__new__(MiniMaxM3VLModel)
    torch.nn.Module.__init__(model)
    model.pre_process = False
    model.config = SimpleNamespace(sequence_parallel=False, _pg_collection=None)
    model.language_model = LanguageModel()

    monkeypatch.setattr(
        "megatron.bridge.models.minimax_m3.modeling_minimax_m3_vl.slice_batch_for_context_parallel",
        lambda **kwargs: (
            kwargs["inputs_embeds"],
            kwargs["labels"],
            kwargs["loss_mask"],
            kwargs["position_ids"],
            kwargs["attention_mask"],
        ),
    )
    inference_context = object()
    inference_params = object()
    extra_block_kwargs = {"key": "value"}

    sliced_loss_mask = torch.ones(1, 1)
    output, returned_loss_mask = model(
        mm_token_type_ids=torch.zeros(1, 1, dtype=torch.int32),
        inference_context=inference_context,
        inference_params=inference_params,
        extra_block_kwargs=extra_block_kwargs,
        loss_mask=sliced_loss_mask,
    )

    assert model.decoder is model.language_model.decoder
    torch.testing.assert_close(output, torch.ones(1))
    assert returned_loss_mask is sliced_loss_mask
    assert captured["inference_context"] is inference_context
    assert captured["inference_params"] is inference_params
    assert captured["extra_block_kwargs"] == extra_block_kwargs


def test_wrapper_preserves_gpt_step_tensor_return_contract(monkeypatch):
    class LanguageModel(torch.nn.Module):
        def forward(self, **_kwargs):
            return torch.ones(1)

    model = MiniMaxM3VLModel.__new__(MiniMaxM3VLModel)
    torch.nn.Module.__init__(model)
    model.pre_process = False
    model.config = SimpleNamespace(sequence_parallel=False, _pg_collection=None)
    model.language_model = LanguageModel()

    monkeypatch.setattr(
        "megatron.bridge.models.minimax_m3.modeling_minimax_m3_vl.slice_batch_for_context_parallel",
        lambda **kwargs: (
            kwargs["inputs_embeds"],
            kwargs["labels"],
            kwargs["loss_mask"],
            kwargs["position_ids"],
            kwargs["attention_mask"],
        ),
    )

    output = model()

    assert isinstance(output, torch.Tensor)
    torch.testing.assert_close(output, torch.ones(1))
