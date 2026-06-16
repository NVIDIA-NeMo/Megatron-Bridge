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

import copy
import datetime
import os
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerConfig,
)

from megatron.bridge.models.qwen_omni.modeling_qwen3_omni.model import Qwen3OmniModel
from megatron.bridge.models.qwen_omni.modeling_qwen3_omni.rope import get_rope_index
from megatron.bridge.models.qwen_omni.modeling_qwen3_omni.thinker_model import (
    Qwen3OmniThinkerModel,
    _trim_feature_sequence,
)
from megatron.bridge.models.qwen_omni.modeling_qwen3_omni.transformer_config import (
    Qwen3OmniTransformerConfig,
)


HIDDEN_SIZE = 128
IMAGE_TOKEN_ID = 900
VIDEO_TOKEN_ID = 901
AUDIO_TOKEN_ID = 902
VISION_START_TOKEN_ID = 903
AUDIO_START_TOKEN_ID = 904


def _make_toy_thinker_config():
    return Qwen3OmniMoeThinkerConfig(
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "out_hidden_size": HIDDEN_SIZE,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0],
        },
        audio_config={
            "num_mel_bins": 8,
            "d_model": 32,
            "encoder_attention_heads": 4,
            "encoder_ffn_dim": 64,
            "encoder_layers": 2,
            "output_dim": HIDDEN_SIZE,
            "downsample_hidden_size": 16,
        },
        text_config={
            "num_hidden_layers": 2,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": 256,
            "moe_intermediate_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "vocab_size": 1000,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-6,
            "attention_bias": False,
            "rope_theta": 1000000.0,
            "rope_scaling": {"rope_type": "default", "mrope_section": [4, 6, 6]},
        },
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        audio_token_id=AUDIO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        audio_start_token_id=AUDIO_START_TOKEN_ID,
    )


@pytest.fixture(scope="module")
def thinker_config():
    return _make_toy_thinker_config()


class TestQwen3OmniModel:
    _original_env: dict[str, str | None] = {}

    @classmethod
    def setup_class(cls):
        if not dist.is_initialized():
            cls._original_env = {
                key: os.environ.get(key) for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE")
            }

            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = cls._find_free_port()
            os.environ["RANK"] = "0"
            os.environ["LOCAL_RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"

            device_count = torch.cuda.device_count()
            if device_count > 0:
                torch.cuda.set_device(0)

            dist.init_process_group(
                backend="nccl" if device_count > 0 else "gloo",
                world_size=1,
                rank=0,
                timeout=datetime.timedelta(minutes=30),
            )

    @classmethod
    def teardown_class(cls):
        if dist.is_initialized():
            dist.destroy_process_group()
        for key, value in cls._original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    @staticmethod
    def _find_free_port() -> str:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return str(sock.getsockname()[1])

    def _setup_parallel_state(self, tp_size=1, pp_size=1):
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            virtual_pipeline_model_parallel_size=None,
            context_parallel_size=1,
        )
        if torch.cuda.is_available():
            model_parallel_cuda_manual_seed(123)
        else:
            torch.manual_seed(123)

    def teardown_method(self):
        parallel_state.destroy_model_parallel()

    @staticmethod
    def _make_language_config():
        return Qwen3OmniTransformerConfig(
            num_layers=2,
            hidden_size=HIDDEN_SIZE,
            num_attention_heads=4,
            num_query_groups=2,
            kv_channels=HIDDEN_SIZE // 4,
            ffn_hidden_size=256,
            moe_ffn_hidden_size=64,
            num_moe_experts=8,
            moe_router_topk=2,
            vocab_size=1000,
            language_max_sequence_length=128,
            normalization="RMSNorm",
            activation_func=F.silu,
            gated_linear_unit=True,
            add_bias_linear=False,
            add_qkv_bias=False,
            qk_layernorm=True,
            layernorm_epsilon=1e-6,
            bf16=False,
            use_cpu_initialization=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            mrope_section=[4, 6, 6],
            image_token_id=IMAGE_TOKEN_ID,
            video_token_id=VIDEO_TOKEN_ID,
            audio_token_id=AUDIO_TOKEN_ID,
            vision_start_token_id=VISION_START_TOKEN_ID,
            audio_start_token_id=AUDIO_START_TOKEN_ID,
            position_id_per_seconds=25,
            seconds_per_chunk=2,
        )

    @staticmethod
    def _make_layer_spec():
        if not torch.cuda.is_available():
            return get_gpt_layer_local_spec(
                num_experts=8,
                moe_grouped_gemm=True,
                qk_layernorm=True,
                normalization="RMSNorm",
            )
        return get_gpt_layer_with_transformer_engine_spec(
            num_experts=8,
            moe_grouped_gemm=True,
            qk_layernorm=True,
            fp8=False,
        )

    def _build_model(self, thinker_config):
        self._setup_parallel_state(tp_size=1, pp_size=1)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        return Qwen3OmniModel(
            language_transformer_config=self._make_language_config(),
            language_transformer_layer_spec=self._make_layer_spec(),
            thinker_transformer_config=thinker_config,
            parallel_output=True,
            pre_process=True,
            post_process=True,
            pg_collection=pg_collection,
        )

    def test_multimodal_runtime_switches_are_applied(self, thinker_config, monkeypatch):
        self._setup_parallel_state(tp_size=1, pp_size=1)
        thinker_config = copy.deepcopy(thinker_config)

        calls = {"vision_gc": 0, "audio_gc": 0, "vision_inputs_grad": 0, "audio_inputs_grad": 0}

        class _FakeTower(torch.nn.Module):
            def __init__(self, gc_key: str, input_key: str, embed_attr: str):
                super().__init__()
                self.config = SimpleNamespace()
                self._gc_key = gc_key
                self._input_key = input_key
                setattr(self, embed_attr, torch.nn.Linear(4, 4, bias=False))

            def gradient_checkpointing_enable(self):
                calls[self._gc_key] += 1

            def enable_input_require_grads(self):
                calls[self._input_key] += 1

        monkeypatch.setattr(
            "megatron.bridge.models.qwen_omni.modeling_qwen3_omni.thinker_model.Qwen3OmniMoeVisionEncoderHF._from_config",
            lambda _cfg: _FakeTower("vision_gc", "vision_inputs_grad", "patch_embed"),
        )
        monkeypatch.setattr(
            "megatron.bridge.models.qwen_omni.modeling_qwen3_omni.thinker_model.Qwen3OmniMoeAudioEncoderHF._from_config",
            lambda _cfg: _FakeTower("audio_gc", "audio_inputs_grad", "conv_out"),
        )

        cfg = self._make_language_config()
        cfg.vit_gradient_checkpointing = True
        cfg.multimodal_attn_impl = "flash_attention_2"

        model = Qwen3OmniModel(
            language_transformer_config=cfg,
            language_transformer_layer_spec=self._make_layer_spec(),
            thinker_transformer_config=thinker_config,
            parallel_output=True,
            pre_process=True,
            post_process=True,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )

        assert calls["vision_gc"] == 1
        assert calls["audio_gc"] == 1
        assert calls["vision_inputs_grad"] == 1
        assert calls["audio_inputs_grad"] == 1
        assert getattr(model.thinker.visual.config, "_attn_implementation") == "flash_attention_2"
        assert getattr(model.thinker.audio_model.config, "_attn_implementation") == "flash_attention_2"
        assert getattr(model.thinker.visual.config, "use_flash_attn") is True
        assert getattr(model.thinker.audio_model.config, "use_flash_attn") is True
        assert model.thinker.visual.get_input_embeddings() is model.thinker.visual.patch_embed
        assert model.thinker.audio_model.get_input_embeddings() is model.thinker.audio_model.conv_out

    def test_model_freeze_api(self, thinker_config):
        model = self._build_model(thinker_config)
        model.freeze(freeze_language_model=True)

        for name, param in model.named_parameters():
            if name.startswith("thinker.language_model"):
                assert param.requires_grad is False

    def test_set_input_tensor(self, thinker_config):
        model = self._build_model(thinker_config)
        test_tensor = torch.randn(2, 4, HIDDEN_SIZE)
        model.set_input_tensor([test_tensor])
        assert model.thinker.encoder_hidden_state is not None

    def test_text_only_forward(self, thinker_config):
        model = self._build_model(thinker_config)
        if torch.cuda.is_available():
            model = model.to("cuda")
            device = "cuda"
        else:
            device = "cpu"

        input_ids = torch.randint(0, 1000, (1, 16), device=device)
        labels = torch.randint(0, 1000, (1, 16), device=device)
        output = model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=None,
        )
        assert output is not None

    def test_image_forward(self, thinker_config):
        model = self._build_model(thinker_config)
        if torch.cuda.is_available():
            model = model.to("cuda")
            device = "cuda"
        else:
            device = "cpu"

        input_ids = torch.tensor(
            [[VISION_START_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 12, 13, 14]],
            device=device,
        )
        labels = torch.randint(0, 1000, input_ids.shape, device=device)
        pixel_values = torch.randn(4, 3 * 1 * 2 * 2, device=device)
        image_grid_thw = torch.tensor([[1, 2, 2]], device=device)

        output = model(
            input_ids=input_ids,
            labels=labels,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        assert output is not None

    def test_image_forward_sequence_parallel_path(self, thinker_config, monkeypatch):
        model = self._build_model(thinker_config)
        model.thinker.config.sequence_parallel = True

        calls = {"scatter": 0, "scatter_group": None, "split": 0}

        def _identity_scatter(x, *, group=None):
            calls["scatter"] += 1
            calls["scatter_group"] = group
            return x

        def _identity_split(visual_pos_masks, deepstack_visual_embeds, **kwargs):
            calls["split"] += 1
            return visual_pos_masks, deepstack_visual_embeds

        monkeypatch.setattr(
            "megatron.bridge.models.qwen_omni.modeling_qwen3_omni.thinker_model.tensor_parallel.scatter_to_sequence_parallel_region",
            _identity_scatter,
        )
        monkeypatch.setattr(
            "megatron.bridge.models.qwen_omni.modeling_qwen3_omni.thinker_model.split_deepstack_embs",
            _identity_split,
        )

        if torch.cuda.is_available():
            model = model.to("cuda")
            device = "cuda"
        else:
            device = "cpu"

        input_ids = torch.tensor(
            [[VISION_START_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 12, 13, 14]],
            device=device,
        )
        labels = torch.randint(0, 1000, input_ids.shape, device=device)
        pixel_values = torch.randn(4, 3 * 1 * 2 * 2, device=device)
        image_grid_thw = torch.tensor([[1, 2, 2]], device=device)

        output = model(
            input_ids=input_ids,
            labels=labels,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

        assert output is not None
        assert calls["scatter"] == 1
        assert calls["scatter_group"] is model.thinker.pg_collection.tp
        assert calls["split"] == 1

    def test_cp_mrope_uses_full_input_without_local_attention_mask(self, thinker_config, monkeypatch):
        model = self._build_model(thinker_config)

        class _MockProcessGroup:
            def __init__(self, size=1, rank=0):
                self._size = size
                self._rank = rank

            def size(self):
                return self._size

            def rank(self):
                return self._rank

        model.thinker.pg_collection = SimpleNamespace(
            cp=_MockProcessGroup(size=2),
            tp=_MockProcessGroup(size=1),
        )

        class _FakeLanguageModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.forward_kwargs = None

            def embedding(self, input_ids, position_ids):  # noqa: ARG002
                return torch.zeros(input_ids.size(1), input_ids.size(0), HIDDEN_SIZE, device=input_ids.device)

            def forward(self, **kwargs):
                self.forward_kwargs = kwargs
                return torch.tensor(0.0, device=kwargs["decoder_input"].device)

        fake_language_model = _FakeLanguageModel()
        model.thinker.language_model = fake_language_model

        rope_calls = {}

        def _fake_get_rope_index(*args, **kwargs):  # noqa: ARG001
            input_ids = args[7]
            rope_calls["attention_mask"] = kwargs.get("attention_mask")
            position_ids = torch.zeros(
                3, input_ids.size(0), input_ids.size(1), dtype=torch.long, device=input_ids.device
            )
            return position_ids, torch.zeros(input_ids.size(0), 1, dtype=torch.long, device=input_ids.device)

        monkeypatch.setattr(
            "megatron.bridge.models.qwen_omni.modeling_qwen3_omni.thinker_model.get_rope_index",
            _fake_get_rope_index,
        )
        monkeypatch.setattr(
            "megatron.bridge.models.qwen_omni.modeling_qwen3_omni.thinker_model.split_data_cp_rank",
            lambda x, *args, **kwargs: x,
        )
        monkeypatch.setattr(
            "megatron.bridge.models.qwen_omni.modeling_qwen3_omni.thinker_model.split_deepstack_embs",
            lambda visual_pos_masks, deepstack_visual_embeds, **kwargs: (visual_pos_masks, deepstack_visual_embeds),
        )

        input_ids = torch.tensor([[AUDIO_START_TOKEN_ID, AUDIO_TOKEN_ID, 12, 13]])
        attention_mask = torch.tensor([[1, 1]])
        output = model.thinker(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_feature_lengths=torch.tensor([1]),
        )

        assert output is not None
        assert rope_calls["attention_mask"] is None
        assert fake_language_model.forward_kwargs["attention_mask"] is attention_mask

    def test_packed_forward_resets_rotary_thd_state(self):
        class _MockProcessGroup:
            def size(self):
                return 1

            def rank(self):
                return 0

        class _FakeLanguageModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rotary_pos_emb = SimpleNamespace(is_thd_format=False)
                self.forward_kwargs = []

            def forward(self, **kwargs):
                self.forward_kwargs.append(kwargs)
                return torch.tensor(0.0)

        fake_language_model = _FakeLanguageModel()
        thinker = SimpleNamespace(
            pg_collection=SimpleNamespace(cp=_MockProcessGroup(), tp=_MockProcessGroup()),
            config=SimpleNamespace(sequence_parallel=False),
            pre_process=False,
            language_model=fake_language_model,
        )

        input_ids = torch.tensor([[1, 2, 3, 4]])
        position_ids = torch.arange(input_ids.size(1)).view(1, 1, -1).expand(3, input_ids.size(0), -1)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        Qwen3OmniThinkerModel.forward(
            thinker,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            packed_seq_params=object(),
        )
        assert fake_language_model.rotary_pos_emb.is_thd_format is True

        Qwen3OmniThinkerModel.forward(
            thinker,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            packed_seq_params=None,
        )
        assert fake_language_model.rotary_pos_emb.is_thd_format is False

    def test_audio_forward(self, thinker_config):
        model = self._build_model(thinker_config)
        if torch.cuda.is_available():
            model = model.to("cuda")
            device = "cuda"
        else:
            device = "cpu"

        input_ids = torch.tensor(
            [[AUDIO_START_TOKEN_ID, AUDIO_TOKEN_ID, AUDIO_TOKEN_ID, 21, 22, 23]],
            device=device,
        )
        labels = torch.randint(0, 1000, input_ids.shape, device=device)
        input_features = torch.randn(1, 8, 10, device=device)
        feature_attention_mask = torch.ones(1, 10, dtype=torch.long, device=device)

        output = model(
            input_ids=input_ids,
            labels=labels,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
        )
        assert output is not None

    def test_audio_feature_trim_handles_multiple_audio_samples(self):
        class _FakeAudioModel:
            dtype = torch.float32

            def __call__(self, input_features, feature_lens):
                assert input_features.shape == (8, 3648)
                assert feature_lens.tolist() == [3000, 648]
                return SimpleNamespace(last_hidden_state=torch.arange(474 * 2, dtype=torch.float32).view(474, 2))

        thinker = SimpleNamespace(audio_model=_FakeAudioModel())
        input_features = torch.randn(2, 8, 3000)
        feature_attention_mask = torch.zeros(2, 3000, dtype=torch.long)
        feature_attention_mask[0, :3000] = 1
        feature_attention_mask[1, :648] = 1

        audio_embeds = Qwen3OmniThinkerModel.get_audio_features(
            thinker,
            input_features,
            feature_attention_mask=feature_attention_mask,
            expected_audio_token_counts=torch.tensor([390, 84]),
        )

        assert audio_embeds.shape == (474, 2)

    def test_audio_only_rope_index(self):
        input_ids = torch.tensor([[AUDIO_START_TOKEN_ID, AUDIO_TOKEN_ID, AUDIO_TOKEN_ID, 17, 18]])
        audio_seqlens = torch.tensor([16])

        position_ids, mrope_position_deltas = get_rope_index(
            spatial_merge_size=1,
            image_token_id=IMAGE_TOKEN_ID,
            video_token_id=VIDEO_TOKEN_ID,
            audio_token_id=AUDIO_TOKEN_ID,
            vision_start_token_id=VISION_START_TOKEN_ID,
            audio_start_token_id=AUDIO_START_TOKEN_ID,
            position_id_per_seconds=25,
            input_ids=input_ids,
            audio_seqlens=audio_seqlens,
        )

        assert position_ids.shape == (3, 1, input_ids.shape[1])
        assert mrope_position_deltas.shape == (1, 1)

    def test_video_rope_index_requires_second_per_grid(self):
        input_ids = torch.tensor([[VISION_START_TOKEN_ID, VIDEO_TOKEN_ID, VIDEO_TOKEN_ID, 17, 18]])
        video_grid_thw = torch.tensor([[1, 1, 2]])
        video_second_per_grid = torch.tensor([2.0])

        position_ids, mrope_position_deltas = get_rope_index(
            spatial_merge_size=1,
            image_token_id=IMAGE_TOKEN_ID,
            video_token_id=VIDEO_TOKEN_ID,
            audio_token_id=AUDIO_TOKEN_ID,
            vision_start_token_id=VISION_START_TOKEN_ID,
            audio_start_token_id=AUDIO_START_TOKEN_ID,
            position_id_per_seconds=25,
            input_ids=input_ids,
            video_grid_thw=video_grid_thw,
            second_per_grids=video_second_per_grid,
        )

        assert position_ids.shape == (3, 1, input_ids.shape[1])
        assert mrope_position_deltas.shape == (1, 1)

    def test_audio_rope_index_accepts_multidim_attention_mask(self):
        input_ids = torch.tensor([[AUDIO_START_TOKEN_ID, AUDIO_TOKEN_ID, AUDIO_TOKEN_ID, 17, 18]])
        audio_seqlens = torch.tensor([16])
        attention_mask = torch.zeros(1, 1, input_ids.shape[1], input_ids.shape[1], dtype=torch.long)
        attention_mask[:, :, :, : input_ids.shape[1]] = 1

        position_ids, mrope_position_deltas = get_rope_index(
            spatial_merge_size=1,
            image_token_id=IMAGE_TOKEN_ID,
            video_token_id=VIDEO_TOKEN_ID,
            audio_token_id=AUDIO_TOKEN_ID,
            vision_start_token_id=VISION_START_TOKEN_ID,
            audio_start_token_id=AUDIO_START_TOKEN_ID,
            position_id_per_seconds=25,
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_seqlens=audio_seqlens,
        )

        assert position_ids.shape == (3, 1, input_ids.shape[1])
        assert mrope_position_deltas.shape == (1, 1)

    def test_audio_rope_index_truncates_positions_to_surviving_tokens(self):
        seq_len = 8
        input_ids = torch.tensor([[AUDIO_START_TOKEN_ID] + [AUDIO_TOKEN_ID] * (seq_len - 1)])
        attention_mask = torch.ones(1, seq_len, dtype=torch.long)
        audio_seqlens = torch.tensor([512])

        position_ids, mrope_position_deltas = get_rope_index(
            spatial_merge_size=1,
            image_token_id=IMAGE_TOKEN_ID,
            video_token_id=VIDEO_TOKEN_ID,
            audio_token_id=AUDIO_TOKEN_ID,
            vision_start_token_id=VISION_START_TOKEN_ID,
            audio_start_token_id=AUDIO_START_TOKEN_ID,
            position_id_per_seconds=25,
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_seqlens=audio_seqlens,
        )

        assert position_ids.shape == (3, 1, seq_len)
        assert mrope_position_deltas.shape == (1, 1)

    def test_trim_feature_sequence_clips_to_surviving_placeholders(self):
        features = torch.randn(12, HIDDEN_SIZE)
        multiscale = [torch.randn(12, HIDDEN_SIZE), torch.randn(12, HIDDEN_SIZE)]

        trimmed_features, trimmed_multiscale = _trim_feature_sequence(
            features,
            multiscale,
            expected_tokens=5,
            feature_name="Image features",
        )

        assert trimmed_features is not None
        assert trimmed_multiscale is not None
        assert trimmed_features.shape == (5, HIDDEN_SIZE)
        assert all(feature.shape == (5, HIDDEN_SIZE) for feature in trimmed_multiscale)

    def test_model_wrapper_exposes_pipeline_flags(self, thinker_config):
        model = self._build_model(thinker_config)

        assert model.pre_process is True
        assert model.post_process is True
        assert model.share_embeddings_and_output_weights == model.thinker.share_embeddings_and_output_weights
