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

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from megatron.bridge.diffusion.models.wan.flow_matching.flow_inference_pipeline import (
    FlowInferencePipeline,
    _encode_text,
)


pytestmark = [pytest.mark.unit]

PIPELINE_MODULE = "megatron.bridge.diffusion.models.wan.flow_matching.flow_inference_pipeline"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_pipeline(**overrides):
    """Create a lightweight FlowInferencePipeline skipping __init__."""
    pip = object.__new__(FlowInferencePipeline)
    defaults = dict(
        device=torch.device("cpu"),
        rank=0,
        t5_cpu=False,
        tensor_parallel_size=1,
        context_parallel_size=1,
        pipeline_parallel_size=1,
        sequence_parallel=False,
        pipeline_dtype=torch.float32,
        num_train_timesteps=1000,
        param_dtype=torch.float32,
        text_len=512,
        sp_size=1,
        model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        sample_neg_prompt="",
    )
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(pip, k, v)
    return pip


def _make_mock_model(hidden_size=16):
    """Create a mock model that echoes its first positional argument."""

    _hs = hidden_size  # avoid class-scope lookup issue

    class _Cfg:
        hidden_size = _hs
        qkv_format = "thd"

    class _Model:
        config = _Cfg()
        patch_size = (1, 2, 2)

        def __call__(self, x, **kwargs):
            return x

        def parameters(self):
            return iter([torch.zeros(1)])

        def set_input_tensor(self, x):
            pass

        def to(self, device):
            return self

        def cpu(self):
            return self

        def no_sync(self):
            from contextlib import contextmanager

            @contextmanager
            def _ns():
                yield

            return _ns()

    return _Model()


# ===========================================================================
# Tests for _encode_text
# ===========================================================================
class TestEncodeText:
    def test_output_shape_trimmed_to_true_length(self):
        hidden_dim = 32
        true_len = 5

        mock_tokenizer = MagicMock()
        attn_mask = torch.zeros(1, 512)
        attn_mask[0, :true_len] = 1.0
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, 512, dtype=torch.long),
            "attention_mask": attn_mask,
        }

        encoder_output = torch.randn(1, 512, hidden_dim)
        mock_text_encoder = MagicMock()
        mock_text_encoder.return_value.last_hidden_state = encoder_output

        result = _encode_text(mock_tokenizer, mock_text_encoder, "cpu", "a test caption")

        assert result.shape == (true_len, hidden_dim)

    def test_tokenizer_called_with_correct_args(self):
        mock_tokenizer = MagicMock()
        attn_mask = torch.ones(1, 512)
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, 512, dtype=torch.long),
            "attention_mask": attn_mask,
        }
        mock_text_encoder = MagicMock()
        mock_text_encoder.return_value.last_hidden_state = torch.randn(1, 512, 16)

        _encode_text(mock_tokenizer, mock_text_encoder, "cpu", "  hello world  ")

        mock_tokenizer.assert_called_once_with(
            "hello world",
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

    def test_full_length_no_padding(self):
        hidden_dim = 16
        max_len = 512
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, max_len, dtype=torch.long),
            "attention_mask": torch.ones(1, max_len),
        }
        mock_text_encoder = MagicMock()
        mock_text_encoder.return_value.last_hidden_state = torch.randn(1, max_len, hidden_dim)

        result = _encode_text(mock_tokenizer, mock_text_encoder, "cpu", "text")
        assert result.shape == (max_len, hidden_dim)


# ===========================================================================
# Tests for _select_checkpoint_dir
# ===========================================================================
class TestSelectCheckpointDir:
    def test_latest_checkpoint(self, tmp_path):
        base = tmp_path / "ckpts"
        os.makedirs(base / "iter_0000100")
        os.makedirs(base / "iter_0000200")

        pip = _make_pipeline()
        latest = pip._select_checkpoint_dir(str(base), checkpoint_step=None)
        assert latest.endswith("iter_0000200")

    def test_specific_step(self, tmp_path):
        base = tmp_path / "ckpts"
        os.makedirs(base / "iter_0000100")
        os.makedirs(base / "iter_0000200")

        pip = _make_pipeline()
        specific = pip._select_checkpoint_dir(str(base), checkpoint_step=100)
        assert specific.endswith("iter_0000100")

    def test_missing_step_raises(self, tmp_path):
        base = tmp_path / "ckpts"
        os.makedirs(base / "iter_0000100")

        pip = _make_pipeline()
        with pytest.raises(FileNotFoundError, match="not found"):
            pip._select_checkpoint_dir(str(base), checkpoint_step=999)

    def test_nonexistent_base_dir_raises(self, tmp_path):
        pip = _make_pipeline()
        with pytest.raises(FileNotFoundError, match="does not exist"):
            pip._select_checkpoint_dir(str(tmp_path / "nonexistent"), checkpoint_step=None)

    def test_empty_base_dir_no_checkpoints(self, tmp_path):
        base = tmp_path / "ckpts"
        os.makedirs(base)

        pip = _make_pipeline()
        with pytest.raises(FileNotFoundError, match="No checkpoints found"):
            pip._select_checkpoint_dir(str(base), checkpoint_step=None)

    def test_ignores_non_iter_directories(self, tmp_path):
        base = tmp_path / "ckpts"
        os.makedirs(base / "iter_0000050")
        os.makedirs(base / "random_dir")
        (base / "some_file.txt").write_text("x")

        pip = _make_pipeline()
        result = pip._select_checkpoint_dir(str(base), checkpoint_step=None)
        assert result.endswith("iter_0000050")

    def test_step_zero_padded(self, tmp_path):
        base = tmp_path / "ckpts"
        os.makedirs(base / "iter_0000005")

        pip = _make_pipeline()
        result = pip._select_checkpoint_dir(str(base), checkpoint_step=5)
        assert result.endswith("iter_0000005")


# ===========================================================================
# Tests for forward_pp_step
# ===========================================================================
class TestForwardPPStep:
    def test_no_pp(self, monkeypatch):
        """PP=1: model is called directly, output echoes input."""
        from megatron.core import parallel_state

        monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_world_size", lambda: 1, raising=False)

        pip = _make_pipeline()
        pip.model = _make_mock_model()

        S, B, H = 8, 1, 16
        latent = torch.randn(S, B, H)
        grid = [(2, 2, 2)]
        ts = torch.tensor([10.0])

        out = pip.forward_pp_step(latent, grid_sizes=grid, max_video_seq_len=S, timestep=ts, arg_c={})
        assert out.shape == latent.shape
        assert torch.allclose(out, latent)

    def test_pp_first_stage(self, monkeypatch):
        """PP>1, first stage: should call model, send to next, and return broadcast result."""
        from megatron.core import parallel_state

        monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_world_size", lambda: 2, raising=False)
        monkeypatch.setattr(parallel_state, "is_pipeline_first_stage", lambda ignore_virtual=True: True, raising=False)
        monkeypatch.setattr(parallel_state, "is_pipeline_last_stage", lambda ignore_virtual=True: False, raising=False)

        S, B, H = 8, 2, 16
        expected_output = torch.randn(S, B, H)

        pip = _make_pipeline()
        pip.model = _make_mock_model(hidden_size=H)

        send_mock = MagicMock()
        broadcast_mock = MagicMock(return_value=expected_output)
        monkeypatch.setattr(f"{PIPELINE_MODULE}.send_to_next_pipeline_rank", send_mock)
        monkeypatch.setattr(f"{PIPELINE_MODULE}.broadcast_from_last_pipeline_stage", broadcast_mock)

        latent = torch.randn(S, B, H)
        out = pip.forward_pp_step(
            latent, grid_sizes=[(2, 2, 2)], max_video_seq_len=S, timestep=torch.tensor([10.0, 10.0]), arg_c={}
        )

        send_mock.assert_called_once()
        broadcast_mock.assert_called_once()
        assert torch.equal(out, expected_output)

    def test_pp_last_stage(self, monkeypatch):
        """PP>1, last stage: should recv, call model, and broadcast."""
        from megatron.core import parallel_state

        monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_world_size", lambda: 2, raising=False)
        monkeypatch.setattr(
            parallel_state, "is_pipeline_first_stage", lambda ignore_virtual=True: False, raising=False
        )
        monkeypatch.setattr(parallel_state, "is_pipeline_last_stage", lambda ignore_virtual=True: True, raising=False)

        S, B, H = 8, 2, 16
        expected_output = torch.randn(S, B, H)

        pip = _make_pipeline()
        pip.model = _make_mock_model(hidden_size=H)

        recv_mock = MagicMock()
        broadcast_mock = MagicMock(return_value=expected_output)
        monkeypatch.setattr(f"{PIPELINE_MODULE}.recv_from_prev_pipeline_rank_", recv_mock)
        monkeypatch.setattr(f"{PIPELINE_MODULE}.broadcast_from_last_pipeline_stage", broadcast_mock)

        latent = torch.randn(S, B, H)
        out = pip.forward_pp_step(
            latent, grid_sizes=[(2, 2, 2)], max_video_seq_len=S, timestep=torch.tensor([10.0, 10.0]), arg_c={}
        )

        recv_mock.assert_called_once()
        broadcast_mock.assert_called_once()
        assert torch.equal(out, expected_output)

    def test_pp_intermediate_stage(self, monkeypatch):
        """PP>1, intermediate stage: recv -> model -> send -> receive broadcast."""
        from megatron.core import parallel_state

        monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_world_size", lambda: 3, raising=False)
        monkeypatch.setattr(
            parallel_state, "is_pipeline_first_stage", lambda ignore_virtual=True: False, raising=False
        )
        monkeypatch.setattr(parallel_state, "is_pipeline_last_stage", lambda ignore_virtual=True: False, raising=False)

        S, B, H = 8, 2, 16
        expected_output = torch.randn(S, B, H)

        pip = _make_pipeline()
        pip.model = _make_mock_model(hidden_size=H)

        recv_mock = MagicMock()
        send_mock = MagicMock()
        broadcast_mock = MagicMock(return_value=expected_output)
        monkeypatch.setattr(f"{PIPELINE_MODULE}.recv_from_prev_pipeline_rank_", recv_mock)
        monkeypatch.setattr(f"{PIPELINE_MODULE}.send_to_next_pipeline_rank", send_mock)
        monkeypatch.setattr(f"{PIPELINE_MODULE}.broadcast_from_last_pipeline_stage", broadcast_mock)

        latent = torch.randn(S, B, H)
        out = pip.forward_pp_step(
            latent, grid_sizes=[(2, 2, 2)], max_video_seq_len=S, timestep=torch.tensor([10.0, 10.0]), arg_c={}
        )

        recv_mock.assert_called_once()
        send_mock.assert_called_once()
        broadcast_mock.assert_called_once()
        assert torch.equal(out, expected_output)


# ===========================================================================
# Tests for setup_model_from_checkpoint
# ===========================================================================
class TestSetupModelFromCheckpoint:
    def test_calls_provider_and_loads_checkpoint(self, monkeypatch):
        mock_provider_cls = MagicMock()
        mock_provider_instance = MagicMock()
        mock_provider_cls.return_value = mock_provider_instance

        fake_model = MagicMock(spec=[])  # spec=[] so hasattr(model, "module") is False
        mock_load = MagicMock(return_value=fake_model)

        monkeypatch.setattr(f"{PIPELINE_MODULE}.WanModelProvider", mock_provider_cls)
        monkeypatch.setattr(f"{PIPELINE_MODULE}._load_megatron_model", mock_load)

        pip = _make_pipeline(
            tensor_parallel_size=2, pipeline_parallel_size=4, context_parallel_size=2, sequence_parallel=True
        )
        result = pip.setup_model_from_checkpoint("/fake/checkpoint")

        mock_provider_instance.finalize.assert_called_once()
        mock_provider_instance.initialize_model_parallel.assert_called_once_with(seed=0)
        mock_load.assert_called_once()
        assert result is fake_model

    def test_unwraps_list_model(self, monkeypatch):
        inner_model = MagicMock(spec=[])  # no 'module' attr
        mock_load = MagicMock(return_value=[inner_model])

        monkeypatch.setattr(f"{PIPELINE_MODULE}.WanModelProvider", MagicMock())
        monkeypatch.setattr(f"{PIPELINE_MODULE}._load_megatron_model", mock_load)

        pip = _make_pipeline()
        result = pip.setup_model_from_checkpoint("/fake/checkpoint")
        assert result is inner_model

    def test_unwraps_module_attribute(self, monkeypatch):
        inner_module = MagicMock(spec=[])
        wrapper = MagicMock()
        wrapper.module = inner_module
        mock_load = MagicMock(return_value=wrapper)

        monkeypatch.setattr(f"{PIPELINE_MODULE}.WanModelProvider", MagicMock())
        monkeypatch.setattr(f"{PIPELINE_MODULE}._load_megatron_model", mock_load)

        pip = _make_pipeline()
        result = pip.setup_model_from_checkpoint("/fake/checkpoint")
        assert result is inner_module

    def test_propagates_parallelism_config(self, monkeypatch):
        mock_provider_cls = MagicMock()
        mock_provider_instance = MagicMock()
        mock_provider_cls.return_value = mock_provider_instance

        monkeypatch.setattr(f"{PIPELINE_MODULE}.WanModelProvider", mock_provider_cls)
        monkeypatch.setattr(f"{PIPELINE_MODULE}._load_megatron_model", MagicMock(return_value=MagicMock(spec=[])))

        pip = _make_pipeline(
            tensor_parallel_size=4,
            pipeline_parallel_size=2,
            context_parallel_size=8,
            sequence_parallel=True,
            pipeline_dtype=torch.bfloat16,
        )
        pip.setup_model_from_checkpoint("/fake/ckpt")

        assert mock_provider_instance.tensor_model_parallel_size == 4
        assert mock_provider_instance.pipeline_model_parallel_size == 2
        assert mock_provider_instance.context_parallel_size == 8
        assert mock_provider_instance.sequence_parallel is True
        assert mock_provider_instance.pipeline_dtype == torch.bfloat16


# ===========================================================================
# Tests for generate
# ===========================================================================
class TestGenerate:
    @pytest.fixture
    def pipeline_for_generate(self, monkeypatch):
        """Build a fully-mocked pipeline that can run the generate loop on CPU."""
        from megatron.core import parallel_state

        monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_world_size", lambda: 1, raising=False)

        text_len = 16
        text_hidden = 32

        pip = _make_pipeline(text_len=text_len)

        mock_model = _make_mock_model(hidden_size=16)
        pip.model = mock_model
        pip.inference_cfg = SimpleNamespace(
            num_train_timesteps=1000,
            param_dtype=torch.float32,
            t5_dtype=torch.float32,
            vae_stride=(4, 8, 8),
            patch_size=(1, 2, 2),
            text_len=text_len,
            english_sample_neg_prompt="",
        )
        pip.vae_stride = (4, 8, 8)
        pip.patch_size = (1, 2, 2)

        vae_cfg = SimpleNamespace(z_dim=16, latents_mean=[0.0] * 16, latents_std=[1.0] * 16)
        mock_vae = MagicMock()
        mock_vae.config = vae_cfg
        mock_vae.dtype = torch.float32
        mock_vae.decode.return_value = SimpleNamespace(sample=torch.randn(1, 3, 5, 16, 16))
        pip.vae = mock_vae

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, text_len, dtype=torch.long),
            "attention_mask": torch.cat([torch.ones(1, 4), torch.zeros(1, text_len - 4)], dim=1),
        }
        pip.tokenizer = mock_tokenizer

        mock_text_encoder = MagicMock()
        mock_text_encoder.return_value.last_hidden_state = torch.randn(1, text_len, text_hidden)
        mock_text_encoder.to = MagicMock(return_value=mock_text_encoder)
        mock_text_encoder.cpu = MagicMock(return_value=mock_text_encoder)
        pip.text_encoder = mock_text_encoder

        # Patch schedulers to avoid downloading pretrained weights
        mock_base_sched = MagicMock()
        mock_base_sched.config = {"solver_order": 2}
        mock_sched = MagicMock()
        mock_sched.timesteps = torch.linspace(999, 0, 2)  # 2 steps for speed
        mock_sched.step.return_value = (torch.randn(1, 16, 2, 2, 2),)

        monkeypatch.setattr(
            f"{PIPELINE_MODULE}.FlowMatchEulerDiscreteScheduler.from_pretrained",
            lambda *a, **kw: mock_base_sched,
        )
        monkeypatch.setattr(
            f"{PIPELINE_MODULE}.UniPCMultistepScheduler.from_config",
            lambda *a, **kw: mock_sched,
        )

        # Patch dist.is_initialized / dist.barrier
        monkeypatch.setattr(f"{PIPELINE_MODULE}.dist.is_initialized", lambda: False)

        return pip

    def test_generate_returns_tensor_for_rank0(self, pipeline_for_generate):
        pip = pipeline_for_generate
        result = pip.generate(
            prompts=["a cat"],
            sizes=[(16, 16)],
            frame_nums=[5],
            sampling_steps=2,
            seed=42,
            offload_model=False,
        )
        assert isinstance(result, torch.Tensor)

    def test_generate_returns_none_for_nonzero_rank(self, pipeline_for_generate):
        pip = pipeline_for_generate
        pip.rank = 1
        result = pip.generate(
            prompts=["a cat"],
            sizes=[(16, 16)],
            frame_nums=[5],
            sampling_steps=2,
            seed=42,
            offload_model=False,
        )
        assert result is None

    def test_generate_uses_custom_seed(self, pipeline_for_generate):
        pip = pipeline_for_generate
        pip.generate(
            prompts=["a cat"],
            sizes=[(16, 16)],
            frame_nums=[5],
            sampling_steps=2,
            seed=12345,
            offload_model=False,
        )

    def test_generate_uses_negative_prompt(self, pipeline_for_generate):
        pip = pipeline_for_generate
        pip.generate(
            prompts=["a cat"],
            sizes=[(16, 16)],
            frame_nums=[5],
            sampling_steps=2,
            n_prompt="bad quality",
            seed=0,
            offload_model=False,
        )
        calls = pip.tokenizer.call_args_list
        _ = [c[0][0] if c[0] else c[1].get("text", "") for c in calls]
        assert any("bad quality" in str(c) for c in calls)

    def test_generate_offload_model(self, pipeline_for_generate):
        pip = pipeline_for_generate
        pip.generate(
            prompts=["a dog"],
            sizes=[(16, 16)],
            frame_nums=[5],
            sampling_steps=2,
            seed=0,
            offload_model=True,
        )
        pip.text_encoder.cpu.assert_called()

    def test_generate_batch_of_two(self, pipeline_for_generate):
        pip = pipeline_for_generate
        pip.vae.decode.return_value = SimpleNamespace(sample=torch.randn(2, 3, 5, 16, 16))
        result = pip.generate(
            prompts=["a cat", "a dog"],
            sizes=[(16, 16), (16, 16)],
            frame_nums=[5, 5],
            sampling_steps=2,
            seed=42,
            offload_model=False,
        )
        assert isinstance(result, torch.Tensor)
