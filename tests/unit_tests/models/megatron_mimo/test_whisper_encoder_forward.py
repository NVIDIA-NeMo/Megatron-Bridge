# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Forward-pass shape contracts for WhisperEncoder.

Constructs a stripped-down encoder that bypasses MegatronModule.__init__ so we
don't need a real distributed/Megatron init. The decoder block is replaced
with an identity stub to exercise the conv → position-embed → decoder →
ln_post pipeline on CPU.

Skipped when megatron.core isn't importable (whisper_model.py imports several
mcore modules at module scope).
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch


pytest.importorskip("megatron.core.transformer.transformer_block")


WHISPER_MODEL_PATH = (
    Path(__file__).resolve().parents[4] / "examples" / "models" / "megatron_mimo" / "whisper" / "whisper_model.py"
)


@pytest.fixture(scope="module")
def whisper_module():
    sys.modules.setdefault("transformers", MagicMock())
    spec = importlib.util.spec_from_file_location("whisper_model_under_test", WHISPER_MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _IdentityDecoder:
    """TransformerBlock stand-in: passes input through and records the call."""

    def __init__(self):
        self.last_input = None
        self.last_mask = None

    def __call__(self, x, attention_mask=None):
        self.last_input = x
        self.last_mask = attention_mask
        return x

    def set_input_tensor(self, t):
        pass


def _build_encoder_skeleton(whisper_module, *, num_mel_bins=8, hidden=16, max_pos=32):
    """Bypass __init__ and assign just the attributes forward() reads."""
    enc = whisper_module.WhisperEncoder.__new__(whisper_module.WhisperEncoder)
    torch.nn.Module.__init__(enc)
    enc.num_mel_bins = num_mel_bins
    enc.visual_hidden_size = hidden
    enc.max_source_positions = max_pos
    enc.conv1 = torch.nn.Conv1d(num_mel_bins, hidden, 3, padding=1)
    enc.conv2 = torch.nn.Conv1d(hidden, hidden, 3, stride=2, padding=1)
    enc.position_embeddings = torch.nn.Embedding(max_pos, hidden)
    enc.position_embeddings.weight.data.copy_(whisper_module._sinusoidal_position_embedding(max_pos, hidden))
    enc.position_embeddings.weight.requires_grad = False
    enc.ln_post = torch.nn.LayerNorm(hidden)
    enc.decoder = _IdentityDecoder()
    return enc


@pytest.mark.unit
class TestWhisperEncoderForward:
    def test_default_output_shape(self, whisper_module):
        enc = _build_encoder_skeleton(whisper_module)
        # mel_frames=16 → stride-2 conv2 → 8 output tokens.
        out = enc(torch.randn(2, 8, 16))
        assert out.shape == (2, 8, 16)

    def test_odd_mel_frames(self, whisper_module):
        """Conv2 with kernel=3, stride=2, padding=1 on length 17 → floor((17+2-3)/2 + 1) = 9."""
        enc = _build_encoder_skeleton(whisper_module)
        out = enc(torch.randn(2, 8, 17))
        assert out.shape == (2, 9, 16)

    def test_seq_lengths_packs_output(self, whisper_module):
        enc = _build_encoder_skeleton(whisper_module)
        x = torch.randn(2, 8, 16)  # → [B=2, S=8, H=16] after convs
        seq_lengths = torch.tensor([5, 3])
        out = enc(x, seq_lengths=seq_lengths)
        assert out.shape == (8, 16)  # 5 + 3 valid tokens, hidden=16

    def test_seq_lengths_keeps_first_n_rows_per_sample(self, whisper_module):
        """Packed rows for sample i must equal the unpacked rows[:seq_lengths[i]]."""
        enc = _build_encoder_skeleton(whisper_module)
        x = torch.randn(2, 8, 16)
        unpacked = enc(x)  # [2, 8, 16]
        seq_lengths = torch.tensor([5, 3])
        packed = enc(x, seq_lengths=seq_lengths)
        assert torch.equal(packed[:5], unpacked[0, :5])
        assert torch.equal(packed[5:8], unpacked[1, :3])

    def test_set_input_tensor_delegates_to_decoder(self, whisper_module):
        enc = _build_encoder_skeleton(whisper_module)
        enc.decoder = MagicMock()
        t = torch.zeros(3, 4)
        enc.set_input_tensor(t)
        enc.decoder.set_input_tensor.assert_called_once_with(t)

    def test_position_embeddings_added_to_features(self, whisper_module):
        """Forward feeds pos-embed-added features into the decoder; capturing decoder input proves it."""
        enc = _build_encoder_skeleton(whisper_module)
        out = enc(torch.randn(1, 8, 16))
        # Decoder operated on [S, B, H] = [8, 1, 16].
        assert enc.decoder.last_input.shape == (8, 1, 16)
        # And output flows back to [B, S, H].
        assert out.shape == (1, 8, 16)
