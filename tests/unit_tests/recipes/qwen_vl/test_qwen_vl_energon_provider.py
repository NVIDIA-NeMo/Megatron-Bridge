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

#
# Test purpose:
# - Verify QwenVLEnergonProvider.finalize() propagates dataset-config knobs
#   (seq_length, seq_len, min_pixels, max_pixels, max_num_images, max_num_frames,
#   max_visual_tokens) onto the eagerly-built task encoder. finalize() is called by
#   ConfigContainer.validate() after CLI/YAML overrides are merged, which is what makes
#   these fields overridable.
#

from megatron.bridge.recipes.qwen_vl.qwen3_vl import QwenVLEnergonProvider


class _FakeTaskEncoder:
    """Minimal stand-in for QwenVLTaskEncoder — only carries the synced fields."""

    def __init__(self):
        # Initialize to sentinel values that differ from the provider defaults so we
        # can confirm the assignments happened.
        self.seq_len = -1
        self.seq_length = -1
        self.min_pixels = -1
        self.max_pixels = -1
        self.max_num_images = -1
        self.max_num_frames = -1
        self.max_visual_tokens = -1


def _make_provider(task_encoder, **overrides):
    """Construct a QwenVLEnergonProvider with safe defaults for unit tests."""
    defaults = dict(
        path="/tmp/fake-energon-path",
        seq_length=4096,
        micro_batch_size=1,
        global_batch_size=1,
        num_workers=0,
        task_encoder=task_encoder,
    )
    defaults.update(overrides)
    return QwenVLEnergonProvider(**defaults)


def test_finalize_syncs_all_fields_to_task_encoder():
    encoder = _FakeTaskEncoder()
    provider = _make_provider(
        encoder,
        seq_length=2048,
        min_pixels=12345,
        max_pixels=67890,
        max_num_images=4,
        max_num_frames=16,
        max_visual_tokens=999,
    )

    provider.finalize()

    # seq_length / seq_len and the Qwen-VL-specific knobs are all synced by finalize().
    assert encoder.seq_len == 2048
    assert encoder.seq_length == 2048
    assert encoder.min_pixels == 12345
    assert encoder.max_pixels == 67890
    assert encoder.max_num_images == 4
    assert encoder.max_num_frames == 16
    assert encoder.max_visual_tokens == 999


def test_finalize_no_op_when_task_encoder_is_none():
    provider = _make_provider(task_encoder=None)
    # Should not raise even though task_encoder is None.
    provider.finalize()


def test_provider_default_field_values():
    """Defaults should match the documented per-sample budget."""
    provider = _make_provider(task_encoder=_FakeTaskEncoder())
    assert provider.min_pixels == 200704
    assert provider.max_pixels == 1003520
    assert provider.max_num_images == 10
    assert provider.max_num_frames == 60
    assert provider.max_visual_tokens == 16384


def test_provider_accepts_none_for_unbounded_limits():
    """None should be accepted for the optional limit fields (disables the budget)."""
    provider = _make_provider(
        task_encoder=_FakeTaskEncoder(),
        max_num_images=None,
        max_num_frames=None,
        max_visual_tokens=None,
    )
    assert provider.max_num_images is None
    assert provider.max_num_frames is None
    assert provider.max_visual_tokens is None


def test_finalize_propagates_none_limits():
    encoder = _FakeTaskEncoder()
    provider = _make_provider(
        encoder,
        max_num_images=None,
        max_num_frames=None,
        max_visual_tokens=None,
    )

    provider.finalize()

    assert encoder.max_num_images is None
    assert encoder.max_num_frames is None
    assert encoder.max_visual_tokens is None
