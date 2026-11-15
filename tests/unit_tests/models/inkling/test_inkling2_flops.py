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

"""Unit tests for the Inkling2 training FLOPs estimate."""

from types import SimpleNamespace

import pytest

from megatron.bridge.models.inkling import Inkling2Bridge, Inkling2Config
from megatron.bridge.training.utils.flop_utils import num_floating_point_operations


pytestmark = pytest.mark.unit

_TEXT = {
    "hidden_size": 64,
    "num_hidden_layers": 6,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 512,
    "unpadded_vocab_size": 500,
    "d_rel": 8,
    "rel_extent": 32,
    "log_scaling_n_floor": 8192,
    "log_scaling_alpha": 0.1,
    "rms_norm_eps": 1e-6,
    "pre_norm_pre_affine": False,
    "final_norm_pre_affine": False,
    "use_value_norm_no_gamma": True,
    "qk_l2_clip_threshold": 22.627416997969522,
    "rel_l2_clip_threshold": 8.0,
    "local_layer_ids": [0, 1, 2, 3, 4],
    "sconv_kernel_size": 4,
    "final_logit_softcapping": 30.0,
    "sliding_window_size": 16,
    "n_layers_per_attn_res_block": 6,
    "attn_res_normalize_values": True,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "n_shared_experts": 0,
    "intermediate_size": 32,
    "mlp_gate_softcap": 4.0,
    "mlp_up_softcap": 25.0,
    "route_scale": 8.0,
    "use_gate_bias": True,
    "torch_dtype": "bfloat16",
}


def _model_config(**text_overrides):
    hf_config = Inkling2Config(text_config={**_TEXT, **text_overrides}, vision_config={})
    bridge = Inkling2Bridge()
    bridge.hf_config = hf_config
    config = bridge.hf_config_to_model_config(hf_config)
    config.seq_length = 64
    return config


def _expected_flops(*, tokens: int, mean_length: float) -> float:
    """Hand-expanded estimate for ``_TEXT``: five sliding-window layers and one full-attention layer."""
    hidden, heads, kv_heads, head_dim, rel_dim, window, rel_extent, kernel = 64, 4, 2, 16, 8, 16, 32, 4
    query_width, kv_width = heads * head_dim, kv_heads * head_dim
    projections = hidden * (2 * query_width + 2 * kv_width + heads * rel_dim + heads) + query_width * hidden
    relative_logits = heads * rel_dim * (5 * window + 1 * rel_extent)
    dynamic_convolution = 2 * hidden * kv_heads * kernel
    local_context = window - window * (window - 1) / (2 * mean_length) if window < mean_length else mean_length / 2
    core_attention = 2 * query_width * (1 * mean_length / 2 + 5 * local_context)
    experts = 6 * (hidden * 8 + 3 * hidden * 32 * 2)
    logits = hidden * 512
    per_token = 6 * projections + relative_logits + dynamic_convolution + core_attention + experts + logits
    return 6 * tokens * per_token


def test_full_length_flops_match_hand_expansion():
    config = _model_config()

    flops = config._get_num_floating_point_operations_with_runtime_stats(
        batch_size=2, seqlen_sum=None, seqlen_squared_sum=None
    )

    assert flops == pytest.approx(_expected_flops(tokens=2 * 64, mean_length=64))


def test_packed_subsequences_use_their_lengths():
    config = _model_config()

    # Two packs of four 16-token documents: each document fits within the 16-token window.
    flops = config._get_num_floating_point_operations_with_runtime_stats(
        batch_size=2, seqlen_sum=8 * 16, seqlen_squared_sum=8 * 16**2
    )

    assert flops == pytest.approx(_expected_flops(tokens=8 * 16, mean_length=16))
    full_length = config._get_num_floating_point_operations_with_runtime_stats(
        batch_size=2, seqlen_sum=None, seqlen_squared_sum=None
    )
    assert flops < full_length


def test_sliding_window_layers_cost_less_than_full_attention():
    windowed = _model_config()
    all_full = _model_config(local_layer_ids=[])
    kwargs = {"batch_size": 1, "seqlen_sum": 4096, "seqlen_squared_sum": 4096**2}

    assert windowed._get_num_floating_point_operations_with_runtime_stats(
        **kwargs
    ) < all_full._get_num_floating_point_operations_with_runtime_stats(**kwargs)


def test_partial_sequence_statistics_are_rejected():
    config = _model_config()

    with pytest.raises(ValueError, match="both sequence-length sums"):
        config._get_num_floating_point_operations_with_runtime_stats(
            batch_size=1, seqlen_sum=64, seqlen_squared_sum=None
        )


def test_training_flops_dispatch_to_the_inkling2_estimate():
    config = _model_config()
    cfg = SimpleNamespace(model=config, peft=None)

    assert num_floating_point_operations(cfg, batch_size=2) == pytest.approx(
        _expected_flops(tokens=2 * 64, mean_length=64)
    )
