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

"""HF ↔ Megatron conversion for Shensi against a tiny randomly initialised checkpoint."""

import pytest
import torch

from megatron.bridge import AutoBridge

from . import make_tiny_hf_model


def _tiny_hf_dir(tmp_path, name: str = "shensi_tiny"):
    model = make_tiny_hf_model()
    out = tmp_path / name
    model.save_pretrained(out, safe_serialization=False)
    return out, model


def _tiny_provider(hf_dir, tp: int = 1, pp: int = 1):
    bridge = AutoBridge.from_hf_pretrained(hf_dir)
    provider = bridge.to_megatron_provider()
    provider.tensor_model_parallel_size = tp
    provider.pipeline_model_parallel_size = pp
    # No fused DSA kernels in the test environment: the PyTorch fallback path is used.
    provider.dsa_kernel_backend = "none"
    provider.finalize()
    return bridge, provider


@pytest.mark.parametrize("tp,pp", [(1, 1)])
def test_hf_to_megatron_loads_every_parameter(tmp_path, tp, pp):
    hf_dir, hf_model = _tiny_hf_dir(tmp_path)

    bridge, provider = _tiny_provider(hf_dir, tp, pp)
    model = provider.provide_distributed_model(wrap_with_ddp=False)

    hf_state = hf_model.state_dict()
    # load_hf_weights raises on unmapped parameters, so reaching the next line means all
    bridge.load_hf_weights(model)

    megatron_state = dict(model[0].named_parameters())
    assert megatron_state, "no Megatron parameters were loaded"

    # Spot-check shapes by suffix so the assertions do not depend on layer prefixes.
    def param_with_suffix(suffix: str):
        hits = [(n, p) for n, p in megatron_state.items() if n.endswith(suffix)]
        assert hits, f"no parameter ends with {suffix}; have {list(megatron_state)[:5]}"
        return hits[0][1]

    assert tuple(param_with_suffix("word_embeddings.weight").shape) == tuple(
        hf_state["model.embed_tokens.weight"].shape
    )
    assert len(megatron_state) > 50, f"too few parameters ({len(megatron_state)})"


def test_megatron_to_hf_export_round_trips_shapes_and_dtypes(tmp_path):
    hf_dir, hf_model = _tiny_hf_dir(tmp_path)
    hf_state = hf_model.state_dict()

    bridge, provider = _tiny_provider(hf_dir)
    model = provider.provide_distributed_model(wrap_with_ddp=False)
    bridge.load_hf_weights(model)

    seen = 0
    for name, tensor in bridge.export_hf_weights(model, cpu=True):
        hf_name = name if name in hf_state else f"model.{name}"
        assert hf_name in hf_state, f"exported name {name} is missing from the HF state dict"
        reference = hf_state[hf_name]
        assert tuple(tensor.shape) == tuple(reference.shape), hf_name
        assert tensor.dtype in (reference.dtype, torch.float32), hf_name
        seen += 1

    assert seen > 0, "no tensors were exported"
