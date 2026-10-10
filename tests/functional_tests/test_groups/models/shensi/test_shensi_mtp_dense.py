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

"""Dense-only MTP depths build, run forward and receive gradients without Transformer Engine.

The latent-MoE MTP test needs TE, but the MTP container, the stream flow and the inner-layer
contract do not. The last decoder layer is indexed attention, so the MTP depth either runs a full
indexer or (when the last layer reuses a preceding full indexer) dense causal attention.
"""

import pytest
import torch
from megatron.core.extensions.transformer_engine import HAVE_TE
from safetensors.torch import load_file, save_file

from megatron.bridge import AutoBridge
from megatron.bridge.models.shensi.modeling_shensi import ShensiTransformerLayer

from . import TINY_SHENSI_CONFIG


if not HAVE_TE:
    # MCore's inference-linear type check is not TE-less-safe; this model has no inference layers,
    # and the check only enables a symmetric-memory barrier for them.
    import megatron.core.transformer.multi_token_prediction as _mtp

    _mtp.is_inference_column_parallel_linear = lambda module: False


def _dense_mtp_checkpoint(tmp_path, *, indexer_types, last_linear: bool):
    from transformers.models.shensi.configuration_shensi import ShensiConfig
    from transformers.models.shensi.modeling_shensi import ShensiForCausalLM

    cfg_dict = dict(TINY_SHENSI_CONFIG)
    cfg_dict.update(
        num_hidden_layers=4,
        mlp_layer_types=["dense"] * 4,
        first_k_dense_replace=4,
        attn_res_block_layer_types=["block_write_layer", "block_read_layer"] * 2,
        attn_res_block_size=2,
        layer_types=["indexed_attention", "linear_attention"] * 2,
        rosa_layer_ids=[],
        indexer_types=indexer_types,
    )
    if last_linear:
        cfg_dict["layer_types"] = ["indexed_attention", "indexed_attention", "indexed_attention", "linear_attention"]
        cfg_dict["attn_res_block_layer_types"] = [
            "block_write_layer",
            "block_read_layer",
            "block_write_layer",
            "block_read_layer",
        ]
    else:
        # All-indexed: the last layer's `indexer_types` entry selects between a full indexer and a
        # shared one, which is the dense-causal MTP fallback.
        cfg_dict["layer_types"] = ["indexed_attention"] * 4
        cfg_dict["attn_res_block_layer_types"] = ["block_write_layer"] * 4
        cfg_dict["attn_res_block_size"] = 1
    cfg = ShensiConfig(**cfg_dict)
    torch.manual_seed(0)
    hf_model = ShensiForCausalLM(cfg).eval().to("cuda")
    hf_model.save_pretrained(tmp_path, safe_serialization=False)
    # The reference dropped MTP from its config surface; the checkpoint convention keeps a
    # `mtp.{depth}` subtree, so record the depth in config.json for the bridge to map it.
    import json

    config_path = tmp_path / "config.json"
    config_data = json.loads(config_path.read_text())
    config_data["num_mtp_layers"] = 1
    config_path.write_text(json.dumps(config_data))

    state = dict(load_file(tmp_path / "model.safetensors"))
    # Use the in-memory state so tied duplicates (DeepEmbed projections, routers) are present
    # under the last layer as well; the MTP depth mirrors that layer.
    full_state = {key: tensor.detach().float().cpu() for key, tensor in hf_model.state_dict().items()}
    source = f"model.layers.{cfg.num_hidden_layers - 1}."
    inner = {key[len(source) :]: tensor.clone() for key, tensor in full_state.items() if key.startswith(source)}
    hidden = cfg.hidden_size
    generator = torch.Generator().manual_seed(1)
    state["mtp.0.enorm.weight"] = torch.randn(hidden, generator=generator) * 0.02
    state["mtp.0.hnorm.weight"] = torch.randn(hidden, generator=generator) * 0.02
    state["mtp.0.norm.weight"] = torch.randn(hidden, generator=generator) * 0.02
    state["mtp.0.eh_proj.weight"] = torch.randn(hidden, 2 * hidden, generator=generator) * 0.02
    for suffix, tensor in inner.items():
        state["mtp.0." + suffix] = tensor
    save_file(state, tmp_path / "model.safetensors")
    return cfg


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the Megatron model is built on CUDA")
@pytest.mark.parametrize(
    "indexer_types,last_linear",
    [
        (["full", "full", "full", "full"], False),
        (["full", "shared", "full", "shared"], False),
        (["full", "full", "full", "full"], True),
    ],
)
def test_dense_mtp_forward_and_backward(tmp_path, indexer_types, last_linear: bool) -> None:
    if last_linear:
        pytest.importorskip("fla")

    cfg = _dense_mtp_checkpoint(tmp_path, indexer_types=indexer_types, last_linear=last_linear)
    bridge = AutoBridge.from_hf_pretrained(tmp_path)
    provider = bridge.to_megatron_provider()
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.dsa_kernel_backend = "none"
    provider.gradient_accumulation_fusion = False
    provider.finalize()
    model = provider.provide_distributed_model(wrap_with_ddp=False)
    bridge.load_hf_weights(model)

    unwrapped = model[0].module if hasattr(model[0], "module") else model[0]
    inner = unwrapped.mtp.layers[0].mtp_model_layer.layers[0]
    assert isinstance(inner, ShensiTransformerLayer)
    assert inner.is_write == unwrapped.decoder.layers[-1].is_write
    assert inner.is_dense == unwrapped.decoder.layers[-1].is_dense
    attention = inner.self_attention
    if inner.is_write:
        assert attention.mtp_dense_attention == (indexer_types[-1] == "shared")

    tokens = torch.randint(0, cfg.vocab_size, (1, 8), device="cuda")
    position_ids = torch.arange(8, device="cuda").unsqueeze(0)
    attention_mask = torch.ones((1, 8), dtype=torch.bool, device="cuda")
    logits = model[0](tokens, position_ids, attention_mask=attention_mask)
    output = logits[0] if isinstance(logits, tuple) else logits
    assert torch.isfinite(output).all()
    output.float().sum().backward()

    grads = [param.grad for param in unwrapped.mtp.layers[0].parameters() if param.grad is not None]
    assert grads, "the MTP depth received no gradient"
    assert all(torch.isfinite(grad).all() for grad in grads)
