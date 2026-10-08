# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Tiny published-layout checkpoint round trips and independent text-model parity."""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from megatron.core import dist_checkpointing
from safetensors.torch import save_file
from transformers import InklingForCausalLM, InklingTextConfig

from megatron.bridge import AutoBridge
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.training.checkpointing import apply_peft_adapter_filter_to_state_dict


logger = logging.getLogger(__name__)


def tiny_config() -> dict:
    raw = json.loads(
        (Path(__file__).resolve().parents[4] / "unit_tests/models/inkling/published_config.json").read_text()
    )
    raw["text_config"].update(
        num_hidden_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=16,
        dense_intermediate_size=128,
        intermediate_size=64,
        n_routed_experts=4,
        num_experts_per_tok=2,
        vocab_size=128,
        unpadded_vocab_size=120,
        model_max_length=128,
        dense_mlp_idx=1,
        d_rel=8,
        rel_extent=16,
        sliding_window_size=8,
        local_layer_ids=[0],
        log_scaling_n_floor=8,
        torch_dtype="float32",
    )
    return raw


def published_weights(model: InklingForCausalLM) -> dict[str, torch.Tensor]:
    """Encode the real checkpoint schema independently of Bridge's mappings."""
    state = model.state_dict()
    raw = {
        "model.llm.embed.weight": state["model.embed_tokens.weight"],
        "model.llm.embed_norm.weight": state["model.embed_norm.weight"],
        "model.llm.norm.weight": state["model.norm.weight"],
        "model.llm.unembed.weight": state["lm_head.weight"],
    }
    for i in range(model.config.num_hidden_layers):
        src, dst = f"model.layers.{i}", f"model.llm.layers.{i}"
        for hf, checkpoint in [
            ("input_layernorm.weight", "attn_norm.weight"),
            ("post_attention_layernorm.weight", "mlp_norm.weight"),
            ("attn_sconv.conv1d.weight", "attn_sconv.weight"),
            ("mlp_sconv.conv1d.weight", "mlp_sconv.weight"),
            ("self_attn.k_sconv.conv1d.weight", "attn.k_sconv.weight"),
            ("self_attn.v_sconv.conv1d.weight", "attn.v_sconv.weight"),
            ("self_attn.q_norm.weight", "attn.q_norm.weight"),
            ("self_attn.k_norm.weight", "attn.k_norm.weight"),
            ("self_attn.rel_logits_proj.proj", "attn.rel_logits_proj.proj"),
            *[
                (f"self_attn.{hf}_proj.weight", f"attn.{ckpt}.weight")
                for hf, ckpt in [("q", "wq_du"), ("k", "wk_dv"), ("v", "wv_dv"), ("r", "wr_du"), ("o", "wo_ud")]
            ],
        ]:
            raw[f"{dst}.{checkpoint}"] = state[f"{src}.{hf}"]
        if model.config.mlp_layer_types[i] == "dense":
            gate, up = state[f"{src}.mlp.gate_proj.weight"], state[f"{src}.mlp.up_proj.weight"]
            raw[f"{dst}.mlp.w13_dn.weight"] = torch.stack((gate, up), dim=1).flatten(0, 1)
            raw[f"{dst}.mlp.w2_md.weight"] = state[f"{src}.mlp.down_proj.weight"]
            raw[f"{dst}.mlp.global_scale"] = state[f"{src}.mlp.global_scale"]
        else:
            for name in ("weight", "global_scale"):
                raw[f"{dst}.mlp.gate.{name}"] = state[f"{src}.mlp.gate.{name}"]
            raw[f"{dst}.mlp.gate.bias"] = state[f"{src}.mlp.gate.e_score_correction_bias"]
            gate, up = state[f"{src}.mlp.experts.gate_up_proj"].chunk(2, dim=1)
            raw[f"{dst}.mlp.experts.w13_weight"] = torch.stack((gate, up), dim=2).flatten(1, 2)
            raw[f"{dst}.mlp.experts.w2_weight"] = state[f"{src}.mlp.experts.down_proj"]
            gate, up = state[f"{src}.mlp.shared_experts.gate_proj"], state[f"{src}.mlp.shared_experts.up_proj"]
            raw[f"{dst}.mlp.shared_experts.shared_w13_weight"] = torch.stack((gate, up), dim=2).flatten(1, 2)
            raw[f"{dst}.mlp.shared_experts.shared_w2_weight"] = state[f"{src}.mlp.shared_experts.down_proj"]
    return {name: tensor.contiguous().cpu() for name, tensor in raw.items()}


def run_parity(checkpoint: Path, tp: int, ep: int) -> None:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group("nccl")
    torch.manual_seed(41)
    raw = tiny_config()
    # The raw checkpoint's intermediate_size is the expert width. Transformers
    # 5.16 replaces that field with dense_intermediate_size during normalization.
    config = InklingTextConfig(**raw["text_config"], moe_intermediate_size=raw["text_config"]["intermediate_size"])
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    reference = InklingForCausalLM(config)
    original = published_weights(reference)
    if torch.distributed.get_rank() == 0:
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "config.json").write_text(json.dumps(raw))
        save_file(original, checkpoint / "model.safetensors")
    torch.distributed.barrier()
    bridge = AutoBridge.from_hf_pretrained(checkpoint, trust_remote_code=False, torch_dtype=torch.float32)
    provider = bridge.to_megatron_provider()
    provider.tensor_model_parallel_size = tp
    provider.expert_model_parallel_size = ep
    provider.expert_tensor_parallel_size = 1
    provider.sequence_parallel = tp > 1
    provider.parallel_output = False
    provider.attention_backend = "fused"
    provider.params_dtype = torch.float32
    provider.autocast_dtype = torch.float32
    provider.bf16 = False
    provider.gradient_accumulation_fusion = False
    provider.recompute_granularity = "full"
    provider.recompute_method = "uniform"
    provider.recompute_num_layers = 1
    provider.finalize()
    models = provider.provide_distributed_model(wrap_with_ddp=False)
    exported = dict(bridge.export_hf_weights(models, cpu=True))
    if torch.distributed.get_rank() == 0:
        assert exported.keys() == original.keys()
        for name, tensor in original.items():
            torch.testing.assert_close(exported[name], tensor, atol=0, rtol=0)
        logger.info("Exact roundtrip: %d/%d tensors", len(original), len(original))
    reference = reference.cuda()
    torch.manual_seed(42)
    tokens = torch.randint(0, 120, (2, 32), device="cuda")
    positions = torch.arange(32, device="cuda").expand_as(tokens)
    valid = torch.ones_like(tokens, dtype=torch.bool)
    valid[1, 24:] = False
    actual = models[0](tokens, positions, ~valid[:, None, None, :])
    expected = reference(tokens, attention_mask=valid, use_cache=False).logits
    torch.testing.assert_close(actual[valid], expected[valid], atol=2e-5, rtol=2e-3)
    assert torch.equal(actual[valid].argmax(-1), expected[valid].argmax(-1))
    native_loss = actual[valid].float().square().mean()
    native_loss.backward()
    expected[valid].square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in models[0].parameters() if p.grad is not None)
    if tp == ep == 1:
        model = models[0]
        torch.testing.assert_close(
            model.embedding.word_embeddings.weight.grad, reference.model.embed_tokens.weight.grad, atol=2e-6, rtol=2e-3
        )
        torch.testing.assert_close(
            model.decoder.layers[1].self_attention.linear_r.weight.grad,
            reference.model.layers[1].self_attn.r_proj.weight.grad,
            atol=2e-6,
            rtol=2e-3,
        )
    logger.info("Forward/backward parity: TP=%d, EP=%d, loss=%s", tp, ep, native_loss.item())
    native_checkpoint = checkpoint / "native"
    native_checkpoint.mkdir(exist_ok=True)
    dist_checkpointing.save({"model": models[0].sharded_state_dict()}, str(native_checkpoint))
    with torch.no_grad():
        for parameter in models[0].parameters():
            parameter.zero_()
    restored_state = dist_checkpointing.load({"model": models[0].sharded_state_dict()}, str(native_checkpoint))
    models[0].load_state_dict(restored_state["model"])
    with torch.no_grad():
        restored_logits = models[0](tokens, positions, ~valid[:, None, None, :])
        torch.testing.assert_close(restored_logits[valid], actual[valid], atol=0, rtol=0)
    logger.info("Native base checkpoint restore passed: TP=%d, EP=%d", tp, ep)
    lora = LoRA(target_modules=["linear_qkv", "linear_r", "linear_proj"], dim=4, alpha=4, lora_dtype=torch.float32)
    models = lora(models, training=True)
    lora.set_params_to_save(models)
    if tp == ep == 1:
        optimizer = torch.optim.SGD([p for p in models[0].parameters() if p.requires_grad], lr=0.1)
        optimizer.zero_grad(set_to_none=True)
        logits = models[0](tokens, positions, ~valid[:, None, None, :])
        torch.nn.functional.cross_entropy(logits[valid].float(), tokens[valid]).backward()
        optimizer.step()
    else:
        # Nonzero and nonuniform B tests distributed adapter gathering/splitting.
        with torch.no_grad():
            for name, parameter in models[0].named_parameters():
                if ".adapter.linear_out.weight" in name:
                    parameter.copy_(
                        torch.arange(parameter.numel(), device=parameter.device).reshape_as(parameter) * 1e-4
                    )
    adapters = dict(bridge.export_adapter_weights(models, cpu=True, show_progress=False))
    projection_names = {"wq_du": "q_proj", "wk_dv": "k_proj", "wv_dv": "v_proj", "wr_du": "r_proj", "wo_ud": "o_proj"}
    expected_names = {
        f"model.llm.layers.{i}.attn.{name}.lora_{side}.weight"
        for i in range(2)
        for name in projection_names
        for side in ("A", "B")
    }
    assert adapters.keys() == expected_names, {
        "missing": sorted(expected_names - adapters.keys()),
        "extra": sorted(adapters.keys() - expected_names),
    }
    assert all(tensor.count_nonzero() for name, tensor in adapters.items() if ".lora_B." in name)
    with torch.no_grad():
        for i in range(2):
            for checkpoint_name, hf_name in projection_names.items():
                prefix = f"model.llm.layers.{i}.attn.{checkpoint_name}"
                delta = adapters[f"{prefix}.lora_B.weight"].cuda() @ adapters[f"{prefix}.lora_A.weight"].cuda()
                getattr(reference.model.layers[i].self_attn, hf_name).weight.add_(delta)
        updated = models[0](tokens, positions, ~valid[:, None, None, :])
        restored = reference(tokens, attention_mask=valid, use_cache=False).logits
        torch.testing.assert_close(updated[valid], restored[valid], atol=2e-5, rtol=2e-3)
    logger.info("Native LoRA export parity: %d/%d tensors", len(adapters), len(expected_names))
    merged = dict(bridge.export_hf_weights(models, cpu=True))
    for name, tensor in published_weights(reference).items():
        torch.testing.assert_close(merged[name], tensor.cpu(), atol=2e-7, rtol=2e-5)
    native_checkpoint = checkpoint / "adapter"
    native_checkpoint.mkdir(exist_ok=True)
    state = apply_peft_adapter_filter_to_state_dict({"model": models[0].sharded_state_dict()}, lora)
    dist_checkpointing.save(state, str(native_checkpoint))
    with torch.no_grad():
        for parameter in models[0].parameters():
            if parameter.requires_grad:
                parameter.zero_()
    state = apply_peft_adapter_filter_to_state_dict({"model": models[0].sharded_state_dict()}, lora)
    restored_state = dist_checkpointing.load(state, str(native_checkpoint))
    _, unexpected = models[0].load_state_dict(restored_state["model"], strict=False)
    assert not unexpected
    restored_adapters = dict(bridge.export_adapter_weights(models, cpu=True, show_progress=False))
    assert restored_adapters.keys() == adapters.keys()
    for name in adapters:
        torch.testing.assert_close(restored_adapters[name], adapters[name], atol=0, rtol=0)
    with torch.no_grad():
        restored_logits = models[0](tokens, positions, ~valid[:, None, None, :])
        torch.testing.assert_close(restored_logits[valid], updated[valid], atol=0, rtol=0)
    logger.info("Merged export and native checkpoint restore passed: TP=%d, EP=%d", tp, ep)
    torch.distributed.destroy_process_group()


@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize("tp,ep", [(1, 1), (2, 1), (1, 2)])
def test_inkling_raw_checkpoint_parity(tmp_path, tp, ep):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={max(tp, ep)}",
            __file__,
            "--checkpoint",
            str(tmp_path / "inkling"),
            "--tp",
            str(tp),
            "--ep",
            str(ep),
        ],
        check=True,
        timeout=600,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    args = parser.parse_args()
    run_parity(args.checkpoint, args.tp, args.ep)
