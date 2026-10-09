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

"""Tiny MiMo-V2.6 text conversion, numerical parity and native checkpoint tests."""

import argparse
import hashlib
import inspect
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from megatron.core import dist_checkpointing
from megatron.core.utils import unwrap_model
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM

from megatron.bridge import AutoBridge
from megatron.bridge.peft.lora import LoRA


logger = logging.getLogger(__name__)
HF_MODEL = "XiaomiMiMo/MiMo-V2.6-Flash-RL"
HF_REVISION = "5711b268169967567844e1e560e8a3966da959b1"


def checkpoint_weights(model, checkpoint_tp_size=4):
    """Encode published TP-chunked QKV rows independently of Bridge's mappings."""
    result = {}
    config = model.config
    for name, tensor in model.state_dict().items():
        if name.endswith(".self_attn.qkv_proj.weight"):
            layer = int(name.split(".")[2])
            groups = (
                config.swa_num_key_value_heads if config.hybrid_layer_pattern[layer] else config.num_key_value_heads
            )
            sizes = (
                config.num_attention_heads * config.head_dim,
                groups * config.head_dim,
                groups * config.v_head_dim,
            )
            projections = tensor.split(sizes)
            tensor = torch.cat(
                [part for shard in zip(*(p.chunk(checkpoint_tp_size) for p in projections)) for part in shard]
            )
        result[name] = tensor.detach().contiguous().cpu()
    return result


def create_tiny_checkpoint(checkpoint: Path, hf_source: str, dtype: torch.dtype) -> dict:
    """Use unchanged official HF classes; keep global/SWA and dense/MoE layers."""
    config = AutoConfig.from_pretrained(hf_source, revision=HF_REVISION, trust_remote_code=True)
    overrides = dict(
        num_hidden_layers=2,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=8,
        swa_num_attention_heads=8,
        num_key_value_heads=4,
        swa_num_key_value_heads=8,
        head_dim=192,
        swa_head_dim=192,
        v_head_dim=128,
        swa_v_head_dim=128,
        vocab_size=128,
        max_position_embeddings=256,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        hybrid_layer_pattern=[0, 1],
        moe_layer_freq=[0, 1],
        vision_config={},
        audio_config={},
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=False,
    )
    for name, value in overrides.items():
        if not hasattr(config, name):
            raise ValueError(f"Official MiMo config has no field {name}")
        setattr(config, name, value)
    del config.quantization_config
    config.dtype = dtype
    config._attn_implementation = "eager"
    torch.manual_seed(41)
    reference = (
        AutoModelForCausalLM.from_config(config, trust_remote_code=True, code_revision=HF_REVISION).to(dtype).eval()
    )
    # These custom parameters use torch.empty in the official implementation.
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if name.endswith("mlp.gate.weight"):
                parameter.normal_(0, 0.02)
            elif name.endswith("e_score_correction_bias"):
                parameter.copy_(torch.linspace(-0.01, 0.01, parameter.numel()).reshape_as(parameter))
            elif name.endswith("attention_sink_bias"):
                parameter.copy_(torch.linspace(-0.2, 0.2, parameter.numel()).reshape_as(parameter))
    assert all(torch.isfinite(p).all() for p in reference.parameters())
    canonical, raw = checkpoint / "hf", checkpoint / "raw"
    canonical.mkdir(parents=True)
    raw.mkdir()
    reference.save_pretrained(canonical, safe_serialization=True)
    # Preserve official remote code byte-for-byte in both local fixtures.
    source_directory = Path(inspect.getfile(type(reference))).parent
    code_hashes = {}
    for name in ("configuration_mimo_v2.py", "modeling_mimo_v2.py"):
        source = source_directory / name
        code_hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        for destination in (canonical, raw):
            shutil.copyfile(source, destination / name)
    config.save_pretrained(raw)
    weights = checkpoint_weights(reference)
    save_file(weights, raw / "model.safetensors")
    (raw / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"tp_size": 4}, "weight_map": {name: "model.safetensors" for name in weights}})
    )
    receipt = dict(
        hf_model=HF_MODEL,
        revision=HF_REVISION,
        code_sha256=code_hashes,
        parameters=sum(p.numel() for p in reference.parameters()),
        tensors=len(weights),
        dtype=str(dtype),
        checkpoint_tp_size=4,
        config_overrides=overrides,
        reference_attention="official eager forward with explicit full/sliding attention-mask dictionary",
    )
    (checkpoint / "fixture.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def reference_masks(tokens, dtype):
    """Supported HF mask dictionary: causal and 128-token windows including self."""
    positions = torch.arange(tokens.shape[1], device=tokens.device)
    future = positions[None, :] > positions[:, None]
    outside_window = positions[None, :] < positions[:, None] - 127
    return {
        name: torch.zeros(1, 1, tokens.shape[1], tokens.shape[1], dtype=dtype, device=tokens.device).masked_fill(
            blocked, torch.finfo(dtype).min
        )
        for name, blocked in (("full_attention", future), ("sliding_window_attention", future | outside_window))
    }


def compare_logits(actual, expected):
    """Use the repository's cosine/next-token gate; absolute errors are diagnostics."""
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    difference = (actual.float() - expected.float()).abs()
    cosine = F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0).item()
    next_token_matches = torch.equal(actual[:, -1].argmax(-1), expected[:, -1].argmax(-1))
    assert cosine >= 0.99, cosine
    assert next_token_matches
    return dict(
        cosine=cosine,
        max_abs=difference.max().item(),
        mean_abs=difference.mean().item(),
        next_token_matches=next_token_matches,
        all_position_argmax_match_fraction=(actual.argmax(-1) == expected.argmax(-1)).float().mean().item(),
    )


def _exact_weights(actual, expected):
    assert actual.keys() == expected.keys()
    for name in expected:
        # Native router and expert-bias storage is FP32, even for BF16 HF weights.
        torch.testing.assert_close(actual[name].float(), expected[name].float(), atol=0, rtol=0)
    return dict(
        tensors=len(expected),
        dtype_differences={
            name: [str(expected[name].dtype), str(actual[name].dtype)]
            for name in expected
            if actual[name].dtype != expected[name].dtype
        },
    )


def run_parity(checkpoint: Path, tp: int, ep: int) -> None:
    """Run one bounded distributed layout with the prepared local tiny checkpoint."""
    from megatron.bridge.training.checkpointing import apply_peft_adapter_filter_to_state_dict

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group("nccl")
    torch.backends.cuda.matmul.allow_tf32 = False
    raw = checkpoint / "raw"
    reference = AutoModelForCausalLM.from_pretrained(
        checkpoint / "hf", trust_remote_code=True, attn_implementation="eager", dtype="auto"
    )
    dtype = next(reference.parameters()).dtype
    original = load_file(raw / "model.safetensors")
    bridge = AutoBridge.from_hf_pretrained(raw, trust_remote_code=True, torch_dtype=dtype)
    provider = bridge.to_megatron_provider()
    provider.tensor_model_parallel_size = tp
    provider.expert_model_parallel_size = ep
    provider.expert_tensor_parallel_size = 1
    provider.sequence_parallel = tp > 1
    provider.parallel_output = False
    provider.attention_backend = "fused" if dtype == torch.bfloat16 else "unfused"
    provider.params_dtype = dtype
    provider.autocast_dtype = dtype
    provider.bf16 = dtype == torch.bfloat16
    provider.gradient_accumulation_fusion = False
    provider.finalize()
    models = provider.provide_distributed_model(wrap_with_ddp=False)
    model = models[0]
    core_model = unwrap_model(models)[0]
    model.eval()
    metrics = dict(tp=tp, ep=ep, dtype=str(dtype), sequence_length=160, batch_size=2)
    exported = dict(bridge.export_hf_weights(models, cpu=True))
    if torch.distributed.get_rank() == 0:
        metrics["roundtrip"] = _exact_weights(exported, original)
        logger.info("Exact checkpoint roundtrip: %s", metrics["roundtrip"])
    reference = reference.cuda().eval()  # Official noaux_tc rejects training mode; autograd stays enabled.
    torch.manual_seed(42)
    tokens = torch.randint(0, 128, (2, 160), device="cuda")
    positions = torch.arange(tokens.shape[1], device="cuda").expand_as(tokens)
    masks = reference_masks(tokens, dtype)
    actual = model(tokens, positions, None)
    expected = reference(tokens, attention_mask=masks, use_cache=False).logits
    metrics["base_logits"] = compare_logits(actual, expected)
    logger.info("Base logits: %s", metrics["base_logits"])
    native_loss = F.cross_entropy(actual[:, :-1].float().flatten(0, 1), tokens[:, 1:].flatten())
    hf_loss = F.cross_entropy(expected[:, :-1].float().flatten(0, 1), tokens[:, 1:].flatten())
    native_loss.backward()
    hf_loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    metrics["backward"] = dict(
        native_loss=native_loss.item(), hf_loss=hf_loss.item(), finite_gradient_tensors=len(grads)
    )
    if tp == ep == 1:
        metrics["backward"]["gradient_cosines"] = {}
        for name, native_param, hf_param in (
            ("embedding", core_model.embedding.word_embeddings.weight, reference.model.embed_tokens.weight),
            ("output", core_model.output_layer.weight, reference.lm_head.weight),
            ("router", core_model.decoder.layers[1].mlp.router.weight, reference.model.layers[1].mlp.gate.weight),
        ):
            assert native_param.grad is not None and hf_param.grad is not None
            cosine = F.cosine_similarity(
                native_param.grad.float().flatten(), hf_param.grad.float().flatten(), dim=0
            ).item()
            assert cosine >= 0.99, (name, cosine)
            metrics["backward"]["gradient_cosines"][name] = cosine
    model.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    native_checkpoint = checkpoint / f"native_tp{tp}_ep{ep}"
    native_checkpoint.mkdir(exist_ok=True)
    torch.distributed.barrier()
    dist_checkpointing.save({"model": model.sharded_state_dict()}, str(native_checkpoint))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    restored = dist_checkpointing.load({"model": model.sharded_state_dict()}, str(native_checkpoint))
    model.load_state_dict(restored["model"])
    restored_export = dict(bridge.export_hf_weights(models, cpu=True))
    if torch.distributed.get_rank() == 0:
        metrics["base_checkpoint_restore"] = _exact_weights(restored_export, original)
        logger.info("Base backward and checkpoint restore: %s", metrics["backward"])

    lora = LoRA(target_modules=["linear_qkv", "linear_proj"], dim=4, alpha=4, lora_dtype=dtype)
    models = lora(models, training=True)
    lora.set_params_to_save(models)
    model = models[0]
    model.eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".adapter.linear_out.weight" in name:
                values = (torch.arange(parameter.numel(), device=parameter.device) % 31 - 15) * 1e-4
                parameter.copy_(values.reshape_as(parameter))
    adapters = dict(bridge.export_adapter_weights(models, cpu=True, show_progress=False))
    expected_names = {
        f"model.layers.{layer}.self_attn.{projection}.lora_{side}.weight"
        for layer in range(2)
        for projection in ("qkv_proj", "o_proj")
        for side in ("A", "B")
    }
    assert adapters.keys() == expected_names
    assert all(weight.count_nonzero() for name, weight in adapters.items() if ".lora_B." in name)
    with torch.no_grad():
        for layer in range(2):
            for projection in ("qkv_proj", "o_proj"):
                prefix = f"model.layers.{layer}.self_attn.{projection}"
                delta = adapters[f"{prefix}.lora_B.weight"].float() @ adapters[f"{prefix}.lora_A.weight"].float()
                weight = getattr(reference.model.layers[layer].self_attn, projection).weight
                weight.copy_((weight.float() + delta.to(weight.device)).to(weight.dtype))
        updated = models[0](tokens, positions, None)
        expected_updated = reference(tokens, attention_mask=masks, use_cache=False).logits
        metrics["lora_logits"] = compare_logits(updated, expected_updated)
        assert not torch.equal(updated, actual)
    merged = dict(bridge.export_hf_weights(models, cpu=True))
    if torch.distributed.get_rank() == 0:
        metrics["merged_lora_export"] = _exact_weights(merged, checkpoint_weights(reference))
    adapter_checkpoint = checkpoint / f"adapter_tp{tp}_ep{ep}"
    adapter_checkpoint.mkdir(exist_ok=True)
    torch.distributed.barrier()
    state = apply_peft_adapter_filter_to_state_dict({"model": model.sharded_state_dict()}, lora)
    dist_checkpointing.save(state, str(adapter_checkpoint))
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.zero_()
    state = apply_peft_adapter_filter_to_state_dict({"model": model.sharded_state_dict()}, lora)
    restored = dist_checkpointing.load(state, str(adapter_checkpoint))
    _, unexpected = unwrap_model(models)[0].load_state_dict(restored["model"], strict=False)
    assert not unexpected
    restored_adapters = dict(bridge.export_adapter_weights(models, cpu=True, show_progress=False))
    metrics["adapter_checkpoint_restore"] = _exact_weights(restored_adapters, adapters)
    with torch.no_grad():
        metrics["adapter_restore_logits"] = compare_logits(models[0](tokens, positions, None), updated)
    if torch.distributed.get_rank() == 0:
        (checkpoint / f"results_tp{tp}_ep{ep}.json").write_text(json.dumps(metrics, indent=2) + "\n")
        logger.info("MiMo-V2 parity passed: %s", metrics)
    torch.distributed.destroy_process_group()


@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize("tp,ep", [(1, 1), (2, 1), (1, 2)])
def test_mimo_v2_text_conversion(tmp_path, tp, ep):
    source = os.environ.get("MIMO_V2_HF_SOURCE", HF_MODEL)
    checkpoint = tmp_path / "mimo_v2"
    create_tiny_checkpoint(checkpoint, source, torch.bfloat16)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={max(tp, ep)}",
            __file__,
            "--checkpoint",
            str(checkpoint),
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
    parser.add_argument("--create-fixture", action="store_true")
    parser.add_argument("--hf-source", default=HF_MODEL)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    args = parser.parse_args()
    if args.create_fixture:
        logger.info("Fixture: %s", create_tiny_checkpoint(args.checkpoint, args.hf_source, getattr(torch, args.dtype)))
    else:
        run_parity(args.checkpoint, args.tp, args.ep)
