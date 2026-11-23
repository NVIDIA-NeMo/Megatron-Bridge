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
"""Compare legacy GPT and native Hybrid K3 with real routing and all parameter gradients.

Run on one GPU using torch.distributed.run --nproc_per_node=1. This synthetic
six-block fixture exercises dense/shared/routed MLPs, KDA, MLA, and three
attention-residual blocks. It does not download or execute a public HF model.
Set NVIDIA_TF32_OVERRIDE=0 and FLA_DISABLE_BACKEND_DISPATCH=1 for the FP32 control.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.tuning import AutotunePolicy, install
from transformers import PretrainedConfig

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.hf_pretrained.state import DictStateSource, StateDict
from megatron.bridge.models.kimi.kimi_k3_bridge import KimiK3Bridge
from megatron.bridge.models.kimi.kimi_k3_gpt_bridge import KimiK3GPTBridge


logger = logging.getLogger(__name__)


def _hf_config() -> PretrainedConfig:
    text = PretrainedConfig(
        attention_bias=False,
        attn_res_block_size=2,
        first_k_dense_replace=1,
        head_dim=64,
        hidden_act="situ",
        hidden_size=128,
        initializer_range=0.02,
        intermediate_size=256,
        kv_lora_rank=64,
        latent_moe_use_norm=True,
        linear_attn_config=dict(
            gate_lower_bound=-5.0, head_dim=64, kda_layers=[1, 2, 4, 5], num_heads=2, short_conv_kernel_size=4
        ),
        max_position_embeddings=256,
        moe_intermediate_size=64,
        moe_layer_freq=1,
        moe_router_activation_func="sigmoid",
        num_attention_heads=2,
        num_expert_group=1,
        num_experts=4,
        num_experts_per_token=2,
        num_hidden_layers=6,
        num_key_value_heads=2,
        num_shared_experts=1,
        q_lora_rank=64,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        rms_norm_eps=1e-5,
        routed_expert_hidden_size=64,
        routed_scaling_factor=1.0,
        tie_word_embeddings=False,
        topk_group=1,
        torch_dtype="float32",
        use_grouped_topk=True,
        v_head_dim=64,
        vocab_size=256,
    )
    return PretrainedConfig(architectures=["KimiK3ForConditionalGeneration"], model_type="kimi_k3", text_config=text)


def _logical(bridge, model, hf_config):
    result = {"state": {}, "gradients": {}, "gradient_presence": {}}
    tasks = MegatronModelBridge.build_conversion_tasks(bridge, hf_config, [model])
    covered = {id(task.param_weight) for task in tasks}
    assert all(id(p) in covered for p in model.parameters()), "Unmapped trainable parameter"
    persistent = set(model.state_dict())
    missing = [name for name, buffer in model.named_buffers() if name in persistent and id(buffer) not in covered]
    assert not missing, f"Unmapped persistent buffers: {missing}"
    # Router token counters are intentionally absent from HF checkpoints. Compare
    # their values too, in source-block order, independently of renamed modules.
    result["nonpersistent_buffers"] = {
        f"{index}:{name.rsplit('.', 1)[-1]}": buffer.detach().cpu().clone()
        for index, (name, buffer) in enumerate((item for item in model.named_buffers() if item[0] not in persistent))
    }
    for task in tasks:
        weight = task.param_weight
        parameter = isinstance(weight, torch.nn.Parameter)
        present = parameter and weight.grad is not None
        fields = {
            "state": weight,
            "gradients": weight.grad if present else torch.zeros_like(weight),
            "gradient_presence": torch.full_like(weight, int(present) if parameter else -1),
        }
        for field, value in fields.items():
            converted = task.mapping.megatron_to_hf(value, task.megatron_module)
            assert not result[field].keys() & converted.keys(), task.param_name
            result[field].update({n: t.detach().cpu().clone() for n, t in converted.items()})
    return result


def _build(bridge, hf_config, *, dtype, sequence_length, recompute):
    model_parallel_cuda_manual_seed(7391)
    torch.manual_seed(7391)
    config = bridge.hf_config_to_model_config(hf_config)
    config.params_dtype = dtype
    config.bf16 = dtype == torch.bfloat16
    config.fp16 = False
    config.seq_length = sequence_length
    config.parallel_output = False
    config.gradient_accumulation_fusion = False
    config.moe_permute_fusion = False
    config.cross_entropy_loss_fusion = False
    config.moe_grouped_gemm = True
    config.masked_softmax_fusion = False
    config.use_cpu_initialization = False
    if recompute != "none":
        config.recompute_granularity = recompute
        if recompute == "full":
            config.recompute_method = "uniform"
            config.recompute_num_layers = 3
        else:
            config.recompute_modules = ["core_attn", "moe"]
    config.finalize()
    return config.get_builder_cls()(config).build_model(ProcessGroupCollection.use_mpu_process_groups()).cuda()


def _run(model, ids, packed):
    embeddings = []

    def capture_embedding(_module, _inputs, output):
        output.retain_grad()
        embeddings.append(output)

    handle = model.embedding.register_forward_hook(capture_embedding)
    positions = torch.arange(ids.shape[-1], device=ids.device).unsqueeze(0)
    logits = model(ids, positions, None, packed_seq_params=packed)
    loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, 256), ids.roll(-1, -1).reshape(-1))
    loss.backward()
    handle.remove()
    return {
        "logits": logits.detach().cpu(),
        "loss": loss.detach().cpu(),
        "embedding_gradient": embeddings[0].grad.detach().cpu(),
    }


def _compare(expected, actual, *, dtype, exact):
    assert expected.shape == actual.shape
    assert expected.dtype == actual.dtype
    assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
    atol, rtol = (0.0, 0.0) if exact else ((2e-5, 2e-4) if dtype == torch.float32 else (2e-3, 2e-2))
    difference = (actual.double() - expected.double()).abs()
    relative_l2 = difference.norm() / expected.double().norm().clamp_min(1e-30)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    if dtype == torch.bfloat16 and not exact:
        assert relative_l2 <= 0.02, f"Relative L2 error {relative_l2} exceeds .02"
    return {
        "max_abs": difference.max().item(),
        "relative_l2": relative_l2.item(),
        "exact": torch.equal(actual, expected),
    }


def main() -> None:
    """Run the comparison and optionally save per-tensor numerical metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--recompute", choices=("none", "full", "selective"), default="none")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group("nccl")
    assert torch.distributed.get_world_size() == 1, "This focused regression uses one GPU"
    parallel_state.initialize_model_parallel()
    install(AutotunePolicy(mode="pinned", modules=("megatron.core", "fla", "transformer_engine", "mamba_ssm")))
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    sequence_length = 257 if args.packed else 128
    hf_config = _hf_config()
    old_bridge, new_bridge = KimiK3GPTBridge(), KimiK3Bridge()
    old = _build(old_bridge, hf_config, dtype=dtype, sequence_length=sequence_length, recompute=args.recompute)
    generator = torch.Generator(device="cuda").manual_seed(7392)
    with torch.no_grad():
        for name, parameter in old.named_parameters():
            noise = torch.randn(parameter.shape, device=parameter.device, generator=generator)
            if "norm" in name and parameter.ndim == 1:
                parameter.copy_(1 + noise * 0.05)
            elif name.endswith(("A_log", "dt_bias")):
                parameter.copy_(noise * 0.1)
            else:
                parameter.copy_(noise * 0.02)
    initial = _logical(old_bridge, old, hf_config)
    native = _build(new_bridge, hf_config, dtype=dtype, sequence_length=sequence_length, recompute=args.recompute)
    source = SimpleNamespace(
        config=hf_config, model_name_or_path="synthetic-kimi-k3", state=StateDict(DictStateSource(initial["state"]))
    )
    new_bridge.load_weights_hf_to_megatron(source, native)
    restored = _logical(new_bridge, native, hf_config)
    assert initial["state"].keys() == restored["state"].keys()
    for name, expected in initial["state"].items():
        _compare(expected, restored["state"][name], dtype=dtype, exact=True)
    ids = torch.randint(
        0, 256, (1, sequence_length), device="cuda", generator=torch.Generator(device="cuda").manual_seed(7393)
    )
    packed = None
    if args.packed:
        boundaries = torch.tensor([0, 85, 257], device="cuda", dtype=torch.int32)
        packed = PackedSeqParams(
            qkv_format="thd", cu_seqlens_q=boundaries, cu_seqlens_kv=boundaries, max_seqlen_q=172, max_seqlen_kv=172
        )
    before = _run(old, ids, packed)
    after = _run(native, ids, packed)
    old_logical, new_logical = _logical(old_bridge, old, hf_config), _logical(new_bridge, native, hf_config)
    report = {}
    for field in old_logical:
        assert old_logical[field].keys() == new_logical[field].keys(), field
        report[field] = {
            name: _compare(value, new_logical[field][name], dtype=dtype, exact=field != "gradients")
            for name, value in old_logical[field].items()
        }
    report["outputs"] = {
        name: _compare(value, after[name], dtype=dtype, exact=False) for name, value in before.items()
    }
    report["passed"] = True
    if args.report:
        args.report.write_text(json.dumps(report, indent=2))
    logger.info("Kimi K3 forward/backward parity passed for every logical tensor")
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
