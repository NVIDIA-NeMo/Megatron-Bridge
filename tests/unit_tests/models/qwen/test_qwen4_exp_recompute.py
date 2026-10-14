"""Narrow recompute admission and opt-in native RC4 MoE backward comparison."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit
_SOURCE = Path(__file__).resolve().parents[4] / "src/megatron/bridge/models/qwen/modeling_qwen4_exp/model_config.py"


@pytest.fixture
def validate_recompute():
    # Execute the production dependency-free guard, without loading CUDA extensions.
    tree = ast.parse(_SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen4ExpModelConfig")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_validate_qwen4_recompute")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(_SOURCE), "exec"), namespace)
    return namespace[method.name]


def _config(**changes):
    values = dict(
        cuda_graph_impl="none",
        vision_recompute_granularity=None,
        recompute_granularity="selective",
        recompute_modules=["moe_act", "shared_experts"],
        recompute_method=None,
        recompute_num_layers=None,
        num_moe_experts=4,
        moe_grouped_gemm=True,
        moe_shared_expert_intermediate_size=32,
        moe_shared_expert_overlap=False,
        overlap_moe_expert_parallel_comm=False,
        delay_wgrad_compute=False,
        cpu_offloading=False,
        fine_grained_activation_offloading=False,
        distribute_saved_activations=False,
    )
    assert changes.keys() <= values.keys()
    return SimpleNamespace(**(values | changes))


@pytest.mark.parametrize("modules", [["moe_act"], ["shared_experts"], ["moe_act", "shared_experts"]])
def test_selective_allowlist(validate_recompute, modules):
    validate_recompute(_config(recompute_modules=modules))


@pytest.mark.parametrize(
    "modules",
    [
        None,
        [],
        "moe_act",
        ["core_attn"],
        ["moe"],
        ["layernorm"],
        ["mhc"],
        ["gdn_norm_out"],
        ["residual_stream"],
        ["moe_act", "core_attn"],
    ],
)
def test_unsupported_modules_rejected(validate_recompute, modules):
    with pytest.raises(NotImplementedError):
        validate_recompute(_config(recompute_modules=modules))


@pytest.mark.parametrize(
    "changes",
    [
        {"recompute_granularity": "full"},
        {"cuda_graph_impl": "local"},
        {"vision_recompute_granularity": "inherit"},
        {"vision_recompute_granularity": "selective"},
        {"recompute_method": "uniform"},
        {"recompute_num_layers": 1},
        {"num_moe_experts": None},
        {"moe_grouped_gemm": False},
        {"moe_shared_expert_intermediate_size": None},
        *[
            {name: True}
            for name in (
                "moe_shared_expert_overlap",
                "overlap_moe_expert_parallel_comm",
                "delay_wgrad_compute",
                "cpu_offloading",
                "fine_grained_activation_offloading",
                "distribute_saved_activations",
            )
        ],
    ],
)
def test_unsupported_execution_modes_rejected(validate_recompute, changes):
    with pytest.raises((NotImplementedError, ValueError)):
        validate_recompute(_config(**changes))


def test_disabled_policy_preserved_and_vision_defaults_disabled(validate_recompute):
    validate_recompute(_config(recompute_granularity=None, recompute_modules=["core_attn"]))
    tree = ast.parse(_SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen4ExpTransformerConfig")
    field = next(n for n in cls.body if isinstance(n, ast.AnnAssign) and n.target.id == "vision_recompute_granularity")
    assert ast.literal_eval(field.value) is None


@pytest.fixture
def native_moe_cuda(tmp_path):
    """Actual one-GPU native MoE; not a TP/EP or full-model qualification."""
    if os.environ.get("RUN_QWEN4_RECOMPUTE_CUDA_TESTS") != "1":
        pytest.skip("Set RUN_QWEN4_RECOMPUTE_CUDA_TESTS=1 for native CUDA qualification")
    import torch
    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.spec_utils import build_module

    from megatron.bridge.models.qwen.modeling_qwen4_exp.layer_specs import get_qwen4_exp_block_spec
    from megatron.bridge.models.qwen.modeling_qwen4_exp.model_config import Qwen4ExpTransformerConfig

    assert torch.cuda.is_available(), "Native qualification requires CUDA"
    assert not torch.distributed.is_initialized(), "Fixture owns its one-rank process groups"
    torch.cuda.set_device(0)
    torch.distributed.init_process_group("nccl", init_method=(tmp_path / "rdzv").as_uri(), rank=0, world_size=1)
    try:
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)
        model_parallel_cuda_manual_seed(731)

        def build(modules, *, fusion):
            cfg = Qwen4ExpTransformerConfig(
                num_layers=4,
                hidden_size=32,
                num_attention_heads=2,
                num_query_groups=1,
                kv_channels=16,
                ffn_hidden_size=64,
                params_dtype=torch.bfloat16,
                bf16=True,
                normalization="RMSNorm",
                add_bias_linear=False,
                hidden_dropout=0.0,
                attention_dropout=0.0,
                gated_linear_unit=True,
                activation_func=torch.nn.functional.silu,
                gradient_accumulation_fusion=False,
                bias_activation_fusion=fusion,
                num_moe_experts=4,
                moe_router_topk=2,
                moe_ffn_hidden_size=32,
                moe_shared_expert_intermediate_size=32,
                moe_shared_expert_gate=True,
                moe_grouped_gemm=True,
                moe_router_pre_softmax=False,
                moe_router_dtype="fp32",
                moe_aux_loss_coeff=0.0,
                moe_token_dispatcher_type="allgather",
                moe_permute_fusion=False,
                experimental_attention_variant="gdn",
                linear_attention_freq=4,
                recompute_granularity="selective" if modules else None,
                recompute_modules=modules or [],
            )
            # Both GDN and ordinary-attention positions use the original native MoE builder.
            specs = get_qwen4_exp_block_spec(cfg).layer_specs
            assert len(specs) == 4
            module = (
                build_module(
                    specs[0].submodules.mlp,
                    config=cfg,
                    layer_number=1,
                    pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
                )
                .cuda()
                .train()
            )
            assert module.experts.activation_recompute == ("moe_act" in modules)
            assert module.shared_experts_recompute == ("shared_experts" in modules)
            return module

        yield build
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("modules", [["moe_act"], ["shared_experts"], ["moe_act", "shared_experts"]])
@pytest.mark.parametrize("fusion", [False, True])
def test_native_moe_outputs_all_gradients_and_accumulated_update(native_moe_cuda, monkeypatch, modules, fusion):
    """Two forward/backward cycles; native router, shared and expert VJPs stay live."""
    import torch
    from megatron.core import tensor_parallel

    baseline, replay = native_moe_cuda([], fusion=fusion), native_moe_cuda(modules, fusion=fusion)
    replay.load_state_dict(baseline.state_dict(), strict=True)
    torch.manual_seed(94)
    values = [torch.randn(32, 1, 32, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    cotangents = [torch.randn_like(x) for x in values]
    calls = {"moe_act": 0, "shared_experts": 0}
    original_without = tensor_parallel.CheckpointWithoutOutput.checkpoint
    original_with = tensor_parallel.checkpoint

    def counted(kind, function):
        def run(*args, **kwargs):
            calls[kind] += 1
            return function(*args, **kwargs)

        return run

    def without_output(owner, function, *args, **kwargs):
        return original_without(owner, counted("moe_act", function), *args, **kwargs)

    def with_output(function, *args, **kwargs):
        return original_with(counted("shared_experts", function), *args, **kwargs)

    monkeypatch.setattr(tensor_parallel.CheckpointWithoutOutput, "checkpoint", without_output)
    monkeypatch.setattr(tensor_parallel, "checkpoint", with_output)

    def run(module):
        module.zero_grad(set_to_none=True)
        probabilities = []

        def retain_probabilities(owner, args, output):
            probs = output[0]
            assert probs.requires_grad
            probs.retain_grad()
            probabilities.append(probs)

        handle = module.router.register_forward_hook(retain_probabilities)
        inputs, outputs = [], []
        try:
            for value, cotangent in zip(values, cotangents):
                x = value.detach().clone().requires_grad_()
                output, bias = module(x)
                assert bias is None
                output.backward(cotangent)
                inputs.append(x.grad.detach().clone())
                outputs.append(output.detach().clone())
        finally:
            handle.remove()
        grads = {n: None if p.grad is None else p.grad.detach().clone() for n, p in module.named_parameters()}
        assert len(probabilities) == 2
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all() and p.grad.count_nonzero() for p in probabilities
        )
        for prefix in ("router.", "experts.", "shared_experts."):
            assert any(g is not None and g.count_nonzero() for n, g in grads.items() if n.startswith(prefix))
        # Exercise an actual optimizer update after both accumulated backwards.
        torch.optim.SGD(module.parameters(), lr=1e-3).step()
        updates = {n: p.detach().clone() for n, p in module.named_parameters()}
        return outputs, inputs, [p.grad.detach().clone() for p in probabilities], grads, updates

    expected, actual = run(baseline), run(replay)
    for left, right in zip(expected[:3], actual[:3]):
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    for left, right in zip(expected[3:], actual[3:]):
        assert left.keys() == right.keys()
        for name, a in left.items():
            b = right[name]
            if a is None:
                assert b is None, name
            else:
                assert torch.isfinite(a).all() and torch.isfinite(b).all(), name
                torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)
    assert calls == {name: 4 if name in modules else 0 for name in calls}, (
        "Native backward must recompute each checkpoint"
    )
