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

"""GPU test of the ERNIE 5.5 MoE layer at EP4: grouped-GEMM experts behind the alltoall dispatcher.

Needs one node with 4 GPUs and Transformer Engine. Launch it on all four ranks::

    uv run python -m torch.distributed.run --nproc-per-node 4 -m pytest -rA \\
        -o log_cli=true -o log_cli_level=INFO tests/unit_tests/models/ernie/test_ernie55_moe_gpu.py

Without CUDA, Transformer Engine or a 4-rank launch, the tests skip.

Each rank routes its own tokens through the ``E`` layer's MoE module from ``ernie55_hybrid_stack_spec``:
``TEGroupedMLP`` with 4 of the 16 experts per rank, the alltoall dispatcher without permute fusion, the
ungated shared expert, bf16 parameters and the fp32 router. After the backward, every rank compares
against a dense per-expert fp64 autograd reference on CPU that uses the layer's routing decisions:

* the output and the input gradient;
* the router (fp32), shared-expert and latent-projection gradients, which come from local tokens only
  (no DDP here);
* the routed-expert weight gradients, which the alltoall backward accumulates from every rank's tokens,
  against the reference gradients summed over the EP group;
* the routing probabilities (fp32 GEMMs, TF32 off) to 1e-5 and the token counts exactly;
* the expert-bias update over the EP group (an NCCL all-reduce of the int64 counts).

bf16 tensors must reach a cosine similarity of at least 0.9999 with the reference; every rank logs all of
its cosines. Cases: SwiGLU (the proxy), SiTU-GLU through Megatron-Core's clamp knobs, and latent MoE with
the latent norm.

A separate check, which needs one GPU and no EP launch, compares Megatron-Core's SiTU-GLU kernels on the
GPU with the reference's fp32 single-rounding contract, which the CPU measurements in
``modeling_ernie55.moe`` found bitwise equal.
"""

from __future__ import annotations

import logging
import math
import os
import re
import zlib

import pytest
import torch


logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.run_only_on("GPU"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the ERNIE 5.5 MoE GPU test needs CUDA devices"),
]

EP = 4
HIDDEN = 256
EXPERTS = 16
TOPK = 4
MOE_FFN = 128
SHARED_FFN = 192
LATENT = 128
TOKENS = 96
EPS = 1e-6
RATE = 1e-4
SEED = 1234
COSINE = 0.9999
PROB_TOL = 1e-5

CASES = {
    "swiglu": {},
    "situ": dict(activation_func_tanh_clamp_scale=1.0),
    "latent": dict(moe_latent_size=LATENT, moe_use_norm_before_up_proj=True),
}

# Local routed-expert weights: TEGroupedMLP names them "linear_fcX.weight<i>", SequentialMLP
# "local_experts.<i>.linear_fcX.weight".
_EXPERT_PARAM = re.compile(
    r"^experts\.(?:linear_(fc[12])\.weight(\d+)|local_experts\.(\d+)\.linear_(fc[12])\.weight)$"
)


def _missing_dependencies() -> tuple[str, ...]:
    """Transformer Engine with grouped linear layers, which TEGroupedMLP needs."""
    try:
        from megatron.core.extensions import transformer_engine as te_extensions
    except Exception as error:  # noqa: BLE001 - report any import failure as a skip reason
        return (f"Megatron-Core's Transformer Engine extension ({type(error).__name__}: {error})",)
    if not te_extensions.HAVE_TE or te_extensions.TEColumnParallelGroupedLinear is None:
        return ("Transformer Engine with grouped linear layers",)
    return ()


@pytest.fixture(scope="module")
def ep4():
    """EP4 model-parallel state over the four ranks of a ``torch.distributed.run`` launch."""
    missing = _missing_dependencies()
    if missing:
        pytest.skip("the ERNIE 5.5 MoE GPU test needs " + "; ".join(missing))
    if int(os.environ.get("WORLD_SIZE", "1")) != EP or torch.cuda.device_count() < EP:
        pytest.skip(f"the ERNIE 5.5 MoE GPU test runs on {EP} GPUs: torch.distributed.run --nproc-per-node {EP}")
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    created_group = False
    if not dist.is_initialized():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group(backend="nccl")
        created_group = True
    if dist.get_world_size() != EP:
        pytest.skip(f"the ERNIE 5.5 MoE GPU test runs on exactly {EP} ranks")
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, expert_model_parallel_size=EP
    )
    model_parallel_cuda_manual_seed(SEED)
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    # The router's fp32 GEMMs are part of its contract; TF32 would round them.
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    yield ProcessGroupCollection.use_mpu_process_groups()
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
    parallel_state.destroy_model_parallel()
    if created_group:
        dist.destroy_process_group()


@pytest.fixture(autouse=True)
def _fresh_moe_metrics():
    from megatron.core.transformer.moe import moe_logging

    saved = moe_logging.get_moe_metrics_tracker()
    moe_logging.set_moe_metrics_tracker(moe_logging.MoEMetricsTracker())
    yield
    moe_logging.set_moe_metrics_tracker(saved)


# ---------------------------------------------------------------------------------------------
# Builders (device independent)
# ---------------------------------------------------------------------------------------------


def make_config(**overrides):
    """Finalized ERNIE 5.5 config for the EP4 MoE layer (bf16 parameters, alltoall, grouped experts)."""
    import torch.nn.functional as F

    from megatron.bridge.models.ernie.ernie55_config import Ernie55TransformerConfig

    values = dict(
        num_layers=2,
        hidden_size=HIDDEN,
        normalization="RMSNorm",
        layernorm_epsilon=EPS,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        gradient_accumulation_fusion=False,
        # The D-layer envelope that Ernie55TransformerConfig.finalize validates.
        num_attention_heads=4,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_head_dim=16,
        qk_pos_emb_head_dim=16,
        v_head_dim=16,
        ernie_vha_rank=2,
        experimental_attention_variant="dsa",
        dsa_kernel_backend="none",
        dsa_indexer_n_heads=4,
        dsa_indexer_head_dim=32,
        dsa_indexer_topk=8,
        dsa_indexer_k_norm_fp32=True,
        csa_window_size=4,
        # MoE.
        ffn_hidden_size=4 * HIDDEN,
        gated_linear_unit=True,
        activation_func=F.silu,
        num_moe_experts=EXPERTS,
        moe_router_topk=TOPK,
        moe_ffn_hidden_size=MOE_FFN,
        moe_shared_expert_intermediate_size=SHARED_FFN,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_bias_update_rate=RATE,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_router_dtype="fp32",
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=True,
        moe_permute_fusion=False,
        expert_model_parallel_size=EP,
    )
    values.update(overrides)
    config = Ernie55TransformerConfig(**values)
    config.finalize()
    return config


def _global_name(name: str, *, ep_rank: int, local_experts: int) -> str:
    """Parameter name with local routed-expert indices replaced by global ones (``experts.<e>.linear_fcX.weight``)."""
    match = _EXPERT_PARAM.match(name)
    if match is None:
        return name
    fc, index = (match.group(1), match.group(2)) if match.group(1) else (match.group(4), match.group(3))
    return f"experts.{ep_rank * local_experts + int(index)}.linear_{fc}.weight"


def _values(name: str, shape: torch.Size | tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Deterministic O(1) values for a tensor, keyed by its global name (the same on every rank)."""
    generator = torch.Generator().manual_seed(SEED * 1_000_003 + zlib.crc32(name.encode()))
    noise = torch.randn(shape, generator=generator, dtype=torch.float64)
    if name.endswith("routed_scaling_factor_param"):
        values = 1.0 + 0.3 * noise.clamp(-1.5, 1.5)
    elif name.endswith("expert_bias"):
        values = 0.05 * noise
    elif "norm" in name:
        values = 1.0 + 0.2 * noise
    else:
        values = noise / math.sqrt(shape[-1])
    return values.to(dtype)


def build_moe(config, pg_collection, *, device, builder=None):
    """The ``E`` layer's MoE module with deterministic, layout-independent parameter values."""
    from megatron.bridge.models.ernie.modeling_ernie55 import ernie55_hybrid_stack_spec

    if builder is None:
        builder = ernie55_hybrid_stack_spec(config).submodules.moe_layer.submodules.mlp
    moe = builder(
        config=config, pg_collection=pg_collection, is_mtp_layer=False, name=None, hash_moe_layer_threshold=None
    )
    moe.set_layer_number(1)
    moe = moe.to(device)
    ep_rank = pg_collection.ep.rank()
    with torch.no_grad():
        for name, param in moe.named_parameters():
            key = _global_name(name, ep_rank=ep_rank, local_experts=moe.num_local_experts)
            param.copy_(_values(key, param.shape, param.dtype))
        moe.router.expert_bias.copy_(_values("router.expert_bias", moe.router.expert_bias.shape, torch.float32))
    return moe.train()


# ---------------------------------------------------------------------------------------------
# Reference and comparison
# ---------------------------------------------------------------------------------------------


def _activation(config):
    import torch.nn.functional as F

    beta = config.activation_func_tanh_clamp_scale
    linear_beta = config.activation_func_tanh_clamp_scale_linear

    def activation(z: torch.Tensor) -> torch.Tensor:
        gate, up = z.chunk(2, dim=-1)
        if beta is None:
            return F.silu(gate) * up
        gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
        if linear_beta is not None:
            up = linear_beta * torch.tanh(up / linear_beta)
        return gate * up

    return activation


def reference_moe(moe, config, hidden: torch.Tensor, routing_map: torch.Tensor):
    """fp64 CPU reference of the MoE output for ``routing_map``; returns (output, gates, leaves by global name)."""
    params = {name: param for name, param in moe.named_parameters() if not _EXPERT_PARAM.match(name)}
    leaves = {"input": hidden.detach().cpu().double().requires_grad_(True)}
    for name, param in params.items():
        leaves[name] = param.detach().cpu().double().requires_grad_(True)
    expert_dtype = next(param.dtype for name, param in moe.named_parameters() if _EXPERT_PARAM.match(name))
    expert_input_size = config.moe_latent_size or config.hidden_size
    for expert in range(config.num_moe_experts):
        for fc, shape in (
            ("fc1", (2 * config.moe_ffn_hidden_size, expert_input_size)),
            ("fc2", (expert_input_size, config.moe_ffn_hidden_size)),
        ):
            name = f"experts.{expert}.linear_{fc}.weight"
            leaves[name] = _values(name, shape, expert_dtype).double().requires_grad_(True)

    activation = _activation(config)
    x = leaves["input"].reshape(-1, config.hidden_size)
    scores = torch.sigmoid(x @ leaves["router.weight"].T) + torch.sigmoid(x @ leaves["router.weight_1"].T)
    gates = torch.where(routing_map.cpu(), scores, torch.zeros_like(scores))
    if config.ernie_norm_topk_prob and config.moe_router_topk > 1:
        gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-20)
    gates = gates * leaves["router.routed_scaling_factor_param"]
    expert_input = x @ leaves["fc1_latent_proj.weight"].T if config.moe_latent_size else x
    routed = 0.0
    for expert in range(config.num_moe_experts):
        fc1, fc2 = leaves[f"experts.{expert}.linear_fc1.weight"], leaves[f"experts.{expert}.linear_fc2.weight"]
        routed = routed + (gates[:, expert : expert + 1] * activation(expert_input @ fc1.T)) @ fc2.T
    if config.moe_latent_size:
        if config.moe_use_norm_before_up_proj:
            routed = routed * torch.rsqrt(routed.pow(2).mean(dim=-1, keepdim=True) + config.layernorm_epsilon)
            routed = routed * leaves["fc2_norm.weight"]
        routed = routed @ leaves["fc2_latent_proj.weight"].T
    shared = (
        activation(x @ leaves["shared_experts.linear_fc1.weight"].T) @ leaves["shared_experts.linear_fc2.weight"].T
    )
    return (routed + shared).reshape(hidden.shape), gates, leaves


def cosine(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual, expected = actual.detach().double().cpu().flatten(), expected.detach().double().cpu().flatten()
    return (actual @ expected / (actual.norm() * expected.norm()).clamp_min(1e-300)).item()


def run_case(case: str, pg_collection, *, device, builder=None, **config_overrides) -> dict[str, float]:
    """Forward and backward of one EP case against the reference; asserts every gate, returns the cosines."""
    import torch.distributed as dist
    from megatron.core.distributed.finalize_model_grads import _update_router_expert_bias

    config = make_config(**{**CASES[case], **config_overrides})
    moe = build_moe(config, pg_collection, device=device, builder=builder)
    ep_group, ep_rank = pg_collection.ep, pg_collection.ep.rank()
    local_experts = moe.num_local_experts
    assert local_experts * ep_group.size() == EXPERTS

    routed = {}
    moe.router.register_forward_hook(lambda module, args, output: routed.update(probs=output[0], map=output[1]))
    generator = torch.Generator().manual_seed(SEED + 17 * dist.get_rank())
    hidden = torch.randn(TOKENS, 1, HIDDEN, generator=generator, dtype=torch.float64)
    hidden = hidden.to(device=device, dtype=config.params_dtype).requires_grad_(True)
    output, bias = moe(hidden)
    assert bias is None and output.dtype == config.params_dtype
    cotangent = torch.randn(output.shape, generator=generator, dtype=torch.float64)
    output.backward(cotangent.to(device=device, dtype=output.dtype))

    routing_map = routed["map"].detach()
    assert routing_map.dtype == torch.bool and bool((routing_map.sum(dim=-1) == TOPK).all())
    assert torch.equal(moe.router.local_tokens_per_expert.cpu(), routing_map.sum(dim=0).cpu())
    expected, gates, leaves = reference_moe(moe, config, hidden, routing_map)
    assert (routed["probs"].detach().cpu().double() - gates.detach()).abs().max().item() <= PROB_TOL
    expected.backward(cotangent.reshape(expected.shape))

    metrics = {"output": cosine(output, expected), "d input": cosine(hidden.grad, leaves["input"].grad)}
    # The alltoall backward sums every rank's tokens into the local experts, so the per-expert reference
    # gradients (from this rank's tokens) are summed over the EP group before taking the local experts.
    expert_grads = {}
    for fc in ("fc1", "fc2"):
        stacked = torch.stack([leaves[f"experts.{expert}.linear_{fc}.weight"].grad for expert in range(EXPERTS)])
        stacked = stacked.to(device)
        dist.all_reduce(stacked, group=ep_group)
        expert_grads[fc] = stacked.cpu()
    for name, param in moe.named_parameters():
        key = _global_name(name, ep_rank=ep_rank, local_experts=local_experts)
        assert torch.equal(param.detach().cpu(), _values(key, param.shape, param.dtype)), f"{name} changed"
        assert param.grad is not None, f"{name}: no gradient"
        expert = re.fullmatch(r"experts\.(\d+)\.linear_(fc[12])\.weight", key)
        reference_grad = expert_grads[expert.group(2)][int(expert.group(1))] if expert else leaves[key].grad
        metrics[f"d {name}"] = cosine(param.grad, reference_grad)

    # Expert-bias update over the EP group, from the global token counts.
    counts = routing_map.sum(dim=0).to(torch.int64)
    dist.all_reduce(counts, group=pg_collection.tp_dp_cp)
    before = moe.router.expert_bias.detach().double().cpu()
    _update_router_expert_bias([moe], config, tp_dp_cp_group=pg_collection.tp_dp_cp)
    direction = torch.sign(counts.double().mean() - counts.double()).cpu()
    # fp32 rounding of |b + rate| < 0.25 is at most 2^-26; a wrong or missing update is off by 1e-4.
    torch.testing.assert_close(moe.router.expert_bias.double().cpu(), before + RATE * direction, rtol=0, atol=2**-26)

    logger.info(
        "ERNIE 5.5 MoE EP4 %s, rank %d, cosines in ascending order: %s",
        case,
        dist.get_rank(),
        ", ".join(f"{name} {value:.7f}" for name, value in sorted(metrics.items(), key=lambda item: item[1])),
    )
    failed = {name: value for name, value in metrics.items() if value < COSINE}
    assert not failed, f"{case}: cosine below {COSINE}: {failed}"
    return metrics


# ---------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------


def test_grouped_expert_parameter_names_and_dtypes(ep4):
    """TEGroupedMLP names the local experts ``linear_fcX.weight<i>``, as the reference checkpoint layout does."""
    from megatron.core.transformer.moe.experts import TEGroupedMLP

    config = make_config()
    moe = build_moe(config, ep4, device=torch.cuda.current_device())
    assert isinstance(moe.experts, TEGroupedMLP)
    local = EXPERTS // EP
    expected = {
        "router.weight": ((EXPERTS, HIDDEN), torch.float32),
        "router.weight_1": ((EXPERTS, HIDDEN), torch.float32),
        "router.routed_scaling_factor_param": ((EXPERTS,), torch.float32),
        "shared_experts.linear_fc1.weight": ((2 * SHARED_FFN, HIDDEN), torch.bfloat16),
        "shared_experts.linear_fc2.weight": ((HIDDEN, SHARED_FFN), torch.bfloat16),
    }
    for expert in range(local):
        expected[f"experts.linear_fc1.weight{expert}"] = ((2 * MOE_FFN, HIDDEN), torch.bfloat16)
        expected[f"experts.linear_fc2.weight{expert}"] = ((HIDDEN, MOE_FFN), torch.bfloat16)
    manifest = {name: (tuple(param.shape), param.dtype) for name, param in moe.named_parameters()}
    assert manifest == expected


@pytest.mark.parametrize("case", sorted(CASES))
def test_ep4_moe_matches_dense_reference(ep4, case):
    run_case(case, ep4, device=torch.cuda.current_device())


def _bf16_ulps(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    """Distance in bf16 ulps between finite bf16 tensors, through the monotone int16 ordering."""

    def ordered(values: torch.Tensor) -> torch.Tensor:
        bits = values.view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, -32768 - bits, bits)

    return (ordered(actual) - ordered(expected)).abs()


def _situ_contract(z: torch.Tensor, beta: float, linear_beta: float | None) -> torch.Tensor:
    """SiTU-GLU computed in fp32 and rounded once to the input dtype (the reference contract)."""
    gate, up = z.float().chunk(2, dim=-1)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (gate * up).to(z.dtype)


def test_situ_kernels_follow_the_fp32_single_rounding_contract():
    """Megatron-Core's SiTU-GLU kernels on this GPU against the reference contract (fp32, one rounding).

    The dense and shared MLPs run ``situ_glu`` (``bias_activation_fusion=False``) or the fused
    ``bias_swiglu_impl`` (True). On CPU both are bitwise equal to the contract under torch.compile
    (``modeling_ernie55.moe``); GPU kernels lower tanh and sigmoid differently. With the default
    ``ernie_fused_ops=True``, ``fused_dispatch`` replaces CUDA calls of ``situ_glu`` with the reference's Triton
    kernel, so the ``situ_glu`` checked here runs in training only with ``model.ernie_fused_ops=false``. Gates:
    every output within one bf16 ulp, at most 1% of outputs and of input gradients different (without the jit
    fuser, 31-44% of outputs differ). Not gated, only logged: the fused kernels that round once where
    the reference rounds twice (routed SiTU and every SwiGLU), the reason a parity run keeps
    ``bias_activation_fusion=False``.
    """
    import torch.nn.functional as F
    from megatron.core import activations
    from megatron.core.fusions import fused_bias_swiglu

    device = torch.cuda.current_device()
    generator = torch.Generator().manual_seed(SEED)
    rows, width = 4096, 512
    for sigma in (1.0, 4.0):
        z = (sigma * torch.randn(rows, 2 * width, generator=generator)).to(device=device, dtype=torch.bfloat16)
        probs = (0.05 + 0.9 * torch.rand(rows, 1, generator=generator)).to(device)
        cotangent = torch.randn(rows, width, generator=generator).to(device=device, dtype=torch.bfloat16)

        gate, up = z.chunk(2, dim=-1)
        swiglu_reference = F.silu(gate) * up
        swiglu_shares = (
            (fused_bias_swiglu.bias_swiglu_impl(z, None) != swiglu_reference).float().mean().item(),
            (
                fused_bias_swiglu.weighted_bias_swiglu_impl(z, None, probs, False, None)
                != (swiglu_reference * probs).to(torch.bfloat16)
            )
            .float()
            .mean()
            .item(),
        )
        logger.info(
            "sigma %s: fused SwiGLU differs from the unfused reference in %.2f%% (dense) and %.2f%% (routed) of outputs",
            sigma,
            100 * swiglu_shares[0],
            100 * swiglu_shares[1],
        )

        for beta, linear_beta in ((1.0, None), (4.0, None), (1.0, 2.0), (4.0, 25.0)):
            setting = f"sigma {sigma}, beta {beta}, linear_beta {linear_beta}"
            inputs = [z.clone().requires_grad_(True) for _ in range(3)]
            expected = _situ_contract(inputs[0], beta, linear_beta)
            expected.backward(cotangent)
            kernels = {
                "situ_glu": activations.situ_glu(inputs[1], beta, linear_beta, 0.0),
                "fused bias_swiglu_impl": fused_bias_swiglu.bias_swiglu_impl(
                    inputs[2], None, gate_clamp_scale=beta, linear_clamp_scale=linear_beta
                ),
            }
            for (name, output), x in zip(kernels.items(), inputs[1:]):
                output.backward(cotangent)
                ulps = _bf16_ulps(output.detach(), expected.detach())
                output_share = (ulps > 0).float().mean().item()
                grad_share = (x.grad != inputs[0].grad).float().mean().item()
                logger.info(
                    "%s, %s: %.4f%% of outputs differ (max %d ulp), %.4f%% of input gradients differ",
                    setting,
                    name,
                    100 * output_share,
                    ulps.max().item(),
                    100 * grad_share,
                )
                assert ulps.max().item() <= 1 and output_share <= 0.01, f"{setting}, {name}: outputs"
                assert grad_share <= 0.01, f"{setting}, {name}: input gradients"

            fused_routed = fused_bias_swiglu.weighted_bias_swiglu_impl(
                z, None, probs, False, None, gate_clamp_scale=beta, linear_clamp_scale=linear_beta
            )
            routed_share = (fused_routed != (expected.detach() * probs).to(torch.bfloat16)).float().mean().item()
            logger.info(
                "%s: fused routed SiTU differs from the reference in %.2f%% of outputs", setting, 100 * routed_share
            )
