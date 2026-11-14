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

"""End-to-end training loop: Float16Module + DDP + distributed optimizer + the MCore schedule.

What is covered:

* ``test_shensi_trains``: five iterations of the real ``get_forward_backward_func`` schedule under
  ``Float16Module``, DDP (distributed optimizer, APEX fused wgrad) with the latent MoE, the router
  load-balancing loss, the ERC auxiliary loss and a ROSA layer (auxiliary next-token loss). The LM
  loss falls, the optimizer steps every iteration, and every parameter receives a finite gradient —
  with three documented exceptions that have no gradient path by design:

  - the HISA indexer (its top-k output is non-differentiable);
  - routed-stream adapters that the per-token stream routing never selected in that step;
  - a routed expert that received no tokens in the first step (the load-balancing loss spreads
    tokens across all experts from the second step on).

  Grad reduce runs synchronously: with ``overlap_grad_reduce=True`` Megatron-Core's DDP requires
  every bucket parameter to register a gradient every step, which sparse routing and the
  non-differentiable indexer do not guarantee.
* ``test_shensi_trains_with_mtp``: the same loop with an MTP depth; the MTP parameters train and
  the MTP loss is logged.

The batch draws tokens from a small alphabet so the ROSA suffix automaton produces predictions
(a full-vocabulary batch has no repeated suffixes and the ROSA branch would be inactive).
"""

import importlib.util
import logging
import math
import os
from pathlib import Path

import pytest
import torch


logger = logging.getLogger(__name__)

_ITERATIONS = 5
_SEQ_LEN = 16
_MICRO_BATCHES = 2
_PROBE_VOCAB = 8


def _tiny_config_dict() -> dict:
    """Load the shared tiny Shensi fixture from the functional test group by path."""
    path = Path(__file__).parents[3] / "functional_tests" / "test_groups" / "models" / "shensi" / "__init__.py"
    spec = importlib.util.spec_from_file_location("shensi_functional_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.TINY_SHENSI_CONFIG)


@pytest.fixture(scope="module")
def distributed(tmp_path_factory):
    """Single-rank process group and Megatron-Core model-parallel state (TP = PP = CP = EP = 1)."""
    if not torch.cuda.is_available():
        pytest.skip("the Shensi training-loop test needs a GPU")
    import torch.distributed as dist
    from megatron.core import parallel_state

    created_group = False
    if dist.is_initialized():
        if dist.get_world_size() != 1:
            pytest.skip("the Shensi training-loop test runs on a single rank")
    else:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        store = tmp_path_factory.mktemp("shensi_train_loop_dist") / "store"
        dist.init_process_group(backend="nccl", init_method=f"file://{store}", rank=0, world_size=1)
        created_group = True
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
    )
    # The provider's own init seeds the CUDA RNG tracker; replicate it for the manual init.
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(0)
    yield
    parallel_state.destroy_model_parallel()
    if created_group:
        dist.destroy_process_group()


@pytest.fixture(autouse=True)
def _fresh_aux_loss_state():
    """Reset Megatron-Core's auxiliary-loss autoscalers and the MoE metrics tracker."""
    from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossAutoScaler
    from megatron.core.transformer.moe import moe_logging
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

    saved_tracker = moe_logging.get_moe_metrics_tracker()
    moe_logging.set_moe_metrics_tracker(moe_logging.MoEMetricsTracker())
    MoEAuxLossAutoScaler.main_loss_backward_scale = None
    DSAIndexerLossAutoScaler.main_loss_backward_scale = None
    yield
    MoEAuxLossAutoScaler.main_loss_backward_scale = None
    DSAIndexerLossAutoScaler.main_loss_backward_scale = None
    moe_logging.set_moe_metrics_tracker(saved_tracker)


def _provider(num_mtp_layers: int = 0):
    from transformers.models.shensi.configuration_shensi import ShensiConfig

    from megatron.bridge import AutoBridge

    cfg_dict = _tiny_config_dict()
    cfg_dict["rosa_layer_ids"] = [2]
    cfg_dict["rosa_loss_coef"] = 0.05
    cfg_dict["output_router_logits"] = True
    # Full-width stream update: every adapter exists only in the conditional configuration, which
    # test_shensi_conditional_path_train covers slot by slot; here every parameter that exists
    # must carry a gradient every step (the indexer excepted).
    cfg_dict["hc_active_streams"] = cfg_dict["hc_mult"]
    cfg = ShensiConfig(**cfg_dict)
    provider = AutoBridge.from_hf_config(cfg).to_megatron_provider(load_weights=False)
    # MTP is bridge-side: the reference no longer carries it in the config.
    provider.mtp_num_layers = int(num_mtp_layers) or None
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.dsa_kernel_backend = "none"
    provider.bf16 = True
    provider.params_dtype = torch.bfloat16
    provider.rosa_loss_coef = 0.05
    assert provider.gradient_accumulation_fusion, "APEX fused wgrad is part of the acceleration contract"
    provider.finalize()
    return provider


_TRAIN_WITH_DISTRIBUTED_OPTIMIZER = True


def _training_setup(provider, *, seed: int, lr: float = 1e-3):
    """The tiny model under Float16Module and DDP (overlapped grad reduce), with the distributed optimizer."""
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig, finalize_model_grads
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.transformer.module import Float16Module

    model = provider.provide_distributed_model(wrap_with_ddp=False)[0]
    while hasattr(model, "module"):  # the provider already casts bf16 weights
        model = model.module
    config = provider
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(seed)
    model = Float16Module(config, model)

    # Seed and initialize BEFORE the optimizer captures the weights: the distributed/float16
    # optimizer snapshots the parameters at construction, so an initialization applied after
    # construction would be silently discarded at the first step.
    torch.manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0.0, 0.02)

    dist_opt = _TRAIN_WITH_DISTRIBUTED_OPTIMIZER
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=False,
        use_distributed_optimizer=dist_opt,
    )
    model = DistributedDataParallel(config, ddp_config, model)
    optimizer_config = OptimizerConfig(
        optimizer="adam",
        lr=lr,
        min_lr=lr,
        weight_decay=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=dist_opt,
        clip_grad=1.0,
    )
    optimizer = get_megatron_optimizer(optimizer_config, [model])
    config.no_sync_func = model.no_sync
    config.finalize_model_grads_func = finalize_model_grads
    config.grad_scale_func = optimizer.scale_loss
    return model, optimizer


def _batch(seed: int, vocab_size: int, device):
    generator = torch.Generator().manual_seed(seed)
    # A small alphabet guarantees repeated suffixes, so the ROSA automaton predicts.
    tokens = torch.randint(1, _PROBE_VOCAB + 1, (1, _SEQ_LEN), generator=generator)
    labels = tokens.roll(-1, dims=1)
    loss_mask = torch.ones(1, _SEQ_LEN)
    loss_mask[0, -1] = 0.0
    position_ids = torch.arange(_SEQ_LEN).unsqueeze(0)
    return {
        name: tensor.to(device)
        for name, tensor in (
            ("tokens", tokens),
            ("labels", labels),
            ("loss_mask", loss_mask),
            ("position_ids", position_ids),
        )
    }


def _forward_step(data_iterator, model):
    batch = next(data_iterator)
    losses = model(
        input_ids=batch["tokens"],
        position_ids=batch["position_ids"],
        attention_mask=None,
        labels=batch["labels"],
        loss_mask=batch["loss_mask"],
    )
    losses = losses[0] if isinstance(losses, tuple) else losses
    loss_mask = batch["loss_mask"]

    def loss_func(output_tensor):
        loss = (output_tensor.float().view(-1) * loss_mask.view(-1)).sum() / loss_mask.sum()
        return loss, {"lm loss": loss.detach()}

    return losses, loss_func


def _run_schedule(model, batches) -> float:
    from megatron.core.pipeline_parallel import get_forward_backward_func

    model.zero_grad_buffer()
    outputs = get_forward_backward_func()(
        forward_step_func=_forward_step,
        data_iterator=iter(batches),
        model=[model],
        num_microbatches=len(batches),
        seq_length=batches[0]["tokens"].size(1),
        micro_batch_size=1,
        forward_only=False,
    )
    return torch.stack([item["lm loss"] for item in outputs]).mean().item()


def _unwrap(model):
    """DDP -> Float16Module -> ShensiModel."""
    return model.module.module


def _no_grad_allowlist(name: str) -> bool:
    """Parameters allowed a zero gradient in a given step.

    Every parameter in this configuration trains: the HISA indexer through its KL auxiliary loss
    (top-k is non-differentiable; the scores carry the signal), the router/expert pool through
    the load-balancing and ERC losses, and ROSA through the LM loss. A first-step exception is
    documented separately for routed experts that received no tokens before the load-balancing
    loss spreads them (``_assert_experts_train``).
    """
    return False


def _assert_experts_train(inner) -> None:
    """After the first step the load-balancing loss routes tokens to every expert."""
    for name, param in inner.named_parameters():
        if ".experts." not in name:
            continue
        grad = param.main_grad
        assert grad is not None and bool(torch.isfinite(grad).all()), name
        assert torch.any(grad != 0), f"{name} never received tokens or gradients"


def _train(provider, *, iterations: int = _ITERATIONS):
    batches = [_batch(seed, provider.vocab_size, torch.cuda.current_device()) for seed in range(_MICRO_BATCHES)]
    model, optimizer = _training_setup(provider, seed=7)
    inner = _unwrap(model)

    lm_losses = []
    for iteration in range(iterations):
        optimizer.zero_grad()
        lm_loss = _run_schedule(model, batches)
        grads = {
            name: param.main_grad.detach().float().clone()
            for name, param in inner.named_parameters()
            if param.requires_grad
        }
        assert math.isfinite(lm_loss), f"iteration {iteration}: LM loss {lm_loss}"
        non_finite = [name for name, grad in grads.items() if not bool(torch.isfinite(grad).all())]
        assert not non_finite, f"iteration {iteration}: non-finite gradients for {non_finite}"
        without_grad = sorted(name for name, grad in grads.items() if not torch.any(grad != 0))
        violations = [name for name in without_grad if not _no_grad_allowlist(name)]
        if iteration == 0:
            # The load-balancing loss needs one step to spread tokens over every expert.
            violations = [name for name in violations if ".experts." not in name]
        assert not violations, f"iteration {iteration}: no gradient for {violations}"

        update_successful, grad_norm, _ = optimizer.step()
        assert update_successful, f"iteration {iteration}: the optimizer step was skipped"
        assert grad_norm is None or math.isfinite(float(grad_norm)), f"iteration {iteration}: grad norm {grad_norm}"
        lm_losses.append(lm_loss)
        logger.info("[shensi-train-loop] iteration %d: lm=%.6f", iteration, lm_loss)

    _assert_experts_train(inner)
    assert lm_losses[-1] < lm_losses[0], lm_losses
    return lm_losses


def test_shensi_trains(distributed):
    """The full loop trains: LM loss falls, ERC and the ROSA auxiliary loss carry gradient."""
    _train(_provider(num_mtp_layers=0))


def test_boundary_modules_follow_the_pipeline_stage(distributed):
    """A non-terminal stage builds neither the stream embeddings nor the output-side modules."""
    from megatron.core.process_groups_config import ProcessGroupCollection

    provider = _provider(num_mtp_layers=0)
    # ROSA ties its reference table to the shared token embedding, which a middle stage does not own.
    provider.rosa_layer_ids = []
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    provider._pg_collection = pg_collection
    middle = provider.provide(pre_process=False, post_process=False)
    assert middle.stream_embeddings is None
    assert middle.hc_head is None
    assert middle.norm is None
    assert middle.output_attn_res is None
    assert middle.attn_bank is not None

    provider._pg_collection = pg_collection
    full = provider.provide(pre_process=True, post_process=True)
    middle_names = {name for name, _ in middle.named_parameters()}
    full_names = {name for name, _ in full.named_parameters()}
    assert middle_names < full_names


def test_shensi_trains_with_mtp(distributed):
    """An MTP depth trains in the same loop and logs its loss."""
    from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

    provider = _provider(num_mtp_layers=1)
    batches = [_batch(seed, provider.vocab_size, torch.cuda.current_device()) for seed in range(_MICRO_BATCHES)]
    model, optimizer = _training_setup(provider, seed=7)
    inner = _unwrap(model)
    assert inner.mtp is not None and len(inner.mtp.layers) == 1
    mtp_params = [name for name, _ in inner.named_parameters() if name.startswith("mtp.")]
    mtp_experts_with_grad: set[str] = set()

    lm_losses = []
    for iteration in range(2):
        optimizer.zero_grad()
        lm_loss = _run_schedule(model, batches)
        assert math.isfinite(lm_loss)
        grads = {
            name: param.main_grad.detach().float() for name, param in inner.named_parameters() if param.requires_grad
        }
        non_finite = [name for name, grad in grads.items() if not bool(torch.isfinite(grad).all())]
        assert not non_finite, f"non-finite gradients for {non_finite}"
        mtp_grads = {name: grad for name, grad in grads.items() if name.startswith("mtp.")}
        assert len(mtp_grads) == len(mtp_params) and mtp_grads
        # From the second step the load-balancing loss has routed tokens to every expert, so the
        # whole MTP depth (including its attention-residual block routing) must carry a gradient.
        # Routed experts follow the same per-step sparsity as any MoE: an expert that received no
        # token legitimately has no gradient in that step, so they are checked across the loop.
        if iteration > 0:
            without = sorted(
                name for name, grad in mtp_grads.items() if ".experts." not in name and not torch.any(grad != 0)
            )
            assert not without, f"the MTP depth has untrained parameters: {without}"
        for name, grad in mtp_grads.items():
            if ".experts." in name and torch.any(grad != 0):
                mtp_experts_with_grad.add(name)
        optimizer.step()
        lm_losses.append(lm_loss)
    MTPLossLoggingHelper.tracker.clear()
    mtp_experts = {name for name in mtp_params if ".experts." in name}
    assert mtp_experts_with_grad == mtp_experts, (
        f"MTP experts that never trained: {sorted(mtp_experts - mtp_experts_with_grad)}"
    )
    assert lm_losses[-1] <= lm_losses[0] + 1.0, lm_losses


def test_shensi_conditional_path_train(distributed):
    """Every routed-stream adapter, the route projection and the decay train when selected.

    The end-to-end loop uses the full-width stream configuration (no conditional path); this test
    drives ``ShensiHyperConnection.write`` directly with each routed stream selected for some
    tokens, so every conditional parameter demonstrates its gradient.
    """
    from types import SimpleNamespace

    from megatron.bridge.models.shensi.modeling_shensi import ShensiHyperConnection

    hidden, streams_n, seq, batch = 8, 4, 6, 1
    config = SimpleNamespace(
        hidden_size=hidden,
        hc_mult=streams_n,
        hc_active_streams=2,
        hc_fixed_streams=1,
        hc_adapter_rank=2,
        hc_conv_kernels=[],
        layernorm_epsilon=1e-6,
        sequence_parallel=False,
        calculate_per_token_loss=False,
    )
    module = ShensiHyperConnection(config, is_mlp=False)
    # Reproduce the checkpoint initialization, then make every coefficient row non-zero: the
    # reference starts with zero pre/post rows and a zero adapter up-projection, paths that only
    # become live after a step. The test is about the conditional routing mechanism itself.
    module.init_shensi_weights()
    with torch.no_grad():
        module.fn.normal_(0.0, 0.02)
        for adapter in module.adapters:
            adapter[1].weight.normal_(0.0, 0.1)

    torch.manual_seed(0)
    streams = torch.randn(seq, batch, streams_n, hidden, requires_grad=True)
    output = torch.randn(seq, batch, hidden)
    residual = torch.randn(seq, batch, hidden)

    collapsed, state = module(streams)
    # Token t selects routed stream (t % 3): every adapter sees tokens.
    routed = torch.arange(seq, device=streams.device) % (streams_n - config.hc_fixed_streams)
    slot = config.hc_fixed_streams
    # Only the selection is forced; the scale stays the live score so the route projection's
    # gradient path (adapter updates it scales) participates.
    state["indices"] = state["indices"].detach().clone()
    state["indices"][:, 0, slot] = routed + slot

    updated = module.write_back(output, state, residual)
    updated.sum().backward()

    for name, param in module.named_parameters():
        if name.endswith("pre.weight") or name.startswith("convs."):
            continue  # pre is zero-init by design; this module has no convolutions
        grad = param.grad
        assert grad is not None and bool(torch.isfinite(grad).all()), name
        assert torch.any(grad != 0), f"{name} received no gradient"
    assert collapsed.shape == (seq, batch, hidden)


def test_full_recompute_matches_no_recompute(distributed):
    """Full activation recompute reproduces the losses and every gradient of the eager run.

    The Shensi stream loop carries per-layer state (decoded streams, re-read bank, top-k chain),
    so MCore's built-in full recompute cannot drive it; the stack checkpoints one loop iteration
    at a time. The recompute leg also drops the distributed optimizer and APEX fused wgrad: their
    grad-buffer bookkeeping assumes one backward per forward, which the checkpoint's
    forward-then-recompute breaks.
    """
    losses_per_run, grads_per_run = {}, {}
    # `selective` must behave like `full` for Shensi: Megatron-Core's selective recompute wraps
    # only its own submodules, so the stack maps both settings to per-iteration checkpointing.
    for label, recompute in (("eager", False), ("full_recompute", True), ("selective_recompute", "selective")):
        provider = _provider(num_mtp_layers=0)
        global _TRAIN_WITH_DISTRIBUTED_OPTIMIZER
        _TRAIN_WITH_DISTRIBUTED_OPTIMIZER = not recompute
        provider.gradient_accumulation_fusion = False
        if recompute:
            provider.recompute_granularity = "full" if recompute is True else recompute
            provider.recompute_method = "block"
            provider.recompute_num_layers = 1
        else:
            provider.recompute_granularity = "none"
        batches = [_batch(seed, provider.vocab_size, torch.cuda.current_device()) for seed in range(_MICRO_BATCHES)]
        model, optimizer = _training_setup(provider, seed=7)
        inner = _unwrap(model)

        losses, grads = [], None
        for _ in range(2):
            optimizer.zero_grad()
            lm_loss = _run_schedule(model, batches)
            losses.append(float(lm_loss))
            grads = {
                name: param.main_grad.detach().float().clone()
                for name, param in inner.named_parameters()
                if param.requires_grad
            }
            optimizer.step()
        losses_per_run[label] = losses
        grads_per_run[label] = grads
    _TRAIN_WITH_DISTRIBUTED_OPTIMIZER = True

    assert losses_per_run["eager"] == losses_per_run["full_recompute"], losses_per_run
    assert losses_per_run["eager"] == losses_per_run["selective_recompute"], losses_per_run
    eager_grads, ckpt_grads = grads_per_run["eager"], grads_per_run["full_recompute"]
    assert set(eager_grads) == set(ckpt_grads)
    for name in eager_grads:
        torch.testing.assert_close(
            ckpt_grads[name], eager_grads[name], rtol=1e-5, atol=1e-6, msg=lambda m: f"{name}: {m}"
        )


def test_indexer_trains_through_kl_aux_loss(distributed):
    """The HISA indexer trains: KL auxiliary loss gives it gradients, forward output is unchanged.

    The top-k selection is non-differentiable, so the indexer cannot train from the LM loss. It
    trains against the attention distribution over the same candidates (DeepSeek-V3
    lightning-indexer style): a KL auxiliary loss attached through the auxiliary-loss autoscaler.
    """
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

    outputs, indexer_grads = {}, {}
    for label, coeff in (("off", 0.0), ("on", 0.05)):
        MoEAuxLossAutoScaler.main_loss_backward_scale = None
        provider = _provider(num_mtp_layers=0)
        provider.indexer_loss_coeff = coeff
        provider.finalize()
        batches = [_batch(seed, provider.vocab_size, torch.cuda.current_device()) for seed in range(1)]
        model, optimizer = _training_setup(provider, seed=7)
        inner = _unwrap(model)
        model.eval()
        with torch.no_grad():
            outputs[label] = model(batches[0]["tokens"], batches[0]["position_ids"], attention_mask=None)
        model.train()
        optimizer.zero_grad()
        _run_schedule(model, batches)
        grads = {
            name: param.main_grad.detach().float().clone()
            for name, param in inner.named_parameters()
            if ".indexer." in name
        }
        assert grads, "no indexer parameters found"
        indexer_grads[label] = grads
        MoEAuxLossAutoScaler.main_loss_backward_scale = None

    # Same weights, same inputs: the auxiliary loss must not change the forward output.
    torch.testing.assert_close(outputs["on"], outputs["off"], rtol=0, atol=0)

    total_off = sum(float(g.abs().sum()) for g in indexer_grads["off"].values())
    total_on = sum(float(g.abs().sum()) for g in indexer_grads["on"].values())
    assert total_off == 0.0, "the indexer must not receive gradient from the LM loss alone"
    assert total_on > 0.0, "the indexer KL auxiliary loss produced no gradient"
    for name, grad in indexer_grads["on"].items():
        assert bool(torch.isfinite(grad).all()), name


def test_hc_router_balance_spreads_the_streams():
    """The routed-stream balancing loss makes every adapter reachable.

    Per-token top-k stream routing starves an adapter when its stream never wins. The Switch-style
    balancing loss (``hc_router_aux_loss_coef``) lowers the gates of over-subscribed streams; this
    test optimizes only that loss and checks every routed stream is selected afterwards.
    """
    from types import SimpleNamespace

    from megatron.bridge.models.shensi.modeling_shensi import ShensiHyperConnection

    hidden, streams_n, seq, batch = 8, 4, 64, 4
    config = SimpleNamespace(
        hidden_size=hidden,
        hc_mult=streams_n,
        hc_active_streams=2,
        hc_fixed_streams=1,
        hc_adapter_rank=2,
        hc_conv_kernels=[],
        layernorm_epsilon=1e-6,
        sequence_parallel=False,
        calculate_per_token_loss=False,
        hc_router_aux_loss_coef=1.0,
    )
    module = ShensiHyperConnection(config, is_mlp=False)
    module.init_shensi_weights()
    with torch.no_grad():
        module.fn.normal_(0.0, 0.02)
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-2)

    torch.manual_seed(0)
    streams = torch.randn(seq, batch, streams_n, hidden)
    for _ in range(40):
        optimizer.zero_grad()
        collapsed, _ = module(streams)
        # `collapsed` depends only on the pre rows of `fn`; the route rows are trained solely by
        # the balancing loss attached to this activation.
        collapsed.sum().backward()
        optimizer.step()

    with torch.no_grad():
        _, state = module(torch.randn(seq, batch, streams_n, hidden))
    selected = set(state["indices"][..., 1:].reshape(-1).tolist())
    assert selected == set(range(1, streams_n - 1 + 1)), selected
