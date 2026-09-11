# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import schedules
from megatron.core.utils import get_model_config

import megatron.bridge.training.setup as training_setup
from megatron.bridge.models.gpt.gpt_builder import GPTModelConfig
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hybrid.hybrid_builder import HybridModelConfig
from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.models.nemotron_omni.nemotron_omni_provider import NemotronOmniModelProvider
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.bridge.training.state import GlobalState


pytestmark = pytest.mark.unit


class FakeDDP:
    def __init__(self, config: TransformerConfig) -> None:
        self.module = SimpleNamespace(config=config, model_type=ModelType.encoder_or_decoder)
        self.no_sync = Mock(side_effect=contextlib.nullcontext)
        self.start_grad_sync = Mock()
        self.start_param_sync = Mock()


def _make_configs(
    kind: str,
) -> tuple[GPTModelProvider | HybridModelProvider | GPTModelConfig | HybridModelConfig, TransformerConfig]:
    dimensions = dict(num_layers=1, hidden_size=8, num_attention_heads=2)
    if kind == "copied_omni":
        provider = NemotronOmniModelProvider(**dimensions)
        return provider, provider._copy_config_without_runtime_process_groups(deep=True)
    if kind in ("gpt_provider", "hybrid_provider"):
        provider = (GPTModelProvider if kind == "gpt_provider" else HybridModelProvider)(**dimensions)
        return provider, provider
    runtime = TransformerConfig(**dimensions)
    builder = GPTModelConfig if kind == "gpt_builder" else HybridModelConfig
    return builder(transformer=runtime, vocab_size=32), runtime


def _run_setup(
    provider: GPTModelProvider | HybridModelProvider | GPTModelConfig | HybridModelConfig,
    model: list[FakeDDP],
    ddp_config: SimpleNamespace,
) -> tuple[MagicMock, SimpleNamespace]:
    """Exercise the real setup call and installer without constructing a training job."""
    cfg = SimpleNamespace(
        _checkpoint_load_required=False,
        checkpoint=SimpleNamespace(
            finetune=False,
            load="/checkpoint",
            load_optim=True,
            load_rng=True,
            pretrained_checkpoint=None,
            save=None,
        ),
        dataset=SimpleNamespace(),
        ddp=SimpleNamespace(),
        dist=SimpleNamespace(
            align_grad_reduce=True,
            disable_jit_fuser=False,
            enable_megatron_core_experimental=False,
            use_decentralized_pg=False,
            use_gloo_process_groups=False,
            use_torch_fsdp2=False,
        ),
        ft=None,
        logger=SimpleNamespace(
            filter_warnings=False,
            log_progress=False,
            logging_level="INFO",
            modules_to_filter=[],
            set_level_for_all_loggers=False,
        ),
        model=SimpleNamespace(
            fine_grained_activation_offloading=False,
            restore_modelopt_state=False,
            should_pad_vocab=False,
            vocab_size=32,
        ),
        optimizer=SimpleNamespace(overlap_param_gather_with_optimizer_step=False),
        optimizer_config_override_provider=None,
        peft=None,
        profiling=SimpleNamespace(),
        rng=SimpleNamespace(data_parallel_random_init=False),
        scheduler=SimpleNamespace(),
        tensor_inspect=SimpleNamespace(),
        tokenizer=SimpleNamespace(use_tokenizer_vocab_size=False),
        train=SimpleNamespace(micro_batch_size=1, num_epochs=None),
    )
    timer = MagicMock()
    state = SimpleNamespace(
        _eval_pgs=None,
        cfg=cfg,
        comet_logger=None,
        initialize_async_checkpoint_worker=Mock(),
        start_time=0.0,
        tensorboard_logger=None,
        timers=Mock(return_value=timer),
        train_state=SimpleNamespace(step=1),
        wandb_logger=None,
    )
    pg_collection = SimpleNamespace(dp=object())
    checkpoint_manager = MagicMock(checkpointing_context={})
    optimizer = MagicMock()
    scheduler = MagicMock()
    cfg.model = provider
    cfg.ddp = ddp_config
    pg_collection.cp = SimpleNamespace(size=lambda: 1)
    pg_collection.tp = SimpleNamespace(size=lambda: 1)
    start_time_tensor = Mock()
    start_time_tensor.item.return_value = 0.0
    with (
        patch.multiple(
            training_setup,
            DistributedDataParallel=FakeDDP,
            _build_distributed_model=Mock(return_value=model),
            _should_load_checkpoint=Mock(return_value=False),
            _validate_and_set_vocab_size=Mock(return_value=(32, False)),
            barrier_and_log=Mock(),
            build_tokenizer=Mock(return_value=SimpleNamespace(vocab_size=32)),
            create_checkpoint_manager=Mock(return_value=checkpoint_manager),
            finalize_tensor_inspect_post_model_initialization=Mock(),
            initialize_megatron=Mock(return_value=pg_collection),
            initialize_tensor_inspect_pre_model_initialization=Mock(),
            maybe_load_dataloader_state=Mock(),
            maybe_log_and_save_config=Mock(),
            print_rank_0=Mock(),
            set_experimental_flag=Mock(),
            set_jit_fusion_options=Mock(),
            setup_data_iterators=Mock(return_value=(None, None, None)),
            setup_logging=Mock(),
            setup_optimizer=Mock(return_value=(optimizer, scheduler)),
            start_memory_history_recording=Mock(),
        ),
        patch.object(torch, "tensor", return_value=start_time_tensor),
        patch.object(torch.distributed, "all_reduce"),
    ):
        training_setup.setup(state, Mock())
    return optimizer, pg_collection


@pytest.mark.parametrize("kind", ["copied_omni", "gpt_provider", "hybrid_provider", "gpt_builder", "hybrid_builder"])
@pytest.mark.parametrize("chunk_count", [1, 2])
@pytest.mark.parametrize("overlap", [False, True])
def test_setup_installs_callbacks_on_scheduler_config(kind: str, chunk_count: int, overlap: bool) -> None:
    provider, runtime = _make_configs(kind)
    # The second chunk may have an independent config; the interleaved scheduler
    # reads its callback lists from the first chunk's config.
    chunks = [FakeDDP(runtime)]
    if chunk_count == 2:
        chunks.append(FakeDDP(TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=2)))
    ddp_config = SimpleNamespace(overlap_grad_reduce=overlap, overlap_param_gather=overlap, align_param_gather=True)
    optimizer, pg_collection = _run_setup(provider, chunks, ddp_config)

    config = get_model_config(chunks[0])
    assert config.finalize_model_grads_func is not None
    assert config.finalize_model_grads_func.func is training_setup.finalize_model_grads
    assert config.finalize_model_grads_func.keywords["pg_collection"] is pg_collection
    assert config.grad_scale_func == optimizer.scale_loss
    for name, method in (
        ("no_sync_func", "no_sync"),
        ("grad_sync_func", "start_grad_sync"),
        ("param_sync_func", "start_param_sync"),
    ):
        callbacks = [getattr(chunk, method) for chunk in chunks]
        expected = (callbacks[0] if chunk_count == 1 else callbacks) if overlap else None
        assert getattr(config, name) == expected
    if kind == "copied_omni":
        assert config is not provider
        assert provider.finalize_model_grads_func is None
        assert provider.no_sync_func is None


@pytest.mark.parametrize("copied", [False, True])
def test_restart_binds_callbacks_to_rebuilt_model(copied: bool) -> None:
    kind = "copied_omni" if copied else "gpt_provider"
    provider, runtime = _make_configs(kind)
    ddp_config = SimpleNamespace(overlap_grad_reduce=True, overlap_param_gather=True, align_param_gather=True)
    first = FakeDDP(runtime)
    first_optimizer, _ = _run_setup(provider, [first], ddp_config)
    state = GlobalState()
    state._cfg = SimpleNamespace(model=provider)
    state.reset_for_restart()
    rebuilt_config = provider._copy_config_without_runtime_process_groups(deep=True) if copied else provider
    rebuilt = FakeDDP(rebuilt_config)
    rebuilt_optimizer, _ = _run_setup(provider, [rebuilt], ddp_config)
    assert rebuilt_config.no_sync_func is rebuilt.no_sync
    assert rebuilt_config.grad_sync_func is rebuilt.start_grad_sync
    assert rebuilt_config.param_sync_func is rebuilt.start_param_sync
    assert rebuilt_config.grad_scale_func == rebuilt_optimizer.scale_loss
    assert rebuilt_config.grad_scale_func != first_optimizer.scale_loss


@pytest.mark.parametrize("forward_only", [False, True])
@pytest.mark.parametrize("microbatches", [1, 3])
def test_scheduler_finalizes_once_after_setup(forward_only: bool, microbatches: int) -> None:
    provider, runtime = _make_configs("copied_omni")
    model = FakeDDP(runtime)
    ddp_config = SimpleNamespace(overlap_grad_reduce=False, overlap_param_gather=False)
    finalizer = Mock()
    with patch.object(training_setup, "finalize_model_grads", finalizer):
        _, pg_collection = _run_setup(provider, [model], ddp_config)
    runtime.timers = None
    with (
        patch.object(schedules.torch, "zeros", return_value=torch.tensor(0)),
        patch.object(schedules, "forward_step", return_value=(None, 0)),
        patch.object(schedules, "backward_step") as backward,
    ):
        schedules.forward_backward_no_pipelining(
            forward_step_func=Mock(),
            data_iterator=iter(()),
            model=[model],
            num_microbatches=microbatches,
            seq_length=1,
            micro_batch_size=1,
            forward_only=forward_only,
            pg_collection=pg_collection,
        )
    assert finalizer.call_count == int(not forward_only)
    assert backward.call_count == (0 if forward_only else microbatches)
