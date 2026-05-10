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
"""Functional test for MegatronMIMO heterogeneous parallel training.

Exercises pretrain_megatron_mimo -> setup_megatron_mimo -> train_megatron_mimo on 2 GPUs with
synthetic data. Requires torchrun with --nproc_per_node=2.

Run:
    torchrun --nproc_per_node=2 -m pytest -v -s -x \
        tests/functional_tests/test_groups/training/megatron_mimo/test_pretrain_megatron_mimo.py
"""

from __future__ import annotations

import contextlib
import os

import pytest
import torch
import torch.distributed as dist
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.data.megatron_mimo.mock_provider import MockMegatronMIMOProvider
from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)
from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import MegatronMIMOProvider
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    LoggerConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from megatron.bridge.training.megatron_mimo_step import forward_step as megatron_mimo_forward_step
from megatron.bridge.training.pretrain_megatron_mimo import pretrain_megatron_mimo
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from tests.functional_tests.utils import initialize_distributed


# ── Constants ────────────────────────────────────────────────────────────────

_ENCODER_SEQ_LEN = 197  # (224/16)^2 = 196 patches + 1 class token
_SPECIAL_TOKEN_ID = 32000
_VOCAB_SIZE = 50304
_SEQ_LENGTH = 256
_IMG_SIZE = 224
_PATCH_DIM = 16
_TRAIN_ITERS = 5


# ── Model helpers ────────────────────────────────────────────────────────────


def _make_vision_config(
    *,
    dtype: str = "bf16",
    per_token_loss: bool = False,
    dropout: float = 0.0,
    deterministic: bool = False,
) -> TransformerConfig:
    if dtype == "bf16":
        bf16, fp16 = True, False
        pipeline_dtype = torch.bfloat16
    elif dtype == "fp32":
        bf16, fp16 = False, False
        pipeline_dtype = torch.float32
    else:
        raise ValueError(f"unsupported dtype {dtype!r}")
    cfg = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        ffn_hidden_size=256,
        num_attention_heads=4,
        pipeline_dtype=pipeline_dtype,
        bf16=bf16,
        fp16=fp16,
        variable_seq_lengths=True,
        moe_token_dispatcher_type="alltoall",
        calculate_per_token_loss=per_token_loss,
    )
    cfg.add_bias_linear = True
    cfg.add_qkv_bias = True
    cfg.hidden_dropout = dropout
    cfg.attention_dropout = dropout
    cfg.gated_linear_unit = False
    cfg.layernorm_zero_centered_gamma = False
    cfg.apply_query_key_layer_scaling = False
    cfg.bias_activation_fusion = False
    cfg.bias_dropout_fusion = False
    cfg.attention_softmax_in_fp32 = True
    cfg.normalization = "LayerNorm"
    cfg.apply_rope_fusion = False
    if deterministic:
        cfg.attention_backend = AttnBackend.unfused
        cfg.deterministic_mode = True
        cfg.recompute_granularity = "full"
        cfg.recompute_method = "uniform"
        cfg.recompute_num_layers = 1
    return cfg


def _make_language_config(
    *,
    dtype: str = "bf16",
    per_token_loss: bool = False,
    dropout: float = 0.0,
    deterministic: bool = False,
) -> TransformerConfig:
    """Language ``TransformerConfig`` for the functional tests.

    ``hidden_dropout`` / ``attention_dropout`` default to 0.1 in
    ``TransformerConfig``. Most smoke tests keep dropout at 0 for cheap
    deterministic training. The checkpoint-resume parity tests pass
    ``dropout=0.1`` with ``save_rng=True`` to verify that RNG-consuming forward
    paths replay the same masks after restore.

    ``dtype`` lets callers switch the language stack to fp32 — needed for the
    parity test where bf16 noise (~1e-3) overlaps the optimizer-reset signature
    (~1e-3 from Adam bias-correction skew at t=1 vs t=4). fp32 drops the noise
    floor to ~1e-6 so a tight tolerance can actually discriminate the two.

    ``per_token_loss`` flips ``calculate_per_token_loss`` on the
    TransformerConfig. The heterogeneous-DP smoke must use ``True`` to exercise
    the production-supported path (pure-SUM DDP + ``1/N_global`` finalize math
    for MegatronMIMO); ``False`` makes mcore's schedule do its own local
    averaging and ``finalize_model_grads_multimodule`` skips the per-token
    divisor.

    ``dropout`` controls both hidden and attention dropout for the language
    stack, matching the vision config's test knob.
    """
    if dtype == "bf16":
        bf16, fp16 = True, False
        pipeline_dtype = torch.bfloat16
    elif dtype == "fp32":
        bf16, fp16 = False, False
        pipeline_dtype = torch.float32
    else:
        raise ValueError(f"unsupported dtype {dtype!r}")
    cfg = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        ffn_hidden_size=256,
        num_attention_heads=4,
        pipeline_dtype=pipeline_dtype,
        bf16=bf16,
        fp16=fp16,
        hidden_dropout=dropout,
        attention_dropout=dropout,
        variable_seq_lengths=True,
        moe_token_dispatcher_type="alltoall",
        cross_entropy_loss_fusion=not deterministic,
        calculate_per_token_loss=per_token_loss,
    )
    if deterministic:
        cfg.attention_backend = AttnBackend.unfused
        cfg.deterministic_mode = True
        cfg.recompute_granularity = "full"
        cfg.recompute_method = "uniform"
        cfg.recompute_num_layers = 1
    return cfg


def _build_model_specs(
    *, dtype: str = "bf16", per_token_loss: bool = False, dropout: float = 0.0, deterministic: bool = False
):
    """Return (language_model_spec, modality_submodules_spec, special_token_ids).

    ``dtype="fp32"`` is used by the heterogeneous parity test where bf16
    reduction noise (~1e-3) overlaps the optimizer-state-reset signature.

    ``per_token_loss=True`` exercises the production-supported pure-SUM DDP +
    ``1/N_global`` finalize path; required for the heterogeneous smoke and
    parity tests (where DDP's default ``1/dp_size`` averaging on asymmetric
    DP would use a different scaling path).
    """
    vision_encoder = ModuleSpec(
        module=CLIPViTModel,
        params={
            "transformer_config": _make_vision_config(
                dtype=dtype,
                per_token_loss=per_token_loss,
                dropout=dropout,
                deterministic=deterministic,
            ),
            "transformer_layer_spec": get_vit_layer_with_transformer_engine_spec(),
            "patch_dim": _PATCH_DIM,
            "img_h": _IMG_SIZE,
            "img_w": _IMG_SIZE,
        },
    )
    vision_submodule_spec = ModuleSpec(
        module=VisionModalitySubmodules,
        params={},
        submodules={"encoders": {"clip": vision_encoder}},
    )
    language_model_spec = ModuleSpec(
        module=GPTModel,
        params={
            "config": _make_language_config(
                dtype=dtype,
                per_token_loss=per_token_loss,
                dropout=dropout,
                deterministic=deterministic,
            ),
            "transformer_layer_spec": get_gpt_layer_with_transformer_engine_spec(),
            "vocab_size": _VOCAB_SIZE,
            "max_sequence_length": _SEQ_LENGTH,
        },
    )
    return language_model_spec, {"vision": vision_submodule_spec}, {"vision": _SPECIAL_TOKEN_ID}


# ── Data helpers ─────────────────────────────────────────────────────────────


class _CLIPImageProcessor:
    """Minimal image processor that produces pixel_values in the shape CLIP ViT expects.

    Avoids depending on the openai/clip-vit-base-patch16 HF processor which may
    not be available in all CI environments.
    """

    def __call__(self, image, return_tensors="pt"):
        # CLIP ViT expects [3, img_h, img_w] normalized float tensors.
        import numpy as np

        arr = np.array(image, dtype=np.float32) / 255.0  # [H, W, 3]
        arr = arr.transpose(2, 0, 1)  # [3, H, W]
        t = torch.tensor(arr)
        if return_tensors == "pt":
            t = t.unsqueeze(0)  # [1, 3, H, W] — batch dim removed by MegatronMIMODataset
        return {"pixel_values": t}


def _build_mock_data_provider() -> MockMegatronMIMOProvider:
    provider = MockMegatronMIMOProvider(
        seq_length=_SEQ_LENGTH,
        processor_paths={},
        tokenizer_path="gpt2",
        special_token_ids={"vision": _SPECIAL_TOKEN_ID},
        encoder_seq_lengths={"vision": _ENCODER_SEQ_LEN},
        modality_configs={"vision": {"type": "image", "width": _IMG_SIZE, "height": _IMG_SIZE}},
    )
    provider.drop_last = True
    # Inject our minimal CLIP-compatible processor so MegatronMIMODataset uses it.
    object.__setattr__(provider, "_processors", {"vision": _CLIPImageProcessor()})
    return provider


def _wrap_iter(loader_iter, *, pixel_dtype: torch.dtype = torch.bfloat16):
    """Adapt data-loader batches for the MegatronMIMO model.

    Remaps modality_inputs["vision"]["pixel_values"] to
    modality_inputs["vision"]["clip"]["x"] for CLIPViTModel.
    """
    for batch in loader_iter:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.cuda(non_blocking=True)
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        value[k] = v.cuda(non_blocking=True)
                    elif isinstance(v, dict):
                        for kk, vv in v.items():
                            if isinstance(vv, torch.Tensor):
                                value[k][kk] = vv.cuda(non_blocking=True)

        mi = batch.get("modality_inputs")
        if mi and "vision" in mi:
            pv = mi["vision"].get("pixel_values")
            if pv is not None:
                mi["vision"] = {"clip": {"x": pv.to(pixel_dtype)}}

        if "loss_mask" not in batch or batch["loss_mask"] is None:
            batch["loss_mask"] = torch.ones_like(batch["input_ids"], dtype=torch.float)

        batch["attention_mask"] = None
        yield batch


def _build_data_iterators(cfg, megatron_mimo_infra, *, train_state=None):
    """Build data iterators compatible with pretrain_megatron_mimo's build_data_iterators_fn.

    Accepts an optional ``train_state`` so consumed-sample offsets from a restored
    checkpoint are honored during resume. ``setup_megatron_mimo`` introspects the
    signature and passes ``train_state`` when ``train_state.step > 0``.
    """
    from megatron.bridge.data.megatron_mimo.loaders import build_megatron_mimo_data_loaders
    from megatron.bridge.training.state import TrainState

    if train_state is None:
        train_state = TrainState()
    train_samples = cfg.train.train_iters * cfg.train.global_batch_size

    train_loader, _, _ = build_megatron_mimo_data_loaders(
        cfg=cfg,
        train_state=train_state,
        megatron_mimo_provider=cfg.dataset,
        train_samples=max(train_samples, 100),
        valid_samples=0,
        test_samples=0,
    )

    pixel_dtype = torch.bfloat16 if getattr(cfg.model, "bf16", False) else torch.float32
    train_iter = _wrap_iter(train_loader, pixel_dtype=pixel_dtype) if train_loader is not None else None
    return train_iter, None


# ── Config builder ───────────────────────────────────────────────────────────


def _build_config(
    parallelism_config: MegatronMIMOParallelismConfig,
    train_iters: int = _TRAIN_ITERS,
    *,
    micro_batch_size: int = 1,
    global_batch_size: int = 1,
    per_token_loss: bool = False,
    deterministic: bool = False,
) -> ConfigContainer:
    dtype = "fp32" if deterministic else "bf16"
    language_model_spec, modality_submodules_spec, special_token_ids = _build_model_specs(
        dtype=dtype,
        per_token_loss=per_token_loss,
        deterministic=deterministic,
    )

    megatron_mimo_provider = MegatronMIMOProvider(
        language_model_spec=language_model_spec,
        modality_submodules_spec=modality_submodules_spec,
        special_token_ids=special_token_ids,
        megatron_mimo_parallelism_config=parallelism_config,
        topology={"vision": ["language"], "language": []},
        # MegatronMIMOProvider casts the whole model via .bfloat16() after build,
        # overriding per-submodule TransformerConfig dtypes — flip it off in
        # deterministic mode so the model stays fp32.
        bf16=not deterministic,
    )

    train_cfg = TrainingConfig(
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        train_iters=train_iters,
    )
    train_cfg.num_microbatches = 1

    opt_config = OptimizerConfig(
        bf16=not deterministic,
        use_distributed_optimizer=True,
        lr=1e-4,
        min_lr=0.0,
    )

    cfg = ConfigContainer(
        train=train_cfg,
        model=megatron_mimo_provider,
        optimizer=opt_config,
        scheduler=SchedulerConfig(start_weight_decay=0.0, end_weight_decay=0.0),
        dataset=_build_mock_data_provider(),
        logger=LoggerConfig(),
        tokenizer=TokenizerConfig(),
        # Smoke tests do not save checkpoints, so skip RNG checkpoint payloads.
        checkpoint=CheckpointConfig(save_rng=False),
    )
    # Mirrors the --deterministic flag plumbing in
    # examples/models/megatron_mimo/megatron_mimo_training_llava.py: fp32 grad
    # reduction is the part of "deterministic mode" that lives on DDP rather
    # than TransformerConfig.
    cfg.ddp.grad_reduce_in_fp32 = deterministic
    return cfg


# ── Index tracing for checkpoint-resume test ─────────────────────────────────


_RESUME_TEST_CONSUMED_INDICES: list[int] = []


class _IndexTaggedDataset(torch.utils.data.Dataset):
    """Wrap a Dataset so each sample carries its global index separately.

    Used by the checkpoint-resume L2 test to trace which samples were consumed
    across save/resume runs without modifying model inputs.
    """

    def __init__(self, inner: torch.utils.data.Dataset):
        self._inner = inner

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx):
        sample = self._inner[idx]
        sample["sample_index"] = idx
        return sample


class _TraceableMockProvider(MockMegatronMIMOProvider):
    """Test-only provider that wraps MockMegatronMIMOProvider's datasets with
    ``_IndexTaggedDataset`` so samples carry their global index."""

    def build_datasets(self, context):
        train, valid, test = super().build_datasets(context)
        wrap = lambda ds: _IndexTaggedDataset(ds) if ds is not None else None
        return wrap(train), wrap(valid), wrap(test)

    def get_collate_fn(self):
        base_collate_fn = super().get_collate_fn()

        def _collate_with_sample_index(batch):
            collated = base_collate_fn(batch)
            sample_indices = [sample["sample_index"] for sample in batch]
            collated["sample_index"] = torch.tensor(sample_indices, dtype=torch.long)
            return collated

        return _collate_with_sample_index


def _tracing_wrap_iter(loader_iter, *, pixel_dtype: torch.dtype = torch.bfloat16):
    """Like ``_wrap_iter`` but records batch sample-indices into the module-level
    ``_RESUME_TEST_CONSUMED_INDICES`` list before yielding."""
    for batch in _wrap_iter(loader_iter, pixel_dtype=pixel_dtype):
        sample_index = batch.pop("sample_index")
        _RESUME_TEST_CONSUMED_INDICES.extend(sample_index.cpu().tolist())
        yield batch


@contextlib.contextmanager
def _capture_post_step_params_into(target: dict):
    """Snapshot every parameter into ``target`` after each train step.

    Monkey-patches ``train_step_megatron_mimo`` so the wrapper captures
    ``inner.named_parameters()`` after ``optimizer.step()`` runs each
    iteration. Last iteration's snapshot is the post-train state — the
    artifact the resume parity check compares.

    Captures per rank: each rank holds its own TP shard, and both compared
    runs go through the identical TP rank assignment, so per-rank dicts are
    apples-to-apples.
    """
    import megatron.bridge.training.train_megatron_mimo as _tmm
    from megatron.bridge.training.megatron_mimo_parallel_utils import unwrap_megatron_mimo_model

    original = _tmm.train_step_megatron_mimo

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        model = kwargs.get("model")
        if model is None and len(args) > 2:
            model = args[2]
        inner = unwrap_megatron_mimo_model(model)
        target.clear()
        for name, param in inner.named_parameters():
            target[name] = param.detach().clone().cpu()
        return result

    _tmm.train_step_megatron_mimo = wrapped
    try:
        yield
    finally:
        _tmm.train_step_megatron_mimo = original


def _build_tracing_data_iterators(cfg, megatron_mimo_infra, *, train_state=None):
    """Same as ``_build_data_iterators`` but uses the tracing iter wrapper."""
    from megatron.bridge.data.megatron_mimo.loaders import build_megatron_mimo_data_loaders
    from megatron.bridge.training.state import TrainState

    if train_state is None:
        train_state = TrainState()
    train_samples = cfg.train.train_iters * cfg.train.global_batch_size

    train_loader, _, _ = build_megatron_mimo_data_loaders(
        cfg=cfg,
        train_state=train_state,
        megatron_mimo_provider=cfg.dataset,
        train_samples=max(train_samples, 100),
        valid_samples=0,
        test_samples=0,
    )
    pixel_dtype = torch.bfloat16 if getattr(cfg.model, "bf16", False) else torch.float32
    train_iter = _tracing_wrap_iter(train_loader, pixel_dtype=pixel_dtype) if train_loader is not None else None
    return train_iter, None


def _build_traceable_mock_provider() -> _TraceableMockProvider:
    """Like ``_build_mock_data_provider`` but returns a provider whose datasets
    include each sample's global index."""
    provider = _TraceableMockProvider(
        seq_length=_SEQ_LENGTH,
        processor_paths={},
        tokenizer_path="gpt2",
        special_token_ids={"vision": _SPECIAL_TOKEN_ID},
        encoder_seq_lengths={"vision": _ENCODER_SEQ_LEN},
        modality_configs={"vision": {"type": "image", "width": _IMG_SIZE, "height": _IMG_SIZE}},
    )
    provider.drop_last = True
    object.__setattr__(provider, "_processors", {"vision": _CLIPImageProcessor()})
    return provider


def _build_resume_config(
    parallelism_config: MegatronMIMOParallelismConfig,
    *,
    train_iters: int,
    save_interval: int,
    ckpt_save_dir: str,
    ckpt_load_dir: str | None = None,
    micro_batch_size: int = 1,
    global_batch_size: int = 1,
    fully_parallel_save: bool = True,
    dtype: str = "bf16",
    per_token_loss: bool = False,
    save_rng: bool = False,
    dropout: float = 0.0,
    freeze_language_model: bool = False,
    freeze_modality_encoders: dict[str, bool] | None = None,
    freeze_modality_projections: dict[str, bool] | None = None,
) -> ConfigContainer:
    """Config builder for the checkpoint-resume L2 test.

    Mirrors ``_build_config`` but wires the save/load directory into
    ``CheckpointConfig`` and uses the traceable mock data provider.

    ``micro_batch_size``/``global_batch_size`` default to the symmetric-shape
    values; the heterogeneous-shape resume test passes 2/2 because encoder DP=2
    requires MBS divisible by 2.

    ``dtype="fp32"`` switches both the model TransformerConfigs and the
    optimizer to fp32 — the heterogeneous parity test needs the noise floor
    well below the optimizer-state-reset signature so the parity tolerance
    actually discriminates the two.

    ``per_token_loss=True`` is required for heterogeneous DP layouts so the
    run uses the production-supported pure-SUM DDP plus global token
    denominator path.
    """
    language_model_spec, modality_submodules_spec, special_token_ids = _build_model_specs(
        dtype=dtype,
        per_token_loss=per_token_loss,
        dropout=dropout,
    )

    megatron_mimo_provider = MegatronMIMOProvider(
        language_model_spec=language_model_spec,
        modality_submodules_spec=modality_submodules_spec,
        special_token_ids=special_token_ids,
        megatron_mimo_parallelism_config=parallelism_config,
        topology={"vision": ["language"], "language": []},
        bf16=(dtype == "bf16"),
        fp16=False,
        freeze_language_model=freeze_language_model,
        freeze_modality_encoders=freeze_modality_encoders or {},
        freeze_modality_projections=freeze_modality_projections or {},
    )

    train_cfg = TrainingConfig(
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        train_iters=train_iters,
    )
    train_cfg.num_microbatches = 1

    opt_config = OptimizerConfig(
        bf16=(dtype == "bf16"),
        use_distributed_optimizer=True,
        lr=1e-4,
        min_lr=0.0,
    )

    ckpt_cfg = CheckpointConfig(
        save_interval=save_interval,
        save=ckpt_save_dir,
        ckpt_format="torch_dist",
        fully_parallel_save=fully_parallel_save,
        dist_ckpt_optim_fully_reshardable=True,
        save_rng=save_rng,
    )
    if ckpt_load_dir is not None:
        ckpt_cfg.load = ckpt_load_dir

    return ConfigContainer(
        train=train_cfg,
        model=megatron_mimo_provider,
        optimizer=opt_config,
        scheduler=SchedulerConfig(start_weight_decay=0.0, end_weight_decay=0.0),
        dataset=_build_traceable_mock_provider(),
        logger=LoggerConfig(),
        tokenizer=TokenizerConfig(),
        checkpoint=ckpt_cfg,
    )


def _run_checkpoint_resume_param_parity(
    *,
    tmp_path,
    parallelism_config: MegatronMIMOParallelismConfig,
    checkpoint_prefix: str,
    micro_batch_size: int,
    global_batch_size: int,
    fully_parallel_save: bool,
    per_token_loss: bool,
    dropout: float,
    freeze_language_model: bool = False,
    freeze_modality_encoders: dict[str, bool] | None = None,
    freeze_modality_projections: dict[str, bool] | None = None,
) -> None:
    """Compare a continuous run against save+resume with RNG state restored.

    A continuous run trains ``total_steps`` and captures the final parameters.
    A save run trains ``save_steps`` and writes a checkpoint. A resume run loads
    that checkpoint and trains to ``total_steps``. With ``save_rng=True`` and
    dropout enabled, the resume run only matches the continuous run if the
    checkpoint restores the same CUDA RNG stream(s) that the continuous run
    would have used for later steps.
    """
    from megatron.bridge.training.state import GlobalState

    save_steps = 3
    total_steps = 6
    dtype = "fp32"

    # Use tmp_path on rank 0 and broadcast so all ranks share the same dirs.
    # The continuous run needs its own dir even though it never saves, because
    # CheckpointConfig.save is a required field; setting save_interval >
    # train_iters means no actual write.
    rank_zero_dirs = (
        [str(tmp_path / f"{checkpoint_prefix}_continuous"), str(tmp_path / f"{checkpoint_prefix}_resume")]
        if dist.get_rank() == 0
        else [None, None]
    )
    dist.broadcast_object_list(rank_zero_dirs, src=0)
    continuous_dir, resume_dir = rank_zero_dirs

    # Continuous run, no save: parity baseline.
    _RESUME_TEST_CONSUMED_INDICES.clear()
    continuous_params: dict = {}
    cfg_continuous = _build_resume_config(
        parallelism_config,
        train_iters=total_steps,
        # save_interval > train_iters means no checkpoint write during the
        # run. The continuous run must not leave a checkpoint that would mask the resume run's
        # actual load behavior.
        save_interval=total_steps + 100,
        ckpt_save_dir=continuous_dir,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        fully_parallel_save=fully_parallel_save,
        dtype=dtype,
        per_token_loss=per_token_loss,
        save_rng=True,
        dropout=dropout,
        freeze_language_model=freeze_language_model,
        freeze_modality_encoders=freeze_modality_encoders,
        freeze_modality_projections=freeze_modality_projections,
    )
    state_continuous = GlobalState()
    with _capture_post_step_params_into(continuous_params):
        pretrain_megatron_mimo(
            cfg=cfg_continuous,
            forward_step_func=megatron_mimo_forward_step,
            build_data_iterators_fn=_build_tracing_data_iterators,
            global_state=state_continuous,
        )
    continuous_indices = list(_RESUME_TEST_CONSUMED_INDICES)
    assert state_continuous.train_state.step == total_steps
    assert len(continuous_params) > 0, "continuous run captured no parameters"
    dist.barrier()

    # Save run: train SAVE_STEPS iters, save checkpoint.
    _RESUME_TEST_CONSUMED_INDICES.clear()
    cfg_save = _build_resume_config(
        parallelism_config,
        train_iters=save_steps,
        save_interval=save_steps,
        ckpt_save_dir=resume_dir,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        fully_parallel_save=fully_parallel_save,
        dtype=dtype,
        per_token_loss=per_token_loss,
        save_rng=True,
        dropout=dropout,
        freeze_language_model=freeze_language_model,
        freeze_modality_encoders=freeze_modality_encoders,
        freeze_modality_projections=freeze_modality_projections,
    )
    # The save run intentionally stops early to produce the checkpoint, but its LR
    # schedule must match the first ``save_steps`` iterations of the continuous
    # ``total_steps`` run. Otherwise the saved weights are already on a
    # different optimizer trajectory before resume starts.
    cfg_save.scheduler.lr_decay_iters = total_steps
    state_save = GlobalState()
    pretrain_megatron_mimo(
        cfg=cfg_save,
        forward_step_func=megatron_mimo_forward_step,
        build_data_iterators_fn=_build_tracing_data_iterators,
        global_state=state_save,
    )
    save_indices = list(_RESUME_TEST_CONSUMED_INDICES)
    saved_consumed = state_save.train_state.consumed_train_samples
    assert state_save.train_state.step == save_steps
    assert saved_consumed > 0
    dist.barrier()

    # Resume run: load checkpoint, train to TOTAL_STEPS.
    _RESUME_TEST_CONSUMED_INDICES.clear()
    resume_params: dict = {}
    cfg_resume = _build_resume_config(
        parallelism_config,
        train_iters=total_steps,
        save_interval=total_steps + 100,
        ckpt_save_dir=resume_dir,
        ckpt_load_dir=resume_dir,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        fully_parallel_save=fully_parallel_save,
        dtype=dtype,
        per_token_loss=per_token_loss,
        save_rng=True,
        dropout=dropout,
        freeze_language_model=freeze_language_model,
        freeze_modality_encoders=freeze_modality_encoders,
        freeze_modality_projections=freeze_modality_projections,
    )
    cfg_resume.scheduler.override_opt_param_scheduler = True

    state_resume = GlobalState()
    with _capture_post_step_params_into(resume_params):
        pretrain_megatron_mimo(
            cfg=cfg_resume,
            forward_step_func=megatron_mimo_forward_step,
            build_data_iterators_fn=_build_tracing_data_iterators,
            global_state=state_resume,
        )
    resume_indices = list(_RESUME_TEST_CONSUMED_INDICES)

    assert state_resume.train_state.step == total_steps, (
        f"Step continuity broken: resume run ended at step={state_resume.train_state.step}, expected {total_steps}"
    )

    expected_consumed = saved_consumed + (total_steps - save_steps) * cfg_resume.train.global_batch_size
    assert state_resume.train_state.consumed_train_samples == expected_consumed, (
        f"consumed_train_samples not restored correctly: save run saved {saved_consumed}, "
        f"resume run ended at {state_resume.train_state.consumed_train_samples}, "
        f"expected {expected_consumed}"
    )

    overlap = set(save_indices) & set(resume_indices)
    assert not overlap, (
        f"Resume regression: resumed loader re-consumed samples {sorted(overlap)} "
        f"(save run saw {save_indices}, resume run saw {resume_indices})"
    )

    # The continuous run saw the whole sample sequence; save + resume must replay the
    # exact same ordered trajectory. Set equality is insufficient because
    # different sample order changes the optimizer trajectory.
    assert continuous_indices == save_indices + resume_indices, (
        f"Continuous-run indices {continuous_indices} != save {save_indices} + resume {resume_indices}; "
        f"resumed dataloader saw a different sample trajectory than the continuous run."
    )

    assert continuous_params.keys() == resume_params.keys(), (
        f"Parameter key sets diverge between continuous and resume runs — set difference: "
        f"{set(continuous_params) ^ set(resume_params)}"
    )

    # With fp32 and restored RNG, continuous and resume runs should match well below
    # the optimizer-state-reset signature (~1e-3). If RNG restore is wrong,
    # dropout masks diverge and this blows past the tolerance.
    atol = 1e-5
    max_diff = 0.0
    worst_param = None
    for name in continuous_params:
        p0 = continuous_params[name]
        p2 = resume_params[name]
        assert p0.shape == p2.shape, f"shape mismatch on {name}: {p0.shape} vs {p2.shape}"
        diff = (p0.float() - p2.float()).abs().max().item()
        if diff > max_diff:
            max_diff = diff
            worst_param = name

    assert max_diff < atol, (
        f"Parameter parity violated: continuous-run vs resumed-run max abs diff "
        f"{max_diff} on '{worst_param}' exceeds {atol}. The resumed model trajectory "
        f"diverged from the continuous baseline — likely indicates RNG, optimizer state, "
        f"or weight load failure."
    )


# ── Test class ───────────────────────────────────────────────────────────────


class TestMegatronMIMOTraining:
    """Functional tests for MegatronMIMO heterogeneous parallel training.

    Requires 2 GPUs. Run with:
        torchrun --nproc_per_node=2 -m pytest -v -s -x \\
            tests/functional_tests/test_groups/training/megatron_mimo/test_pretrain_megatron_mimo.py
    """

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize("deterministic", [False, True], ids=["default", "deterministic"])
    def test_megatron_mimo_tp1_both(self, deterministic):
        """Smoke test: MegatronMIMO training with TP=1 for both LLM and vision.

        LLM on rank 0 (TP=1, DP=1), vision on rank 1 (TP=1, DP=1).
        Trains for 5 iterations with synthetic data and verifies completion.

        Parametrized over the ``--deterministic`` code path exposed by
        ``examples/models/megatron_mimo/megatron_mimo_training_llava.py`` to
        guard against regressions in the deterministic config knobs (FP32
        dtypes, unfused attention, deterministic_mode, recompute, fp32 grad
        reduction, and process-wide torch deterministic algorithms).
        """
        initialize_distributed()

        world_size = dist.get_world_size()
        if world_size != 2:
            pytest.skip(f"MegatronMIMO test requires exactly 2 GPUs, got {world_size}")

        # Monkey-patch: report_theoretical_memory crashes on MegatronMIMO models
        # because cfg.model is MegatronMIMOProvider (no kv_channels).
        import megatron.bridge.training.utils.train_utils as _tu

        _tu.report_theoretical_memory = lambda *a, **kw: None

        par_cfg = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                    data_parallel_size=1,
                    rank_offset=0,
                ),
                "vision": ModuleParallelismConfig(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                    data_parallel_size=1,
                    rank_offset=1,
                ),
            },
        )

        cfg = _build_config(par_cfg, deterministic=deterministic)

        # Process-wide torch knobs and env vars the --deterministic flag flips.
        # Toggle in a try/finally because they leak across pytest cases sharing
        # this process. The env vars in particular are required:
        #   - NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 is checked by Transformer
        #     Engine's TEDotProductAttention.__init__ when deterministic_mode
        #     is set, and would raise otherwise.
        #   - CUBLAS_WORKSPACE_CONFIG=:4096:8 is required by
        #     torch.use_deterministic_algorithms(True) for some cuBLAS ops.
        _DET_ENV = {
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT": "0",
        }
        prev_use_deterministic = torch.are_deterministic_algorithms_enabled()
        prev_cudnn_benchmark = torch.backends.cudnn.benchmark
        prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        prev_env = {k: os.environ.get(k) for k in _DET_ENV}
        if deterministic:
            for k, v in _DET_ENV.items():
                os.environ[k] = v
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        try:
            pretrain_megatron_mimo(
                cfg=cfg,
                forward_step_func=megatron_mimo_forward_step,
                build_data_iterators_fn=_build_data_iterators,
            )
        finally:
            if deterministic:
                torch.use_deterministic_algorithms(prev_use_deterministic)
                torch.backends.cudnn.benchmark = prev_cudnn_benchmark
                torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
                torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
                for k, v in prev_env.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

    @pytest.mark.run_only_on("GPU")
    def test_megatron_mimo_colocated_equal_dp_smoke(self):
        """Smoke test: colocated MegatronMIMO with symmetric DP/TP, LLM PP=1.

        Both modules share rank range [0, 2): TP=1, DP=2 each. Symmetric TP/DP
        passes the asymmetric-DP / asymmetric-TP validator guards. The
        ColocatedBridgeCommunicator runs in pass-through mode (DP equal, TP
        equal). Trains for 5 iters with synthetic data.

        First L2 test for the colocated path. Catches:
        - R10: DDP wrap skipped for a module on a colocated rank → would
          raise during gradient finalization.
        - Path coverage: provider grid construction, setup_megatron_mimo's
          colocated-canonical PG bridge, train_megatron_mimo's threaded
          active_module_name/local_pg_collection, MimoModel with
          cp_group/tp_group from the LLM pg_collection.

        Numerical correctness (R3 wrong DP reduction group, R4 wrong loss
        reduction group) is deferred to a follow-up oracle test that builds
        a non-distributed reference and compares per-rank outputs.
        """
        initialize_distributed()

        world_size = dist.get_world_size()
        if world_size != 2:
            pytest.skip(f"MegatronMIMO test requires exactly 2 GPUs, got {world_size}")

        # Monkey-patch: report_theoretical_memory crashes on MegatronMIMO models.
        import megatron.bridge.training.utils.train_utils as _tu

        _tu.report_theoretical_memory = lambda *a, **kw: None

        par_cfg = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                    data_parallel_size=2,
                    rank_offset=0,
                ),
                "vision": ModuleParallelismConfig(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                    data_parallel_size=2,
                    rank_offset=0,
                ),
            },
        )

        # MBS must be divisible by every module's DP (here: 2). GBS = MBS *
        # num_microbatches because the sampler uses dp_size=1 per the
        # MegatronMIMO data contract — every data-loading rank sees the same
        # global micro-batch and slices per-module at forward time.
        cfg = _build_config(par_cfg, micro_batch_size=2, global_batch_size=2)

        pretrain_megatron_mimo(
            cfg=cfg,
            forward_step_func=megatron_mimo_forward_step,
            build_data_iterators_fn=_build_data_iterators,
        )

    @pytest.mark.run_only_on("GPU")
    def test_megatron_mimo_colocated_heterogeneous_tp_dp_smoke(self):
        """Smoke test: colocated MegatronMIMO at heterogeneous shape ``enc(TP=1,DP=2) × llm(TP=2,DP=1)``.

        Mirrors ``test_megatron_mimo_colocated_equal_dp_smoke`` but exercises
        the heterogeneous TP/DP path: encoder fully DP-parallel, LLM fully
        TP-parallel, both colocated on the same 2 ranks. ``_TRAIN_ITERS`` iters
        with synthetic data, MBS=2/GBS=2 (encoder DP=2 requires MBS divisibility).

        Cheapest end-to-end signal that colocated heterogeneous TP/DP is wired up correctly. Failures
        here point at, in order:
        - ``align_embeddings_by_token_positions`` shape mismatch → per-key
          per-module slicing is wrong.
        - DDP wrap / MimoOptimizer construction errors → provider's
          heterogeneous-TP path is wrong.
        - Crash inside ``ColocatedBridgeCommunicator`` fan-in/fan-out adjoint
          → mcore-side communicator regression.

        Asserts the training loop ran the expected number of iterations and
        consumed the expected number of samples. Numerical correctness (loss
        sanity, gradient parity) is the L2a oracle's job — 5 iters of a
        randomly-initialized small model don't expose a useful loss threshold.
        """
        from megatron.bridge.training.state import GlobalState

        initialize_distributed()

        world_size = dist.get_world_size()
        if world_size != 2:
            pytest.skip(f"MegatronMIMO test requires exactly 2 GPUs, got {world_size}")

        # Monkey-patch: report_theoretical_memory crashes on MegatronMIMO models.
        import megatron.bridge.training.utils.train_utils as _tu

        _tu.report_theoretical_memory = lambda *a, **kw: None

        par_cfg = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(
                    tensor_model_parallel_size=2,
                    pipeline_model_parallel_size=1,
                    data_parallel_size=1,
                    rank_offset=0,
                ),
                "vision": ModuleParallelismConfig(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                    data_parallel_size=2,
                    rank_offset=0,
                ),
            },
        )

        # Encoder DP=2 → MBS must be divisible by 2; sampler uses dp_size=1 per
        # the MegatronMIMO contract, so GBS = MBS.
        global_batch_size = 2
        # ``per_token_loss=True`` is required to exercise the production-supported
        # heterogeneous-DP path: pure-SUM DDP + ``1/N_global`` finalize math
        # for MegatronMIMO. Without it, mcore's schedule applies its own local averaging
        # and ``finalize_model_grads_multimodule`` skips the per-token divisor —
        # a code path that doesn't match what production runs at this shape.
        cfg = _build_config(
            par_cfg,
            micro_batch_size=2,
            global_batch_size=global_batch_size,
            per_token_loss=True,
        )

        state = GlobalState()
        pretrain_megatron_mimo(
            cfg=cfg,
            forward_step_func=megatron_mimo_forward_step,
            build_data_iterators_fn=_build_data_iterators,
            global_state=state,
        )

        assert state.train_state.step == _TRAIN_ITERS, (
            f"Training loop did not reach the configured train_iters: "
            f"step={state.train_state.step}, expected {_TRAIN_ITERS}. "
            f"A truncated loop is a silent failure mode (e.g. checkpoint exit-on-load, "
            f"early break in train_megatron_mimo) that pure no-crash smoke would miss."
        )
        assert state.train_state.consumed_train_samples == _TRAIN_ITERS * global_batch_size, (
            f"consumed_train_samples mismatch: got {state.train_state.consumed_train_samples}, "
            f"expected {_TRAIN_ITERS * global_batch_size}. Indicates the training loop "
            f"either skipped iterations or accounted samples wrong."
        )

    @pytest.mark.run_only_on("GPU")
    def test_megatron_mimo_colocated_language_cp2_smoke(self, monkeypatch):
        """Smoke test: colocated MegatronMIMO with language CP=2.

        Minimal 2-GPU CP shape:

        * vision:   TP=1, CP=1, PP=1, DP=2
        * language: TP=1, CP=2, PP=1, DP=1

        This keeps Bridge data slicing DP-only while MCore's PartitionAdapter
        shards the language sequence. ``MIMO_CHECK_DATA_ALIGNMENT`` stays on so
        the colocated data guard confirms CP does not affect the batch dimension.
        """
        from megatron.bridge.training import megatron_mimo_step as _mms
        from megatron.bridge.training.state import GlobalState

        initialize_distributed()

        world_size = dist.get_world_size()
        if world_size != 2:
            pytest.skip(f"MegatronMIMO test requires exactly 2 GPUs, got {world_size}")

        # Monkey-patch: report_theoretical_memory crashes on MegatronMIMO models.
        import megatron.bridge.training.utils.train_utils as _tu

        _tu.report_theoretical_memory = lambda *a, **kw: None

        monkeypatch.setenv("MIMO_CHECK_DATA_ALIGNMENT", "true")
        monkeypatch.setenv("MIMO_CHECK_DATA_ALIGNMENT_STEPS", "1")
        _mms._DATA_ALIGNMENT_CHECK_COUNT = 0

        par_cfg = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                    context_parallel_size=2,
                    data_parallel_size=1,
                    rank_offset=0,
                ),
                "vision": ModuleParallelismConfig(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                    context_parallel_size=1,
                    data_parallel_size=2,
                    rank_offset=0,
                ),
            },
        )

        # Vision DP=2 requires MBS divisible by 2. Sequence length is 256,
        # divisible by 2 * language_cp = 4 for MCore PartitionAdapter.
        global_batch_size = 2
        cfg = _build_config(
            par_cfg,
            micro_batch_size=2,
            global_batch_size=global_batch_size,
            per_token_loss=True,
        )

        state = GlobalState()
        pretrain_megatron_mimo(
            cfg=cfg,
            forward_step_func=megatron_mimo_forward_step,
            build_data_iterators_fn=_build_data_iterators,
            global_state=state,
        )

        assert state.train_state.step == _TRAIN_ITERS, (
            f"Training loop did not reach the configured train_iters: "
            f"step={state.train_state.step}, expected {_TRAIN_ITERS}."
        )
        assert state.train_state.consumed_train_samples == _TRAIN_ITERS * global_batch_size, (
            f"consumed_train_samples mismatch: got {state.train_state.consumed_train_samples}, "
            f"expected {_TRAIN_ITERS * global_batch_size}."
        )
        assert _mms._DATA_ALIGNMENT_CHECK_COUNT == 1

    @pytest.mark.run_only_on("GPU")
    def test_megatron_mimo_colocated_language_pp_smoke(self):
        """Smoke test: colocated MegatronMIMO with language PP=2.

        This is the minimal end-to-end signal for the colocated language-PP
        adapter: encoder forward over colocated ranks, language non-interleaved
        pipeline schedule, encoder backward, and deferred multimodule gradient
        finalization. The shape fits on two GPUs:

        * vision:   TP=1, DP=2, PP=1
        * language: TP=1, DP=1, PP=2

        Validation/eval stays disabled; this test only covers training.
        """
        from megatron.bridge.training.state import GlobalState

        initialize_distributed()

        world_size = dist.get_world_size()
        if world_size != 2:
            pytest.skip(f"MegatronMIMO test requires exactly 2 GPUs, got {world_size}")

        # Monkey-patch: report_theoretical_memory crashes on MegatronMIMO models.
        import megatron.bridge.training.utils.train_utils as _tu

        _tu.report_theoretical_memory = lambda *a, **kw: None

        par_cfg = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=2,
                    data_parallel_size=1,
                    rank_offset=0,
                ),
                "vision": ModuleParallelismConfig(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                    data_parallel_size=2,
                    rank_offset=0,
                ),
            },
        )

        # Vision DP=2 requires MBS divisible by 2. The sampler uses dp_size=1
        # for MegatronMIMO, so GBS=MBS for this one-microbatch smoke.
        global_batch_size = 2
        cfg = _build_config(
            par_cfg,
            micro_batch_size=2,
            global_batch_size=global_batch_size,
            per_token_loss=True,
        )

        state = GlobalState()
        pretrain_megatron_mimo(
            cfg=cfg,
            forward_step_func=megatron_mimo_forward_step,
            build_data_iterators_fn=_build_data_iterators,
            global_state=state,
        )

        assert state.train_state.step == _TRAIN_ITERS, (
            f"Training loop did not reach the configured train_iters: "
            f"step={state.train_state.step}, expected {_TRAIN_ITERS}."
        )
        assert state.train_state.consumed_train_samples == _TRAIN_ITERS * global_batch_size, (
            f"consumed_train_samples mismatch: got {state.train_state.consumed_train_samples}, "
            f"expected {_TRAIN_ITERS * global_batch_size}."
        )

    @pytest.mark.run_only_on("GPU")
    def test_megatron_mimo_checkpoint_resume_dp1_both(self, tmp_path):
        """Checkpoint-resume baseline for non-colocated MegatronMIMO.

        Uses ``save_rng=True`` with dropout enabled. This is the explicit
        no-regression baseline: non-colocated MIMO should continue to use the
        standard singleton CUDA RNG checkpoint path and match a continuous run.
        """
        initialize_distributed()

        world_size = dist.get_world_size()
        if world_size != 2:
            pytest.skip(f"MegatronMIMO test requires exactly 2 GPUs, got {world_size}")

        # Monkey-patch: report_theoretical_memory crashes on MegatronMIMO models.
        import megatron.bridge.training.utils.train_utils as _tu

        _tu.report_theoretical_memory = lambda *a, **kw: None

        par_cfg = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                    data_parallel_size=1,
                    rank_offset=0,
                ),
                "vision": ModuleParallelismConfig(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                    data_parallel_size=1,
                    rank_offset=1,
                ),
            },
        )

        _run_checkpoint_resume_param_parity(
            tmp_path=tmp_path,
            parallelism_config=par_cfg,
            checkpoint_prefix="ckpt_non_colocated_rng",
            micro_batch_size=1,
            global_batch_size=1,
            fully_parallel_save=True,
            per_token_loss=False,
            dropout=0.1,
        )

    @pytest.mark.run_only_on("GPU")
    def test_megatron_mimo_checkpoint_resume_frozen_language_save_rng(self, tmp_path):
        """Non-colocated checkpoint-resume when rank 0 owns no active scheduler.

        The language module runs on rank 0 and is frozen; the vision module on
        rank 1 remains trainable. This reproduces the L3 failure where checkpoint
        save omitted ``opt_param_scheduler`` because rank 0 had no optimizer
        param groups, while resume on the trainable nonzero rank expected the
        shared scheduler state.
        """
        initialize_distributed()

        world_size = dist.get_world_size()
        if world_size != 2:
            pytest.skip(f"MegatronMIMO test requires exactly 2 GPUs, got {world_size}")

        import megatron.bridge.training.utils.train_utils as _tu

        _tu.report_theoretical_memory = lambda *a, **kw: None

        par_cfg = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                    data_parallel_size=1,
                    rank_offset=0,
                ),
                "vision": ModuleParallelismConfig(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                    data_parallel_size=1,
                    rank_offset=1,
                ),
            },
        )

        _run_checkpoint_resume_param_parity(
            tmp_path=tmp_path,
            parallelism_config=par_cfg,
            checkpoint_prefix="ckpt_non_colocated_frozen_language_rng",
            micro_batch_size=1,
            global_batch_size=1,
            fully_parallel_save=True,
            per_token_loss=False,
            dropout=0.1,
            freeze_language_model=True,
        )

    @pytest.mark.run_only_on("GPU")
    def test_megatron_mimo_colocated_symmetric_checkpoint_resume_save_rng(self, tmp_path):
        """Checkpoint-resume for symmetric colocated MIMO in singleton RNG mode.

        Both modules share ranks [0, 2) with identical TP. ``save_rng=True``
        and dropout enabled verify that the migration away from per-module RNG
        plumbing for symmetric colocated layouts works end-to-end.
        """
        initialize_distributed()

        world_size = dist.get_world_size()
        if world_size != 2:
            pytest.skip(f"MegatronMIMO test requires exactly 2 GPUs, got {world_size}")

        # Monkey-patch: report_theoretical_memory crashes on MegatronMIMO models.
        import megatron.bridge.training.utils.train_utils as _tu

        _tu.report_theoretical_memory = lambda *a, **kw: None

        par_cfg = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                    data_parallel_size=2,
                    rank_offset=0,
                ),
                "vision": ModuleParallelismConfig(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                    data_parallel_size=2,
                    rank_offset=0,
                ),
            },
        )

        _run_checkpoint_resume_param_parity(
            tmp_path=tmp_path,
            parallelism_config=par_cfg,
            checkpoint_prefix="ckpt_symmetric_colocated_rng",
            micro_batch_size=2,
            global_batch_size=2,
            fully_parallel_save=True,
            per_token_loss=False,
            dropout=0.1,
        )

    @pytest.mark.run_only_on("GPU")
    def test_megatron_mimo_colocated_heterogeneous_checkpoint_resume_2gpu(self, tmp_path):
        """Same-layout checkpoint resume at heterogeneous shape ``enc(TP=1,DP=2) × llm(TP=2,DP=1)``.

        Uses ``save_rng=True`` with dropout enabled. This is the load-bearing
        end-to-end proof for per-module CUDA RNG checkpoint save/load: the
        encoder and language module use different TP/DP layouts on the same
        physical ranks, so the resume run only matches the continuous run if
        the module-keyed RNG tracker states are saved and restored correctly.
        """
        initialize_distributed()

        world_size = dist.get_world_size()
        if world_size != 2:
            pytest.skip(f"MegatronMIMO test requires exactly 2 GPUs, got {world_size}")

        # Monkey-patch: report_theoretical_memory crashes on MegatronMIMO models.
        import megatron.bridge.training.utils.train_utils as _tu

        _tu.report_theoretical_memory = lambda *a, **kw: None

        par_cfg = MegatronMIMOParallelismConfig(
            module_parallelisms={
                "language": ModuleParallelismConfig(
                    tensor_model_parallel_size=2,
                    pipeline_model_parallel_size=1,
                    data_parallel_size=1,
                    rank_offset=0,
                ),
                "vision": ModuleParallelismConfig(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                    data_parallel_size=2,
                    rank_offset=0,
                ),
            },
        )

        # fully_parallel_save=True is incompatible with the prepend-axis vs
        # encoder-DP sharding interaction at the heterogeneous shape — the
        # validator flags duplicate replica claims on encoder shards because the
        # dp_cp_group passed to FullyParallelSaveStrategyWrapper is the language
        # pg's. mcore special-cases MegatronMIMO + fully_parallel_save=False to
        # skip the integrity check; per-module sharding strategy is a follow-up.
        fully_parallel_save = False

        _run_checkpoint_resume_param_parity(
            tmp_path=tmp_path,
            parallelism_config=par_cfg,
            checkpoint_prefix="ckpt_heterogeneous_rng",
            micro_batch_size=2,
            global_batch_size=2,
            fully_parallel_save=fully_parallel_save,
            per_token_loss=True,
            dropout=0.1,
        )
