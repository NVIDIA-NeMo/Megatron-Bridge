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
        tests/functional_tests/test_groups/training/test_pretrain_megatron_mimo.py
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
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


def _make_vision_config() -> TransformerConfig:
    cfg = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        ffn_hidden_size=256,
        num_attention_heads=4,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        variable_seq_lengths=True,
        moe_token_dispatcher_type="alltoall",
    )
    cfg.add_bias_linear = True
    cfg.add_qkv_bias = True
    cfg.hidden_dropout = 0.0
    cfg.attention_dropout = 0.0
    cfg.gated_linear_unit = False
    cfg.layernorm_zero_centered_gamma = False
    cfg.apply_query_key_layer_scaling = False
    cfg.bias_activation_fusion = False
    cfg.bias_dropout_fusion = False
    cfg.attention_softmax_in_fp32 = True
    cfg.normalization = "LayerNorm"
    cfg.apply_rope_fusion = False
    return cfg


def _make_language_config() -> TransformerConfig:
    return TransformerConfig(
        num_layers=2,
        hidden_size=64,
        ffn_hidden_size=256,
        num_attention_heads=4,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        variable_seq_lengths=True,
        moe_token_dispatcher_type="alltoall",
        cross_entropy_loss_fusion=True,
    )


def _build_model_specs():
    """Return (language_model_spec, modality_submodules_spec, special_token_ids)."""
    vision_encoder = ModuleSpec(
        module=CLIPViTModel,
        params={
            "transformer_config": _make_vision_config(),
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
            "config": _make_language_config(),
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


def _wrap_iter(loader_iter):
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
                mi["vision"] = {"clip": {"x": pv.to(torch.bfloat16)}}

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

    train_iter = _wrap_iter(train_loader) if train_loader is not None else None
    return train_iter, None


# ── Config builder ───────────────────────────────────────────────────────────


def _build_config(
    parallelism_config: MegatronMIMOParallelismConfig,
    train_iters: int = _TRAIN_ITERS,
    *,
    micro_batch_size: int = 1,
    global_batch_size: int = 1,
) -> ConfigContainer:
    language_model_spec, modality_submodules_spec, special_token_ids = _build_model_specs()

    megatron_mimo_provider = MegatronMIMOProvider(
        language_model_spec=language_model_spec,
        modality_submodules_spec=modality_submodules_spec,
        special_token_ids=special_token_ids,
        megatron_mimo_parallelism_config=parallelism_config,
        topology={"vision": ["language"], "language": []},
    )
    if not hasattr(megatron_mimo_provider, "num_moe_experts"):
        megatron_mimo_provider.num_moe_experts = None

    train_cfg = TrainingConfig(
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        train_iters=train_iters,
    )
    train_cfg.num_microbatches = 1

    opt_config = OptimizerConfig(
        bf16=True,
        use_distributed_optimizer=True,
        lr=1e-4,
        min_lr=0.0,
    )

    return ConfigContainer(
        train=train_cfg,
        model=megatron_mimo_provider,
        optimizer=opt_config,
        scheduler=SchedulerConfig(start_weight_decay=0.0, end_weight_decay=0.0),
        dataset=_build_mock_data_provider(),
        logger=LoggerConfig(),
        tokenizer=TokenizerConfig(),
        checkpoint=CheckpointConfig(),
    )


# ── Index tracing for checkpoint-resume test ─────────────────────────────────


_RESUME_TEST_CONSUMED_INDICES: list[int] = []


class _IndexTaggedDataset(torch.utils.data.Dataset):
    """Wrap a Dataset so each sample's ``input_ids[-1]`` holds its global index.

    Used by the checkpoint-resume L2 test to trace which samples were consumed
    across save/resume phases. Read ``batch["input_ids"][:, -1]`` to get the
    indices fed into a step.

    The tag goes in the *last* text position (not position 0) because positions
    [0, _ENCODER_SEQ_LEN) hold vision placeholder tokens that MimoModel's
    ``align_embeddings_by_token_positions`` counts against the vision encoder's
    output (mismatch → ValueError).
    """

    def __init__(self, inner: torch.utils.data.Dataset):
        self._inner = inner

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx):
        sample = self._inner[idx]
        sample["input_ids"][-1] = idx
        return sample


class _TraceableMockProvider(MockMegatronMIMOProvider):
    """Test-only provider that wraps MockMegatronMIMOProvider's datasets with
    ``_IndexTaggedDataset`` so samples carry their global index."""

    def build_datasets(self, context):
        train, valid, test = super().build_datasets(context)
        wrap = lambda ds: _IndexTaggedDataset(ds) if ds is not None else None
        return wrap(train), wrap(valid), wrap(test)


def _tracing_wrap_iter(loader_iter):
    """Like ``_wrap_iter`` but records batch sample-indices into the module-level
    ``_RESUME_TEST_CONSUMED_INDICES`` list before yielding."""
    for batch in _wrap_iter(loader_iter):
        # input_ids was moved to cuda by _wrap_iter.
        _RESUME_TEST_CONSUMED_INDICES.extend(batch["input_ids"][:, -1].cpu().tolist())
        yield batch


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
    train_iter = _tracing_wrap_iter(train_loader) if train_loader is not None else None
    return train_iter, None


def _build_traceable_mock_provider() -> _TraceableMockProvider:
    """Like ``_build_mock_data_provider`` but returns a provider whose datasets
    embed each sample's global index in ``input_ids[0]``."""
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
) -> ConfigContainer:
    """Config builder for the checkpoint-resume L2 test.

    Mirrors ``_build_config`` but wires the save/load directory into
    ``CheckpointConfig`` and uses the traceable mock data provider.
    """
    language_model_spec, modality_submodules_spec, special_token_ids = _build_model_specs()

    megatron_mimo_provider = MegatronMIMOProvider(
        language_model_spec=language_model_spec,
        modality_submodules_spec=modality_submodules_spec,
        special_token_ids=special_token_ids,
        megatron_mimo_parallelism_config=parallelism_config,
        topology={"vision": ["language"], "language": []},
    )
    if not hasattr(megatron_mimo_provider, "num_moe_experts"):
        megatron_mimo_provider.num_moe_experts = None

    train_cfg = TrainingConfig(
        micro_batch_size=1,
        global_batch_size=1,
        train_iters=train_iters,
    )
    train_cfg.num_microbatches = 1

    opt_config = OptimizerConfig(
        bf16=True,
        use_distributed_optimizer=True,
        lr=1e-4,
        min_lr=0.0,
    )

    ckpt_cfg = CheckpointConfig(
        save_interval=save_interval,
        save=ckpt_save_dir,
        ckpt_format="torch_dist",
        fully_parallel_save=True,  # llm pp==1 in this test
        dist_ckpt_optim_fully_reshardable=True,
        save_rng=False,  # MegatronMIMO RNG save produces duplicate shard keys; disable
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


# ── Test class ───────────────────────────────────────────────────────────────


class TestMegatronMIMOTraining:
    """Functional tests for MegatronMIMO heterogeneous parallel training.

    Requires 2 GPUs. Run with:
        torchrun --nproc_per_node=2 -m pytest -v -s -x \\
            tests/functional_tests/test_groups/training/test_pretrain_megatron_mimo.py
    """

    @pytest.mark.run_only_on("GPU")
    def test_megatron_mimo_tp1_both(self):
        """Smoke test: MegatronMIMO training with TP=1 for both LLM and vision.

        LLM on rank 0 (TP=1, DP=1), vision on rank 1 (TP=1, DP=1).
        Trains for 5 iterations with synthetic data and verifies completion.
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

        cfg = _build_config(par_cfg)

        pretrain_megatron_mimo(
            cfg=cfg,
            forward_step_func=megatron_mimo_forward_step,
            build_data_iterators_fn=_build_data_iterators,
        )

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
    def test_megatron_mimo_checkpoint_resume_dp1_both(self, tmp_path):
        """Fast checkpoint-resume check (issue #11 regression guard).

        Single torchrun, single process-group init. Runs two ``pretrain_megatron_mimo``
        calls back-to-back: phase 1 trains for ``SAVE_STEPS`` iters and writes a
        checkpoint; phase 2 loads the same checkpoint and trains to ``TOTAL_STEPS``.
        ``_IndexTaggedDataset`` puts each sample's global index into
        ``input_ids[0]``, and ``_tracing_wrap_iter`` records those indices as the
        sampler yields them. Asserts:

        * ``train_state.step`` goes 0 → SAVE_STEPS after phase 1, SAVE_STEPS → TOTAL_STEPS after phase 2
        * ``train_state.consumed_train_samples`` is correctly restored
        * Phase-1 and phase-2 consumed indices are disjoint (the resumed loader
          skips samples already seen before the crash)

        Only needs 2 GPUs and ~30–40s — much faster than the 8 GPU L3 ``dp4_both``
        checkpoint-resume test, and catches the same class of bug every PR.
        """
        from megatron.bridge.training.state import GlobalState

        save_steps = 3
        total_steps = 6

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

        # Use tmp_path on rank 0 and broadcast so all ranks share the same dir.
        ckpt_dir = [str(tmp_path / "ckpt")] if dist.get_rank() == 0 else [None]
        dist.broadcast_object_list(ckpt_dir, src=0)
        ckpt_dir = ckpt_dir[0]

        # ── Phase 1: train SAVE_STEPS iters, save checkpoint ────────────────
        _RESUME_TEST_CONSUMED_INDICES.clear()
        cfg_save = _build_resume_config(
            par_cfg,
            train_iters=save_steps,
            save_interval=save_steps,
            ckpt_save_dir=ckpt_dir,
        )
        state_save = GlobalState()
        pretrain_megatron_mimo(
            cfg=cfg_save,
            forward_step_func=megatron_mimo_forward_step,
            build_data_iterators_fn=_build_tracing_data_iterators,
            global_state=state_save,
        )
        phase1_indices = sorted(set(_RESUME_TEST_CONSUMED_INDICES))
        phase1_consumed = state_save.train_state.consumed_train_samples
        assert state_save.train_state.step == save_steps
        assert phase1_consumed > 0
        dist.barrier()

        # ── Phase 2: resume from checkpoint, train to TOTAL_STEPS ────────────
        _RESUME_TEST_CONSUMED_INDICES.clear()
        cfg_resume = _build_resume_config(
            par_cfg,
            train_iters=total_steps,
            save_interval=total_steps,
            ckpt_save_dir=ckpt_dir,
            ckpt_load_dir=ckpt_dir,
        )
        # Save phase used train_iters=save_steps so the checkpoint's scheduler
        # state doesn't match total_steps; override so the resumed scheduler uses
        # the new values without asserting against the checkpoint.
        cfg_resume.scheduler.override_opt_param_scheduler = True

        state_resume = GlobalState()
        pretrain_megatron_mimo(
            cfg=cfg_resume,
            forward_step_func=megatron_mimo_forward_step,
            build_data_iterators_fn=_build_tracing_data_iterators,
            global_state=state_resume,
        )
        phase2_indices = sorted(set(_RESUME_TEST_CONSUMED_INDICES))

        # Step counter continues from the checkpoint.
        assert state_resume.train_state.step == total_steps, (
            f"Step continuity broken: phase 2 ended at step={state_resume.train_state.step}, expected {total_steps}"
        )

        # consumed_train_samples was restored and then incremented by the extra iters.
        expected_consumed = phase1_consumed + (total_steps - save_steps) * cfg_resume.train.global_batch_size
        assert state_resume.train_state.consumed_train_samples == expected_consumed, (
            f"consumed_train_samples not restored correctly: phase 1 saved {phase1_consumed}, "
            f"phase 2 ended at {state_resume.train_state.consumed_train_samples}, "
            f"expected {expected_consumed}"
        )

        # The data loader honored the restored offset: phase 2's first batch picks
        # up where phase 1 left off, so indices are disjoint.
        overlap = set(phase1_indices) & set(phase2_indices)
        assert not overlap, (
            f"Issue #11 regression: resumed loader re-consumed samples {sorted(overlap)} "
            f"(phase 1 saw {phase1_indices}, phase 2 saw {phase2_indices})"
        )
