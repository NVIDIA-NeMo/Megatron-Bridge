# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.


from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

import torch
import torch.distributed as dist
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig


class CLIPViTNoCLS(CLIPViTModel):
    """CLIPViTModel that drops the CLS token to match HF LLaVA (mm_vision_select_feature='patch')."""

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = super().forward(x, attention_mask=attention_mask)
        return x[:, self.class_token_len :, :]


# ---------------------------------------------------------------------------
# LLaVA model configs (Vicuna-7B + CLIP ViT-L/14 + MLP projection)
# ---------------------------------------------------------------------------

IMAGE_SPECIAL_TOKEN_ID = 32000
VOCAB_SIZE = 32256
CLIP_OUTPUT_DIM = 1024  # CLIP ViT-L/14 hidden size
MAX_SEQ_LENGTH = 4096
_IMG_SIZE = 336
_PATCH_DIM = 14
# CLIP ViT-L/14 @ 336×336: (336/14)^2 = 576 patches (CLS token dropped per HF LLaVA)
_ENCODER_SEQ_LEN = 576


def _make_vision_config() -> TransformerConfig:
    """CLIP ViT-L/14 vision encoder config (23 layers = penultimate layer output per HF LLaVA)."""
    cfg = TransformerConfig(
        num_layers=23,
        hidden_size=1024,
        ffn_hidden_size=4096,
        num_attention_heads=16,
        use_cpu_initialization=True,
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
    # CLIP uses "quick_gelu", not standard gelu
    cfg.activation_func = lambda x: x * torch.sigmoid(1.702 * x)
    cfg.calculate_per_token_loss = True

    return cfg


def _make_language_config() -> TransformerConfig:
    """Vicuna-7B language model config (same arch as Llama-7B)."""
    cfg = TransformerConfig(
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        use_cpu_initialization=True,
    )

    cfg.ffn_hidden_size = 11008
    cfg.activation_func = torch.nn.functional.silu
    cfg.gated_linear_unit = True

    cfg.normalization = "RMSNorm"
    cfg.rms_norm_eps = 1e-5

    cfg.position_embedding_type = "rope"
    cfg.rotary_base = 10000
    cfg.rotary_percent = 1.0

    cfg.seq_length = MAX_SEQ_LENGTH
    cfg.max_position_embeddings = MAX_SEQ_LENGTH

    cfg.attention_dropout = 0.0
    cfg.hidden_dropout = 0.0

    cfg.num_query_groups = 32
    cfg.add_bias_linear = False
    cfg.untie_embeddings_and_output_weights = True

    cfg.bias_activation_fusion = True
    cfg.masked_softmax_fusion = True
    cfg.persist_layer_norm = True
    cfg.bias_dropout_fusion = True
    cfg.apply_rope_fusion = True

    cfg.pipeline_dtype = torch.bfloat16
    cfg.bf16 = True
    cfg.cross_entropy_loss_fusion = True
    cfg.variable_seq_lengths = True
    cfg.calculate_per_token_loss = True

    return cfg


def _make_projection_config(hidden_size: int = 4096) -> TransformerConfig:
    """Vision→language projection MLP config."""
    cfg = TransformerConfig(num_layers=1, hidden_size=hidden_size, num_attention_heads=1, use_cpu_initialization=True)
    cfg.ffn_hidden_size = 4096
    cfg.bias_activation_fusion = True
    cfg.add_bias_linear = True
    cfg.activation_func = torch.nn.functional.gelu
    cfg.calculate_per_token_loss = True

    return cfg


def _build_model_specs():
    """Return (language_model_spec, modality_submodules_spec, special_token_ids)."""
    vision_config = _make_vision_config()
    language_config = _make_language_config()
    projection_config = _make_projection_config(hidden_size=language_config.hidden_size)

    # CLIP ViT-L/14 encoder
    vision_encoder = ModuleSpec(
        module=CLIPViTNoCLS,
        params={
            "transformer_config": vision_config,
            "transformer_layer_spec": get_vit_layer_with_transformer_engine_spec(),
            "patch_dim": _PATCH_DIM,
            "img_h": _IMG_SIZE,
            "img_w": _IMG_SIZE,
        },
    )

    # Vision→language projection MLP
    vision_projection = ModuleSpec(
        module=MultimodalProjector,
        params={
            "config": projection_config,
            "submodules": MLPSubmodules(
                linear_fc1=TEColumnParallelLinear,
                linear_fc2=TERowParallelLinear,
            ),
            "projector_type": "mlp",
            "input_size": CLIP_OUTPUT_DIM,
        },
    )

    vision_submodule_spec = ModuleSpec(
        module=VisionModalitySubmodules,
        params={},
        submodules={
            "encoders": {"clip": vision_encoder},
            "input_projections": [vision_projection],
        },
    )

    language_model_spec = ModuleSpec(
        module=GPTModel,
        params={
            "config": language_config,
            "transformer_layer_spec": get_gpt_layer_with_transformer_engine_spec(),
            "vocab_size": VOCAB_SIZE,
            "max_sequence_length": MAX_SEQ_LENGTH,
            "position_embedding_type": "rope",
        },
    )

    modality_submodules_spec = {"images": vision_submodule_spec}
    special_token_ids = {"images": IMAGE_SPECIAL_TOKEN_ID}
    return language_model_spec, modality_submodules_spec, special_token_ids


# ---------------------------------------------------------------------------
# Per-submodule checkpoint loading
# ---------------------------------------------------------------------------


def _get_pp_layer_offset(module: torch.nn.Module) -> int:
    """Return the global layer offset for a module's PP stage.

    Inspects the first transformer layer's ``layer_number`` attribute (1-based
    global index set by Megatron) and compares it with the local ModuleList
    index to derive the offset.  Returns 0 when PP=1 or for non-transformer
    modules (e.g. the vision encoder).
    """
    decoder = getattr(module, "decoder", None)
    if decoder is None:
        return 0
    layers = getattr(decoder, "layers", None)
    if not layers or len(layers) == 0:
        return 0
    first_layer = layers[0]
    global_layer_number = getattr(first_layer, "layer_number", None)
    if global_layer_number is None:
        return 0
    # layer_number is 1-based; local index is 0-based
    return global_layer_number - 1


def _remap_checkpoint_for_pp(
    state_dict: dict[str, torch.Tensor],
    module: torch.nn.Module,
    layer_offset: int,
) -> dict[str, torch.Tensor]:
    """Remap globally-numbered checkpoint layer keys to local PP stage indices.

    The HF→Megatron converters produce globally-numbered keys
    (``decoder.layers.0`` … ``decoder.layers.31``), but each PP stage stores
    layers in a ``nn.ModuleList`` with 0-based local indices.  For PP stage 1
    with offset=16, checkpoint key ``decoder.layers.16`` must become
    ``decoder.layers.0`` so it matches the module's state dict.

    Non-layer keys (embedding, output_layer, final_layernorm) are passed
    through unchanged, then filtered to keys the module actually owns.
    """
    import re

    module_keys = set(module.state_dict().keys())
    remapped = {}

    for key, value in state_dict.items():
        m = re.match(r"^(decoder\.layers\.)(\d+)(\..*)", key)
        if m:
            global_idx = int(m.group(2))
            local_idx = global_idx - layer_offset
            if local_idx < 0:
                continue  # belongs to an earlier PP stage
            new_key = f"{m.group(1)}{local_idx}{m.group(3)}"
            if new_key in module_keys:
                remapped[new_key] = value
        else:
            # Non-layer key (embedding, output_layer, final_layernorm, etc.)
            if key in module_keys:
                remapped[key] = value

    return remapped


def _load_tp_rank_weights(
    module: torch.nn.Module,
    ckpt_dir: str,
    tp_rank: int,
    label: str,
) -> None:
    """Load per-TP-rank ``.pt`` weights produced by the HF→Megatron converters.

    Both ``convert_hf_clip_to_megatron.py`` and ``convert_hf_llama_to_megatron.py``
    write the same layout::

        {ckpt_dir}/tp_rank_{NN}/model_weights.pt   →  {"model": {key: tensor}}

    When pipeline parallelism (PP) > 1, checkpoint layer keys are globally
    numbered but the module uses local 0-based indices.  We remap
    ``decoder.layers.<global_idx>`` → ``decoder.layers.<local_idx>`` using
    the PP stage's layer offset so each stage loads the correct layer weights.

    After loading, a spot-check compares up to 5 parameter tensors against the
    file to verify the weights actually landed in the module.
    """
    ckpt_file = os.path.join(ckpt_dir, f"tp_rank_{tp_rank:02d}", "model_weights.pt")
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"[{label}] Checkpoint not found: {ckpt_file}")

    saved = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    state_dict = {k: v for k, v in saved["model"].items() if v is not None}

    # With pipeline parallelism (PP > 1), checkpoint layer keys are globally
    # numbered (decoder.layers.0 … decoder.layers.31) but each PP stage's
    # nn.ModuleList uses local 0-based indices.  Remap before loading.
    layer_offset = _get_pp_layer_offset(module)
    state_dict = _remap_checkpoint_for_pp(state_dict, module, layer_offset)
    if layer_offset > 0:
        print(f"[{label}] PP layer offset={layer_offset}, remapped checkpoint keys to local indices")

    incompat = module.load_state_dict(state_dict, strict=False)
    unexpected = [k for k in incompat.unexpected_keys if "_extra_state" not in k]
    missing = [k for k in incompat.missing_keys if "_extra_state" not in k]
    if unexpected or missing:
        raise RuntimeError(f"[{label}] load_state_dict mismatch.\n  Missing:    {missing}\n  Unexpected: {unexpected}")

    # Spot-check: re-read module state and compare against checkpoint tensors
    model_sd = module.state_dict()
    checked = 0
    for key, ref_tensor in state_dict.items():
        if key not in model_sd or ref_tensor is None:
            continue
        if not torch.equal(model_sd[key].float().cpu(), ref_tensor.float().cpu()):
            max_diff = (model_sd[key].float().cpu() - ref_tensor.float().cpu()).abs().max().item()
            raise RuntimeError(f"[{label}] Weight verification FAILED for '{key}': max abs diff = {max_diff}")
        checked += 1
        if checked >= 5:
            break
    if checked == 0:
        raise RuntimeError(f"[{label}] Weight verification found 0 overlapping keys to check")
    print(f"[{label}] Loaded and verified from {ckpt_file} ({checked} keys spot-checked)")


def _make_checkpoint_loader_hook(
    language_model_ckpt: str | None = None,
    vision_encoder_ckpt: str | None = None,
):
    """Return a ``pre_wrap_hook`` that loads per-module checkpoints.

    In hetero MIMO each rank only materialises the modules it participates in
    (``MimoModel.language_model`` is ``None`` on encoder-only ranks, and the
    vision submodule is absent on LLM-only ranks).  The hook therefore guards
    every load with an existence check so it is safe to call on all ranks.

    Both checkpoint dirs are expected to contain per-TP-rank ``.pt`` files
    produced by ``convert_hf_llama_to_megatron.py`` / ``convert_hf_clip_to_megatron.py``.
    """

    def _hook(model_list):
        model = model_list[0]
        grids = model.mimo_config.module_to_grid_map

        if language_model_ckpt and model.language_model is not None:
            tp_group = grids["language"].get_pg(["tp"])
            tp_rank = dist.get_rank(tp_group)
            tp_size = dist.get_world_size(tp_group)
            _load_tp_rank_weights(
                model.language_model,
                language_model_ckpt,
                tp_rank,
                label=f"LLM tp_rank={tp_rank}/{tp_size}",
            )

        if vision_encoder_ckpt and "images" in model.modality_submodules:
            images_sub = model.modality_submodules["images"]
            encoder = getattr(images_sub.encoders, "clip", None) if hasattr(images_sub, "encoders") else None
            if encoder is not None:
                tp_group = grids["images"].get_pg(["tp"])
                tp_rank = dist.get_rank(tp_group)
                tp_size = dist.get_world_size(tp_group)
                _load_tp_rank_weights(
                    encoder,
                    vision_encoder_ckpt,
                    tp_rank,
                    label=f"CLIP tp_rank={tp_rank}/{tp_size}",
                )

        return model_list

    return _hook


# ---------------------------------------------------------------------------
# Parallelism config (8 GPUs: TP=4 for both modules)
# ---------------------------------------------------------------------------

from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)


def _build_parallelism_config() -> MegatronMIMOParallelismConfig:
    return MegatronMIMOParallelismConfig(
        module_parallelisms={
            "language": ModuleParallelismConfig(
                tensor_model_parallel_size=int(os.environ.get("MIMO_LLM_TP", 4)),
                pipeline_model_parallel_size=int(os.environ.get("MIMO_LLM_PP", 1)),
                data_parallel_size=int(os.environ.get("MIMO_LLM_DP", 1)),
                rank_offset=int(os.environ.get("MIMO_LLM_OFFSET", 0)),
            ),
            "images": ModuleParallelismConfig(
                tensor_model_parallel_size=int(os.environ.get("MIMO_VISION_TP", 4)),
                pipeline_model_parallel_size=int(os.environ.get("MIMO_VISION_PP", 1)),
                data_parallel_size=int(os.environ.get("MIMO_VISION_DP", 1)),
                rank_offset=int(os.environ.get("MIMO_VISION_OFFSET", 4)),
            ),
        },
    )


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

from megatron.bridge.data.megatron_mimo.dataset import MegatronMIMODataset
from megatron.bridge.data.megatron_mimo.hf_provider import HFMegatronMIMODatasetProvider


def _llava_preprocess(example, dataset_root):
    """Convert LLaVA conversations format to plain text and resolve image paths.

    Emits the full conversation (human + gpt turns) as ``text`` so the LM
    conditions on the human prompt during training. Loss masking to the
    assistant-answer tokens is applied by ``_AnswerMaskedMimoDataset``,
    matching HF LLaVA's ``preprocess_plain`` and the Megatron-LM
    examples/mimo task encoders.
    """
    conversations = example.get("conversations", [])
    text_parts = [turn.get("value", "") for turn in conversations]
    example["text"] = " ".join(text_parts).replace("<image>", "").strip()
    # Resolve relative image paths to absolute paths
    if "image" in example and example["image"] and not os.path.isabs(example["image"]):
        example["image"] = os.path.join(dataset_root, example["image"])
    return example


def _find_token_span(
    seq: torch.Tensor, pattern: torch.Tensor, start_idx: int = 0, allow_first_mismatch: bool = False
) -> tuple[int, int]:
    """Return (start, end) of the first occurrence of ``pattern`` in ``seq``.

    Mirrors ``_find_pattern_indices`` in Megatron-LM examples/mimo task encoders.
    ``allow_first_mismatch`` handles SentencePiece boundary differences when the
    answer is tokenized standalone vs. embedded in the full prompt.
    Returns (-1, -1) if not found.
    """
    n, p = seq.size(0), pattern.size(0)
    if p == 0 or p > n:
        return -1, -1
    for i in range(start_idx, n - p + 1):
        match = seq[i : i + p] == pattern
        if torch.all(match) or (allow_first_mismatch and torch.all(match[1:])):
            return i, i + p
    return -1, -1


class _AnswerMaskedMimoDataset(MegatronMIMODataset):
    """MegatronMIMODataset variant that masks loss to assistant-answer tokens only.

    The base class sets ``loss_mask=1`` for every non-placeholder, non-pad
    position, which trains the LM on the human instruction as well as the
    caption. For LLaVA-Pretrain loss must be computed on the assistant ("gpt")
    turn only — the HF LLaVA ``preprocess_plain`` contract, also implemented
    by the Megatron-LM examples/mimo task encoders.
    """

    def __getitem__(self, idx):
        item = super().__getitem__(idx)

        raw = self.examples[idx]  # HF datasets return a fresh dict per access
        answers = [t.get("value", "") for t in raw.get("conversations", []) if t.get("from") == "gpt"]
        if not any(a.strip() for a in answers):
            return item

        input_ids = item["input_ids"]
        labels = torch.full_like(input_ids, -100)
        search_idx = 0
        for ans in answers:
            ans = ans.replace("<image>", "").strip()
            if not ans:
                continue
            ans_ids = self.tokenizer(ans, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)
            if ans_ids.numel() == 0:
                continue
            s, e = _find_token_span(input_ids, ans_ids, start_idx=search_idx, allow_first_mismatch=True)
            if s < 0:
                # Answer span not found (e.g. truncated); skip this answer.
                continue
            # labels[i] predicts input_ids[i+1]; answer tokens at input_ids[s:e]
            # are predicted at positions [s-1, e-1).
            lo, hi = max(0, s - 1), e - 1
            if hi > lo:
                labels[lo:hi] = input_ids[lo + 1 : hi + 1]
            search_idx = e

        item["labels"] = labels
        item["loss_mask"] = (labels != -100).to(item["loss_mask"].dtype)
        return item


class _AnswerMaskedHFMimoProvider(HFMegatronMIMODatasetProvider):
    """HFMegatronMIMODatasetProvider that builds ``_AnswerMaskedMimoDataset`` instances."""

    def _build_split_dataset(self, split, target_samples, processors, tokenizer):
        if target_samples <= 0:
            return None
        hf_dataset = self._load_hf_dataset(split)
        if hf_dataset is None:
            return None
        return _AnswerMaskedMimoDataset(
            examples=hf_dataset,
            processors=processors,
            tokenizer=tokenizer,
            seq_length=self.seq_length,
            special_token_ids=self.special_token_ids,
            encoder_seq_lengths=self.encoder_seq_lengths,
            modality_columns=self.modality_columns,
            text_column=self.text_column,
            max_samples=target_samples,
            preprocess_fn=self.preprocess_fn,
        )


def _build_hf_data_provider(dataset_root: str) -> HFMegatronMIMODatasetProvider:
    """Build an HFMegatronMIMODatasetProvider for liuhaotian/LLaVA-Pretrain."""
    provider = _AnswerMaskedHFMimoProvider(
        seq_length=MAX_SEQ_LENGTH,
        hf_dataset_path=dataset_root,
        hf_data_files="blip_laion_cc_sbu_558k.json",
        hf_tokenizer_path="llava-hf/llava-1.5-7b-hf",
        processor_paths={"images": "openai/clip-vit-large-patch14-336"},
        special_token_ids={"images": IMAGE_SPECIAL_TOKEN_ID},
        encoder_seq_lengths={"images": _ENCODER_SEQ_LEN},
        modality_columns={"images": "image"},
        text_column="text",
        train_split="train",
        preprocess_fn=lambda example: _llava_preprocess(example, dataset_root),
    )
    provider.drop_last = True

    return provider


def _wrap_iter(loader_iter):
    """Adapt data-loader batches for the MIMO model.

    Transforms:
    - modality_inputs["images"]["pixel_values"] → modality_inputs["images"]["clip"]["x"]
      so VisionModalitySubmodules.encode() finds the "clip" encoder key and
      CLIPViTModel.forward() receives ``x=...``.
    - Sets attention_mask=None (not needed for this test).
    - Generates loss_mask if not present.
    """
    for batch in loader_iter:
        # Move tensors to GPU
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

        # Rewrap modality_inputs: {"images": {"pixel_values": t}} → {"images": {"clip": {"x": t}}}
        # Cast to bfloat16 to match model weights
        mi = batch.get("modality_inputs")
        if mi and "images" in mi:
            pv = mi["images"].get("pixel_values")
            if pv is not None:
                mi["images"] = {"clip": {"x": pv.to(torch.bfloat16)}}

        # Ensure loss_mask exists
        if "loss_mask" not in batch or batch["loss_mask"] is None:
            batch["loss_mask"] = torch.ones_like(batch["input_ids"], dtype=torch.float)

        # Drop attention_mask (not needed)
        batch["attention_mask"] = None

        yield batch


def _build_data_iterators(cfg, _megatron_mimo_infra, *, train_state=None):
    """Build data iterators compatible with setup_megatron_mimo's build_data_iterators_fn.

    Signature: (cfg, megatron_mimo_infra, *, train_state=None) -> (train_iter, valid_iter)
    Uses build_megatron_mimo_data_loaders which auto-detects MIMO path via cfg.model.
    Accepts optional train_state for resume support.
    """
    from megatron.bridge.data.megatron_mimo.loaders import build_megatron_mimo_data_loaders
    from megatron.bridge.training.state import TrainState

    if train_state is None:
        train_state = TrainState()

    # Compute sample counts
    train_samples = cfg.train.train_iters * cfg.train.global_batch_size
    valid_samples = 0
    test_samples = 0

    train_loader, _, _ = build_megatron_mimo_data_loaders(
        cfg=cfg,
        train_state=train_state,
        megatron_mimo_provider=cfg.dataset,
        train_samples=max(train_samples, 10),  # min 10 samples
        valid_samples=valid_samples,
        test_samples=test_samples,
    )

    train_iter = _wrap_iter(train_loader) if train_loader is not None else None
    valid_iter = None
    return train_iter, valid_iter


# ---------------------------------------------------------------------------
# Config assembly
# ---------------------------------------------------------------------------


from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import MegatronMIMOProvider
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    LoggerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from megatron.bridge.training.config import OptimizerConfig as BridgeOptimizerConfig
from megatron.bridge.training.tokenizers.config import TokenizerConfig


def _build_config(
    megatron_mimo_provider: MegatronMIMOProvider,
    data_provider: HFMegatronMIMODatasetProvider,
    opt_config: BridgeOptimizerConfig,
    micro_batch_size: int = 1,
    global_batch_size: int = 1,
    train_iters: int = 2,
    log_interval: int = 1,
    wandb_project: str | None = None,
    wandb_exp_name: str | None = None,
    wandb_entity: str | None = None,
    wandb_save_dir: str | None = None,
    lr_warmup_iters: int = 0,
    seed: int = 42,
) -> ConfigContainer:
    train_cfg = TrainingConfig(
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        train_iters=train_iters,
    )
    # Runtime patches for MIMO
    train_cfg.num_microbatches = 1
    train_cfg.grad_reduce_in_fp32 = False
    train_cfg.overlap_grad_reduce = False
    train_cfg.use_distributed_optimizer = True
    train_cfg.check_for_nan_in_grad = False
    train_cfg.log_interval = log_interval

    logger_cfg = LoggerConfig()
    logger_cfg.log_timers_to_tensorboard = True
    logger_cfg.log_interval = log_interval
    logger_cfg.wandb_project = wandb_project
    logger_cfg.wandb_exp_name = wandb_exp_name
    logger_cfg.wandb_entity = wandb_entity
    logger_cfg.wandb_save_dir = wandb_save_dir
    logger_cfg.tensorboard_dir = os.path.join(wandb_save_dir or "/tmp/tb_logs", "tb_logs") if wandb_project else None

    scheduler_cfg = SchedulerConfig(
        lr_decay_style="cosine",
        lr_warmup_iters=lr_warmup_iters,
        lr_warmup_init=opt_config.min_lr,
        start_weight_decay=opt_config.weight_decay,
        end_weight_decay=opt_config.weight_decay,
    )

    cfg = ConfigContainer(
        train=train_cfg,
        model=megatron_mimo_provider,
        optimizer=opt_config,
        scheduler=scheduler_cfg,
        dataset=data_provider,
        logger=logger_cfg,
        tokenizer=TokenizerConfig(),
        checkpoint=CheckpointConfig(),
    )
    cfg.rng.seed = seed
    # data_parallel_size=1 because the sampler does not shard by DP.
    # All data-loading ranks receive identical global micro-batches;
    # per-module DP sub-sharding is handled by slice_batch_for_megatron_mimo in the
    # forward step.  num_microbatches = global_batch_size / micro_batch_size.
    cfg.data_parallel_size = 1
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

from megatron.bridge.training.megatron_mimo_step import forward_step as megatron_mimo_forward_step
from megatron.bridge.training.pretrain_megatron_mimo import pretrain_megatron_mimo


_rank_log_file = None


def _log(msg):
    """Write with rank prefix to per-rank log file and flush."""
    global _rank_log_file
    rank = dist.get_rank() if dist.is_initialized() else "?"
    line = f"[Rank {rank}] {msg}\n"
    if _rank_log_file:
        _rank_log_file.write(line)
        _rank_log_file.flush()
    print(line, end="", flush=True)


def _str2bool(v):
    """Parse boolean values from command line arguments."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")


def parse_args():
    """Parse command-line arguments for the MegatronMIMO LLaVA training example."""
    parser = argparse.ArgumentParser(description="MegatronMIMO LLaVA training")
    parser.add_argument("--micro-batch-size", type=int, default=1, help="Micro batch size per GPU")
    parser.add_argument("--global-batch-size", type=int, default=1, help="Global batch size across all GPUs")
    parser.add_argument("--train-iters", type=int, default=2, help="Number of training iterations")
    parser.add_argument("--min-lr", type=float, default=2.0e-5)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=None, help="Checkpoint save interval (iterations)")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint output directory")
    parser.add_argument("--load-checkpoint", type=str, default=None, help="Checkpoint directory to resume from")
    parser.add_argument(
        "--language-model-checkpoint",
        type=str,
        default=None,
        help="Path to a Megatron distributed checkpoint to load into the language model only",
    )
    parser.add_argument(
        "--vision-encoder-checkpoint",
        type=str,
        default=None,
        help="Path to a Megatron distributed checkpoint to load into the vision encoder only",
    )
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--wandb-project", type=str, default="Megatron-Bridge-MIMO", help="W&B project name")
    parser.add_argument("--wandb-exp-name", type=str, default="omni-modal-llava-e2e-test", help="W&B experiment name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity")
    parser.add_argument("--wandb-save-dir", type=str, default="/tmp/wandb", help="W&B save directory")
    parser.add_argument(
        "--lr-warmup-iters", type=int, default=20, help="Number of iterations to linearly warmup learning rate"
    )
    parser.add_argument("--dataset-root", type=str, required=True, help="Root directory of the LLaVA-Pretrain dataset")
    parser.add_argument(
        "--freeze-vision", type=_str2bool, default=True, help="Freeze the vision encoder (default: True)"
    )
    parser.add_argument("--freeze-llm", type=_str2bool, default=True, help="Freeze the language model (default: True)")
    parser.add_argument(
        "--freeze-projector", type=_str2bool, default=False, help="Freeze the projector (default: False)"
    )
    return parser.parse_args()


def main():
    """Entry point for the MegatronMIMO LLaVA training example."""
    global _rank_log_file

    args = parse_args()

    # 1. Initialize distributed first so we know rank
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # Seed all RNGs for reproducible weight initialization.
    # NOTE: _set_megatron_mimo_random_seeds() in setup_megatron_mimo re-seeds all RNGs with
    # cfg.rng.seed before model construction, so cfg.rng.seed (set below)
    # is what actually determines projection weight initialization.
    seed = 42

    # Open per-rank log file
    log_dir = os.environ.get("OMNI_MODAL_LOG_DIR", "/tmp/megatron_mimo_llava_logs")
    os.makedirs(log_dir, exist_ok=True)
    _rank_log_file = open(f"{log_dir}/rank_{rank}.log", "w")

    logging.basicConfig(
        level=logging.INFO,
        format=f"[Rank {rank}] %(name)s: %(message)s",
        handlers=[logging.FileHandler(f"{log_dir}/rank_{rank}_full.log", mode="w"), logging.StreamHandler(sys.stderr)],
        force=True,
    )
    # Enable debug logging for bridge communicator to trace P2P ops
    logging.getLogger("megatron.core.pipeline_parallel.bridge_communicator").setLevel(logging.DEBUG)
    logging.getLogger("megatron.core.pipeline_parallel.multimodule_communicator").setLevel(logging.DEBUG)

    _log(f"distributed initialized (world_size={dist.get_world_size()})")

    succeeded = False
    # No parallel_state.initialize_model_parallel() — MIMO manages its own
    # parallelism via HyperCommGrids and pg_collections. Float16Module is
    # skipped (direct bf16 cast), and cross_entropy_loss_fusion=True ensures
    # the fused CE path uses pg_collection.tp instead of global parallel_state.

    # 2. Build model provider
    _log("building model specs")
    language_model_spec, modality_submodules_spec, special_token_ids = _build_model_specs()
    megatron_mimo_parallelism_config = _build_parallelism_config()

    # Propagate per-module pipeline parallelism size into the TransformerConfig
    # so that get_num_layers_to_build() and get_transformer_layer_offset() in
    # Megatron-LM produce the correct layer count and offset for each PP stage.
    # Without this, config.pipeline_model_parallel_size defaults to 1 and every
    # stage builds *all* num_layers — duplicating the model across PP stages.
    llm_pp_size = megatron_mimo_parallelism_config.module_parallelisms["language"].pipeline_model_parallel_size
    language_model_spec.params["config"].pipeline_model_parallel_size = llm_pp_size

    megatron_mimo_provider = MegatronMIMOProvider(
        language_model_spec=language_model_spec,
        modality_submodules_spec=modality_submodules_spec,
        special_token_ids=special_token_ids,
        megatron_mimo_parallelism_config=megatron_mimo_parallelism_config,
        topology={"images": ["language"], "language": []},
        use_cpu_initialization=True,
        bf16=True,
        freeze_language_model=args.freeze_llm,
        freeze_modality_encoders={"images": args.freeze_vision},
        freeze_modality_projections={"images": args.freeze_projector},
    )
    # Register per-module checkpoint loading hook (runs before DDP wrapping)
    if args.language_model_checkpoint or args.vision_encoder_checkpoint:
        megatron_mimo_provider.register_pre_wrap_hook(
            _make_checkpoint_loader_hook(
                language_model_ckpt=args.language_model_checkpoint,
                vision_encoder_ckpt=args.vision_encoder_checkpoint,
            )
        )
        _log(
            f"Registered checkpoint hooks: "
            f"LLM={args.language_model_checkpoint}, "
            f"vision={args.vision_encoder_checkpoint}"
        )

    # Patch: training_log accesses config.model.num_moe_experts
    if not hasattr(megatron_mimo_provider, "num_moe_experts"):
        megatron_mimo_provider.num_moe_experts = None

    # 4. Build data provider
    _log("building data provider")
    data_provider = _build_hf_data_provider(args.dataset_root)

    # 5. Build optimizer config
    _log("building optimizer config")
    print_rank_0 = lambda msg: _log(msg) if dist.get_rank() == 0 else None
    print_rank_0(
        f"Optimizer config: lr={args.lr}, min_lr={args.min_lr}, weight_decay={args.weight_decay}, "
        f"adam_beta1={args.adam_beta1}, adam_beta2={args.adam_beta2}, clip_grad={args.clip_grad}"
    )
    bridge_opt_config = BridgeOptimizerConfig(
        optimizer="adam",
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        clip_grad=args.clip_grad,
        bf16=True,
        use_distributed_optimizer=True,
    )

    # 6. Build config container
    _log("building config")
    cfg = _build_config(
        megatron_mimo_provider,
        data_provider,
        bridge_opt_config,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
        train_iters=args.train_iters,
        log_interval=args.log_interval,
        wandb_project=args.wandb_project,
        wandb_exp_name=args.wandb_exp_name,
        wandb_entity=args.wandb_entity,
        wandb_save_dir=args.wandb_save_dir,
        lr_warmup_iters=args.lr_warmup_iters,
        seed=seed,
    )

    # Configure checkpointing from CLI args
    if args.checkpoint_interval is not None:
        cfg.checkpoint.save_interval = args.checkpoint_interval
    if args.checkpoint_dir is not None:
        cfg.checkpoint.save = args.checkpoint_dir
    if args.load_checkpoint is not None:
        cfg.checkpoint.load = args.load_checkpoint
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True
    cfg.checkpoint.dist_ckpt_optim_fully_reshardable = True
    # MiMo RNG save is not yet supported: each module produces ShardedObject
    # with key "rng_state" using module-local PP/TP/DP ranks, causing
    # duplicate shard keys across modules.  Disable until upstream fix.
    cfg.checkpoint.save_rng = False

    # 7. Run training
    _log("launching pretrain_megatron_mimo")
    pretrain_megatron_mimo(
        cfg=cfg,
        forward_step_func=megatron_mimo_forward_step,
        build_data_iterators_fn=_build_data_iterators,
    )

    _log("PASSED")
    succeeded = True

    # 8. Cleanup — only tear down NCCL on success; on failure torchrun
    # handles cleanup via SIGTERM (destroy_process_group deadlocks when
    # other ranks are stuck in collectives).
    if succeeded:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
