# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.


from __future__ import annotations

import argparse
import logging
import os
import sys

import torch
import torch.distributed as dist
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo.submodules.audio import AudioModalitySubmodules
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

# Shared LLaVA (Vicuna-7B + CLIP ViT-L/14) configs and checkpoint helpers.
from megatron_mimo_training_llava import (
    _ENCODER_SEQ_LEN,
    _IMG_SIZE,
    _PATCH_DIM,
    CLIP_OUTPUT_DIM,
    IMAGE_SPECIAL_TOKEN_ID,
    MAX_SEQ_LENGTH,
    VOCAB_SIZE,
    CLIPViTNoCLS,
    _build_config,
    _load_tp_rank_weights,
    _make_language_config,
    _make_projection_config,
    _make_vision_config,
    _str2bool,
)
from whisper.whisper_layer_specs import get_whisper_layer_with_transformer_engine_spec
from whisper.whisper_model import WhisperEncoder


# ---------------------------------------------------------------------------
# Audio-specific constants (Whisper-base encoder)
# ---------------------------------------------------------------------------

AUDIO_SPECIAL_TOKEN_ID = 32002
WHISPER_OUTPUT_DIM = 512  # Whisper-base hidden size
# Whisper-base: 30s padded audio → 3000 mel frames → 1500 encoder output tokens
_AUDIO_ENCODER_SEQ_LEN = 1500
_AUDIO_NUM_MEL_BINS = 80
_AUDIO_MAX_SOURCE_POSITIONS = 1500


def _make_audio_config() -> TransformerConfig:
    """Whisper-base audio encoder config (6 encoder layers, d_model=512)."""
    cfg = TransformerConfig(
        num_layers=6,
        hidden_size=512,
        ffn_hidden_size=2048,
        num_attention_heads=8,
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
    cfg.calculate_per_token_loss = True
    return cfg


def _build_model_specs():  # pragma: no cover
    """Return (language_model_spec, modality_submodules_spec, special_token_ids)."""
    vision_config = _make_vision_config()
    audio_config = _make_audio_config()
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

    # Megatron-native Whisper encoder (TP-shardable)
    audio_encoder = ModuleSpec(
        module=WhisperEncoder,
        params={
            "transformer_config": audio_config,
            "transformer_layer_spec": get_whisper_layer_with_transformer_engine_spec(),
            "num_mel_bins": _AUDIO_NUM_MEL_BINS,
            "max_source_positions": _AUDIO_MAX_SOURCE_POSITIONS,
        },
    )

    # Audio→language projection MLP
    audio_projection_config = _make_projection_config(hidden_size=language_config.hidden_size)
    audio_projection = ModuleSpec(
        module=MultimodalProjector,
        params={
            "config": audio_projection_config,
            "submodules": MLPSubmodules(
                linear_fc1=TEColumnParallelLinear,
                linear_fc2=TERowParallelLinear,
            ),
            "projector_type": "mlp",
            "input_size": WHISPER_OUTPUT_DIM,
        },
    )

    audio_submodule_spec = ModuleSpec(
        module=AudioModalitySubmodules,
        params={},
        submodules={
            "encoders": {"whisper": audio_encoder},
            "input_projections": [audio_projection],
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

    modality_submodules_spec = {"images": vision_submodule_spec, "audios": audio_submodule_spec}
    special_token_ids = {"images": IMAGE_SPECIAL_TOKEN_ID, "audios": AUDIO_SPECIAL_TOKEN_ID}
    return language_model_spec, modality_submodules_spec, special_token_ids


# ---------------------------------------------------------------------------
# Per-submodule checkpoint loading
# ---------------------------------------------------------------------------


def _make_checkpoint_loader_hook(
    language_model_ckpt: str | None = None,
    vision_encoder_ckpt: str | None = None,
    audio_encoder_ckpt: str | None = None,
):
    """Return a ``pre_wrap_hook`` that loads per-module checkpoints.

    In hetero MIMO each rank only materialises the modules it participates in
    (``MimoModel.language_model`` is ``None`` on encoder-only ranks, and the
    vision/audio submodules are absent on LLM-only ranks).  The hook therefore
    guards every load with an existence check so it is safe to call on all ranks.

    Checkpoint dirs are expected to contain per-TP-rank ``.pt`` files
    produced by HF→Megatron converters.
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

        if audio_encoder_ckpt and "audios" in model.modality_submodules:
            audios_sub = model.modality_submodules["audios"]
            encoder = getattr(audios_sub.encoders, "whisper", None) if hasattr(audios_sub, "encoders") else None
            if encoder is not None:
                tp_group = grids["audios"].get_pg(["tp"])
                tp_rank = dist.get_rank(tp_group)
                tp_size = dist.get_world_size(tp_group)
                _load_tp_rank_weights(
                    encoder,
                    audio_encoder_ckpt,
                    tp_rank,
                    label=f"Whisper tp_rank={tp_rank}/{tp_size}",
                )

        return model_list

    return _hook


# ---------------------------------------------------------------------------
# Parallelism config (8 GPUs: LLM TP=4, vision TP=2, audio TP=2)
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
                tensor_model_parallel_size=int(os.environ.get("MIMO_VISION_TP", 2)),
                pipeline_model_parallel_size=int(os.environ.get("MIMO_VISION_PP", 1)),
                data_parallel_size=int(os.environ.get("MIMO_VISION_DP", 1)),
                rank_offset=int(os.environ.get("MIMO_VISION_OFFSET", 4)),
            ),
            "audios": ModuleParallelismConfig(
                tensor_model_parallel_size=int(os.environ.get("MIMO_AUDIO_TP", 2)),
                pipeline_model_parallel_size=int(os.environ.get("MIMO_AUDIO_PP", 1)),
                data_parallel_size=int(os.environ.get("MIMO_AUDIO_DP", 1)),
                rank_offset=int(os.environ.get("MIMO_AUDIO_OFFSET", 6)),
            ),
        },
    )


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

from megatron.bridge.data.megatron_mimo.dataset import MegatronMIMODataset
from megatron.bridge.data.megatron_mimo.hf_provider import HFMegatronMIMODatasetProvider


def _llava_preprocess(example, dataset_root):
    """Convert LLaVA conversations format to plain text and resolve media paths.

    Emits the full conversation (human + gpt turns) as ``text`` so the LM
    conditions on the human prompt during training. Loss masking to the
    assistant-answer tokens is applied by ``_AnswerMaskedMegatronMIMODataset``,
    matching HF LLaVA's ``preprocess_plain`` and the Megatron-LM
    examples/mimo task encoders.
    """
    conversations = example.get("conversations", [])
    text_parts = [turn.get("value", "") for turn in conversations]
    example["text"] = " ".join(text_parts).replace("<image>", "").replace("<audio>", "").strip()
    # Resolve relative image paths to absolute paths
    if "image" in example and example["image"] and not os.path.isabs(example["image"]):
        example["image"] = os.path.join(dataset_root, example["image"])
    # Load audio from file path into a numpy array for WhisperProcessor
    if "audio" in example and example["audio"]:
        audio_val = example["audio"]
        if isinstance(audio_val, str):
            audio_path = audio_val if os.path.isabs(audio_val) else os.path.join(dataset_root, audio_val)
            import soundfile as sf

            audio_array, sr = sf.read(audio_path)
            if sr != 16000:
                raise ValueError(
                    f"Whisper expects 16 kHz audio but {audio_path} has sample rate {sr}. "
                    "Resample the dataset to 16 kHz before training."
                )
            example["audio"] = audio_array
        elif isinstance(audio_val, dict) and "array" in audio_val:
            # HuggingFace Audio feature format
            example["audio"] = audio_val["array"]
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


class _AnswerMaskedMegatronMIMODataset(MegatronMIMODataset):
    """MegatronMIMODataset variant that masks loss to assistant-answer tokens only.

    The base class sets ``loss_mask=1`` for every non-placeholder, non-pad
    position, which trains the LM on the human instruction as well as the
    caption. For LLaVA-Pretrain loss must be computed on the assistant ("gpt")
    turn only — the HF LLaVA ``preprocess_plain`` contract, also implemented
    by the Megatron-LM examples/mimo task encoders.

    Works identically for vision-only and audio-augmented variants: the audio
    placeholders (if any) fall outside the answer span and remain ``-100`` /
    ``loss_mask=0``.
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
            ans = ans.replace("<image>", "").replace("<audio>", "").strip()
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


class _AnswerMaskedHFMegatronMIMOProvider(HFMegatronMIMODatasetProvider):
    """HFMegatronMIMODatasetProvider that builds ``_AnswerMaskedMegatronMIMODataset`` instances."""

    def _build_split_dataset(self, split, target_samples, processors, tokenizer):  # pragma: no cover
        if target_samples <= 0:
            return None
        hf_dataset = self._load_hf_dataset(split)
        if hf_dataset is None:
            return None
        return _AnswerMaskedMegatronMIMODataset(
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


def _build_hf_data_provider(
    dataset_root: str,
    audio_column: str | None = None,
    hf_data_files: str = "blip_laion_cc_sbu_558k.json",
) -> HFMegatronMIMODatasetProvider:  # pragma: no cover
    """Build an HFMegatronMIMODatasetProvider for LLaVA-Pretrain with optional audio."""
    processor_paths = {"images": "openai/clip-vit-large-patch14-336"}
    special_token_ids = {"images": IMAGE_SPECIAL_TOKEN_ID}
    encoder_seq_lengths = {"images": _ENCODER_SEQ_LEN}
    modality_columns = {"images": "image"}

    if audio_column:
        processor_paths["audios"] = "openai/whisper-base"
        special_token_ids["audios"] = AUDIO_SPECIAL_TOKEN_ID
        encoder_seq_lengths["audios"] = _AUDIO_ENCODER_SEQ_LEN
        modality_columns["audios"] = audio_column

    provider = _AnswerMaskedHFMegatronMIMOProvider(
        seq_length=MAX_SEQ_LENGTH,
        hf_dataset_path=dataset_root,
        hf_data_files=hf_data_files,
        hf_tokenizer_path="llava-hf/llava-1.5-7b-hf",
        processor_paths=processor_paths,
        special_token_ids=special_token_ids,
        encoder_seq_lengths=encoder_seq_lengths,
        modality_columns=modality_columns,
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
    - modality_inputs["audios"]["input_features"] → modality_inputs["audios"]["whisper"]["input_features"]
      so AudioModalitySubmodules.encode() finds the "whisper" encoder key and
      WhisperEncoder.forward() receives ``input_features=...``.
    - Drops padding-derived audio encoder tokens: computes valid frame counts
      from the mel spectrogram (padding frames are zero), derives per-sample
      encoder output lengths, trims excess audio placeholder tokens from
      input_ids, and passes seq_lengths to the encoder.
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

        # Rewrap modality_inputs to encoder-keyed dicts and cast to bfloat16
        mi = batch.get("modality_inputs")
        if mi and "images" in mi:
            pv = mi["images"].get("pixel_values")
            if pv is not None:
                mi["images"] = {"clip": {"x": pv.to(torch.bfloat16)}}

        if mi and "audios" in mi:
            af = mi["audios"].get("input_features")
            if af is not None:
                audio_kwargs = {"input_features": af.to(torch.bfloat16)}

                # Compute per-sample valid encoder output lengths.
                # WhisperFeatureExtractor pads mel spectrograms with zeros;
                # real frames always have non-zero energy in at least one bin.
                frame_energy = af.abs().sum(dim=-2)  # [B, mel_frames], sum over mel bins
                valid_frames = (frame_energy > 0).sum(dim=-1)  # [B]
                # Conv2 uses stride=2: output_len = (input_len - 1) // 2 + 1
                seq_lengths = ((valid_frames - 1) // 2 + 1).clamp(min=0).long()
                audio_kwargs["seq_lengths"] = seq_lengths

                # Replace excess audio placeholder tokens in input_ids so that
                # align_embeddings_by_token_positions sees the correct count.
                input_ids = batch["input_ids"]
                for i in range(input_ids.size(0)):
                    positions = (input_ids[i] == AUDIO_SPECIAL_TOKEN_ID).nonzero(as_tuple=True)[0]
                    n_valid = seq_lengths[i].item()
                    if n_valid < len(positions):
                        input_ids[i, positions[n_valid:]] = 0  # replace with pad token

                mi["audios"] = {"whisper": audio_kwargs}

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
# Main
# ---------------------------------------------------------------------------

from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import MegatronMIMOProvider
from megatron.bridge.training.config import OptimizerConfig as BridgeOptimizerConfig
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


def parse_args():  # pragma: no cover
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
    parser.add_argument(
        "--audio-encoder-checkpoint",
        type=str,
        default=None,
        help="Path to a Megatron distributed checkpoint to load into the audio encoder only",
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
        "--hf-data-files",
        type=str,
        default="blip_laion_cc_sbu_558k.json",
        help="JSON file under --dataset-root to load (e.g. the audio-augmented variant).",
    )
    parser.add_argument(
        "--audio-column",
        type=str,
        default=None,
        help="Dataset column name for audio data (e.g. 'audio'). Enables the audio encoder when set.",
    )
    parser.add_argument(
        "--freeze-vision", type=_str2bool, default=True, help="Freeze the vision encoder (default: True)"
    )
    parser.add_argument("--freeze-llm", type=_str2bool, default=True, help="Freeze the language model (default: True)")
    parser.add_argument(
        "--freeze-vision-projector", type=_str2bool, default=False, help="Freeze the vision projector (default: False)"
    )
    parser.add_argument(
        "--freeze-audio", type=_str2bool, default=True, help="Freeze the audio encoder (default: True)"
    )
    parser.add_argument(
        "--freeze-audio-projector", type=_str2bool, default=False, help="Freeze the audio projector (default: False)"
    )
    return parser.parse_args()


def main():  # pragma: no cover
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
        topology={"images": ["language"], "audios": ["language"], "language": []},
        use_cpu_initialization=True,
        bf16=True,
        freeze_language_model=args.freeze_llm,
        freeze_modality_encoders={"images": args.freeze_vision, "audios": args.freeze_audio},
        freeze_modality_projections={"images": args.freeze_vision_projector, "audios": args.freeze_audio_projector},
    )
    # Register per-module checkpoint loading hook (runs before DDP wrapping)
    if args.language_model_checkpoint or args.vision_encoder_checkpoint or args.audio_encoder_checkpoint:
        megatron_mimo_provider.register_pre_wrap_hook(
            _make_checkpoint_loader_hook(
                language_model_ckpt=args.language_model_checkpoint,
                vision_encoder_ckpt=args.vision_encoder_checkpoint,
                audio_encoder_ckpt=args.audio_encoder_checkpoint,
            )
        )
        _log(
            f"Registered checkpoint hooks: "
            f"LLM={args.language_model_checkpoint}, "
            f"vision={args.vision_encoder_checkpoint}, "
            f"audio={args.audio_encoder_checkpoint}"
        )

    # Patch: training_log accesses config.model.num_moe_experts
    if not hasattr(megatron_mimo_provider, "num_moe_experts"):
        megatron_mimo_provider.num_moe_experts = None

    # 4. Build data provider
    _log("building data provider")
    data_provider = _build_hf_data_provider(
        args.dataset_root,
        audio_column=args.audio_column,
        hf_data_files=args.hf_data_files,
    )

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
