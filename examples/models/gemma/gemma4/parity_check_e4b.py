#!/usr/bin/env python3
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

"""
Logit parity check: Megatron Gemma-4 E4B vs HF Gemma-4 E4B.

Supports three modes (via --mode or GEMMA4_CONVERSION_MODE env var):

  text  : text-only checkpoint
          Megatron: Gemma4DenseProvider → GPTModel
          HF:       AutoModelForCausalLM (Gemma4ForCausalLM)

  vl    : VL checkpoint, full image encoder forward
          Megatron: Gemma4DenseVLProvider → Gemma4VLModel forward with
                    pixel_values and image_token_id positions
          HF:       AutoModelForVision2Seq (Gemma4ForConditionalGeneration)
                    with pixel_values

  audio : VL+Audio checkpoint, full audio encoder forward
          Megatron: Gemma4DenseVLProvider (with audio_config) → Gemma4VLModel
                    forward with input_features and audio_token_id positions
          HF:       AutoModelForVision2Seq with input_features

          Audio tower architecture (from checkpoint):
            input  : [B, T, 128]   mel-spectrogram (128-bin, 10 ms frames)
            subsample: 2× stride-2 Conv2D → T/4 frames
            encoder: 12-layer transformer, hidden=1024
            output_proj: 1024 → 1536
            embed_audio: 1536 → 2560 (text hidden)
          So T input frames → T/4 audio tokens in the sequence.

Run from Megatron-Bridge root via:
    CUDA_DEVICE_MAX_CONNECTIONS=1 uv run --no-sync python -m torch.distributed.run --nproc_per_node=2 \\
        examples/models/gemma/gemma4/parity_check_e4b.py \\
        --hf-dir ~/models/gemma-4-E4B-it \\
        --megatron-ckpt /path/to/gemma4-e4b-megatron \\
        [--mode text|vl|audio]
"""

import argparse
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BRIDGE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../.."))
MEGATRON_LM_ROOT = os.environ.get("MEGATRON_LM_ROOT", os.getcwd())

sys.path.insert(0, os.path.join(BRIDGE_ROOT, "src"))
sys.path.insert(0, MEGATRON_LM_ROOT)

import torch
import torch.distributed as dist
from megatron.training import print_rank_0


SEQ = 16
BATCH = 1
FULL_VOCAB = 262144

# Audio-mode constants (based on audio_tower checkpoint analysis)
AUDIO_MEL_BINS = 128  # mel-spectrogram frequency bins
AUDIO_SUBSAMPLING = 4  # two stride-2 Conv2D stages → 4× time reduction
AUDIO_TOKEN_ID = 258_881  # audio_token_id from HF config
AUDIO_NUM_TOKENS = 12  # desired audio tokens in test sequence
AUDIO_INPUT_FRAMES = AUDIO_NUM_TOKENS * AUDIO_SUBSAMPLING  # 48 input time frames
AUDIO_SEQ = AUDIO_NUM_TOKENS + (SEQ - AUDIO_NUM_TOKENS)  # same total seq length

# VL-mode constants. Gemma4 image processor defaults to 280 soft tokens.
IMAGE_TOKEN_ID = 258_880
IMAGE_NUM_TOKENS = 280
IMAGE_PATCH_SIZE = 16
IMAGE_POOLING_KERNEL_SIZE = 3
IMAGE_PATCH_GRID_H = 42
IMAGE_PATCH_GRID_W = 60
IMAGE_NUM_PATCHES = IMAGE_PATCH_GRID_H * IMAGE_PATCH_GRID_W  # 2520 = 280 * 3^2
IMAGE_PATCH_DIM = 3 * IMAGE_PATCH_SIZE * IMAGE_PATCH_SIZE  # flattened RGB patch
VL_TEXT_TOKENS = 4
VL_SEQ = IMAGE_NUM_TOKENS + VL_TEXT_TOKENS


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--hf-dir", required=True)
    p.add_argument("--megatron-ckpt", required=True)
    p.add_argument("--atol", type=float, default=1.0, help="Max absolute logit difference. ~1.0 fp32, ~3.0 bf16.")
    p.add_argument("--tp", type=int, default=2, choices=[1, 2], help="Tensor parallel size.")
    p.add_argument("--bf16", action="store_true", help="Use bf16 (default: float32).")
    _default_mode = os.environ.get("GEMMA4_CONVERSION_MODE", "text").lower()
    if _default_mode not in ("text", "vl", "auto", "audio"):
        _default_mode = "text"
    if _default_mode == "auto":
        _default_mode = "vl"
    # "audio" stays as "audio" — triggers full audio forward test
    p.add_argument(
        "--mode",
        choices=["text", "vl", "audio"],
        default=_default_mode,
        help="Parity mode. Default: $GEMMA4_CONVERSION_MODE or 'text'.",
    )
    p.add_argument(
        "--vl-image-tokens",
        type=int,
        default=IMAGE_NUM_TOKENS,
        help=(
            "Number of soft image tokens for VL parity. "
            "Reduced counts (e.g. 14, 70) let you verify that max |diff| "
            "scales with token count (bf16 accumulated error)."
        ),
    )
    return p.parse_args()


def _build_megatron_argv(ckpt, tp=2, bf16=False, seq=SEQ):
    return [
        "parity",
        "--use-mcore-models",
        "--num-layers",
        "42",
        "--hidden-size",
        "2560",
        "--ffn-hidden-size",
        "10240",
        "--num-attention-heads",
        "8",
        "--group-query-attention",
        "--num-query-groups",
        "2",
        "--kv-channels",
        "256",
        "--seq-length",
        str(seq),
        "--max-position-embeddings",
        "131072",
        "--position-embedding-type",
        "rope",
        "--rotary-percent",
        "1.0",
        "--window-size",
        "511,0",
        "--window-attn-skip-freq",
        "6",
        "--normalization",
        "RMSNorm",
        "--norm-epsilon",
        "1e-6",
        "--attention-dropout",
        "0.0",
        "--hidden-dropout",
        "0.0",
        "--disable-bias-linear",
        "--vocab-size",
        "262143",
        "--make-vocab-size-divisible-by",
        "128",
        "--transformer-impl",
        "local",
        "--attention-backend",
        "unfused",
        "--tensor-model-parallel-size",
        str(tp),
        "--pipeline-model-parallel-size",
        "1",
        "--context-parallel-size",
        "1",
        "--no-rope-fusion",
        "--no-persist-layer-norm",
        "--no-masked-softmax-fusion",
        "--no-gradient-accumulation-fusion",
        "--load",
        ckpt,
        "--finetune",
        "--no-load-optim",
        "--no-load-rng",
        "--init-method-std",
        "0.02",
        "--micro-batch-size",
        str(BATCH),
        "--global-batch-size",
        str(BATCH),
        "--train-iters",
        "1",
        "--tokenizer-type",
        "NullTokenizer",
        "--mock-data",
        "--no-create-attention-mask-in-dataloader",
        "--no-mmap-bin-files",
        "--num-workers",
        "0",
        "--lr",
        "1e-4",
        "--distributed-timeout-minutes",
        "10",
        "--log-interval",
        "1",
        "--eval-iters",
        "0",
        "--eval-interval",
        "1000",
        "--no-save-optim",
        "--no-save-rng",
    ] + (["--bf16"] if bf16 else [])


# ---------------------------------------------------------------------------
# Shared VL provider builder (used by both vl and audio modes)
# ---------------------------------------------------------------------------


def _seq_len_for_mode(mode: str) -> int:
    if mode == "audio":
        return AUDIO_SEQ
    if mode == "vl":
        return VL_SEQ
    return SEQ


def _make_vl_provider(args, hf_cfg, seq_len: int = AUDIO_SEQ, include_audio: bool = False):
    from megatron.bridge.models.gemma_vl.gemma4_vl_provider import Gemma4DenseVLProvider

    model_dtype = torch.bfloat16 if args.bf16 else torch.float32
    return Gemma4DenseVLProvider(
        num_layers=42,
        hidden_size=2560,
        ffn_hidden_size=10240,
        num_attention_heads=8,
        num_query_groups=2,
        kv_channels=256,
        global_kv_channels=512,
        num_global_query_groups=2,
        seq_length=seq_len,
        vocab_size=262143,
        make_vocab_size_divisible_by=128,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        window_attn_skip_freq=6,
        sliding_window_rope_base=10000.0,
        full_attention_rope_base=1000000.0,
        full_attention_rope_partial_factor=0.25,
        num_kv_shared_layers=18,
        per_layer_embed_vocab_size=262144,
        per_layer_embed_dim=256,
        vision_config=hf_cfg.vision_config,
        text_config=hf_cfg.text_config,
        audio_config=hf_cfg.audio_config if include_audio else None,
        audio_token_id=getattr(hf_cfg, "audio_token_id", AUDIO_TOKEN_ID),
        image_token_id=getattr(hf_cfg, "image_token_id", IMAGE_TOKEN_ID),
        bf16=args.bf16,
        params_dtype=model_dtype,
        autocast_dtype=model_dtype,
    )


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def _build_text_models(args):
    """Text mode: GPTModel via Gemma4DenseProvider."""
    from megatron.core.enums import ModelType
    from megatron.training import get_model

    from megatron.bridge.models.gemma.gemma4_provider import Gemma4DenseProvider

    model_dtype = torch.bfloat16 if args.bf16 else torch.float32
    provider = Gemma4DenseProvider(
        bf16=args.bf16,
        params_dtype=model_dtype,
        autocast_dtype=model_dtype,
    )
    return get_model(
        lambda pre_process=True, post_process=True, config=None, pg_collection=None: provider.build(
            pre_process=pre_process, post_process=post_process
        ),
        ModelType.encoder_or_decoder,
    )


def _build_vl_models(args, seq_len: int = AUDIO_SEQ, include_audio: bool = False):
    """VL / Audio mode: Gemma4VLModel via Gemma4DenseVLProvider."""
    from megatron.core.enums import ModelType
    from megatron.training import get_model
    from transformers import AutoConfig

    hf_cfg = AutoConfig.from_pretrained(args.hf_dir)
    provider = _make_vl_provider(args, hf_cfg, seq_len=seq_len, include_audio=include_audio)
    return get_model(
        lambda pre_process=True, post_process=True, config=None, pg_collection=None: provider.provide(
            pre_process=pre_process, post_process=post_process
        ),
        ModelType.encoder_or_decoder,
    )


# ---------------------------------------------------------------------------
# Forward passes
# ---------------------------------------------------------------------------


def _unwrap(model):
    """Peel DDP / Float16Module / any .module wrappers to reach the real model."""
    inner = model
    while hasattr(inner, "module"):
        inner = inner.module
    return inner


def _batch_first_logits(logits, seq_len):
    if logits.shape[0] == seq_len and logits.shape[1] == BATCH:
        logits = logits.permute(1, 0, 2)
    return logits


def _forward_text(model, tokens):
    """GPTModel forward → logits [BATCH, SEQ, vocab/tp]."""
    with torch.no_grad():
        out = model(input_ids=tokens, position_ids=None, attention_mask=None)
    logits = out[0] if isinstance(out, tuple) else out
    return _batch_first_logits(logits, SEQ)


def _forward_vl(model, input_ids_vl, pixel_values, image_position_ids):
    """VL mode: full Gemma4VLModel forward with image input."""
    inner = _unwrap(model)
    with torch.no_grad():
        out = inner(
            input_ids=input_ids_vl,
            attention_mask=None,
            position_ids=None,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
        )
    logits = out[0] if isinstance(out, tuple) else out
    return _batch_first_logits(logits, VL_SEQ)


def _forward_audio(model, input_ids_audio, audio_features):
    """Audio mode: full Gemma4VLModel forward with audio input.

    Routes through audio_tower → embed_audio → language_model.
    """
    inner = _unwrap(model)
    with torch.no_grad():
        out = inner(
            input_ids=input_ids_audio,
            attention_mask=None,
            position_ids=None,
            input_features=audio_features,
            pixel_values=None,
        )
    logits = out[0] if isinstance(out, tuple) else out
    return _batch_first_logits(logits, AUDIO_SEQ)


# ---------------------------------------------------------------------------
# Logit gathering
# ---------------------------------------------------------------------------


def _gather_logits(logits, mpu):
    """All-gather TP vocab shards and trim to FULL_VOCAB."""
    tp = mpu.get_tensor_model_parallel_world_size()
    if tp > 1:
        parts = [torch.zeros_like(logits) for _ in range(tp)]
        dist.all_gather(parts, logits.contiguous(), group=mpu.get_tensor_model_parallel_group())
        logits = torch.cat(parts, dim=-1)
    return logits[..., :FULL_VOCAB].cpu().float()


# ---------------------------------------------------------------------------
# HF reference logits
# ---------------------------------------------------------------------------


def _hf_logits_text(args, tokens):
    from transformers import AutoModelForCausalLM

    hf_dtype = torch.bfloat16 if args.bf16 else torch.float32
    print_rank_0(f"\nLoading HF model (CausalLM) from {args.hf_dir} ...")
    hf = AutoModelForCausalLM.from_pretrained(args.hf_dir, torch_dtype=hf_dtype, device_map="cuda:0")
    hf.eval()
    with torch.no_grad():
        logits = hf(input_ids=tokens, output_hidden_states=False).logits
    del hf
    torch.cuda.empty_cache()
    return logits[..., :FULL_VOCAB].cpu().float()


def _load_hf_conditional_generation(hf_dir, dtype):
    """Load Gemma4ForConditionalGeneration regardless of transformers version.

    - transformers >= 4.46: AutoModelForVision2Seq is available.
    - older versions: fall back to direct class import.
    """
    try:
        from transformers import AutoModelForVision2Seq

        return AutoModelForVision2Seq.from_pretrained(hf_dir, torch_dtype=dtype, device_map="cuda:0")
    except ImportError:
        pass
    # Fallback: import the class from the models submodule directly
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

    return Gemma4ForConditionalGeneration.from_pretrained(hf_dir, torch_dtype=dtype, device_map="cuda:0")


def _hf_logits_vl(args, input_ids_vl, pixel_values, image_position_ids):
    hf_dtype = torch.bfloat16 if args.bf16 else torch.float32
    print_rank_0(f"\nLoading HF model (VL) from {args.hf_dir} ...")
    hf = _load_hf_conditional_generation(args.hf_dir, hf_dtype)
    hf.eval()
    hf_input_ids = input_ids_vl.to("cuda:0")
    hf_pixel_values = pixel_values.to("cuda:0", hf_dtype)
    hf_image_position_ids = image_position_ids.to("cuda:0")
    mm_token_type_ids = torch.zeros_like(hf_input_ids)
    mm_token_type_ids[hf_input_ids == getattr(hf.config, "image_token_id", IMAGE_TOKEN_ID)] = 1
    with torch.no_grad():
        logits = hf(
            input_ids=hf_input_ids,
            pixel_values=hf_pixel_values,
            image_position_ids=hf_image_position_ids,
            mm_token_type_ids=mm_token_type_ids,
        ).logits
    del hf
    torch.cuda.empty_cache()
    return logits[..., :FULL_VOCAB].cpu().float()


def _hf_logits_audio(args, input_ids_audio, audio_features):
    """HF audio parity: Gemma4ForConditionalGeneration with input_features."""
    hf_dtype = torch.bfloat16 if args.bf16 else torch.float32
    print_rank_0(f"\nLoading HF model (VL+Audio) from {args.hf_dir} ...")
    hf = _load_hf_conditional_generation(args.hf_dir, hf_dtype)
    hf.eval()
    hf_audio = audio_features.to("cuda:0", hf_dtype)
    hf_audio_mask = torch.ones(
        hf_audio.shape[:2],
        dtype=torch.bool,
        device=hf_audio.device,
    )
    with torch.no_grad():
        logits = hf(
            input_ids=input_ids_audio,
            input_features=hf_audio,
            input_features_mask=hf_audio_mask,
            pixel_values=None,
        ).logits
    del hf
    torch.cuda.empty_cache()
    return logits[..., :FULL_VOCAB].cpu().float()


# ---------------------------------------------------------------------------
# Synthetic multimodal parity inputs
# ---------------------------------------------------------------------------


def _vl_grid_for_tokens(n_tokens: int):
    """Return (grid_h, grid_w) preserving the standard 42:60 (=7:10) aspect ratio.

    The standard grid is 42×60 → 280 tokens.  For other counts we find (H,W)
    with H*W=n_tokens and H/W≈7/10, then multiply by 3 to get the patch grid.
    Falls back to a 3×(3*n_tokens) horizontal strip if no factorisation fits.
    """
    target_ratio = 42 / 60  # 0.7
    best = None
    best_err = float("inf")
    for h in range(1, n_tokens + 1):
        if n_tokens % h == 0:
            w = n_tokens // h
            err = abs(h / w - target_ratio)
            if err < best_err:
                best_err = err
                best = (h, w)
    h_tok, w_tok = best
    return 3 * h_tok, 3 * w_tok


def _make_vl_inputs(dtype, n_tokens: int = IMAGE_NUM_TOKENS):
    """Create synthetic patch tensors for VL parity with ``n_tokens`` image tokens.

    The patch grid is chosen to preserve the 42:60 aspect ratio so that
    pixel_position_ids stay comparable across different token counts.
    The default (280) uses the standard Gemma4 42×60 grid.
    """
    grid_h, grid_w = _vl_grid_for_tokens(n_tokens)
    num_patches = grid_h * grid_w

    image_pos = torch.full((BATCH, n_tokens), IMAGE_TOKEN_ID, dtype=torch.long)
    text_pos = torch.arange(VL_TEXT_TOKENS, dtype=torch.long).unsqueeze(0)
    input_ids_vl = torch.cat([image_pos, text_pos], dim=1).cuda()

    torch.manual_seed(42)
    pixel_values = torch.rand(BATCH, num_patches, IMAGE_PATCH_DIM, dtype=dtype).cuda()

    grid_x, grid_y = torch.meshgrid(
        torch.arange(grid_w),
        torch.arange(grid_h),
        indexing="xy",
    )
    image_position_ids = torch.stack([grid_x, grid_y], dim=-1)
    image_position_ids = image_position_ids.reshape(1, num_patches, 2).cuda()
    return input_ids_vl, pixel_values, image_position_ids


def _make_audio_inputs(dtype):
    # input_ids: first AUDIO_NUM_TOKENS are audio_token_id, rest are normal text tokens
    audio_pos = torch.full((BATCH, AUDIO_NUM_TOKENS), AUDIO_TOKEN_ID, dtype=torch.long)
    text_pos = torch.arange(SEQ - AUDIO_NUM_TOKENS, dtype=torch.long).unsqueeze(0)
    input_ids_audio = torch.cat([audio_pos, text_pos], dim=1).cuda()

    # Fixed dummy mel-spectrogram: [BATCH, AUDIO_INPUT_FRAMES, AUDIO_MEL_BINS]
    torch.manual_seed(42)
    audio_features = torch.randn(
        BATCH,
        AUDIO_INPUT_FRAMES,
        AUDIO_MEL_BINS,
        dtype=dtype,
    ).cuda()
    return input_ids_audio, audio_features


# ---------------------------------------------------------------------------
# Comparison reporting
# ---------------------------------------------------------------------------


def _report(mode, megatron_logits, hf_logits, atol, seq_len=None):
    if seq_len is None:
        seq_len = SEQ
    mode_labels = {
        "text": "Megatron GPTModel (text)              vs  HF Gemma4ForCausalLM",
        "vl": "Megatron Gemma4VLModel (image forward)  vs  HF Gemma4ForConditionalGeneration",
        "audio": "Megatron Gemma4VLModel (audio forward)  vs  HF Gemma4ForConditionalGeneration",
    }
    diff = (megatron_logits - hf_logits).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    per_token_max = diff[0].max(dim=-1).values
    top3 = per_token_max.topk(min(3, seq_len))

    print_rank_0(f"\n{'=' * 70}")
    print_rank_0(f"  Parity [{mode.upper()}]: {mode_labels[mode]}")
    print_rank_0(f"  seq={seq_len}  batch={BATCH}  vocab={FULL_VOCAB}")
    print_rank_0(f"  max |diff|  : {max_diff:.6f}  (atol={atol})")
    print_rank_0(f"  mean |diff| : {mean_diff:.6f}")
    print_rank_0(
        f"  worst token positions: {top3.indices.tolist()} (diffs: {[f'{v:.4f}' for v in top3.values.tolist()]})"
    )
    status = "PASSED" if max_diff <= atol else "FAILED"
    print_rank_0(f"  --> {status}")
    print_rank_0(f"{'=' * 70}\n")
    return status == "PASSED"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Run the requested Gemma4 parity check."""
    args = _parse()

    pretrain_gpt = os.path.join(MEGATRON_LM_ROOT, "pretrain_gpt.py")
    if not os.path.isfile(pretrain_gpt):
        sys.exit(f"Error: Megatron-LM root not found: {MEGATRON_LM_ROOT}")
    os.chdir(MEGATRON_LM_ROOT)

    vl_n_tokens = args.vl_image_tokens  # may differ from IMAGE_NUM_TOKENS
    if args.mode == "vl":
        seq_len = vl_n_tokens + VL_TEXT_TOKENS
    else:
        seq_len = _seq_len_for_mode(args.mode)
    sys.argv = _build_megatron_argv(args.megatron_ckpt, tp=args.tp, bf16=args.bf16, seq=seq_len)

    from megatron.core import mpu
    from megatron.training.arguments import parse_and_validate_args
    from megatron.training.checkpointing import load_checkpoint
    from megatron.training.initialize import initialize_megatron

    parse_and_validate_args()
    initialize_megatron()
    rank = dist.get_rank()

    print_rank_0(f"Parity mode: {args.mode.upper()}", rank=rank)

    # Build model
    if args.mode == "text":
        models = _build_text_models(args)
    elif args.mode == "vl":
        models = _build_vl_models(args, seq_len=seq_len, include_audio=False)
    else:  # audio
        models = _build_vl_models(args, seq_len=seq_len, include_audio=True)

    model = models[0]
    load_checkpoint(models, None, None)
    model.eval()

    # Prepare inputs
    tokens = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).cuda()
    input_dtype = torch.bfloat16 if args.bf16 else torch.float32

    if args.mode == "vl":
        input_ids_vl, pixel_values, image_position_ids = _make_vl_inputs(input_dtype, n_tokens=vl_n_tokens)
    elif args.mode == "audio":
        input_ids_audio, audio_features = _make_audio_inputs(input_dtype)

    # Megatron forward
    if args.mode == "text":
        logits = _forward_text(model, tokens)
    elif args.mode == "vl":
        logits = _forward_vl(model, input_ids_vl, pixel_values, image_position_ids)
    else:
        logits = _forward_audio(model, input_ids_audio, audio_features)

    megatron_logits = _gather_logits(logits, mpu)

    del model, models, logits
    torch.cuda.empty_cache()

    fail_flag = torch.tensor([0], dtype=torch.int32).cuda()

    if rank == 0:
        if args.mode == "text":
            hf_logits = _hf_logits_text(args, tokens)
        elif args.mode == "vl":
            hf_logits = _hf_logits_vl(args, input_ids_vl, pixel_values.cpu(), image_position_ids.cpu())
        else:
            hf_logits = _hf_logits_audio(args, input_ids_audio, audio_features.cpu())

        passed = _report(args.mode, megatron_logits, hf_logits, args.atol, seq_len=seq_len)
        if not passed:
            fail_flag.fill_(1)

    dist.broadcast(fail_flag, src=0)
    dist.barrier()
    if fail_flag.item() == 1:
        sys.exit(1)


if __name__ == "__main__":
    main()
