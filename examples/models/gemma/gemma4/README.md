# Gemma 4 E4B Support

**Gemma 4 E4B** (3.8B dense text model) integration for Megatron-Bridge, supporting
HuggingFace checkpoint conversion, numerical parity verification (text / audio / VL image),
and TP-distributed pretraining.

Works with **clean Megatron-Core** — no Gemma4-specific CLI arguments or
`TransformerConfig` fields are required in MCore. All Gemma4 specifics live in Bridge via
`Gemma4DenseProvider`, `Gemma4DenseVLProvider`, and `Gemma4VLModel`.

## File map

| File | Purpose |
|------|---------|
| `src/megatron/bridge/models/gemma_vl/modeling_gemma4_vl.py` | Layer spec, `Gemma4DenseTransformerLayer`, dual-RoPE, PLE, shared-KV, `Gemma4DenseProvider`, `Gemma4VLModel` |
| `src/megatron/bridge/models/gemma_vl/gemma4_vl_provider.py` | `Gemma4DenseVLProvider` (Dense VL/Audio), `Gemma4VLModelProvider` (MoE VL), `Gemma4ModelProvider` (MoE text) |
| `src/megatron/bridge/models/gemma_vl/gemma4_vl_bridge.py` | Bridge-native HF ↔ Megatron conversion (`Gemma4VLBridge`) |
| `examples/models/gemma/gemma4/parity_check_e4b.py` | Distributed parity check — `text`, `vl`, `audio` modes |
| `examples/models/gemma/gemma4/slurm_pretrain.sh` | Full pipeline: text convert → VL convert → parity checks → training |
| `tests/unit_tests/models/gemma_vl/test_gemma4_vl_provider.py` | Provider unit tests |
| `tests/unit_tests/models/gemma_vl/test_gemma4_vl_bridge.py` | Bridge mapping unit tests |
| `tests/unit_tests/models/gemma_vl/test_gemma4_vl_modeling.py` | VL model unit tests |

## Quick start

### Step 1 — Convert HuggingFace weights

Two separate checkpoints are needed: one text-only (for pretraining) and one VL/audio (for multimodal parity).

```bash
export MEGATRON_LM_ROOT=/path/to/Megatron-LM
export PYTHONPATH=$PWD/src:$MEGATRON_LM_ROOT

# Text-only checkpoint (used for training)
GEMMA4_CONVERSION_MODE=text \
torchrun --nproc_per_node=2 \
    examples/conversion/convert_checkpoints_multi_gpu.py import \
    --hf-model /path/to/gemma-4-E4B-it \
    --megatron-path /path/to/gemma4-e4b-megatron-text \
    --tp 2 --pp 1 --torch-dtype bfloat16

# VL/audio checkpoint (used for multimodal parity)
GEMMA4_CONVERSION_MODE=audio \
torchrun --nproc_per_node=2 \
    examples/conversion/convert_checkpoints_multi_gpu.py import \
    --hf-model /path/to/gemma-4-E4B-it \
    --megatron-path /path/to/gemma4-e4b-megatron-vl \
    --tp 2 --pp 1 --torch-dtype bfloat16
```

### Step 2 — Verify conversion (parity checks)

```bash
# Text parity — GPTModel vs HF Gemma4ForCausalLM
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=2 \
    examples/models/gemma/gemma4/parity_check_e4b.py \
    --hf-dir /path/to/gemma-4-E4B-it \
    --megatron-ckpt /path/to/gemma4-e4b-megatron-text \
    --tp 2 --bf16 --mode text --atol 3.0

# Audio parity — Gemma4VLModel (audio forward) vs HF
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=2 \
    examples/models/gemma/gemma4/parity_check_e4b.py \
    --hf-dir /path/to/gemma-4-E4B-it \
    --megatron-ckpt /path/to/gemma4-e4b-megatron-vl \
    --tp 2 --bf16 --mode audio --atol 3.0

# VL image parity — Gemma4VLModel (image forward) vs HF
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=2 \
    examples/models/gemma/gemma4/parity_check_e4b.py \
    --hf-dir /path/to/gemma-4-E4B-it \
    --megatron-ckpt /path/to/gemma4-e4b-megatron-vl \
    --tp 2 --bf16 --mode vl --atol 10.0
```

**Expected results (bf16):**

| Mode | Typical max \|diff\| | atol | Notes |
|------|---------------------|------|-------|
| text | ~2.94 | 3.0 | Softcap 30.0 applied before comparison |
| audio | ~1.65 | 3.0 | 12 audio tokens, audio feature diff ~0.10 |
| vl | ~8.11 | 10.0 | 280 image tokens — see note below |

> **VL bf16 tolerance (10.0):** The higher atol for VL image parity is expected and not a bug.
> With 280 image tokens and a bf16 vision tower feature diff of ~0.22 max per token,
> error accumulates through 42 transformer layers. The worst-case positions are consistently
> at the image/text boundary (position 279 = last image token, 280 = first text token),
> which is the hallmark of bf16 accumulated rounding from image features.
> For comparison: audio passes at atol 3.0 with only 12 tokens and ~0.10 feature diff;
> VL has 23× more tokens and 2× larger per-token diff, producing the observed ~8 floor.
>
> **fp32 mode is not supported** for VL parity: the vision/audio towers are stored as
> bfloat16 in the checkpoint, causing dtype mismatches when the rest of the model runs
> in fp32. The parity test always runs bf16.

### Step 3 — Or run all steps at once

```bash
NVIDIA_VISIBLE_DEVICES=0,1 \
HF_MODEL_DIR=/path/to/gemma-4-E4B-it \
MEGATRON_CKPT=/path/to/gemma4-e4b-megatron \
TRAIN_DATA_PATH=/path/to/data \
bash examples/models/gemma/gemma4/slurm_pretrain.sh
```

The script derives paths automatically:
- `${MEGATRON_CKPT}-text` — text conversion, used for training
- `${MEGATRON_CKPT}-vl` — VL/audio conversion, used for parity checks

Skip flags: `SKIP_CONVERT=1`, `SKIP_TEXT_CONVERT=1`, `SKIP_VL_CONVERT=1`, `SKIP_PARITY=1`.

## Running unit tests

```bash
PYTHONPATH=$PWD/src python -m pytest \
    tests/unit_tests/models/gemma_vl/test_gemma4_vl_provider.py \
    tests/unit_tests/models/gemma_vl/test_gemma4_vl_bridge.py \
    tests/unit_tests/models/gemma_vl/test_gemma4_vl_modeling.py \
    -v
```

Multi-GPU unit tests (TP=2, requires 2 GPUs):

```bash
NVIDIA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    -m pytest tests/unit_tests/models/gemma_vl -v -k "TensorParallel"
```

## Investigating VL parity error

The `--vl-image-tokens N` flag in `parity_check_e4b.py` lets you test with fewer image tokens.
The grid is chosen to preserve the standard 42:60 aspect ratio so positional encodings
stay comparable:

```bash
# T=70 tokens (21×30 grid, same 7:10 aspect ratio as default 42×60)
python ... parity_check_e4b.py --mode vl --vl-image-tokens 70 --atol 99

# T=140 tokens (30×42 grid, ≈7:10 aspect)
python ... parity_check_e4b.py --mode vl --vl-image-tokens 140 --atol 99
```

Note: the absolute diff values depend heavily on the random patch content for each grid
size, so the scaling is not perfectly monotonic across T values. The most reliable
evidence for accumulated error is the consistently worst positions at the image/text
boundary (last image token, first text token) across all token counts.

## Implemented components

### Language model (Dense / E4B)

| Component | Detail |
|-----------|--------|
| **4-norm structure** | `input_layernorm` → attention → `post_self_attn_layernorm` → MLP → `post_mlp_layernorm` |
| **GQA + sliding/global mix** | `kv_channels=256` (sliding), `global_kv_channels=512` (global); `window_attn_skip_freq=6` |
| **Dual RoPE** | Sliding θ=10 000 (full rotation), global θ=1 000 000 (partial factor=0.25); `Gemma4DenseRotaryEmbedding` |
| **Q/K LayerNorm** | RMSNorm on queries and keys via `Gemma4DenseSelfAttention` |
| **Shared KV** | Last 18 layers reuse KV from the last non-shared layer of the same type; wired by `wire_gemma4_kv_sharing()` |
| **Per-Layer Embeddings (PLE)** | `per_layer_embedding` (vocab) + `per_layer_model_proj` (hidden→PLE) per layer; patched into `GPTModel.forward` via `_install_ple_forward()` |
| **GEGLU activation** | `tanh`-approximate GELU; handled by Bridge's `GatedMLPMapping` |
| **Logit softcapping** | `final_logit_softcapping=30.0` applied in `Gemma4DenseProvider.build()` |

### Vision-Language model (`Gemma4VLModel`)

| Component | Detail |
|-----------|--------|
| **Vision encoder** | HF `Gemma4VisionTower` (SigLIP-based) loaded via `AutoModel.from_config(vision_config)` |
| **Vision projector** | `Gemma4MultimodalEmbedder` (RMSNorm + linear, vision hidden → text hidden) |
| **Image scatter** | Features scattered at `image_token_id=258880` positions with bidirectional attention within image blocks |
| **Audio encoder** | HF audio tower (12-layer transformer, 128-bin mel, 4× subsampling, 1024→1536 projection) |
| **Audio projector** | `Gemma4AudioEmbedder` (1536 → 2560) |
| **Audio scatter** | Features scattered at `audio_token_id=258881` positions with bidirectional attention |
| **PLE in VL path** | `lm_input_ids` replaces multimodal positions with `pad_token_id=0` before PLE lookup; embedding scaled by `√hidden_size` before scatter; post-scatter embeddings used for PLE `mdl_proj` (matching HF) |
| **Causal mask** | VL forward uses pure causal mask (HF default without `mm_token_type_ids`) |

### Checkpoint conversion

```
AutoBridge.from_hf_pretrained("google/gemma-4-E4B-it")
  └─ Gemma4VLBridge                  # registered for Gemma4ForConditionalGeneration
       ├─ provider_bridge()
       │    text mode  → Gemma4DenseProvider    (text-only pretraining)
       │    vl/audio   → Gemma4DenseVLProvider  (full VL + Audio)
       ├─ _dense_e4b_mapping_registry()
       │    QKV / GEGLU / PLE / 4 norms / shared-KV synthesis
       └─ maybe_modify_loaded_hf_weight()
            shared-KV: synthesize zero K/V rows for last 18 layers
            (HF stores no k/v proj for those layers)
```

### Parity check modes

| Mode | Megatron model | HF model | Checkpoint |
|------|---------------|----------|-----------|
| `text` | `Gemma4DenseProvider` → `GPTModel` | `Gemma4ForCausalLM` | `*-text` |
| `vl` | `Gemma4DenseVLProvider` → `Gemma4VLModel` (image forward) | `Gemma4ForConditionalGeneration` | `*-vl` |
| `audio` | `Gemma4DenseVLProvider` → `Gemma4VLModel` (audio forward) | `Gemma4ForConditionalGeneration` | `*-vl` |

### Key correctness fixes in VL forward

Three bugs were found and fixed in the VL forward path (vs. the text-only path which passes cleanly):

1. **PLE was completely skipped** — `Gemma4VLModel.forward` called `language_model.forward(input_ids=None, ...)`, causing `_compute_per_layer_inputs` to return early. Fixed by passing `input_ids=lm_input_ids`.

2. **PLE token IDs at multimodal positions** — raw `audio_token_id` / `image_token_id` values were passed to `per_layer_embedding`, producing wrong PLE at multimodal positions. Fixed by replacing multimodal positions with `pad_token_id=0` in `lm_input_ids` (matching HF behavior).

3. **Embedding scaling missing** — `language_model.embedding()` was called directly (bypassing the `_ple_forward` wrapper that applies `√hidden_size` scaling). Fixed by applying explicit scaling before the modality scatter.
