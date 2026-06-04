# Gemma 4 E4B Support

**Gemma 4 E4B** (3.8B dense text model) integration for Megatron-Bridge, including HuggingFace checkpoint conversion, numerical parity verification, and TP-distributed training.

Works with **clean Megatron-Core** — no Gemma4-specific CLI arguments or `TransformerConfig` fields are required in MCore. All Gemma4 specifics live in Bridge via `Gemma4DenseProvider`, `Gemma4DenseVLProvider`, and `Gemma4VLModel`.

## What's included

| File | Purpose |
|------|---------|
| `src/megatron/bridge/models/gemma_vl/modeling_gemma4_vl.py` | Layer spec, attention, dual-RoPE, PLE, shared-KV, `Gemma4DenseProvider`, `Gemma4VLModel` |
| `src/megatron/bridge/models/gemma_vl/gemma4_vl_provider.py` | `Gemma4DenseVLProvider` (Dense VL), `Gemma4VLModelProvider` (MoE VL), `Gemma4ModelProvider` (MoE text) |
| `src/megatron/bridge/models/gemma_vl/gemma4_vl_bridge.py` | Bridge-native HF↔Megatron conversion (`Gemma4VLBridge` for E4B HF checkpoints) |
| `examples/models/gemma/gemma4/parity_check_e4b.py` | Distributed parity check — text, vl, and audio modes |
| `examples/models/gemma/gemma4/slurm_pretrain.sh` | Full pipeline: text convert → vl/audio convert → parity checks → training |
| `tests/unit_tests/models/gemma_vl/test_gemma4_vl_provider.py` | Provider unit tests |
| `tests/unit_tests/models/gemma_vl/test_gemma4_vl_bridge.py` | Bridge mapping unit tests |
| `tests/unit_tests/models/gemma_vl/test_gemma4_vl_modeling.py` | VL model unit tests |

## Quick start

**Step 1a — Convert HuggingFace weights (text-only, for training):**

```bash
export MEGATRON_LM_ROOT=/path/to/Megatron-LM
export PYTHONPATH=$PWD/src:$MEGATRON_LM_ROOT
export GEMMA4_CONVERSION_MODE=text

torchrun --nproc_per_node=2 \
    examples/conversion/convert_checkpoints_multi_gpu.py import \
    --hf-model /path/to/gemma-4-E4B-it \
    --megatron-path /path/to/gemma4-e4b-megatron-text \
    --tp 2 --pp 1 --torch-dtype bfloat16
```

**Step 1b — Convert HuggingFace weights (VL/audio, for multimodal parity):**

```bash
export GEMMA4_CONVERSION_MODE=audio

torchrun --nproc_per_node=2 \
    examples/conversion/convert_checkpoints_multi_gpu.py import \
    --hf-model /path/to/gemma-4-E4B-it \
    --megatron-path /path/to/gemma4-e4b-megatron-vl \
    --tp 2 --pp 1 --torch-dtype bfloat16
```

**Step 2 — Verify conversion (logit parity, all 3 modalities):**

```bash
# Text parity (GPTModel vs HF Gemma4ForCausalLM)
CUDA_DEVICE_MAX_CONNECTIONS=1 \
torchrun --nproc_per_node=2 \
    examples/models/gemma/gemma4/parity_check_e4b.py \
    --hf-dir /path/to/gemma-4-E4B-it \
    --megatron-ckpt /path/to/gemma4-e4b-megatron-text \
    --tp 2 --bf16 --mode text --atol 3.0

# VL parity (language_model path of Gemma4VLModel vs HF conditional generation)
CUDA_DEVICE_MAX_CONNECTIONS=1 \
torchrun --nproc_per_node=2 \
    examples/models/gemma/gemma4/parity_check_e4b.py \
    --hf-dir /path/to/gemma-4-E4B-it \
    --megatron-ckpt /path/to/gemma4-e4b-megatron-vl \
    --tp 2 --bf16 --mode vl --atol 3.0

# Audio parity (full audio forward of Gemma4VLModel vs HF conditional generation)
CUDA_DEVICE_MAX_CONNECTIONS=1 \
torchrun --nproc_per_node=2 \
    examples/models/gemma/gemma4/parity_check_e4b.py \
    --hf-dir /path/to/gemma-4-E4B-it \
    --megatron-ckpt /path/to/gemma4-e4b-megatron-vl \
    --tp 2 --bf16 --mode audio --atol 3.0
```

Expected results:
- fp32: `max |diff|: ~0.15  (atol=0.3)  --> PASSED`
- bf16: `max |diff|: ~2.94  (atol=3.0)  --> PASSED`

**Or run all steps at once (convert → parity → training):**

```bash
NVIDIA_VISIBLE_DEVICES=0,1 \
HF_MODEL_DIR=/path/to/gemma-4-E4B-it \
MEGATRON_CKPT=/path/to/gemma4-e4b-megatron \
TRAIN_DATA_PATH=/path/to/data \
bash examples/models/gemma/gemma4/slurm_pretrain.sh
```

The script derives two checkpoint paths automatically:
- `${MEGATRON_CKPT}-text` — text-only conversion, used for training
- `${MEGATRON_CKPT}-vl` — VL/audio conversion, used for vl and audio parity checks

## Running tests

Provider and bridge unit tests:

```bash
PYTHONPATH=$PWD/src python -m pytest \
    tests/unit_tests/models/gemma_vl/test_gemma4_vl_provider.py \
    tests/unit_tests/models/gemma_vl/test_gemma4_vl_bridge.py \
    tests/unit_tests/models/gemma_vl/test_gemma4_vl_modeling.py \
    -v
```

Multi-GPU tests (TP=2, requires 2 GPUs):

```bash
NVIDIA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    -m pytest tests/unit_tests/models/gemma_vl -v -k "TensorParallel"
```

## Implemented components

- **Attention**: GQA, mixed sliding-window / full-attention, layer-dependent head dimension (`kv_channels=256` sliding, `global_kv_channels=512` global), attention normalization (q/k layernorm)
- **RoPE**: dual RoPE (sliding θ=10000 full rotation, global θ=1000000 partial-factor=0.25), handled by `Gemma4DenseRotaryEmbedding` in `modeling_gemma4_vl.py`
- **Per-Layer Embeddings (PLE)**: `embed_tokens_per_layer` weight mapping; per-layer projection forwarded through transformer blocks via MCore's generic `per_layer_inputs` hook in `TransformerBlock`
- **Shared KV layers**: last 18 layers reuse KV from earlier layers, wired post-construction by `wire_gemma4_kv_sharing()`
- **GEGLU activation**: tanh-approximate GELU matching HF `gelu_pytorch_tanh`, handled automatically by Bridge's `GatedMLPMapping` (interleaved TP split)
- **Logit softcapping**: `final_logit_softcapping=30.0` applied inside `Gemma4DenseProvider`
- **Vision support**: HF vision tower + `Gemma4MultimodalEmbedder`, features scattered at `image_token_id` positions; bidirectional attention mask within image blocks
- **Audio support**: HF audio tower (12-layer transformer, 128-bin mel input, 4× subsampling, 1024→1536 projection) + `Gemma4AudioEmbedder` (1536→2560); features scattered at `audio_token_id` positions with bidirectional attention mask
- **Checkpoint conversion**: Bridge-native via `Gemma4VLBridge` registered for `Gemma4ForConditionalGeneration`; QKV/GEGLU/PLE handled by `GatedMLPMapping`, `_Gemma4E4BQKVMapping`, `AutoMapping`
- **`Gemma4DenseProvider`**: builds `TransformerConfig`, injects Gemma4 attrs, replaces `rotary_pos_emb`, attaches PLE modules, patches `forward()` for PLE computation, wires shared-KV
- **`Gemma4DenseVLProvider`**: wraps `Gemma4DenseProvider` inside `Gemma4VLModel` to add vision/audio encoders and multimodal scatter logic

## Bridge conversion architecture

```
AutoBridge.from_hf_pretrained("google/gemma-4-E4B-it")
  └─ Gemma4VLBridge                      # registered for Gemma4ForConditionalGeneration
       ├─ provider_bridge()               # text mode → Gemma4DenseProvider (text-only pretraining)
       │                                  # vl/audio mode → Gemma4DenseVLProvider (full VL+Audio)
       ├─ _dense_e4b_mapping_registry()   # language mappings (4 norms, QKV, GEGLU, PLE, ...)
       └─ maybe_modify_loaded_hf_weight() # shared-KV: synthesize zero K/V rows
                                          # (last 18 layers have no k/v proj in HF)
```

### Parity check modes

| Mode | Megatron model | HF model | Checkpoint |
|------|---------------|----------|-----------|
| `text` | `Gemma4DenseProvider` → `GPTModel` | `Gemma4ForCausalLM` | `*-text` |
| `vl` | `Gemma4DenseVLProvider` → `Gemma4VLModel.language_model` | `Gemma4ForConditionalGeneration` (pixel_values=None) | `*-vl` |
| `audio` | `Gemma4DenseVLProvider` → `Gemma4VLModel` (full forward) | `Gemma4ForConditionalGeneration` (with input_features) | `*-vl` |
