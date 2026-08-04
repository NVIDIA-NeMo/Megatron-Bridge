# Gemma 4 E4B Examples

This directory contains example scripts for the Gemma 4 E4B dense model.

Gemma 4 E4B is a dense Gemma 4 variant with text, vision, and audio support in
the Hugging Face checkpoint. The Bridge implementation keeps the text-only path
and the vision/audio path separated:

- `Gemma4ForCausalLM` is handled by `Gemma4Bridge` in
  `megatron.bridge.models.gemma`.
- `Gemma4ForConditionalGeneration` is handled by `Gemma4VLBridge` in
  `megatron.bridge.models.gemma_vl`.
- Shared language-model modules live under `megatron.bridge.models.gemma`; VL
  modules extend that implementation without introducing a reverse dependency.

## Requirements

Gemma 4 requires a Megatron-Core checkout on `PYTHONPATH`. The Bridge Gemma 4
provider is designed to work with a clean Megatron-Core checkout: Gemma 4
specific features such as dual RoPE, per-layer embeddings, shared KV, and
embedding scaling are implemented or patched on the Bridge side rather than as
Gemma 4 specific Megatron-Core arguments or `TransformerConfig` fields.

Set `MEGATRON_LM_ROOT` to your Megatron-LM repository:

```bash
export MEGATRON_LM_ROOT=/path/to/Megatron-LM
export PYTHONPATH=$PWD/src:${MEGATRON_LM_ROOT}:${PYTHONPATH:-}
```

Gemma 4 checkpoints may require a recent `transformers` version:

```bash
uv pip install -q --upgrade 'transformers>=5.5.0'
```

The conversion and inference scripts use `uv run --no-sync` where they depend on
the current Python environment package versions. Distributed launch examples use
`uv run python -m torch.distributed.run`, following the repository convention.

## Workspace Configuration

The examples below use a `WORKSPACE` environment variable to keep checkpoints,
logs, and results in one place:

```bash
export WORKSPACE=/your/custom/path
```

Suggested directory structure:
- `${WORKSPACE}/models/` - Converted Megatron checkpoints
- `${WORKSPACE}/results/` - Training outputs and experiment results
- `${WORKSPACE}/logs/` - Parity and training logs

`slurm_pretrain.sh` also requires `GEMMA4_LOG_ROOT` for parity and training
logs:

```bash
export GEMMA4_LOG_ROOT=${WORKSPACE}/logs
```

## Checkpoint Conversion

Gemma 4 E4B has two useful conversion modes:

- `GEMMA4_CONVERSION_MODE=text` imports the text-only GPTModel path, used for
  text pretraining and text generation.
- `GEMMA4_CONVERSION_MODE=audio` imports the full VL/audio model path, used for
  multimodal parity checks.

### Import HF → Megatron (text)

```bash
GEMMA4_CONVERSION_MODE=text \
./scripts/conversion/convert.sh import \
    --hf-model google/gemma-4-E4B-it \
    --megatron-path ${WORKSPACE}/models/gemma-4-E4B-it
```

### Import HF → Megatron (VL/audio)

```bash
GEMMA4_CONVERSION_MODE=audio \
./scripts/conversion/convert.sh import \
    --hf-model google/gemma-4-E4B-it \
    --megatron-path ${WORKSPACE}/models/gemma-4-E4B-it-vl
```

### Export Megatron → HF

```bash
./scripts/conversion/convert.sh export \
    --hf-model google/gemma-4-E4B-it \
    --megatron-path ${WORKSPACE}/models/gemma-4-E4B-it/iter_0000000 \
    --hf-path ${WORKSPACE}/models/gemma-4-E4B-it-hf-export
```

### Round-trip validation

```bash
GEMMA4_CONVERSION_MODE=text \
uv run --no-sync python -m torch.distributed.run --nproc_per_node=2 \
    examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id google/gemma-4-E4B-it \
    --output-dir ${WORKSPACE}/results/gemma-4-E4B-it-roundtrip \
    --tp 2 --pp 1
```

See [conversion.sh](conversion.sh) for the full text-only import, export, and
round-trip workflow.

## Inference

Text-only inference uses `hf_to_megatron_generate_text.py` with
`GEMMA4_CONVERSION_MODE=text` so the bridge selects `Gemma4Bridge` and builds a
`GPTModel`, not the full `Gemma4VLModel`.

### Text generation from HF weights

```bash
GEMMA4_CONVERSION_MODE=text \
uv run --no-sync python -m torch.distributed.run --nproc_per_node=2 \
    examples/conversion/hf_to_megatron_generate_text.py \
    --hf_model_path google/gemma-4-E4B-it \
    --prompt $'<start_of_turn>user\nWhat is the capital of France?<end_of_turn>\n<start_of_turn>model\n' \
    --max_new_tokens 20 \
    --tp 2 --pp 1
```

### Text generation from imported Megatron checkpoint

```bash
GEMMA4_CONVERSION_MODE=text \
uv run --no-sync python -m torch.distributed.run --nproc_per_node=2 \
    examples/conversion/hf_to_megatron_generate_text.py \
    --hf_model_path google/gemma-4-E4B-it \
    --megatron_model_path ${WORKSPACE}/models/gemma-4-E4B-it/iter_0000000 \
    --prompt $'<start_of_turn>user\nExplain entropy in one sentence.<end_of_turn>\n<start_of_turn>model\n' \
    --max_new_tokens 50 \
    --tp 2 --pp 1
```

See [inference.sh](inference.sh) for both examples.

> **Note:** `google/gemma-4-E4B-it` is instruction tuned. For high-quality
> assistant-style responses, use prompts and tokenization compatible with the
> model's chat template. The simple generation script is intended as a Bridge
> smoke test, not a production serving path.

## Parity Checks

[parity_check_e4b.py](parity_check_e4b.py) compares Megatron logits against the
Hugging Face model in three modes:

| Mode | Megatron model | HF model | Checkpoint |
|------|---------------|----------|------------|
| `text` | `Gemma4DenseProvider` → `GPTModel` | `Gemma4ForCausalLM` | text checkpoint |
| `vl` | `Gemma4DenseVLProvider` → `Gemma4VLModel` | `Gemma4ForConditionalGeneration` | VL/audio checkpoint |
| `audio` | `Gemma4DenseVLProvider` → `Gemma4VLModel` | `Gemma4ForConditionalGeneration` | VL/audio checkpoint |

### Text parity

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 uv run --no-sync python -m torch.distributed.run --nproc_per_node=2 \
    examples/models/gemma/gemma4/parity_check_e4b.py \
    --hf-dir /path/to/gemma-4-E4B-it \
    --megatron-ckpt ${WORKSPACE}/models/gemma-4-E4B-it \
    --tp 2 --bf16 --mode text --atol 3.0
```

### Audio parity

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 uv run --no-sync python -m torch.distributed.run --nproc_per_node=2 \
    examples/models/gemma/gemma4/parity_check_e4b.py \
    --hf-dir /path/to/gemma-4-E4B-it \
    --megatron-ckpt ${WORKSPACE}/models/gemma-4-E4B-it-vl \
    --tp 2 --bf16 --mode audio --atol 3.0
```

### Vision parity

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 uv run --no-sync python -m torch.distributed.run --nproc_per_node=2 \
    examples/models/gemma/gemma4/parity_check_e4b.py \
    --hf-dir /path/to/gemma-4-E4B-it \
    --megatron-ckpt ${WORKSPACE}/models/gemma-4-E4B-it-vl \
    --tp 2 --bf16 --mode vl --atol 6.0
```

Expected bf16 results:

| Mode | Typical max \|diff\| | atol | Notes |
|------|----------------------|------|-------|
| text | ~2.94 | 3.0 | Softcap 30.0 applied before comparison |
| audio | ~1.65 | 3.0 | 12 audio tokens |
| vl | ~5.47 | 6.0 | 280 image tokens |

The higher VL tolerance is expected. The image path injects many more modality
tokens than the audio path, and bf16 vision feature differences accumulate
through the language model. The worst positions are usually at the image/text
boundary.

## Pretraining

[slurm_pretrain.sh](slurm_pretrain.sh) runs the full workflow:

1. Convert the text checkpoint.
2. Convert the VL/audio checkpoint.
3. Run text, audio, and VL parity checks.
4. Launch Gemma 4 E4B text pretraining.

```bash
HF_MODEL_DIR=/path/to/gemma-4-E4B-it \
MEGATRON_CKPT=${WORKSPACE}/models/gemma4-e4b-megatron \
GEMMA4_LOG_ROOT=${WORKSPACE}/logs \
TRAIN_DATA_PATH=/path/to/data \
bash examples/models/gemma/gemma4/slurm_pretrain.sh
```

The script derives paths automatically:
- `${MEGATRON_CKPT}-text` - text conversion, used for training
- `${MEGATRON_CKPT}-vl` - VL/audio conversion, used for parity checks

Skip flags:
- `SKIP_CONVERT=1`
- `SKIP_TEXT_CONVERT=1`
- `SKIP_VL_CONVERT=1`
- `SKIP_PARITY=1`

## Evaluation

Use the parity checks above as the primary conversion sanity tests. The text
mode verifies the pure LLM path, while the `vl` and `audio` modes verify that
the multimodal wrapper preserves the Hugging Face behavior.

For generation sanity checks, run [inference.sh](inference.sh). For production
serving, export the checkpoint to Hugging Face format and run it with a serving
runtime that supports the Gemma 4 chat template and multimodal preprocessing.

## Running Unit Tests

```bash
PYTHONPATH=$PWD/src:${MEGATRON_LM_ROOT}:${PYTHONPATH:-} uv run --no-sync python -m pytest \
    tests/unit_tests/models/gemma/test_gemma4_bridge.py \
    tests/unit_tests/models/gemma/test_gemma4_provider.py \
    tests/unit_tests/models/gemma_vl/test_gemma4_vl_provider.py \
    tests/unit_tests/models/gemma_vl/test_gemma4_vl_bridge.py \
    tests/unit_tests/models/gemma_vl/test_gemma4_vl_modeling.py \
    tests/unit_tests/recipes/test_gemma4_recipe.py \
    -v
```

Multi-GPU unit tests (TP=2, requires 2 GPUs):

```bash
NVIDIA_VISIBLE_DEVICES=0,1 uv run --no-sync python -m torch.distributed.run --nproc_per_node=2 \
    -m pytest tests/unit_tests/models/gemma_vl -v -k "TensorParallel"
```

## Architecture Notes

### Clean Megatron-Core Compatibility

Gemma 4 keeps model-specific behavior in Bridge:

- `Gemma4DenseProvider` builds a standard `GPTModel`, then installs Gemma 4
  dual RoPE, shared-KV wiring, PLE modules, and checkpoint load aliases.
- `modeling_gemma4.py` patches only the created Gemma 4 decoder instance to
  thread `per_layer_inputs` through clean Megatron-Core's generic
  `extra_block_kwargs` path.
- No Gemma 4 specific Megatron-Core CLI arguments or `TransformerConfig` fields
  are required for the dense text path.

### Text and VL Separation

The text-only implementation lives in `megatron.bridge.models.gemma`:

- `modeling_gemma4.py` contains Dense/MoE layers, attention, dual RoPE, PLE,
  shared-KV wiring, and output softcapping.
- `gemma4_provider.py` contains `Gemma4DenseProvider` and
  `Gemma4ModelProvider`.
- `gemma4_bridge.py` registers `Gemma4ForCausalLM` and defines text checkpoint
  mappings.

The VL implementation lives in `megatron.bridge.models.gemma_vl`:

- `modeling_gemma4_vl.py` contains only `Gemma4VLModel` and VL/audio forward
  helpers.
- `gemma4_vl_provider.py` contains `Gemma4DenseVLProvider` and
  `Gemma4VLModelProvider`.
- `gemma4_vl_bridge.py` registers `Gemma4ForConditionalGeneration` and adds
  vision/audio mappings on top of the text mappings.

`gemma_vl` imports from `gemma`; `gemma` does not import from `gemma_vl`.

### Dense E4B Language Model

| Component | Detail |
|-----------|--------|
| 4-norm structure | `input_layernorm` → attention → `post_self_attn_layernorm` → MLP → `post_mlp_layernorm` |
| GQA + sliding/global mix | Sliding layers use 256-dim heads; global layers use 512-dim heads |
| Dual RoPE | Sliding θ=10 000; global θ=1 000 000 with partial factor 0.25 |
| Shared KV | Last 18 layers reuse KV from the last non-shared layer of the same attention type |
| Per-Layer Embeddings | PLE modules are attached after `GPTModel` construction and threaded through `forward()` |
| Logit softcapping | `final_logit_softcapping=30.0` is applied by the Gemma4 output layer |

### VL and Audio Path

`Gemma4VLModel` wraps the language model with HF vision/audio modules:

- Vision tower and projector weights are mapped under `vision_tower.*` and
  `embed_vision.*`.
- Audio tower and projector weights are mapped under `audio_tower.*` and
  `embed_audio.*`.
- Multimodal token positions are replaced with pad token IDs before PLE lookup,
  matching Hugging Face behavior.
