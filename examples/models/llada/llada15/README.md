# LLaDA1.5 — Megatron Bridge Integration

LLaDA1.5 is `GSAI-ML/LLaDA-1.5` — a dense 8B masked-diffusion (MDM)
language model with a LLaMA-style block, OLMo-style parameter naming,
and full RoPE. The Megatron Bridge integration converts the HuggingFace
checkpoint, exposes the model as a Megatron-Core `GPTModel` (with NVIDIA
Transformer Engine attention), and provides a block-diffusion generation
loop that matches the reference ML-GSAI sampler token-for-token.

## At a glance

| | Value |
|---|---|
| Architecture | Dense LLaMA-style block, 32 layers, hidden 4096, MLP 12288, full MHA (32 Q heads = 32 KV heads) |
| Activation / Norm | SwiGLU + SiLU / RMSNorm (`eps=1e-5`) |
| Positional | Full RoPE, `rotary_base = 500000` |
| Embeddings | `vocab = embedding = 126464`; `weight_tying = false` (separate `lm_head`) |
| Tokens | `mask_token_id = 126336`, `eos = 126081`, `pad = 126081` |
| HF arch class | `LLaDAModelLM` (trust_remote_code; not in stock `transformers`) |
| Attention pattern | Fully bidirectional at training and inference (reference uses zero attention bias) |
| Block diffusion | Sampling-time only — the block structure governs *which positions get unmasked per step*, not the attention mask |

## Bridge files

```
src/megatron/bridge/diffusion/
  models/llada15/
    __init__.py
    llada15_provider.py      LLaDA15ModelProvider (RoPE on, bidirectional attention via custom core)
    llada15_attention.py     TE attention shim: forces AttnMaskType.no_mask
    inference_llada15.py     Block-diffusion denoising loop (no attention-mask manipulation)
  conversion/llada15/
    __init__.py
    llada15_bridge.py        Registered under source="LLaDAModelLM"
```

One-line addition to `src/megatron/bridge/models/conversion/auto_bridge.py`:
`"LLaDAModelLM"` is now in `SUPPORTED_HF_ARCHITECTURES` so `AutoBridge`
resolves the trust_remote_code class.

## Workflow

```
HF safetensors  →  AutoBridge.to_megatron_model()  →  save_megatron_model()
                                                              ↓
                                                    iter_0000000/*.distcp
                                                              ↓
                                                    load_megatron_model()
                                                              ↓
                                                    generate_block_diffusion()
                                                    (TE attention; MDM denoising)
```

The model uses NVIDIA Transformer Engine (TE) fused attention via
`LLaDA15TEDotProductAttention`, a thin subclass of
`megatron.core.extensions.transformer_engine.TEDotProductAttention`.
The only model-specific change is forcing `AttnMaskType.no_mask` so the
default causal mask of MCore's GPT layer spec is overridden — matching
LLaDA1.5's reference `get_bidirectional_attention_bias` (a zero tensor).

## Scripts

All scripts live under `examples/models/llada/llada15/`. They use the container's
bundled Bridge install (`/opt/Megatron-Bridge/src`) because that path has a
working `megatron.bridge.training` against the installed Megatron-Core.

| Script | Purpose | GPU? |
|---|---|---|
| `convert_llada15_hf_to_megatron.py` | One-shot HF → Megatron checkpoint conversion + save to disk | 1 GPU |
| `run_llada15_chat.py` | Loads the saved Megatron checkpoint, applies the Llama3-style chat template, runs block-diffusion generation | 1 GPU |

Automated tests for the bridge live under `tests/` (no HF checkpoint or GPU
required for the unit tests):

| Test | Purpose |
|---|---|
| `tests/unit_tests/diffusion/model/llada15/test_inference.py` | Block-diffusion generation loop (mask scheduling, batched EOS early-stop) |
| `tests/functional_tests/test_groups/models/llada15/test_llada15_generate.py` | End-to-end generation on a converted checkpoint |

Environment expected in the rest of this doc:

```bash
export HF_PATH=/path/to/huggingface/hub/models--GSAI-ML--LLaDA-1.5/snapshots/<commit-hash>
export CKPT_PATH=/path/to/llada15_megatron_ckpt
export PYTHONPATH=/opt/Megatron-Bridge/src
```

## Quick start

### 1. Verify the bridge (testing)

The Bridge already supports LLaDA1.5. Run the unit tests (no HF checkpoint
or GPU required) to confirm the generation loop on your machine:

```bash
uv run python -m pytest tests/unit_tests/diffusion/model/llada15/test_inference.py
```

For end-to-end generation on a converted checkpoint, run the functional
test:

```bash
uv run python -m pytest tests/functional_tests/test_groups/models/llada15/test_llada15_generate.py
```

To re-validate HF ↔ Megatron numerical parity (cosine similarity, argmax
agreement, generation token-for-token match), compare the bridge output
against the reference [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA)
`generate.py` sampler on the same prompt, seed, and sampling settings.

### 2. Convert the checkpoint (one-time)

```bash
python3 examples/models/llada/llada15/convert_llada15_hf_to_megatron.py \
    --hf-path  "$HF_PATH" \
    --out-path "$CKPT_PATH"
```

This builds the Megatron `GPTModel` via `LLaDA15Bridge`, loads HF weights,
and saves a torch-dist sharded checkpoint plus tokenizer metadata. Output
layout:

```
$CKPT_PATH/
  iter_0000000/
    __0_0.distcp         # sharded weights (~15 GB bf16)
    common.pt
    metadata.json
    run_config.yaml      # serialized GPTModelProvider config
    train_state.pt
    tokenizer/           # embedded HF tokenizer + chat template
  latest_checkpointed_iteration.txt
  latest_train_state.pt
```

### 3. Chat / generate from the saved checkpoint

```bash
python3 examples/models/llada/llada15/run_llada15_chat.py \
    --ckpt-path      "$CKPT_PATH" \
    --tokenizer-path "$HF_PATH" \
    --gen-length 128 --block-length 32 --steps 64
```

Runs `generate_block_diffusion()` over a built-in prompt set. To supply
custom prompts:

```bash
python3 examples/models/llada/llada15/run_llada15_chat.py \
    --ckpt-path "$CKPT_PATH" --tokenizer-path "$HF_PATH" \
    --prompts \
      "Explain BFS in one sentence." \
      "Write a Python function that returns the n-th Fibonacci number."
```

The chat script applies the LLaDA1.5 Llama3-style chat template
(`<|start_header_id|>user<|end_header_id|>...<|eot_id|>`) before
generation and trims the reply at EOS.

## Sampling controls

| Flag | Default | Notes |
|---|---|---|
| `--gen-length` | 64 (chat) / 32 (parity) | Tokens to generate after the prompt. Higher → longer replies, more denoising iterations. |
| `--block-length` | 32 | Tokens unmasked per outer denoising block. The model still attends bidirectionally over the whole sequence; this only governs which positions are eligible to be transferred per step. |
| `--steps` | 32–64 | Total denoising steps. Internally divided across blocks. `steps / num_blocks` becomes steps per block. |
| `--temperature` | 0 | Greedy. `>0` activates gumbel-noise sampling (matches official sampler). |
| `--top-k`, `--top-p` | unset | Active only when `temperature > 0`. |

Greedy decoding with small `--gen-length` will frequently terminate with
EOS padding (visible in the parity test as `[..., 126081, 126081, ...]`).
For more varied output, use `--temperature 0.3 --top-p 0.9` and a larger
`--gen-length`.

## Current verification status

| Check | Verdict |
|---|---|
| Level 1 — exact state-dict round-trip | PASS (291/291 tensors match, atol=1e-6) |
| Level 2 — forward parity (unmasked) | PASS (cos_sim 0.99979, all argmax + top-5 match) |
| Level 2 — forward parity (masked, MDM path) | PASS (cos_sim 0.99994, all argmax + top-5 match) |
| Generation parity vs official ML-GSAI sampler | PASS (32/32 token agreement, decoded ` Paris.`) |
| End-to-end chat on a saved checkpoint | PASS (coherent on-topic replies) |
| Multi-GPU (TP/PP/EP > 1) | Untested |
| FP8 training path | Untested |
| MDM-style training step + backward parity | Not implemented |

## Design notes

**Why `position_embedding_type = "rope"` (not `"none"`).**
LLaDA1.5 uses full RoPE — `rotate_half` over the entire `head_dim=128`,
no slicing. Megatron's built-in rotary path is correct out of the box; the
custom attention does not need to rotate Q/K itself. (Contrast with
LLaDA2, which uses partial RoPE and disables Megatron's RoPE.)

**Why `AttnMaskType.no_mask` instead of `arbitrary` with a block mask.**
The reference `modeling_llada.py:get_bidirectional_attention_bias`
returns a zero tensor — i.e. fully bidirectional attention at training
*and* at inference. The block structure exists only in the sampling
schedule (which positions get unmasked per step). Imposing a
block-diagonal attention mask would diverge from the reference.

**Why `QKVMapping` (separate q/k/v), not `ConcatenatedQKVMapping`.**
The safetensors layout has three separate `q_proj`, `k_proj`, `v_proj`
tensors per layer (OLMo-style), not a single fused `query_key_value`.

**Why `GatedMLPMapping(gate=ff_proj, up=up_proj)`.**
`LLaDALlamaBlock.forward` computes `act(ff_proj(x)) * up_proj(x)`, so
`ff_proj` is the gate (SiLU is applied to it) and `up_proj` is the linear
up input. This is the inverse of the naming convention in most
HuggingFace LLaMA derivatives — verify against the reference code, not
intuition.

**Transformer Engine attention (`LLaDA15TEDotProductAttention`).**
Subclasses `megatron.core.extensions.transformer_engine.TEDotProductAttention`,
which in turn wraps
`transformer_engine.pytorch.attention.dot_product_attention.DotProductAttention`
— so attention dispatches to TE's fused kernels (FlashAttention 2/3 or
cuDNN). The shim adds nothing except the `AttnMaskType.no_mask` override
and two unused state hooks (`set_block_mask`, `reset_inference_state`)
reserved for users who want to experiment with LLaDA2-style block-diagonal
attention.

**Transformers-5.x compatibility shims.**
The bundled `modeling_llada.py` was written for `transformers 4.46.3`.
On 5.x, three shims are needed when loading the HF reference (only for
parity testing — the bridge itself reads safetensors directly and does
not invoke the trust_remote_code class):

- `cls.all_tied_weights_keys = {}` — new attribute checked by
  `_finalize_model_loading`. `weight_tying=False` so it is a no-op.
- Wrap `tie_weights()` to swallow `missing_keys=` / `recompute_mapping=`
  kwargs.
- Set `model.config.use_cache = False` (no longer auto-populated).

These shims are only needed when loading the HF reference for parity
testing against the [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA)
implementation; the bridge itself does not require them.

## Limitations

- **TP / PP / EP / CP > 1 untested.** Single-GPU TP=PP=1 works; multi-GPU
  configurations may surface scatter/gather bugs in the QKV or MLP
  mappings that don't appear single-GPU.
- **No training step / loss function yet.** LLaDA training uses a
  masked-diffusion loss (not next-token prediction). The training step
  needs to be implemented and validated against the LLaDA paper recipe
  before any real training run.
- **No recipe under `src/megatron/bridge/recipes/`.** Pretrain / SFT /
  PEFT presets for LLaDA1.5 are a follow-up.
- **FP8 path untested.** TE supports FP8 but it was not exercised here.

## Reference

- Model card: <https://huggingface.co/GSAI-ML/LLaDA-1.5>
- Reference implementation: <https://github.com/ML-GSAI/LLaDA>
- LLaDA paper: <https://arxiv.org/abs/2502.09992>
