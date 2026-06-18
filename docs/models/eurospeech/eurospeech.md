# EuroSpeech (Canary + EuroLLM SpeechLLM)

EuroSpeech is a SpeechLLM that maps **(audio + text instruction) → text**. It
combines NVIDIA's **Canary** audio encoder, a trainable **audio projector**, and
an **EuroLLM** (Llama-architecture) language backbone, trained with the Megatron
audio-LM step so it benefits from TP/PP/SP, the distributed optimizer, and BF16/FP8.

## Architecture

```
audio waveform → Canary encoder → audio projector ─┐
                                                    ▼
text instruction (with <extra_id_0> placeholders) → EuroLLM embeddings
                                                    │  (scatter audio at placeholder positions)
                                                    ▼
                                             EuroLLM decoder → token logits
```

The audio features are scattered into the text embedding sequence at
`audio_token_id` positions, so `scatter_embedding_sequence_parallel` is forced
**off** (audio embeddings are dense and inserted mid-sequence).

## Supported variants

| Backbone | HF path | Default recipes |
|----------|---------|-----------------|
| EuroLLM-1.7B | `utter-project/EuroLLM-1.7B-Instruct` | `eurospeech_1_7b_sft_config`, `eurospeech_1_7b_peft_config` |
| EuroLLM-9B | `utter-project/EuroLLM-9B-Instruct` | `eurospeech_9b_sft_config`, `eurospeech_9b_peft_config` |

## Conversion / assembly

EuroSpeech has **no combined HuggingFace checkpoint** and therefore **no
`AutoBridge` registration** (registering a bridge against `LlamaForCausalLM`
would collide with the existing `LlamaBridge`). Instead the model is assembled
from three sources:

- **EuroLLM** language weights — derived from EuroLLM's HF config via `AutoBridge`
  and loaded into the `language_model` sub-module with the stock Llama mappings.
- **Canary** encoder — loaded from a `.nemo` file (Path A; requires
  `nemo_toolkit[asr]` as an optional extra) or replaced by a dependency-free stub.
- **Audio projector** — trained from scratch.

Assemble a trainable Megatron checkpoint on a single GPU (loads EuroLLM into the
language sub-module, loads Canary, smoke-tests, and saves the combined model):

```bash
uv run python examples/models/eurospeech/bootstrap_and_smoke.py \
    --eurollm utter-project/EuroLLM-1.7B-Instruct \
    --canary-nemo-path /data/canary-1b.nemo \
    --audio-hidden-size 1024 \
    --output-dir /workspace/eurospeech-1.7b-init
```

Omit `--canary-nemo-path` to use the stub encoder (no NeMo dependency). The audio
placeholder defaults to `<extra_id_0>` (id 5), a reserved EuroLLM token — id 4 is
EOS and must not be used. Point a recipe's `checkpoint.pretrained_checkpoint` at
`--output-dir` so training starts from the assembled weights.

## Training

EuroSpeech uses the audio-LM training step — pass `--step_func audio_lm_step`.

### SFT (1.7B, single GPU)

```bash
uv run python -m torch.distributed.run --nproc_per_node=1 scripts/training/run_recipe.py \
    --recipe eurospeech_1_7b_sft_config \
    --step_func audio_lm_step \
    checkpoint.pretrained_checkpoint=/workspace/eurospeech-1.7b-init \
    model.canary_nemo_path=/data/canary-1b.nemo \
    model.audio_hidden_size=1024 \
    dataset.maker_name=<your_audio_dataset_maker> \
    train.train_iters=1000
```

### PEFT / LoRA (9B, multi-GPU)

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 scripts/training/run_recipe.py \
    --recipe eurospeech_9b_peft_config \
    --step_func audio_lm_step \
    checkpoint.pretrained_checkpoint=/workspace/eurospeech-9b-init \
    model.canary_nemo_path=/data/canary-1b.nemo \
    dataset.maker_name=<your_audio_dataset_maker> \
    train.train_iters=1000
```

### Staged freeze schedule

| Stage | Frozen | Trained | How |
|-------|--------|---------|-----|
| 1 — Alignment | Canary, EuroLLM | Projector only | `model.freeze_language_model=true` |
| 2 — SFT | Canary | Projector + EuroLLM | `*_sft_config` (default) |
| 2-alt — PEFT | Canary, most of EuroLLM | Projector + LoRA | `*_peft_config` |

## Known limitations

- No HF round-trip export (not a registered HF architecture); Megatron-format
  checkpoints only. The trainable checkpoint comes from the bootstrap step.
- No bundled audio dataset — supply your own dataset maker via CLI.
- The audio placeholder defaults to `<extra_id_0>` (id 5), a reserved EuroLLM
  token, so no vocab resize is needed. Never use id 4 (EOS).
- Long-form audio (>~40s) requires chunking — out of scope for v1.
