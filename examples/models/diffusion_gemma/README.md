# DiffusionGemma validation examples

These scripts validate the DiffusionGemma bridge and training step. They use
public checkpoints or randomly initialized toy models with synthetic inputs.

| Script | Purpose | Hardware |
| --- | --- | --- |
| `toy_parity.py` | Tiny HF ↔ Megatron conversion and encoder parity. Shared helpers for the other toy checks. | 1 GPU |
| `toy_training_parity.py` | Tiny loss, logits, gradient, TP/EP, long-prompt, and overfit checks. Used by the L0 functional test. | 1–2 GPUs |
| `training_loop_smoke.py` | Real `pretrain()` loop, distributed optimizer, validation, checkpoint save, and fresh-process resume. Used by the L1 functional test. | 2 GPUs (tiny) or 8 GPUs (26B) |
| `real_encoder_parity.py` | 26B encoder parity helpers and diagnostics. | 1 H200 |
| `real_fp32_parity.py` | 26B FP32 encoder and decoder parity against Hugging Face. | 1 H200 |
| `real_training_step.py` | One-GPU 26B BF16 corruption, self-conditioning, backward, and optimizer smoke. | 1 H200 |
| `recipe_long_context_smoke.py` | 26B recipe on a synthetic long-context multimodal JSONL split through `DiffusionGemmaJSONLDataset`; reports memory. | 8 H200 |

Run the tiny functional tests with:

```bash
bash tests/functional_tests/launch_scripts/h100/active/L0_Launch_models_diffusion_gemma.sh
bash tests/functional_tests/launch_scripts/h100/active/L1_Launch_training_diffusion_gemma_resume.sh
```

Run a 26B check with a local checkpoint path:

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 \
  examples/models/diffusion_gemma/recipe_long_context_smoke.py \
  --model-path /path/to/diffusiongemma-26B-A4B-it \
  --work-dir /tmp/diffusiongemma-recipe-smoke \
  --report /tmp/diffusiongemma-recipe-smoke/report.json
```
