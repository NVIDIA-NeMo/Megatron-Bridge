# MiniMax-M2

[MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2) from **MiniMax AI**
is a large sparse Mixture-of-Experts model with **456B total parameters
(45.9B active)**, **256 experts**, and **FP8 block-wise quantization**.

The same Bridge supports the closely related
[MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5) and
[MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7) checkpoints —
they share the `MiniMaxM2ForCausalLM` architecture and route through the same
[`MiniMaxM2Bridge`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/minimax_m2/minimax_m2_bridge.py).

## Hardware requirements

MiniMax-M2 requires **at least 2 nodes (16 GPUs)** for inference and
checkpoint conversion. It cannot fit on a single 8-GPU node:

- `TEGroupedMLP` workspace is proportional to `num_experts / EP`. With `EP=8`
  on a single node the workspace alone OOMs.
- Tensor parallelism does **not** reduce expert memory — use **expert
  parallelism** instead.
- Minimum recommended layout: `TP=1, EP=16, PP=1` (2 nodes × 8 GPUs).

## FP8 block-wise quantization

The HF checkpoint ships in FP8 with 128×128 block-wise scaling. The Bridge
dequantizes weights to BF16 on import using the block scale tensor stored
next to each FP8 weight — there is no separate "FP8 mode" you need to enable
on the Bridge side. On export the Bridge writes BF16 by default; passing
through the original FP8 quantization is not currently supported.

## Conversion with 🤗 Hugging Face

### Load HF → Megatron

```python
from megatron.bridge import AutoBridge

# Works for M2 / M2.5 / M2.7 — same MiniMaxM2ForCausalLM architecture
bridge = AutoBridge.from_hf_pretrained("MiniMaxAI/MiniMax-M2")
provider = bridge.to_megatron_provider()

# Required parallelism shape on 2 × H100 nodes
provider.tensor_model_parallel_size = 1
provider.expert_model_parallel_size = 16
provider.pipeline_model_parallel_size = 1
provider.sequence_parallel = True

provider.finalize()
model = provider.provide_distributed_model(wrap_with_ddp=False)
```

For larger clusters, `(TP, PP, EP)` sweeps verified end-to-end against
the round-trip test in `examples/models/minimax_m2/slurm_conversion.sh`
include `2,1,8`, `1,2,8`, and `2,2,4`.

### Export Megatron → HF

```python
bridge.export_ckpt(
    megatron_path="/results/minimax_m2/checkpoints/iter_0010000",
    hf_path="./minimax-m2-hf-export",
)
```

## Examples

The end-to-end Slurm scripts live in
[`examples/models/minimax_m2/`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/minimax_m2):

- [`slurm_conversion.sh`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/minimax_m2/slurm_conversion.sh)
  — sweeps multiple `TP/PP/EP` configs and verifies HF↔Megatron round-trip
- [`slurm_inference.sh`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/minimax_m2/slurm_inference.sh)
  — text generation on the full FP8 checkpoint with `TP=1, EP=16`

## Recipes

MiniMax-M2 currently ships **conversion + inference** through the Bridge.
Pretrain / SFT / PEFT recipes are not yet included in
`src/megatron/bridge/recipes/`. If you need to train this model family,
build your config on top of the provider returned by
`AutoBridge.from_hf_pretrained(...)` and look at the multi-node Slurm
patterns under `examples/models/minimax_m2/` as a starting point.

For more context on parallelism shape, see
[`../../parallelisms.md`](../../parallelisms.md) and
[`../../training/moe-optimization.md`](../../training/moe-optimization.md).
