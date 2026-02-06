# NeMo2 → Megatron Bridge Conversion

This tutorial shows how to turn a NeMo2 checkpoint into a Megatron Bridge
checkpoint by running two lightweight helper scripts in sequence.

## Prerequisites

- `00_save_nemo2_weights_torch.py` must run inside the NeMo training
  environment that can import `nemo.lightning` and access the original
  checkpoint directory.
- `01_torch_weights_to_megatron.py` only needs CPU PyTorch, Hugging Face
  Transformers, and the Megatron Bridge Python package.

## Step 1: dump NeMo2 weights

```bash
python tutorials/convert_nemo2_checkpoint_to_bridge/00_save_nemo2_weights_torch.py \
  --nemo-checkpoint /path/to/nemo_ckpt \
  --output /tmp/model_weights.pt
```

The script loads the checkpoint via `ModelConnector`, optionally moves tensors
to CPU, and stores a flat `state_dict` with `torch.save`. Pass
`--keep-device` to keep the original device placements.

## Step 2: build a Megatron checkpoint

```bash
python tutorials/convert_nemo2_checkpoint_to_bridge/01_torch_weights_to_megatron.py \
  --hf-model-id meta-llama/Llama-3.2-1B \
  --torch-weights /tmp/model_weights.pt \
  --output-dir /tmp/megatron_ckpt
```

This loads the torch file, infers dtype, instantiates `AutoBridge` from the HF
config, and writes a Megatron Bridge checkpoint to `--output-dir`. The script
reports any missing or unexpected keys before saving.

After these steps, the resulting directory is ready for downstream Megatron
Bridge tools (sharding, inference runners, etc.).

