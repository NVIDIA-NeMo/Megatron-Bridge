# Megatron-Bridge Diffusion

Diffusion Foundation Models (DFM) integrated into Megatron-Bridge. This module provides
Megatron-based implementations of diffusion models including DiT, FLUX, and WAN.

## Directory Structure

```
diffusion/
├── models/           # Model implementations (architecture, layers, forward steps)
│   ├── common/       # Shared modules (attention, embeddings, normalization)
│   ├── dit/          # DiT model (with EDM pipeline)
│   ├── flux/         # FLUX model (MMDiT, flow matching)
│   └── wan/          # WAN model (video generation, flow matching, inference)
├── conversion/       # HF ↔ Megatron checkpoint conversion bridges
│   ├── flux/         # FLUX bridge and HF pretrained adapter
│   └── wan/          # WAN bridge and HF pretrained adapter
├── data/             # Data loading and task encoders
│   ├── common/       # Shared data modules (energon, diffusion samples, sequence packing)
│   ├── dit/          # DiT task encoder and mock data
│   ├── flux/         # FLUX task encoder and data modules
│   └── wan/          # WAN task encoder and data modules
├── recipes/          # Training recipe configurations
│   ├── dit/          # DiT pretraining recipe
│   ├── flux/         # FLUX pretraining recipe
│   └── wan/          # WAN pretraining recipe
├── common/           # Shared utilities (video saving, tokenizers, batch ops)
└── base/             # Base module placeholder
```

## Supported Models

- **DiT**: Diffusion Transformer with EDM (Elucidating Diffusion Models) pipeline
- **FLUX**: State-of-the-art text-to-image model using MMDiT-style transformer blocks
- **WAN**: Video generation model with 3D rotary embeddings and flow matching

## Examples

Training examples and configuration files are in `examples/diffusion/`.

