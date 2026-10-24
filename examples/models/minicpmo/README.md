# MiniCPM-o 4.5 understanding checkpoint conversion

Use a supported Megatron Bridge environment, including its CUDA/Transformer Engine
build dependencies. These commands use the shared conversion launcher; no
model-specific conversion script is needed.

```bash
./scripts/conversion/convert.sh import \
  --executor local --device gpu --nodes 1 --gpus-per-node 2 \
  --hf-model openbmb/MiniCPM-o-4_5 --trust-remote-code \
  --megatron-path ./checkpoints/minicpmo-4.5 \
  --torch-dtype bfloat16 --tp 2 --pp 1

./scripts/conversion/convert.sh export \
  --executor local --device gpu --nodes 1 --gpus-per-node 2 \
  --hf-model openbmb/MiniCPM-o-4_5 --trust-remote-code \
  --megatron-path ./checkpoints/minicpmo-4.5/iter_0000000 \
  --hf-path ./checkpoints/minicpmo-4.5-hf \
  --torch-dtype bfloat16 --tp 2 --pp 1
```

Only the understanding-stage weights are imported/exported. The resulting HF
config retains `MiniCPMO` identity and disables TTS. The original source is needed
for custom model code and tokenizer/processor files.

Model construction, input contracts, freezing, and limitations are documented in
[MiniCPM-o 4.5](../../../docs/models/minicpmo/minicpmo-4.5.md). This entry does not
include an SFT/PEFT recipe or incremental generation implementation.
