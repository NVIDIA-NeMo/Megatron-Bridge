# Qwen3.5-VL MegatronMIMO Examples

This directory contains Qwen3.5-VL examples for **MegatronMIMO**, a
multimodal training path that lets the vision encoder and language model use
different parallelism layouts.

If you are new to this feature, start with one of the tutorials below. The
tutorials explain the training mode, show the commands to run, and state what
has been validated.


## Tutorials

| Goal | Tutorial | Status |
|---|---|---|
| Fine-tune Qwen3.5-VL 27B with non-colocated MegatronMIMO SFT | [`qwen35-vl-non-colocated-sft.md`](../../../tutorials/megatron_mimo/qwen35-vl-non-colocated-sft.md) | Available |
| Fine-tune with colocated MegatronMIMO training | Coming soon | Planned |


## What This Example Covers

The current Qwen3.5-VL example focuses on one validated workflow:

- Dense Qwen3.5-VL models.
- Two MegatronMIMO components: `language` and `images`.
- Hugging Face to MegatronMIMO checkpoint conversion.
- Full-parameter SFT on direct Hugging Face SFT data.
- Non-colocated training, where the language model and image encoder run on
  disjoint rank ranges.
- Optional Multi-Token Prediction (MTP) training with `--enable-mtp`; see
  [MTP training](#mtp-training).

The reported performance and loss-parity evidence currently applies to the
validated 27B non-colocated workflow in the tutorial. Re-measure performance and
check loss behavior if you adapt the scripts to a different model size, dataset,
or parallelism layout.

Select a built-in source with `DATASET_NAME`; `cord_v2`, `rdr`, and `medpix`
each bind their validated Hub path to the matching schema adapter. The Python
entry point also accepts a custom `--dataset-path` with an optional explicit
`--schema-adapter`. Custom sources default to training-only; pass
`--do-validation` when the custom source exposes a `validation` split.


## Files

These files are used by the tutorials:

| File | Purpose |
|---|---|
| `conversion.sh` | Converts dense HF Qwen3.5-VL checkpoints to MegatronMIMO format and exports back to HF for a round-trip check. |
| `finetune_qwen35_vl.py` | MegatronMIMO runner for direct Hugging Face SFT data. |
| `slurm_sft.sh` | Multi-node Slurm launcher for the validated 27B non-colocated SFT layout. |
| `verify_mtp_parity.py` | Checks that MegatronMIMO computes MTP like the standard Qwen3.5-VL path. |


## Relationship to Standard Qwen3.5-VL

MegatronMIMO changes the parallelism layout and launcher for multimodal
training. The standard non-MIMO Qwen3.5-VL examples remain available at
`examples/models/qwen/qwen35_vl/`.

Use the standard examples when you want the regular Megatron-Bridge Qwen3.5-VL
path. Use this directory when you want to try MegatronMIMO layouts for
Qwen3.5-VL.


## MTP Training

Qwen3.5-VL checkpoints ship one MTP layer. MegatronMIMO builds it on the last
language pipeline stage, as standard training does:

- `conversion.sh` imports and exports the `mtp.*` weights, so the HF round trip
  is lossless. Checkpoints converted before MTP support have no MTP weights;
  convert them again before training MTP.
- `finetune_qwen35_vl.py` keeps MTP off by default. Pass `--enable-mtp` (or set
  `ENABLE_MTP=1` for `slurm_sft.sh`) to train the MTP layer; the last language
  stage then also reads `input_ids`.
- The log reports `mtp_1 loss` next to `lm loss`.

MegatronMIMO excludes image-placeholder tokens from MTP conditioning, while the
standard Qwen3.5-VL path does not, so `mtp_1 loss` can differ slightly between
the two when image tokens contribute to the loss. The performance and
loss-parity results in the tutorial were measured with MTP off.

To check MTP against the standard path, `verify_mtp_parity.py` loads one
checkpoint into both and runs one forward and backward pass on the same
synthetic image-text batch. The standard step applies MegatronMIMO's
image-token mask, so the weights, language-model loss, MTP loss and every
gradient must match; it also reports the MTP loss without the mask. Both
models run with tensor and pipeline parallel size 1:

```bash
OUT=/tmp/qwen35_vl_mtp_parity
uv run python -m torch.distributed.run --nproc_per_node=1 \
    examples/megatron_mimo/qwen35_vl/verify_mtp_parity.py standard \
    --hf-model Qwen/Qwen3.5-0.8B --output-dir ${OUT}
uv run python -m torch.distributed.run --nproc_per_node=2 \
    examples/megatron_mimo/qwen35_vl/verify_mtp_parity.py mimo \
    --hf-model Qwen/Qwen3.5-0.8B --output-dir ${OUT}
uv run python examples/megatron_mimo/qwen35_vl/verify_mtp_parity.py compare --output-dir ${OUT}
```

`compare` prints `PASS` or lists each mismatch and exits with status 1. A NaN or
infinite loss or gradient also fails, even when both paths agree.


## More Coverage Coming

Future examples and tutorials will expand coverage for:

- Qwen3.5-VL MoE variants.
- Packed sequences.
- Energon datasets.
- Evaluation or matched validation/test artifacts.
- MIMO throughput/FLOPs logging.

See the tutorial for the exact command sequence and the validated 27B
non-colocated setup.
