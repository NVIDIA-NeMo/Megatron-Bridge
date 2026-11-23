# Qwen3.8-Flash-Next training

**EARLY, UNOPTIMIZED recipes.** The registered recipes use the full checkpoint
through the standard public training launcher. Text pretraining freezes
the EP-sharded PLE n-gram tables and uses Muon. Image SFT trains the language
model, vision encoder and projector with PLE n-gram lookup tables frozen and
EP-sharded. Hard-selection QSA indexer weights are also frozen.

Use the released **`nvcr.io/nvidia/nemo:26.10`** container and
**Qwen/Qwen3.8-Flash-Next** at revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540` for config, tokenizer, processor and
weights. The model requires Transformers 5.16.1 or later with Qwen4-Exp support.
Provide scheduler settings, container mounts and cache mappings through the
launchers' external setup. The portable `work/...` paths below identify inputs
and outputs; create output parents before running.

Verification is split by mode:

- [Text-only card](../../../model_verification_cards/qwen3.8-flash-next-text-only/card.yaml): fixed-4K frozen-PLE Muon pretraining.
- [VLM card](../../../model_verification_cards/qwen3.8-flash-next/card.yaml): full language/vision conversion and image SFT.

## Convert the checkpoint

The vision-language and standalone text models have different native checkpoint
namespaces. Import them separately; do not pass the VL native checkpoint to the
text-only recipe. The conversion topology need not equal the training topology.

```bash
./scripts/conversion/convert.sh import --executor slurm --device gpu \
  --nodes 2 --gpus-per-node 4 --tp 4 --pp 1 --ep 8 --etp 1 \
  --hf-model Qwen/Qwen3.8-Flash-Next \
  --hf-revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --torch-dtype bfloat16 \
  --megatron-path work/checkpoints/qwen3.8-flash-next-vl

./scripts/conversion/convert.sh import --executor slurm --device gpu \
  --nodes 2 --gpus-per-node 4 --tp 4 --pp 1 --ep 8 --etp 1 \
  --hf-model Qwen/Qwen3.8-Flash-Next \
  --hf-revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --torch-dtype bfloat16 --text-only --detach \
  --megatron-path work/checkpoints/qwen3.8-flash-next-text
```

The text-only import submits asynchronously. Wait for that job to finish
successfully and verify its persisted output before starting pretraining;
returning from its launcher confirms submission only. The VL import above
waits for completion.

Export uses a fresh process to reload the persisted native VL checkpoint:

```bash
./scripts/conversion/convert.sh export --executor slurm --device gpu \
  --nodes 2 --gpus-per-node 4 --tp 4 --pp 1 --ep 8 --etp 1 \
  --hf-model Qwen/Qwen3.8-Flash-Next \
  --hf-revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --torch-dtype bfloat16 --export-weight-dtype bfloat16 --distributed-save \
  --megatron-path work/checkpoints/qwen3.8-flash-next-vl \
  --hf-path work/exports/exported-hf
```

The full-VL roundtrip preserves all 1,627 supported tensors exactly by key,
shape, dtype and bytes, including 333 vision tensors and three integer PLE
buffers. Config, tokenizer and image-processor reload checks also pass. The
original export wrapper failed in a supplementary checker while formatting an
integer config key, after export and exact weight comparison had succeeded;
the repaired checker passed separately on the same preserved output. No
conversion rerun or changed output was needed. MTP is unsupported and excluded.
BF16 export restores
the original storage dtype for router weights promoted to FP32 in the native
model; integer PLE buffers retain their integer dtype. Weight equality does not
qualify inference, optimizer-state resume or a training trajectory.

## Text-only pretraining with frozen PLE

The library recipe
[`qwen38_flash_next_text_frozen_ngram_pretrain_32gpu_gb200_bf16_muon_config`](../../../../src/megatron/bridge/recipes/qwen/gb200/qwen38_flash_next.py)
uses `text_only=True`, TP2/SP, EP32/ETP1, PP1/CP1, sequence 4096,
MBS1/GBS32 on 32 GB200 GPUs. PLE projections, norms, convolution and other
language parameters remain trainable. It starts a fresh optimizer from the
imported text-only checkpoint, including native TP-to-EP table resharding.

```bash
./scripts/training/train.sh --nodes 8 --gpus-per-node 4 \
  --recipe qwen38_flash_next_text_frozen_ngram_pretrain_32gpu_gb200_bf16_muon_config \
  --mode pretrain --max_steps 30 \
  --pretrained_checkpoint work/checkpoints/qwen3.8-flash-next-text \
  'dataset.data_path=[work/data/wikitext2_qwen38_train_text_document]' \
  logger.save_config_filepath=work/model-verification/qwen3.8-flash-next/pretrain/resolved-config.yaml \
  logger.tensorboard_dir=work/model-verification/qwen3.8-flash-next/pretrain/tensorboard
```

The verification input is one million indexed tokens from 9,460 training
texts in Salesforce/wikitext, wikitext-2-raw-v1 revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, tokenized with this model's pinned
tokenizer. Preparation keeps nonblank rows in source order, appends EOS token
248046 per document, and trims the final document to exactly one million
tokens. Supply the matching indexed prefix at the logical path above. Dataset
index files must be visible to every rank: use shared data/cache storage, or
prepare identical indexes on every node when staging the data locally. The
30 updates consume 3,932,160 token slots, reusing this bounded input; this is not
a held-out evaluation.

The 30-update schedule has five-step warmup, constant LR 1e-4, zero weight decay
and clipping 1. Muon uses layer-wise distribution, duplicated full-matrix TP
orthogonalization, momentum 0.95, five Newton-Schulz steps, `unit_rms_norm`
scaling with factor 0.2, unsplit QKV and Adam fallback. This is a short functional
qualification schedule, not a recommendation for full training convergence.

## Image SFT

The library recipe
[`qwen38_flash_next_sft_64gpu_gb200_bf16_config`](../../../../src/megatron/bridge/recipes/qwen_vl/qwen38_flash_next.py)
uses the Qwen VL forward step and native Energon loader. Current defaults use
fixed 4096-token batches, frozen EP-sharded PLE lookup tables and synchronized
manual garbage collection every 100 steps. Prepare DataComp image-caption
conversations with the [maintained DataComp tutorial](../../../../tutorials/data/datacomp/README.md)
and supply the ChatML Energon directory at `work/data/datacomp/energon`.
These defaults are awaiting a new full-checkpoint verification; the recorded
2K results below retain their original recipe and trainable-table settings.

The historical image-SFT run used the first 8,192 complete supervised
conversations from the pinned ECD portion of
**nvidia/Nemotron-Image-Training-v3**, with images from
**ChartFoundation/ECD-10k-Images**:

The measured subset was prepared with the
[pinned subset preparer](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c7ee2cf6eaf977b853ef0b65e127a60a11fde81f/scripts/data/prepare_nemotron_image_v3_subset.py).
It remains available at that verification revision. Reuse an equivalent prepared
Energon dataset at `work/data/nemotron-image-v3-ecd`; general Nemotron conversion
is provided by `tutorials/data/energon/prepare_nemotron_image_v3.py`.

The `work/cache/qwen3.8-flash-next` alias must contain the complete local HF
checkpoint and processor assets from the public model and immutable revision
above. The checkpoint warm-start option takes an existing directory; it does
not resolve a Hub model ID. The historical preparer uses those pinned processor assets and
writes 32 shards plus a selection/asset manifest. It skips overlength
conversations whole, without truncation. There is no validation split. An
existing subset with matching preprocessing and manifest can be reused.

```bash
./scripts/training/train.sh --nodes 16 --gpus-per-node 4 \
  --recipe qwen38_flash_next_sft_64gpu_gb200_bf16_config --mode sft --step-func qwen3_vl_step \
  --max_steps 100 --pretrained_checkpoint work/cache/qwen3.8-flash-next \
  dataset.path=work/data/datacomp/energon \
  logger.save_config_filepath=work/model-verification/qwen3.8-flash-next/sft/resolved-config.yaml \
  logger.tensorboard_dir=work/model-verification/qwen3.8-flash-next/sft/tensorboard
```

Use the recipe-owned Energon configuration. Defaults are 64 GB200 GPUs, TP4/SP, EP64/ETP1, PP1/CP1,
MBS1/GBS16, unpacked sequences padded to 4096, image limits 65,536–262,144 pixels,
and distributed Adam with constant LR 1e-6, zero weight decay and clipping 1.

Both recipes use BF16 with FP32 gradient reduction, natural routing, HybridEP,
grouped GEMM and MoE fusions. Forced balancing, recomputation, CUDA graphs,
shared-expert overlap, evaluation and checkpoint saving are disabled. Place
the expert group within a supported NVLink domain and supply eight OpenMP
threads per worker through the external runtime setup. These public recipes
are still **EARLY, UNOPTIMIZED**.

## Verification results

Released NeMo 26.10 verification completed on 2026-11-22.

| Workload | GPUs / sequence / updates | Initial → final loss | Final-ten mean step | Final-ten mean TFLOP/s/GPU | Token slots/s/GPU |
| --- | --- | --- | --- | --- | --- |
| Frozen-PLE text / Muon | 32 / 4096 / 30 | 1.927702 → 1.366627 | 3.637 s | 47.46 | 1126.30 |
| Full image SFT / Adam | 64 / 2048 / 100 | 0.351589 → 0.167965 | 5.458 s | 5.89 | 93.81 |

All ranks completed finite, unskipped updates. Frozen text PLE table hashes
were unchanged; selected language and optimizer samples changed. SFT observed
image inputs, encoder execution and selected language/vision updates.

The
[verification card](../../../model_verification_cards/qwen3.8-flash-next/card.yaml)
records the exact measured source, commands, losses and final-ten metrics.
Text uses updates 21–30; SFT uses 91–100. Native timing includes lightweight
verification callbacks; there is no profiler capture, and startup weight hashing
is outside the final-ten window. TFLOP/s is the arithmetic mean of the
native per-update model-work estimate. Token throughput counts sequence slots,
including SFT padding, rather than supervised tokens. The 4K QSA FLOP estimate
uses a dense surrogate and does not accurately count indexer score work;
forward-only indexer projections are counted. Norms, lookups, PLE depthwise
convolution, communication and optimizer work are outside the matrix-work
estimate. At 2K, QSA takes the
dense shortcut and does not run the indexer. These are functional observations,
not hardware-counter utilization or optimized benchmark results.

The historical 5,349-step image-SFT run remains separate valid evidence: loss
0.31087→0.01342 on the same bounded 8,192-sample subset. Its
[loss curve](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6354#issuecomment-6160448312)
was captured with the earlier runtime and configuration; it did not capture
trustworthy TFLOP/s. “Unoptimized” qualifies recipe performance, not that loss
curve. Repeated-subset loss does not establish held-out quality.

Current launcher usage sets `--step-func qwen3_vl_step` explicitly for SFT.
Cluster staging belongs to the caller's allocation setup; this model PR does
not change the shared conversion or training scripts. The card retains the
exact commands and source commits of the completed verification runs.

FLOPs accounting now lives in the model's `estimate_flops` functions, shared
between LM and VLM and called by training accounting. Standard blocks now use
the pinned Core estimators, including its causal-attention convention. QSA
still uses a dense surrogate and GDN recurrence remains approximate. Historical
TFLOP/s figures retain their original estimator and source revisions; they have
not been recomputed with these new functions.
