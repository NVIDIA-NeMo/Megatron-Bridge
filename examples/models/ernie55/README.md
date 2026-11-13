## Introduction

**ERNIE 5.5 Flash** is an efficiency-focused model in the ERNIE 5.5 family: a sparse Mixture-of-Experts (MoE) language model with **280B total parameters and 14B actived parameters per token**, pre-trained from scratch on **33.5T tokens** with a training context of up to **128K**; fine-tuning and inference support a **1M-token** context. Different domains, reasoning tasks and agentic applications call for different data, training objectives and behavioral policies; by releasing the weights at the end of pre-training, we let developers build on the model's general capabilities and choose for themselves where to take it next. Because further training and agentic use keep adding to compute and memory cost, we made efficiency a key design goal from the start, revisiting each component, from attention and residual connections to the MoE layers and the output head, and treating activated compute, KV cache and long-context cost as part of one overall design. The same long-term view shaped the data and the training system: pre-training data emphasizes capabilities that are hard to build in afterwards, such as repository-scale code and long-context understanding, and the training system was co-designed with the architecture so that its efficiency choices also train efficiently at scale.

**ERNIE-5.5-Flash-Base** is the checkpoint at the end of pre-training, before any mid- or post-training, and can serve as a common starting point for domain adaptation, instruction tuning and reinforcement-learning research. Under a unified evaluation pipeline it delivers strong results across knowledge, reasoning, math, code, multilingual and long-context benchmarks.

This directory provides a recipe and instructions for supervised fine-tuning (SFT) of an ERNIE 5.5 Hugging Face
checkpoint with Megatron Bridge, covering data preparation, training, and export to Hugging Face format. The
recipe runs on 16 nodes with four GB300 GPUs per node.

## Requirements

* **Megatron-Core**: Use Megatron-Core from the `dev` branch at commit `f78809f6d` or later. Prepend the
  Megatron-Core checkout and this repository's `src` directory to `PYTHONPATH`, in that order, to ensure that the
  checkout takes precedence over any installed `megatron-core` package.

  ```bash
  export PYTHONPATH=/path/to/Megatron-LM:/path/to/Megatron-Bridge/src:${PYTHONPATH}
  ```

* **A Hugging Face ERNIE 5.5 checkpoint**: a directory with `config.json`, the safetensors shards and the tokenizer
  files (`tokenizer.json`, `tokenizer_config.json`). The examples below call it `$ERNIE55_HF_PATH`:

  ```bash
  export ERNIE55_HF_PATH=/path/to/ernie55_hf
  ```

  The recipe builds the model from its `config.json` and starts from its weights.
* **GPUs**: the full model trains on 16 nodes of 4 GB300 GPUs (see [Hardware](#hardware)).

## Recipe

The recipe is in `megatron.bridge.recipes.ernie.ernie55` and trains with the ERNIE 5.5 forward step
(`--step_func ernie55_step`).

|Recipe|GPUs|Parallelism|Row length|Global batch|Use it for|
|:---|:---|:---|:---|:---|:---|
|`ernie55_sft_64gpu_gb300_bf16_config`|16 x 4|TP1 PP1 CP64 EP64|262144|2|Fine-tuning the full model from `$ERNIE55_HF_PATH`|

* The `sft` recipe needs `$ERNIE55_HF_PATH` (or the recipe's `hf_path` argument). Without either, it builds a
  placeholder config with no weights, and fine-tuning stops with an error when it starts (`slurm_train.sh` checks
  for the path up front). `run_recipe.py` runs it in Bridge's `finetune` loop, which it infers from the `_sft_` in
  its name.

## Fine-tune a Hugging Face checkpoint

### 1. Prepare the data

`dataset.path` is a JSONL file with one sample per line, in any of three formats (they may be mixed):

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "...", "reasoning_content": "..."}], "tools": [...]}
{"input": "What is SFT?", "output": "Supervised fine-tuning."}
{"input_ids": [7, 1023, 8], "labels": [-100, -100, 8]}
```

* **Chat rows** (`messages`, with an optional tool list) and **prompt/completion rows** (`input`/`output`) are text:
  they need a tokenizer, `dataset.hf_tokenizer=$ERNIE55_HF_PATH`. The dataset renders ERNIE 5.5's SFT chat template
  (`ernie55_chat_template.jinja`) and supervises the assistant turns only. The template uses the released
  tokenizer's special tags (`<|ernie:im_start|>`, `<|ernie:im_end|>`, `<ernie:think>`, ...); reasoning written inline
  in a message's content must use `<ernie:think>`/`</ernie:think>` too. Add `"loss": false` to a history assistant
  message to leave it out. Template variables go in `dataset.chat_template_kwargs`, for example
  `'dataset.chat_template_kwargs={enable_thinking: false}'`.
* **Pre-tokenized rows** carry `input_ids` and `labels` in the Hugging Face convention: equal to `input_ids` where
  supervised, `-100` elsewhere. Bridge shifts them; set `dataset.labels_are_shifted=true` if yours are already
  aligned with their inputs. They need no tokenizer.

Text rows are encoded on every rank when training starts. For a large corpus, encode once with
`convert_sft_jsonl.py`, which uses the same template, and train on its output:

```bash
python examples/models/ernie55/convert_sft_jsonl.py \
    --input /path/to/sft.jsonl --output /path/to/train.jsonl \
    --tokenizer $ERNIE55_HF_PATH
```

A directory `--input` converts every `*.jsonl` directly inside it into the `--output` directory. The options
`--prompt-column`, `--completion-column`, `--enable-thinking`, `--preserve-thinking`, `--loss` and `--chat-template`
match the dataset settings.

The recipe makes each sample its own row and splits samples longer than 262144 tokens. A sample with a token id at
or above the model's `vocab_size` raises a `ValueError` when training first reads it, not when the dataset is built.

### 2. Train

#### Multi-node: the full model on 16 nodes

[slurm_train.sh](slurm_train.sh) runs one task per GPU in a Pyxis/enroot container. By default it fine-tunes
`$ERNIE55_HF_PATH` with `ernie55_sft_64gpu_gb300_bf16_config`. The paths must be visible inside the container
(`CONTAINER_MOUNTS`), and the job needs a QoS that admits 16 nodes:

```bash
ERNIE55_HF_PATH=/path/to/ernie55_hf DATA_PATH=/path/to/train.jsonl \
CONTAINER_IMAGE=/path/to/image.sqsh MEGATRON_LM_PATH=/path/to/Megatron-LM \
MEGATRON_BRIDGE_PATH=/path/to/Megatron-Bridge WORKSPACE=/path/to/workspace CONTAINER_MOUNTS=/path:/path \
    sbatch --account=<account> --partition=<partition> --qos=<qos> \
    examples/models/ernie55/slurm_train.sh
```

* Text rows are tokenized with the checkpoint's tokenizer (`HF_TOKENIZER`, default `$ERNIE55_HF_PATH`).
* The script saves to and resumes from `$CHECKPOINT_DIR` (default `$WORKSPACE/checkpoints/<recipe without _config>`,
  e.g. `$WORKSPACE/checkpoints/ernie55_sft_64gpu_gb300_bf16`): the first submission starts from the Hugging Face
  weights, and a resubmission continues from the last saved step. Use a new `CHECKPOINT_DIR` for a fresh run.
* `TRAIN_ITERS` overrides the number of steps (default 2065, with 103 warmup steps). The cosine schedule still
  decays over 2065 steps; to change that too, also pass `scheduler.lr_decay_iters=<N>`.
* Pipeline parallelism (`model.pipeline_model_parallel_size=<N>`): `Ernie55ModelConfig` splits the layer pattern at
  ERNIE-layer boundaries and enables `variable_seq_lengths`, because the SFT rows have data-dependent lengths and the
  stages must exchange tensor shapes before each transfer (this rules out the `allgather` MoE token dispatcher; the
  default is `alltoall`); the hyper-connections also require `model.pipeline_dtype=torch.float32` (the stages
  exchange the fp32 residual streams).

Without SLURM, start the same command on every node with `torch.distributed.run`:

```bash
python -m torch.distributed.run --nnodes 16 --nproc_per_node 4 --node_rank <0..15> \
    --master_addr <first node> --master_port 29500 \
    scripts/training/run_recipe.py --recipe ernie55_sft_64gpu_gb300_bf16_config --step_func ernie55_step \
    dataset.path=/path/to/train.jsonl dataset.hf_tokenizer=$ERNIE55_HF_PATH \
    checkpoint.save=/path/to/checkpoints checkpoint.load=/path/to/checkpoints
```

The context-parallel split is Megatron-Core's contiguous partition, the DSv4 layout. Each D layer converts its
input to the zigzag partition and back. Every rank holds 4096 of a row's 262144 tokens.

### 3. Export to Hugging Face

Convert a saved step back with `scripts/conversion/run_conversion.py export`, passing the original checkpoint as
`--hf-model` (it supplies the config and tokenizer):

```bash
python -m torch.distributed.run --nproc_per_node 4 --nnodes 16 ... scripts/conversion/run_conversion.py export \
    --device gpu --hf-model $ERNIE55_HF_PATH --megatron-path /path/to/checkpoints/iter_0002065 \
    --hf-path /path/to/export --ep 64
```

The export holds the model's tensors with their values unchanged. The tensors kept in fp32 (router,
hyper-connections, sinks and similar) are written in F32 even when the source checkpoint stored them in bf16, and the
shared MTP body is written once, not repeated under the MTP layer. Its
`tokenizer_config.json` names `PreTrainedTokenizerFast`, so transformers 4 and 5 both load it.

## Training settings

The recipe uses the reference implementation's SFT settings:

* **Loss.** Per-row LM and MTP losses (each row averaged over its supervised tokens, then the rows averaged). MTP is
  scaled by 0.1. The sparse DSA indexer KL is weighted 0.01.
* **MoE.** No auxiliary loss; the z loss has coefficient 0 but is still logged. The router expert bias updates at
  rate 1e-4. The router runs in fp32. Dispatch is all-to-all with grouped GEMMs.
* **Kernels.** The cuDNN DSA backend and the reference's Triton kernels for SiTU-GLU, the multimax head and the
  context-parallel RoPE (`model.ernie_fused_ops=false` runs the eager code).
* **Recompute.** Full recompute of every layer.
* **Optimizer.** `dist_muon`: Muon with momentum 0.95, Nesterov, spectral scale, extra scale 0.2 and 5 quintic
  Newton-Schulz steps. Like the reference, Muon orthogonalizes the attention projections per head (and per output
  group) and every gated fc1 as its gate and up halves (`optimizer.muon_split_qkv`, on by default). AdamW (0.9,
  0.95, 1e-8) handles the embedding, the output layer, the router matrices, the compressors' APE and every 1-D
  tensor. Weight decay 0.1 on every matrix.
* **Schedule.** Cosine from 2e-5 to 2e-6, warming up from 0. Gradient clipping at 1.0; the multimax head is clipped
  by its own norm.

Logged losses: `lm loss` is the mean over micro-batches of each row's supervised-token mean (the reference's
`loss`); the MTP loss follows the same convention before the 0.1 scaling; the indexer loss is the sum over D layers
divided by the number of decoder layers (H and D) plus MTP depths; the z loss is divided by the number of MoE layers.

## Hardware

The recipes are validated on NVIDIA Blackwell (GB200 at SM100, GB300 at SM103). Every component also supports
Hopper (SM90), but SM90 is not validated.

|Component|Architectures|Function|
|:---|:---|:---|
|ERNIE Triton kernels (SiTU-GLU, multimax, CP RoPE)|SM90 and later.|Fuse the SiTU-GLU activation, the multimax output head and the RoPE of context-parallel shards.|
|FlashMLA sparse forward|SM90 and later.|Computes the forward of the D layers' sparse attention over the top-k selected positions.|
|cuDNN DSA backward|SM90 and later.|Computes the backward of the D layers' sparse attention.|
|cuDNN DSA indexer|SM90 and later.|Scores the keys and selects the top-k positions each query attends to in the D layers.|
|Transformer Engine, DeepEP/HybridEP|SM90 and later.|Transformer Engine provides the linear layers; DeepEP/HybridEP dispatches tokens to experts.|
|MoE dispatcher|SM90 and later.|Exchanges tokens between expert-parallel ranks: `alltoall` over NCCL, or HybridEP over NVLink and RDMA.|

## License

The ERNIE 5.5 code in Megatron Bridge is under the Apache License 2.0, as is the rest of the repository. The Triton
kernels in `modeling_ernie55/fused_ops.py` are adapted from the reference implementation (Apache-2.0, Copyright
(c) 2026 PaddlePaddle Authors) and keep its header.
