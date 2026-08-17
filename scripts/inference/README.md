# Inference launcher

`infer.sh` submits Bridge-backed offline text or vision-language generation and
HF/Megatron model comparison from a Slurm login node. It uses NeMo Run's Slurm
executor and launches one srun-native process per GPU; users should not enter an
allocation or wrap it in `srun`, `torchrun`, or `sbatch`.

```bash
./scripts/inference/infer.sh \
  --nodes 1 \
  --gpus-per-node 1 \
  --account ACCOUNT \
  --partition PARTITION \
  --time 00:30:00 \
  --container-image /path/to/megatron-bridge.sqsh \
  --mount /path/to/Megatron-Bridge:/opt/Megatron-Bridge \
  --env HF_TOKEN \
  --hf-model-path meta-llama/Llama-3.2-1B \
  --prompt "Megatron Bridge inference is" \
  --max_new_tokens 32
```

The launcher owns only Slurm resources, the container, mounts, explicitly
forwarded environment variables, and submission behavior. All other arguments
are forwarded unchanged to the selected inference entry point. Text generation
uses `text_generation.py`; `--task vlm-generation` selects
`vlm_generation.py` for multimodal generation, and `--task model-comparison`
selects `examples/conversion/compare_hf_and_megatron/compare.py` for forward-logit
parity. `--task legacy-full-prefix-generation` selects a slow, non-optimized
compatibility path and requires `--legacy-full-prefix`. It recomputes the
accumulated prefix for every decoding step for models such as GLM-5 whose
AbsorbedMLA attention does not yet support cached inference.

```bash
./scripts/inference/infer.sh \
  --task vlm-generation \
  --nodes 1 --gpus-per-node 4 \
  --account ACCOUNT --partition PARTITION \
  --container-image /path/to/megatron-bridge.sqsh \
  --mount /path/to/Megatron-Bridge:/opt/Megatron-Bridge \
  --env HF_TOKEN \
  --hf_model_path Qwen/Qwen2.5-VL-3B-Instruct \
  --image_path /shared/image.jpg \
  --prompt "Describe this image." \
  --tp 4
```

For a one-step Hugging Face/Megatron forward-logit comparison, select the
comparison task and pass the comparison entry point's arguments unchanged:

```bash
./scripts/inference/infer.sh \
  --task model-comparison \
  --nodes 1 --gpus-per-node 2 \
  --account ACCOUNT --partition PARTITION \
  --container-image /path/to/megatron-bridge.sqsh \
  --mount /path/to/Megatron-Bridge:/opt/Megatron-Bridge \
  --env HF_TOKEN \
  --hf_model_path Qwen/Qwen3-1.7B \
  --prompt "Hello world" \
  --tp 2
```

## Model and checkpoint inputs

Use `--hf-model-path` for the Hugging Face model ID or local directory that
provides the model configuration, tokenizer, and (when no Megatron checkpoint
is given) weights. Add `--megatron-model-path` to load an existing Megatron
Bridge checkpoint. The Hugging Face path may be omitted when the checkpoint's
`run_config.yaml` records `model.hf_model_id`.

```bash
./scripts/inference/infer.sh \
  --nodes 1 --gpus-per-node 8 \
  --account ACCOUNT --partition PARTITION \
  --container-image /path/to/megatron-bridge.sqsh \
  --mount /path/to/Megatron-Bridge:/opt/Megatron-Bridge \
  --mount /shared/checkpoints \
  --megatron-model-path /shared/checkpoints/model/iter_0000000 \
  --tp 8 \
  --prompt "Megatron Bridge inference is"
```

Mount every local model, checkpoint, prompt, cache, and output path used by the
job. Every host-side mount, including the repository mount, must be on storage
visible at the same path from every allocated compute node. Repeat `--prompt`
for inline prompts, or pass a mounted line-oriented or JSONL file with
`--prompt-file`. JSONL records may use a `text`, `prompt`, or `input` field.

Run the selected entry point with `--help` in a configured Megatron Bridge
environment for its full CLI.

Use `--task hf-inference` to reload an exported Hugging Face checkpoint and run
one deterministic inference. Its forwarded model argument is `--hf-model`, and
multimodal verification additionally uses `--image` and `--chat-template`.

## Environment and cluster options

`--env` accepts names only. Export a value in the login-node environment, then
repeat `--env NAME` to forward it without placing the value in generated Slurm
scripts. `CONTAINER_IMAGE`, `SLURM_ACCOUNT`, and `SLURM_PARTITION` provide
defaults for their corresponding launcher options.

No cluster-specific srun flags are added implicitly. Repeat `--srun-arg=ARG`
for each required argument; the `=` form is required when `ARG` starts with
`-`. On clusters that allocate complete GPU nodes without an explicit GPU
request, use `--no-gpu-resource-request` while retaining the requested number
of inference tasks. Small inference jobs share nodes by default; use
`--exclusive` only when the model or cluster requires exclusive nodes. Use
`--mem` to request a specific amount of host memory when the cluster default is
not sufficient.

## Output, failures, and detached jobs

By default the command waits for inference, tails the Slurm logs, prints the
rank-zero generated text in the invoking terminal, and exits nonzero if the
NeMo Run task fails. The experiment and scheduler logs remain in NeMo Run's
experiment directory under `~/.nemo_run/experiments`.

Pass `--detach` to return as soon as Slurm accepts the job. NeMo Run prints the
experiment and job identifiers needed to inspect the logs later. Use
`--submission-dry-run` (or `--dry-run`) to render the experiment without
submitting it.
