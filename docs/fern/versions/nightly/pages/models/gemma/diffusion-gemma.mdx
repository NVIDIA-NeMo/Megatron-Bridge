# DiffusionGemma SFT

DiffusionGemma support in Megatron Bridge provides offline supervised fine-tuning
of the text model using existing prompt/completion datasets. Training uses native
Gemma4 modules, the Bridge trainer, and native Megatron checkpoints. It does not
require a rollout server or a generation loop.

## Supported configuration

The loader targets the BF16 text weights of `google/diffusiongemma-26B-A4B-it`.
It recognizes the Hugging Face `DiffusionGemmaForBlockDiffusion` architecture
and derives the native provider from its text configuration.

| Setting | Support |
| --- | --- |
| Training | Full-parameter text SFT, with MoE routers frozen |
| Precision | BF16, with Transformer Engine |
| Parallelism | Data parallelism; TP, PP, EP, expert TP, and CP must each be 1 |
| Initialization | Unquantized HF weights or a matching native DiffusionGemma checkpoint |
| Training checkpoints | Native model, optimizer, scheduler, and per-DP-rank RNG state |

Full-size 26B training has not been validated. The functional checks described
below use tiny models. The launch command documents the training interface;
it is not a verified full-model hardware sizing recommendation. Data parallelism
alone does not shard the model across GPUs.

## Prepare weights and data

Use the repository's supported training environment and Transformers version.
For offline execution, make the checkpoint and tokenizer available locally before
launching. The tokenizer must match the pretrained model and provide its EOS token.
The example requires an existing local tokenizer directory even when the HF
checkpoint is specified by model ID.

Create a dataset directory containing `training.jsonl` and `validation.jsonl`.
Each line must have nonempty `input` and `output` strings, for example:

```json
{"input": "What is the capital of France?", "output": "Paris."}
```

The recipe reuses `GPTSFTDatasetConfig` and prompt/completion preprocessing. It
inserts a space between input and output, appends EOS, and uses unpacked rows.
Each row must retain a nonempty prompt and one contiguous, fully supervised final
response. Choose `--seq-length` large enough to retain both; truncated rows that
lose the final EOS or the required shifted labels are rejected.

## Run SFT

From the repository root, initialize directly from a local HF checkpoint:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run python -m torch.distributed.run --nproc_per_node=1 \
  examples/models/diffusion_gemma/sft.py \
  --hf-model /workspace/models/diffusiongemma \
  --tokenizer-model /workspace/models/diffusiongemma \
  --dataset-root /workspace/data/diffusiongemma-sft \
  --experiment-dir /workspace/runs/diffusiongemma-sft \
  --seq-length 2048 \
  --canvas-length 256 \
  --micro-batch-size 1 \
  --global-batch-size 8 \
  --train-iters 1000 \
  --encoder-loss-weight 1.0
```

The paths above represent locally prepared artifacts. Without the offline
environment variables, `--hf-model` can also be an accessible HF model ID.
This initialization uses AutoBridge before distributed wrapping; a separate
conversion recipe is not required.

To initialize from native pretrained weights, replace `--hf-model` with
`--pretrained-checkpoint /workspace/checkpoints/diffusiongemma`. The checkpoint
must match the native provider. The CLI uses the default 26B provider for this
path; smaller fixtures use the recipe's Python `model` override.

Select exactly one initialization mode. `--allow-random-init` is only for
fixtures and does not demonstrate pretrained-model quality. Increasing
`--nproc_per_node` enables data parallelism when sufficient GPU memory is
available; keep the global batch size compatible with the DP size and microbatch
size. `--canvas-length` must be positive and no larger than `--seq-length`.

### Checkpointing and resume

The recipe saves native checkpoints under `<experiment-dir>/checkpoints` and
TensorBoard logs under `<experiment-dir>/tb_logs`. It also uses the checkpoint
directory as the resume location. Repeat the launch with the same experiment
directory, initialization arguments, tokenizer, data, and model settings to
resume an existing checkpoint. Increase `--train-iters` to extend the run.
Use a new experiment directory for a fresh training run.

HF initialization supplies initial weights; subsequent training saves and resumes
use native state. Exact BF16 restart has only been established for the controlled
tiny fixture described below.

## Architecture and training objective

`DiffusionGemmaModelProvider` composes a shared Gemma4 text stack with a causal
clean encoder and a response-canvas decoder. Decoder tokens attend bidirectionally
within the selected canvas and to earlier clean prefix tokens. Full and
sliding-window layers retain their respective attention rules.

`DiffusionGemmaStep` reconstructs the clean row from the standard shifted GPTSFT
batch, EOS-fills the final response block, and samples one response canvas per
row. It corrupts canvas tokens with uniformly sampled vocabulary IDs rather than
a mask token. A detached preview pass supplies self-conditioning to a randomly
selected subset of rows before the gradient-bearing decoder pass.

The objective combines flat cross-entropy over every canvas token, including
retained tokens and EOS fill, with a separately weighted clean-encoder next-token
loss. The encoder objective includes prompt tokens as well as response tokens;
it is not restricted to the completion loss mask. The two token means are
normalized independently across DP ranks for each microbatch, then averaged by
the existing accumulation schedule. This is not one token mean across the
entire accumulation window. The recipe disables MoE auxiliary loss and freezes
routers while training the remaining text parameters.

`DiffusionGemmaBridge` reuses Gemma4 MoE parameter mappings and global K=V
projection synthesis. Decoder weights are canonical for shared text tensors;
encoder layer scalars and decoder self-conditioning weights have explicit
mappings. Conflicting shared aliases are rejected. Vision weights remain outside
the native text model.

Implementation files are under `src/megatron/bridge/diffusion/`: the
`models/diffusion_gemma` provider, model, and step; the
`conversion/diffusion_gemma` bridge; and the `recipes/diffusion_gemma` recipe.

## Validation and limitations

The tiny BF16 functional suite passed on two H100s and two B200s. It checks HF
text-weight loading and logits parity, training updates with frozen routers,
self-conditioning updates, and native model/optimizer/scheduler resume. Run the
same test group on a prepared two-GPU node:

```bash
uv run python -m pytest -v -x \
  tests/functional_tests/test_groups/diffusion/diffusion_gemma
```

Exact-resume tests set deterministic model, NCCL, and Transformer Engine controls
and use Megatron-Core's original eager `fast_gelu` expression. Production retains
the default compiled activation, so the fixture does not establish bitwise
restart equivalence for every production execution path. B200 validation also
does not establish GB200-system coverage.

This implementation rejects model parallelism, sequence parallelism, packing,
PEFT, activation recomputation/offloading, CUDA graphs, overlap schedule plans,
FP8/FP4 training, quantized or clipped-linear checkpoints, and nonzero dropout.
Multimodal training, inference/rollouts, a standalone conversion recipe,
full-size training validation, and model-quality evaluation are outside this
SFT integration.
