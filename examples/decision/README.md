# Typed decision post-training

These examples preserve each publisher's finite option ordering and report accuracy, utility, NLL, Brier score and calibration. They evaluate a released baseline, tune its existing decision head, and optionally optimize expected utility with a KL reference. They do not generate answer text.

| Entry point | Training scope |
|---|---|
| `clm_posttrain.py` | Frozen native Qwen3 encoder; CLM state/action projection heads |
| `clef_posttrain.py --head-only` | Frozen native Qwen3.5 backbone; CLEF joint schema head |
| `clef_posttrain.py` | Native backbone LoRA plus CLEF joint schema head |
| `d1_posttrain.py` | D1 answer-token row adapter or Omni marker head; experimental, GPU acceptance pending |

The [tutorials](../../tutorials/decision_models/README.md) generate a new synthetic fixture and provide notebook entry points. No organization data, pretrained weights or recorded training results are included. The small checked-in fixture tests mechanics; it does not establish model quality.

## CLM and CLEF

From the repository root, generate separate training, validation and test data:

```bash
PYTHONPATH=tutorials/decision_models uv run python -m decision_tutorial.data \
  --output-dir runs/typed-decisions/data
```

Choose the number of GPUs and tensor parallelism to fit the backbone. For CLM, the first invocation caches embeddings; subsequent stages reuse them. The cache identity includes the encoder ID, revision, truncation length and input texts.

```bash
uv run python -m torch.distributed.run --nproc_per_node=1 examples/decision/clm_posttrain.py \
  --mode eval --data-dir runs/typed-decisions/data --output-dir runs/clm/base
uv run python -m torch.distributed.run --nproc_per_node=1 examples/decision/clm_posttrain.py \
  --mode train --loss-mode sft --data-dir runs/typed-decisions/data --output-dir runs/clm/sft
uv run python -m torch.distributed.run --nproc_per_node=1 examples/decision/clm_posttrain.py \
  --mode train --loss-mode rl --heads runs/clm/sft/heads.pt --eval-metric reward \
  --data-dir runs/typed-decisions/data --output-dir runs/clm/utility
```

CLEF requires enough GPU memory for its 27B backbone. This example uses four ranks with TP=4. Pass `--head-only` consistently for frozen backbone tuning; omit it for LoRA plus head tuning.

```bash
uv run python -m torch.distributed.run --nproc_per_node=4 examples/decision/clef_posttrain.py \
  --mode eval --tp 4 --head-only --data-dir runs/typed-decisions/data --output-dir runs/clef/base
uv run python -m torch.distributed.run --nproc_per_node=4 examples/decision/clef_posttrain.py \
  --mode train --loss-mode sft --tp 4 --head-only \
  --data-dir runs/typed-decisions/data --output-dir runs/clef/sft
uv run python -m torch.distributed.run --nproc_per_node=4 examples/decision/clef_posttrain.py \
  --mode train --loss-mode rl --tp 4 --head-only --init-from runs/clef/sft/best --eval-metric reward \
  --data-dir runs/typed-decisions/data --output-dir runs/clef/utility
```

Validation selects checkpoints. Every stage writes `metrics_validation.json` and `metrics_test.json`; compare the untouched baseline with each selected stage. Expected utility can increase while NLL or calibration worsens, so report those measures together. `expected_reward` enumerates the finite option set exactly; optional `grpo` samples choices. These are adaptation objectives, not claims that the publishers released these RL recipes.

Default Hub sources are pinned:

| Source | Revision |
|---|---|
| `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |
| `Contrastive-LM/CLM-v0.1-8B` heads | `e939398d4556fcd9400c76fa8c5a513202f42b0a` |
| `Cloudflare/clef` | `ed3eed331870db2eff4b0db01237128ede8a00ce` |

Override with `--encoder-revision` / `--heads-revision` for CLM, or `--revision` for CLEF, when using another checkpoint. A checkpoint revision does not pin the CUDA runtime or installed dependency versions.

The declarative `models.clm.decision_head` and `models.clef.decision_head` APIs share their architectures with these recipe utilities. Frozen-head InfoNCE and experimental calibration-aware continuation functions live in `training.post_training.decision`; those objectives remain distinct from the SFT and expected-utility recipe flags above.

## D1 experimental support

The native D1 integration is provided under this repository's Apache-2.0 source license. Target-GPU acceptance remains pending. Runtime download of the publisher's custom code requires explicit `trust_remote_code=True` in the recipe. No publisher source snapshot, weights or trained readout is bundled; independently downloaded artifacts retain their accompanying publisher terms.

The native language modules currently require TP=PP=CP=1 with sequence parallelism disabled. The recipe supports text adaptation; image/audio encoders are preserved and frozen. Its [strict validators](../../tests/functional_tests/decision_models/d1/README.md) and GPU tests use pinned publisher downloads as references, including non-unit normalization weights, media boundaries and backward gradients. Those checks must run on the target hardware before making parity or post-training quality claims. CPU checker tests establish coverage and rejection behavior, not publisher numerical parity.

```bash
uv run python -m torch.distributed.run --nproc_per_node=1 examples/decision/d1_posttrain.py \
  --model d1-omni --mode base --dtype float32 --data-dir runs/typed-decisions/data \
  --cache-dir runs/d1/cache --output-dir runs/d1/base
```

Use `--model d1-3b` for the causal VL backbone. SFT uses `--mode sft`; utility continuation uses `--mode rl --input-dir <selected-sft-stage>`. Float32 is the default. The Omni publisher recommends float32 or float16 and warns that bfloat16 can change answers. Cache identity records precision and source hashes.

## Publisher attribution

CLM's projection layout and prompting follow [Contrastive-LM/CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) and the Apache-2.0 [reference package](https://github.com/Contrastive-LM/CLM). The CLM checkpoint is published under Apache-2.0; preserve the terms accompanying independently obtained artifacts.

The attributed section in `src/megatron/bridge/decision/clef_head.py` is adapted from [Cloudflare CLEF](https://huggingface.co/Cloudflare/clef), Apache-2.0. Its accompanying release license contains the notice `Copyright 2026 Alibaba Cloud`, retained in the adapted module. Modified areas are identified in that file. The typed head wrapper delegates to that implementation.

[LiquidAI d1-3B](https://huggingface.co/LiquidAI/d1-3B) and [d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M) model artifacts carry the LFM Open License v1.0. This source contribution redistributes no model weights or publisher code snapshot and does not change the terms accompanying independently obtained publisher artifacts or runtime downloads.
