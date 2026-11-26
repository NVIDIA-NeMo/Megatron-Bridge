# SkyRL Megatron weekly CI

This directory runs [SkyRL](https://github.com/NovaSky-AI/SkyRL)'s Megatron
model-parity test against the Megatron-Bridge and Megatron-Core revisions checked
out by [`skyrl-e2e-weekly.yml`](../skyrl-e2e-weekly.yml). SkyRL uses
Megatron-Bridge for HF <-> Megatron conversion in its Megatron training backend,
and streams the converted weights into vLLM after every training step.

## What the test checks

`tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_models.py::test_logprobs_matching_roundtrip`
runs, for one model and one parallel layout:

1. greedy GSM8K generation in vLLM, recording per-token logprobs;
2. a Megatron forward pass on the same sequences, with the model loaded from the
   HF checkpoint through `AutoBridge` (packed THD sequences), and checks that the
   mean |logprob difference| from vLLM is below a threshold for that row;
3. a weight sync that exports the Megatron weights back to HF names through the
   bridge's `export_hf_weights` / `stream_weights_megatron_to_hf` and loads them
   into the running vLLM engine;
4. generation again, and checks the vLLM logprobs match the pre-sync run.

Together these cover both directions of the conversion: the HF -> Megatron
mapping (step 2) and Megatron -> HF export, including TP/EP gathers (steps 3-4),
across TP, PP, CP and EP layouts. A wrong mapping in either direction usually
runs without raising but gives different logprobs, and this test catches that.

## Rows

| Row ID | Model | Layout | GPUs | Bridge exercised |
| --- | --- | --- | --- | --- |
| `qwen3-moe_tp2_ep2` | `eatang/qwen3-moe-tiny-random` | TP2 EP2 | 4 | `Qwen3MoEBridge` |
| `qwen3-moe_pp2_cp2` | `eatang/qwen3-moe-tiny-random` | PP2 CP2 | 4 | `Qwen3MoEBridge` |
| `qwen3.5-0.8b-dense_tp2` | `Qwen/Qwen3.5-0.8B` | TP2 | 2 | `Qwen35Bridge` |
| `qwen3.5-35b-a3b_h100_tp4_ep4` | `Qwen/Qwen3.5-35B-A3B` | TP4 EP4 | 4 | `Qwen35MoEBridge` |
| `nemotron3.5-lightning_tp4_ep4_h100` | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | TP4 EP4 | 4 | NemotronH bridge |

The Qwen3.5 rows load the language model only. SkyRL subclasses the upstream
Qwen3.5 bridges to move the `model.language_model.*` prefix and register them
for the text-only architectures; the parameter mappings are the upstream ones.

Rows of the same test that are deliberately **not** run here:

- `glm-5.3-flash-4layer_*`: the GLM-5.3-Flash bridge, provider and Megatron-Core
  modules (KDA, mHC, k-pool DSA) are carried in SkyRL until the upstream
  Megatron-LM PRs land. A failure would point at SkyRL code, not Bridge.
- `kimi-k2.5-2layer-int4-qat_*`: depends on SkyRL's fake-INT4 QAT and its own
  Kimi K2.5 text bridge.
- `*_full_fp8`, `*_fp8_param`: mostly exercise SkyRL's FP8 weight-sync format.
  Candidates for a later addition.
- `*_mxfp8*`: need SM100 on x86_64. The SkyRL Megatron image has no aarch64
  build of all its CUDA extensions.

Run a different subset with `workflow_dispatch` (`rows` input) or the
`SKYRL_TEST_ROWS` environment variable.

## Pinned environment and version injection

The workflow pins the SkyRL source SHA (`SKYRL_REF`) and the image digest SkyRL's
own Megatron GPU CI uses. SkyRL pins its own megatron-bridge / megatron-core
revisions in `uv.lock`, so installing SkyRL alone would test those pins rather
than this checkout. The runner swaps in the checkout under test the same way the
NeMo-RL suite does:

1. Mount the Megatron-Bridge checkout and its `3rdparty/Megatron-LM` submodule.
2. Put their source roots first in `PYTHONPATH` and set
   `SKYRL_PYTHONPATH_EXPORT=1`; SkyRL's GPU test fixtures then forward
   `PYTHONPATH` into every Ray worker's runtime env.
3. Before the test, import `megatron.bridge` and `megatron.core` in the driver
   **and in a Ray worker** (each worker builds its own environment through Ray's
   uv runtime-env hook) and require both to resolve under the mounted roots.
4. Print and verify the Bridge, MCore and SkyRL SHAs.

The rest of the stack (torch, Transformer Engine, vLLM, flash-attn, ...) comes
from SkyRL's `uv.lock` at the pinned SHA. A Bridge change that needs newer
versions of those shows up as an import error, not a parity failure.

SkyRL carries a few runtime patches against the Megatron-Bridge / Megatron-Core
revisions it pins (`skyrl/backends/skyrl_train/patches/megatron/`). Each one
checks whether the upstream fix is already installed, and skips itself or fails
with a message naming the patch, so a failure there identifies the SkyRL patch
that needs updating.

## Runner requirements

- One 8x H100 node (`nemo-ci-gcp-gpu-x8`). Every row needs at most 4 GPUs; the
  rows run one after another.
- Network access to PyPI, GitHub and the Hugging Face Hub. `uv` builds SkyRL's
  environment at run time; `UV_CACHE_DIR` and `HF_HOME` point at the shared
  test-data volume so later runs reuse the wheels and checkpoints. The volume
  must be writable by uid 1000 (`ray`, the image's user).
- The first run downloads about 130 GB of checkpoints (Qwen3.5-35B-A3B and
  Nemotron-3.5-Lightning) and SkyRL's Python environment; later runs take far
  less time.

## Updating the pins

Bump `SKYRL_REF` when SkyRL bumps its own megatron-bridge pin, so the pinned
SkyRL is never far behind Bridge `main`. If a row's threshold changes upstream
in SkyRL, it comes in with the new SHA; this directory has no copy to keep in
sync.
