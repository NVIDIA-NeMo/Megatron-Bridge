# Deterministic startup

Bridge uses `megatron.determinism.configure_determinism` to establish the
same process-wide startup policy as Megatron Core training. An MCore revision
containing that API is required when deterministic mode is requested. Older
revisions fail with an upgrade message. Runs with deterministic mode disabled
continue to work without the new API when no early policy has been applied.

## Enable the policy

Configure the process before importing Bridge or Core. Their GPU dependencies
can initialize CUDA during import, before a recipe exists:

```python
from megatron.determinism import configure_determinism

configure_determinism({"deterministic_mode": True})

from megatron.bridge.recipes.utils.determinism_utils import apply_determinism_overrides
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain

cfg = my_recipe()
apply_determinism_overrides(cfg)
cfg.rng.seed = 1234
pretrain(cfg, forward_step)
```

Alternatively, start an existing script with `python -m megatron.determinism
train.py ...`, or use `torchrun --nproc-per-node=2 --module megatron.determinism
train.py ...` for fresh distributed workers. The script must still enable the
model's deterministic mode, for example with the recipe helper shown above.
The launcher preserves the script's arguments and applies policy before imports.

The helper enables `model.deterministic_mode`, disables fused cross entropy and
TP communication overlap in both the model and communication config, and disables
MoE auxiliary-loss fusion. Router fusion can remain enabled when MCore exposes
the separate auxiliary-loss option. The helper prepares config and environment
defaults; it does not initialize CUDA, seed RNGs, or apply process-wide policy.

`pretrain` resolves mixed precision and communication options, then rechecks the
early policy at the start of `ConfigContainer.validate()`, before model or other
sub-config finalization. `initialize_megatron()` also checks the policy before its
first CUDA probe, including for direct callers. Explicit shell/launcher environment
settings take precedence over recipe defaults, and MCore rejects incompatible
settings instead of silently replacing them. If `cfg.env_vars` requests a different
policy value after early setup, Bridge warns and retains the established value.
To select another supported setting (for example `CUBLAS_WORKSPACE_CONFIG=:16:8`),
export it before starting the launcher; recipe construction happens too late.

`scripts/training/run_recipe.py --deterministic` configures the early policy
before Bridge imports. Recipe-environment re-execution preserves the determinism
launcher, including when the outer command supplied that launcher directly.

MCore supplies unset `NCCL_ALGO=Ring`, `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`, and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, enables strict Torch deterministic algorithms,
disables cuDNN benchmarking, and enables cuDNN deterministic mode. Its shared
validator owns accepted NCCL choices, SSM settings, and Triton cache requirements.
In particular, `NCCL_ALGO=Tree` is no longer accepted by Bridge's deterministic
training path. A DEBUG-level `Determinism policy:` record describes effective
settings.

All deterministic Bridge entrypoints need the early call or launcher before
Bridge imports, using compatible environment values and final model options.
The first call after initialization fails and requires a fresh process. Repeated
calls after initialization are allowed only if the same process established the
policy early and the checked environment and strict Torch settings have not
changed. Changing a recipe from deterministic to ordinary execution should use
a fresh process; disabling the config flag after early setup is rejected.

## Evidence and limits

Startup settings do not certify a recipe as deterministic. Keep data order,
seeds, parallel layout, software, hardware and topology fixed, and compare
independent executions and checkpoint resumes. A fixed NCCL algorithm does not
fix the physical collective topology. Record and verify Triton cache contents
when using a cache, and measure overhead on the original recipe without test
instrumentation.

The startup integration supports providers with a top-level deterministic-mode
config and GPT/Hybrid model configs that proxy their transformer settings.
MegatronMIMO's per-module providers need a separate policy adapter; its existing
top-level check rejects early deterministic startup with an explicit unsupported-provider error. Inference,
conversion and user code that executes kernels before training startup also need
their own early API call.

The CPU adapter tests exercise the actual MCore policy and selected Bridge
function bodies without importing GPU dependencies:

```bash
uv run python -m pytest --confcutdir=tests/unit_tests/training/determinism \
  tests/unit_tests/training/determinism/test_startup_policy.py
```

When testing an MCore change before updating Bridge's submodule, set
`MCORE_SOURCE_ROOT` to the checkout providing that API. This only selects the
public policy package for CPU tests; it does not change Bridge's installed MCore package.
The functional startup tests use real public imports, config finalization,
single-rank NCCL initialization and a small output/gradient byte comparison in
fresh processes. Provider and builder configs (including the early launcher), late setup, and
post-initialization drift are checked in the H100 and GB200 L0 launchers. They require the matching
MCore revision in the GPU environment and do not qualify full-model training or
multi-rank replay.
