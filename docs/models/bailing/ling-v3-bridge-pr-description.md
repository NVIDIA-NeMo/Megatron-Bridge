# Suggested PR title

`feat(model): add Ling 3.0 Tiny and Flash Bridge support`

# What does this PR do ?

Add Megatron Bridge support for converting the public Hugging Face Ling 3.0 Tiny and Flash checkpoints to native Megatron distributed checkpoints (DCP), with architecture-aware mappings for KDA, gated MLA, MoE, and Flash MTP.

# Changelog

- Register one architecture-level `BailingMoeV3Bridge` for `BailingMoeV3ForCausalLM` / `bailing_hybrid`.
- Validate the public Tiny and Flash configurations and select their physical HybridModel layer schedules from the HF configuration.
- Add local KDA fused-projection and convolution mappings, including tensor parallel split/gather behavior for the semantic KDA sections.
- Add the Flash direct-Q MLA module-spec adapter and the serializable provider fields required for DCP `run_config.yaml` reload.
- Add a model-only HF-to-DCP conversion example for both public variants.
- Extend the shared hybrid FLOPs calculator with MLA and KDA accounting, including ragged dense-attention sequence statistics and unit tests for the formulas.
- Add focused bridge, registration, mapping, provider, and FLOPs unit tests.
- Add model documentation covering the architecture contract, mapping design, conversion commands, validation results, and known limitations.

# GitHub Actions CI

See the [CI section](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/CONTRIBUTING.md#-running-github-ci) in the Contributing doc for how to trigger the CI. An NVIDIA developer will need to approve and trigger the CI for external contributors.

# Before your PR is "Ready for review"

**Pre checks**:

- [x] Read and followed the [Contributor guidelines](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/CONTRIBUTING.md).
- [x] Added focused unit tests for the new bridge, mappings, provider/spec behavior, registration, and FLOPs formulas.
- [x] Added and updated model documentation and a real-checkpoint conversion example.
- [x] No new required dependency or `pyproject.toml` / `uv.lock` change is introduced.
- [x] Update the `3rdparty/Megatron-LM` development submodule pin and `.dev.commit` to the official Ling-capable draft commit `f62b8bf20ee5a03c2fd77a28362e568a0451257e`.

The GPU functional test is intentionally not included in this temporary Bridge integration scope. A manual AIStudio training smoke has been run and is recorded below; a persistent Bridge functional test can be added when the MCore pin and training support are formalized.

# Validation

## Targeted remote unit tests

Executed in the configured Linux/AIStudio runtime with the official Ling-capable MCore draft commit `f62b8bf20ee5a03c2fd77a28362e568a0451257e`:

```text
163 passed, 14 warnings in 52.22s
```

The run covered:

- `tests/unit_tests/models/bailing/test_bailing_moe3_bridge.py`
- `tests/unit_tests/models/test_autobridge_registration_matrix.py`
- `tests/unit_tests/training/utils/test_flop_utils.py`

Ruff check and format checks passed for all changed Python files. The repository pre-commit hooks also passed in an isolated hook environment.

## Manual checkpoint validation

- Tiny HF source audit covered 32 BF16 shards and 9,283 indexed tensors.
- Tiny HF-to-HybridModel conversion produced native DCP through the target model's `sharded_state_dict()`.
- Tiny TP1 strict DCP reload succeeded without factory-key or expert-offset repair.
- Tiny DCP-to-HF export covered all 9,283 source tensors with zero mismatches and maximum absolute difference `0.0`.
- Tiny EP8 reload and TP2 save/reload were exercised.
- Tiny one-step forward/backward/update/save/reload completed with finite loss and gradients and a nonzero parameter update.
- The full public Flash checkpoint was converted to native DCP and strictly reloaded.

These are manual environment validations, not a committed GPU functional test or a convergence claim.

# Known limitations and follow-up

- The Bridge submodule development pin is the official MCore draft commit `f62b8bf20ee5a03c2fd77a28362e568a0451257e`; `.main.commit` remains at the production MCore pin `24bad8e` until the MCore draft PR is merged.
- No training recipe is added in this change. The scope is model registration, checkpoint conversion, DCP reload, and focused unit coverage.
- The temporary runtime has unrelated missing Transformers symbols in the normal full Bridge import path; this does not affect the isolated Ling mapping tests.

# Additional Information

- Related issue: [NVIDIA-NeMo/Megatron-Bridge#5602](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/5602)
- Suggested labels: `feature`, `area:model`, `area:ckpt`, `high-complexity`, `needs-more-tests`
- This PR should remain a draft until the MCore draft PR is merged or its final commit is confirmed; if the final SHA changes, refresh the submodule pin and rerun the targeted validation.
