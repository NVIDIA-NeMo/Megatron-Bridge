# D1 publisher-comparison checks

These independent validators preserve the strict local acceptance checks for the
experimental D1 implementation. These integration and validation sources use the
repository's Apache-2.0 source license. Independently downloaded publisher artifacts
retain their accompanying license terms. No publisher source snapshot, model
weights, historical logs or training data are included. Explicitly invoking a
validator fetches and executes code at its immutable public publisher revision.

From the Bridge repository root, in the supported CUDA/Transformer Engine
environment, with one GPU visible:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python tests/functional_tests/decision_models/d1/verify_bridge_d1_3b_model.py --device cuda
CUDA_VISIBLE_DEVICES=0 uv run python tests/functional_tests/decision_models/d1/verify_bridge_d1_omni_model.py --device cuda
```

Each check constructs tiny random publisher models, changes their affine RMSNorm
weights away from one, converts every native parameter **and persistent buffer**
through the registered mappings, and requires exact full `reference.state_dict()`
key, dtype and tensor coverage on export. D1-3B also checks tied input/output weight
identity and FP32/BF16 text/image logits and mapped parameter gradients, retaining
intermediate activation and per-parameter failure diagnostics. Omni checks exact
image/audio prefixes, public text/image/audio decision probabilities for choice,
score and yes/no readouts, temperature/polarity conventions, and differentiable
text decisions. Omni uses the pinned publisher class's own configuration; a
separate AutoModel checkout is unnecessary.

The existing `test_groups/models/d1/test_full_models.py` tests additionally compare
Omni parameter gradients through text, image and audio in FP32. These checks
complement rather than replace that suite. Neither tiny random-model comparison
nor a small sampled public-baseline check establishes complete released-size
checkpoint parity or production distributed training acceptance.

The validators have not been rerun on CUDA as part of this source preparation.
CPU coverage checks only exercise the strict-checker's rejection of missing
buffers/keys, modified tensors/dtypes and insufficient non-unit norm fixtures:

```bash
uv run python -m pytest --noconftest -o addopts= tests/functional_tests/decision_models/d1/test_validation_contract.py
```

Imports and `--help` perform no publisher download or CUDA initialization. Actual
runs own and clean a fresh standalone one-GPU process group, including on errors;
invocation within an existing distributed job is rejected. Keep this directory,
the D1 provider/mapping/model/registry/tests/example and D1 tutorial together
when splitting the contribution.
