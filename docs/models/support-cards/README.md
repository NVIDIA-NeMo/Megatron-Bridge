# Model Support Cards (Draft)

> This page is an illustrative design draft. It does not add or change a
> release support claim for any model.

Model support cards make the scope and evidence behind a support claim
reviewable. A card is release-scoped and describes one exact model target. It
does not infer support from the presence of a bridge, recipe, example, or test
file.

The first placeholder card below uses the dense `Qwen/Qwen3-8B` checkpoint to
make the proposed fields concrete. Qwen3-MoE, Qwen3-Next, other Qwen3 sizes,
quantized checkpoints, and instruction-tuned variants require separate cards
when their verification evidence differs.

## Card identity

A card identifies one verification target with these fields:

| Field | Requirement |
|---|---|
| Release | Megatron Bridge release or POR being qualified. |
| Hugging Face model | Exact public repository or checkpoint path. |
| Hugging Face revision | Immutable commit revision used by verification. Required for `Verified`. |
| Architecture | Exact Hugging Face architecture class. |
| Modality | Text, vision-language, audio, omni, diffusion, or another explicit modality. |
| Checkpoint scope | Full checkpoint or a named component-only scope. |
| Precision / format | BF16, FP8, FP4, or another format when it changes the support claim. |
| Owner | Team or handle responsible for keeping the card current. |

Cards must be split when architecture, model size, Base/Instruct flavor,
modality, checkpoint scope, precision, or supported workflow changes the
evidence. Family pages may aggregate cards, but family membership alone cannot
promote a target to `Verified`.

## Support and verification

Each capability records an independent support declaration and verification
state. This keeps implementation presence, product intent, and qualification
evidence from being conflated.

Support declaration:

| Support | Meaning |
|---|---|
| `TBD` | The POR support decision has not been made for this exact target and release. |
| `Supported` | The capability is in the intended support scope. Verification is recorded separately. |
| `Unsupported` | The capability is intentionally outside the support scope; the card must explain why. |
| `N/A` | The capability does not apply to this target; the card must explain why. |

Verification state:

| Verification | Meaning |
|---|---|
| `Unverified` | There is no current qualifying pass for the exact card scope. |
| `Verified` | The latest qualifying run passed the capability's acceptance contract on the named release stack, and evidence is linked. |

Partial support is expressed by narrowing or splitting the card scope and
documenting the excluded configurations. A capability can be `Supported` and
`Unverified`; implementation or test presence alone does not imply either
state.

There is no intermediate launch-only verification state. A short run may be
recorded as evidence, but it does not produce `Verified` unless it satisfies
that capability's acceptance contract.

Test execution outcomes are separate from support status:

| Run outcome | Meaning |
|---|---|
| `Pass` | The recorded command met its stated acceptance criteria. |
| `Fail` | The command ran and failed an acceptance criterion. |
| `Blocked` | The command could not run because a prerequisite was unavailable. |
| `Not run` | No attempt was made for this release target. |

A blocked or failed run does not by itself mean `Unsupported`. It leaves the
support declaration unchanged and the capability `Unverified`.

Verification is current only for the exact release and stack recorded by the
card. A newer release starts `Unverified`. If a later qualifying run fails on
the same card, verification returns to `Unverified`, the failed run becomes the
latest outcome, and the prior passing evidence remains in history. This makes
staleness and regression visible without adding another verification level.

## Required capabilities

Every card records all of these capabilities so omission is not confused with
support:

1. Hugging Face to Megatron checkpoint conversion
2. Megatron to Hugging Face checkpoint conversion
3. Megatron checkpoint inference for the applicable modality
4. Pretraining
5. Supervised fine-tuning
6. Parameter-efficient fine-tuning
7. Checkpoint save, load, and resume
8. Convergence, scoped to a named training mode
9. Optimized performance, scoped to hardware, precision, and topology

Examples, recipes, unit tests, functional tests, and CI launchers are
references or evidence for these capabilities. Their existence is not a
separate capability and does not automatically produce `Verified`.

## Evidence contract

A `Verified` capability must link a reviewable result that records:

- exact Hugging Face model and immutable revision;
- Megatron Bridge and Megatron Core commit SHAs;
- container image or digest and relevant dependency versions;
- hardware, GPU count, precision, and parallelism topology;
- exact command, test, or certification workload;
- result date and run outcome;
- comparison method, tolerance, metrics, and skipped values where applicable;
- input and output artifact identity for checkpoint workflows; and
- known limitations and the scope that was not exercised.

Each capability therefore stores an acceptance-contract identifier, its
evidence history, and the latest qualifying run. `Unsupported` and `N/A`
support declarations require a reason. `Verified` is invalid without an
immutable Hugging Face revision, a contract, and at least one current passing
evidence record.

Historical results remain useful evidence, but do not qualify a different
release. CI tier (`L0`, `L1`, or `L2`) records scheduling frequency and cost;
it is not a verification strength.

## Proposed machine-readable shape

This shape is included for design review only. A schema, validator, renderer,
and generated public matrix are follow-up work after the fields are agreed.

```json
{
  "schema_version": 1,
  "id": "qwen3-8b-bf16-26-08",
  "display_name": "Qwen3 8B",
  "release": "26.08",
  "target": {
    "hf_model_id": "Qwen/Qwen3-8B",
    "hf_revision": null,
    "architecture": "Qwen3ForCausalLM"
  },
  "scope": {
    "modalities": ["text"],
    "checkpoint": "full",
    "precision": "bf16"
  },
  "owner": null,
  "capabilities": [
    {
      "id": "hf_to_megatron",
      "support": "tbd",
      "verification": "unverified",
      "acceptance_contract": null,
      "evidence": [],
      "latest_run": null,
      "reason": null
    },
    {
      "id": "megatron_to_hf",
      "support": "tbd",
      "verification": "unverified",
      "acceptance_contract": null,
      "evidence": [],
      "latest_run": null,
      "reason": null
    },
    {
      "id": "inference",
      "support": "tbd",
      "verification": "unverified",
      "acceptance_contract": null,
      "evidence": [],
      "latest_run": null,
      "reason": null
    },
    {
      "id": "pretrain",
      "support": "tbd",
      "verification": "unverified",
      "acceptance_contract": null,
      "evidence": [],
      "latest_run": null,
      "reason": null
    },
    {
      "id": "sft",
      "support": "tbd",
      "verification": "unverified",
      "acceptance_contract": null,
      "evidence": [],
      "latest_run": null,
      "reason": null
    },
    {
      "id": "peft",
      "support": "tbd",
      "verification": "unverified",
      "acceptance_contract": null,
      "evidence": [],
      "latest_run": null,
      "reason": null
    },
    {
      "id": "checkpoint_resume",
      "support": "tbd",
      "verification": "unverified",
      "acceptance_contract": null,
      "evidence": [],
      "latest_run": null,
      "reason": null
    },
    {
      "id": "convergence",
      "support": "tbd",
      "verification": "unverified",
      "acceptance_contract": null,
      "evidence": [],
      "latest_run": null,
      "reason": null
    },
    {
      "id": "performance",
      "support": "tbd",
      "verification": "unverified",
      "acceptance_contract": null,
      "evidence": [],
      "latest_run": null,
      "reason": null
    }
  ],
  "references": [
    "docs/models/qwen/qwen.md",
    "src/megatron/bridge/models/qwen/qwen3_bridge.py",
    "src/megatron/bridge/recipes/qwen/h100/qwen3.py"
  ],
  "known_limitations": [
    "Verification evidence and an immutable Hugging Face revision are pending."
  ]
}
```

## Qwen3 8B placeholder

This example inventories repository references without treating them as a
release certification result.

### Identity

| Field | Value |
|---|---|
| Release | 26.08 |
| Family | Qwen3 dense |
| Hugging Face model | `Qwen/Qwen3-8B` |
| Hugging Face revision | TBD |
| Architecture | `Qwen3ForCausalLM` |
| Modality | Text |
| Checkpoint scope | Full checkpoint |
| Precision / format | BF16 |
| Owner | TBD |
| POR qualification | TBD |

### Capability profile

| Capability | Support | Verification | Repository references | Missing qualification evidence |
|---|---|---|---|---|
| Hugging Face to Megatron | `TBD` | `Unverified` | Bridge, unit test, toy conversion test, active H100/GB200 L0 launchers | Immutable HF revision and current-release real-checkpoint result |
| Megatron to Hugging Face | `TBD` | `Unverified` | Bridge, toy round-trip test, active H100/GB200 L0 launchers | Immutable HF revision, exported artifact inventory, and current-release result |
| Megatron inference | `TBD` | `Unverified` | Qwen guide and generic text-generation workflow | Current-release result from the imported Megatron checkpoint |
| Pretraining | `TBD` | `Unverified` | Qwen3 8B recipe and downsized Qwen recipe test | Exact-target run, checkpoint artifact, and acceptance result |
| SFT | `TBD` | `Unverified` | Qwen3 8B recipe | Exact-target run, checkpoint artifact, and acceptance result |
| PEFT | `TBD` | `Unverified` | Qwen3 8B LoRA/DoRA recipe; LoRA-only toy export and verification tests | Exact-target LoRA/DoRA training and adapter verification result |
| Checkpoint save/load/resume | `TBD` | `Unverified` | Related Qwen3-4B checkpoint compatibility coverage | Exact Qwen3-8B scope and current-release result |
| Convergence | `TBD` | `Unverified` | None linked | Mode-specific workload, metrics, thresholds, and result |
| Optimized performance | `TBD` | `Unverified` | None linked | Hardware-specific runnable recipe, measurements, and recommended topology |

### Repository reference inventory

These references establish implementation and regression-test coverage only:

- [Qwen3 bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/qwen/qwen3_bridge.py)
- [Qwen3 recipes](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/qwen/h100/qwen3.py)
- [Qwen model guide](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/models/qwen/qwen.md)
- [Qwen3 bridge unit test](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/tests/unit_tests/models/qwen/test_qwen3_bridge.py)
- [Qwen3 conversion functional test](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/tests/functional_tests/test_groups/models/qwen/test_qwen3_conversion.py)
- [H100 active L0 launcher](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/tests/functional_tests/launch_scripts/h100/active/L0_Launch_models_qwen3.sh)
- [GB200 active L0 launcher](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/tests/functional_tests/launch_scripts/gb200/active/L0_Launch_models_qwen3.sh)
- [Qwen recipe test](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/tests/functional_tests/test_groups/recipes/test_qwen_recipes_pretrain.py)
- [Qwen3 PEFT export test](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/tests/functional_tests/test_groups/models/qwen/test_qwen3_peft_export.py)
- [Qwen3 PEFT verification test](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/tests/functional_tests/test_groups/models/qwen/test_qwen3_peft_verify.py)
- [Qwen3 checkpoint compatibility test](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/tests/functional_tests/test_groups/ckpts/qwen3_4b/test_qwen3_4b_ckpt.py)

### Release evidence placeholders

| Evidence field | Value |
|---|---|
| Hugging Face revision | TBD |
| Megatron Bridge commit | TBD |
| Megatron Core commit | TBD |
| Container image / digest | TBD |
| Hardware and topology | TBD |
| Conversion result and comparison criteria | TBD |
| Inference result | TBD |
| Training and checkpoint-resume result | TBD |
| Convergence result | TBD |
| Performance report | TBD |

## Proposed public summary

Once cards are machine-readable, the public page can be generated as a compact
index. Detail and evidence remain on the card.

Each capability cell uses `support / verification`; no multi-capability rollup
is implied.

| Model target | Capability | Support / verification | Latest run | POR |
|---|---|---|---|---|
| Qwen3 8B · BF16 · 26.08 | HF to Megatron | `TBD / Unverified` | Not run | TBD |
|  | Megatron to HF | `TBD / Unverified` | Not run |  |
|  | Inference | `TBD / Unverified` | Not run |  |
|  | Pretraining | `TBD / Unverified` | Not run |  |
|  | SFT | `TBD / Unverified` | Not run |  |
|  | PEFT | `TBD / Unverified` | Not run |  |
|  | Checkpoint resume | `TBD / Unverified` | Not run |  |
|  | Convergence | `TBD / Unverified` | Not run |  |
|  | Performance | `TBD / Unverified` | Not run |  |

## Review questions

1. Is exact checkpoint plus release the right card granularity?
2. Are independent support declarations and binary verification sufficient without an intermediate verification level?
3. Does the proposed latest-run behavior preserve regressions and historical evidence clearly enough?
4. Do the nine required capabilities and evidence fields capture the support claims we intend to publish?
