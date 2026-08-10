# Nemotron 3.5 Lightning on Megatron Bridge 0.5.1

This directory is the release-specific entrypoint for
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` on Megatron Bridge 0.5.1.
Use it with the immutable NeMo 26.06.01 runtime documented below and a checkout
that contains this directory.

The unmodified 26.06.01 image does not contain this model support or the newer
generic `scripts/training/train.sh` and `scripts/conversion/convert.sh`
wrappers. It does contain the Python entrypoints used here. These model-local
wrappers are intentionally scoped to the 0.5.1 release line; Megatron Bridge
0.6 and newer use the generic launchers instead.

## Contents

| File | Purpose |
|---|---|
| `conversion.sh` | Import a Hugging Face checkpoint or export a Megatron checkpoint |
| `inference.sh` | Generate from the HF checkpoint or an imported Megatron checkpoint |
| `pretrain.sh` | Pretraining recipe with H100 and GB200 profiles |
| `sft.sh` | Packed OpenMathInstruct-2 full SFT with H100 and GB200 profiles |
| `lora.sh` | Rank-32 LoRA fine-tuning with H100 and GB200 profiles |
| `adapter.sh` | Export a LoRA adapter or merge a Megatron LoRA checkpoint |

All commands in this README use Docker directly and are independent of
external launch tooling.

## Fresh-checkout bootstrap

Clone the release branch and initialize the Megatron-LM submodule before
starting the container:

```bash
git clone --branch nemotron-3.5-lightning-mb-0.5.1 \
  https://github.com/NVIDIA-NeMo/Megatron-Bridge.git megatron-bridge
cd megatron-bridge
git submodule update --init 3rdparty/Megatron-LM

test -f 3rdparty/Megatron-LM/megatron/core/__init__.py
git rev-parse HEAD
git submodule status 3rdparty/Megatron-LM
```

Run the remaining host commands from this repository root. Keep the workspace
outside the checkout so generated files do not dirty the source tree:

```bash
export REPO_ROOT="$PWD"
export LIGHTNING_STATE="${LIGHTNING_STATE:-$PWD/../nemotron-3.5-lightning-workspace}"
mkdir -p "$LIGHTNING_STATE"/{cache,models,results}
```

Plan storage before downloading or training. Approximate observed sizes are:

| Artifact | Space |
|---|---:|
| Pinned HF source cache | 70 GiB |
| Imported Megatron checkpoint | 70 GiB |
| Each full-model HF export or merged checkpoint | 70 GiB |
| Full-SFT checkpoint | 70 GiB |
| LoRA checkpoint and exported adapter | 5 GiB |
| 100-step pretraining checkpoint with optimizer state | 450 GiB |

Allow at least 160 GiB for import plus export, or about 1 TiB to retain every
documented artifact and both dataset caches at once.

## Build the immutable runtime image

The validated 26.06.01 image digest is pinned so rebuilding this small local
runtime tag cannot silently select different base contents:

```bash
export NEMO_IMAGE="nvcr.io/nvidia/nemo@sha256:912033288c982a8c4af05df46a1d670c34350f1427c758f5da9c485bdec57264"
export LIGHTNING_IMAGE="nemotron-3.5-lightning-mb-0.5.1:26.06.01"

docker pull "$NEMO_IMAGE"
docker build --pull=false \
  -t "$LIGHTNING_IMAGE" -f - . <<'DOCKERFILE'
FROM nvcr.io/nvidia/nemo@sha256:912033288c982a8c4af05df46a1d670c34350f1427c758f5da9c485bdec57264
WORKDIR /opt/Megatron-Bridge
DOCKERFILE

docker image inspect "$LIGHTNING_IMAGE" --format '{{.Id}} {{json .RepoDigests}}'
```

Start a fresh container with the checkout read-only and all model, dataset,
and output state on the persistent workspace mount:

```bash
docker run --rm -it --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$REPO_ROOT:/opt/Megatron-Bridge:ro" \
  -v "$LIGHTNING_STATE:/workspace" \
  -e HF_HOME=/workspace/cache/huggingface \
  -e HF_HUB_CACHE=/workspace/cache/huggingface/hub \
  -e HF_DATASETS_CACHE=/workspace/cache/huggingface/datasets \
  -e NEMO_HOME=/workspace/cache/nemo \
  -e NEMO_DATASETS_CACHE=/workspace/cache/nemo/datasets \
  -w /opt/Megatron-Bridge \
  "$LIGHTNING_IMAGE" bash
```

In the container, activate the bundled environment and export the paths used
by every subsequent command:

```bash
source /opt/venv/bin/activate
cd /opt/Megatron-Bridge

export PYTHONPATH="$PWD/src:$PWD/3rdparty/Megatron-LM:${PYTHONPATH:-}"
export WORKSPACE=/workspace
export HF_HOME=/workspace/cache/huggingface
export HF_HUB_CACHE=/workspace/cache/huggingface/hub
export HF_DATASETS_CACHE=/workspace/cache/huggingface/datasets
export NEMO_HOME=/workspace/cache/nemo
export NEMO_DATASETS_CACHE=/workspace/cache/nemo/datasets
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$NEMO_DATASETS_CACHE" /workspace/{models,results}
```

The scripts use `uv run --active --no-sync` with the bundled environment. No
package installation or dependency change is required.

## Prepare immutable inputs

The model and both training datasets are pinned to immutable revisions:

```bash
export HF_MODEL_ID="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
export HF_MODEL_REVISION="b3caaabed0263651a17dc1f2d4ce97e794f76c44"
export OPENMATH_REVISION="469216e3f46f4dacf476b382e192485ea51a143e"
export SQUAD_REVISION="7b6d24c440a36b6815f21b70d25016731768db1f"
export HF_MODEL_PATH_FILE=/workspace/models/nemotron-3.5-lightning-hf-source.path
```

Download the model once into the standard persistent HF cache and record the
resolved snapshot path:

```bash
uv run --active --no-sync python - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

snapshot = snapshot_download(
    repo_id=os.environ["HF_MODEL_ID"],
    revision=os.environ["HF_MODEL_REVISION"],
    cache_dir=os.environ["HF_HUB_CACHE"],
)
Path(os.environ["HF_MODEL_PATH_FILE"]).write_text(f"{Path(snapshot).resolve()}\n")
print(snapshot)
PY

export HF_MODEL="$(<"$HF_MODEL_PATH_FILE")"
test -f "$HF_MODEL/config.json"
```

Prefetch the exact dataset revisions into the same paths used by the SFT and
LoRA builders. This step downloads raw data; packing occurs when training first
uses each dataset.

```bash
uv run --active --no-sync python - <<'PY'
import os
from pathlib import Path

from datasets import load_dataset

cache_root = Path(os.environ["NEMO_DATASETS_CACHE"])
load_dataset(
    "nvidia/OpenMathInstruct-2",
    split="train_1M",
    revision=os.environ["OPENMATH_REVISION"],
    cache_dir=str(cache_root / "nvidia/OpenMathInstruct-2"),
)
load_dataset(
    "rajpurkar/squad",
    revision=os.environ["SQUAD_REVISION"],
    cache_dir=str(cache_root / "rajpurkar/squad"),
)
PY
```

## Checkpoint conversion

Import the pinned local HF snapshot with TP1/EP8. Eight local ranks are
required:

```bash
export BASE_MEGATRON_ROOT=/workspace/models/nemotron-3.5-lightning-megatron
export BASE_MEGATRON_CHECKPOINT="$BASE_MEGATRON_ROOT/iter_0000000"

NPROC_PER_NODE=8 HF_MODEL="$HF_MODEL" \
MEGATRON_PATH="$BASE_MEGATRON_ROOT" TP=1 EP=8 \
  ./examples/models/nemotron/nemotron_3_5_lightning/conversion.sh import

printf '%s\n' "$HF_MODEL_REVISION" > "$BASE_MEGATRON_ROOT/source_hf_revision.txt"
test "$(<"$BASE_MEGATRON_ROOT/latest_checkpointed_iteration.txt")" = 0
test -f "$BASE_MEGATRON_CHECKPOINT/run_config.yaml"
```

Export that checkpoint back to native HF format:

```bash
export BASE_HF_EXPORT=/workspace/models/nemotron-3.5-lightning-hf-export

NPROC_PER_NODE=8 HF_MODEL="$HF_MODEL" \
MEGATRON_PATH="$BASE_MEGATRON_CHECKPOINT" HF_EXPORT_PATH="$BASE_HF_EXPORT" \
TP=1 EP=8 \
  ./examples/models/nemotron/nemotron_3_5_lightning/conversion.sh export

test -f "$BASE_HF_EXPORT/model.safetensors.index.json"
```

### Exact round-trip tensor audit

The following verifier compares the source and round-trip HF artifacts by
tensor name, shape, native dtype, and exact value. It exits nonzero on the
first mismatch and includes MTP tensors in the same audit:

```bash
export HF_EXPORT_PATH="$BASE_HF_EXPORT"
uv run --active --no-sync python - <<'PY'
import json
import os
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open


def weight_map(root: Path) -> dict[str, str]:
    indexes = sorted(root.glob("*.safetensors.index.json"))
    if len(indexes) == 1:
        return json.loads(indexes[0].read_text())["weight_map"]
    if indexes:
        raise RuntimeError(f"Expected one safetensors index in {root}, found {len(indexes)}")

    result = {}
    for shard in sorted(root.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            result.update({name: shard.name for name in handle.keys()})
    if not result:
        raise RuntimeError(f"No safetensors weights found in {root}")
    return result


source = Path(os.environ["HF_MODEL"])
exported = Path(os.environ["HF_EXPORT_PATH"])
source_map = weight_map(source)
exported_map = weight_map(exported)
if source_map.keys() != exported_map.keys():
    missing = sorted(source_map.keys() - exported_map.keys())
    extra = sorted(exported_map.keys() - source_map.keys())
    raise SystemExit(f"Tensor schema mismatch: missing={missing[:10]}, extra={extra[:10]}")

with ExitStack() as stack:
    source_handles = {
        shard: stack.enter_context(safe_open(source / shard, framework="pt", device="cpu"))
        for shard in set(source_map.values())
    }
    exported_handles = {
        shard: stack.enter_context(safe_open(exported / shard, framework="pt", device="cpu"))
        for shard in set(exported_map.values())
    }
    for name in sorted(source_map):
        expected = source_handles[source_map[name]].get_tensor(name)
        actual = exported_handles[exported_map[name]].get_tensor(name)
        if expected.shape != actual.shape or expected.dtype != actual.dtype or not torch.equal(expected, actual):
            raise SystemExit(
                f"Tensor mismatch: {name}; source={expected.shape}/{expected.dtype}, "
                f"export={actual.shape}/{actual.dtype}"
            )

print(f"PASS: {len(source_map)} tensors match by name, shape, dtype, and exact value")
PY
```

## Inference

Generate from the imported Megatron checkpoint:

```bash
NPROC_PER_NODE=8 HF_MODEL="$HF_MODEL" \
MEGATRON_PATH="$BASE_MEGATRON_CHECKPOINT" \
PROMPT="The capital of France is" TP=1 EP=8 \
  ./examples/models/nemotron/nemotron_3_5_lightning/inference.sh
```

Generate by importing the pinned HF snapshot in memory:

```bash
NPROC_PER_NODE=8 HF_MODEL="$HF_MODEL" MEGATRON_PATH= \
PROMPT="The capital of France is" TP=1 EP=8 \
  ./examples/models/nemotron/nemotron_3_5_lightning/inference.sh
```

Both commands should complete without an exception and produce a continuation
beginning with `The capital of France is Paris.`.

## Select a training profile

Choose one hardware profile in the container. The profile determines the
recipe, local process count, and full-SFT tensor parallelism; EP is 8 for all
documented workflows.

```bash
# Choose exactly one value: h100 or gb200.
export HARDWARE=gb200

case "$HARDWARE" in
  h100)
    export PRETRAIN_NPROC=16 SFT_NPROC=16 SFT_TP=2
    ;;
  gb200)
    export PRETRAIN_NPROC=8 SFT_NPROC=8 SFT_TP=1
    ;;
  *)
    echo "HARDWARE must be h100 or gb200" >&2
    return 2
    ;;
esac
export LORA_NPROC=8
```

The GB200 SFT and LoRA profiles select HybridEP exactly as implemented by the
scripts. The commands below do not apply an undocumented dispatcher fallback.

## Pretraining

The default dataset is mock data. The 100-step default saves optimizer state
and can create a checkpoint of approximately 450 GiB.

```bash
export PRETRAIN_OUTPUT=/workspace/results/nemotron-3.5-lightning-pretrain-$HARDWARE

NPROC_PER_NODE="$PRETRAIN_NPROC" HARDWARE="$HARDWARE" \
WORKSPACE=/workspace OUTPUT_DIR="$PRETRAIN_OUTPUT" \
  ./examples/models/nemotron/nemotron_3_5_lightning/pretrain.sh

test "$(<"$PRETRAIN_OUTPUT/latest_checkpointed_iteration.txt")" = 100
test -f "$PRETRAIN_OUTPUT/iter_0000100/run_config.yaml"
```

Pass normal Hydra-style recipe overrides after `pretrain.sh` to select real
indexed data. For example, set `dataset.blend` to the mounted dataset prefix.

## Full SFT

The SFT workflow uses packed OpenMathInstruct-2 at the pinned revision prepared
above. Its input is the imported `iter_0000000` checkpoint.

```bash
export SFT_OUTPUT=/workspace/results/nemotron-3.5-lightning-sft-$HARDWARE
export SFT_CHECKPOINT="$SFT_OUTPUT/iter_0000100"

NPROC_PER_NODE="$SFT_NPROC" HARDWARE="$HARDWARE" \
PRETRAINED_CHECKPOINT="$BASE_MEGATRON_CHECKPOINT" OUTPUT_DIR="$SFT_OUTPUT" \
  ./examples/models/nemotron/nemotron_3_5_lightning/sft.sh \
  "dataset.hf_kwargs={revision:${OPENMATH_REVISION}}"

test "$(<"$SFT_OUTPUT/latest_checkpointed_iteration.txt")" = 100
test -f "$SFT_CHECKPOINT/run_config.yaml"
```

Export the exact iteration-100 full-SFT checkpoint with the TP value selected
for the profile:

```bash
export SFT_HF_EXPORT=/workspace/models/nemotron-3.5-lightning-sft-$HARDWARE-hf

NPROC_PER_NODE="$SFT_NPROC" HF_MODEL="$HF_MODEL" \
MEGATRON_PATH="$SFT_CHECKPOINT" HF_EXPORT_PATH="$SFT_HF_EXPORT" \
TP="$SFT_TP" EP=8 \
  ./examples/models/nemotron/nemotron_3_5_lightning/conversion.sh export

test -f "$SFT_HF_EXPORT/model.safetensors.index.json"
```

Verify both forms of the trained artifact by generating from each:

```bash
NPROC_PER_NODE="$SFT_NPROC" HF_MODEL="$HF_MODEL" \
MEGATRON_PATH="$SFT_CHECKPOINT" TP="$SFT_TP" EP=8 \
  ./examples/models/nemotron/nemotron_3_5_lightning/inference.sh

NPROC_PER_NODE=8 HF_MODEL="$SFT_HF_EXPORT" MEGATRON_PATH= TP=1 EP=8 \
  ./examples/models/nemotron/nemotron_3_5_lightning/inference.sh
```

## LoRA

The LoRA recipe uses rank 32, alpha 32, zero dropout, and targets attention,
MLP, Mamba, expert, shared-expert, and MTP linear layers. Its packed SQuAD
dataset is explicitly pinned by the final override below.

```bash
export LORA_OUTPUT=/workspace/results/nemotron-3.5-lightning-lora-$HARDWARE
export LORA_CHECKPOINT="$LORA_OUTPUT/iter_0000100"

NPROC_PER_NODE="$LORA_NPROC" HARDWARE="$HARDWARE" \
PRETRAINED_CHECKPOINT="$BASE_MEGATRON_CHECKPOINT" OUTPUT_DIR="$LORA_OUTPUT" \
  ./examples/models/nemotron/nemotron_3_5_lightning/lora.sh \
  "dataset.hf_kwargs={revision:${SQUAD_REVISION}}"

test "$(<"$LORA_OUTPUT/latest_checkpointed_iteration.txt")" = 100
test -f "$LORA_CHECKPOINT/run_config.yaml"
```

Export either hardware profile's Megatron adapter to the standard HF PEFT
layout:

```bash
export HF_ADAPTER_PATH=/workspace/models/nemotron-3.5-lightning-lora-$HARDWARE-adapter

NPROC_PER_NODE=8 HF_MODEL="$HF_MODEL" \
LORA_CHECKPOINT="$LORA_CHECKPOINT" HF_ADAPTER_PATH="$HF_ADAPTER_PATH" \
TP=1 EP=8 \
  ./examples/models/nemotron/nemotron_3_5_lightning/adapter.sh export
```

Check the exported adapter schema without loading model weights:

```bash
uv run --active --no-sync python - <<'PY'
import json
import os
from pathlib import Path

from safetensors import safe_open

root = Path(os.environ["HF_ADAPTER_PATH"])
config = json.loads((root / "adapter_config.json").read_text())
if config.get("peft_type") != "LORA":
    raise SystemExit(f"Unexpected PEFT type: {config.get('peft_type')}")
with safe_open(root / "adapter_model.safetensors", framework="pt", device="cpu") as handle:
    keys = list(handle.keys())
if not keys or not any("lora_A" in key for key in keys) or not any("lora_B" in key for key in keys):
    raise SystemExit("Adapter does not contain both LoRA A and B tensors")
print(f"PASS: standard PEFT config and {len(keys)} adapter tensors found")
PY
```

Merge the same Megatron LoRA checkpoint with its imported Megatron base into a
standalone HF checkpoint:

```bash
export HF_MERGED_PATH=/workspace/models/nemotron-3.5-lightning-lora-$HARDWARE-merged

NPROC_PER_NODE=8 HF_MODEL="$HF_MODEL" \
PRETRAINED_CHECKPOINT="$BASE_MEGATRON_CHECKPOINT" \
LORA_CHECKPOINT="$LORA_CHECKPOINT" HF_MERGED_PATH="$HF_MERGED_PATH" \
TP=1 EP=8 \
  ./examples/models/nemotron/nemotron_3_5_lightning/adapter.sh merge

test -f "$HF_MERGED_PATH/model.safetensors.index.json"
```

Verify that the merged standalone artifact loads and generates:

```bash
NPROC_PER_NODE=8 HF_MODEL="$HF_MERGED_PATH" MEGATRON_PATH= TP=1 EP=8 \
  ./examples/models/nemotron/nemotron_3_5_lightning/inference.sh
```

## Enforced offline operation

Complete the immutable input preparation once while network access is
available. The model snapshot, raw datasets, and generated packed datasets then
remain under `/workspace/cache`.

Enable offline enforcement in the same container shell:

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_MODEL="$(<"$HF_MODEL_PATH_FILE")"
```

Confirm that the pinned model and datasets resolve locally before starting an
offline workflow:

```bash
uv run --active --no-sync python - <<'PY'
import os
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["HF_MODEL_ID"],
    revision=os.environ["HF_MODEL_REVISION"],
    cache_dir=os.environ["HF_HUB_CACHE"],
    local_files_only=True,
)
cache_root = Path(os.environ["NEMO_DATASETS_CACHE"])
load_dataset(
    "nvidia/OpenMathInstruct-2",
    split="train_1M",
    revision=os.environ["OPENMATH_REVISION"],
    cache_dir=str(cache_root / "nvidia/OpenMathInstruct-2"),
)
load_dataset(
    "rajpurkar/squad",
    revision=os.environ["SQUAD_REVISION"],
    cache_dir=str(cache_root / "rajpurkar/squad"),
)
print("PASS: all pinned inputs resolved with offline mode enforced")
PY
```

With these variables set, run the import, export, inference, SFT, LoRA,
adapter-export, or merge commands above unchanged. To return to online mode:

```bash
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
```

## Artifact inventory and cleanup

Inspect free space, cache size, outputs, and recorded checkpoint iterations:

```bash
df -h /workspace
du -sh /workspace/cache /workspace/models /workspace/results
find /workspace/models /workspace/results -name latest_checkpointed_iteration.txt \
  -exec sh -c 'printf "%s: " "$1"; cat "$1"' _ {} \;
find /workspace/models /workspace/results -maxdepth 2 -type d -name 'iter_*' -print
```

The host retains the mounted workspace after the container exits. Review the
inventory above, then remove only the task-scoped outputs you no longer need:

```bash
test "$WORKSPACE" = /workspace
case "$HARDWARE" in
  h100|gb200) ;;
  *) echo "HARDWARE must be h100 or gb200" >&2; return 2 ;;
esac
rm -rf -- \
  /workspace/models/nemotron-3.5-lightning-megatron \
  /workspace/models/nemotron-3.5-lightning-hf-export \
  "/workspace/models/nemotron-3.5-lightning-sft-$HARDWARE-hf" \
  "/workspace/models/nemotron-3.5-lightning-lora-$HARDWARE-adapter" \
  "/workspace/models/nemotron-3.5-lightning-lora-$HARDWARE-merged" \
  "/workspace/results/nemotron-3.5-lightning-pretrain-$HARDWARE" \
  "/workspace/results/nemotron-3.5-lightning-sft-$HARDWARE" \
  "/workspace/results/nemotron-3.5-lightning-lora-$HARDWARE"
```

Remove the persistent caches only when no retained workflow needs them:

```bash
test "$WORKSPACE" = /workspace
rm -rf -- /workspace/cache/huggingface /workspace/cache/nemo
```
