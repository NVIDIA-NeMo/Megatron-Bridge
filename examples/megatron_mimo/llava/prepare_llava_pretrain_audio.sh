#!/usr/bin/env bash
# Prepare an audio-augmented LLaVA-Pretrain dataset for training with
# examples/megatron_mimo/llava/megatron_mimo_training_llava_audio.py --audio-column audio.
#
# Writes a fresh AUGMENTED_DATASET_DIR tree containing:
#   blip_laion_cc_sbu_558k.json              — records with image paths absolutised to DATASET_ROOT
#   blip_laion_cc_sbu_558k_with_audio.json   — same records + "audio" field (emitted by merge)
#   audio/<prefix>/<id>.flac                 — synthesized 16 kHz FLACs
#   audio_manifest/shard_NNNNN.jsonl         — per-shard manifest ({id, audio, text, ...})
#   shard_logs/shard_NNNNN.log               — per-shard stdout+stderr (multi-GPU runs)
#
# Image paths inside the JSON are absolutised so we do not have to copy or
# symlink the ~107 GB image tree into AUGMENTED_DATASET_DIR.
#
# Run inside the project container (nemo-toolkit[tts] + soundfile + scipy).
#
# NeMo's from_pretrained() path has been flaking with md5 validation errors on
# this cluster, so we resolve the NGC URLs from NeMo's own model registry and
# fetch the .nemo files via curl (cached under $TTS_CACHE).  Set TTS_NEMO /
# VOCODER_NEMO to existing local .nemo files to bypass the download entirely.
#
# Multi-GPU: NUM_SHARDS defaults to the number of visible GPUs, with one shard
# per GPU launched in parallel and bound via CUDA_VISIBLE_DEVICES.  Resume-safe
# — reruns skip existing FLACs, so killing a shard and restarting just fills
# the gaps.  NOTE: --limit is applied *per shard*, so LIMIT=1000 with
# NUM_SHARDS=8 produces 8000 total samples; set NUM_SHARDS=1 for calibration
# runs where you want exactly LIMIT samples.

set -euo pipefail

# Empty DATASET_ROOT triggers an auto-download of LLaVA-Pretrain (captions JSON +
# extracted images) into DATASET_DOWNLOAD_DIR; set DATASET_ROOT to use an existing
# local copy.
DATASET_ROOT=${DATASET_ROOT:-""}
DATASET_DOWNLOAD_DIR=${DATASET_DOWNLOAD_DIR:-/workspace/llava_pretrain}
LLAVA_PRETRAIN_REPO=${LLAVA_PRETRAIN_REPO:-liuhaotian/LLaVA-Pretrain}
AUGMENTED_DATASET_DIR=${AUGMENTED_DATASET_DIR:-/workspace/llava_pretrain_audio_augmented}
# REPO defaults to the repo root inferred from this script's location
# ($REPO/examples/megatron_mimo/llava/prepare_llava_pretrain_audio.sh).
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
LIMIT=${LIMIT:-}  # empty = use all 558k records
TTS_CACHE=${TTS_CACHE:-$AUGMENTED_DATASET_DIR/.tts_cache}
TTS_NEMO=${TTS_NEMO:-}
VOCODER_NEMO=${VOCODER_NEMO:-}
TTS_MODEL_NAME=${TTS_MODEL_NAME:-tts_en_fastpitch}
VOCODER_MODEL_NAME=${VOCODER_MODEL_NAME:-tts_en_lj_hifigan_ft_mixertts}

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
NUM_GPUS=${NUM_GPUS:-1}
NUM_SHARDS=${NUM_SHARDS:-$NUM_GPUS}
if (( NUM_SHARDS < 1 )); then NUM_SHARDS=1; fi

mkdir -p "$AUGMENTED_DATASET_DIR" "$TTS_CACHE"

# -1. Install NeMo speech (TTS) into the container if it isn't importable.
#     The NeMo 26.0x containers are optimized for the LLM/Multimodal domains and
#     ship the source tree at /opt/NeMo WITHOUT the speech (ASR/TTS) collections
#     installed, so `import nemo.collections.tts` fails out of the box
#     (ModuleNotFoundError: nemo / lightning).  The documented fix is to add the
#     speech extra.  We install the shipped source editable with [tts]:
#     `pip install -e /opt/NeMo[tts]` registers nemo AND pulls the TTS deps
#     (lightning, soundfile, scipy, librosa, ...) that NeMo resolves dynamically
#     via nemo_dependencies.py.  The venv has system site-packages, so torch /
#     megatron-core / Transformer Engine already in the image are treated as
#     satisfied and left untouched.  Overrides:
#       NEMO_SRC=/path/to/NeMo        point at an alternate checkout
#       NEMO_EXTRAS=tts               extra(s) to install, e.g. "asr,tts"
#       IGNORE_INSTALLED_PKGS="..."   apt-managed pkgs to reinstall (no RECORD)
#       SKIP_NEMO_BOOTSTRAP=1         disable this step entirely
NEMO_SRC=${NEMO_SRC:-/opt/NeMo}
NEMO_EXTRAS=${NEMO_EXTRAS:-tts}
# NB: do NOT name this PIP_IGNORE_INSTALLED — pip reads PIP_* env vars and would
# turn on --ignore-installed globally (reinstalling torch et al.).
IGNORE_INSTALLED_PKGS=${IGNORE_INSTALLED_PKGS:-PyYAML}
bootstrap_nemo() {
    [[ "${SKIP_NEMO_BOOTSTRAP:-0}" == 1 ]] && return
    if python -c "import nemo.collections.tts" 2>/dev/null; then
        echo "[prepare] nemo.collections.tts already importable; skipping speech install"
        return
    fi
    echo "[prepare] nemo.collections.tts not importable; installing NeMo speech ($NEMO_EXTRAS) from $NEMO_SRC"
    if [[ ! -d "$NEMO_SRC" ]]; then
        echo "[prepare] ERROR: NEMO_SRC=$NEMO_SRC not found; set NEMO_SRC or 'pip install nemo_toolkit[$NEMO_EXTRAS]'" >&2
        exit 1
    fi
    # Some base images install Python packages via apt (e.g. PyYAML) without a
    # RECORD file, so pip can't uninstall them to satisfy NeMo's pins and dies
    # with "uninstall-no-record-file".  Pre-install fresh copies into the venv
    # (they shadow the Debian ones) so the editable install never uninstalls
    # them.  Add to IGNORE_INSTALLED_PKGS if another apt package trips this.
    if [[ -n "$IGNORE_INSTALLED_PKGS" ]]; then
        echo "[prepare]   pip install --ignore-installed $IGNORE_INSTALLED_PKGS"
        pip install --ignore-installed $IGNORE_INSTALLED_PKGS
    fi
    echo "[prepare]   pip install -e $NEMO_SRC[$NEMO_EXTRAS]"
    pip install -e "$NEMO_SRC[$NEMO_EXTRAS]"
    if ! python -c "import nemo.collections.tts" 2>/dev/null; then
        echo "[prepare] ERROR: nemo.collections.tts still not importable after install; full traceback:" >&2
        python -c "import nemo.collections.tts" || true
        exit 1
    fi
    echo "[prepare] nemo.collections.tts ready"
}

bootstrap_nemo

# 0. Ensure the source LLaVA-Pretrain dataset is available. When DATASET_ROOT is
#    empty, download liuhaotian/LLaVA-Pretrain (captions JSON + images.zip) into
#    DATASET_DOWNLOAD_DIR and extract images.zip there, then point DATASET_ROOT at
#    it. The JSON's image paths (e.g. "00453/004531425.jpg") are absolutised
#    against DATASET_ROOT below, so images.zip is extracted directly into it. Both
#    the download and the extraction are skipped if already present.
prepare_dataset() {
    if [[ -n "$DATASET_ROOT" ]]; then
        echo "[prepare] using DATASET_ROOT=$DATASET_ROOT"
        return
    fi

    DATASET_ROOT="$DATASET_DOWNLOAD_DIR"
    local json_file="$DATASET_ROOT/blip_laion_cc_sbu_558k.json"
    local images_zip="$DATASET_ROOT/images.zip"

    # Already prepared: captions JSON + extracted image shards present. This
    # holds even if images.zip was deleted post-extraction.
    if [[ -f "$json_file" && -d "$DATASET_ROOT/00000" ]]; then
        echo "[prepare] using cached LLaVA-Pretrain dataset at $DATASET_ROOT"
        return
    fi

    echo "[prepare] DATASET_ROOT not set; preparing $LLAVA_PRETRAIN_REPO under $DATASET_ROOT"
    mkdir -p "$DATASET_ROOT"

    if [[ -f "$json_file" && -f "$images_zip" ]]; then
        echo "[prepare]   using cached download in $DATASET_ROOT"
    else
        echo "[prepare]   downloading $LLAVA_PRETRAIN_REPO (this can take a while)..."
        LLAVA_PRETRAIN_REPO="$LLAVA_PRETRAIN_REPO" DATASET_ROOT="$DATASET_ROOT" python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id=os.environ["LLAVA_PRETRAIN_REPO"], repo_type="dataset", local_dir=os.environ["DATASET_ROOT"])
PY
    fi

    # images.zip extracts to 5-digit shard dirs (00000, 00001, ...) at the root.
    if [[ -d "$DATASET_ROOT/00000" ]]; then
        echo "[prepare]   images already extracted."
    else
        echo "[prepare]   extracting images.zip..."
        unzip -q -o "$images_zip" -d "$DATASET_ROOT"
    fi

    echo "[prepare]   dataset ready at $DATASET_ROOT"
}

prepare_dataset

# 1. Write the working JSON with absolute image paths (so training can still
#    resolve images without copying the 107 GB image tree into AUGMENTED_DATASET_DIR).  If LIMIT
#    is set, first-N records are used; otherwise all 558k are kept.
DATASET_ROOT="$DATASET_ROOT" AUGMENTED_DATASET_DIR="$AUGMENTED_DATASET_DIR" LIMIT="$LIMIT" python - <<'PY'
import json, os
src, dst = os.environ["DATASET_ROOT"], os.environ["AUGMENTED_DATASET_DIR"]
limit = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
recs = json.load(open(os.path.join(src, "blip_laion_cc_sbu_558k.json")))
if limit is not None:
    recs = recs[:limit]
for r in recs:
    if r.get("image") and not os.path.isabs(r["image"]):
        r["image"] = os.path.join(src, r["image"])
out = os.path.join(dst, "blip_laion_cc_sbu_558k.json")
json.dump(recs, open(out, "w"))
print(f"[prepare] wrote {len(recs)} records -> {out}")
PY

# 2. Resolve local .nemo paths: user override → cached file → fresh curl.
_fetch_nemo() {
    local cache_name=$1 registry_cls=$2 registry_name=$3
    local dest=$TTS_CACHE/$cache_name
    if [[ -s "$dest" ]]; then
        echo "$dest"
        return 0
    fi
    local url
    url=$(TTS_CLS="$registry_cls" TTS_NAME="$registry_name" python - <<'PY'
import os
from nemo.collections.tts import models as tts_models
cls = getattr(tts_models, os.environ["TTS_CLS"])
name = os.environ["TTS_NAME"]
for m in cls.list_available_models() or []:
    if m.pretrained_model_name == name:
        print(m.location)
        break
else:
    raise SystemExit(f"{name} not found in {cls.__name__} registry")
PY
)
    echo "[prepare] downloading $cache_name from $url" >&2
    curl -fL --retry 3 --retry-delay 2 "$url" -o "$dest.tmp"
    mv "$dest.tmp" "$dest"
    echo "$dest"
}

if [[ -z "$TTS_NEMO" ]]; then
    TTS_NEMO=$(_fetch_nemo fastpitch.nemo FastPitchModel "$TTS_MODEL_NAME")
fi
if [[ -z "$VOCODER_NEMO" ]]; then
    VOCODER_NEMO=$(_fetch_nemo hifigan.nemo HifiGanModel "$VOCODER_MODEL_NAME")
fi
echo "[prepare] using TTS_NEMO=$TTS_NEMO"
echo "[prepare] using VOCODER_NEMO=$VOCODER_NEMO"

# 3. Synthesize FLACs under AUGMENTED_DATASET_DIR/audio and per-shard manifests under AUGMENTED_DATASET_DIR/audio_manifest.
#    One process per GPU, launched in parallel and bound via CUDA_VISIBLE_DEVICES.
LOG_DIR=$AUGMENTED_DATASET_DIR/shard_logs
mkdir -p "$LOG_DIR"

echo "[prepare] launching $NUM_SHARDS synth shard(s) across $NUM_GPUS GPU(s)"
echo "[prepare] follow with: tail -F $LOG_DIR/shard_*.log"

pids=()
for (( s=0; s<NUM_SHARDS; s++ )); do
    gpu=$(( s % NUM_GPUS ))
    log=$(printf "%s/shard_%05d.log" "$LOG_DIR" "$s")
    echo "[prepare]   shard $s -> GPU $gpu, log=$log"
    CUDA_VISIBLE_DEVICES=$gpu \
      python "$REPO/examples/megatron_mimo/llava/synthesize_llava_pretrain_audio.py" \
        --dataset-root "$AUGMENTED_DATASET_DIR" \
        --shard-index "$s" --num-shards "$NUM_SHARDS" \
        ${LIMIT:+--limit "$LIMIT"} \
        --tts-model "$TTS_NEMO" --vocoder-model "$VOCODER_NEMO" \
        >"$log" 2>&1 &
    pids+=("$!")
done

fail=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        fail=$((fail + 1))
    fi
done
if (( fail > 0 )); then
    echo "[prepare] ERROR: $fail shard(s) failed; see $LOG_DIR/" >&2
    exit 1
fi
echo "[prepare] all $NUM_SHARDS shard(s) completed successfully"

# 4. Merge into AUGMENTED_DATASET_DIR/blip_laion_cc_sbu_558k_with_audio.json (what the test consumes).
python "$REPO/examples/megatron_mimo/llava/synthesize_llava_pretrain_audio.py" --mode merge \
    --dataset-root "$AUGMENTED_DATASET_DIR"
