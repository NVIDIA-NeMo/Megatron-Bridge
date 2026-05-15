#!/usr/bin/env bash
# Prepare an audio-augmented LLaVA-Pretrain dataset for training with
# examples/models/megatron_mimo/megatron_mimo_training_llava_audio.py --audio-column audio.
#
# Writes a fresh DST tree containing:
#   blip_laion_cc_sbu_558k.json              — records with image paths absolutised to SRC
#   blip_laion_cc_sbu_558k_with_audio.json   — same records + "audio" field (emitted by merge)
#   audio/<prefix>/<id>.flac                 — synthesized 16 kHz FLACs
#   audio_manifest/shard_NNNNN.jsonl         — per-shard manifest ({id, audio, text, ...})
#   shard_logs/shard_NNNNN.log               — per-shard stdout+stderr (multi-GPU runs)
#
# Image paths inside the JSON are absolutised so we do not have to copy or
# symlink the ~107 GB image tree into DST.
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

SRC=${SRC:-/path/to/LLaVA-Pretrain}
DST=${DST:-/path/to/LLaVA-Pretrain-Audio-Augmented}
REPO=${REPO:-/path/to/Megatron-Bridge}
LIMIT=${LIMIT:-}  # empty = use all 558k records
TTS_CACHE=${TTS_CACHE:-$DST/.tts_cache}
TTS_NEMO=${TTS_NEMO:-}
VOCODER_NEMO=${VOCODER_NEMO:-}
TTS_MODEL_NAME=${TTS_MODEL_NAME:-tts_en_fastpitch}
VOCODER_MODEL_NAME=${VOCODER_MODEL_NAME:-tts_en_lj_hifigan_ft_mixertts}

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
NUM_GPUS=${NUM_GPUS:-1}
NUM_SHARDS=${NUM_SHARDS:-$NUM_GPUS}
if (( NUM_SHARDS < 1 )); then NUM_SHARDS=1; fi

mkdir -p "$DST" "$TTS_CACHE"

# 1. Write the working JSON with absolute image paths (so training can still
#    resolve images without copying the 107 GB image tree into DST).  If LIMIT
#    is set, first-N records are used; otherwise all 558k are kept.
SRC="$SRC" DST="$DST" LIMIT="$LIMIT" python - <<'PY'
import json, os
src, dst = os.environ["SRC"], os.environ["DST"]
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

# 3. Synthesize FLACs under DST/audio and per-shard manifests under DST/audio_manifest.
#    One process per GPU, launched in parallel and bound via CUDA_VISIBLE_DEVICES.
LOG_DIR=$DST/shard_logs
mkdir -p "$LOG_DIR"

echo "[prepare] launching $NUM_SHARDS synth shard(s) across $NUM_GPUS GPU(s)"
echo "[prepare] follow with: tail -F $LOG_DIR/shard_*.log"

pids=()
for (( s=0; s<NUM_SHARDS; s++ )); do
    gpu=$(( s % NUM_GPUS ))
    log=$(printf "%s/shard_%05d.log" "$LOG_DIR" "$s")
    echo "[prepare]   shard $s -> GPU $gpu, log=$log"
    CUDA_VISIBLE_DEVICES=$gpu \
      python "$REPO/examples/models/megatron_mimo/synthesize_llava_pretrain_audio.py" \
        --dataset-root "$DST" \
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

# 4. Merge into DST/blip_laion_cc_sbu_558k_with_audio.json (what the test consumes).
python "$REPO/examples/models/megatron_mimo/synthesize_llava_pretrain_audio.py" --mode merge \
    --dataset-root "$DST"
