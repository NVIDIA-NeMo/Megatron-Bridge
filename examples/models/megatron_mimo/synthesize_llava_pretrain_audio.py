# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Synthesize TTS audio for LLaVA-Pretrain and emit an augmented JSON.

For each record in <dataset_root>/<input_json>, pull the first ``human`` turn
from ``conversations``, synthesize 16 kHz mono speech via NeMo FastPitch +
HiFiGAN, and write:

  <dataset_root>/<output_subdir>/<prefix>/<id>.flac
  <dataset_root>/<manifest_subdir>/shard_{idx:05d}.jsonl

Sharded via --shard-index / --num-shards; resumable (non-empty FLAC files are
kept). After all shards finish, rerun with --mode merge to produce
<dataset_root>/<output_json> (the original JSON plus an ``audio`` field per
record), which test_mimo_training_llava.py consumes via --hf-data-files.

Run inside the project container where nemo-toolkit[tts] is available.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path


logger = logging.getLogger("synthesize_llava_pretrain_audio")

SPECIAL_TOKEN_RE = re.compile(r"<(?:image|audio)>")
WHITESPACE_RE = re.compile(r"\s+")


def extract_human_text(record: dict) -> str | None:
    """Return the first ``human`` turn's text with ``<image>``/``<audio>`` tokens stripped."""
    for turn in record.get("conversations", []):
        if turn.get("from") != "human":
            continue
        cleaned = SPECIAL_TOKEN_RE.sub(" ", turn.get("value", ""))
        cleaned = WHITESPACE_RE.sub(" ", cleaned).strip()
        return cleaned or None
    return None


def audio_rel_path(record: dict, output_subdir: str) -> str:
    """Return the ``<output_subdir>/<prefix>/<id>.flac`` path relative to ``dataset_root``."""
    # Keep only the last directory component of the image path so that an
    # absolutised image path (e.g. /abs/.../00453/foo.jpg) still yields a
    # relative audio path inside <dataset_root>/<output_subdir>/00453/.
    rec_id = record["id"]
    image = record.get("image") or ""
    prefix = os.path.basename(os.path.dirname(image)) or rec_id[:5]
    return os.path.join(output_subdir, prefix, f"{rec_id}.flac")


def run_synth(args: argparse.Namespace) -> None:  # pragma: no cover
    """Synthesize FLAC audio for this shard's records and write a JSONL manifest."""
    # Defer heavy imports so --mode merge (and --help) do not require the
    # audio/TTS stack, which only lives inside the project container.
    import numpy as np
    import soundfile as sf
    import torch
    from nemo.collections.tts.models import FastPitchModel, HifiGanModel
    from scipy.signal import resample as scipy_resample

    def resample_to(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        if src_sr == dst_sr:
            return audio.astype(np.float32, copy=False)
        num = int(round(len(audio) * dst_sr / src_sr))
        return scipy_resample(audio, num).astype(np.float32)

    dataset_root = Path(args.dataset_root)
    input_path = dataset_root / args.input_json
    manifest_root = dataset_root / args.manifest_subdir
    manifest_root.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s", input_path)
    with open(input_path) as f:
        records = json.load(f)

    shard = records[args.shard_index :: args.num_shards]
    if args.limit is not None:
        shard = shard[: args.limit]
    logger.info(
        "Shard %d/%d → %d records of %d total%s",
        args.shard_index,
        args.num_shards,
        len(shard),
        len(records),
        f", limited to {args.limit}" if args.limit else "",
    )

    def _load(model_cls, name_or_path: str, label: str):
        if os.path.isfile(name_or_path):
            logger.info("Loading %s from local file: %s", label, name_or_path)
            return model_cls.restore_from(name_or_path)
        logger.info("Loading %s from NGC: %s", label, name_or_path)
        return model_cls.from_pretrained(name_or_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec_gen = _load(FastPitchModel, args.tts_model, "FastPitch").to(device).eval()
    vocoder = _load(HifiGanModel, args.vocoder_model, "HiFiGAN").to(device).eval()

    src_sr = int(getattr(spec_gen._cfg, "sample_rate", 22050))
    max_samples = int(args.max_duration_s * args.target_sample_rate)

    manifest_path = manifest_root / f"shard_{args.shard_index:05d}.jsonl"
    n_written = n_skipped = n_failed = 0

    with open(manifest_path, "w") as mf, torch.inference_mode():
        for i, record in enumerate(shard):
            rec_id = record.get("id")
            if not rec_id:
                n_failed += 1
                continue
            rel_audio = audio_rel_path(record, args.output_subdir)
            abs_audio = dataset_root / rel_audio

            if abs_audio.exists() and abs_audio.stat().st_size > 0:
                mf.write(json.dumps({"id": rec_id, "audio": rel_audio}) + "\n")
                n_skipped += 1
                continue

            text = extract_human_text(record)
            if not text:
                n_failed += 1
                continue

            try:
                tokens = spec_gen.parse(text).to(device)
                spec = spec_gen.generate_spectrogram(tokens=tokens)
                audio = vocoder.convert_spectrogram_to_audio(spec=spec)
            except Exception as exc:
                logger.warning("synth failed id=%s text=%r: %s", rec_id, text, exc)
                n_failed += 1
                continue

            audio_np = audio.squeeze().detach().cpu().numpy().astype(np.float32)
            audio_np = resample_to(audio_np, src_sr, args.target_sample_rate)
            if len(audio_np) > max_samples:
                audio_np = audio_np[:max_samples]

            abs_audio.parent.mkdir(parents=True, exist_ok=True)
            sf.write(
                str(abs_audio),
                audio_np,
                args.target_sample_rate,
                format="FLAC",
                subtype="PCM_16",
            )
            mf.write(
                json.dumps(
                    {
                        "id": rec_id,
                        "audio": rel_audio,
                        "text": text,
                        "num_samples": int(len(audio_np)),
                        "duration_s": float(len(audio_np) / args.target_sample_rate),
                    }
                )
                + "\n"
            )
            n_written += 1

            if (i + 1) % args.log_every == 0:
                logger.info(
                    "Shard %d progress: %d/%d (written=%d skipped=%d failed=%d)",
                    args.shard_index,
                    i + 1,
                    len(shard),
                    n_written,
                    n_skipped,
                    n_failed,
                )
            mf.flush()

    logger.info(
        "Shard %d done: written=%d skipped=%d failed=%d manifest=%s",
        args.shard_index,
        n_written,
        n_skipped,
        n_failed,
        manifest_path,
    )


def run_merge(args: argparse.Namespace) -> None:
    """Combine per-shard manifests into a single JSON augmented with an ``audio`` field."""
    dataset_root = Path(args.dataset_root)
    input_path = dataset_root / args.input_json
    output_path = dataset_root / args.output_json
    manifest_root = dataset_root / args.manifest_subdir

    logger.info("Loading %s", input_path)
    with open(input_path) as f:
        records = json.load(f)

    id_to_audio: dict[str, str] = {}
    shard_files = sorted(manifest_root.glob("shard_*.jsonl"))
    for shard_file in shard_files:
        with open(shard_file) as f:
            for line in f:
                entry = json.loads(line)
                id_to_audio[entry["id"]] = entry["audio"]
    logger.info(
        "Collected %d audio entries from %d shard manifest(s)",
        len(id_to_audio),
        len(shard_files),
    )

    augmented: list[dict] = []
    dropped = 0
    for record in records:
        rec_id = record.get("id")
        if rec_id and rec_id in id_to_audio:
            record = dict(record)
            record["audio"] = id_to_audio[rec_id]
            augmented.append(record)
        else:
            dropped += 1

    logger.info(
        "Writing %s: kept=%d dropped=%d (of %d)",
        output_path,
        len(augmented),
        dropped,
        len(records),
    )
    with open(output_path, "w") as f:
        json.dump(augmented, f)


def parse_args() -> argparse.Namespace:  # pragma: no cover
    """Parse command-line arguments for the synth/merge entry points."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", choices=["synth", "merge"], default="synth")
    p.add_argument("--dataset-root", required=True, help="LLaVA-Pretrain root directory")
    p.add_argument("--input-json", default="blip_laion_cc_sbu_558k.json")
    p.add_argument("--output-json", default="blip_laion_cc_sbu_558k_with_audio.json")
    p.add_argument("--output-subdir", default="audio", help="Relative FLAC output dir under dataset-root")
    p.add_argument("--manifest-subdir", default="audio_manifest")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=None, help="Cap records per shard (for smoke tests)")
    p.add_argument("--tts-model", default="tts_en_fastpitch")
    p.add_argument("--vocoder-model", default="tts_en_lj_hifigan_ft_mixertts")
    p.add_argument("--target-sample-rate", type=int, default=16000)
    p.add_argument("--max-duration-s", type=float, default=30.0, help="Whisper hard cap")
    p.add_argument("--log-every", type=int, default=50)
    return p.parse_args()


def main() -> None:
    """Configure logging and dispatch to ``run_synth`` or ``run_merge`` based on ``--mode``."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s: %(message)s",
    )
    args = parse_args()
    if args.mode == "synth":
        run_synth(args)
    else:
        run_merge(args)


if __name__ == "__main__":
    main()
