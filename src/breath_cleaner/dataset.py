from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .audio_io import WavAudio, read_audio, write_wav
from .detect import BreathSegment, DetectionConfig, detect_breaths


def main() -> int:
    parser = argparse.ArgumentParser(description="Create candidate clips for building a personal breath dataset.")
    parser.add_argument("input", help="Input local audio file.")
    parser.add_argument("--out-dir", default="dataset/candidates", help="Directory for candidate clips and labels.csv.")
    parser.add_argument("--context-ms", type=float, default=120.0, help="Extra audio around each detected candidate.")
    parser.add_argument("--min-score", type=float, default=0.55, help="Rules detector threshold.")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    audio = read_audio(input_path)
    segments = detect_breaths(audio.samples, audio.sample_rate, DetectionConfig(min_score=args.min_score))

    rows = []
    for index, segment in enumerate(segments, start=1):
        clip_name = f"{input_path.stem}_candidate_{index:04d}.wav"
        clip_path = clips_dir / clip_name
        clip_audio = _extract_clip(audio, segment, args.context_ms)
        write_wav(clip_path, clip_audio)
        rows.append(
            {
                "clip": str(clip_path),
                "source": str(input_path),
                "start": f"{segment.start:.3f}",
                "end": f"{segment.end:.3f}",
                "score": f"{segment.score:.6f}",
                "label": "unknown",
                "notes": "",
            }
        )

    labels_path = out_dir / "labels.csv"
    _write_labels(labels_path, rows)

    manifest = {
        "input": str(input_path),
        "sample_rate": audio.sample_rate,
        "candidate_count": len(rows),
        "labels_csv": str(labels_path),
        "clips_dir": str(clips_dir),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    return 0


def _extract_clip(audio: WavAudio, segment: BreathSegment, context_ms: float) -> WavAudio:
    context_samples = int(audio.sample_rate * context_ms / 1000.0)
    start = max(0, int(segment.start * audio.sample_rate) - context_samples)
    end = min(audio.samples.size, int(segment.end * audio.sample_rate) + context_samples)
    return WavAudio(sample_rate=audio.sample_rate, samples=np.asarray(audio.samples[start:end], dtype=np.float32))


def _write_labels(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["clip", "source", "start", "end", "score", "label", "notes"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
