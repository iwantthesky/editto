from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from breath_cleaner.audio_io import WavAudio, read_audio, write_wav  # noqa: E402
from breath_cleaner.detect import BreathSegment, DetectionConfig, detect_breaths  # noqa: E402


FIELDNAMES = ["clip", "source", "start", "end", "score", "label", "notes"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Import a local audio file and append detected candidates as unknown.")
    parser.add_argument("input", help="Input audio file.")
    parser.add_argument("--labels", default="dataset/candidates/labels.csv")
    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument("--context-ms", type=float, default=120.0)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    labels_path = Path(args.labels)
    raw_dir = labels_path.parent.parent / "raw"
    clips_dir = labels_path.parent / "clips"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    source_path = _unique_path(raw_dir / _safe_filename(input_path.name))
    shutil.copyfile(input_path, source_path)

    audio = read_audio(source_path)
    segments = detect_breaths(audio.samples, audio.sample_rate, DetectionConfig(min_score=args.min_score))
    rows = _read_rows(labels_path)
    existing_clips = {Path(row.get("clip", "")).as_posix() for row in rows}

    added = 0
    for segment in segments:
        clip_path = _auto_clip_path(clips_dir, source_path)
        if clip_path.as_posix() in existing_clips:
            continue
        clip_audio = _extract_clip(audio, segment, context_ms=args.context_ms)
        write_wav(clip_path, clip_audio)
        rows.append(
            {
                "clip": str(clip_path),
                "source": str(source_path),
                "start": f"{segment.start:.3f}",
                "end": f"{segment.end:.3f}",
                "score": f"{segment.score:.6f}",
                "label": "unknown",
                "notes": "",
            }
        )
        existing_clips.add(clip_path.as_posix())
        added += 1

    _write_rows(labels_path, rows)
    print(f"source {source_path}")
    print(f"candidates {added}")
    return 0


def _extract_clip(audio: WavAudio, segment: BreathSegment, context_ms: float) -> WavAudio:
    context_samples = int(audio.sample_rate * context_ms / 1000.0)
    start = max(0, int(segment.start * audio.sample_rate) - context_samples)
    end = min(audio.samples.size, int(segment.end * audio.sample_rate) + context_samples)
    return WavAudio(sample_rate=audio.sample_rate, samples=np.asarray(audio.samples[start:end], dtype=np.float32))


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _safe_filename(filename: str) -> str:
    path = Path(filename)
    stem = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in path.stem.strip())
    return f"{stem or 'audio'}{path.suffix.lower() or '.audio'}"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    index = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def _auto_clip_path(clips_dir: Path, source_path: Path) -> Path:
    stem = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in source_path.stem.strip())
    next_index = len(list(clips_dir.glob(f"{stem}_candidate_*.wav"))) + 1
    while True:
        candidate = clips_dir / f"{stem}_candidate_{next_index:04d}.wav"
        if not candidate.exists():
            return candidate
        next_index += 1


if __name__ == "__main__":
    raise SystemExit(main())
