from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audio_io import WavAudio, read_audio, write_wav
from .detect import DetectionConfig, detect_breaths
from .process import duck_segments


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect and reduce breath-like sounds in a local audio file.")
    parser.add_argument("input", help="Input audio file. WAV is native; other formats require ffmpeg.")
    parser.add_argument("-o", "--output", help="Output WAV path. Defaults to <input>.ducked.wav.")
    parser.add_argument("--segments-json", help="Write detected breath segments to this JSON file.")
    parser.add_argument("--gain-db", type=float, default=-18.0, help="Gain applied to detected segments.")
    parser.add_argument("--fade-ms", type=float, default=25.0, help="Fade in/out around detected segments.")
    parser.add_argument("--min-score", type=float, default=0.55, help="Detection confidence threshold.")
    parser.add_argument("--dry-run", action="store_true", help="Only detect and print segments; do not write audio.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".ducked.wav")

    audio = read_audio(input_path)
    config = DetectionConfig(min_score=args.min_score)
    segments = detect_breaths(audio.samples, audio.sample_rate, config)

    payload = {
        "input": str(input_path),
        "sample_rate": audio.sample_rate,
        "backend": "rules",
        "segments": [segment.to_dict() for segment in segments],
    }

    print(json.dumps(payload, indent=2))

    if args.segments_json:
        json_path = Path(args.segments_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not args.dry_run:
        processed = duck_segments(audio.samples, audio.sample_rate, segments, args.gain_db, args.fade_ms)
        write_wav(output_path, WavAudio(sample_rate=audio.sample_rate, samples=processed))
        print(f"Wrote {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
