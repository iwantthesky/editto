from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from breath_cleaner.audio_io import WavAudio, write_wav
from breath_cleaner.detect import DetectionConfig, detect_breaths
from breath_cleaner.process import duck_segments


def main() -> int:
    sample_rate = 16_000
    duration = 3.0
    t = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate

    speech_like = 0.18 * np.sin(2 * math.pi * 180 * t)
    audio = speech_like.copy()

    breath_start = int(1.2 * sample_rate)
    breath_end = int(1.55 * sample_rate)
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 0.035, breath_end - breath_start).astype(np.float32)
    highpassish = noise - np.roll(noise, 1)
    audio[breath_start:breath_end] = highpassish

    config = DetectionConfig(min_score=0.45)
    segments = detect_breaths(audio, sample_rate, config)
    assert segments, "Expected at least one synthetic breath segment"

    processed = duck_segments(audio, sample_rate, segments)
    out_dir = Path("tmp")
    write_wav(out_dir / "smoke_input.wav", WavAudio(sample_rate, audio))
    write_wav(out_dir / "smoke_output.wav", WavAudio(sample_rate, processed))

    print(f"Detected {len(segments)} segment(s)")
    for segment in segments:
        print(segment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
