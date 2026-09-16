from __future__ import annotations

import numpy as np

from .detect import BreathSegment


def duck_segments(
    samples: np.ndarray,
    sample_rate: int,
    segments: list[BreathSegment],
    gain_db: float = -18.0,
    fade_ms: float = 25.0,
    low_confidence_gain_db: float | None = None,
    confidence_floor: float = 0.75,
) -> np.ndarray:
    output = np.asarray(samples, dtype=np.float32).copy()
    fade_samples = max(1, int(sample_rate * fade_ms / 1000.0))

    for segment in segments:
        start = max(0, int(segment.start * sample_rate))
        end = min(output.shape[0], int(segment.end * sample_rate))
        if end <= start:
            continue

        selected_gain_db = gain_db
        if low_confidence_gain_db is not None:
            confidence = np.clip((float(segment.score) - confidence_floor) / max(1e-6, 1.0 - confidence_floor), 0.0, 1.0)
            selected_gain_db = low_confidence_gain_db + confidence * (gain_db - low_confidence_gain_db)
        gain = float(10.0 ** (selected_gain_db / 20.0))
        envelope = np.full(end - start, gain, dtype=np.float32)
        fade_len = min(fade_samples, envelope.size // 2)
        if fade_len > 0:
            phase = np.linspace(0.0, np.pi, fade_len, dtype=np.float32)
            fade_in = gain + (1.0 - gain) * (1.0 + np.cos(phase)) / 2.0
            envelope[:fade_len] = fade_in
            envelope[-fade_len:] = fade_in[::-1]

        if output.ndim == 2:
            envelope = envelope[:, None]
        output[start:end] *= envelope

    return output
