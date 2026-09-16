from __future__ import annotations

from pathlib import Path

import numpy as np

from .audio_io import read_audio


FEATURE_NAMES = [
    "duration",
    "rms_mean",
    "rms_std",
    "rms_max",
    "rms_p90",
    "zcr_mean",
    "zcr_std",
    "zcr_max",
    "centroid_mean",
    "centroid_std",
    "centroid_max",
    "rolloff_mean",
    "rolloff_std",
    "flatness_mean",
    "flatness_std",
    "low_band_ratio",
    "mid_band_ratio",
    "high_band_ratio",
    "peak",
]


def extract_features(path: str | Path) -> np.ndarray:
    audio = read_audio(path)
    return extract_features_from_audio(audio.samples, audio.sample_rate)


def extract_features_from_audio(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    duration = samples.size / sample_rate

    if samples.size < 512:
        samples = np.pad(samples, (0, 512 - samples.size))

    frame_size = int(sample_rate * 0.032)
    hop_size = int(sample_rate * 0.010)
    frames = _frames(samples, frame_size, hop_size)
    window = np.hanning(frame_size).astype(np.float32)
    windowed = frames * window

    eps = 1e-9
    rms = np.sqrt(np.mean(np.square(frames), axis=1) + eps)
    zcr = np.mean(np.signbit(frames[:, 1:]) != np.signbit(frames[:, :-1]), axis=1)

    spectrum = np.abs(np.fft.rfft(windowed, axis=1)) + eps
    power = spectrum * spectrum
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)

    energy = np.sum(power, axis=1) + eps
    centroid = np.sum(power * freqs[None, :], axis=1) / energy

    cumulative = np.cumsum(power, axis=1)
    rolloff_idx = np.argmax(cumulative >= (0.85 * energy[:, None]), axis=1)
    rolloff = freqs[rolloff_idx]

    geometric = np.exp(np.mean(np.log(spectrum), axis=1))
    arithmetic = np.mean(spectrum, axis=1) + eps
    flatness = geometric / arithmetic

    band_energy = _band_ratios(power, freqs)

    return np.asarray(
        [
            duration,
            float(np.mean(rms)),
            float(np.std(rms)),
            float(np.max(rms)),
            float(np.percentile(rms, 90)),
            float(np.mean(zcr)),
            float(np.std(zcr)),
            float(np.max(zcr)),
            float(np.mean(centroid)),
            float(np.std(centroid)),
            float(np.max(centroid)),
            float(np.mean(rolloff)),
            float(np.std(rolloff)),
            float(np.mean(flatness)),
            float(np.std(flatness)),
            band_energy[0],
            band_energy[1],
            band_energy[2],
            float(np.max(np.abs(samples))),
        ],
        dtype=np.float32,
    )


def _frames(samples: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if samples.size <= frame_size:
        return np.expand_dims(np.pad(samples, (0, frame_size - samples.size)), axis=0)

    starts = np.arange(0, samples.size - frame_size + 1, hop_size)
    return np.stack([samples[start : start + frame_size] for start in starts])


def _band_ratios(power: np.ndarray, freqs: np.ndarray) -> tuple[float, float, float]:
    eps = 1e-9
    total = float(np.sum(power) + eps)
    low = float(np.sum(power[:, freqs < 500.0]) / total)
    mid = float(np.sum(power[:, (freqs >= 500.0) & (freqs < 2500.0)]) / total)
    high = float(np.sum(power[:, freqs >= 2500.0]) / total)
    return low, mid, high
