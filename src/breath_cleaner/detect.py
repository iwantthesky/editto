from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class BreathSegment:
    start: float
    end: float
    score: float
    rms_db: float
    zcr: float
    centroid_hz: float
    rolloff_hz: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class DetectionConfig:
    frame_ms: float = 32.0
    hop_ms: float = 10.0
    min_duration_ms: float = 120.0
    max_duration_ms: float = 950.0
    merge_gap_ms: float = 80.0
    low_energy_db: float = -52.0  # Default, will be overridden by adaptive if possible
    max_adaptive_low_energy_db: float = -35.0
    high_energy_db: float = -16.0
    min_zcr: float = 0.045
    min_centroid_hz: float = 850.0
    max_rolloff_hz: float = 8000.0 # Breaths usually don't have high energy above 8k
    min_score: float = 0.55


def detect_breaths(samples: np.ndarray, sample_rate: int, config: DetectionConfig) -> list[BreathSegment]:      
    mono = np.asarray(samples, dtype=np.float32)
    frame_size = max(1, int(sample_rate * config.frame_ms / 1000.0))
    hop_size = max(1, int(sample_rate * config.hop_ms / 1000.0))

    if mono.size < frame_size:
        return []

    # Adaptive Noise Floor calculation
    # Sample every 100ms to get a representative distribution of noise
    noise_sample_hop = int(sample_rate * 0.1)
    noise_rms_list = []
    for start in range(0, mono.size - frame_size + 1, noise_sample_hop):
        frame = mono[start : start + frame_size]
        rms = np.sqrt(np.mean(np.square(frame)) + 1e-9)
        noise_rms_list.append(20.0 * np.log10(rms))
    
    if noise_rms_list:
        # Use the 10th percentile as the noise floor
        noise_floor_db = float(np.percentile(noise_rms_list, 10))
        # A recording dominated by speech can have an unrealistically high
        # percentile floor. Cap it so quieter breaths are not rejected first.
        adaptive_low_energy = min(
            config.max_adaptive_low_energy_db,
            max(config.low_energy_db, noise_floor_db + 6.0),
        )
    else:
        adaptive_low_energy = config.low_energy_db

    candidates: list[BreathSegment] = []
    active_start: int | None = None
    active_scores: list[tuple[float, float, float, float, float]] = []

    for start in range(0, mono.size - frame_size + 1, hop_size):
        frame = mono[start : start + frame_size]
        features = _frame_features(frame, sample_rate)
        
        # Override config temporarily with adaptive value for scoring
        score = _breath_score(features, config, adaptive_low_energy)

        if score >= config.min_score:
            if active_start is None:
                active_start = start
            active_scores.append((score, features["rms_db"], features["zcr"], features["centroid_hz"], features["rolloff_hz"]))
        elif active_start is not None:
            segment = _make_segment(active_start, start + frame_size, sample_rate, active_scores)
            if _duration_ok(segment, config):
                candidates.append(segment)
            active_start = None
            active_scores = []

    if active_start is not None:
        segment = _make_segment(active_start, mono.size, sample_rate, active_scores)
        if _duration_ok(segment, config):
            candidates.append(segment)

    return _merge_segments(candidates, config)


def _frame_features(frame: np.ndarray, sample_rate: int) -> dict[str, float]:
    eps = 1e-9
    rms = float(np.sqrt(np.mean(np.square(frame)) + eps))
    rms_db = float(20.0 * np.log10(rms + eps))

    signs = np.signbit(frame)
    zcr = float(np.mean(signs[1:] != signs[:-1])) if frame.size > 1 else 0.0

    windowed = frame * np.hanning(frame.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(frame.size, d=1.0 / sample_rate)
    
    sum_spec = np.sum(spectrum) + eps
    centroid = float(np.sum(freqs * spectrum) / sum_spec)
    
    # Spectral Rolloff (85% energy)
    cumulative_energy = np.cumsum(spectrum)
    rolloff_idx = np.searchsorted(cumulative_energy, 0.85 * cumulative_energy[-1])
    rolloff = float(freqs[rolloff_idx])

    return {"rms_db": rms_db, "zcr": zcr, "centroid_hz": centroid, "rolloff_hz": rolloff}


def _breath_score(features: dict[str, float], config: DetectionConfig, adaptive_low_energy: float) -> float:
    rms_db = features["rms_db"]
    zcr = features["zcr"]
    centroid = features["centroid_hz"]
    rolloff = features["rolloff_hz"]

    if rms_db < adaptive_low_energy or rms_db > config.high_energy_db:
        return 0.0

    energy_score = 1.0 - abs(rms_db - (-34.0)) / 28.0
    zcr_score = (zcr - config.min_zcr) / 0.12
    centroid_score = (centroid - config.min_centroid_hz) / 2500.0
    
    # Penalty for very high roll-off (sibilants like 's')
    rolloff_penalty = max(0, (rolloff - 6000.0) / 4000.0)
    
    score = 0.30 * energy_score + 0.30 * zcr_score + 0.30 * centroid_score - 0.10 * rolloff_penalty
    return float(np.clip(score, 0.0, 1.0))


def _make_segment(
    start_sample: int,
    end_sample: int,
    sample_rate: int,
    scores: list[tuple[float, float, float, float, float]],
) -> BreathSegment:
    arr = np.asarray(scores, dtype=np.float32)
    means = arr.mean(axis=0) if arr.size else np.zeros(5, dtype=np.float32)
    return BreathSegment(
        start=start_sample / sample_rate,
        end=end_sample / sample_rate,
        score=float(means[0]),
        rms_db=float(means[1]),
        zcr=float(means[2]),
        centroid_hz=float(means[3]),
        rolloff_hz=float(means[4]),
    )


def _duration_ok(segment: BreathSegment, config: DetectionConfig) -> bool:
    duration_ms = (segment.end - segment.start) * 1000.0
    return config.min_duration_ms <= duration_ms <= config.max_duration_ms


def _merge_segments(segments: list[BreathSegment], config: DetectionConfig) -> list[BreathSegment]:
    if not segments:
        return []

    merged: list[BreathSegment] = [segments[0]]
    max_gap = config.merge_gap_ms / 1000.0

    for segment in segments[1:]:
        previous = merged[-1]
        if segment.start - previous.end <= max_gap:
            merged[-1] = BreathSegment(
                start=previous.start,
                end=segment.end,
                score=max(previous.score, segment.score),
                rms_db=(previous.rms_db + segment.rms_db) / 2.0,
                zcr=(previous.zcr + segment.zcr) / 2.0,
                centroid_hz=(previous.centroid_hz + segment.centroid_hz) / 2.0,
                rolloff_hz=(previous.rolloff_hz + segment.rolloff_hz) / 2.0,
            )
        else:
            merged.append(segment)

    return [segment for segment in merged if _duration_ok(segment, config)]
