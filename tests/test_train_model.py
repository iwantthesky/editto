from __future__ import annotations

import unittest
import math

import numpy as np

from breath_cleaner.train_model import _group_stratified_split, _recommended_threshold
from breath_cleaner.labeler import _merge_detection_payloads
from breath_cleaner.detect import DetectionConfig, detect_breaths


class SourceGroupedSplitTests(unittest.TestCase):
    def test_sources_never_overlap(self) -> None:
        rows = []
        labels = []
        for source_index in range(12):
            for label in (0.0, 1.0):
                rows.append({"source": f"recording-{source_index}.wav", "clip": f"{source_index}-{label}.wav"})
                labels.append(label)
        y = np.asarray(labels, dtype=np.float32)

        train_idx, validation_idx = _group_stratified_split(y, rows, seed=42)

        train_sources = {rows[index]["source"] for index in train_idx}
        validation_sources = {rows[index]["source"] for index in validation_idx}
        self.assertFalse(train_sources & validation_sources)
        self.assertEqual({0.0, 1.0}, set(y[train_idx]))
        self.assertEqual({0.0, 1.0}, set(y[validation_idx]))

    def test_threshold_recommendation_favors_speech_safety(self) -> None:
        y = np.asarray([1, 1, 0, 0], dtype=np.float32)
        probabilities = np.asarray([0.95, 0.80, 0.74, 0.10], dtype=np.float32)
        self.assertGreaterEqual(_recommended_threshold(y, probabilities), 0.75)

    def test_duplicate_detection_regions_are_merged(self) -> None:
        merged = _merge_detection_payloads([
            {"start": 1.0, "end": 1.2, "model_probability": 0.8, "label": "breath"},
            {"start": 1.0, "end": 1.3, "model_probability": 0.9, "label": "breath"},
            {"start": 2.0, "end": 2.2, "model_probability": 0.7, "label": "breath"},
        ])
        self.assertEqual(2, len(merged))
        self.assertEqual(1.3, merged[0]["end"])
        self.assertEqual(0.9, merged[0]["model_probability"])

    def test_quiet_breath_is_not_lost_in_speech_dominated_audio(self) -> None:
        sample_rate = 16_000
        time = np.arange(sample_rate * 3, dtype=np.float32) / sample_rate
        audio = 0.18 * np.sin(2 * math.pi * 180 * time)
        start = int(1.2 * sample_rate)
        end = int(1.55 * sample_rate)
        noise = np.random.default_rng(42).normal(0.0, 0.035, end - start).astype(np.float32)
        audio[start:end] = noise - np.roll(noise, 1)

        segments = detect_breaths(audio, sample_rate, DetectionConfig(min_score=0.45))

        self.assertTrue(any(segment.start < 1.35 < segment.end for segment in segments))


if __name__ == "__main__":
    unittest.main()
