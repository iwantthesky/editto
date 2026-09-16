from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES, extract_features


POSITIVE_LABEL = "breath"
NEGATIVE_LABELS = {"speech", "noise", "silence"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the first local breath classifier.")
    parser.add_argument("--labels", default="dataset/candidates/labels.csv", help="Path to labels.csv.")
    parser.add_argument("--models-dir", default="models", help="Directory where model files are saved.")
    parser.add_argument("--epochs", type=int, default=1200, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=0.08, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    args = parser.parse_args()

    rows = _load_rows(Path(args.labels))
    x, y, used_rows = _load_dataset(rows)
    if len(np.unique(y)) < 2:
        raise ValueError("Need both breath and non-breath labels before training.")

    train_idx, val_idx = _group_stratified_split(y, used_rows, seed=args.seed)
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-6] = 1.0
    x_train_s = (x_train - mean) / std
    x_val_s = (x_val - mean) / std

    weights, bias, history = _train_logistic_regression(
        x_train_s,
        y_train,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )

    train_metrics = _metrics(y_train, _predict_proba(x_train_s, weights, bias))
    val_metrics = _metrics(y_val, _predict_proba(x_val_s, weights, bias))
    recommended_threshold = _recommended_threshold(y_val, _predict_proba(x_val_s, weights, bias))

    train_sources = {used_rows[index].get("source", "") for index in train_idx}
    validation_sources = {used_rows[index].get("source", "") for index in val_idx}

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    version = datetime.now().strftime("v%Y%m%d_%H%M%S")
    model_path = models_dir / f"breath_logreg_{version}.json"
    latest_path = models_dir / "breath_logreg_latest.json"

    payload = {
        "model_type": "numpy_logistic_regression",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "positive_label": POSITIVE_LABEL,
        "negative_labels": sorted(NEGATIVE_LABELS),
        "feature_names": FEATURE_NAMES,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": weights.tolist(),
        "bias": float(bias),
        "train_count": int(len(train_idx)),
        "validation_count": int(len(val_idx)),
        "split_strategy": "source_grouped_stratified",
        "train_source_count": len(train_sources),
        "validation_source_count": len(validation_sources),
        "source_overlap_count": len(train_sources & validation_sources),
        "recommended_threshold": recommended_threshold,
        "label_counts": _label_counts(used_rows),
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "loss_start": float(history[0]),
        "loss_end": float(history[-1]),
    }

    model_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps({"model": str(model_path), "latest": str(latest_path), **payload}, indent=2))
    return 0


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_dataset(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    features = []
    labels = []
    used_rows = []
    for row in rows:
        label = row.get("label", "unknown")
        if label == POSITIVE_LABEL:
            target = 1.0
        elif label in NEGATIVE_LABELS:
            target = 0.0
        else:
            continue

        clip = Path(row["clip"])
        if not clip.exists():
            continue
        features.append(extract_features(clip))
        labels.append(target)
        used_rows.append(row)

    return np.stack(features).astype(np.float32), np.asarray(labels, dtype=np.float32), used_rows


def _stratified_split(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    for label in [0.0, 1.0]:
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        val_count = max(1, int(round(len(idx) * 0.2)))
        val_parts.append(idx[:val_count])
        train_parts.append(idx[val_count:])

    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def _group_stratified_split(
    y: np.ndarray,
    rows: list[dict[str, str]],
    seed: int,
    validation_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Split by source so clips from one recording never leak across sets."""
    if len(y) != len(rows):
        raise ValueError("Labels and rows must have the same length.")

    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        key = row.get("source", "").strip() or row.get("clip", "").strip() or f"row:{index}"
        groups.setdefault(key, []).append(index)

    if len(groups) < 2:
        return _stratified_split(y, seed)

    rng = np.random.default_rng(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    label_totals = np.asarray([np.sum(y == 0.0), np.sum(y == 1.0)], dtype=np.int64)
    targets = np.maximum(1, np.rint(label_totals * validation_fraction).astype(np.int64))
    validation_counts = np.zeros(2, dtype=np.int64)
    validation_groups: set[str] = set()

    for key, indices in group_items:
        counts = np.asarray([
            np.sum(y[indices] == 0.0),
            np.sum(y[indices] == 1.0),
        ], dtype=np.int64)
        if np.any(label_totals - (validation_counts + counts) < 1):
            continue
        current_error = np.sum(np.abs(targets - validation_counts))
        next_error = np.sum(np.abs(targets - (validation_counts + counts)))
        if next_error < current_error:
            validation_groups.add(key)
            validation_counts += counts

    for label in (0, 1):
        if validation_counts[label] > 0:
            continue
        candidates = [
            (key, indices)
            for key, indices in group_items
            if key not in validation_groups and np.any(y[indices] == float(label))
        ]
        if not candidates:
            raise ValueError("Could not create a source-grouped validation split for both labels.")
        key, indices = min(candidates, key=lambda item: len(item[1]))
        validation_groups.add(key)
        validation_counts += np.asarray([
            np.sum(y[indices] == 0.0),
            np.sum(y[indices] == 1.0),
        ], dtype=np.int64)

    val_idx = np.asarray(
        [index for key, indices in group_items if key in validation_groups for index in indices],
        dtype=np.int64,
    )
    train_idx = np.asarray(
        [index for key, indices in group_items if key not in validation_groups for index in indices],
        dtype=np.int64,
    )
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Could not create non-empty source-grouped train and validation sets.")
    if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[val_idx])) < 2:
        raise ValueError("Source-grouped split must contain both labels in both sets.")
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def _train_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    learning_rate: float,
) -> tuple[np.ndarray, float, list[float]]:
    weights = np.zeros(x.shape[1], dtype=np.float32)
    bias = 0.0
    pos_count = float(np.sum(y == 1.0))
    neg_count = float(np.sum(y == 0.0))
    pos_weight = (pos_count + neg_count) / max(1.0, 2.0 * pos_count)
    neg_weight = (pos_count + neg_count) / max(1.0, 2.0 * neg_count)
    sample_weights = np.where(y == 1.0, pos_weight, neg_weight).astype(np.float32)
    history = []

    for _ in range(epochs):
        probabilities = _predict_proba(x, weights, bias)
        error = (probabilities - y) * sample_weights
        grad_w = (x.T @ error) / x.shape[0] + 0.002 * weights
        grad_b = float(np.mean(error))
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
        history.append(_weighted_loss(y, probabilities, sample_weights, weights))

    return weights, bias, history


def _predict_proba(x: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    logits = np.clip(x @ weights + bias, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _weighted_loss(y: np.ndarray, probabilities: np.ndarray, sample_weights: np.ndarray, weights: np.ndarray) -> float:
    eps = 1e-7
    loss = -(y * np.log(probabilities + eps) + (1.0 - y) * np.log(1.0 - probabilities + eps))
    return float(np.mean(loss * sample_weights) + 0.001 * np.sum(weights * weights))


def _metrics(y: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(np.float32)
    tp = int(np.sum((predictions == 1.0) & (y == 1.0)))
    fp = int(np.sum((predictions == 1.0) & (y == 0.0)))
    tn = int(np.sum((predictions == 0.0) & (y == 0.0)))
    fn = int(np.sum((predictions == 0.0) & (y == 1.0)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    accuracy = (tp + tn) / max(1, len(y))
    return {
        "count": int(len(y)),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _recommended_threshold(y: np.ndarray, probabilities: np.ndarray) -> float:
    """Prefer precision over recall so speech is less likely to be removed."""
    best_threshold = 0.75
    best_score = -1.0
    for threshold in np.arange(0.50, 0.951, 0.01):
        metrics = _metrics(y, probabilities, float(threshold))
        precision = float(metrics["precision"])
        recall = float(metrics["recall"])
        beta_squared = 0.25
        score = (1.0 + beta_squared) * precision * recall / max(beta_squared * precision + recall, 1e-9)
        if score > best_score or (math.isclose(score, best_score) and threshold > best_threshold):
            best_score = score
            best_threshold = float(threshold)
    return round(best_threshold, 2)


def _label_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = row.get("label", "unknown")
        counts[label] = counts.get(label, 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
