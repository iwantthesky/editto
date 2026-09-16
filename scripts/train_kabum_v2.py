from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
import numpy as np

from breath_cleaner.features import FEATURE_NAMES, extract_features

POSITIVE_LABEL = "breath"
NEGATIVE_LABELS = {"speech", "noise", "silence"}


def main() -> int:
    print("=== kabum_v2 Eğitimi Başlatılıyor ===")
    base_model_path = Path("models/kabum_v1.json")
    if not base_model_path.exists():
        raise FileNotFoundError("models/kabum_v1.json bulunamadı!")
    base_model = json.loads(base_model_path.read_text(encoding="utf-8"))

    labels_path = Path("dataset/candidates/labels.csv")
    rows = _load_rows(labels_path)
    print(f"Toplam etiketli satır sayısı: {len(rows)}")

    # Feature cache
    cache_path = Path("tmp/kabum_v2_features.npz")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        print("Önbellekten özellikler yükleniyor...")
        cached = np.load(cache_path, allow_pickle=True)
        x = cached["x"]
        y = cached["y"]
        used_rows = [dict(zip(cached["row_keys"], vals)) for vals in cached["rows"]]
    else:
        print("Ses kliplerinden 19 akustik özellik çıkarılıyor...")
        x, y, used_rows = _load_dataset(rows)
        # Cache for fast re-runs
        row_keys = list(used_rows[0].keys())
        rows_matrix = np.array([[r.get(k, "") for k in row_keys] for r in used_rows])
        np.savez_compressed(cache_path, x=x, y=y, row_keys=row_keys, rows=rows_matrix)
        print(f"Özellikler önbelleğe kaydedildi: {cache_path}")

    print(f"Kullanılan toplam örnek: {len(y)} (Nefes: {int(np.sum(y == 1.0))}, Konuşma/Diğer: {int(np.sum(y == 0.0))})")

    # Short breath analysis
    dur_idx = FEATURE_NAMES.index("duration")
    short_breath_mask = (y == 1.0) & (x[:, dur_idx] < 0.38)
    print(f"Kısa nefes (< 0.38s) sayısı: {int(np.sum(short_breath_mask))}")

    # Split by source
    train_idx, val_idx = _group_stratified_split(y, used_rows, seed=42, validation_fraction=0.20)
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    train_rows = [used_rows[i] for i in train_idx]

    # Oversample short breaths in training set to boost sensitivity
    train_short_breaths = np.where((y_train == 1.0) & (x_train[:, dur_idx] < 0.38))[0]
    if len(train_short_breaths) > 0:
        # Repeat short breaths 2x with slight jitter on features to prevent overfitting
        extra_x = []
        extra_y = []
        rng = np.random.default_rng(42)
        for idx in train_short_breaths:
            feat = x_train[idx].copy()
            for _ in range(2):
                noise = rng.normal(0, 0.02, size=feat.shape) * np.abs(feat)
                extra_x.append(feat + noise)
                extra_y.append(1.0)
        x_train = np.vstack([x_train, np.array(extra_x, dtype=np.float32)])
        y_train = np.concatenate([y_train, np.array(extra_y, dtype=np.float32)])
        print(f"Eğitim kümesine {len(extra_x)} adet güçlendirilmiş kısa nefes örneği eklendi.")

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-6] = 1.0
    x_train_s = (x_train - mean) / std
    x_val_s = (x_val - mean) / std

    # Train with higher penalty for false positives (speech cut off)
    print("kabum_v2 lojistik regresyon optimize ediliyor...")
    weights, bias, history = _train_kabum_v2(
        x_train_s,
        y_train,
        base_weights=np.array(base_model["weights"], dtype=np.float32),
        base_bias=float(base_model["bias"]),
        epochs=1500,
        learning_rate=0.05,
        fp_penalty=2.2,  # Stronger penalty against cutting speech
    )

    # Evaluate validation
    val_probs = _predict_proba(x_val_s, weights, bias)
    rec_thresh = _optimize_threshold(y_val, val_probs)
    print(f"Önerilen Eşik: {rec_thresh}")

    train_metrics = _metrics(y_train, _predict_proba(x_train_s, weights, bias), rec_thresh)
    val_metrics = _metrics(y_val, val_probs, rec_thresh)

    print(f"Eğitim Metrikleri: Doğruluk={train_metrics['accuracy']:.4f}, Kesinlik={train_metrics['precision']:.4f}, Yakalama={train_metrics['recall']:.4f}")
    print(f"Doğrulama Metrikleri: Doğruluk={val_metrics['accuracy']:.4f}, Kesinlik={val_metrics['precision']:.4f}, Yakalama={val_metrics['recall']:.4f}")
    print(f"Doğrulama Hataları: FP (Konuşma kesme)={val_metrics['fp']}, FN (Nefes kaçırma)={val_metrics['fn']}")

    # Save kabum_v2
    out_model = Path("models/kabum_v2.json")
    payload = {
        "model_type": "numpy_logistic_regression",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "version": "kabum_v2",
        "base_model": "kabum_v1",
        "positive_label": POSITIVE_LABEL,
        "negative_labels": sorted(NEGATIVE_LABELS),
        "feature_names": FEATURE_NAMES,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": weights.tolist(),
        "bias": float(bias),
        "train_count": int(len(train_idx)),
        "validation_count": int(len(val_idx)),
        "recommended_threshold": rec_thresh,
        "label_counts": {
            "speech": int(np.sum(y == 0.0)),
            "breath": int(np.sum(y == 1.0)),
        },
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "short_breath_enhanced": True,
    }

    out_model.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Yeni model kaydedildi: {out_model}")
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
            break
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
        return _stratified_split(y, seed)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def _train_kabum_v2(
    x: np.ndarray,
    y: np.ndarray,
    base_weights: np.ndarray,
    base_bias: float,
    epochs: int = 1500,
    learning_rate: float = 0.05,
    fp_penalty: float = 2.2,
) -> tuple[np.ndarray, float, list[float]]:
    # Initialize near base_weights to preserve learned acoustic structure
    weights = base_weights.copy()
    bias = base_bias

    pos_count = float(np.sum(y == 1.0))
    neg_count = float(np.sum(y == 0.0))
    pos_weight = (pos_count + neg_count) / max(1.0, 2.0 * pos_count)
    neg_weight = ((pos_count + neg_count) / max(1.0, 2.0 * neg_count)) * fp_penalty
    sample_weights = np.where(y == 1.0, pos_weight, neg_weight).astype(np.float32)

    history = []
    best_loss = 1e9
    best_w = weights.copy()
    best_b = bias

    # Adam-like momentum
    m_w = np.zeros_like(weights)
    v_w = np.zeros_like(weights)
    m_b = 0.0
    v_b = 0.0
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    for step in range(1, epochs + 1):
        probabilities = _predict_proba(x, weights, bias)
        error = (probabilities - y) * sample_weights

        grad_w = (x.T @ error) / x.shape[0] + 0.001 * weights
        grad_b = float(np.mean(error))

        m_w = beta1 * m_w + (1 - beta1) * grad_w
        v_w = beta2 * v_w + (1 - beta2) * (grad_w ** 2)
        m_w_hat = m_w / (1 - beta1 ** step)
        v_w_hat = v_w / (1 - beta2 ** step)

        m_b = beta1 * m_b + (1 - beta1) * grad_b
        v_b = beta2 * v_b + (1 - beta2) * (grad_b ** 2)
        m_b_hat = m_b / (1 - beta1 ** step)
        v_b_hat = v_b / (1 - beta2 ** step)

        weights -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + eps)
        bias -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + eps)

        loss = _weighted_loss(y, probabilities, sample_weights, weights)
        history.append(loss)
        if loss < best_loss:
            best_loss = loss
            best_w = weights.copy()
            best_b = bias

    return best_w, best_b, history


def _predict_proba(x: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    logits = np.clip(x @ weights + bias, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _weighted_loss(y: np.ndarray, probabilities: np.ndarray, sample_weights: np.ndarray, weights: np.ndarray) -> float:
    eps = 1e-7
    loss = -(y * np.log(probabilities + eps) + (1.0 - y) * np.log(1.0 - probabilities + eps))
    return float(np.mean(loss * sample_weights) + 0.0005 * np.sum(weights * weights))


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


def _optimize_threshold(y: np.ndarray, probabilities: np.ndarray) -> float:
    """Find threshold that keeps recall >= 0.99 while maximizing precision."""
    best_thresh = 0.70
    best_score = -1.0
    for t in np.arange(0.40, 0.90, 0.01):
        m = _metrics(y, probabilities, float(t))
        # Penalty if recall drops below 98.5%
        rec = float(m["recall"])
        prec = float(m["precision"])
        if rec < 0.985:
            score = rec * 0.5
        else:
            score = prec * 2.0 + rec
        if score > best_score:
            best_score = score
            best_thresh = float(t)
    return round(best_thresh, 2)


if __name__ == "__main__":
    raise SystemExit(main())
