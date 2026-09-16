from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .features import extract_features


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a saved breath classifier on labeled clips.")
    parser.add_argument("--model", default="models/breath_logreg_latest.json", help="Model JSON path.")
    parser.add_argument("--labels", default="dataset/candidates/labels.csv", help="Labels CSV path.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Breath probability threshold.")
    parser.add_argument("--predictions", help="Optional CSV path for per-clip predictions.")
    args = parser.parse_args()

    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    rows = _read_rows(Path(args.labels))
    results = []

    for row in rows:
        label = row.get("label", "unknown")
        if label == "breath":
            expected = "breath"
        elif label in {"speech", "noise", "silence"}:
            expected = "non_breath"
        else:
            continue

        clip = Path(row["clip"])
        if not clip.exists():
            continue
        probability = _predict(model, extract_features(clip))
        predicted = "breath" if probability >= args.threshold else "non_breath"
        results.append(
            {
                "clip": row["clip"],
                "label": label,
                "expected": expected,
                "predicted": predicted,
                "breath_probability": probability,
                "correct": expected == predicted,
            }
        )

    summary = _summary(results)
    if args.predictions:
        _write_predictions(Path(args.predictions), results)

    print(json.dumps(summary, indent=2))
    return 0


def _predict(model: dict, features: np.ndarray) -> float:
    mean = np.asarray(model["mean"], dtype=np.float32)
    std = np.asarray(model["std"], dtype=np.float32)
    weights = np.asarray(model["weights"], dtype=np.float32)
    bias = float(model["bias"])
    x = (features - mean) / std
    logit = float(np.clip(x @ weights + bias, -40.0, 40.0))
    return float(1.0 / (1.0 + np.exp(-logit)))


def _summary(results: list[dict]) -> dict:
    total = len(results)
    correct = sum(1 for item in results if item["correct"])
    wrong = total - correct
    tp = sum(1 for item in results if item["expected"] == "breath" and item["predicted"] == "breath")
    fp = sum(1 for item in results if item["expected"] == "non_breath" and item["predicted"] == "breath")
    tn = sum(1 for item in results if item["expected"] == "non_breath" and item["predicted"] == "non_breath")
    fn = sum(1 for item in results if item["expected"] == "breath" and item["predicted"] == "non_breath")
    by_label: dict[str, dict[str, int]] = {}
    for item in results:
        label = item["label"]
        if label not in by_label:
            by_label[label] = {"total": 0, "correct": 0, "wrong": 0}
        by_label[label]["total"] += 1
        if item["correct"]:
            by_label[label]["correct"] += 1
        else:
            by_label[label]["wrong"] += 1

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "accuracy": round(correct / max(1, total), 4),
        "breath_precision": round(precision, 4),
        "breath_recall": round(recall, 4),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "by_original_label": by_label,
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_predictions(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["clip", "label", "expected", "predicted", "breath_probability", "correct"],
        )
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    raise SystemExit(main())
