from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    labels = "dataset/candidates/labels.csv"
    rows = []
    for path in sorted(Path("models").glob("*.json")):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
            if meta.get("model_type") != "numpy_logistic_regression":
                continue
        except Exception:
            continue
        name = path.stem
        model = path.as_posix()
        summary = _evaluate(model, labels)
        confusion = summary["confusion"]
        rows.append(
            {
                "name": name,
                "model": model,
                "accuracy": summary["accuracy"],
                "precision": summary["breath_precision"],
                "recall": summary["breath_recall"],
                "wrong": summary["wrong"],
                "tp": confusion["tp"],
                "fp": confusion["fp"],
                "tn": confusion["tn"],
                "fn": confusion["fn"],
            }
        )

    rows.sort(key=lambda item: (item["accuracy"], item["precision"], item["recall"]), reverse=True)
    out = Path("outputs/model_comparison.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['name']}: acc={row['accuracy']:.4f} "
            f"precision={row['precision']:.4f} recall={row['recall']:.4f} "
            f"wrong={row['wrong']} fp={row['fp']} fn={row['fn']}"
        )
    print(f"wrote {out}")
    return 0


def _evaluate(model: str, labels: str) -> dict:
    command = [
        sys.executable,
        "-m",
        "breath_cleaner.evaluate_model",
        "--model",
        model,
        "--labels",
        labels,
    ]
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = "src"
    result = subprocess.run(command, cwd=Path.cwd(), env=env, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
