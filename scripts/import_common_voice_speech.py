from __future__ import annotations

import argparse
import csv
import tarfile
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from breath_cleaner.audio_io import read_audio, write_wav  # noqa: E402


FIELDNAMES = ["clip", "source", "start", "end", "score", "label", "notes"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Import Common Voice Turkish clips as speech negatives.")
    parser.add_argument("archive", help="Common Voice Turkish .tar.gz archive.")
    parser.add_argument("--labels", default="dataset/candidates/labels.csv")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--label", default="speech", choices=["speech", "unknown"])
    args = parser.parse_args()

    archive = Path(args.archive)
    if not archive.exists():
        raise FileNotFoundError(archive)

    labels_path = Path(args.labels)
    raw_dir = labels_path.parent.parent / "external" / "common_voice_tr"
    clips_dir = labels_path.parent / "clips" / "common_voice_tr"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(labels_path)
    existing_clips = {Path(row.get("clip", "")).as_posix() for row in rows}
    existing_sources = {Path(row.get("source", "")).as_posix() for row in rows}

    imported = 0
    skipped = 0
    failed = 0

    with tarfile.open(archive, "r:gz") as tar:
        members = [
            member
            for member in tar
            if member.isfile() and "/clips/" in member.name and member.name.lower().endswith(".mp3")
        ]

        for member in members:
            if imported >= args.limit:
                break

            name = Path(member.name).name
            raw_path = raw_dir / name
            clip_path = clips_dir / f"{Path(name).stem}.wav"
            if clip_path.as_posix() in existing_clips or raw_path.as_posix() in existing_sources:
                skipped += 1
                continue

            try:
                if not raw_path.exists():
                    source = tar.extractfile(member)
                    if source is None:
                        raise RuntimeError("archive member has no file data")
                    with raw_path.open("wb") as handle:
                        handle.write(source.read())

                audio = read_audio(raw_path)
                write_wav(clip_path, audio)
                duration = audio.samples.size / float(audio.sample_rate)
            except Exception as exc:
                failed += 1
                print(f"skip {name}: {exc}", file=sys.stderr)
                continue

            row = {
                "clip": str(clip_path),
                "source": str(raw_path),
                "start": "0.000",
                "end": f"{duration:.3f}",
                "score": "common_voice",
                "label": args.label,
                "notes": "Common Voice Turkish speech negative",
            }
            _append_row(labels_path, row)
            existing_clips.add(clip_path.as_posix())
            existing_sources.add(raw_path.as_posix())
            imported += 1
            print(f"imported {imported}/{args.limit} {name}", flush=True)

    print(f"imported {imported}")
    print(f"skipped {skipped}")
    print(f"failed {failed}")
    return 0


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _append_row(path: Path, row: dict[str, str]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writerow(row)


if __name__ == "__main__":
    raise SystemExit(main())
