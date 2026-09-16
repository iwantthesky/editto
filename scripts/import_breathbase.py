from __future__ import annotations

import sys
try:
    import argparse
    import csv
    import json
    import shutil
    import urllib.request
    import wave
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    FIELDNAMES = ["clip", "source", "start", "end", "score", "label", "notes"]
    ZENODO_API = "https://zenodo.org/api/records/3841039"
    ZENODO_RECORD = "https://zenodo.org/records/3841039/files"

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from breath_cleaner.audio_io import read_audio, write_wav  # noqa: E402
except Exception as e:
    print(f"FATAL ERROR during imports: {e}")
    sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Import BreathBase WAV files into the local labeler dataset.")
    parser.add_argument("--labels", default="dataset/candidates/labels.csv", help="Path to labels.csv.")
    parser.add_argument("--limit", type=int, default=200, help="Number of files to import. Use 0 for all files.")
    parser.add_argument("--label", default="unknown", choices=["unknown", "breath"], help="Initial label for imported clips.")
    parser.add_argument("--force", action="store_true", help="Re-download files even when already present.")
    parser.add_argument("--existing-only", action="store_true", help="Import already downloaded BreathBase files without downloading more.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel download/convert worker count.")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.csv not found: {labels_path}")

    raw_dir = labels_path.parent.parent / "external" / "breathbase"
    clips_dir = labels_path.parent / "clips" / "breathbase"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    if args.existing_only:
        wav_files = [
            {"name": path.name, "download_url": ""}
            for path in sorted(raw_dir.glob("*.wav"))
        ]
    else:
        files = _zenodo_files()
        if not files:
            files = _fallback_files(args.limit)
        wav_files = [item for item in files if item["name"].lower().endswith(".wav")]
        if args.limit > 0:
            wav_files = wav_files[: args.limit]

    rows = _read_rows(labels_path)
    existing_clips = {Path(row.get("clip", "")).as_posix() for row in rows}
    existing_raw = {Path(row.get("source", "")).as_posix() for row in rows}
    imported = 0
    skipped = 0
    failed = 0
    pending = []

    for item in wav_files:
        raw_path = raw_dir / item["name"]
        clip_path = clips_dir / item["name"]
        clip_key = clip_path.as_posix()
        raw_key = raw_path.as_posix()
        if clip_key in existing_clips or raw_key in existing_raw:
            skipped += 1
            continue
        pending.append((item, raw_path, clip_path))

    workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _prepare_import,
                item,
                raw_path,
                clip_path,
                args.label,
                args.force,
                args.existing_only,
            )
            for item, raw_path, clip_path in pending
        ]

        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception as exc:
                failed += 1
                print(f"skip import: {exc}", file=sys.stderr)
                continue

            clip_key = Path(row["clip"]).as_posix()
            raw_key = Path(row["source"]).as_posix()
            current_rows = _read_rows(labels_path)
            current_clips = {Path(item.get("clip", "")).as_posix() for item in current_rows}
            current_raw = {Path(item.get("source", "")).as_posix() for item in current_rows}
            if clip_key in current_clips or raw_key in current_raw:
                skipped += 1
                existing_clips.add(clip_key)
                existing_raw.add(raw_key)
                continue

            existing_clips.add(clip_key)
            existing_raw.add(raw_key)
            imported += 1
            _append_row(labels_path, row)
            print(
                json.dumps(
                    {
                        "imported": imported,
                        "skipped": skipped,
                        "failed": failed,
                        "clip": row["clip"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    print(
        json.dumps(
            {
                "ok": True,
                "downloaded_or_checked": len(wav_files),
                "imported": imported,
                "skipped": skipped,
                "failed": failed,
            },
            indent=2,
        )
    )
    return 0


def _prepare_import(
    item: dict[str, str],
    raw_path: Path,
    clip_path: Path,
    label: str,
    force: bool,
    existing_only: bool,
) -> dict[str, str]:
    if force or not raw_path.exists():
        if existing_only:
            raise FileNotFoundError(f"{item['name']} is not downloaded yet")
        _download(item["download_url"], raw_path)

    if force or not clip_path.exists() or not _is_readable_pcm_wav(clip_path):
        audio = read_audio(raw_path)
        write_wav(clip_path, audio)

    duration = _wav_duration_seconds(clip_path)
    return {
        "clip": str(clip_path),
        "source": str(raw_path),
        "start": "0.000",
        "end": f"{duration:.3f}",
        "score": "breathbase",
        "label": label,
        "notes": "BreathBase Zenodo 3841039",
    }


def _zenodo_files() -> list[dict[str, str]]:
    try:
        request = urllib.request.Request(
            ZENODO_API, 
            headers={"User-Agent": "Editto-BreathBase-Importer/1.0"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"Zenodo API error: {e}")
        return []

    result = []
    for item in payload.get("files", []):
        name = item.get("key") or item.get("filename")
        links = item.get("links", {})
        download_url = links.get("self") or links.get("download")
        if name and download_url:
            result.append({"name": str(name), "download_url": str(download_url)})
    return sorted(result, key=lambda value: value["name"])


def _fallback_files(limit: int) -> list[dict[str, str]]:
    target = limit if limit > 0 else 5070
    result = []
    for speaker in range(1, 21):
        for condition in range(1, 7):
            for index in range(1, 500):
                name = f"{speaker:02d}_{condition}_{index:03d}.wav"
                result.append({"name": name, "download_url": f"{ZENODO_RECORD}/{name}?download=1"})
                if len(result) >= target:
                    return result
    return result


def _download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Editto-BreathBase-Importer/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    temporary.replace(destination)


def _wav_duration_seconds(path: Path) -> float:
    audio = read_audio(path)
    return audio.samples.size / float(audio.sample_rate)


def _is_readable_pcm_wav(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getsampwidth() == 2
    except Exception:
        return False


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _append_row(path: Path, row: dict[str, str]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writerow(row)

if __name__ == '__main__':
    raise SystemExit(main())
