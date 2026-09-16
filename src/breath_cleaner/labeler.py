from __future__ import annotations

import argparse
import ast
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, message="'cgi' is deprecated.*")

import cgi
import csv
import json
import mimetypes
import subprocess
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

try:
    from .audio_io import WavAudio, read_audio, read_audio_native, write_wav
    from .detect import BreathSegment, DetectionConfig, detect_breaths
    from .features import extract_features
    from .process import duck_segments
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from breath_cleaner.audio_io import WavAudio, read_audio, read_audio_native, write_wav
    from breath_cleaner.detect import BreathSegment, DetectionConfig, detect_breaths
    from breath_cleaner.features import extract_features
    from breath_cleaner.process import duck_segments


LABELS = ["breath", "speech", "noise", "silence", "bad"]
FIELDNAMES = ["clip", "source", "start", "end", "score", "label", "notes"]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
MODEL_CACHE: dict[tuple[str, float, float], float | None] = {}
MODEL_DISPLAY_NAMES = {
    "kabum_v2": "Editto Precision v2",
    "kabum_v1": "Editto Precision v1",
    "kabum": "Editto Classic",
    "babum": "Editto Legacy",
    "breath_cnn_personal_v3": "Editto Neural v3 (Deneysel)",
    "breath_cnn_v2": "Editto Neural v2 (Deneysel)",
}
PROCESS_MODES = {
    "cut": {"gain_db": 0.0, "fade_ms": 12.0, "label": "cut"},
    "mute": {"gain_db": -96.0, "fade_ms": 18.0, "label": "mute"},
    "strong": {"gain_db": -28.0, "low_confidence_gain_db": -16.0, "fade_ms": 42.0, "label": "strong"},
    "gentle": {"gain_db": -16.0, "low_confidence_gain_db": -8.0, "fade_ms": 52.0, "label": "gentle"},
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local breath dataset labeler.")
    parser.add_argument("--labels", default="dataset/candidates/labels.csv", help="Path to labels.csv.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8765, help="Bind port.")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.csv not found: {labels_path}")

    handler = _make_handler(labels_path)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Labeler running at http://{args.host}:{args.port}")
    print(f"Labels file: {labels_path}")
    server.serve_forever()
    return 0


def _make_handler(labels_path: Path) -> type[BaseHTTPRequestHandler]:
    class LabelerHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_text(_html_payload("INDEX_HTML", INDEX_HTML), "text/html; charset=utf-8")
            elif parsed.path == "/manual":
                self._send_text(_html_payload("MANUAL_HTML", MANUAL_HTML), "text/html; charset=utf-8")
            elif parsed.path == "/test":
                self._send_text(_html_payload("TEST_HTML", TEST_HTML), "text/html; charset=utf-8")
            elif parsed.path == "/model":
                self._send_text(_html_payload("MODEL_HTML", MODEL_HTML), "text/html; charset=utf-8")
            elif parsed.path == "/api/items":
                self._send_json(_items_payload(labels_path))
            elif parsed.path == "/api/audio":
                query = parse_qs(parsed.query)
                self._send_audio(labels_path, query)
            elif parsed.path == "/api/sources":
                self._send_json(_sources_payload(labels_path))
            elif parsed.path == "/api/source-audio":
                query = parse_qs(parsed.query)
                self._send_source_audio(labels_path, query)
            elif parsed.path == "/api/processed-audio":
                query = parse_qs(parsed.query)
                self._send_processed_audio(query)
            elif parsed.path == "/api/test-detections":
                query = parse_qs(parsed.query)
                try:
                    self._send_json(_test_detections_payload(labels_path, query))
                except (ValueError, RuntimeError, OSError) as exc:
                    self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            elif parsed.path == "/api/models":
                choices = []
                for name in ("kabum_v2", "kabum_v1", "kabum", "babum", "breath_cnn_personal_v3", "breath_cnn_v2"):
                    model = _load_test_model(name)
                    if model:
                        choices.append({"id": name, "display_name": MODEL_DISPLAY_NAMES.get(name, name),
                            "status": model.get("status", "legacy"),
                            "recommended_threshold": model.get("recommended_threshold", 0.75),
                            "version": model.get("version", name),
                            "accepted": model.get("accepted", False)})
                self._send_json({"models": choices, "default": "kabum_v2"})
            elif parsed.path == "/api/evaluate":
                self._send_json(_run_model_command("evaluate", labels_path))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/label":
                self._handle_candidate_label(labels_path)
            elif parsed.path == "/api/manual-label":
                self._handle_manual_label(labels_path)
            elif parsed.path == "/api/upload":
                self._handle_upload(labels_path)
            elif parsed.path == "/api/trim-candidate":
                self._handle_trim_candidate(labels_path)
            elif parsed.path == "/api/add-selection":
                self._handle_add_selection(labels_path)
            elif parsed.path == "/api/process-model":
                self._handle_process_model(labels_path)
            elif parsed.path == "/api/train":
                self._send_json(_run_model_command("train", labels_path))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
                return

        def _handle_candidate_label(self, labels_file: Path) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8")
                payload = json.loads(body)
                index = int(payload["index"])
                label = str(payload["label"])
                notes = str(payload.get("notes", ""))
                if label not in LABELS:
                    raise ValueError(f"Invalid label: {label}")
                rows = _read_rows(labels_path)
                if index < 0 or index >= len(rows):
                    raise IndexError(f"Invalid item index: {index}")
                rows[index]["label"] = label
                rows[index]["notes"] = notes
                _write_rows(labels_file, rows)
                model = _load_latest_model()
                self._send_json({"ok": True, "item": _item_payload(index, rows[index], model), "summary": _summary(rows)})
            except Exception as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))

        def _handle_manual_label(self, labels_file: Path) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8")
                payload = json.loads(body)
                source_index = int(payload["source_index"])
                start = float(payload["start"])
                end = float(payload["end"])
                label = str(payload["label"])
                notes = str(payload.get("notes", ""))

                if label not in LABELS:
                    raise ValueError(f"Invalid label: {label}")
                if end <= start:
                    raise ValueError("Selection end must be after start.")
                if end - start < 0.04:
                    raise ValueError("Selection is too short.")

                sources = _sources(labels_file)
                if source_index < 0 or source_index >= len(sources):
                    raise ValueError("Invalid source index.")

                source_path = Path(sources[source_index])
                if not source_path.exists():
                    raise FileNotFoundError(f"Source not found: {source_path}")

                rows = _read_rows(labels_file)
                clip_path = _manual_clip_path(labels_file, source_path)
                _extract_clip_ffmpeg(source_path, clip_path, start, end)

                row = {
                    "clip": str(clip_path),
                    "source": str(source_path),
                    "start": f"{start:.3f}",
                    "end": f"{end:.3f}",
                    "score": "manual",
                    "label": label,
                    "notes": notes,
                }
                rows.append(row)
                _write_rows(labels_file, rows)
                self._send_json({"ok": True, "item": row, "summary": _summary(rows)})
            except Exception as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))

        def _handle_upload(self, labels_file: Path) -> None:
            try:
                content_type = self.headers.get("Content-Type", "")
                if "multipart/form-data" not in content_type:
                    raise ValueError("Expected multipart/form-data upload.")

                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": content_type,
                    },
                )
                fields = form["files"] if "files" in form else []
                if not isinstance(fields, list):
                    fields = [fields]

                results = []
                raw_dir = labels_file.parent.parent / "raw"
                raw_dir.mkdir(parents=True, exist_ok=True)
                extracted_dir = labels_file.parent.parent / "extracted_audio"
                extracted_dir.mkdir(parents=True, exist_ok=True)

                for field in fields:
                    if not getattr(field, "filename", ""):
                        continue
                    filename = _safe_filename(field.filename)
                    destination = _unique_path(raw_dir / filename)
                    with destination.open("wb") as handle:
                        handle.write(field.file.read())

                    candidate_source = _prepare_source_for_detection(destination, extracted_dir)
                    count = _append_candidates_for_source(labels_file, candidate_source)
                    if count == 0:
                        _register_source(labels_file, str(candidate_source))
                    results.append(
                        {
                            "file": str(destination),
                            "source": str(candidate_source),
                            "kind": "video" if _is_video_file(destination) else "audio",
                            "candidates": count,
                        }
                    )

                if not results:
                    raise ValueError("No files were uploaded.")

                MODEL_CACHE.clear()
                rows = _read_rows(labels_file)
                self._send_json({"ok": True, "results": results, "summary": _summary(rows)})
            except Exception as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))

        def _handle_process_model(self, labels_file: Path) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8")
                payload = json.loads(body)
                source_index = int(payload["source_index"])
                mode = str(payload["mode"])
                model_name = str(payload.get("model", "kabum_v1"))
                detections_payload = payload.get("detections", [])

                sources = _sources(labels_file)
                if source_index < 0 or source_index >= len(sources):
                    raise ValueError("Invalid source index.")
                if mode not in PROCESS_MODES:
                    raise ValueError(f"Invalid process mode: {mode}")
                if not isinstance(detections_payload, list):
                    raise ValueError("Detections must be a list.")

                source_path = Path(sources[source_index])
                if not source_path.exists():
                    raise FileNotFoundError(f"Source not found: {source_path}")

                segments = _segments_from_payload(detections_payload)
                output = _process_source_audio(source_path, segments, mode, model_name)
                self._send_json(
                    {
                        "ok": True,
                        "mode": mode,
                        "model": model_name,
                        "segments": len(segments),
                        "output": str(output),
                        "audio": f"/api/processed-audio?file={output.name}",
                    }
                )
            except Exception as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))

        def _handle_trim_candidate(self, labels_file: Path) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8")
                payload = json.loads(body)
                index = int(payload["index"])
                start = float(payload["start"])
                end = float(payload["end"])
                if end <= start:
                    raise ValueError("Trim end must be after start.")
                if end - start < 0.03:
                    raise ValueError("Trim selection is too short.")

                rows = _read_rows(labels_file)
                if index < 0 or index >= len(rows):
                    raise IndexError(f"Invalid item index: {index}")
                clip_path = Path(rows[index]["clip"])
                if not clip_path.exists():
                    raise FileNotFoundError(f"Clip not found: {clip_path}")

                audio = read_audio(clip_path)
                clip_duration = audio.samples.size / audio.sample_rate
                start = max(0.0, min(clip_duration, start))
                end = max(0.0, min(clip_duration, end))
                if end <= start:
                    raise ValueError("Trim selection is outside the clip.")

                start_sample = int(start * audio.sample_rate)
                end_sample = int(end * audio.sample_rate)
                trimmed = WavAudio(sample_rate=audio.sample_rate, samples=audio.samples[start_sample:end_sample])
                trimmed_path = _trimmed_clip_path(clip_path)
                write_wav(trimmed_path, trimmed)

                original_start = float(rows[index]["start"])
                rows[index]["clip"] = str(trimmed_path)
                rows[index]["start"] = f"{original_start + start:.3f}"
                rows[index]["end"] = f"{original_start + end:.3f}"
                rows[index]["notes"] = (rows[index].get("notes", "") + " trimmed").strip()
                _write_rows(labels_file, rows)
                MODEL_CACHE.clear()
                model = _load_latest_model()
                self._send_json({"ok": True, "item": _item_payload(index, rows[index], model), "summary": _summary(rows)})
            except Exception as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))

        def _handle_add_selection(self, labels_file: Path) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8")
                payload = json.loads(body)
                index = int(payload["index"])
                start = float(payload["start"])
                end = float(payload["end"])
                if end <= start:
                    raise ValueError("Selection end must be after start.")
                if end - start < 0.03:
                    raise ValueError("Selection is too short.")

                rows = _read_rows(labels_file)
                if index < 0 or index >= len(rows):
                    raise IndexError(f"Invalid item index: {index}")
                clip_path = Path(rows[index]["clip"])
                if not clip_path.exists():
                    raise FileNotFoundError(f"Clip not found: {clip_path}")

                audio = read_audio(clip_path)
                clip_duration = audio.samples.size / audio.sample_rate
                start = max(0.0, min(clip_duration, start))
                end = max(0.0, min(clip_duration, end))
                if end <= start:
                    raise ValueError("Selection is outside the clip.")

                start_sample = int(start * audio.sample_rate)
                end_sample = int(end * audio.sample_rate)
                selected = WavAudio(sample_rate=audio.sample_rate, samples=audio.samples[start_sample:end_sample])
                selected_path = _selection_clip_path(clip_path)
                write_wav(selected_path, selected)

                original_start = float(rows[index]["start"])
                row = {
                    "clip": str(selected_path),
                    "source": rows[index]["source"],
                    "start": f"{original_start + start:.3f}",
                    "end": f"{original_start + end:.3f}",
                    "score": "manual",
                    "label": "unknown",
                    "notes": "selection",
                }
                new_index = index + 1
                rows.insert(new_index, row)
                _write_rows(labels_file, rows)
                MODEL_CACHE.clear()
                payload = _items_payload(labels_file)
                payload["ok"] = True
                payload["new_index"] = new_index
                self._send_json(payload)
            except Exception as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))

        def log_message(self, format: str, *args: object) -> None:
            return

        def _send_audio(self, labels_file: Path, query: dict[str, list[str]]) -> None:
            rows = _read_rows(labels_file)
            index = int(query.get("index", ["-1"])[0])
            if index < 0 or index >= len(rows):
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            clip_path = Path(rows[index]["clip"])
            if not clip_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, f"Clip not found: {clip_path}")
                return

            data = clip_path.read_bytes()
            mime_type = mimetypes.guess_type(clip_path.name)[0] or "audio/wav"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_source_audio(self, labels_file: Path, query: dict[str, list[str]]) -> None:
            sources = _sources(labels_file)
            index = int(query.get("index", ["-1"])[0])
            if index < 0 or index >= len(sources):
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            source_path = Path(sources[index])
            if not source_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, f"Source not found: {source_path}")
                return

            data = source_path.read_bytes()
            mime_type = mimetypes.guess_type(source_path.name)[0] or "audio/mp4"
            self._send_bytes_with_range(data, mime_type)

        def _send_bytes_with_range(self, data: bytes, mime_type: str) -> None:
            total = len(data)
            range_header = self.headers.get("Range")
            if range_header and range_header.startswith("bytes="):
                spec = range_header[len("bytes="):].split(",", 1)[0].strip()
                start_str, _, end_str = spec.partition("-")
                try:
                    if start_str == "":
                        suffix = int(end_str)
                        if suffix <= 0:
                            raise ValueError
                        start = max(0, total - suffix)
                        end = total - 1
                    else:
                        start = int(start_str)
                        end = int(end_str) if end_str else total - 1
                        if end >= total:
                            end = total - 1
                        if start > end or start >= total:
                            raise ValueError
                except ValueError:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{total}")
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    return

                chunk = data[start:end + 1]
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Type", mime_type)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                self._write_body(chunk)
                return

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(total))
            self.end_headers()
            self._write_body(data)

        def _write_body(self, data: bytes) -> None:
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def _send_json(self, payload: object) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self._send_no_cache_headers()
            self.end_headers()
            self.wfile.write(data)

        def _send_processed_audio(self, query: dict[str, list[str]]) -> None:
            filename = Path(query.get("file", [""])[0]).name
            output_path = Path("outputs/model_processed") / filename
            if not filename or not output_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, f"Processed audio not found: {filename}")
                return

            data = output_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self._send_no_cache_headers()
            self.end_headers()
            self.wfile.write(data)

        def _send_text(self, payload: str, content_type: str) -> None:
            data = payload.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self._send_no_cache_headers()
            self.end_headers()
            self.wfile.write(data)

        def _send_no_cache_headers(self) -> None:
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")

    return LabelerHandler


def _items_payload(labels_path: Path) -> dict[str, object]:
    rows = _read_rows(labels_path)
    model = _load_latest_model()
    return {
        "labels": LABELS,
        "summary": _summary(rows),
        "source_summary": _source_summary(rows),
        "items": [
            _item_payload(index, row, model)
            for index, row in enumerate(rows)
        ],
    }


def _item_payload(index: int, row: dict[str, str], model: dict | None) -> dict[str, object]:
    probability = _model_probability(row.get("clip", ""), model)
    prediction = ""
    if probability is not None:
        prediction = "breath" if probability >= 0.5 else "non_breath"
    return {
        "index": index,
        "clip": row["clip"],
        "source": row["source"],
        "start": row["start"],
        "end": row["end"],
        "score": row["score"],
        "label": row["label"],
        "notes": row["notes"],
        "model_probability": probability,
        "model_prediction": prediction,
        "audio": f"/api/audio?index={index}",
    }


def _load_latest_model() -> dict | None:
    model_path = Path("models/breath_logreg_latest.json")
    if not model_path.exists():
        return None
    try:
        payload = json.loads(model_path.read_text(encoding="utf-8"))
        payload["_mtime"] = model_path.stat().st_mtime
        return payload
    except Exception:
        return None


def _model_probability(clip: str, model: dict | None) -> float | None:
    if model is None:
        return None
    clip_path = Path(clip)
    if not clip_path.exists():
        return None
    model_mtime = float(model.get("_mtime", 0.0))
    cache_key = (str(clip_path), clip_path.stat().st_mtime_ns, model.get("_path", "latest"), model_mtime)
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]
    try:
        from .prediction import predict_clip
        probability = round(predict_clip(clip_path, model), 4)
        MODEL_CACHE[cache_key] = probability
        return probability
    except Exception:
        MODEL_CACHE[cache_key] = None
        return None


def _sources_payload(labels_path: Path) -> dict[str, object]:
    sources = _sources(labels_path)
    return {
        "labels": LABELS,
        "latest_index": max(0, len(sources) - 1),
        "sources": [
            {"index": index, "path": source, "audio": f"/api/source-audio?index={index}"}
            for index, source in enumerate(sources)
        ],
    }


def _test_detections_payload(labels_path: Path, query: dict[str, list[str]]) -> dict[str, object]:
    sources = _sources(labels_path)
    source_index = int(query.get("source_index", [str(max(0, len(sources) - 1))])[0])
    threshold = float(query.get("threshold", ["0.5"])[0])
    model_name = query.get("model", ["latest"])[0]
    respect_labels = query.get("respect_labels", ["0"])[0].lower() in {"1", "true", "yes"}
    if source_index < 0 or source_index >= len(sources):
        raise ValueError("Invalid source index.")

    source = sources[source_index]
    rows = _read_rows(labels_path)
    model = _load_test_model(model_name)
    if model is None:
        raise ValueError("Selected model could not be loaded")
    if model.get("model_type") == "breath_cnn_v2":
        from .prediction import scan_recording
        result = scan_recording(source, model, threshold if "threshold" in query else None)
        result.update({"source_index": source_index, "source": source, "model": model_name,
                       "count": len(result["detections"])})
        return result
    detections = []
    for index, row in enumerate(rows):
        if row.get("source", "") != source:
            continue
        if respect_labels and row.get("label", "") in {"speech", "noise", "silence", "bad"}:
            continue
        probability = _model_probability(row.get("clip", ""), model)
        if probability is None or probability < threshold:
            continue
        detections.append(
            {
                "index": index,
                "start": float(row["start"]),
                "end": float(row["end"]),
                "label": row.get("label", ""),
                "score": row.get("score", ""),
                "model_probability": probability,
                "audio": f"/api/audio?index={index}",
                "model_version": model.get("version", Path(model["_path"]).stem),
                "needs_review": True,
            }
        )

    detections.sort(key=lambda item: (item["start"], item["end"]))
    detections = _merge_detection_payloads(detections)
    return {
        "source_index": source_index,
        "source": source,
        "threshold": threshold,
        "model": model_name,
        "count": len(detections),
        "detections": detections,
        "model_version": model.get("version", Path(model["_path"]).stem),
        "scan_mode": "legacy_candidates",
        "requires_review": True,
    }


def _merge_detection_payloads(detections: list[dict[str, object]]) -> list[dict[str, object]]:
    """Collapse overlapping/duplicate candidates before showing or processing them."""
    if not detections:
        return []
    merged = [dict(detections[0])]
    for detection in detections[1:]:
        previous = merged[-1]
        if float(detection["start"]) <= float(previous["end"]) + 0.015:
            previous["end"] = max(float(previous["end"]), float(detection["end"]))
            previous["model_probability"] = max(
                float(previous.get("model_probability", 0.0)),
                float(detection.get("model_probability", 0.0)),
            )
            if detection.get("label") != previous.get("label"):
                previous["label"] = "mixed"
        else:
            merged.append(dict(detection))
    return merged


def _load_test_model(model_name: str) -> dict | None:
    if model_name == "kabum_v2":
        path = Path("models/kabum_v2.json")
    elif model_name == "breath_cnn_personal_v3":
        path = Path("models/breath_cnn_personal_v3.json")
    elif model_name == "breath_cnn_v2":
        path = Path("models/breath_cnn_v2.json")
    elif model_name == "babum":
        path = Path("models/babum.json")
    elif model_name == "kabum_v1":
        path = Path("models/kabum_v1.json")
    elif model_name == "kabum":
        path = Path("models/kabum.json")
    elif model_name == "kakapun":
        path = Path("models/kakapun.json")
    elif model_name == "previous":
        path = Path("models/breath_logreg_v20260605_130955.json")
    elif model_name == "old":
        path = Path("models/breath_logreg_v20260605_013909.json")
    elif model_name == "new" or model_name == "latest":
        path = Path("models/breath_logreg_latest.json")
    else:
        path = Path(model_name)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_mtime"] = path.stat().st_mtime
        payload["_path"] = str(path)
        return payload
    except Exception:
        return None


def _segments_from_payload(items: list[object]) -> list[BreathSegment]:
    segments: list[BreathSegment] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item["start"])
            end = float(item["end"])
            score = float(item.get("model_probability", item.get("score", 1.0)) or 1.0)
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start or end - start < 0.01:
            continue
        segments.append(
            BreathSegment(
                start=max(0.0, start),
                end=max(0.0, end),
                score=score,
                rms_db=0.0,
                zcr=0.0,
                centroid_hz=0.0,
                rolloff_hz=0.0,
            )
        )
    return _merge_overlapping_segments(sorted(segments, key=lambda segment: (segment.start, segment.end)))


def _merge_overlapping_segments(segments: list[BreathSegment]) -> list[BreathSegment]:
    if not segments:
        return []
    merged = [segments[0]]
    for segment in segments[1:]:
        previous = merged[-1]
        if segment.start <= previous.end + 0.015:
            merged[-1] = BreathSegment(
                start=previous.start,
                end=max(previous.end, segment.end),
                score=max(previous.score, segment.score),
                rms_db=0.0,
                zcr=0.0,
                centroid_hz=0.0,
                rolloff_hz=0.0,
            )
        else:
            merged.append(segment)
    return merged


def _process_source_audio(source_path: Path, segments: list[BreathSegment], mode: str, model_name: str) -> Path:
    config = PROCESS_MODES[mode]
    audio = read_audio_native(source_path)
    duration = audio.samples.shape[0] / audio.sample_rate
    clipped_segments = [
        BreathSegment(
            start=max(0.0, min(duration, segment.start)),
            end=max(0.0, min(duration, segment.end)),
            score=segment.score,
            rms_db=segment.rms_db,
            zcr=segment.zcr,
            centroid_hz=segment.centroid_hz,
            rolloff_hz=segment.rolloff_hz,
        )
        for segment in segments
        if min(duration, segment.end) > max(0.0, segment.start)
    ]
    if mode == "cut":
        processed = _cut_segments(audio.samples, audio.sample_rate, clipped_segments, crossfade_ms=float(config["fade_ms"]))
    else:
        processed = duck_segments(
            audio.samples,
            audio.sample_rate,
            clipped_segments,
            gain_db=float(config["gain_db"]),
            fade_ms=float(config["fade_ms"]),
            low_confidence_gain_db=config.get("low_confidence_gain_db"),
        )
    out_dir = Path("outputs/model_processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(source_path.stem)
    model_stem = _safe_stem(model_name)
    out_path = _unique_path(out_dir / f"{stem}_{model_stem}_{mode}.wav")
    write_wav(out_path, WavAudio(sample_rate=audio.sample_rate, samples=processed))
    return out_path


def _cut_segments(
    samples: np.ndarray,
    sample_rate: int,
    segments: list[BreathSegment],
    crossfade_ms: float = 12.0,
) -> np.ndarray:
    if not segments:
        return np.asarray(samples, dtype=np.float32).copy()

    source_audio = np.asarray(samples, dtype=np.float32)
    frame_count = source_audio.shape[0]
    ranges = []
    for segment in _merge_overlapping_segments(segments):
        start = max(0, int(segment.start * sample_rate))
        end = min(frame_count, int(segment.end * sample_rate))
        if end > start:
            ranges.append((start, end))
    if not ranges:
        return source_audio.copy()

    pieces = []
    cursor = 0
    for start, end in ranges:
        if start > cursor:
            pieces.append(source_audio[cursor:start])
        cursor = max(cursor, end)
    if cursor < frame_count:
        pieces.append(source_audio[cursor:])
    if not pieces:
        return np.zeros((0,) + source_audio.shape[1:], dtype=np.float32)

    output = pieces[0].copy()
    crossfade_samples = max(1, int(sample_rate * crossfade_ms / 1000.0))
    for piece in pieces[1:]:
        if output.shape[0] == 0:
            output = piece.copy()
            continue
        if piece.shape[0] == 0:
            continue
        fade_len = min(crossfade_samples, output.shape[0], piece.shape[0])
        if fade_len <= 1:
            output = np.concatenate([output, piece], axis=0)
            continue
        left = output[-fade_len:]
        right = piece[:fade_len]
        fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        if source_audio.ndim == 2:
            fade_out = fade_out[:, None]
            fade_in = fade_in[:, None]
        joined = left * fade_out + right * fade_in
        output = np.concatenate([output[:-fade_len], joined, piece[fade_len:]], axis=0)
    return output.astype(np.float32, copy=False)


def _sources(labels_path: Path) -> list[str]:
    seen = []
    for row in _read_rows(labels_path):
        source = row.get("source", "")
        if source and source not in seen:
            seen.append(source)
    registry = labels_path.with_suffix('.sources.json')
    if registry.exists():
        for source in json.loads(registry.read_text(encoding='utf-8')):
            if source not in seen:
                seen.append(source)
    return seen


def _register_source(labels_path: Path, source: str) -> None:
    import os
    import tempfile
    registry = labels_path.with_suffix('.sources.json')
    sources = json.loads(registry.read_text(encoding='utf-8')) if registry.exists() else []
    if source in sources:
        return
    sources.append(source)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=registry.parent, delete=False) as f:
        json.dump(sources, f)
        temporary = f.name
    os.replace(temporary, registry)


def _summary(rows: list[dict[str, str]]) -> dict[str, int]:
    summary = {label: 0 for label in ["unknown", *LABELS]}
    for row in rows:
        summary[row.get("label", "unknown")] = summary.get(row.get("label", "unknown"), 0) + 1
    return summary


def _source_summary(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for row in rows:
        source = row.get("source", "")
        label = row.get("label", "unknown")
        if source not in result:
            result[source] = {"total": 0, "unknown": 0, "breath": 0, "speech": 0, "noise": 0, "silence": 0, "bad": 0}
        result[source]["total"] += 1
        result[source][label] = result[source].get(label, 0) + 1
    return result


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def _run_model_command(command_name: str, labels_path: Path) -> dict[str, object]:
    if command_name == "train":
        args = [
            sys.executable,
            "-m",
            "breath_cleaner.train_model",
            "--labels",
            str(labels_path),
        ]
    elif command_name == "evaluate":
        args = [
            sys.executable,
            "-m",
            "breath_cleaner.evaluate_model",
            "--labels",
            str(labels_path),
            "--predictions",
            "outputs/model_predictions.csv",
        ]
    else:
        raise ValueError(f"Unknown model command: {command_name}")

    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = "src"
    result = subprocess.run(args, cwd=Path.cwd(), env=env, capture_output=True, text=True)
    if result.returncode != 0:
        return {
            "ok": False,
            "command": command_name,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        payload = {"raw": result.stdout}
    if command_name == "train":
        MODEL_CACHE.clear()
    return {"ok": True, "command": command_name, "result": payload}


def _append_candidates_for_source(labels_path: Path, source_path: Path) -> int:
    audio = read_audio(source_path)
    segments = detect_breaths(audio.samples, audio.sample_rate, DetectionConfig())
    rows = _read_rows(labels_path)
    clips_dir = labels_path.parent / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    for segment in segments:
        clip_path = _auto_clip_path(clips_dir, source_path)
        clip_audio = _extract_clip(audio, segment, context_ms=120.0)
        write_wav(clip_path, clip_audio)
        rows.append(
            {
                "clip": str(clip_path),
                "source": str(source_path),
                "start": f"{segment.start:.3f}",
                "end": f"{segment.end:.3f}",
                "score": f"{segment.score:.6f}",
                "label": "unknown",
                "notes": "",
            }
        )

    _write_rows(labels_path, rows)
    return len(segments)


def _prepare_source_for_detection(path: Path, extracted_dir: Path) -> Path:
    if not _is_video_file(path):
        return path

    extracted_dir.mkdir(parents=True, exist_ok=True)
    audio_path = _unique_path(extracted_dir / f"{path.stem}_audio.wav")
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        str(audio_path),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return audio_path


def _is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def _extract_clip(audio: WavAudio, segment: BreathSegment, context_ms: float) -> WavAudio:
    context_samples = int(audio.sample_rate * context_ms / 1000.0)
    start = max(0, int(segment.start * audio.sample_rate) - context_samples)
    end = min(audio.samples.size, int(segment.end * audio.sample_rate) + context_samples)
    return WavAudio(sample_rate=audio.sample_rate, samples=np.asarray(audio.samples[start:end], dtype=np.float32))


def _auto_clip_path(clips_dir: Path, source_path: Path) -> Path:
    stem = _safe_stem(source_path.stem)
    next_index = len(list(clips_dir.glob(f"{stem}_candidate_*.wav"))) + 1
    while True:
        candidate = clips_dir / f"{stem}_candidate_{next_index:04d}.wav"
        if not candidate.exists():
            return candidate
        next_index += 1


def _safe_filename(filename: str) -> str:
    path = Path(filename)
    stem = _safe_stem(path.stem)
    suffix = path.suffix.lower() or ".audio"
    return f"{stem}_{int(time.time())}{suffix}"


def _safe_stem(stem: str) -> str:
    safe = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in stem.strip())
    return safe or "audio"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    index = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def _manual_clip_path(labels_path: Path, source_path: Path) -> Path:
    clips_dir = labels_path.parent / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    existing = list(clips_dir.glob("manual_*.wav"))
    next_index = len(existing) + 1
    while True:
        candidate = clips_dir / f"manual_{source_path.stem}_{next_index:04d}.wav"
        if not candidate.exists():
            return candidate
        next_index += 1


def _trimmed_clip_path(clip_path: Path) -> Path:
    next_index = 1
    while True:
        candidate = clip_path.with_name(f"{clip_path.stem}_trimmed_{next_index:02d}.wav")
        if not candidate.exists():
            return candidate
        next_index += 1


def _selection_clip_path(clip_path: Path) -> Path:
    next_index = 1
    while True:
        candidate = clip_path.with_name(f"{clip_path.stem}_selection_{next_index:02d}.wav")
        if not candidate.exists():
            return candidate
        next_index += 1


def _extract_clip_ffmpeg(source_path: Path, clip_path: Path, start: float, end: float) -> None:
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(source_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        str(clip_path),
    ]
    subprocess.run(command, check=True, capture_output=True)


def _html_payload(name: str, fallback: str) -> str:
    try:
        source = Path(__file__).read_text(encoding="utf-8")
        module = ast.parse(source)
        for node in module.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                continue
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return node.value.value
    except Exception:
        return fallback
    return fallback


INDEX_HTML = r"""<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Breath Labeler</title>
  <script src="https://unpkg.com/wavesurfer.js@7"></script>
  <style>
    :root {
      color-scheme: light;
      --bg: #f0f2f5;
      --panel: #ffffff;
      --ink: #111827;
      --muted: #6b7280;
      --line: #e5e7eb;
      --accent: #0f766e;
      --accent-hover: #0d6460;
      --accent-weak: #ccfaf4;
      --danger: #b42318;
      --warn: #8a5a00;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, Arial, sans-serif;
      font-size: 14px;
      line-height: 1.5;
      letter-spacing: 0;
    }
    header {
      height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 20px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      box-shadow: 0 1px 4px rgba(0,0,0,.07);
    }
    h1 {
      margin: 0;
      font-size: 17px;
      font-weight: 700;
      letter-spacing: -0.3px;
    }
    nav {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-left: 18px;
      margin-right: auto;
    }
    nav a {
      color: var(--accent);
      font-size: 13px;
      font-weight: 600;
      text-decoration: none;
      padding: 4px 10px;
      border-radius: 6px;
      transition: background .15s;
    }
    nav a:hover { background: var(--accent-weak); }
    .summary {
      display: flex;
      gap: 5px;
      align-items: center;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .summary span {
      min-width: 68px;
      padding: 3px 8px;
      border: 1px solid var(--line);
      background: var(--bg);
      border-radius: 20px;
      text-align: center;
      font-weight: 600;
    }
    main {
      display: grid;
      grid-template-columns: minmax(280px, 360px) 1fr;
      height: calc(100vh - 56px);
    }
    aside {
      overflow: auto;
      border-right: 1px solid var(--line);
      background: var(--panel);
    }
    .item {
      width: 100%;
      min-height: 58px;
      border: 0;
      border-bottom: 1px solid var(--line);
      background: transparent;
      display: grid;
      grid-template-columns: 44px 1fr 78px;
      gap: 8px;
      align-items: center;
      padding: 9px 12px;
      text-align: left;
      color: var(--ink);
      cursor: pointer;
      transition: background .1s;
    }
    .item:hover { background: #f5f7f9; }
    .item.active { background: var(--accent-weak); }
    .idx {
      font-weight: 700;
      color: var(--muted);
      font-size: 12px;
      text-align: right;
    }
    .time {
      font-size: 13px;
      font-weight: 700;
    }
    .meta {
      color: var(--muted);
      font-size: 11px;
      margin-top: 2px;
    }
    .model-meta {
      color: var(--accent);
      font-size: 11px;
      font-weight: 700;
      margin-top: 2px;
    }
    .model-meta.error { color: var(--danger); }
    .model-meta.warn { color: var(--warn); }
    .badge {
      height: 24px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 0 8px;
      border-radius: 20px;
      border: 1px solid var(--line);
      background: var(--bg);
      font-size: 11px;
      font-weight: 700;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .workspace {
      overflow: auto;
      padding: 20px 24px;
    }
    .surface {
      max-width: 900px;
      margin: 0 auto;
    }
    .upload {
      min-height: 88px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 18px;
      margin-bottom: 18px;
      border: 2px dashed #c5cdd8;
      border-radius: 10px;
      background: var(--panel);
      transition: border-color .15s, background .15s;
    }
    .upload.dragover {
      border-color: var(--accent);
      background: var(--accent-weak);
    }
    .upload-title {
      font-size: 14px;
      font-weight: 700;
      margin-bottom: 3px;
    }
    .upload-copy {
      color: var(--muted);
      font-size: 12px;
    }
    .upload button {
      min-width: 120px;
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--panel);
      font: inherit;
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
      transition: background .15s, border-color .15s;
    }
    .upload button:hover {
      background: var(--bg);
      border-color: #9ca3af;
    }
    .filters {
      display: grid;
      grid-template-columns: 140px minmax(180px, 1fr) 170px 150px 130px 130px 150px;
      gap: 8px;
      align-items: center;
      margin-bottom: 16px;
    }
    .filters select,
    .filters button {
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--panel);
      font: inherit;
      font-size: 13px;
      font-weight: 600;
      transition: border-color .15s, background .15s;
    }
    .filters select:hover,
    .filters button:hover {
      border-color: #9ca3af;
    }
    .filters button {
      cursor: pointer;
    }
    .filters button.active {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }
    .filters button.active:hover {
      background: var(--accent-hover);
    }
    .trim-tools button.active {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }
    .topline {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 16px;
      margin-bottom: 18px;
    }
    .clip-title {
      margin: 0;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: -0.3px;
    }
    .clip-meta {
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    #waveformWrap {
      position: relative;
      width: 100%;
      height: 128px;
      margin-bottom: 10px;
    }
    #waveform {
      width: 100%;
      height: 128px;
      background: #fafbfc;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 1px 4px rgba(0,0,0,.06);
    }
    #trimSelection {
      position: absolute;
      top: 1px;
      bottom: 1px;
      left: 0;
      width: 0;
      background: rgba(20, 184, 166, 0.22);
      border-left: 2px solid #0f766e;
      border-right: 2px solid #0f766e;
      border-radius: 7px;
      pointer-events: none;
      display: none;
    }
    .trim-tools {
      display: grid;
      grid-template-columns: repeat(5, minmax(88px, 1fr));
      gap: 8px 7px;
      margin-bottom: 14px;
      align-items: center;
    }
    .trim-tools > *:nth-child(11) {
      border-color: var(--accent);
      color: var(--accent);
      grid-column: span 2;
    }
    .trim-tools > *:nth-child(11):hover {
      background: var(--accent-weak);
    }
    .trim-tools > *:nth-child(12) {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
      grid-column: span 3;
    }
    .trim-tools > *:nth-child(12):hover {
      background: var(--accent-hover);
    }
    .trim-tools input,
    .trim-tools button {
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--panel);
      font: inherit;
      font-size: 12px;
      font-weight: 600;
      transition: background .15s, border-color .15s;
    }
    .trim-tools button {
      cursor: pointer;
    }
    .trim-tools button:hover {
      background: var(--bg);
      border-color: #9ca3af;
    }
    .labels {
      display: grid;
      grid-template-columns: repeat(5, minmax(94px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .label {
      min-height: 52px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--ink);
      font-size: 14px;
      font-weight: 700;
      cursor: pointer;
      position: relative;
      transition: border-color .15s, box-shadow .15s;
    }
    .label:hover {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-weak);
    }
    .label::after {
      content: attr(data-key);
      position: absolute;
      top: 5px;
      right: 7px;
      font-size: 10px;
      color: var(--muted);
      opacity: 0.7;
    }
    .label.selected {
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
      box-shadow: 0 2px 8px rgba(15,118,110,.25);
    }
    .label.selected::after { color: #fff; opacity: 0.8; }
    .label.bad.selected {
      border-color: var(--danger);
      background: var(--danger);
      box-shadow: 0 2px 8px rgba(180,35,24,.25);
    }
    .label.noise.selected {
      border-color: var(--warn);
      background: var(--warn);
    }
    textarea {
      width: 100%;
      min-height: 80px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 10px 12px;
      font: inherit;
      font-size: 13px;
      margin-bottom: 14px;
      transition: border-color .15s, box-shadow .15s;
    }
    textarea:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-weak);
    }
    .actions {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }
    .nav {
      display: flex;
      gap: 8px;
    }
    .action {
      min-width: 112px;
      min-height: 40px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--panel);
      font: inherit;
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
      transition: background .15s, border-color .15s;
    }
    .action:hover {
      background: var(--bg);
      border-color: #9ca3af;
    }
    .status {
      color: var(--muted);
      font-size: 13px;
      min-height: 20px;
    }
    kbd {
      background: #f3f4f6;
      border-radius: 4px;
      border: 1px solid #d1d5db;
      padding: 2px 5px;
      font-size: 11px;
      font-family: inherit;
    }
    @media (max-width: 780px) {
      header {
        height: auto;
        min-height: 56px;
        align-items: flex-start;
        flex-direction: column;
        gap: 8px;
        padding: 12px;
      }
      .summary {
        flex-wrap: wrap;
        white-space: normal;
      }
      nav {
        margin: 0;
      }
      .upload {
        align-items: stretch;
        flex-direction: column;
      }
      .filters {
        grid-template-columns: 1fr;
      }
      .trim-tools {
        grid-template-columns: 1fr 1fr;
      }
      .trim-tools > *:nth-child(11),
      .trim-tools > *:nth-child(12) {
        grid-column: 1 / -1;
      }
      main {
        grid-template-columns: 1fr;
        height: auto;
      }
      aside {
        max-height: 240px;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
      .labels {
        grid-template-columns: repeat(2, minmax(120px, 1fr));
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>Breath Labeler</h1>
    <nav>
      <a href="/manual">Manual Selection</a>
      <a href="/test">Test Detection</a>
      <a href="/model">Model</a>
    </nav>
    <div id="summary" class="summary"></div>
  </header>
  <main>
    <aside id="list"></aside>
    <section class="workspace">
      <div class="surface">
        <div id="uploadZone" class="upload">
          <div>
            <div class="upload-title">Drop audio or video files here</div>
            <div class="upload-copy">Audio is extracted from videos, then scanned for breath-like candidates.</div>
          </div>
          <div>
            <input id="filePicker" type="file" multiple accept="audio/*,video/*" hidden>
            <button id="pickFiles" type="button">Choose Files</button>
          </div>
        </div>
        <div class="filters">
          <select id="labelFilter">
            <option value="all">All labels</option>
            <option value="unknown">unknown</option>
            <option value="breath">breath</option>
            <option value="speech">speech</option>
            <option value="noise">noise</option>
            <option value="silence">silence</option>
            <option value="bad">bad</option>
          </select>
          <select id="sourceFilter"></select>
          <select id="modelFilter">
            <option value="all">All model scores</option>
            <option value="breath70">model breath > 70%</option>
            <option value="breath90">model breath > 90%</option>
            <option value="low30">model breath < 30%</option>
          </select>
          <button id="mistakesOnly" type="button">Model Errors</button>
          <button id="trainModel" type="button">Train</button>
          <button id="evalModel" type="button">Evaluate</button>
          <button id="negativeMode" type="button">Negatives</button>
        </div>
        <div class="topline">
          <h2 id="title" class="clip-title">Loading</h2>
          <div id="clipMeta" class="clip-meta"></div>
        </div>
        <div id="waveformWrap">
          <div id="waveform"></div>
          <div id="trimSelection"></div>
        </div>
        <div class="trim-tools">
          <input id="trimStart" type="number" min="0" step="0.001" value="0">
          <input id="trimEnd" type="number" min="0" step="0.001" value="0">
          <button id="markTrim" type="button">Mark on Wave</button>
          <button id="setTrimStart" type="button">Set Start Here</button>
          <button id="setTrimEnd" type="button">Set End Here</button>
          <button id="trimStartBack" type="button">Start -10ms</button>
          <button id="trimStartForward" type="button">Start +10ms</button>
          <button id="trimEndBack" type="button">End -10ms</button>
          <button id="trimEndForward" type="button">End +10ms</button>
          <button id="clearTrim" type="button">Clear Trim</button>
          <button id="addSelection" type="button">Add as Unknown</button>
          <button id="trimCandidate" type="button">Trim to Selection</button>
        </div>
        <div id="labels" class="labels"></div>
        <textarea id="notes" placeholder="Notes (Esc to blur)"></textarea>
        <div class="actions">
          <div class="nav">
            <button id="prev" class="action" type="button" title="Left Arrow">Previous</button>
            <button id="next" class="action" type="button" title="Right Arrow">Next</button>
          </div>
          <div id="status" class="status"></div>
          <div class="shortcuts" style="font-size: 11px; color: var(--muted);">
            <kbd>Space</kbd> Play/Pause &nbsp; <kbd>1-5</kbd> Label &nbsp; <kbd>←</kbd> <kbd>→</kbd> Nav
          </div>
        </div>
      </div>
    </section>
  </main>
  <script>
    let state = { items: [], labels: [], current: 0, summary: {}, filters: { label: 'all', source: 'all', model: 'all', mistakesOnly: false }, trim: { active: false, start: 0, end: 0, clicks: 0, dragging: false, dragStart: 0, dragMoved: false } };
    let wavesurfer;
    const STORAGE_KEY = 'edittoBreathLabelerState';
    
    const list = document.getElementById('list');
    const labels = document.getElementById('labels');
    const title = document.getElementById('title');
    const clipMeta = document.getElementById('clipMeta');
    const notes = document.getElementById('notes');
    const status = document.getElementById('status');
    const summary = document.getElementById('summary');
    const uploadZone = document.getElementById('uploadZone');
    const filePicker = document.getElementById('filePicker');
    const pickFiles = document.getElementById('pickFiles');
    const labelFilter = document.getElementById('labelFilter');
    const sourceFilter = document.getElementById('sourceFilter');
    const modelFilter = document.getElementById('modelFilter');
    const mistakesOnly = document.getElementById('mistakesOnly');
    const trainModel = document.getElementById('trainModel');
    const evalModel = document.getElementById('evalModel');
    const negativeMode = document.getElementById('negativeMode');
    const trimStart = document.getElementById('trimStart');
    const trimEnd = document.getElementById('trimEnd');
    const markTrim = document.getElementById('markTrim');
    const setTrimStart = document.getElementById('setTrimStart');
    const setTrimEnd = document.getElementById('setTrimEnd');
    const trimStartBack = document.getElementById('trimStartBack');
    const trimStartForward = document.getElementById('trimStartForward');
    const trimEndBack = document.getElementById('trimEndBack');
    const trimEndForward = document.getElementById('trimEndForward');
    const clearTrim = document.getElementById('clearTrim');
    const addSelection = document.getElementById('addSelection');
    const trimCandidate = document.getElementById('trimCandidate');
    const waveformWrap = document.getElementById('waveformWrap');
    const trimSelection = document.getElementById('trimSelection');

    function initWaveSurfer() {
      if (wavesurfer) wavesurfer.destroy();
      wavesurfer = WaveSurfer.create({
        container: '#waveform',
        waveColor: '#0f766e',
        progressColor: '#0f766e',
        cursorColor: '#17191c',
        height: 128,
        normalize: true,
        interact: false,
      });
      wavesurfer.on('finish', () => wavesurfer.setTime(0));
    }

    async function load() {
      const response = await fetch('/api/items');
      const savedView = loadSavedView();
      state = await response.json();
      state.filters = savedView.filters || { label: 'all', source: 'all', model: 'all', mistakesOnly: false };
      state.current = validItemIndex(savedView.current) ? savedView.current : Math.max(0, state.items.findIndex(item => item.label === 'unknown'));
      if (!validItemIndex(state.current)) state.current = 0;
      initWaveSurfer();
      renderFilterControls();
      ensureCurrentVisible();
      render();
    }

    function loadSavedView() {
      try {
        return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
      } catch {
        return {};
      }
    }

    function saveView() {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ current: state.current, filters: state.filters }));
    }

    function validItemIndex(index) {
      return Number.isInteger(index) && index >= 0 && index < state.items.length;
    }

    function render() {
      renderSummary();
      renderList();
      renderCurrent();
    }

    function renderSummary() {
      const keys = ['unknown', 'breath', 'speech', 'noise', 'silence', 'bad'];
      summary.innerHTML = keys.map(key => `<span>${key}: ${state.summary[key] || 0}</span>`).join('');
    }

    function renderList() {
      const items = filteredItems();
      list.innerHTML = items.map(item => `
        <button class="item ${item.index === state.current ? 'active' : ''}" type="button" onclick="selectItem(${item.index})">
          <span class="idx">${item.index + 1}</span>
          <span>
            <div class="time">${item.start}s - ${item.end}s</div>
            <div class="meta">score ${scoreLine(item.score)}</div>
            <div class="model-meta ${modelStatusClass(item)}">${modelLine(item)}</div>
          </span>
          <span class="badge">${item.label}</span>
        </button>
      `).join('');
      const active = list.querySelector('.active');
      if (active) active.scrollIntoView({ block: 'nearest' });
    }

    function renderFilterControls() {
      const sources = Array.from(new Set(state.items.map(item => item.source))).sort();
      sourceFilter.innerHTML = '<option value="all">All sources</option>' + sources.map(source => {
        const stats = state.source_summary && state.source_summary[source] ? state.source_summary[source] : { total: 0, unknown: 0 };
        return `<option value="${escapeHtml(source)}">${sourceName(source)} (${stats.total}, u:${stats.unknown || 0})</option>`;
      }).join('');
      labelFilter.value = state.filters.label;
      sourceFilter.value = state.filters.source;
      modelFilter.value = state.filters.model;
      mistakesOnly.classList.toggle('active', state.filters.mistakesOnly);
    }

    function filteredItems() {
      return state.items.filter(item => {
        if (state.filters.label !== 'all' && item.label !== state.filters.label) return false;
        if (state.filters.source !== 'all' && item.source !== state.filters.source) return false;
        const probability = item.model_probability;
        if (state.filters.model === 'breath70' && !(probability !== null && probability > 0.70)) return false;
        if (state.filters.model === 'breath90' && !(probability !== null && probability > 0.90)) return false;
        if (state.filters.model === 'low30' && !(probability !== null && probability < 0.30)) return false;
        if (state.filters.mistakesOnly && !isModelMistake(item)) return false;
        return true;
      });
    }

    function isModelMistake(item) {
      if (item.model_prediction === '' || item.model_prediction === undefined) return false;
      if (item.label === 'unknown' || item.label === 'bad') return false;
      const expected = item.label === 'breath' ? 'breath' : 'non_breath';
      return item.model_prediction !== expected;
    }

    function modelErrorType(item) {
      if (!isModelMistake(item)) return '';
      if (item.label === 'breath' && item.model_prediction === 'non_breath') return 'FN';
      if (item.label !== 'breath' && item.model_prediction === 'breath') return 'FP';
      return 'ERR';
    }

    function modelStatusClass(item) {
      const errorType = modelErrorType(item);
      if (errorType === 'FP') return 'error';
      if (errorType === 'FN') return 'warn';
      return '';
    }

    function sourceName(source) {
      return source.split(/[\\/]/).pop();
    }

    function escapeHtml(value) {
      return String(value).replaceAll('&', '&amp;').replaceAll('"', '&quot;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
    }

    function renderCurrent() {
      const item = state.items[state.current];
      if (!item || filteredItems().length === 0) {
        title.textContent = 'No candidates';
        clipMeta.textContent = '';
        if (wavesurfer) wavesurfer.empty();
        notes.value = '';
        labels.innerHTML = '';
        status.textContent = 'No items match the current filters';
        return;
      }
      title.textContent = `Candidate ${item.index + 1}`;
      clipMeta.textContent = `${item.start}s - ${item.end}s${modelDetail(item)}`;
      wavesurfer.load(item.audio);
      resetTrimInputs();
      notes.value = item.notes || '';
      labels.innerHTML = state.labels.map((label, idx) => `
        <button class="label ${label} ${item.label === label ? 'selected' : ''}" 
                type="button" data-key="${idx+1}" 
                onclick="setLabel('${label}')">${label}</button>
      `).join('');
      status.textContent = '';
    }

    function modelLine(item) {
      if (item.model_probability === null || item.model_probability === undefined) return 'model -';
      const probability = `model breath ${(item.model_probability * 100).toFixed(1)}%`;
      const errorType = modelErrorType(item);
      if (!errorType) return probability;
      const expected = item.label === 'breath' ? 'breath' : 'non-breath';
      const predicted = item.model_prediction === 'breath' ? 'breath' : 'non-breath';
      return `${errorType}: expected ${expected}, got ${predicted} | ${probability}`;
    }

    function scoreLine(score) {
      const value = Number(score);
      return Number.isFinite(value) ? value.toFixed(3) : String(score || '-');
    }

    function modelDetail(item) {
      if (item.model_probability === null || item.model_probability === undefined) return '';
      const errorType = modelErrorType(item);
      const mistake = errorType ? ` | ${errorType} model error` : '';
      return ` | model breath ${(item.model_probability * 100).toFixed(1)}%${mistake}`;
    }

    function resetTrimInputs() {
      state.trim = { active: false, start: 0, end: 0, clicks: 0, dragging: false, dragStart: 0, dragMoved: false };
      trimStart.value = '0.000';
      trimEnd.value = '0.000';
      markTrim.classList.remove('active');
      updateTrimSelectionOverlay();
    }

    function syncTrimInputs() {
      const start = Math.min(state.trim.start, state.trim.end);
      const end = Math.max(state.trim.start, state.trim.end);
      state.trim.start = start;
      state.trim.end = end;
      trimStart.value = start.toFixed(3);
      trimEnd.value = end.toFixed(3);
      updateTrimSelectionOverlay();
    }

    function updateTrimFromInputs() {
      state.trim.start = Math.max(0, Number(trimStart.value));
      state.trim.end = Math.max(0, Number(trimEnd.value));
      syncTrimInputs();
    }

    function waveformTimeFromClientX(clientX) {
      const duration = wavesurfer ? wavesurfer.getDuration() : 0;
      const rect = waveformWrap.getBoundingClientRect();
      const x = Math.max(0, Math.min(rect.width, clientX - rect.left));
      if (!duration || !rect.width) return 0;
      return (x / rect.width) * duration;
    }

    function updateTrimSelectionOverlay() {
      const duration = wavesurfer ? wavesurfer.getDuration() : 0;
      if (!duration || state.trim.end <= state.trim.start) {
        trimSelection.style.display = 'none';
        trimSelection.style.left = '0';
        trimSelection.style.width = '0';
        return;
      }
      const left = Math.max(0, Math.min(100, (state.trim.start / duration) * 100));
      const right = Math.max(0, Math.min(100, (state.trim.end / duration) * 100));
      trimSelection.style.display = 'block';
      trimSelection.style.left = `${left}%`;
      trimSelection.style.width = `${Math.max(0, right - left)}%`;
    }

    function beginWaveSelection(event) {
      if (!wavesurfer || !wavesurfer.getDuration()) return;
      if (event.button !== 0) return;
      event.preventDefault();
      event.stopPropagation();
      const time = waveformTimeFromClientX(event.clientX);
      state.trim.dragging = true;
      state.trim.dragStart = time;
      state.trim.dragMoved = false;
      state.trim.start = time;
      state.trim.end = time;
      syncTrimInputs();
    }

    function moveWaveSelection(event) {
      if (!state.trim.dragging) return;
      event.preventDefault();
      const time = waveformTimeFromClientX(event.clientX);
      state.trim.start = state.trim.dragStart;
      state.trim.end = time;
      state.trim.dragMoved = Math.abs(state.trim.end - state.trim.dragStart) > 0.015;
      syncTrimInputs();
    }

    function endWaveSelection(event) {
      if (!state.trim.dragging) return;
      event.preventDefault();
      const time = waveformTimeFromClientX(event.clientX);
      state.trim.start = state.trim.dragStart;
      state.trim.end = time;
      const selected = Math.abs(state.trim.end - state.trim.dragStart);
      state.trim.dragging = false;
      state.trim.active = false;
      state.trim.clicks = 0;
      markTrim.classList.remove('active');
      if (selected <= 0.015) {
        state.trim.start = 0;
        state.trim.end = 0;
        if (wavesurfer) wavesurfer.setTime(time);
        syncTrimInputs();
        status.textContent = 'Playback position moved';
        return;
      }
      syncTrimInputs();
      status.textContent = 'Trim range selected';
    }

    function setTrimEdgeFromPlayhead(edge) {
      const time = wavesurfer ? wavesurfer.getCurrentTime() : 0;
      if (edge === 'start') {
        state.trim.start = Math.max(0, time);
        if (state.trim.end <= state.trim.start) state.trim.end = Math.min(wavesurfer.getDuration(), state.trim.start + 0.2);
      } else {
        state.trim.end = Math.max(0, time);
        if (state.trim.start >= state.trim.end) state.trim.start = Math.max(0, state.trim.end - 0.2);
      }
      syncTrimInputs();
      status.textContent = edge === 'start' ? 'Trim start set' : 'Trim end set';
    }

    function nudgeTrim(edge, delta) {
      updateTrimFromInputs();
      const duration = wavesurfer ? wavesurfer.getDuration() : 0;
      if (edge === 'start') {
        state.trim.start = Math.max(0, Math.min(state.trim.end - 0.01, state.trim.start + delta));
      } else {
        state.trim.end = Math.min(duration, Math.max(state.trim.start + 0.01, state.trim.end + delta));
      }
      syncTrimInputs();
    }

    function selectItem(index) {
      state.current = index;
      saveView();
      render();
    }

    function selectNextFiltered(direction) {
      const items = filteredItems();
      if (items.length === 0) return;
      const currentPos = items.findIndex(item => item.index === state.current);
      let nextPos = currentPos + direction;
      if (currentPos < 0) nextPos = direction > 0 ? 0 : items.length - 1;
      nextPos = Math.max(0, Math.min(items.length - 1, nextPos));
      selectItem(items[nextPos].index);
    }

    async function setLabel(label) {
      const response = await fetch('/api/label', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ index: state.current, label, notes: notes.value })
      });
      if (!response.ok) {
        status.textContent = 'Save failed';
        return;
      }
      const payload = await response.json();
      state.items[state.current] = { ...state.items[state.current], ...payload.item };
      state.summary = payload.summary;
      status.textContent = 'Saved';
      
      // Auto-advance if it was unknown
      if (payload.item.label !== 'unknown') {
         setTimeout(() => {
           const nextUnknown = state.items.findIndex((item, idx) => idx > state.current && item.label === 'unknown');
           if (nextUnknown >= 0) {
             state.current = nextUnknown;
             saveView();
             render();
           } else {
             saveView();
             render();
           }
         }, 150);
      } else {
        saveView();
        render();
      }
    }

    document.getElementById('prev').onclick = () => selectNextFiltered(-1);
    document.getElementById('next').onclick = () => selectNextFiltered(1);
    labelFilter.onchange = () => {
      state.filters.label = labelFilter.value;
      saveView();
      jumpToFirstFiltered();
    };
    sourceFilter.onchange = () => {
      state.filters.source = sourceFilter.value;
      saveView();
      jumpToFirstFiltered();
    };
    modelFilter.onchange = () => {
      state.filters.model = modelFilter.value;
      saveView();
      jumpToFirstFiltered();
    };
    mistakesOnly.onclick = () => {
      state.filters.mistakesOnly = !state.filters.mistakesOnly;
      mistakesOnly.classList.toggle('active', state.filters.mistakesOnly);
      saveView();
      jumpToFirstFiltered();
    };
    trainModel.onclick = () => runModelAction('train');
    evalModel.onclick = () => runModelAction('evaluate');
    negativeMode.onclick = () => {
      state.filters.label = 'all';
      state.filters.model = 'low30';
      state.filters.mistakesOnly = false;
      saveView();
      renderFilterControls();
      jumpToFirstFiltered();
    };
    markTrim.onclick = () => {
      state.trim.active = true;
      state.trim.clicks = 0;
      markTrim.classList.add('active');
      status.textContent = 'Drag on the waveform to select a trim range';
    };
    setTrimStart.onclick = () => setTrimEdgeFromPlayhead('start');
    setTrimEnd.onclick = () => setTrimEdgeFromPlayhead('end');
    trimStartBack.onclick = () => nudgeTrim('start', -0.010);
    trimStartForward.onclick = () => nudgeTrim('start', 0.010);
    trimEndBack.onclick = () => nudgeTrim('end', -0.010);
    trimEndForward.onclick = () => nudgeTrim('end', 0.010);
    clearTrim.onclick = () => {
      resetTrimInputs();
      status.textContent = 'Trim cleared';
    };
    trimStart.onchange = updateTrimFromInputs;
    trimEnd.onchange = updateTrimFromInputs;
    waveformWrap.addEventListener('mousedown', beginWaveSelection, true);
    window.addEventListener('mousemove', moveWaveSelection);
    window.addEventListener('mouseup', endWaveSelection);
    addSelection.onclick = async () => {
      updateTrimFromInputs();
      if (state.trim.end <= state.trim.start) {
        status.textContent = 'Select a valid range first';
        return;
      }
      const response = await fetch('/api/add-selection', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ index: state.current, start: state.trim.start, end: state.trim.end })
      });
      if (!response.ok) {
        status.textContent = 'Add selection failed';
        return;
      }
      const payload = await response.json();
      state = payload;
      state.filters = { label: 'unknown', source: 'all', model: 'all', mistakesOnly: false };
      state.current = payload.new_index;
      saveView();
      renderFilterControls();
      render();
      status.textContent = 'Selection added as unknown';
    };
    trimCandidate.onclick = async () => {
      updateTrimFromInputs();
      if (state.trim.end <= state.trim.start) {
        status.textContent = 'Select a valid trim range first';
        return;
      }
      const response = await fetch('/api/trim-candidate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ index: state.current, start: state.trim.start, end: state.trim.end })
      });
      if (!response.ok) {
        status.textContent = 'Trim failed';
        return;
      }
      const payload = await response.json();
      state.items[state.current] = payload.item;
      state.summary = payload.summary;
      status.textContent = 'Trim saved';
      render();
    };

    function jumpToFirstFiltered() {
      const items = filteredItems();
      if (items.length > 0) state.current = items[0].index;
      saveView();
      render();
    }

    function ensureCurrentVisible() {
      const items = filteredItems();
      if (items.length === 0) return;
      if (!items.some(item => item.index === state.current)) {
        state.current = items[0].index;
        saveView();
      }
    }

    async function runModelAction(action) {
      status.textContent = action === 'train' ? 'Training model' : 'Evaluating model';
      const response = await fetch(action === 'train' ? '/api/train' : '/api/evaluate', { method: action === 'train' ? 'POST' : 'GET' });
      const payload = await response.json();
      if (!payload.ok) {
        status.textContent = `${action} failed`;
        console.error(payload);
        return;
      }
      if (action === 'train') {
        const metrics = payload.result.validation_metrics || {};
        status.textContent = `trained: val f1 ${metrics.f1 ?? '-'} recall ${metrics.recall ?? '-'}`;
      } else {
        status.textContent = `eval: ${payload.result.correct}/${payload.result.total} correct, wrong ${payload.result.wrong}`;
      }
      await load();
    }
    
    // Keyboard shortcuts
    window.addEventListener('keydown', e => {
      if (document.activeElement === notes) {
        if (e.key === 'Escape') notes.blur();
        return;
      }
      
      if (e.key === ' ') { e.preventDefault(); wavesurfer.playPause(); }
      if (e.key === 'ArrowLeft') selectNextFiltered(-1);
      if (e.key === 'ArrowRight') selectNextFiltered(1);
      if (e.key === '[') setTrimEdgeFromPlayhead('start');
      if (e.key === ']') setTrimEdgeFromPlayhead('end');
      
      if (e.key >= '1' && e.key <= '5') {
        const idx = parseInt(e.key) - 1;
        if (state.labels[idx]) setLabel(state.labels[idx]);
      }
    });

    pickFiles.onclick = () => filePicker.click();
    filePicker.onchange = () => uploadFiles(filePicker.files);
    uploadZone.addEventListener('dragover', event => {
      event.preventDefault();
      uploadZone.classList.add('dragover');
    });
    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
    uploadZone.addEventListener('drop', event => {
      event.preventDefault();
      uploadZone.classList.remove('dragover');
      uploadFiles(event.dataTransfer.files);
    });

    async function uploadFiles(files) {
      if (!files || files.length === 0) return;
      status.textContent = 'Uploading and scanning';
      const form = new FormData();
      Array.from(files).forEach(file => form.append('files', file));
      const response = await fetch('/api/upload', { method: 'POST', body: form });
      if (!response.ok) {
        status.textContent = 'Upload failed';
        return;
      }
      const payload = await response.json();
      const total = payload.results.reduce((sum, item) => sum + item.candidates, 0);
      status.textContent = `Added ${total} candidate(s)`;
      await load();
    }

    window.selectItem = selectItem;
    window.setLabel = setLabel;
    load();
  </script>
</body>
</html>
"""


TEST_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Test Detection</title>
  <style>
    :root {
      --bg: #f0f2f5;
      --panel: #ffffff;
      --ink: #111827;
      --muted: #6b7280;
      --line: #e5e7eb;
      --accent: #0f766e;
      --accent-weak: #ccfaf4;
      --danger: #dc2626;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, Arial, sans-serif;
      font-size: 14px;
      letter-spacing: 0;
    }
    header {
      min-height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 20px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      box-shadow: 0 1px 4px rgba(0,0,0,.07);
    }
    h1 { margin: 0; font-size: 17px; }
    a { color: var(--accent); font-weight: 700; text-decoration: none; padding: 4px 10px; border-radius: 6px; }
    a:hover { background: var(--accent-weak); }
    main {
      max-width: 1180px;
      margin: 0 auto;
      padding: 20px 22px;
    }
    .toolbar {
      display: grid;
      grid-template-columns: minmax(260px, 1fr) 150px 130px 130px;
      gap: 10px;
      margin-bottom: 12px;
    }
    select, input, button {
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      font-weight: 650;
      padding: 0 10px;
    }
    button { cursor: pointer; }
    button:hover { border-color: var(--accent); }
    audio { display: none; }
    .transport {
      display: grid;
      grid-template-columns: 64px 64px minmax(170px, 1fr) 96px;
      gap: 8px;
      align-items: center;
      margin: 10px 0 14px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      box-shadow: 0 1px 4px rgba(0,0,0,.05);
    }
    .icon-button {
      min-width: 58px;
      padding: 0;
      font-size: 18px;
      line-height: 1;
    }
    .seek {
      display: grid;
      gap: 5px;
    }
    .seek input, .zoom-row input {
      width: 100%;
      padding: 0;
    }
    .time-line {
      display: flex;
      justify-content: space-between;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }
    .zoom-row {
      display: grid;
      grid-template-columns: 72px minmax(140px, 1fr) 72px minmax(140px, 1fr);
      gap: 8px;
      align-items: center;
      margin-top: 12px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 14px;
      box-shadow: 0 1px 4px rgba(0,0,0,.06);
    }
    canvas {
      display: block;
      width: 100%;
      height: 300px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fbfbfc;
      cursor: grab;
      user-select: none;
    }
    canvas.dragging { cursor: grabbing; }
    .readout {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
    }
    .legend {
      display: flex;
      gap: 12px;
      align-items: center;
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    .legend span::before {
      content: "";
      display: inline-block;
      width: 14px;
      height: 9px;
      margin-right: 6px;
      border-radius: 2px;
    }
    .legend .kakapun::before { background: rgba(220, 38, 38, 0.55); }
    .detections {
      margin-top: 14px;
      display: grid;
      gap: 6px;
      max-height: 260px;
      overflow: auto;
    }
    .detection {
      min-height: 34px;
      display: grid;
      grid-template-columns: 110px 90px 1fr;
      gap: 10px;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fff;
      padding: 6px 10px;
      text-align: left;
    }
    .detection strong { color: var(--danger); }
    .status { color: var(--muted); min-height: 22px; margin-top: 10px; }
    @media (max-width: 760px) {
      .toolbar { grid-template-columns: 1fr; }
      .transport { grid-template-columns: 64px 64px 1fr; }
      .transport select { grid-column: 1 / -1; }
      .zoom-row { grid-template-columns: 1fr; }
      canvas { height: 240px; }
      .detection { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Test Detection</h1>
    <span>
      <a href="/">Candidates</a>
      <a href="/manual">Manual Selection</a>
      <a href="/model">Model</a>
    </span>
  </header>
  <main>
    <div class="toolbar">
      <select id="source"></select>
      <select id="testModel">
        <option value="kabum_v2" selected>Editto Precision v2 (önerilen)</option>
        <option value="kabum_v1">Editto Precision v1</option>
        <option value="kabum">Editto Classic</option>
        <option value="babum">Editto Legacy</option>
        <option value="breath_cnn_personal_v3">Editto Neural v3 (Deneysel)</option>
        <option value="breath_cnn_v2">Editto Neural v2 (Deneysel)</option>
      </select>
      <input id="threshold" type="number" min="0" max="1" step="0.05" value="0.75" title="Model threshold">
      <button id="reload" type="button">Reload</button>
    </div>
    <audio id="audio"></audio>
    <div class="transport">
      <button id="playPause" class="icon-button" type="button" title="Play/Pause">Play</button>
      <button id="stop" class="icon-button" type="button" title="Stop">Stop</button>
      <div class="seek">
        <input id="seekSlider" type="range" min="0" max="1" step="0.001" value="0" title="Seek">
        <div class="time-line">
          <span id="currentTime">0.000s</span>
          <span id="totalTime">0.000s</span>
        </div>
      </div>
      <select id="speed" title="Playback speed">
        <option value="0.5">0.5x</option>
        <option value="0.75">0.75x</option>
        <option value="1" selected>1x</option>
        <option value="1.25">1.25x</option>
        <option value="1.5">1.5x</option>
      </select>
    </div>
    <div class="panel">
      <canvas id="wave"></canvas>
      <div class="zoom-row">
        <span>Zoom</span>
        <input id="zoomSlider" type="range" min="1" max="40" step="0.5" value="1">
        <span>Window</span>
        <input id="panSlider" type="range" min="0" max="0" step="0.001" value="0">
      </div>
      <div class="readout">
        <span id="duration">duration: -</span>
        <span id="summary">detections: -</span>
        <span id="playhead">playhead: 0.000s</span>
      </div>
      <div class="legend">
        <span class="kakapun" id="legendModel">kabum</span>
      </div>
    </div>
    <div id="status" class="status"></div>
    <div id="detections" class="detections"></div>
  </main>
  <script>
    const sourceSelect = document.getElementById('source');
    const testModelSelect = document.getElementById('testModel');
    const thresholdInput = document.getElementById('threshold');
    const reloadButton = document.getElementById('reload');
    const audio = document.getElementById('audio');
    const playPauseButton = document.getElementById('playPause');
    const stopButton = document.getElementById('stop');
    const seekSlider = document.getElementById('seekSlider');
    const currentTimeEl = document.getElementById('currentTime');
    const totalTimeEl = document.getElementById('totalTime');
    const speedSelect = document.getElementById('speed');
    const zoomSlider = document.getElementById('zoomSlider');
    const panSlider = document.getElementById('panSlider');
    const canvas = document.getElementById('wave');
    const ctx = canvas.getContext('2d');
    const statusEl = document.getElementById('status');
    const durationEl = document.getElementById('duration');
    const summaryEl = document.getElementById('summary');
    const playheadEl = document.getElementById('playhead');
    const detectionsEl = document.getElementById('detections');

    let state = {
      sources: [],
      sourceIndex: 0,
      modelName: 'kabum_v2',
      duration: 0,
      peaks: [],
      detections: [],
      playhead: 0,
      viewStart: 0,
      viewDuration: 0,
      isDraggingWave: false,
      isDraggingSeek: false,
      isDraggingPan: false,
      followPlayhead: true,
      pendingSeekTime: null,
      pendingSeekUntil: 0,
      dragStartX: 0,
      dragStartView: 0
    };

    let availableModels = [];

    function displayModelName(id) {
      const model = availableModels.find(item => item.id === id);
      return model?.display_name || id;
    }

    async function load() {
      statusEl.textContent = 'Loading sources and models';
      const [sourcesRes, modelsRes] = await Promise.all([
        fetch('/api/sources').then(r => r.json()),
        fetch('/api/models').then(r => r.json()).catch(() => null)
      ]);
      state.sources = sourcesRes.sources;
      sourceSelect.innerHTML = state.sources.map(source => `<option value="${source.index}">${source.path}</option>`).join('');
      if (modelsRes && modelsRes.models) {
        availableModels = modelsRes.models;
        testModelSelect.innerHTML = availableModels.map(m =>
          `<option value="${m.id}" ${m.id === (modelsRes.default || 'kabum_v2') ? 'selected' : ''}>${m.display_name || m.id} (${m.status})</option>`
        ).join('');
        const chosen = availableModels.find(m => m.id === (modelsRes.default || 'kabum_v2'));
        if (chosen && chosen.recommended_threshold) {
          thresholdInput.value = String(chosen.recommended_threshold);
        }
      }
      await loadSource(sourcesRes.latest_index || 0);
    }

    async function loadSource(index) {
      state.sourceIndex = Number(index);
      state.modelName = testModelSelect.value;
      sourceSelect.value = String(state.sourceIndex);
      const source = state.sources[state.sourceIndex];
      if (!source) return;
      statusEl.textContent = 'Loading full audio and model detections';
      audio.src = source.audio;
      audio.currentTime = 0;
      audio.playbackRate = Number(speedSelect.value || 1);
      state.playhead = 0;
      state.viewStart = 0;

      const audioResponse = await fetch(source.audio);
      const arrayBuffer = await audioResponse.arrayBuffer();
      const audioContext = new AudioContext();
      const decoded = await audioContext.decodeAudioData(arrayBuffer.slice(0));
      state.duration = decoded.duration;
      state.peaks = makePeaks(decoded.getChannelData(0), 2200);
      updateViewFromZoom(true);

      const threshold = Number(thresholdInput.value || 0.5);
      const detectionsResponse = await fetch(`/api/test-detections?source_index=${state.sourceIndex}&threshold=${threshold}&model=${encodeURIComponent(state.modelName)}`);
      const detectionsPayload = await detectionsResponse.json();
      state.detections = detectionsPayload.detections || [];

      renderDetections();
      draw();
      statusEl.textContent = 'Test mode only marks detections; it does not reduce, cut, or alter audio.';
    }

    function updateViewFromZoom(resetPan = false) {
      const zoom = Number(zoomSlider.value || 1);
      state.viewDuration = Math.max(0.25, state.duration / zoom);
      if (resetPan) {
        state.viewStart = 0;
      } else {
        state.viewStart = clamp(state.viewStart, 0, maxViewStart());
      }
      panSlider.max = String(maxViewStart());
      panSlider.step = String(Math.max(0.001, state.duration / 1000));
      panSlider.value = String(state.viewStart);
    }

    function maxViewStart() {
      return Math.max(0, state.duration - state.viewDuration);
    }

    function setViewStart(value) {
      state.viewStart = clamp(value, 0, maxViewStart());
      panSlider.value = String(state.viewStart);
      draw();
    }

    function clamp(value, min, max) {
      return Math.max(min, Math.min(max, value));
    }

    function makePeaks(samples, width) {
      const block = Math.max(1, Math.floor(samples.length / width));
      const peaks = [];
      for (let i = 0; i < width; i++) {
        let min = 0;
        let max = 0;
        const start = i * block;
        const end = Math.min(samples.length, start + block);
        for (let j = start; j < end; j++) {
          const value = samples[j];
          if (value < min) min = value;
          if (value > max) max = value;
        }
        peaks.push([min, max]);
      }
      return peaks;
    }

    function resizeCanvas() {
      const rect = canvas.getBoundingClientRect();
      const scale = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * scale));
      canvas.height = Math.max(1, Math.floor(rect.height * scale));
      ctx.setTransform(scale, 0, 0, scale, 0, 0);
    }

    function draw() {
      resizeCanvas();
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);
      ctx.fillStyle = '#fbfbfc';
      ctx.fillRect(0, 0, rect.width, rect.height);

      drawDetectionMarkers(state.detections, rect, '#dc2626', 16);

      const mid = rect.height / 2;
      ctx.strokeStyle = '#0f766e';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let x = 0; x < rect.width; x++) {
        const time = state.viewStart + (x / Math.max(1, rect.width)) * state.viewDuration;
        const index = Math.floor((time / Math.max(0.001, state.duration)) * state.peaks.length);
        const peak = state.peaks[index] || [0, 0];
        ctx.moveTo(x, mid + peak[0] * mid * 0.92);
        ctx.lineTo(x, mid + peak[1] * mid * 0.92);
      }
      ctx.stroke();

      drawDetectionEdges(state.detections, rect, '#dc2626');

      const playheadX = timeToX(state.playhead, rect.width);
      ctx.strokeStyle = '#111827';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(playheadX, 0);
      ctx.lineTo(playheadX, rect.height);
      ctx.stroke();

      durationEl.textContent = `duration: ${state.duration.toFixed(3)}s`;
      summaryEl.textContent = `${displayModelName(state.modelName)} detections: ${state.detections.length}`;
      document.getElementById('legendModel').textContent = displayModelName(state.modelName);
      playheadEl.textContent = `playhead: ${state.playhead.toFixed(3)}s`;
      currentTimeEl.textContent = `${state.playhead.toFixed(3)}s`;
      totalTimeEl.textContent = `${state.duration.toFixed(3)}s`;
      seekSlider.max = String(Math.max(0.001, state.duration));
      if (!state.isDraggingSeek) seekSlider.value = String(state.playhead);
    }

    function drawDetectionMarkers(detections, rect, color, y) {
      for (const detection of detections) {
        const x1 = timeToX(detection.start, rect.width);
        const x2 = timeToX(detection.end, rect.width);
        if (x2 < 0 || x1 > rect.width) continue;
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.86;
        ctx.fillRect(Math.max(0, x1), y, Math.max(3, Math.min(rect.width, x2) - Math.max(0, x1)), 7);
        ctx.globalAlpha = 1;
      }
    }

    function drawDetectionEdges(detections, rect, color) {
      for (const detection of detections) {
        const x1 = timeToX(detection.start, rect.width);
        const x2 = timeToX(detection.end, rect.width);
        if (x2 < 0 || x1 > rect.width) continue;
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 4]);
        ctx.beginPath();
        ctx.moveTo(x1, 0);
        ctx.lineTo(x1, rect.height);
        ctx.moveTo(x2, 0);
        ctx.lineTo(x2, rect.height);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    function timeToX(time, width) {
      return ((time - state.viewStart) / Math.max(0.001, state.viewDuration)) * width;
    }

    function xToTime(clientX) {
      const rect = canvas.getBoundingClientRect();
      const x = Math.max(0, Math.min(rect.width, clientX - rect.left));
      return clamp(state.viewStart + (x / Math.max(1, rect.width)) * state.viewDuration, 0, state.duration);
    }

    function renderDetections() {
      const items = state.detections.map((detection, idx) => detectionRow(state.modelName, idx, detection)).join('');
      detectionsEl.innerHTML = `
        <div><strong>${displayModelName(state.modelName)} (${state.detections.length})</strong></div>
        ${items}
      `;
    }

    function detectionRow(kind, idx, detection) {
      return `
        <button class="detection" type="button" onclick="seekTo(${detection.start})">
          <strong>${kind} ${idx + 1}. ${detection.start.toFixed(3)}s-${detection.end.toFixed(3)}s</strong>
          <span>${(detection.model_probability * 100).toFixed(1)}%</span>
          <span>label: ${detection.label || '-'}</span>
        </button>
      `;
    }

    function seekTo(time, follow = true) {
      const nextTime = clamp(time, 0, state.duration);
      state.playhead = nextTime;
      state.followPlayhead = follow;
      state.pendingSeekTime = nextTime;
      state.pendingSeekUntil = performance.now() + 1200;
      try {
        audio.currentTime = nextTime;
      } catch {
        state.pendingSeekTime = null;
      }
      if (follow) keepPlayheadVisible();
      draw();
    }

    function syncPlayheadFromAudio() {
      if (state.pendingSeekTime !== null) {
        const closeEnough = Math.abs(audio.currentTime - state.pendingSeekTime) < 0.12;
        const expired = performance.now() > state.pendingSeekUntil;
        if (!closeEnough && !expired) {
          state.playhead = state.pendingSeekTime;
          draw();
          return;
        }
        state.pendingSeekTime = null;
      }
      state.playhead = audio.currentTime;
      keepPlayheadVisible();
      draw();
    }

    function keepPlayheadVisible() {
      if (!state.followPlayhead || state.isDraggingPan || state.isDraggingWave || state.isDraggingSeek) return;
      if (state.playhead < state.viewStart) {
        setViewStart(state.playhead);
      } else if (state.playhead > state.viewStart + state.viewDuration) {
        setViewStart(state.playhead - state.viewDuration * 0.85);
      }
    }

    canvas.addEventListener('pointerdown', event => {
      state.isDraggingWave = true;
      state.followPlayhead = false;
      state.dragStartX = event.clientX;
      state.dragStartView = state.viewStart;
      canvas.classList.add('dragging');
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener('pointermove', event => {
      if (!state.isDraggingWave) return;
      const rect = canvas.getBoundingClientRect();
      const deltaSeconds = ((event.clientX - state.dragStartX) / Math.max(1, rect.width)) * state.viewDuration;
      setViewStart(state.dragStartView - deltaSeconds);
    });
    canvas.addEventListener('pointerup', event => {
      if (!state.isDraggingWave) return;
      const moved = Math.abs(event.clientX - state.dragStartX);
      state.isDraggingWave = false;
      canvas.classList.remove('dragging');
      if (moved < 5) seekTo(xToTime(event.clientX));
    });
    canvas.addEventListener('pointerleave', () => {
      if (!state.isDraggingWave) return;
      state.isDraggingWave = false;
      canvas.classList.remove('dragging');
    });
    audio.addEventListener('timeupdate', () => {
      state.playhead = audio.currentTime;
      keepPlayheadVisible();
      draw();
    });
    audio.addEventListener('seeking', () => {
      state.playhead = audio.currentTime;
      keepPlayheadVisible();
      draw();
    });
    audio.addEventListener('play', () => {
      playPauseButton.textContent = 'Pause';
    });
    audio.addEventListener('pause', () => {
      playPauseButton.textContent = 'Play';
    });
    playPauseButton.addEventListener('click', async () => {
      if (audio.paused) {
        await audio.play();
      } else {
        audio.pause();
      }
    });
    stopButton.addEventListener('click', () => {
      audio.pause();
      seekTo(0);
    });
    function startSeekDrag() {
      state.isDraggingSeek = true;
      state.followPlayhead = true;
    }

    function commitSeekDrag() {
      state.isDraggingSeek = false;
      seekTo(Number(seekSlider.value), true);
    }

    seekSlider.addEventListener('pointerdown', startSeekDrag);
    seekSlider.addEventListener('mousedown', startSeekDrag);
    seekSlider.addEventListener('touchstart', startSeekDrag);
    seekSlider.addEventListener('input', event => {
      state.isDraggingSeek = true;
      seekTo(Number(event.target.value), false);
    });
    seekSlider.addEventListener('change', commitSeekDrag);
    seekSlider.addEventListener('pointerup', commitSeekDrag);
    seekSlider.addEventListener('mouseup', commitSeekDrag);
    seekSlider.addEventListener('touchend', commitSeekDrag);
    seekSlider.addEventListener('blur', () => {
      if (state.isDraggingSeek) commitSeekDrag();
    });
    speedSelect.addEventListener('change', () => {
      audio.playbackRate = Number(speedSelect.value || 1);
    });
    zoomSlider.addEventListener('input', () => {
      const center = state.playhead || (state.viewStart + state.viewDuration / 2);
      updateViewFromZoom(false);
      setViewStart(center - state.viewDuration / 2);
    });
    panSlider.addEventListener('pointerdown', () => {
      state.isDraggingPan = true;
      state.followPlayhead = false;
    });
    panSlider.addEventListener('input', event => {
      state.followPlayhead = false;
      setViewStart(Number(event.target.value));
    });
    panSlider.addEventListener('pointerup', () => {
      state.isDraggingPan = false;
    });
    panSlider.addEventListener('change', () => {
      state.isDraggingPan = false;
    });
    sourceSelect.addEventListener('change', event => loadSource(Number(event.target.value)));
    testModelSelect.addEventListener('change', () => {
      const chosen = availableModels.find(m => m.id === testModelSelect.value);
      if (chosen && chosen.recommended_threshold) {
        thresholdInput.value = String(chosen.recommended_threshold);
      }
      loadSource(state.sourceIndex);
    });
    thresholdInput.addEventListener('change', () => loadSource(state.sourceIndex));
    reloadButton.addEventListener('click', () => loadSource(state.sourceIndex));
    window.addEventListener('resize', draw);
    window.seekTo = seekTo;
    load();
  </script>
</body>
</html>
"""


MODEL_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Editto</title>
  <style>
    :root {
      --bg: #f0f2f5;
      --panel: #ffffff;
      --ink: #111827;
      --muted: #6b7280;
      --line: #e5e7eb;
      --accent: #0f766e;
      --accent-weak: #ccfaf4;
      --danger: #dc2626;
      --danger-weak: #fee2e2;
      --warn: #8a5a00;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, Arial, sans-serif;
      font-size: 14px;
      letter-spacing: 0;
      overflow-x: hidden;
    }
    header {
      min-height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 20px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      box-shadow: 0 1px 4px rgba(0,0,0,.07);
    }
    h1 { margin: 0; font-size: 17px; }
    a { color: var(--accent); font-weight: 700; text-decoration: none; padding: 4px 10px; border-radius: 6px; }
    a:hover { background: var(--accent-weak); }
    main {
      max-width: 1220px;
      margin: 0 auto;
      padding: 20px 22px;
    }
    .toolbar, .modebar {
      display: grid;
      gap: 10px;
      margin-bottom: 12px;
    }
    .toolbar { grid-template-columns: minmax(150px, .8fr) minmax(240px, 1.8fr) 140px 120px 120px; }
    .modebar { grid-template-columns: repeat(auto-fit, minmax(145px, 1fr)); }
    .uploadzone {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      padding: 12px 16px;
      margin-bottom: 12px;
      border: 2px dashed var(--line);
      border-radius: 10px;
      background: var(--panel);
      transition: border-color .15s, background .15s;
      flex-wrap: wrap;
    }
    .uploadzone.dragover { border-color: var(--accent); background: var(--accent-weak); }
    .uploadzone .uz-title { font-weight: 700; }
    .uploadzone .uz-copy { color: var(--muted); font-size: 12px; margin-top: 2px; }
    select, input, button {
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      font-weight: 650;
      padding: 0 10px;
    }
    button { cursor: pointer; }
    button:hover { border-color: var(--accent); }
    button.active {
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
    }
    button.danger.active, button.danger:hover {
      border-color: var(--danger);
      background: var(--danger);
      color: #fff;
    }
    audio { display: none; }
    .transport {
      display: grid;
      grid-template-columns: 64px 64px minmax(170px, 1fr) 96px;
      gap: 8px;
      align-items: center;
      margin: 10px 0 14px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      box-shadow: 0 1px 4px rgba(0,0,0,.05);
    }
    .icon-button {
      min-width: 58px;
      padding: 0;
      line-height: 1;
    }
    .seek { display: grid; gap: 5px; }
    .seek input, .zoom-row input { width: 100%; padding: 0; }
    .time-line {
      display: flex;
      justify-content: space-between;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }
    .panel, .output {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 14px;
      box-shadow: 0 1px 4px rgba(0,0,0,.06);
    }
    canvas {
      display: block;
      width: 100%;
      height: 300px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fbfbfc;
      cursor: grab;
      user-select: none;
    }
    canvas.eraser { cursor: crosshair; }
    canvas.dragging { cursor: grabbing; }
    .zoom-row {
      display: grid;
      grid-template-columns: 72px minmax(140px, 1fr) 72px minmax(140px, 1fr);
      gap: 8px;
      align-items: center;
      margin-top: 12px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    .readout {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
    }
    .legend {
      display: flex;
      gap: 12px;
      align-items: center;
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    .legend span::before {
      content: "";
      display: inline-block;
      width: 14px;
      height: 9px;
      margin-right: 6px;
      border-radius: 2px;
      background: rgba(220, 38, 38, 0.60);
    }
    .detections {
      margin-top: 14px;
      display: grid;
      gap: 6px;
      max-height: 280px;
      overflow: auto;
    }
    .detection {
      min-height: 36px;
      display: grid;
      grid-template-columns: 150px 90px 1fr 72px 88px;
      gap: 10px;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fff;
      padding: 6px 10px;
      text-align: left;
    }
    .detection strong { color: var(--danger); }
    .detection button {
      min-height: 28px;
    }
    .detection .remove {
      color: var(--danger);
      background: var(--danger-weak);
      border-color: var(--danger-weak);
    }
    .status { color: var(--muted); min-height: 22px; margin-top: 10px; }
    .safety-note {
      margin: 0 0 12px;
      padding: 10px 12px;
      border: 1px solid #bae6fd;
      border-radius: 8px;
      background: #f0f9ff;
      color: #0c4a6e;
      line-height: 1.45;
    }
    .field { display: grid; gap: 4px; min-width: 0; }
    .field > span { color: var(--muted); font-size: 11px; font-weight: 750; }
    .field select, .field input { width: 100%; min-width: 0; }
    .output {
      margin-top: 14px;
      display: none;
      gap: 10px;
    }
    .output.visible { display: grid; }
    .output audio {
      display: block;
      width: 100%;
      height: 42px;
    }
    .output-meta {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    @media (max-width: 820px) {
      .toolbar, .modebar, .zoom-row { grid-template-columns: 1fr; }
      .transport { grid-template-columns: 64px 64px 1fr; }
      .transport select { grid-column: 1 / -1; }
      canvas { height: 240px; }
      .detection { grid-template-columns: 1fr; }
      .readout, .output-meta { flex-direction: column; }
      header { align-items: flex-start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Editto</h1>
    <span>
      <a href="/">Eğitim Klipleri</a>
      <a href="/manual">Elle Seçim</a>
      <a href="/test">Algılama Testi</a>
    </span>
  </header>
  <main>
    <div id="uploadZone" class="uploadzone">
      <div>
        <div class="uz-title">Ses veya video dosyanı buraya bırak</div>
        <div class="uz-copy">Dosya eklenince nefes adayları otomatik bulunur. Kaynak dosya değiştirilmez.</div>
      </div>
      <div>
        <input id="filePicker" type="file" multiple accept="audio/*,video/*" hidden>
        <button id="pickFiles" type="button">Dosya Seç</button>
      </div>
    </div>
    <div class="safety-note"><strong>Güvenli başlangıç:</strong> “Akıllı Hafif” ve 0,85 eşik varsayılandır. Kırmızı bölgeleri dinleyip yanlış olanları kaldır. Çıktı her zaman yeni bir WAV dosyasına yazılır.</div>
    <div id="modelNotice" class="safety-note" hidden></div>
    <div class="toolbar">
      <label class="field"><span>KAYIT ARA</span><input id="sourceSearch" type="search" placeholder="Dosya adı yaz…"></label>
      <label class="field"><span>KAYNAK KAYIT</span><select id="source"></select></label>
      <label class="field"><span>MODEL</span><select id="modelName">
        <option value="kabum_v2" selected>Editto Precision v2</option>
        <option value="kabum_v1">Editto Precision v1</option>
        <option value="kabum">Editto Classic</option>
        <option value="babum">Editto Legacy</option>
      </select></label>
      <label class="field"><span>GÜVEN EŞİĞİ</span><input id="threshold" type="number" min="0.5" max="0.99" step="0.05" value="0.75" title="Yüksek değer daha az ama daha güvenli kesit seçer"></label>
      <button id="reload" type="button">Yeniden Algıla</button>
    </div>
    <div class="modebar">
      <button class="mode active" type="button" data-mode="gentle">Akıllı Hafif (-8…-16 dB)</button>
      <button class="mode" type="button" data-mode="strong">Akıllı Güçlü (-16…-28 dB)</button>
      <button class="mode" type="button" data-mode="mute">Sessize Al</button>
      <button class="mode" type="button" data-mode="cut">Tamamen Kes (riskli)</button>
      <button id="eraser" class="danger" type="button">Yanlışı Sil</button>
      <button id="restore" type="button">Seçimleri Geri Getir</button>
      <button id="process" class="active" type="button">Çıktı Oluştur</button>
    </div>
    <audio id="audio"></audio>
    <div class="transport">
      <button id="playPause" class="icon-button" type="button" title="Oynat/Duraklat">Oynat</button>
      <button id="stop" class="icon-button" type="button" title="Durdur">Durdur</button>
      <div class="seek">
        <input id="seekSlider" type="range" min="0" max="1" step="0.001" value="0" title="Seek">
        <div class="time-line">
          <span id="currentTime">0.000s</span>
          <span id="totalTime">0.000s</span>
        </div>
      </div>
      <select id="speed" title="Playback speed">
        <option value="0.5">0.5x</option>
        <option value="0.75">0.75x</option>
        <option value="1" selected>1x</option>
        <option value="1.25">1.25x</option>
        <option value="1.5">1.5x</option>
      </select>
    </div>
    <div class="panel">
      <canvas id="wave"></canvas>
      <div class="zoom-row">
        <span>Yakınlaştır</span>
        <input id="zoomSlider" type="range" min="1" max="40" step="0.5" value="1">
        <span>Görünüm</span>
        <input id="panSlider" type="range" min="0" max="0" step="0.001" value="0">
      </div>
      <div class="readout">
        <span id="duration">süre: -</span>
        <span id="summary">tespit: -</span>
        <span id="playhead">konum: 0.000s</span>
      </div>
      <div class="legend">
        <span id="legendModel">Editto Precision v2 nefes adayları</span>
      </div>
    </div>
    <div id="status" class="status"></div>
    <div id="output" class="output">
      <audio id="outputAudio" controls></audio>
      <div class="output-meta">
        <span id="outputPath"></span>
        <a id="downloadOutput" href="#" download>WAV Dosyasını İndir</a>
      </div>
    </div>
    <div id="detections" class="detections"></div>
  </main>
  <script>
    const sourceSelect = document.getElementById('source');
    const sourceSearch = document.getElementById('sourceSearch');
    const modelNameSelect = document.getElementById('modelName');
    const thresholdInput = document.getElementById('threshold');
    const reloadButton = document.getElementById('reload');
    const uploadZone = document.getElementById('uploadZone');
    const filePicker = document.getElementById('filePicker');
    const pickFiles = document.getElementById('pickFiles');
    const modeButtons = Array.from(document.querySelectorAll('.mode'));
    const eraserButton = document.getElementById('eraser');
    const restoreButton = document.getElementById('restore');
    const processButton = document.getElementById('process');
    const audio = document.getElementById('audio');
    const playPauseButton = document.getElementById('playPause');
    const stopButton = document.getElementById('stop');
    const seekSlider = document.getElementById('seekSlider');
    const currentTimeEl = document.getElementById('currentTime');
    const totalTimeEl = document.getElementById('totalTime');
    const speedSelect = document.getElementById('speed');
    const zoomSlider = document.getElementById('zoomSlider');
    const panSlider = document.getElementById('panSlider');
    const canvas = document.getElementById('wave');
    const ctx = canvas.getContext('2d');
    const statusEl = document.getElementById('status');
    const durationEl = document.getElementById('duration');
    const summaryEl = document.getElementById('summary');
    const playheadEl = document.getElementById('playhead');
    const detectionsEl = document.getElementById('detections');
    const outputEl = document.getElementById('output');
    const outputAudio = document.getElementById('outputAudio');
    const outputPathEl = document.getElementById('outputPath');
    const downloadOutput = document.getElementById('downloadOutput');

    let state = {
      sources: [],
      sourceIndex: 0,
      modelName: 'kabum_v2',
      mode: 'gentle',
      duration: 0,
      peaks: [],
      detections: [],
      originalDetections: [],
      playhead: 0,
      viewStart: 0,
      viewDuration: 0,
      eraser: false,
      isDraggingWave: false,
      isDraggingSeek: false,
      isDraggingPan: false,
      followPlayhead: true,
      pendingSeekTime: null,
      pendingSeekUntil: 0,
      pendingAutoplay: false,
      dragStartX: 0,
      dragStartView: 0
    };

    function displayModelName(id) {
      const model = (state.models || []).find(item => item.id === id);
      return model?.display_name || id;
    }

    async function load() {
      statusEl.textContent = 'Kayıtlar yükleniyor…';
      const modelsResponse = await fetch('/api/models');
      if (!modelsResponse.ok) throw new Error('Model listesi alınamadı.');
      const modelPayload = await modelsResponse.json();
      state.models = modelPayload.models;
      modelNameSelect.replaceChildren(...state.models.map(model => {
        let label = model.display_name || model.id;
        if (model.id === modelPayload.default) label += ' — Önerilen';
        const option = new Option(label, model.id);
        return option;
      }));
      modelNameSelect.value = modelPayload.default;
      const defaultModel = state.models.find(model => model.id === modelPayload.default);
      if (defaultModel && defaultModel.recommended_threshold) {
        thresholdInput.value = String(defaultModel.recommended_threshold);
      }
      const response = await fetch('/api/sources');
      if (!response.ok) throw new Error('Kayıt listesi alınamadı.');
      const payload = await response.json();
      state.sources = payload.sources;
      renderSourceOptions();
      await loadSource(payload.latest_index || 0);
    }

    function isTrainingLibrary(source) {
      return source.path.toLowerCase().includes('dataset\\external\\');
    }

    function renderSourceOptions(query = '') {
      const normalized = query.trim().toLocaleLowerCase('tr-TR');
      let visible = state.sources.filter(source => normalized
        ? source.path.toLocaleLowerCase('tr-TR').includes(normalized)
        : !isTrainingLibrary(source));
      const selected = state.sources.find(source => source.index === state.sourceIndex);
      if (selected && !visible.some(source => source.index === selected.index)) visible = [selected, ...visible];
      sourceSelect.innerHTML = visible
        .slice(0, 300)
        .map(source => `<option value="${source.index}">${source.path}</option>`)
        .join('');
      if (selected) sourceSelect.value = String(selected.index);
      sourceSearch.title = normalized && visible.length > 300
        ? `${visible.length} eşleşmeden ilk 300 tanesi gösteriliyor.`
        : `${visible.length} kayıt gösteriliyor.`;
    }

    async function loadSource(index) {
      state.sourceIndex = Number(index);
      state.modelName = modelNameSelect.value;
      sourceSelect.value = String(state.sourceIndex);
      const source = state.sources[state.sourceIndex];
      if (!source) return;
      statusEl.textContent = 'Ses ve nefes adayları yükleniyor…';
      processButton.disabled = true;
      state.detections = [];
      document.querySelectorAll('.toolbar select, .toolbar input, #reload').forEach(el => el.disabled = true);
      const notice = document.getElementById('modelNotice');
      const cnnCandidate = state.modelName === 'breath_cnn_v2' || state.modelName === 'breath_cnn_personal_v3';
      notice.hidden = !cnnCandidate;
      notice.textContent = state.modelName === 'breath_cnn_personal_v3'
        ? 'Kişisel CNN v3 ek aday bulucudur. Dört kayda göre iyileşti ama konuşma yanlış alarmı ve sınırlar bağımsız kayıtta doğrulanmadı; tüm bölgeleri dinle.'
        : 'CNN v2 deneysel adaydır. Tüm kayıt taranır; işaretli bölgeleri dinleyerek kontrol et. Genel kullanım ve kesim sınırları henüz doğrulanmadı.';
      outputEl.classList.remove('visible');
      audio.src = source.audio;
      audio.currentTime = 0;
      audio.playbackRate = Number(speedSelect.value || 1);
      state.playhead = 0;
      state.viewStart = 0;

      try {
        const audioResponse = await fetch(source.audio);
        if (!audioResponse.ok) throw new Error('Ses dosyası okunamadı.');
        const arrayBuffer = await audioResponse.arrayBuffer();
        const audioContext = new AudioContext();
        const decoded = await audioContext.decodeAudioData(arrayBuffer.slice(0));
        state.duration = decoded.duration;
        state.peaks = makePeaks(decoded.getChannelData(0), 2400);
        await audioContext.close();
        updateViewFromZoom(true);
        await loadDetections();
        statusEl.textContent = 'Hazır. Kırmızı bölgeleri dinle; yanlış olanları sil ve çıktıyı oluştur.';
        processButton.disabled = false;
      } catch (error) {
        state.duration = 0;
        state.peaks = [];
        state.detections = [];
        renderDetections();
        statusEl.textContent = `Yükleme hatası: ${error.message || error}`;
      } finally {
        document.querySelectorAll('.toolbar select, .toolbar input, #reload').forEach(el => el.disabled = false);
      }
    }

    async function loadDetections() {
      const threshold = Number(thresholdInput.value || 0.75);
      const response = await fetch(`/api/test-detections?source_index=${state.sourceIndex}&threshold=${threshold}&model=${encodeURIComponent(state.modelName)}&respect_labels=0`);
      if (!response.ok) throw new Error('Model tespitleri alınamadı.');
      const payload = await response.json();
      state.detections = (payload.detections || []).map((item, index) => ({ ...item, localId: `${Date.now()}_${index}` }));
      state.originalDetections = state.detections.map(item => ({ ...item }));
      renderDetections();
      draw();
    }

    function makePeaks(samples, width) {
      const block = Math.max(1, Math.floor(samples.length / width));
      const peaks = [];
      for (let i = 0; i < width; i++) {
        let min = 0;
        let max = 0;
        const start = i * block;
        const end = Math.min(samples.length, start + block);
        for (let j = start; j < end; j++) {
          const value = samples[j];
          if (value < min) min = value;
          if (value > max) max = value;
        }
        peaks.push([min, max]);
      }
      return peaks;
    }

    function updateViewFromZoom(resetPan = false) {
      const zoom = Number(zoomSlider.value || 1);
      state.viewDuration = Math.max(0.25, state.duration / zoom);
      state.viewStart = resetPan ? 0 : clamp(state.viewStart, 0, maxViewStart());
      panSlider.max = String(maxViewStart());
      panSlider.step = String(Math.max(0.001, state.duration / 1000));
      if (!state.isDraggingPan) panSlider.value = String(state.viewStart);
    }

    function maxViewStart() {
      return Math.max(0, state.duration - state.viewDuration);
    }

    function setViewStart(value) {
      state.viewStart = clamp(value, 0, maxViewStart());
      if (!state.isDraggingPan) panSlider.value = String(state.viewStart);
      draw();
    }

    function clamp(value, min, max) {
      return Math.max(min, Math.min(max, value));
    }

    function resizeCanvas() {
      const rect = canvas.getBoundingClientRect();
      const scale = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * scale));
      canvas.height = Math.max(1, Math.floor(rect.height * scale));
      ctx.setTransform(scale, 0, 0, scale, 0, 0);
    }

    function draw() {
      resizeCanvas();
      canvas.classList.toggle('eraser', state.eraser);
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);
      ctx.fillStyle = '#fbfbfc';
      ctx.fillRect(0, 0, rect.width, rect.height);

      drawDetectionMarkers(rect);
      drawWaveform(rect);
      drawDetectionEdges(rect);
      drawPlayhead(rect);

      durationEl.textContent = `süre: ${state.duration.toFixed(3)} sn`;
      summaryEl.textContent = `${displayModelName(state.modelName)}: ${state.detections.length} etkin nefes adayı`;
      document.getElementById('legendModel').textContent = `${displayModelName(state.modelName)} nefes adayları`;
      playheadEl.textContent = `konum: ${state.playhead.toFixed(3)} sn`;
      currentTimeEl.textContent = `${state.playhead.toFixed(3)}s`;
      totalTimeEl.textContent = `${state.duration.toFixed(3)}s`;
      seekSlider.max = String(Math.max(0.001, state.duration));
      if (!state.isDraggingSeek) seekSlider.value = String(state.playhead);
    }

    function drawWaveform(rect) {
      const mid = rect.height / 2;
      ctx.strokeStyle = '#0f766e';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let x = 0; x < rect.width; x++) {
        const time = state.viewStart + (x / Math.max(1, rect.width)) * state.viewDuration;
        const index = Math.floor((time / Math.max(0.001, state.duration)) * state.peaks.length);
        const peak = state.peaks[index] || [0, 0];
        ctx.moveTo(x, mid + peak[0] * mid * 0.92);
        ctx.lineTo(x, mid + peak[1] * mid * 0.92);
      }
      ctx.stroke();
    }

    function drawDetectionMarkers(rect) {
      for (const detection of state.detections) {
        const x1 = timeToX(detection.start, rect.width);
        const x2 = timeToX(detection.end, rect.width);
        if (x2 < 0 || x1 > rect.width) continue;
        ctx.fillStyle = '#dc2626';
        ctx.globalAlpha = 0.70;
        ctx.fillRect(Math.max(0, x1), 14, Math.max(4, Math.min(rect.width, x2) - Math.max(0, x1)), rect.height - 28);
        ctx.globalAlpha = 1;
      }
    }

    function drawDetectionEdges(rect) {
      for (const detection of state.detections) {
        const x1 = timeToX(detection.start, rect.width);
        const x2 = timeToX(detection.end, rect.width);
        if (x2 < 0 || x1 > rect.width) continue;
        ctx.strokeStyle = '#b91c1c';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 4]);
        ctx.beginPath();
        ctx.moveTo(x1, 0);
        ctx.lineTo(x1, rect.height);
        ctx.moveTo(x2, 0);
        ctx.lineTo(x2, rect.height);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    function drawPlayhead(rect) {
      const playheadX = timeToX(state.playhead, rect.width);
      ctx.strokeStyle = '#111827';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(playheadX, 0);
      ctx.lineTo(playheadX, rect.height);
      ctx.stroke();
    }

    function timeToX(time, width) {
      return ((time - state.viewStart) / Math.max(0.001, state.viewDuration)) * width;
    }

    function xToTime(clientX) {
      const rect = canvas.getBoundingClientRect();
      const x = Math.max(0, Math.min(rect.width, clientX - rect.left));
      return clamp(state.viewStart + (x / Math.max(1, rect.width)) * state.viewDuration, 0, state.duration);
    }

    function renderDetections() {
      const sorted = state.detections
        .map((detection, index) => ({ detection, index }))
        .sort((a, b) => {
          const pa = Number(a.detection.model_probability ?? 0);
          const pb = Number(b.detection.model_probability ?? 0);
          if (pa !== pb) return pa - pb;
          return Number(a.detection.start) - Number(b.detection.start);
        });
      const items = sorted.map((item, order) => detectionRow(item.index, item.detection, order)).join('');
      detectionsEl.innerHTML = `
        <div><strong>Etkin bölgeler (${state.detections.length}) — en düşük güvenli olanlar önce</strong></div>
        ${items || '<div class="status">Etkin nefes adayı kalmadı.</div>'}
      `;
      draw();
    }

    function detectionRow(index, detection, order) {
      return `
        <div class="detection">
          <strong>${order + 1}. ${detection.start.toFixed(3)}s-${detection.end.toFixed(3)}s</strong>
          <span>${(detection.model_probability * 100).toFixed(1)}%</span>
          <span>etiket: ${detection.label || '-'}</span>
          <button type="button" onclick="seekTo(${detection.start}, true, true)">Dinle</button>
          <button class="remove" type="button" onclick="removeDetection(${index})">Kaldır</button>
        </div>
      `;
    }

    function removeDetection(index) {
      const removed = state.detections.splice(index, 1)[0];
      renderDetections();
      statusEl.textContent = removed ? `${removed.start.toFixed(3)}-${removed.end.toFixed(3)} sn bölgesi kaldırıldı.` : 'Kaldırılacak bölge yok.';
    }

    function removeDetectionAtTime(time) {
      const tolerance = Math.max(0.025, state.viewDuration / Math.max(1, canvas.getBoundingClientRect().width) * 8);
      const index = state.detections.findIndex(detection => time >= detection.start - tolerance && time <= detection.end + tolerance);
      if (index >= 0) removeDetection(index);
      else statusEl.textContent = 'Silginin altında bir nefes adayı yok.';
    }

    function seekTo(time, follow = true, autoplay = false) {
      const nextTime = clamp(time, 0, state.duration);
      state.playhead = nextTime;
      state.followPlayhead = follow;
      state.pendingSeekTime = nextTime;
      state.pendingSeekUntil = performance.now() + 1200;
      state.pendingAutoplay = autoplay;
      try {
        audio.currentTime = nextTime;
      } catch {
        state.pendingSeekTime = null;
      }
      if (autoplay) {
        const playResult = audio.play();
        if (playResult && typeof playResult.catch === 'function') playResult.catch(() => {});
      }
      if (follow) keepPlayheadVisible();
      draw();
    }

    function syncPlayheadFromAudio() {
      if (state.pendingSeekTime !== null) {
        const closeEnough = Math.abs(audio.currentTime - state.pendingSeekTime) < 0.12;
        const expired = performance.now() > state.pendingSeekUntil;
        if (!closeEnough && !expired) {
          state.playhead = state.pendingSeekTime;
          draw();
          return;
        }
        state.pendingSeekTime = null;
      }
      state.playhead = audio.currentTime;
      keepPlayheadVisible();
      draw();
    }

    function keepPlayheadVisible() {
      if (!state.followPlayhead || state.isDraggingPan || state.isDraggingWave || state.isDraggingSeek) return;
      if (state.playhead < state.viewStart) {
        setViewStart(state.playhead);
      } else if (state.playhead > state.viewStart + state.viewDuration) {
        setViewStart(state.playhead - state.viewDuration * 0.85);
      }
    }

    canvas.addEventListener('pointerdown', event => {
      if (state.eraser) {
        removeDetectionAtTime(xToTime(event.clientX));
        return;
      }
      state.isDraggingWave = true;
      state.followPlayhead = false;
      state.dragStartX = event.clientX;
      state.dragStartView = state.viewStart;
      canvas.classList.add('dragging');
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener('pointermove', event => {
      if (!state.isDraggingWave) return;
      const rect = canvas.getBoundingClientRect();
      const deltaSeconds = ((event.clientX - state.dragStartX) / Math.max(1, rect.width)) * state.viewDuration;
      setViewStart(state.dragStartView - deltaSeconds);
    });
    canvas.addEventListener('pointerup', event => {
      if (!state.isDraggingWave) return;
      const moved = Math.abs(event.clientX - state.dragStartX);
      state.isDraggingWave = false;
      canvas.classList.remove('dragging');
      if (moved < 5) seekTo(xToTime(event.clientX));
    });
    canvas.addEventListener('pointerleave', () => {
      if (!state.isDraggingWave) return;
      state.isDraggingWave = false;
      canvas.classList.remove('dragging');
    });

    audio.addEventListener('timeupdate', () => {
      if (state.isDraggingSeek) return;
      syncPlayheadFromAudio();
    });
    audio.addEventListener('seeking', () => {
      if (state.isDraggingSeek) return;
      syncPlayheadFromAudio();
    });
    audio.addEventListener('seeked', () => {
      if (state.isDraggingSeek) return;
      state.pendingSeekTime = null;
      if (state.pendingAutoplay) {
        state.pendingAutoplay = false;
        const playResult = audio.play();
        if (playResult && typeof playResult.catch === 'function') playResult.catch(() => {});
      }
      syncPlayheadFromAudio();
    });
    audio.addEventListener('play', () => { playPauseButton.textContent = 'Duraklat'; });
    audio.addEventListener('pause', () => { playPauseButton.textContent = 'Oynat'; });
    playPauseButton.addEventListener('click', async () => {
      if (audio.paused) await audio.play();
      else audio.pause();
    });
    stopButton.addEventListener('click', () => {
      audio.pause();
      seekTo(0);
    });

    function startSeekDrag() {
      state.isDraggingSeek = true;
      state.followPlayhead = true;
    }
    function commitSeekDrag() {
      state.isDraggingSeek = false;
      seekTo(Number(seekSlider.value), true);
    }
    seekSlider.addEventListener('pointerdown', startSeekDrag);
    seekSlider.addEventListener('mousedown', startSeekDrag);
    seekSlider.addEventListener('touchstart', startSeekDrag);
    seekSlider.addEventListener('input', event => {
      state.isDraggingSeek = true;
      seekTo(Number(event.target.value), false);
    });
    seekSlider.addEventListener('change', commitSeekDrag);
    seekSlider.addEventListener('pointerup', commitSeekDrag);
    seekSlider.addEventListener('mouseup', commitSeekDrag);
    seekSlider.addEventListener('touchend', commitSeekDrag);
    seekSlider.addEventListener('blur', () => {
      if (state.isDraggingSeek) commitSeekDrag();
    });
    speedSelect.addEventListener('change', () => {
      audio.playbackRate = Number(speedSelect.value || 1);
    });
    zoomSlider.addEventListener('input', () => {
      const center = state.playhead || (state.viewStart + state.viewDuration / 2);
      updateViewFromZoom(false);
      setViewStart(center - state.viewDuration / 2);
    });
    panSlider.addEventListener('pointerdown', () => {
      state.isDraggingPan = true;
      state.followPlayhead = false;
    });
    panSlider.addEventListener('input', event => {
      state.followPlayhead = false;
      state.viewStart = clamp(Number(event.target.value), 0, maxViewStart());
      draw();
    });
    panSlider.addEventListener('pointerup', () => {
      state.isDraggingPan = false;
      panSlider.value = String(state.viewStart);
    });
    panSlider.addEventListener('change', () => {
      state.isDraggingPan = false;
      state.viewStart = clamp(Number(panSlider.value), 0, maxViewStart());
      draw();
    });

    sourceSelect.addEventListener('change', event => loadSource(Number(event.target.value)));
    sourceSearch.addEventListener('input', event => renderSourceOptions(event.target.value));
    modelNameSelect.addEventListener('change', () => {
      const chosen = (state.models || []).find(model => model.id === modelNameSelect.value);
      thresholdInput.max = '1';
      thresholdInput.step = '0.001';
      thresholdInput.value = String(chosen ? chosen.recommended_threshold : 0.75);
      loadSource(state.sourceIndex);
    });
    thresholdInput.addEventListener('change', () => loadSource(state.sourceIndex));
    reloadButton.addEventListener('click', () => loadSource(state.sourceIndex));
    pickFiles.addEventListener('click', () => filePicker.click());
    filePicker.addEventListener('change', () => uploadFiles(filePicker.files));
    uploadZone.addEventListener('dragover', event => {
      event.preventDefault();
      uploadZone.classList.add('dragover');
    });
    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
    uploadZone.addEventListener('drop', event => {
      event.preventDefault();
      uploadZone.classList.remove('dragover');
      uploadFiles(event.dataTransfer.files);
    });

    async function uploadFiles(files) {
      if (!files || files.length === 0) return;
      statusEl.textContent = 'Dosya ekleniyor ve taranıyor…';
      const form = new FormData();
      Array.from(files).forEach(file => form.append('files', file));
      let payload;
      try {
        const response = await fetch('/api/upload', { method: 'POST', body: form });
        if (!response.ok) throw new Error((await response.text()) || response.statusText);
        payload = await response.json();
      } catch (err) {
        statusEl.textContent = 'Dosya eklenemedi: ' + (err && err.message ? err.message : err);
        return;
      }
      const results = payload.results || [];
      const uploadedSource = results.length ? results[0].source : null;
      const sourcesResponse = await fetch('/api/sources');
      const sourcesPayload = await sourcesResponse.json();
      state.sources = sourcesPayload.sources;
      sourceSearch.value = '';
      renderSourceOptions();
      let target = uploadedSource ? state.sources.find(source => source.path === uploadedSource) : null;
      if (!target) target = state.sources[sourcesPayload.latest_index] || state.sources[state.sources.length - 1];
      if (target) await loadSource(target.index);
      const total = results.reduce((sum, item) => sum + (item.candidates || 0), 0);
      statusEl.textContent = `${results.length} dosya eklendi, ${total} nefes adayı bulundu. Hazır.`;
      filePicker.value = '';
    }

    modeButtons.forEach(button => {
      button.addEventListener('click', () => {
        state.mode = button.dataset.mode;
        modeButtons.forEach(item => item.classList.toggle('active', item === button));
        statusEl.textContent = state.mode === 'cut'
          ? 'Dikkat: Tamamen Kes modu konuşma parçalarını da silebilir. Önce kırmızı bölgeleri dinle.'
          : 'Güvenli azaltma modu seçildi. Kaynak dosya değiştirilmez.';
      });
    });
    eraserButton.addEventListener('click', () => {
      state.eraser = !state.eraser;
      eraserButton.classList.toggle('active', state.eraser);
      draw();
    });
    restoreButton.addEventListener('click', () => {
      state.detections = state.originalDetections.map(item => ({ ...item }));
      renderDetections();
      statusEl.textContent = 'Modelin ilk tespitleri geri getirildi.';
    });
    processButton.addEventListener('click', async () => {
      if (!state.detections.length) {
        statusEl.textContent = 'İşlenecek etkin nefes adayı yok.';
        return;
      }
      if (state.mode === 'cut' && !window.confirm('Tamamen Kes modu seçili. Yanlış bir tespit konuşmayı da kesebilir. Yeni bir WAV çıktısı oluşturulsun mu?')) return;
      statusEl.textContent = 'Yeni WAV çıktısı oluşturuluyor…';
      processButton.disabled = true;
      try {
        const response = await fetch('/api/process-model', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            source_index: state.sourceIndex,
            model: state.modelName,
            mode: state.mode,
            detections: state.detections,
          })
        });
        if (!response.ok) {
          statusEl.textContent = 'Çıktı oluşturulamadı: ' + (await response.text());
          return;
        }
        const payload = await response.json();
        outputAudio.src = `${payload.audio}&t=${Date.now()}`;
        downloadOutput.href = payload.audio;
        outputPathEl.textContent = payload.output;
        outputEl.classList.add('visible');
        statusEl.textContent = `Tamamlandı: ${payload.segments} bölge işlendi (${payload.mode}).`;
      } catch (error) {
        statusEl.textContent = `İşlem hatası: ${error.message || error}`;
      } finally {
        processButton.disabled = false;
      }
    });
    window.addEventListener('resize', draw);
    window.removeDetection = removeDetection;
    load().catch(error => { statusEl.textContent = `Başlatma hatası: ${error.message || error}`; });
  </script>
</body>
</html>
"""



MANUAL_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Manual Segment Labeler</title>
  <style>
    :root {
      --bg: #f0f2f5;
      --panel: #ffffff;
      --ink: #111827;
      --muted: #6b7280;
      --line: #e5e7eb;
      --accent: #0f766e;
      --accent-hover: #0d6460;
      --accent-weak: #ccfaf4;
      --danger: #b42318;
      --warn: #8a5a00;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, Arial, sans-serif;
      font-size: 14px;
      line-height: 1.5;
      letter-spacing: 0;
    }
    header {
      min-height: 56px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 12px 20px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      box-shadow: 0 1px 4px rgba(0,0,0,.07);
    }
    h1 { margin: 0; font-size: 17px; font-weight: 700; letter-spacing: -0.3px; }
    a { color: var(--accent); font-weight: 600; text-decoration: none; padding: 4px 10px; border-radius: 6px; transition: background .15s; }
    a:hover { background: var(--accent-weak); }
    main {
      max-width: 1120px;
      margin: 0 auto;
      padding: 20px 22px;
    }
    .toolbar {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) 160px 160px;
      gap: 8px;
      margin-bottom: 12px;
    }
    select, input, textarea, button {
      font: inherit;
      border-radius: 7px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      transition: border-color .15s;
    }
    select, input {
      min-height: 38px;
      padding: 0 10px;
    }
    select:hover, input:hover { border-color: #9ca3af; }
    select:focus, input:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-weak);
    }
    audio {
      width: 100%;
      height: 42px;
      margin: 8px 0 12px;
    }
    .wave-wrap {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 14px;
      margin-bottom: 14px;
      box-shadow: 0 1px 4px rgba(0,0,0,.06);
    }
    .zoom-controls {
      display: grid;
      grid-template-columns: 88px 1fr 88px 1fr 88px;
      gap: 8px;
      align-items: center;
      margin-bottom: 10px;
    }
    .zoom-controls label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    .zoom-controls input[type="range"] {
      width: 100%;
      min-height: 28px;
      padding: 0;
      border: none;
      background: transparent;
      box-shadow: none;
    }
    .zoom-controls button {
      min-height: 34px;
      font-weight: 700;
      cursor: pointer;
    }
    .zoom-controls button:hover {
      background: var(--bg);
      border-color: #9ca3af;
    }
    canvas {
      display: block;
      width: 100%;
      height: 260px;
      cursor: crosshair;
      background: #fafbfc;
      border: 1px solid var(--line);
      border-radius: 6px;
    }
    .hint {
      color: var(--muted);
      font-size: 12px;
      margin-top: 8px;
    }
    .readout {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 12px;
      margin-top: 8px;
    }
    .manual-adjust {
      display: grid;
      grid-template-columns: repeat(6, minmax(82px, 1fr));
      gap: 7px;
      margin-top: 10px;
    }
    .manual-adjust button,
    .manual-adjust label {
      min-height: 32px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--panel);
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
      transition: background .15s, border-color .15s;
    }
    .manual-adjust button:hover {
      background: var(--bg);
      border-color: #9ca3af;
    }
    .labels {
      display: grid;
      grid-template-columns: repeat(5, minmax(100px, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }
    .label {
      min-height: 48px;
      font-weight: 700;
      cursor: pointer;
      border-radius: 8px;
      transition: border-color .15s, box-shadow .15s;
    }
    .label:hover {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-weak);
    }
    .label.selected {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
      box-shadow: 0 2px 8px rgba(15,118,110,.25);
    }
    .label.bad.selected { background: var(--danger); border-color: var(--danger); box-shadow: 0 2px 8px rgba(180,35,24,.25); }
    .label.noise.selected { background: var(--warn); border-color: var(--warn); }
    textarea {
      width: 100%;
      min-height: 80px;
      padding: 10px 12px;
      resize: vertical;
      margin-bottom: 12px;
      font-size: 13px;
    }
    textarea:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-weak);
    }
    .actions {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }
    .action {
      min-width: 120px;
      min-height: 40px;
      font-weight: 700;
      font-size: 13px;
      cursor: pointer;
      transition: background .15s, border-color .15s;
    }
    .action:hover {
      background: var(--bg);
      border-color: #9ca3af;
    }
    .primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }
    .primary:hover {
      background: var(--accent-hover);
      border-color: var(--accent-hover);
    }
    .status {
      color: var(--muted);
      min-height: 20px;
      font-size: 13px;
      margin-top: 10px;
    }
    @media (max-width: 760px) {
      .toolbar { grid-template-columns: 1fr; }
      .zoom-controls { grid-template-columns: 1fr; }
      .manual-adjust { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
      .labels { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
      canvas { height: 220px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Manual Segment Labeler</h1>
    <span>
      <a href="/">Candidates</a>
      <a href="/test">Test Detection</a>
      <a href="/model">Model</a>
    </span>
  </header>
  <main>
    <div class="toolbar">
      <select id="source"></select>
      <input id="start" type="number" step="0.001" min="0" value="0">
      <input id="end" type="number" step="0.001" min="0" value="0">
    </div>
    <audio id="audio" controls></audio>
    <div class="wave-wrap">
      <div class="zoom-controls">
        <label for="zoom">Zoom</label>
        <input id="zoom" type="range" min="1" max="80" step="1" value="1">
        <button id="zoomReset" type="button">Reset</button>
        <input id="pan" type="range" min="0" max="0" step="0.001" value="0">
        <label for="pan">Pan</label>
      </div>
      <canvas id="wave"></canvas>
      <div class="hint">Drag to select a segment. Click the waveform to move playback. The red line follows the current audio position.</div>
      <div class="readout">
        <span id="duration">duration: -</span>
        <span id="selection">selection: -</span>
      </div>
      <div class="manual-adjust">
        <button id="startBack" type="button">Start -10ms</button>
        <button id="startForward" type="button">Start +10ms</button>
        <button id="endBack" type="button">End -10ms</button>
        <button id="endForward" type="button">End +10ms</button>
        <button id="zoomToSelection" type="button">Zoom Selection</button>
        <label><input id="loopSelection" type="checkbox"> Loop</label>
      </div>
    </div>
    <div id="labels" class="labels"></div>
    <textarea id="notes" placeholder="Notes"></textarea>
    <div class="actions">
      <div>
        <button id="playSelection" class="action" type="button">Play Selection</button>
        <button id="clear" class="action" type="button">Clear</button>
      </div>
      <button id="save" class="action primary" type="button">Save Clip</button>
    </div>
    <div id="status" class="status"></div>
  </main>
  <script>
    let state = {
      sources: [],
      labels: [],
      selectedLabel: 'breath',
      sourceIndex: 0,
      duration: 0,
      peaks: [],
      playhead: 0,
      viewStart: 0,
      viewDuration: 0,
      dragStart: null,
      selection: { start: 0, end: 0 }
    };

    const sourceSelect = document.getElementById('source');
    const audio = document.getElementById('audio');
    const canvas = document.getElementById('wave');
    const ctx = canvas.getContext('2d');
    const startInput = document.getElementById('start');
    const endInput = document.getElementById('end');
    const labelsEl = document.getElementById('labels');
    const notes = document.getElementById('notes');
    const status = document.getElementById('status');
    const durationEl = document.getElementById('duration');
    const selectionEl = document.getElementById('selection');
    const zoomInput = document.getElementById('zoom');
    const panInput = document.getElementById('pan');
    const zoomReset = document.getElementById('zoomReset');
    const loopSelection = document.getElementById('loopSelection');

    async function load() {
      const response = await fetch('/api/sources');
      const payload = await response.json();
      state.sources = payload.sources;
      state.labels = payload.labels;
      sourceSelect.innerHTML = state.sources.map(source => `<option value="${source.index}">${source.path}</option>`).join('');
      renderLabels();
      await loadSource(payload.latest_index || 0);
    }

    async function loadSource(index) {
      state.sourceIndex = index;
      sourceSelect.value = String(index);
      const source = state.sources[index];
      audio.src = source.audio;
      audio.currentTime = 0;
      state.playhead = 0;
      status.textContent = 'Loading waveform';
      const response = await fetch(source.audio);
      const arrayBuffer = await response.arrayBuffer();
      const audioContext = new AudioContext();
      const decoded = await audioContext.decodeAudioData(arrayBuffer.slice(0));
      state.duration = decoded.duration;
      state.peaks = makePeaks(decoded.getChannelData(0), 1800);
      state.viewStart = 0;
      state.viewDuration = state.duration;
      state.selection = { start: 0, end: Math.min(0.5, state.duration) };
      zoomInput.value = '1';
      updatePanControl();
      syncInputs();
      draw();
      status.textContent = '';
    }

    function makePeaks(samples, width) {
      const block = Math.max(1, Math.floor(samples.length / width));
      const peaks = [];
      for (let i = 0; i < width; i++) {
        let min = 0;
        let max = 0;
        const start = i * block;
        const end = Math.min(samples.length, start + block);
        for (let j = start; j < end; j++) {
          const v = samples[j];
          if (v < min) min = v;
          if (v > max) max = v;
        }
        peaks.push([min, max]);
      }
      return peaks;
    }

    function resizeCanvas() {
      const rect = canvas.getBoundingClientRect();
      const scale = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * scale));
      canvas.height = Math.max(1, Math.floor(rect.height * scale));
      ctx.setTransform(scale, 0, 0, scale, 0, 0);
    }

    function draw() {
      resizeCanvas();
      const rect = canvas.getBoundingClientRect();
      clampView();
      ctx.clearRect(0, 0, rect.width, rect.height);
      ctx.fillStyle = '#fbfbfc';
      ctx.fillRect(0, 0, rect.width, rect.height);
      const mid = rect.height / 2;
      ctx.strokeStyle = '#0f766e';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let x = 0; x < rect.width; x++) {
        const time = state.viewStart + (x / Math.max(1, rect.width)) * state.viewDuration;
        const idx = Math.floor(time / Math.max(0.001, state.duration) * state.peaks.length);
        const peak = state.peaks[idx] || [0, 0];
        ctx.moveTo(x, mid + peak[0] * mid * 0.95);
        ctx.lineTo(x, mid + peak[1] * mid * 0.95);
      }
      ctx.stroke();

      const x1 = timeToX(state.selection.start);
      const x2 = timeToX(state.selection.end);
      const visibleX1 = Math.max(0, Math.min(rect.width, x1));
      const visibleX2 = Math.max(0, Math.min(rect.width, x2));
      if (Math.max(x1, x2) >= 0 && Math.min(x1, x2) <= rect.width) {
        ctx.fillStyle = 'rgba(15, 118, 110, 0.22)';
        ctx.fillRect(Math.min(visibleX1, visibleX2), 0, Math.abs(visibleX2 - visibleX1), rect.height);
        ctx.strokeStyle = '#0f766e';
        ctx.lineWidth = 2;
        ctx.beginPath();
        if (x1 >= 0 && x1 <= rect.width) {
          ctx.moveTo(x1, 0);
          ctx.lineTo(x1, rect.height);
        }
        if (x2 >= 0 && x2 <= rect.width) {
          ctx.moveTo(x2, 0);
          ctx.lineTo(x2, rect.height);
        }
        ctx.stroke();
      }

      const playheadX = timeToX(state.playhead);
      if (playheadX >= 0 && playheadX <= rect.width) {
        ctx.strokeStyle = '#dc2626';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(playheadX, 0);
        ctx.lineTo(playheadX, rect.height);
        ctx.stroke();
      }

      durationEl.textContent = `duration: ${state.duration.toFixed(3)}s`;
      selectionEl.textContent = `view: ${state.viewStart.toFixed(3)}s - ${(state.viewStart + state.viewDuration).toFixed(3)}s | selection: ${state.selection.start.toFixed(3)}s - ${state.selection.end.toFixed(3)}s | playhead: ${state.playhead.toFixed(3)}s`;
    }

    function xToTime(clientX) {
      const rect = canvas.getBoundingClientRect();
      const x = Math.min(rect.width, Math.max(0, clientX - rect.left));
      return state.viewStart + (x / Math.max(1, rect.width)) * state.viewDuration;
    }

    function timeToX(time) {
      const rect = canvas.getBoundingClientRect();
      return (time - state.viewStart) / Math.max(0.001, state.viewDuration) * rect.width;
    }

    function clampView() {
      state.viewDuration = Math.max(0.15, Math.min(state.duration || 0.15, state.viewDuration || state.duration || 0.15));
      const maxStart = Math.max(0, state.duration - state.viewDuration);
      state.viewStart = Math.max(0, Math.min(maxStart, state.viewStart || 0));
    }

    function updatePanControl() {
      clampView();
      const maxStart = Math.max(0, state.duration - state.viewDuration);
      panInput.max = maxStart.toFixed(3);
      panInput.step = Math.max(0.001, state.viewDuration / 1000).toFixed(3);
      panInput.value = state.viewStart.toFixed(3);
      panInput.disabled = maxStart <= 0.001;
    }

    function setZoom(level) {
      const previousCenter = state.playhead > state.viewStart && state.playhead < state.viewStart + state.viewDuration
        ? state.playhead
        : state.viewStart + state.viewDuration / 2;
      const zoom = Math.max(1, Number(level));
      state.viewDuration = Math.max(0.15, state.duration / zoom);
      state.viewStart = previousCenter - state.viewDuration / 2;
      updatePanControl();
      draw();
    }

    function syncInputs() {
      const start = Math.min(state.selection.start, state.selection.end);
      const end = Math.max(state.selection.start, state.selection.end);
      state.selection = { start, end };
      startInput.value = start.toFixed(3);
      endInput.value = end.toFixed(3);
    }

    function renderLabels() {
      labelsEl.innerHTML = state.labels.map(label => `
        <button class="label ${label} ${state.selectedLabel === label ? 'selected' : ''}" type="button" onclick="selectLabel('${label}')">${label}</button>
      `).join('');
    }

    function selectLabel(label) {
      state.selectedLabel = label;
      renderLabels();
    }

    canvas.addEventListener('mousedown', event => {
      state.dragStart = xToTime(event.clientX);
      state.selection = { start: state.dragStart, end: state.dragStart };
      syncInputs();
      draw();
    });
    window.addEventListener('mousemove', event => {
      if (state.dragStart === null) return;
      state.selection.end = xToTime(event.clientX);
      syncInputs();
      draw();
    });
    window.addEventListener('mouseup', () => {
      if (state.dragStart === null) return;
      state.dragStart = null;
      syncInputs();
      draw();
    });
    canvas.addEventListener('click', event => {
      if (Math.abs(state.selection.end - state.selection.start) > 0.02) return;
      audio.currentTime = xToTime(event.clientX);
      state.playhead = audio.currentTime;
      draw();
    });

    startInput.addEventListener('change', () => {
      state.selection.start = Math.max(0, Math.min(state.duration, Number(startInput.value)));
      syncInputs();
      draw();
    });
    endInput.addEventListener('change', () => {
      state.selection.end = Math.max(0, Math.min(state.duration, Number(endInput.value)));
      syncInputs();
      draw();
    });
    sourceSelect.addEventListener('change', event => loadSource(Number(event.target.value)));
    zoomInput.addEventListener('input', () => setZoom(zoomInput.value));
    panInput.addEventListener('input', () => {
      state.viewStart = Number(panInput.value);
      clampView();
      draw();
    });
    zoomReset.addEventListener('click', () => {
      zoomInput.value = '1';
      state.viewStart = 0;
      state.viewDuration = state.duration;
      updatePanControl();
      draw();
    });
    audio.addEventListener('timeupdate', () => {
      state.playhead = audio.currentTime;
      draw();
    });
    audio.addEventListener('seeking', () => {
      state.playhead = audio.currentTime;
      draw();
    });
    audio.addEventListener('play', tickPlayhead);

    function tickPlayhead() {
      state.playhead = audio.currentTime;
      keepPlayheadVisible();
      draw();
      if (!audio.paused && !audio.ended) requestAnimationFrame(tickPlayhead);
    }

    function keepPlayheadVisible() {
      if (state.viewDuration >= state.duration) return;
      const margin = state.viewDuration * 0.12;
      if (state.playhead < state.viewStart + margin) {
        state.viewStart = state.playhead - margin;
        updatePanControl();
      } else if (state.playhead > state.viewStart + state.viewDuration - margin) {
        state.viewStart = state.playhead - state.viewDuration + margin;
        updatePanControl();
      }
    }

    document.getElementById('playSelection').onclick = () => {
      audio.currentTime = state.selection.start;
      audio.play();
      const stopAt = state.selection.end;
      const timer = setInterval(() => {
        if (audio.currentTime >= stopAt && loopSelection.checked) {
          audio.currentTime = state.selection.start;
        } else if (audio.currentTime >= stopAt || audio.paused) {
          audio.pause();
          clearInterval(timer);
        }
      }, 25);
    };

    document.getElementById('startBack').onclick = () => nudgeSelection('start', -0.010);
    document.getElementById('startForward').onclick = () => nudgeSelection('start', 0.010);
    document.getElementById('endBack').onclick = () => nudgeSelection('end', -0.010);
    document.getElementById('endForward').onclick = () => nudgeSelection('end', 0.010);
    document.getElementById('zoomToSelection').onclick = () => {
      const width = Math.max(0.15, (state.selection.end - state.selection.start) * 3);
      const center = (state.selection.start + state.selection.end) / 2;
      state.viewDuration = Math.min(state.duration, width);
      state.viewStart = center - state.viewDuration / 2;
      zoomInput.value = String(Math.max(1, Math.round(state.duration / state.viewDuration)));
      updatePanControl();
      draw();
    };

    function nudgeSelection(edge, delta) {
      if (edge === 'start') {
        state.selection.start = Math.max(0, Math.min(state.selection.end - 0.010, state.selection.start + delta));
      } else {
        state.selection.end = Math.min(state.duration, Math.max(state.selection.start + 0.010, state.selection.end + delta));
      }
      syncInputs();
      draw();
    }

    document.getElementById('clear').onclick = () => {
      state.selection = { start: 0, end: Math.min(0.5, state.duration) };
      notes.value = '';
      syncInputs();
      draw();
    };

    window.addEventListener('keydown', event => {
      if (document.activeElement === notes) return;
      if (event.key === 'a' || event.key === 'A') nudgeSelection('start', -0.010);
      if (event.key === 'd' || event.key === 'D') nudgeSelection('start', 0.010);
      if (event.key === 'j' || event.key === 'J') nudgeSelection('end', -0.010);
      if (event.key === 'l' || event.key === 'L') nudgeSelection('end', 0.010);
      if (event.key === 'z' || event.key === 'Z') document.getElementById('zoomToSelection').click();
    });

    document.getElementById('save').onclick = async () => {
      status.textContent = 'Saving';
      const response = await fetch('/api/manual-label', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source_index: state.sourceIndex,
          start: state.selection.start,
          end: state.selection.end,
          label: state.selectedLabel,
          notes: notes.value
        })
      });
      if (!response.ok) {
        status.textContent = 'Save failed';
        return;
      }
      const payload = await response.json();
      status.textContent = `Saved ${payload.item.label}: ${payload.item.start}s - ${payload.item.end}s`;
    };

    window.selectLabel = selectLabel;
    window.addEventListener('resize', draw);
    load();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
