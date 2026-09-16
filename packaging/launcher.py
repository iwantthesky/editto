from __future__ import annotations

import csv
import ctypes
import os
import signal
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path


APP_NAME = "Editto"
HOST = "127.0.0.1"
PORT = 8765
URL = f"http://{HOST}:{PORT}/model"
FIELDNAMES = ["clip", "source", "start", "end", "score", "label", "notes"]


def resource_root() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def data_root() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    return base / APP_NAME


def show_error(message: str) -> None:
    ctypes.windll.user32.MessageBoxW(None, message, APP_NAME, 0x10)


def prepare_workspace() -> tuple[Path, Path]:
    root = data_root()
    labels = root / "dataset" / "candidates" / "labels.csv"
    (root / "dataset" / "raw").mkdir(parents=True, exist_ok=True)
    labels.parent.mkdir(parents=True, exist_ok=True)
    (root / "outputs" / "model_processed").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(parents=True, exist_ok=True)
    if not labels.exists():
        with labels.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=FIELDNAMES).writeheader()

    packaged_models = resource_root() / "resources" / "models"
    for packaged_model in packaged_models.iterdir():
        if packaged_model.is_file() and packaged_model.suffix.lower() in {".json", ".pt"}:
            (root / "models" / packaged_model.name).write_bytes(packaged_model.read_bytes())

    tools = resource_root() / "resources" / "bin"
    if tools.exists():
        os.environ["PATH"] = str(tools) + os.pathsep + os.environ.get("PATH", "")
    os.chdir(root)
    return root, labels


def server_is_ready(timeout: float = 0.15) -> bool:
    try:
        with urllib.request.urlopen(f"http://{HOST}:{PORT}/api/models", timeout=timeout) as response:
            body = response.read(4096)
        return response.status == 200 and b'"default": "kabum_v2"' in body
    except Exception:
        return False


def open_when_ready() -> None:
    for _ in range(100):
        if server_is_ready():
            webbrowser.open(URL)
            return
        time.sleep(0.1)


def process_image(pid: int) -> str:
    process_query_limited_information = 0x1000
    handle = ctypes.windll.kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return ""
    try:
        size = ctypes.c_ulong(32768)
        buffer = ctypes.create_unicode_buffer(size.value)
        ok = ctypes.windll.kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size))
        return buffer.value if ok else ""
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def stop_server() -> int:
    pid_file = data_root() / "editto.pid"
    if not pid_file.exists():
        return 0
    try:
        pid = int(pid_file.read_text(encoding="ascii").strip())
        image = process_image(pid).lower()
        if pid > 0 and pid != os.getpid() and image.endswith("editto.exe"):
            os.kill(pid, signal.SIGTERM)
    except (OSError, ValueError):
        pass
    try:
        pid_file.unlink(missing_ok=True)
    except OSError:
        pass
    return 0


def main() -> int:
    args = {arg.lower() for arg in sys.argv[1:]}
    if "--stop" in args:
        return stop_server()
    if "--open-outputs" in args:
        target = data_root() / "outputs" / "model_processed"
        target.mkdir(parents=True, exist_ok=True)
        os.startfile(target)
        return 0
    if server_is_ready():
        if "--no-browser" not in args:
            webbrowser.open(URL)
        return 0

    try:
        root, labels = prepare_workspace()
        pid_file = root / "editto.pid"
        pid_file.write_text(str(os.getpid()), encoding="ascii")
        from breath_cleaner.labeler import _make_handler
        from http.server import ThreadingHTTPServer

        server = ThreadingHTTPServer((HOST, PORT), _make_handler(labels))
        if "--no-browser" not in args:
            threading.Thread(target=open_when_ready, daemon=True).start()
        try:
            server.serve_forever()
        finally:
            server.server_close()
            pid_file.unlink(missing_ok=True)
        return 0
    except Exception as exc:
        show_error(f"Editto başlatılamadı.\n\n{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
