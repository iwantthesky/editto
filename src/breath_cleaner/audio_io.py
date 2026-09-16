from __future__ import annotations

import json
import wave
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class WavAudio:
    sample_rate: int
    samples: np.ndarray


def read_wav(path: str | Path, preserve_channels: bool = False) -> WavAudio:
    path = Path(path)
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        raw = wav.readframes(frames)

    if sample_width != 2:
        raise ValueError(
            f"Only 16-bit PCM WAV is supported for the first prototype; got {sample_width * 8}-bit."
        )

    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels)
        if not preserve_channels:
            data = data.mean(axis=1)

    return WavAudio(sample_rate=sample_rate, samples=data)


def read_audio(path: str | Path, target_sample_rate: int = 16_000) -> WavAudio:
    path = Path(path)
    if path.suffix.lower() == ".wav":
        try:
            audio = read_wav(path)
            if audio.sample_rate == target_sample_rate:
                return audio
        except (wave.Error, ValueError):
            pass

    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(target_sample_rate),
        "-f",
        "s16le",
        "pipe:1",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required for non-WAV files or WAV resampling.") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to decode {path}: {message}") from exc

    samples = np.frombuffer(result.stdout, dtype="<i2").astype(np.float32) / 32768.0
    return WavAudio(sample_rate=target_sample_rate, samples=samples)


def read_audio_native(path: str | Path) -> WavAudio:
    """Decode at the source sample rate and retain every channel for final output."""
    path = Path(path)
    if path.suffix.lower() == ".wav":
        try:
            return read_wav(path, preserve_channels=True)
        except (wave.Error, ValueError):
            pass

    probe_command = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels", "-of", "json", str(path),
    ]
    try:
        probe = subprocess.run(probe_command, check=True, capture_output=True, text=True)
        stream = json.loads(probe.stdout)["streams"][0]
        sample_rate = int(stream["sample_rate"])
        channels = int(stream["channels"])
        if sample_rate <= 0 or channels <= 0:
            raise ValueError("Invalid source audio format")
        command = [
            "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-vn",
            "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
        ]
        result = subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg and ffprobe are required for native-quality output.") from exc
    except (subprocess.CalledProcessError, KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
        message = getattr(exc, "stderr", "")
        if isinstance(message, bytes):
            message = message.decode("utf-8", errors="replace")
        raise RuntimeError(f"Could not decode native-quality audio {path}: {str(message).strip() or exc}") from exc

    samples = np.frombuffer(result.stdout, dtype="<f4").copy()
    if channels > 1:
        if samples.size % channels:
            raise RuntimeError("Decoded audio channel data is incomplete.")
        samples = samples.reshape(-1, channels)
    if not np.isfinite(samples).all():
        raise RuntimeError("Decoded audio contains non-finite samples.")
    return WavAudio(sample_rate=sample_rate, samples=samples)


def write_wav(path: str | Path, audio: WavAudio) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    samples = np.asarray(audio.samples, dtype=np.float32)
    if samples.ndim not in (1, 2) or (samples.ndim == 2 and samples.shape[1] < 1):
        raise ValueError("Audio samples must be mono or frames-by-channels.")
    if not np.isfinite(samples).all():
        raise ValueError("Audio contains non-finite samples.")
    channels = 1 if samples.ndim == 1 else samples.shape[1]
    clipped = np.clip(samples, -1.0, 1.0)
    scale = np.where(clipped < 0, 32768.0, 32767.0)
    pcm = np.rint(clipped * scale).astype("<i2")

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(audio.sample_rate)
        wav.writeframes(pcm.tobytes())
