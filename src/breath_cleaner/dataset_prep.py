from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

try:
    from .audio_io import read_audio
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from breath_cleaner.audio_io import read_audio


def prepare_training_data(labels_csv: Path, output_dir: Path, sample_rate: int = 16000):
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = output_dir / "features"
    features_dir.mkdir(exist_ok=True)

    with labels_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    metadata = []
    
    # Filter only labeled items (exclude 'unknown')
    labeled_rows = [r for r in rows if r["label"] != "unknown"]
    
    print(f"Processing {len(labeled_rows)} labeled clips...")

    for i, row in enumerate(labeled_rows):
        clip_path = Path(row["clip"])
        if not clip_path.is_absolute():
            clip_path = labels_csv.parent.parent / clip_path
        
        if not clip_path.exists():
            print(f"Warning: Clip not found {clip_path}")
            continue

        try:
            audio = read_audio(clip_path)
            
            # Use 500ms window (8000 samples at 16kHz)
            target_len = int(0.5 * sample_rate)
            samples = audio.samples
            
            if len(samples) > target_len:
                # Center crop
                start = (len(samples) - target_len) // 2
                samples = samples[start : start + target_len]
            elif len(samples) < target_len:
                # Pad with zeros
                samples = np.pad(samples, (0, target_len - len(samples)))
            
            mel = compute_mel_spectrogram(samples, sample_rate)
            
            # Save as .npy
            feat_filename = f"feat_{i:05d}.npy"
            np.save(features_dir / feat_filename, mel.astype(np.float32))
            
            metadata.append({
                "feature_file": feat_filename,
                "label": row["label"],
                "source": row["source"],
                "start": row["start"]
            })
        except Exception as e:
            print(f"Error processing {clip_path}: {e}")

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset preparation complete. Saved to {output_dir}")


def compute_mel_spectrogram(samples: np.ndarray, sr: int, n_mels: int = 40, n_fft: int = 1024, hop_length: int = 256):
    # 1. Windowing and FFT
    window = np.hanning(n_fft)
    # Pad samples to center frames
    pad_samples = np.pad(samples, (n_fft // 2, n_fft // 2), mode='reflect')
    
    frames = []
    # Calculate number of frames
    for i in range(0, len(samples), hop_length):
        frame = pad_samples[i : i + n_fft]
        if len(frame) < n_fft:
             frame = np.pad(frame, (0, n_fft - len(frame)))
        
        # Power spectrum
        mag = np.abs(np.fft.rfft(frame * window))
        frames.append(mag ** 2)
    
    spec = np.array(frames).T # (freq_bins, time_frames)
    
    # 2. Mel Filterbank
    mel_fb = _create_mel_filterbank(sr, n_fft, n_mels)
    mel_spec = np.dot(mel_fb, spec)
    
    # 3. Log scaling
    log_mel_spec = 10.0 * np.log10(mel_spec + 1e-9)
    return log_mel_spec


def _create_mel_filterbank(sr: int, n_fft: int, n_mels: int):
    # Convert Hz to Mel (Slaney/Librosa default)
    def hz_to_mel(hz): return 2595 * np.log10(1 + hz / 700)
    def mel_to_hz(mel): return 700 * (10**(mel / 2595) - 1)
    
    min_hz = 0
    max_hz = sr / 2
    
    mel_pts = np.linspace(hz_to_mel(min_hz), hz_to_mel(max_hz), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        # Left slope
        for k in range(bin_pts[m-1], bin_pts[m]):
            fb[m-1, k] = (k - bin_pts[m-1]) / (bin_pts[m] - bin_pts[m-1])
        # Right slope
        for k in range(bin_pts[m], bin_pts[m+1]):
            fb[m-1, k] = (bin_pts[m+1] - k) / (bin_pts[m+1] - bin_pts[m])
            
    # Normalize filters (area = 1) for energy consistency
    enorm = 2.0 / (hz_pts[2:n_mels+2] - hz_pts[0:n_mels])
    fb *= enorm[:, np.newaxis]
    
    return fb


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="dataset/candidates/labels.csv")
    parser.add_argument("--output", default="dataset/processed")
    args = parser.parse_args()
    
    prepare_training_data(Path(args.labels), Path(args.output))
