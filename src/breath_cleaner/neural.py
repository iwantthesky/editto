"""Versioned CPU breath classifier; preprocessing shared by training and inference."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from torch import nn

SR = 16000
WINDOW = 8000
PREPROCESS = {'sample_rate': SR, 'window_samples': WINDOW, 'n_fft': 512,
              'hop_length': 160, 'n_mels': 64, 'normalization': 'peak_then_logmel_v1'}
SCAN = {'hop_samples':1600, 'minimum_consecutive_windows':3,
        'minimum_rms_dbfs':-65, 'region_start_offset':.20, 'region_end_offset':.30}

def fixed_window(samples):
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not np.isfinite(x).all(): raise ValueError('Non-finite audio')
    if len(x) > WINDOW:
        start = (len(x) - WINDOW) // 2
        x = x[start:start + WINDOW]
    if len(x) < WINDOW:
        left = (WINDOW - len(x)) // 2
        x = np.pad(x, (left, WINDOW - len(x) - left))
    return x

class LogMel(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.linspace(0, SR / 2, 257)
        m = torch.linspace(0, 2595 * np.log10(1 + (SR / 2) / 700), 66)
        hz = 700 * (10 ** (m / 2595) - 1)
        bank = torch.minimum((f[None] - hz[:-2, None]) / (hz[1:-1] - hz[:-2])[:, None],
                             (hz[2:, None] - f[None]) / (hz[2:] - hz[1:-1])[:, None]).clamp_min(0)
        self.register_buffer('bank', bank)
        self.register_buffer('window', torch.hann_window(512))

    def forward(self, audio):
        audio = audio / audio.abs().amax(-1, keepdim=True).clamp_min(1e-4)
        power = torch.stft(audio, 512, hop_length=160, window=self.window,
                           return_complex=True, center=True).abs().square()
        mel = torch.log10(torch.matmul(self.bank, power).clamp_min(1e-8))
        return ((mel + 4) / 4).clamp(-2, 2).unsqueeze(1)

class BreathCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        channels = 1
        for out in (8, 16, 32):
            layers += [nn.Conv2d(channels, out, 3, padding=1), nn.BatchNorm2d(out), nn.ReLU(), nn.MaxPool2d(2)]
            channels = out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(.2), nn.Linear(32, 1))

    def forward(self, mel): return self.head(self.features(mel)).squeeze(-1)

def load_cnn(metadata):
    if metadata['preprocess'] != PREPROCESS: raise ValueError('Unsupported neural preprocessing')
    path = Path(metadata['_path']).parent / metadata['weights_file']
    if path.parent.resolve() != Path(metadata['_path']).parent.resolve(): raise ValueError('Invalid model path')
    if metadata.get('weights_sha256') and hashlib.sha256(path.read_bytes()).hexdigest()!=metadata['weights_sha256']:
        raise ValueError('Model weights checksum mismatch')
    model = BreathCNN()
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    model.eval()
    return model

@torch.inference_mode()
def probabilities(model, audio, batch_size=128):
    front = LogMel()
    results = []
    for start in range(0, len(audio), batch_size):
        x = torch.as_tensor(np.asarray(audio[start:start+batch_size]), dtype=torch.float32)
        results.extend(torch.sigmoid(model(front(x))).cpu().numpy().tolist())
    return np.asarray(results)

def regions_from_scores(starts, scores, duration, threshold, hop=.1):
    """Three consecutive high windows required. Ends trimmed to window centers."""
    if not .0 < threshold <= 1: raise ValueError('Invalid threshold')
    regions = []
    active = []
    for start, score in list(zip(starts, scores)) + [(duration + 1, 0.)]:
        if score >= threshold:
            active.append((float(start), float(score)))
        else:
            if len(active) >= 3:
                left = active[0][0] + .20
                right = min(duration, active[-1][0] + .30)
                if right > left:
                    regions.append({'start': left, 'end': right,
                                    'model_probability': max(s for _, s in active),
                                    'needs_review': True, 'label': 'unknown'})
            active = []
    return regions
