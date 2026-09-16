"""Common legacy and CNN inference. No labels are used for model evaluation."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from .audio_io import read_audio
from .features import extract_features

@lru_cache(maxsize=3)
def _network(path, stamp, size):
    import json
    import torch
    from .neural import load_cnn
    torch.set_num_threads(min(4, torch.get_num_threads()))
    metadata=json.loads(Path(path).read_text(encoding='utf-8'))
    metadata['_path']=path
    return load_cnn(metadata)

def network(metadata):
    path=Path(metadata['_path']).resolve()
    weights=path.parent/metadata['weights_file']
    return _network(str(path), (path.stat().st_mtime_ns,weights.stat().st_mtime_ns),weights.stat().st_size)

def predict_clip(path,metadata):
    if metadata.get('model_type')=='breath_cnn_v2':
        from .neural import fixed_window, probabilities
        audio=read_audio(path)
        return float(probabilities(network(metadata), [fixed_window(audio.samples)])[0])
    x=extract_features(path)
    mean=np.asarray(metadata['mean'],np.float32);std=np.asarray(metadata['std'],np.float32)
    if np.any(std<=0):raise ValueError('Invalid model standard deviation')
    z=np.clip(((x-mean)/std)@np.asarray(metadata['weights'])+metadata['bias'],-40,40)
    return float(1/(1+np.exp(-z)))

def scan_recording(path, metadata, threshold=None):
    from .neural import fixed_window, probabilities, regions_from_scores, WINDOW, SR, SCAN
    if metadata.get('scan',SCAN)!=SCAN:raise ValueError('Unsupported scan configuration')
    audio=read_audio(path);duration=len(audio.samples)/SR
    selected=float(metadata['recommended_threshold'] if threshold is None else threshold)
    if not 0<selected<=1:raise ValueError('Threshold must be between 0 and 1')
    hop=SCAN['hop_samples']
    starts=np.arange(0,max(1,len(audio.samples)-WINDOW+1),hop)
    if len(audio.samples)>WINDOW and starts[-1]!=len(audio.samples)-WINDOW:
        starts=np.append(starts,len(audio.samples)-WINDOW)
    model=network(metadata);scores=[]
    for offset in range(0,len(starts),128):
        windows=[fixed_window(audio.samples[s:s+WINDOW]) for s in starts[offset:offset+128]]
        p=probabilities(model, windows)
        rms=np.sqrt(np.mean(np.asarray(windows)**2,axis=1))
        p[rms<10**(SCAN['minimum_rms_dbfs']/20)]=0
        scores.extend(p)
    detections=regions_from_scores(starts/SR,scores,duration,selected)
    for region in detections:
        region['model_version']=metadata['version']
        region['needs_review']=not bool(metadata.get('accepted'))
    return {'detections':detections,'threshold':selected,'model_version':metadata['version'],
            'model_status':metadata.get('status','candidate'), 'scan_mode':'full_recording',
            'window_count':len(starts),'boundary_accuracy':'unverified',
            'requires_review':not bool(metadata.get('accepted'))}
