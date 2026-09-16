"""Adapt the general CNN to real sliding-window context without cross-recording leakage."""
from __future__ import annotations
import copy,csv,hashlib,json,math,os,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;PROJECT=ROOT
sys.path.insert(0,str(ROOT/'src'))
from breath_cleaner.audio_io import read_audio
from breath_cleaner.neural import BreathCNN,LogMel,PREPROCESS,SCAN,SR,WINDOW,fixed_window,probabilities,regions_from_scores

def window_at(audio,center,shift=0):
    middle=int((center+shift)*SR);start=middle-WINDOW//2;end=start+WINDOW
    left=max(0,-start);right=max(0,end-len(audio));piece=audio[max(0,start):min(len(audio),end)]
    return np.pad(piece,(left,right)).astype(np.float32)[:WINDOW]

def metric(y,p,t):
    y=np.asarray(y)==1;pred=np.asarray(p)>=t
    tp=int(np.sum(y&pred));fp=int(np.sum(~y&pred));fn=int(np.sum(y&~pred));tn=int(np.sum(~y&~pred))
    return {'n':len(y),'tp':tp,'fp':fp,'fn':fn,'tn':tn,
      'precision':None if tp+fp==0 else tp/(tp+fp),'recall':None if tp+fn==0 else tp/(tp+fn),
      'speech_fpr':None if fp+tn==0 else fp/(fp+tn)}

def choose_threshold(y,p,max_fpr=.04):
    options=[]
    for t in np.linspace(.05,.995,300):
        m=metric(y,p,t);prec=m['precision'] or 0;rec=m['recall'] or 0
        safe=m['speech_fpr']<=max_fpr and m['tp']>0
        f05=1.25*prec*rec/max(.25*prec+rec,1e-9)
        options.append(((safe,rec if safe else f05,prec,t),float(t),m))
    return max(options,key=lambda x:x[0])[1:]

def augment(a,rng):
    gain=10**(rng.numpy.uniform(-9,6)/20);a=a*gain
    rms=torch.sqrt(torch.mean(a*a,dim=1,keepdim=True)).clamp_min(1e-5)
    snr=10**(torch.empty((len(a),1)).uniform_(22,38,generator=rng.torch_generator)/20)
    return torch.clamp(a+torch.randn(a.shape,generator=rng.torch_generator)*rms/snr,-1,1)

class Randoms:
    def __init__(self,seed):
        self.numpy=np.random.default_rng(seed);self.torch_generator=torch.Generator().manual_seed(seed)

def train_model(base_state,audio,y,epochs,seed):
    torch.manual_seed(seed);rng=Randoms(seed);model=BreathCNN();model.load_state_dict(base_state)
    front=LogMel();optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=2e-3)
    pos=float(np.sum(y==1));neg=float(np.sum(y==0));lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg/max(1,pos)))
    history=[];best=None;best_loss=float('inf')
    indices=np.arange(len(y))
    for epoch in range(epochs):
        model.train();losses=[]
        for batch in np.array_split(rng.numpy.permutation(indices),max(1,math.ceil(len(indices)/64))):
            a=augment(torch.from_numpy(audio[batch].copy()),rng)
            with torch.no_grad():z=front(a);width=int(rng.numpy.integers(0,6))
            if width:
                start=int(rng.numpy.integers(0,65-width));z[:,:,start:start+width,:]=0
            optimizer.zero_grad();loss=lossfn(model(z),torch.from_numpy(y[batch]))
            loss.backward();optimizer.step();losses.append(float(loss.detach()))
        value=float(np.mean(losses));history.append(value)
        if value<best_loss:best_loss=value;best=copy.deepcopy(model.state_dict())
    model.load_state_dict(best);model.eval();return model,history

def scan(model,audio,threshold):
    hop=SCAN['hop_samples'];starts=np.arange(0,max(1,len(audio)-WINDOW+1),hop)
    if len(audio)>WINDOW and starts[-1]!=len(audio)-WINDOW:starts=np.append(starts,len(audio)-WINDOW)
    windows=np.stack([fixed_window(audio[s:s+WINDOW]) for s in starts]);scores=probabilities(model,windows)
    rms=np.sqrt(np.mean(windows*windows,axis=1));scores[rms<10**(SCAN['minimum_rms_dbfs']/20)]=0
    return regions_from_scores(starts/SR,scores,len(audio)/SR,threshold),scores

def overlap(a,b):return max(0,min(a[1],b[1])-max(a[0],b[0]))

def one_to_one_matches(truth,detections,min_iou=.10):
    pairs=[]
    for ti,t in enumerate(truth):
        for di,d in enumerate(detections):
            intersection=overlap(t,d);union=max(t[1],d[1])-min(t[0],d[0])
            iou=intersection/max(union,1e-9)
            if iou>=min_iou:pairs.append((iou,ti,di))
    used_truth=set();used_detections=set()
    for _,ti,di in sorted(pairs,reverse=True):
        if ti not in used_truth and di not in used_detections:
            used_truth.add(ti);used_detections.add(di)
    return len(used_truth)

def main():
    os.chdir(PROJECT);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    rows=[r for r in csv.DictReader((PROJECT/'dataset/candidates/labels.csv').open(encoding='utf-8'))
          if r['label'] in {'breath','speech','noise','silence'} and 'external' not in r['source'].lower()]
    sources=sorted({r['source'] for r in rows if Path(r['source']).exists()})
    decoded={s:read_audio(s).samples for s in sources};records=[];centered=[];expanded=[];expanded_y=[];expanded_source=[]
    for row in rows:
        source=row['source'];
        if source not in decoded:continue
        center=(float(row['start'])+float(row['end']))/2;label=1 if row['label']=='breath' else 0
        records.append(row);centered.append(window_at(decoded[source],center));
        for shift in [-.16,-.08,0,.08,.16]:
            expanded.append(window_at(decoded[source],center,shift));expanded_y.append(label);expanded_source.append(source)
    centered=np.stack(centered);expanded=np.stack(expanded);expanded_y=np.array(expanded_y,np.float32);expanded_source=np.array(expanded_source)
    labels=np.array([1 if r['label']=='breath' else 0 for r in records],np.float32);record_sources=np.array([r['source'] for r in records])
    base_meta=json.loads((ROOT/'models/breath_cnn_v2.json').read_text());base_path=ROOT/'models'/base_meta['weights_file']
    base_state=torch.load(base_path,map_location='cpu',weights_only=True)
    seed_reports=[]
    for seed in [17,43,89]:
        oof=np.zeros(len(records),np.float32);folds={};fullscan={}
        for hold in sources:
            train_mask=expanded_source!=hold;val_mask=record_sources==hold
            model,history=train_model(base_state,expanded[train_mask],expanded_y[train_mask],12,seed)
            train_center_mask=record_sources!=hold;train_p=probabilities(model,centered[train_center_mask])
            t,_=choose_threshold(labels[train_center_mask],train_p)
            oof[val_mask]=probabilities(model,centered[val_mask])
            detections,_=scan(model,decoded[hold],t)
            truths=[(float(r['start']),float(r['end'])) for r in records if r['source']==hold and r['label']=='breath']
            speech=[(float(r['start']),float(r['end'])) for r in records if r['source']==hold and r['label']!='breath']
            fullscan[hold]={'threshold_from_other_recordings':t,'detections':len(detections),'truth_breaths':len(truths),
              'truth_events_matched_one_to_one_iou_0_10':one_to_one_matches(truths,[(d['start'],d['end']) for d in detections]),
              'detections_overlapping_labeled_nonbreath':sum(any(overlap(x,(d['start'],d['end']))>0 for x in speech) for d in detections),
              'detections_in_unannotated_time':sum(not any(overlap(x,(d['start'],d['end']))>0 for x in truths+speech) for d in detections)}
            folds[hold]={'candidate_metrics':metric(labels[val_mask],oof[val_mask],t),'train_loss':history,'threshold':t}
        threshold,overall=choose_threshold(labels,oof)
        report={'seed':seed,'threshold':threshold,'candidate_oof':metric(labels,oof,threshold),'folds':folds,'full_scan':fullscan,'oof':oof}
        seed_reports.append(report);print('SEED',seed,report['candidate_oof'],fullscan,flush=True)
    def rank(r):
        m=r['candidate_oof'];safe=m['speech_fpr']<=.04
        return(safe,m['recall'] if safe else 0,m['precision'] or 0)
    chosen=max(seed_reports,key=rank);seed=chosen['seed']
    final,history=train_model(base_state,expanded,expanded_y,15,seed)
    weight_name=f'breath_cnn_personal_v3_seed{seed}.pt';torch.save(final.state_dict(),ROOT/'models'/weight_name)
    weight_hash=hashlib.sha256((ROOT/'models'/weight_name).read_bytes()).hexdigest()
    metadata={'model_type':'breath_cnn_v2','version':'breath_cnn_personal_v3','created_at':'2026-09-15',
      'preprocess':PREPROCESS,'scan':SCAN,'weights_file':weight_name,'weights_sha256':weight_hash,
      'recommended_threshold':chosen['threshold'],'seed':seed,'status':'candidate','accepted':False,'requires_review':True,
      'base_model':'editto-cnn-20260907-v2','personal_recordings':len(sources),'personal_labeled_candidates':len(records),
      'context_windows_per_candidate':5,'augmentation':{'gain_db':[-9,6],'snr_db':[22,38],'time_shifts_seconds':[-.16,-.08,0,.08,.16]},
      'recording_held_out_candidate_validation':chosen['candidate_oof'],
      'limitations':['One speaker across four recordings; not general-use evidence.',
        'Non-breath intervals are candidate labels, not continuous full-recording annotation.',
        'Full-scan detections in unannotated time cannot be counted as false positives.',
        'A new locked recording is required before promotion.']}
    (ROOT/'models/breath_cnn_personal_v3.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    serial=[]
    for r in seed_reports:
        c=dict(r);c.pop('oof');serial.append(c)
    output={'selected_seed':seed,'recommended_threshold':chosen['threshold'],
      'event_matching_protocol':'Greedy one-to-one highest IoU, minimum IoU 0.10; fixed before final candidate report.', 'seeds':serial,
      'final_train_loss':history,'metadata':metadata}
    (ROOT/'personal_cnn_v3_report.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print('CHOSEN',json.dumps(metadata,indent=2),flush=True)

if __name__=='__main__':main()
