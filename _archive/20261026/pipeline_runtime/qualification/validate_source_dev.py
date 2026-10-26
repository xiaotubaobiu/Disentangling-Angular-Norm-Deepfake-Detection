from pathlib import Path
import json,sys,time
import numpy as np,pandas as pd,torch
from reliability_v2.models import build_model
from reliability_v2.trainer import policy,digest,save_json,source_metrics,loader,fp64_margin,empirical_threshold,code_hash
root=Path(sys.argv[1]);backbone=sys.argv[2];policy()
gate=json.loads((root/'qualification/gpu_gate.json').read_text())
if gate['status']!='PASS' or gate['training_code_sha256']!=code_hash():raise ValueError('stale GPU qualification')
config=json.loads((root/'configs/base.json').read_text());weight=config['backbones'][backbone]
freeze=json.loads((root/'data/D/freeze.json').read_text())
for name in ['dev.parquet','source_cal.parquet']:
 if digest(root/'data/D'/name)!=freeze['manifest_sha256'][name]:raise ValueError('source manifest hash changed')
model=build_model(backbone,weight['weight_path']).cuda()
state=torch.load(root/'qualification'/backbone/'resume_epoch1.pt',map_location='cpu',weights_only=False)
model.load_state_dict(state['model']);del state
micro=gate['backbones'][backbone]['microbatch'];dev=pd.read_parquet(root/'data/D/dev.parquet')
if not dev.domain.eq('ff').all() or not dev.role.eq('dev').all():raise ValueError('source-only DEV guard')
started=time.time();metrics,scores=source_metrics(model,dev,backbone,micro,'cuda',4)
folder=root/'qualification'/backbone;scores.to_parquet(folder/'source_dev_video_scores.parquet',index=False)
cal=pd.read_parquet(root/'data/D/source_cal.parquet')
if not cal.label.eq(0).all() or not cal.role.eq('source_cal').all():raise ValueError('source-cal must be real-only')
parts=[];model.eval()
with torch.inference_mode():
 for x,y in loader(cal,backbone,micro,workers=4):parts.append(fp64_margin(model.encoder(x.cuda()).float(),model.classifier.weight,model.classifier.bias).cpu().numpy())
cal=cal[['video_uid','label']].copy();cal['score']=np.concatenate(parts)
video=cal.groupby(['video_uid','label'],as_index=False).score.mean();threshold=empirical_threshold(video.score,.05)
fp=int((video.score>=threshold).sum());allow=int(np.floor(.05*len(video)))
if fp>allow:raise ValueError('source real-only threshold failed FPR bound')
feasible=[float(np.nextafter(v,np.inf)) for v in sorted(set(video.score)) if (video.score>=np.nextafter(v,np.inf)).sum()<=allow]
if threshold!=min(feasible):raise ValueError('independent threshold minimal-boundary mismatch')
video.to_parquet(folder/'source_cal_video_scores.parquet',index=False)
save_json(folder/'source_dev_cal_acceptance.json',{'status':'PASS','backbone':backbone,'training_code_sha256':code_hash(),'data_freeze_sha256':digest(root/'data/D/freeze.json'),'source_dev_videos':len(scores),'source_dev_macro_auc':metrics['macro_auc'],'source_dev_method_count':len(metrics['method_auc']),'independent_AUC_recomputed':True,'source_cal_videos':len(video),'source_cal_threshold':threshold,'source_cal_FP':fp,'source_cal_allowed_FP':allow,'source_cal_AUC_called':False,'threshold_independently_verified':True,'target_used':False,'seconds':time.time()-started,'scientific_result':False})
print(backbone,'SOURCE_DEV_CAL_PASS',flush=True)
