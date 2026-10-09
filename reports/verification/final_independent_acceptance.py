"""Interactive CPU acceptance only, run after original supervisor terminal COMPLETE."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
import pathlib,sys,json,hashlib,datetime,subprocess,urllib.request
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score
P=pathlib.Path('/home/z/project/deepfake_detector');sys.path.insert(0,str(P/'src'))
from deepfake_detector.orchestration import code_hash,container_state
from deepfake_detector.stages import stage_identity,completed_stage
from deepfake_detector.manifest import digest
C=json.loads((P/'configs/v1.1.json').read_text());R=pathlib.Path(C['artifact_root']);OUT=P/'reports/final_independent_acceptance.json'
E=json.loads((R/'execution_status.json').read_text());Q=json.loads((R/'queue_status.json').read_text())
if E.get('status')!='COMPLETE':
 print(json.dumps({'status':'NOT_READY','execution_status':E.get('status'),'phase':E.get('phase')}));sys.exit(0)
CODE='7d56c6e5803450369852b5865130a3ed815d87e3e6a91226d276f0fd092611b5'
assert code_hash(P)==CODE==E['code_sha256']
assert digest(R/'data/freeze_complete.json')=='c3044b9219a8ef18dd1c25f7f75de0d38dd93642bbe4cc0e19381218bf820b5d'
assert digest(pathlib.Path('/home/z/obsidian/paper/deepfake-detector/实验重设计_3backbone_5seed_v1.1.md'))=='61e583963a4d661ccf83d02798600c83eb4aca2238d2984c5422bab1553ecc6b'
IDS=[f'{b}_seed{s}' for b in C['backbones'] for s in C['seeds']]
assert set(Q['runs'])==set(IDS) and all(v['state']=='COMPLETE' and v['returncode']==0 for v in Q['runs'].values())
models=json.loads((P/'reports/completed_model_state_receipts.json').read_text())['completed_models']
caches=json.loads((P/'reports/completed_run_cache_receipts.json').read_text())['completed_runs']
assert set(models)==set(caches)==set(IDS)
for rid in IDS:
 m=models[rid]
 assert m['epoch']==15 and m['draws']==3750000 and m['attempts']==29310
 assert m['successes']+m['AMP_skips']==m['attempts']
 assert m['all_epoch_sequences_exactly_reconstructed'] and m['final_model_matches_full_state'] and m['model_and_optimizer_finite'] and m['full_Python_NumPy_CPU_CUDA_RNG_present']
 assert caches[rid]['canonical_verified_run'] and caches[rid]['rows']==679977
 assert digest(R/'runs'/rid/'final.pt')==m['final_model_sha256']
 assert digest(R/'runs'/rid/'state.pt')==m['state_sha256']
REPORT=json.loads((R/'report/report.json').read_text());assert REPORT['status']=='ANALYSIS_REPORT_COMPLETE' and REPORT['runs']==15 and REPORT['seeds']==C['seeds']
assert REPORT['all_primary_metrics_reconstructed_from_saved_scores'] and REPORT['TPR95']=='UNAVAILABLE' and not REPORT['target_selection_performed']

def close(a,b,atol=1e-10):
 assert np.allclose(np.asarray(a),np.asarray(b),rtol=0,atol=atol,equal_nan=True),(float(np.nanmax(np.abs(np.asarray(a)-np.asarray(b)))),'numeric mismatch')
def weights(df):
 w=np.ones(len(df));mask=df.evaluation_unit.eq('video_frame').to_numpy()
 if mask.any():w[mask]=1/df.loc[mask].groupby(['domain','method_id','video_id']).video_id.transform('size').to_numpy()
 return w

def aggregate(df,z):
 out=df[['sample_id','domain','experiment_role','method_id','method_type','evaluation_unit','video_id','group_id','label']].copy();out['score']=z;out['probability']=expit(z)
 v=out[out.evaluation_unit.eq('video_frame')].groupby(['domain','method_id','video_id'],sort=True).agg(label=('label','first'),method_type=('method_type','first'),group_id=('group_id','first'),experiment_role=('experiment_role','first'),score=('score','mean'),probability=('probability','mean')).reset_index()
 v['sample_id']=v[['domain','method_id','video_id']].astype(str).agg('/'.join,axis=1);v['evaluation_unit']='video'
 l=v.copy();l['probability']=expit(l.score);p=v.copy();p['score']=p.probability
 return {'image':out,'L':l,'P':p}

def threshold(x,w,budget=.05):
 vals,code=np.unique(np.asarray(x),return_inverse=True);mass=np.bincount(code,weights=w,minlength=len(vals));above=np.r_[np.cumsum(mass[::-1])[::-1][1:],0.]/sum(mass)
 for i,v in enumerate(vals):
  if above[i]+mass[i]/sum(mass)<=budget+1e-14:return float(v)
  if above[i]<=budget+1e-14:return float(np.nextafter(v,np.inf))
 raise AssertionError('threshold candidate missing')

def summary(x,record):
 x=np.asarray(x,dtype=float);v=x[np.isfinite(x)];assert record['replicates']==len(x) and record['valid']==len(v) and record['invalid']==len(x)-len(v)
 if len(v):close(np.median(v),record['median']);close(np.quantile(v,[.025,.975]),record['conditional_95_interval'])
 else:assert record['median'] is None and record['conditional_95_interval'] is None

def audit_metrics(df,record):
 y=df.label.to_numpy();s=df.score.to_numpy();p=df.probability.to_numpy();w=weights(df);real=y==0;fake=~real;mv=df.method_id.to_numpy();methods=sorted(set(mv[fake]));t=float(df.threshold.iloc[0]);pred=s>=t
 out={'pooled_AUC':roc_auc_score(y,s,sample_weight=w),'method_macro_AUC':np.mean([roc_auc_score(y[real|(mv==m)],s[real|(mv==m)],sample_weight=w[real|(mv==m)]) for m in methods]),'FPR':np.average(pred[real],weights=w[real]),'macro_TPR':np.mean([np.average(pred[mv==m],weights=w[mv==m]) for m in methods]),'real_Brier':np.average(p[real]**2,weights=w[real]),'fake_Brier':np.average((1-p[fake])**2,weights=w[fake])}
 out['macro_BA']=(out['macro_TPR']+1-out['FPR'])/2;out['class_balanced_Brier']=(out['real_Brier']+out['fake_Brier'])/2;out['native_Brier']=np.average((p-y)**2,weights=w)
 for key,value in out.items():close(value,record[key],1e-12)
 return out

SUPPORT=json.loads((P/'reports/bootstrap_support_structure.json').read_text())
assert not SUPPORT['score_values_used'] and not SUPPORT['protocol_or_input_changes']
for role,sha in SUPPORT['frozen_manifest_sha256'].items():assert digest(R/'data'/(role+'.parquet'))==sha
SUPPORT_MAP={(x['role'],x['unit']):x for x in SUPPORT['bootstrap_support_structure']}
receipts=[];heads=[];control_receipts=[]
for backbone in C['backbones']:
 for seed in C['seeds']:
  rid=f'{backbone}_seed{seed}';A=R/'analysis'/rid;j=json.loads((A/'analysis.json').read_text());inputs={'freeze':R/'data/freeze_complete.json','checkpoint':R/'runs'/rid/'final.pt'}
  inputs.update({r:R/'features'/rid/r/'complete.json' for r in ['train','ff_test','cdf_test','calibration']})
  assert completed_stage(A,stage_identity('analysis_'+rid,C,P,inputs))
  assert not j['target_selection_performed'] and j['FPR5']['available'] and not j['TPR95']['available']
  assert len(j['B']['scale_controls'])==80 and len(j['B']['length_controls'])==24 and len(j['B']['pair_transitions'])==8
  assert len(j['B']['affine_controls'])==12 and all(r['prediction_mismatches']==0 for r in j['B']['affine_controls'])
  for pair in j['B']['pair_transitions']:close(pair['ledger_delta'],pair['direct_delta']);assert np.asarray(pair['table']).shape==(3,3)
  # These controls are specified by the original unchanged v1.1 mechanism design.
  scale_error=[];length_error=[]
  for domain in ['ff','cdf']:
   scales=[x for x in j['B']['scale_controls'] if x['domain']==domain]
   length=[x for x in j['B']['length_controls'] if x['domain']==domain]
   for metric in ['method_macro_AUC','pooled_AUC']:
    for kind in ['raw','direction']:
     for agg in ['image','L']:
      chosen=[x for x in scales if x['kind']==kind and x['aggregation']==agg];base=next(x[metric] for x in chosen if x['scale']==1)
      scale_error.extend(abs(x[metric]-base) for x in chosen)
    for agg in ['image','L']:
     chosen={x['condition']:x for x in length if x['aggregation']==agg}
     length_error.extend([abs(chosen['h'][metric]-chosen['h_over_c'][metric]),abs(chosen['u'][metric]-chosen['c_u'][metric])])
  assert max(scale_error)<=1e-10 and max(length_error)<=1e-10,(rid,'prespecified positive-affine ranking control')
  post=[x for x in j['B']['scale_controls'] if x['aggregation']=='post_mean_sigmoid']
  control_receipts.append({'run_id':rid,'scale_image_and_mean_logit_AUC_max_abs_difference':max(scale_error),'global_length_rescale_image_and_mean_logit_AUC_max_abs_difference':max(length_error),'synchronized_affine_prediction_mismatches':0,'post_mean_sigmoid_max_abs_finite_precision_AUC_delta':max(abs(x['finite_precision_AUC_delta']) for x in post),'post_mean_sigmoid_conditions_with_abs_delta_gt_1e_12':sum(abs(x['finite_precision_AUC_delta'])>1e-12 for x in post),'post_mean_sigmoid_max_exact_tied_video_pairs':max(x['exact_tied_video_pairs'] for x in post)})
  cal={};train_norms=[];maxdiff=0.
  for role,domain in [('train',None),('calibration','calibration'),('ff_test','ff'),('cdf_test','cdf')]:
   F=R/'features'/rid/role;meta=json.loads((F/'complete.json').read_text());df=pd.read_parquet(F/'manifest.parquet');W=np.asarray(meta['head']['weight'],dtype=float);B=np.asarray(meta['head']['bias'],dtype=float);w=W[1]-W[0];b=B[1]-B[0]
   raw=[];direction=[];norms=[];mh={k:[] for k in ['raw','direction']}
   for chunk in meta['chunks']:
    h=np.load(F/chunk['file'],allow_pickle=False).astype(np.float64);n=np.linalg.norm(h,axis=1);u=h/np.maximum(n,1e-12)[:,None];raw.append(h@w+b);direction.append(u@w+b);norms.append(n)
    if role!='train':
     for kind,x in [('raw',h),('direction',u)]:
      head=j['matched_heads'][kind];mh[kind].append(x@np.asarray(head['weight'])+head['bias'])
   norms=np.concatenate(norms)
   if role=='train':
    nt=pd.read_parquet(A/'training_norms.parquet');assert np.array_equal(nt.sample_id.to_numpy(),df.sample_id.to_numpy());close(nt.norm,norms)
    q=df.sampling_mass.to_numpy(dtype=float);close(nt.sampling_mass,q)
    order=np.argsort(norms,kind='stable');median=norms[order[np.searchsorted(np.cumsum(q[order]),.5,side='left')]];close(median,j['B']['source_training_median']);continue
   tables={}
   for kind,z in [('raw',np.concatenate(raw)),('direction',np.concatenate(direction))]:
    for agg,t in aggregate(df,z).items():tables['image_'+kind if agg=='image' else 'video_'+('R' if kind=='raw' else 'D')+agg]=t
   for flow,t in tables.items():
    saved=pd.read_parquet(A/'scores'/domain/(flow+'.parquet'));assert np.array_equal(saved.sample_id,t.sample_id)
    for col in ['label','method_id','group_id','experiment_role','evaluation_unit']:assert np.array_equal(saved[col],t[col]),(rid,role,flow,col)
    close(saved.score,t.score);close(saved.probability,t.probability);close(saved.weight,weights(t));maxdiff=max(maxdiff,float(np.max(np.abs(saved.score-t.score))))
    thr=j['FPR5']['thresholds'][flow]['threshold'];assert saved.threshold.eq(thr).all() and np.array_equal(saved.prediction.to_numpy(),(saved.score.to_numpy()>=thr).astype(int))
    if role=='calibration':
     assert saved.label.eq(0).all() and saved.domain.eq('ff').all() and saved.experiment_role.eq('calibration').all();assert len(saved)==(1120 if flow.startswith('image') else 140)
     assert threshold(saved.score,weights(saved))==thr
     close(np.average(saved.score>=thr,weights=weights(saved)),j['FPR5']['thresholds'][flow]['source_FPR']);assert j['FPR5']['thresholds'][flow]['source_FPR']<=.05+1e-14;cal[flow]=saved
    else:
     unit,name=flow.split('_',1);audit_metrics(saved,j['A'][domain][unit][name])
   if role!='calibration':
    for kind in ['raw','direction']:
     for agg,t in aggregate(df,np.concatenate(mh[kind])).items():
      saved=pd.read_parquet(A/'matched'/kind/domain/(agg+'.parquet'));assert np.array_equal(saved.sample_id,t.sample_id);close(saved.score,t.score);close(saved.probability,t.probability)
  for kind,head in j['matched_heads'].items():
   assert kind in ['raw','direction'] and head['L2']==1e-4 and not head['bias_penalized'] and head['initialization']=='zero' and not head['target_labels_used'] and not head['encoder_updated'] and not head['normalization_running_state_updated'];assert 0<=head['iterations']<=2000 and np.isfinite(head['weight']).all() and np.isfinite(head['bias']) and np.isfinite(head['objective'])
   heads.append({'run_id':rid,'kind':kind,**{k:head[k] for k in ['converged','iterations','message','objective']}})
  for domain in ['ff','cdf']:
   for unit in ['video','image']:
    scope=domain if unit=='video' else domain+'_image';m=j['D']['test_resampling'][scope];assert m['iterations']==2000 and not m['multiplicity_adjusted'];support=SUPPORT_MAP[(domain+'_test',unit)];assert m['groups']==support['groups'];expected_invalid=support['support_by_seed'][str(seed)]['macro_support_invalid']
    with np.load(A/'bootstrap'/f'test_{domain}_{unit}.npz',allow_pickle=False) as arrays:
     for flow,values in m['metrics'].items():
      for key,record in values.items():
       value=arrays[flow+'__'+key];assert value.shape==(2000,);summary(value,record);assert record['invalid']==m['invalid_replicates'][flow][key];assert record['invalid']==(expected_invalid if key in ['method_macro_AUC','macro_TPR','macro_BA'] else 0)
     for name,record in m['paired_effects'].items():
      metric,effect=name.split('__',1)
      if unit=='image':expected=arrays['direction__'+metric]-arrays['raw__'+metric]
      else:
       rl,rp,dl,dp=[arrays[f+'__'+metric] for f in ['RL','RP','DL','DP']];expected={'delta_raw':rp-rl,'delta_dir':dp-dl,'delta_norm_L':dl-rl,'delta_norm_P':dp-rp,'interaction':dp-dl-rp+rl}[effect]
      close(arrays['effect__'+name],expected);summary(expected,record)
  for unit in ['video','image']:
   table=pd.read_parquet(A/'bootstrap'/f'calibration_{unit}.parquet');m=j['D']['calibration_resampling'][unit];assert m['iterations']==1000
   flows=['RL','RP','DL','DP'] if unit=='video' else ['raw','direction'];assert set(table.flow)==set(flows)
   for flow in flows:
    t=table[table.flow.eq(flow)].sort_values('replicate');assert np.array_equal(t.replicate,np.arange(1000)) and t.valid.all()
    for key,record in m['summaries'][flow].items():summary(t[key].to_numpy(),record)
    source=cal[unit+'_'+flow];groups,codes=np.unique(source.group_id.astype(str),return_inverse=True);draws=np.random.default_rng(seed).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=1000)
    vals=source.score.to_numpy();ww=weights(source);independent=[]
    for d in draws:
     keep=d[codes]>0;independent.append(threshold(vals[keep],ww[keep]*d[codes[keep]]))
    assert np.array_equal(t.threshold.to_numpy(),independent),(rid,unit,flow,'calibration resample threshold')
  receipts.append({'run_id':rid,'saved_primary_scores_reconstructed_from_raw_cache':True,'max_absolute_primary_logit_difference':maxdiff,'source_only_frozen_thresholds_reconstructed':True,'primary_metrics_reconstructed':True,'all_stage_outputs_hash_verified':True,'paired_test_bootstrap_arrays_summaries_invalid_counts_verified':True,'independent_source_calibration_resample_thresholds_verified':True,'matched_head_scores_reconstructed':True})
  print(json.dumps({'verified_run':rid,'heads_converged':[j['matched_heads'][k]['converged'] for k in ['raw','direction']]}),flush=True)
for file in ['metrics_by_seed.csv','metrics_five_seed_summary.csv','metrics_by_seed_method.csv','paired_flow_differences.csv','paired_five_seed_summary.csv','scale_controls.csv','length_controls.csv','matched_heads.csv','pair_transitions.csv','test_bootstrap.csv','calibration_bootstrap.csv','image_FS_FR_EFS_by_seed.csv']:
 t=pd.read_csv(R/'report'/file);assert len(t)>0 and set(t.backbone)==set(C['backbones']) and all(set(g.seed)==set(C['seeds']) for _,g in t.groupby('backbone')) if 'seed' in t else len(t)>0
summary_table=pd.read_csv(R/'report/metrics_five_seed_summary.csv');assert summary_table.n.eq(5).all()
from PIL import Image
figures={}
for name in REPORT['stability_and_source_threshold_relative_figures']:
 path=R/'report'/name
 with Image.open(path) as im:assert im.format=='PNG' and min(im.size)>=1000;figures[name]={'pixels':im.size,'sha256':digest(path)};im.verify()
assert len(figures)==6 and len(heads)==30
original=json.loads((R/'service_state.json').read_text())['original'];actual=container_state();assert actual['id']==original['id']=='0de7633bc4eb9b0d12b41dd2d05e3bb42961310e62bcb1b78f0f2ed0f54aad5c' and actual['state']['Running'] and actual['restart']==original['restart']
with urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=15) as response:assert response.status==200
record={'status':'INDEPENDENT_FINAL_ACCEPTANCE_PASS','checked_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_code_sha256':CODE,'runs':receipts,'matched_heads':heads,'unconverged_matched_head_count':sum(not h['converged'] for h in heads),'figures':figures,'mechanism_negative_controls':control_receipts,'LLM_original_container_running_same_identity_and_restart_policy':True,'LLM_actual_health_HTTP':200,'scope_limitations':['unknown ancestry is not certified','TPR95 unavailable without complete independent fake calibration','individual AMP skipped-step event positions/scales not logged by frozen training source'],'acceptance_helper_sha256':os.environ.get('DF40_ACCEPTANCE_HELPER_SHA256')}
# The invoking operator records this helper source hash separately; no queue/source mutation.
tmp=OUT.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(OUT)
with (P/'reports/audit.log').open('a') as f:f.write(json.dumps({'event':'independent_final_acceptance','checked_UTC':record['checked_UTC'],'runs':15,'heads':30,'unconverged_heads':record['unconverged_matched_head_count']})+'\n')
print(json.dumps({'status':record['status'],'runs':15,'heads':30,'unconverged_heads':record['unconverged_matched_head_count'],'LLM_health_HTTP':200}))