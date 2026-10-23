"""CPU-only, fixed-score Detector calibration analysis. No model loading/inference."""
from pathlib import Path
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import argparse,collections,datetime,hashlib,importlib.util,json,math,platform,re,sys,time
import numpy as np
import pandas as pd
from scipy.special import expit

PROJECT=Path('/home/z/project/deepfake_detector')
ART=PROJECT.joinpath('artifacts').resolve()
BACKBONES=('clip_vit_b16','dinov2_vit_b14','efficientnet_b0')
SEEDS=(7,42,123,2026,3407)
FLOWS={'Raw':'video_RL','L2':'video_DL'}
SIZES=(10,20,40,80)
REPEATS=200
RNG_SEED=20261022
SALT='Detector-CDF-real-calibration-20261022-v1'
META=['sample_id','domain','experiment_role','method_id','method_type','evaluation_unit','video_id','group_id','label','weight']

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 b=(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode();atomic(p,b)
def atomic(p,b):
 p=Path(p);tmp=p.with_name(p.name+'.tmp');assert not p.exists()and not tmp.exists()
 with tmp.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p);assert p.read_bytes()==b
def csv(p,df):atomic(p,df.to_csv(index=False,float_format='%.17g').encode())
def rank(s):return hashlib.sha256((SALT+'|'+s).encode()).hexdigest()
def tokens(v):return set(re.findall(r'(?:^|_)(id\d+)(?=_|$)',str(v)))
def load_meta():
 q=ART/'analysis/clip_vit_b16_seed7/scores/cdf/video_RL.parquet'
 df=pd.read_parquet(q,columns=META).sort_values('sample_id').reset_index(drop=True)
 assert len(df)==13053 and not df.sample_id.duplicated().any() and (df.weight==1).all()
 assert df.domain.eq('cdf').all()and df.evaluation_unit.eq('video').all()
 assert (df.label.eq(0)==df.method_id.eq('real')).all()
 return df
def independent_threshold(x,budget=.05):
 x=np.asarray(x,dtype=np.float64)
 if x.ndim!=1 or len(x)==0 or not np.isfinite(x).all():raise ValueError('finite real-only calibration required')
 k=math.floor(len(x)*budget+1e-12)
 boundary=np.sort(x)[len(x)-k-1]
 return float(np.nextafter(boundary,np.inf))
def original_threshold():
 spec=importlib.util.spec_from_file_location('detector_original_scoring',PROJECT/'src/deepfake_detector/scoring.py')
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m.fpr_threshold
def self_checks():
 ref=original_threshold();checks=0
 for n in [10,20,40,80,140]:
  for x in [np.arange(n,dtype=float),np.zeros(n),np.arange(n,dtype=float)//3,np.linspace(-50,50,n)]:
   t=independent_threshold(x);assert t==ref(x)and np.mean(x>=t)<=.05+1e-14
   assert t==independent_threshold(x[::-1]);checks+=1
 assert independent_threshold(np.arange(10.))>9
 assert np.sum(np.arange(20.)>=independent_threshold(np.arange(20.)))==1
 z=np.array([-1000.,-1.,0.,1.,1000.]);assert np.array_equal(expit(z)>=.5,z>=0)
 for bad in [[],[np.nan],[np.inf]]:
  try:independent_threshold(bad)
  except ValueError:checks+=1
  else:raise AssertionError('invalid calibration accepted')
 return {'threshold_cases':checks,'original_algorithm_bitwise_equal':True,'tie_and_10_sample_zero_empirical_FPR_checked':True,'probability_half_equals_logit_zero':True}

def lock(out):
 assert not(out/'protocol.json').exists()
 df=load_meta();real=df[df.label.eq(0)];assert len(real)==177
 # Metadata only; never inspect scores while defining groups, pool or test set.
 frames=pd.read_parquet(ART/'data/cdf_test.parquet',columns=['domain','method_id','video_id','group_id','label','evaluation_unit','image_path','rgb_sha256','content_sha256'])
 frames=frames[frames.evaluation_unit.eq('video_frame')].copy()
 frames['video_sample_id']=frames.domain+'/'+frames.method_id+'/'+frames.video_id
 assert set(frames.video_sample_id)==set(df.sample_id)
 assert frames.groupby('video_sample_id').size().eq(8).all()
 rm=frames[frames.label.eq(0)].drop_duplicates('video_sample_id').set_index('video_sample_id')
 celeb=[];youtube=[];groups=collections.defaultdict(list)
 for sid in real.sample_id:
  row=rm.loc[sid];tt=tokens(row.video_id)
  if '/Celeb-real/'in row.image_path:
   assert len(tt)==1;g='celeb_name_proxy:'+next(iter(tt));celeb.append(sid)
  else:
   assert '/YouTube-real/'in row.image_path and not tt;g='youtube_video_proxy:'+row.video_id;youtube.append(sid)
  groups[g].append(sid)
 assert len(celeb)==107 and len(youtube)==70
 # Preserve origin mix approximately: 48 Celeb + 32 YouTube = 80.
 # Choose complete Celeb name-proxy groups, in predeclared hash order, exactly48.
 gs=sorted([g for g in groups if g.startswith('celeb_')],key=rank)
 feasible=[set()for _ in range(len(gs)+1)];feasible[-1]={0}
 for i in range(len(gs)-1,-1,-1):feasible[i]=feasible[i+1]|{v+len(groups[gs[i]])for v in feasible[i+1]if v+len(groups[gs[i]])<=48}
 remaining=48;chosen=[]
 for i,g in enumerate(gs):
  n=len(groups[g])
  if remaining>=n and remaining-n in feasible[i+1]:chosen.append(g);remaining-=n
 assert remaining==0
 selected=sum([groups[g]for g in chosen],[])+sorted(youtube,key=rank)[:32]
 selected=sorted(selected);assert len(selected)==len(set(selected))==80
 cal_ids=set(selected);cal_frame=frames[frames.video_sample_id.isin(cal_ids)]
 cal_groups=set(cal_frame.group_id);cal_rgb=set(cal_frame.rgb_sha256);cal_bytes=set(cal_frame.content_sha256)
 cal_tokens=set().union(*(tokens(v)for v in df.loc[df.sample_id.isin(cal_ids),'video_id']))
 content_related=set(frames.loc[frames.group_id.isin(cal_groups)|frames.rgb_sha256.isin(cal_rgb)|frames.content_sha256.isin(cal_bytes),'video_sample_id'])
 reasons={}
 for row in df.itertuples():
  why=[]
  if row.sample_id in cal_ids:why.append('calibration_pool')
  if row.sample_id in content_related:why.append('frozen_group_or_content_link')
  if tokens(row.video_id)&cal_tokens:why.append('shared_id_name_proxy')
  reasons[row.sample_id]=why
 test=df[[not reasons[sid]for sid in df.sample_id]].copy()
 assert test.label.eq(0).sum()==97
 methods=sorted(df.loc[df.label.eq(1),'method_id'].unique());assert len(methods)==21
 assert sorted(test.loc[test.label.eq(1),'method_id'].unique())==methods
 tf=frames[frames.video_sample_id.isin(test.sample_id)]
 assert not(set(tf.group_id)&cal_groups or set(tf.rgb_sha256)&cal_rgb or set(tf.content_sha256)&cal_bytes)
 assert not set().union(*(tokens(v)for v in test.video_id))&cal_tokens
 memberships=df.copy();memberships['role']=['calibration_pool'if sid in cal_ids else'heldout_test'if not reasons[sid]else'excluded_associated'for sid in df.sample_id]
 memberships['exclusion_reasons']=[';'.join(reasons[sid])for sid in df.sample_id]
 memberships['identity_name_proxy_tokens']=[';'.join(sorted(tokens(v)))for v in df.video_id]
 permutations=np.array([np.random.default_rng(np.random.SeedSequence([RNG_SEED,r])).permutation(80)for r in range(REPEATS)],dtype=np.int16)
 assert np.all(np.sort(permutations,axis=1)==np.arange(80))
 models=[]
 receipts=read(PROJECT/'reports/completed_model_state_receipts.json')['completed_models']
 for backbone in BACKBONES:
  for seed in SEEDS:
   name=f'{backbone}_seed{seed}';root=ART/'analysis'/name;complete=read(root/'complete.json');ident=read(root/'run.json')
   receipt=receipts[name];cksha=complete['identity']['inputs_sha256']['checkpoint']
   assert complete['status']=='COMPLETE'and receipt['epoch']==15 and cksha==receipt['final_model_sha256']
   assert ident['backbone']==backbone and ident['seed']==seed
   inputs={}
   for role in ['calibration','cdf']:
    for flow in FLOWS.values():
     rel=f'scores/{role}/{flow}.parquet';q=root/rel;h=sha(q);assert h==complete['outputs_sha256'][rel];inputs[rel]=h
   th=root/'thresholds.json';assert sha(th)==complete['outputs_sha256']['thresholds.json'];inputs['thresholds.json']=sha(th)
   models.append({'run_id':name,'backbone':backbone,'seed':seed,'checkpoint_sha256':cksha,'epoch':15,'analysis_complete_sha256':sha(root/'complete.json'),'run_identity_sha256':sha(root/'run.json'),'input_sha256':inputs})
 csv(out/'video_split.csv',memberships)
 csv(out/'calibration_pool.csv',pd.DataFrame({'pool_index':range(80),'sample_id':selected,'origin':['Celeb-real'if sid in celeb else'YouTube-real'for sid in selected]}))
 with (out/'calibration_permutations.npy').open('xb')as f:np.save(f,permutations,allow_pickle=False)
 protocol={'version':'Detector_CDF_calibration_fixed_scores_v1','locked_UTC':now(),'project':str(PROJECT),'project_git_commit':'674fa4a362f2b7ce6382992238a9b975e7f9fe0a','stage':'locked_before_new_test_metrics','backbones':BACKBONES,'model_seeds':SEEDS,'models':models,'flows':FLOWS,'aggregation':'mean of eight frozen RGB frame logits, then sigmoid for probability; original head unchanged','Raw_definition':'mean(w dot h + b)','L2_definition':'mean(w dot (h/max(norm2(h),1e-12)) + b), inference-only direction ablation; not L2 training regularization or a refitted head','fixed_probability_threshold':.5,'equivalent_fixed_logit_threshold':0.,'n0':'reuse each model/flow original source-real FPR5 threshold, independently recomputed from140 FF-val videos','target_FPR':.05,'threshold_rule':'minimum observed value or nextafter successor giving empirical inclusive tail <=5%; independently equals nextafter(sorted_real[n-floor(.05*n)-1],+inf); no interpolation/randomization','prediction':'fake iff native mean-logit score >= native threshold; probability>=0.5 for fixed0.5 row','ties':'retain inclusive>= and record boundary multiplicity; never break ties randomly; sigmoid-transformed thresholds are descriptive only due to floating-point saturation','calibration_sizes':SIZES,'repetitions':REPEATS,'sampling_seed':RNG_SEED,'sampling':'per-repeat SeedSequence([20261022,r]), uniform permutation of fixed80; nested prefixes10/20/40/80; shared by all models/flows; n80 identical across repeats','pool_selection':'score-blind stable SHA256 order with salt '+SALT+';48 Celeb-real videos from whole id-name-proxy groups via deterministic feasible subset,32 YouTube-real videos; unchanged test set for all six rows','pool_videos':80,'test_real_videos':int(test.label.eq(0).sum()),'test_fake_videos':int(test.label.eq(1).sum()),'test_fake_methods':methods,'test_per_method':{str(k):int(v)for k,v in test.groupby('method_id').size().items()},'split_roles':{str(k):int(v)for k,v in memberships.role.value_counts().items()},'excluded_id_name_tokens':sorted(cal_tokens),'split_independence':'disjoint video units, frozen content-connected group_id, cached byte/RGB hashes and conservative id-name tokens; name tokens are exclusion proxies, NOT authenticated identity metadata; unknown YouTube identity/driver relations remain unverified','primary_metrics':'FPR on97 shared real videos; macroTPR equal mean across21 fixed fake methods; macroBA=(1-FPR+macroTPR)/2; percent displayed','uncertainty':'per model first average200 sampling replicates; main mean and sampleSD across5 model-seed means separately for each backbone; sampling SD retained separately; not pooled15 IID runs, not a population CI','historical_scope':'retrospective new target-real calibration analysis explicitly authorized2026-10-22; original historical report remains source-only and unchanged; no target-fake threshold tuning, model selection, training or inference','hashes':{'cdf_manifest':sha(ART/'data/cdf_test.parquet'),'code':sha(Path(__file__)),'video_split.csv':sha(out/'video_split.csv'),'calibration_pool.csv':sha(out/'calibration_pool.csv'),'calibration_permutations.npy':sha(out/'calibration_permutations.npy')},'source_code_sha256':{n:sha(PROJECT/'src/deepfake_detector'/n)for n in ['scoring.py','analysis.py','experiments.py']},'self_checks':self_checks()}
 save(out/'protocol.json',protocol)
 print(json.dumps({'stage':'locked','UTC':now(),'protocol_sha256':sha(out/'protocol.json'),'pool':80,'test_real':protocol['test_real_videos'],'test_fake':protocol['test_fake_videos'],'per_method':protocol['test_per_method'],'role_counts':protocol['split_roles'],'excluded_identity_name_tokens':len(cal_tokens)},ensure_ascii=False),flush=True)

def direct_metrics(z,y,method,threshold,methods):
 pred=np.asarray(z)>=threshold
 fpr=float(pred[y==0].mean());tprs=np.array([pred[method==m].mean()for m in methods]);tpr=float(tprs.mean())
 return np.array([.5*(1-fpr+tpr),fpr,tpr]),tprs

def calculate(out):
 protocol=read(out/'protocol.json');assert protocol['hashes']['code']==sha(Path(__file__))
 assert not(out/'complete.json').exists()and not(out/'by_seed.csv').exists()
 for n in ['video_split.csv','calibration_pool.csv','calibration_permutations.npy']:assert sha(out/n)==protocol['hashes'][n]
 split=pd.read_csv(out/'video_split.csv',dtype={'video_id':str});pool=pd.read_csv(out/'calibration_pool.csv');perms=np.load(out/'calibration_permutations.npy',allow_pickle=False)
 test_ids=split.loc[split.role.eq('heldout_test'),'sample_id'].tolist();pool_ids=pool.sample_id.tolist();methods=protocol['test_fake_methods']
 source_ref=original_threshold();meta=load_meta();byseed=[];bymethod=[];rep_records=[];binding=[];maxerr=0.;start=time.monotonic()
 for spec in protocol['models']:
  root=ART/'analysis'/spec['run_id'];thresholds=read(root/'thresholds.json')
  assert sha(root/'complete.json')==spec['analysis_complete_sha256']
  for name,flow in FLOWS.items():
   for role in ['calibration','cdf']:
    rel=f'scores/{role}/{flow}.parquet';assert sha(root/rel)==spec['input_sha256'][rel]
   full=pd.read_parquet(root/f'scores/cdf/{flow}.parquet').sort_values('sample_id').reset_index(drop=True)
   pd.testing.assert_frame_equal(full[META],meta,check_dtype=True)
   assert np.isfinite(full.score).all()and np.isfinite(full.probability).all()
   assert np.array_equal(expit(full.score.to_numpy()),full.probability.to_numpy())
   assert np.array_equal(full.probability.to_numpy()>=.5,full.score.to_numpy()>=0)
   source=pd.read_parquet(root/f'scores/calibration/{flow}.parquet');assert len(source)==140 and source.label.eq(0).all()and source.experiment_role.eq('calibration').all()and source.weight.eq(1).all()
   t0=independent_threshold(source.score.to_numpy());stored=thresholds['FPR5']['thresholds'][flow]['threshold']
   assert t0==source_ref(source.score.to_numpy())==stored and np.mean(source.score>=t0)==.05
   indexed=full.set_index('sample_id');test=indexed.loc[test_ids];cal=indexed.loc[pool_ids]
   assert cal.label.eq(0).all()and len(cal)==80 and len(test)==len(test_ids)
   z=test.score.to_numpy();pr=test.probability.to_numpy();y=test.label.to_numpy();method=test.method_id.to_numpy();cs=cal.score.to_numpy()
   thresholds_all={'p0.5':np.array([0.]),'n0':np.array([t0])}
   for n in SIZES:thresholds_all[f'n{n}']=np.array([independent_threshold(cs[order[:n]])for order in perms])
   sorted_real=np.sort(z[y==0]);sorted_fake={m:np.sort(z[method==m])for m in methods}
   row_binding={'run_id':spec['run_id'],'flow':flow,'checkpoint_sha256':spec['checkpoint_sha256'],'source_threshold':t0,'source_FPR':float(np.mean(source.score>=t0)),'score_min':float(full.score.min()),'score_max':float(full.score.max()),'probability_exact_zero':int(full.probability.eq(0).sum()),'probability_exact_one':int(full.probability.eq(1).sum())}
   binding.append(row_binding)
   for rowkey,ts in thresholds_all.items():
    n=0 if rowkey in ['p0.5','n0']else int(rowkey[1:]);vals=[];pm=[];tie=[];cf=[];prob_mismatches=[]
    for r,t in enumerate(ts):
     fpr=(len(sorted_real)-np.searchsorted(sorted_real,t,side='left'))/len(sorted_real)
     tpr_m=np.array([(len(sorted_fake[m])-np.searchsorted(sorted_fake[m],t,side='left'))/len(sorted_fake[m])for m in methods]);v=np.array([.5*(1-fpr+tpr_m.mean()),fpr,tpr_m.mean()])
     # Independent direct Boolean recomputation of every repeat/seed/flow.
     v2,tm2=direct_metrics(pr if rowkey=='p0.5'else z,y,method,.5 if rowkey=='p0.5'else t,methods)
     e=max(float(np.max(np.abs(v-v2))),float(np.max(np.abs(tpr_m-tm2))));assert e<=1e-14;maxerr=max(maxerr,e)
     if n:
      cx=cs[perms[r,:n]];assert t==source_ref(cx)
      achieved=float(np.mean(cx>=t));assert achieved<=.05+1e-14
      boundary=np.sort(cx)[n-math.floor(.05*n+1e-12)-1];ties=int(np.sum(cx==boundary))
     else:achieved=float(np.mean(source.score>=t))if rowkey=='n0'else None;ties=0
     mismatch=int(np.sum((pr>=expit(t))!=(z>=t)))
     vals.append(v);pm.append(tpr_m);tie.append(ties);cf.append(achieved);prob_mismatches.append(mismatch)
     rep_records.append((spec['run_id'],name,rowkey,r,float(t),*v.tolist(),ties,achieved,mismatch))
    va=np.array(vals);pp=np.array(pm);mean=va.mean(0);sd=va.std(0,ddof=1)if len(va)>1 else np.zeros(3)
    item={'backbone':spec['backbone'],'model_seed':spec['seed'],'flow':name,'row':rowkey,'n_CDF_calibration':n,'calibration_repeats':len(va),'threshold_mean':float(ts.mean()),'threshold_min':float(ts.min()),'threshold_max':float(ts.max()),'calibration_FPR_mean':float(np.mean([v for v in cf if v is not None]))if any(v is not None for v in cf)else None,'max_calibration_boundary_ties':max(tie),'max_sigmoid_threshold_prediction_mismatches':max(prob_mismatches)}
    for k,v,sdev in zip(['BA','FPR','TPR'],mean,sd):item[k+'_mean']=float(v);item[k+'_sampling_SD']=float(sdev)
    byseed.append(item)
    for m,mu in zip(methods,pp.mean(0)):bymethod.append({'backbone':spec['backbone'],'model_seed':spec['seed'],'flow':name,'row':rowkey,'method':m,'TPR_mean':float(mu),'fake_test_videos':len(sorted_fake[m])})
  print(json.dumps({'run_complete':spec['run_id'],'elapsed_seconds':time.monotonic()-start}),flush=True)
 bs=pd.DataFrame(byseed);summary=[]
 for (backbone,flow,rowkey),g in bs.groupby(['backbone','flow','row'],sort=False):
  assert sorted(g.model_seed)==list(SEEDS);item={'backbone':backbone,'flow':flow,'row':rowkey,'model_seeds':5,'calibration_repeats':int(g.calibration_repeats.iloc[0])}
  for metric in ['BA','FPR','TPR']:
   item[metric+'_mean']=float(g[metric+'_mean'].mean());item[metric+'_model_seed_SD']=float(g[metric+'_mean'].std(ddof=1));item[metric+'_sampling_SD_RMS']=float(np.sqrt(np.mean(g[metric+'_sampling_SD']**2)))
  summary.append(item)
 sm=pd.DataFrame(summary);csv(out/'by_seed.csv',bs);csv(out/'summary_by_backbone.csv',sm);csv(out/'per_method_by_seed.csv',pd.DataFrame(bymethod))
 rr=pd.DataFrame(rep_records,columns=['run_id','flow','row','repeat','threshold_logit','BA','FPR','TPR','calibration_boundary_ties','calibration_FPR','sigmoid_threshold_prediction_mismatches'])
 rr.to_parquet(out/'paired_replicates.parquet',index=False)
 save(out/'input_binding_and_probability_audit.json',binding)
 labels={'p0.5':'固定概率阈值0.5','n0':'0（FF++源阈值）','n10':'10','n20':'20','n40':'40','n80':'80'}
 title={'clip_vit_b16':'CLIP ViT-B/16','dinov2_vit_b14':'DINOv2 ViT-B/14','efficientnet_b0':'EfficientNet-B0'}
 text='# Detector：CDF 真实视频校准表\n\n每格为 **宏BA / FPR / 宏TPR（%）**。各骨干分别汇总5个固定模型种子；10/20/40/80行先在各模型内平均200次共同的配对抽样，再平均5模型。Raw=原特征原head的8帧mean-logit；L2=推理时特征单位方向、原head不变，同样mean-logit。不是重新训练或重拟合head。\n\n'
 for backbone in BACKBONES:
  text+='## '+title[backbone]+'\n\n| CDF真实校准视频数 | Raw：BA / FPR / TPR | L2：BA / FPR / TPR |\n|---|---:|---:|\n'
  for rowkey in labels:
   cells=[]
   for flow in FLOWS:
    q=sm[(sm.backbone==backbone)&(sm.flow==flow)&(sm.row==rowkey)].iloc[0];cells.append(' / '.join(f'{q[k+"_mean"]*100:.2f}'for k in ['BA','FPR','TPR']))
   text+=f'| {labels[rowkey]} | {cells[0]} | {cells[1]} |\n'
  text+='\n'
 text+=f'''统一测试集：{protocol['test_real_videos']}条真实、{protocol['test_fake_videos']}条伪造视频，21种方法均保留。真实校准池80条（48 Celeb-real、32 YouTube-real），全部六行都使用同一剩余测试集合，因此第0行不能与原完整CDF测试主表直接混比。采用固定分割，未根据新测试性能优化任何设置。原报告历史结论不改写；本表是用户2026-10-22新授权的回顾性目标域真实样本校准分析。

校准与测试按视频去重，并排除共享既有来源组、Byte/RGB内容哈希和`idN`名称标记的视频。名称标记只是保守排除代理，不认证人物身份；YouTube人物/未知驱动祖先缺口仍存在，不能称人口独立盲测。

0.5行直接计算`sigmoid(mean_logit)>=0.5`，并验证与`mean_logit>=0`逐条一致，绝非logit阈值0.5。0行按原协议从140条FF真实视频独立重算并核对原冻结阈值。10/20/40/80行只使用对应数量CDF真实分数；无CDF伪造视频参与阈值选择。阈值为使校准池经验FPR≤5%的最小观测值或其FP64严格后继，预测Fake采用`score>=threshold`，ties不随机拆分。10条时误报分辨率10%，故校准经验FPR必须为0%；20/40/80条最多允许1/2/4个误报。目标测试FPR不保证正好5%。阈值在原生logit域比较，避免sigmoid饱和/舍入使相邻阈值合并；转换后的概率阈值仅作说明。

抽样：固定80池内每次无放回排列，10/20/40/80为嵌套前缀；200次重复在所有骨干、模型种子和Raw/L2之间完全共享。80行不随排列变化，其抽样SD应为0。`summary_by_backbone.csv`保留5模型均值、模型种子样本SD，以及模型内校准抽样SD的RMS，二者不混合；不是总体置信区间。`by_seed.csv`保留180个种子×流程×行结果，`paired_replicates.parquet`保留全部重复、阈值、ties与经验校准FPR；`per_method_by_seed.csv`保留逐方法TPR。没有把三骨干当15个同分布种子合并。

每个种子的完整代码/源数据/分数/模型checkpoint绑定见`protocol.json`和`input_binding_and_probability_audit.json`；分割见`video_split.csv`、`calibration_pool.csv`及`calibration_permutations.npy`。冻结模型都是epoch15 final。原有计分定义已从项目代码核对；本次只复用保存分数，零GPU推理、零训练、零下载，不修改Jev论文。

复现：在同一固定项目与原始分数仍在的环境，复制本目录的`calibration_table.py`到新的输出目录，先运行`python -B calibration_table.py --output NEW_DIR lock`锁定，再运行`python -B calibration_table.py --output NEW_DIR calculate`。输出目录须提前建立并先通过容量准入；不得覆盖本次证据。`verify`重新从分数与冻结分割独立核验已保存重复及汇总，零训练。
'''
 atomic(out/'结果与方法.md',text.encode())
 sizes={str(p.relative_to(out)):p.stat().st_size for p in out.rglob('*')if p.is_file()}
 result={'status':'computed_pending_independent_readback','UTC':now(),'elapsed_seconds':time.monotonic()-start,'model_flow_rows':len(bs),'summary_rows':len(sm),'replicates':len(rr),'all_repeats_independent_boolean_crosscheck_max_error':maxerr,'protocol_sha256':sha(out/'protocol.json'),'source_project_unchanged':True,'GPU_used':False,'training_or_inference':False,'outputs_sha256':{n:sha(out/n)for n in sizes},'output_bytes':sum(sizes.values()),'environment':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'platform':platform.platform()}}
 save(out/'computed.json',result)
 print(json.dumps({'stage':'computed','summary':summary,'proof':{k:v for k,v in result.items()if k not in ['outputs_sha256','environment']}},ensure_ascii=False),flush=True)

def verify(out):
 protocol=read(out/'protocol.json');computed=read(out/'computed.json')
 for n,h in computed['outputs_sha256'].items():assert sha(out/n)==h,(n,'hash changed')
 assert computed['protocol_sha256']==sha(out/'protocol.json')
 rep=pd.read_parquet(out/'paired_replicates.parquet');bs=pd.read_csv(out/'by_seed.csv');summary=pd.read_csv(out/'summary_by_backbone.csv')
 split=pd.read_csv(out/'video_split.csv',dtype={'video_id':str});pool=pd.read_csv(out/'calibration_pool.csv');perms=np.load(out/'calibration_permutations.npy',allow_pickle=False);test_ids=split.loc[split.role.eq('heldout_test'),'sample_id'];methods=protocol['test_fake_methods']
 assert len(bs)==180 and len(summary)==36 and len(rep)==24060
 max_error=0.;cases=0
 for model in protocol['models']:
  for flow,stored_flow in FLOWS.items():
   full=pd.read_parquet(ART/'analysis'/model['run_id']/f'scores/cdf/{stored_flow}.parquet').set_index('sample_id')
   test=full.loc[test_ids];cal=full.loc[pool.sample_id]
   r=rep[(rep.run_id==model['run_id'])&(rep.flow==flow)]
   for row in r.itertuples():
    if row.row.startswith('n')and row.row!='n0':
     n=int(row.row[1:]);cx=cal.score.to_numpy()[perms[row.repeat,:n]];assert independent_threshold(cx)==row.threshold_logit
    v,_=direct_metrics(test.probability.to_numpy()if row.row=='p0.5'else test.score.to_numpy(),test.label.to_numpy(),test.method_id.to_numpy(),.5 if row.row=='p0.5'else row.threshold_logit,methods)
    e=float(np.max(np.abs(v-[row.BA,row.FPR,row.TPR])));assert e<=1e-14;max_error=max(e,max_error);cases+=1
   for rowkey,g in r.groupby('row'):
    saved=bs[(bs.backbone==model['backbone'])&(bs.model_seed==model['seed'])&(bs.flow==flow)&(bs.row==rowkey)].iloc[0]
    for metric in ['BA','FPR','TPR']:
     assert abs(g[metric].mean()-saved[metric+'_mean'])<1e-13
     expected=g[metric].std(ddof=1)if len(g)>1 else 0.;assert abs(expected-saved[metric+'_sampling_SD'])<1e-13
 for group,g in bs.groupby(['backbone','flow','row']):
  row=summary[(summary.backbone==group[0])&(summary.flow==group[1])&(summary.row==group[2])].iloc[0]
  for metric in ['BA','FPR','TPR']:
   assert abs(g[metric+'_mean'].mean()-row[metric+'_mean'])<1e-13
   assert abs(g[metric+'_mean'].std(ddof=1)-row[metric+'_model_seed_SD'])<1e-13
 assert bs.loc[bs.row.eq('n80'),['BA_sampling_SD','FPR_sampling_SD','TPR_sampling_SD']].to_numpy().max()<1e-13
 proof={'status':'complete_independent_CPU_recalculation_and_readback_passed','UTC':now(),'verified_replicate_rows':cases,'maximum_metric_error':max_error,'complete_hash_readback':True,'same_test_for_all_six_rows':True,'all_models_and_flows_share_calibration_draws':True,'independent_metrics_and_seed_summary_verified':True,'model_seeds':list(SEEDS),'backbones':list(BACKBONES),'no_training_inference_or_GPU':True,'protocol_sha256':sha(out/'protocol.json'),'computed_sha256':sha(out/'computed.json'),'outputs_sha256':{str(p.relative_to(out)):sha(p)for p in out.rglob('*')if p.is_file()}}
 save(out/'complete.json',proof);print(json.dumps(proof,ensure_ascii=False),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('stage',choices=['lock','calculate','verify']);args=ap.parse_args();out=Path(args.output).resolve();assert out.is_dir()
 assert out.is_relative_to(PROJECT/'reports')
 {'lock':lock,'calculate':calculate,'verify':verify}[args.stage](out)
