"""Small full-pipeline qualification, isolated from all 15 formal runs."""
from pathlib import Path
import json,copy,os,subprocess
import pandas as pd
from .manifest import digest,save_json,METHODS
from .orchestration import exclusive_lock,llm_session,preflight,code_hash,termination_guard,owned_process
from .stages import config_identity
from .freezing import canonical_training_pool

def small_role(df,images=4):
    parts=[]
    for method,g in df.groupby('method_id',sort=True):
        if g.evaluation_unit.eq('video_frame').all():
            vids=sorted(set(g.video_id))[:2 if method=='real' else 1]
            parts.append(g[g.video_id.isin(vids)])
        else:parts.append(g.sort_values('sample_id').head(images))
    return pd.concat(parts,ignore_index=True)

def prepare_pilot(config,project):
    root=Path(config['artifact_root']);folder=root/'pilot';data=folder/'data';data.mkdir(parents=True,exist_ok=True)
    main=json.loads((root/'data/freeze_complete.json').read_text());mapping={}
    associations=pd.read_parquet(root/'data/training_associations.parquet')
    train,_=canonical_training_pool(small_role(associations,images=8));train.to_parquet(data/'train.parquet',index=False);mapping['train.parquet']=digest(data/'train.parquet')
    for role in ['ff_test','cdf_test','calibration']:
        df=small_role(pd.read_parquet(root/'data'/f'{role}.parquet'))
        df.to_parquet(data/f'{role}.parquet',index=False);mapping[f'{role}.parquet']=digest(data/f'{role}.parquet')
    freeze=copy.deepcopy(main);freeze.update(manifest_sha256=mapping,pilot_only=True,formal_data_modified=False)
    save_json(data/'freeze_complete.json',freeze)
    save_json(folder/'prepare_complete.json',json.loads((root/'prepare_complete.json').read_text()))
    pc=copy.deepcopy(config);pc.update(artifact_root=str(folder),epochs=1,test_bootstrap=10,calibration_bootstrap=10,matched_head_max_iter=25)
    save_json(folder/'config.json',pc)
    preflight(pc,project)
    probe=json.loads((root/'probe_complete.json').read_text())
    probe.update(freeze_sha256=digest(data/'freeze_complete.json'),config_identity=config_identity(pc),code_sha256=code_hash(project),pilot_only=True)
    save_json(folder/'probe_complete.json',probe)
    return pc

def run_pilot(config,project):
    root=Path(config['artifact_root']);folder=root/'pilot';marker=root/'pilot_complete.json'
    if marker.exists():
        old=json.loads(marker.read_text())
        if old['code_sha256']!=code_hash(project) or old['freeze_sha256']!=digest(root/'data/freeze_complete.json') or old['probe_sha256']!=digest(root/'probe_complete.json'):raise ValueError('Pilot completion binding changed')
        if digest(folder/'analysis/clip_vit_b16_seed7/complete.json')!=old['analysis_marker_sha256']:raise ValueError('Pilot evidence changed')
        return old
    if (folder/'runs').exists():raise ValueError('Partial pilot exists; preserve it for diagnosis before any retry')
    pc=prepare_pilot(config,project)
    with termination_guard(),exclusive_lock(root/'queue.lock'),llm_session(root,config['physical_gpus']) as session:
        # Pilot configuration has its own freeze/probe/CPU evidence; same service owner/UUID.
        save_json(folder/'service_state.json',session)
        uid=session['device_uuids'][str(config['physical_gpus'][0])]
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=uid,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
        args=[str(Path(project)/'.venv/bin/python'),'-B',str(Path(project)/'run.py'),'worker','--config',str(folder/'config.json'),
              '--backbone','clip_vit_b16','--seed','7','--device-uuid',uid,'--session-id',session['session_id']]
        with (folder/'worker.log').open('w') as f:p=owned_process(args,cwd=project,env=env,stdout=f,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError('End-to-end GPU pilot failed; '+str(folder/'worker.log'))
    from .reporting import analyze_run
    result=analyze_run(pc,project,'clip_vit_b16',7)
    from .cache import load_features
    import torch,numpy as np
    state=torch.load(folder/'runs/clip_vit_b16_seed7/final.pt',map_location='cpu',weights_only=True)
    df,h,w,b=load_features(folder/'features/clip_vit_b16_seed7/train')
    head=state['model'];rw=(head['classifier.weight'][1].double()-head['classifier.weight'][0].double()).numpy();rb=float(head['classifier.bias'][1].double()-head['classifier.bias'][0].double())
    if not np.array_equal(w,rw.astype(float)) or b!=rb:raise ValueError('Pilot cached signed head differs from final checkpoint')
    record={'status':'END_TO_END_PILOT_PASS','formal_runs_started':False,'epochs':1,'training_inputs':len(df),
        'training_cache_signed_head_exact':True,'four_flow_analysis_complete':True,'target_outcomes_used_for_selection':False,
        'code_sha256':code_hash(project),'freeze_sha256':digest(root/'data/freeze_complete.json'),'probe_sha256':digest(root/'probe_complete.json'),
        'analysis_marker_sha256':digest(folder/'analysis/clip_vit_b16_seed7/complete.json')}
    save_json(marker,record);return record
