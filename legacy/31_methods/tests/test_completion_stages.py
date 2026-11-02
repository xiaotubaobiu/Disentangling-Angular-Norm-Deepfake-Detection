import json
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
from deepfake_detector.freezing import source_role_filter

def test_unknown_video_sources_keep_released_roles_with_explicit_limits():
    df=pd.DataFrame([dict(domain='ff',released_role=role,label=1,method_type='FR',source_ids='[]') for role in ['train','test']])
    q=source_role_filter(df,{})
    assert list(q.experiment_role)==['train','ff_test']

def test_cli_exposes_dry_run_without_freeze_or_cuda(tmp_path,monkeypatch,capsys):
    from deepfake_detector.cli import main
    config={'backbones':{k:{'learning_rate':lr} for k,lr in [('clip_vit_b16',1e-5),('dinov2_vit_b14',5e-5),('efficientnet_b0',1e-4)]},
      'seeds':[7,42,123,2026,3407],'epochs':15,'effective_batch':128,'physical_gpus':[4,5,6,7],
      'checkpoint_selection':'final_epoch_only','artifact_root':str(tmp_path/'artifacts')}
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps(config))
    monkeypatch.setattr('sys.argv',['run.py','dry-run','--config',str(cfg)])
    main(tmp_path)
    assert len(json.loads(capsys.readouterr().out)['jobs'])==15

def test_numeric_policy_is_applied_not_just_recorded():
    from deepfake_detector.qualification import apply_numeric_policy
    import torch,os
    apply_numeric_policy()
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.benchmark is False
    assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'

def test_report_keeps_exactly_five_seeds_per_backbone_and_pairs(tmp_path):
    from deepfake_detector.reporting import summarize_runs
    records=[]
    for backbone in ['a','b','c']:
      for seed in [7,42,123,2026,3407]:
        records.append({'backbone':backbone,'seed':seed,'domain':'ff','unit':'video','flow':'RL','method_macro_AUC':.7+seed/1e6})
    table=summarize_runs(pd.DataFrame(records),[7,42,123,2026,3407])
    assert list(table.n)==[5,5,5] and set(table.backbone)=={'a','b','c'}
    with pytest.raises(ValueError,match='seed'):summarize_runs(pd.DataFrame(records[:-1]),[7,42,123,2026,3407])

def test_annotation_join_suffixes_require_proof_before_normalizing():
    from deepfake_detector.freezing import normalize_audit_manifest
    df=pd.DataFrame({'label_x':[0,1],'label_y':[0,1],'image_path_x':['a','b'],'image_path_y':['a','b']})
    q=normalize_audit_manifest(df)
    assert list(q.label)==[0,1] and list(q.image_path)==['a','b']
    df.loc[1,'image_path_y']='WRONG'
    with pytest.raises(ValueError,match='conflict'):normalize_audit_manifest(df)

def test_formal_queue_uses_four_cards_after_separate_verified_pilot(tmp_path):
    import subprocess,sys
    from deepfake_detector.queueing import schedule_jobs
    jobs=[{'run_id':str(i)} for i in range(4)]
    launched=set()
    def launch(job,device):
        launched.add(job['run_id'])
        return subprocess.Popen([sys.executable,'-c',"import time;time.sleep(.15)"])
    result=schedule_jobs(jobs,[4,5,6,7],launch,lambda job:job['run_id'] in launched,tmp_path/'queue.json',{'same':True},poll_seconds=.01,pilot_verified=True)
    assert result['peak_workers']==4
    assert [e['state'] for e in result['events'][:4]]==['RUNNING']*4

def test_unresolved_restore_intent_is_not_overwritten(tmp_path,monkeypatch):
    from deepfake_detector import orchestration as o
    old={'owner_pid':99999999,'state':'STOPPED','original':{'id':'original','state':{'Running':True}}}
    path=tmp_path/'service_state.json';path.write_text(json.dumps(old));before=path.read_bytes()
    monkeypatch.setattr(o,'require_gpu_gates',lambda *a:None)
    monkeypatch.setattr(o,'container_state',lambda:{'id':'original','state':{'Running':False}})
    monkeypatch.setattr(o,'gpu_inventory',lambda:({i:{'uuid':str(i),'memory_MiB':0,'utilization':0} for i in [4,5,6,7]},[]))
    with pytest.raises(ValueError,match='restore intent'):o.pause_llm(tmp_path,[4,5,6,7])
    assert path.read_bytes()==before

def test_sigterm_joins_owned_child_before_lifecycle_restoration(tmp_path):
    import subprocess,sys,time,os,signal
    from deepfake_detector.orchestration import owned_process,termination_guard
    childpid=tmp_path/'childpid';restored=tmp_path/'restored'
    script="""from pathlib import Path
import contextlib,sys
from deepfake_detector.orchestration import owned_process,termination_guard
@contextlib.contextmanager
def lifecycle():
 try:yield
 finally:Path(sys.argv[2]).write_text('restored')
with termination_guard(),lifecycle():
 owned_process([sys.executable,'-c',"import os,time;from pathlib import Path;Path("+repr(sys.argv[1])+").write_text(str(os.getpid()));time.sleep(30)"])
"""
    env=os.environ.copy()
    parent=subprocess.Popen([sys.executable,'-c',script,str(childpid),str(restored)],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        deadline=time.time()+5
        while not childpid.exists() and time.time()<deadline:time.sleep(.01)
        assert childpid.exists()
        os.kill(parent.pid,signal.SIGTERM);parent.wait(timeout=5)
        assert restored.read_text()=='restored'
        assert not Path('/proc/'+childpid.read_text()).exists()
    finally:
        if parent.poll() is None:parent.terminate();parent.wait(timeout=5)

def test_efs_content_aliases_connect_all_storage_groups_before_selection():
    from deepfake_detector.freezing import select_with_groups
    rows=[]
    for bucket,rgb in [('A','shared'),('B','shared'),('A','uniqueA'),('B','uniqueB')]:
        rows.append(dict(domain='ff',method_id='SG2',method_type='EFS',evaluation_unit='image',video_id='',storage_group_id=bucket,source_ids='[]',rgb_sha256=rgb,image_path=bucket+'/'+rgb,experiment_role='train',label=1))
    selected=select_with_groups(pd.DataFrame(rows),{})
    assert len(selected)==3 and selected.group_id.nunique()==1

def test_consolidated_report_keeps_controls_heads_pairs_and_bootstrap_scope():
    from deepfake_detector.reporting import consolidate_results
    result={'B':{'scale_controls':[{'scale':1,'aggregation':'L','method_macro_AUC':.7}],'length_controls':[{'condition':'u','aggregation':'L','method_macro_AUC':.6}],
              'pair_transitions':[{'domain':'cdf','before':'RL','after':'RP','ledger_delta':.01,'table':[[1,0,0],[0,1,0],[0,0,1]]}]},
            'matched_heads':{'raw':{'converged':False,'iterations':25,'ff':{'L':{'method_macro_AUC':.7}}}},
            'D':{'test_resampling':{'cdf':{'conditioning':'fixed model','iterations':2000,'metrics':{'RL':{'method_macro_AUC':{'replicates':2000,'invalid':4,'valid':1996,'conditional_95_interval':[.6,.8]}}}}},
                 'calibration_resampling':{'available':True,'video':{'iterations':1000,'conditioning':'fixed targets','summaries':{'RL':{'threshold':{'invalid':0,'median':1.}}}}}}}
    tables=consolidate_results([{'backbone':'clip','seed':7,'result':result}])
    assert set(tables)>={'scale_controls','length_controls','matched_heads','pair_transitions','test_bootstrap','calibration_bootstrap'}
    assert not tables['matched_heads'].iloc[0].converged
    assert 'invalid' in tables['test_bootstrap'].columns

def test_preflight_collects_only_this_project_suite(tmp_path,monkeypatch):
    import subprocess
    from deepfake_detector import orchestration as o
    (tmp_path/'src').mkdir();(tmp_path/'configs').mkdir();(tmp_path/'reports').mkdir()
    (tmp_path/'configs/v1.1.json').write_text('{}')
    data=tmp_path/'artifacts/data';data.mkdir(parents=True);(data/'freeze_complete.json').write_text('{}')
    monkeypatch.setattr(o,'verify_selected_source_stats',lambda *a:{'selected_paths_stat_checked':1})
    def fake(args,**kwargs):
        assert 'tests' in args, 'must not collect artifact/reference upstream generation scripts'
        return subprocess.CompletedProcess(args,0,'1 passed','')
    monkeypatch.setattr(o.subprocess,'run',fake)
    o.preflight({'artifact_root':str(tmp_path/'artifacts')},tmp_path)

def test_dino_cpu_position_interpolation_preserves_value_and_original_grid_gradient():
    import torch
    from timm.layers import resample_abs_pos_embed
    from deepfake_detector.models import resample_dino_positions
    x=torch.arange(1370*8,dtype=torch.float32).reshape(1,1370,8)/1000
    x.requires_grad_()
    expected=resample_abs_pos_embed(x,[16,16],[37,37],num_prefix_tokens=1)
    actual=resample_dino_positions(x,[16,16],[37,37],1)
    assert torch.equal(actual,expected)
    actual.square().sum().backward()
    assert x.grad.shape==(1,1370,8) and torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad[:,1:])>1000
