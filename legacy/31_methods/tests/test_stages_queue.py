from pathlib import Path
import json,sys,subprocess
import pytest
from deepfake_detector import stages as s,queueing as q


def test_stage_identity_rejects_changed_input_and_corrupt_output(tmp_path):
    assert hasattr(s,'stage_identity'), 'hash-bound stages missing'
    project=tmp_path/'project';(project/'src').mkdir(parents=True)
    (project/'src/a.py').write_text('stable')
    source=tmp_path/'source';source.write_text('input')
    out=tmp_path/'stage';out.mkdir();(out/'scores.csv').write_text('scores')
    identity=s.stage_identity('analysis',{'seed':7},project,{'source':source})
    s.finish_stage(out,identity,['scores.csv'])
    assert s.completed_stage(out,identity)
    source.write_text('changed')
    changed=s.stage_identity('analysis',{'seed':7},project,{'source':source})
    with pytest.raises(ValueError,match='identity'):s.completed_stage(out,changed)
    (out/'scores.csv').write_text('corrupt')
    with pytest.raises(ValueError,match='hash'):s.completed_stage(out,identity)


def test_dry_plan_has_all15_final_epoch_runs_without_cuda_or_freeze(tmp_path,monkeypatch):
    assert hasattr(q,'dry_plan'), 'dry run plan missing'
    from deepfake_detector import orchestration as o
    monkeypatch.setattr(o,'gpu_inventory',lambda:pytest.fail('dry run must never inspect or initialize GPUs'))
    config={'backbones':{k:{'learning_rate':lr} for k,lr in [('clip_vit_b16',1e-5),('dinov2_vit_b14',5e-5),('efficientnet_b0',1e-4)]},
      'seeds':[7,42,123,2026,3407],'epochs':15,'effective_batch':128,'physical_gpus':[4,5,6,7],
      'checkpoint_selection':'final_epoch_only','artifact_root':str(tmp_path/'artifacts'),'data_policy':'AWAITING_SCIENTIFIC_RULING'}
    plan=q.dry_plan(config,tmp_path)
    assert len(plan['jobs'])==15 and len({j['run_id'] for j in plan['jobs']})==15
    assert plan['pilot_run_id']=='clip_vit_b16_seed7'
    assert plan['maximum_concurrent_workers']==4 and plan['GPU_initialized']==False
    assert plan['blocking_gates']==['scientific_role_freeze','CPU_preflight','GPU_microbatch_probe','end_to_end_pilot']


def test_real_cpu_process_queue_pilot_first_max4_and_resume(tmp_path):
    assert hasattr(q,'schedule_jobs'), 'bounded queue scheduler missing'
    jobs=[{'run_id':str(i)} for i in range(7)]
    def complete(job):return (tmp_path/(job['run_id']+'.done')).exists()
    def launch(job,device):
        script="from pathlib import Path;import time;time.sleep(.04);Path("+repr(str(tmp_path/(job['run_id']+'.done')))+").write_text('complete')"
        return subprocess.Popen([sys.executable,'-c',script],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    journal=tmp_path/'queue.json'
    result=q.schedule_jobs(jobs,[4,5,6,7],launch,complete,journal,{'protocol':'same'},poll_seconds=.01)
    assert result['status']=='COMPLETE' and result['peak_workers']<=4
    events=result['events'];pilot_done=next(i for i,e in enumerate(events) if e['run_id']=='0' and e['state']=='COMPLETE')
    assert all(e['run_id']=='0' for e in events[:pilot_done+1])
    reused=q.schedule_jobs(jobs,[4,5,6,7],lambda *args:pytest.fail('completed CPU jobs must not relaunch'),complete,journal,{'protocol':'same'},poll_seconds=.01)
    assert all(v['state']=='REUSED_COMPLETE' for v in reused['runs'].values())
    with pytest.raises(ValueError,match='identity'):q.schedule_jobs(jobs,[4,5,6,7],launch,complete,journal,{'protocol':'changed'})


def test_worker_exit_zero_without_verified_outputs_is_failure(tmp_path):
    assert hasattr(q,'schedule_jobs'), 'bounded queue scheduler missing'
    def launch(job,device):return subprocess.Popen([sys.executable,'-c','pass'])
    with pytest.raises(RuntimeError,match='verified completion'):
        q.schedule_jobs([{'run_id':'pilot'}],[4],launch,lambda job:False,tmp_path/'queue.json',{'protocol':'same'},poll_seconds=.01)
    record=json.loads((tmp_path/'queue.json').read_text())
    assert record['status']=='FAILED' and record['runs']['pilot']['state']=='FAILED'
