"""Final-commit issuer fix confirmation; only synthetic/tmp evidence.
The owner-plan write prepares, and successful cache_permit publication commits.
Postcommit budget changes are checked by the real owner startup contract.
"""
from pathlib import Path
from types import SimpleNamespace
import json,pytest
import independent_source_best_cache_permit_probes as base
import independent_source_best_cache_permit_publication_probes as pub
m=base.m

def no_permit(p):assert not list(p['root'].glob('qualification/M_cache_production/*/*/cache_permit.json'))
def plans(p):return list(p['root'].glob('qualification/M_cache_production/*/*/cache_owner_plan.json'))

def test_fixed_public_issue_resume_all45_real_preflights(tmp_path,monkeypatch):
 base.test_public_issue_resume_real_owner_bridge_and_all45_cache_preflights(tmp_path,monkeypatch)

@pytest.mark.parametrize('problem',['missing_D_record','wrong_source_SHA','reordered_queue'])
def test_fixed_source_and_strict_phase_before_any_publication(tmp_path,monkeypatch,problem):
 pub.test_source_and_strict_phase_fail_before_any_publication(tmp_path,monkeypatch,problem)

def test_fixed_known_ownerplan_collision_before_new_permit(tmp_path,monkeypatch):
 pub.test_owner_plan_collision_must_not_publish_new_accepted_permit(tmp_path,monkeypatch)

def test_fixed_capacity_fails_after_plan_before_permit(tmp_path,monkeypatch):
 p=base.corpus(tmp_path/'artifacts',monkeypatch);required=m.analysis.remaining_required(p['root'],m.owner.proof(p['root'],p['budget']));checks=[]
 def free(path):
  prepared=bool(plans(p));value=required-1 if prepared else 500*base.GiB;checks.append((prepared,value));return SimpleNamespace(free=value)
 monkeypatch.setattr(m.shutil,'disk_usage',free)
 with pytest.raises(ValueError,match='actual free capacity below complete joint remaining budget'):base.public(p)
 no_permit(p);assert len(plans(p))==1
 print('PRECOMMIT_BUDGET_REJECT',json.dumps({'required':required,'checks':checks,'uncommitted_plan_preserved':True}))

def test_fixed_phase_fails_after_plan_before_permit(tmp_path,monkeypatch):
 # The same old plan-publication callback now runs BEFORE permit commit.
 pub.test_phase_failure_after_publish_must_not_leave_accepted_success_permit(tmp_path,monkeypatch)

def test_fixed_source_after_plan_refuses_then_same_request_resumes(tmp_path,monkeypatch):
 p=base.corpus(tmp_path/'artifacts',monkeypatch);source=Path(p['source']['path']);before=source.read_bytes();publish=m.runtime.publish
 def drift(path,value):
  result=publish(path,value)
  if Path(path).name=='cache_owner_plan.json':source.write_text('{}')
  return result
 monkeypatch.setattr(m.runtime,'publish',drift)
 with pytest.raises(ValueError,match='input/code hash drift'):base.public(p)
 no_permit(p);assert len(plans(p))==1;plan=plans(p)[0];stamp=(m.owner.digest(plan),plan.stat().st_mtime_ns)
 source.write_bytes(before);monkeypatch.setattr(m.runtime,'publish',publish);result=base.public(p);base.scope_preflight(p,result)
 assert (m.owner.digest(plan),plan.stat().st_mtime_ns)==stamp
 print('PRECOMMIT_SOURCE_REJECT_AND_EXACT_RESUME')

@pytest.mark.parametrize('extra',[1024**2-1,1024**2])
def test_fixed_publication_reserve_boundary(tmp_path,monkeypatch,extra):
 p=base.corpus(tmp_path/'artifacts',monkeypatch);required=m.analysis.remaining_required(p['root'],m.owner.proof(p['root'],p['budget']))
 monkeypatch.setattr(m.shutil,'disk_usage',lambda path:SimpleNamespace(free=required+extra))
 if extra<1024**2:
  with pytest.raises(ValueError,match='complete joint budget plus1MiB publication allocation required'):base.public(p)
  no_permit(p);assert len(plans(p))==1
 else:
  result=base.public(p);base.scope_preflight(p,result)
 print('PUBLICATION_RESERVE_BOUNDARY',extra)

def test_fixed_committed_permit_low_future_budget_rejected_by_owner_startup(tmp_path,monkeypatch):
 p=base.corpus(tmp_path/'artifacts',monkeypatch);result=base.public(p);base.scope_preflight(p,result)
 bounds={Path(v['path']):(m.owner.digest(v['path']),Path(v['path']).stat().st_mtime_ns) for v in (result['cache_permit'],result['owner_plan'])}
 required=m.analysis.remaining_required(p['root'],m.owner.proof(p['root'],p['budget']));monkeypatch.setattr(m.shutil,'disk_usage',lambda path:SimpleNamespace(free=required-1))
 with pytest.raises(ValueError,match='full initial or runtime resource budget failed'):
  m.owner.validate_plan(p['root'],result['owner_plan']['path'],result['owner_plan']['sha256'])
 assert all((m.owner.digest(path),path.stat().st_mtime_ns)==prior for path,prior in bounds.items())
 print('POSTCOMMIT_OWNER_INITIAL_BUDGET_REFUSAL',required-1)

@pytest.mark.parametrize('when',['before_plan_write','after_plan_write'])
def test_fixed_plan_interruption_no_permit_then_exact_resume(tmp_path,monkeypatch,when):
 p=base.corpus(tmp_path/'artifacts',monkeypatch);publish=m.runtime.publish
 def interrupt(path,value):
  if Path(path).name=='cache_owner_plan.json':
   if when=='after_plan_write':publish(path,value)
   raise OSError('synthetic prepared-plan interruption')
  return publish(path,value)
 monkeypatch.setattr(m.runtime,'publish',interrupt)
 with pytest.raises(OSError,match='synthetic prepared-plan interruption'):base.public(p)
 no_permit(p);saved={path:(m.owner.digest(path),path.stat().st_mtime_ns) for path in plans(p)}
 if when=='after_plan_write':
  assert len(saved)==1;path=next(iter(saved))
  with pytest.raises(ValueError,match='regular input file required'):
   m.owner.validate_plan(p['root'],str(path),m.owner.digest(path))
 monkeypatch.setattr(m.runtime,'publish',publish);result=base.public(p);base.scope_preflight(p,result)
 assert all((m.owner.digest(path),path.stat().st_mtime_ns)==prior for path,prior in saved.items())
 print('PRECOMMIT_INTERRUPT_EXACT_RESUME',when)

def test_fixed_gate_drift_after_assemble_refuses(tmp_path,monkeypatch):
 base.test_public_gate_drift_after_first_assemble_before_locked_check(tmp_path,monkeypatch)

def test_fixed_foreign_permit_collision_preserved(tmp_path,monkeypatch):
 base.test_public_append_only_collision_preserves_foreign_proof(tmp_path,monkeypatch)
