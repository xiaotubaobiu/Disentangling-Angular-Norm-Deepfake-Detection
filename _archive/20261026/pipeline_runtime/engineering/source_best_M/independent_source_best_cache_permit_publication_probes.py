"""Separate post-source-fix publication/capacity probes. Entirely synthetic/tmp."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,pytest
import independent_source_best_cache_permit_probes as base
m=base.m

def accepted_permits(p):
 paths=[]
 for path in p['root'].glob('qualification/M_cache_production/*/*/cache_permit.json'):
  value=json.loads(path.read_text())
  if value.get('status')=='PASS_PRODUCTION_CACHE_ADMISSION':
   ctx=SimpleNamespace(root=p['root'],admission_sha=p['input']['sha256'],base={'parent_run_id':m.analysis.expected_feature_runs()[0]})
   assert m.cache._permit(ctx,str(path),m.owner.digest(path))['target_inference_authorized'] is True;paths.append(str(path))
 return paths

@pytest.mark.parametrize('problem',['missing_D_record','wrong_source_SHA','reordered_queue'])
def test_source_and_strict_phase_fail_before_any_publication(tmp_path,monkeypatch,problem):
 p=base.corpus(tmp_path/'artifacts',monkeypatch,problem if problem!='reordered_queue' else None)
 if problem=='reordered_queue':
  q=json.loads(Path(p['queue']['path']).read_text());q['detail']['run_ids'].reverse();Path(p['queue']['path']).write_text(json.dumps(q));pattern='exact original GPU queue terminal/empty occupancy required'
 else:pattern='exact45 D15/H30 original-source admission required before publication' if problem=='missing_D_record' else 'input/code hash drift'
 with pytest.raises(ValueError,match=pattern):base.public(p)
 assert not list(p['root'].glob('qualification/M_cache_production/*/*/cache_permit.json'))
 print('PREPUBLICATION_FIXED',problem)


def test_late_capacity_failure_must_not_leave_accepted_success_permit(tmp_path,monkeypatch):
 p=base.corpus(tmp_path/'artifacts',monkeypatch);required=m.analysis.remaining_required(p['root'],m.owner.proof(p['root'],p['budget']));checks=[]
 def free(path):
  published=bool(list(p['root'].glob('qualification/M_cache_production/*/*/cache_permit.json')))
  value=required-1 if published else required;checks.append((published,value));return SimpleNamespace(free=value)
 monkeypatch.setattr(m.shutil,'disk_usage',free)
 with pytest.raises(ValueError,match='full initial or runtime resource budget failed'):base.public(p)
 stale=accepted_permits(p)
 print('LATE_CAPACITY',json.dumps({'required':required,'free_after_publish':required-1,'accepted_success_permits':stale,'space_checks':checks}))
 assert stale==[],'BLOCKER: issuer final full-budget rejection left a successful permit accepted at the lower cache runtime floor'


def test_owner_plan_collision_must_not_publish_new_accepted_permit(tmp_path,monkeypatch):
 p=base.corpus(tmp_path/'artifacts',monkeypatch);before=m.analysis.implementations()|{str(Path(m.__file__)):m.owner.digest(m.__file__),str(Path(m.owner.__file__)):m.owner.digest(m.owner.__file__)}
 key=hashlib.sha256(m.runtime.sink.typed(dict(input=p['input'],gates=p['gates'],bridge=p['bridge'],budget=p['budget'],source=before))).hexdigest();folder=p['root']/'qualification/M_cache_production'/p['input']['sha256']/key;m.runtime.sink._dirs(p['root'],folder);foreign=folder/'cache_owner_plan.json';foreign.write_text('{"foreign":true}')
 with pytest.raises(ValueError,match='append-only permit namespace collision'):base.public(p)
 stale=accepted_permits(p);assert foreign.read_text()=='{"foreign":true}'
 print('PLAN_COLLISION',json.dumps({'foreign_preserved':True,'new_accepted_success_permits':stale}))
 assert stale==[],'BLOCKER: known foreign owner-plan collision was discovered only after a new accepted permit was published'


def test_phase_failure_after_publish_must_not_leave_accepted_success_permit(tmp_path,monkeypatch):
 p=base.corpus(tmp_path/'artifacts',monkeypatch);publish=m.runtime.publish
 def change_phase(path,value):
  answer=publish(path,value)
  if Path(path).name=='cache_owner_plan.json':
   q=json.loads(Path(p['queue']['path']).read_text());q['detail']['run_ids'].reverse();Path(p['queue']['path']).write_text(json.dumps(q))
  return answer
 monkeypatch.setattr(m.runtime,'publish',change_phase)
 with pytest.raises(ValueError,match='exact original GPU queue terminal/empty occupancy required'):base.public(p)
 stale=accepted_permits(p)
 print('LATE_PHASE',json.dumps({'accepted_success_permits':stale}))
 assert stale==[],'BLOCKER: issuer raised strict original phase rejection but left a successful permit accepted by cache._permit'
