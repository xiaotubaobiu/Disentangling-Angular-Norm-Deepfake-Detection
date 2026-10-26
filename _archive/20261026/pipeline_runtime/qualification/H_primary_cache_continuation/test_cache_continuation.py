from pathlib import Path
import importlib.util,json,pytest
p=Path(__file__).parent/'cache_continuation.py';s=importlib.util.spec_from_file_location('cache_cont',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_effective_registry_requires_all3_and_exact30():
 r=registry();assert len(m.effective_ids(r))==30
 r['target_export_allowed']=False
 with pytest.raises(ValueError):m.effective_ids(r)

@pytest.mark.parametrize('change',['missing','duplicates','precision','swapped'])
def test_effective_registry_drift_rejected(change):
 r=registry()
 if change=='missing':r['effective_runs'].pop('efficientnet_b0')
 if change=='duplicates':r['effective_runs']['clip_vit_b16']['run_ids'][0]=r['effective_runs']['clip_vit_b16']['run_ids'][1]
 if change=='precision':r['effective_runs']['clip_vit_b16']['precision']='INVALID'
 if change=='swapped':r['effective_runs']['clip_vit_b16']['run_ids'].reverse()
 with pytest.raises(ValueError):m.effective_ids(r)

def registry():
 f=[f'H_formal_{b}_{h}_seed{s}' for b in m.BACKBONES for s in m.SEEDS for h in m.HISTORIES]
 return dict(schema='H_RUN_REGISTRY_V2',status='H_FORMAL_COMPLETE_TARGET_CACHE_PENDING',formal_model_count=30,formal_run_ids=f,target_export_allowed=True,effective_runs={b:dict(precision='AMP',run_ids=[x for x in f if x.startswith('H_formal_'+b+'_')]) for b in m.BACKBONES})

def test_marker_alone_is_never_completion(tmp_path):
 assert m.verify_worker_receipt(tmp_path,'run','nonce',{},missing_ok=True) is False

@pytest.fixture
def receipt(tmp_path):
 r=tmp_path;run='H_formal_clip_vit_b16_B_seed7';folder=r/'features/H'/run;folder.mkdir(parents=True);roles={}
 for role in m.ROLES:
  p=folder/role/'complete.json';p.parent.mkdir(parents=True);p.write_text('{}');roles[role]=dict(path=str(p.parent),rows=1,complete_sha256=m.sha(p))
 cp=folder/'complete.json';cp.write_text(json.dumps(dict(status='CACHE_COMPLETE',run_id=run,roles=roles)))
 binding=dict(config_sha256='c',registry_sha256='r',entry_sha256='e')
 out=m.worker_receipt_path(r,'nonce',run);out.parent.mkdir(parents=True);d=dict(status='ORIGINAL_EXTRACT_RETURNED',nonce='nonce',run_id=run,bindings=binding,aggregate_marker_sha256=m.sha(cp),roles=roles);out.write_text(json.dumps(d))
 return r,run,binding,out

def test_current_receipt_and7markers_pass(receipt):
 r,run,binding,out=receipt;assert m.verify_worker_receipt(r,run,'nonce',binding)

@pytest.mark.parametrize('change',['nonce','binding','aggregate','role'])
def test_nonce_identity_or_marker_drift_rejected(receipt,change):
 r,run,binding,out=receipt;d=json.loads(out.read_text())
 if change=='nonce':d['nonce']='old';out.write_text(json.dumps(d))
 if change=='binding':binding['registry_sha256']='changed'
 if change=='aggregate':(r/'features/H'/run/'complete.json').write_text('tampered')
 if change=='role':(r/'features/H'/run/m.ROLES[0]/'complete.json').write_text('tampered')
 with pytest.raises(ValueError):m.verify_worker_receipt(r,run,'nonce',binding)

@pytest.mark.parametrize('free,required',[(100,101),(0,1)])
def test_live_capacity_failure_stops_launch(free,required):
 with pytest.raises(ValueError):m.capacity_floor(free,required)

def test_capacity_does_not_subtract_outputs_or_reserve():
 assert m.capacity_floor(100,66)==dict(free_bytes=100,required_bytes=66,margin_bytes=34)

def test_adapter_inherits_original_batch_and_calls_original_devices(tmp_path,monkeypatch):
 from types import SimpleNamespace
 class Base:
  def __init__(self):self.root=tmp_path;self.calls=0
  def batch(self,*args):pass
  def _devices(self):self.calls+=1;return [(0,'GPU0')],{'ready':True}
 budget=tmp_path/'budget.json';budget.write_text('{}');expected={'p':'h'}
 monkeypatch.setattr(m,'bindings',lambda root:expected);monkeypatch.setattr(m.shutil,'disk_usage',lambda root:SimpleNamespace(free=100))
 cls=m.queue_type(SimpleNamespace(HQueue=Base),'nonce',expected,66,budget,m.sha(budget));q=cls()
 assert cls.batch is Base.batch
 assert q._devices()==([(0,'GPU0')],{'ready':True}) and q.calls==1
 monkeypatch.setattr(m.shutil,'disk_usage',lambda root:SimpleNamespace(free=65))
 with pytest.raises(ValueError):q._devices()
 assert q.calls==1

def test_launch_uses_only_original_cache_worker_and_gpu_boundary(tmp_path,monkeypatch):
 from types import SimpleNamespace
 class Base:
  def __init__(self):
   self.root=tmp_path;self.path=tmp_path/'config.json';self.code=tmp_path;self.logs=tmp_path/'logs';self.logs.mkdir();self.config={'runs':{'run':{'backbone':'clip_vit_b16'}}};self.jobs={}
 budget=tmp_path/'budget.json';budget.write_text('{}');expected={'p':'h'};seen=[]
 monkeypatch.setattr(m,'bindings',lambda root:expected);monkeypatch.setattr(m,'effective_ids',lambda reg:['run']);monkeypatch.setattr(m,'read',lambda p:{});monkeypatch.setattr(m.shutil,'disk_usage',lambda root:SimpleNamespace(free=100))
 monkeypatch.setattr(m.subprocess,'Popen',lambda command,**kw:seen.append((command,kw)) or SimpleNamespace(pid=1))
 q=m.queue_type(SimpleNamespace(HQueue=Base),'nonce',expected,66,budget,m.sha(budget))()
 with pytest.raises(ValueError):q._launch('run',4,'PROTECTED4')
 assert not seen
 q._launch('run',0,'UUID0')
 command,kw=seen[0];assert '--worker' in command and 'reliability_v2.worker_h' not in command
 assert kw['env']['CUDA_VISIBLE_DEVICES']=='UUID0' and q.jobs['run']['gpu']==0

def test_original_extract_failure_never_writes_operational_receipt(tmp_path,monkeypatch):
 from types import SimpleNamespace
 import sys
 monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'bindings',lambda root:{'x':'h'});monkeypatch.setattr(m,'effective_ids',lambda reg:['run']);monkeypatch.setattr(m,'read',lambda p:{})
 def fail(*a):raise RuntimeError('original_extract_rejected')
 monkeypatch.setitem(sys.modules,'reliability_v2',SimpleNamespace(cache_h=SimpleNamespace(ROLES=m.ROLES,extract=fail)))
 with pytest.raises(RuntimeError,match='original_extract_rejected'):m.worker(str(tmp_path/'configs/H_execution.json'),'run','nonce',json.dumps({'x':'h'}))
 assert not m.worker_receipt_path(tmp_path,'nonce','run').exists()

@pytest.mark.parametrize('name',['cache_h.py','cache_d.py','scoring_d.py','trainer.py','queue_h.py','queue_d.py','worker_h.py','data_h.py'])
def test_qualified_extractor_and_helpers_reject_pre_admission_drift(tmp_path,name):
 import shutil
 source=tmp_path/'source';(source/'reliability_v2').mkdir(parents=True)
 manifest=json.loads((Path(m.__file__).parent/'qualified_source_manifest.json').read_text())
 for rel in manifest['source_sha256']:shutil.copyfile(m.SCIENCE/rel,source/rel)
 (source/'reliability_v2'/name).write_text('arbitrary_changed_source')
 with pytest.raises(ValueError):m.qualified_source_bindings(source)

@pytest.mark.parametrize('later',[False,True])
def test_budget_change_after_devices_before_any_launch_is_rejected(tmp_path,monkeypatch,later):
 from types import SimpleNamespace
 class Base:
  def __init__(self):
   self.root=tmp_path;self.path=tmp_path/'config.json';self.code=tmp_path;self.logs=tmp_path/'logs';self.logs.mkdir();self.config={'runs':{r:{'backbone':'clip_vit_b16'} for r in ['run0','run1']}};self.jobs={}
  def _devices(self):return [(0,'U0'),(1,'U1')],{'ready':True}
 budget=tmp_path/'budget.json';budget.write_text('{}');expected={'p':'h'};seen=[]
 monkeypatch.setattr(m,'bindings',lambda root:expected);monkeypatch.setattr(m,'effective_ids',lambda reg:['run0','run1']);monkeypatch.setattr(m,'read',lambda p:{});monkeypatch.setattr(m.shutil,'disk_usage',lambda root:SimpleNamespace(free=100));monkeypatch.setattr(m.subprocess,'Popen',lambda command,**kw:seen.append(command) or SimpleNamespace(pid=1))
 q=m.queue_type(SimpleNamespace(HQueue=Base),'nonce',expected,66,budget,m.sha(budget))();q._devices()
 if later:q._launch('run0',0,'U0')
 budget.write_text('drift')
 with pytest.raises(ValueError):q._launch('run1',1,'U1')
 assert len(seen)==int(later)

def test_exact_original_source_manifest_passes():
 pins=m.qualified_source_bindings();assert len(pins)==22
 assert pins[str(m.SCIENCE/'reliability_v2/cache_h.py')]=='9119200359f8d696f1366e6c0137bca533c63089c32e505971e69f84ebaf40fb'
