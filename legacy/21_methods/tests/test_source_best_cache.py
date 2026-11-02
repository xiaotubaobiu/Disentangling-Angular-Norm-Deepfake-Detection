import copy,hashlib,json,os
from pathlib import Path
import numpy as np,pandas as pd,pytest,torch
from PIL import Image
from reliability_v2 import trainer,cache_d,cache_h,source_best_files as sf
from reliability_v2 import source_best_cache as m

class Encoder(torch.nn.Module):
 def __init__(self):super().__init__();self.gain=torch.nn.Parameter(torch.tensor(1.))
 def forward(self,x):
  assert not self.training and x.dtype==torch.float32 and not torch.is_autocast_enabled(x.device.type)
  return (x.mean((1,2,3))*self.gain)[:,None].expand(-1,512).contiguous()
class Model(torch.nn.Module):
 def __init__(self):super().__init__();self.encoder=Encoder();self.classifier=torch.nn.Linear(512,2)
def save(p,v):trainer.save_json(p,v)
def sha(p):return trainer.digest(p)
def rgb(p):
 with Image.open(p) as im:x=np.asarray(im,dtype=np.uint8)
 return hashlib.sha256(str(x.shape).encode()+b'\0'+x.tobytes()).hexdigest()

@pytest.fixture(params=['D','H'])
def project(tmp_path,monkeypatch,request):
 route=request.param;root=tmp_path/'artifacts';(root/'configs').mkdir(parents=True);run=f'{route}_formal_clip_vit_b16_'+('B_' if route=='H' else '')+'seed7';folder=root/'runs'/run;folder.mkdir(parents=True)
 model=Model()
 with torch.no_grad():model.encoder.gain.fill_(3.);model.classifier.weight[0].fill_(1.);model.classifier.weight[1].fill_(2.);model.classifier.bias.copy_(torch.tensor([.25,-.25]))
 training={'source':'synthetic-original-identity'};torch.save(dict(model=model.state_dict(),identity=training,epoch=1),folder/'source_best.pt')
 last=Model();torch.save(dict(model=last.state_dict(),identity=training,epoch=2),folder/'final.pt')
 roles=m.ROLES[route];tables={};frames={}
 for role in roles:
  rows=[];methods=['real'] if 'cal' in role.lower() else ['real','DeepFakeDetection'] if route=='H' and 'TARGET' in role else ['real','Deepfakes','Face2Face','FaceSwap','NeuralTextures'] if route=='H' else ['real','m']
  for method in methods:
   for slot in range(8):
    image=root/'images'/role/method/f'{slot}.png';image.parent.mkdir(parents=True,exist_ok=True);Image.new('RGB',(7,9),(20+slot*20,30+slot,40)).save(image);uid=role+'/'+method;mother=uid+'/'+str(slot)
    rows.append(dict(image_path=str(image),label=int(method!='real'),method_id=method,video_uid=uid,group_id=uid,domain='target' if 'target' in role.lower() else 'source',role=role.split('/')[-1],sample_id=mother,mother_id=mother,mother_rgb_sha256=hashlib.sha256(mother.encode()).hexdigest(),mother_frame_index=slot,mother_timestamp_seconds=float(slot),processing_version='B',content_sha256=sha(image),rgb_sha256=rgb(image),bytes=image.stat().st_size,mtime_ns=image.stat().st_mtime_ns))
  df=pd.DataFrame(rows);path=root/f'data/{route}/{role}.parquet';path.parent.mkdir(parents=True,exist_ok=True);df.to_parquet(path,index=False);frames[role]=df;tables[role]=dict(path=str(path),sha256=sha(path),rows=len(df),videos=df.video_uid.nunique())
 freeze=dict(status='FROZEN',protocol='synthetic-original',target_scores_used=False,manifest_sha256={r+'.parquet':x['sha256'] for r,x in tables.items()},tables=tables)
 freeze_path=root/f'data/{route}/freeze.json';save(freeze_path,freeze);gate=root/'qualification/gate.json';save(gate,dict(status='PASS',D_manifest_freeze_sha256=sha(freeze_path)))
 init=root/'init.bin';init.write_bytes(b'no-network-fixture');task=dict(route=route,backbone='clip_vit_b16',seed=7,history='B' if route=='H' else None,epochs=2,phase='formal',recipe_frozen=True,microbatch=2,qualification_gate=str(gate.relative_to(root)),H_data_freeze_sha256=sha(freeze_path));config=dict(artifact_root=str(root),backbones={'clip_vit_b16':{'weight_path':str(init)}},cache_batch_size=8,cache_workers=0,cache_chunk_rows=4096,runs={run:task},allowed_gpus=[0,1,2,3],allowed_gpu_uuids=['u0','u1','u2','u3'])
 cfg=root/('configs/execution.json' if route=='D' else 'configs/H_execution.json');save(cfg,config)
 source=dict(schema='M_SOURCE_BEST_FILE_TENSOR_CONTEXT_V1',status='SOURCE_BEST_BYTES_TENSORS_HEAD_PASS_NO_CACHE_AUTHORIZATION',source_best_bytes_tensors_head_verified=True,parent_run_id=run,endpoint_id='M_source_best_'+run,route=route,backbone='clip_vit_b16',seed=7,history=task['history'],dimension=512,training_identity=training,checkpoint_path=str(folder/'source_best.pt'),checkpoint_sha256=sha(folder/'source_best.pt'),head_sha256=hashlib.sha256(model.classifier.weight.detach().numpy().tobytes()+model.classifier.bias.detach().numpy().tobytes()).hexdigest(),best={'epoch':1,'attempts':1,'successful_updates':1,'AMP_skips':0},roles=list(roles),input_sha256={str(folder/'source_best.pt'):sha(folder/'source_best.pt'),str(cfg):sha(cfg),str(gate):sha(gate)},actual_cache_context_qualified=False)
 parent=dict(training_identity=training,checkpoint_path=str(folder/'final.pt'),checkpoint_sha256=sha(folder/'final.pt'),config_path=str(cfg),config_sha256=sha(cfg));legacy=cache_d if route=='D' else cache_h
 monkeypatch.setattr(sf,'validate_source_file',lambda *a,**kw:copy.deepcopy(source));monkeypatch.setattr(m,'_primary',lambda *a:(legacy,copy.deepcopy(parent),copy.deepcopy(config),copy.deepcopy(task)));monkeypatch.setattr(m,'_build_model',lambda *a:Model())
 admission=root/'input.json';save(admission,dict(schema='M_SOURCE_BEST_CACHE_INPUT_ADMISSION_V1',artifact_root=str(root),cache_code_sha256=sha(m.__file__),source_file_admission={'path':str(root/'source.json'),'sha256':'0'*64},source_input_scope='SELECTED_M_D6_H3_FIVE_SEEDS',target_inference_authorized=False))
 context=m.prepare_context(root,admission,sha(admission),run)
 return dict(root=root,run=run,context=context,model=model,frames=frames,source=source,config=config,admission=admission,admission_sha=sha(admission),route=route,parent=parent)

def test_context_is_source_best_specific_and_not_execution_permission(project):
 e=project;c=e['context'];assert c.base['checkpoint_selection']=='source_best' and c.base['source_best_epoch']==1 and c.base['parent_run_id']==e['run'];assert c.base['target_inference_authorized'] is False
 assert tuple(c.roles)==m.ROLES[e['route']] and 'dev' not in c.roles

def test_storage_uses_original_raw_features_and_original_head(project):
 e=project;c=e['context'];role=c.roles[0];frame=e['frames'][role];h=np.arange(len(frame)*512,dtype=np.float32).reshape(len(frame),512)
 m.write_role(c,role,e['model'].classifier.weight.detach().numpy(),e['model'].classifier.bias.detach().numpy(),lambda start:[h[start:]])
 meta,df,x,w,b=m.load_role(c,role);assert meta['schema']==m.SCHEMA and np.array_equal(x,h) and np.array_equal(w,e['model'].classifier.weight.detach().numpy()) and np.array_equal(b,e['model'].classifier.bias.detach().numpy())
 assert len(df)==len(frame) and '/features/source_best_M/' in str(m.role_path(c,role))

def test_resume_does_not_recompute_completed_arrays(project):
 e=project;c=e['context'];role=c.roles[0];h=np.ones((len(e['frames'][role]),512),np.float32);w=e['model'].classifier.weight.detach().numpy();b=e['model'].classifier.bias.detach().numpy();m.write_role(c,role,w,b,lambda start:[h[start:]])
 def forbidden(start):raise AssertionError('completed cache recomputed')
 before=sha(m.role_path(c,role)/'complete.json');m.write_role(c,role,w,b,forbidden);assert sha(m.role_path(c,role)/'complete.json')==before

@pytest.mark.parametrize('bad',['dev','common/AB/TARGET_TEST','../target_test'])
def test_unselected_or_unsafe_roles_refuse(project,bad):
 with pytest.raises(ValueError):m.role_path(project['context'],bad)

def test_checkpoint_or_manifest_change_refuses(project):
 e=project;c=e['context'];Path(e['source']['checkpoint_path']).write_bytes(b'changed')
 with pytest.raises(ValueError):c.check()

def test_mutable_context_payload_refuses(project):
 c=project['context'];c.base['source_best_epoch']=2
 with pytest.raises(ValueError):c.check()

def test_production_permit_required_before_encoder_or_output(project,monkeypatch):
 e=project
 def forbidden(*a):raise AssertionError('model allocated without permit')
 monkeypatch.setattr(m,'_build_model',forbidden)
 with pytest.raises((ValueError,FileNotFoundError)):m.extract(e['root'],e['admission'],e['admission_sha'],e['run'],permit_path=e['root']/'absent.json',permit_sha256='0'*64,device='cpu')
 assert not (e['root']/'features').exists()

def test_legacy_module_globals_remain_unchanged(project):
 e=project;c=e['context'];legacy=cache_d if e['route']=='D' else cache_h;before={k:id(getattr(legacy,k)) for k in ['_cache_role','verify_cache','_read_manifest','SCHEMA','_validate_live']};role=c.roles[0];h=np.ones((len(e['frames'][role]),512),np.float32)
 m.write_role(c,role,e['model'].classifier.weight.detach().numpy(),e['model'].classifier.bias.detach().numpy(),lambda start:[h[start:]]);m.verify_role(c,role)
 assert before=={k:id(getattr(legacy,k)) for k in before}


def test_cpu_extraction_loads_selected_encoder_not_parent_final(project,monkeypatch):
 e=project
 # Only the execution-permit validator is stubbed for this synthetic unit test.
 # Source checkpoint, original reader/loader, FP32 encoder and all cache checks run.
 monkeypatch.setattr(m,'_permit',lambda *a:{'synthetic_test_only':True})
 result=m.extract(e['root'],e['admission'],e['admission_sha'],e['run'],permit_path='synthetic-only',permit_sha256='0'*64,device='cpu')
 assert set(result['roles'])==set(m.ROLES[e['route']])
 for role in m.ROLES[e['route']]:
  _,frame,h,w,b=m.load_role(e['context'],role)
  batch=trainer.loader(frame,e['config']['runs'][e['run']]['backbone'],8,workers=0)
  expected=[]
  with torch.inference_mode():
   for x,_ in batch:expected.append(e['model'].eval().encoder(x).numpy())
  assert np.array_equal(h,np.concatenate(expected))
  assert not torch.cuda.is_initialized()

def test_completed_resume_never_constructs_encoder(project,monkeypatch):
 e=project;monkeypatch.setattr(m,'_permit',lambda *a:{})
 m.extract(e['root'],e['admission'],e['admission_sha'],e['run'],permit_path='synthetic-only',permit_sha256='0'*64,device='cpu')
 def forbidden(*a):raise AssertionError('completed model cache recomputed')
 monkeypatch.setattr(m,'_build_model',forbidden)
 m.extract(e['root'],e['admission'],e['admission_sha'],e['run'],permit_path='synthetic-only',permit_sha256='0'*64,device='cpu')

def test_changed_manifest_or_freeze_refuses_before_encoding(project):
 e=project;c=e['context'];p=Path(c.tables[c.roles[0]]['path']);p.write_bytes(b'drift')
 with pytest.raises(ValueError):c.check()

def test_foreign_head_preserved_and_refused(project):
 e=project;c=e['context'];folder=m.role_path(c,c.roles[0]);folder.mkdir(parents=True);x=e['model'].classifier.weight.detach().numpy().copy();x[0,0]+=1
 np.save(folder/'head_weight.npy',x);before=sha(folder/'head_weight.npy')
 with pytest.raises(ValueError):m.write_role(c,c.roles[0],e['model'].classifier.weight.detach().numpy(),e['model'].classifier.bias.detach().numpy(),lambda start:[])
 assert sha(folder/'head_weight.npy')==before

def test_huge_or_wrong_array_header_refused_before_np_load(project,monkeypatch):
 e=project;c=e['context'];folder=m.role_path(c,c.roles[0]);folder.mkdir(parents=True)
 with (folder/'head_weight.npy').open('wb') as f:np.lib.format.write_array_header_1_0(f,dict(descr='<f4',fortran_order=False,shape=(2,10**12)))
 def forbidden(*a,**kw):raise AssertionError('unbounded array load')
 monkeypatch.setattr(m.np,'load',forbidden)
 with pytest.raises(ValueError):m._precheck(c,c.roles[0])

def test_typed_identity_bool_integer_collision_refused(project):
 e=project;c=e['context'];role=c.roles[0];h=np.ones((len(e['frames'][role]),512),np.float32)
 m.write_role(c,role,e['model'].classifier.weight.detach().numpy(),e['model'].classifier.bias.detach().numpy(),lambda start:[h[start:]])
 p=m.role_path(c,role)/'identity.json';x=json.loads(p.read_text());x['source_best_epoch']=True;save(p,x)
 with pytest.raises(ValueError):m.verify_role(c,role)

def test_gpu_bridge_refuses_without_allocating_cuda(project,monkeypatch):
 e=project;monkeypatch.setattr(m,'_permit',lambda *a:{})
 def forbidden(*a):raise AssertionError('unqualified GPU allocation')
 monkeypatch.setattr(m,'_build_model',forbidden)
 with pytest.raises(ValueError,match='phase-lease'):m.extract(e['root'],e['admission'],e['admission_sha'],e['run'],permit_path='synthetic-only',permit_sha256='0'*64,device='cuda')
 assert not torch.cuda.is_initialized() and not (e['root']/'features').exists()

def test_candidate_namespace_symlink_refuses(project,tmp_path):
 e=project;c=e['context'];p=e['root']/'features';p.symlink_to(tmp_path/'elsewhere')
 with pytest.raises(ValueError):m.role_path(c,c.roles[0])


def test_partial_chunk_resume_retains_committed_bytes(project):
 e=project;role=e['context'].roles[0];pieces=[]
 for n in range(8448//len(e['frames'][role])):
  frame=e['frames'][role].copy()
  for field in ('sample_id','mother_id','video_uid','group_id'):frame[field]=frame[field].astype(str)+'_repeat'+str(n)
  pieces.append(frame)
 frame=pd.concat(pieces,ignore_index=True);path=Path(e['context'].tables[role]['path']);frame.to_parquet(path,index=False)
 freeze_path=e['root']/f'data/{e["route"]}/freeze.json';freeze=json.loads(freeze_path.read_text());freeze['manifest_sha256'][role+'.parquet']=sha(path);freeze['tables'][role]=dict(path=str(path),sha256=sha(path),rows=len(frame),videos=int(frame.video_uid.nunique()));save(freeze_path,freeze)
 gate=e['root']/'qualification/gate.json';g=json.loads(gate.read_text());g['D_manifest_freeze_sha256']=sha(freeze_path);save(gate,g)
 e['config']['runs'][e['run']]['H_data_freeze_sha256']=sha(freeze_path);cfg=Path(e['parent']['config_path']);save(cfg,e['config']);e['parent']['config_sha256']=sha(cfg)
 e['source']['input_sha256'][str(cfg)]=sha(cfg);e['source']['input_sha256'][str(gate)]=sha(gate)
 c=m.prepare_context(e['root'],e['admission'],e['admission_sha'],e['run']);h=np.arange(len(frame)*512,dtype=np.float32).reshape(len(frame),512);w=e['model'].classifier.weight.detach().numpy();b=e['model'].classifier.bias.detach().numpy()
 def interrupted(start):
  assert start==0;yield h[:4096];raise RuntimeError('synthetic interruption')
 with pytest.raises(RuntimeError,match='synthetic interruption'):m.write_role(c,role,w,b,interrupted)
 folder=m.role_path(c,role);first=folder/'chunk_000000.npy';before=(sha(first),first.stat().st_mtime_ns);starts=[]
 def resume(start):starts.append(start);yield h[start:]
 m.write_role(c,role,w,b,resume);assert starts==[4096] and before==(sha(first),first.stat().st_mtime_ns)
 assert np.array_equal(m.load_role(c,role)[2],h)

def test_foreign_orphan_chunk_is_preserved(project):
 e=project;c=e['context'];role=c.roles[0];folder=m.role_path(c,role);folder.mkdir(parents=True);h=np.ones((len(e['frames'][role]),512),np.float32)
 np.save(folder/'chunk_000000.npy',h+1);before=sha(folder/'chunk_000000.npy')
 with pytest.raises(ValueError,match='bit identity'):m.write_role(c,role,e['model'].classifier.weight.detach().numpy(),e['model'].classifier.bias.detach().numpy(),lambda start:[h[start:]])
 assert sha(folder/'chunk_000000.npy')==before and not (folder/'complete.json').exists()

def test_role_duplicate_writer_lock_refuses(project):
 e=project;c=e['context'];role=c.roles[0];folder=m.role_path(c,role)
 with trainer.exclusive_file(folder/'write.lock'):
  with pytest.raises(BlockingIOError):m.write_role(c,role,e['model'].classifier.weight.detach().numpy(),e['model'].classifier.bias.detach().numpy(),lambda start:[])
 assert not (folder/'complete.json').exists()

def test_publish_preserves_signed_zero_bits(tmp_path):
 p=tmp_path/'a.npy';x=np.array([0.],np.float32);m._array(p,x);before=sha(p)
 with pytest.raises(ValueError,match='bit identity'):m._array(p,np.array([-0.],np.float32))
 assert sha(p)==before
