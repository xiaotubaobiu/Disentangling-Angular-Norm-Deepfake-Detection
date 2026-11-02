import copy,hashlib,json
from pathlib import Path
import numpy as np,pytest,torch
from reliability_v2 import source_best_files as f,trainer

def write(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return p

def hashfile(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

@pytest.fixture
def sample(tmp_path):
 root=tmp_path/'study';run='D_formal_clip_vit_b16_seed7';folder=root/'runs'/run;folder.mkdir(parents=True)
 task=dict(route='D',phase='formal',recipe_frozen=True,selection='frozen_last',backbone='clip_vit_b16',seed=7,epochs=2,precision='AMP',lr=1e-5,microbatch=32,train_manifest='data/D/train.parquet',dev_manifest='data/D/dev.parquet',qualification_gate='qualification/D_gate.json')
 for name in ['data/D/train.parquet','data/D/dev.parquet','configs/requirements.execution.lock.txt','initial.pt']:
  p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture_locked_source')
 gate=write(root/'qualification/D_gate.json',{'status':'PASS','training_code_sha256':trainer.code_hash()})
 config={'artifact_root':str(root),'formal_seeds':[7,42,123,2026,3407],'backbones':{'clip_vit_b16':{'weight_path':str(root/'initial.pt'),'weight_sha256':hashfile(root/'initial.pt')}},'runs':{run:task}}
 cfg=write(root/'configs/execution.json',config)
 identity=dict(config=hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest(),train_manifest=hashfile(root/'data/D/train.parquet'),dev_manifest=hashfile(root/'data/D/dev.parquet'),initialization=hashfile(root/'initial.pt'),environment=hashfile(root/'configs/requirements.execution.lock.txt'),code=trainer.code_hash(),qualification=hashfile(gate))
 h=[dict(epoch=i+1,attempts=10,success=9 if i==0 else 10,skip=1 if i==0 else 0,draw_count=1280,dev={'macro_auc':a}) for i,a in enumerate([.5,.8])]
 state={'encoder.x':torch.arange(8,dtype=torch.float32),'classifier.weight':torch.arange(1024,dtype=torch.float32).reshape(2,512)/1000,'classifier.bias':torch.tensor([.1,-.2],dtype=torch.float32)}
 torch.save(dict(model=state,identity=identity,epoch=2),folder/'source_best.pt');torch.save(dict(model=state,identity=identity,epoch=2),folder/'final.pt')
 write(folder/'identity.json',identity);write(folder/'resolved_config.json',task);write(folder/'history.json',h);write(folder/'complete.json',dict(status='TRAINING_COMPLETE',identity=identity,epochs=2,attempts=20,success=19,skip=1,selection='frozen_last',target_used=False,source_best_epoch=2,source_best_sha256=hashfile(folder/'source_best.pt'),final_sha256=hashfile(folder/'final.pt')))
 audit=write(root/'reports/accepted_D_audit.json',{'status':'PASS_SYNTHETIC_TERMINAL_PROOF'})
 core={name:{'sha256':hashfile(folder/name),'bytes':(folder/name).stat().st_size} for name in ['identity.json','resolved_config.json','history.json','complete.json','source_best.pt','final.pt']}
 proof=write(root/'reports/D_manifest.json',dict(status='PASS_ALL_FIFTEEN_D_FORMAL_TRAINING_TERMINALS_ONLY',all_D_formal_training_complete=True,no_target_performance=True,acceptance_audit=str(audit),acceptance_audit_sha256=hashfile(audit),completed_formal=[dict(run_id=run,epochs=2,identity=identity,source_best_epoch=2,source_DEV_tables_independently_accepted=2,final_equal_terminal_state_model=True,source_best_equal_saved_state_model=True,core_file_inventory=core)]))
 admission=write(root/'engineering/admission.json',dict(schema='M_SOURCE_BEST_FILE_ADMISSION_V1',artifact_root=str(root),target_inference_authorized=False,endpoint_code_sha256=hashfile(f.__file__),H_registry=None,records={run:dict(route='D',config_relative='configs/execution.json',config_sha256=hashfile(cfg),proof_relative=str(proof.relative_to(root)),proof_sha256=hashfile(proof))}))
 return dict(root=root,run=run,folder=folder,cfg=cfg,proof=proof,admission=admission,state=state)

def call(s,**kw):return f.validate_source_file(s['root'],s['admission'],hashfile(s['admission']),s['run'],**kw)

def rebind(s):
 d=json.loads(s['proof'].read_text());row=d['completed_formal'][0]
 for name,record in row['core_file_inventory'].items():record.update(sha256=hashfile(s['folder']/name),bytes=(s['folder']/name).stat().st_size)
 write(s['proof'],d);a=json.loads(s['admission'].read_text());a['records'][s['run']]['proof_sha256']=hashfile(s['proof']);write(s['admission'],a)

def test_real_torch_files_preserve_original_FP32_head_and_endpoint(sample):
 out=call(sample);w=sample['state']['classifier.weight'].numpy();b=sample['state']['classifier.bias'].numpy()
 assert out['source_best_bytes_tensors_head_verified'] is True and out['target_inference_authorized'] is False
 assert out['tensor_count']==3 and out['dimension']==512 and out['head_sha256']==hashlib.sha256(w.tobytes()+b.tobytes()).hexdigest()
 assert out['best_final_tensor_equality_checked'] is True and out['best_equals_final_tensors'] is True
 assert not torch.cuda.is_initialized()

@pytest.mark.parametrize('name',['source_best.pt','history.json','identity.json','resolved_config.json','complete.json'])
def test_actual_source_file_drift_rejected(sample,name):
 with (sample['folder']/name).open('ab') as stream:stream.write(b'drift')
 with pytest.raises(ValueError):call(sample)

@pytest.mark.parametrize('change',['checkpoint_epoch','checkpoint_identity','missing_head','head_dtype','head_shape','nan_head','nan_encoder','non_tensor','complex','sparse'])
def test_self_consistent_file_hashes_do_not_hide_bad_tensor_or_metadata(sample,change):
 p=sample['folder']/'source_best.pt';c=torch.load(p,map_location='cpu',weights_only=True)
 if change=='checkpoint_epoch':c['epoch']=1
 if change=='checkpoint_identity':c['identity']['code']='f'*64
 if change=='missing_head':del c['model']['classifier.bias']
 if change=='head_dtype':c['model']['classifier.weight']=c['model']['classifier.weight'].double()
 if change=='head_shape':c['model']['classifier.weight']=torch.zeros(3,512)
 if change=='nan_head':c['model']['classifier.weight'][0,0]=float('nan')
 if change=='nan_encoder':c['model']['encoder.x'][0]=float('inf')
 if change=='non_tensor':c['model']['unexpected']='not_tensor'
 if change=='complex':c['model']['encoder.x']=torch.ones(8,dtype=torch.complex64)
 if change=='sparse':c['model']['encoder.x']=torch.ones(8).to_sparse()
 torch.save(c,p);receipt=json.loads((sample['folder']/'complete.json').read_text());receipt['source_best_sha256']=hashfile(p);write(sample['folder']/'complete.json',receipt);rebind(sample)
 with pytest.raises(ValueError):call(sample)

@pytest.mark.parametrize('change',['foreign_config','unaccepted_proof','bad_proof_relation','proof_sha','endpoint_code'])
def test_foreign_config_proof_or_loader_version_rejected(sample,change):
 a=json.loads(sample['admission'].read_text());record=a['records'][sample['run']]
 if change=='foreign_config':record['config_relative']='configs/foreign.json'
 if change=='proof_sha':record['proof_sha256']='0'*64
 if change=='endpoint_code':a['endpoint_code_sha256']='0'*64
 if change in ['unaccepted_proof','bad_proof_relation']:
  d=json.loads(sample['proof'].read_text())
  if change=='unaccepted_proof':d['status']='PENDING'
  else:d['completed_formal'][0]['source_best_equal_saved_state_model']=False
  write(sample['proof'],d);record['proof_sha256']=hashfile(sample['proof'])
 write(sample['admission'],a)
 with pytest.raises(ValueError):call(sample)

@pytest.mark.parametrize('limit',[True,0,-1,1024])
def test_checkpoint_byte_resource_limit_rejected(sample,limit):
 with pytest.raises(ValueError):call(sample,max_checkpoint_bytes=limit)

def test_symlink_to_even_identical_checkpoint_rejected(sample):
 p=sample['folder']/'source_best.pt';p.rename(sample['folder']/'real.pt');p.symlink_to(sample['folder']/'real.pt')
 with pytest.raises(ValueError):call(sample)

def test_separate_source_best_is_never_relabeled_as_final(sample):
 out=call(sample);assert out['checkpoint_path'].endswith('/source_best.pt') and out['endpoint_id'].startswith('M_source_best_')
 assert out['actual_cache_context_qualified'] is False and out['reuse_capacity_credit_bytes']==0

@pytest.fixture
def Hsample(sample,monkeypatch):
 root=sample['root'];run='H_formal_clip_vit_b16_B_seed7';folder=root/'runs'/run;folder.mkdir();task=dict(route='H',history='B',phase='formal',recipe_frozen=True,selection='frozen_last',backbone='clip_vit_b16',seed=7,epochs=30,precision='AMP',lr=1e-5,microbatch=32,draw_count=1280,train_manifest='data/H/B/train.parquet',dev_manifest='data/H/B/dev.parquet',qualification_gate='qualification/H_gate.json',H_data_freeze_sha256='d'*64,h_code_bundle_sha256='e'*64,source_recipe_sha256='f'*64)
 for rel in ['data/H/B/train.parquet','data/H/B/dev.parquet']:
  q=root/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b'H_fixture_locked_source')
 gate=write(root/'qualification/H_gate.json',{'status':'PASS'});cfg0=json.loads(sample['cfg'].read_text());cfg0['schema']='H_EXECUTION_V2';cfg0['runs']={run:task};cfg=write(root/'configs/H_execution.json',cfg0)
 identity=dict(config=hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest(),train_manifest=hashfile(root/task['train_manifest']),dev_manifest=hashfile(root/task['dev_manifest']),initialization=hashfile(root/'initial.pt'),environment=hashfile(root/'configs/requirements.execution.lock.txt'),code=trainer.code_hash(),qualification=hashfile(gate))
 h=[dict(epoch=i,attempts=10,success=10,skip=0,draw_count=1280,dev={'macro_auc':.9 if i==2 else .6}) for i in range(1,31)]
 for name in f.worker_h._completion_artifacts(30):(folder/name).write_bytes(b'{}')
 torch.save(dict(model=sample['state'],identity=identity,epoch=2),folder/'source_best.pt');torch.save(dict(model=sample['state'],identity=identity,epoch=30),folder/'final.pt');write(folder/'identity.json',identity);write(folder/'resolved_config.json',task);write(folder/'history.json',h);cp=write(folder/'complete.json',dict(status='TRAINING_COMPLETE',identity=identity,epochs=30,attempts=300,success=300,skip=0,selection='frozen_last',target_used=False,source_best_epoch=2,source_best_sha256=hashfile(folder/'source_best.pt'),final_sha256=hashfile(folder/'final.pt')))
 seal=write(folder/'H_completion_seal.json',dict(status='SEALED',run_id=run,artifact_root=str(root),config_path=str(cfg),target_used=False,H_data_freeze_sha256=task['H_data_freeze_sha256'],h_code_bundle_sha256=task['h_code_bundle_sha256'],artifact_sha256={name:hashfile(folder/name) for name in f.worker_h._completion_artifacts(30)}))
 row=dict(run_id=run,terminal_seal_sha256=hashfile(seal),terminal_complete_sha256=hashfile(cp),identity_sha256=hashfile(folder/'identity.json'),source_best_matches_best_state=True,final_matches_complete_state=True,source_DEV_tables_independently_recomputed=30,source_only=True,target_used=False,source_best_epoch=2,source_recipe_sha256=task['source_recipe_sha256'])
 proof=write(root/'reports/H_pair_audit.json',dict(status='PASS_SELECTED_H_FORMAL_AMP_TERMINAL_FULL_CHECKPOINTS_SOURCE_DEV_AND_PAIRED_STREAMS',source_only=True,CUDA_initialized=False,inputs={'H_execution_sha256':hashfile(cfg)},selected_run_ids=[run],records=[row]))
 bones=('clip_vit_b16','dinov2_vit_b14','efficientnet_b0');seeds=(7,42,123,2026,3407);ids=lambda b:[f'H_formal_{b}_{h}_seed{s}' for s in seeds for h in ['B','AB']]
 registry=write(root/'H_run_registry.json',dict(schema='H_RUN_REGISTRY_V2',status='H_FORMAL_COMPLETE_TARGET_CACHE_PENDING',formal_model_count=30,formal_run_ids=[x for b in bones for x in ids(b)],target_export_allowed=True,effective_runs={b:dict(precision='AMP',run_ids=ids(b)) for b in bones}))
 admission=write(root/'engineering/H_admission.json',dict(schema='M_SOURCE_BEST_FILE_ADMISSION_V1',artifact_root=str(root),target_inference_authorized=False,endpoint_code_sha256=hashfile(f.__file__),H_registry={'relative':'H_run_registry.json','sha256':hashfile(registry)},records={run:dict(route='H',config_relative='configs/H_execution.json',config_sha256=hashfile(cfg),proof_relative=str(proof.relative_to(root)),proof_sha256=hashfile(proof))}))
 original_context=f._source_context
 monkeypatch.setattr(f,'_source_context',lambda path,r,route:(cfg0,task,identity) if route=='H' else original_context(path,r,route))
 monkeypatch.setattr(f,'_source_bridge_evidence',lambda path,r,route:{})
 return dict(root=root,run=run,folder=folder,cfg=cfg,proof=proof,admission=admission,seal=seal,task=task,registry=registry)

def test_H_real_files_use_current_seal_and_effective_registry(Hsample):
 out=call(Hsample);assert out['source_best_bytes_tensors_head_verified'] is True and out['effective_precision']=='AMP' and out['best']['attempts']==20 and out['best_final_tensor_equality_checked'] is False
 assert out['roles']==['B/SOURCE_CAL','B/SOURCE_TEST','B/TARGET_TEST'] and out['target_inference_authorized'] is False

@pytest.mark.parametrize('field',['H_data_freeze_sha256','h_code_bundle_sha256'])
def test_H_seal_header_must_match_current_task_even_with_updated_proof(Hsample,field):
 s=Hsample;seal=json.loads(s['seal'].read_text());seal[field]='a'*64;write(s['seal'],seal);d=json.loads(s['proof'].read_text());d['records'][0]['terminal_seal_sha256']=hashfile(s['seal']);write(s['proof'],d);a=json.loads(s['admission'].read_text());a['records'][s['run']]['proof_sha256']=hashfile(s['proof']);write(s['admission'],a)
 with pytest.raises(ValueError):call(s)

def test_H_registry_file_drift_rejected(Hsample):
 with Hsample['registry'].open('a') as out:out.write(' ')
 with pytest.raises(ValueError):call(Hsample)

def test_H_seal_or_selected_bytes_cannot_be_dropped(Hsample):
 s=Hsample;seal=json.loads(s['seal'].read_text());seal['artifact_sha256'].pop('epoch_30_updates.jsonl');write(s['seal'],seal);d=json.loads(s['proof'].read_text());d['records'][0]['terminal_seal_sha256']=hashfile(s['seal']);write(s['proof'],d);a=json.loads(s['admission'].read_text());a['records'][s['run']]['proof_sha256']=hashfile(s['proof']);write(s['admission'],a)
 with pytest.raises(ValueError):call(s)

def test_accepted_initialization_symlink_keeps_original_byte_identity(sample):
 root=sample['root'];p=root/'initial.pt';p.rename(root/'canonical_initial.pt');p.symlink_to(root/'canonical_initial.pt')
 out=call(sample);assert out['training_identity']['initialization']==hashfile(root/'canonical_initial.pt') and out['target_inference_authorized'] is False

@pytest.mark.parametrize('relative',['configs/H_frozen_recipes.json','data/H/freeze.json','qualification/H_gate_dependency.json'])
def test_H_source_validator_prerequisite_change_during_load_is_rejected(Hsample,monkeypatch,relative):
 s=Hsample;prerequisite=write(s['root']/relative,{'immutable':'v1'});expected=hashfile(prerequisite);original_context=f._source_context;original_load=f._load
 def guarded(*a):
  if hashfile(prerequisite)!=expected:raise ValueError('source validator prerequisite drift')
  return original_context(*a)
 def change(path):
  value=original_load(path)
  if Path(path).name=='source_best.pt':write(prerequisite,{'immutable':'changed'})
  return value
 monkeypatch.setattr(f,'_source_context',guarded);monkeypatch.setattr(f,'_load',change)
 with pytest.raises(ValueError):call(s)

def test_best_final_tensor_equivalence_requires_same_encoder_dtype(sample):
 p=sample['folder']/'final.pt';c=torch.load(p,map_location='cpu',weights_only=True);c['model']['encoder.x']=c['model']['encoder.x'].double();torch.save(c,p);receipt=json.loads((sample['folder']/'complete.json').read_text());receipt['final_sha256']=hashfile(p);write(sample['folder']/'complete.json',receipt);rebind(sample)
 with pytest.raises(ValueError):call(sample)
