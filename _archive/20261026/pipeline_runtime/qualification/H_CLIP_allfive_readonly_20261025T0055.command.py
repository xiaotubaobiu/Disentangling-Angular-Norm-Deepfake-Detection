# Finite read-only original five-pair audit; no target qualification writer.
from pathlib import Path
import os,json,hashlib,datetime,time,sys
import torch
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
V=Path('/home/z/workspace/deepfake_detector_publish_20261023/studies/pipeline_reliability/v2')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert all(os.environ.get(n)=='2' for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
torch.set_num_threads(2);assert not torch.cuda.is_initialized()
config=R/'configs/H_execution.json';registry=R/'H_run_registry.json';queue=V/'reliability_v2/queue_h.py'
assert sha(config)=='ec344230ac000185312eb5cd069c5ef1e3917ddc82b65f466fd8180fccc7ee9c' and sha(registry)=='c8965b2f754fd12ba30bc0a1ee037cb42ba85439c861f27c4d3769a8c6ce1a6f' and sha(queue)=='a2792361443e64622391fb6ae766ee5e6f887789d662afdfb6d809f9a5ea8176'
report=R/'reports/H_CLIP_allfive_readonly_source_ledger_20261025T0055.json'
assert not report.exists() and not report.is_symlink()
accepted=[];bound={};accepted_reports=[]
for basename in ('H_CLIP_first4_terminal_acceptance_20261024T2245.json','H_CLIP_second4_terminal_acceptance_20261024T2350.json','H_CLIP_last_pair_terminal_acceptance_20261025T0055.json'):
 path=R/'reports'/basename;receipt=json.loads(path.read_text())
 assert receipt['status']=='PASS_SELECTED_H_FORMAL_AMP_TERMINAL_FULL_CHECKPOINTS_SOURCE_DEV_AND_PAIRED_STREAMS'
 assert receipt['inputs']['H_execution_sha256']=='ec344230ac000185312eb5cd069c5ef1e3917ddc82b65f466fd8180fccc7ee9c'
 assert receipt['inputs']['exporter_script_sha256']=='b76806a40ee85233123d17efceee8b10f6882600fa74507ff19d603d4685fa02'
 accepted.extend(receipt['selected_run_ids']);bound[str(path)]=sha(path);accepted_reports.append(str(path))
 for record in receipt['records']:
  folder=R/'runs'/record['run_id']
  assert sha(folder/'complete.json')==record['terminal_complete_sha256']
  assert sha(folder/'H_completion_seal.json')==record['terminal_seal_sha256']
  bound[str(folder/'complete.json')]=sha(folder/'complete.json')
  bound[str(folder/'H_completion_seal.json')]=sha(folder/'H_completion_seal.json')
expected=[f'H_formal_clip_vit_b16_{h}_seed{s}' for s in (7,42,123,2026,3407) for h in ('B','AB')]
assert len(accepted)==len(set(accepted))==10 and set(accepted)==set(expected)
print('All10 independent E30 terminal acceptances/seals bound; original fresh default verifier starts',flush=True)
sys.path.insert(0,str(V))
from reliability_v2 import queue_h
start=time.monotonic()
ledger=queue_h.audit_paired_ledgers(config,'clip_vit_b16',precision='AMP')
assert ledger['status'] in ('MATCHED_SUCCESSFUL_UPDATE_STREAMS','REQUIRES_UNIFORM_FP32_FALLBACK')
assert ledger['formal_seeds']==[7,42,123,2026,3407] and len(ledger['pairs'])==5
assert ledger['backbone']=='clip_vit_b16' and ledger['precision']=='AMP'
assert ledger['before_target_export'] and ledger['target_performance_used'] is False
assert sha(config)=='ec344230ac000185312eb5cd069c5ef1e3917ddc82b65f466fd8180fccc7ee9c' and sha(registry)=='c8965b2f754fd12ba30bc0a1ee037cb42ba85439c861f27c4d3769a8c6ce1a6f' and sha(queue)=='a2792361443e64622391fb6ae766ee5e6f887789d662afdfb6d809f9a5ea8176'
assert all(sha(p)==v for p,v in bound.items()) and not torch.cuda.is_initialized()
current=json.loads(registry.read_text());assert not current['effective_runs'] and current['target_export_allowed'] is False
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='SOURCE_ONLY_ALLFIVE_MATCHED_GLOBAL_TARGET_PHASE_PENDING' if ledger['status']=='MATCHED_SUCCESSFUL_UPDATE_STREAMS' else 'SOURCE_ONLY_ALLFIVE_REQUIRES_UNIFORM_FP32_PENDING_PHASE_AND_CAPACITY',original_frozen_ledger=ledger,CPU_audit_seconds=time.monotonic()-start,independent_terminal_models=10,input_sha256=bound,H_execution_sha256='ec344230ac000185312eb5cd069c5ef1e3917ddc82b65f466fd8180fccc7ee9c',H_run_registry_sha256='c8965b2f754fd12ba30bc0a1ee037cb42ba85439c861f27c4d3769a8c6ce1a6f',queue_source_sha256='a2792361443e64622391fb6ae766ee5e6f887789d662afdfb6d809f9a5ea8176',caller_sha256=sha(Path(__file__)),used_original_default_fresh_verifier=True,verify_override_supplied=False,target_performance_read=False,qualification_gate_written=False,registry_modified=False,FP32_reruns_started=False,GPU_export_started=False,CUDA_initialized=False,scope='Read-only allfive source numerical ledger after allten independent E30 terminal acceptances. This is not the canonical global post-AMP ledger/gate writer. Global effective registry and original four-GPU phase remain pending; no target export or competing dispatch.')
with report.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
print(json.dumps(dict(status=result['status'],ledger_status=ledger['status'],report=str(report),seconds=result['CPU_audit_seconds'])),flush=True)
