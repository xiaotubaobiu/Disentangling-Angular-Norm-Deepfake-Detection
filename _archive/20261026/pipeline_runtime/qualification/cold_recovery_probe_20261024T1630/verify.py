from pathlib import Path
import json,hashlib,os,torch,numpy as np,shutil,datetime,subprocess
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023');P=R/'reports';M=P/'D_closed_cold_copy_manifest_20261024T1510.json';m=json.loads(M.read_text());root=Path(__file__).parent
assert not torch.cuda.is_initialized();torch.set_num_threads(2)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
entries=m['entries'];records=[];mapping=[];deltas=[]
for e in entries:
 source=R/e['relative'];target=Path(e['staged_path']);a=source.stat();b=target.stat()
 assert source.is_file() and not source.is_symlink() and target.is_file() and not target.is_symlink() and a.st_size==b.st_size==e['bytes'] and a.st_mtime_ns==e['mtime_ns'] and a.st_ino==e['source_inode']
 deltas.append(b.st_mtime_ns-a.st_mtime_ns)
 mapping.append(dict(source=str(source),verified_copy=str(target),sha256=e['sha256'],bytes=e['bytes'],original_mtime_ns=e['mtime_ns'],original_inode=e['source_inode'],inverse_restore='Copy verified bytes to a private file on original filesystem, set original mtime_ns, SHA verify, then atomically restore canonical path. No current relocation performed.'))
for backbone in ('clip_vit_b16','dinov2_vit_b14','efficientnet_b0'):
 e=next(e for e in entries if e['relative']==f'runs/D_formal_{backbone}_seed7/state.pt');target=Path(e['staged_path']);link=root/(backbone+'_state_probe.pt');link.symlink_to(target)
 assert link.resolve()==target.resolve() and sha(link)==e['sha256']
 state=torch.load(link,map_location='cpu',weights_only=False);assert state['epoch']==30 and state['scheduler_clock']=='attempt_count' and state['attempts']==23970
 total=0
 def finite(x):
  global total
  if torch.is_tensor(x):assert torch.isfinite(x).all();total+=1
  elif isinstance(x,dict):
   for v in x.values():finite(v)
  elif isinstance(x,(tuple,list)):
   for v in x:finite(v)
 for key in ('model','best_model','optimizer'):finite(state[key])
 records.append(dict(backbone=backbone,probe_path=str(link),verified_copy=str(target),source_untouched=True,sha256=e['sha256'],epoch=state['epoch'],attempts=state['attempts'],identity=state['identity'],finite_tensor_count=total,CUDA_initialized=torch.cuda.is_initialized()))
 del state
for backbone in ('clip_vit_b16','dinov2_vit_b14','efficientnet_b0'):
 e=next(e for e in entries if e['relative'].startswith(f'runs/D_formal_{backbone}_seed7/') and e['relative'].endswith('_methods.npy'));target=Path(e['staged_path']);restored=root/(backbone+'_methods_restored.npy');shutil.copyfile(target,restored);os.utime(restored,ns=(e['mtime_ns'],e['mtime_ns']));assert sha(restored)==e['sha256'] and restored.stat().st_mtime_ns==e['mtime_ns']
 arr=np.load(restored,allow_pickle=False);assert len(arr)==102272
 records.append(dict(backbone=backbone,restored_probe=str(restored),sha256=e['sha256'],array_shape=list(arr.shape),dtype=str(arr.dtype),original_mtime_exactly_restored_on_SSD=True))
assert not torch.cuda.is_initialized()
report=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS_VERIFIED_COLD_COPY_RECOVERY_MAPPING_AND_REPRESENTATIVE_CPU_READS',cold_manifest_sha256=sha(M),file_count=len(entries),total_bytes=sum(e['bytes'] for e in entries),mapping=mapping,representative_read_and_restore_probes=records,target_mtime_differing_files=sum(d!=0 for d in deltas),target_mtime_delta_ns_min=min(deltas),target_mtime_delta_ns_max=max(deltas),source_originals_retained=True,SSD_released_bytes=0,actual_canonical_relocation_performed=False,forward_canonical_consumer_compatibility_complete=False,limitation='Cold targets preserve bytes but789 mtimes differ; representative restoration to original filesystem recovers exact original timestamps. Current source files and gates untouched. Before actual cold relocation, check every downstream consumer against candidate symlink timestamp/path behavior; archived qualification fingerprints must not be silently weakened.')
f=P/'cold_copy_recoverability_20261024T1630.json';assert not f.exists();f.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('status','file_count','target_mtime_differing_files','target_mtime_delta_ns_min','target_mtime_delta_ns_max','SSD_released_bytes')}))
