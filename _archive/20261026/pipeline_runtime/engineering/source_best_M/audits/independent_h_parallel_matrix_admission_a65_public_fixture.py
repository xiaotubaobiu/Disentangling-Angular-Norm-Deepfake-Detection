from pathlib import Path
import os,json,traceback,contextlib,datetime,hashlib
A=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023/engineering/source_best_M/audits')
source=(A/'independent_h_parallel_matrix_admission_public_fixture.py').read_text().split("\nif __name__=='__main__':")[0]
exec(compile(source,str(A/'independent_h_parallel_matrix_admission_public_fixture.py'),'exec'),globals())
results=[]
cases=[('prototype_positive','dinov2_vit_b14',7,True),('clip_positive','clip_vit_b16',3407,True),('snapshot_clip_positive','clip_vit_b16',3407,True),('live_true_birth','dinov2_vit_b14',7,False),('live_birth_plus1','dinov2_vit_b14',7,False),('live_birth_minus1','dinov2_vit_b14',7,False),('live_birth_later_same_era','dinov2_vit_b14',7,False),('proven_later_PID_reuse','dinov2_vit_b14',7,True),('qualified_bad_Q','dinov2_vit_b14',7,False),('qualified_DINO_direct_owned','dinov2_vit_b14',7,False),('clip_missing_Q','clip_vit_b16',3407,False),('clip_wrong_Q_sha','clip_vit_b16',3407,False),('clip_wrong_Q_path','clip_vit_b16',3407,False),('clip_changed_Q_bytes','clip_vit_b16',3407,False),('protected_pin_conflict','dinov2_vit_b14',7,False),('run_metadata_drift','dinov2_vit_b14',7,False),('seal_drift','dinov2_vit_b14',7,False),('time_exit1','dinov2_vit_b14',7,False),('log_complete_first','dinov2_vit_b14',7,False)]
for case,bone,seed,expect in cases:
 row=dict(case=case,expected_accept=expect);entered=[]
 try:
  m,root,b,mode,names=fixture(case,bone,seed)
  # All public cases use this non-acquiring fake context: no real or temporary
  # flock/slot function is called and no actual shared slot is opened.
  @contextlib.contextmanager
  def fake_slot(*args):
   entered.append([str(v) for v in args]);yield
  m.old.SHARED.cpu_slot=fake_slot
  refs=json.loads(b.read_text())
  if bone=='clip_vit_b16':
   q=root/'reports/H_parallel_CPU_actual_qualification_clip_vit_b16_20261025T0830.json';refs['proofs']['CPU_qualification']=dict(path=str(q),sha256=H(q))
   if case=='clip_missing_Q':refs['proofs'].pop('CPU_qualification')
   if case=='clip_wrong_Q_sha':refs['proofs']['CPU_qualification']['sha256']='0'*64
   if case=='clip_wrong_Q_path':refs['proofs']['CPU_qualification']['path']=str(root/'reports/wrong.json')
   if case=='clip_changed_Q_bytes':q.write_bytes(q.read_bytes()+b'\n');refs['proofs']['CPU_qualification']['sha256']=H(q)
  if case=='snapshot_clip_positive':
   snapshot=root/'engineering/source_best_M/qualification_snapshots'/H(SOURCE)/'entry.py';snapshot.parent.mkdir(parents=True);snapshot.write_bytes(SOURCE.read_bytes());m.__file__=str(snapshot)
  if case in ('live_birth_minus1','live_birth_later_same_era','proven_later_PID_reuse'):
   L=json.loads(names['launch'].read_text());w=json.loads(names['process_witness'].read_text());pid=os.getpid();tick=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]);boot=int(next(x.split()[1] for x in Path('/proc/stat').read_text().splitlines() if x.startswith('btime ')));hz=os.sysconf('SC_CLK_TCK');oldtick=tick-1000 if case=='proven_later_PID_reuse' else tick+(-1 if case=='live_birth_minus1' else -100)
   L.update(pid=pid,utc=datetime.datetime.fromtimestamp(boot+oldtick/hz,datetime.timezone.utc).isoformat());dump(names['launch'],L);w['launch_sha256']=H(names['launch']);w['wrapper_pid']=pid;w['observed_process_tree'][0]['pid']=pid
   for n in w['observed_process_tree']:n['start_ticks']=oldtick
   if case=='proven_later_PID_reuse':w['utc']=datetime.datetime.fromtimestamp(boot+oldtick/hz+.25,datetime.timezone.utc).isoformat()
   dump(names['process_witness'],w);refs['proofs']['launch']['sha256']=H(names['launch']);refs['proofs']['process_witness']['sha256']=H(names['process_witness'])
  dump(b,refs)
  if case=='qualified_DINO_direct_owned':m.audit_owned(dict(path=str(b),sha256=H(b)),'qualified_pair',1)
  else:
   result=m.main(['--execute','--binding',str(b),'--binding-sha',H(b),'--mode',mode,'--slot','1' if bone=='dinov2_vit_b14' else '0']);d=json.loads(Path(result['output']).read_text())
   assert len(d['canonical_shared107_pin_sha256'])==107 and len(d['role_marker_checks'])==14 and len(d['role_manifest_footer_metadata'])==7 and d['source_checks_in_ledger_receipts']==140
   assert result['CPU_qualification_issued'] is False and d['actual_CPU_qualification_issued'] is False and d['M_authorized'] is False and d['production_GPU_allowed'] is False and d['scientific_metric_or_bootstrap_results_accepted'] is False
   assert d['bracketed_readonly_evidence_sha256'][m.__file__]==H(m.__file__)
   if bone=='clip_vit_b16':assert d['bracketed_readonly_evidence_sha256'][str(q)]=='0ae309b3366de860cbd77f539c24a05aae3ec1180dbbd84bbbff95ed3111d7f7'
   row.update(output=result['output'],pins=107,roles=14,footers=7,source_checks=140,Q_issued=False,self_path=m.__file__,self_sha=H(m.__file__))
  row['accepted']=True
 except BaseException as e:row.update(accepted=False,error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc())
 row['fake_slot_entries']=len(entered)
 if case.startswith('qualified_'):assert not entered,'disabled DINO reached even fake CPU slot'
 row['matches_expected']=row['accepted']==expect;results.append(row)
print(json.dumps(results,indent=2));(OUT/'fixed_public_results.json').write_text(json.dumps(results,indent=2)+'\n')
assert all(x['matches_expected'] for x in results),'public boundary regression'
assert 'torch' not in __import__('sys').modules
print('PASS_PUBLIC_BOUNDARIES',len(results),'NO_TORCH_ALL_LOCKS_FAKE')
