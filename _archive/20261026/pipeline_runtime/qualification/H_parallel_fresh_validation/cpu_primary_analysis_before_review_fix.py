"""Finite original primary H statistics after actual CPU/cache qualification.

No GPU/cache dispatcher, scientific kernel changes or live owner edits.
Default is a nonexecuting plan. Actual outputs still need independent audit.
"""
from pathlib import Path
from contextlib import contextmanager
import argparse,datetime,fcntl,hashlib,json,os,sys,time,shutil
ROOT=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
CODE=Path('/home/z/workspace/deepfake_detector_publish_20261023/studies/pipeline_reliability/v2')
BACKBONES=('clip_vit_b16','dinov2_vit_b14','efficientnet_b0');SEEDS=(7,42,123,2026,3407)
ROLES=('B/SOURCE_CAL','B/SOURCE_TEST','B/TARGET_TEST','common/B/SOURCE_TEST','common/B/TARGET_TEST','common/AB/SOURCE_TEST','common/AB/TARGET_TEST')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_proof(path,pin):
 p=Path(path)
 if p.is_symlink() or p.resolve()!=p or ROOT/'reports' not in p.parents:raise ValueError('canonical reports proof required')
 if not p.is_file():raise FileNotFoundError(p)
 if p.stat().st_size>16*1024**2 or sha(p)!=pin:raise ValueError('proof SHA/size mismatch')
 return json.loads(p.read_text())
def validate_qualification(q,backbone):
 if (q.get('schema')!='H_FRESH_PARALLEL_CPU_VALIDATION_ACCEPTANCE_V1' or q.get('status')!='PASS_ACTUAL_ORIGINAL14ROLE_FRESH140SOURCE_EXACT_LEDGER_EQUIVALENCE'
     or not isinstance(q.get('allowed_backbones'),list) or backbone not in q['allowed_backbones'] or not isinstance(q.get('allowed_use'),list) or 'primary_pair_analysis' not in q['allowed_use']
     or type(q.get('original_role_checks')) is not int or q['original_role_checks']!=14 or type(q.get('fresh_default_source_checks')) is not int or q['fresh_default_source_checks']!=140
     or q.get('all_original_ledger_checks_exact_except_utc') is not True or q.get('independent_review_pass') is not True or q.get('production_GPU_allowed') is not False):raise ValueError('complete actual independently accepted CPU equivalence required')
def validate_pair(d,backbone,seed,expected_runs):
 identity=d.get('identity',{});markers=identity.get('cache_completion_sha256',{})
 if (d.get('status') not in ('PASS_ORIGINAL_FULL_14_ROLE_PAIR_CONTEXT_INDEPENDENT_OF_CACHE_WORKERS','PASS_ORIGINAL_FULL14ROLE_PAIR_CONTEXT_WITH_FRESH_PARALLEL_ORIGINAL_LEDGERS')
     or d.get('backbone')!=backbone or type(d.get('seed')) is not int or d['seed']!=seed or type(d.get('models')) is not int or d['models']!=2 or type(d.get('roles')) is not int or d['roles']!=14
     or identity.get('run_ids')!=expected_runs or set(markers)!={history+'/'+role for history in ('B','AB') for role in ROLES}):raise ValueError('exact original fully admitted pair required')
def legacy_slot_alive(slot):
 entry=(ROOT/'qualification/H_primary_cache_independent_admission/admit_clip_seed7_20261025T0750.py') if slot==0 else ROOT/'qualification/H_parallel_fresh_validation/cpu_ledger_admission.py'
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:args=(p/'cmdline').read_bytes().split(b'\x00')
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  if str(entry).encode() in args:return True
 return False
@contextmanager
def cpu_slot(slot):
 if legacy_slot_alive(slot):raise RuntimeError('existing independent admission still owns CPU slot')
 folder=ROOT/'analysis/H/.cpu_slots';folder.mkdir(parents=True,exist_ok=True)
 with (folder/f'slot{slot}.lock').open('a+b') as stream:
  try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError as error:raise RuntimeError('CPU analysis slot busy') from error
  try:
   if legacy_slot_alive(slot):raise RuntimeError('legacy admission CPU slot not released')
   yield
  finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--backbone',choices=BACKBONES,default='clip_vit_b16');p.add_argument('--seed',type=int,choices=SEEDS,default=42);p.add_argument('--slot',type=int,choices=(0,1),default=1)
 for name in ('qualification','qualification-sha','admission','admission-sha'):p.add_argument('--'+name)
 a=p.parse_args(argv)
 if not a.execute:return dict(execute=False,GPU_launch_attempted=False,scope='original primary H analyze_pair only after actual parallel qualification and exact cache-pair admission; max2CPU slots')
 if not all(getattr(a,k) for k in ('qualification','qualification_sha','admission','admission_sha')):p.error('execute requires external qualification and pair admission SHA pins')
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(os.environ.get(k)!='2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')):raise ValueError('empty CUDA and bounded2threads required')
 q=read_proof(a.qualification,a.qualification_sha);d=read_proof(a.admission,a.admission_sha);validate_qualification(q,a.backbone)
 folder=ROOT/'qualification/H_parallel_fresh_validation'
 if Path(__file__).resolve().parent!=folder:raise ValueError('canonical CPU entry required')
 sys.path.insert(0,str(folder));import cpu_ledger_admission as admission
 pins,registry=admission.input_pins(a.backbone)
 for name in ('parallel_source.py','cpu_ledger_adapter.py','cpu_ledger_admission.py'):
  if q.get('implementation_sha256',{}).get(name)!=sha(folder/name):raise ValueError('actual CPU qualification implementation changed')
 for arg,pin in [(a.qualification,a.qualification_sha),(a.admission,a.admission_sha)]:pins[str(Path(arg))]=pin
 pins[str(Path(__file__).resolve())]=sha(__file__)
 config_path=ROOT/'configs/H_execution.json';config=json.loads(config_path.read_text());effective=registry['effective_runs'][a.backbone]
 expected={config['runs'][run]['history']:run for run in effective['run_ids'] if config['runs'][run]['seed']==a.seed};validate_pair(d,a.backbone,a.seed,expected)
 for path,pin in d.get('pinned_sha256',{}).items():
  if sha(path)!=pin:raise ValueError('independent admission input drift')
 for history,run in expected.items():
  for role in ROLES:
   if sha(ROOT/'features/H'/run/role/'complete.json')!=d['identity']['cache_completion_sha256'][history+'/'+role]:raise ValueError('independently admitted role marker changed')
 budget_path=ROOT/'reports/full_capacity_check_pre_H_cache_20261025T0555.json'
 if sha(budget_path)!='3ce2d435d51895b9cfe53ac2f4bde6df12db244a7a2365c89a3f31b418b502cd':raise ValueError('original full remaining primary budget changed')
 required=json.loads(budget_path.read_text())['normal_required_bytes']
 if shutil.disk_usage(ROOT).free<=required:raise RuntimeError('full remaining primary+temp+reserve budget not covered')
 output=ROOT/f'reports/H_primary_analysis_execution_{a.backbone}_seed{a.seed}.json'
 if output.exists():raise FileExistsError(output)
 if any(n=='reliability_v2' or n.startswith('reliability_v2.') for n in sys.modules):raise ValueError('new process scientific imports required')
 sys.path.insert(0,str(CODE));from reliability_v2 import analysis_h,queue_h,worker_h
 from cpu_ledger_adapter import installed_fresh_audit
 from parallel_source import publish_json_new
 admission.check_pins(pins);original_complete=ROOT/f'analysis/H/{a.backbone}/seed{a.seed}/complete.json';was_complete=original_complete.exists();start=time.monotonic();counter=[0]
 def ids_for(backbone,precision):
  if backbone!=a.backbone or precision!=effective['precision']:raise ValueError('unexpected ledger request')
  counter[0]+=1;print('BEGIN_PRIMARY_FRESH_LEDGER',counter[0],backbone,precision,flush=True);return effective['run_ids']
 def ledger_for(backbone,precision):return ROOT/f'reports/H_{backbone}_{precision}_paired_ledger.json'
 with cpu_slot(a.slot):
  with installed_fresh_audit(queue_h,config_path,worker_h.verify_run_completion,ids_for,ledger_for,pins) as ledgers:result=analysis_h.analyze_pair(config_path,a.backbone,a.seed)
  admission.check_pins(pins)
  if len(ledgers)!=(14 if was_complete else 42):raise ValueError('original full source/cache checks not completed')
  record=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='ORIGINAL_PRIMARY_H_ANALYSIS_EXECUTION_COMPLETE_REQUIRES_INDEPENDENT_METRIC_BOOTSTRAP_AUDIT',backbone=a.backbone,seed=a.seed,source_qualification_sha256=a.qualification_sha,pair_admission_sha256=a.admission_sha,pinned_sha256=pins,CPU_slot=a.slot,original_analysis_complete_sha256=sha(original_complete),original_analysis_identity=result.get('identity'),fresh_ledger_invocations=ledgers,elapsed_seconds=time.monotonic()-start,GPU_launch_attempted=False,primary_cache_owner_modified=False,M_results_generated=False,scientific_final_acceptance=False)
  publish_json_new(output,record);print('COMPLETE_ORIGINAL_PRIMARY_H_ANALYSIS_PENDING_INDEPENDENT_AUDIT',output,flush=True);return record
if __name__=='__main__':main()
