"""Finite original full H pair-cache admission in a NEW bounded CPU process.

Preserves every original role/PNG/RGB/source/checkpoint check. Only independent
source-completion checks inside each original ledger are scheduled with2 threads.
No cache dispatcher, GPU, scoring or cross-invocation receipt reuse exists here.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,os,sys,time
ROOT=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
CODE=Path('/home/z/workspace/deepfake_detector_publish_20261023/studies/pipeline_reliability/v2')
BACKBONES=('clip_vit_b16','dinov2_vit_b14','efficientnet_b0')
SEEDS=(7,42,123,2026,3407)
HELPER_PINS={'parallel_source.py':'a6133c824913ab3bf361095b415eefc9ef75be39ab32fb49ab62e2656dc2624a','cpu_ledger_adapter.py':'c6b98535e5a28123baae595ab1f4b208f1feab43ff4018e740be94366d044287'}
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def report_path(backbone,seed):return ROOT/f'reports/H_{backbone}_seed{seed}_independent_7role_parallel_candidate_20261025T0645.json'
def check_pins(pins):
 if any(digest(p)!=pin for p,pin in pins.items()):raise ValueError('canonical CPU admission pinned inputs changed')
def input_pins(backbone):
 if ROOT!=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023') or ROOT.resolve()!=ROOT or CODE.resolve()!=CODE:raise ValueError('canonical primary roots required')
 folder=ROOT/'qualification/H_parallel_fresh_validation'
 if Path(__file__).resolve().parent!=folder:raise ValueError('canonical reviewed CPU entry required')
 manifest=ROOT/'qualification/H_primary_cache_continuation/qualified_source_manifest.json'
 if digest(manifest)!='977431237b901d5ac88172f98bc2591d3cd7992c35be50546cda34ad2276a747':raise ValueError('qualified20source inventory changed')
 original=json.loads(manifest.read_text())['source_sha256']
 if len(original)!=20:raise ValueError('exact qualified20source inventory required')
 pins={str(CODE/rel):pin for rel,pin in original.items()}
 pins.update({str(folder/name):pin for name,pin in HELPER_PINS.items()});pins[str(manifest)]=digest(manifest);pins[str(Path(__file__).resolve())]=digest(__file__)
 accepted=ROOT/'reports/H_final_all3_effective_registry_acceptance_20261025T0555.json'
 if digest(accepted)!='49a620f6de1c58e259d8a214ce6cae188c911e2118d22a3d5f5fcf39c24f83ed':raise ValueError('final three-backbone admission proof changed')
 registry=ROOT/'H_run_registry.json'
 if digest(registry)!='93d9336f5fe57c8af09b0d0177f0e1f9dd1bbab51331347127dec95e4e8dda27':raise ValueError('final effective registry changed')
 record=json.loads(registry.read_text());ids=record['effective_runs'][backbone]['run_ids']
 paths=[accepted,registry,ROOT/'configs/H_execution.json',ROOT/'data/H/freeze.json',ROOT/'qualification/H_gate.json',ROOT/'configs/H_frozen_recipes.json',ROOT/'configs/requirements.execution.lock.txt']
 for b in BACKBONES:
  precision=record['effective_runs'][b]['precision'];paths.extend([ROOT/f'qualification/H_target_export_{b}.json',ROOT/f'reports/H_{b}_{precision}_paired_ledger.json'])
 for run in ids:
  for name in ('identity.json','resolved_config.json','history.json','complete.json','H_completion_seal.json'):paths.append(ROOT/'runs'/run/name)
 pins.update({str(path):digest(path) for path in paths});check_pins(pins);return pins,record

def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');parser.add_argument('--backbone',choices=BACKBONES,default='clip_vit_b16');parser.add_argument('--seed',type=int,choices=SEEDS,default=42);args=parser.parse_args(argv)
 if not args.execute:return dict(execute=False,GPU_launch_attempted=False,scope='original full14role pair admission, fresh original10source checks per ledger in2threads; review and actual qualification required')
 output=report_path(args.backbone,args.seed)
 if output.exists() or output.is_symlink():raise FileExistsError(output)
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(os.environ.get(k)!='2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')):raise ValueError('CPU-only empty CUDA and exactly2threads required')
 pins,registry=input_pins(args.backbone)
 if any(name=='reliability_v2' or name.startswith('reliability_v2.') for name in sys.modules):raise ValueError('canonical science imports must begin in a new process')
 sys.path.insert(0,str(CODE));sys.path.insert(0,str(Path(__file__).resolve().parent))
 from reliability_v2 import analysis_h,queue_h,worker_h
 from cpu_ledger_adapter import installed_fresh_audit
 from parallel_source import publish_json_new
 for module in (analysis_h,queue_h,worker_h):
  if Path(module.__file__).resolve()!=CODE/'reliability_v2'/Path(module.__file__).name:raise ValueError('foreign scientific module path')
 config=ROOT/'configs/H_execution.json';effective=registry['effective_runs'][args.backbone];precision=effective['precision']
 expected=[queue_h.formal_id(args.backbone,h,s,precision) for s in SEEDS for h in ('B','AB')]
 if effective['run_ids']!=expected:raise ValueError('exact all five effective pairs required')
 def ids_for(backbone,p):
  if backbone!=args.backbone or p!=precision:raise ValueError('unexpected backbone/precision')
  return expected
 def ledger_for(backbone,p):return ROOT/f'reports/H_{backbone}_{p}_paired_ledger.json'
 check_pins(pins);start=time.monotonic();print('BEGIN_ACTUAL_CANONICAL_ORIGINAL_PAIR_CONTEXT_WITH_FRESH_PARALLEL_LEDGERS',args.backbone,args.seed,flush=True)
 with installed_fresh_audit(queue_h,config,worker_h.verify_run_completion,ids_for,ledger_for,pins) as ledgers:
  root,configuration,freeze,runs,identity=analysis_h._pair_context(config,args.backbone,args.seed)
 check_pins(pins)
 if len(ledgers)!=14 or len(identity['cache_completion_sha256'])!=14 or any(row['fresh_source_completions']!=10 for row in ledgers):raise ValueError('original full14role/140fresh source checks were not completed')
 result=dict(status='PASS_ORIGINAL_FULL14ROLE_PAIR_CONTEXT_WITH_FRESH_PARALLEL_ORIGINAL_LEDGERS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),backbone=args.backbone,seed=args.seed,models=2,roles=14,identity=identity,elapsed_seconds=time.monotonic()-start,pinned_sha256=pins,fresh_ledger_invocations=ledgers,all_source_completions_fresh=140,cross_invocation_receipt_reuse=False,production_scoring_executed=False,GPU_launch_attempted=False,live_cache_owner_changed=False,scope='Original full role verification, exact original paired numerical ledgers and every PNG byte/RGB check. Only independent source-completion scheduling changed inside this NEW CPU process.')
 publish_json_new(output,result);print('PASS_ACTUAL_CANONICAL_ORIGINAL_PAIR_CONTEXT',output,flush=True);return result
if __name__=='__main__':main()
