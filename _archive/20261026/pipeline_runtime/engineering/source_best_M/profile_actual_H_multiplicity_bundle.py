"""One-shot transport probe for independently accepted real H LAST payloads.
No inference, resampling, production M permit, original edits or background watch.
"""
from pathlib import Path
import argparse,datetime,gc,hashlib,json,os,resource,shutil,time
from reliability_v2 import multiplicity_bundle as bundle
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023');E=R/'engineering/source_best_M'
W=E/'code/studies/pipeline_reliability/v2'
def require(ok,why):
 if not ok:raise ValueError(why)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 return h.hexdigest()
def safe(path):
 p=Path(path).absolute();require(p.is_relative_to(R) and p.resolve()==p and p.is_file(),'canonical local regular input required');return p
def canonical(value):
 h=hashlib.sha256();n=0
 for text in json.JSONEncoder(ensure_ascii=False,indent=2,allow_nan=False).iterencode(value):
  x=text.encode();h.update(x);n+=len(x)
 h.update(b'\n');return h.hexdigest(),n+1

def main():
 p=argparse.ArgumentParser();p.add_argument('--backbone',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--marker-sha',required=True)
 p.add_argument('--independent-acceptance',required=True);p.add_argument('--acceptance-sha',required=True);p.add_argument('--bundle-sha',required=True)
 p.add_argument('--primary-budget',required=True);p.add_argument('--budget-sha',required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
 require(a.backbone in ('clip_vit_b16','dinov2_vit_b14','efficientnet_b0') and a.seed in (7,42,123,2026,3407),'registered primary model required')
 source=R/f'analysis/H/{a.backbone}/seed{a.seed}';marker=safe(source/'complete.json');require(sha(marker)==a.marker_sha,'primary H terminal marker drift');complete=json.loads(marker.read_text())
 require(complete.get('schema')=='H_PAIR_ANALYSIS_COMPLETE_V1' and complete.get('status') in ('ANALYSIS_COMPLETE','ANALYSIS_COMPLETE_WITH_NUMERIC_NA'),'real H primary analysis terminal required')
 accepted=safe(a.independent_acceptance);require(sha(accepted)==a.acceptance_sha,'independent acceptance changed');proof=json.loads(accepted.read_text())
 require(proof.get('source_complete_sha256')==a.marker_sha and proof.get('all_saved_metrics_independently_recomputed') is True and proof.get('all_reportable_bootstrap_draws_independently_verified') is True,'independent real H metric/draw acceptance required')
 code=W/'reliability_v2/multiplicity_bundle.py';require(sha(code)==a.bundle_sha,'accepted bundle code changed')
 bp=safe(a.primary_budget);require(sha(bp)==a.budget_sha,'primary budget changed');budget=json.loads(bp.read_text());primary=budget.get('normal_required_bytes',budget.get('remaining_primary_conservative_bytes'))
 require(type(primary) is int and primary>=50*1024**3,'complete primary remaining/temp/reserve budget required')
 selected=[f'{aggregation}/{domain}/all_B/conditional_bootstrap.json' for aggregation in ('mean_logit','mean_probability') for domain in ('SOURCE_TEST','TARGET_TEST')]
 pins={str(marker):a.marker_sha,str(accepted):a.acceptance_sha,str(bp):a.budget_sha,str(code):a.bundle_sha,str(Path(__file__).resolve()):sha(__file__)}
 limits=dict(max_cells=40000000,max_json_bytes=2*1024**3,max_array_bytes=512*1024**2)
 for key in selected:
  source_file=safe(source/key);expected=complete['artifact_sha256'][key];require(sha(source_file)==expected,'accepted H source payload changed');pins[str(source_file)]=expected
 # Worst-case codec limits, not a compression estimate. Existing source files
 # are not moved/deleted and no reclaimed-SSD credit is assumed.
 worst_stored=len(selected)*(limits['max_json_bytes']+limits['max_array_bytes']+8*1024**2)
 require(shutil.disk_usage(R).free>=primary+worst_stored,'full primary plus worst-case probe capacity insufficient')
 check=dict(status='READY_TRANSPORT_PROBE_ONLY',source=str(source),marker_sha256=a.marker_sha,files=selected,source_raw_bytes=sum((source/key).stat().st_size for key in selected),primary_remaining_plus_temp_reserve_bytes=primary,additional_probe_worst_case_bytes=worst_stored,production_M_admitted=False)
 if a.preflight_only:print(json.dumps(check));return
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(x)=='2' for x in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'bounded CPU/empty CUDA environment required')
 resource.setrlimit(resource.RLIMIT_AS,(64*1024**3,64*1024**3));out=E/f'storage_profiles/H_{a.backbone}_seed{a.seed}_atomic_bundle';require(not out.exists(),'use a fresh probe namespace; never replace an earlier probe');out.mkdir(parents=True)
 records=[];start=time.monotonic()
 for n,key in enumerate(selected):
  source_file=source/key;expected=pins[str(source_file)];print('BEGIN_ACTUAL_H',key,flush=True);value=json.loads(source_file.read_bytes())
  identity=dict(scope='ENGINEERING_TRANSPORT_ONLY_ACCEPTED_LAST_H_NATIVE_B_PAYLOAD',source_relative=key,source_sha256=expected,source_complete_sha256=a.marker_sha,source_best_results=False)
  receipt=bundle.write_bundle(out/str(n),value,identity=identity,**limits)
  decoded=bundle.read_bundle(out/str(n),expected_complete_sha256=receipt['complete_sha256'],expected_identity=identity,**limits)
  require(decoded==value and canonical(decoded)==(expected,source_file.stat().st_size),'real H full ordered JSON byte roundtrip failed')
  require(sha(source_file)==expected and sha(marker)==a.marker_sha,'actual input changed during probe')
  records.append(dict(source_relative=key,original_bytes=source_file.stat().st_size,receipt=receipt,all_original_fields_and_ordered_JSON_bytes_exact=True));print('PASS_ACTUAL_H',key,receipt['stored_bytes'],flush=True);del value,decoded;gc.collect()
 for path,pin in pins.items():require(sha(path)==pin,'probe input/code changed')
 result=dict(check,status='PASS_ACTUAL_H_NATIVE_B_ORDERED_JSON_TRANSPORT_PROFILE_ONLY',results=records,seconds=time.monotonic()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,input_sha256=pins,source_originals_mutated=False,new_scientific_results=False,source_best_results_generated=False,joint_M_capacity_admitted=False)
 report=E/'actual_H_atomic_bundle_profile.json'
 with report.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
 print('ACTUAL_H_PROFILE_COMPLETE',str(report),flush=True)
if __name__=='__main__':main()
