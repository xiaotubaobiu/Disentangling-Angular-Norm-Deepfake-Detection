"""CPU-only source equivalence benchmark. Default invocation is inert."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time,threading
ROOT=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
CODE=Path('/home/z/workspace/deepfake_detector_publish_20261023/studies/pipeline_reliability/v2')
REPORT_RELATIVE=Path('qualification/H_parallel_fresh_validation/real_source_equivalence_20261025T0120.json')

def run_source_benchmark():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(os.environ.get(k)!='2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')):
        raise ValueError('CPU-only empty CUDA and bounded threads required')
    output=ROOT/REPORT_RELATIVE
    if output.exists() or output.is_symlink():raise FileExistsError(output)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    supervision=ROOT/'reports/check_20261025T0055_recorded_summary.json'
    if sha(supervision)!='44c5bb13fd9a94cc7696fe0788588e6b56690b9157ef15dd3ce19a4c0da62140':raise ValueError('Accepted serial supervision changed')
    accepted=json.loads(supervision.read_text())
    refpath=ROOT/'reports/H_CLIP_allfive_readonly_source_ledger_20261025T0055.json'
    if sha(refpath)!=accepted['evidence_sha256'][str(refpath)]:raise ValueError('Serial reference changed')
    reference=json.loads(refpath.read_text());original=reference['original_frozen_ledger']
    if reference['status']!='SOURCE_ONLY_ALLFIVE_MATCHED_GLOBAL_TARGET_PHASE_PENDING':raise ValueError('Incomplete serial source reference')
    config_path=ROOT/'configs/H_execution.json';registry=ROOT/'H_run_registry.json';queue=CODE/'reliability_v2/queue_h.py'
    if sha(config_path)!=reference['H_execution_sha256'] or sha(registry)!=reference['H_run_registry_sha256'] or sha(queue)!=reference['queue_source_sha256']:raise ValueError('Reference/configuration/code drift')
    candidate=ROOT/'qualification/H_parallel_fresh_validation';unit_path=candidate/'candidate_acceptance.json';unit=json.loads(unit_path.read_text())
    if sha(candidate/'parallel_source.py')!=unit['source_sha256'] or sha(Path(__file__))!=unit['entry_sha256'] or sha(candidate/'tests/test_parallel_source.py')!=unit['tests_sha256']:raise ValueError('Candidate test binding changed')
    sys.path.insert(0,str(candidate));sys.path.insert(0,str(CODE))
    from parallel_source import collect_fresh,FreshLookup,assert_same_ledger
    import torch
    from reliability_v2 import queue_h,worker_h
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():raise ValueError('CUDA initialized unexpectedly')
    config=json.loads(config_path.read_text());reg=json.loads(registry.read_text())
    if reg['effective_runs'] or reg['target_export_allowed'] is not False:raise ValueError('This held source-only benchmark requires the current unexported phase')
    if worker_h.h_code_hash()!=config['h_code_bundle_sha256']:raise ValueError('Frozen H training code drift')
    expected=[queue_h.formal_id('clip_vit_b16',h,s,'AMP') for s in queue_h.FORMAL_SEEDS for h in queue_h.HISTORIES]
    if len(expected)!=10 or len(set(expected))!=10:raise ValueError('Expected allten formal source models')
    bound=dict(reference['input_sha256']);bound.update({str(supervision):sha(supervision),str(refpath):sha(refpath),str(config_path):sha(config_path),str(registry):sha(registry),str(queue):sha(queue),str(candidate/'parallel_source.py'):sha(candidate/'parallel_source.py'),str(Path(__file__)):sha(Path(__file__)),str(unit_path):sha(unit_path)})
    if not all(sha(p)==v for p,v in bound.items()):raise ValueError('Accepted serial terminal evidence changed')
    paths=[]
    for run in expected:
        folder=ROOT/'runs'/run;sealpath=folder/'H_completion_seal.json';seal=json.loads(sealpath.read_text())
        if set(seal['artifact_sha256'])!=set(worker_h._completion_artifacts(30)):raise ValueError('Incomplete sealed inventory')
        paths.extend(folder/name for name in seal['artifact_sha256']);paths.append(sealpath)
    metadata={}
    for p in paths:
        if p.is_symlink() or not p.is_file():raise ValueError('Unexpected sealed path')
        stat=p.stat();metadata[str(p)]=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
    completed=[];active=0;peak=0;lock=threading.Lock();started=time.monotonic()
    def full_verify(config_arg,run):
        nonlocal active,peak
        with lock:
            active+=1;peak=max(peak,active);print('FRESH_ORIGINAL_BEGIN '+run,flush=True)
        try:receipt=worker_h.verify_run_completion(config_arg,run)
        finally:
            with lock:active-=1
        with lock:completed.append(run);print('FRESH_ORIGINAL_END '+run,flush=True)
        return receipt
    fresh=collect_fresh(config_path,expected,full_verify,workers=2)
    lookup=FreshLookup(config_path,fresh)
    ledger=queue_h.audit_paired_ledgers(config_path,'clip_vit_b16',precision='AMP',verify=lookup)
    lookup.finish();assert_same_ledger(original,ledger)
    if len(completed)!=10 or set(completed)!=set(expected) or peak>2 or active:raise ValueError('Fresh verification completeness/concurrency drift')
    for p,old in metadata.items():
        st=Path(p).stat()
        if old!=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns):raise ValueError('Sealed input changed during parallel audit')
    if not all(sha(p)==v for p,v in bound.items()) or torch.cuda.is_initialized():raise ValueError('End-of-audit evidence/CUDA drift')
    elapsed=time.monotonic()-started
    result=dict(utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),status='PARALLEL_SOURCE_EQUIVALENCE_PASS_PRODUCTION_INTEGRATION_HELD',CPU_only=True,maximum_verifier_threads=2,observed_peak_verifiers=peak,completed_fresh_original_verifications=completed,serial_reference_sha256=sha(refpath),serial_seconds=reference['CPU_audit_seconds'],parallel_seconds=elapsed,single_observed_speed_ratio=reference['CPU_audit_seconds']/elapsed,speed_scope='Single source-only measurement under concurrent training; no end-to-end cache or cross-backbone promise.',original_parallel_ledger=ledger,all_original_ledger_fields_equal_except_top_level_utc=True,full_original_verifier_called_fresh_for_every_model=True,receipts_reused_between_invocations=False,queue_verify_callback_is_current_fresh_receipts=True,sealed_artifact_stat_snapshots_checked=len(metadata),input_sha256=bound,source_best_HOLD=True,target_export_started=False,gate_or_registry_written=False,production_code_changed=False,production_integration_accepted=False,independent_review_pending=True)
    with output.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    return result

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');args=parser.parse_args(argv)
    if not args.execute:
        result=dict(status='CHECK_ONLY_EXPLICIT_EXECUTE_REQUIRED',executed=False,production_integration=False);print(json.dumps(result));return result
    result=run_source_benchmark();print(json.dumps({k:result[k] for k in ('status','parallel_seconds','serial_seconds','single_observed_speed_ratio')}));return result
if __name__=='__main__':main()
