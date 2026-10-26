"""Independent EOF-only verification; preserves the original review receipt."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

R = Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
V = Path('/home/z/workspace/deepfake_detector_publish_20261023/studies/pipeline_reliability/v2')
L = R/'reports/h_kernel_independent_review_logs_20261023'
original_report = R/'reports/H_kernel_independent_review.json'
root_addendum = R/'reports/H_coverage_format_addendum.json'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

original_report_sha = sha(original_report)
assert original_report_sha == 'b81685ebaa8167c48fa288d01eb8560b4e963c228ac8f197fd776536d11942bc'
previous = json.loads(original_report.read_text())
assert previous['status'] == 'PASSED_SYNTHETIC_CPU_KERNEL_REVIEW'
root_evidence = json.loads(root_addendum.read_text())
expected = {
    'reliability_v2/coverage_h.py': (
        'a858e384591e2aaadf9232faee2e18c73964a384ec65868f783b5368bb33b7e0',
        '268a77b2ddb5458a9a85276347c2e43d51f86d0a2c4ff719b95d7b15d235deee'),
    'tests/test_coverage_h.py': (
        '6e82ae7c9a5ca4aa665e9152be3b9eaa3d5c3fd2190387bbb588a1c8831527f6',
        '993cece73b6344020ccc15990eb40d8bf1908ff02450e35aa49bfedfb73b0ea4')}
verified = []
for change in root_evidence['changes']:
    relative = change['path']
    old_sha, new_sha = expected[relative]
    old_path = Path(change['old_backup'])
    new_path = V/relative
    old, new = old_path.read_bytes(), new_path.read_bytes()
    assert hashlib.sha256(old).hexdigest() == old_sha == change['old_sha256']
    assert hashlib.sha256(new).hexdigest() == new_sha == change['new_sha256']
    assert previous['reviewed_input_sha256'][str(new_path)] == old_sha
    expected_bytes = old.rstrip(b'\r\n') + b'\n'
    assert new == expected_bytes, f'Non-EOF byte change: {relative}'
    old_ast = ast.dump(ast.parse(old.decode('utf-8'), type_comments=True), include_attributes=False)
    new_ast = ast.dump(ast.parse(new.decode('utf-8'), type_comments=True), include_attributes=False)
    assert old_ast == new_ast, f'AST changed: {relative}'
    verified.append({'path':str(new_path),'old_backup':str(old_path),'old_sha256':old_sha,'new_sha256':new_sha,
        'backup_matches_original_independent_review':True,'ast_equivalent_with_type_comments':True,
        'ast_sha256':hashlib.sha256(old_ast.encode()).hexdigest(),
        'raw_bytes_change_only_trailing_CR_LF_sequence_to_one_LF':True,
        'old_bytes':len(old),'new_bytes':len(new)})
assert len(verified)==len(expected)==2
unchanged = {}
for path, digest in previous['reviewed_input_sha256'].items():
    if path in {str(V/relative) for relative in expected}: continue
    assert sha(path)==digest, f'Other reviewed input changed: {path}'
    unchanged[path]=digest

env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
    NUMEXPR_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(V))
command = [str(R/'environment/bin/python'),'-m','pytest','-q','-p','no:cacheprovider',
    f'--basetemp={L}/pytest_format_addendum_temp',str(V/'tests/test_stats_h.py'),str(V/'tests/test_coverage_h.py')]
started=time.perf_counter()
run=subprocess.run(command,cwd=V,env=env,capture_output=True,text=True)
elapsed=time.perf_counter()-started
stdout=L/'format_addendum_focused_pytest_stdout.txt'
stdout.write_text(run.stdout+run.stderr)
assert run.returncode==0, run.stdout+run.stderr
assert re.search(r'\b56 passed\b',run.stdout),run.stdout
for item in verified:
    assert sha(item['path'])==item['new_sha256']
    assert sha(item['old_backup'])==item['old_sha256']
for path,digest in unchanged.items(): assert sha(path)==digest
assert sha(original_report)==original_report_sha

report = {'schema':'H_KERNEL_INDEPENDENT_FORMAT_ADDENDUM_V1','status':'PASS',
    'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    'scope':'Independent original-byte hash, exact EOF-only transformation, AST equivalence, and 56 focused test rerun',
    'original_independent_review':{'path':str(original_report),'sha256':original_report_sha,'preserved_unchanged':True},
    'root_format_addendum':{'path':str(root_addendum),'sha256':sha(root_addendum)},
    'verified_changes':verified,'unchanged_reviewed_inputs':unchanged,
    'focused_tests':{'status':'PASS','passed':56,'exit_code':0,'wall_seconds':elapsed,
        'stdout':{'path':str(stdout),'sha256':sha(stdout)},'command':command,
        'environment':{key:env[key] for key in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','PYTHONDONTWRITEBYTECODE','PYTHONPATH')}},
    'blocking_kernel_findings':[],'source_or_test_edits_performed':False,'gpu_work_performed':False,
    'production_h_data_accepted':False,'production_results_generated':False,
    'remaining_blocker':'Production H data freeze is separate and was not accepted by this formatting review',
    'supersession_policy':'Preserves original independent review; only the two current coverage file identities are updated here',
    'verification_script':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)}}
target=R/'reports/H_kernel_independent_review_format_addendum.json'
target.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
assert json.loads(target.read_text())==report
print(run.stdout,end='')
print(json.dumps({'status':'PASS','report':str(target),'sha256':sha(target),
    'verified_original_backups':2,'exact_EOF_only_and_AST_equal':True,'focused_tests_passed':56,
    'original_review_preserved':sha(original_report)==original_report_sha},sort_keys=True))
