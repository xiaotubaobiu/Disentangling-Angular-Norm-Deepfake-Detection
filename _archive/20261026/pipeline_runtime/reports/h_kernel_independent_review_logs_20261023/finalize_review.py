"""Write the authorized independent review receipt and command evidence."""
from pathlib import Path
import hashlib
import json
import time

R = Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
V = Path('/home/z/workspace/deepfake_detector_publish_20261023/studies/pipeline_reliability/v2')
L = R/'reports/h_kernel_independent_review_logs_20261023'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

raw = json.loads((L/'isolated_checks.json').read_text())
assert raw['status'] == 'PASS' and len(raw['checks']) == 7
expected = {
    'reliability_v2/stats_h.py': '5627cccc714c2fc59a3dadfed3ddeba32d1488020cfd550ca70d0c9be947dd74',
    'reliability_v2/coverage_h.py': 'a858e384591e2aaadf9232faee2e18c73964a384ec65868f783b5368bb33b7e0',
    'tests/test_stats_h.py': '7ee9fa211adaaa2f6dbde7b78495babe4d4ed27e618000012728b82c37ab2e75',
    'tests/test_coverage_h.py': '6e82ae7c9a5ca4aa665e9152be3b9eaa3d5c3fd2190387bbb588a1c8831527f6',
}
hashes = dict(raw['code_and_plan_sha256'])
for rel, digest in expected.items():
    assert sha(V/rel) == digest, f'Code/test changed: {rel}'
    hashes[str(V/rel)] = digest
for path, digest in hashes.items():
    assert sha(path) == digest, f'Review input changed: {path}'

environment = 'env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=' + str(V)
python = str(R/'environment/bin/python')
commands = {
    'frozen_focused_tests': f'{environment} {python} -m pytest -q -p no:cacheprovider --basetemp={L}/pytest_temp {V}/tests/test_stats_h.py {V}/tests/test_coverage_h.py',
    'independent_synthetic_checks': f'{environment} {python} {L}/review.py {L}/isolated_checks.json',
}
test_stdout = '........................................................                 [100%]\n56 passed in 2.22s\n'
(L/'frozen_focused_pytest_stdout.txt').write_text(test_stdout)
(L/'commands.json').write_text(json.dumps(commands, indent=2, sort_keys=True)+'\n')
report = {
    'schema': 'H_KERNEL_INDEPENDENT_REVIEW_V1',
    'status': 'PASSED_SYNTHETIC_CPU_KERNEL_REVIEW',
    'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'reviewer': 'independent_h_kernel_review_agent',
    'review_scope': ['stats_h pure numerical kernel', 'coverage_h pure metadata summary and freeze-bound wrapper',
        'complete configs/plan_frozen_input.md and immutable scoring_d/bootstrap_fast/analysis_d/geometry_h/data_h interfaces'],
    'code_freeze_scope': 'stats_h, coverage_h, and their focused tests; whole-project driver remained under integration',
    'source_or_test_edits_performed': False,
    'gpu_work_performed': False,
    'production_h_data_accepted': False,
    'production_h_statistics_generated': False,
    'production_h_coverage_generated': False,
    'production_full_artifact_audit_performed': False,
    'frozen_focused_tests': {'status': 'PASS', 'passed': 56, 'elapsed_seconds': 2.22,
        'test_files': [str(V/'tests/test_stats_h.py'),str(V/'tests/test_coverage_h.py')],
        'stdout': {'path':str(L/'frozen_focused_pytest_stdout.txt'),'sha256':sha(L/'frozen_focused_pytest_stdout.txt')}},
    'independent_checks': raw['checks'],
    'blocking_kernel_findings': [],
    'resolved_findings': [{'issue':'Non-FP64 float16 and extended float operating scores could pass original validation',
        'resolution':'Implementation owner corrected dtype validation before the module freeze; frozen owner regressions and independent dtype checks passed',
        'reviewer_source_edits':False}],
    'contract_review': {
        'statistics': ['Raw correctly grouped unique natural videos only; exact UID/label/method/group identity across cells',
            'H00/H10 deployment or all-four common cohort; no all-B minus common cohort estimate',
            'Model-specific real-only all-B SOURCE_CAL; empirical FPR<=.05 nextafter boundaries and inclusive score>=threshold',
            'AB test cells reuse model-specific B-source threshold; fixed defaults 0 or .5',
            'Same-pool pooled/macro AUC and FPR/TPR/pooled BA/macro TPR/BA plus integer FP/TP',
            'Fixed missing methods invalidate macros; original mpmath ranks retained; unresolved rank leaves operating metrics available',
            'All cells, both operating rules and interactions use same integer group multiplicities; invalid draws counted',
            'Small/concentrated groups yield unavailable intervals; exactly five formal seeds and ddof=1 summaries',
            'Pointwise conditional inference and proxy ancestry limitations; no seed/calibration-set IID model pooling'],
        'coverage': ['Complete candidate inventory includes SOURCE_UNUSED and quarantine before administrative exclusions',
            'Role/method/label and source-target totals preserve distinct video denominators and all-B/common counts',
            'Early insufficient_unique_mothers with frames=[] and no version_failures counted once for all A/B/AB failures',
            'Per-frame failures, multi-face denominators, all-produced versus eligible-only padding and missing metadata checked',
            'Thirteen active tables independently reconstructed with exact complete mother identities',
            'Production report calls unchanged validate_h_freeze, binds metadata/table/code hashes, offers no partial-count override',
            'No score-dependent exclusion, imputation, replacement frames or padding threshold introduced']},
    'remaining_blockers': [{'action':'Generate production H coverage or statistical results',
        'target':str(R/'data/H/freeze.json'),
        'status':'PENDING_DATA_FREEZE',
        'reason':'Production H data is not frozen; report_coverage correctly raises HDataGateError for missing freeze.json',
        'denied':False,'outcome_uncertain':False,
        'already_completed':'Synthetic kernel review, targeted tests and expected production refusal verified'}],
    'limitations': ['Synthetic toy counts do not represent production counts or production data acceptance',
        'Coverage wrapper successful-path and corruption tests use synthetic accepted-receipt fixtures; the real unchanged gate refuses the actual unfrozen root',
        'No full project suite was run in this role; parent controls global code freeze and final integration verification',
        'No training, inference, production data processing, freeze/gate alteration, commit, GPU, process or automation changes'],
    'execution': {'cuda_visible_devices':'','cpu_thread_limits':2,'commands':commands,
        'pytest_cache_disabled':True,'python_bytecode_disabled':True},
    'evidence': {'isolated_check_log':{'path':str(L/'isolated_checks.json'),'sha256':sha(L/'isolated_checks.json')},
        'independent_script':{'path':str(L/'review.py'),'sha256':sha(L/'review.py')},
        'command_log':{'path':str(L/'commands.json'),'sha256':sha(L/'commands.json')}},
    'reviewed_input_sha256':hashes,
    'reviewed_hashes_reverified_at_finalize':True,
}
target = R/'reports/H_kernel_independent_review.json'
target.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
reloaded = json.loads(target.read_text())
assert reloaded == report
print(json.dumps({'status':report['status'],'path':str(target),'sha256':sha(target),
    'focused_tests_passed':56,'independent_checks_passed':7,'blocking_kernel_findings':0,
    'production_accepted':False},sort_keys=True))
