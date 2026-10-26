"""Record implementation evidence only; never creates production data/runtime gates."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys

R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
V=Path('/home/z/workspace/deepfake_detector_publish_20261023/studies/pipeline_reliability/v2')
sys.path.insert(0,str(V))
from reliability_v2 import trainer,worker_h,queue_h


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert trainer.code_hash()=='336f3097434edf903a5ff402162e29afeca2442ca50f9c7d9883ca60136b8335'
    assert digest(R/'configs/execution.json')=='8f12b0285593b30cb4fc8e2d50f125dc7f1c9ea690f36a286e9fdca24f6df1f4'
    assert digest(R/'qualification/D_gate.json')=='c2a9a09d208d2671b7a123df7ee29e83b8abedcac08d9221d5ad26f3948700aa'
    assert digest(V/'reliability_v2/geometry_h.py')=='161ac4e837463ff16e7916ca42eeb5d13acc44aab9bcc1c397025b9426ad9532'
    full=R/'reports/H_implementation_full_suite_final.log'
    assert '199 passed in 184.99s' in full.read_text()
    post=R/'reports/H_qualification_post_format_tests.log'
    assert '22 passed' in post.read_text()
    pipeline=json.loads((R/'reports/H_pipeline_independent_review.json').read_text())
    data=json.loads((R/'reports/H_data_qualification_independent_review.json').read_text())
    assert data['blockers']==[] and data['status']=='PASS_CODE_REVIEW_ONLY'
    assert pipeline['status']=='CODE_REVIEW_COMPLETE_EXECUTION_GATES_PENDING'
    readiness=queue_h.readiness(R)
    assert not readiness['H_freeze_present'] and not readiness['H_gate_present'] and not readiness['GPU_launch_attempted']
    assert not list((R/'runs').glob('H_*'))
    files=['reliability_v2/data_h.py','reliability_v2/qualification_h.py','reliability_v2/queue_h.py','reliability_v2/worker_h.py',
        'tests/test_data_h.py','tests/test_qualification_h.py','tests/test_queue_h.py','tests/test_worker_h.py',
        'docs/h_training_queue_usage.md','docs/manuscript_preparation/verified_additional_references.bib','docs/manuscript_preparation/additional_reference_audit.json']
    evidence=['reports/H_implementation_full_suite_final.log','reports/H_worker_missing_start_red.log','reports/H_qualification_post_format_tests.log',
        'reports/H_pipeline_independent_review.json','reports/H_data_qualification_independent_review.json','reports/H_format_only_addendum.json',
        'reports/qualification_h_implementation.json','reports/qualification_h_source_geometry_integration.json','reports/data_h_gate_implementation.json']
    addenda=list((R/'reports').glob('H*format*review*.json'))
    evidence += [str(p.relative_to(R)) for p in addenda]
    report=dict(schema='H_IMPLEMENTATION_ACCEPTANCE_V1',status='IMPLEMENTED_CPU_VERIFIED_INDEPENDENTLY_REVIEWED_RUNTIME_PENDING',
        utc=datetime.now(timezone.utc).isoformat(),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=V,text=True).strip(),
        training_bundle_sha256=trainer.code_hash(),H_bundle_sha256=worker_h.h_code_hash(),source_sha256={name:digest(V/name) for name in files},
        evidence_sha256={name:digest(R/name) for name in evidence},CPU_full_suite=dict(passed=199,seconds=184.99,scope='CPU-only synthetic/unit/integration tests; no production pretrained H GPU qualification'),
        actual_source_integration='10B videos/80PNG byte+RGB/schema checked; all10 legacyABversions ineligible and excluded, not repaired/accepted',
        formatting_only_followup='Two EOF-only changes; old/newbytes and identicalAST bound in addendum;22qualifier tests repeated on finalbytes',
        readiness=readiness,production_H_run_ids=[],production_H_freeze_created=False,production_H_gate_created=False,D_config_modified=False,D_H_existing_processes_restarted=False,
        registered_counts=dict(standard_development=20,formal=30,maximum_extra_DINO_development=8,maximum_conditional_FP32_formal_reruns=30),
        capacity_policy='Initial>200GiB+50GiB; later50GiBreserve; conditionalFP32additional1.2xretained10AMPmodelbytes',
        scheduling='Hqualification/trainingwaitDgpu/cacheterminalstatusandfreshidleGPU0-3physicalUUID/compute/memory/lifetime-lockchecks;noGPU4-7',
        pending=['Finishall8431Hpixels','FreshrepairedfullcandidateauditwithSOURCE_UNUSED','Independentproductiondata/cohortreviewthenfreeze',
            'Dgpu/cachequeuecompletion','RealsourceAMP+FP32qualificationsixjobsbothhistories','H20developmentconditional8DINO+30formalconditionaluniformFP32',
            'Htargetcache/scoringandallB/common2x2analysis','FormalDanalysis/fiveseedtables/figures','FullCN/ENpapersandmasterreadback'],scientific_results_complete=False,journal_manuscripts_complete=False)
    trainer.save_json(R/'reports/H_implementation_acceptance.json',report)
    print(json.dumps({k:report[k] for k in ['status','commit','H_bundle_sha256','CPU_full_suite','readiness']}))


if __name__=='__main__':main()
