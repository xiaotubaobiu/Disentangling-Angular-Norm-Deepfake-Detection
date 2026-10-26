"""One finite original primary-H admission, reusing qualified shared slot code."""
from pathlib import Path
import os,json,hashlib,sys,runpy,importlib.util,shutil
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023');F=R/'qualification/H_parallel_fresh_validation'
PINS={'cpu_primary_analysis.py':'c081c6ffeb14dc6bb0219851fd3fd11df25b28d2eed26ac4b7e06ef1e6e4c6ce','cpu_ledger_admission.py':'bd202e85a4990f9246b96a7081815d07cf364041e5d3498f49bfe41d8779b213','parallel_source.py':'a6133c824913ab3bf361095b415eefc9ef75be39ab32fb49ab62e2656dc2624a','cpu_ledger_adapter.py':'c6b98535e5a28123baae595ab1f4b208f1feab43ff4018e740be94366d044287'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name,pin in PINS.items():
 if sha(F/name)!=pin:raise ValueError('qualified original CPU wrapper dependency drift')
q=R/'reports/H_parallel_CPU_actual_qualification_clip_vit_b16_20261025T0830.json'
if sha(q)!='0ae309b3366de860cbd77f539c24a05aae3ec1180dbbd84bbbff95ed3111d7f7':raise ValueError('actual CLIP CPU qualification drift')
output=R/'reports/H_clip_vit_b16_seed123_independent_7role_parallel_candidate_20261025T0645.json'
if output.exists():raise FileExistsError(output)
for history in ('B','AB'):
 for role in ('B/SOURCE_CAL','B/SOURCE_TEST','B/TARGET_TEST','common/B/SOURCE_TEST','common/B/TARGET_TEST','common/AB/SOURCE_TEST','common/AB/TARGET_TEST'):
  p=R/'features/H'/f'H_formal_clip_vit_b16_{history}_seed123'/role/'complete.json'
  if p.is_symlink() or not p.is_file():raise ValueError('complete 14 role cache prerequisites missing')
b=R/'reports/full_capacity_check_pre_H_cache_20261025T0555.json'
if sha(b)!='3ce2d435d51895b9cfe53ac2f4bde6df12db244a7a2365c89a3f31b418b502cd' or shutil.disk_usage(R).free<json.loads(b.read_text())['normal_required_bytes']:raise ValueError('full primary remaining space gate failed')
spec=importlib.util.spec_from_file_location('_qualified_original_primary_cpu_slot',F/'cpu_primary_analysis.py');entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
with entry.cpu_slot(1):
 print('ACQUIRED_EXISTING_PRIMARY_SHARED_CPU_SLOT_1_ONE_FINITE_ADMISSION',flush=True)
 sys.argv=[str(F/'cpu_ledger_admission.py'),'--execute','--backbone','clip_vit_b16','--seed','123']
 runpy.run_path(str(F/'cpu_ledger_admission.py'),run_name='__main__')
