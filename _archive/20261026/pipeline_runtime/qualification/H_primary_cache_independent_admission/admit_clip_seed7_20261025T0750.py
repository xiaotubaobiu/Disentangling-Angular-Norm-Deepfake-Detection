from pathlib import Path
import json,time,hashlib,datetime,os
from reliability_v2 import analysis_h
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
V=Path('/home/z/workspace/deepfake_detector_publish_20261023/studies/pipeline_reliability/v2')
manifest_path=R/'qualification/H_primary_cache_continuation/qualified_source_manifest.json'
manifest=json.loads(manifest_path.read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
print('MANIFEST_KEYS',list(manifest),flush=True)
assert os.environ['CUDA_VISIBLE_DEVICES']==''
report=R/'reports/H_clip_seed7_independent_7role_admission_20261025T0750.json'
assert not report.exists()
# The canonical frozen source inventory is independently SHA pinned here.
assert digest(manifest_path)=='977431237b901d5ac88172f98bc2591d3cd7992c35be50546cda34ad2276a747'
pins=manifest.get('source_sha256',manifest.get('files'))
if pins is None:raise ValueError('unknown qualified source manifest schema')
for name,expected in pins.items():
 path=Path(name) if Path(name).is_absolute() else V/name
 assert digest(path)==(expected['sha256'] if isinstance(expected,dict) else expected)
start=time.monotonic();print('BEGIN_ORIGINAL_FULL_14_ROLE_PAIR_CONTEXT',datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
root,config,freeze,runs,identity=analysis_h._pair_context(R/'configs/H_execution.json','clip_vit_b16',7)
for name,expected in pins.items():
 path=Path(name) if Path(name).is_absolute() else V/name
 assert digest(path)==(expected['sha256'] if isinstance(expected,dict) else expected)
assert digest(manifest_path)=='977431237b901d5ac88172f98bc2591d3cd7992c35be50546cda34ad2276a747'
assert len(identity['cache_completion_sha256'])==14
payload=dict(status='PASS_ORIGINAL_FULL_14_ROLE_PAIR_CONTEXT_INDEPENDENT_OF_CACHE_WORKERS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),models=2,roles=14,backbone='clip_vit_b16',seed=7,identity=identity,elapsed_seconds=time.monotonic()-start,analysis_executed=False,GPU_launch_attempted=False,qualified_source_manifest_sha256=digest(manifest_path))
with report.open('x') as stream:json.dump(payload,stream,ensure_ascii=False,indent=2);stream.write('\n')
print('PASS_INDEPENDENT_7ROLE_MODELS',runs,'elapsed_seconds',payload['elapsed_seconds'],flush=True)
