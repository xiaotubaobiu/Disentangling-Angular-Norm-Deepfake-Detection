from pathlib import Path
import tempfile,importlib.util,sys,json,hashlib,time,resource
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023');W=R/'engineering/source_best_M/code/studies/pipeline_reliability/v2'
sys.path[:0]=[str(W),str(W/'tests')]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths=[W/'tools/m_source_best_d_numeric.py',W/'tests/test_m_source_best_d_numeric.py',W/'tools/m_source_best_result_audit.py']
pins={str(p):sha(p) for p in paths};start=time.monotonic()
a=load('integration_m_consumer',paths[2]);dtest=load('integration_m_D_fixture',paths[1]);from reliability_v2 import source_best_sink as sink
with tempfile.TemporaryDirectory(prefix='detector_M_D60_private_') as tmp:
 root=Path(tmp)
 class Factory:
  def mktemp(self,name):
   p=root/name;p.mkdir();return p
 print('BEGIN_SYNTHETIC_D60_FIXTURE',flush=True)
 values=dtest.synthetic.__wrapped__(Factory());inventory=sink.declared_inventory('D')
 assert set(values)==set(inventory)
 ident={'schema':'PRIVATE_SYNTHETIC_D60_SERIALIZATION_ONLY','purpose':'No actual M source cache, numerical/resource/scientific admission'}
 token=hashlib.sha256(sink.typed(ident)).hexdigest();output=root/'analysis/source_best_M'/token/'SYNTHETIC_D60'
 limits=dict(max_total_bytes=128*1024**2,max_temporary_bytes=128*1024**2,max_artifact_bytes=64*1024**2,max_json_bytes=64*1024**2,max_json_array_bytes=32*1024**2,max_json_cells=8*1024**2,max_bundle_members=64,max_table_rows=100000,max_decoded_bytes=64*1024**2)
 print('BEGIN_PRIVATE_PRODUCTION_SINK_SERIALIZATION',flush=True)
 with sink.CompactSink(root,output,ident,inventory=inventory,limits=limits) as emit:
  for name,kind in inventory.items():emit(name,kind,values[name])
  emit.finish()
 allocated=sum(max(p.stat().st_size,p.stat().st_blocks*512) for p in (root,*root.rglob('*')))
 print('BEGIN_PRIVATE_READ_ONLY_FULL_D60_NUMERIC',allocated,flush=True)
 reader=a.ReadOnlyArtifacts(output,sha(output/'complete.json'),route='D')
 result=a.audit_numeric(reader,{'job':{'seed':7}});reader.verify_all_decodable()
 assert result['accepted_for_selected_core'] is True and result['draws_recomputed']==28000 and result['calibration_rows_recomputed']==3208
 assert reader.decoded_names==set(inventory)
 assert {str(p):sha(p) for p in paths}==pins
 print(json.dumps(dict(schema='M_PRIVATE_D60_SERIALIZED_CORE_INTEGRATION_V1',status='PASS',source_sha256=pins,logical_assets=len(reader),draws_checked=result['draws_recomputed'],CAL_rows_checked=result['calibration_rows_recomputed'],synthetic_allocated_bytes=allocated,seconds=time.monotonic()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,actual_M_source_cache_or_shared_slot_used=False,GPU_used=False,real_statistic_canary_or_resource_qualification=False,scientific_final_acceptance=False),allow_nan=False),flush=True)
