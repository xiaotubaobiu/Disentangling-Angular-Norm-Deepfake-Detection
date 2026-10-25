import copy,hashlib,json,os
from pathlib import Path
import numpy as np,pandas as pd,pytest
from reliability_v2 import analysis_d as io,source_best_sink as m
from test_source_best_cache import project

@pytest.fixture
def case(tmp_path):
 identity=dict(schema='M_STATISTIC_STATIC_IDENTITY_V1',scope='SYNTHETIC_TRANSPORT_UNIT_NO_PRODUCTION_AUTHORITY',checkpoint_selection='source_best',inputs={'model':'a'*64},CAL_repeats=200,TEST_repeats=2000)
 token=hashlib.sha256(m.typed(identity)).hexdigest();out=tmp_path/'analysis/source_best_M'/token/'D_formal_clip_vit_b16_seed7'
 limits=dict(max_total_bytes=32*1024**2,max_temporary_bytes=16*1024**2,max_artifact_bytes=4*1024**2,max_json_bytes=4*1024**2,max_json_array_bytes=4*1024**2,max_json_cells=1000000,max_bundle_members=64,max_table_rows=1000000,max_decoded_bytes=8*1024**2)
 return tmp_path,out,identity,limits

def sink(case,inventory):return m.CompactSink(*case[:3],inventory=inventory,limits=case[3])
def test_all_artifact_kinds_read_back_and_complete(case):
 table=pd.DataFrame({'a':[1,2],'value':[.5,np.nan]});array=np.array([0.,1.25],np.float64);payload={'draws':[dict(repeat=i,group_draw_counts={'zero':0,'g':1},estimate=None if i%2 else -.25) for i in range(2000)],'repeats':2000}
 inventory={'metrics.csv':'csv','rows.parquet':'parquet','training_radii.npy':'array','conditional_bootstrap.json':'json'}
 with sink(case,inventory) as s:
  pd.testing.assert_frame_equal(s('metrics.csv','csv',table),table)
  pd.testing.assert_frame_equal(s('rows.parquet','parquet',table),table)
  assert np.array_equal(s('training_radii.npy','array',array),array)
  assert s('conditional_bootstrap.json','json',payload)==payload
  receipt=s.finish();assert receipt['status']=='ALL_DECLARED_ARTIFACTS_COMMITTED_REQUIRES_SCIENTIFIC_ADMISSION'
 with sink(case,inventory) as s:assert s.completed and s.finish()==receipt

def test_only_declared_kinds_and_complete_inventory(case):
 with sink(case,{'a.json':'json','b.csv':'csv'}) as s:
  with pytest.raises(ValueError):s('../escape.json','json',{})
  with pytest.raises(ValueError):s('a.json','csv',pd.DataFrame({'x':[1]}))
  s('a.json','json',{'ok':True})
  with pytest.raises(ValueError):s.finish()
 assert not (case[1]/'complete.json').exists()

def test_committed_resumption_preserves_values_and_mtime(case):
 value={'x':-1,'draws':[dict(group_draw_counts={'g':1}) for _ in range(4)]};inv={'a.json':'json'}
 with sink(case,inv) as s:s('a.json','json',value);s.finish()
 files={str(p):p.stat().st_mtime_ns for p in case[1].rglob('*') if p.is_file()}
 with sink(case,inv) as s:
  assert s('a.json','json',copy.deepcopy(value))==value
  s.finish()
 assert files=={str(p):p.stat().st_mtime_ns for p in case[1].rglob('*') if p.is_file()}

def test_changed_scientific_value_refuses_without_overwrite(case):
 with sink(case,{'a.json':'json'}) as s:s('a.json','json',{'estimate':.5,'preparation_seconds':1.})
 before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in case[1].rglob('*') if p.is_file()}
 with sink(case,{'a.json':'json'}) as s:
  with pytest.raises(ValueError):s('a.json','json',{'estimate':.6,'preparation_seconds':1.})
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in before.items())

def test_bootstrap_resume_retains_original_timing_fields_only(case):
 name='target_full/normal_bootstrap_raw_mean_logit.json';one={'preparation_seconds':1.,'bootstrap_seconds':2.,'draws':[dict(group_draw_counts={'g':1},estimate=-.2)],'repeats':2000};two=dict(one,preparation_seconds=9.,bootstrap_seconds=12.)
 with sink(case,{name:'json'}) as s:s(name,'json',one)
 with sink(case,{name:'json'}) as s:
  assert s(name,'json',two)==one
  bad=copy.deepcopy(two);bad['draws'][0]['estimate']=.9
  with pytest.raises(ValueError):s(name,'json',bad)

def test_exclusive_writer_and_namespace_symlink_refusal(case,tmp_path):
 with sink(case,{'a.json':'json'}):
  with pytest.raises(BlockingIOError):
   with sink(case,{'a.json':'json'}):pass
 other=tmp_path/'other';other.mkdir();(case[0]/'analysis').rename(case[0]/'old_analysis');(case[0]/'analysis').symlink_to(other)
 with pytest.raises(ValueError):
  with sink(case,{'a.json':'json'}):pass
 assert list(other.iterdir())==[]

def test_corrupt_existing_payload_refused(case):
 with sink(case,{'a.csv':'csv'}) as s:s('a.csv','csv',pd.DataFrame({'x':[1]}));s.finish()
 p=case[1]/'artifacts/a.csv';p.write_bytes(b'changed')
 with pytest.raises(ValueError):
  with sink(case,{'a.csv':'csv'}):pass

def test_persistent_quota_refuses_before_publication_preserves_staging(case):
 case[3]['max_total_bytes']=1024**2
 with sink(case,{'training_radii.npy':'array'}) as s:
  with pytest.raises(ValueError):s('training_radii.npy','array',np.ones(300000,np.float64))
 assert not (case[1]/'artifacts/training_radii.npy').exists() and not (case[1]/'complete.json').exists()
 assert any(p.is_file() and p.stat().st_size>1024**2 for p in (case[1]/'.staging').rglob('*'))

def test_artifact_cap_cannot_be_bypassed_by_numpy_fileno(case):
 case[3]['max_artifact_bytes']=1024
 with sink(case,{'training_radii.npy':'array'}) as s:
  with pytest.raises((ValueError,OSError)):s('training_radii.npy','array',np.ones(10000,np.float64))
 assert not (case[1]/'artifacts/training_radii.npy').exists()
 assert all(p.stat().st_size<=1024 for p in (case[1]/'.staging').rglob('*') if p.is_file())

def test_code_identity_or_limits_change_refuses_mutation(case):
 with sink(case,{'a.json':'json'}) as s:
  s.identity['scope']='changed'
  with pytest.raises(ValueError):s('a.json','json',{})


def test_exact_driver_artifact_inventories():
 assert len(m.declared_inventory('D'))==60 and len(m.declared_inventory('H'))==37
 with pytest.raises(ValueError):m.declared_inventory('DEV')

def test_H_driver_roundtrip_into_compact_sink(case,tmp_path,monkeypatch):
 from test_source_best_stats import h_data
 from reliability_v2 import source_best_stats
 data,selections=h_data(tmp_path,monkeypatch)
 with sink(case,m.declared_inventory('H')) as out:
  result=source_best_stats.h_driver(data,selections,out);receipt=out.finish()
 assert result['update_matched_primary_claim'] is False and len(receipt['receipts'])==37
 with sink(case,m.declared_inventory('H')) as out:
  source_best_stats.h_driver(data,selections,out);assert out.finish()==receipt

def test_D_driver_roundtrip_into_compact_sink(project,case):
 from test_source_best_stats import d_data
 from reliability_v2 import source_best_stats
 data,selected,freeze=d_data(project)
 case[3]['max_total_bytes']=256*1024**2
 case[3]['max_temporary_bytes']=256*1024**2
 case[3]['max_json_bytes']=64*1024**2
 case[3]['max_json_array_bytes']=32*1024**2
 with sink(case,m.declared_inventory('D')) as out:
  source_best_stats.d_driver(data,selected,freeze,out);receipt=out.finish()
 assert len(receipt['receipts'])==60

def test_rq4_bootstrap_timing_resume_known_actual_filename(case):
 name='rq4/raw_mean_logit/primary_n20_vs_source_bootstrap.json';one={'preparation_seconds':1.,'bootstrap_seconds':2.,'draws':[{'x':1}]};two=dict(one,preparation_seconds=3.,bootstrap_seconds=9.)
 with sink(case,{name:'json'}) as out:out(name,'json',one)
 with sink(case,{name:'json'}) as out:assert out(name,'json',two)==one


@pytest.mark.parametrize('kind',['csv','parquet'])
def test_decoded_row_quota_rejects_before_commit(case,kind):
 case[3]['max_table_rows']=1;name='rows.'+kind
 with sink(case,{name:kind}) as out:
  with pytest.raises(ValueError):out(name,kind,pd.DataFrame({'x':[1,2,3]}))
 assert not (case[1]/'artifacts'/name).exists()
 assert not any((case[1]/'.receipts').iterdir())

def test_scores_row_quota_rejects_before_commit(case):
 from test_source_best_stats import cache_fixture
 from reliability_v2 import source_best_stats,source_best_cache
 full=cache_fixture();native={h:{r:v for r,v in rs.items() if r in source_best_cache.ROLES['H']} for h,rs in full.items()}
 q=source_best_stats.native_H_pools(native)['SOURCE_TEST']['H00'];assert len(q)>1
 case[3]['max_table_rows']=1
 with sink(case,{'scores.csv':'scores'}) as out:
  with pytest.raises(ValueError):out('scores.csv','scores',q)
 assert not (case[1]/'artifacts/scores.csv').exists()

@pytest.mark.parametrize('kind',['csv','parquet'])
def test_decoded_byte_quota_rejects_before_commit(case,kind):
 case[3]['max_decoded_bytes']=512;name='rows.'+kind
 with sink(case,{name:kind}) as out:
  with pytest.raises(ValueError):out(name,kind,pd.DataFrame({'x':['same'*100]*500}))
 assert not (case[1]/'artifacts'/name).exists()


@pytest.mark.parametrize('inventory',[{'a.json':'json','a.json.bundle':'csv'},{'a.csv':'csv','a.csv/b.csv':'csv'}])
def test_mapped_inventory_collision_rejected(case,inventory):
 with pytest.raises(ValueError):sink(case,inventory)

def test_temporary_quota_includes_allocated_block_rounding(case):
 case[3]['max_artifact_bytes']=1500;case[3]['max_temporary_bytes']=1500
 with pytest.raises(ValueError):
  with sink(case,{'a.csv':'csv'}) as out:out('a.csv','csv',pd.DataFrame({'x':[1]}))
 assert not any((case[1]/'.staging').iterdir())

def test_exact_bootstrap_input_still_rejects_invalid_timing(case):
 name='conditional_bootstrap.json';one={'preparation_seconds':-1.,'draws':[]}
 with sink(case,{name:'json'}) as out:
  with pytest.raises(ValueError):out(name,'json',one)
 assert not (case[1]/'artifacts/conditional_bootstrap.json.bundle').exists()


@pytest.mark.parametrize('quota',[16384,24576,28672,32768,36864,65536])
def test_exact_completion_allocation_quota_is_resumable(case,quota):
 case[3]['max_total_bytes']=quota
 try:
  with sink(case,{'r.npy':'array'}) as out:
   out('r.npy','array',np.ones(1,np.float64));out.finish()
 except ValueError:
  # Directory blocks now count too: insufficient quotas must reject BEFORE
  # committing the artifact, never strand a valid receipt without completion.
  assert not (case[1]/'artifacts/r.npy').exists()
 else:
  with sink(case,{'r.npy':'array'}) as out:assert out.completed
  assert m._usage(case[1])-m._usage(case[1]/'.staging')<=quota

def test_usage_includes_directory_allocation(case):
 with sink(case,{'a/b/c/r.npy':'array'}) as out:
  expected=sum(max(p.stat().st_size,p.stat().st_blocks*512) for p in [out.output,*out.output.rglob('*')])
  assert m._usage(out.output)==expected
