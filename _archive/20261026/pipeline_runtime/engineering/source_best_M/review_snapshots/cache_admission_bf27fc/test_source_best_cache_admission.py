import importlib.util,sys,copy,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pytest
from test_source_best_cache import project
from reliability_v2 import source_best_cache as cache,source_best_analysis as analysis
P=Path(__file__).parents[1]/'tools/source_best_cache_admission.py';spec=importlib.util.spec_from_file_location('selected_m_admission_under_test',P);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

@pytest.fixture
def cached(project):
 c=cache.prepare_context(project['root'],project['admission'],project['admission_sha'],project['run']);w=project['model'].classifier.weight.detach().numpy();b=project['model'].classifier.bias.detach().numpy()
 for role in c.roles:
  h=np.arange(c.tables[role]['rows']*512,dtype=np.float32).reshape(-1,512)/100
  cache.write_role(c,role,w,b,lambda offset,h=h:iter([h[offset:]]))
 return project,c

def test_owned_role_proves_header_manifest_head_without_claiming_input_admission(cached):
 project,c=cached;role=c.roles[0];pin=m.digest(cache.role_path(c,role)/'complete.json');before=dict(c.pins)
 record=m.owned_role(c.root,c.admission_sha,c.base['parent_run_id'],role,pin,content_checksums=True)
 assert record['rows']==c.tables[role]['rows'] and record['head_sha256']==c.source['head_sha256'] and record['owned_FP32_format_verified'] is True and record['fresh_source_image_validation'] is False
 assert before==c.pins

@pytest.mark.parametrize('mutation',['marker','foreignfile','head','manifest','symlink'])
def test_owned_role_rejects_unowned_or_changed_bytes(cached,mutation,tmp_path):
 project,c=cached;role=c.roles[0];folder=cache.role_path(c,role);pin=m.digest(folder/'complete.json')
 if mutation=='marker':(folder/'complete.json').write_text('{}')
 elif mutation=='foreignfile':(folder/'unregistered').write_bytes(b'keep')
 elif mutation=='head':a=np.load(folder/'head_weight.npy');a[0,0]+=1;np.save(folder/'head_weight.npy',a)
 elif mutation=='manifest':(folder/'manifest.parquet').write_bytes(b'wrong')
 else:(folder/'chunk_000000.npy').rename(folder/'saved.npy');(folder/'chunk_000000.npy').symlink_to(folder/'saved.npy')
 with pytest.raises(ValueError):m.owned_role(c.root,c.admission_sha,c.base['parent_run_id'],role,pin,content_checksums=True)

def test_fresh_job_admission_preserves_exact_D6_and_source_prefix(cached):
 project,c=cached
 if project['route']!='D':pytest.skip('D full single-model fixture only')
 spec=analysis.job('D','clip_vit_b16',7);result=m.admit_contexts(spec,{'D':c})
 assert result['independent_role_and_source_best_head_validation_passed'] is True and len(result['cache_completion_sha256'])==6 and result['source_contexts']['D']['checkpoint_selection']=='source_best'
 assert result['scientific_metric_or_bootstrap_acceptance'] is False

@pytest.mark.parametrize('mutation',['missingrole','wrongjob','contextchanged'])
def test_admit_contexts_fails_closed(cached,mutation):
 project,c=cached
 if project['route']!='D':pytest.skip('D full single-model fixture only')
 spec=analysis.job('D','clip_vit_b16',7)
 if mutation=='missingrole':(cache.role_path(c,c.roles[-1])/'complete.json').unlink()
 elif mutation=='wrongjob':spec=analysis.job('D','clip_vit_b16',42)
 else:c.base['source_best_epoch']+=1
 with pytest.raises((ValueError,FileNotFoundError)):m.admit_contexts(spec,{'D':c})


def test_capacity_closure_rebinds_exact_current_owner_and_format_report(tmp_path,monkeypatch):
 source=tmp_path/'owner.json';source.write_text('{}');stage_ref={'path':str(source),'sha256':m.digest(source)}
 input_sha='a'*64;folder=tmp_path/'reports/M_feature_capacity_closure'/input_sha;folder.mkdir(parents=True);path=folder/'complete.json'
 fresh={'schema':m.CLOSURE_SCHEMA,'input_admission_sha256':input_sha,'owner_stage':stage_ref,'collector_code_sha256':m.digest(m.__file__),'future_scheduled_feature_write_bytes':0}
 monkeypatch.setattr(m,'_feature_capacity_report',lambda root,ref:dict(fresh))
 path.write_text(json.dumps(fresh));ref={'path':str(path),'sha256':m.digest(path)}
 assert m.validate_feature_capacity(tmp_path,ref)==0
 changed=dict(fresh,foreign_future_write_claim=True);path.write_text(json.dumps(changed));bad={'path':str(path),'sha256':m.digest(path)}
 with pytest.raises(ValueError):m.validate_feature_capacity(tmp_path,bad)
 with pytest.raises(ValueError):m.validate_feature_capacity(tmp_path,ref)
 other=tmp_path/'elsewhere.json';other.write_text(json.dumps(fresh))
 with pytest.raises(ValueError):m.validate_feature_capacity(tmp_path,{'path':str(other),'sha256':m.digest(other)})
