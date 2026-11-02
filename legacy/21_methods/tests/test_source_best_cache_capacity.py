from pathlib import Path
import numpy as np,pytest
from test_source_best_cache import project,sha
from reliability_v2 import source_best_cache as m

def test_nonempty_role_lock_refused_without_truncation(project):
 e=project;c=e['context'];role=c.roles[0];folder=m.role_path(c,role);folder.mkdir(parents=True);p=folder/'write.lock';p.write_bytes(b'x'*(2*1024**2));before=sha(p)
 with pytest.raises(ValueError):m.write_role(c,role,e['model'].classifier.weight.detach().numpy(),e['model'].classifier.bias.detach().numpy(),lambda start:[np.ones((len(e['frames'][role]),512),np.float32)])
 assert sha(p)==before and not (folder/'complete.json').exists()

def test_nonempty_model_lock_refused_without_truncation(project,monkeypatch):
 e=project;c=e['context'];base=e['root']/'features/source_best_M'/e['admission_sha']/e['run'];base.mkdir(parents=True);p=base/'extract.lock';p.write_bytes(b'x'*(2*1024**2));before=sha(p);monkeypatch.setattr(m,'_permit',lambda *a:{})
 with pytest.raises(ValueError):m.extract(e['root'],e['admission'],e['admission_sha'],e['run'],permit_path='synthetic-only',permit_sha256='0'*64,device='cpu')
 assert sha(p)==before

def test_model_unknown_bytes_refuse_namespace_and_preserve(project):
 e=project;c=e['context'];base=e['root']/'features/source_best_M'/e['admission_sha']/e['run'];base.mkdir(parents=True);p=base/'unregistered_evidence.bin';p.write_bytes(b'x'*(2*1024**2));before=sha(p)
 with pytest.raises(ValueError):m.role_path(c,c.roles[0])
 assert sha(p)==before

def test_unselected_model_subtree_refuses_preserves(project):
 e=project;c=e['context'];base=e['root']/'features/source_best_M'/e['admission_sha']/e['run'];p=base/('dev' if e['route']=='D' else 'B/EXTRA');p.mkdir(parents=True);evidence=p/'preserve.txt';evidence.write_text('foreign evidence')
 with pytest.raises(ValueError):m.role_path(c,c.roles[0])
 assert evidence.read_text()=='foreign evidence'
