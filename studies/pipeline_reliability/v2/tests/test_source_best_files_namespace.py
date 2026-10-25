from pathlib import Path
from types import SimpleNamespace
import importlib,shutil,sys
import pytest,torch
from reliability_v2 import source_best_files as f

def copied(tmp_path):
 root=tmp_path/'qualified_primary';(root/'reliability_v2').mkdir(parents=True);original=Path(f.__file__).parent.parent
 for rel in f.BASE_SOURCE_PINS:shutil.copyfile(original/rel,root/rel)
 return root

def test_canonical_namespace_preserves_exact_qualified_file_locations(tmp_path):
 root=copied(tmp_path);worker=f.canonical_primary_worker(root)
 assert Path(worker.__file__).resolve()==root/'reliability_v2/worker_h.py'
 data=importlib.import_module(worker.__package__+'.data_h');geometry=importlib.import_module(worker.__package__+'.geometry_h')
 assert Path(data.__file__).resolve()==root/'reliability_v2/data_h.py'
 assert Path(geometry.__file__).resolve()==root/'reliability_v2/geometry_h.py'
 assert worker.h_code_hash()==f.worker_h.h_code_hash() and not torch.cuda.is_initialized()

def test_canonical_source_root_rejects_unqualified_code_before_import(tmp_path):
 root=copied(tmp_path);(root/'reliability_v2/cache_h.py').write_text('arbitrary_changed_code')
 with pytest.raises(ValueError):f.canonical_primary_worker(root)

def test_foreign_preloaded_private_package_is_rejected(tmp_path,monkeypatch):
 root=copied(tmp_path);alias=f.canonical_primary_alias(root);monkeypatch.setitem(sys.modules,alias,SimpleNamespace(__file__='/foreign/__init__.py'))
 with pytest.raises(ValueError):f.canonical_primary_worker(root)

def test_matching_preloaded_package_must_not_redirect_search_and_execute_foreign_worker(tmp_path,monkeypatch):
 import types
 root=copied(tmp_path);alias=f.canonical_primary_alias(root);foreign=tmp_path/'foreign';foreign.mkdir();sentinel=tmp_path/'foreign_executed'
 (foreign/'worker_h.py').write_text('from pathlib import Path\nPath('+repr(str(sentinel))+').write_text("EXECUTED")\n')
 package=types.ModuleType(alias);package.__file__=str(root/'reliability_v2/__init__.py');package._M_primary_root=str(root);package.__path__=[str(foreign)];monkeypatch.setitem(sys.modules,alias,package)
 with pytest.raises(ValueError):f.canonical_primary_worker(root)
 assert not sentinel.exists()

def test_owned_package_search_path_cannot_change(tmp_path,monkeypatch):
 root=copied(tmp_path);worker=f.canonical_primary_worker(root);package=sys.modules[worker.__package__];monkeypatch.setattr(package,'__path__',[str(tmp_path/'foreign')])
 with pytest.raises(ValueError):f.canonical_primary_worker(root)

def test_owned_package_spec_search_path_cannot_change(tmp_path,monkeypatch):
 root=copied(tmp_path);worker=f.canonical_primary_worker(root);package=sys.modules[worker.__package__];monkeypatch.setattr(package.__spec__,'submodule_search_locations',[str(tmp_path/'foreign')])
 with pytest.raises(ValueError):f.canonical_primary_worker(root)
