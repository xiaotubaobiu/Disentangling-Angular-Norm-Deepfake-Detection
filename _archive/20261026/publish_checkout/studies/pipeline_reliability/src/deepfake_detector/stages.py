"""Hash-bound stage identities and completion records."""
from pathlib import Path
import hashlib,json
from .manifest import digest,save_json
from .orchestration import code_hash


def config_identity(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()


def stage_identity(stage,config,project,inputs):
    return {'stage':stage,'config_sha256':config_identity(config),'code_sha256':code_hash(project),
            'inputs_sha256':{name:digest(path) for name,path in sorted(inputs.items())}}


def _output(root,name):
    name=Path(name);root=Path(root).resolve();path=(root/name).resolve()
    if name.is_absolute() or not path.is_relative_to(root):raise ValueError('Unsafe stage output path')
    return path


def finish_stage(root,identity,outputs):
    root=Path(root);files={str(name):digest(_output(root,name)) for name in outputs}
    if not files:raise ValueError('A completed stage needs saved evidence')
    marker={'status':'COMPLETE','identity':identity,'outputs_sha256':files}
    save_json(root/'complete.json',marker);return marker


def completed_stage(root,identity):
    root=Path(root);marker=root/'complete.json'
    if not marker.exists():return False
    old=json.loads(marker.read_text())
    if old.get('status')!='COMPLETE' or old.get('identity')!=identity:raise ValueError('Completed stage identity mismatch')
    if not old.get('outputs_sha256'):raise ValueError('Completed stage has no output evidence')
    for name,sha in old['outputs_sha256'].items():
        p=_output(root,name)
        if not p.is_file() or digest(p)!=sha:raise ValueError('Completed stage output hash mismatch')
    return True
