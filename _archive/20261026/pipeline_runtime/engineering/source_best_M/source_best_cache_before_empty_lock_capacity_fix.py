"""Selected-M source-best FP32 candidate caches, separate from primary caches.

No production permit is issued by this module. The accepted primary validators
remain unchanged; their functions run with private copied globals. A cache marker
alone never authorizes a production statistic or target inference.
"""
from pathlib import Path
from types import FunctionType, SimpleNamespace
import contextlib, copy, fcntl, hashlib, json, os, re, shutil, stat, sys, tempfile
import numpy as np
import pandas as pd
import torch
from . import cache_d, source_best_files as sf, trainer

SCHEMA = 'M_SOURCE_BEST_RAW_FP32_CACHE_V1'
ROLES = {'D': ('train','source_cal','source_test','target_full','target_cal','target_test'),
         'H': ('B/SOURCE_CAL','B/SOURCE_TEST','B/TARGET_TEST')}
INPUT_SCHEMA = 'M_SOURCE_BEST_CACHE_INPUT_ADMISSION_V1'
PERMIT_SCHEMA = 'M_SOURCE_BEST_PRODUCTION_CACHE_PERMIT_V1'
CHUNK_ROWS = 4096
ROLE_METADATA_BYTES = 1024**2
require = sf.require

def _typed(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()

def _json(path):
    path=Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=32*1024**2,'bounded regular metadata required')
    def pairs(items):
        out={}
        for k,v in items:
            require(k not in out,'duplicate metadata key');out[k]=v
        return out
    return json.loads(path.read_text(),object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite metadata')))

def _inside(path,root,*,file=False):
    path=Path(path).absolute();root=Path(root).absolute()
    require(path.is_relative_to(root) and path.resolve().is_relative_to(root),'canonical path escape')
    for q in (path,*path.parents):
        require(not q.is_symlink(),'symlink in cache/input namespace')
        if q==root:break
    if file:require(path.is_file(),'regular input file required:'+str(path))
    return path


@contextlib.contextmanager
def _lock(path,root):
    # Walk using directory fds and O_NOFOLLOW, before mkdir/open can follow a
    # symlink. The lock fd and its inode remain owned for this full operation.
    path=_inside(path,root);root=Path(root);parts=path.relative_to(root).parts
    directory=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    stream=None
    try:
        for name in parts[:-1]:
            try:os.mkdir(name,dir_fd=directory)
            except FileExistsError:pass
            child=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=directory)
            os.close(directory);directory=child
        fd=os.open(parts[-1],os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600,dir_fd=directory)
        stream=os.fdopen(fd,'a+')
        info=os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and info.st_nlink==1,'regular private lock inode required')
        _inside(path,root,file=True)
        current=os.stat(parts[-1],dir_fd=directory,follow_symlinks=False)
        require((current.st_dev,current.st_ino)==(info.st_dev,info.st_ino),'lock namespace/inode changed')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield stream
        finally:fcntl.flock(fd,fcntl.LOCK_UN)
    finally:
        if stream is not None:stream.close()
        os.close(directory)

def _inventory(context,role):
    folder=role_path(context,role)
    if not folder.exists():return
    require(folder.is_dir(),'candidate role directory required')
    count=(context.tables[role]['rows']+CHUNK_ROWS-1)//CHUNK_ROWS
    ordinary={'identity.json','manifest.parquet','head_weight.npy','head_bias.npy','complete.json','write.lock'}
    metadata_bytes=0
    for path in folder.iterdir():
        _inside(path,context.root,file=True)
        require(stat.S_ISREG(path.lstat().st_mode),'regular candidate artifact required')
        match=re.fullmatch(r'chunk_([0-9]{6})\.(json|npy)',path.name)
        if match:
            number=int(match[1]);require(number<count,'chunk outside frozen role capacity')
            if match[2]=='npy':
                rows=min(CHUNK_ROWS,context.tables[role]['rows']-number*CHUNK_ROWS)
                _npy(path,(rows,context.base['dimension']))
        else:require(path.name in ordinary,'unregistered candidate/orphan artifact; preserve evidence and resolve before resume')
        if path.suffix=='.json':metadata_bytes+=path.stat().st_size
    require(metadata_bytes<=ROLE_METADATA_BYTES,'cumulative role JSON metadata capacity exceeded')

def _pin(path,pin,bound,root=None):
    path=Path(path).absolute()
    if root is not None:_inside(path,root,file=True)
    require(not path.is_symlink() and path.is_file(),'regular pinned file required')
    require(sf.digest(path)==pin,'input/code hash drift:'+str(path));bound[str(path)]=pin
    return path

def _primary(route,config_path,run):
    if route=='D':legacy=cache_d
    else:
        source,_,_=sf._H_primary_source(config_path,run)
        worker=sf.canonical_primary_worker(source)
        legacy=sys.modules[worker.__package__+'.cache_h']
    base,config,task=legacy._context(config_path,run)
    return legacy,base,config,task

def _build_model(backbone,weight_path):
    from .models import build_model
    return build_model(backbone,weight_path)

class BoundContext:
    def __init__(self,root,admission_path,admission_sha,source,legacy,parent,config,task,tables,pins):
        self.root=Path(root);self.admission_path=Path(admission_path);self.admission_sha=admission_sha
        self.source=copy.deepcopy(source);self.legacy=legacy;self.parent=copy.deepcopy(parent)
        self.config=copy.deepcopy(config);self.task=copy.deepcopy(task);self.tables=copy.deepcopy(tables);self.pins=copy.deepcopy(pins)
        self.roles=tuple(ROLES[source['route']])
        self.base=dict(parent,schema=SCHEMA,artifact_root=str(root),run_id=source['parent_run_id'],parent_run_id=source['parent_run_id'],
            endpoint_id=source['endpoint_id'],route=source['route'],dimension=source['dimension'],checkpoint_selection='source_best',
            source_best_epoch=source['best']['epoch'],source_best_prefix=source['best'],checkpoint_path=source['checkpoint_path'],
            checkpoint_sha256=source['checkpoint_sha256'],original_head_sha256=source['head_sha256'],
            parent_final_checkpoint_path=parent['checkpoint_path'],parent_final_checkpoint_sha256=parent['checkpoint_sha256'],
            cache_code_sha256=sf.digest(__file__),input_admission_path=str(admission_path),input_admission_sha256=admission_sha,
            source_file_admission_sha256=_json(admission_path)['source_file_admission']['sha256'],
            target_inference_authorized=False,cache_kind='SELECTED_M_CANDIDATE_ONLY',production_qualification_required=True)
        self._seal=hashlib.sha256(self._payload()).hexdigest()
    def _payload(self):
        return _typed(dict(base=self.base,source=self.source,parent=self.parent,config=self.config,task=self.task,tables=self.tables,pins=self.pins,roles=self.roles,
                           root=str(self.root),admission=str(self.admission_path),admission_sha=self.admission_sha,legacy_file=self.legacy.__file__))
    def check(self,*,full=False):
        require(hashlib.sha256(self._payload()).hexdigest()==self._seal,'bound context mutated')
        for path,pin in self.pins.items():_pin(path,pin,{})
        if full:
            module,parent,config,task=_primary(self.source['route'],self.base['config_path'],self.base['parent_run_id'])
            require(module is self.legacy and _typed(parent)==_typed(self.parent) and _typed(config)==_typed(self.config) and _typed(task)==_typed(self.task),'primary live eligibility/context changed')
        return self

def prepare_context(root,input_admission_path,admission_sha,parent_run_id):
    root=Path(root).absolute();require(root.resolve()==root,'canonical artifact root required')
    require(isinstance(parent_run_id,str) and re.fullmatch(r'[DH]_formal_[A-Za-z0-9_]+',parent_run_id),'registered formal endpoint ID required')
    pins={};ap=_pin(input_admission_path,admission_sha,pins,root);admission=_json(ap)
    require(admission.get('schema')==INPUT_SCHEMA and admission.get('artifact_root')==str(root),'M cache input capsule required')
    require(admission.get('source_input_scope')=='SELECTED_M_D6_H3_FIVE_SEEDS' and admission.get('target_inference_authorized') is False,'selected source-only M scope required')
    require(admission.get('cache_code_sha256')==sf.digest(__file__),'cache implementation admission changed')
    ref=admission['source_file_admission'];source=sf.validate_source_file(root,ref['path'],ref['sha256'],parent_run_id)
    require(source.get('schema')=='M_SOURCE_BEST_FILE_TENSOR_CONTEXT_V1' and source.get('source_best_bytes_tensors_head_verified') is True,'actual source-best tensor context required')
    require(source.get('parent_run_id')==parent_run_id and source['route'] in ROLES and tuple(source['roles'])==ROLES[source['route']],'M endpoint/role selection changed')
    route=source['route'];cfg=root/('configs/execution.json' if route=='D' else 'configs/H_execution.json')
    legacy,parent,config,task=_primary(route,cfg,parent_run_id)
    require(task.get('route')==route and task.get('phase')=='formal' and task.get('recipe_frozen') is True and config.get('artifact_root')==str(root),'frozen registered parent required')
    require(source['training_identity']==parent['training_identity'] and source['backbone']==task['backbone'] and source['dimension']==cache_d.DIMENSIONS[task['backbone']],'selected source-best/parent identity mismatch')
    require(source['checkpoint_path']==str(root/'runs'/parent_run_id/'source_best.pt'),'canonical selected source-best path required')
    require(type(source['best']['epoch']) is int and 1<=source['best']['epoch']<=task['epochs'],'source-best epoch budget invalid')
    for path,pin in source['input_sha256'].items():_pin(path,pin,pins)
    _pin(cfg,parent['config_sha256'],pins,root);_pin(parent['checkpoint_path'],parent['checkpoint_sha256'],pins,root)
    _pin(source['checkpoint_path'],source['checkpoint_sha256'],pins,root);_pin(__file__,sf.digest(__file__),pins)
    gatepath=root/task['qualification_gate'];gate=_json(gatepath);require(gate.get('status')=='PASS','parent qualification required')
    _pin(gatepath,sf.digest(gatepath),pins,root)
    freeze_path=root/f'data/{route}/freeze.json';freeze=_json(freeze_path);require(freeze.get('status')=='FROZEN','frozen data required')
    expected=gate.get('D_manifest_freeze_sha256') if route=='D' else task.get('H_data_freeze_sha256')
    _pin(freeze_path,expected,pins,root)
    require(freeze.get('target_scores_used',False) is False and freeze.get('target_performance_used',False) is False,'target-dependent data selection refused')
    tables={}
    for role in ROLES[route]:
        path=root/f'data/{route}'/(role+'.parquet')
        if route=='H':
            table=freeze['tables'][role];require(table['path']==str(path),'frozen H table location changed');pin=table['sha256']
        else:pin=freeze['manifest_sha256'][role+'.parquet'];table={}
        _pin(path,pin,pins,root)
        frame=legacy._read_manifest(path,role) if route=='H' else legacy._read_manifest(path)
        require(len(frame)>0 and 'video_uid' in frame,'nonempty original role table required')
        if 'cal' in role.lower():require(frame.label.eq(0).all(),'calibration pool must contain only reals')
        if route=='H':require(len(frame)==table['rows'] and frame.video_uid.nunique()==table['videos'],'frozen H accepted table coverage mismatch')
        tables[role]=dict(path=str(path),sha256=pin,rows=len(frame),videos=int(frame.video_uid.nunique()))
    context=BoundContext(root,ap,admission_sha,source,legacy,parent,config,task,tables,pins)
    return context.check(full=True)

def role_path(context,role):
    require(role in context.roles,'role not selected by M')
    return _inside(context.root/'features/source_best_M'/context.admission_sha/context.base['parent_run_id']/role,context.root)

def _identity(context,role):
    table=context.tables[role]
    return dict(context.base,role=role,source_manifest_path=table['path'],source_manifest_sha256=table['sha256'],
                accepted_table_rows=table['rows'],accepted_table_videos=table['videos'])

def _head_hash(weight,bias,dimension):
    w,b=cache_d._validate_head(weight,bias,dimension)
    return hashlib.sha256(w.tobytes(order='C')+b.tobytes(order='C')).hexdigest()

def _npy(path,shape):
    path=Path(path);require(path.is_file() and not path.is_symlink(),'regular array required')
    with path.open('rb') as stream:
        version=np.lib.format.read_magic(stream);require(version==(1,0),'bounded NPY v1 required')
        got,fortran,dtype=np.lib.format.read_array_header_1_0(stream,max_header_size=4096)
        offset=stream.tell()
    require(tuple(got)==tuple(shape) and dtype==np.dtype('float32') and not fortran,'array header shape/dtype/order mismatch')
    require(offset<=4096 and path.stat().st_size==offset+int(np.prod(shape,dtype=np.int64))*4,'array header/file byte budget mismatch')
    return np.load(path,mmap_mode='r',allow_pickle=False)

def _fsync_dir(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def _publish_file(temp,path):
    try:os.link(temp,path)
    except FileExistsError:
        require(sf.digest(temp)==sf.digest(path),'refuse replacement of an existing committed artifact')
    _fsync_dir(path.parent)
    Path(temp).unlink()

def _array(path,x):
    path=Path(path);x=np.asarray(x);require(x.dtype==np.float32 and x.flags.c_contiguous,'FP32 C-order array required')
    if path.exists():
        saved=_npy(path,x.shape);require(saved.tobytes(order='C')==x.tobytes(order='C'),'resumed array bit identity mismatch');return
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.'+path.name+'.',suffix='.tmp',delete=False) as stream:
        np.save(stream,x,allow_pickle=False);stream.flush();os.fsync(stream.fileno());temp=stream.name
    _publish_file(temp,path)

def _save_json(path,value):
    path=Path(path);payload=_typed(value)
    total=sum(p.stat().st_size for p in path.parent.glob('*.json') if p!=path)
    require(len(payload)+total<=ROLE_METADATA_BYTES,'cumulative role JSON metadata capacity exceeded')
    if path.exists():require(_typed(_json(path))==payload,'refuse replacement of committed JSON');return
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.'+path.name+'.',suffix='.tmp',delete=False) as stream:
        stream.write(payload);stream.flush();os.fsync(stream.fileno());temp=stream.name
    _publish_file(temp,path)

def _precheck(context,role):
    _inventory(context,role)
    folder=role_path(context,role);identity=_identity(context,role);dimension=identity['dimension']
    for filename,shape in [('head_weight.npy',(2,dimension)),('head_bias.npy',(2,))]:
        if (folder/filename).exists():_npy(folder/filename,shape)
    if all((folder/f'head_{key}.npy').exists() for key in ('weight','bias')):
        require(_head_hash(_npy(folder/'head_weight.npy',(2,dimension)),_npy(folder/'head_bias.npy',(2,)),dimension)==identity['original_head_sha256'],'selected checkpoint head bits changed')
    if (folder/'identity.json').exists():require(_typed(_json(folder/'identity.json'))==_typed(identity),'typed cache identity mismatch')
    if (folder/'complete.json').exists():
        complete=_json(folder/'complete.json')
        fields={'schema','status','identity','rows','dimension','precision','autocast','epsilon','zero_norm_count','chunk_rows','cached_manifest_sha256','head','chunks'}
        if context.source['route']=='H':fields.add('input_validation')
        require(set(complete)==fields,'unexpected cache completion fields')
        require(set(complete['head'])=={'weight_file','bias_file','weight_sha256','bias_sha256'},'unexpected head metadata fields')
        require(_typed(complete['identity'])==_typed(identity),'typed cache completion identity mismatch')
    for marker in sorted(folder.glob('chunk_*.json')):
        item=_json(marker);number=int(marker.stem.split('_')[-1])
        require(set(item)=={'file','start','rows','dimension','dtype','identity_sha256','row_sha256','zero_norm_count','sha256'},'unexpected chunk metadata fields')
        require(marker.name==f'chunk_{number:06d}.json' and item.get('file')==f'chunk_{number:06d}.npy','unsafe chunk file')
        rows=min(CHUNK_ROWS,context.tables[role]['rows']-number*CHUNK_ROWS)
        require(rows>0,'chunk beyond selected manifest');_npy(folder/item['file'],(rows,dimension))
    return folder,identity

def _clone(function,overrides):
    result=FunctionType(function.__code__,dict(function.__globals__,**overrides),function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=function.__kwdefaults__
    return result

def _functions(context,role):
    original=context.legacy
    def canonical(folder,identity):
        require(Path(folder).absolute()==role_path(context,role) and _typed(identity)==_typed(_identity(context,role)),'candidate canonical cache identity changed')
    def live(identity):
        context.check(full=True);canonical(role_path(context,role),identity)
    def chunks(folder,frame,identity,chunk_rows):
        require(type(chunk_rows) is int and chunk_rows==CHUNK_ROWS,'frozen M chunk budget changed')
        _precheck(context,role)
        return cache_d._read_chunks(folder,frame,identity,chunk_rows)
    proxy=SimpleNamespace(**dict(vars(original.trainer),save_json=_save_json))
    overrides=dict(SCHEMA=SCHEMA,ROLES=context.roles,_canonical_cache=canonical,_validate_live=live,
                   _read_chunks=chunks,_atomic_array=_array,trainer=proxy)
    return _clone(original._cache_role,overrides),_clone(original.verify_cache,overrides)

def write_role(context,role,weight,bias,feature_batches):
    """Append/resume a candidate from caller-provided raw FP32 batches.

    This low-level writer provides no inference or statistic authorization.
    """
    context.check(full=True);folder=role_path(context,role)
    require(_head_hash(weight,bias,context.base['dimension'])==context.base['original_head_sha256'],'selected source-best head required')
    with _lock(folder/'write.lock',context.root):
        _precheck(context,role)
        for key,x in [('weight',weight),('bias',bias)]:
            if (folder/f'head_{key}.npy').exists():
                saved=_npy(folder/f'head_{key}.npy',np.asarray(x).shape)
                require(saved.tobytes(order='C')==np.asarray(x).tobytes(order='C'),'resumed original head bit identity mismatch')
        if (folder/'complete.json').exists():return verify_role(context,role)[0]
        # Prepublish the exact original manifest append-only; original writers
        # therefore never enter their overwrite-capable copy branch.
        manifest=Path(context.tables[role]['path']);cached=folder/'manifest.parquet'
        if cached.exists():require(sf.digest(cached)==context.tables[role]['sha256'],'cached manifest changed')
        else:
            with tempfile.NamedTemporaryFile(dir=folder,prefix='.manifest.',suffix='.tmp',delete=False) as stream:
                with manifest.open('rb') as source:shutil.copyfileobj(source,stream)
                stream.flush();os.fsync(stream.fileno());temp=stream.name
            _publish_file(temp,cached)
        writer,_=_functions(context,role)
        result=writer(folder,manifest,_identity(context,role),weight,bias,feature_batches,chunk_rows=CHUNK_ROWS)
        context.check(full=True);verify_role(context,role)
        return result

def verify_role(context,role):
    folder,identity=_precheck(context,role)
    _,verify=_functions(context,role);meta,frame=verify(folder,identity)
    require(len(frame)==context.tables[role]['rows'] and frame.video_uid.nunique()==context.tables[role]['videos'],'candidate accepted row/video coverage mismatch')
    context.check();return meta,frame

def load_role(context,role):
    meta,frame=verify_role(context,role);folder=role_path(context,role)
    arrays=[_npy(folder/item['file'],(item['rows'],context.base['dimension'])) for item in meta['chunks']]
    h=np.concatenate(arrays);w=_npy(folder/'head_weight.npy',(2,context.base['dimension']));b=_npy(folder/'head_bias.npy',(2,))
    return meta,frame,h,w,b

def _permit(context,path,pin):
    require(path is not None and isinstance(pin,str),'separate M production cache permit required')
    _pin(path,pin,{},context.root);permit=_json(path)
    require(permit.get('schema')==PERMIT_SCHEMA and permit.get('status')=='PASS_PRODUCTION_CACHE_ADMISSION','qualified production cache permit required')
    require(permit.get('artifact_root')==str(context.root) and permit.get('input_admission_sha256')==context.admission_sha and permit.get('cache_code_sha256')==sf.digest(__file__),'permit input/code context drift')
    require(permit.get('target_inference_authorized') is True and permit.get('four_core_comparisons_only') is True and context.base['parent_run_id'] in permit.get('parent_run_ids',[]),'endpoint not authorized by selected-M permit')
    # No issuer is provided here. Each gate must be an independently accepted
    # real artifact, byte-bound and explicitly positive, before inference.
    for name in ('cache_qualification','actual_D_storage','actual_H_storage','joint_capacity'):
        ref=permit.get(name);require(isinstance(ref,dict),'missing production gate:'+name)
        _pin(ref['path'],ref['sha256'],{},context.root);proof=_json(ref['path'])
        require(proof.get('M_production_cache_admission') is True and proof.get('cache_code_sha256')==sf.digest(__file__) and proof.get('input_admission_sha256')==context.admission_sha,'production gate not accepted:'+name)
    budget=permit['joint_capacity'];proof=_json(budget['path'])
    require(type(proof.get('required_remaining_bytes')) is int and proof['required_remaining_bytes']>0,'remaining combined byte budget required')
    require(shutil.disk_usage(context.root).free>=proof['required_remaining_bytes'],'remaining joint capacity gate failed')
    return permit

def extract(root,input_admission_path,admission_sha,parent_run_id,*,permit_path=None,permit_sha256=None,device='cpu',phase_lease_fd=None):
    """Permit-gated encoder execution. CUDA additionally requires phase/GPU leases.

    The original primary owner currently retains the phase lock. This module
    contains no dispatcher and cannot issue a permit or preempt another owner.
    """
    context=prepare_context(root,input_admission_path,admission_sha,parent_run_id)
    _permit(context,permit_path,permit_sha256)
    cpu=torch.device(device).type=='cpu'
    if not cpu:
        # A future qualified owner must pass a live inherited phase lock fd.
        # Until that bridge is independently qualified, fail closed.
        raise ValueError('M GPU phase-lease bridge not yet qualified; no CUDA dispatch')
    require(not torch.cuda.is_initialized(),'CPU M extraction must not initialize CUDA')
    batch=context.config.get('cache_batch_size',context.task['microbatch'])
    workers=context.config.get('cache_workers',context.config.get('workers',4))
    require(type(batch) is int and batch>0 and type(workers) is int and workers>=0 and context.config.get('cache_chunk_rows',CHUNK_ROWS)==CHUNK_ROWS,'frozen M loader/chunk budget invalid')
    with _lock(context.root/'features/source_best_M'/context.admission_sha/parent_run_id/'extract.lock',context.root):
        # Fresh original readers validate every selected input before model allocation.
        frames={role:(context.legacy._read_manifest(context.tables[role]['path'],role) if context.source['route']=='H' else context.legacy._read_manifest(context.tables[role]['path'])) for role in context.roles}
        model=None;results={};trainer.policy()
        for role in context.roles:
            _permit(context,permit_path,permit_sha256)
            context.check(full=True)
            if (role_path(context,role)/'complete.json').exists():
                results[role]=verify_role(context,role)[0];_permit(context,permit_path,permit_sha256);continue
            if model is None:
                checkpoint=sf._load(context.source['checkpoint_path'])
                require(checkpoint.get('identity')==context.source['training_identity'] and checkpoint.get('epoch')==context.source['best']['epoch'],'selected checkpoint tensor identity changed')
                sf._tensor_check(checkpoint['model'],2*1024**3);context.check()
                w=checkpoint['model']['classifier.weight'].detach().numpy().copy();b=checkpoint['model']['classifier.bias'].detach().numpy().copy()
                require(_head_hash(w,b,context.base['dimension'])==context.base['original_head_sha256'],'selected original FP32 head changed')
                model=_build_model(context.task['backbone'],context.config['backbones'][context.task['backbone']]['weight_path'])
                model.load_state_dict(checkpoint['model'],strict=True);model=model.to(device='cpu',dtype=torch.float32).eval()
            def batches(start,frame=frames[role]):
                data=trainer.loader(frame.iloc[start:].reset_index(drop=True),context.task['backbone'],batch,workers=workers)
                with torch.inference_mode(),torch.autocast(device_type='cpu',enabled=False):
                    for x,_ in data:
                        _permit(context,permit_path,permit_sha256)
                        h=model.encoder(x.to(device='cpu',dtype=torch.float32))
                        require(h.dtype==torch.float32,'encoder must export original FP32 features')
                        _permit(context,permit_path,permit_sha256)
                        yield h.detach().cpu().numpy().copy()
            results[role]=write_role(context,role,w,b,batches)
            _permit(context,permit_path,permit_sha256)
        _permit(context,permit_path,permit_sha256)
        context.check(full=True);require(not torch.cuda.is_initialized(),'CPU extraction initialized CUDA')
        return dict(schema=SCHEMA,status='M_SOURCE_BEST_CANDIDATE_ROLES_COMPLETE_REQUIRES_INDEPENDENT_ADMISSION',run_id=parent_run_id,roles=results)
