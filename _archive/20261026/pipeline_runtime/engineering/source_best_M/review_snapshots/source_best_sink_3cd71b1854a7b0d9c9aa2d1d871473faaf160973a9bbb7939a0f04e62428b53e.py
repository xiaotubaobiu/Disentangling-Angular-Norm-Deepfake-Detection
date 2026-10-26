"""Bounded append-only transport sink for secondary statistics.

Complete means declared bytes roundtrip, never scientific or resource admission.
Writer leases cover the whole namespace; failed staging is retained and charged.
"""
from pathlib import Path,PurePosixPath
import copy,csv,ctypes,errno,hashlib,io as streamio,json,math,os,stat,uuid
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from . import analysis_d as io,multiplicity_bundle as bundle,source_best_cache as cache

SCHEMA='M_COMPACT_STATISTIC_SINK_V1'
KINDS={'json','csv','scores','array','parquet'}
TIMINGS={'preparation_seconds','bootstrap_seconds'}
LIMITS={'max_total_bytes','max_temporary_bytes','max_artifact_bytes','max_json_bytes','max_json_array_bytes','max_json_cells','max_bundle_members','max_table_rows','max_decoded_bytes'}

def typed(value):
    def strict(x):
        if isinstance(x,dict):
            if any(type(k) is not str for k in x):raise ValueError('string JSON keys required')
            for v in x.values():strict(v)
        elif isinstance(x,list):
            for v in x:strict(v)
        elif type(x) not in (str,int,float,bool,type(None)):raise ValueError('plain typed JSON required')
    strict(value)
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def _ordered(value):return json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def _sha(path):return io._digest(path)
def _name(name):
    if type(name) is not str or not name or '\\' in name:raise ValueError('relative declared artifact required')
    p=PurePosixPath(name)
    if p.is_absolute() or any(x in ('','.','..') for x in name.split('/')):raise ValueError('unsafe artifact name')
    return name

def _codes():return {Path(x.__file__).name:_sha(x.__file__) for x in (io,bundle,bundle.codec,bundle.io,cache)}|{Path(__file__).name:_sha(__file__)}
def _regular(path):
    if path.is_symlink() or any(x.is_symlink() for x in path.parents):raise ValueError('symlink namespace refused')
    st=path.stat()
    if not stat.S_ISREG(st.st_mode) or st.st_nlink!=1:raise ValueError('private regular file required')
    return st

def _usage(path):
    if not path.exists():return 0
    if path.is_symlink():raise ValueError('symlink namespace refused')
    total=0
    for p in path.rglob('*'):
        if p.is_symlink():raise ValueError('symlink namespace refused')
        if p.is_file():
            st=_regular(p);total+=max(st.st_size,st.st_blocks*512)
        elif not p.is_dir():raise ValueError('unexpected namespace member')
    return total

def _dirs(root,path):
    try:parts=path.relative_to(root).parts
    except ValueError:raise ValueError('outside sink root')
    fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in parts:
            try:os.mkdir(part,mode=0o700,dir_fd=fd)
            except FileExistsError:pass
            nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
    except OSError as e:raise ValueError('unsafe directory namespace') from e
    finally:os.close(fd)

def _publish(source,target):
    if target.exists() or target.is_symlink():raise ValueError('append-only destination already exists')
    if source.is_dir():
        libc=ctypes.CDLL(None,use_errno=True)
        if not hasattr(libc,'renameat2'):raise ValueError('atomic no-replace directory promotion unavailable')
        if libc.renameat2(-100,os.fsencode(source),-100,os.fsencode(target),1)!=0:
            e=ctypes.get_errno();raise OSError(e,os.strerror(e))
    else:
        os.link(source,target,follow_symlinks=False);source.unlink()
    fd=os.open(target.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)

class _Capped(streamio.RawIOBase):
    def __init__(self,raw,limit):super().__init__();self.raw=raw;self.limit=limit
    def writable(self):return True
    def readable(self):return False
    def seekable(self):return True
    def tell(self):return self.raw.tell()
    def seek(self,offset,whence=0):
        p=self.raw.seek(offset,whence)
        if p<0 or p>self.limit:raise ValueError('artifact seek budget exceeded')
        return p
    def write(self,value):
        if self.tell()+len(value)>self.limit:raise ValueError('artifact byte budget exceeded')
        return self.raw.write(value)
    def flush(self):
        if not self.raw.closed:self.raw.flush()
    def fileno(self):raise streamio.UnsupportedOperation('bounded stream has no exposed descriptor')

class CompactSink:
    def __init__(self,root,output,identity,*,inventory,limits):
        self.root=Path(root).absolute();self.output=Path(output).absolute();self.identity=copy.deepcopy(identity)
        self.inventory=copy.deepcopy(inventory);self.limits=copy.deepcopy(limits);self.completed=False;self._lease=None
        if not isinstance(identity,dict) or not identity:raise ValueError('static context required')
        if not isinstance(inventory,dict) or not inventory or any(_name(k)!=k or v not in KINDS for k,v in inventory.items()):raise ValueError('declared inventory required')
        if set(limits)!=LIMITS or any(type(v) is not int or v<=0 for v in limits.values()):raise ValueError('explicit positive limits required')
        mapped=[PurePosixPath(k+('.bundle' if v=='json' else '')) for k,v in inventory.items()]
        if any(a==b or a in b.parents or b in a.parents for i,a in enumerate(mapped) for b in mapped[i+1:]):raise ValueError('mapped artifact namespace collision')
        token=hashlib.sha256(typed(identity)).hexdigest();expected=self.root/'analysis/source_best_M'/token
        if self.output.parent!=expected or not self.output.name or self.output.name.startswith('.'):raise ValueError('identity-bound sink namespace required')
        self.claim=dict(schema=SCHEMA,identity=self.identity,inventory=self.inventory,limits=self.limits,implementations=_codes())
        self._claim_bytes=typed(self.claim);self._initial=typed([self.identity,self.inventory,self.limits])
        self.artifacts=self.output/'artifacts';self.receipts=self.output/'.receipts';self.staging=self.output/'.staging'
    def __enter__(self):
        try:
            self._lease=cache._lock(self.output/'writer.lock',self.root);self._lease.__enter__()
            claim=self.output/'identity.json'
            if claim.exists():
                if _regular(claim).st_size>1024**2 or typed(json.loads(claim.read_bytes()))!=self._claim_bytes:raise ValueError('sink identity/code/limits mismatch')
            else:
                if set(p.name for p in self.output.iterdir())!={'writer.lock'}:raise ValueError('unclaimed namespace preserved')
                self._small(claim,self.claim)
            for p in (self.artifacts,self.receipts,self.staging):_dirs(self.root,p)
            self._guard();self._inventory_check()
            for name in self.inventory:
                if self._receipt_path(name).exists():self._read(name)
            if (self.output/'complete.json').exists():
                actual=self._small_read(self.output/'complete.json')
                if typed(actual)!=typed(self._completion()):raise ValueError('complete inventory mismatch')
                self.completed=True
            return self
        except BaseException:
            if self._lease is not None:self._lease.__exit__(None,None,None);self._lease=None
            raise
    def __exit__(self,*args):
        if self._lease is not None:self._lease.__exit__(*args);self._lease=None
    def _guard(self):
        if self._lease is None or typed([self.identity,self.inventory,self.limits])!=self._initial or typed(dict(self.claim,implementations=_codes()))!=self._claim_bytes:raise ValueError('sink context/code/limits changed')
        if typed(self._small_read(self.output/'identity.json'))!=self._claim_bytes:raise ValueError('sink claim changed')
        if _usage(self.staging)>self.limits['max_temporary_bytes']:raise ValueError('retained staging exceeds temporary budget')
        if self._persistent()>self.limits['max_total_bytes']:raise ValueError('persistent budget exceeded')
    def _persistent(self):return _usage(self.output)-_usage(self.staging)
    def _small_read(self,path):
        if _regular(path).st_size>1024**2:raise ValueError('evidence oversized')
        value=json.loads(path.read_bytes());typed(value);return value
    def _small(self,path,value):
        raw=typed(value)+b'\n'
        if len(raw)>1024**2:raise ValueError('small evidence budget exceeded')
        if self._persistent()+len(raw)+4096>self.limits['max_total_bytes']:raise ValueError('persistent evidence budget exceeded')
        tmp=self.output/('.evidence-'+uuid.uuid4().hex)
        with tmp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        _publish(tmp,path)
    def _receipt_path(self,name):return self.receipts/(hashlib.sha256(name.encode()).hexdigest()+'.json')
    def _path(self,name):return self.artifacts/(name+('.bundle' if self.inventory[name]=='json' else ''))
    def _identity(self,name):return dict(sink_identity_sha256=hashlib.sha256(self._claim_bytes).hexdigest(),artifact=name,kind=self.inventory[name])
    def _inventory_check(self):
        if set(p.name for p in self.output.iterdir())-{'identity.json','writer.lock','artifacts','.receipts','.staging','complete.json'}:raise ValueError('foreign sink member preserved')
        expected_receipts={self._receipt_path(n).name for n in self.inventory}
        if any(p.name not in expected_receipts or not p.is_file() for p in self.receipts.iterdir()):raise ValueError('foreign artifact receipt')
        allowed=set()
        for n in self.inventory:
            p=self._path(n);allowed.add(p)
            allowed.update(x for x in p.parents if x!=self.artifacts and self.artifacts in x.parents)
            if self.inventory[n]=='json' and p.exists():allowed.update(p.rglob('*'))
        for p in self.artifacts.rglob('*'):
            if p not in allowed:raise ValueError('undeclared artifact payload')
        for n in self.inventory:
            if self._path(n).exists() and not self._receipt_path(n).exists():raise ValueError('uncommitted publication preserved; recovery required')
    def _files(self,path,destination=None):
        paths=sorted(path.rglob('*')) if path.is_dir() else [path]
        def key(p):
            if destination is None:return str(p.relative_to(self.artifacts))
            return str(destination.relative_to(self.artifacts)/(p.relative_to(path))) if path.is_dir() else str(destination.relative_to(self.artifacts))
        return {key(p):_sha(p) for p in paths if p.is_file()}
    def _table_probe(self,p):
        # Stream one CSV row at a time, before pandas materializes the table.
        # This is a conservative decoded-table bound, not peak-process RSS.
        old=csv.field_size_limit();csv.field_size_limit(min(self.limits['max_artifact_bytes'],self.limits['max_decoded_bytes']))
        try:
            with p.open(encoding='utf-8',newline='') as f:
                reader=csv.reader(f);header=next(reader,None)
                if not header or len(header)!=len(set(header)):raise ValueError('distinct table headers required')
                rows=0;decoded=4096+sum(128+4*len(v.encode()) for v in header)
                for row in reader:
                    rows+=1
                    if len(row)!=len(header) or rows>self.limits['max_table_rows']:raise ValueError('CSV row/column budget exceeded')
                    decoded+=8+sum(128+4*len(v.encode()) for v in row)
                    if decoded>self.limits['max_decoded_bytes']:raise ValueError('CSV decoded-table budget exceeded')
        except csv.Error as error:raise ValueError('CSV field budget exceeded') from error
        finally:csv.field_size_limit(old)
    def _decode(self,name,p,br):
        kind=self.inventory[name]
        if kind=='json':
            return bundle.read_bundle(p,expected_complete_sha256=br['complete_sha256'],expected_identity=self._identity(name),max_cells=self.limits['max_json_cells'],max_json_bytes=self.limits['max_json_bytes'],max_array_bytes=self.limits['max_json_array_bytes'])
        if _regular(p).st_size>self.limits['max_artifact_bytes']:raise ValueError('artifact size exceeds cap')
        if kind=='array':
            with p.open('rb') as f:
                version=np.lib.format.read_magic(f)
                if version!=(1,0):raise ValueError('registered NPY1 header required')
                shape,fortran,dtype=np.lib.format.read_array_header_1_0(f,max_header_size=4096)
                if any(type(x) is not int or x<0 for x in shape):raise ValueError('invalid NPY dimensions')
                size=math.prod(shape)*dtype.itemsize
                if dtype.hasobject or fortran or size>self.limits['max_decoded_bytes'] or f.tell()+size!=p.stat().st_size:raise ValueError('NPY allocation cap/type mismatch')
            return np.load(p,allow_pickle=False)
        if kind=='parquet':
            meta=pq.ParquetFile(p).metadata;decoded=4096+meta.num_rows*(meta.num_columns*128+8)
            if meta.num_rows>self.limits['max_table_rows']:raise ValueError('Parquet row budget exceeded')
            for i in range(meta.num_row_groups):
                group=meta.row_group(i)
                for j in range(group.num_columns):
                    col=group.column(j)
                    # Owned transport writes plain uncompressed values. Refuse
                    # dictionary/RLE value inflation rather than extrapolating
                    # decoded bytes from a tiny dictionary page.
                    if col.compression!='UNCOMPRESSED' or not set(col.encodings)<={'PLAIN','RLE'} or 'PLAIN' not in col.encodings:raise ValueError('unregistered Parquet value encoding')
                    decoded+=4*col.total_uncompressed_size+128*col.num_values
            if decoded>self.limits['max_decoded_bytes']:raise ValueError('Parquet decoded-table budget exceeded')
            q=pd.read_parquet(p)
        else:
            self._table_probe(p)
            q=io.read_scores(p) if kind=='scores' else pd.read_csv(p,float_precision='round_trip',nrows=self.limits['max_table_rows']+1)
        if len(q)>self.limits['max_table_rows'] or q.memory_usage(deep=True).sum()>self.limits['max_decoded_bytes']:raise ValueError('table decoded budget exceeded')
        return q
    def _read(self,name):
        rec=self._small_read(self._receipt_path(name));p=self._path(name)
        if rec.get('schema')!=SCHEMA or rec.get('identity')!=self._identity(name) or rec.get('files')!=self._files(p):raise ValueError('artifact digest/context mismatch')
        return self._decode(name,p,rec.get('bundle'))
    def _equivalent(self,name,kind,saved,value):
        if kind=='json':
            expected=io._jsonable(value)
            base=PurePosixPath(name).name
            known=base in {'normal_bootstrap_'+r+'_'+a+'.json' for r in ('raw','l2') for a in ('mean_logit','mean_probability')} or base in {'closure_bootstrap.json','conditional_bootstrap.json','primary_n20_vs_source_bootstrap.json'}
            if known and isinstance(saved,dict) and isinstance(expected,dict):
                for key in TIMINGS:
                    if (key in saved)!=(key in expected):raise ValueError('bootstrap timing inventory differs')
                    for obj in (saved,expected):
                        if key in obj and (type(obj[key]) not in (int,float) or not math.isfinite(obj[key]) or obj[key]<0):raise ValueError('invalid engineering timing')
                if _ordered({k:v for k,v in saved.items() if k not in TIMINGS})==_ordered({k:v for k,v in expected.items() if k not in TIMINGS}):return
            if _ordered(saved)==_ordered(expected):return
            raise ValueError('committed scientific JSON differs')
        if kind=='array':
            a=np.asarray(value)
            if a.dtype!=saved.dtype or a.shape!=saved.shape or a.tobytes()!=saved.tobytes():raise ValueError('committed array differs')
        elif kind=='parquet':pd.testing.assert_frame_equal(saved,value,check_exact=True)
        else:
            text=streamio.StringIO()
            if kind=='csv':io._csv(text,value)
            else:
                q=io.score.precision_safe_score_table(value)
                for c in io.TUPLE_COLUMNS:
                    if c in q:q[c]=q[c].map(lambda v:json.dumps(list(v),ensure_ascii=False))
                q.to_csv(text,index=False,float_format='%.17g')
            expected=pd.read_csv(streamio.StringIO(text.getvalue()),float_precision='round_trip') if kind=='csv' else io.read_scores(streamio.StringIO(text.getvalue()))
            pd.testing.assert_frame_equal(saved,expected,check_exact=True)
    def __call__(self,name,kind,value):
        self._guard();_name(name)
        if self.inventory.get(name)!=kind:raise ValueError('artifact absent from declared kind inventory')
        if self._receipt_path(name).exists():
            saved=self._read(name);self._equivalent(name,kind,saved,value);return saved
        if self.completed:raise ValueError('complete sink cannot extend')
        if self._path(name).exists():raise ValueError('unreceipted payload preserved')
        limit=self.limits['max_artifact_bytes']
        if kind=='json':
            value=io._jsonable(value);self._equivalent(name,kind,value,value)
            encoded=bundle.codec.encode_counts(value,max_cells=min(self.limits['max_json_cells'],self.limits['max_json_array_bytes']//8))
            if len(encoded['arrays'])>self.limits['max_bundle_members']:raise ValueError('bundle member budget exceeded')
            meta={k:encoded[k] for k in ('schema','residual','tables')}
            bundle._array_specs(meta,self.limits['max_json_cells'],self.limits['max_json_array_bytes'])
            _,meta_bytes=bundle._json_identity(meta,self.limits['max_json_bytes'])
            array_bytes=sum(a.nbytes for a in encoded['arrays'].values())
            reserve=meta_bytes+meta_bytes//100+array_bytes+array_bytes//100+4096*len(encoded['arrays'])+4*1024**2
            del encoded,meta
        else:reserve=limit
        reserve=((reserve+4095)//4096)*4096
        if _usage(self.staging)+reserve>self.limits['max_temporary_bytes']:raise ValueError('staging reservation unavailable')
        stage=self.staging/uuid.uuid4().hex
        if kind=='json':
            br=bundle.write_bundle(stage,value,identity=self._identity(name),max_cells=self.limits['max_json_cells'],max_json_bytes=self.limits['max_json_bytes'],max_array_bytes=self.limits['max_json_array_bytes'])
        else:
            br=None
            with stage.open('xb') as raw:
                capped=_Capped(raw,limit)
                if kind=='array':np.save(capped,np.asarray(value),allow_pickle=False)
                elif kind=='parquet':value.to_parquet(capped,index=False,compression=None,use_dictionary=False,row_group_size=2048,write_batch_size=256,data_page_size=65536)
                else:
                    text=streamio.TextIOWrapper(capped,encoding='utf-8',write_through=True)
                    if kind=='csv':io._csv(text,value)
                    else:
                        q=io.score.precision_safe_score_table(value)
                        for c in io.TUPLE_COLUMNS:
                            if c in q:q[c]=q[c].map(lambda v:json.dumps(list(v),ensure_ascii=False))
                        q.to_csv(text,index=False,float_format='%.17g')
                    text.flush();text.detach()
                capped.flush();os.fsync(raw.fileno())
        self._guard();saved=self._decode(name,stage,br);self._equivalent(name,kind,saved,value)
        size=_usage(stage) if stage.is_dir() else max(stage.stat().st_size,stage.stat().st_blocks*512)
        target=self._path(name)
        rec=dict(schema=SCHEMA,identity=self._identity(name),files=self._files(stage,target),bundle=br)
        receipt_reserve=((len(typed(rec))+4096)//4096)*4096
        future_complete=dict(schema=SCHEMA,status='ALL_DECLARED_ARTIFACTS_COMMITTED_REQUIRES_SCIENTIFIC_ADMISSION',identity_sha256='0'*64,receipts={n:'0'*64 for n in self.inventory},scientific_admitted=False)
        complete_reserve=((len(typed(future_complete))+4096)//4096)*4096
        if self._persistent()+size+receipt_reserve+complete_reserve>self.limits['max_total_bytes']:raise ValueError('persistent publication quota exceeded')
        _dirs(self.root,target.parent);_publish(stage,target)
        self._small(self._receipt_path(name),rec)
        return self._read(name)
    def _completion(self):
        receipts={}
        for name in self.inventory:
            p=self._receipt_path(name)
            if not p.exists():raise ValueError('declared artifact inventory incomplete')
            self._read(name);receipts[name]=_sha(p)
        return dict(schema=SCHEMA,status='ALL_DECLARED_ARTIFACTS_COMMITTED_REQUIRES_SCIENTIFIC_ADMISSION',identity_sha256=hashlib.sha256(self._claim_bytes).hexdigest(),receipts=receipts,scientific_admitted=False)
    def finish(self):
        self._guard();self._inventory_check();result=self._completion();p=self.output/'complete.json'
        if p.exists():
            if typed(self._small_read(p))!=typed(result):raise ValueError('complete marker differs')
        else:self._small(p,result)
        self.completed=True;return result


def declared_inventory(route):
    """Fixed complete selected-M output contracts, no optional omitted cells."""
    out={}
    if route=='H':
        for aggregation in ('mean_logit','mean_probability'):
            for history in ('B','AB'):out[f'{aggregation}/source_cal_{history}.csv']='scores'
            for domain in ('SOURCE_TEST','TARGET_TEST'):
                folder=f'{aggregation}/{domain}/all_B'
                for cell in ('H00','H10'):out[f'{folder}/{cell}_scores.csv']='scores'
                for name in ('metrics','comparisons'):out[f'{folder}/{name}.csv']='csv'
                for name in ('thresholds','independent_saved_score_checks','conditional_bootstrap','independent_kernel_checks'):out[f'{folder}/{name}.json']='json'
    elif route=='D':
        out.update({'training_radii.npy':'array','training_reference.json':'json'})
        for role in ('source_test','target_full'):
            out[f'{role}/frame_margins.parquet']='parquet'
            for name in ('video_scores','control_video_scores'):out[f'{role}/{name}.csv']='scores'
            for name in ('metrics','effects','video_diagnostics','method_diagnostics','numerical_audits'):out[f'{role}/{name}.csv']='csv'
            for name in ('pair_transition_ledgers','closure_bootstrap'):out[f'{role}/{name}.json']='json'
            for r in ('raw','l2'):
                for a in ('mean_logit','mean_probability'):out[f'{role}/normal_bootstrap_{r}_{a}.json']='json'
        for role in ('source_cal','target_cal','target_test'):out[f'rq4/{role}_frame_margins.parquet']='parquet'
        out['rq4/scope.json']='json'
        for r in ('raw','l2'):
            for a in ('mean_logit','mean_probability'):
                folder=f'rq4/{r}_{a}'
                for role in ('source_cal','target_cal','target_test'):out[f'{folder}/{role}_scores.csv']='scores'
                out[f'{folder}/calibration.csv']='csv'
                for name in ('independent_calibration_recompute','primary_n20_vs_source_bootstrap'):out[f'{folder}/{name}.json']='json'
        out['independent_rank_recomputation.json']='json'
    else:raise ValueError('only frozen D or H selected-M routes allowed')
    out['limitations.json']='json'
    return out
