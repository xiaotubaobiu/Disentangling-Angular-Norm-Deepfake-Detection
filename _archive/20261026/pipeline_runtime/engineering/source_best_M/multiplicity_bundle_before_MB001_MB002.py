"""Authenticated exact multiplicity bundles for isolated secondary outputs.

This is a file codec, never a scientific/model/data/resource authorization.
All source JSON fields and ordered count mappings roundtrip. Each reader needs
an external complete SHA, expected context and explicit allocation budgets.
Partial same-input transactions can resume; existing children never overwrite.
"""
from pathlib import Path
from contextlib import contextmanager
import fcntl,gzip,hashlib,json,math,os,tempfile,zipfile
import numpy as np
from . import multiplicity_codec as codec,sensitivity_io as io
SCHEMA='M_EXACT_MULTIPLICITY_BUNDLE_V1'
_MAX_ARRAYS=64
_MAX_HEADER=4096
_SMALL_JSON=1024*1024

def _sha(path):return io._digest(path)
def _limits(cells,json_bytes,array_bytes):
 for v in (cells,json_bytes,array_bytes):
  if type(v) is not int or v<=0:raise ValueError('explicit positive integer budgets required')
 if array_bytes<8:raise ValueError('at least8array bytes required')

def _json_parts(value):
 for token in json.JSONEncoder(ensure_ascii=False,indent=2,allow_nan=False).iterencode(value):yield token.encode('utf-8')
 yield b'\n'

def _json_identity(value,limit):
 h=hashlib.sha256();n=0
 for block in _json_parts(value):
  n+=len(block)
  if n>limit:raise ValueError('canonical JSON byte budget exceeded')
  h.update(block)
 return h.hexdigest(),n

def _safe(path,directory=False):
 path=Path(path)
 if path.is_symlink() or any(x.is_symlink() for x in path.parents):raise ValueError('canonical nonsymlink bundle paths required')
 if path.exists() and (not path.is_dir() if directory else not path.is_file()):raise ValueError('unexpected bundle path type')
 return path

def _small_json(path,limit):
 path=_safe(path)
 if not path.is_file() or path.stat().st_size>min(limit,_SMALL_JSON):raise ValueError('small JSON evidence missing or oversized')
 def bad(value):raise ValueError('nonfinite JSON '+value)
 def finite(value):
  number=float(value)
  if not math.isfinite(number):bad(value)
  return number
 try:return json.loads(path.read_bytes(),parse_constant=bad,parse_float=finite)
 except (OSError,UnicodeError,json.JSONDecodeError) as error:raise ValueError('invalid JSON evidence') from error

def _implementations():
 return {Path(module.__file__).name:_sha(module.__file__) for module in (codec,io)}|{Path(__file__).name:_sha(__file__)}

def _atomic_json(path,value):
 with tempfile.NamedTemporaryFile(mode='w+b',prefix=path.name+'.',suffix='.partial',dir=path.parent,delete=False) as stream:
  temporary=Path(stream.name)
  for block in _json_parts(value):stream.write(block)
  stream.flush();os.fsync(stream.fileno())
 os.link(temporary,path)
 try:temporary.unlink()
 except OSError:pass

def _publish_npz(path,arrays):
 with tempfile.NamedTemporaryFile(mode='w+b',prefix=path.name+'.',suffix='.partial',dir=path.parent,delete=False) as stream:
  temporary=Path(stream.name);np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
 os.link(temporary,path)
 try:temporary.unlink()
 except OSError:pass

@contextmanager
def _writer_lock(path):
 lock=_safe(path/'writer.lock')
 with lock.open('a+b') as stream:
  try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError as error:raise RuntimeError('bundle writer already active') from error
  try:yield
  finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)

def _array_specs(meta,max_cells,max_array_bytes):
 if not isinstance(meta,dict) or set(meta)!={'schema','residual','tables'} or meta['schema']!=codec.SCHEMA or not isinstance(meta['tables'],list):raise ValueError('invalid compact metadata')
 if len(meta['tables'])*2>_MAX_ARRAYS:raise ValueError('array-member budget exceeded')
 specs={};cells=0;size=0
 required={'list_path','field','row_indices','groups','counts_array','order_array'}
 for i,table in enumerate(meta['tables']):
  if not isinstance(table,dict) or set(table)!=required or table['field'] not in codec.FIELDS:raise ValueError('invalid multiplicity table')
  if not isinstance(table['row_indices'],list) or not isinstance(table['groups'],list):raise ValueError('invalid table dimensions')
  rows,columns=len(table['row_indices']),len(table['groups']);cells+=rows*columns;size+=rows*columns*8
  if cells>max_cells or size>max_array_bytes:raise ValueError('summed array cell/byte budget exceeded')
  for prefix,dtype in [('counts',np.dtype(np.uint32)),('order',np.dtype(np.int32))]:
   key=f'{prefix}_{i}'
   if table[prefix+'_array']!=key:raise ValueError('array metadata identity mismatch')
   specs[key]=dict(dtype=dtype.str,shape=[rows,columns],bytes=rows*columns*4)
 return specs

def _header(stream,info,spec):
 try:
  version=np.lib.format.read_magic(stream)
  if version!=(1,0):raise ValueError('registered arrays require NPY1.0header')
  shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream,max_header_size=_MAX_HEADER)
 except (ValueError,EOFError,UnicodeError,TypeError) as error:raise ValueError('invalid numeric array header') from error
 if (not isinstance(shape,tuple) or any(type(x) is not int for x in shape) or list(shape)!=spec['shape']
     or fortran is not False or dtype.hasobject or dtype.str!=spec['dtype'] or info.file_size!=stream.tell()+spec['bytes']):raise ValueError('array header dtype/shape/byte budget mismatch')

def _read_npz(path,expected_sha,specs,max_array_bytes):
 path=_safe(path)
 bound=max_array_bytes+max_array_bytes//100+_MAX_HEADER*len(specs)+_SMALL_JSON
 if not path.is_file() or path.stat().st_size>bound or _sha(path)!=expected_sha:raise ValueError('NPZ file hash/size mismatch')
 expected={key+'.npy':key for key in specs};arrays={}
 try:
  with zipfile.ZipFile(path,'r') as archive:
   infos=archive.infolist();names=[x.filename for x in infos]
   if len(names)!=len(set(names)) or set(names)!=set(expected):raise ValueError('unexpected or duplicate NPZ members')
   if sum(x.file_size for x in infos)>max_array_bytes+_MAX_HEADER*len(infos):raise ValueError('NPZ inflation budget exceeded')
   for info in infos:
    spec=specs[expected[info.filename]]
    if info.compress_type!=zipfile.ZIP_DEFLATED or info.flag_bits&1 or info.file_size>spec['bytes']+_MAX_HEADER:raise ValueError('unregistered or oversized NPZ entry')
    with archive.open(info) as stream:_header(stream,info,spec)
   # Allocate only from a validated, bounded payload, never np.load on an
   # untrusted header. Recheck each header immediately before its payload.
   for info in infos:
    key=expected[info.filename];spec=specs[key]
    with archive.open(info) as stream:
     _header(stream,info,spec);payload=stream.read(spec['bytes']+1)
     if len(payload)!=spec['bytes']:raise ValueError('array payload length mismatch')
    arrays[key]=np.frombuffer(payload,dtype=np.dtype(spec['dtype'])).reshape(spec['shape'])
 except (zipfile.BadZipFile,EOFError,OSError,RuntimeError) as error:raise ValueError('invalid or incomplete NPZ') from error
 if _sha(path)!=expected_sha:raise ValueError('NPZ changed during read')
 return arrays

def _receipt(path,complete):
 return dict(schema=SCHEMA,complete_sha256=_sha(path/'complete.json'),raw_json_sha256=complete['raw_json_sha256'],
             raw_json_bytes=complete['raw_json_bytes'],stored_bytes=sum((path/name).stat().st_size for name in ('metadata.json.gz','multiplicities.npz','complete.json','transaction.json')),
             production_admitted=False)

def read_bundle(path,*,expected_complete_sha256,expected_identity,max_cells,max_json_bytes,max_array_bytes):
 _limits(max_cells,max_json_bytes,max_array_bytes);io._hash_identity(expected_complete_sha256);path=_safe(path,directory=True)
 complete_path=_safe(path/'complete.json')
 if not complete_path.is_file() or _sha(complete_path)!=expected_complete_sha256:raise ValueError('external complete SHA mismatch')
 complete=_small_json(complete_path,max_json_bytes)
 required={'schema','identity','raw_json_sha256','raw_json_bytes','implementation_sha256','metadata','npz','array_specs'}
 if not isinstance(complete,dict) or set(complete)!=required or complete['schema']!=SCHEMA or complete['identity']!=expected_identity or complete['implementation_sha256']!=_implementations():raise ValueError('bundle context/schema/implementation mismatch')
 io._hash_identity(complete['raw_json_sha256'])
 if type(complete['raw_json_bytes']) is not int or not 0<complete['raw_json_bytes']<=max_json_bytes:raise ValueError('original JSON budget mismatch')
 meta_info=complete['metadata'];npz_info=complete['npz']
 if not isinstance(meta_info,dict) or not isinstance(npz_info,dict):raise ValueError('component receipt missing')
 meta_path=_safe(path/'metadata.json.gz');npz_path=_safe(path/'multiplicities.npz')
 if meta_path.stat().st_size!=meta_info['compressed_bytes'] or meta_path.stat().st_size>max_json_bytes+max_json_bytes//100+_SMALL_JSON:raise ValueError('metadata stored byte budget mismatch')
 meta=io.read_json(meta_path,compressed_sha256=meta_info['compressed_sha256'],uncompressed_sha256=meta_info['uncompressed_sha256'],max_uncompressed_bytes=max_json_bytes)
 specs=_array_specs(meta,max_cells,max_array_bytes)
 if specs!=complete['array_specs'] or npz_path.stat().st_size!=npz_info['bytes']:raise ValueError('numeric component identity mismatch')
 arrays=_read_npz(npz_path,npz_info['sha256'],specs,max_array_bytes)
 value=codec.decode_counts(meta|{'arrays':arrays},max_cells=max_cells)
 raw_sha,raw_bytes=_json_identity(value,max_json_bytes)
 if raw_sha!=complete['raw_json_sha256'] or raw_bytes!=complete['raw_json_bytes']:raise ValueError('complete original JSON roundtrip mismatch')
 if (_sha(complete_path)!=expected_complete_sha256 or _sha(meta_path)!=meta_info['compressed_sha256'] or _sha(npz_path)!=npz_info['sha256']
     or _implementations()!=complete['implementation_sha256'] or complete['identity']!=expected_identity):raise ValueError('bundle source/context/component changed during read')
 return value

def write_bundle(path,value,*,identity,max_cells,max_json_bytes,max_array_bytes,resume=False):
 _limits(max_cells,max_json_bytes,max_array_bytes);path=_safe(path,directory=True)
 if not isinstance(identity,dict) or not identity:raise ValueError('explicit nonempty context identity required')
 implementations=_implementations();identity_sha,_=_json_identity(identity,min(max_json_bytes,_SMALL_JSON))
 context=json.loads(json.dumps(identity,ensure_ascii=False,allow_nan=False))
 existed=path.exists()
 if existed and not resume:raise FileExistsError(path)
 if existed and not _safe(path/'transaction.json').is_file():raise ValueError('foreign directory has no owned transaction')
 if existed:_safe(path/'complete.json')
 raw_sha,raw_bytes=_json_identity(value,max_json_bytes)
 encoded=codec.encode_counts(value,max_cells=min(max_cells,max_array_bytes//8));meta={k:encoded[k] for k in ('schema','residual','tables')};arrays=encoded['arrays']
 specs=_array_specs(meta,max_cells,max_array_bytes);meta_sha,meta_bytes=_json_identity(meta,max_json_bytes)
 transaction=dict(schema=SCHEMA,identity=context,raw_json_sha256=raw_sha,raw_json_bytes=raw_bytes,implementation_sha256=implementations)
 if not existed:path.mkdir(parents=True,exist_ok=False)
 with _writer_lock(path):
  transaction_path=_safe(path/'transaction.json')
  if transaction_path.exists():
   if _small_json(transaction_path,max_json_bytes)!=transaction:raise ValueError('resume input/context/implementation differs')
  else:_atomic_json(transaction_path,transaction)
  complete_path=_safe(path/'complete.json')
  if complete_path.exists():
   complete=_small_json(complete_path,max_json_bytes)
   restored=read_bundle(path,expected_complete_sha256=_sha(complete_path),expected_identity=identity,max_cells=max_cells,max_json_bytes=max_json_bytes,max_array_bytes=max_array_bytes)
   if _json_identity(restored,max_json_bytes)!=(raw_sha,raw_bytes):raise ValueError('resume original JSON input differs')
   return _receipt(path,complete)
  meta_path=_safe(path/'metadata.json.gz');npz_path=_safe(path/'multiplicities.npz')
  if not meta_path.exists():meta_info=io.write_json(meta_path,meta)
  else:
   meta_info=dict(compressed_sha256=_sha(meta_path),uncompressed_sha256=meta_sha,compressed_bytes=meta_path.stat().st_size,uncompressed_bytes=meta_bytes)
  restored_meta=io.read_json(meta_path,compressed_sha256=meta_info['compressed_sha256'],uncompressed_sha256=meta_sha,max_uncompressed_bytes=max_json_bytes)
  if _json_identity(restored_meta,max_json_bytes)!=(meta_sha,meta_bytes):raise ValueError('partial metadata differs')
  if not npz_path.exists():_publish_npz(npz_path,arrays)
  npz_sha=_sha(npz_path);restored_arrays=_read_npz(npz_path,npz_sha,specs,max_array_bytes)
  restored=codec.decode_counts(restored_meta|{'arrays':restored_arrays},max_cells=max_cells)
  if _json_identity(restored,max_json_bytes)!=(raw_sha,raw_bytes) or _json_identity(value,max_json_bytes)!=(raw_sha,raw_bytes):raise ValueError('input or stored value changed during transaction')
  if (_implementations()!=implementations or _json_identity(identity,min(max_json_bytes,_SMALL_JSON))[0]!=identity_sha
      or _sha(meta_path)!=meta_info['compressed_sha256'] or _sha(npz_path)!=npz_sha):raise ValueError('bundle source/context/component changed during transaction')
  complete=dict(schema=SCHEMA,identity=context,raw_json_sha256=raw_sha,raw_json_bytes=raw_bytes,implementation_sha256=implementations,
      metadata={k:meta_info[k] for k in ('compressed_sha256','uncompressed_sha256','compressed_bytes','uncompressed_bytes')},npz=dict(sha256=npz_sha,bytes=npz_path.stat().st_size),array_specs=specs)
  _atomic_json(complete_path,complete);return _receipt(path,complete)
