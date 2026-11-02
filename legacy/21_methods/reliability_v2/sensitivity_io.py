"""Lossless gzip transport for isolated secondary analyses.

No primary artifact is moved/deleted. Both stored and uncompressed byte SHA256
identities are mandatory on reads. A failed write leaves only a .partial file;
atomic hard-link publication never overwrites an already committed output.
This module grants no data/model/GPU qualification or resource admission.
"""
from __future__ import annotations
import gzip,hashlib,json,math,os,re,tempfile,zlib
from pathlib import Path

_CHUNK=1024*1024

def _digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(_CHUNK),b''):h.update(block)
    return h.hexdigest()

def _hash_identity(value):
    if not isinstance(value,str) or re.fullmatch('[0-9a-f]{64}',value) is None:
        raise ValueError('expected SHA256 identity is malformed')
    return value

def _output(path):
    p=Path(path)
    if not p.name.endswith('.json.gz'):raise ValueError('secondary output requires .json.gz suffix')
    if p.is_symlink() or any(a.is_symlink() for a in p.parents):raise ValueError('symlink output is not a bound artifact')
    if p.exists():raise FileExistsError(str(p))
    p.parent.mkdir(parents=True,exist_ok=True)
    return p

def _publish(path,chunks,guard=None):
    p=_output(path);h=hashlib.sha256();n=0
    with tempfile.NamedTemporaryFile(mode='w+b',prefix=p.name+'.',suffix='.partial',dir=p.parent,delete=False) as stream:
        temporary=Path(stream.name)
        with gzip.GzipFile(filename='',mode='wb',compresslevel=1,mtime=0,fileobj=stream) as packed:
            for block in chunks():
                h.update(block);n+=len(block);packed.write(block)
        stream.flush();os.fsync(stream.fileno())
        if guard is not None:guard(h.hexdigest())
        receipt={'schema':'SECONDARY_JSON_GZIP_V1','codec':'gzip','zlib_version':zlib.ZLIB_VERSION,
                 'compressed_sha256':_digest(temporary),'uncompressed_sha256':h.hexdigest(),
                 'compressed_bytes':temporary.stat().st_size,'uncompressed_bytes':n}
    os.link(temporary,p)
    # Only unlink the duplicate temporary name: the committed inode is retained.
    receipt['temporary_cleanup_pending']=False
    try:temporary.unlink()
    except OSError:
        receipt['temporary_cleanup_pending']=True
        receipt['duplicate_temporary_path']=str(temporary)
    return receipt

def write_json(path,value):
    """Stream the complete JSON value to a new atomic gzip artifact."""
    encoder=json.JSONEncoder(ensure_ascii=False,indent=2,allow_nan=False)
    def chunks():
        for text in encoder.iterencode(value):yield text.encode('utf-8')
        yield b'\n'
    return _publish(path,chunks)

def pack_existing(source,path,*,expected_source_sha256):
    """Copy an already hash-bound regular artifact byte-for-byte into gzip."""
    source=Path(source);expected=_hash_identity(expected_source_sha256)
    if source.is_symlink() or not source.is_file():raise ValueError('source must be a bound regular file')
    if _digest(source)!=expected:raise ValueError('source SHA256 differs before packing')
    stat=source.stat();fingerprint=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
    def chunks():
        with source.open('rb') as stream:
            yield from iter(lambda:stream.read(_CHUNK),b'')
    def guard(raw_sha):
        stat=source.stat();current=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
        if raw_sha!=expected or current!=fingerprint or _digest(source)!=expected:
            raise ValueError('source changed during packing')
    return _publish(path,chunks,guard)

def read_json(path,*,compressed_sha256,uncompressed_sha256,max_uncompressed_bytes=None):
    """Decode a hash-bound stream with an explicit raw-byte budget.

    This bounds collected uncompressed bytes, not parsed-object RSS. Callers
    still need real peak-memory qualification and namespace/resource gates.
    """
    if type(max_uncompressed_bytes) is not int or max_uncompressed_bytes<=0:
        raise ValueError('an explicit positive integer uncompressed byte budget is required')
    path=Path(path);stored=_hash_identity(compressed_sha256);raw_expected=_hash_identity(uncompressed_sha256)
    if path.is_symlink() or not path.is_file():raise ValueError('compressed artifact must be a regular file')
    if _digest(path)!=stored:raise ValueError('compressed SHA256 identity mismatch')
    chunks=[];n=0;h=hashlib.sha256()
    try:
        with gzip.open(path,'rb') as stream:
            while True:
                block=stream.read(min(_CHUNK,max_uncompressed_bytes-n+1))
                if not block:break
                n+=len(block)
                if n>max_uncompressed_bytes:raise ValueError('uncompressed byte budget exceeded')
                h.update(block);chunks.append(block)
    except (OSError,EOFError,zlib.error) as error:
        raise ValueError('invalid or incomplete gzip stream') from error
    if _digest(path)!=stored:raise ValueError('compressed artifact changed while decoding')
    if h.hexdigest()!=raw_expected:raise ValueError('uncompressed SHA256 identity mismatch')
    def reject(value):raise ValueError('nonfinite JSON value '+value)
    def finite_float(value):
        number=float(value)
        if not math.isfinite(number):reject(value)
        return number
    return json.loads(b''.join(chunks),parse_constant=reject,parse_float=finite_float)
