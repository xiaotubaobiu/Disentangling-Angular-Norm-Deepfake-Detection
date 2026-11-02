"""CPU leases and global retained-staging accounting; no production permit issuer."""
from pathlib import Path
from contextlib import contextmanager
import fcntl,hashlib,json,os,re,stat,uuid
from . import source_best_cache as cache,source_best_sink as sink,source_best_lease as lease

SCHEMA='M_GLOBAL_TEMPORARY_RESERVATION_V1'

def require(ok,text):
 if not ok:raise ValueError(text)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 cache._inside(p,Path(p).parents[0],file=True)
 require(p.stat().st_size<=65536 and stat.S_ISREG(p.stat().st_mode) and p.stat().st_nlink==1,'bounded private reservation evidence required')
 d=json.loads(p.read_bytes());sink.typed(d);return d

def publish(p,value):
 raw=sink.typed(value)+b'\n';require(len(raw)<=65536,'reservation evidence too large')
 tmp=p.parent/('.publish-'+uuid.uuid4().hex)
 with tmp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 os.link(tmp,p);tmp.unlink()
 fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:os.fsync(fd)
 finally:os.close(fd)

@contextmanager
def cpu_slot(root,slot):
 require(type(slot) is int and slot in (0,1),'shared primary CPU slot0/1 required')
 fd=lease._open_lock(Path(root),f'analysis/H/.cpu_slots/slot{slot}.lock')
 try:
  fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  yield fd
 finally:
  # Inherited descriptions retain exclusion: never explicitly LOCK_UN.
  os.close(fd)

def output_path(root,path):
 root=Path(root).absolute();p=cache._inside(Path(path),root);base=root/'analysis/source_best_M'
 require(p.parent.parent==base and re.fullmatch('[0-9a-f]{64}',p.parent.name) is not None and re.fullmatch('[A-Za-z0-9_]+',p.name) is not None,'canonical registered statistic output required')
 return p

def live(record):
 try:
  fields=Path(f'/proc/{record["pid"]}/stat').read_text().rpartition(') ')[2].split()
  return fields[0] not in ('Z','X') and int(fields[19])==record['birth']
 except (FileNotFoundError,ProcessLookupError,PermissionError,ValueError,IndexError):return False

def staging_usage(root):
 root=Path(root).absolute();home=root/'analysis/source_best_M';stages={}
 if not home.exists():return stages
 for parent in home.iterdir():
  if parent.name in ('.resource_pool.lock','.resource_policy.json','.resource_claims'):continue
  cache._inside(parent,root);require(parent.is_dir() and re.fullmatch('[0-9a-f]{64}',parent.name) is not None,'foreign statistic namespace')
  for out in parent.iterdir():
   output_path(root,out);require(out.is_dir() and not out.is_symlink(),'canonical statistic job directory required')
   p=out/'.staging';require(not p.is_symlink(),'dangling or linked staging refused')
   if p.exists():require(p.is_dir(),'staging directory required');stages[str(out)]=sink._usage(p)
 return stages

class TemporaryPool:
 def __init__(self,root,output,reserve_bytes,pool_bytes):
  self.root=Path(root).absolute();self.output=output_path(self.root,output)
  require(type(reserve_bytes) is int and type(pool_bytes) is int and 4096<=reserve_bytes<=pool_bytes,'positive bounded global temporary reservation required')
  self.reserve=reserve_bytes;self.total=pool_bytes;self.home=self.root/'analysis/source_best_M';self.folder=self.home/'.resource_claims';self.claim_path=None;self.owner=(os.getpid(),lease._start(os.getpid()))
 def _lock(self):return cache._lock(self.home/'.resource_pool.lock',self.root)
 def _policy(self):
  path=self.home/'.resource_policy.json';expected=dict(schema=SCHEMA,artifact_root=str(self.root),pool_bytes=self.total)
  if path.exists():require(read(path)==expected,'global pool size/schema changed')
  else:publish(path,expected)
  sink._dirs(self.root,self.folder)
 def _state(self):
  paths=list(self.folder.iterdir());require(len(paths)<=8192,'reservation history budget exceeded')
  known=set();claims=[]
  for p in paths:
   if re.fullmatch(r'[0-9a-f]{32}\.json',p.name):
    d=read(p);require(set(d)=={'schema','output','pid','birth','reservation_bytes','pool_bytes','nonce'} and d['schema']==SCHEMA,'invalid reservation claim')
    require(type(d['pid']) is int and d['pid']>0 and type(d['birth']) is int and d['birth']>0 and type(d['reservation_bytes']) is int and 4096<=d['reservation_bytes']<=self.total and d['pool_bytes']==self.total and d['nonce']==p.stem,'invalid reservation identity/counts')
    output_path(self.root,d['output']);known.add(p);release=p.with_suffix('.released.json');released=False
    if release.exists():
     require(read(release)==dict(schema=SCHEMA,claim_sha256=sha(p),pid=d['pid'],birth=d['birth']),'invalid reservation release')
     known.add(release);released=True
    if not released and live(d):claims.append(d)
  require(set(paths)==known,'unclaimed resource evidence preserved; explicit recovery required')
  stages=staging_usage(self.root)
  for d in claims:
   key=d['output'];require(not any(x['output']==key for x in claims if x is not d),'duplicate live output claims')
   require(stages.get(key,0)<=d['reservation_bytes'],'live job staging exceeds admitted reservation')
  charged=sum(stages.values())+sum(max(0,d['reservation_bytes']-stages.get(d['output'],0)) for d in claims)
  return claims,stages,charged
 def __enter__(self):
  require((os.getpid(),lease._start(os.getpid()))==self.owner,'pool must enter in its constructing process')
  require(self.claim_path is None,'pool already entered')
  with self._lock():
   self._policy();claims,stages,charged=self._state();require(not any(d['output']==str(self.output) for d in claims),'output has a live reservation')
   own=stages.get(str(self.output),0);require(own<=self.reserve,'retained own staging exceeds per-job quota')
   require(charged+self.reserve-own<=self.total,'global temporary pool unavailable; retained failed stages are charged')
   nonce=uuid.uuid4().hex;record=dict(schema=SCHEMA,output=str(self.output),pid=self.owner[0],birth=self.owner[1],reservation_bytes=self.reserve,pool_bytes=self.total,nonce=nonce)
   self.claim_path=self.folder/(nonce+'.json');publish(self.claim_path,record);self.claim_sha=sha(self.claim_path)
  return self
 def __exit__(self,*args):
  require((os.getpid(),lease._start(os.getpid()))==self.owner,'only original process may release its claim')
  if self.claim_path is None:return
  with self._lock():
   require(sha(self.claim_path)==self.claim_sha,'reservation claim changed')
   release=self.claim_path.with_suffix('.released.json');value=dict(schema=SCHEMA,claim_sha256=self.claim_sha,pid=self.owner[0],birth=self.owner[1])
   if release.exists():require(read(release)==value,'reservation release changed')
   else:publish(release,value)
  self.claim_path=None
