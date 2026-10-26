"""Finite closed-D auxiliary relocation. No scientific code or live run is edited."""
import os,json,pathlib,hashlib,fcntl,re,tempfile,datetime
PATTERN=re.compile(r'runs/D_(?:formal|dev)_[A-Za-z0-9_.-]+/(?:state[.]pt|epoch_[0-9]{2}_methods[.]npy)')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def syncdir(p):
 fd=os.open(str(pathlib.Path(p)),os.O_RDONLY);os.fsync(fd);os.close(fd)
def save(p,d):
 p=pathlib.Path(p);tmp=p.with_name(p.name+'.tmp')
 with tmp.open('w') as f:json.dump(d,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
 fd=os.open(str(p.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
def locations(root,cold,e,require_cold=True):
 root=pathlib.Path(root).resolve();cold=pathlib.Path(cold).resolve();rel=e['relative']
 if not PATTERN.fullmatch(rel):raise ValueError('only registered closed D state/method auxiliary paths allowed')
 s=root/rel;t=pathlib.Path(e['staged_path'])
 if s.parent.resolve()!=root/pathlib.Path(rel).parent or not t.resolve().is_relative_to(cold) or t.is_symlink():raise ValueError('path boundary/symlink target violation')
 if require_cold and (not t.is_file() or t.stat().st_size!=e['bytes']):raise ValueError('cold size missing/mismatch')
 return s,t
def check(root,cold,e,hashes=True):
 s,t=locations(root,cold,e);st=s.lstat()
 if s.is_symlink() or not s.is_file() or st.st_nlink!=1 or (st.st_ino,st.st_size,st.st_mtime_ns)!=(e['source_inode'],e['bytes'],e['mtime_ns']):raise ValueError('source fingerprint mismatch')
 if hashes and (sha(s)!=e['sha256'] or sha(t)!=e['sha256']):raise ValueError('source/cold SHA mismatch')
 return s,t
class locked:
 def __init__(self,root):self.p=pathlib.Path(root)/'.cold_auxiliary_migration.lock'
 def __enter__(self):
  self.f=self.p.open('a+')
  try:fcntl.flock(self.f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BaseException:self.f.close();raise
  return self
 def __exit__(self,*args):self.f.close()
def _restore(data,journal):
 for rec in reversed(data['records']):
  e=rec['entry'];s,t=locations(data['root'],data['cold'],e,require_cold=False);b=pathlib.Path(rec['backup']);a=pathlib.Path(rec['alias'])
  if b.exists():
   if b.is_symlink() or b.stat().st_ino!=e['source_inode'] or sha(b)!=e['sha256']:raise ValueError('backup mismatch, preserved for diagnosis')
   if s.is_symlink():
    if s.resolve()!=t.resolve():raise ValueError('unknown canonical alias, preserved')
    os.replace(b,s);syncdir(s.parent)
   elif s.exists() and s.stat().st_ino==b.stat().st_ino:b.unlink();syncdir(s.parent)
   else:raise ValueError('unknown canonical content, preserved')
  elif s.is_symlink():
   if s.resolve()!=t.resolve() or sha(t)!=e['sha256']:raise ValueError('cold restore source mismatch')
   fd,tmp=tempfile.mkstemp(prefix='.'+s.name+'.restore.',dir=s.parent)
   try:
    os.fchmod(fd,rec.get('original_mode',0o600))
    with os.fdopen(fd,'wb') as out,t.open('rb') as src:
     for chunk in iter(lambda:src.read(4*1024*1024),b''):out.write(chunk)
     out.flush();os.fsync(out.fileno())
    os.utime(tmp,ns=(e['mtime_ns'],e['mtime_ns']))
    with open(tmp,'rb') as synced:os.fsync(synced.fileno())
    if sha(tmp)!=e['sha256']:raise ValueError('restored bytes mismatch')
    os.replace(tmp,s);syncdir(s.parent)
   finally:
    if os.path.exists(tmp):os.unlink(tmp)
  if a.is_symlink():
   if a.resolve()!=t.resolve():raise ValueError('unknown temporary alias')
   a.unlink();syncdir(s.parent)
  if not s.is_file() or s.is_symlink() or s.stat().st_size!=e['bytes'] or s.stat().st_mtime_ns!=e['mtime_ns'] or sha(s)!=e['sha256']:raise ValueError('canonical restoration failed')
 data['phase']='RESTORED_REGULAR_SSD_FILES';data['SSD_released_bytes']=0;save(journal,data)
def restore(journal):
 data=json.loads(pathlib.Path(journal).read_text())
 with locked(data['root']):_restore(data,journal)
 return data

def relocate(root,cold,entries,journal,validator,commit=False):
 root=pathlib.Path(root).resolve();cold=pathlib.Path(cold).resolve();journal=pathlib.Path(journal)
 with locked(root):
  if journal.exists():raise FileExistsError('existing transaction: inspect/restore; do not repeat')
  records=[]
  for e in entries:
   s,t=check(root,cold,e);b=s.with_name('.'+s.name+'.cold-original.'+journal.stem);a=s.with_name('.'+s.name+'.cold-alias.'+journal.stem)
   if b.exists() or b.is_symlink() or a.exists() or a.is_symlink():raise ValueError('preexisting temporary path')
   records.append({'entry':e,'backup':str(b),'alias':str(a),'original_mode':s.stat().st_mode & 0o777})
  data={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(root),'cold':str(cold),'records':records,'phase':'PREPARED','SSD_released_bytes':0,'logical_bytes':sum(e['bytes'] for e in entries)};save(journal,data)
  try:
   for rec in records:
    e=rec['entry'];s,t=check(root,cold,e);b=pathlib.Path(rec['backup']);a=pathlib.Path(rec['alias']);os.link(s,b);syncdir(s.parent);a.symlink_to(t);os.replace(a,s);syncdir(s.parent)
   data['phase']='ALIASES_INSTALLED_ORIGINALS_RETAINED';save(journal,data)
   validator([root/e['relative'] for e in entries])
   for rec in records:
    s,t=locations(root,cold,rec['entry']);b=pathlib.Path(rec['backup'])
    if not s.is_symlink() or s.resolve()!=t.resolve() or sha(s)!=rec['entry']['sha256'] or sha(b)!=rec['entry']['sha256']:raise ValueError('post-validation canonical or backup changed')
   if not commit:_restore(data,journal);return data
   data['phase']='COMMITTING_VERIFIED_COLD_RELOCATION';save(journal,data)
   for rec in records:
    backup=pathlib.Path(rec['backup']);backup.unlink();syncdir(backup.parent)
   data['phase']='COMMITTED_COLD_ALIASES';data['SSD_backing_logical_bytes_detached']=data['logical_bytes'];data['SSD_released_bytes']=None;data['actual_free_bytes_after']=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize;data['release_measurement_note']='Fresh filesystem availability is authoritative; concurrent writes prevent attributing net free delta to detached logical bytes.';save(journal,data);return data
  except BaseException:
   _restore(data,journal);raise
