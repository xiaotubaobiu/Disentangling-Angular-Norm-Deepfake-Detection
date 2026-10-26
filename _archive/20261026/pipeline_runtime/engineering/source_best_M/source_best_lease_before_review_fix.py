"""Kernel leases for a future selected-M owner, with no production authority.

Uses the unchanged primary H phase/native GPU lock filenames. Closing an owner
fd never explicitly unlocks a description still held by an inherited worker fd.
No dispatcher, CUDA initialization, original process control or permit issuance.
"""
from pathlib import Path
import contextlib,csv,ctypes,fcntl,json,os,re,signal,stat,subprocess,uuid

def require(ok,why):
    if not ok:raise ValueError(why)

def _root(root):
    root=Path(root).absolute();require(root.is_dir() and root.resolve()==root,'canonical existing root required');return root

def _open_lock(root,relative):
    root=_root(root);parts=Path(relative).parts
    require(parts and all(p not in ('','..','.') for p in parts) and not Path(relative).is_absolute(),'relative lease path required')
    directory=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            try:os.mkdir(part,dir_fd=directory)
            except FileExistsError:pass
            child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=directory);os.close(directory);directory=child
        fd=os.open(parts[-1],os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW|os.O_NONBLOCK,0o600,dir_fd=directory)
        try:
            info=os.fstat(fd);require(stat.S_ISREG(info.st_mode) and info.st_nlink==1 and info.st_size==0,'empty private regular lease required')
            current=os.stat(parts[-1],dir_fd=directory,follow_symlinks=False)
            require((current.st_dev,current.st_ino)==(info.st_dev,info.st_ino),'lease inode changed');return fd
        except BaseException:os.close(fd);raise
    finally:os.close(directory)

def _start(pid):
    return int(Path(f'/proc/{pid}/stat').read_text().rpartition(') ')[2].split()[19])

def _held(fd,pid):
    info=os.fstat(fd);device=(os.major(info.st_dev),os.minor(info.st_dev),info.st_ino)
    for line in Path('/proc/locks').read_text().splitlines():
        parts=line.split()
        if len(parts)<8 or parts[1:4]!=['FLOCK','ADVISORY','WRITE']:continue
        if int(parts[4])!=pid:continue
        major,minor,inode=parts[5].split(':')
        if (int(major,16),int(minor,16),int(inode))==device:return True
    return False

class PhaseLease:
    def __init__(self,root):self.root=_root(root);self.fd=None
    def __enter__(self):
        require(self.fd is None,'phase lease already entered');fd=_open_lock(self.root,'H_pipeline.lock')
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:os.close(fd);raise
        self.fd=fd;self.owner=os.getpid();self.birth=_start(self.owner);self.nonce=uuid.uuid4().hex
        try:self.check()
        except BaseException:
            os.close(fd);self.fd=None;raise
        return self
    def check(self):
        require(self.fd is not None and self.owner==os.getpid() and self.birth==_start(self.owner),'owner process/lease changed')
        require(_held(self.fd,self.owner),'exclusive original phase lease required');return self
    def proof(self):
        self.check();info=os.fstat(self.fd)
        return dict(schema='M_INHERITED_PHASE_LEASE_V1',artifact_root=str(self.root),owner_pid=self.owner,owner_start_ticks=self.birth,owner_fd=self.fd,device=info.st_dev,inode=info.st_ino,nonce=self.nonce)
    def __exit__(self,*unused):
        # Do not LOCK_UN: that would also unlock all inherited child fds.
        fd=self.fd;self.fd=None
        if fd is not None:os.close(fd)

def verify_worker(root,proof,fd,*,arm_parent_death=False):
    root=_root(root)
    keys={'schema','artifact_root','owner_pid','owner_start_ticks','owner_fd','device','inode','nonce'}
    require(type(proof) is dict and set(proof)==keys and proof['schema']=='M_INHERITED_PHASE_LEASE_V1' and proof['artifact_root']==str(root),'exact phase lease context required')
    require(all(type(proof[k]) is int and proof[k]>=0 for k in ('owner_pid','owner_start_ticks','owner_fd','device','inode')) and type(fd) is int and fd>=0,'typed phase lease identity required')
    require(isinstance(proof['nonce'],str) and re.fullmatch('[0-9a-f]{32}',proof['nonce']),'phase nonce required')
    owner=proof['owner_pid'];require(owner!=os.getpid() and os.getppid()==owner and _start(owner)==proof['owner_start_ticks'],'live direct owner identity required')
    if arm_parent_death:
        libc=ctypes.CDLL(None,use_errno=True)
        require(libc.prctl(1,signal.SIGTERM,0,0,0)==0,'parent-death signal setup failed')
        require(os.getppid()==owner and _start(owner)==proof['owner_start_ticks'],'owner changed during parent-death setup')
    current=(root/'H_pipeline.lock');require(current.resolve()==current and not current.is_symlink(),'canonical original phase lock required')
    info=os.fstat(fd);parent=os.stat(f'/proc/{owner}/fd/{proof["owner_fd"]}');live=current.stat()
    require(stat.S_ISREG(info.st_mode) and info.st_size==0 and info.st_nlink==1,'empty private inherited lease required')
    require((info.st_dev,info.st_ino)==(parent.st_dev,parent.st_ino)==(live.st_dev,live.st_ino)==(proof['device'],proof['inode']),'inherited/owner/canonical phase inode mismatch')
    require(_held(fd,owner),'original phase kernel lock must belong to owner')
    # Same inode is insufficient. A separately opened fd contends; only the
    # inherited open-file description can reacquire this already-held flock.
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require(_held(fd,owner),'inherited phase description identity lost')
    return proof

def _query(args):
    return subprocess.check_output(['nvidia-smi',*args,'--format=csv,noheader,nounits'],text=True)

@contextlib.contextmanager
def native_gpu(root,config,proof,phase_fd):
    root=_root(root);verify_worker(root,proof,phase_fd,arm_parent_death=True)
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    require(config.get('allowed_gpus')==[0,1,2,3] and visible in config.get('allowed_gpu_uuids',[]) and re.fullmatch('GPU-[A-Za-z0-9-]{1,80}',visible),'single approved GPU0-3 UUID required')
    initial=list(csv.reader(_query(['--query-gpu=index,uuid,memory.used']).splitlines(),skipinitialspace=True))
    mapping={u:int(index) for index,u,memory in initial}
    require(visible in mapping and mapping[visible] in (0,1,2,3),'selected UUID maps outside GPU0-3')
    fd=_open_lock(root,Path('gpu_locks')/(visible+'.lock'))
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rows=list(csv.reader(_query(['--query-gpu=index,uuid,memory.used']).splitlines(),skipinitialspace=True))
        inventory={u:(int(index),int(memory)) for index,u,memory in rows}
        require(visible in inventory and inventory[visible][0] in (0,1,2,3),'selected UUID maps outside GPU0-3')
        occupied={row[0].strip() for row in csv.reader(_query(['--query-compute-apps=gpu_uuid,pid']).splitlines(),skipinitialspace=True) if row}
        require(visible not in occupied and inventory[visible][1]<100,'assigned GPU is not idle; refuse to share')
        verify_worker(root,proof,phase_fd);yield
        verify_worker(root,proof,phase_fd)
    finally:
        # Child is sole owner of its native GPU description; closing releases it.
        os.close(fd)
