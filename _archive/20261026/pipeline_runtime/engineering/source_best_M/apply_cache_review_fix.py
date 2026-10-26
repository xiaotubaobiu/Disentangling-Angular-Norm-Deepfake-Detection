from pathlib import Path
import hashlib
E=Path(__file__).resolve().parent;p=E/'code/studies/pipeline_reliability/v2/reliability_v2/source_best_cache.py'
s=p.read_text();assert hashlib.sha256(p.read_bytes()).hexdigest()=='b57bd4ff02a06e8db72a265de56a62b1f5dadef160862fa3ebba5e3f7f9107af'
assert 'Exit status: 0' in (E/'source_best_cache_full_suite_time_v.log').read_text(),'preserve stable pre-fix full-suite process until terminal'
backup=E/'source_best_cache_before_review_fix.py';assert not backup.exists();backup.write_bytes(p.read_bytes())
s=s.replace('import contextlib, copy, hashlib, json, os, re, shutil, sys, tempfile','import contextlib, copy, fcntl, hashlib, json, os, re, shutil, stat, sys, tempfile')
s=s.replace('CHUNK_ROWS = 4096','CHUNK_ROWS = 4096\nROLE_METADATA_BYTES = 1024**2')
insert="""
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
        match=re.fullmatch(r'chunk_([0-9]{6})\\.(json|npy)',path.name)
        if match:
            number=int(match[1]);require(number<count,'chunk outside frozen role capacity')
            if match[2]=='npy':
                rows=min(CHUNK_ROWS,context.tables[role]['rows']-number*CHUNK_ROWS)
                _npy(path,(rows,context.base['dimension']))
        else:require(path.name in ordinary,'unregistered candidate/orphan artifact; preserve evidence and resolve before resume')
        if path.suffix=='.json':metadata_bytes+=path.stat().st_size
    require(metadata_bytes<=ROLE_METADATA_BYTES,'cumulative role JSON metadata capacity exceeded')

"""
pos=s.index('def _pin(');s=s[:pos]+insert+s[pos:]
s=s.replace("    path=Path(path);payload=_typed(value)\n", "    path=Path(path);payload=_typed(value)\n    total=sum(p.stat().st_size for p in path.parent.glob('*.json') if p!=path)\n    require(len(payload)+total<=ROLE_METADATA_BYTES,'cumulative role JSON metadata capacity exceeded')\n")
s=s.replace("    folder=role_path(context,role);identity=_identity(context,role);dimension=identity['dimension']", "    _inventory(context,role)\n    folder=role_path(context,role);identity=_identity(context,role);dimension=identity['dimension']")
s=s.replace("    if (folder/'complete.json').exists():require(_typed(_json(folder/'complete.json')['identity'])==_typed(identity),'typed cache completion identity mismatch')", """    if (folder/'complete.json').exists():
        complete=_json(folder/'complete.json')
        fields={'schema','status','identity','rows','dimension','precision','autocast','epsilon','zero_norm_count','chunk_rows','cached_manifest_sha256','head','chunks'}
        if context.source['route']=='H':fields.add('input_validation')
        require(set(complete)==fields,'unexpected cache completion fields')
        require(set(complete['head'])=={'weight_file','bias_file','weight_sha256','bias_sha256'},'unexpected head metadata fields')
        require(_typed(complete['identity'])==_typed(identity),'typed cache completion identity mismatch')""")
s=s.replace("        item=_json(marker);number=int(marker.stem.split('_')[-1])", "        item=_json(marker);number=int(marker.stem.split('_')[-1])\n        require(set(item)=={'file','start','rows','dimension','dtype','identity_sha256','row_sha256','zero_norm_count','sha256'},'unexpected chunk metadata fields')")
s=s.replace("with trainer.exclusive_file(folder/'write.lock'):","with _lock(folder/'write.lock',context.root):")
s=s.replace("with trainer.exclusive_file(context.root/'features/source_best_M'/context.admission_sha/parent_run_id/'extract.lock'):","with _lock(context.root/'features/source_best_M'/context.admission_sha/parent_run_id/'extract.lock',context.root):")
s=s.replace("        for role in context.roles:\n            context.check(full=True)\n", "        for role in context.roles:\n            _permit(context,permit_path,permit_sha256)\n            context.check(full=True)\n")
s=s.replace("                results[role]=verify_role(context,role)[0];continue", "                results[role]=verify_role(context,role)[0];_permit(context,permit_path,permit_sha256);continue")
s=s.replace("                    for x,_ in data:\n                        h=model.encoder", "                    for x,_ in data:\n                        _permit(context,permit_path,permit_sha256)\n                        h=model.encoder")
s=s.replace("                        require(h.dtype==torch.float32,'encoder must export original FP32 features')\n                        yield", "                        require(h.dtype==torch.float32,'encoder must export original FP32 features')\n                        _permit(context,permit_path,permit_sha256)\n                        yield")
s=s.replace("            results[role]=write_role(context,role,w,b,batches)\n        context.check(full=True);", "            results[role]=write_role(context,role,w,b,batches)\n            _permit(context,permit_path,permit_sha256)\n        _permit(context,permit_path,permit_sha256)\n        context.check(full=True);")
p.write_text(s);print('APPLIED_REVIEW_FIX',hashlib.sha256(p.read_bytes()).hexdigest())
