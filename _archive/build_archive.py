#!/usr/bin/env python3
"""Read original Git objects; build a self-contained source archive elsewhere."""
import collections, hashlib, json, os, re, stat, subprocess, tarfile, time
from pathlib import Path

BASE = Path('/home/z/project_backups/2026-10-01-simplify')
OUT = Path('/tmp/jev-archive-20261026/legacy-staging')
DEST = OUT/'archive.git'
LEGACY = BASE/'archive/deepfake_detection_legacy'
REPOS = {'legacy': LEGACY/'.git', 'jev-nested': BASE/'nested-git/deepfake_visual_jev/experiments/2026-09-28_e4_antifakeprompt/upstream/git-metadata', 'outer': BASE/'outer-git'}
ENV = dict(os.environ, GIT_OPTIONAL_LOCKS='0', GIT_CONFIG_NOSYSTEM='1')
def git(repo, *args, data=None, env=None):
    return subprocess.check_output(['git', '--git-dir='+str(repo), *args], input=data, env=env or ENV)
def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2)+'\n')
def hobj(data, kind='blob'):
    return git(DEST, 'hash-object','-w','-t',kind,'--stdin',data=data).decode().strip()

SOURCE_EXT = {'.py','.pyi','.sh','.bash','.zsh','.c','.h','.cpp','.hpp','.cu','.cuh','.rs','.go','.java','.js','.jsx','.ts','.tsx','.vue','.svelte','.r','.rmd','.jl','.m','.f','.f90','.ipynb','.yaml','.yml','.toml','.ini','.cfg','.conf','.json','.xml','.md','.rst','.tex','.bib','.txt','.template','.in','.bat','.ps1','.sql'}
EXCLUDED_EXT = {'.pth','.pt','.ckpt','.bin','.safetensors','.onnx','.npy','.npz','.parquet','.dat','.pkl','.pickle','.h5','.hdf5','.pyc','.pyo','.so','.o','.a','.whl','.zip','.tar','.mp4','.avi','.mov','.webm','.tmp','.lock'}
SECRET_NAMES = {'.env','.netrc','.npmrc','.pypirc','credentials.json','credentials','id_rsa','id_ed25519','service-account.json'}
SECRET_PATTERNS = [
    ('github-token', re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})')),
    ('huggingface-token',re.compile(rb'\bhf_[A-Za-z0-9]{25,}\b')),
    ('openai-token',re.compile(rb'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}\b')),
    ('google-api-key',re.compile(rb'\bAIza[0-9A-Za-z_-]{35}\b')),
    ('aws-access-key',re.compile(rb'\b(?:AKIA|ASIA)[0-9A-Z]{16}\b')),
    ('private-key',re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----')),
    ('credential-url',re.compile(rb'https?://[^\s/:@]{1,80}:[^\s/@]{8,}@')),
]
exclusions=[]; secret_findings=[]; tree_checks=[]; snapshots=[]; copied=set(); seen_original_blobs=collections.defaultdict(set)
def scan(data, context):
    for label,pattern in SECRET_PATTERNS:
        if pattern.search(data): secret_findings.append({'context':context,'pattern':label})
def reason(path, size, mode='100644'):
    pp=Path(path); parts=pp.parts
    if '.git' in parts: return 'git-internal-metadata'
    if any(p in {'__pycache__','.pytest_cache','.mypy_cache','.ruff_cache'} for p in parts): return 'runtime-cache'
    if pp.name.lower() in SECRET_NAMES or pp.name.lower().endswith(('.pem','.key')): return 'credential-like-filename'
    if mode=='120000': return None
    if pp.suffix.lower() in EXCLUDED_EXT:return 'model-data-or-runtime-binary'
    if size>20_000_000:
        if pp.suffix.lower() in SOURCE_EXT: raise RuntimeError('Oversized possible source needs review: '+path)
        return 'large-nonsource-artifact'
    return None

class Original:
    def __init__(self,name,repo):
        self.name,self.repo=name,repo
        self.inventory={}
        for line in (OUT/(name+'-objects.txt')).read_text().splitlines():
            oid,kind,size=line.split();self.inventory[oid]=(kind,int(size))
        self.proc=subprocess.Popen(['git','--git-dir='+str(repo),'cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,env=ENV)
        self.treecache={};self.commitmap={};self.originalcommits={};self.missingparents=[]
        self.refs=[]
        for line in git(repo,'for-each-ref','--format=%(objectname) %(objecttype) %(refname)').decode().splitlines():
            oid,kind,ref=line.split(' ',2);self.refs.append({'oid':oid,'type':kind,'ref':ref})
        self.head=git(repo,'rev-parse','HEAD').decode().strip()
    def read(self,oid):
        self.proc.stdin.write((oid+'\n').encode());self.proc.stdin.flush()
        hdr=self.proc.stdout.readline().split()
        if len(hdr)!=3:raise RuntimeError(hdr)
        data=self.proc.stdout.read(int(hdr[2]));assert self.proc.stdout.read(1)==b'\n';return data
    def blob(self,oid,path,mode):
        size=self.inventory[oid][1];why=reason(path,size,mode)
        if why:
            exclusions.append({'repo':self.name,'path':path,'oid':oid,'bytes':size,'reason':why});return None
        seen_original_blobs[self.name].add(oid)
        if oid not in copied:
            data=self.read(oid);scan(data, self.name+':'+path+':'+oid)
            assert hobj(data)==oid;copied.add(oid)
        return oid
    def tree(self,oid,prefix='',outer_scope=False):
        key=(oid,prefix if outer_scope else '')
        if key in self.treecache:return self.treecache[key]
        raw=self.read(oid);pos=0;entries=[];selected=[]
        while pos<len(raw):
            j=raw.index(b' ',pos);mode=raw[pos:j].decode();k=raw.index(b'\0',j);name=raw[j+1:k].decode('utf8','surrogateescape');child=raw[k+1:k+21].hex();pos=k+21
            full=prefix+name
            if outer_scope and not prefix and name not in {'.gitignore','ORGANIZATION.txt','deepfake_detection','deepfake_feature_geometry','deepfake_visual_jev','archive','docs','scripts'}:
                exclusions.append({'repo':self.name,'path':full,'oid':child,'reason':'outside-detection-scope'});continue
            if outer_scope and prefix=='archive/' and name!='deepfake_detection_legacy':continue
            if mode=='40000':
                new=self.tree(child,full+'/',outer_scope)
                if new=='4b825dc642cb6eb9a060e54bf8d69288fbee4904':continue
            elif mode=='160000':new=child
            else:
                new=self.blob(child,full,mode)
                if not new:continue
            entries.append(mode.encode()+b' '+name.encode('utf8','surrogateescape')+b'\0'+bytes.fromhex(new));selected.append((mode,name,new))
        new=hobj(b''.join(entries),'tree');self.treecache[key]=new
        tree_checks.append({'repo':self.name,'original':oid,'filtered':new,'prefix':prefix,'entries':selected})
        return new
    def commit(self,oid):
        if oid in self.commitmap:return self.commitmap[oid]
        raw=self.read(oid);self.originalcommits[oid]=raw.decode('utf8','surrogateescape');scan(raw,self.name+':commit:'+oid)
        hdr,msg=raw.split(b'\n\n',1);new=[];skip=False
        for line in hdr.splitlines():
            if line.startswith(b' '):
                if not skip:new.append(line)
                continue
            skip=False
            if line.startswith(b'tree '):new.append(b'tree '+self.tree(line[5:].decode(),outer_scope=self.name=='outer').encode())
            elif line.startswith(b'parent '):
                par=line[7:].decode()
                if par in self.inventory:new.append(b'parent '+self.commit(par).encode())
                else:self.missingparents.append({'commit':oid,'missing_parent':par})
            elif line.startswith((b'gpgsig ',b'mergetag ')):skip=True
            else:new.append(line)
        rewritten=hobj(b'\n'.join(new)+b'\n\n'+msg,'commit');self.commitmap[oid]=rewritten;return rewritten
    def index(self,path,label):
        env=dict(ENV,GIT_INDEX_FILE=str(path))
        records=git(self.repo,'ls-files','--stage','-z',env=env);files={};rawrecords=[]
        for rec in records.split(b'\0'):
            if not rec:continue
            info,name=rec.split(b'\t',1);mode,oid,stage=info.decode().split();name=name.decode('utf8','surrogateescape');rawrecords.append({'mode':mode,'oid':oid,'stage':stage,'path':name})
            keep=self.blob(oid,name,mode) if mode!='160000' else oid
            if keep:files[(('conflict-stage-'+stage+'/') if stage!='0' else '')+name]=(mode,keep)
        dump(OUT/(label+'-original-index.json'),rawrecords)
        return make_tree(files)

def make_tree(files):
    root={}
    for path,val in files.items():
        d=root;parts=path.split('/')
        for p in parts[:-1]:d=d.setdefault(p,{})
        d[parts[-1]]=val
    def write(d):
        ents=[]
        for name,v in d.items():
            mode,oid=('40000',write(v)) if isinstance(v,dict) else v
            ents.append((name.encode('utf8','surrogateescape'),mode,oid))
        ents.sort(key=lambda x:x[0]+(b'/' if x[1]=='40000' else b''))
        return hobj(b''.join(mode.encode()+b' '+name+b'\0'+bytes.fromhex(oid) for name,mode,oid in ents),'tree')
    return write(root)
def commit_snapshot(tree,label,parents=()):
    headers=[b'tree '+tree.encode()]+[b'parent '+p.encode() for p in dict.fromkeys(parents)]
    ident=b'Archive Preservation <archive@localhost> 1793001600 +0000'
    headers += [b'author '+ident,b'committer '+ident]
    return hobj(b'\n'.join(headers)+b'\n\n'+label.encode()+b'\n','commit')
def fs_snapshot(root,label,parents=()):
    files={};manifest=[]
    for dp,dirs,names in os.walk(root,followlinks=False):
        dirs[:]=[d for d in dirs if d not in {'.git','__pycache__','.pytest_cache','.mypy_cache','.ruff_cache'}]
        for d in dirs[:]:
            if (Path(dp)/d).is_symlink():names.append(d);dirs.remove(d)
        for name in names:
            p=Path(dp)/name;rel=str(p.relative_to(root));s=p.lstat();mode='120000' if stat.S_ISLNK(s.st_mode) else ('100755' if s.st_mode&0o111 else '100644')
            why=reason(rel,s.st_size,mode)
            if why:
                exclusions.append({'repo':label,'path':rel,'bytes':s.st_size,'reason':why});continue
            data=os.readlink(p).encode() if mode=='120000' else p.read_bytes();scan(data,label+':'+rel);oid=hobj(data)
            files[rel]=(mode,oid);manifest.append({'path':rel,'mode':mode,'oid':oid,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    tree=make_tree(files);commit=commit_snapshot(tree,'Preserve uncommitted filesystem snapshot: '+label,parents)
    dump(OUT/(label+'-filesystem-manifest.json'),manifest);snapshots.append({'label':label,'root':str(root),'tree':tree,'commit':commit,'files':len(files)})
    return commit,files

def main():
    if DEST.exists():raise RuntimeError('Destination already exists; refuse overwrite')
    subprocess.run(['git','init','--bare',str(DEST)],check=True,capture_output=True)
    originals={n:Original(n,p) for n,p in REPOS.items()};retain=[];branches={};summary={}
    for name,o in originals.items():
        for oid,(kind,size) in o.inventory.items():
            if kind=='commit':o.commit(oid)
        tip_commits=set(o.commitmap.values())
        # Every object is considered: this includes unreachable commit and tree objects.
        child_trees=set();commit_trees=set()
        for oid,(kind,size) in o.inventory.items():
            if kind=='commit':commit_trees.add(o.read(oid).splitlines()[0][5:].decode())
            if kind=='tree':
                data=o.read(oid);pos=0
                while pos<len(data):
                    j=data.index(b' ',pos);k=data.index(b'\0',j);mode=data[pos:j];child=data[k+1:k+21].hex();pos=k+21
                    if mode==b'40000':child_trees.add(child)
        tree_roots=sorted(oid for oid,(kind,size) in o.inventory.items() if kind=='tree' and oid not in child_trees and oid not in commit_trees)
        orphan_trees=[]
        for oid in tree_roots:
            new=o.tree(oid,outer_scope=name=='outer');c=commit_snapshot(new,'Preserve standalone original tree '+name+':'+oid)
            orphan_trees.append({'original_tree':oid,'tree':new,'commit':c});tip_commits.add(c)
        index_files=[(o.repo/'index',name+'-index')]
        if name=='legacy':index_files += [(p/'index',name+'-'+p.name+'-index') for p in (o.repo/'worktrees').iterdir() if (p/'index').exists()]
        indexes=[]
        for index,label in index_files:
            tree=o.index(index,label);c=commit_snapshot(tree,'Preserve original staged index '+label,[o.commitmap[o.head]])
            indexes.append({'label':label,'tree':tree,'commit':c});tip_commits.add(c)
        orphanfiles={}
        # Unnamed source-like blobs are retained under their original OID even if
        # no surviving tree/index provides a pathname. Outer mixed-project blobs
        # remain exclusively in the original repository, which is not deleted.
        if name!='outer':
            for oid,(kind,size) in o.inventory.items():
                if kind!='blob' or oid in seen_original_blobs[name] or size>20_000_000:continue
                data=o.read(oid)
                if b'\0' in data:continue
                try:data.decode('utf8')
                except UnicodeDecodeError:continue
                scan(data,name+':unreferenced-text:'+oid);assert hobj(data)==oid
                orphanfiles['recovered-unreferenced-text-blobs/'+oid+'.txt']=('100644',oid)
        if orphanfiles:
            tree=make_tree(orphanfiles);c=commit_snapshot(tree,'Preserve unnamed text blobs from '+name);tip_commits.add(c)
        # Final merge below makes every original commit reachable, including
        # commits previously reachable only through reflogs or no refs at all.
        retain.extend(sorted(tip_commits))
        dump(OUT/(name+'-commit-map.json'),o.commitmap)
        dump(OUT/(name+'-original-commits.json'),o.originalcommits)
        dump(OUT/(name+'-original-refs.json'),{'head':o.head,'refs':o.refs,'shallow':(o.repo/'shallow').read_text() if (o.repo/'shallow').exists() else '', 'missing_parents':o.missingparents})
        dump(OUT/(name+'-standalone-trees.json'),orphan_trees)
        summary[name]={'original_commits':len(o.commitmap),'original_refs':len(o.refs),'standalone_tree_snapshots':len(orphan_trees),'index_snapshots':indexes,'recovered_unnamed_text_blobs':len(orphanfiles),'missing_parents':o.missingparents}
        if name=='jev-nested':branches['archive/legacy-jev-upstream-20261026']=o.commitmap[o.head]
        if name=='outer':branches['archive/legacy-outer-detection-20261026']=commit_snapshot(o.tree(o.read(o.head).splitlines()[0][5:].decode(),outer_scope=True),'Preserve outer repository detection-scoped history',sorted(tip_commits))
    latest,latest_files=fs_snapshot(LEGACY,'legacy-working-directory',[originals['legacy'].commitmap[originals['legacy'].head]])
    retain.append(latest)
    for p in sorted((LEGACY/'.claude/worktrees').iterdir()):
        if not p.is_dir():continue
        c,_=fs_snapshot(p,'legacy-worktree-'+p.name,[originals['legacy'].commitmap[originals['legacy'].head]])
        branches['archive/legacy-worktree-'+p.name+'-20261026']=c;retain.append(c)
    # Preserve deleted configuration files embedded in the small historical tar.
    tar=LEGACY/'.deleted_configs_backup_20260729.tar.gz'
    if tar.exists():
        files={}
        with tarfile.open(tar,'r:gz') as tf:
            for member in tf.getmembers():
                if member.isfile():
                    data=tf.extractfile(member).read();scan(data,'deleted-config-tar:'+member.name)
                    files[member.name]=('100644',hobj(data))
        c=commit_snapshot(make_tree(files),'Recover deleted configuration files from historical tar');retain.append(c);summary['deleted_config_tar_files']=len(files)
    # Source-related cleanup originals only; do not read/copy deidentification.
    edited=BASE/'edited-originals'
    for p in sorted(edited.iterdir()):
        if p.name in {'deepfake_feature_geometry','deepfake_visual_jev'}:
            c,_=fs_snapshot(p,'edited-originals-'+p.name);retain.append(c)
    dump(OUT/'snapshots.json',snapshots);dump(OUT/'exclusions.json',exclusions);dump(OUT/'secret-scan.json',secret_findings);dump(OUT/'tree-checks.json',tree_checks);dump(OUT/'summary.json',summary)
    if secret_findings:raise RuntimeError('Potential secrets detected; see secret-scan.json (paths only)')
    # Exact filtered tree/blob comparison: no included byte content is rewritten.
    verified=0
    for row in tree_checks:
        raw=git(DEST,'cat-file','tree',row['filtered']);entries=[];pos=0
        while pos<len(raw):
            j=raw.index(b' ',pos);k=raw.index(b'\0',j);entries.append((raw[pos:j].decode(),raw[j+1:k].decode('utf8','surrogateescape'),raw[k+1:k+21].hex()));pos=k+21
        assert entries==row['entries'];verified+=1
    audit_files={}
    for p in OUT.iterdir():
        if p.is_file() and p.suffix in {'.json','.py','.txt'}:
            data=p.read_bytes();scan(data,'archive-metadata:'+p.name);audit_files['_archive/'+p.name]=('100644',hobj(data))
    for name in ['RESTORE.txt','ORGANIZATION.txt','metadata_moves.json','repository_history.json','missing_original_tracked_source.json','verification.json','path_edits.json']:
        data=(BASE/name).read_bytes();scan(data,'archive-original-records:'+name);audit_files['_archive/original-records/'+name]=('100644',hobj(data))
    readme='''# Legacy source-history archive (2026-10-26)\n\nAll local legacy commit objects, including previously unreachable commits, have\nbeen reconstructed with original source bytes and file modes. Models, numeric\narray datasets, caches, and large nonsource artifacts were intentionally omitted.\nAll exclusions are listed in `_archive/exclusions.json`. Original -> archived\ncommit maps, original refs and commit metadata are in `_archive/`. Every mapped\ncommit is reachable through this archive commit's parents. Standalone tree\nobjects, original indexes (including worktree staged state), unnamed text blobs,\nand existing worktree filesystem snapshots are also reachable.\n\nUse `git show <mapped-commit>:<path>` or `git worktree add <dir> <mapped-commit>`\nto recover a historical version. Use `_archive/snapshots.json` for uncommitted\nfilesystem versions and `_archive/summary.json` for index snapshots. Original\nblob OIDs are preserved: source content was not rewritten. The small deleted\nconfiguration tar is retained and its contents have an additional commit.\n\nThe JEV AntiFakePrompt upstream repository was already shallow: its single\nlocally available commit is preserved; absent upstream parents are recorded.\nThe third legacy worktree no longer exists on disk; its HEAD and staged index\nare preserved. Outer Git contains other projects, so only detection-scoped\ntrees are copied and the original outer Git must remain locally. Deidentification\nrepositories and working directories were neither changed nor copied.\n\nThis is a source archive, not a backup of model weights or experiment arrays.\n'''
    latest_files.update(audit_files);latest_files['_archive/README.md']=('100644',hobj(readme.encode()))
    if secret_findings:dump(OUT/'secret-scan.json',secret_findings);raise RuntimeError('Potential secret in archive metadata')
    archive=commit_snapshot(make_tree(latest_files),'Archive every recoverable legacy source version and uncommitted snapshot',retain)
    branches['archive/legacy-history-20261026']=archive
    for name,oid in branches.items():git(DEST,'update-ref','refs/heads/'+name,oid)
    git(DEST,'symbolic-ref','HEAD','refs/heads/archive/legacy-history-20261026')
    reachable=set(git(DEST,'rev-list','--all').decode().splitlines())
    for o in originals.values():assert set(o.commitmap.values())<=reachable
    fsck=subprocess.run(['git','--git-dir='+str(DEST),'fsck','--full'],capture_output=True,text=True,env=ENV)
    (OUT/'fsck.txt').write_text(fsck.stdout+fsck.stderr);assert fsck.returncode==0
    verification={'branches':branches,'verified_filtered_trees':verified,'all_mapped_commits_reachable':True,'reachable_commits':len(reachable),'secret_findings':len(secret_findings),'fsck_exit':fsck.returncode,'original_repositories_modified':False}
    dump(OUT/'verification-final.json',verification)
    print(json.dumps(verification,indent=2))
if __name__=='__main__':main()
