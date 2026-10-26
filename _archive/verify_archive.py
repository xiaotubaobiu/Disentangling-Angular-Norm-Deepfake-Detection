#!/usr/bin/env python3
"""Independent read-only verification against originals and optional remote clone."""
import hashlib,json,os,stat,subprocess,sys
from pathlib import Path
HERE=Path(__file__).parent
BASE=Path('/home/z/project_backups/2026-10-01-simplify')
DEST=Path(sys.argv[1]) if len(sys.argv)>1 else HERE/'archive.git'
REPOS={'legacy':BASE/'archive/deepfake_detection_legacy/.git','jev-nested':BASE/'nested-git/deepfake_visual_jev/experiments/2026-09-28_e4_antifakeprompt/upstream/git-metadata','outer':BASE/'outer-git'}
SOURCE={'.py','.pyi','.sh','.bash','.c','.h','.cpp','.hpp','.cu','.cuh','.js','.ts','.jsx','.tsx','.r','.m','.jl','.java','.rs','.go','.yaml','.yml','.json','.toml','.ini','.cfg','.conf','.xml','.md','.rst','.txt','.ipynb','.tex','.bib','.template','.in','.bat','.ps1','.sql'}
def git(repo,*args):return subprocess.check_output(['git','--git-dir='+str(repo),*args],env=dict(os.environ,GIT_OPTIONAL_LOCKS='0'))
def files(repo,tree):
    result={}
    for line in git(repo,'ls-tree','-rz',tree).split(b'\0'):
        if not line:continue
        info,name=line.split(b'\t',1);mode,kind,oid=info.decode().split();result[name.decode('utf8','surrogateescape')]=(mode,oid)
    return result
def source(path):
    p=Path(path)
    return p.suffix.lower() in SOURCE or p.name in {'Makefile','Dockerfile','.gitignore','LICENSE','LICENSE.txt','CODEOWNERS'}
checked=0;commits=0;errors=[];fs_checked=0;source_files=0
reachable=set(git(DEST,'rev-list','--all').decode().splitlines())
for name,repo in REPOS.items():
    mapping=json.loads((HERE/(name+'-commit-map.json')).read_text())
    original_commits=set(json.loads((HERE/(name+'-all-commits.json')).read_text()))
    if original_commits!=set(mapping):errors.append({'repo':name,'error':'original commit mapping incomplete'})
    for old,new in mapping.items():
        if new not in reachable:errors.append({'repo':name,'commit':old,'error':'mapped commit unreachable'})
        original=files(repo,old);archived=files(DEST,new)
        for path,pair in original.items():
            if source(path):
                checked+=1
                if archived.get(path)!=pair:errors.append({'repo':name,'commit':old,'path':path,'error':'source mode or blob OID changed/missing'})
        # Every retained file must have exactly the original blob hash and mode.
        for path,pair in archived.items():
            if original.get(path)!=pair:errors.append({'repo':name,'commit':old,'path':path,'error':'unexpected archive content'})
        commits+=1
for row in json.loads((HERE/'snapshots.json').read_text()):
    root=Path(row['root']);archived=files(DEST,row['commit'])
    manifest=json.loads((HERE/(row['label']+'-filesystem-manifest.json')).read_text())
    if set(archived)!={r['path'] for r in manifest}:errors.append({'snapshot':row['label'],'error':'snapshot inventory mismatch'})
    for rec in manifest:
        path=root/rec['path'];st=path.lstat();data=os.readlink(path).encode() if path.is_symlink() else path.read_bytes()
        oid=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        mode='120000' if path.is_symlink() else ('100755' if st.st_mode&0o111 else '100644')
        if archived.get(rec['path'])!=(mode,oid) or hashlib.sha256(data).hexdigest()!=rec['sha256']:
            errors.append({'snapshot':row['label'],'path':rec['path'],'error':'filesystem bytes/mode mismatch'})
        fs_checked+=1
    for dp,dirs,names in os.walk(root,followlinks=False):
        dirs[:]=[d for d in dirs if d not in {'.git','__pycache__','.pytest_cache','.mypy_cache','.ruff_cache'}]
        for name in names:
            path=Path(dp)/name;rel=str(path.relative_to(root))
            if source(rel):
                source_files+=1
                if rel not in archived:errors.append({'snapshot':row['label'],'path':rel,'error':'filesystem source missing'})
result={'archive_git':str(DEST),'original_commits_checked':commits,'historical_source_file_versions_checked':checked,'filesystem_files_byte_verified':fs_checked,'filesystem_source_coverage_checked':source_files,'error_count':len(errors),'errors':errors}
print(json.dumps(result,indent=2));(HERE/('remote-independent-verification.json' if len(sys.argv)>1 else 'independent-verification.json')).write_text(json.dumps(result,indent=2)+'\n')
raise SystemExit(bool(errors))
