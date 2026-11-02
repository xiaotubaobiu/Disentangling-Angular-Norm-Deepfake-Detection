"""Role-aware source audit. Audit never chooses experiment roles or touches data."""
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import datetime,hashlib,json,os,re,sqlite3,time
import numpy as np
import pandas as pd
from PIL import Image

METHODS={
 'FS':['fsgan','faceswap','simswap','inswap','blendface','uniface','mobileswap','e4s','facedancer'],
 'FR':['fomm','facevid2vid','wav2lip','MRAA','one_shot_free','pirender','tpsm','lia','danet','sadtalker','mcnet','hyperreenact'],
 'EFS':['VQGAN','StyleGAN2','StyleGAN3','StyleGANXL','sd2.1','ddim','rddm','pixart','DiT','SiT']}


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def save_json(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    os.replace(tmp,p)


def resolve_path(root,original,method,kind,released_role,label):
    root=Path(root);parts=Path(original.replace('\\','/')).parts
    if parts and parts[0]=='deepfakes_detection_datasets':parts=parts[1:]
    if not parts or '..' in parts:raise ValueError('Unsafe source path')
    if not label:return root.joinpath(*parts)
    physical_method=_physical_method(str(root),method,released_role)
    if kind=='EFS' and released_role=='test' and len(parts)>3 and parts[2]=='cdf':
        parts=parts[:3]+(_cdf_component(str(root),physical_method,parts[3]),)+parts[4:]
    if parts[0]=='DF40_train':
        candidates=[root.joinpath('DF40_train',physical_method,*parts[2:])]
    elif parts[0]=='DF40':
        if released_role=='train':
            tail=parts[3:] if len(parts)>2 and parts[2]=='ff' else parts[2:]
            candidates=[root.joinpath('DF40_train',physical_method,*tail)]
        else:candidates=[root.joinpath('DF40_test',physical_method,*parts[2:])]
    else:raise ValueError(f'Unknown fake prefix {parts[0]}')
    # Some RDDM installations keep the lower-case name; explicit alias only.
    if method=='rddm':
        candidates += [Path(str(p).replace('/RDDM/','/rddm/')) for p in candidates]
    if method=='rddm':
        for p in candidates:
            if p.parent.parent.is_dir():return p
    return candidates[0]


@lru_cache(maxsize=256)
def _physical_method(root,method,role):
    if method!='rddm':return method
    base=Path(root)/('DF40_train' if role=='train' else 'DF40_test')
    return 'RDDM' if (base/'RDDM').is_dir() else 'rddm'


@lru_cache(maxsize=256)
def _cdf_component(root,method,component):
    base=Path(root)/'DF40_test'/method/'cdf'
    if (base/component).is_dir():return component
    aliases={'Celeb-real':'Fake_from_Celeb-real','YouTube-real':'Fake_from_Youtube-real'}
    alternate=aliases.get(component,component)
    return alternate if (base/alternate).is_dir() else component


def select_eight(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r.get('domain',''),r['method_id'],r['video_id'])].append(r)
    out=[]
    for key,group in sorted(groups.items()):
        valid=[];seen=set()
        for r in sorted(group,key=lambda q:(q['frame_number'],q['image_path'])):
            if not r.get('decode_ok',True):continue
            h=r.get('rgb_sha256') or r['content_sha256']
            if h not in seen:valid.append(r);seen.add(h)
        if len(valid)>=8:
            indices=np.linspace(0,len(valid)-1,8).astype(int)
            out.extend(valid[i] for i in indices)
    return out


def source_ids_from_video(domain,kind,video):
    if domain!='ff' or kind=='EFS':return []
    match=re.match(r'^(\d{3})(?=_|$)(?:_(\d{3})(?=_|$))?',video)
    return [v for v in match.groups() if v] if match else []


def index_rows(config):
    root=Path(config['data_root']);out=[];sources={}
    for kind,methods in METHODS.items():
        for method in methods:
            print(f'index {method} ({kind})',flush=True)
            for domain in ['ff','cdf']:
                p=root/'dataset_json_update'/f'{method}_{domain}.json'
                obj=json.loads(p.read_text());sources[str(p)]=digest(p)
                if len(obj)!=1:raise ValueError(f'Unexpected index envelope {p}')
                for labelkey,roles in next(iter(obj.values())).items():
                    label=int(labelkey.endswith('_Fake'))
                    if not labelkey.endswith(('_Fake','_Real')):raise ValueError(labelkey)
                    for role,videos in roles.items():
                        if role not in ['train','test']:raise ValueError(f'Unexpected published role {role}')
                        if domain=='cdf' and role=='train' and videos:raise ValueError('CDF training pool prohibited')
                        for original_id,record in videos.items():
                            if record['label']!=labelkey:raise ValueError('Index label conflict')
                            for original in record['frames']:
                                path=resolve_path(root,original,method,kind,role,label)
                                video=path.parent.name
                                suffix=path.stem
                                frame=int(suffix) if suffix.isdigit() else -1
                                sources_id=source_ids_from_video(domain,kind if label else 'real',video)
                                out.append({'domain':domain,'released_role':role,'physical_role':'train' if '/DF40_train/' in str(path) else 'test' if '/DF40_test/' in str(path) else 'shared_real',
                                  'upstream_source_role':'unverified','experiment_role':'UNASSIGNED','label':label,'method_id':method if label else 'real',
                                  'index_method':method,'method_type':kind if label else 'real','evaluation_unit':'image' if label and kind=='EFS' else 'video_frame',
                                  'video_id':video if not(label and kind=='EFS') else '', 'storage_group_id':video if label and kind=='EFS' else '',
                                  'source_ids':json.dumps(sources_id),'driving_source_id':'unknown','generator_id':method if label and kind=='EFS' else '',
                                  'generation_batch_id':'unknown','image_path':str(path),'original_image_path':original,'original_index_id':original_id,
                                  'index_path':str(p),'frame_number':frame,'include':False,'exclusion_reason':'policy_not_frozen'})
    df=pd.DataFrame(out)
    # Canonical shared real rows: retain exact source-index provenance separately.
    identity=['domain','released_role','method_id','image_path']
    provenance=df[df.method_id=='real'][['domain','released_role','image_path','index_method','index_path','original_index_id']].copy()
    df=df.drop_duplicates(identity).reset_index(drop=True)
    return df,provenance,sources


def inspect_image(path):
    try:
        p=Path(path);before=p.stat();sha=digest(p)
        with Image.open(p) as im:
            rgb=im.convert('RGB');rgb.load();width,height=rgb.size
            pixel=hashlib.sha256(f'{width}x{height}:RGB:'.encode()+rgb.tobytes()).hexdigest()
        after=p.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Input changed during byte/RGB audit')
        return (path,sha,pixel,width,height,1,'',after.st_size,after.st_mtime_ns)
    except Exception as e:return (path,'','',0,0,0,f'{type(e).__name__}: {e}',0,0)


def cache_stat_matches(path,size,mtime_ns):
    """A metadata freshness check, not a proof that bytes never changed."""
    try:
        s=Path(path).stat()
        return s.st_size==size and s.st_mtime_ns==mtime_ns and Path(path).is_file()
    except OSError:return False


def audit(config,artifact_root):
    start=time.time();dest=Path(artifact_root)/'data';dest.mkdir(parents=True,exist_ok=True)
    df,provenance,hashes=index_rows(config)
    df.to_parquet(dest/'released_manifest.parquet',index=False);provenance.to_parquet(dest/'real_index_provenance.parquet',index=False)
    db=sqlite3.connect(dest/'image_audit.sqlite');db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS images(path TEXT PRIMARY KEY,sha TEXT,rgb TEXT,width INTEGER,height INTEGER,ok INTEGER,error TEXT,bytes INTEGER,mtime_ns INTEGER)')
    paths=sorted(set(df.image_path));current=set(paths)
    known={p for p,size,mtime in db.execute('SELECT path,bytes,mtime_ns FROM images WHERE ok=1') if p in current and cache_stat_matches(p,size,mtime)}
    todo=[x for x in paths if x not in known]
    save_json(dest/'audit_progress.json',{'status':'RUNNING','total_unique_paths':len(paths),'cached':len(known),'remaining':len(todo),'workers':config['audit_workers'],'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    with ThreadPoolExecutor(max_workers=config['audit_workers']) as executor:
        for i,result in enumerate(executor.map(inspect_image,todo),1):
            db.execute('INSERT OR REPLACE INTO images VALUES(?,?,?,?,?,?,?,?,?)',result)
            if i%1000==0:
                db.commit();save_json(dest/'audit_progress.json',{'status':'RUNNING','total_unique_paths':len(paths),'audited':len(known)+i,'elapsed_seconds':time.time()-start})
                print(f'audit {len(known)+i}/{len(paths)} elapsed={time.time()-start:.0f}s',flush=True)
    db.commit();checks=pd.read_sql_query('SELECT * FROM images',db);db.close()
    checks=checks.rename(columns={'path':'image_path','sha':'content_sha256','rgb':'rgb_sha256','ok':'decode_ok'})
    joined=df.merge(checks,on='image_path',validate='many_to_one');joined.to_parquet(dest/'audited_manifest.parquet',index=False)
    errors=joined[~joined.decode_ok.astype(bool)];errors.to_parquet(dest/'invalid_inputs.parquet',index=False)
    valid=joined[joined.decode_ok.astype(bool)]
    conflicts=[]
    for hashcol in ['content_sha256','rgb_sha256']:
        stats=valid.groupby(hashcol).agg(labels=('label','nunique'),roles=('released_role','nunique'),domains=('domain','nunique'),rows=('image_path','size'))
        selected=set(stats.index[(stats.labels>1)|(stats.roles>1)])
        q=valid[valid[hashcol].isin(selected)].copy();q['audit_hash_type']=hashcol;conflicts.append(q)
    pd.concat(conflicts,ignore_index=True).to_parquet(dest/'cross_role_or_label_conflicts.parquet',index=False)
    counts=joined.groupby(['domain','released_role','method_id','evaluation_unit']).agg(rows=('image_path','size'),decode_ok=('decode_ok','sum'),unique_bytes=('content_sha256','nunique')).reset_index()
    counts.to_csv(dest/'published_counts.csv',index=False)
    summary={'status':'AUDITED_POLICY_NOT_FROZEN','source_indices_sha256':hashes,'canonical_rows':len(df),'unique_paths':len(paths),'invalid_rows':len(errors),
      'cross_role_or_label_rows_by_hash':{name:len(q) for name,q in zip(['bytes','RGB'],conflicts)},'method_count':31,
      'roles_changed':False,'original_data_modified':False,'target_scores_used':False,'elapsed_seconds':time.time()-start,
      'policy_gate':'Do not train until duplicate disposition/source-role policy and all31method coverage are frozen; source calibration remains unverified.'}
    save_json(dest/'audit_summary.json',summary);save_json(dest/'audit_progress.json',{**summary,'status':'COMPLETE','audit_complete':True})
    print(json.dumps(summary),flush=True)
    return summary
