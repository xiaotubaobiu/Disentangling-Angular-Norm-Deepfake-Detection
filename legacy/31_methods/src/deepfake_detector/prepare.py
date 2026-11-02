"""CPU-only public initialization and environment preparation; no target access."""
from pathlib import Path
import datetime,hashlib,importlib.metadata,json,os,re,sys,urllib.request
from .manifest import digest,save_json


def validate_weight(path,expected_prefix=None):
    sha=digest(path)
    if expected_prefix and not sha.startswith(expected_prefix):raise ValueError(f'Initialization checksum mismatch: {path}')
    return sha


def source_split_roles(splits):
    roles={}
    for role,pairs in splits.items():
        for pair in pairs:
            for source in pair:
                if source in roles and roles[source]!=role:raise ValueError('Conflicting original-source split roles')
                roles[source]=role
    return roles


def fetch_public(url,path,expected_prefix=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():return validate_weight(path,expected_prefix)
    if not url.startswith(('https://dl.fbaipublicfiles.com/','https://download.pytorch.org/','https://raw.githubusercontent.com/')):
        raise ValueError('Only explicitly public initialization/source-metadata hosts permitted')
    tmp=path.with_name(path.name+'.download');n=0
    with urllib.request.urlopen(url,timeout=60) as response,tmp.open('wb') as f:
        while True:
            block=response.read(1024*1024)
            if not block:break
            f.write(block);n+=len(block)
            if n%(50*1024*1024)==0:print(f'download {path.name}: {n//1024//1024}MiB',flush=True)
    sha=validate_weight(tmp,expected_prefix);os.replace(tmp,path);return sha


def prepare(config,root):
    root=Path(root);artifact=Path(config['artifact_root']);out=artifact/'weights';out.mkdir(exist_ok=True)
    import clip
    from torchvision.models import EfficientNet_B0_Weights
    clip_sha=clip.clip._MODELS['ViT-B/16'].split('/')[-2]
    public={'clip_vit_b16':{'path':config['backbones']['clip_vit_b16']['weight_path'],'source':clip.clip._MODELS['ViT-B/16'],'expected':clip_sha},
      'dinov2_vit_b14':{'path':str(out/'dinov2_vitb14_pretrain.pth'),'source':'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth','expected':None},
      'efficientnet_b0':{'path':str(out/'efficientnet_b0.pth'),'source':EfficientNet_B0_Weights.DEFAULT.url,'expected':re.search(r'-([0-9a-f]+)\.pth',EfficientNet_B0_Weights.DEFAULT.url).group(1)}}
    for name,v in public.items():
        if name=='clip_vit_b16':v['sha256']=validate_weight(v['path'],v['expected'])
        else:v['sha256']=fetch_public(v['source'],v['path'],v['expected'])
        config['backbones'][name]['weight_path']=v['path'];config['backbones'][name]['weight_sha256']=v['sha256'];print('initialization_ready',name,v['sha256'],flush=True)
    split_dir=artifact/'data/source_splits';splits={};records={}
    for role in ['train','val','test']:
        url=f'https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/{role}.json';path=split_dir/f'{role}.json'
        sha=fetch_public(url,path);splits[role]=json.loads(path.read_text());records[role]={'URL':url,'sha256':sha,'source_ids':sum(len(x) for x in splits[role])}
    save_json(split_dir/'source_roles.json',source_split_roles(splits));save_json(split_dir/'provenance.json',records)
    packages=['torch','torchvision','numpy','scipy','Pillow','pandas','pyarrow','scikit-learn','albumentations','timm','openai-clip','pytest','opencv-python-headless']
    versions={name:importlib.metadata.version(name) for name in packages};save_json(artifact/'environment_versions.json',{'python':sys.version,'packages':versions,'CUDA_initialized':__import__('torch').cuda.is_initialized()})
    (root/'requirements.lock.txt').write_text('\n'.join(f'{k}=={v}' for k,v in versions.items())+'\n')
    from .training import augmentation
    augmentation_defaults={name:augmentation(name,7).to_dict() for name in config['backbones']};save_json(artifact/'augmentation_defaults.json',augmentation_defaults)
    config['augmentation_defaults_sha256']=digest(artifact/'augmentation_defaults.json');config['environment_versions_sha256']=digest(artifact/'environment_versions.json')
    save_json(root/'configs/v1.1.json',config)
    report={'status':'PUBLIC_ASSETS_READY','weights':public,'original_FF_source_splits':records,'config_sha256':digest(root/'configs/v1.1.json'),
       'data_roles_changed':False,'GPU_allocated':False,'note':'Independent calibration is not certified by merely downloading source split annotations.'}
    save_json(artifact/'prepare_complete.json',report);print(json.dumps(report),flush=True)
    return report
