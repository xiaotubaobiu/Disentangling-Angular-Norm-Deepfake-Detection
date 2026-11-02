"""Journal H: RetinaFace-five/336 then independent dlib-68/224 histories.

Only explicit local checkpoints are loaded. RGB inputs and pixel-centre forward
affines are used throughout; saved A PNG bytes are read before full B inference.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import os
import fcntl
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RETINAFACE_SOURCE = REPOSITORY_ROOT/'third_party/h_preprocessing/Pytorch_Retinaface'
RETINAFACE_COMMIT = 'b984b4b775b2c4dced95c1eadd195a5c7d32a60b'
RETINAFACE_SHA256 = '2979b33ffafda5d74b6948cd7a5b9a7a62f62b949cef24e95fd15d2883a65220'
DLIB68_SHA256 = 'fbdc2cb80eb9aa7a758672cbfdda32ba6300efe9b6e6c7a299ff7e736b11b92f'
TEMPLATE_112 = np.asarray([[38.2946,51.6963],[73.5318,51.5014],
                          [56.0252,71.7366],[41.5493,92.3655],
                          [70.7299,92.2041]],dtype=np.float64)
VERSIONS = ('A','B','AB')
METHODS = ('Deepfakes','Face2Face','FaceSwap','NeuralTextures')


class GeometryFailure(ValueError):
    """A prespecified frame/video eligibility failure, retained in manifests."""


@dataclass
class Face:
    bbox: np.ndarray
    score: float
    landmarks: np.ndarray


@dataclass
class MotherFrame:
    index: int
    timestamp: float
    rgb: np.ndarray
    rgb_sha256: str


@dataclass
class Crop:
    rgb: np.ndarray
    valid_mask: np.ndarray
    matrix: np.ndarray

    @property
    def padding_ratio(self):
        return float(1-np.count_nonzero(self.valid_mask)/self.valid_mask.size)


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def rgb_hash(rgb):
    rgb=np.ascontiguousarray(rgb,dtype=np.uint8)
    return hashlib.sha256(str(rgb.shape).encode()+b'\0'+rgb.tobytes()).hexdigest()


def mother_id(video_uid,mother):
    return f'{video_uid}/mother/{mother.index:09d}/{mother.rgb_sha256}'


def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n')
    os.replace(temporary,path)


def read_rgb(path):
    bgr=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if bgr is None: raise GeometryFailure('png_read_failed')
    return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)


def select_face(faces,threshold=0.):
    valid=[]
    for face in faces:
        box=np.asarray(face.bbox,dtype=float)
        if (box.shape==(4,) and np.isfinite(box).all() and box[2]>box[0] and box[3]>box[1]
                and np.isfinite(face.score) and face.score>=threshold): valid.append(face)
    if not valid: raise GeometryFailure('no_qualified_face')
    def key(face):
        x1,y1,x2,y2=map(float,face.bbox)
        return (-(x2-x1)*(y2-y1),-float(face.score),x1,y1,x2,y2,
                *np.nan_to_num(face.landmarks,nan=np.inf).ravel().tolist())
    return min(valid,key=key),len(valid)


def similarity_matrix(source,target):
    source=np.asarray(source,dtype=np.float64); target=np.asarray(target,dtype=np.float64)
    if source.shape!=target.shape or source.ndim!=2 or source.shape[1]!=2 or not np.isfinite(source).all():
        raise GeometryFailure('nonfinite_or_missing_landmarks')
    if np.linalg.matrix_rank(source-source.mean(0),tol=1e-8)<2:
        raise GeometryFailure('degenerate_landmarks')
    design=np.zeros((source.size,4),dtype=np.float64)
    design[0::2,0]=source[:,0]; design[0::2,1]=-source[:,1]; design[0::2,2]=1
    design[1::2,0]=source[:,1]; design[1::2,1]=source[:,0]; design[1::2,3]=1
    a,b,tx,ty=np.linalg.lstsq(design,target.ravel(),rcond=None)[0]
    matrix=np.asarray([[a,-b,tx],[b,a,ty]])
    if not np.isfinite(matrix).all() or a*a+b*b<1e-12: raise GeometryFailure('degenerate_affine')
    return matrix


def a_alignment_matrix(face):
    points=np.asarray(face.landmarks,dtype=np.float64)
    if points.shape!=(5,2): raise GeometryFailure('A_requires_5_landmarks')
    return similarity_matrix(points,TEMPLATE_112*3)


def b_alignment_matrix(face):
    """dlib's canonical 68-point chip at size224/padding.25, forward affine.

    Mirrors dlib get_mapping_to_chip: rotate extraction-rectangle corners by
    chip.angle, then map them to output corner pixel centres. Rendering remains
    OpenCV bicubic so the saved affine exactly determines RGB and mask warps.
    """
    import dlib
    points=np.asarray(face.landmarks,dtype=np.float64)
    if points.shape!=(68,2) or not np.isfinite(points).all():
        raise GeometryFailure('B_requires_68_finite_landmarks')
    if np.linalg.matrix_rank(points-points.mean(0),tol=1e-8)<2:
        raise GeometryFailure('degenerate_68_landmarks')
    x1,y1,x2,y2=np.rint(face.bbox).astype(int)
    shape=dlib.full_object_detection(dlib.rectangle(int(x1),int(y1),int(x2),int(y2)),
        [dlib.point(int(x),int(y)) for x,y in np.rint(points).astype(int)])
    chip=dlib.get_face_chip_details(shape,size=224,padding=.25)
    rect=chip.rect
    corners=np.array([[rect.left(),rect.top()],[rect.right(),rect.top()],
                      [rect.right(),rect.bottom()]],dtype=np.float64)
    centre=np.array([(rect.left()+rect.right())/2,(rect.top()+rect.bottom())/2])
    c,s=np.cos(chip.angle),np.sin(chip.angle)
    rotated=(corners-centre)@np.array([[c,s],[-s,c]])+centre
    try:
        matrix=np.linalg.solve(np.c_[rotated,np.ones(3)],np.array([[0,0],[223,0],[223,223]],float)).T
    except np.linalg.LinAlgError as exc: raise GeometryFailure('degenerate_68_affine') from exc
    if not np.isfinite(matrix).all() or np.linalg.det(matrix[:,:2])<1e-12:
        raise GeometryFailure('degenerate_68_affine')
    return matrix


def warp(rgb,matrix,size,valid_mask=None):
    matrix=np.asarray(matrix,dtype=np.float64)
    if matrix.shape!=(2,3) or not np.isfinite(matrix).all(): raise GeometryFailure('invalid_affine')
    if valid_mask is None: valid_mask=np.full(rgb.shape[:2],255,np.uint8)
    if valid_mask.shape!=rgb.shape[:2] or not np.isin(valid_mask,[0,255]).all():
        raise GeometryFailure('invalid_inherited_mask')
    output=cv2.warpAffine(rgb,matrix,(size,size),flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    mask=cv2.warpAffine(valid_mask,matrix,(size,size),flags=cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    return Crop(output,mask,matrix)


def save_crop(crop,directory,version):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'{version}.png'; mask_path=directory/f'{version}_mask.png'
    params=[cv2.IMWRITE_PNG_COMPRESSION,3]
    if not cv2.imwrite(str(path),cv2.cvtColor(crop.rgb,cv2.COLOR_RGB2BGR),params):
        raise GeometryFailure('png_write_failed')
    if not cv2.imwrite(str(mask_path),crop.valid_mask,params): raise GeometryFailure('mask_write_failed')
    return {'ok':True,'path':str(path),'mask_path':str(mask_path),
        'rgb_sha256':rgb_hash(crop.rgb),'byte_sha256':sha256_file(path),
        'mask_byte_sha256':sha256_file(mask_path),'matrix':crop.matrix.tolist(),
        'padding_ratio':crop.padding_ratio,'png_bytes':path.stat().st_size,
        'mask_bytes':mask_path.stat().st_size,'shape':list(crop.rgb.shape)}


def face_json(face,count):
    return {'bbox':np.asarray(face.bbox).tolist(),'score':float(face.score),
            'landmarks':np.asarray(face.landmarks).tolist(),
            'qualified_face_count':count,'multiface':count>1}


def process_frame(mother,detector_a,detector_b,directory):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    path=directory/'mother.png'
    if not cv2.imwrite(str(path),cv2.cvtColor(mother.rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_PNG_COMPRESSION,3]):
        raise GeometryFailure('mother_png_write_failed')
    result={'mother_frame_index':mother.index,'mother_timestamp_seconds':mother.timestamp,
        'mother_rgb_sha256':mother.rgb_sha256,'mother_path':str(path),
        'mother_byte_sha256':sha256_file(path),'mother_png_bytes':path.stat().st_size,
        'mother_shape':list(mother.rgb.shape)}
    try:
        face,count=select_face(detector_b.detect(mother.rgb),0.)
        result['B']=save_crop(warp(mother.rgb,b_alignment_matrix(face),224),directory,'B')
        result['B'].update(input_rgb_sha256=mother.rgb_sha256,detection=face_json(face,count))
    except (GeometryFailure,ValueError,cv2.error,RuntimeError) as exc:
        result['B']={'ok':False,'stage':'B_original','reason':str(exc)}
    try:
        face,count=select_face(detector_a.detect(mother.rgb),.6)
        aligned=warp(mother.rgb,a_alignment_matrix(face),336)
        result['A']=save_crop(aligned,directory,'A')
        result['A'].update(input_rgb_sha256=mother.rgb_sha256,detection=face_json(face,count))
    except (GeometryFailure,ValueError,cv2.error,RuntimeError) as exc:
        result['A']={'ok':False,'stage':'A_original','reason':str(exc)}
        result['AB']={'ok':False,'stage':'A_parent','reason':'A_failed'}
        return result
    try:
        # Both A image and mask are read back. No original B coordinates enter.
        a_rgb=read_rgb(result['A']['path'])
        a_mask=cv2.imread(result['A']['mask_path'],cv2.IMREAD_GRAYSCALE)
        if rgb_hash(a_rgb)!=result['A']['rgb_sha256']: raise GeometryFailure('A_png_rgb_changed')
        face,count=select_face(detector_b.detect(a_rgb),0.)
        result['AB']=save_crop(warp(a_rgb,b_alignment_matrix(face),224,a_mask),directory,'AB')
        result['AB'].update(input_rgb_sha256=rgb_hash(a_rgb),parent_byte_sha256=result['A']['byte_sha256'],
                            detection=face_json(face,count))
    except (GeometryFailure,ValueError,cv2.error,RuntimeError) as exc:
        result['AB']={'ok':False,'stage':'B_redetection','reason':str(exc)}
    return result


def select_mothers(frames,count=8):
    unique=[]; seen=set()
    for frame in sorted(frames,key=lambda f:(f.timestamp,f.index)):
        if frame.rgb_sha256 not in seen: unique.append(frame); seen.add(frame.rgb_sha256)
    if len(unique)<count: raise GeometryFailure('insufficient_unique_mothers')
    return [unique[i] for i in np.rint(np.linspace(0,len(unique)-1,count)).astype(int)]


def _full_decode_mothers(path):
    capture=_video_capture(path); seen=set(); unique=[]; index=0
    while True:
        ok,bgr=capture.read()
        if not ok: break
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); digest=rgb_hash(rgb)
        if digest not in seen:
            seen.add(digest); unique.append((index,float(capture.get(cv2.CAP_PROP_POS_MSEC))/1000,digest))
        index+=1
    capture.release()
    if len(unique)<8: raise GeometryFailure('insufficient_unique_mothers')
    wanted={unique[i][0]:unique[i] for i in np.rint(np.linspace(0,len(unique)-1,8)).astype(int)}
    capture=_video_capture(path); selected=[]
    for i in range(max(wanted)+1):
        ok,bgr=capture.read()
        if not ok: capture.release(); raise GeometryFailure('selected_mother_decode_failed')
        if i in wanted:
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); index,pts,digest=wanted[i]
            if rgb_hash(rgb)!=digest: capture.release(); raise GeometryFailure('decode_rgb_not_reproducible')
            selected.append(MotherFrame(index,pts,rgb,digest))
    capture.release()
    return selected,len(unique)


def decode_mothers(path):
    """At most32 uniformly spaced time candidates, then full sequential fallback."""
    started=time.monotonic()
    probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=duration,avg_frame_rate,nb_frames:format=duration','-of','json',str(path)],
        check=True,capture_output=True,text=True,timeout=90)
    info=json.loads(probe.stdout); stream=info['streams'][0]
    duration=float(stream.get('duration',info.get('format',{}).get('duration',0)))
    if not np.isfinite(duration) or duration<=0: raise GeometryFailure('invalid_video_duration')
    capture=_video_capture(path)
    if not capture.isOpened(): raise GeometryFailure('decode_open_failed')
    fps=capture.get(cv2.CAP_PROP_FPS); end=max(0.,duration-(1/fps if fps>0 else .001))
    candidates=[]; targets=np.linspace(0,end,32); cursor=0
    # One chronological decoder pass avoids decoding the same H.264 GOP32 times.
    # Only the32 target frames are converted to RGB; unselected frames have no
    # RGB hashing or retention. Target rule is fixed for every domain/label.
    while cursor<len(targets) and capture.grab():
        pts=float(capture.get(cv2.CAP_PROP_POS_MSEC))/1000
        if pts+1e-7<targets[cursor]: continue
        ok,bgr=capture.retrieve()
        if not ok: continue
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); digest=rgb_hash(rgb)
        index=int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))-1
        while cursor<len(targets) and pts+1e-7>=targets[cursor]:
            candidates.append(MotherFrame(index,pts,rgb,digest)); cursor+=1
    failures=[float(t) for t in targets[cursor:]]
    capture.release()
    unique_count=len({f.rgb_sha256 for f in candidates}); strategy='uniform_time_candidates_32_sequential_first_pts_at_or_after'
    try: selected=select_mothers(candidates)
    except GeometryFailure:
        candidates.clear(); selected,unique_count=_full_decode_mothers(path)
        strategy='full_chronological_decode_fallback'
    return selected,{'strategy':strategy,'candidate_count':32,'unique_candidate_rgb':unique_count,
        'candidate_decode_failure_timestamps':failures,'duration_seconds':duration,
        'decode_seconds':time.monotonic()-started,'selected_frames':[
            {'frame_index':m.index,'timestamp_seconds':m.timestamp,'rgb_sha256':m.rgb_sha256} for m in selected]}


def _video_capture(path):
    # Limit decoder threads as each persistent worker already owns one CPU.
    return cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])


def video_directory(record,artifact_root):
    parts=[record['dataset'],record['method_id'],record['video_id']]
    if any(not p or '/' in p or p in ('.','..') for p in parts): raise ValueError('Unsafe H video path')
    return Path(artifact_root).resolve().joinpath('derived/H',*parts)


def _verify_file(path,expected):
    if not Path(path).is_file() or sha256_file(path)!=expected:
        raise GeometryFailure('persisted_artifact_changed: '+str(path))


def verify_frame_artifacts(frame,deep=False):
    _verify_file(frame['mother_path'],frame['mother_byte_sha256'])
    if deep and rgb_hash(read_rgb(frame['mother_path']))!=frame['mother_rgb_sha256']:
        raise GeometryFailure('persisted_mother_rgb_changed')
    for version in VERSIONS:
        item=frame.get(version,{})
        if not item.get('ok'): continue
        _verify_file(item['path'],item['byte_sha256']); _verify_file(item['mask_path'],item['mask_byte_sha256'])
        if deep:
            size=336 if version=='A' else 224; rgb=read_rgb(item['path'])
            mask=cv2.imread(item['mask_path'],cv2.IMREAD_GRAYSCALE)
            if rgb.shape!=(size,size,3) or rgb_hash(rgb)!=item['rgb_sha256']:
                raise GeometryFailure('persisted_output_rgb_changed')
            if mask is None or mask.shape!=(size,size) or not np.isin(mask,[0,255]).all():
                raise GeometryFailure('persisted_mask_invalid')


def audit_video_mothers(record,artifact_root,protocol_sha256=None):
    """Persist the eight mothers before inference, including unused source nodes.

    Audit scope is the full candidate VIDEO graph and its selected mothers;
    unselected video frames are not pixel-audited. Full video bytes are hashed.
    """
    directory=video_directory(record,artifact_root); directory.mkdir(parents=True,exist_ok=True)
    identity=protocol_sha256 or protocol_hash(); path=directory/'mothers.json'
    with (directory/'mothers.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        source_sha=sha256_file(record['path'])
        if path.exists():
            audit=json.loads(path.read_text())
            if (audit['protocol_sha256']!=identity or audit['video_uid']!=record['video_uid']
                    or audit['source_sha256']!=source_sha): raise GeometryFailure('persisted_mother_identity_changed')
            for frame in audit.get('frames',[]): _verify_file(frame['mother_path'],frame['mother_byte_sha256'])
            return audit
        audit={'video_uid':record['video_uid'],'protocol_sha256':identity,
            'source_path':str(Path(record['path']).resolve()),'source_sha256':source_sha,
            'source_bytes':Path(record['path']).stat().st_size,'selected_frames':[],'frames':[],
            'audit_scope':'ALL_CANDIDATE_VIDEOS_SELECTED_EIGHT_MOTHERS_AND_FULL_VIDEO_BYTE_HASH'}
        try:
            mothers,selection=decode_mothers(record['path']); audit.update(selection)
            for slot,mother in enumerate(mothers):
                output=directory/f'{slot:03d}'; output.mkdir(parents=True,exist_ok=True)
                image=output/'mother.png'
                if not cv2.imwrite(str(image),cv2.cvtColor(mother.rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_PNG_COMPRESSION,3]):
                    raise GeometryFailure('mother_png_write_failed')
                audit['frames'].append({'mother_id':mother_id(record['video_uid'],mother),
                    'mother_frame_index':mother.index,'mother_timestamp_seconds':mother.timestamp,
                    'mother_rgb_sha256':mother.rgb_sha256,'mother_path':str(image),
                    'mother_byte_sha256':sha256_file(image),'mother_png_bytes':image.stat().st_size})
            audit['status']='MOTHERS_SELECTED'
        except (GeometryFailure,subprocess.SubprocessError,KeyError,ValueError) as exc:
            audit.update(status='MOTHER_REJECTED',failure=str(exc),frames=[],selected_frames=[])
        write_json(path,audit)
        return audit


def process_video(record,detector_a,detector_b,artifact_root,protocol_sha256=None):
    """Resume independent videos/frames; completed records fail on altered bytes."""
    directory=video_directory(record,artifact_root); directory.mkdir(parents=True,exist_ok=True)
    identity=protocol_sha256 or protocol_hash(); path=directory/'manifest.json'; started=time.monotonic()
    with (directory/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        audit=audit_video_mothers(record,artifact_root,identity)
        if path.exists():
            manifest=json.loads(path.read_text())
            if manifest['protocol_sha256']!=identity or manifest['source_sha256']!=audit['source_sha256']:
                raise GeometryFailure('persisted_video_identity_changed')
            for frame in manifest['frames']: verify_frame_artifacts(frame)
            return manifest
        manifest={'schema':'JOURNAL_H_VIDEO_MANIFEST_V1','record':record,
            'video_uid':record['video_uid'],'protocol_sha256':identity,'source_sha256':audit['source_sha256'],
            'source_bytes':audit['source_bytes'],'mother_audit_sha256':sha256_file(directory/'mothers.json'),
            'mother_selection':{key:value for key,value in audit.items() if key not in ('frames','source_sha256')},
            'frames':[]}
        if audit['status']=='MOTHERS_SELECTED':
            for slot,saved in enumerate(audit['frames']):
                frame_path=directory/f'{slot:03d}'/'frame_manifest.json'
                if frame_path.exists():
                    frame=json.loads(frame_path.read_text())
                    if frame.get('protocol_sha256')!=identity or frame.get('mother_id')!=saved['mother_id']:
                        raise GeometryFailure('persisted_frame_identity_changed')
                    verify_frame_artifacts(frame)
                else:
                    rgb=read_rgb(saved['mother_path'])
                    if rgb_hash(rgb)!=saved['mother_rgb_sha256']: raise GeometryFailure('persisted_mother_rgb_changed')
                    mother=MotherFrame(saved['mother_frame_index'],saved['mother_timestamp_seconds'],rgb,saved['mother_rgb_sha256'])
                    # Only incomplete slots can be regenerated. Their orphan files
                    # never become active successful entries after a failed rerun.
                    for version in VERSIONS:
                        for suffix in ('.png','_mask.png'):
                            (frame_path.parent/f'{version}{suffix}').unlink(missing_ok=True)
                    frame=process_frame(mother,detector_a,detector_b,frame_path.parent)
                    frame.update(mother_id=saved['mother_id'],protocol_sha256=identity)
                    write_json(frame_path,frame)
                manifest['frames'].append(frame)
        else: manifest['failure']=audit['failure']
        eligible,failures=video_eligibility(manifest['frames'])
        manifest.update(status='COMPLETE',eligible=eligible,version_failures=failures,
            common_B_AB=bool(eligible['B'] and eligible['AB']),processing_seconds=time.monotonic()-started)
        write_json(path,manifest)
        return manifest


def video_eligibility(frames):
    eligible={}; failures={}
    for version in VERSIONS:
        errors=[f'{i}:{f[version].get("stage",version)}:{f[version]["reason"]}'
                for i,f in enumerate(frames) if not f[version]['ok']]
        hashes=[f[version]['rgb_sha256'] for f in frames if f[version]['ok']]
        if len(frames)!=8: errors.append('insufficient_mother_slots')
        if len(hashes)==8 and len(set(hashes))!=8: errors.append('insufficient_unique_final_rgb')
        eligible[version]=not errors; failures[version]=errors
    return eligible,failures


class RetinaFace5Detector:
    """Pinned primary PyTorch RetinaFace/MobileNet0.25 inference on CPU."""
    def __init__(self,weight,device='cpu'):
        import torch
        if sha256_file(weight)!=RETINAFACE_SHA256: raise ValueError('RetinaFace weight hash mismatch')
        provenance=json.loads((RETINAFACE_SOURCE.parent/'PROVENANCE.json').read_text())
        if provenance['commit']!=RETINAFACE_COMMIT: raise ValueError('RetinaFace source revision mismatch')
        for name,expected in provenance['files_sha256'].items():
            if sha256_file(RETINAFACE_SOURCE/name)!=expected: raise ValueError('RetinaFace source file changed: '+name)
        source=str(RETINAFACE_SOURCE)
        if source not in sys.path: sys.path.insert(0,source)
        cfg=dict(importlib.import_module('data.config').cfg_mnet); cfg['pretrain']=False
        model=importlib.import_module('models.retinaface').RetinaFace(cfg=cfg,phase='test')
        state=torch.load(weight,map_location='cpu',weights_only=True)
        if 'state_dict' in state: state=state['state_dict']
        model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
        self.net=model.eval().to(device); self.cfg=cfg; self.device=torch.device(device)
        self.prior=importlib.import_module('layers.functions.prior_box').PriorBox
        boxes=importlib.import_module('utils.box_utils')
        self.decode=boxes.decode; self.decode_landm=boxes.decode_landm
        self.nms=importlib.import_module('utils.nms.py_cpu_nms').py_cpu_nms
        self.priors={}

    def detect(self,rgb):
        import torch
        height,width=rgb.shape[:2]; scale=min(1.,1024/max(height,width))
        w,h=max(1,int(round(width*scale))),max(1,int(round(height*scale)))
        image=cv2.resize(rgb,(w,h),interpolation=cv2.INTER_LINEAR) if (w,h)!=(width,height) else rgb
        bgr=image[:,:,::-1].astype(np.float32); bgr-=(104,117,123)
        tensor=torch.from_numpy(np.ascontiguousarray(bgr.transpose(2,0,1))).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            loc,conf,landmarks=self.net(tensor)
            if (h,w) not in self.priors:
                self.priors[(h,w)]=self.prior(self.cfg,image_size=(h,w)).forward().to(self.device)
            priors=self.priors[(h,w)]
            boxes=self.decode(loc.squeeze(0),priors,self.cfg['variance']).cpu().numpy()*[w,h,w,h]
            points=self.decode_landm(landmarks.squeeze(0),priors,self.cfg['variance']).cpu().numpy().reshape(-1,5,2)*[w,h]
            scores=conf.squeeze(0)[:,1].cpu().numpy()
        take=np.flatnonzero(scores>=.6)
        take=take[np.argsort(-scores[take],kind='stable')[:5000]]
        boxes=boxes[take]; points=points[take]; scores=scores[take]
        keep=self.nms(np.c_[boxes,scores].astype(np.float32),.4)[:750]
        return [Face(boxes[i]/[w/width,h/height,w/width,h/height],float(scores[i]),
                     points[i]/[w/width,h/height]) for i in keep]


class Dlib68Detector:
    def __init__(self,weight):
        import dlib
        if sha256_file(weight)!=DLIB68_SHA256: raise ValueError('dlib68 weight hash mismatch')
        self.detector=dlib.get_frontal_face_detector(); self.predictor=dlib.shape_predictor(str(weight))

    def detect(self,rgb):
        import dlib
        height,width=rgb.shape[:2]; scale=min(1.,1024/max(height,width))
        w,h=max(1,int(round(width*scale))),max(1,int(round(height*scale)))
        image=cv2.resize(rgb,(w,h),interpolation=cv2.INTER_LINEAR) if (w,h)!=(width,height) else rgb
        rectangles,scores,_=self.detector.run(np.ascontiguousarray(image),1,0.)
        faces=[]
        for rect,score in zip(rectangles,scores):
            box=np.array([rect.left(),rect.top(),rect.right(),rect.bottom()],float)/[w/width,h/height,w/width,h/height]
            x1,y1,x2,y2=np.rint(box).astype(int)
            shape=self.predictor(np.ascontiguousarray(rgb),dlib.rectangle(int(x1),int(y1),int(x2),int(y2)))
            points=np.array([[shape.part(i).x,shape.part(i).y] for i in range(68)],float)
            faces.append(Face(box,float(score),points))
        return faces


def protocol():
    import dlib
    import torch
    return {'schema':'JOURNAL_H_RETINAFACE5_336_DLIB68_224_V1',
        'mother_selection':'32 uniform-time targets; sequential firstPTS at/after target; chronological RGB dedup; rint-linspace8; full decode fallback',
        'mother_selection_before_detection':True,'mothers_per_video':8,
        'A':{'detector':'biubug6/Pytorch_Retinaface','commit':RETINAFACE_COMMIT,
             'architecture':'MobileNet0.25','weight_sha256':RETINAFACE_SHA256,'landmarks':5,
             'score_threshold':.6,'nms_iou':.4,'pre_nms_top_k':5000,'post_nms_top_k':750,
             'detector_max_long_side':1024,'upsampling':False,'output_size':336,
             'alignment':'orientation-preserving least-squares similarity to ArcFace112 template x3',
             'template_pixel_centres':(TEMPLATE_112*3).tolist()},
        'B':{'detector':'dlib.get_frontal_face_detector HOG/SVM','dlib_version':dlib.__version__,
             'weight_sha256':DLIB68_SHA256,'landmarks':68,'detector_max_long_side':1024,
             'upsample_num_times':1,'adjust_threshold':0.,'score_threshold':0.,'output_size':224,
             'alignment':'dlib.get_face_chip_details(68points,size=224,padding=0.25); get_mapping_to_chip geometry'},
        'AB':'read saved A.png AND A_mask.png; complete fresh B detector+68predictor+alignment',
        'face_selection':'largest bbox area; score descending; bbox lexicographic; landmarks lexicographic',
        'rgb_interpolation':'OpenCV INTER_CUBIC','mask_interpolation':'OpenCV INTER_NEAREST',
        'padding':'BORDER_CONSTANT zero','mask':'binary validity of mapped pixel centres; AB inherits A validity',
        'png_compression':3,'eligibility':'8 unique mothers and all8 successful unique finalRGB per version; no replacement',
        'versions':{'opencv':cv2.__version__,'numpy':np.__version__,'torch':torch.__version__},
        'primary_source_provenance_sha256':sha256_file(RETINAFACE_SOURCE.parent/'PROVENANCE.json'),
        'geometry_code_sha256':sha256_file(__file__)}


def protocol_hash():
    return hashlib.sha256(json.dumps(protocol(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
