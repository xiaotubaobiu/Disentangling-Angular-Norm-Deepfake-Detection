#!/usr/bin/env python3
"""Persistent journal H generation: qualify → source freeze → remaining test.

The standalone operation deliberately leaves trainer/registry/config code alone.
Resume the same command after interruption; per-video and per-frame records are
independent and completed artifacts are verified before reuse.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

from journal_detector import geometry_h as g
from journal_detector import data_h as h
import pandas as pd

_ROOT=None
_IDENTITY=None
_DETECTORS=None


def initialize(root,identity):
    global _ROOT,_IDENTITY
    _ROOT=Path(root); _IDENTITY=identity
    g.cv2.setNumThreads(1)
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    from h_png_io import atomic_imwrite
    g.cv2.imwrite=atomic_imwrite


def worker(job):
    global _DETECTORS
    action,record=job; started=time.monotonic()
    if g.protocol_hash()!=_IDENTITY: raise RuntimeError('H geometry changed during persistent preparation')
    prior=g.video_directory(record,_ROOT)/('mothers.json' if action=='audit' else 'manifest.json')
    cached=prior.exists()
    if action=='audit':
        obj=g.audit_video_mothers(record,_ROOT,_IDENTITY)
        return {'video_uid':record['video_uid'],'action':action,'status':obj['status'],
                'cached':cached,'seconds':time.monotonic()-started}
    if _DETECTORS is None:
        weights=_ROOT/'weights/preprocessing'
        _DETECTORS=(g.RetinaFace5Detector(weights/'mobilenet0.25_Final.pth'),
                    g.Dlib68Detector(weights/'shape_predictor_68_face_landmarks.dat'))
    obj=g.process_video(record,*_DETECTORS,_ROOT,_IDENTITY)
    return {'video_uid':record['video_uid'],'action':action,'status':obj['status'],
            'eligible':obj['eligible'],'common':obj['common_B_AB'],'cached':cached,'seconds':time.monotonic()-started}


def phase(pool,jobs,root,name):
    started=time.monotonic(); counts=Counter(); timing=[]; errors=[]; fresh_timing=[]
    pending={pool.submit(worker,job):job[1]['video_uid'] for job in jobs}
    for done,future in enumerate(as_completed(pending),1):
        try: result=future.result()
        except Exception as exc:
            errors.append({'video_uid':pending[future],'error':repr(exc)})
            print(json.dumps({'status':'VIDEO_ERROR','stage':name,**errors[-1]}),flush=True)
            continue
        counts[result['status']]+=1; timing.append(result['seconds'])
        counts['cached' if result['cached'] else 'fresh']+=1
        if not result['cached']: fresh_timing.append(result['seconds'])
        if result.get('common'): counts['common']+=1
        if done==1 or done%10==0 or done==len(jobs):
            progress={'stage':name,'done':done,'total':len(jobs),'elapsed_seconds':time.monotonic()-started,
                'counts':dict(counts),'last':result,'mean_task_seconds':sum(timing)/len(timing),
                'mean_fresh_task_seconds':sum(fresh_timing)/len(fresh_timing) if fresh_timing else None}
            g.write_json(root/'operations/h_progress.json',progress)
            print(json.dumps(progress,ensure_ascii=False),flush=True)
    if errors:
        g.write_json(root/'operations/h_video_errors.json',{'stage':name,'errors':errors})
        raise RuntimeError(f'{len(errors)} H video errors; see operations/h_video_errors.json')
    return {'stage':name,'tasks':len(jobs),'elapsed_seconds':time.monotonic()-started,'counts':dict(counts)}


def load_artifacts(records,root):
    audits={}; manifests={}; paths={}
    for record in records:
        directory=g.video_directory(record,root); uid=record['video_uid']
        if (directory/'mothers.json').exists(): audits[uid]=json.loads((directory/'mothers.json').read_text())
        if (directory/'manifest.json').exists():
            manifests[uid]=json.loads((directory/'manifest.json').read_text()); paths[uid]=str(directory/'manifest.json')
    return audits,manifests,paths


def qualify(pool,records,root,identity):
    path=root/'data/H/geometry_frozen.json'
    if path.exists():
        receipt=json.loads(path.read_text())
        if receipt['protocol_sha256']!=identity: raise ValueError('Existing H geometry qualification differs')
        return receipt
    cells={}
    for record in records:
        key=(record['role'],record['method_id'])
        if record['role'] in ('train','dev') and key not in cells: cells[key]=record
    chosen=[cells[key] for key in sorted(cells)]
    result=phase(pool,[('process',r) for r in chosen],root,'source_geometry_qualification')
    audits,manifests,_=load_artifacts(chosen,root)
    receipt={'status':'FROZEN_BEFORE_TARGET_PREPROCESSING','protocol_sha256':identity,
        'protocol':g.protocol(),'qualification_scope':'source TRAIN/DEV, one video per method/role cell, no target decoded',
        'videos':[{'video_uid':r['video_uid'],'eligible':manifests[r['video_uid']]['eligible'],
                   'common':manifests[r['video_uid']]['common_B_AB']} for r in chosen],
        'runtime':result,'rules_changed_from_target_scores':False,
        'geometry_notes':str(root/'operations/h_pipeline_notes.md')}
    if len(chosen)!=10 or not all(manifests[r['video_uid']].get('frames') for r in chosen):
        raise ValueError('Source-only geometry qualification lacks10 decodable method/role cells')
    g.write_json(path,receipt)
    print(json.dumps({'status':'SOURCE_GEOMETRY_FROZEN','path':str(path),'protocol_sha256':identity}),flush=True)
    return receipt


def save_coverage(root,records,manifests,graph,stage):
    frame,counts=h.coverage(records,manifests)
    frame['quarantined']=frame.video_uid.isin(graph['quarantined_video_uids'])
    directory=root/'data/H'; frame.to_csv(directory/f'coverage_videos_{stage}.csv',index=False)
    stable=directory/('coverage_source.csv' if stage=='source' else 'coverage.csv')
    frame.to_csv(stable,index=False)
    counts.to_csv(directory/f'coverage_counts_{stage}.csv',index=False)
    failures=Counter(); quality=[]
    by_uid={r['video_uid']:r for r in records}
    for uid,manifest in manifests.items():
        row=by_uid[uid]
        for slot,frame in enumerate(manifest.get('frames',[])):
            for version in ('A','B','AB'):
                item=frame[version]
                if not item['ok']: failures[(row['role'],row['method_id'],version,item.get('stage',''),item['reason'])]+=1
                quality.append({'video_uid':uid,'role':row['role'],'method_id':row['method_id'],'slot':slot,
                    'version':version,'ok':item['ok'],'padding_ratio':item.get('padding_ratio'),
                    'multiface':item.get('detection',{}).get('multiface'),
                    'failure_stage':item.get('stage',''),'failure_reason':item.get('reason','')})
    pd.DataFrame(quality).to_csv(directory/f'coverage_frames_{stage}.csv',index=False)
    g.write_json(directory/f'failure_counts_{stage}.json',[
        {'role':role,'method_id':method,'version':version,'stage':failure_stage,'reason':reason,'count':count}
        for (role,method,version,failure_stage,reason),count in sorted(failures.items())])


def materialize(root,records,scope):
    audits,manifests,paths=load_artifacts(records,root)
    if len(audits)!=len(records): raise ValueError('Incomplete H selected-mother candidate audit')
    graph=h.source_graph_audit(records,audits,manifests)
    stage='source' if scope=='SOURCE_TRAIN_DEV_CAL' else 'all'
    graph_path=root/f'data/H/cohort_graph_{stage}.json'; g.write_json(graph_path,graph)
    blocked=set(graph['quarantined_video_uids']); active=[]
    for record in records:
        if record['video_uid'] not in blocked:
            active.append({**record,'group_id':graph['components'][record['video_uid']]})
    tables=h.build_tables(active,manifests)
    if scope=='SOURCE_TRAIN_DEV_CAL':
        tables={key:value for key,value in tables.items() if key.split('/')[-1] in ('train','dev','source_cal')}
    else:
        required=[r['video_uid'] for r in records if r['role']!='source_unused']
        if any(uid not in manifests for uid in required): raise ValueError('Incomplete H geometry generation')
    save_coverage(root,records,manifests,graph,stage)
    selected_paths={uid:path for uid,path in paths.items()
                    if scope!='SOURCE_TRAIN_DEV_CAL' or next(r['role'] for r in records if r['video_uid']==uid) in ('train','dev','source_cal')}
    evidence={str(path):g.sha256_file(path) for path in [root/'data/H/inventory.json',root/'data/H/geometry_protocol.json',
                root/'data/H/geometry_frozen.json',graph_path,Path(__file__).resolve(),Path(__file__).with_name('h_png_io.py').resolve()]}
    coverage_path=root/'data/H'/('coverage_source.csv' if stage=='source' else 'coverage.csv')
    evidence[str(coverage_path)]=g.sha256_file(coverage_path)
    evidence.update({str(g.video_directory(r,root)/'mothers.json'):g.sha256_file(g.video_directory(r,root)/'mothers.json') for r in records})
    receipt=h.write_table_freeze(root,tables,scope,metadata={
        'candidate_videos':len(records),'processed_geometry_videos':len(manifests),
        'quarantined_videos':len(blocked),'graph_path':str(graph_path),'evidence_files':evidence,
        'coverage':{'path':str(coverage_path),'sha256':g.sha256_file(coverage_path),
                    'candidate_videos':len(records),'schema':'JOURNAL_H_VIDEO_COVERAGE_V1',
                    'denominator_scope':'all source candidates, including unused/pending' if stage=='source' else 'all8431 candidates, including failures/unused',
                    'eligibility_flag_scope':'A/B/AB/common are geometric eligibility; quarantined excludes active cohort'},
        'inventory':{'path':str(root/'data/H/inventory.json'),'sha256':g.sha256_file(root/'data/H/inventory.json'),
                     'candidate_videos':8431,'schema':'JOURNAL_H_CANDIDATE_VIDEO_INVENTORY_V1'},
        'selected_mother_audit_scope':graph['pixel_scope'],'coverage_directory':str(root/'data/H'),
        'DFD_parent_identity_certified':False},manifest_paths=selected_paths)
    g.write_json(root/f'operations/h_{stage}_ready.json',{'status':'SOURCE_READY' if stage=='source' else 'H_READY',
        'freeze_path':str(root/'data/H'/('train_dev_freeze.json' if stage=='source' else 'freeze.json')),
        'draw_count':receipt['draw_count'],'tables':receipt['tables']})
    print(json.dumps({'status':'SOURCE_READY' if stage=='source' else 'H_READY','draw_count':receipt['draw_count'],
        'quarantined_videos':len(blocked),'tables':{key:{'rows':v['rows'],'videos':v['videos']} for key,v in receipt['tables'].items()}}),flush=True)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,default=h.DATA_ROOT)
    parser.add_argument('--workers',type=int,default=32)
    parser.add_argument('--stage',choices=['qualify','source','all'],default='all')
    args=parser.parse_args(); root=args.artifact_root.resolve()
    (root/'operations').mkdir(parents=True,exist_ok=True); (root/'data/H').mkdir(parents=True,exist_ok=True)
    with (root/'operations/h_preparation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=g.protocol_hash(); inventory=root/'data/H/inventory.json'; records=h.build_inventory(args.data_root)
        if inventory.exists() and json.loads(inventory.read_text())!=records: raise ValueError('H inventory changed')
        g.write_json(inventory,records); g.write_json(root/'data/H/geometry_protocol.json',g.protocol())
        g.write_json(root/'operations/h_process.json',{'pid':os.getpid(),'workers':args.workers,
            'command':sys.argv,'protocol_sha256':identity,'device':'CPU_ONLY','start_time':time.time()})
        print(json.dumps({'status':'STARTED','pid':os.getpid(),'workers':args.workers,'candidate_videos':len(records),
                          'protocol_sha256':identity}),flush=True)
        context=multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=context,
                initializer=initialize,initargs=(str(root),identity)) as pool:
            qualify(pool,records,root,identity)
            if args.stage=='qualify': return
            source=[r for r in records if r['dataset']=='FFPP']
            if not (root/'data/H/train_dev_freeze.json').exists():
                jobs=[('process' if r['role'] in ('train','dev','source_cal') else 'audit',r) for r in source]
                # Audit-only candidates first preserve allsource bridges before
                # source readiness; source geometry follows without a new pool.
                jobs.sort(key=lambda job:(job[0]!='audit',job[1]['role']!='source_cal',job[1]['role']!='dev',job[1]['video_uid']))
                phase(pool,jobs,root,'source_train_dev_cal_plus_allsource_mother_audit')
                materialize(root,source,'SOURCE_TRAIN_DEV_CAL')
            else: h.validate_h_freeze(root,source_only=True)
            if args.stage=='source': return
            remaining=[r for r in records if r['role'] in ('source_test','target_test','source_unused')]
            phase(pool,[('process',r) for r in remaining],root,'remaining_source_test_DFD_unused_geometry')
            materialize(root,records,'ALL_H_ROLES')
        print(json.dumps({'status':'COMPLETE','protocol_sha256':identity}),flush=True)


if __name__=='__main__': main()
