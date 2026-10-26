"""Selected-M worker bridge. No dispatcher, production permit or GPU probe.

The reviewed CPU cache component remains byte-unchanged. This worker is usable
only by a new owner after the original 30-model primary cache stage has closed,
with an inherited genuine phase lease, a shared native GPU lease, independently
qualified bridge and an issued full-M resource/code/cache permit.
"""
from pathlib import Path
import argparse,json,os,time,resource
import numpy as np
import torch
from . import source_best_cache as cache,source_best_lease as lease,source_best_files as sf,trainer
require=sf.require

def _bridge_gate(context,permit,*,full=True):
    gate=permit.get('gpu_bridge');require(isinstance(gate,dict),'qualified selected-M GPU bridge required')
    pins={str(Path(__file__)):sf.digest(__file__),str(Path(lease.__file__)):sf.digest(lease.__file__),str(Path(cache.__file__)):sf.digest(cache.__file__)}
    require(gate.get('source_sha256')==pins,'GPU bridge/source implementation admission changed')
    ref=gate.get('qualification');require(isinstance(ref,dict),'real GPU bridge qualification required')
    cache._pin(ref['path'],ref['sha256'],{},context.root);accepted=cache._json(ref['path'])
    require(accepted.get('schema')=='M_GPU_BRIDGE_QUALIFICATION_V1' and accepted.get('status')=='PASS' and accepted.get('real_source_only_GPU_qualification_passed') is True and accepted.get('phase_native_mutual_exclusion_verified') is True and accepted.get('source_sha256')==pins,'bridge has not passed real source-only GPU qualification')
    ref=gate.get('primary_cache_stage');require(isinstance(ref,dict),'closed original primary cache stage required')
    require(Path(ref['path'])==context.root/'reports/H_primary_cache_stage_complete.json','canonical original stage receipt required')
    cache._pin(ref['path'],ref['sha256'],{},context.root);stage=cache._json(ref['path'])
    require(stage.get('status')=='PRIMARY30_CACHE_WORKERS_RETURNED_210ROLE_MARKERS_PENDING_INDEPENDENT_ADMISSION' and stage.get('role_count')==210,'original 30-model GPU cache stage must finish first')
    source_ref=cache._json(context.admission_path)['source_file_admission'];admission=cache._json(source_ref['path'])
    expected=sorted(key for key,record in admission['records'].items() if record['route']=='H')
    require(len(expected)==30 and sorted(stage['run_ids'])==expected and len(stage['run_ids'])==30,'original effective primary H model coverage changed')
    queue=cache._json(context.root/'H_queue_status.json')
    require(not queue.get('active') and str(queue.get('stage','')).endswith('_COMPLETE'),'original primary owner still dispatches/holds workers')
    if full:
        for path,pin in stage['bindings'].items():cache._pin(path,pin,{})
    return gate

def _extract_owned(context,device,guard):
    """Exact original FP32 encoder/writer path; caller supplies live authority.

    The CPU form is for bounded synthetic/source-only qualification, and does
    not issue permits or authorize actual target evaluation.
    """
    guard(True);context.check(full=True);task=context.task;config=context.config
    batch=config.get('cache_batch_size',task['microbatch']);workers=config.get('cache_workers',config.get('workers',4))
    require(type(batch) is int and batch>0 and type(workers) is int and workers>=0 and config.get('cache_chunk_rows',4096)==4096,'frozen M loader/chunk budget invalid')
    frames={role:(context.legacy._read_manifest(context.tables[role]['path'],role) if context.source['route']=='H' else context.legacy._read_manifest(context.tables[role]['path'])) for role in context.roles}
    trainer.policy();model=None;results={};start=time.monotonic()
    for role in context.roles:
        guard(True);context.check(full=True);folder=cache.role_path(context,role)
        if (folder/'complete.json').exists():meta=cache.verify_role(context,role)[0]
        else:
            if model is None:
                cp=sf._load(context.source['checkpoint_path']);require(cp.get('identity')==context.source['training_identity'] and cp.get('epoch')==context.source['best']['epoch'],'selected source-best state identity changed')
                sf._tensor_check(cp['model'],2*1024**3);context.check()
                w=cp['model']['classifier.weight'].detach().cpu().numpy().copy();b=cp['model']['classifier.bias'].detach().cpu().numpy().copy()
                require(cache._head_hash(w,b,context.base['dimension'])==context.base['original_head_sha256'],'original selected head changed')
                guard(False);model=cache._build_model(task['backbone'],config['backbones'][task['backbone']]['weight_path'])
                model.load_state_dict(cp['model'],strict=True);model=model.to(device=device,dtype=torch.float32).eval();guard(False)
            def batches(offset,frame=frames[role]):
                data=trainer.loader(frame.iloc[offset:].reset_index(drop=True),task['backbone'],batch,workers=workers)
                with torch.inference_mode(),torch.autocast(device_type=torch.device(device).type,enabled=False):
                    for x,_ in data:
                        guard(False);h=model.encoder(x.to(device=device,dtype=torch.float32));require(h.dtype==torch.float32,'original raw FP32 encoder output required');guard(False)
                        yield h.detach().cpu().numpy().copy()
            meta=cache.write_role(context,role,w,b,batches)
        results[role]=dict(path=str(folder),rows=meta['rows'],complete_sha256=sf.digest(folder/'complete.json'))
        guard(True)
    context.check(full=True);guard(True)
    return dict(schema='M_SOURCE_BEST_OPERATION_RETURN_V1',status='ALL_SELECTED_CANDIDATE_ROLES_VERIFIED_REQUIRES_INDEPENDENT_ADMISSION',parent_run_id=context.base['parent_run_id'],input_admission_sha256=context.admission_sha,checkpoint_sha256=context.base['checkpoint_sha256'],source_best_epoch=context.base['source_best_epoch'],source_best_prefix=context.base['source_best_prefix'],roles=results,seconds=time.monotonic()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,independent_scientific_admission=False)

def extract_worker(root,input_admission_path,input_sha,parent_run_id,*,permit_path,permit_sha256,phase_proof,phase_fd):
    lease.verify_worker(root,phase_proof,phase_fd,arm_parent_death=True)
    try:
        context=cache.prepare_context(root,input_admission_path,input_sha,parent_run_id)
        def guard(full=False):
            lease.verify_worker(root,phase_proof,phase_fd)
            permit=cache._permit(context,permit_path,permit_sha256);_bridge_gate(context,permit,full=full)
        guard(True)
        with lease.native_gpu(root,context.config,phase_proof,phase_fd):
            with cache._lock(context.root/'features/source_best_M'/context.admission_sha/parent_run_id/'extract.lock',context.root):
                result=_extract_owned(context,'cuda',guard)
        guard(True)
        receipt=dict(result,worker_pid=os.getpid(),phase_owner_pid=phase_proof['owner_pid'],phase_nonce=phase_proof['nonce'],permit_path=str(permit_path),permit_sha256=permit_sha256,source_sha256={str(Path(__file__)):sf.digest(__file__),str(Path(lease.__file__)):sf.digest(lease.__file__),str(Path(cache.__file__)):sf.digest(cache.__file__)})
        destination=cache._inside(context.root/'reports/M_source_best_workers'/phase_proof['nonce']/(parent_run_id+'.json'),context.root)
        with cache._lock(destination.with_suffix('.lock'),context.root):cache._save_json(destination,receipt)
        guard(True);return receipt
    finally:
        # This fd is inherited, never explicitly LOCK_UN it.
        os.close(phase_fd)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','input-admission','input-sha','run-id','permit','permit-sha','phase-proof'):p.add_argument('--'+name,required=True)
    p.add_argument('--phase-fd',required=True,type=int);a=p.parse_args()
    result=extract_worker(a.root,a.input_admission,a.input_sha,a.run_id,permit_path=a.permit,permit_sha256=a.permit_sha,phase_proof=json.loads(a.phase_proof),phase_fd=a.phase_fd)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
