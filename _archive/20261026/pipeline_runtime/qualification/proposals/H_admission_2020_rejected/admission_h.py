"""Qualification-only post-lock admission during the original D batch drain.

No generic inter-queue check/launch race is claimed solved. Admission is restricted
by an after-lease witness: the unchanged D batch has no pending jobs and still
owns a child lifetime lock, hence cannot have entered its next dispatch batch.
Future D dispatch checks the same already-held per-UUID flock. No CUDA is used
before this witness, and the lease lasts until the whole qualification returns.
"""
from contextlib import contextmanager
from pathlib import Path
import datetime
import hashlib
import json
import os
import time
from . import trainer

SCHEMA='H_MICROQUALIFICATION_D_DRAIN_LEASE_V1'
D_QUEUE_SHA256='3db3d1ebb63fbe233fa19116934268ab999be94faaff59fe1f2d52fdbe1949b8'
STAGE='P2_D_STANDARD_DEVELOPMENT'
RUNS={f'D_dev_efficientnet_b0_lr1e-04_seed{s}_e30' for s in (1007,1042)}


def process_identity(pid):
    p=Path('/proc')/str(int(pid))
    def stat():
        raw=(p/'stat').read_text();parts=raw[raw.rfind(')')+2:].split()
        if parts[0] in ('Z','X'):raise ValueError('process is retired')
        return dict(ppid=int(parts[1]),starttime=int(parts[19]))
    before=stat();args=(p/'cmdline').read_bytes().split(b'\0')
    args=[x.decode() for x in args if x]
    after=stat()
    if before!=after:raise ValueError('process identity changed')
    return dict(pid=int(pid),**before,args=args)


def kernel_lock_owner(path,pid):
    try:
        s=Path(path).stat()
        key=f'{os.major(s.st_dev):02x}:{os.minor(s.st_dev):02x}:{s.st_ino}'
        return any(row.split()[1:6]==['FLOCK','ADVISORY','WRITE',str(pid),key]
                   for row in Path('/proc/locks').read_text().splitlines())
    except FileNotFoundError:return False


def _queue(root):return json.loads((Path(root)/'queue_status.json').read_text())


def create_policy(root):
    """Pin only this live owner/code/config; no queue, GPU or training mutation."""
    root=Path(root).resolve();q=_queue(root);owner=process_identity(q['owner_pid'])
    if q.get('stage')!=STAGE or q.get('detail',{}).get('pending')!=[]:
        raise ValueError('D is not in the permitted pending-empty drain')
    args=owner['args']
    if len(args)<3 or args[1:3]!=['-u','-m']:
        # Existing owner is normally python -u -m; accept python -m too.
        if len(args)<3 or args[1:3]!=['-m','reliability_v2.queue_d']:
            raise ValueError('unrecognized D owner command')
    if 'reliability_v2.queue_d' not in args or str(root) not in args:
        raise ValueError('D owner command does not match artifact root')
    if not kernel_lock_owner(root/'queue.lock',owner['pid']):
        raise ValueError('D owner does not own queue lock')
    code=Path(__file__).with_name('queue_d.py')
    if trainer.digest(code)!=D_QUEUE_SHA256:raise ValueError('D dispatch implementation changed')
    return dict(schema=SCHEMA,scope='H_SOURCE_MICROQUALIFICATION_ONLY',artifact_root=str(root),
        allowed_physical_GPUs=[2,3],D_owner=owner,D_queue_code_sha256=D_QUEUE_SHA256,
        D_execution_sha256=trainer.digest(root/'configs/execution.json'),
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        no_D_mutation=True,no_expiry_time_safety_assumption=True)


def drain_witness(root,config,policy):
    """Must be invoked after H owns its exact lifetime GPU flock."""
    root=Path(root).resolve();visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not kernel_lock_owner(root/'gpu_locks'/f'{visible}.lock',os.getpid()):
        raise ValueError('H lifetime GPU lease is not held before drain witness')
    if policy.get('schema')!=SCHEMA or policy.get('scope')!='H_SOURCE_MICROQUALIFICATION_ONLY':
        raise ValueError('unrecognized qualification-only admission policy')
    if str(root)!=policy.get('artifact_root') or str(root)!=str(Path(config['artifact_root']).resolve()):
        raise ValueError('admission root changed')
    if policy.get('allowed_physical_GPUs')!=[2,3] or visible not in [config['allowed_gpu_uuids'][i] for i in [2,3]]:
        raise ValueError('drain qualification only permits idle physical GPU2/3')
    if policy.get('D_queue_code_sha256')!=D_QUEUE_SHA256 or trainer.digest(Path(__file__).with_name('queue_d.py'))!=D_QUEUE_SHA256:
        raise ValueError('D dispatch implementation changed')
    if trainer.digest(root/'configs/execution.json')!=policy.get('D_execution_sha256'):
        raise ValueError('D configuration already transitioned')
    q=_queue(root);stamp=datetime.datetime.fromisoformat(q['utc']).timestamp()
    if not -2<=time.time()-stamp<=45:raise ValueError('D queue status is stale')
    if q.get('stage')!=STAGE or q.get('detail',{}).get('pending')!=[]:
        raise ValueError('D is not in pending-empty standard drain')
    owner=process_identity(q['owner_pid'])
    if owner!=policy.get('D_owner') or not kernel_lock_owner(root/'queue.lock',owner['pid']):
        raise ValueError('D owner identity or lifetime queue lock changed')
    active=q.get('active',[])
    if not 1<=len(active)<=2 or any(x['gpu'] not in [0,1] or x['run_id'] not in RUNS for x in active):
        raise ValueError('D active membership is not the approved two-card drain')
    witnesses=[]
    for item in active:
        run=item['run_id'];pid=item['pid'];uuid=item['uuid']
        if uuid!=config['allowed_gpu_uuids'][item['gpu']]:raise ValueError('D child UUID changed')
        child=process_identity(pid);args=child['args']
        if child['ppid']!=owner['pid'] or 'reliability_v2.trainer' not in args or run not in args or str(root/'configs/execution.json') not in args:
            raise ValueError('D child command/parent identity differs')
        runlock=root/'runs'/run/'run.lock';gpulock=root/'gpu_locks'/f'{uuid}.lock'
        if (root/'runs'/run/'complete.json').exists():continue
        # A child still owning both lifetime locks after the H claim proves
        # this pending-empty batch cannot already have returned to Queue.run.
        if kernel_lock_owner(runlock,pid) and kernel_lock_owner(gpulock,pid):
            if process_identity(pid)!=child:raise ValueError('D child identity changed')
            witnesses.append(dict(**item,process=child))
    if not witnesses:raise ValueError('no post-lease live D batch witness; refuse admission')
    return dict(status='ADMITTED',schema=SCHEMA,source_only=True,target_used=False,
        physical_gpu=config['allowed_gpu_uuids'].index(visible),device_uuid=visible,
        lease_holder_pid=os.getpid(),D_owner=owner,D_batch_stage=STAGE,D_pending=[],
        post_lease_D_child_witnesses=witnesses,policy=policy,
        policy_sha256=hashlib.sha256(json.dumps(policy,sort_keys=True).encode()).hexdigest(),
        admitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())


@contextmanager
def owned_drain_qualification(root,config,policy):
    from .worker_h import verify_resource_boundary
    verify_resource_boundary(config,require_D_priority=False)
    with trainer.owned_gpu(root,config):
        witness=drain_witness(root,config,policy)
        yield witness
