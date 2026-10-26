"""Single explicit stage dispatcher, anchored to project config."""
import argparse,json,os
from pathlib import Path

def main(root):
    root=Path(root)
    parser=argparse.ArgumentParser(description='DF40 v1.1 reproducible experiments')
    parser.add_argument('stage',choices=['audit','prepare','propose','evidence','status','freeze','preflight','dry-run','probe','pilot','probe-worker','execute','run','resume','worker','features','analyze','report','restore'])
    parser.add_argument('--config',type=Path,default=root/'configs/v1.1.json')
    parser.add_argument('--approval',type=Path)
    parser.add_argument('--backbone')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--device-uuid')
    parser.add_argument('--session-id')
    parser.add_argument('--microbatch',type=int)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();config=json.loads(args.config.read_text());artifact=Path(config['artifact_root'])
    cpu=['audit','prepare','propose','evidence','status','freeze','preflight','dry-run','analyze','report']
    if args.stage in cpu:os.environ['CUDA_VISIBLE_DEVICES']=''
    result=None
    if args.stage in ['audit','prepare']:
        from .orchestration import exclusive_lock
        with exclusive_lock(artifact/(args.stage+'.lock')):
            if args.stage=='audit':
                from .manifest import audit
                result=audit(config,artifact)
            else:
                from .prepare import prepare
                result=prepare(config,root)
    elif args.stage=='propose':
        from .proposals import propose
        result=propose(config,root)
    elif args.stage=='evidence':
        from .evidence import write_evidence
        result=write_evidence(config,root)
    elif args.stage=='freeze':
        from .freezing import freeze
        from .orchestration import exclusive_lock
        if args.approval is None:parser.error('freeze requires saved authorization evidence --approval')
        with exclusive_lock(artifact/'freeze.lock'):result=freeze(config,root,json.loads(args.approval.read_text()))
    elif args.stage=='preflight':
        from .orchestration import preflight
        result=preflight(config,root)
    elif args.stage=='dry-run':
        from .queueing import dry_plan
        result=dry_plan(config,root)
    elif args.stage=='probe':
        from .qualification import probe
        result=probe(config,root,args.config.resolve())
    elif args.stage=='pilot':
        from .pilot import run_pilot
        result=run_pilot(config,root)
    elif args.stage=='probe-worker':
        from .qualification import probe_worker
        probe_worker(config,args.backbone,args.microbatch,args.device_uuid,args.session_id,args.output)
    elif args.stage=='execute':
        from .execution import execute
        result=execute(config,root,args.config.resolve())
    elif args.stage in ['run','resume']:
        from .queueing import run_queue
        result=run_queue(config,root,args.config)
    elif args.stage in ['worker','features']:
        from .queueing import worker
        worker(config,args.backbone,args.seed,args.device_uuid,args.session_id)
    elif args.stage in ['analyze','report']:
        from .reporting import analyze_run,report
        from .orchestration import exclusive_lock
        with exclusive_lock(artifact/'analysis.lock'):
            result=analyze_run(config,root,args.backbone,args.seed) if args.stage=='analyze' else report(config,root)
    elif args.stage=='restore':
        from .orchestration import restore_llm
        journal=json.loads((artifact/'service_state.json').read_text())
        if Path(f'/proc/{journal["owner_pid"]}').exists():raise ValueError('Service owner alive: do not restore during active GPU work')
        restore_llm(artifact,journal);result=journal
    elif args.stage=='status':
        result={}
        for name in ['data/audit_progress.json','data/audit_summary.json','data/freeze_complete.json','cpu_preflight_complete.json','probe_complete.json','queue_status.json','service_state.json','report/report.json']:
            p=artifact/name
            if p.exists():result[name]=json.loads(p.read_text())
    if result is not None:print(json.dumps(result,ensure_ascii=False,default=str),flush=True)
