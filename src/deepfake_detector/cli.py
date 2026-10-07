"""Single stage dispatcher; no implicit training or service changes."""
import argparse,json,os
from pathlib import Path


def main(root):
    parser=argparse.ArgumentParser(description='DF40 v1.1 staged reproducible experiments')
    parser.add_argument('stage',choices=['audit','prepare','propose','evidence','status'])
    parser.add_argument('--config',type=Path,default=root/'configs/v1.1.json')
    args=parser.parse_args();config=json.loads(args.config.read_text())
    if args.stage in ['audit','prepare']:
        os.environ['CUDA_VISIBLE_DEVICES']=''
        import fcntl
        from .manifest import audit
        lock=Path(config['artifact_root'])/(args.stage+'.lock')
        with lock.open('w') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if args.stage=='audit':audit(config,config['artifact_root'])
            else:
                from .prepare import prepare
                prepare(config,root)
    elif args.stage=='propose':
        os.environ['CUDA_VISIBLE_DEVICES']=''
        from .proposals import propose
        report=propose(config,root)
        print(json.dumps({'status':report['status'],'capacity_upper_bound':report['capacity_upper_bound']}))
    elif args.stage=='evidence':
        os.environ['CUDA_VISIBLE_DEVICES']=''
        from .evidence import write_evidence
        print(json.dumps(write_evidence(config,root)))
    elif args.stage=='status':
        for name in ['audit_progress.json','audit_summary.json','freeze_complete.json']:
            p=Path(config['artifact_root'])/'data'/name
            if p.exists():print(name,p.read_text())
