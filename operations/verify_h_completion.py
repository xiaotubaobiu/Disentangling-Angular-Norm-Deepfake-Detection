#!/usr/bin/env python3
"""Wait for full H freeze, verify all PNGs/masks, and preserve source identity."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
import traceback

for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import pandas as pd
from journal_detector import geometry_h as g,data_h as h


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root',type=Path,required=True)
    p.add_argument('--preparation-pid',type=int,required=True)
    args=p.parse_args();root=args.artifact_root.resolve()
    source=root/'data/H/train_dev_freeze.json';full=root/'data/H/freeze.json'
    expected=g.sha256_file(source);status=root/'operations/h_full_freeze_validation.json'
    report={'status':'WAITING_FOR_FULL_FREEZE','pid':os.getpid(),'preparation_pid':args.preparation_pid,
        'source_freeze_sha256':expected,'full_freeze_path':str(full)}
    g.write_json(status,report);print(json.dumps(report),flush=True)
    try:
        while not full.exists():
            try:os.kill(args.preparation_pid,0)
            except ProcessLookupError:raise RuntimeError('H preparation exited before publishing the full freeze')
            time.sleep(15)
        begun=time.monotonic()
        receipt=h.validate_h_freeze(root,deep=True)
        source_receipt=h.validate_h_freeze(root,source_only=True)
        if g.sha256_file(source)!=expected:raise ValueError('Source freeze changed during remaining geometry generation')
        metadata=receipt['metadata'];coverage=pd.read_csv(metadata['coverage']['path'])
        inventory=json.loads(Path(metadata['inventory']['path']).read_text())
        if len(coverage)!=8431 or coverage.video_uid.nunique()!=8431 or len(inventory)!=8431:
            raise ValueError('Full H candidate denominator is not8431 unique videos')
        inventory=pd.DataFrame(inventory)
        columns=['video_uid','role','label','method_id']
        if not coverage[columns].sort_values('video_uid').reset_index(drop=True).equals(
                inventory[columns].sort_values('video_uid').reset_index(drop=True)):
            raise ValueError('Coverage and candidate inventory identities differ')
        for role in ('source_test','target_test'):
            b=pd.read_parquet(root/f'data/H/common/B/{role}.parquet')
            ab=pd.read_parquet(root/f'data/H/common/AB/{role}.parquet')
            if b.mother_id.tolist()!=ab.mother_id.tolist():raise ValueError('Common test mother pairing changed: '+role)
        grouped=[]
        for (role,method),rows in coverage.groupby(['role','method_id']):
            grouped.append({'role':role,'method_id':method,'candidates':len(rows),'processed':int(rows.processed.sum()),
                'B_geometric':int(rows.B.sum()),'AB_geometric':int(rows.AB.sum()),'common_geometric':int(rows.common.sum()),
                'B_active':int((rows.B&~rows.quarantined).sum()),
                'common_active':int((rows.common&~rows.quarantined).sum()),'quarantined':int(rows.quarantined.sum())})
        report.update(status='PASS',scope='FULL_H_ALL_PNG_MASK_BYTES_AND_DECODED_RGB_PLUS_SOURCE_FREEZE_REVALIDATION',
            elapsed_seconds=time.monotonic()-begun,full_freeze_sha256=g.sha256_file(full),
            source_freeze_unchanged=True,source_revalidation_status=source_receipt['status'],
            candidate_videos=len(coverage),coverage_sha256=g.sha256_file(metadata['coverage']['path']),coverage=grouped)
        g.write_json(status,report)
        with (root/'operations/h_pipeline_notes.md').open('a') as stream:
            stream.write('\nFULL_H_COMPLETION_VERIFIED: all frozen table/manifest/audit evidence hashes and every successful A/B/AB PNG/mask/RGB verified; common test mother pairing verified; all8431 inventory/coverage identities match. Original source freeze SHA256 remains '+expected+'. Source-only validation passes after fullfreeze. See h_full_freeze_validation.json for all-B/common and method failure denominators.\n')
        print(json.dumps(report,ensure_ascii=False),flush=True)
    except Exception as exc:
        report.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc())
        g.write_json(status,report);print(json.dumps(report),flush=True);raise


if __name__=='__main__':main()
