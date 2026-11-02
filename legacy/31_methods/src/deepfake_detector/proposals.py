"""Count-only scientific policy proposals; never emit usable training manifests."""
from pathlib import Path
import hashlib,json,os,sqlite3
import numpy as np
import pandas as pd
from .manifest import METHODS,save_json,digest,inspect_image,source_ids_from_video


def propose(config,root):
    root=Path(root);data=Path(config['artifact_root'])/'data';base=data/'released_manifest.parquet'
    df=pd.read_parquet(base);roles=json.loads((data/'source_splits/source_roles.json').read_text())
    # Work at declared video/storage units for annotations, not at repeated frames.
    keys=['domain','released_role','method_id','method_type','evaluation_unit','video_id','storage_group_id','source_ids']
    units=df.groupby(keys,dropna=False).agg(indexed_rows=('image_path','size')).reset_index()
    def annotate(value):
        ids=json.loads(value)
        if not ids:return 'unknown'
        found={roles.get(i,'unknown') for i in ids}
        return next(iter(found)) if len(found)==1 else 'mixed:'+','.join(sorted(found))
    # Re-annotate exact tokens without changing roles or content audits. Earlier
    # audit snapshots remain intact; abnormal five-digit prefixes stay unknown.
    units['source_ids']=units.apply(lambda r:json.dumps(source_ids_from_video(r.domain,r.method_type,r.video_id)),axis=1)
    units['source_annotation']=units.source_ids.map(annotate)
    units.to_csv(root/'reports/source_role_annotations.csv',index=False)
    summary=[]
    for kind,methods in METHODS.items():
        for method in methods:
            q=units[(units.domain=='ff')&(units.method_id==method)]
            train=q[q.released_role=='train'];test=q[q.released_role=='test']
            row={'method':method,'type':kind,'count_unit':'indexed_storage_group_NOT_source' if kind=='EFS' else 'video',
              'published_train_units':len(train),'published_test_units':len(test),'published_train_images':int(train.indexed_rows.sum()),'published_test_images':int(test.indexed_rows.sum()),
              'proposed_train_known_original_train_units':int((train.source_annotation=='train').sum()),
              'train_known_val_sources_to_exclude':int((train.source_annotation=='val').sum()),'train_known_test_sources_to_exclude':int((train.source_annotation=='test').sum()),
              'train_unknown_source_units':int((train.source_annotation=='unknown').sum()),'train_mixed_source_units':int(train.source_annotation.str.startswith('mixed:').sum()),
              'proposed_test_known_original_test_units':int((test.source_annotation=='test').sum()),'test_non_test_known_source_units':int(test.source_annotation.isin(['train','val']).sum()),
              'test_unknown_source_units':int((test.source_annotation=='unknown').sum()),'test_mixed_source_units':int(test.source_annotation.str.startswith('mixed:').sum()),
              'fake_calibration_units':0,'proposed_policy_applied':False}
            summary.append(row)
    pd.DataFrame(summary).to_csv(root/'reports/proposed_role_counts.csv',index=False)
    # Capacity upper bound before de-duplication/filtering; keep invalid inputs in
    # metadata but not presumed usable. Each video at most8 features, EFS images.
    unique=df.drop_duplicates(['domain','released_role','method_id','image_path'])
    videos=unique[unique.evaluation_unit=='video_frame'].drop_duplicates(['domain','released_role','method_id','video_id'])
    image_units=int((unique.evaluation_unit=='image').sum());upper=8*len(videos)+image_units
    cache_bytes=upper*4*5*(512+768+1280)
    report={'status':'COUNT_ONLY_PROPOSAL_NOT_APPROVED','released_manifest_sha256':digest(base),'methods':summary,'no_test_or_CDF_to_train_or_calibration':True,
      'source_annotations_are_not_identities':True,'EFS_buckets_are_not_sources':True,'fake_calibration':'unavailable',
      'policy_A':'Keep released roles: inspect byte/RGB conflicts; unresolved cross-role content blocks affected full31-method training/evaluation.',
      'policy_B':'Only original released TRAIN may train, after removing approved known calibration/test sources and content conflicts; only original released TEST may test. Unknown driver/EFS ancestry remains explicitly unknown, not magically isolated. Source-real calibration candidates come from original FF val, not existing FF/CDF test.',
      'independence_limit':'Known source-ID annotations do not certify unknown EFS conditioning or audio drivers. Independent calibration claim requires a stated certified scope; otherwise experiment C remains unavailable.',
      'capacity_upper_bound':{'indexed_video_units':len(videos),'indexed_EFS_image_rows':image_units,'maximum_selected_feature_rows':upper,'FP32_feature_bytes_all15models':cache_bytes,'FP32_feature_GiB':cache_bytes/2**30,'available_bytes':__import__('shutil').disk_usage(root/'artifacts').free,'checkpoints_scores_statistics_extra':'not included; measure model parameter counts before final bound'},
      'content_conflict_counts':'Full byte/RGB audit pending; do not substitute filename/source overlap for proven content copies.'}
    save_json(root/'reports/role_policy_proposal.json',report)
    lines=['# 数据角色方案：仅计数，不执行迁移','',
      'A保留发布角色；B只从原发布train选训练、只从原发布test选测试，并按批准的已知来源隔离及内容副本政策收缩。不会把原test/CDF回灌训练或校准。未知EFS条件/音频驱动不靠桶ID认证。',
      '', '| 方法 | 单位 | 原train | 原FFtest | 已知train来源的训练候选 | train中val/test来源 | train未知 | 已知test来源的FF测试候选 |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for q in summary:lines.append(f"| {q['method']} | {'存储组，非来源' if q['type']=='EFS' else '视频'} | {q['published_train_units']} | {q['published_test_units']} | {q['proposed_train_known_original_train_units']} | {q['train_known_val_sources_to_exclude']}/{q['train_known_test_sources_to_exclude']} | {q['train_unknown_source_units']} | {q['proposed_test_known_original_test_units']} |")
    lines+=['','以上是索引/已知来源上界，不是内容审计后有效train/test计数；EFS须改按去重图片计数，不以存储组制定来源政策。fake calibration为0，TPR95不可用。真实校准140来源视频只是候选，还需来源与内容隔离核验。',
      f"\n按当前全部索引、视频最多8帧、EFS图片未去重计算，15模型FP32特征上界约{cache_bytes/2**30:.1f}GiB；不含模型状态/分数/统计，最终按冻结清单复算。\n"]
    (root/'reports/数据角色提案.md').write_text('\n'.join(lines)+'\n')
    return report
