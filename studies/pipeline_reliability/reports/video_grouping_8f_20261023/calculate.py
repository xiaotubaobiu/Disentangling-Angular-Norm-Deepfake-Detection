"""CPU-only 2x2x2 scoring diagnostic using immutable saved frame predictions."""
from pathlib import Path
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import platform

import numpy as np
import pandas as pd
import scipy
from scipy.special import expit
import sklearn
from sklearn.metrics import roc_auc_score

KEYS = ['domain', 'method_id', 'video_id']
IDENTITY = ['sample_id', 'domain', 'method_id', 'video_id', 'group_id', 'label']
COLUMNS = IDENTITY + ['evaluation_unit', 'score', 'probability']
TOL = 1e-10


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def aggregate(frame, grouping):
    if grouping not in {'correct', 'wrong'}:
        raise ValueError('Unknown grouping')
    if frame.empty or frame.domain.nunique() != 1 or frame.sample_id.duplicated().any():
        raise ValueError('Empty, mixed-domain, or duplicate frame inputs')
    if not set(frame.label) <= {0, 1} or set(frame.label) != {0, 1}:
        raise ValueError('Both binary labels required')
    if not np.isfinite(frame[['score', 'probability']].to_numpy()).all():
        raise ValueError('Nonfinite frame predictions')
    if not frame.probability.between(0, 1).all():
        raise ValueError('Invalid frame probability')
    original = frame.groupby(KEYS, sort=True)
    if not original.size().eq(8).all() or not original.label.nunique().eq(1).all():
        raise ValueError('Expected eight aligned frames per method/video')
    if frame.groupby('video_id').label.nunique().gt(1).any():
        raise ValueError('Bare video ID has conflicting real/fake labels')
    keys = KEYS if grouping == 'correct' else ['domain', 'video_id']
    group = frame.groupby(keys, sort=True)
    result = group.agg(label=('label', 'first'), score=('score', 'mean'),
        probability=('probability', 'mean'), n_frames=('sample_id', 'size'),
        n_methods=('method_id', 'nunique')).reset_index()
    if not result.n_frames.eq(8 * result.n_methods).all():
        raise ValueError('Merged group denominator must include eight frames per method')
    return result


def auroc(labels, scores):
    return float(roc_auc_score(np.asarray(labels), np.asarray(scores)))


def rank_auc(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    real = np.sort(scores[labels == 0])
    fake = scores[labels == 1]
    if not len(real) or not len(fake):
        raise ValueError('Both classes required')
    left = np.searchsorted(real, fake, side='left')
    right = np.searchsorted(real, fake, side='right')
    return float(np.mean((left + .5 * (right - left)) / len(real)))


def verify_means(frame, table, grouping):
    """Separate dictionary + fsum reference; numeric means compared with tolerance."""
    keys = KEYS if grouping == 'correct' else ['domain', 'video_id']
    records = {}
    for row in frame.itertuples(index=False):
        key = tuple(getattr(row, name) for name in keys)
        target = records.setdefault(key, {'z': [], 'p': [], 'methods': set()})
        target['z'].append(float(row.score))
        target['p'].append(float(row.probability))
        target['methods'].add(row.method_id)
    max_error = 0.
    if len(records) != len(table):
        raise ValueError('Reference group count mismatch')
    for row in table.itertuples(index=False):
        key = tuple(getattr(row, name) for name in keys)
        values = records[key]
        n = len(values['z'])
        if n != row.n_frames or len(values['methods']) != row.n_methods:
            raise ValueError('Reference membership/count mismatch')
        error = max(abs(math.fsum(values['z']) / n - row.score),
                    abs(math.fsum(values['p']) / n - row.probability))
        max_error = max(max_error, error)
    if max_error > 1e-12:
        raise ValueError('Independent mean reconstruction failed')
    return max_error


def compare_baseline(normal, path, column):
    old = pd.read_parquet(path).sort_values(KEYS).reset_index(drop=True)
    new = normal.sort_values(KEYS).reset_index(drop=True)
    if not new[KEYS + ['label']].equals(old[KEYS + ['label']]):
        raise ValueError('Original normal-video keys/labels differ')
    np.testing.assert_allclose(new[column], old.score, rtol=0, atol=1e-12)
    delta = abs(auroc(new.label, new[column]) - auroc(old.label, old.score))
    if delta > TOL:
        raise ValueError('Normal-group pooled AUROC does not reproduce original baseline')
    return delta


def report(output, summary, counts):
    lines = ['# 8帧视频分组错误补充实验', '',
        '复用原15个模型、21种视频方法与已保存逐帧分数；零训练、零推理、零GPU。',
        '所有单元格均为 **pooled AUROC**，每个评价单位等权。原论文主表的方法宏平均AUC不是这里的比较基准。',
        '本实验为8帧版，不是另拟的旧16方法/32帧方案。', '',
        '正常分组使用 domain＋method_id＋video_id；错误分组在单域内仅按video_id。',
        'Mean-logit平均原始margin；Mean-probability平均保存的逐帧sigmoid概率，不是sigmoid(mean-logit)。',
        '两种分组共用同一批预测；同一骨干五种子分别汇总，±为模型种子样本标准差。', '']
    for domain in ['cdf', 'ff']:
        c = counts[domain]
        lines += [f'## {domain.upper()}', '',
            f"真实视频：{c['real_units']}（两种分组相同）；伪造：正常{c['correct_fake_units']}个视频，错误{c['wrong_fake_units']}个组。",
            f"错误组中{c['mixed_fake_groups']}组含多个方法，{c['single_method_fake_groups']}组只有一个方法；最大{c['max_methods']}方法/组。", '',
            '| 骨干 | 聚合 | 正常Raw AUC % | 正常L2 AUC % | 错误Raw AUC % | 错误L2 AUC % | 正常Δ pp | 错误Δ pp | 指定方向反转 |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in summary[summary.domain.eq(domain)].to_dict('records'):
            label = {'clip_vit_b16': 'CLIP', 'dinov2_vit_b14': 'DINOv2', 'efficientnet_b0': 'EfficientNet'}[row['backbone']]
            values = [f"{100*row[name+'_mean']:.2f} ± {100*row[name+'_SD']:.2f}" for name in ['correct_raw','correct_l2','wrong_raw','wrong_l2','delta_correct','delta_wrong']]
            lines.append('| ' + ' | '.join([label, row['aggregation'], *values, f"{row['reversal_count']}/5"]) + ' |')
        lines.append('')
    lines += ['## 解释范围', '',
        'Δ_correct=L2_correct−Raw_correct；Δ_wrong=L2_wrong−Raw_wrong。指定方向反转要求同一个模型的Δ_correct<−1e−10且Δ_wrong>1e−10。',
        '错误分组改变评价单位数量与权重，所以这是评价流程对相对结论的影响，不是同一个评价总体上的模型提升。',
        '当前模型采用31方法训练。本实验不能单独归因已删除旧模型的历史提升，也不能单独证明Sigmoid饱和机制。',
        '本次不使用0.5阈值或任何阈值校准，AUC直接依据分数排序；没有调整种子、尺度或方法集合以寻找反转。', '',
        '主结果：auc_by_seed.csv；完整八组合：auc_cells.csv；五种子汇总：summary_by_backbone.csv；输入身份和检查：verification.json。', '']
    (output / '结果与方法.md').write_text('\n'.join(lines))


def run(source, output):
    if (output / 'complete.json').exists():
        raise ValueError('Completed experiment already exists; use a new output directory')
    config = json.loads((source / 'configs/v1.1.json').read_text())
    report_root = source / 'artifacts/analysis'
    bindings = []; paired = []; cells = []; counts = {}; identity_hashes = {}
    max_mean_error = 0.; max_baseline_auc_error = 0.; max_rank_error = 0.
    protocol = {'version': 'video_grouping_8f_21methods_v1', 'frames_per_video': 8,
        'domains': ['ff', 'cdf'], 'backbones': list(config['backbones']), 'seeds': config['seeds'],
        'scope': 'Complete missing wrong-grouping cells using current 8-frame 21-method baselines',
        'grouping_correct': KEYS, 'grouping_wrong': ['domain', 'video_id'],
        'metric': 'pooled AUROC; each video/group has weight one; ties half credit',
        'aggregation': ['mean_logit', 'mean_probability'], 'precision': 'saved FP64 scores/probabilities; pandas group mean for both groupings',
        'no_training': True, 'no_inference': True, 'no_GPU': True,
        'auc_zero_tolerance': TOL, 'source_root': str(source),
        'script_sha256': digest(Path(__file__)), 'config_sha256': digest(source / 'configs/v1.1.json'),
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__}}
    save_json(output / 'protocol.json', protocol)
    for backbone in config['backbones']:
        for seed in config['seeds']:
            run_id = f'{backbone}_seed{seed}'; run_root = report_root / run_id
            complete = json.loads((run_root / 'complete.json').read_text())
            if complete['status'] != 'COMPLETE':
                raise ValueError('Source analysis not complete')
            binding = {'run_id': run_id, 'analysis_complete_sha256': digest(run_root / 'complete.json'),
                'checkpoint_sha256_from_analysis_identity': complete['identity']['inputs_sha256']['checkpoint'], 'files': {}}
            for domain in ['ff', 'cdf']:
                tables = {}; frames_by_kind = {}
                for kind, letter in [('raw','R'),('direction','D')]:
                    path = run_root / 'scores' / domain / f'image_{kind}.parquet'
                    binding['files'][str(path.relative_to(source))] = digest(path)
                    saved = pd.read_parquet(path, columns=COLUMNS)
                    frame = saved[saved.evaluation_unit.eq('video_frame')].reset_index(drop=True)
                    if not np.allclose(frame.probability, expit(frame.score), rtol=0, atol=1e-15):
                        raise ValueError('Stored probabilities do not match frame logits')
                    current_hash = hashlib.sha256(pd.util.hash_pandas_object(frame[IDENTITY], index=False).values.tobytes()).hexdigest()
                    if identity_hashes.setdefault(domain,current_hash) != current_hash:
                        raise ValueError('Frame identity/order differs across models or Raw/L2')
                    frames_by_kind[kind] = frame
                    normal = aggregate(frame,'correct'); wrong = aggregate(frame,'wrong')
                    for grouping, table in [('correct',normal),('wrong',wrong)]:
                        max_mean_error = max(max_mean_error, verify_means(frame,table,grouping))
                        for agg, column in [('mean_logit','score'),('mean_probability','probability')]:
                            value = auroc(table.label,table[column]); other = rank_auc(table.label,table[column])
                            max_rank_error = max(max_rank_error,abs(value-other))
                            if abs(value-other) > 1e-12: raise ValueError('Independent rank-AUC check failed')
                            tables[(kind,grouping,agg)] = value
                            cells.append(dict(backbone=backbone,seed=seed,domain=domain,feature='Raw' if kind=='raw' else 'L2',grouping=grouping,aggregation=agg,pooled_AUC=value,real_units=int(table.label.eq(0).sum()),fake_units=int(table.label.eq(1).sum())))
                    for suffix, column in [('L','score'),('P','probability')]:
                        old = run_root/'scores'/domain/f'video_{letter}{suffix}.parquet'
                        binding['files'][str(old.relative_to(source))] = digest(old)
                        max_baseline_auc_error = max(max_baseline_auc_error,compare_baseline(normal,old,column))
                    real = normal[normal.label.eq(0)].sort_values('video_id').reset_index(drop=True)
                    wr = wrong[wrong.label.eq(0)].sort_values('video_id').reset_index(drop=True)
                    if not np.array_equal(real.video_id,wr.video_id) or not np.array_equal(real[['score','probability']],wr[['score','probability']]):
                        raise ValueError('Real-video predictions changed')
                    one = wrong[wrong.n_methods.eq(1)].merge(normal,on=['domain','video_id','label'],suffixes=('_wrong','_correct'))
                    if not np.array_equal(one[['score_wrong','probability_wrong']].to_numpy(),one[['score_correct','probability_correct']].to_numpy()):
                        raise ValueError('Single-method group scores changed')
                    target=output/'videos'/run_id/domain;target.mkdir(parents=True,exist_ok=True)
                    wrong.to_parquet(target/f'wrong_{kind}.parquet',index=False)
                    c = {'frame_rows':len(frame),'real_units':len(real),'correct_fake_units':int(normal.label.eq(1).sum()),'wrong_fake_units':int(wrong.label.eq(1).sum()),'mixed_fake_groups':int((wrong.label.eq(1)&wrong.n_methods.gt(1)).sum()),'single_method_fake_groups':int((wrong.label.eq(1)&wrong.n_methods.eq(1)).sum()),'max_methods':int(wrong.n_methods.max()),'fake_method_ids':sorted(set(frame.loc[frame.label.eq(1),'method_id']))}
                    if len(c['fake_method_ids']) != 21:raise ValueError('Expected complete 21-method video coverage')
                    if counts.setdefault(domain,c) != c:raise ValueError('Group populations differ across conditions')
                if not frames_by_kind['raw'][IDENTITY].equals(frames_by_kind['direction'][IDENTITY]):
                    raise ValueError('Raw/L2 row alignment failed')
                for agg in ['mean_logit','mean_probability']:
                    a=tables[('raw','correct',agg)];b=tables[('raw','wrong',agg)];c=tables[('direction','correct',agg)];d=tables[('direction','wrong',agg)]
                    dc=c-a;dw=d-b
                    paired.append(dict(backbone=backbone,seed=seed,domain=domain,aggregation=agg,correct_raw=a,wrong_raw=b,correct_l2=c,wrong_l2=d,delta_correct=dc,delta_wrong=dw,interaction=dw-dc,reversal=bool(dc < -TOL and dw > TOL),reverse_reversal=bool(dc > TOL and dw < -TOL)))
            bindings.append(binding)
            save_json(output/'progress.json',{'completed_models':len(bindings),'expected_models':15,'last_run':run_id})
            print(json.dumps({'completed_models':len(bindings),'run_id':run_id}),flush=True)
    per_seed=pd.DataFrame(paired); cell_table=pd.DataFrame(cells); summaries=[]
    for (backbone,domain,aggregation), group in per_seed.groupby(['backbone','domain','aggregation'],sort=True):
        if len(group)!=5 or set(group.seed)!=set(config['seeds']):raise ValueError('Missing model seeds')
        row=dict(backbone=backbone,domain=domain,aggregation=aggregation,seeds=5,reversal_count=int(group.reversal.sum()),reverse_reversal_count=int(group.reverse_reversal.sum()))
        for column in ['correct_raw','correct_l2','wrong_raw','wrong_l2','delta_correct','delta_wrong','interaction']:
            row.update({column+'_mean':float(group[column].mean()),column+'_SD':float(group[column].std(ddof=1)),column+'_min':float(group[column].min()),column+'_max':float(group[column].max())})
        summaries.append(row)
    summary=pd.DataFrame(summaries)
    per_seed.to_csv(output/'auc_by_seed.csv',index=False);cell_table.to_csv(output/'auc_cells.csv',index=False);summary.to_csv(output/'summary_by_backbone.csv',index=False)
    save_json(output/'input_bindings.json',bindings);save_json(output/'group_counts.json',counts)
    report(output,summary,counts)
    outputs={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file() and (p.suffix in {'.parquet','.csv','.md'} or p.name in {'protocol.json','input_bindings.json','group_counts.json'})}
    verification={'status':'COMPUTED_AND_CHECKED','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model_count':len(bindings),'auc_cells':len(cells),'paired_rows':len(paired),'frame_identity_sha256':identity_hashes,'max_normal_baseline_AUC_error':max_baseline_auc_error,'max_independent_mean_error':max_mean_error,'max_independent_rank_AUC_error':max_rank_error,'real_and_single_method_scores_unchanged':True,'raw_l2_and_all_models_share_frames':True,'outputs_sha256':outputs}
    save_json(output/'verification.json',verification)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True);parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'execution.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args.source_root.resolve(),args.output.resolve())
