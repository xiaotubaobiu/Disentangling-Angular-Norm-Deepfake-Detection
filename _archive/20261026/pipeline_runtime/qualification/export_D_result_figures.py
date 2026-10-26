"""Frozen D result figures from accepted saved tables, CPU only.

No model, scoring, calibration threshold, bootstrap, or training implementation.
Production requires all three independent five-seed acceptances. Synthetic roots
are confined to this exporter's validation directory and visibly labelled.
"""
from pathlib import Path
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import stat

import numpy as np

R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
VALIDATION=R/'qualification/D_result_figures_validation'
SCIENCE=Path('/home/z/workspace/deepfake_detector_publish_20261023/studies/pipeline_reliability/v2/reliability_v2')
BACKBONES=('clip_vit_b16','dinov2_vit_b14','efficientnet_b0')
SEEDS=(7,42,123,2026,3407)
BUDGETS=(10,20,40,80)
TABLES=('per_seed_metrics','five_seed_metrics','per_seed_effects','five_seed_effects',
        'per_seed_calibration','five_seed_calibration','per_seed_calibration_comparisons',
        'five_seed_calibration_comparisons','per_seed_oracles','five_seed_oracles',
        'per_seed_numerical','five_seed_numerical')
CODE=('analysis_d.py','scoring_d.py','stats_h.py','cache_d.py','bootstrap_fast.py','summary_d.py')
METRICS=('fpr','tpr','pooled_ba')
ACCEPT='PASS_INDEPENDENT_REAL_FIVE_SEED_SUMMARY_STDLIB_RECOMPUTATION'


def require(condition,message):
    if not condition:
        raise ValueError(message)


def safe_path(path,base,*,missing=False,directory=False):
    """Reject escapes, every symlink component and non-regular inputs."""
    path=Path(path).absolute();base=Path(base).absolute()
    require(path.is_relative_to(base),'path escape')
    for current in (path,*path.parents):
        require(not current.is_symlink(),'symlink path rejected')
        if current==base:
            break
    require(path.resolve().is_relative_to(base.resolve()),'resolved path escape')
    if not path.exists():
        require(missing,'missing required input: '+str(path))
    else:
        mode=path.stat().st_mode
        require(stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode),'regular path required')
    return path


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):
            result.update(part)
    return result.hexdigest()


def context(root):
    root=Path(root).absolute()
    require(not root.is_symlink(),'root symlink rejected')
    if root==R:
        safe_path(root,R,directory=True)
        return root,False
    safe_path(root,VALIDATION,directory=True)
    marker=safe_path(root/'SYNTHETIC_FIXTURE.json',root)
    require(json.loads(marker.read_text()).get('status')=='SYNTHETIC_TEST_INPUTS_NOT_RESULTS','unmarked fixture')
    return root,True


class Inputs:
    def __init__(self,root,synthetic):
        self.root=root;self.synthetic=synthetic;self.hashes={}

    def bind(self,path,expected=None,*,science=False):
        base=(self.root/'science' if self.synthetic else SCIENCE) if science else self.root
        path=safe_path(path,base)
        actual=digest(path)
        require(expected is None or actual==expected,'stale/tampered source hash: '+str(path))
        old=self.hashes.setdefault(str(path),actual)
        require(old==actual,'input changed while reading')
        return path

    def json(self,path,expected=None):
        return json.loads(self.bind(path,expected).read_text())

    def csv(self,path,expected=None):
        with self.bind(path,expected).open(newline='') as stream:
            reader=csv.DictReader(stream)
            require(reader.fieldnames and len(set(reader.fieldnames))==len(reader.fieldnames),'CSV headers required')
            fields=list(reader.fieldnames);rows=list(reader)
        require(all(None not in row for row in rows),'malformed CSV')
        return fields,rows


def ecdf_coordinates(values):
    """Unique FP64 support and right-continuous empirical cumulative masses."""
    a=np.asarray(values,dtype=np.float64)
    require(a.ndim==1 and len(a)>0 and np.isfinite(a).all(),'nonempty finite scores required')
    x,count=np.unique(a,return_counts=True)
    return x.tolist(),(np.cumsum(count,dtype=np.int64)/len(a)).tolist()


def _natural(rows,role,representation,*,real_only=False):
    require(rows,'nonempty natural-video score table required')
    ids=[]
    for row in rows:
        require(row['role']==role and row['grouping']=='correct' and
                row['representation']==representation and row['aggregation']=='mean_logit','score role/condition mismatch')
        uid=row['video_uid'];require(uid and uid not in ids,'duplicate natural video_uid')
        if 'member_video_uids' in row:
            require(json.loads(row['member_video_uids'])==[uid],'natural-video grouping required')
        label=int(row['label']);require(label in (0,1) and (not real_only or label==0),'class mismatch')
        require(math.isfinite(float(row['score'])),'finite FP64 scores required')
        ids.append(uid)
    return set(ids)


def _accepted_summary(io,backbone,config_hash,config):
    root=io.root;folder=root/'analysis/D'/backbone/'five_seed_summary'
    cp=folder/'complete.json';complete=io.json(cp)
    require(complete.get('schema')=='D_FIVE_SEED_SUMMARY_COMPLETE_V1' and
            complete.get('status')=='SUMMARY_COMPLETE' and complete.get('backbone')==backbone and
            complete.get('training_seed_count')==5,'complete accepted backbone summary required')
    identity=complete['identity']
    require(identity.get('backbone')==backbone and identity.get('seeds')==list(SEEDS) and
            identity.get('config_sha256')==config_hash and
            set(identity.get('analysis_complete_sha256',{}))=={str(s) for s in SEEDS},'exact current five-seed summary identity required')
    audit_paths=sorted((root/'reports').glob(backbone+'_five_seed_summary_independent_acceptance*.json'))
    matching=[]
    for path in audit_paths:
        audit=io.json(path)
        if audit.get('status')==ACCEPT and audit.get('summary_completion_sha256')==digest(cp):
            matching.append((path,audit))
    require(matching,'missing/current independent summary acceptance required')
    for path,audit in matching:
        require(audit.get('backbone')==backbone and audit.get('training_seed_count')==5 and
                audit.get('seeds')==list(SEEDS),'independent five-seed identity mismatch')
        require(audit.get('source_files_bound_sha256'),'independent source hash inventory required')
        require(set(audit.get('tables',{}))==set(TABLES),'independent audit must cover all12 accepted tables')
        for path,expected in audit['source_files_bound_sha256'].items():
            io.bind(path,expected)
    require(set(identity.get('code_sha256',{}))==set(CODE),'summary scientific code identity required')
    for name,expected in identity['code_sha256'].items():
        io.bind((root/'science' if io.synthetic else SCIENCE)/name,expected,science=True)
    for name,expected in complete['artifact_sha256'].items():
        io.bind(folder/name,expected)
    require(io.json(folder/'identity.json')==identity,'summary identity file mismatch')
    tables={}
    for name in TABLES:
        require(name+'.csv' in complete['artifact_sha256'],'all12 accepted tables required')
        fields,rows=io.csv(folder/(name+'.csv'))
        require(rows and all(row.get('backbone')==backbone for row in rows),'table backbone identity required')
        require(all(audit['tables'][name].get('rows')==len(rows) and audit['tables'][name].get('fields_verified',0)>0 for _,audit in matching),'independent table row coverage mismatch')
        if name.startswith('per_seed'):
            require({int(row['seed']) for row in rows}==set(SEEDS),'table missing registered seed')
        else:
            for row in rows:
                if 'n_seeds' in row:require(int(row['n_seeds'])==5,'four-seed summary forbidden')
                if 'expected_seeds' in row:require(json.loads(row['expected_seeds'])==list(SEEDS),'aggregate seed identity mismatch')
        tables[name]={'fields':fields,'rows':rows}
    runs={}
    registered=[(run,t) for run,t in config['runs'].items()
                if t.get('route')=='D' and t.get('phase')=='formal' and t.get('backbone')==backbone]
    require(len(registered)==5 and {t['seed'] for _,t in registered}==set(SEEDS),'exact5 frozen formal runs required')
    for run,task in registered:
        require(task.get('recipe_frozen') is True,'formal recipe must be frozen')
        seed=task['seed'];folder=root/'analysis/D'/run
        complete=io.json(folder/'complete.json',identity['analysis_complete_sha256'][str(seed)])
        require(complete.get('status') in ('ANALYSIS_COMPLETE','ANALYSIS_COMPLETE_WITH_NUMERIC_NA') and
                complete.get('run_id')==run and complete.get('seed')==seed and
                complete.get('backbone')==backbone,'accepted formal analysis identity mismatch')
        a=complete['identity']
        require(a.get('calibration_repeats')==200 and a.get('test_bootstrap_repeats')==2000 and
                a.get('cache_context',{}).get('config_sha256')==config_hash,'frozen200CAL/2000TEST/current config required')
        io.bind(root/'data/D/freeze.json',a['D_freeze_sha256'])
        for name,expected in complete['artifact_sha256'].items():
            io.bind(folder/name,expected)
        required=[f'{role}/video_scores.csv' for role in ('source_test','target_full')]
        required += [f'rq4/{rep}_mean_logit/{name}.csv' for rep in ('raw','l2')
                     for name in ('calibration','target_cal_scores','target_test_scores')]
        require(all(p in complete['artifact_sha256'] for p in required),'missing bound figure source')
        runs[seed]=(run,folder)
    return tables,runs


def _calibration(io,folder,backbone,seed,rep):
    leaf=folder/'rq4'/f'{rep}_mean_logit'
    _,cal_scores=io.csv(leaf/'target_cal_scores.csv');pool=_natural(cal_scores,'target_cal',rep,real_only=True)
    _,test_scores=io.csv(leaf/'target_test_scores.csv');test=_natural(test_scores,'target_test',rep)
    require(not pool&test,'target CAL/TEST overlap forbidden')
    test_hash=hashlib.sha256('\n'.join(sorted(test)).encode()).hexdigest()
    _,rows=io.csv(leaf/'calibration.csv')
    require(len(rows)==802 and all(row.get('test_id_hash')==test_hash for row in rows),'802 rows on fixed TEST hash required')
    require({row['calibration_kind'] for row in rows}=={'default','source_real','target_real'},'unexpected calibration kind')
    for kind in ('default','source_real'):
        q=[row for row in rows if row['calibration_kind']==kind]
        require(len(q)==1 and int(q[0]['repeat'])==-1,'one default/source row required')
    target=[row for row in rows if row['calibration_kind']=='target_real']
    require({int(row['budget']) for row in target}==set(BUDGETS),'all4 CAL budgets required')
    boxes=[];schedule={};previous={};seen_operating={}
    for budget in BUDGETS:
        q=sorted([row for row in target if int(row['budget'])==budget],key=lambda row:int(row['repeat']))
        require(len(q)==200 and [int(row['repeat']) for row in q]==list(range(200)),'exact200 CAL repeats required')
        sets=[];values={name:[] for name in METRICS}
        for row in q:
            repeat=int(row['repeat']);ids=json.loads(row['calibration_ids'])
            require(isinstance(ids,list) and len(ids)==len(set(ids)) and set(ids)<=pool,'valid unique CAL IDs required')
            available=len(pool)>=budget
            require(row['status']==('ok' if available else 'unavailable'),'legitimate availability must follow frozen pool size')
            require(len(ids)==(budget if available else 0),'CAL subset must have exact available budget')
            if available:
                require(row.get('operating_score_field')=='score','FP64 operating score domain required')
                require(set(previous.get(repeat,()))<=set(ids),'CAL budgets must be nested within repeat')
                previous[repeat]=ids
            for metric in METRICS:
                raw=row.get(metric,'')
                if available:
                    value=float(raw);require(math.isfinite(value) and 0<=value<=1,'finite operating metric in [0,1] required')
                else:
                    require(raw in ('','NA','NaN','nan'),'unavailable budget must preserve NA')
                    value=None
                values[metric].append(value)
            key=tuple(sorted(ids))
            if available:
                point=tuple(values[name][-1] for name in METRICS)
                require(seen_operating.setdefault(key,point)==point,'identical CAL set/fixed model and TEST has inconsistent saved operating metrics')
            sets.append(key);schedule[(budget,repeat)]=key
        distinct=len(set(sets)) if len(pool)>=budget else 0
        require(all(int(float(row['unique_set_count']))==distinct for row in q),'saved unique count disagrees with actual subsets')
        boxes.append(dict(backbone=backbone,seed=seed,representation=rep,budget=budget,
                          CAL_repeats=200,model_count=1,unique_calibration_sets=distinct,
                          status='ok' if len(pool)>=budget else 'unavailable',
                          NA_reason=None if len(pool)>=budget else f'budget_{budget}_exceeds_frozen_pool_{len(pool)}',
                          test_id_hash=test_hash,metrics=values,calibration_set_ids=[list(s) for s in sets]))
    return boxes,rows,schedule,test_hash,pool,test


def build_packet(root):
    """Hash-gate all 3x5 accepted packets; preserve saved statistical tables."""
    root,synthetic=context(root);io=Inputs(root,synthetic)
    if synthetic:io.bind(root/'SYNTHETIC_FIXTURE.json')
    config_path=root/'configs/execution.json';config=io.json(config_path);config_hash=digest(config_path)
    require(config.get('artifact_root')==str(root) and config.get('formal_seeds')==list(SEEDS),'registered root/formal seed contract required')
    tables={name:{'fields':None,'rows':[]} for name in TABLES};runs={}
    # Admission is complete for all three backbones before any plotting/source selection.
    for backbone in BACKBONES:
        part,runs[backbone]=_accepted_summary(io,backbone,config_hash,config)
        for name,t in part.items():
            if tables[name]['fields'] is None:tables[name]['fields']=t['fields']
            require(tables[name]['fields']==t['fields'],'12-table schema differs across accepted backbones')
            tables[name]['rows'].extend(t['rows'])
    ecdf=[];boxes=[];cal_rows=[];common_schedule=None;common_test=None;common_pool=None
    for backbone in BACKBONES:
        for seed in SEEDS:
            run,folder=runs[backbone][seed]
            for role in ('source_test','target_full'):
                _,all_rows=io.csv(folder/role/'video_scores.csv')
                for rep in ('raw','l2'):
                    selected=[r for r in all_rows if r['grouping']=='correct' and r['aggregation']=='mean_logit' and r['representation']==rep]
                    _natural(selected,role,rep)
                    require({int(r['label']) for r in selected}=={0,1},'both real/fake classes required')
                    expected_domain='ff' if role=='source_test' else 'cdf'
                    require(all(r.get('domain')==expected_domain for r in selected),'score domain mismatch')
                    for label in (0,1):
                        q=[r for r in selected if int(r['label'])==label];scores=[float(r['score']) for r in q];x,y=ecdf_coordinates(scores)
                        ecdf.append(dict(backbone=backbone,seed=seed,run_id=run,role=role,domain=expected_domain,
                                         representation=rep,aggregation='mean_logit',grouping='correct',label=label,
                                         score_field='score',score_domain='saved_FP64_mean_logit_margin',
                                         natural_video_uids=[r['video_uid'] for r in q],
                                         score_strings=[r['score'] for r in q],scores=scores,x=x,cdf=y))
            for rep in ('raw','l2'):
                part,rows,schedule,test_hash,pool,test=_calibration(io,folder,backbone,seed,rep)
                if common_schedule is None:common_schedule=schedule;common_test=test_hash;common_pool=pool
                require(schedule==common_schedule and test_hash==common_test and pool==common_pool,
                        'paired/common CAL IDs and fixed TEST required across representations/seeds/backbones')
                boxes.extend(part);cal_rows.append(dict(backbone=backbone,seed=seed,representation=rep,aggregation='mean_logit',rows=rows))
    packet=dict(schema='D_ACCEPTED_RESULT_PLOT_DATA_V1',root=str(root),synthetic=synthetic,
                training_seed_count_per_backbone=5,backbones=list(BACKBONES),seeds=list(SEEDS),
                ecdf=ecdf,calibration_boxes=boxes,calibration_full_rows=cal_rows,tables=tables,
                input_sha256=io.hashes,uncertainty_scope='Five training seeds are descriptive; 200 nested CAL repeats condition on each fixed model/TEST; 2000 TEST bootstrap conditional CIs remain referenced in accepted tables. No pooled15-model estimator.')
    verify_inputs(packet)
    return packet


def verify_inputs(packet):
    root,synthetic=context(packet['root'])
    for path,expected in packet['input_sha256'].items():
        path=Path(path)
        base=root if path.is_relative_to(root) else SCIENCE
        require(not synthetic or base==root,'fixture read escape')
        safe_path(path,base)
        require(digest(path)==expected,'input changed since accepted snapshot: '+str(path))


def _captions(synthetic):
    prefix='SYNTHETIC TEST FIXTURE ONLY — NOT SCIENTIFIC RESULTS.\n\n' if synthetic else ''
    return prefix+"""# D result figure captions / D结果图注

**ECDF.** Saved FP64 mean-logit margins (score field), correct natural-video grouping only. Source source_test and target target_full are separate figures; the target_full RQ2/RQ3 ranking cohort is not the fixed RQ4 target_test cohort and is never reconstructed as a CAL+TEST union. Real/fake colours and the five registered seed line styles remain separate in each backbone and raw/L2 panel. No CAL observations, rank_score_decimal ordering metric, model/seed pooling or score-based example selection enters these ECDFs.

**分数ECDF。** 仅显示正确自然视频分组下保存的FP64 mean-logit margin（score字段），源source_test与目标target_full分图展示。target_full是既定RQ2/RQ3排序队列，不是固定RQ4 target_test，也不重新合并CAL与TEST。三骨干、Raw/L2、真假和五训练种子分别保留；不使用CAL观测、rank_score_decimal排序值、模型/种子混池或按分数选样。

**Calibration variation.** Each paired raw/L2 box shows 200 saved common nested target-real CAL repetitions for one model on its fixed target_test ID hash, at budgets10/20/40/80. Boxes show median, quartiles and full min–max range, not TEST confidence intervals or across-training-model SD. Five seed positions remain distinct within each backbone; no 15-model estimator is formed. Labels u=raw/L2 report actual distinct calibration sets; a full80 pool has u=1 despite200 repeat rows. Legitimate unavailable budgets retain all200 NA rows and the seed position. No thresholds, predictions or inferential statistics are re-estimated. Accepted twelve tables are concatenated with exact fields/rows/NA and backbone identity, preserving default/source-real operating points, mean-probability sensitivity, oracle score-domain labels and original conditional CI references. ECDF target_full and RQ4 fixed target_test retain their separate roles. Training-seed mean±sampleSD, CAL spread and 2000 TEST bootstrap pointwise conditional intervals remain distinct.

**校准波动。** 每个Raw/L2配对箱体来自单一固定训练模型、固定target_test ID哈希上的200次共同嵌套真实CAL预算10/20/40/80。箱体显示中位数、四分位数及完整最小至最大范围，不是TEST置信区间或跨模型训练种子SD。每骨干保留五个种子位置，不构造15模型总体估计；u=Raw/L2为实际独特校准集数，使用完整80池时即使有200行重复也只有u=1。不可用预算保留200个NA和种子位置。仅展示已保存结果，不重新估阈值、预测或推断。十二张已验收表按原字段逐行连接，保留NA、骨干/种子、默认/源真实阈值、mean-probability敏感性、oracle计分域及条件CI引用；target_full和固定target_test不混同。五训练种子均值±样本SD、CAL条件波动及2000次TEST bootstrap逐比较条件区间分开解释。
"""


def render(packet,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.hashsalt':'D-frozen-result-figures'})
    prefix='SYNTHETIC FIXTURE | ' if packet['synthetic'] else ''
    titles={'clip_vit_b16':'CLIP ViT-B/16','dinov2_vit_b14':'DINOv2 ViT-B/14','efficientnet_b0':'EfficientNet-B0'}
    colours={0:'#34699a',1:'#c66728'};repcol={'raw':'#34699a','l2':'#c66728'}
    styles=['solid','dashed','dashdot','dotted',(0,(5,1,1,1,1,1))]
    for role in ('source_test','target_full'):
        fig,axes=plt.subplots(3,2,figsize=(12.6,10.4),squeeze=False)
        fig.subplots_adjust(left=.085,right=.98,bottom=.12,top=.90,wspace=.22,hspace=.35)
        for i,bb in enumerate(BACKBONES):
            for j,rep in enumerate(('raw','l2')):
                ax=axes[i,j]
                for curve in packet['ecdf']:
                    if (curve['role'],curve['backbone'],curve['representation'])!=(role,bb,rep):continue
                    x,y=curve['x'],curve['cdf']
                    ax.step([x[0],*x],[0.,*y],where='post',color=colours[curve['label']],
                            linestyle=styles[SEEDS.index(curve['seed'])],linewidth=1)
                ax.set_title(titles[bb]+' | '+rep.upper(),fontsize=10)
                ax.set_ylim(0,1.02);ax.set_ylabel('Empirical cumulative probability')
                ax.set_xlabel('Saved FP64 mean-logit margin (score)');ax.grid(alpha=.18)
        handles=[Line2D([0],[0],color=colours[v],label=['Real','Fake'][v]) for v in (0,1)]
        handles += [Line2D([0],[0],color='#333333',linestyle=styles[i],label='seed '+str(seed)) for i,seed in enumerate(SEEDS)]
        fig.legend(handles=handles,loc='lower center',ncol=7,frameon=False,bbox_to_anchor=(.5,.025),fontsize=8)
        fig.suptitle(prefix+role+' | correct natural videos | Raw / L2',fontsize=12,y=.97)
        _save_figure(fig,out/('ECDF_'+role));plt.close(fig)
    for bb in BACKBONES:
        fig,axes=plt.subplots(4,3,figsize=(13.4,12.2),squeeze=False)
        fig.subplots_adjust(left=.075,right=.99,bottom=.095,top=.90,hspace=.62,wspace=.25)
        for i,budget in enumerate(BUDGETS):
            for j,metric in enumerate(METRICS):
                ax=axes[i,j];labels=[]
                for k,seed in enumerate(SEEDS,1):
                    cells=[c for c in packet['calibration_boxes'] if c['backbone']==bb and c['seed']==seed and c['budget']==budget]
                    counts={}
                    for rep,offset in [('raw',-.16),('l2',.16)]:
                        cell=next(c for c in cells if c['representation']==rep);counts[rep]=cell['unique_calibration_sets']
                        values=cell['metrics'][metric]
                        if cell['status']=='ok':
                            ax.boxplot([values],positions=[k+offset],widths=.25,whis=(0,100),showfliers=False,
                                       patch_artist=True,manage_ticks=False,
                                       boxprops={'facecolor':repcol[rep],'alpha':.6},
                                       medianprops={'color':'#222222','linewidth':1},
                                       whiskerprops={'color':repcol[rep]},capprops={'color':repcol[rep]})
                        else:ax.text(k+offset,.5,'NA',rotation=90,ha='center',va='center',fontsize=7,color=repcol[rep])
                    labels.append(str(seed)+'\nu='+str(counts['raw'])+'/'+str(counts['l2']))
                ax.set_xticks(range(1,6),labels,fontsize=7);ax.set_xlim(.5,5.5);ax.set_ylim(-.03,1.03)
                ax.set_title('n='+str(budget)+' | '+{'fpr':'FPR','tpr':'TPR','pooled_ba':'Pooled BA'}[metric],fontsize=10)
                ax.set_ylabel('Fixed TEST operating metric');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        fig.suptitle(prefix+titles[bb]+' | nested target-real CAL variation',fontsize=12,y=.975)
        fig.legend(handles=[Patch(facecolor=repcol[r],alpha=.6,label=r.upper()) for r in ('raw','l2')],
                   loc='lower center',bbox_to_anchor=(.5,.038),ncol=2,frameon=False)
        fig.text(.5,.018,'Five model positions; u = distinct Raw/L2 CAL sets; boxes = quartiles / full range, not TEST CI',
                 ha='center',fontsize=8)
        _save_figure(fig,out/('CAL_variation_'+bb));plt.close(fig)


def _save_figure(fig,stem):
    for fmt in ('png','svg'):
        fig.savefig(str(stem)+'.'+fmt,dpi=220,metadata={'Date':None} if fmt=='svg' else None)


def export_packet(root,output=None,*,admit_accepted_production=False):
    root,synthetic=context(root)
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only CUDA_VISIBLE_DEVICES empty required')
    if not synthetic:require(admit_accepted_production,'controller production admission flag required')
    out=Path(output) if output is not None else root/'manuscript_figures/D_accepted_results'
    if not out.is_absolute():out=root/out
    safe_path(out,root,missing=True,directory=True)
    require(synthetic or out==root/'manuscript_figures/D_accepted_results','canonical production output only')
    require(not out.exists(),'never overwrite existing output')
    packet=build_packet(root)
    verify_inputs(packet)
    out.parent.mkdir(parents=True,exist_ok=True)
    safe_path(out.parent,root,directory=True)
    out.mkdir()  # exclusive creation; a second exporter fails before writing
    render(packet,out)
    (out/'tables').mkdir()
    for name,table in packet['tables'].items():
        with (out/'tables'/(name+'.csv')).open('x',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=table['fields']);w.writeheader();w.writerows(table['rows'])
    (out/'plot_data.json').write_text(json.dumps(packet,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
    (out/'captions_cn_en.md').write_text(_captions(synthetic))
    verify_inputs(packet)
    files={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}
    record=dict(schema='D_ACCEPTED_RESULT_FIGURES_COMPLETE_V1',
                status='SYNTHETIC_FIXTURE_RENDERED_NOT_SCIENTIFIC_RESULTS' if synthetic else 'EXPORTED_RENDER_REVIEW_PENDING',
                utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),synthetic=synthetic,
                exporter_path=str(Path(__file__).absolute()),exporter_sha256=digest(__file__),
                input_sha256=packet['input_sha256'],artifacts_sha256=files,
                training_seed_count_per_backbone=5,CAL_repeats_per_model=200,TEST_bootstrap_repeats=2000,
                scope=packet['uncertainty_scope'],inference_or_training_run=False)
    with (out/'complete.json').open('x') as stream:json.dump(record,stream,ensure_ascii=False,indent=2);stream.write('\n')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=R)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--export',action='store_true')
    parser.add_argument('--admit-accepted-production',action='store_true')
    args=parser.parse_args()
    if args.export:
        result=export_packet(args.root,args.output,admit_accepted_production=args.admit_accepted_production)
        print(json.dumps({'status':result['status'],'synthetic':result['synthetic']}))
    else:
        result=build_packet(args.root)
        print(json.dumps({'status':'ALL_THREE_ACCEPTED_INPUTS_VERIFIED','synthetic':result['synthetic'],
                          'ECDF_curves':len(result['ecdf']),'CAL_boxes':len(result['calibration_boxes']),
                          'production_exported':False}))
if __name__=='__main__':main()
