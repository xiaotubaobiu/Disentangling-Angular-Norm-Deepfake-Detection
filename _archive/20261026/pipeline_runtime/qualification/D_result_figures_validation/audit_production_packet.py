"""Independent stdlib-only D figure data/lineage audit.

Does not import the exporter, production helpers, NumPy, or scientific kernels.
Only source CSV values, ECDF counting, subset identities and file hashes are
recomputed. Actual production invocation is owned by the controller; preparation
and --fixture-only checks do not constitute scientific acceptance.
"""
from pathlib import Path
import argparse
import bisect
import csv
import datetime
import hashlib
import itertools
import json
import math
import os
import stat
import sys
import time

R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
SCIENCE=Path('/home/z/workspace/deepfake_detector_publish_20261023/studies/pipeline_reliability/v2/reliability_v2')
VALIDATION=R/'qualification/D_result_figures_validation'
EXPORTER_SHA='b7a0da32a42476f38d59491b1e4729b32f9b6924612bbfd3ab56893153ac513c'
BACKBONES=('clip_vit_b16','dinov2_vit_b14','efficientnet_b0')
SEEDS=(7,42,123,2026,3407)
REPS=('raw','l2')
BUDGETS=(10,20,40,80)
METRICS=('fpr','tpr','pooled_ba')
TABLES=('per_seed_metrics','five_seed_metrics','per_seed_effects','five_seed_effects',
        'per_seed_calibration','five_seed_calibration','per_seed_calibration_comparisons',
        'five_seed_calibration_comparisons','per_seed_oracles','five_seed_oracles',
        'per_seed_numerical','five_seed_numerical')
ACCEPT='PASS_INDEPENDENT_REAL_FIVE_SEED_SUMMARY_STDLIB_RECOMPUTATION'
REPORT=R/'reports/D_result_figures_real_data_independent_acceptance_20261024.json'


def need(condition,message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(4*1024*1024),b''):
            h.update(part)
    return h.hexdigest()


def safe(path,base):
    path=Path(path).absolute();base=Path(base).absolute()
    need(path.is_relative_to(base) and path.resolve().is_relative_to(base.resolve()),'path escape: '+str(path))
    for current in (path,*path.parents):
        need(not current.is_symlink(),'symlink rejected: '+str(current))
        if current==base:
            break
    need(path.is_file() and stat.S_ISREG(path.stat().st_mode),'regular input required: '+str(path))
    return path


class ReadSet:
    def __init__(self,root):
        self.root=root;self.hashes={}
    def bind(self,path,expected=None):
        path=Path(path).absolute()
        base=self.root if path.is_relative_to(self.root) else SCIENCE
        if path==R/'qualification/export_D_result_figures.py':
            base=R
        safe(path,base)
        actual=sha(path)
        need(expected is None or actual==expected,'hash mismatch: '+str(path))
        previous=self.hashes.setdefault(str(path),actual)
        need(previous==actual,'input changed during audit: '+str(path))
        return path
    def js(self,path,expected=None):
        return json.loads(self.bind(path,expected).read_text())
    def csv(self,path,expected=None):
        with self.bind(path,expected).open(newline='') as f:
            reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
        need(fields and len(fields)==len(set(fields)) and all(None not in x and None not in x.values() for x in rows),'malformed CSV')
        return fields,rows
    def verify(self):
        for path,expected in self.hashes.items():
            p=Path(path)
            base=self.root if p.is_relative_to(self.root) else SCIENCE
            if p==R/'qualification/export_D_result_figures.py':
                base=R
            safe(p,base)
            need(sha(p)==expected,'file changed before audit completion: '+path)


def natural(rows,role,rep,real_only=False):
    need(rows,'empty natural-video input')
    uids=set()
    for row in rows:
        need(row['role']==role and row['representation']==rep and row['grouping']=='correct'
             and row['aggregation']=='mean_logit','natural video role/condition mismatch')
        uid=row['video_uid']
        need(uid and uid not in uids,'duplicate natural video identity')
        if 'member_video_uids' in row:
            need(json.loads(row['member_video_uids'])==[uid],'non-natural video membership')
        label=int(row['label'])
        need(label in (0,1) and (not real_only or label==0),'natural video label mismatch')
        need(math.isfinite(float(row['score'])),'nonfinite saved FP64 score')
        uids.add(uid)
    return uids


def check_ecdf_curve(curve,rows):
    """Count ties independently with Python sorted/set and bisect_right."""
    natural(rows,curve['role'],curve['representation'])
    need(all(int(row['label'])==curve['label'] for row in rows),'curve class mixing')
    need(curve['grouping']=='correct' and curve['aggregation']=='mean_logit'
         and curve['score_field']=='score' and curve['score_domain']=='saved_FP64_mean_logit_margin','ECDF score domain mismatch')
    values=[float(row['score']) for row in rows]
    ordered=sorted(values);support=sorted(set(ordered))
    cumulative=[bisect.bisect_right(ordered,value)/len(ordered) for value in support]
    need(curve['natural_video_uids']==[row['video_uid'] for row in rows],'ECDF natural video identity/order mismatch')
    need(curve['score_strings']==[row['score'] for row in rows] and curve['scores']==values,'ECDF saved FP64 values mismatch')
    need(curve['x']==support,'ECDF support mismatch')
    need(curve['cdf']==cumulative,'ECDF cumulative mismatch')
    return len(values)


def indexed(items,key,expected,label):
    result={}
    for item in items:
        identity=key(item)
        need(identity not in result,'duplicate '+label)
        result[identity]=item
    need(set(result)==set(expected),'missing/extra '+label)
    return result


def check_calibration_group(rows,pool,test,boxes):
    """Validate saved values and subset identities, without estimating thresholds."""
    need(len(rows)==802,'CAL requires802 saved rows')
    test_hash=hashlib.sha256('\n'.join(sorted(test)).encode()).hexdigest()
    need(not pool&test and all(x['test_id_hash']==test_hash for x in rows),'CAL/TEST boundary or fixed TEST hash mismatch')
    need({x['calibration_kind'] for x in rows}=={'default','source_real','target_real'},'CAL kinds mismatch')
    for kind in ('default','source_real'):
        fixed=[x for x in rows if x['calibration_kind']==kind]
        need(len(fixed)==1 and int(fixed[0]['repeat'])==-1,'default/source CAL row mismatch')
    target=[x for x in rows if x['calibration_kind']=='target_real']
    by_key=indexed(target,lambda x:(int(x['budget']),int(x['repeat'])),
                   itertools.product(BUDGETS,range(200)),'CAL budget/repeat')
    need(set(boxes)==set(BUDGETS),'CAL boxes missing budget')
    previous={};schedule={};same_set={};metric_values=0
    for budget in BUDGETS:
        available=len(pool)>=budget
        subsets=[];values={name:[] for name in METRICS}
        for repeat in range(200):
            row=by_key[budget,repeat];ids=json.loads(row['calibration_ids'])
            need(isinstance(ids,list) and len(ids)==len(set(ids)) and set(ids)<=pool,'invalid CAL IDs')
            need(row['status']==('ok' if available else 'unavailable'),'forged/missing CAL availability')
            need(len(ids)==(budget if available else 0),'CAL budget cardinality mismatch')
            if available:
                need(row['operating_score_field']=='score','CAL is not saved FP64 operating score')
                need(previous.get(repeat,set())<=set(ids),'CAL IDs are not nested')
                previous[repeat]=set(ids)
            identity=tuple(sorted(ids));subsets.append(list(identity));schedule[budget,repeat]=identity
            point=[]
            for metric in METRICS:
                raw=row[metric]
                if available:
                    value=float(raw)
                    need(math.isfinite(value) and 0<=value<=1,'invalid saved operating metric')
                else:
                    need(raw in ('','NA','NaN','nan'),'unavailable CAL must preserve NA')
                    value=None
                values[metric].append(value);point.append(value);metric_values+=1
            if available:
                need(same_set.setdefault(identity,tuple(point))==tuple(point),'same CAL set/fixed TEST has inconsistent operating metrics')
        count=len({tuple(x) for x in subsets}) if available else 0
        need(all(float(by_key[budget,repeat]['unique_set_count'])==count for repeat in range(200)),'saved distinct CAL count mismatch')
        cell=boxes[budget]
        need(cell['CAL_repeats']==200 and cell['model_count']==1,'CAL repetitions treated as models')
        need(cell['status']==('ok' if available else 'unavailable'),'CAL box availability mismatch')
        reason=None if available else f'budget_{budget}_exceeds_frozen_pool_{len(pool)}'
        need(cell['NA_reason']==reason,'CAL NA reason mismatch')
        need(cell['test_id_hash']==test_hash and cell['unique_calibration_sets']==count,'CAL box TEST/distinct count mismatch')
        need(cell['calibration_set_ids']==subsets,'CAL plotted subset identity mismatch')
        need(cell['metrics']==values,'CAL plotted operating values mismatch')
        if budget==80 and len(pool)==80:
            need(count==1,'full80 pool must have1 distinct subset')
    return schedule,test_hash,metric_values


def audit_packet(output,*,fixture=False):
    started=time.monotonic();output=Path(output).absolute()
    safe(output/'complete.json',R)
    receipt_raw=json.loads((output/'complete.json').read_text())
    data_raw=json.loads((output/'plot_data.json').read_text())
    root=Path(data_raw['root']).absolute()
    if fixture:
        need(root.is_relative_to(VALIDATION) and root!=VALIDATION,'fixture root boundary')
        need(data_raw.get('synthetic') is True and receipt_raw.get('synthetic') is True,'fixture must remain explicitly synthetic')
        need(receipt_raw.get('status')=='SYNTHETIC_FIXTURE_RENDERED_NOT_SCIENTIFIC_RESULTS','fixture receipt status mismatch')
    else:
        need(root==R and output==R/'manuscript_figures/D_accepted_results','canonical real packet required')
        need(data_raw.get('synthetic') is False and receipt_raw.get('synthetic') is False,'synthetic inputs cannot pass real audit')
        need(receipt_raw.get('status')=='EXPORTED_RENDER_REVIEW_PENDING','real export completion required')
    io=ReadSet(root)
    receipt=io.js(output/'complete.json');packet=io.js(output/'plot_data.json')
    need(packet['schema']=='D_ACCEPTED_RESULT_PLOT_DATA_V1' and receipt['schema']=='D_ACCEPTED_RESULT_FIGURES_COMPLETE_V1','packet/receipt schema mismatch')
    expected_keys={'schema','root','synthetic','training_seed_count_per_backbone','backbones','seeds','ecdf','calibration_boxes','calibration_full_rows','tables','input_sha256','uncertainty_scope'}
    need(set(packet)==expected_keys and packet['backbones']==list(BACKBONES) and packet['seeds']==list(SEEDS)
         and packet['training_seed_count_per_backbone']==5,'seed/backbone structure or unexpected pooled result')
    need('No pooled15-model estimator.' in packet['uncertainty_scope'] and '2000 TEST bootstrap' in packet['uncertainty_scope'],'uncertainty scopes missing')
    need(receipt['training_seed_count_per_backbone']==5 and receipt['CAL_repeats_per_model']==200
         and receipt['TEST_bootstrap_repeats']==2000 and receipt['inference_or_training_run'] is False,'receipt scientific scope mismatch')
    need(receipt['input_sha256']==packet['input_sha256'],'receipt/plot input inventory mismatch')
    need(receipt['exporter_path']==str(R/'qualification/export_D_result_figures.py') and receipt['exporter_sha256']==EXPORTER_SHA,'reviewed exporter identity mismatch')
    io.bind(receipt['exporter_path'],EXPORTER_SHA)
    expected_files={'plot_data.json','captions_cn_en.md'}
    expected_files|={f'ECDF_{role}.{ext}' for role in ('source_test','target_full') for ext in ('png','svg')}
    expected_files|={f'CAL_variation_{bb}.{ext}' for bb in BACKBONES for ext in ('png','svg')}
    expected_files|={'tables/'+name+'.csv' for name in TABLES}
    need(set(receipt['artifacts_sha256'])==expected_files,'incomplete/extra figure artifact inventory')
    actual_files={str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()}-{'complete.json'}
    need(actual_files==expected_files,'unexpected/missing physical figure files')
    for rel,h in receipt['artifacts_sha256'].items():
        io.bind(output/rel,h)
    for i,(path,h) in enumerate(packet['input_sha256'].items(),1):
        io.bind(path,h)
        if i%100==0:
            print('FRESH_INPUT_HASHES',i,'OF',len(packet['input_sha256']),flush=True)
    config_path=root/'configs/execution.json'
    need(str(config_path) in packet['input_sha256'],'config missing from bound input inventory')
    config=io.js(config_path,packet['input_sha256'][str(config_path)])
    need(config['artifact_root']==str(root) and config['formal_seeds']==list(SEEDS),'registered config root/seeds mismatch')
    curves=indexed(packet['ecdf'],lambda x:(x['backbone'],x['seed'],x['role'],x['representation'],x['label']),
                   itertools.product(BACKBONES,SEEDS,('source_test','target_full'),REPS,(0,1)),'ECDF curve')
    boxes=indexed(packet['calibration_boxes'],lambda x:(x['backbone'],x['seed'],x['representation'],x['budget']),
                  itertools.product(BACKBONES,SEEDS,REPS,BUDGETS),'CAL cell')
    full=indexed(packet['calibration_full_rows'],lambda x:(x['backbone'],x['seed'],x['representation']),
                 itertools.product(BACKBONES,SEEDS,REPS),'full CAL row group')
    joined={name:{'fields':None,'rows':[]} for name in TABLES}
    points=0;metric_count=0;cal_row_count=0;table_counts={};common=None;summary_hashes={}
    for bb in BACKBONES:
        summary=root/'analysis/D'/bb/'five_seed_summary'
        cp=summary/'complete.json'
        need(str(cp) in packet['input_sha256'],'summary completion not input-bound')
        completion=io.js(cp,packet['input_sha256'][str(cp)]);summary_hashes[bb]=sha(cp)
        ident=completion['identity']
        need(completion.get('schema')=='D_FIVE_SEED_SUMMARY_COMPLETE_V1' and completion['status']=='SUMMARY_COMPLETE'
             and completion['backbone']==bb and completion['training_seed_count']==5,'accepted summary completion mismatch')
        need(ident['seeds']==list(SEEDS) and ident['backbone']==bb and ident['config_sha256']==sha(config_path),'accepted summary seed/config mismatch')
        need(io.js(summary/'identity.json',completion['artifact_sha256']['identity.json'])==ident,'summary identity file mismatch')
        science_root=root/'science' if fixture else SCIENCE
        need(set(ident['code_sha256'])=={'analysis_d.py','scoring_d.py','stats_h.py','cache_d.py','bootstrap_fast.py','summary_d.py'},'summary scientific code coverage mismatch')
        for name,h in ident['code_sha256'].items():
            io.bind(science_root/name,h)
        audits=[]
        for p in sorted((root/'reports').glob(bb+'_five_seed_summary_independent_acceptance*.json')):
            if str(p) not in packet['input_sha256']:
                continue
            a=io.js(p,packet['input_sha256'][str(p)])
            if a.get('status')==ACCEPT and a.get('summary_completion_sha256')==summary_hashes[bb]:
                need(a.get('backbone')==bb and a.get('seeds')==list(SEEDS) and a.get('training_seed_count')==5,'independent summary identity mismatch')
                need(set(a.get('tables',{}))==set(TABLES) and a.get('source_files_bound_sha256'),'independent table/source coverage missing')
                for path,h in a['source_files_bound_sha256'].items():
                    need(packet['input_sha256'].get(path)==h,'independent summary source not bound by exported packet')
                audits.append(a)
        need(audits,'no exact current independently accepted five-seed summary')
        for name in TABLES:
            rel=name+'.csv';need(rel in completion['artifact_sha256'],'summary table not completion-bound')
            fields,rows=io.csv(summary/rel,completion['artifact_sha256'][rel])
            need(rows and all(row['backbone']==bb for row in rows),'joined table backbone mismatch')
            need(all(a['tables'][name]['rows']==len(rows) and a['tables'][name].get('fields_verified',0)>0 for a in audits),'independent table row-count coverage mismatch')
            if name.startswith('per_seed'):
                need({int(x['seed']) for x in rows}==set(SEEDS),'missing seed in table')
            if joined[name]['fields'] is None:
                joined[name]['fields']=fields
            need(joined[name]['fields']==fields,'cross-backbone table field order changed')
            joined[name]['rows'].extend(rows)
        registered={task['seed']:(run,task) for run,task in config['runs'].items()
                    if task.get('route')=='D' and task.get('phase')=='formal' and task.get('backbone')==bb}
        need(set(registered)==set(SEEDS),'registered formal run seed coverage mismatch')
        for seed in SEEDS:
            run,task=registered[seed];need(task.get('recipe_frozen') is True,'unfrozen formal recipe')
            folder=root/'analysis/D'/run
            complete=io.js(folder/'complete.json',ident['analysis_complete_sha256'][str(seed)])
            need(complete['status'] in ('ANALYSIS_COMPLETE','ANALYSIS_COMPLETE_WITH_NUMERIC_NA')
                 and complete['run_id']==run and complete['seed']==seed and complete['backbone']==bb,'analysis identity mismatch')
            ai=complete['identity']
            need(ai['calibration_repeats']==200 and ai['test_bootstrap_repeats']==2000
                 and ai['cache_context']['config_sha256']==sha(config_path),'analysis scientific scope mismatch')
            io.bind(root/'data/D/freeze.json',ai['D_freeze_sha256'])
            for role in ('source_test','target_full'):
                rel=role+'/video_scores.csv'
                _,saved=io.csv(folder/rel,complete['artifact_sha256'][rel])
                for rep in REPS:
                    selected=[x for x in saved if x['grouping']=='correct' and x['aggregation']=='mean_logit' and x['representation']==rep]
                    natural(selected,role,rep)
                    domain='ff' if role=='source_test' else 'cdf'
                    need(all(x['domain']==domain for x in selected) and {int(x['label']) for x in selected}=={0,1},'ECDF role/domain/class coverage mismatch')
                    for label in (0,1):
                        curve=curves[bb,seed,role,rep,label]
                        need(curve['run_id']==run and curve['domain']==domain,'ECDF model/domain lineage mismatch')
                        points+=check_ecdf_curve(curve,[x for x in selected if int(x['label'])==label])
                del saved
            for rep in REPS:
                leaf=folder/'rq4'/f'{rep}_mean_logit'
                rowkey=f'rq4/{rep}_mean_logit/'
                _,cal=io.csv(leaf/'target_cal_scores.csv',complete['artifact_sha256'][rowkey+'target_cal_scores.csv'])
                _,test=io.csv(leaf/'target_test_scores.csv',complete['artifact_sha256'][rowkey+'target_test_scores.csv'])
                pool=natural(cal,'target_cal',rep,True);test_ids=natural(test,'target_test',rep)
                need(all(x.get('domain')=='cdf' for x in cal+test) and {int(x['label']) for x in test}=={0,1},'RQ4 target cohort mismatch')
                _,rows=io.csv(leaf/'calibration.csv',complete['artifact_sha256'][rowkey+'calibration.csv'])
                group=full[bb,seed,rep]
                need(group['aggregation']=='mean_logit' and group['rows']==rows,'saved CAL full rows altered/dropped')
                schedule,test_hash,n=check_calibration_group(rows,pool,test_ids,{budget:boxes[bb,seed,rep,budget] for budget in BUDGETS})
                signature=(schedule,test_hash,pool,test_ids)
                if common is None:
                    common=signature
                need(signature==common,'CAL schedules/pools/TEST not paired across model/representation')
                metric_count+=n;cal_row_count+=len(rows)
    need(set(packet['tables'])==set(TABLES),'missing/extra joined tables')
    for name in TABLES:
        need(packet['tables'][name]==joined[name],'plot JSON changed accepted table fields/rows/NA: '+name)
        fields,rows=io.csv(output/'tables'/(name+'.csv'),receipt['artifacts_sha256']['tables/'+name+'.csv'])
        need(fields==joined[name]['fields'] and rows==joined[name]['rows'],'exported CSV changed accepted table fields/rows/NA: '+name)
        table_counts[name]=len(rows)
    for path,h in io.hashes.items():
        p=Path(path)
        if not p.is_relative_to(output) and p!=R/'qualification/export_D_result_figures.py':
            need(packet['input_sha256'].get(path)==h,'audited source missing from exported input bindings: '+path)
    io.verify()
    return dict(schema='D_RESULT_FIGURES_INDEPENDENT_DATA_AUDIT_V1',
                status='PASS_SYNTHETIC_AUDITOR_SELF_CHECK_NOT_SCIENTIFIC_RESULTS' if fixture else 'PASS_INDEPENDENT_REAL_FIGURE_PACKET_DATA_AND_LINEAGE',
                utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),synthetic=fixture,
                packet_directory=str(output),packet_completion_sha256=sha(output/'complete.json'),
                plotting_data_sha256=sha(output/'plot_data.json'),exporter_sha256=EXPORTER_SHA,
                auditor_sha256=sha(__file__),summary_completion_sha256=summary_hashes,
                input_files=len(packet['input_sha256']),output_files=len(expected_files),
                ECDF_curves=len(curves),ECDF_saved_score_values=points,CAL_cells=len(boxes),
                CAL_metric_values=metric_count,CAL_saved_rows=cal_row_count,
                table_row_counts=table_counts,all12_tables_exact_string_cells_and_NA=True,
                source_test_target_full_and_RQ4_target_test_separate=True,
                five_seeds_per_backbone=True,pooled_15_model_estimator=False,
                score_domain='saved FP64 score; no rank-score substitution',
                independent_ECDF_method='stdlib sorted/set support + bisect_right cumulative counts',
                CAL_scope='Saved FPR/TPR/pooledBA and subset identities only; no threshold or scientific metric kernel recomputation',
                fresh_hashes_before_and_after=True,source_or_output_files_modified=False,
                GPU_or_model_inference=False,scientific_metric_kernel_reimplementation_claim=False,
                seconds=time.monotonic()-started)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--packet',type=Path,required=True)
    ap.add_argument('--fixture-only',action='store_true')
    ap.add_argument('--report',type=Path)
    args=ap.parse_args()
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CUDA_VISIBLE_DEVICES must be empty')
    report=args.report if args.report is not None else (None if args.fixture_only else REPORT)
    if report is not None:
        need(not args.fixture_only and report.absolute()==REPORT,'real acceptance report path is fixed')
        need(not report.exists(),'never overwrite prior acceptance')
    result=audit_packet(args.packet,fixture=args.fixture_only)
    if report is not None:
        with report.open('x') as f:
            json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
