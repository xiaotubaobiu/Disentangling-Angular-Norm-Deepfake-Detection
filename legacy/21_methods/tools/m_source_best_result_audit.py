"""Read-only independent selected-four-core source-best M result consumer.

Default is preparation only. An actual run needs an external SHA-bound request,
the real six-role admission, actual producer wait/exit0 and GNU time witnesses,
and the original lifetime shared CPU slot. No writer sink, output/staging mkdir,
model, feature-dot reconstruction, training, target inference or permit issuer.
CLI emits a report to stdout; its caller owns any later append-only publication.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math, os, re, resource, stat, sys
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CODE=Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:sys.path.insert(0,str(CODE))
from reliability_v2 import multiplicity_bundle as bundle
SCHEMA='M_INDEPENDENT_SAVED_RESULT_AUDIT_V1'
SINK_SCHEMA='M_COMPACT_STATISTIC_SINK_V1'
REQUEST_SCHEMA='M_INDEPENDENT_SAVED_RESULT_AUDIT_REQUEST_V1'
ROOT=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
H_KERNEL=ROOT/'engineering/source_best_M/audits/h_pair_result_audit.py'
H_SHA='6cd3bff403cc2f7c27a870d092023e4c8677f5d9386668abbc439829eadaa4e5'
D_KERNEL=Path(__file__).absolute().parent/'m_source_best_d_numeric.py'
D_SHA='a7b079965fd34acbf2e974480aa197ded476cbc21f1447aeac523e1c4aab1dd0'
SEEDS=(7,42,123,2026,3407)
BACKBONES=('clip_vit_b16','dinov2_vit_b14','efficientnet_b0')
AGGREGATIONS=('mean_logit','mean_probability')
H_METHODS=('Deepfakes','Face2Face','FaceSwap','NeuralTextures')
LIMIT_KEYS={'max_total_bytes','max_temporary_bytes','max_artifact_bytes','max_json_bytes','max_json_array_bytes','max_json_cells','max_bundle_members','max_table_rows','max_decoded_bytes'}
KINDS={'json','csv','scores','array','parquet'}
PROOF_KEYS={'input_admission','cache_admission','production_permit','producer_execution','producer_launch','producer_exit','producer_stdout','producer_time_v'}
CORE_ENDPOINTS={
 'RQ1':dict(domain='DFD_TARGET_TEST',cohort='all_B',aggregation='mean_logit',comparison='H10-H00',source_FF_validation=True,mean_probability_sensitivity=True),
 'RQ2':dict(role='target_full',grouping='correct',aggregation='mean_logit',metric='pooled_auc',comparison='L2-Raw'),
 'RQ3':dict(role='target_full',aggregations=list(AGGREGATIONS),metric='pooled_auc',comparison='delta_collision-delta_correct'),
 'RQ4':dict(role='target_test',calibration_budget=20,comparison='target_real-source_real',metrics=['fpr','tpr','pooled_ba'],CAL_repeats=200,TEST_repeats=2000)}
LAYERS=dict(training_seeds=list(SEEDS),CAL_subsets=200,conditional_TEST_bootstrap=2000,resampling_seed=20261023,training_seeds_resampled=False,intervals='95% pointwise conditional percentile; no global significance claim',five_seed_summary_computed_here=False)

def require(ok,why):
    if not ok:raise ValueError(why)

def sha(path):
    h=hashlib.sha256()
    with safe(path,file=True).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()

def safe(path,*,file=False,root=None):
    p=Path(path).absolute()
    require(p.resolve()==p and not any(x.is_symlink() for x in (p,*p.parents)),'canonical nonsymlink path required')
    if root is not None:require(p.is_relative_to(Path(root).absolute()),'path escape')
    if file:
        require(p.exists(),'regular file missing')
        st=p.stat();require(stat.S_ISREG(st.st_mode) and st.st_nlink==1,'private regular file required')
    return p

def typed(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def hash_value(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None,'external SHA256 required')
    return value

def json_read(path,limit=1024**2):
    p=safe(path,file=True);require(p.stat().st_size<=limit,'bounded JSON evidence required')
    def pairs(items):
        d={}
        for k,v in items:
            require(k not in d,'duplicate JSON key');d[k]=v
        return d
    def bad(x):raise ValueError('nonfinite JSON evidence')
    d=json.loads(p.read_bytes(),object_pairs_hook=pairs,parse_constant=bad)
    typed(d);return d

def artifact_name(name):
    require(type(name) is str and name and '\\' not in name and not Path(name).is_absolute() and all(x not in ('','.','..') for x in name.split('/')),'safe relative artifact name required')
    return name

class Pins:
    """Every expected digest is sealed before computation and never refreshed."""
    def __init__(self):self.expected={}
    def add(self,path,digest):
        path=str(safe(path,file=True));hash_value(digest)
        require(path not in self.expected or self.expected[path]==digest,'conflicting fixed SHA')
        require(sha(path)==digest,'fixed SHA drift '+path);self.expected[path]=digest
        return Path(path)
    def check(self):
        for path,digest in self.expected.items():require(sha(path)==digest,'initial fixed SHA drift '+path)

def expected_inventory(route):
    out={}
    if route=='H':
        for aggregation in AGGREGATIONS:
            for h in ('B','AB'):out[f'{aggregation}/source_cal_{h}.csv']='scores'
            for domain in ('SOURCE_TEST','TARGET_TEST'):
                folder=f'{aggregation}/{domain}/all_B'
                for c in ('H00','H10'):out[f'{folder}/{c}_scores.csv']='scores'
                for n in ('metrics','comparisons'):out[f'{folder}/{n}.csv']='csv'
                for n in ('thresholds','independent_saved_score_checks','conditional_bootstrap','independent_kernel_checks'):out[f'{folder}/{n}.json']='json'
    elif route=='D':
        out.update({'training_radii.npy':'array','training_reference.json':'json'})
        for role in ('source_test','target_full'):
            out[f'{role}/frame_margins.parquet']='parquet'
            for n in ('video_scores','control_video_scores'):out[f'{role}/{n}.csv']='scores'
            for n in ('metrics','effects','video_diagnostics','method_diagnostics','numerical_audits'):out[f'{role}/{n}.csv']='csv'
            for n in ('pair_transition_ledgers','closure_bootstrap'):out[f'{role}/{n}.json']='json'
            for r in ('raw','l2'):
                for a in AGGREGATIONS:out[f'{role}/normal_bootstrap_{r}_{a}.json']='json'
        for role in ('source_cal','target_cal','target_test'):out[f'rq4/{role}_frame_margins.parquet']='parquet'
        out['rq4/scope.json']='json'
        for r in ('raw','l2'):
            for a in AGGREGATIONS:
                folder=f'rq4/{r}_{a}'
                for role in ('source_cal','target_cal','target_test'):out[f'{folder}/{role}_scores.csv']='scores'
                out[f'{folder}/calibration.csv']='csv'
                for n in ('independent_calibration_recompute','primary_n20_vs_source_bootstrap'):out[f'{folder}/{n}.json']='json'
        out['independent_rank_recomputation.json']='json'
    else:raise ValueError('frozen D/H selected-M route required')
    out['limitations.json']='json';return out

def producer_package_codes():
    return {str(p):sha(p) for p in sorted((CODE/'reliability_v2').glob('*.py'))}

def producer_transport_codes():
    names=('analysis_d.py','multiplicity_bundle.py','multiplicity_codec.py','sensitivity_io.py','source_best_cache.py','source_best_sink.py')
    return {name:sha(CODE/'reliability_v2'/name) for name in names}

ENTRY_SHA=sha(__file__)
_BASE_SOURCES={str(Path(__file__).absolute()):ENTRY_SHA,str(H_KERNEL):H_SHA}
_BASE_SOURCES.update({str(Path(m.__file__).absolute()):sha(m.__file__) for m in (bundle,bundle.codec,bundle.io)})
_H=None

def dependency_pins(*,include_D=True):
    pins=dict(_BASE_SOURCES)
    if include_D:pins[str(D_KERNEL)]=D_SHA
    return pins

def h_kernel():
    global _H
    require(sha(H_KERNEL)==H_SHA,'frozen independent H kernel changed')
    if _H is None:
        spec=importlib.util.spec_from_file_location('m_independent_frozen_H',H_KERNEL)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);_H=module
    require(sha(H_KERNEL)==H_SHA,'H kernel changed during import');return _H

class ReadOnlyArtifacts(Mapping):
    """Authenticated inventory with per-asset decoding and no retained payload cache."""
    def __init__(self,output,complete_sha256,*,route):
        self.output=safe(output);require(self.output.is_dir(),'completed result directory required')
        self.route=route;self.inventory=expected_inventory(route);self.pins=Pins();self.decoded_assets=0;self.decoded_names=set()
        self.complete_sha256=hash_value(complete_sha256)
        self.pins.add(self.output/'complete.json',complete_sha256)
        self.complete=json_read(self.output/'complete.json')
        self.claim=json_read(self.output/'identity.json');self.pins.add(self.output/'identity.json',sha(self.output/'identity.json'))
        require(json_read(self.output/'identity.json')==self.claim,'identity changed while sealing initial SHA')
        require(self.claim.get('schema')==SINK_SCHEMA and self.claim.get('inventory')==self.inventory,'exact D60/H37 kind inventory required')
        require(self.claim.get('implementations')==producer_transport_codes(),'transport/code SHA inventory drift')
        self.limits=self.claim['limits']
        require(set(self.limits)==LIMIT_KEYS and all(type(x) is int and x>0 for x in self.limits.values()),'explicit positive decoding limits required')
        self.identity=self.claim['identity'];self.claim_sha=hashlib.sha256(typed(self.claim)).hexdigest()
        require(self.complete==dict(schema=SINK_SCHEMA,status='ALL_DECLARED_ARTIFACTS_COMMITTED_REQUIRES_SCIENTIFIC_ADMISSION',identity_sha256=self.claim_sha,receipts=self.complete.get('receipts'),scientific_admitted=False),'unadmitted transport completion contract required')
        require(type(self.complete['receipts']) is dict and set(self.complete['receipts'])==set(self.inventory),'complete exact receipt inventory required')
        self.records={};self.paths={};payloads=set()
        for name,kind in self.inventory.items():
            artifact_name(name);path=self.output/'artifacts'/(name+('.bundle' if kind=='json' else ''))
            rp=self.output/'.receipts'/(hashlib.sha256(name.encode()).hexdigest()+'.json')
            self.pins.add(rp,self.complete['receipts'][name]);record=json_read(rp)
            aid=dict(sink_identity_sha256=self.claim_sha,artifact=name,kind=kind)
            require(set(record)=={'schema','identity','files','bundle'} and record['schema']==SINK_SCHEMA and record['identity']==aid,'artifact receipt schema/context mismatch')
            files=record['files'];require(type(files) is dict and files,'nonempty chunk/file SHA manifest required')
            actual=set(x for x in path.rglob('*') if x.is_file()) if kind=='json' else {path}
            wanted=set()
            for rel,pin in files.items():
                artifact_name(rel);target=safe(self.output/'artifacts'/rel,file=True,root=self.output/'artifacts')
                require((target.is_relative_to(path) if kind=='json' else target==path),'receipt path escape/asset mismatch')
                require(target not in payloads,'duplicate asset file');wanted.add(target);payloads.add(target);self.pins.add(target,pin)
            require(actual==wanted,'asset manifest/chunk inventory differs')
            if kind=='json':
                require(type(record['bundle']) is dict and record['bundle'].get('complete_sha256')==files[str((path/'complete.json').relative_to(self.output/'artifacts'))],'typedcodec completion SHA mismatch')
                require(set(p.name for p in path.iterdir())=={'complete.json','transaction.json','metadata.json.gz','multiplicities.npz','writer.lock'},'typedcodec member inventory differs')
                require((path/'writer.lock').stat().st_size==0,'nonempty bundle writer lock')
            else:require(record['bundle'] is None,'non-JSON bundle receipt forbidden')
            self.records[name]=record;self.paths[name]=path
        require(set(self.output.iterdir())=={self.output/x for x in ('identity.json','complete.json','artifacts','.receipts','.staging','writer.lock')},'foreign/missing result member')
        require(not any((self.output/'.staging').iterdir()) and (self.output/'writer.lock').stat().st_size==0,'completed idle result with empty retained staging required')
        self._namespace()
        self.check()
    def __iter__(self):return iter(self.inventory)
    def __len__(self):return len(self.inventory)
    def _namespace(self):
        require(set(self.output.iterdir())=={self.output/x for x in ('identity.json','complete.json','artifacts','.receipts','.staging','writer.lock')},'result namespace changed')
        require(not any((self.output/'.staging').iterdir()) and (self.output/'writer.lock').stat().st_size==0,'completed idle namespace/staging changed')
        expected_dirs={self.output/'artifacts'}
        for path in self.pins.expected:
            p=Path(path)
            if p.is_relative_to(self.output/'artifacts'):
                expected_dirs.update(x for x in p.parents if x.is_relative_to(self.output/'artifacts'))
        for p in self.output.rglob('*'):
            safe(p,root=self.output)
            require(p.is_dir() or p.is_file(),'unexpected nonregular namespace member')
            if p.is_relative_to(self.output/'artifacts') and p.is_dir():require(p in expected_dirs,'undeclared artifact directory')

        expected_payload={p for p in self.pins.expected if Path(p).is_relative_to(self.output/'artifacts')}
        require({str(p) for p in (self.output/'artifacts').rglob('*') if p.is_file()}==expected_payload,'undeclared artifact payload')
        require({p.name for p in (self.output/'.receipts').iterdir()}=={hashlib.sha256(n.encode()).hexdigest()+'.json' for n in self.inventory},'foreign artifact receipt')
    def check(self):
        self._namespace();self.pins.check()
        allocated=sum(max(p.stat().st_size,p.stat().st_blocks*512) for p in (self.output,*self.output.rglob('*')))
        require(allocated<=self.limits['max_total_bytes'],'actual output allocation exceeds fixed cap')

    def _table_bound(self,path):
        old=csv.field_size_limit();csv.field_size_limit(min(self.limits['max_artifact_bytes'],self.limits['max_decoded_bytes']))
        try:
            with path.open(newline='',encoding='utf-8') as stream:
                rows=csv.reader(stream);header=next(rows,None)
                require(header and len(header)==len(set(header)),'distinct table header required')
                decoded=4096+sum(128+4*len(x.encode()) for x in header)
                for i,row in enumerate(rows):
                    require(i<self.limits['max_table_rows'] and len(row)==len(header),'CSV row/width budget')
                    decoded+=8+sum(128+4*len(x.encode()) for x in row)
                    require(decoded<=self.limits['max_decoded_bytes'],'CSV decoded allocation budget')
        finally:csv.field_size_limit(old)
    def __getitem__(self,name):
        if name not in self.inventory:raise KeyError(name)
        path=self.paths[name];kind=self.inventory[name];record=self.records[name]
        for rel,pin in record['files'].items():require(sha(self.output/'artifacts'/rel)==pin,'asset initial SHA drift')
        if kind=='json':
            require(len(json_read(path/'complete.json').get('array_specs',{}))<=self.limits['max_bundle_members'],'typedcodec numeric member cap')
            value=bundle.read_bundle(path,expected_complete_sha256=record['bundle']['complete_sha256'],expected_identity=record['identity'],max_cells=self.limits['max_json_cells'],max_json_bytes=self.limits['max_json_bytes'],max_array_bytes=self.limits['max_json_array_bytes'])
        else:
            require(path.stat().st_size<=self.limits['max_artifact_bytes'],'artifact byte bound')
            if kind=='array':
                with path.open('rb') as f:
                    require(np.lib.format.read_magic(f)==(1,0),'NPY1 header required')
                    shape,fortran,dtype=np.lib.format.read_array_header_1_0(f,max_header_size=4096)
                    require(not dtype.hasobject and not fortran and all(type(x) is int and x>=0 for x in shape),'safe numeric NPY shape/type')
                    size=math.prod(shape)*dtype.itemsize
                    require(size<=self.limits['max_decoded_bytes'] and f.tell()+size==path.stat().st_size,'array allocation/length bound')
                value=np.load(path,allow_pickle=False)
            elif kind=='parquet':
                m=pq.ParquetFile(path).metadata;decoded=4096+m.num_rows*(m.num_columns*128+8)
                require(m.num_rows<=self.limits['max_table_rows'],'Parquet row bound')
                for i in range(m.num_row_groups):
                    for j in range(m.row_group(i).num_columns):
                        c=m.row_group(i).column(j)
                        require(c.compression=='UNCOMPRESSED' and set(c.encodings)<={'PLAIN','RLE'} and 'PLAIN' in c.encodings,'unregistered Parquet encoding')
                        decoded+=4*c.total_uncompressed_size+128*c.num_values
                require(decoded<=self.limits['max_decoded_bytes'],'Parquet decoded allocation bound');value=pd.read_parquet(path)
            else:
                self._table_bound(path)
                value=pd.read_csv(path,float_precision='round_trip',dtype={'rank_score_decimal':str,'video_id':str},keep_default_na=kind!='scores')
                if kind=='scores':
                    for c in ('member_video_uids','member_group_ids','precision_history'):
                        if c in value:value[c]=value[c].map(lambda x:tuple(json.loads(x)))
            if isinstance(value,pd.DataFrame):require(len(value)<=self.limits['max_table_rows'] and value.memory_usage(deep=True).sum()<=self.limits['max_decoded_bytes'],'decoded table allocation bound')
        for rel,pin in record['files'].items():require(sha(self.output/'artifacts'/rel)==pin,'asset changed during read')
        self.decoded_assets+=1;self.decoded_names.add(name);return value
    def verify_all_decodable(self):
        for name in self:
            if name not in self.decoded_names:self[name]
        self.check()

def h_rows(table,aggregation,*,cal=False):
    k=h_kernel();required=set(k.IDS)|{'score','rank_score_kind','rank_score_decimal','rank_score_export_digits','ranking_usable','representation','grouping','aggregation'}
    require(isinstance(table,pd.DataFrame) and required<=set(table) and not {'weight','draw_id'}&set(table),'saved natural score columns required')
    out=[]
    for row in table.to_dict('records'):
        require(row['representation']=='raw' and row['grouping']=='correct' and row['aggregation']==aggregation,'H native Raw correct aggregation')
        require(int(row['rank_score_export_digits'])>=350,'original rank precision retained')
        y=int(row['label']);s=float(row['score']);require(y in (0,1) and y==row['label'] and math.isfinite(s),'finite binary score')
        require(aggregation!='mean_probability' or 0<=s<=1,'native mean-probability score range')
        if row['rank_score_kind']=='float64':require(float(row['rank_score_decimal'])==s,'FP64 rank agrees with saved operating score')

        require(all(type(row[x]) is str and row[x] for x in ('video_uid','method_id','group_id')),'natural score identity')
        require((row['method_id']=='real')==(y==0) and (not cal or y==0),'real-only CAL / method agreement')
        flag=row['ranking_usable'];require(isinstance(flag,(bool,np.bool_)),'typed ranking validity')
        out.append(dict(video_uid=row['video_uid'],label=y,method_id=row['method_id'],group_id=row['group_id'],score=s,rank=k.parse_rank(row['rank_score_kind'],str(row['rank_score_decimal'])),ranking_usable=bool(flag)))
    require(out and len({r['video_uid'] for r in out})==len(out),'nonempty unique natural videos')
    return sorted(out,key=lambda r:r['video_uid'])

def audit_H_artifacts(artifacts,*,manifest_ids=None):
    require(set(artifacts)==set(expected_inventory('H')),'complete H37 inventory required')
    k=h_kernel();summaries=[]
    limitations=artifacts['limitations.json']
    require(limitations.get('checkpoint_selection')=='source_best' and limitations.get('update_matched_primary_claim') is False and limitations.get('test_bootstrap_repeats')==2000 and limitations.get('resampling_seed')==20261023 and limitations.get('source_CAL_resampled') is False and limitations.get('training_seeds_resampled') is False,'frozen H sensitivity/CAL/TEST policy required')
    for aggregation in AGGREGATIONS:
        cal={h:h_rows(artifacts[f'{aggregation}/source_cal_{h}.csv'],aggregation,cal=True) for h in ('B','AB')}
        if manifest_ids is not None:
            for rows in cal.values():require(k.identities(rows)==manifest_ids['B/SOURCE_CAL'],'saved CAL exact admitted native IDs')
        for domain in ('SOURCE_TEST','TARGET_TEST'):
            folder=f'{aggregation}/{domain}/all_B';methods=H_METHODS if domain=='SOURCE_TEST' else ('DeepFakeDetection',)
            cells={c:h_rows(artifacts[f'{folder}/{c}_scores.csv'],aggregation) for c in ('H00','H10')}
            if manifest_ids is not None:
                for rows in cells.values():require(k.identities(rows)==manifest_ids['B/'+domain],'saved TEST exact admitted native IDs')
            _,groups,_,thresholds,evaluate=k.prepare(cells,cal,methods,aggregation)
            values=evaluate(np.ones(len(groups),dtype=np.int64));point=k.flatten(values)
            saved_thresholds=artifacts[folder+'/thresholds.json'];require(set(saved_thresholds)=={'B','AB'},'exact two source thresholds')
            for h,t in thresholds.items():k.close(saved_thresholds[h],t,'source CAL threshold')
            metrics=artifacts[folder+'/metrics.csv'].to_dict('records')
            require(len(metrics)==4 and {(x['cell'],x['threshold_kind']) for x in metrics}=={(c,t) for c in cells for t in k.KINDS},'H point row inventory')
            for row in metrics:
                v=values[row['cell']][row['threshold_kind']]
                for name in k.METRICS+k.COUNTS+('threshold',):k.close(row[name],v[name],'H point '+name)
                require(row['model_history']==k.HISTORY[row['cell']],'native input trained history')
                for flag in ('macro_valid','macro_operating_valid','ranking_usable'):require(bool(row[flag])==v[flag] and isinstance(row[flag],(bool,np.bool_)),'H point validity')
            expected={x:v for x,v in point.items() if ':comparison:' in x};seen=set()
            comparisons=artifacts[folder+'/comparisons.csv'].to_dict('records')
            require(len(comparisons)==len(expected),'exact H10-H00 effect inventory')
            for row in comparisons:
                key=f'{row["threshold_kind"]}:comparison:{row["comparison"]}:{row["metric"]}'
                require(key in expected and key not in seen,'H signed effect inventory');seen.add(key);k.close(row['estimate'],expected[key],key)
            for suffix,kind in [('independent_kernel_checks','kernel'),('independent_saved_score_checks','saved')]:
                checks=artifacts[f'{folder}/{suffix}.json']
                require(len(checks)==4 and {(x['cell'],x['threshold_kind']) for x in checks}=={(c,t) for c in cells for t in k.KINDS},'producer guard row inventory')
                for check in checks:
                    usable=values[check['cell']][check['threshold_kind']]['ranking_usable']
                    expected_status=('PASSED' if usable else 'OPERATING_PASSED_RANKING_UNRESOLVED') if kind=='kernel' else ('PASS' if usable else 'PASS_OPERATING_ONLY_NUMERIC_AUC_NA')
                    require(check['status']==expected_status,'producer guard retains numerical NA')
                    if kind=='kernel':require(math.isfinite(check['max_abs_error']) and 0<=check['max_abs_error']<=1e-10,'producer guard finite error')
            summary=k.check_bootstrap(artifacts[folder+'/conditional_bootstrap.json'],cells,cal,methods,aggregation,repeats=2000,seed=20261023)
            summary.update(source_cal={h:dict(n_real=len(q),false_positives=sum(r['score']>=thresholds[h] for r in q),allowed_false_positives=math.floor(.05*len(q)+1e-12),calibration_uid_sha256=hashlib.sha256('\n'.join(sorted(r['video_uid'] for r in q)).encode()).hexdigest()) for h,q in cal.items()},aggregation=aggregation,domain=domain,cohort='all_B',cells=['H00','H10'],comparisons=['H10-H00'],numeric_NA_cells=[c for c in cells if not values[c]['default']['ranking_usable']],independent_point_estimates={x:float(v) if math.isfinite(v) else None for x,v in point.items()},source_real_thresholds=thresholds)
            summaries.append(summary)
    return dict(status='NUMERIC_SELECTED_NATIVE_B_CORE_RECOMPUTED',audited_scope='SELECTED_FOUR_CORE_ONLY',accepted_for_selected_core=True,cohorts=summaries,coverage=['All four H native-B cohorts, points, source thresholds, signed H10-H00, exact applicable2000 draws/NA/validcounts/pointwise intervals'],unverified_numeric_artifacts=[f'{a}/{d}/all_B/{n}.json' for a in AGGREGATIONS for d in ('SOURCE_TEST','TARGET_TEST') for n in ('independent_kernel_checks','independent_saved_score_checks')],unverified_numeric_detail='Producer self-check error magnitudes are bounded/status-checked; independent core point/draw/interval values are recomputed directly instead.',all_output_metrics_numerically_recomputed=False,blockers_for_selected_core=[])

def job_spec(route,backbone,seed):
    require(route in ('D','H') and backbone in BACKBONES and type(seed) is int and seed in SEEDS,'registered five-seed selected job')
    runs={'D':f'D_formal_{backbone}_seed{seed}'} if route=='D' else {h:f'H_formal_{backbone}_{h}_seed{seed}' for h in ('B','AB')}
    return dict(job_id=f'{route}_{backbone}_seed{seed}',route=route,backbone=backbone,seed=seed,runs=runs)

def check_selection(job,contexts,limitations):
    require(job==job_spec(job['route'],job['backbone'],job['seed']) and set(contexts)==set(job['runs']),'frozen selected job/contexts')
    disclosed={}
    for label,c in contexts.items():
        # Native cache contexts bind the seed via the registered formal run ID.
        # They do not contain a redundant top-level seed; reject conflicts if supplied.
        require(c.get('schema')=='M_SOURCE_BEST_RAW_FP32_CACHE_V1' and c.get('checkpoint_selection')=='source_best' and c.get('route')==job['route'] and c.get('backbone')==job['backbone'] and c.get('run_id')==c.get('parent_run_id')==job['runs'][label],'source-only registered source-best context')
        require('seed' not in c or (type(c['seed']) is int and c['seed']==job['seed']),'conflicting redundant source context seed')
        best=c['source_best_prefix'];epoch=best['epoch']
        require(type(epoch) is int and 1<=epoch<=c['epochs'] and c['source_best_epoch']==epoch,'source DEV selected prefix within frozen budget')
        require(all(type(best[x]) is int and best[x]>=0 for x in ('attempts','successful_updates','AMP_skips')) and best['successful_updates']+best['AMP_skips']==best['attempts'],'selected prefix attempt/success/skip ledger')
        disclosed[label]=dict(seed=job['seed'],epoch=epoch,attempts=best['attempts'],successful_updates=best['successful_updates'],AMP_skips=best['AMP_skips'])
    wanted=disclosed if job['route']=='H' else disclosed['D']
    require(limitations.get('checkpoint_selection')=='source_best' and limitations.get('update_matched_primary_claim') is False and limitations.get('source_only_selected_prefix')==wanted,'saved sensitivity selection/prefix disclosure')
    epochs={h:q['epoch'] for h,q in disclosed.items()}
    require((limitations.get('source_best_epochs')==epochs if job['route']=='H' else limitations.get('source_best_epoch')==epochs['D']),'saved selected epochs')
    return disclosed

def _ref(pins,ref):
    require(type(ref) is dict and set(ref)=={'path','sha256'},'exact external proof reference')
    return pins.add(ref['path'],ref['sha256'])

def load_request(path,pin):
    pins=Pins();pins.add(path,pin);request=json_read(path)
    require(set(request)=={'schema','root','output','complete_sha256','job','proofs','consumer_dependencies_sha256'} and request['schema']==REQUEST_SCHEMA and request['root']==str(ROOT),'explicit actual audit request schema/root')
    require(request['job']==job_spec(request['job']['route'],request['job']['backbone'],request['job']['seed']),'request frozen job')
    require(set(request['proofs'])==PROOF_KEYS,'all real producer/admission evidence references required')
    require(request['consumer_dependencies_sha256']==dependency_pins(),'external consumer/dependency source inventory')
    for p,h in request['consumer_dependencies_sha256'].items():pins.add(p,h)
    return request,pins

def _identity_ids(path):
    q=pd.read_parquet(path,columns=['video_uid','label','method_id','group_id'])
    require(not q.empty and not q.isna().any().any(),'complete admitted natural identities')
    q=q.drop_duplicates()
    require(not q.video_uid.duplicated().any(),'manifest identity inconsistent across frames')
    return sorted((str(r.video_uid),int(r.label),str(r.method_id),str(r.group_id)) for r in q.itertuples())

def bind_saved_frame_metadata(manifest,saved):
    # Collision IDs, source edges and RGB hashes are scientific bootstrap inputs.
    # Bind EVERY original manifest column, not just natural-video identity.
    require(isinstance(manifest,pd.DataFrame) and isinstance(saved,pd.DataFrame) and set(manifest)<=set(saved),'all original frame metadata columns required')
    try:pd.testing.assert_frame_equal(manifest.reset_index(drop=True),saved[list(manifest)].reset_index(drop=True),check_dtype=False,check_exact=True)
    except AssertionError as error:raise ValueError('saved frame metadata differs from admitted manifest') from error

def metadata_chain(request,reader,pins):
    refs=request['proofs'];docs={}
    for name,ref in refs.items():
        p=_ref(pins,ref)
        if name not in ('producer_stdout','producer_time_v'):docs[name]=json_read(p,4*1024**2)
    job=request['job'];ident=reader.identity;ad=docs['cache_admission'];inp=docs['input_admission'];permit=docs['production_permit'];producer=docs['producer_execution'];launch=docs['producer_launch'];exit_doc=docs['producer_exit']
    require(ident.get('schema')=='M_STATISTIC_STATIC_IDENTITY_V1' and ident.get('scope')=='SELECTED_FOUR_CORE_SOURCE_BEST_DESCRIPTIVE_SENSITIVITY' and ident.get('job')==job and ident.get('checkpoint_selection')=='source_best' and ident.get('CAL_repeats')==200 and ident.get('TEST_repeats')==2000 and ident.get('resampling_seed')==20261023 and ident.get('update_matched_primary_claim') is False,'fixed statistic identity/scope')
    package=producer_package_codes();require(ident.get('code_sha256')==package,'fixed producer whole-package SHA inventory')
    for path,digest in package.items():pins.add(path,digest)
    require(inp.get('schema')=='M_SOURCE_BEST_CACHE_INPUT_ADMISSION_V1' and inp.get('artifact_root')==str(ROOT) and inp.get('source_input_scope')=='SELECTED_M_D6_H3_FIVE_SEEDS' and inp.get('target_inference_authorized') is False and ident['input_admission_sha256']==refs['input_admission']['sha256'],'bound source-only input capsule')
    require(ad.get('schema')=='M_SOURCE_BEST_JOB_CACHE_INDEPENDENT_ADMISSION_V1' and ad.get('status')=='PASS' and ad.get('artifact_root')==str(ROOT) and ad.get('job')==job and ad.get('input_admission_sha256')==refs['input_admission']['sha256'] and ad.get('independent_role_and_source_best_head_validation_passed') is True and ad.get('fresh_original_source_context_and_all_selected_role_checks') is True and ad.get('source_contexts')==ident['source_contexts'] and ad.get('cache_completion_sha256')==ident['cache_completion_sha256'],'actual independent six-role admission/context required')
    require(ad.get('cache_code_sha256')==inp.get('cache_code_sha256')==package[str(CODE/'reliability_v2/source_best_cache.py')],'bound cache implementation')
    require(permit.get('schema') in ('M_PRODUCTION_STATISTIC_ADMISSION_V1','M_BOUNDED_STATISTIC_CANARY_ADMISSION_V1') and permit.get('status')=='PASS' and permit.get('artifact_root')==str(ROOT) and permit.get('input_admission_sha256')==refs['input_admission']['sha256'] and permit.get('four_core_comparisons_only') is True and permit.get('all_five_seeds_required') is True and job['job_id'] in permit['job_ids'] and permit.get('code_sha256')==package,'actual production permit for exact job/core/code required')
    require(permit['limits'][job['route']]==reader.limits,'producer decoding/resource limit identity')
    all_jobs={job_spec(r,b,s)['job_id'] for r in ('D','H') for b in BACKBONES for s in SEEDS}
    allowed_jobs={'D_clip_vit_b16_seed7','H_clip_vit_b16_seed7'} if permit['schema']=='M_BOUNDED_STATISTIC_CANARY_ADMISSION_V1' else all_jobs
    require(type(permit['job_ids']) is list and len(permit['job_ids'])==len(allowed_jobs) and set(permit['job_ids'])==allowed_jobs,'exact bounded canary2 or full30 producer permission')

    for key in ('qualification','capacity','cache_permit'):
        ref=permit[key];p=_ref(pins,ref);value=json_read(p,4*1024**2)
        if key=='qualification':
            require(value.get('schema')=='M_STATISTIC_RESOURCE_AND_FORMAT_ACCEPTANCE_V1' and value.get('status')=='PASS' and value.get('code_sha256')==package and value.get('input_admission_sha256')==ident['input_admission_sha256'] and all(value.get(x) is True for x in ('actual_D_transport_accepted','actual_H_terminal_bound_transport_accepted','independent_entry_review_passed')),'qualified bounded producer entry')
            require((value.get('actual_entry_resource_qualified') is False and value.get('bounded_canary_resource_plan_independently_accepted') is True) if permit['schema']=='M_BOUNDED_STATISTIC_CANARY_ADMISSION_V1' else value.get('actual_entry_resource_qualified') is True,'canary versus real-resource qualification distinction')
        if key=='cache_permit':
            require(value.get('schema')=='M_SOURCE_BEST_PRODUCTION_CACHE_PERMIT_V1' and value.get('status')=='PASS_PRODUCTION_CACHE_ADMISSION' and value.get('artifact_root')==str(ROOT) and value.get('input_admission_sha256')==ident['input_admission_sha256'] and value.get('cache_code_sha256')==ad['cache_code_sha256'] and value.get('target_inference_authorized') is True and value.get('four_core_comparisons_only') is True and set(job['runs'].values())<=set(value.get('parent_run_ids',[])),'real selected endpoint CAL/TEST cache permission')
            for name in ('cache_qualification','actual_D_storage','actual_H_storage','joint_capacity'):
                gate=json_read(_ref(pins,value[name]),4*1024**2)
                require(gate.get('M_production_cache_admission') is True and gate.get('cache_code_sha256')==ad['cache_code_sha256'] and gate.get('input_admission_sha256')==ident['input_admission_sha256'],'accepted actual cache gate '+name)
    expected_output=ROOT/'analysis/source_best_M'/hashlib.sha256(typed(ident)).hexdigest()/job['job_id']
    require(reader.output==expected_output and request['output']==str(expected_output),'canonical identity-bound completed output')
    require(producer.get('schema')=='M_STATISTIC_EXECUTION_RETURN_V1' and producer.get('status')=='COMPLETE_DECLARED_TRANSPORT_PENDING_INDEPENDENT_SAVED_RESULT_AUDIT' and producer.get('job')==job and producer.get('output')==str(reader.output) and producer.get('complete_sha256')==reader.complete_sha256 and producer.get('identity_sha256')==reader.output.parent.name and producer.get('independent_cache_admission_sha256')==refs['cache_admission']['sha256'] and producer.get('production_permit_sha256')==refs['production_permit']['sha256'] and producer.get('scientific_final_acceptance') is False and producer.get('source_best_only') is True and producer.get('GPU_or_model_training_started') is False,'real producer execution return binds terminal/permit/admission')
    require(launch.get('schema')=='M_STATISTIC_PRODUCER_LAUNCH_V1' and launch.get('job')==job and launch.get('output')==str(reader.output) and launch.get('CPU_slot')==producer.get('CPU_slot') and type(launch.get('CPU_slot')) is int and launch['CPU_slot'] in (0,1),'actual producer launch identity/slot')
    env=launch.get('environment',{});require(env.get('CUDA_VISIBLE_DEVICES')=='' and all(env.get(x)=='2' for x in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')) and launch.get('address_space_limit_bytes')==32*1024**3,'actual CPU-only qualified2threads/AS32GiB launch')
    require(launch.get('cache_admission')==refs['cache_admission'] and launch.get('production_permit')==refs['production_permit'] and launch.get('input_admission')==refs['input_admission'],'actual launch proof references')
    require(launch.get('stdout')==refs['producer_stdout']['path'] and launch.get('time_v')==refs['producer_time_v']['path'],'actual stdout/GNUtime targets')
    command=['/usr/bin/time','-v','-o',launch['time_v'],str(ROOT/'environment/bin/python'),'-B','-u','-m','reliability_v2.source_best_analysis','--execute','--root',str(ROOT),'--input',refs['input_admission']['path'],'--input-sha',refs['input_admission']['sha256'],'--admission',refs['cache_admission']['path'],'--admission-sha',refs['cache_admission']['sha256'],'--permit',refs['production_permit']['path'],'--permit-sha',refs['production_permit']['sha256'],'--slot',str(launch['CPU_slot'])]
    require(launch.get('command')==command and launch.get('cwd')==str(CODE),'exact actual qualified producer CLI/cwd')
    require(exit_doc.get('schema')=='M_STATISTIC_PRODUCER_EXIT_V1' and exit_doc.get('job')==job and exit_doc.get('launch_sha256')==refs['producer_launch']['sha256'] and type(exit_doc.get('exit_code')) is int and exit_doc['exit_code']==0 and exit_doc.get('returncode_observed_by_wait') is True and exit_doc.get('wrapper_and_child_true_exit_observed') is True,'real wrapper wait/child exit0 evidence required')
    require(all(Path(refs[key]['path']).stat().st_size<=32*1024**2 for key in ('producer_stdout','producer_time_v')),'bounded real producer logs')
    timing=Path(refs['producer_time_v']['path']).read_text()
    require(len(re.findall(r'^\s*Exit status: 0\s*$',timing,re.M))==1 and len(re.findall(r'^\s*Exit status:',timing,re.M))==1,'real GNUtime exit0')
    rss=re.findall(r'Maximum resident set size \(kbytes\):\s*(\d+)',timing)
    require(len(rss)==1 and 0<int(rss[0])<=32*1024**2,'observed producer RSS bounded32GiB')
    lines=[x for x in Path(refs['producer_stdout']['path']).read_text().splitlines() if x.strip()]
    require(lines and json.loads(lines[-1])==producer,'actual producer final stdout equals external execution receipt')
    roles=('train','source_cal','source_test','target_full','target_cal','target_test') if job['route']=='D' else ('B/SOURCE_CAL','B/SOURCE_TEST','B/TARGET_TEST')
    expected={h+'/'+role for h in job['runs'] for role in roles}
    require(set(ident['cache_completion_sha256'])==expected and len(expected)==6 and set(ad['original_head_sha256_by_role'])==expected,'exact six admitted roles/head inventory')
    # Bind prior independent source proofs by their ORIGINAL fixed hashes.
    require(ad.get('pinned_source_proof_sha256'),'actual source/DEV/selection proof pins required')
    for path,digest in ad['pinned_source_proof_sha256'].items():pins.add(path,digest)
    manifest_ids={};frames={}
    for h,run in job['runs'].items():
        context=ident['source_contexts'][h]
        for role in roles:
            folder=ROOT/'features/source_best_M'/ident['input_admission_sha256']/run/role
            marker_path=folder/'complete.json';pins.add(marker_path,ident['cache_completion_sha256'][h+'/'+role]);m=json_read(marker_path);v=m['identity']
            require(m.get('status')=='CACHE_COMPLETE' and m.get('precision')=='FP32' and m.get('autocast') is False and v.get('checkpoint_selection')=='source_best' and v.get('parent_run_id')==run and v.get('role')==role and v.get('original_head_sha256')==ad['original_head_sha256_by_role'][h+'/'+role] and all(v.get(x)==value for x,value in context.items()),'admitted selected role marker/context/head')
            manifest=ROOT/f'data/{job["route"]}'/(role+'.parquet');require(v['source_manifest_path']==str(manifest),'native role manifest path')
            pins.add(manifest,v['source_manifest_sha256']);pins.add(folder/'manifest.parquet',v['source_manifest_sha256'])
            require(m.get('cached_manifest_sha256')==v['source_manifest_sha256'],'cached manifest SHA')
            head_metadata=m.get('head')
            require(type(head_metadata) is dict and set(head_metadata)=={'weight_file','bias_file','weight_sha256','bias_sha256'} and head_metadata['weight_file']=='head_weight.npy' and head_metadata['bias_file']=='head_bias.npy','frozen complete head file/SHA contract')
            head=[]
            for key,shape in [('weight',(2,m['dimension'])),('bias',(2,))]:
                hp=folder/head_metadata[key+'_file'];pins.add(hp,head_metadata[key+'_sha256'])
                with hp.open('rb') as stream:
                    require(np.lib.format.read_magic(stream)==(1,0),'admitted head NPY1')
                    got,fortran,dtype=np.lib.format.read_array_header_1_0(stream,max_header_size=4096)
                    require(tuple(got)==shape and not fortran and dtype==np.dtype('float32') and stream.tell()+math.prod(shape)*4==hp.stat().st_size,'admitted original head shape/type/length')
                head.append(np.load(hp,allow_pickle=False))
            require(hashlib.sha256(head[0].tobytes(order='C')+head[1].tobytes(order='C')).hexdigest()==v['original_head_sha256'],'original source-best head bytes against independent admission')
            require(m['dimension']=={'clip_vit_b16':512,'dinov2_vit_b14':768,'efficientnet_b0':1280}[job['backbone']] and m['rows']==pq.ParquetFile(manifest).metadata.num_rows,'frozen backbone dimension/manifest row count')
            ids=_identity_ids(manifest)
            if job['route']=='H':
                if role in manifest_ids:require(manifest_ids[role]==ids,'two trained histories same native-B IDs')
                manifest_ids[role]=ids
            else:frames[role]=manifest
            chunks=m.get('chunks');require(type(chunks) is list and len(chunks)==math.ceil(m['rows']/4096) and m.get('chunk_rows')==4096,'complete contiguous source chunk manifest')
            position=0
            for i,item in enumerate(chunks):
                count=min(4096,m['rows']-position)
                require(item['file']==f'chunk_{i:06d}.npy' and item['start']==position and item['rows']==count and item['dimension']==m['dimension'] and item['dtype']=='float32','source chunk shape/position')
                pins.add(folder/item['file'],item['sha256']);cp=folder/f'chunk_{i:06d}.json';pins.add(cp,sha(cp));require(json_read(cp)==item,'source chunk metadata agrees with pinned complete')
                position+=count
    selection=check_selection(job,ident['source_contexts'],reader['limitations.json'])
    if job['route']=='D':
        freeze=ROOT/'data/D/freeze.json';require(str(freeze) in pins.expected,'D frozen methods/scope externally pinned')
        f=json_read(freeze);require(f.get('status')=='FROZEN' and f.get('method_count')==21 and f.get('target_scores_used') is False,'frozen source-only21method protocol')
        fixed=sorted(f['method_registry']['FS9']+f['method_registry']['FR12'])
        require(len(fixed)==len(set(fixed))==21,'fixed21 method registry')
        for role in ('source_test','target_full','source_cal','target_cal','target_test'):
            saved=reader[(role+'/frame_margins.parquet') if role in ('source_test','target_full') else f'rq4/{role}_frame_margins.parquet']
            q=pd.read_parquet(frames[role])
            bind_saved_frame_metadata(q,saved)
        require(reader['rq4/scope.json']==f['cdf_calibration_scope'],'RQ4 exact frozen CAL/TEST permission scope')
    else:fixed=None
    pins.check();reader.check()
    return dict(job=job,fixed_methods=fixed,source_contexts=ident['source_contexts'],manifest_ids=manifest_ids,selection=selection,producer_CPU_slot=launch['CPU_slot'],producer_peak_RSS_KiB=int(rss[0]))

def _actual_slot(root,slot):
    # The original runtime lock is the only actual CPU admission mechanism.
    from reliability_v2 import source_best_runtime
    return source_best_runtime.cpu_slot(root,slot)

def audit_numeric(reader,context):
    if reader.route=='H':return audit_H_artifacts(reader,manifest_ids=context.get('manifest_ids'))
    path=D_KERNEL
    require(sha(path)==D_SHA,'fixed independent D source SHA before import')
    spec=importlib.util.spec_from_file_location('m_independent_D_numeric',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    require(sha(path)==D_SHA,'D independent source drift during import')
    result=module.audit_D_artifacts(reader,repeats=2000,cal_repeats=200,resampling_seed=20261023,context=context)
    require(sha(path)==D_SHA,'D independent source drift during compute')
    return result

def execute(request_path,request_sha256,slot,*,emit=None):
    request,pins=load_request(request_path,request_sha256)
    require(type(slot) is int and slot in (0,1),'actual original shared CPU slot0/1 required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(x) in ('1','2') for x in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'bounded CPU-only audit environment')
    soft,hard=resource.getrlimit(resource.RLIMIT_AS);require(0<soft<=32*1024**3,'actual audit AS32GiB bound required')
    with _actual_slot(ROOT,slot):
        # All decoding, actual source/chunk hashing and numerical work owns slot.
        reader=ReadOnlyArtifacts(request['output'],request['complete_sha256'],route=request['job']['route'])
        context=metadata_chain(request,reader,pins);initial=dict(pins.expected)
        result=audit_numeric(reader,context);reader.verify_all_decodable();pins.check()
        require(pins.expected==initial,'fixed audit pin inventory drift')
        accepted=result.get('accepted_for_selected_core') is True and not result.get('blockers_for_selected_core',result.get('blockers',[]))
        report=dict(schema=SCHEMA,status='SELECTED_CORE_METADATA_AND_NUMERIC_ACCEPTED' if accepted else 'PENDING_SELECTED_CORE_NUMERIC_COVERAGE',audited_scope='SELECTED_FOUR_CORE_ONLY',accepted_for_selected_core=accepted,scientific_final_acceptance=False,all_output_metrics_numerically_recomputed=result.get('all_output_metrics_numerically_recomputed',False),all_five_seed_conclusions_available=False,job=request['job'],core_endpoints=CORE_ENDPOINTS,layers=LAYERS,selected_prefix=context['selection'],external_complete_sha256=reader.complete_sha256,artifact_count=len(reader),all_declared_artifacts_SHA_and_codec_verified=True,metadata_status='ACTUAL_BOUND_SIX_ROLE_PRODUCER_EXIT0_AND_RESOURCE_EVIDENCE_RECHECKED',numeric=result,initial_fixed_sha256=initial,result_initial_fixed_sha256=reader.pins.expected,consumer_dependencies_sha256=request['consumer_dependencies_sha256'],lifetime_shared_CPU_slot=slot,producer_peak_RSS_KiB=context['producer_peak_RSS_KiB'],actual_fresh_source_image_or_feature_dot_recompute=False,model_or_GPU_inference_started=False,output_or_staging_created=False,publication='stdout only; caller must preserve this exact report and its fixed pins')
        # Last check directly precedes returning the reviewable report to caller.
        pins.check();reader.check();report['fixed_SHA_rechecked_before_return']=True
        report['audit_peak_RSS_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['audit_address_space_limit_bytes']=soft
        report['stdout_publication_inside_shared_slot']=emit is not None
        if emit is not None:
            pins.check();reader.check();report['fixed_SHA_rechecked_before_publication']=True
            emit(report)
        return report

def main(argv=None,*,emit=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--request');parser.add_argument('--request-sha');parser.add_argument('--slot',type=int,choices=(0,1));args=parser.parse_args(argv)
    if not args.execute:return dict(schema=SCHEMA,status='PREPARE_ONLY',production_audit_started=False,accepted_for_selected_core=False,scientific_final_acceptance=False,inventory_counts={'D':60,'H':37},audited_scope='SELECTED_FOUR_CORE_ONLY',core_endpoints=CORE_ENDPOINTS,layers=LAYERS,required_external_request_schema=REQUEST_SCHEMA,consumer_dependencies_sha256=dependency_pins())
    if not all((args.request,args.request_sha,args.slot is not None)):parser.error('execute requires external request SHA and actual shared CPU slot')
    return execute(args.request,args.request_sha,args.slot,emit=emit)

if __name__=='__main__':
    printed=[]
    def publish_stdout(result):
        print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True);printed.append(True)
    result=main(emit=publish_stdout)
    if not printed:publish_stdout(result)

