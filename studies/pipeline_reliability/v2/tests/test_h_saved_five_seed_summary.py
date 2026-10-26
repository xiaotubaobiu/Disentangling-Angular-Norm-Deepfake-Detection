from pathlib import Path
import ast,contextlib,copy,csv,hashlib,importlib.util,io,json,math,statistics,datetime
from fractions import Fraction
import pytest
P=Path(__file__).parents[1]/'tools/h_saved_five_seed_summary.py'
def module():
 assert P.is_file(),'saved-result five-seed generator missing'
 sp=importlib.util.spec_from_file_location('saved_five_seed_test',P);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m

def rows(values):return [dict(seed=s,backbone='clip_vit_b16',domain='TARGET_TEST',cohort='all_B',aggregation='mean_logit',comparison='H10-H00',threshold_kind='source_real',metric='pooled_ba',estimate=v) for s,v in zip((7,42,123,2026,3407),values)]
def test_sample_sd_closed_form_and_negative_results():
 m=module();q=m.reduce_seed_comparisons(rows([-2,-1,0,1,2]))[0];mu=sum(map(Fraction,[-2,-1,0,1,2]))/5;variance=sum((Fraction(x)-mu)**2 for x in [-2,-1,0,1,2])/4;assert q['estimate_mean']==float(mu) and q['estimate_sample_sd']==pytest.approx(math.sqrt(float(variance)));assert (q['n_positive'],q['n_negative'],q['n_zero_within_tolerance'])==(2,2,1);assert q['expected_seeds']==(7,42,123,2026,3407)
 n=m.reduce_seed_comparisons(rows([-.1,-.2,-.3,-.4,-.5]))[0];assert n['estimate_mean']==pytest.approx(-.3) and n['n_negative']==5

@pytest.mark.parametrize('missing',[None,float('nan'),float('inf')])
def test_any_NA_keeps_all_five_but_mean_sd_unavailable(missing):
 m=module();q=m.reduce_seed_comparisons(rows([1,2,missing,-1,0]))[0];assert q['n_seeds']==5 and q['n_valid_seeds']==4 and q['estimate_mean'] is None and q['estimate_sample_sd'] is None and q['status']=='unavailable_seed_estimate';assert (q['n_positive'],q['n_negative'],q['n_zero_within_tolerance'])==(2,1,1)

@pytest.mark.parametrize('bad',['missing','duplicate','foreign','collapse'])
def test_seed_or_stratum_incompleteness_rejected(bad):
 m=module();q=rows([1,2,3,4,5]);group=None
 if bad=='missing':q.pop()
 elif bad=='duplicate':q[-1]['seed']=7
 elif bad=='foreign':q[-1]['seed']=999
 else:group=['comparison','threshold_kind','metric']
 with pytest.raises((ValueError,AssertionError)):m.reduce_seed_comparisons(q,group_columns=group)

def test_original_H_reducer_interface_reference_only():
 m=module();import pandas as pd,numpy as np
 source=Path(m.DEFINITIONS['stats_h']['path']).read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='summarize_seed_comparisons');ns=dict(pd=pd,np=np,FORMAL_SEEDS=m.SEEDS);exec(compile(ast.Module(body=[node],type_ignores=[]),'<reference-only>','exec'),ns)
 for values in ([1,2,3,4,5],[.1,None,-.2,0,.3]):
  expected=ns['summarize_seed_comparisons'](pd.DataFrame(rows(values))).to_dict('records')[0];actual=m.reduce_seed_comparisons(rows(values))[0];assert set(actual)==set(expected)
  for k,v in expected.items():
   if isinstance(v,float) and not math.isfinite(v):assert actual[k] is None
   elif isinstance(v,float):assert actual[k]==pytest.approx(v,abs=1e-13)
   else:assert actual[k]==v

def fixture(tmp_path,monkeypatch,bone='clip_vit_b16',NA=False):
 m=module();m.ROOT=tmp_path/'project';m.HERE=m.ROOT/'engineering/source_best_M/audits';m.HERE.mkdir(parents=True);root=m.ROOT;H=m.digest
 def dump(p,x):
  p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');return dict(path=str(p),sha256=H(p))
 def csvfile(p,items):
  p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
 def inventory():
  out={'coverage.json','identity.json','independent_saved_score_recompute.json','limitations.json'}
  for agg in ('mean_logit','mean_probability'):
   out.update(f'{agg}/source_cal_{h}.csv' for h in ('B','AB'))
   for dom in ('SOURCE_TEST','TARGET_TEST'):
    for co,cells in [('all_B',('H00','H10')),('common',('H00','H01','H10','H11'))]:
     pre=f'{agg}/{dom}/{co}/';out.update(pre+n for n in ('metrics.csv','comparisons.csv','conditional_bootstrap.json','independent_kernel_checks.json','thresholds.json'));out.update(pre+f'{c}_scores.csv' for c in cells)
  assert len(out)==72;return out
 assert m.pair_inventory()==inventory()
 inputs={}
 reg={'status':'H_FORMAL_COMPLETE_TARGET_CACHE_PENDING','target_export_allowed':True,'effective_runs':{b:dict(precision='AMP',run_ids=[f'H_formal_{b}_{h}_seed{s}' for s in m.SEEDS for h in ('B','AB')]) for b in m.BACKBONES}}
 for rel,obj in [('configs/H_execution.json',{'frozen':True}),('H_run_registry.json',reg),(f'qualification/H_target_export_{bone}.json',{'accepted':True}),(f'reports/H_{bone}_AMP_paired_ledger.json',{'accepted':True}),('data/H/freeze.json',{'frozen':True})]:ref=dump(root/rel,obj);inputs[ref['path']]=ref['sha256']
 manifest=json.loads(Path(m.DEFINITIONS['qualified_manifest']['path']).read_text());code={n:manifest['source_sha256']['reliability_v2/'+n] for n in m.ANALYSIS_CODE_NAMES};pairs=[];metric_paths=[];audit_paths=[]
 for index,seed in enumerate(m.SEEDS):
  folder=root/f'analysis/H/{bone}/seed{seed}';I=dict(schema='H_PAIR_ANALYSIS_IDENTITY_V1',backbone=bone,seed=seed,effective_precision='AMP',run_ids={h:f'H_formal_{bone}_{h}_seed{seed}' for h in ('B','AB')},inputs_sha256=inputs,cache_completion_sha256={h+'/'+r:'a'*64 for h in ('B','AB') for r in m.ROLES},code_sha256=code,bootstrap_repeats=2000,resampling_seed=20261023,source_threshold='each_model_allB_SOURCE_CAL_empiricalFPR5',primary='DFD_allB_Raw_mean_logit_H10_minus_H00',test_cohorts='allB_and_common_reported_separately')
  for n in inventory():
   p=folder/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}\n')
  iref=dump(folder/'identity.json',I);cohorts=[]
  for agg in ('mean_logit','mean_probability'):
   for dom in ('SOURCE_TEST','TARGET_TEST'):
    for co,cells in [('all_B',('H00','H10')),('common',('H00','H01','H10','H11'))]:
     metrics=[];comparisons=[];point={};names=['H10-H00'] if co=='all_B' else ['H10-H00','H01-H00','H11-H01','H11-H10','(H11-H10)-(H01-H00)']
     for cell in cells:
      for kind in ('default','source_real'):
       row=dict(cell=cell,model_history='AB' if cell in ('H10','H11') else 'B',threshold_kind=kind,n_real=2,n_fake=3,threshold=.5,extra_column='retained')
       for metric in m.METRICS:
        value=.4+.01*index
        if NA and seed==7 and cell=='H00' and metric=='macro_auc' and agg=='mean_logit' and dom=='TARGET_TEST' and co=='all_B' and kind=='default':value=None
        row[metric]='' if value is None else value;point[f'{kind}:cell:{cell}:{metric}']=value
       metrics.append(row)
     for name in names:
      for kind in ('default','source_real'):
       for metric in m.METRICS:
        value=(index-2)/10;comparisons.append(dict(comparison=name,threshold_kind=kind,metric=metric,estimate=value));point[f'{kind}:comparison:{name}:{metric}']=value
     base=folder/agg/dom/co;csvfile(base/'metrics.csv',metrics);csvfile(base/'comparisons.csv',comparisons);metric_paths.append(base/'metrics.csv');cohorts.append(dict(aggregation=agg,domain=dom,cohort=co,draws_checked=2000,point_values_checked=len(point),intervals_checked=len(point),status='ok',numeric_NA_cells=[],independent_point_estimates=point))
  hashes={n:H(folder/n) for n in inventory()};cref=dump(folder/'complete.json',dict(schema='H_PAIR_ANALYSIS_COMPLETE_V1',status='ANALYSIS_COMPLETE_WITH_NUMERIC_NA' if NA and seed==7 else 'ANALYSIS_COMPLETE',backbone=bone,seed=seed,effective_precision='AMP',identity=I,independent_recompute='PASSED_FOR_ALL_REPORTABLE_CELLS',artifact_sha256=hashes))
  consumer_sha=m.KERNEL_SHA if bone=='clip_vit_b16' and seed==7 else m.LEGACY42_SHA if bone=='clip_vit_b16' and seed==42 else m.RESUMED_SHA if bone=='clip_vit_b16' and seed==3407 else m.MATRIX_SHA
  source=m.CONSUMERS[consumer_sha];A=dict(schema='INDEPENDENT_ORIGINAL_H_LAST_PAIR_RESULT_AUDIT_V1',status='SAVED_RESULT_NUMERICS_AND_BOUND_PRIOR_ROLE_ADMISSION_PASS',backbone=bone,seed=seed,source_checkpoint='original_LAST_final.pt',terminal_sha256=cref['sha256'],identity_sha256=iref['sha256'],audit_code_sha256=consumer_sha,artifact_count=72,current_pins_sha256={**inputs,**{str(m.ORIGINAL/'reliability_v2'/n):h for n,h in code.items()}},cohorts=cohorts,cpu_slot=0,lifetime_shared_cpu_slot_held=True,fresh_pixel_or_source_context_validation=False,fresh_feature_dot_product_recompute=False,weight_or_model_loading=False,M_source_best_used=False,M_production_authorized=False,GPU_or_target_inference_started=False,snapshot_promoted=False)
  if consumer_sha==m.KERNEL_SHA:A['evidence_route']='ORIGINAL_SERIAL14_ACCEPTED'
  if consumer_sha in (m.MATRIX_SHA,m.RESUMED_SHA):A.update(independent_numeric_kernel_sha256=m.KERNEL_SHA,evidence_route='ACCEPTED_PARALLEL14_METADATA_REBOUND_ORIGINAL42FRESH_LEDGER_PRODUCER',all_five_seed_conclusions_available=False)
  if consumer_sha==m.RESUMED_SHA:
   proof=dict(consumer=dict(path=source['path'],initial_sha256=consumer_sha,observed_sha256=consumer_sha),original_bb=dict(path=m.CONSUMERS[m.MATRIX_SHA]['path'],initial_sha256=m.MATRIX_SHA,observed_sha256=m.MATRIX_SHA));A.update(consumer_source_path=source['path'],consumer_source_sha256=consumer_sha,original_bb299e_source_path=m.CONSUMERS[m.MATRIX_SHA]['path'],original_bb299e_source_sha256=m.MATRIX_SHA,source_attestation_initial=proof,source_attestation_after_audit=proof,source_attestation_before_publication=proof,launch_fact_route='explicit_bound_user_resume',M_statistics_started_field_present=False,missing_field_not_synthesized=True)
  ap=m.HERE/f'H_LAST_{"clip" if bone=="clip_vit_b16" else bone}_seed{seed}_independent_audit.json';aref=dump(ap,A);audit_paths.append(ap);pairs.append(dict(seed=seed,identity=iref,complete=cref,independent_numeric_audit=aref,audit_source=source))
 request=dict(schema='H_SAVED_FIVE_SEED_SUMMARY_REQUEST_V1',root=str(root),backbone=bone,source_checkpoint='original_LAST_final.pt',seeds=list(m.SEEDS),summary_source=dict(path=str(P),sha256=m.INITIAL_SHA),definitions=m.DEFINITIONS,pairs=pairs);rp=m.HERE/f'H_SAVED_FIVE_SEED_{bone}_request.json';rr=dump(rp,request)
 evidence=dump(root/'reports/full_budget_evidence.json',{'schema':'M_FIXED_REMAINING_WRITE_PLAN_V1','accepted_full_remaining':True});budget=dict(schema='H_SAVED_SUMMARY_OPERATIONAL_BUDGET_V1',root=str(root),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='VERIFIED_FULL_REMAINING_PRIMARY_WITH_PEAKS_RESERVE',full_remaining_with_peaks_reserve_bytes=150215493801,reserve_already_included_bytes=50*1024**3,primary_temporary_already_included_bytes=16*1024**3,summary_output_cap_covered_bytes=16*1024**2,summary_cap_already_included=True,unverified_compression_credit_bytes=0,partial_feature_credit_bytes=0,evidence=[evidence]);budget.update(capacity_fixture(m,root,dump));br=dump(root/'reports/summary_budget.json',budget);entries=[]
 @contextlib.contextmanager
 def no_lock(*args):entries.append(True);yield
 monkeypatch.setattr(m,'cpu_slot',no_lock);monkeypatch.setattr(m,'live_free_bytes',lambda:200*1024**3)
 return m,rr,br,request,entries,metric_paths,audit_paths,dump

def execute(m,rr,br):return m.main(['--execute','--request',rr['path'],'--request-sha',rr['sha256'],'--budget',br['path'],'--budget-sha',br['sha256'],'--slot','1'])
@pytest.mark.parametrize('bone',['clip_vit_b16','dinov2_vit_b14','efficientnet_b0'])
def test_all_five_saved_summary_atomic_idempotent_no_context(tmp_path,monkeypatch,bone):
 m,rr,br,request,entries,paths,audits,dump=fixture(tmp_path,monkeypatch,bone);result=execute(m,rr,br);out=Path(result['output']);assert result['reused'] is False and len(entries)==1;assert {p.name for p in out.iterdir()}==set(m.OUTPUTS)|{'complete.json'};raw=list(csv.DictReader((out/'per_seed_metrics.csv').open()));assert len({int(r['seed']) for r in raw})==5 and all(r['extra_column']=='retained' for r in raw);assert 'n_real' in raw[0]
 again=execute(m,rr,br);assert again['reused'] is True and len(entries)==2 and again['sha256']==result['sha256'];assert not (m.ROOT/f'analysis/H/{bone}/five_seed_summary').exists()

def test_NA_summary_retains_negative_comparisons_and_counts(tmp_path,monkeypatch):
 m,rr,br,request,entries,paths,audits,dump=fixture(tmp_path,monkeypatch,NA=True);out=Path(execute(m,rr,br)['output']);summary=list(csv.DictReader((out/'five_seed_metrics.csv').open()));q=[r for r in summary if r['aggregation']=='mean_logit' and r['domain']=='TARGET_TEST' and r['cohort']=='all_B' and r['cell']=='H00' and r['threshold_kind']=='default' and r['metric']=='macro_auc'][0];assert q['estimate_mean']=='' and q['estimate_sample_sd']=='' and q['n_valid_seeds']=='4';per=list(csv.DictReader((out/'per_seed_comparisons.csv').open()));assert any(float(r['estimate'])<0 for r in per)

@pytest.mark.parametrize('bad',['missing_seed','duplicate_seed','wrong_seed','source_best','wrong_identity','wrong_source_HEAD','audit_missing_false','bad_audit_source','bad_complete_inventory','path_escape','bad_requestSHA','tamper_input','budget_under','budget_credit'])
def test_bad_evidence_is_rejected_without_output(tmp_path,monkeypatch,bad):
 m,rr,br,request,entries,paths,audits,dump=fixture(tmp_path,monkeypatch);out=m.ROOT/'analysis/H/clip_vit_b16/five_seed_saved_result_summary'
 if bad=='missing_seed':request['pairs'].pop()
 elif bad=='duplicate_seed':request['pairs'][-1]['seed']=7
 elif bad=='wrong_seed':request['seeds'][-1]=999
 elif bad=='source_best':request['source_checkpoint']='source_best.pt'
 elif bad=='wrong_identity':
  p=Path(request['pairs'][0]['identity']['path']);d=json.loads(p.read_text());d['run_ids']['B']='M_source_best';request['pairs'][0]['identity']=dump(p,d)
 elif bad=='wrong_source_HEAD':request['definitions']=copy.deepcopy(request['definitions']);request['definitions']['qualified_manifest']['sha256']='0'*64
 elif bad=='audit_missing_false':
  p=audits[0];d=json.loads(p.read_text());d.pop('M_source_best_used');request['pairs'][0]['independent_numeric_audit']=dump(p,d)
 elif bad=='bad_audit_source':request['pairs'][1]['audit_source']=request['pairs'][0]['audit_source']
 elif bad=='bad_complete_inventory':
  p=Path(request['pairs'][0]['complete']['path']);d=json.loads(p.read_text());d['artifact_sha256'].pop('coverage.json');request['pairs'][0]['complete']=dump(p,d)
 elif bad=='path_escape':request['pairs'][0]['identity']['path']='/tmp/foreign_identity.json'
 elif bad=='tamper_input':paths[0].write_text(paths[0].read_text()+'\ncorrupted')
 elif bad in ('budget_under','budget_credit'):
  d=json.loads(Path(br['path']).read_text());d['full_remaining_with_peaks_reserve_bytes']=500*1024**3 if bad=='budget_under' else d['full_remaining_with_peaks_reserve_bytes'];d['partial_feature_credit_bytes']=1 if bad=='budget_credit' else 0;br=dump(br['path'],d)
 rr=dump(rr['path'],request)
 if bad=='bad_requestSHA':rr['sha256']='0'*64
 with pytest.raises((AssertionError,ValueError,KeyError,FileNotFoundError)):execute(m,rr,br)
 assert not out.exists()

def test_existing_artifact_and_receipt_tampering_is_not_trusted(tmp_path,monkeypatch):
 m,rr,br,request,entries,paths,audits,dump=fixture(tmp_path,monkeypatch);out=Path(execute(m,rr,br)['output']);p=out/'five_seed_comparisons.csv';p.write_text(p.read_text()+'\nforged');complete=json.loads((out/'complete.json').read_text());complete['artifact_sha256'][p.name]=m.digest(p);dump(out/'complete.json',complete)
 with pytest.raises((AssertionError,ValueError)):execute(m,rr,br)


@pytest.mark.parametrize('target',['summary','D_reducer','shared_slot','matrix'])
def test_digest_only_late_source_drift_rejected_without_source_file_mutation(tmp_path,monkeypatch,target):
 m,rr,br,request,entries,paths,audits,dump=fixture(tmp_path,monkeypatch);original=m.digest;build=m.build;active=[False]
 path=str(P) if target=='summary' else m.DEFINITIONS['independent_D_reducer']['path'] if target=='D_reducer' else m.DEFINITIONS['shared_slot']['path'] if target=='shared_slot' else m.CONSUMERS[m.MATRIX_SHA]['path']
 def mock_digest(p):return '0'*64 if active[0] and str(p)==path else original(p)
 def late(*args):q=build(*args);active[0]=True;return q
 monkeypatch.setattr(m,'digest',mock_digest);monkeypatch.setattr(m,'build',late)
 with pytest.raises(AssertionError,match='SHA drift'):execute(m,rr,br)
 assert not (m.ROOT/'analysis/H/clip_vit_b16/five_seed_saved_result_summary').exists()

@pytest.mark.parametrize('bad',['cohort_duplicate','missing_comparison','point_disagree','bool_artifact_count','bool_seed','wrong_kernel','wrong_legacy_profile','resumed_attestation','conflict_pin'])
def test_additional_accepted_report_semantic_boundaries(tmp_path,monkeypatch,bad):
 m,rr,br,request,entries,paths,audits,dump=fixture(tmp_path,monkeypatch);index=4 if bad=='resumed_attestation' else 0;ap=audits[index];a=json.loads(ap.read_text())
 if bad=='cohort_duplicate':a['cohorts'][-1]=a['cohorts'][0]
 elif bad=='missing_comparison':a['cohorts'][0]['independent_point_estimates'].pop(next(k for k in a['cohorts'][0]['independent_point_estimates'] if ':comparison:' in k))
 elif bad=='point_disagree':a['cohorts'][0]['independent_point_estimates'][next(iter(a['cohorts'][0]['independent_point_estimates']))]=.999
 elif bad=='bool_artifact_count':a['artifact_count']=True
 elif bad=='bool_seed':a['seed']=True
 elif bad=='wrong_kernel':a['independent_numeric_kernel_sha256']='0'*64
 elif bad=='wrong_legacy_profile':a['evidence_route']='PARALLEL'
 elif bad=='resumed_attestation':a['source_attestation_before_publication']['consumer']['initial_sha256']='0'*64
 elif bad=='conflict_pin':a['current_pins_sha256'][m.DEFINITIONS['stats_h']['path']]='0'*64
 request['pairs'][index]['independent_numeric_audit']=dump(ap,a);rr=dump(rr['path'],request)
 with pytest.raises((AssertionError,KeyError,ValueError)):execute(m,rr,br)
 assert not (m.ROOT/'analysis/H/clip_vit_b16/five_seed_saved_result_summary').exists()


def test_budget_refresh_is_operational_not_statistical_identity(tmp_path,monkeypatch):
 m,rr,br,request,entries,paths,audits,dump=fixture(tmp_path,monkeypatch);first=execute(m,rr,br);b=json.loads(Path(br['path']).read_text());b['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();br=dump(m.ROOT/'reports/summary_budget_refreshed.json',b);second=execute(m,rr,br);assert second['reused'] is True and first['sha256']==second['sha256']


def test_historical_weight_pin_retained_without_byte_read(tmp_path,monkeypatch):
 m,rr,br,request,entries,paths,audits,dump=fixture(tmp_path,monkeypatch);p=str(m.ROOT/'runs/H_formal_clip_vit_b16_B_seed7/final.pt');a=json.loads(audits[0].read_text());a['current_pins_sha256'][p]='a'*64;request['pairs'][0]['independent_numeric_audit']=dump(audits[0],a);rr=dump(rr['path'],request);original=m.digest
 def reject_weight(path):assert str(path)!=p,'must not read historical model weight';return original(path)
 monkeypatch.setattr(m,'digest',reject_weight);out=Path(execute(m,rr,br)['output']);identity=json.loads((out/'identity.json').read_text());assert p not in identity['fresh_checked_input_sha256'] and identity['pairs'][0]['historical_nonconsumed_source_sha256']=={p:'a'*64}


def test_original_H_summary_metric_column_order_and_explicit_NA_counts(tmp_path,monkeypatch):
 m,rr,br,request,entries,paths,audits,dump=fixture(tmp_path,monkeypatch,NA=True);out=Path(execute(m,rr,br)['output']);columns=next(csv.reader((out/'five_seed_metrics.csv').open()));assert columns[:8]==['backbone','domain','cohort','aggregation','cell','threshold_kind','metric','n_seeds'];lim=json.loads((out/'limitations.json').read_text());assert len(lim['NA_strata'])==1 and lim['NA_strata'][0]['NA_count']==1 and lim['NA_strata'][0]['valid_count']==4


@pytest.mark.parametrize('bad',['stale','missing_cap','bool_total','no_evidence','not_included'])
def test_budget_requires_fresh_verified_inclusive_evidence(tmp_path,monkeypatch,bad):
 m,rr,br,request,entries,paths,audits,dump=fixture(tmp_path,monkeypatch);b=json.loads(Path(br['path']).read_text())
 if bad=='stale':b['utc']=(datetime.datetime.now(datetime.timezone.utc)-datetime.timedelta(hours=1)).isoformat()
 elif bad=='missing_cap':b['summary_output_cap_covered_bytes']=0
 elif bad=='bool_total':b['full_remaining_with_peaks_reserve_bytes']=True
 elif bad=='no_evidence':b['evidence']=[]
 elif bad=='not_included':b['summary_cap_already_included']=False
 br=dump(br['path'],b)
 with pytest.raises(AssertionError):execute(m,rr,br)
 assert not entries


@pytest.mark.parametrize('bad',['bool_slot','false_lifetime','budget_identity','extra_field'])
def test_existing_receipt_operational_claims_cannot_drift(tmp_path,monkeypatch,bad):
 m,rr,br,request,entries,paths,audits,dump=fixture(tmp_path,monkeypatch);out=Path(execute(m,rr,br)['output']);cp=out/'complete.json';c=json.loads(cp.read_text())
 if bad=='bool_slot':c['cpu_slot']=True
 elif bad=='false_lifetime':c['lifetime_shared_cpu_slot_held']=False
 elif bad=='budget_identity':c['budget_excluded_from_statistical_identity']=False
 elif bad=='extra_field':c['scientific_significance_claim']=True
 dump(cp,c)
 with pytest.raises(AssertionError):execute(m,rr,br)


def test_default_is_read_only_no_slot_or_output(tmp_path,monkeypatch):
 m,rr,br,request,entries,paths,audits,dump=fixture(tmp_path,monkeypatch);result=m.main(['--request',rr['path'],'--request-sha',rr['sha256']]);assert result['status']=='READ_ONLY_REQUEST_BOUND_NO_SUMMARY_EXECUTED' and not entries and not (m.ROOT/'analysis/H/clip_vit_b16/five_seed_saved_result_summary').exists()


def test_busy_slot_no_wait_or_output(tmp_path,monkeypatch):
 m,rr,br,request,entries,paths,audits,dump=fixture(tmp_path,monkeypatch)
 @contextlib.contextmanager
 def busy(*args):entries.append(True);raise RuntimeError('busy no wait');yield
 monkeypatch.setattr(m,'cpu_slot',busy)
 with pytest.raises(RuntimeError,match='busy no wait'):execute(m,rr,br)
 assert len(entries)==1 and not (m.ROOT/'analysis/H/clip_vit_b16/five_seed_saved_result_summary').exists()


def test_late_consumed_evidence_digest_drift_prevents_complete_marker(tmp_path,monkeypatch):
 m,rr,br,request,entries,paths,audits,dump=fixture(tmp_path,monkeypatch);original=m.digest;guard=m.publication_guard;count=[0];path=request['pairs'][0]['identity']['path']
 def fault(p):return '0'*64 if count[0]>=4 and str(p)==path else original(p)
 def boundary(*args):count[0]+=1;return guard(*args)
 monkeypatch.setattr(m,'digest',fault);monkeypatch.setattr(m,'publication_guard',boundary)
 with pytest.raises(AssertionError,match='SHA drift'):execute(m,rr,br)
 out=m.ROOT/'analysis/H/clip_vit_b16/five_seed_saved_result_summary';assert not (out/'complete.json').exists()



def capacity_fixture(m,root,dump):
 G=1024**3;parts=dict(remaining_H_primary9_full_caps=9*G,primary_logs_manifests_figures_audits=16*G,primary_temporary_peak=16*G,primary_reserve=50*G,M_temporary=6*G,M_extra=5*G,all_M45_feature_cap=18413684905,all_M30_statistic_caps=20132659200);base=root/'engineering/source_best_M';admission='8d8e25e767fe5ee396c2d6773d4c65d002268e19bbdfa4151b0764e6f6ca475a'
 runs=[f'D_formal_{b}_seed{s}' for b in m.BACKBONES for s in m.SEEDS]+[f'H_formal_{b}_{h}_seed{s}' for b in m.BACKBONES for s in m.SEEDS for h in ('B','AB')];jobs={f'{d}_{b}_seed{s}':dict(cap_bytes=(G if d=='D' else G//4),output=None) for d in ('D','H') for b in m.BACKBONES for s in m.SEEDS}
 plan=dump(base/'plans/M_two_canary_remaining_write_plan_after_H6_20261026_candidate.json',dict(schema='M_FIXED_REMAINING_WRITE_PLAN_V1',input_admission_sha256=admission,base_future_and_reserve_bytes=109521666048,runtime_floor_bytes=50*G,global_M_temporary_bytes=6*G,feature_cap_bytes=18413684905,feature_runs=runs,jobs=jobs))
 bone=next((root/'analysis/H').iterdir()).name;label='clip' if bone=='clip_vit_b16' else bone;seed=3407;item=dict(backbone=bone,seed=seed,artifact_count=72,independently_recomputed_cohorts=8,applicable_draws_recomputed=16000,native_source_checkpoints='original_LAST_final.pt',source_best_or_M_used=False,complete=dict(path=str(root/f'analysis/H/{bone}/seed{seed}/complete.json'),sha256=m.digest(root/f'analysis/H/{bone}/seed{seed}/complete.json')),independent_audit=dict(path=str(m.HERE/f'H_LAST_{label}_seed{seed}_independent_audit.json'),sha256=m.digest(m.HERE/f'H_LAST_{label}_seed{seed}_independent_audit.json')))
 for key in ('producer_execution','producer_launch','producer_time_v','audit_launch','audit_true_exit'):item[key]=dump(root/f'reports/synthetic_closure_{key}.json',dict(metadata_only=True))
 proof=dump(root/'reports/H_resumed_CLIP3407_DINO7_independent_saved_result_acceptance_20261026.json',dict(schema='H_RESUMED_ORIGINAL_LAST_TWO_PAIR_INDEPENDENT_ACCEPTANCE_V1',status='PASS_ACTUAL_TERMINAL_AND_INDEPENDENT_NUMERIC_BOOTSTRAP_AUDIT',H_total_independently_accepted_pairs=6,pairs=[item]));review=dump(base/'independent_M_capacity_after_H6_reassessment_20261026.json',dict(schema='INDEPENDENT_M_CAPACITY_AFTER_H6_REASSESSMENT_V1',status='PASS_CLOSED_PRIMARY_FUTURE_WRITE_REASSESSMENT_ONLY',bindings=dict(new_plan=plan,new_H_acceptance=proof),budget_breakdown_bytes=parts,all45_feature_runs_preserved=True,all30_job_caps_preserved=True,required_total_bytes=148068010153,remaining_primary_future_pair_count=9,H_independently_accepted_pair_count=6,partial_M_feature_credit_bytes=0,M_statistic_credit_bytes=0,physical_SSD_release_bytes=0))
 acceptance=dump(base/'M_remaining_after_H6_capacity_acceptance_20261026.json',dict(schema='M_REMAINING_FUTURE_WRITE_AFTER_H6_CAPACITY_ACCEPTANCE_V1',status='PASS_REVIEWED_FIXED_FUTURE_WRITE_PLAN_NOT_PRODUCTION_PERMISSION',artifact_root=str(root),input_admission_sha256=admission,write_plan=plan,independent_review=review,actual_H6_acceptance=proof,full_required_remaining_with_peak_and_reserve_bytes=148068010153,summary_small_outputs_included_in_existing_primary16GiB_logs_figures_audits_and_M5GiB_extra=True,primary_temporary_peak_retained_bytes=16*G,M_temporary_peak_retained_bytes=6*G,runtime_floor_bytes=50*G,full_M_feature_bound_retained_bytes=18413684905,all_M30_job_caps_retained_bytes=20132659200,partial_M_feature_credit_bytes=0,physical_SSD_released_bytes=0))
 m.CAPACITY_REFS=dict(plan=plan,acceptance=acceptance,independent_review=review)
 return dict(full_remaining_with_peaks_reserve_bytes=148068010153,M_temporary_already_included_bytes=6*G,all_M45_feature_cap_bytes=18413684905,all_M30_statistic_caps_bytes=20132659200,decomposition_bytes=parts,capacity_evidence=m.CAPACITY_REFS,evidence=list(m.CAPACITY_REFS.values()))


def test_HSUM001_committed_transaction_has_no_cleanup_failure(tmp_path,monkeypatch):
 m,rr,br,request,entries,paths,audits,dump=fixture(tmp_path,monkeypatch);unlink=Path.unlink
 def fail_stage(p,*args,**kwargs):
  if any('.stage_' in part for part in p.parts):raise OSError('injected stage cleanup failure')
  return unlink(p,*args,**kwargs)
 monkeypatch.setattr(Path,'unlink',fail_stage);one=execute(m,rr,br);out=Path(one['output']);two=execute(m,rr,br);assert two['reused'] and one['sha256']==two['sha256'];assert {p.name for p in out.iterdir()}==set(m.OUTPUTS)|{'complete.json'}


def test_HSUM002_legal_runtime_slot_change_is_receipt_tampering(tmp_path,monkeypatch):
 m,rr,br,request,entries,paths,audits,dump=fixture(tmp_path,monkeypatch);one=execute(m,rr,br);out=Path(one['output']);c=json.loads((out/'complete.json').read_text());c['cpu_slot']=0;dump(out/'complete.json',c)
 with pytest.raises(AssertionError):execute(m,rr,br)


def test_HSUM003_logical_near_cap_must_charge_blocks_directories_peak(tmp_path,monkeypatch):
 m,rr,br,request,entries,paths,audits,dump=fixture(tmp_path,monkeypatch);out=m.ROOT/'analysis/H/clip_vit_b16/five_seed_saved_result_summary';payloads={n:b'x' for n in m.OUTPUTS};payloads[m.OUTPUTS[0]]=b'x'*(m.CAP-4096-5)
 with pytest.raises(AssertionError,match='allocat|cap|peak'):m.publish(out,payloads,rr['sha256'],{},m.INITIAL_SHA,br['path'],br['sha256'],1)
 assert not out.exists()


def test_HSUM003_KEEP_SIZE_extra_allocation_without_SHA_change_rejects_reuse(tmp_path,monkeypatch):
 import ctypes,os
 m,rr,br,request,entries,paths,audits,dump=fixture(tmp_path,monkeypatch);out=Path(execute(m,rr,br)['output']);p=out/'per_seed_metrics.csv';before=m.digest(p);libc=ctypes.CDLL(None,use_errno=True)
 fd=os.open(p,os.O_RDWR)
 try:r=libc.fallocate(fd,1,ctypes.c_longlong(0),ctypes.c_longlong(m.CAP+4096))
 finally:os.close(fd)
 if r:pytest.skip('filesystem lacks FALLOC_FL_KEEP_SIZE')
 assert m.digest(p)==before and p.stat().st_blocks*512>m.CAP
 with pytest.raises(AssertionError,match='allocat|cap|peak'):execute(m,rr,br)


def test_HSUM004_conflicting_historical_path_rejected_without_reading_source(tmp_path,monkeypatch):
 m,rr,br,request,entries,paths,audits,dump=fixture(tmp_path,monkeypatch);path=str(m.ROOT/'data/H/historical.png')
 for i,h in [(0,'a'*64),(1,'b'*64)]:
  a=json.loads(audits[i].read_text());a['current_pins_sha256'][path]=h;request['pairs'][i]['independent_numeric_audit']=dump(audits[i],a)
 rr=dump(rr['path'],request)
 with pytest.raises(AssertionError,match='historical|conflict'):execute(m,rr,br)
 assert not (m.ROOT/'analysis/H/clip_vit_b16/five_seed_saved_result_summary').exists()


@pytest.mark.parametrize('bad',['minimum_only','no_M_temp','arbitrary_refs','omit_future45','drop_M30','wrong_plan_SHA','missing_acceptance','FAILED_review'])
def test_HSUM005_full_accepted_capacity_chain_required(tmp_path,monkeypatch,bad):
 m,rr,br,request,entries,paths,audits,dump=fixture(tmp_path,monkeypatch);b=json.loads(Path(br['path']).read_text())
 if bad=='minimum_only':b['full_remaining_with_peaks_reserve_bytes']=66*1024**3+m.CAP
 elif bad=='no_M_temp':b.pop('M_temporary_already_included_bytes')
 elif bad=='arbitrary_refs':b['capacity_evidence']=dict(plan=b['evidence'][0]);b['evidence']=[dump(m.ROOT/'reports/arbitrary.json',dict(verified=True))]
 elif bad=='omit_future45':b['decomposition_bytes']['all_M45_feature_cap']=0;b['all_M45_feature_cap_bytes']=0
 elif bad=='drop_M30':b['decomposition_bytes']['all_M30_statistic_caps']=0;b['all_M30_statistic_caps_bytes']=0
 elif bad=='wrong_plan_SHA':b['capacity_evidence']['plan']['sha256']='0'*64
 elif bad=='missing_acceptance':b['capacity_evidence'].pop('acceptance')
 elif bad=='FAILED_review':
  path=b['capacity_evidence']['independent_review']['path'];d=json.loads(Path(path).read_text());d['status']='FAILED';new=dump(path,d);b['capacity_evidence']['independent_review']=new;b['evidence']=[new if x['path']==path else x for x in b['evidence']]
 br=dump(br['path'],b)
 with pytest.raises((AssertionError,KeyError)):execute(m,rr,br)
 assert not entries



def test_HSUM005_external_reviewed_packet_refresh_not_permanent_SHA_whitelist(tmp_path,monkeypatch):
 m,rr,br,request,entries,paths,audits,dump=fixture(tmp_path,monkeypatch);one=execute(m,rr,br);b=json.loads(Path(br['path']).read_text());anchors=b['capacity_evidence'];plan=json.loads(Path(anchors['plan']['path']).read_text());acceptance=json.loads(Path(anchors['acceptance']['path']).read_text());review=json.loads(Path(anchors['independent_review']['path']).read_text());review['review_generation']=2
 # A new externally SHA-bound reviewed closure packet carries the full remaining
 # H8 bound. No credit is guessed from a running process or partial feature file.
 oldkey='remaining_H_primary9_full_caps';newkey='remaining_H_primary8_full_caps';review['budget_breakdown_bytes'][newkey]=8*1024**3;review['budget_breakdown_bytes'].pop(oldkey);review['remaining_primary_future_pair_count']=8;review['H_independently_accepted_pair_count']=7;review['required_total_bytes']-=1024**3;plan['base_future_and_reserve_bytes']-=1024**3
 base=m.ROOT/'engineering/source_best_M';closed=json.loads(Path(acceptance['actual_H6_acceptance']['path']).read_text());closed['H_total_independently_accepted_pairs']=7;newclosure=dump(m.ROOT/'reports/H_independent_closure_refresh.json',closed);review['bindings']['new_H_acceptance']=newclosure;acceptance['actual_H6_acceptance']=newclosure;newplan=dump(base/'plans/M_reviewed_remaining_write_plan_refresh.json',plan);review['bindings']['new_plan']=newplan;newreview=dump(base/'independent_M_capacity_refresh.json',review);acceptance['write_plan']=newplan;acceptance['independent_review']=newreview;acceptance['full_required_remaining_with_peak_and_reserve_bytes']-=1024**3;newaccept=dump(base/'M_capacity_refresh_acceptance.json',acceptance)
 newanchors=dict(plan=newplan,acceptance=newaccept,independent_review=newreview);b.update(capacity_evidence=newanchors,evidence=list(newanchors.values()),decomposition_bytes=review['budget_breakdown_bytes'],full_remaining_with_peaks_reserve_bytes=review['required_total_bytes'],utc=datetime.datetime.now(datetime.timezone.utc).isoformat());br=dump(m.ROOT/'reports/new_external_reviewed_budget.json',b);two=execute(m,rr,br);assert two['reused'] is True and two['sha256']==one['sha256']


def test_HSUM001_precommit_failure_preserves_owned_stage_and_unknown_remnants(tmp_path,monkeypatch):
 m,rr,br,request,entries,paths,audits,dump=fixture(tmp_path,monkeypatch);original=m.atomic_commit;calls=[]
 def fail(stage,out):calls.append(stage);raise OSError('injected precommit failure')
 monkeypatch.setattr(m,'atomic_commit',fail)
 with pytest.raises(OSError,match='precommit'):execute(m,rr,br)
 out=m.ROOT/'analysis/H/clip_vit_b16/five_seed_saved_result_summary';assert not out.exists() and len(calls)==1 and calls[0].is_dir();before={p.name:m.digest(p) for p in calls[0].iterdir()}
 monkeypatch.setattr(m,'atomic_commit',original);one=execute(m,rr,br);assert Path(one['output'])==out and before=={p.name:m.digest(p) for p in calls[0].iterdir()}


def test_HSUM002_statistical_receipt_is_deterministic_and_runtime_facts_are_operational(tmp_path,monkeypatch):
 m,rr,br,request,entries,paths,audits,dump=fixture(tmp_path,monkeypatch);one=execute(m,rr,br);out=Path(one['output']);c=json.loads((out/'complete.json').read_text());assert not {'cpu_slot','operational_budget_ref','lifetime_shared_cpu_slot_held'}&set(c);assert one['cpu_slot']==1 and one['lifetime_shared_cpu_slot_held'] is True
 two=m.main(['--execute','--request',rr['path'],'--request-sha',rr['sha256'],'--budget',br['path'],'--budget-sha',br['sha256'],'--slot','0']);assert two['cpu_slot']==0 and two['sha256']==one['sha256'] and two['reused'] is True



def test_HSUM003_sparse_unknown_stage_is_preserved_and_charged_on_reuse(tmp_path,monkeypatch):
 m,rr,br,request,entries,paths,audits,dump=fixture(tmp_path,monkeypatch);out=Path(execute(m,rr,br)['output']);stage=out.parent/('.'+out.name+'.stage_unknown');stage.mkdir();sparse=stage/'unknown_sparse_payload'
 with sparse.open('wb') as f:f.truncate(m.CAP+4096)
 assert sparse.stat().st_size>m.CAP and sparse.stat().st_blocks*512<m.CAP
 with pytest.raises(AssertionError,match='allocat|cap|peak'):execute(m,rr,br)
 assert sparse.exists()


def test_HSUM005_closure_proof_counter_must_match_review_not_just_a_hash(tmp_path,monkeypatch):
 m,rr,br,request,entries,paths,audits,dump=fixture(tmp_path,monkeypatch);b=json.loads(Path(br['path']).read_text());anchors=b['capacity_evidence'];a=json.loads(Path(anchors['acceptance']['path']).read_text());r=json.loads(Path(anchors['independent_review']['path']).read_text());p=a['actual_H6_acceptance']['path'];closed=json.loads(Path(p).read_text());closed['H_total_independently_accepted_pairs']=5;wrong=dump(p,closed);a['actual_H6_acceptance']=wrong;r['bindings']['new_H_acceptance']=wrong;newr=dump(anchors['independent_review']['path'],r);a['independent_review']=newr;newa=dump(anchors['acceptance']['path'],a);anchors.update(independent_review=newr,acceptance=newa);b.update(evidence=list(anchors.values()),capacity_evidence=anchors);br=dump(br['path'],b)
 with pytest.raises(AssertionError,match='closure acceptance count'):execute(m,rr,br)
 assert not entries



def test_HSUM001_postcommit_directory_fsync_failure_is_truthful_committed_result(tmp_path,monkeypatch):
 import os
 m,rr,br,request,entries,paths,audits,dump=fixture(tmp_path,monkeypatch);original=os.fsync;parent=m.ROOT/'analysis/H/clip_vit_b16'
 def fail(fd):
  if os.readlink('/proc/self/fd/'+str(fd))==str(parent):raise OSError('postcommit parent fsync unavailable')
  return original(fd)
 monkeypatch.setattr(m.os,'fsync',fail);one=execute(m,rr,br);assert one['reused'] is False and one['directory_fsync_confirmed'] is False;two=execute(m,rr,br);assert two['reused'] is True and two['sha256']==one['sha256']


def test_HSUM001_atomic_commit_refuses_even_concurrently_created_empty_destination(tmp_path,monkeypatch):
 m,rr,br,request,entries,paths,audits,dump=fixture(tmp_path,monkeypatch);original=m.atomic_commit;stages=[]
 def collision(stage,out):stages.append(stage);out.mkdir();return original(stage,out)
 monkeypatch.setattr(m,'atomic_commit',collision)
 with pytest.raises(FileExistsError):execute(m,rr,br)
 out=m.ROOT/'analysis/H/clip_vit_b16/five_seed_saved_result_summary';assert out.is_dir() and not list(out.iterdir()) and stages[0].is_dir() and len(list(stages[0].iterdir()))==7


def test_HSUM003_late_KEEP_SIZE_before_commit_keeps_output_unpublished(tmp_path,monkeypatch):
 import ctypes,os
 m,rr,br,request,entries,paths,audits,dump=fixture(tmp_path,monkeypatch);guard=m.publication_guard;calls=[0];mutated=[]
 def boundary(*args):
  guard(*args);calls[0]+=1
  if calls[0]==4:
   parent=m.ROOT/'analysis/H/clip_vit_b16';stage=next(parent.glob('.five_seed_saved_result_summary.stage_*'));p=stage/'per_seed_metrics.csv';before=m.digest(p);fd=os.open(p,os.O_RDWR)
   try:r=ctypes.CDLL(None,use_errno=True).fallocate(fd,1,ctypes.c_longlong(0),ctypes.c_longlong(m.CAP+4096))
   finally:os.close(fd)
   if r:pytest.skip('filesystem lacks FALLOC_FL_KEEP_SIZE')
   assert before==m.digest(p);mutated.append(p)
 monkeypatch.setattr(m,'publication_guard',boundary)
 with pytest.raises(AssertionError,match='allocat|cap|peak'):execute(m,rr,br)
 assert mutated and mutated[0].exists() and not (m.ROOT/'analysis/H/clip_vit_b16/five_seed_saved_result_summary').exists()


def test_HSUM004_coherent_historical_union_retains_one_identity_without_read(tmp_path,monkeypatch):
 m,rr,br,request,entries,paths,audits,dump=fixture(tmp_path,monkeypatch);path=str(m.ROOT/'data/H/same_history.png')
 for i in (0,1):
  a=json.loads(audits[i].read_text());a['current_pins_sha256'][path]='a'*64;request['pairs'][i]['independent_numeric_audit']=dump(audits[i],a)
 rr=dump(rr['path'],request);out=Path(execute(m,rr,br)['output']);i=json.loads((out/'identity.json').read_text());assert i['historical_nonconsumed_source_sha256']=={path:'a'*64} and path not in i['fresh_checked_input_sha256']



def test_HSUM003_R1_two_normal_publishers_cannot_reserve_same_cap_concurrently(tmp_path,monkeypatch):
 import threading,time
 m,rr,br,request,entries,paths,audits,dump=fixture(tmp_path,monkeypatch);out=m.ROOT/'analysis/H/clip_vit_b16/five_seed_saved_result_summary';payloads={n:b'x' for n in m.OUTPUTS};payloads[m.OUTPUTS[0]]=b'x'*(9*1024**2);B_predicted=threading.Event();allow_B_mkdir=threading.Event();A_final=threading.Event();A_done=threading.Event();B_done=threading.Event();original_mkdir=Path.mkdir;commit=m.atomic_commit;results={};errors={}
 def mkdir(p,*args,**kwargs):
  if threading.current_thread().name=='publisher_B' and p.name.startswith('.'+out.name+'.stage_'):
   B_predicted.set();assert allow_B_mkdir.wait(10),'private B barrier timeout'
  return original_mkdir(p,*args,**kwargs)
 def pause_after_A_last_check(stage,destination):
  if threading.current_thread().name=='publisher_A':A_final.set();assert B_done.wait(10),'private A barrier timeout'
  return commit(stage,destination)
 def run(name,done):
  try:results[name]=m.publish(out,payloads,rr['sha256'],{},m.INITIAL_SHA,br['path'],br['sha256'],1)
  except BaseException as error:errors[name]=error
  finally:done.set()
 monkeypatch.setattr(Path,'mkdir',mkdir);monkeypatch.setattr(m,'atomic_commit',pause_after_A_last_check)
 B=threading.Thread(target=run,args=('B',B_done),name='publisher_B');A=threading.Thread(target=run,args=('A',A_done),name='publisher_A');B.start()
 try:
  assert B_predicted.wait(10);A.start();deadline=time.monotonic()+10
  while not (A_final.is_set() or A_done.is_set()) and time.monotonic()<deadline:A_done.wait(.01)
  assert A_final.is_set() or A_done.is_set();allow_B_mkdir.set();A.join(15);B.join(15)
 finally:
  allow_B_mkdir.set();B_done.set();B.join(15)
  if A.ident is not None:A.join(15)
 assert not A.is_alive() and not B.is_alive();allocated=m.namespace_allocation(out)
 assert allocated<=m.CAP,dict(allocated=allocated,cap=m.CAP,results=results,errors={k:str(v) for k,v in errors.items()})
 assert len(results)==1 and len(errors)==1 and 'writer busy' in str(next(iter(errors.values())))
 assert (out/'complete.json').is_file()



def test_HSUM003_R1_cross_process_busy_then_release_without_lock_or_slot_IO(tmp_path,monkeypatch):
 import subprocess,os,sys
 m,rr,br,request,entries,paths,audits,dump=fixture(tmp_path,monkeypatch);out=m.ROOT/'analysis/H/clip_vit_b16/five_seed_saved_result_summary'
 code="from pathlib import Path;import importlib.util,sys,os;sp=importlib.util.spec_from_file_location('private_writer_child',sys.argv[1]);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)\ntry:\n with m.writer_reservation(Path(sys.argv[2])):print('ACQUIRED',flush=True)\nexcept RuntimeError as e:print(str(e),flush=True);sys.exit(23)\n"
 with m.writer_reservation(out) as lease:
  child=subprocess.run([sys.executable,'-B','-c',code,str(P),str(out)],text=True,capture_output=True,timeout=10);assert child.returncode==23 and 'writer busy' in child.stdout
  with m.writer_reservation(out.parent/'different_summary'):pass
 assert lease['release_confirmed'] is True
 child=subprocess.run([sys.executable,'-B','-c',code,str(P),str(out)],text=True,capture_output=True,timeout=10);assert child.returncode==0 and child.stdout.strip()=='ACQUIRED'
 assert not entries and not out.exists()


def test_HSUM003_R1_process_exit_releases_reservation_without_disk_remnant(tmp_path,monkeypatch):
 import subprocess,sys
 m,rr,br,request,entries,paths,audits,dump=fixture(tmp_path,monkeypatch);out=m.ROOT/'analysis/H/clip_vit_b16/five_seed_saved_result_summary'
 code="from pathlib import Path;import importlib.util,sys,os;sp=importlib.util.spec_from_file_location('private_writer_exit',sys.argv[1]);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)\nwith m.writer_reservation(Path(sys.argv[2])):os._exit(0)\n"
 child=subprocess.run([sys.executable,'-B','-c',code,str(P),str(out)],text=True,capture_output=True,timeout=10);assert child.returncode==0
 with m.writer_reservation(out):pass
 assert not entries and not out.exists()



@pytest.mark.parametrize('same_audit',[False,True])
def test_HSUM004_R1_lexical_alias_conflict_rejected_without_pixel_hash(tmp_path,monkeypatch,same_audit):
 m,rr,br,request,entries,paths,audits,dump=fixture(tmp_path,monkeypatch);canonical=str(m.ROOT/'data/H/shared.png');alias=str(m.ROOT)+'/data/H/./shared.png';first=json.loads(audits[0].read_text());first['current_pins_sha256'][canonical]='a'*64
 if same_audit:first['current_pins_sha256'][alias]='b'*64
 request['pairs'][0]['independent_numeric_audit']=dump(audits[0],first)
 if not same_audit:
  second=json.loads(audits[1].read_text());second['current_pins_sha256'][alias]='b'*64;request['pairs'][1]['independent_numeric_audit']=dump(audits[1],second)
 rr=dump(rr['path'],request);digest=m.digest
 def no_pixels(path):assert Path(path)!=Path(canonical),'historical pixel bytes must not be read';return digest(path)
 monkeypatch.setattr(m,'digest',no_pixels)
 with pytest.raises(AssertionError,match='historical|conflict'):execute(m,rr,br)
 assert not (m.ROOT/'analysis/H/clip_vit_b16/five_seed_saved_result_summary').exists()


def test_HSUM004_R1_coherent_alias_normalizes_one_historical_key_without_read(tmp_path,monkeypatch):
 m,rr,br,request,entries,paths,audits,dump=fixture(tmp_path,monkeypatch);canonical=str(m.ROOT/'data/H/shared.png');alias=str(m.ROOT)+'/data/H/./shared.png'
 for index,path in ((0,canonical),(1,alias)):
  a=json.loads(audits[index].read_text());a['current_pins_sha256'][path]='a'*64;request['pairs'][index]['independent_numeric_audit']=dump(audits[index],a)
 rr=dump(rr['path'],request);digest=m.digest
 def no_pixels(path):assert Path(path)!=Path(canonical),'historical pixel bytes must not be read';return digest(path)
 monkeypatch.setattr(m,'digest',no_pixels);out=Path(execute(m,rr,br)['output']);i=json.loads((out/'identity.json').read_text());assert i['historical_nonconsumed_source_sha256']=={canonical:'a'*64};assert i['pairs'][1]['historical_nonconsumed_source_sha256']=={canonical:'a'*64};assert canonical not in i['fresh_checked_input_sha256']
