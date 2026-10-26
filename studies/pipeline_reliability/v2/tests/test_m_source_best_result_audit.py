"""CPU-only private fixtures; never opens an original cache or CPU slot."""
import copy, hashlib, importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

SPEC=importlib.util.spec_from_file_location('m_saved_consumer',Path(__file__).parents[1]/'tools/m_source_best_result_audit.py')
audit=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

def encoded(q,aggregation):
    q=q.copy()
    q['representation']='raw';q['grouping']='correct';q['aggregation']=aggregation
    q['rank_score_kind']='float64'
    q['rank_score_decimal']=q.score.map(lambda x:format(float(x),'.350g'))
    q['rank_score_export_digits']=350;q['ranking_usable']=True
    return q

@pytest.fixture(scope='module')
def h_values():
    # Fixture producer is deliberately separate from the independent oracle.
    from reliability_v2 import stats_h
    values={}
    for aggregation in audit.AGGREGATIONS:
        def rows(domain,methods,cal=False):
            out=[]
            for group in range(20):
                for method in (('real',) if cal else ('real',)+tuple(methods)):
                    label=int(method!='real')
                    score=(label*2-1)+group*.001 if aggregation=='mean_logit' else .1+label*.8+group*.001
                    out.append(dict(video_uid=f'{domain}/{method}/{group}',label=label,method_id=method,group_id=f'{domain}/g{group}',score=score))
            return encoded(pd.DataFrame(out),aggregation)
        cal={h:rows('cal',(),True) for h in ('B','AB')}
        for h,q in cal.items():values[f'{aggregation}/source_cal_{h}.csv']=q
        for domain,methods in [('SOURCE_TEST',audit.H_METHODS),('TARGET_TEST',('DeepFakeDetection',))]:
            folder=f'{aggregation}/{domain}/all_B'
            cells={c:rows(domain,methods) for c in ('H00','H10')}
            result=stats_h.analyze_cohort(cells,cal,fixed_methods=methods,aggregation=aggregation,repeats=2000,seed=20261023)
            for c,q in cells.items():values[f'{folder}/{c}_scores.csv']=q
            for name,key in [('metrics','point_metrics'),('comparisons','comparisons'),('thresholds','thresholds'),('conditional_bootstrap','bootstrap'),('independent_kernel_checks','independent_checks')]:
                values[f'{folder}/{name}.'+('csv' if name in ('metrics','comparisons') else 'json')]=result[key]
            values[f'{folder}/independent_saved_score_checks.json']=[dict(cell=c,threshold_kind=k,status='PASS') for c in cells for k in ('default','source_real')]
    values['limitations.json']=dict(checkpoint_selection='source_best',source_best_epochs={'B':1,'AB':2},source_only_selected_prefix={h:dict(seed=7,epoch=e,attempts=e*10,successful_updates=e*10,AMP_skips=0) for h,e in [('B',1),('AB',2)]},update_matched_primary_claim=False,test_bootstrap_repeats=2000,resampling_seed=20261023,source_CAL_resampled=False,training_seeds_resampled=False,production_admission_provided_here=False)
    from reliability_v2 import analysis_d
    return {name:analysis_d._jsonable(value) if name.endswith('.json') else value for name,value in values.items()}

def test_fixed_inventory_is_independent_and_exact():
    assert len(audit.expected_inventory('D'))==60
    assert len(audit.expected_inventory('H'))==37
    assert not any(x in n for n in audit.expected_inventory('H') for x in ('common/','H01','H11'))
    with pytest.raises(ValueError):audit.expected_inventory('F')

def test_closed_form_full_2000_draws(h_values):
    result=audit.audit_H_artifacts(h_values)
    assert result['accepted_for_selected_core'] is True
    assert len(result['cohorts'])==4
    assert sum(x['draws_checked'] for x in result['cohorts'])==8000
    for c in result['cohorts']:
        assert c['independent_point_estimates']['default:cell:H00:pooled_auc']==1
        assert c['independent_point_estimates']['default:comparison:H10-H00:pooled_ba']==0
        assert c['source_real_thresholds']['B']> ( -.982 if c['aggregation']=='mean_logit' else .118)

@pytest.mark.parametrize('mutation',['effect','draw','NA','repeats','seed','cal','missing'])
def test_core_mutations_fail(h_values,mutation):
    q=copy.deepcopy(h_values);base='mean_logit/TARGET_TEST/all_B'
    if mutation=='effect':q[base+'/comparisons.csv'].loc[0,'estimate']=.2
    elif mutation=='draw':q[base+'/conditional_bootstrap.json']['draws'][0]['default:comparison:H10-H00:fpr']=.2
    elif mutation=='NA':q[base+'/conditional_bootstrap.json']['point_estimates']['default:cell:H00:pooled_auc']=None
    elif mutation=='repeats':q[base+'/conditional_bootstrap.json']['repeats']=1999
    elif mutation=='seed':q[base+'/conditional_bootstrap.json']['seed']=7
    elif mutation=='cal':q['mean_logit/source_cal_B.csv'].loc[0,'label']=1
    elif mutation=='missing':del q[base+'/metrics.csv']
    with pytest.raises((ValueError,AssertionError,KeyError)):audit.audit_H_artifacts(q)

def fixture_transport(tmp_path,route='H',values=None):
    from reliability_v2 import multiplicity_bundle as bundle
    inventory=audit.expected_inventory(route)
    ident=dict(schema='M_STATISTIC_STATIC_IDENTITY_V1',job=dict(route=route),checkpoint_selection='source_best')
    claim=dict(schema=audit.SINK_SCHEMA,identity=ident,inventory=inventory,limits={k:10000000 for k in audit.LIMIT_KEYS},implementations=audit.producer_transport_codes())
    output=tmp_path/'result';output.mkdir()
    (output/'artifacts').mkdir();(output/'.receipts').mkdir();(output/'.staging').mkdir()
    (output/'writer.lock').touch()
    (output/'identity.json').write_bytes(audit.typed(claim)+b'\n')
    receipts={}
    for name,kind in inventory.items():
        path=output/'artifacts'/(name+('.bundle' if kind=='json' else ''));path.parent.mkdir(parents=True,exist_ok=True)
        aid=dict(sink_identity_sha256=hashlib.sha256(audit.typed(claim)).hexdigest(),artifact=name,kind=kind)
        br=None
        if kind=='json':br=bundle.write_bundle(path,values[name] if values is not None else {'synthetic':name},identity=aid,max_cells=10000000,max_json_bytes=10000000,max_array_bytes=10000000)
        elif kind=='array':np.save(path,np.array([1.]),allow_pickle=False)
        elif kind=='parquet':pd.DataFrame({'x':[1]}).to_parquet(path,index=False,compression=None,use_dictionary=False)
        elif values is not None:values[name].to_csv(path,index=False,float_format='%.17g')
        else:path.write_text('x\n1\n')
        files={str(p.relative_to(output/'artifacts')):audit.sha(p) for p in (path.rglob('*') if path.is_dir() else [path]) if p.is_file()}
        rp=output/'.receipts'/(hashlib.sha256(name.encode()).hexdigest()+'.json')
        rp.write_bytes(audit.typed(dict(schema=audit.SINK_SCHEMA,identity=aid,files=files,bundle=br))+b'\n')
        receipts[name]=audit.sha(rp)
    (output/'complete.json').write_bytes(audit.typed(dict(schema=audit.SINK_SCHEMA,status='ALL_DECLARED_ARTIFACTS_COMMITTED_REQUIRES_SCIENTIFIC_ADMISSION',identity_sha256=hashlib.sha256(audit.typed(claim)).hexdigest(),receipts=receipts,scientific_admitted=False))+b'\n')
    return output

def test_transport_is_lazy_read_only_and_rechecks_initial_pins(tmp_path,monkeypatch):
    output=fixture_transport(tmp_path);before={str(p):audit.sha(p) for p in output.rglob('*') if p.is_file()}
    reader=audit.ReadOnlyArtifacts(output,audit.sha(output/'complete.json'),route='H')
    assert reader.decoded_assets==0
    assert reader['limitations.json']=={'synthetic':'limitations.json'}
    assert reader.decoded_assets==1
    reader.check()
    assert before=={str(p):audit.sha(p) for p in output.rglob('*') if p.is_file()}
    # Private digest mock: never append to or mutate real source files.
    original=audit.sha
    monkeypatch.setattr(audit,'sha',lambda p:'f'*64 if Path(p)==output/'identity.json' else original(p))
    with pytest.raises(ValueError,match='drift'):reader.check()

@pytest.mark.parametrize('mutation',['complete','extra','symlink','receipt'])
def test_transport_rejects_metadata_failures(tmp_path,mutation):
    output=fixture_transport(tmp_path);pin=audit.sha(output/'complete.json')
    if mutation=='extra':(output/'artifacts'/'unregistered.csv').write_text('x\n1\n')
    elif mutation=='symlink':(output/'artifacts'/'unregistered').symlink_to(output/'complete.json')
    elif mutation=='receipt':
        rp=next((output/'.receipts').iterdir());rp.write_text('{}')
    if mutation=='complete':pin='0'*64
    with pytest.raises(ValueError):audit.ReadOnlyArtifacts(output,pin,route='H')

def test_default_entry_prepares_only(monkeypatch):
    monkeypatch.setattr(audit,'execute',lambda *a,**k:pytest.fail('must not execute'))
    r=audit.main([])
    assert r['production_audit_started'] is False and r['accepted_for_selected_core'] is False

def test_selection_prefix_is_bound_and_fails_closed(h_values):
    spec=dict(route='H',backbone='clip_vit_b16',seed=7,job_id='H_clip_vit_b16_seed7',runs={h:f'H_formal_clip_vit_b16_{h}_seed7' for h in ('B','AB')})
    contexts={h:dict(schema='M_SOURCE_BEST_RAW_FP32_CACHE_V1',run_id=run,checkpoint_selection='source_best',route='H',seed=7,backbone='clip_vit_b16',parent_run_id=run,epochs=30,source_best_epoch=e,source_best_prefix=dict(epoch=e,attempts=e*10,successful_updates=e*10,AMP_skips=0)) for (h,run),e in zip(spec['runs'].items(),(1,2))}
    audit.check_selection(spec,contexts,h_values['limitations.json'])
    contexts['B']['source_best_prefix']['AMP_skips']=1
    with pytest.raises(ValueError):audit.check_selection(spec,contexts,h_values['limitations.json'])

def test_actual_requires_external_evidence_before_slot(tmp_path,monkeypatch):
    monkeypatch.setattr(audit,'_actual_slot',lambda *a:pytest.fail('missing evidence must not reach slot'))
    with pytest.raises(ValueError):audit.execute(tmp_path/'missing.json','0'*64,0)

def metadata_fixture(tmp_path,monkeypatch,h_values):
    monkeypatch.setattr(audit,'ROOT',tmp_path)
    input_sha='a'*64
    job=audit.job_spec('H','clip_vit_b16',7)
    contexts={h:dict(schema='M_SOURCE_BEST_RAW_FP32_CACHE_V1',run_id=run,checkpoint_selection='source_best',route='H',seed=7,backbone=job['backbone'],parent_run_id=run,epochs=30,source_best_epoch=e,source_best_prefix=dict(epoch=e,attempts=e*10,successful_updates=e*10,AMP_skips=0)) for (h,run),e in zip(job['runs'].items(),(1,2))}
    counter=[0]
    def proof(name,value):
        p=tmp_path/(name+'.json');p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(audit.typed(value)+b'\n')
        return dict(path=str(p),sha256=audit.sha(p))
    inp=dict(schema='M_SOURCE_BEST_CACHE_INPUT_ADMISSION_V1',artifact_root=str(tmp_path),source_input_scope='SELECTED_M_D6_H3_FIVE_SEEDS',target_inference_authorized=False,cache_code_sha256=audit.producer_package_codes()[str(audit.CODE/'reliability_v2/source_best_cache.py')])
    ir=proof('input',inp);input_sha=ir['sha256']
    markers={};heads={}
    w=np.zeros((2,512),np.float32);b=np.zeros(2,np.float32);head=hashlib.sha256(w.tobytes()+b.tobytes()).hexdigest()
    for h,run in job['runs'].items():
        for role in ('B/SOURCE_CAL','B/SOURCE_TEST','B/TARGET_TEST'):
            leaf=role.split('/')[-1]
            key='mean_logit/source_cal_B.csv' if leaf=='SOURCE_CAL' else f'mean_logit/{leaf}/all_B/H00_scores.csv'
            q=h_values[key][['video_uid','label','method_id','group_id']].copy()
            q['role']=leaf;q['domain']='target' if leaf=='TARGET_TEST' else 'source';q['processing_version']='B'
            mp=tmp_path/'data/H'/(role+'.parquet');mp.parent.mkdir(parents=True,exist_ok=True)
            if not mp.exists():q.to_parquet(mp,index=False)
            folder=tmp_path/'features/source_best_M'/input_sha/run/role;folder.mkdir(parents=True)
            (folder/'manifest.parquet').write_bytes(mp.read_bytes())
            np.save(folder/'head_weight.npy',w,allow_pickle=False);np.save(folder/'head_bias.npy',b,allow_pickle=False)
            np.save(folder/'chunk_000000.npy',np.zeros((len(q),512),np.float32),allow_pickle=False)
            chunk=dict(file='chunk_000000.npy',start=0,rows=len(q),dimension=512,dtype='float32',sha256=audit.sha(folder/'chunk_000000.npy'))
            (folder/'chunk_000000.json').write_bytes(audit.typed(chunk))
            v=dict(contexts[h],checkpoint_selection='source_best',role=role,original_head_sha256=head,source_manifest_path=str(mp),source_manifest_sha256=audit.sha(mp))
            marker=dict(status='CACHE_COMPLETE',precision='FP32',autocast=False,identity=v,rows=len(q),dimension=512,chunk_rows=4096,chunks=[chunk],cached_manifest_sha256=audit.sha(mp),head=dict(weight_file='head_weight.npy',bias_file='head_bias.npy',weight_sha256=audit.sha(folder/'head_weight.npy'),bias_sha256=audit.sha(folder/'head_bias.npy')))
            (folder/'complete.json').write_bytes(audit.typed(marker))
            markers[h+'/'+role]=audit.sha(folder/'complete.json');heads[h+'/'+role]=head
    source_proof=proof('independent_source_proof',{'synthetic_prior_source_proof':True})
    ad=dict(schema='M_SOURCE_BEST_JOB_CACHE_INDEPENDENT_ADMISSION_V1',status='PASS',artifact_root=str(tmp_path),job=job,input_admission_sha256=input_sha,independent_role_and_source_best_head_validation_passed=True,fresh_original_source_context_and_all_selected_role_checks=True,source_contexts=contexts,cache_completion_sha256=markers,original_head_sha256_by_role=heads,pinned_source_proof_sha256={source_proof['path']:source_proof['sha256']},cache_code_sha256=inp['cache_code_sha256'])
    ar=proof('cache_admission',ad);package=audit.producer_package_codes()
    qual=dict(schema='M_STATISTIC_RESOURCE_AND_FORMAT_ACCEPTANCE_V1',status='PASS',code_sha256=package,input_admission_sha256=input_sha,actual_D_transport_accepted=True,actual_H_terminal_bound_transport_accepted=True,independent_entry_review_passed=True,actual_entry_resource_qualified=False,bounded_canary_resource_plan_independently_accepted=True)
    qr=proof('qualification',qual);limits={k:10000000 for k in audit.LIMIT_KEYS}
    permit=dict(schema='M_BOUNDED_STATISTIC_CANARY_ADMISSION_V1',status='PASS',artifact_root=str(tmp_path),input_admission_sha256=input_sha,four_core_comparisons_only=True,all_five_seeds_required=True,job_ids=[job['job_id'],'D_clip_vit_b16_seed7'],code_sha256=package,limits={'H':limits},qualification=qr,capacity=proof('capacity',{'synthetic':True}),cache_permit=proof('cache_permit',dict(schema='M_SOURCE_BEST_PRODUCTION_CACHE_PERMIT_V1',status='PASS_PRODUCTION_CACHE_ADMISSION',artifact_root=str(tmp_path),input_admission_sha256=input_sha,cache_code_sha256=inp['cache_code_sha256'],target_inference_authorized=True,four_core_comparisons_only=True,parent_run_ids=list(job['runs'].values()),**{name:proof(name,dict(M_production_cache_admission=True,cache_code_sha256=inp['cache_code_sha256'],input_admission_sha256=input_sha)) for name in ('cache_qualification','actual_D_storage','actual_H_storage','joint_capacity')})))
    pr=proof('permit',permit)
    ident=dict(schema='M_STATISTIC_STATIC_IDENTITY_V1',scope='SELECTED_FOUR_CORE_SOURCE_BEST_DESCRIPTIVE_SENSITIVITY',job=job,checkpoint_selection='source_best',CAL_repeats=200,TEST_repeats=2000,resampling_seed=20261023,update_matched_primary_claim=False,source_contexts=contexts,input_admission_sha256=input_sha,cache_completion_sha256=markers,code_sha256=package)
    output=tmp_path/'analysis/source_best_M'/hashlib.sha256(audit.typed(ident)).hexdigest()/job['job_id']
    producer=dict(schema='M_STATISTIC_EXECUTION_RETURN_V1',status='COMPLETE_DECLARED_TRANSPORT_PENDING_INDEPENDENT_SAVED_RESULT_AUDIT',job=job,output=str(output),complete_sha256='c'*64,identity_sha256=output.parent.name,independent_cache_admission_sha256=ar['sha256'],production_permit_sha256=pr['sha256'],scientific_final_acceptance=False,source_best_only=True,GPU_or_model_training_started=False,CPU_slot=0)
    er=proof('producer_execution',producer)
    stdout=tmp_path/'stdout.log';stdout.write_bytes(audit.typed(producer)+b'\n')
    timing=tmp_path/'time_v.log';timing.write_text('Maximum resident set size (kbytes): 12345\nExit status: 0\n')
    launch=dict(schema='M_STATISTIC_PRODUCER_LAUNCH_V1',job=job,output=str(output),CPU_slot=0,environment={'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'},address_space_limit_bytes=32*1024**3,cache_admission=ar,production_permit=pr,input_admission=ir,stdout=str(stdout),time_v=str(timing),cwd=str(audit.CODE))
    launch['command']=['/usr/bin/time','-v','-o',str(timing),str(tmp_path/'environment/bin/python'),'-B','-u','-m','reliability_v2.source_best_analysis','--execute','--root',str(tmp_path),'--input',ir['path'],'--input-sha',ir['sha256'],'--admission',ar['path'],'--admission-sha',ar['sha256'],'--permit',pr['path'],'--permit-sha',pr['sha256'],'--slot','0']
    lr=proof('launch',launch)
    xr=proof('exit',dict(schema='M_STATISTIC_PRODUCER_EXIT_V1',job=job,launch_sha256=lr['sha256'],exit_code=0,returncode_observed_by_wait=True,wrapper_and_child_true_exit_observed=True))
    refs=dict(input_admission=ir,cache_admission=ar,production_permit=pr,producer_execution=er,producer_launch=lr,producer_exit=xr,producer_stdout=dict(path=str(stdout),sha256=audit.sha(stdout)),producer_time_v=dict(path=str(timing),sha256=audit.sha(timing)))
    request=dict(job=job,proofs=refs,output=str(output))
    class Reader(dict):
        def check(self):pass
    reader=Reader(h_values);reader.identity=ident;reader.output=output;reader.complete_sha256='c'*64;reader.limits=limits
    return request,reader

def test_private_six_role_metadata_chain_without_slot(tmp_path,monkeypatch,h_values):
    monkeypatch.setattr(audit,'_actual_slot',lambda *a:pytest.fail('synthetic metadata must not touch slot'))
    request,reader=metadata_fixture(tmp_path,monkeypatch,h_values)
    pins=audit.Pins();context=audit.metadata_chain(request,reader,pins)
    assert context['selection']['AB']['epoch']==2 and len(context['manifest_ids'])==3
    assert context['producer_peak_RSS_KiB']==12345
    pins.check()

@pytest.mark.parametrize('mutation',['admission','exit','time','env','command','resource','stdout','chunk','head'])
def test_private_producer_admission_mutations(tmp_path,monkeypatch,h_values,mutation):
    request,reader=metadata_fixture(tmp_path,monkeypatch,h_values);refs=request['proofs']
    if mutation in ('admission','exit','env','command','resource'):
        key={'admission':'cache_admission','exit':'producer_exit','env':'producer_launch','command':'producer_launch','resource':'producer_launch'}[mutation]
        p=Path(refs[key]['path']);doc=json.loads(p.read_text())
        if mutation=='admission':doc['independent_role_and_source_best_head_validation_passed']=False
        elif mutation=='exit':doc['exit_code']=1
        elif mutation=='env':doc['environment']['CUDA_VISIBLE_DEVICES']='4'
        elif mutation=='command':doc['command'][-1]='1'
        else:doc['address_space_limit_bytes']=64*1024**3
        p.write_bytes(audit.typed(doc));refs[key]['sha256']=audit.sha(p)
    elif mutation in ('time','stdout'):
        key='producer_time_v' if mutation=='time' else 'producer_stdout'
        p=Path(refs[key]['path']);p.write_text('Exit status: 1\n' if mutation=='time' else '{}\n');refs[key]['sha256']=audit.sha(p)
    else:
        p=next((tmp_path/'features').rglob('chunk_000000.npy' if mutation=='chunk' else 'head_weight.npy'))
        p.write_bytes(b'private synthetic payload mutation')
    with pytest.raises((ValueError,AssertionError)):audit.metadata_chain(request,reader,audit.Pins())

def test_source_pin_guard_uses_import_time_SHA(monkeypatch):
    original=audit.sha
    monkeypatch.setattr(audit,'sha',lambda p:'f'*64 if Path(p)==Path(audit.__file__) else original(p))
    with pytest.raises(ValueError,match='drift'):
        audit.Pins().add(audit.__file__,audit.ENTRY_SHA)

@pytest.mark.parametrize('case',['negative_effect','declared_numeric_NA'])
def test_negative_effect_and_declared_NA_are_retained(h_values,case):
    from reliability_v2 import stats_h,analysis_d
    values=copy.deepcopy(h_values);base='mean_logit/TARGET_TEST/all_B'
    cells={c:values[f'{base}/{c}_scores.csv'].copy() for c in ('H00','H10')}
    if case=='negative_effect':
        cells['H10']['score']=-cells['H10'].score
        cells['H10']['rank_score_decimal']=cells['H10'].score.map(lambda x:format(float(x),'.350g'))
    else:cells['H10']['ranking_usable']=False
    cal={h:values[f'mean_logit/source_cal_{h}.csv'] for h in ('B','AB')}
    result=stats_h.analyze_cohort(cells,cal,fixed_methods=('DeepFakeDetection',),aggregation='mean_logit',repeats=2000,seed=20261023)
    for c,q in cells.items():values[f'{base}/{c}_scores.csv']=q
    for name,key in [('metrics','point_metrics'),('comparisons','comparisons'),('thresholds','thresholds'),('conditional_bootstrap','bootstrap'),('independent_kernel_checks','independent_checks')]:
        values[f'{base}/{name}.'+('csv' if name in ('metrics','comparisons') else 'json')]=analysis_d._jsonable(result[key]) if name not in ('metrics','comparisons') else result[key]
    values[f'{base}/independent_saved_score_checks.json']=[dict(cell=c,threshold_kind=k,status='PASS_OPERATING_ONLY_NUMERIC_AUC_NA' if c=='H10' and case=='declared_numeric_NA' else 'PASS') for c in cells for k in ('default','source_real')]
    actual=audit.audit_H_artifacts(values)
    row=next(x for x in actual['cohorts'] if x['domain']=='TARGET_TEST' and x['aggregation']=='mean_logit')
    if case=='negative_effect':assert row['independent_point_estimates']['default:comparison:H10-H00:pooled_auc']==-1
    else:
        assert row['numeric_NA_cells']==['H10']
        assert row['independent_point_estimates']['default:cell:H10:pooled_auc'] is None
    assert actual['accepted_for_selected_core'] is True
    assert actual['all_output_metrics_numerically_recomputed'] is False

@pytest.mark.parametrize('case',['codec','escape','empty_directory','staging'])
def test_deep_transport_or_namespace_tamper(tmp_path,case):
    output=fixture_transport(tmp_path)
    if case in ('codec','escape'):
        name='limitations.json';rp=output/'.receipts'/(hashlib.sha256(name.encode()).hexdigest()+'.json')
        record=json.loads(rp.read_text())
        if case=='codec':
            gz=output/'artifacts/limitations.json.bundle/metadata.json.gz';gz.write_bytes(b'private invalid gzip fixture')
            record['files'][str(gz.relative_to(output/'artifacts'))]=audit.sha(gz)
        else:record['files']={'../outside': '0'*64}
        rp.write_bytes(audit.typed(record))
        complete=json.loads((output/'complete.json').read_text());complete['receipts'][name]=audit.sha(rp)
        (output/'complete.json').write_bytes(audit.typed(complete))
    elif case=='empty_directory':(output/'artifacts/undeclared').mkdir()
    else:(output/'.staging/private.tmp').write_text('partial')
    with pytest.raises((ValueError,KeyError)):
        reader=audit.ReadOnlyArtifacts(output,audit.sha(output/'complete.json'),route='H')
        reader['limitations.json']

def test_saved_H37_transport_to_independent_8000_draws(tmp_path,h_values):
    output=fixture_transport(tmp_path,values=h_values)
    before={str(p):audit.sha(p) for p in output.rglob('*') if p.is_file()}
    reader=audit.ReadOnlyArtifacts(output,audit.sha(output/'complete.json'),route='H')
    result=audit.audit_H_artifacts(reader)
    reader.verify_all_decodable()
    assert result['accepted_for_selected_core'] is True
    assert len(reader.decoded_names)==37 and sum(x['draws_checked'] for x in result['cohorts'])==8000
    assert before=={str(p):audit.sha(p) for p in output.rglob('*') if p.is_file()}

def test_stdout_callback_is_inside_original_slot_lifetime(tmp_path,monkeypatch):
    from contextlib import contextmanager
    state={'held':False,'emitted':False}
    @contextmanager
    def fake_slot(root,slot):
        assert not state['held'];state['held']=True
        try:yield
        finally:state['held']=False
    class Reader(dict):
        route='H';complete_sha256='c'*64
        def verify_all_decodable(self):assert state['held']
        def check(self):assert state['held']
        def __len__(self):return 37
    reader=Reader();reader.pins=audit.Pins()
    request=dict(output=str(tmp_path),complete_sha256='c'*64,job=audit.job_spec('H','clip_vit_b16',7),consumer_dependencies_sha256={})
    monkeypatch.setattr(audit,'load_request',lambda *a:(request,audit.Pins()))
    monkeypatch.setattr(audit,'_actual_slot',fake_slot)
    monkeypatch.setattr(audit,'ReadOnlyArtifacts',lambda *a,**k:reader)
    monkeypatch.setattr(audit,'metadata_chain',lambda *a:dict(selection={},producer_peak_RSS_KiB=1))
    monkeypatch.setattr(audit,'audit_numeric',lambda *a:dict(accepted_for_selected_core=True,blockers_for_selected_core=[]))
    monkeypatch.setattr(audit.resource,'getrlimit',lambda *a:(32*1024**3,32*1024**3))
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):monkeypatch.setenv(name,'1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    def emit(report):
        assert state['held'] and report['fixed_SHA_rechecked_before_publication'] is True
        state['emitted']=True
    report=audit.execute(tmp_path/'private.json','0'*64,0,emit=emit)
    assert state=={'held':False,'emitted':True}
    assert report['stdout_publication_inside_shared_slot'] is True

@pytest.mark.parametrize('column',['video_id','rgb_sha256','source_ids','role'])
def test_saved_D_bootstrap_metadata_all_columns_are_admission_bound(column):
    manifest=pd.DataFrame(dict(video_uid=['fake/a'],label=[1],method_id=['fsgan'],group_id=['g0'],video_id=['collision0'],rgb_sha256=['rgb0'],source_ids=[['parent0']],role=['target_full']))
    saved=manifest.copy(deep=True);saved['raw_logit']=.5
    audit.bind_saved_frame_metadata(manifest,saved)
    saved.at[0,column]=['different_parent'] if column=='source_ids' else 'different'
    with pytest.raises(ValueError,match='metadata differs'):audit.bind_saved_frame_metadata(manifest,saved)

def test_MCORE001_fixed_D_SHA_rejects_before_exec_module(monkeypatch):
    from types import SimpleNamespace
    path=Path(audit.__file__).parent/'m_source_best_d_numeric.py'
    fixed='a7b079965fd34acbf2e974480aa197ded476cbc21f1447aeac523e1c4aab1dd0'
    original=audit.sha;state={'exec_module':False,'compute':False}
    monkeypatch.setattr(audit,'sha',lambda p:'f'*64 if Path(p)==path else original(p))
    def fake_compute(*args,**kwargs):
        state['compute']=True
        return dict(accepted_for_selected_core=True)
    def fake_exec(module):
        state['exec_module']=True;module.audit_D_artifacts=fake_compute
    fake_spec=SimpleNamespace(name='m_independent_D_numeric',loader=SimpleNamespace(exec_module=fake_exec))
    monkeypatch.setattr(audit.importlib.util,'spec_from_file_location',lambda *a,**k:fake_spec)
    monkeypatch.setattr(audit.importlib.util,'module_from_spec',lambda *a,**k:SimpleNamespace())
    monkeypatch.setitem(audit.sys.modules,'m_independent_D_numeric',SimpleNamespace())
    dependency_expected=audit.dependency_pins()[str(path)]
    error=None
    try:audit.audit_numeric(SimpleNamespace(route='D'),{})
    except ValueError as caught:error=caught
    assert error is not None and state=={'exec_module':False,'compute':False} and dependency_expected==fixed,dict(error=str(error),state=state,dependency_expected=dependency_expected)
    assert 'before import' in str(error)

def test_MCORE002_complete_head_file_SHA_rejects_private_header_padding(tmp_path,monkeypatch,h_values):
    request,reader=metadata_fixture(tmp_path,monkeypatch,h_values)
    path=next((tmp_path/'features').rglob('head_weight.npy'))
    marker=path.parent/'complete.json';marker_sha=audit.sha(marker)
    declared=json.loads(marker.read_text())['head']['weight_sha256']
    before=np.load(path,allow_pickle=False).tobytes()
    raw=bytearray(path.read_bytes());header_length=int.from_bytes(raw[8:10],'little');last_padding=10+header_length-2
    assert raw[last_padding]==32
    raw[last_padding]=9;path.write_bytes(raw)
    assert np.load(path,allow_pickle=False).tobytes()==before
    assert audit.sha(path)!=declared and audit.sha(marker)==marker_sha
    original_load=audit.np.load;entered=[]
    def guarded_load(p,*args,**kwargs):
        if Path(p)==path:entered.append(True)
        return original_load(p,*args,**kwargs)
    monkeypatch.setattr(audit.np,'load',guarded_load)
    with pytest.raises(ValueError,match='fixed SHA drift'):
        audit.metadata_chain(request,reader,audit.Pins())
    assert entered==[], 'Changed private head must be rejected before numeric np.load'


@pytest.mark.parametrize('route',['D','H'])
@pytest.mark.parametrize('seed',[7,42,123,2026,3407])
def test_actual_context_binds_seed_through_canonical_run_identity(route,seed):
    # Actual M_SOURCE_BEST_RAW_FP32_CACHE_V1 has no redundant top-level seed.
    job=audit.job_spec(route,'clip_vit_b16',seed)
    contexts={}
    disclosed={}
    for label,run in job['runs'].items():
        contexts[label]=dict(schema='M_SOURCE_BEST_RAW_FP32_CACHE_V1',checkpoint_selection='source_best',route=route,backbone=job['backbone'],run_id=run,parent_run_id=run,epochs=30,source_best_epoch=23,source_best_prefix=dict(epoch=23,attempts=18377,successful_updates=18374,AMP_skips=3))
        disclosed[label]=dict(seed=seed,epoch=23,attempts=18377,successful_updates=18374,AMP_skips=3)
    limitations=dict(checkpoint_selection='source_best',update_matched_primary_claim=False,source_only_selected_prefix=disclosed if route=='H' else disclosed['D'])
    limitations.update({'source_best_epochs':{h:23 for h in job['runs']}} if route=='H' else {'source_best_epoch':23})
    assert audit.check_selection(job,contexts,limitations)==disclosed
    for field,value in [('run_id','D_formal_clip_vit_b16_seed42'),('parent_run_id','OTHER'),('schema','OTHER'),('seed',False),('seed',seed+1)]:
        bad=copy.deepcopy(contexts);key=next(iter(bad))
        bad[key][field]=value if value!=bad[key].get(field) else 'OTHER'
        with pytest.raises(ValueError):audit.check_selection(job,bad,limitations)
    with_seed=copy.deepcopy(contexts)
    for c in with_seed.values():c['seed']=seed
    assert audit.check_selection(job,with_seed,limitations)==disclosed


@pytest.mark.parametrize('route,name',[('D','rq4/raw_mean_logit/source_cal_scores.csv'),('H','mean_logit/source_cal_B.csv')])
def test_saved_score_CSV_preserves_lexical_video_identity(tmp_path,route,name):
    ids=['004','010','000']
    q=pd.DataFrame(dict(video_uid=['v/'+v for v in ids],video_id=ids,label=[0,0,0],group_id=['g/'+v for v in ids],score=[.25,.5,.75],rank_score_decimal=['0.2500000000000000001','.5','.75']))
    values={n:({'synthetic':n} if kind=='json' else q) for n,kind in audit.expected_inventory(route).items()}
    output=fixture_transport(tmp_path,route=route,values=values)
    before={str(p):audit.sha(p) for p in output.rglob('*') if p.is_file()}
    reader=audit.ReadOnlyArtifacts(output,audit.sha(output/'complete.json'),route=route)
    actual=reader[name]
    assert actual.video_id.tolist()==ids
    assert all(type(v) is str for v in actual.video_id)
    assert actual.label.tolist()==[0,0,0] and actual.score.tolist()==[.25,.5,.75]
    assert actual.rank_score_decimal.tolist()==q.rank_score_decimal.tolist()
    assert before=={str(p):audit.sha(p) for p in output.rglob('*') if p.is_file()}
