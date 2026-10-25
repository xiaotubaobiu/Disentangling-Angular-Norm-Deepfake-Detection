import copy,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,pytest
from test_analysis_h import cache_fixture,driver_fixture
from test_source_best_cache import project,rgb,sha
from PIL import Image
from reliability_v2 import analysis_h as original,analysis_d as io,source_best_cache as cache,source_best_stats as m

class Sink:
 def __init__(self,path):self.path=path;self.values={};self.corrupt=False
 def __call__(self,name,kind,value):
  p=self.path/name;p.parent.mkdir(parents=True,exist_ok=True)
  if kind=='scores':io.write_scores(p,value);restored=io.read_scores(p)
  elif kind=='csv':io._csv(p,value);restored=pd.read_csv(p,float_precision='round_trip')
  elif kind=='json':io._save_json(p,value);restored=json.loads(p.read_text())
  elif kind=='array':np.save(p,value,allow_pickle=False);restored=np.load(p,allow_pickle=False)
  elif kind=='parquet':value.to_parquet(p,index=False);restored=pd.read_parquet(p)
  else:raise AssertionError(kind)
  if self.corrupt and name.endswith('metrics.csv'):restored.loc[0,'fpr']=.123456
  self.values[name]=restored;return restored

def select(route='H',history='B',head=None):
 return dict(route=route,history=history,seed=7,backbone='clip_vit_b16',checkpoint_selection='source_best',source_best_bytes_tensors_head_verified=True,registered_terminal_epochs=30,best=dict(epoch=1 if history=='B' else 2,attempts=10 if history=='B' else 20,successful_updates=10 if history=='B' else 20,AMP_skips=0),head_sha256=head)

@pytest.mark.parametrize('aggregation',['mean_logit','mean_probability'])
def test_native_H_score_pools_exactly_match_primary_native_cells(aggregation):
 full=cache_fixture();native={hist:{r:v for r,v in roles.items() if r in cache.ROLES['H']} for hist,roles in full.items()};expected=original.score_pools(full,aggregation=aggregation);actual=m.native_H_pools(native,aggregation=aggregation)
 for hist in ('B','AB'):pd.testing.assert_frame_equal(actual['source_cal'][hist],expected['source_cal'][hist])
 for role in ('SOURCE_TEST','TARGET_TEST'):
  assert set(actual[role])=={'H00','H10'}
  for cell in actual[role]:pd.testing.assert_frame_equal(actual[role][cell],expected[role]['all_B'][cell])

@pytest.mark.parametrize('mutation',['common','role','domain','pixels','head','cal_fake'])
def test_H_native_scope_and_same_input_guard(mutation):
 data=cache_fixture();native={hist:{r:v for r,v in roles.items() if r in cache.ROLES['H']} for hist,roles in data.items()}
 if mutation=='common':native['B']['common/B/TARGET_TEST']=data['B']['common/B/TARGET_TEST']
 elif mutation=='head':native['B']['B/TARGET_TEST'][2][1,0]=2
 elif mutation=='cal_fake':native['B']['B/SOURCE_CAL'][0].loc[0,'label']=1
 else:
  field={'role':'role','domain':'domain','pixels':'rgb_sha256'}[mutation];native['AB']['B/TARGET_TEST'][0].loc[0,field]='changed'
 with pytest.raises(ValueError):m.native_H_pools(native)

def h_data(tmp_path,monkeypatch):
 module,path,_=driver_fixture(tmp_path,monkeypatch)
 from reliability_v2 import cache_h
 data={hist:{role:cache_h.load_features(tmp_path/'features/H'/f'H_formal_clip_vit_b16_{hist}_seed7'/role,expected_run_id=f'H_formal_clip_vit_b16_{hist}_seed7',expected_role=role,config_path=path) for role in cache.ROLES['H']} for hist in ('B','AB')}
 selections={hist:select(history=hist,head=cache._head_hash(data[hist]['B/SOURCE_CAL'][2],data[hist]['B/SOURCE_CAL'][3],2)) for hist in data};return data,selections

def test_H_driver_real_saved_score_checks_and_2000_budget_no_common(tmp_path,monkeypatch):
 data,selections=h_data(tmp_path,monkeypatch);sink=Sink(tmp_path/'M');result=m.h_driver(data,selections,sink)
 boot=[x for name,x in sink.values.items() if name.endswith('conditional_bootstrap.json')];assert len(boot)==4 and all(x['repeats']==2000 for x in boot)
 assert not any('common' in name or 'H01' in name or 'H11' in name for name in sink.values)
 assert result['update_matched_primary_claim'] is False and result['source_best_epochs']=={'B':1,'AB':2}
 assert len([n for n in sink.values if n.endswith('independent_saved_score_checks.json')])==4

def test_H_driver_saved_metric_corruption_refused(tmp_path,monkeypatch):
 data,selections=h_data(tmp_path,monkeypatch);sink=Sink(tmp_path/'M');sink.corrupt=True
 with pytest.raises(ValueError,match='independent'):m.h_driver(data,selections,sink)

def test_no_development_or_target_selected_checkpoint_allowed():
 bad=select();bad['checkpoint_selection']='target_best'
 with pytest.raises(ValueError):m.selection(bad,'H')
 bad=select();bad['seed']=99
 with pytest.raises(ValueError):m.selection(bad,'H')

def d_data(project):
 if project['route']!='D':pytest.skip('D fixture only')
 data={r:(df,np.column_stack([np.arange(len(df),dtype=np.float32)/100,*[np.ones(len(df),np.float32)]*511]),project['model'].classifier.weight.detach().numpy(),project['model'].classifier.bias.detach().numpy()) for r,df in project['frames'].items()}
 for role,(frame,h,w,b) in data.items():
  frame['video_id']=frame.video_uid
  frame['domain']='ff' if role in ('train','source_cal','source_test') else 'cdf'
  for i in range(len(frame)):
   path=project['root']/'synthetic_statistic_fixture'/role/(str(i)+'.png');path.parent.mkdir(parents=True,exist_ok=True)
   Image.new('RGB',(7,9),(int(frame.iloc[i].label)*130+i%8,30,40)).save(path)
   frame.loc[i,'image_path']=str(path);frame.loc[i,'content_sha256']=sha(path);frame.loc[i,'rgb_sha256']=rgb(path)
   frame.loc[i,'bytes']=path.stat().st_size;frame.loc[i,'mtime_ns']=path.stat().st_mtime_ns
 data['train'][0]['sampling_mass']=1.0/len(data['train'][0])
 freeze=dict(status='FROZEN',method_count=21,method_registry={'FS9':['fsgan','faceswap','simswap','inswap','blendface','uniface','mobileswap','e4s','facedancer'],'FR12':['fomm','facevid2vid','wav2lip','MRAA','one_shot_free','pirender','tpsm','lia','danet','sadtalker','mcnet','hyperreenact']},target_scores_used=False,cdf_calibration_scope={'scope':'SYNTHETIC_SMALL_UNAVAILABLE_CAL_BUDGET_TEST'})
 return data,select('D',history=None,head=project['source']['head_sha256']),freeze

def test_D_driver_retains_all_score_cells_controls_and_NA_budgets(project,tmp_path):
 data,selection,freeze=d_data(project);sink=Sink(tmp_path/'M');result=m.d_driver(data,selection,freeze,sink)
 assert len([n for n in sink.values if 'normal_bootstrap' in n])==8 and len([n for n in sink.values if 'closure_bootstrap' in n])==2
 assert len([n for n in sink.values if n.endswith('calibration.csv')])==4 and len([n for n in sink.values if n.endswith('primary_n20_vs_source_bootstrap.json')])==4
 assert all(x['status']=='UNAVAILABLE_CALIBRATION_BUDGET' for n,x in sink.values.items() if n.endswith('primary_n20_vs_source_bootstrap.json'))
 assert not any('dev' in n or 'oracle' in n or 'common' in n for n in sink.values)
 assert sink.values['limitations.json']['calibration_repeats']==200 and sink.values['limitations.json']['test_bootstrap_repeats']==2000
 assert result['update_matched_primary_claim'] is False

def test_D_radius_reference_never_reads_target_features(project,tmp_path):
 data,selection,freeze=d_data(project);first=Sink(tmp_path/'first');m.d_driver(data,selection,freeze,first)
 changed=dict(data)
 for role in ('target_full','target_cal','target_test'):
  frame,h,w,b=changed[role];changed[role]=(frame,h*7,w,b)
 second=Sink(tmp_path/'second');m.d_driver(changed,selection,freeze,second)
 assert np.array_equal(first.values['training_radii.npy'],second.values['training_radii.npy'])


def test_H_native_driver_retains_every_actual_2000_draw(tmp_path):
 template=cache_fixture();data={history:{} for history in ('B','AB')}
 for history in data:
  for role in cache.ROLES['H']:
   proto=template[history][role][0].iloc[0].to_dict();leaf=role.split('/')[-1];domain='target' if leaf=='TARGET_TEST' else 'source'
   methods=['real'] if leaf=='SOURCE_CAL' else ['real','DeepFakeDetection'] if domain=='target' else ['real',*original.SOURCE_METHODS];rows=[];features=[]
   for method in methods:
    for video in range(30):
     uid=f'{domain}/{leaf}/{method}/{video}'
     for slot in range(8):
      mother=f'{uid}/{slot}';label=int(method!='real');row=dict(proto,video_uid=uid,video_id=uid,group_id=f'{domain}/{leaf}/g{video}',label=label,method_id=method,mother_id=mother,sample_id=mother,image_path=mother+'.png',rgb_sha256=mother+'/rgb',mother_rgb_sha256=mother+'/mother',content_sha256=mother+'/bytes',role=leaf,domain=domain,processing_version='B');rows.append(row)
      features.append([label*2-1+video*.01+slot*.001+(history=='AB')*.2,1.])
   w=np.array([[0.,0.],[1.,0.]],np.float32);b=np.zeros(2,np.float32);data[history][role]=(pd.DataFrame(rows),np.array(features,np.float32),w,b)
 selected={h:select(history=h,head=cache._head_hash(data[h]['B/SOURCE_CAL'][2],data[h]['B/SOURCE_CAL'][3],2)) for h in data};sink=Sink(tmp_path/'M_actual_2000');m.h_driver(data,selected,sink)
 for name,payload in sink.values.items():
  if not name.endswith('conditional_bootstrap.json'):continue
  assert payload['repeats']==2000 and len(payload['draws'])==2000 and payload['source_cal_resampled'] is False
  assert [row['repeat'] for row in payload['draws']]==list(range(2000))
  assert all(sum(row['group_draw_counts'].values())==30 for row in payload['draws'])

def test_D_all_applicable_CAL_budgets_keep_200_subsets(project,tmp_path):
 data,selected,freeze=d_data(project);original_frame,original_h,w,b=data['target_cal'];frames=[];features=[]
 for video in range(80):
  frame=original_frame.copy();uid='synthetic_target_cal/real/'+str(video);frame['video_uid']=uid;frame['video_id']=uid;frame['group_id']=uid
  frame['rgb_sha256']=[f'{uid}/rgb/{n}' for n in range(len(frame))]
  features.append(original_h+np.float32(video*.01));frames.append(frame)
 data['target_cal']=(pd.concat(frames,ignore_index=True),np.concatenate(features),w,b);sink=Sink(tmp_path/'M_cal_200');m.d_driver(data,selected,freeze,sink)
 for name,table in sink.values.items():
  if not name.endswith('/calibration.csv'):continue
  for budget in (10,20,40,80):
   rows=table[table.calibration_kind.eq('target_real') & table.budget.eq(budget)]
   assert len(rows)==200 and rows.status.eq('ok').all()
