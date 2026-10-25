from pathlib import Path
import numpy as np,pandas as pd,pytest
from test_source_best_cache import project
from test_source_best_stats import Sink,h_data,d_data
from reliability_v2 import source_best_stats as m,analysis_d as io

class BadTableSink(Sink):
 def __init__(self,path,kind):super().__init__(path);self.kind=kind
 def __call__(self,name,kind,value):
  restored=super().__call__(name,kind,value);changed=False
  if name.endswith('comparisons.csv') and self.kind.startswith('H_'):
   field=self.kind[2:];restored.loc[0,field]=123.456 if field=='estimate' else 'unregistered';changed=True
  elif name.endswith('effects.csv') and self.kind.startswith('D_'):
   field=self.kind[2:];restored.loc[0,field]=not bool(restored.loc[0,field]) if field in ('valid','reversal') else 123.456;changed=True
  elif name.endswith('/calibration.csv') and self.kind.startswith('CAL_'):
   if self.kind=='CAL_ulp' and name!='rq4/raw_mean_logit/calibration.csv':return restored
   chosen=restored.calibration_kind.eq('target_real')
   if self.kind=='CAL_all':restored=restored[~chosen].copy()
   elif self.kind=='CAL_one':restored=restored.drop(restored[chosen].index[-1])
   elif self.kind=='CAL_budget':restored.loc[restored[chosen].index[0],'budget']=999
   elif self.kind=='CAL_ulp':
    at=restored[chosen].index[0];restored.loc[at,'threshold']=np.nextafter(float(restored.loc[at,'threshold']),-np.inf)
   else:restored.loc[restored[chosen].index[0],'status']='unregistered'
   changed=True
  if changed:
   io._csv(self.path/name,restored);restored=pd.read_csv(self.path/name,float_precision='round_trip');self.values[name]=restored
  return restored

@pytest.mark.parametrize('field',['estimate','comparison','threshold_kind','metric'])
def test_review_H_saved_comparison_corruption_refuses(tmp_path,monkeypatch,field):
 data,selected=h_data(tmp_path,monkeypatch)
 with pytest.raises(ValueError):m.h_driver(data,selected,BadTableSink(tmp_path/'corrupt','H_'+field))

@pytest.mark.parametrize('field',['delta_correct','interaction','sign_correct','reversal','valid'])
def test_review_D_saved_effect_corruption_refuses(project,tmp_path,field):
 data,selected,freeze=d_data(project)
 with pytest.raises(ValueError):m.d_driver(data,selected,freeze,BadTableSink(tmp_path/'corrupt','D_'+field))

def cal80(data):
 frame,h,w,b=data['target_cal'];frames=[];features=[]
 for number in range(80):
  q=frame.copy();uid='synthetic_cal80/'+str(number);q['video_uid']=uid;q['video_id']=uid;q['group_id']=uid
  q['rgb_sha256']=[uid+'/'+str(i) for i in range(len(q))];frames.append(q);features.append(h+np.float32(number*.01))
 data['target_cal']=(pd.concat(frames,ignore_index=True),np.concatenate(features),w,b)
 return data

@pytest.mark.parametrize('bad',['CAL_all','CAL_one','CAL_budget','CAL_status','CAL_ulp'])
def test_review_D_full_saved_CAL_inventory_required(project,tmp_path,bad):
 data,selected,freeze=d_data(project);cal80(data)
 with pytest.raises(ValueError):m.d_driver(data,selected,freeze,BadTableSink(tmp_path/'corrupt',bad))

@pytest.mark.parametrize('bad',['swap_cal','mislabel_domain','train_role'])
def test_review_D_tuple_role_domain_binding(project,tmp_path,bad):
 data,selected,freeze=d_data(project)
 if bad=='swap_cal':data['source_cal'],data['target_cal']=data['target_cal'],data['source_cal']
 elif bad=='mislabel_domain':data['source_cal'][0]['domain']='cdf'
 else:data['train'][0]['role']='source_test'
 with pytest.raises(ValueError):m.d_driver(data,selected,freeze,Sink(tmp_path/'bad_roles'))
