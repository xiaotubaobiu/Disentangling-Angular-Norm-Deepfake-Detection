from pathlib import Path
from datetime import datetime,timezone
import json,math,statistics,hashlib,os
import pandas as pd
from reliability_v2 import scoring_d as s
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
seeds=[7,42,123,2026,3407];frame=pd.DataFrame([dict(seed=k,aggregation='mean_logit',delta_correct=.1,delta_collision=(-.1 if k==7 else .1),interaction=(-.2 if k==7 else 0.),valid=True,reversal=k==7) for k in seeds])
full=s.summarize_seed_effects(frame).iloc[0].to_dict();assert full['complete_five_seeds'] and full['n_valid_seeds']==5 and full['n_seed_reversals']==1 and not full['mean_reversal']
assert math.isclose(full['delta_collision_mean'],.06,abs_tol=1e-12) and math.isclose(full['delta_collision_sample_sd'],statistics.stdev([-.1,.1,.1,.1,.1]),abs_tol=1e-12)
missing=s.summarize_seed_effects(frame.iloc[:-1]).iloc[0].to_dict();assert not missing['complete_five_seeds'] and missing['n_seeds']==4 and missing['n_valid_seeds']==4 and math.isfinite(missing['delta_correct_mean'])
invalid=frame.copy();invalid.loc[invalid.seed==3407,'valid']=False;invalid.loc[invalid.seed==3407,['delta_correct','delta_collision','interaction']]=float('nan');bad=s.summarize_seed_effects(invalid).iloc[0].to_dict();assert not bad['complete_five_seeds'] and bad['n_seeds']==5 and bad['n_valid_seeds']==4
try:s.summarize_seed_effects(pd.concat([frame,frame.iloc[:1]],ignore_index=True))
except ValueError as e:duplicate=str(e)
else:raise AssertionError('Duplicate seed not rejected')
receipt={'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS_SYNTHETIC_EXISTING_PRIMITIVE_INTERFACE_PROBE_ONLY','cases':{'complete_five_with_one_reversal_and_no_mean_reversal':full,'missing_seed_explicit_incomplete_but_descriptive_subset_mean_present':missing,'one_numeric_NA_seed_explicit_incomplete_nvalid4':bad,'duplicate_seed_rejected':duplicate},'source_module_sha256':hashlib.sha256(Path(s.__file__).read_bytes()).hexdigest(),'probe_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'final_D_five_seed_assembly_entry_implemented':False,'actual_trained_model_results_used':False,'production_modules_or_training_changed':False,'required_export_guard':'Check complete_five_seeds; do not label a valid-subset descriptive mean as the frozen five-seed result. Preserve allseed NA/identity/reason.'}
with (R/'reports/D_seed_summary_interface_probe_20261024T0730.json').open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False)
print(json.dumps(receipt,allow_nan=False))
