from pathlib import Path
import datetime,hashlib,json,collections
from reliability_v2 import source_best_policy as policy
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023');E=R/'engineering/source_best_M';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
registry_path=R/'H_run_registry.json';registry=json.loads(registry_path.read_text());assert registry['target_export_allowed'] is True
rows=[];bound={str(registry_path):sha(registry_path)}
for config_name,prefix in [('execution.json','D_formal_'),('H_execution.json','H_formal_')]:
 cp=R/'configs'/config_name;config=json.loads(cp.read_text());bound[str(cp)]=sha(cp)
 for run,task in config['runs'].items():
  if not run.startswith(prefix) or task['phase']!='formal':continue
  if task['route']=='H' and run not in registry['effective_runs'][task['backbone']]['run_ids']:continue
  folder=R/'runs'/run;hp=folder/'history.json';completep=folder/'complete.json';h=json.loads(hp.read_text());c=json.loads(completep.read_text());bound[str(hp)]=sha(hp);bound[str(completep)]=sha(completep)
  best=policy.select_source_best(h);assert len(h)==c['epochs']==task['epochs'] and best['epoch']==c['source_best_epoch'];assert c['identity']['config']==hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest()
  assert c['attempts']==sum(x['attempts'] for x in h) and c['success']==sum(x['success'] for x in h) and c['skip']==sum(x['skip'] for x in h)
  if task['route']=='H':assert all(x['draw_count']==task['draw_count'] and x['attempts']==task['draw_count']//128 for x in h)
  rows.append(dict(run_id=run,route=task['route'],backbone=task['backbone'],seed=task['seed'],history=task.get('history'),precision=task['precision'],registered_terminal_epochs=task['epochs'],source_best=best,recorded_source_best_sha256=c['source_best_sha256'],best_is_last_epoch=best['epoch']==task['epochs'],checkpoint_bytes_or_tensors_loaded_in_this_check=False))
assert len(rows)==45 and len({x['run_id'] for x in rows})==45
pairs=[]
for b in policy.BACKBONES:
 for seed in policy.SEEDS:
  pair={x['history']:x for x in rows if x['route']=='H' and x['backbone']==b and x['seed']==seed};assert set(pair)=={'B','AB'}
  pairs.append(dict(backbone=b,seed=seed,B_best_epoch=pair['B']['source_best']['epoch'],AB_best_epoch=pair['AB']['source_best']['epoch'],same_best_epoch=pair['B']['source_best']['epoch']==pair['AB']['source_best']['epoch'],equal_update_primary_claim_allowed=False))
for path,pin in bound.items():assert sha(path)==pin
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS_ACTUAL45_SOURCE_HISTORY_COMPLETION_SELECTION_CHECKS_METADATA_ONLY',models=45,rows=rows,H_pairs=pairs,H_pairs_with_different_best_epochs=sum(not x['same_best_epoch'] for x in pairs),best_last_epoch_metadata_candidates=sum(x['best_is_last_epoch'] for x in rows),reuse_capacity_credit_bytes=0,input_sha256=bound,source_policy_sha256=sha(Path(policy.__file__)),target_inference_authorized=False,tensor_head_file_context_qualified=False,scientific_sensitivity_results_generated=False,scope='Only actual source DEV histories and recorded formal completion metadata. No target scores, checkpoint deserialization/tensor/head validation or new inference. Best=earliest strict sourceDEV maximum; later sensitivity is descriptive and unequal epochs/updates are retained.')
(E/'actual45_source_history_selection_ledger.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['rows','H_pairs','input_sha256']},ensure_ascii=False))
