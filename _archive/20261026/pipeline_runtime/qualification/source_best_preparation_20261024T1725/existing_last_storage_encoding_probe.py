from pathlib import Path
import json,hashlib,numpy as np,copy,datetime
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023');root=Path(__file__).parent
source=R/'analysis/D/D_formal_clip_vit_b16_seed7/source_test/normal_bootstrap_raw_mean_logit.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before=sha(source);marker=json.loads((source.parent.parent/'complete.json').read_text());assert marker['artifact_sha256'][str(source.relative_to(source.parent.parent))]==before
original=json.loads(source.read_text());groups=original['source_groups'];draws=original['draws'];assert len(draws)==original['repeats']==2000 and len(groups)==70
assert all(set(d['group_draw_counts'])<=set(groups) and sum(d['group_draw_counts'].values())==70 and all(isinstance(v,int) and 0<=v<=70 for v in d['group_draw_counts'].values()) for d in draws)
counts=np.asarray([[d['group_draw_counts'].get(g,0) for g in groups] for d in draws],dtype=np.uint32)
presence=np.asarray([[g in d['group_draw_counts'] for g in groups] for d in draws],dtype=np.bool_)
arrays=root/'actual_2000x70_draw_counts.npz';np.savez_compressed(arrays,groups=np.asarray(groups),counts=counts,key_present=presence)
residual=copy.deepcopy(original)
for d in residual['draws']:del d['group_draw_counts']
payload=root/'original_remaining_fields.json';payload.write_text(json.dumps(residual,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
recovered=json.loads(payload.read_text())
with np.load(arrays,allow_pickle=False) as a:
 assert a['counts'].dtype==np.uint32 and a['counts'].shape==(2000,70)
 for row,values,mask in zip(recovered['draws'],a['counts'],a['key_present']):row['group_draw_counts']={str(g):int(v) for g,v,p in zip(a['groups'],values,mask) if p}
assert recovered==original and sha(source)==before
compact=arrays.stat().st_size+payload.stat().st_size
report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS_LOSSLESS_EXISTING_LAST_RESAMPLING_ENCODING_FEASIBILITY_ONLY','source':str(source),'source_sha256':before,'original_bytes':source.stat().st_size,'compressed_counts_and_all_remaining_fields_bytes':compact,'ratio_original_to_compact':source.stat().st_size/compact,'repeats_preserved':2000,'ordered_groups':70,'all_140000_group_counts_and_key_presence_verified':True,'all_metric_values_NA_flags_metadata_and_intervals_exactly_reconstructed':True,'source_bytes_unchanged':True,'no_secondary_checkpoint_inference_or_new_bootstrap':True,'source_best_experiment_started':False,'limitation':'One existing last SOURCE_TEST normal-bootstrap schema only. Full normal/closure/RQ4/H schema equivalence, shared-cohort identity and actual resource profiling are required before production adoption. No shared counts may cross nonidentical cohort/group/seed identities.'}
f=R/'reports/source_best_storage_encoding_feasibility_20261024T1725.json';assert not f.exists();f.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
