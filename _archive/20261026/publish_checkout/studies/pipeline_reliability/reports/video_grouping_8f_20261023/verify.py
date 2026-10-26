"""Independent persisted-result readback; no inference and no calculate imports."""
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

root=Path('/home/z/project/deepfake_detector')
out=Path(__file__).resolve().parent
proof=json.loads((out/'verification.json').read_text())
protocol=json.loads((out/'protocol.json').read_text())
bindings=json.loads((out/'input_bindings.json').read_text())
checks=0
for rel,want in proof['outputs_sha256'].items():
 assert hashlib.sha256((out/rel).read_bytes()).hexdigest()==want,rel
 checks+=1
for binding in bindings:
 for rel,want in binding['files'].items():
  assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==want,rel
  checks+=1
cells=pd.read_csv(out/'auc_cells.csv')
paired=pd.read_csv(out/'auc_by_seed.csv')
summary=pd.read_csv(out/'summary_by_backbone.csv')
assert len(cells)==240 and len(paired)==60 and len(summary)==12
max_auc_error=0.
for row in cells.itertuples(index=False):
 rid=f'{row.backbone}_seed{row.seed}'
 if row.grouping=='correct':
  suffix=('R' if row.feature=='Raw' else 'D')+('L' if row.aggregation=='mean_logit' else 'P')
  p=root/'artifacts/analysis'/rid/'scores'/row.domain/f'video_{suffix}.parquet'
  table=pd.read_parquet(p);score=table.score.to_numpy()
 else:
  p=out/'videos'/rid/row.domain/f"wrong_{'raw' if row.feature=='Raw' else 'direction'}.parquet"
  table=pd.read_parquet(p)
  assert table.n_frames.eq(8*table.n_methods).all()
  score=table['score' if row.aggregation=='mean_logit' else 'probability'].to_numpy()
 assert table.label.eq(0).sum()==row.real_units
 assert table.label.eq(1).sum()==row.fake_units
 error=abs(float(roc_auc_score(table.label,score))-row.pooled_AUC)
 max_auc_error=max(max_auc_error,error)
 assert error<1e-12
for row in paired.itertuples(index=False):
 group=cells[(cells.backbone==row.backbone)&(cells.seed==row.seed)&(cells.domain==row.domain)&(cells.aggregation==row.aggregation)]
 values=group.set_index(['feature','grouping']).pooled_AUC
 for field,key in [('correct_raw',('Raw','correct')),('wrong_raw',('Raw','wrong')),('correct_l2',('L2','correct')),('wrong_l2',('L2','wrong'))]:
  assert abs(getattr(row,field)-values[key])<1e-15
 dc=row.correct_l2-row.correct_raw;dw=row.wrong_l2-row.wrong_raw
 assert abs(row.delta_correct-dc)<1e-15 and abs(row.delta_wrong-dw)<1e-15
 assert abs(row.interaction-(dw-dc))<1e-15
 assert row.reversal==(dc < -1e-10 and dw > 1e-10)
for row in summary.itertuples(index=False):
 group=paired[(paired.backbone==row.backbone)&(paired.domain==row.domain)&(paired.aggregation==row.aggregation)]
 assert len(group)==5 and set(group.seed)==set(protocol['seeds'])
 assert group.reversal.sum()==row.reversal_count
 for col in ['correct_raw','correct_l2','wrong_raw','wrong_l2','delta_correct','delta_wrong','interaction']:
  assert abs(group[col].mean()-getattr(row,col+'_mean'))<1e-15
  assert abs(group[col].std(ddof=1)-getattr(row,col+'_SD'))<1e-15
src=hashlib.sha256()
for p in sorted((root/'src').rglob('*.py')):
 src.update(str(p.relative_to(root)).encode());src.update(p.read_bytes())
source_hash=src.hexdigest()
assert source_hash=='7d56c6e5803450369852b5865130a3ed815d87e3e6a91226d276f0fd092611b5'
result={'status':'INDEPENDENT_READBACK_PASS','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'hashes_checked':checks,'models':len(bindings),'AUC_cells_recomputed_from_saved_video_tables':len(cells),
 'paired_contrasts_checked':len(paired),'five_seed_summaries_checked':len(summary),
 'max_AUC_error':max_auc_error,'original_scientific_source_unchanged_sha256':source_hash,
 'no_training_inference_GPU':True}
(out/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
