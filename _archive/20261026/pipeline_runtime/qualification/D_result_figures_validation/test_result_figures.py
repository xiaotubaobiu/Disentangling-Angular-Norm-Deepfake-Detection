"""Behavioral acceptance tests. Fixtures are synthetic and isolated."""
import csv, importlib.util, json, os, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from fixtures import make, rebind, csvwrite, BB, SEEDS, TABLES, sha
HERE=Path(__file__).parent
EXPORTER=HERE.parent/'export_D_result_figures.py'
class Figures(unittest.TestCase):
 def setUp(self):
  self.assertTrue(EXPORTER.is_file(),'remaining result figure exporter is not implemented')
  spec=importlib.util.spec_from_file_location('d_figures',EXPORTER);self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)
  (HERE/'work').mkdir(exist_ok=True);self.tmp=tempfile.TemporaryDirectory(dir=HERE/'work');self.root=make(Path(self.tmp.name))
 def tearDown(self):
  if hasattr(self,'tmp'):self.tmp.cleanup()
 def build(self):return self.m.build_packet(self.root)
 def change_csv(self,relative,change,bb=BB[0],seed=7):
  p=self.root/'analysis/D'/f'D_formal_{bb}_seed{seed}'/relative
  with p.open() as f:rows=list(csv.DictReader(f))
  change(rows);csvwrite(p,rows);rebind(self.root,bb,seed)
 def test_ties_exact_right_continuous(self):
  x,y=self.m.ecdf_coordinates([2,1,2]);self.assertEqual(list(x),[1.,2.]);self.assertEqual(list(y),[1/3,1.])
 def test_no_role_class_seed_pooling_or_rank_score_use(self):
  p=self.build();self.assertEqual(len(p['ecdf']),120)
  c=next(x for x in p['ecdf'] if x['backbone']==BB[0] and x['seed']==7 and x['role']=='source_test' and x['representation']=='raw' and x['label']==1)
  self.assertEqual(c['scores'],[2.,2.]);self.assertEqual(c['x'],[2.]);self.assertEqual(c['cdf'],[1.]);self.assertEqual(c['natural_video_uids'],['source_testf1','source_testf2'])
 def test_duplicate_natural_uid_rejected(self):
  self.change_csv('source_test/video_scores.csv',lambda r:r.append(r[0].copy()))
  with self.assertRaises(ValueError):self.build()
 def test_finite_score_required(self):
  self.change_csv('target_full/video_scores.csv',lambda r:r[0].update(score='nan'))
  with self.assertRaises(ValueError):self.build()
 def test_role_mixing_rejected(self):
  self.change_csv('source_test/video_scores.csv',lambda r:r[0].update(role='target_cal'))
  with self.assertRaises(ValueError):self.build()
 def test_missing_backbone_rejected_before_any_output(self):
  (self.root/'analysis/D'/BB[-1]/'five_seed_summary/complete.json').unlink()
  with self.assertRaises(ValueError):self.build()
  self.assertFalse((self.root/'preview').exists())
 def test_four_seed_summary_rejected(self):
  p=self.root/'analysis/D'/BB[0]/'five_seed_summary/complete.json';a=json.loads(p.read_text());a['identity']['seeds']=list(SEEDS[:-1]);p.write_text(json.dumps(a));rebind(self.root,BB[0])
  with self.assertRaises(ValueError):self.build()
 def test_missing_independent_acceptance_rejected(self):
  (self.root/'reports'/f'{BB[0]}_five_seed_summary_independent_acceptance_fixture.json').unlink()
  with self.assertRaises(ValueError):self.build()
 def test_tampered_completion_rejected(self):
  p=self.root/'analysis/D'/BB[0]/'five_seed_summary/complete.json';p.write_text(p.read_text()+' ')
  with self.assertRaises(ValueError):self.build()
 def test_tampered_source_rejected(self):
  p=self.root/'analysis/D'/f'D_formal_{BB[0]}_seed7/source_test/video_scores.csv';p.write_text(p.read_text()+' ')
  with self.assertRaises(ValueError):self.build()
 def test_missing_CAL_repeat_rejected(self):
  self.change_csv('rq4/raw_mean_logit/calibration.csv',lambda r:r.pop())
  with self.assertRaises(ValueError):self.build()
 def test_paired_CAL_ids_required(self):
  def modify(rows):
   r=rows[2];ids=json.loads(r['calibration_ids']);ids[0]='cal079' if 'cal079' not in ids else 'cal078';r['calibration_ids']=json.dumps(ids)
  self.change_csv('rq4/l2_mean_logit/calibration.csv',modify)
  with self.assertRaises(ValueError):self.build()
 def test_nested_CAL_ids_required(self):
  def modify(rows):
   r=next(r for r in rows if r['budget']=='20' and r['repeat']=='0');ids=json.loads(r['calibration_ids']);ids[0]='cal079' if 'cal079' not in ids else 'cal078';r['calibration_ids']=json.dumps(ids)
  self.change_csv('rq4/raw_mean_logit/calibration.csv',modify)
  with self.assertRaises(ValueError):self.build()
 def test_fixed_TEST_hash_required(self):
  self.change_csv('rq4/raw_mean_logit/calibration.csv',lambda r:r[2].update(test_id_hash='wrong'))
  with self.assertRaises(ValueError):self.build()
 def test_n80_unique_count_is_one(self):
  p=self.build();boxes=[r for r in p['calibration_boxes'] if r['budget']==80];self.assertEqual(len(boxes),30)
  self.assertTrue(all(r['unique_calibration_sets']==1 and r['model_count']==1 and r['CAL_repeats']==200 for r in boxes))
 def test_unique_count_label_cannot_override_data(self):
  self.change_csv('rq4/raw_mean_logit/calibration.csv',lambda r:r[-1].update(unique_set_count='200'))
  with self.assertRaises(ValueError):self.build()
 def test_unavailable_budget_preserves_NA_and_five_seeds(self):
  self.tmp.cleanup();self.tmp=tempfile.TemporaryDirectory(dir=HERE/'work');self.root=make(Path(self.tmp.name),pool=40);p=self.build()
  boxes=[r for r in p['calibration_boxes'] if r['budget']==80];self.assertEqual(len(boxes),30)
  self.assertTrue(all(r['status']=='unavailable' and r['metrics']['fpr']==[None]*200 and r['unique_calibration_sets']==0 for r in boxes))
 def test_summary_rows_NA_and_CI_references_unchanged(self):
  p=self.build()
  for name in TABLES:
   self.assertEqual(len(p['tables'][name]['rows']),15 if name.startswith('per_seed') else 3)
  r=p['tables']['per_seed_metrics']['rows'][0];self.assertEqual(r['estimate'],'');self.assertEqual(r['CI_reference'],'unchanged.json')
  self.assertNotIn('pooled_15_model_mean',p);self.assertIn('2000 TEST bootstrap',p['uncertainty_scope'])
 def test_changed_input_snapshot_rejected(self):
  p=self.build();cfg=self.root/'configs/execution.json';cfg.write_text(cfg.read_text()+' ')
  with self.assertRaises(ValueError):self.m.verify_inputs(p)
 def test_no_overwrite(self):
  out=self.root/'preview';out.mkdir();(out/'sentinel').write_text('keep')
  with self.assertRaises(ValueError):self.m.export_packet(self.root,out)
  self.assertEqual((out/'sentinel').read_text(),'keep')
 def test_escape_output_rejected(self):
  with self.assertRaises(ValueError):self.m.export_packet(self.root,self.root.parent/'outside')
 def test_symlink_input_rejected(self):
  p=self.root/'analysis/D'/BB[0]/'five_seed_summary/complete.json';target=self.root/'copy.json';target.write_bytes(p.read_bytes());p.unlink();p.symlink_to(target)
  with self.assertRaises(ValueError):self.build()
 def test_symlink_output_ancestor_rejected(self):
  link=self.root/'linked';link.symlink_to(self.root,target_is_directory=True)
  with self.assertRaises(ValueError):self.m.export_packet(self.root,link/'preview')
 def test_render_synthetic_five_figures_and_exact_data(self):
  out=self.root/'preview';receipt=self.m.export_packet(self.root,out)
  self.assertTrue(receipt['synthetic']);self.assertEqual(len(list(out.glob('*.png'))),5);self.assertEqual(len(list(out.glob('*.svg'))),5)
  self.assertEqual(len(list((out/'tables').glob('*.csv'))),12)
  data=json.loads((out/'plot_data.json').read_text());self.assertEqual(len(data['ecdf']),120)
  captions=(out/'captions_cn_en.md').read_text();self.assertIn('SYNTHETIC',captions);self.assertIn('校准',captions);self.assertIn('not TEST confidence intervals',captions)
  self.assertEqual(json.loads((out/'complete.json').read_text())['training_seed_count_per_backbone'],5)

 def test_partial_independent_table_audit_rejected(self):
  p=self.root/'reports'/f'{BB[0]}_five_seed_summary_independent_acceptance_fixture.json'
  a=json.loads(p.read_text());a['tables'].pop('five_seed_metrics');p.write_text(json.dumps(a))
  with self.assertRaises(ValueError):self.build()
 def test_available_CAL_cannot_be_marked_NA_to_hide_bad_values(self):
  self.change_csv('rq4/raw_mean_logit/calibration.csv',lambda r:r[2].update(status='unavailable',fpr='',tpr='',pooled_ba=''))
  with self.assertRaises(ValueError):self.build()
 def test_post_render_input_change_leaves_no_complete_receipt(self):
  original=self.m.render
  def mutate_after_real_render(packet,out):
   original(packet,out)
   p=self.root/'configs/execution.json';p.write_text(p.read_text()+' ')
  with patch.object(self.m,'render',side_effect=mutate_after_real_render):
   with self.assertRaises(ValueError):self.m.export_packet(self.root,self.root/'partial')
  self.assertFalse((self.root/'partial/complete.json').exists())


 def test_identical_CAL_set_must_have_identical_saved_operating_metrics(self):
  self.change_csv('rq4/raw_mean_logit/calibration.csv',lambda r:r[-1].update(tpr='0.9'))
  with self.assertRaises(ValueError):self.build()

if __name__=='__main__':unittest.main(verbosity=2)
