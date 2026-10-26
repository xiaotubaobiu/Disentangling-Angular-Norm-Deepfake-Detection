import unittest,math
try:
 import audit
except ModuleNotFoundError:
 audit=None
class IndependentSummaryTests(unittest.TestCase):
 def setUp(self):
  self.assertIsNotNone(audit,'Independent audit implementation missing')
 def test_sample_SD_uses_five_training_seeds_ddof_one(self):
  r=audit.seed_stats([(7,1),(42,2),(123,3),(2026,4),(3407,5)])
  self.assertEqual(r['estimate_mean'],3);self.assertAlmostEqual(r['estimate_sample_sd'],1.5811388300841898)
  self.assertEqual(r['n_valid_seeds'],5);self.assertTrue(r['all_estimates_available'])
 def test_missing_numeric_estimate_does_not_silently_drop_seed(self):
  r=audit.seed_stats([(7,1),(42,2),(123,None),(2026,4),(3407,5)])
  self.assertIsNone(r['estimate_mean']);self.assertIsNone(r['estimate_sample_sd']);self.assertEqual(r['n_valid_seeds'],4)
  self.assertTrue(r['complete_five_seeds']);self.assertFalse(r['all_estimates_available'])
 def test_missing_registered_seed_rejected(self):
  with self.assertRaises(ValueError):audit.seed_stats([(7,1),(42,2),(123,3),(2026,4)])
 def test_duplicate_registered_seed_rejected(self):
  with self.assertRaises(ValueError):audit.seed_stats([(7,1),(42,2),(123,3),(2026,4),(7,5)])
 def test_sign_tolerance_does_not_count_tiny_noise_as_effect(self):
  r=audit.seed_stats([(7,1e-11),(42,-1e-11),(123,0),(2026,1),(3407,-1)])
  self.assertEqual((r['n_positive'],r['n_negative'],r['n_zero_within_tolerance']),(1,1,3))
 def test_CAL_reduces_two_hundred_subsets_with_sample_SD(self):
  r=audit.subset_stats([0]*100+[1]*100,['ok']*200,200)
  self.assertEqual(r['estimate'],0.5);self.assertAlmostEqual(r['subset_sample_sd'],0.5012547071170855)
  self.assertEqual((r['subset_q025'],r['subset_q975']),(0,1));self.assertEqual(r['n_valid_subsets'],200)
 def test_unavailable_CAL_subset_keeps_whole_point_unavailable(self):
  r=audit.subset_stats([0]*200,['ok']*199+['unavailable'],200)
  self.assertIsNone(r['estimate']);self.assertIsNone(r['subset_sample_sd']);self.assertEqual(r['n_valid_subsets'],199)
 def test_source_operating_point_has_no_subset_spread(self):
  r=audit.subset_stats([0.25],['ok'],1)
  self.assertEqual(r['estimate'],0.25);self.assertIsNone(r['subset_sample_sd']);self.assertIsNone(r['subset_q025'])
 def test_row_checker_detects_wrong_SD_even_when_mean_matches(self):
  with self.assertRaises(AssertionError):audit.check_rows([{'key':'one','mean':3,'sd':1.4142135623730951}],[{'key':'one','mean':3,'sd':1.5811388300841898}],['key'])
 def test_row_checker_accepts_equivalent_CSV_numbers_and_booleans(self):
  self.assertEqual(audit.check_rows([{'key':'one','v':'3.0','ok':'True','na':''}],[{'key':'one','v':3,'ok':True,'na':None}],['key']),4)
 def test_row_checker_rejects_duplicate_stratum(self):
  with self.assertRaises(AssertionError):audit.check_rows([{'key':'one','v':1},{'key':'one','v':1}],[{'key':'one','v':1}],['key'])
if __name__=='__main__':unittest.main(verbosity=2)
