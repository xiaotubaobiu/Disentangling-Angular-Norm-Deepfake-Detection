"""Independent stdlib auditor tests; saved values are hand-checkable fixtures."""
import importlib.util
import unittest
from pathlib import Path

SCRIPT=Path(__file__).with_name('audit_production_packet.py')

class Auditor(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.is_file(), 'independent production packet auditor is not implemented')
        spec=importlib.util.spec_from_file_location('independent_D_packet_audit', SCRIPT)
        self.m=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def curve_fixture(self):
        rows=[dict(role='source_test',grouping='correct',representation='raw',aggregation='mean_logit',
                   label='1',video_uid=uid,member_video_uids='["'+uid+'"]',score=score)
              for uid,score in [('a','2'),('b','1'),('c','2')]]
        curve=dict(role='source_test',grouping='correct',representation='raw',aggregation='mean_logit',label=1,
                   score_field='score',score_domain='saved_FP64_mean_logit_margin',
                   natural_video_uids=['a','b','c'],score_strings=['2','1','2'],scores=[2.,1.,2.],x=[1.,2.],cdf=[1/3,1.])
        return rows,curve

    def test_tied_ecdf_uses_independent_bisect_cumulative(self):
        rows,curve=self.curve_fixture()
        self.assertEqual(self.m.check_ecdf_curve(curve,rows),3)

    def test_one_corrupted_ecdf_coordinate_rejected(self):
        rows,curve=self.curve_fixture();curve['cdf'][0]=0.5
        with self.assertRaisesRegex(ValueError,'ECDF cumulative'):
            self.m.check_ecdf_curve(curve,rows)

    def test_duplicate_uid_rejected_even_when_plot_matches(self):
        rows,curve=self.curve_fixture();rows[1]['video_uid']='a'
        with self.assertRaisesRegex(ValueError,'natural video'):
            self.m.check_ecdf_curve(curve,rows)

if __name__=='__main__':
    unittest.main(verbosity=2)
