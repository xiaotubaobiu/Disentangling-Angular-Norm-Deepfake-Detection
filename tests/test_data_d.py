import json
import unittest
import numpy as np
import pandas as pd
from journal_detector.data_d import METHODS, pair_partition, video_uid, attach_components, assign_roles, apply_dispositions, canonical_training_pool, select_eight

class DataDTests(unittest.TestCase):
    def test_pair_partition_deterministic_and_exact(self):
        splits={"train":[[f"{i:03}",f"{i+360:03}"] for i in range(360)],"val":[["900","901"]],"test":[["950","951"]]}
        a,records=pair_partition(splits);b,_=pair_partition({**splits,"train":list(reversed(splits["train"]))})
        self.assertEqual(a,b);self.assertEqual(sum(r["role"]=="train" for r in records),306)
        self.assertEqual(sum(v=="dev" for v in a.values()),108)
        self.assertEqual(a["900"],"source_cal");self.assertEqual(a["950"],"source_test")
        for x,y in splits["train"]: self.assertEqual(a[x],a[y])
    def test_full_graph_unselected_bridge_exposes_leakage(self):
        q=pd.DataFrame([dict(domain="ff",method_id="fsgan",video_id="001_002",source_ids='["001","002"]',video_uid="a",rgb_sha256="shared",label=1,role="train"),dict(domain="cdf",method_id="fsgan",video_id="x",source_ids="[]",video_uid="b",rgb_sha256="shared",label=1,role="target_full")])
        q=attach_components(q,{"train":[["001","002"]]});self.assertEqual(q.group_id.nunique(),1)
        out=apply_dispositions(q);self.assertTrue(out.role.eq("excluded").all());self.assertTrue(out.exclusion_reason.str.contains("cross_role_component").all())
    def test_unknown_source_quarantine_and_method_identity(self):
        q=pd.DataFrame([dict(domain="ff",released_role="train",method_id="sadtalker",video_id="000##x",source_ids="[]",label=1),dict(domain="ff",released_role="train",method_id="e4s",video_id="001_002",source_ids='["001","002"]',label=1)])
        out=assign_roles(q,{"001":"train","002":"train"});self.assertEqual(out.iloc[0].role,"excluded");self.assertEqual(out.iloc[1].role,"train")
        self.assertIn("e4s",METHODS);self.assertNotIn("e4e",METHODS);self.assertNotIn("VQGAN",METHODS)
        self.assertNotEqual(video_uid("ff","e4s","001_002"),video_uid("ff","fsgan","001_002"))
    def test_canonical_weights_preserve_every_method(self):
        rows=[]
        for method in ["real"]+METHODS:
            for video in ["a","b"]:
                for frame in range(8): rows.append(dict(method_id=method,label=int(method!="real"),video_uid=f"{method}/{video}",video_id=video,image_path=f"/{method}/{video}/{frame}",rgb_sha256=(f"fake/{video}/{frame}" if method!="real" else f"real/{video}/{frame}")))
        pool,assoc=canonical_training_pool(pd.DataFrame(rows))
        self.assertAlmostEqual(pool.sampling_mass.sum(),1);self.assertAlmostEqual(pool[pool.label.eq(0)].sampling_mass.sum(),.5)
        self.assertAlmostEqual(assoc[assoc.method_id.eq("e4s")].sampling_contribution.sum(),.5/21)
        for rec in pool.itertuples(): self.assertAlmostEqual(sum(json.loads(rec.method_contributions).values()),rec.sampling_mass)
        members=json.loads(pool[pool.label.eq(1)].iloc[0].method_contributions);self.assertEqual(set(members),set(METHODS))
    def test_select_eight_numeric_distinct(self):
        q=pd.DataFrame([dict(video_uid="v",frame_number=i,image_path=f"/{i}.png",rgb_sha256=str(i//2),decode_ok=1) for i in range(20)])
        out=select_eight(q);self.assertEqual(len(out),8);self.assertEqual(out.rgb_sha256.nunique(),8);self.assertTrue(out.frame_number.is_monotonic_increasing)
    def test_label_conflict_component_quarantined(self):
        q=pd.DataFrame([dict(group_id="g",rgb_sha256="same",label=0,role="train",exclusion_reason=""),dict(group_id="g",rgb_sha256="same",label=1,role="train",exclusion_reason="")])
        out=apply_dispositions(q);self.assertTrue(out.role.eq("excluded").all());self.assertTrue(out.exclusion_reason.str.contains("RGB_label_conflict").all())

if __name__=="__main__":unittest.main()
