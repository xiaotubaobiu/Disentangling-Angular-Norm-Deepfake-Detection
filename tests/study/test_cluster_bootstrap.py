# tests/study/test_cluster_bootstrap.py
import numpy as np, pandas as pd
from deepfake_detection.study.cluster_bootstrap import cluster_bootstrap_metric

def test_ci_covers_stat():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"cluster": np.repeat(range(50), 2),
                       "label": rng.integers(0,2,100),
                       "score": rng.normal(0,1,100)})
    def metric(sub):
        from sklearn.metrics import roc_auc_score
        try: return roc_auc_score(sub["label"], sub["score"])
        except: return np.nan
    out = cluster_bootstrap_metric(df, metric, cluster_col="cluster", n_iter=200, rng_seed=0)
    assert out["ci_low"] <= out["point"] <= out["ci_high"]
