import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

SCRIPT = Path(__file__).with_name('calculate.py')


def module():
    assert SCRIPT.exists(), 'Missing standalone grouping implementation'
    spec = importlib.util.spec_from_file_location('grouping_diagnostic', SCRIPT)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def frames():
    rows = []
    for method, video, label, z in [('real','r',0,0.),('a','x',1,2.),('b','x',1,-1.),('a','y',1,2.)]:
        for i in range(8):
            rows.append(dict(domain='cdf',method_id=method,video_id=video,label=label,
                sample_id=f'{method}/{video}/{i}',score=z,probability=float(expit(z))))
    return pd.DataFrame(rows)


def test_wrong_grouping_merges_methods_but_preserves_real_and_single_method_units():
    m = module(); d = frames()
    normal = m.aggregate(d, 'correct'); wrong = m.aggregate(d, 'wrong')
    assert len(normal) == 4 and len(wrong) == 3
    mixed = wrong.set_index('video_id').loc['x']
    assert mixed.n_frames == 16 and mixed.n_methods == 2
    assert mixed.score == pytest.approx(.5)
    assert mixed.probability == pytest.approx(.5748692496739388)
    assert mixed.probability != pytest.approx(float(expit(mixed.score)))
    assert wrong.set_index('video_id').loc['r','probability'] == .5
    assert wrong.set_index('video_id').loc['y','probability'] == normal[normal.video_id.eq('y')].probability.iloc[0]


def test_pooled_auc_is_over_actual_video_units_and_ties_receive_half_credit():
    m = module()
    normal = m.aggregate(frames(), 'correct'); wrong = m.aggregate(frames(), 'wrong')
    assert m.auroc(normal.label, normal.probability) == pytest.approx(2/3)
    assert m.auroc(wrong.label, wrong.probability) == 1.
    assert m.auroc([0,1], [.5,.5]) == .5
    assert m.rank_auc([0,1,1,1],[.5,.9,.2,.9]) == pytest.approx(2/3)
    assert m.rank_auc([0,1],[.5,.5]) == .5


@pytest.mark.parametrize('defect', ['short_video','duplicate_row','label_collision','nonfinite','mixed_domain'])
def test_rejects_input_changes_that_would_confound_the_grouping_comparison(defect):
    m = module(); d = frames()
    if defect == 'short_video': d = d.iloc[:-1].copy()
    if defect == 'duplicate_row': d = pd.concat([d,d.iloc[[0]]],ignore_index=True)
    if defect == 'label_collision': d.loc[d.method_id.eq('real'),'video_id'] = 'x'
    if defect == 'nonfinite': d.loc[0,'score'] = np.nan
    if defect == 'mixed_domain': d.loc[0,'domain'] = 'ff'
    with pytest.raises(ValueError): m.aggregate(d,'wrong')
