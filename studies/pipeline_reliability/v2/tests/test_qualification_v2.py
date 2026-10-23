import pandas as pd
import pytest
from reliability_v2.qualify import source_probe_rows, assert_equal_tree


def test_probe_rejects_any_target_or_test_input():
    x=pd.DataFrame({'domain':['ff','cdf'],'role':['TRAIN','TARGET_TEST'],'label':[0,1]})
    with pytest.raises(ValueError,match='source-only'):source_probe_rows(x)


def test_probe_requires_class_balanced_source_inputs():
    x=pd.DataFrame({'domain':['ff']*128,'role':['TRAIN']*128,'label':[0]*64+[1]*64})
    assert len(source_probe_rows(x))==128
    with pytest.raises(ValueError,match='128'):source_probe_rows(x.head(127))


def test_recursive_state_compare_rejects_optimizer_change():
    import torch
    assert_equal_tree({'state':[{1:torch.tensor([1.,2.])}]},{'state':[{1:torch.tensor([1.,2.])}]})
    with pytest.raises(AssertionError):assert_equal_tree({'x':torch.tensor(1.)},{'x':torch.tensor(2.)})
