import copy,importlib
import pytest

def api():return importlib.import_module('reliability_v2.multiplicity_codec')

@pytest.mark.parametrize('mapping',[{'a':1},{}])
def test_omitted_count_table_cannot_leave_a_null_placeholder(mapping):
    m=api();value={'draws':[{'group_draw_counts':mapping}]};b=m.encode_counts(value,max_cells=2);b['tables']=[];b['arrays']={}
    with pytest.raises(ValueError,match='placeholder|coverage'):m.decode_counts(b,max_cells=2)

def test_truncated_matching_table_rows_cannot_drop_an_original_draw():
    m=api();v={'draws':[{'group_draw_counts':{'a':1}},{'group_draw_counts':{'b':2}}]};b=m.encode_counts(v,max_cells=4);t=b['tables'][0];t['row_indices']=t['row_indices'][:1]
    for key in ['counts_array','order_array']:b['arrays'][t[key]]=b['arrays'][t[key]][:1].copy()
    with pytest.raises(ValueError,match='placeholder|coverage'):m.decode_counts(b,max_cells=4)
