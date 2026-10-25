import copy,importlib,json
import numpy as np
import pytest

def api():
    try:return importlib.import_module('reliability_v2.multiplicity_codec')
    except ModuleNotFoundError:pytest.fail('Missing lossless multiplicity matrix codec')
def fixture():return {'source_groups':['unused','z','a'],'draws':[{'repeat':0,'group_draw_counts':{'z':0,'a':2},'metric':None,'precision':'1.00000000000000000001'},{'repeat':1,'group_draw_counts':{'a':1},'valid':False}],'nested':{'draws':[{'closure_draw_counts':{'c':1,'b':2},'reason':'missing method'}]}}

def test_nested_normal_closure_and_RQ4_fields_roundtrip_without_mutating_input():
    m=api();value=fixture();before=copy.deepcopy(value);b=m.encode_counts(value,max_cells=6);result=m.decode_counts(b,max_cells=6)
    assert result==before==value and list(result['draws'][0]['group_draw_counts'])==['z','a']
    assert result['source_groups']==['unused','z','a'] and result['draws'][0]['metric'] is None
    assert result['draws'][0]['precision']=='1.00000000000000000001' and type(result['draws'][1]['valid']) is bool

def test_present_zero_and_absent_key_are_not_conflated():
    m=api();value={'draws':[{'group_draw_counts':{'a':0}},{'group_draw_counts':{}}]};b=m.encode_counts(value,max_cells=2);r=m.decode_counts(b,max_cells=2)
    assert 'a' in r['draws'][0]['group_draw_counts'] and 'a' not in r['draws'][1]['group_draw_counts']

def test_all_2000_repeated_group_masses_are_preserved():
    m=api();row={f'g{i}':1 for i in range(70)};row['g0']=2;row['g1']=0;value={'repeats':2000,'draws':[{'repeat':i,'group_draw_counts':dict(row)} for i in range(2000)]};b=m.encode_counts(value,max_cells=140000)
    t=b['tables'][0];a=b['arrays'][t['counts_array']];assert a.dtype==np.uint32 and a.shape==(2000,70) and int(a.sum())==140000
    assert m.decode_counts(b,max_cells=140000)==value

def test_missing_count_field_and_empty_draws_remain_missing_or_empty():
    m=api();v={'draws':[{'group_draw_counts':{}},{'metric':None}],'other':{'draws':[]}};b=m.encode_counts(v,max_cells=1);r=m.decode_counts(b,max_cells=1);assert r==v and 'group_draw_counts' not in r['draws'][1]

def test_JSON_metadata_and_pickle_free_npz_roundtrip(tmp_path):
    m=api();v=fixture();b=m.encode_counts(v,max_cells=6);p=tmp_path/'a.npz';np.savez_compressed(p,**b['arrays']);meta=json.loads(json.dumps({k:v for k,v in b.items() if k!='arrays'}))
    with np.load(p,allow_pickle=False) as arrays:meta['arrays']={k:arrays[k].copy() for k in arrays.files}
    assert m.decode_counts(meta,max_cells=6)==v

@pytest.mark.parametrize('bad',[True,-1,1.5,2**32])
def test_invalid_multiplicities_cannot_be_narrowed_silently(bad):
    m=api()
    with pytest.raises(ValueError):m.encode_counts({'draws':[{'group_draw_counts':{'a':bad}}]},max_cells=10)

@pytest.mark.parametrize('key',[1,''])
def test_nonstring_or_empty_group_identity_is_rejected(key):
    m=api()
    with pytest.raises(ValueError):m.encode_counts({'draws':[{'group_draw_counts':{key:1}}]},max_cells=10)

@pytest.mark.parametrize('limit',[True,0,-1,1.5])
def test_invalid_cell_budget_is_rejected(limit):
    m=api()
    with pytest.raises(ValueError,match='budget'):m.encode_counts(fixture(),max_cells=limit)

def test_total_cell_budget_covers_all_tables_and_cannot_be_bypassed():
    m=api()
    with pytest.raises(ValueError,match='budget'):m.encode_counts(fixture(),max_cells=5)
    b=m.encode_counts(fixture(),max_cells=6)
    with pytest.raises(ValueError,match='budget'):m.decode_counts(b,max_cells=5)

def test_float_count_array_cannot_be_accepted_as_exact_integers():
    m=api();b=m.encode_counts(fixture(),max_cells=6);k=b['tables'][0]['counts_array'];b['arrays'][k]=b['arrays'][k].astype(float)
    with pytest.raises(ValueError):m.decode_counts(b,max_cells=6)

def test_duplicate_group_identity_cannot_alias_columns():
    m=api();b=m.encode_counts(fixture(),max_cells=6);b['tables'][0]['groups']=['a','a']
    with pytest.raises(ValueError):m.decode_counts(b,max_cells=6)

def test_duplicate_order_index_cannot_duplicate_a_draw_group():
    m=api();b=m.encode_counts(fixture(),max_cells=6);b['arrays'][b['tables'][0]['order_array']][0,:]=[0,0]
    with pytest.raises(ValueError):m.decode_counts(b,max_cells=6)

def test_order_indices_cannot_appear_after_missing_padding():
    m=api();b=m.encode_counts(fixture(),max_cells=6);b['arrays'][b['tables'][0]['order_array']][0,:]=[-1,1]
    with pytest.raises(ValueError):m.decode_counts(b,max_cells=6)

def test_counts_for_absent_keys_cannot_be_discarded():
    m=api();b=m.encode_counts(fixture(),max_cells=6);t=b['tables'][0];b['arrays'][t['counts_array']][1,1]=3
    with pytest.raises(ValueError):m.decode_counts(b,max_cells=6)

def test_corrupt_placeholder_path_cannot_overwrite_other_metadata():
    m=api();b=m.encode_counts(fixture(),max_cells=6);b['tables'][0]['list_path']=['source_groups']
    with pytest.raises(ValueError):m.decode_counts(b,max_cells=6)

def test_plain_JSON_without_counts_is_kept_exactly():
    m=api();v={'a':[None,False,{'precision':'0.00000000001'}]};b=m.encode_counts(v,max_cells=1);assert not b['tables'] and not b['arrays'];assert m.decode_counts(b,max_cells=1)==v
