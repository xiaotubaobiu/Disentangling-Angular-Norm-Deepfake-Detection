import hashlib
from deepfake_detector import prepare as p


def test_weight_hash_validation_rejects_wrong_existing_file(tmp_path):
    assert hasattr(p,'validate_weight'), 'initialization hash gate not implemented'
    f=tmp_path/'weights.pt';f.write_bytes(b'public-data')
    h=hashlib.sha256(b'public-data').hexdigest()
    assert p.validate_weight(f,h)==h
    try:p.validate_weight(f,'00000000')
    except ValueError:pass
    else:raise AssertionError('Wrong existing initialization must not be silently accepted')


def test_source_split_annotations_do_not_assign_experiment_roles():
    assert hasattr(p,'source_split_roles'), 'source metadata annotation not implemented'
    roles=p.source_split_roles({'train':[['001','002']],'val':[['003','004']],'test':[['005','006']]})
    assert roles=={'001':'train','002':'train','003':'val','004':'val','005':'test','006':'test'}
    try:p.source_split_roles({'train':[['001','002']],'val':[],'test':[['001','003']]})
    except ValueError:pass
    else:raise AssertionError('Conflicting official source metadata must stop annotation')
