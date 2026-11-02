import pandas as pd
from deepfake_detector import evidence as e


def test_content_proof_distinguishes_bytes_rgb_and_unexamined_inputs():
    assert hasattr(e,'method_content_counts'), 'quantitative evidence not implemented'
    rows=[]
    for role,path,sha,rgb,ok in [('train','a','A','pixels',1),('test','b','B','pixels',1),('train','c','C','other',1),('test','d','C','other',1),('test','e',None,None,None)]:
        rows.append(dict(domain='ff',released_role=role,method_id='m',label=1,image_path=path,content_sha256=sha,rgb_sha256=rgb,decode_ok=ok,evaluation_unit='image',video_id=''))
    q=e.method_content_counts(pd.DataFrame(rows)).iloc[0]
    assert q.byte_cross_role_hashes==1
    assert q.RGB_cross_role_hashes==2
    assert q.RGB_cross_role_hashes_without_byte_copy==1
    assert q.pending_paths==1
    assert q.valid_train_images==2 and q.valid_test_images==2


def test_eight_frame_eligibility_uses_distinct_decoded_pixels_and_method_keys():
    assert hasattr(e,'method_content_counts'), 'quantitative evidence not implemented'
    rows=[]
    for method in ['a','b']:
        for n in range(9):
            rows.append(dict(domain='ff',released_role='train',method_id=method,label=1,image_path=f'{method}/{n}',content_sha256=str(n),rgb_sha256=str(n) if method=='a' else 'same',decode_ok=1,evaluation_unit='video_frame',video_id='same_id'))
    q=e.method_content_counts(pd.DataFrame(rows)).set_index('method_id')
    assert q.loc['a','eligible_train_videos']==1
    assert q.loc['b','eligible_train_videos']==0


def test_candidate_counts_do_not_let_discarded_mobile_test_copies_empty_train():
    assert hasattr(e,'candidate_counts'), 'unapproved isolated-count calculation missing'
    rows=[]
    for role,video in [('train','001_002'),('test','001_002'),('test','003_004')]:
        for n in range(8):
            rows.append(dict(domain='ff',released_role=role,method_id='mobileswap',method_type='FS',label=1,
              image_path=f'{role}/{video}/{n}',content_sha256=f'{video}-{n}',rgb_sha256=f'{video}-{n}',
              decode_ok=1,evaluation_unit='video_frame',video_id=video,storage_group_id=''))
    counts=e.candidate_counts(pd.DataFrame(rows),{'001':'train','002':'train','003':'test','004':'test'})
    q=counts.iloc[0]
    assert q.known_only_train_videos==1 and q.known_only_test_videos==1
    assert q.published_test_non_test_source_videos_excluded==1
    assert q.experiment_roles_changed==False


def test_cross_method_report_counts_hashes_not_basename_matches():
    assert hasattr(e,'cross_method_counts'), 'cross-method proven copy report missing'
    df=pd.DataFrame([
       dict(domain='ff',released_role='train',method_id='a',decode_ok=1,content_sha256='copy',rgb_sha256='pixels'),
       dict(domain='ff',released_role='train',method_id='a',decode_ok=1,content_sha256='differentA',rgb_sha256='sameRGB'),
       dict(domain='ff',released_role='train',method_id='b',decode_ok=1,content_sha256='copy',rgb_sha256='pixels'),
       dict(domain='ff',released_role='train',method_id='b',decode_ok=1,content_sha256='differentB',rgb_sha256='sameRGB')])
    q=e.cross_method_counts(df).iloc[0]
    assert q.shared_byte_hashes==1 and q.shared_RGB_hashes==2
    assert q.method_a=='a' and q.method_b=='b'
