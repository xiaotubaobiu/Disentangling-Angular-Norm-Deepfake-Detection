from pathlib import Path
from deepfake_detector import manifest as m


def test_efs_training_uses_train_copy_not_test_copy(tmp_path):
    assert hasattr(m, 'resolve_path'), 'role-aware resolver not implemented'
    p=tmp_path/'DF40_train/StyleGAN2/446/seed01.png'
    p.parent.mkdir(parents=True);p.write_bytes(b'input')
    result=m.resolve_path(tmp_path,'deepfakes_detection_datasets/DF40/StyleGAN2/ff/446/seed01.png','StyleGAN2','EFS','train',1)
    assert result==p


def test_fs_test_and_real_paths_preserve_method_and_role(tmp_path):
    assert hasattr(m, 'resolve_path'), 'role-aware resolver not implemented'
    p=tmp_path/'DF40_test/blendface/ff/frames/000_001/012.png'
    p.parent.mkdir(parents=True);p.write_bytes(b'input')
    assert m.resolve_path(tmp_path,'deepfakes_detection_datasets/DF40/blendface/ff/frames/000_001/012.png','blendface','FS','test',1)==p


def test_cdf_efs_layout_is_method_specific_not_a_global_replacement(tmp_path):
    a=tmp_path/'DF40_test/DiT/cdf/Fake_from_Celeb-real/id0_0000/11287.png'
    b=tmp_path/'DF40_test/sd2.1/cdf/Celeb-real/id0_0000/generated.png'
    for p in [a,b]:p.parent.mkdir(parents=True);p.write_bytes(b'image')
    assert m.resolve_path(tmp_path,'deepfakes_detection_datasets/DF40/DiT/cdf/Celeb-real/id0_0000/11287.png','DiT','EFS','test',1)==a
    assert m.resolve_path(tmp_path,'deepfakes_detection_datasets/DF40/sd2.1/cdf/Celeb-real/id0_0000/generated.png','sd2.1','EFS','test',1)==b


def test_same_video_id_does_not_mix_methods_and_numeric8():
    assert hasattr(m, 'select_eight'), 'numeric eight-frame selector not implemented'
    rows=[{'method_id':method,'video_id':'v','frame_number':i,'image_path':f'/{method}/{i}.png','content_sha256':f'{method}-{i}','decode_ok':True} for method in ['a','b'] for i in range(32)]
    selected=m.select_eight(rows)
    assert len(selected)==16
    assert {r['method_id'] for r in selected}=={'a','b'}
    assert [r['frame_number'] for r in selected if r['method_id']=='a']==[0,4,8,13,17,22,26,31]


def test_repeated_pixels_do_not_count_as_distinct_valid_frames():
    assert hasattr(m, 'select_eight'), 'numeric eight-frame selector not implemented'
    rows=[{'method_id':'a','video_id':'v','frame_number':i,'image_path':f'/{i}.png','content_sha256':'identical','decode_ok':True} for i in range(9)]
    assert m.select_eight(rows)==[]


def test_source_ids_require_full_tokens_and_never_use_efs_bucket_ids():
    assert hasattr(m,'source_ids_from_video'), 'source token parser not implemented'
    assert m.source_ids_from_video('ff','FR','00001_abnormal')==[]
    assert m.source_ids_from_video('ff','FR','103_audio_50001')==['103']
    assert m.source_ids_from_video('ff','FS','001_002')==['001','002']
    assert m.source_ids_from_video('ff','EFS','446')==[]


def test_audit_cache_rejects_changed_or_missing_files(tmp_path):
    assert hasattr(m,'cache_stat_matches'), 'cached successes are not checked'
    p=tmp_path/'x.png';p.write_bytes(b'original');s=p.stat()
    assert m.cache_stat_matches(p,s.st_size,s.st_mtime_ns)
    p.write_bytes(b'changed content')
    assert not m.cache_stat_matches(p,s.st_size,s.st_mtime_ns)
    p.unlink()
    assert not m.cache_stat_matches(p,s.st_size,s.st_mtime_ns)
