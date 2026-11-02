"""New H history behavior; detector doubles isolate expensive external inference."""
import importlib
from pathlib import Path

import cv2
import numpy as np
import pytest


def geometry():
    return importlib.import_module('journal_detector.geometry_h')


def shape68(size=160):
    # A nondegenerate, rotated face-shaped configuration for dlib chip geometry.
    angle = np.linspace(0, 2*np.pi, 68, endpoint=False)
    points = np.c_[size*(.5+.24*np.cos(angle)), size*(.5+.30*np.sin(angle))]
    points[17:27] = np.c_[np.linspace(.25,.75,10)*size, np.full(10,.35*size)]
    points[27:36] = np.c_[np.linspace(.45,.55,9)*size, np.linspace(.4,.65,9)*size]
    points[36:42] = [[.28*size,.43*size],[.31*size,.41*size],[.35*size,.42*size],
                     [.38*size,.44*size],[.34*size,.46*size],[.30*size,.45*size]]
    points[42:48] = points[36:42] + [size*.29, 0]
    points[48:68] = np.c_[size*(.5+.14*np.cos(angle[:20]*3.4)),
                            size*(.72+.07*np.sin(angle[:20]*3.4))]
    return points


class FivePoint:
    def detect(self, rgb):
        g = geometry()
        h,w = rgb.shape[:2]
        points = np.array([[.34,.4],[.66,.4],[.5,.58],[.37,.75],[.63,.75]]) * [w,h]
        return [g.Face(np.array([.15*w,.1*h,.85*w,.9*h]), .99, points)]


class SixtyEightPoint:
    def __init__(self, reject_a=False):
        self.reject_a = reject_a

    def detect(self, rgb):
        g = geometry()
        h,w = rgb.shape[:2]
        if self.reject_a and (h,w) == (336,336):
            return []
        points = shape68(min(h,w))
        return [g.Face(np.array([.1*w,.1*h,.9*w,.9*h]), .9, points)]


def test_mothers_are_selected_before_detection_and_rgb_deduplicated():
    g = geometry()
    frames = [g.MotherFrame(i,i/25,np.full((8,8,3),i//2,np.uint8),str(i//2)) for i in range(20)]
    selected = g.select_mothers(list(reversed(frames)))
    assert [m.index for m in selected] == [0,2,6,8,10,12,16,18]
    assert len({m.rgb_sha256 for m in selected}) == 8
    with pytest.raises(g.GeometryFailure, match='insufficient_unique_mothers'):
        g.select_mothers(frames[:12])
    assert g.mother_id('FFPP/source/real/001',selected[0]) != g.mother_id('FFPP/source/real/002',selected[0])


def test_warp_mask_tracks_padding_and_inherited_holes():
    g = geometry()
    rgb = np.full((4,4,3), 31, np.uint8)
    mask = np.full((4,4),255,np.uint8); mask[1,0] = 0
    crop = g.warp(rgb,np.array([[1,0,2],[0,1,0]],float),4,mask)
    assert crop.valid_mask.tolist() == [[0,0,255,255],[0,0,0,255],[0,0,255,255],[0,0,255,255]]
    assert crop.rgb[0,:,0].tolist() == [0,0,31,31]


def test_a336_and_b224_and_ab_read_saved_a_with_new_68_landmarks(tmp_path):
    g = geometry()
    rng = np.random.default_rng(7)
    rgb = rng.integers(0,256,(160,160,3),dtype=np.uint8)
    mother = g.MotherFrame(14,.56,rgb,g.rgb_hash(rgb))
    result = g.process_frame(mother,FivePoint(),SixtyEightPoint(),tmp_path)
    assert all(result[v]['ok'] for v in ('A','B','AB'))
    assert g.read_rgb(result['A']['path']).shape == (336,336,3)
    assert g.read_rgb(result['B']['path']).shape == (224,224,3)
    assert g.read_rgb(result['AB']['path']).shape == (224,224,3)
    assert result['AB']['input_rgb_sha256'] == g.rgb_hash(g.read_rgb(tmp_path/'A.png'))
    assert result['AB']['parent_byte_sha256'] == g.sha256_file(tmp_path/'A.png')
    assert len(result['B']['detection']['landmarks']) == 68
    assert len(result['AB']['detection']['landmarks']) == 68
    assert result['B']['detection']['landmarks'] != result['AB']['detection']['landmarks']
    assert abs(result['B']['matrix'][0][1]) > .01
    a_mask = cv2.imread(str(tmp_path/'A_mask.png'),cv2.IMREAD_GRAYSCALE)
    matrix = np.asarray(result['AB']['matrix'])
    expected = cv2.warpAffine(a_mask,matrix,(224,224),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
    assert np.array_equal(expected,cv2.imread(str(tmp_path/'AB_mask.png'),cv2.IMREAD_GRAYSCALE))


def test_failed_b_redetection_never_reuses_original_coordinates(tmp_path):
    g = geometry()
    rgb = np.random.default_rng(3).integers(0,255,(160,160,3),dtype=np.uint8)
    result = g.process_frame(g.MotherFrame(1,.04,rgb,g.rgb_hash(rgb)),FivePoint(),SixtyEightPoint(True),tmp_path)
    assert result['B']['ok'] and result['A']['ok']
    assert result['AB'] == {'ok':False,'stage':'B_redetection','reason':'no_qualified_face'}
    assert not (tmp_path/'AB.png').exists()


def test_b_independent_of_a_failure_and_invalid_68_landmarks_rejected(tmp_path):
    g = geometry()
    class NoFace:
        def detect(self,rgb): return []
    rgb = np.random.default_rng(1).integers(0,255,(160,160,3),dtype=np.uint8)
    result = g.process_frame(g.MotherFrame(1,.04,rgb,g.rgb_hash(rgb)),NoFace(),SixtyEightPoint(),tmp_path)
    assert result['B']['ok'] and not result['A']['ok'] and not result['AB']['ok']
    with pytest.raises(g.GeometryFailure,match='68'):
        g.b_alignment_matrix(g.Face(np.array([0,0,40,40]),1,np.ones((5,2))))


def test_final_rgb_duplicates_reject_a_video_without_replacement_mothers():
    g = geometry()
    frames = [{v:{'ok':True,'rgb_sha256':str(i if v!='AB' else 0)} for v in ('A','B','AB')} for i in range(8)]
    eligible, failures = g.video_eligibility(frames)
    assert eligible == {'A':True,'B':True,'AB':False}
    assert failures['AB'] == ['insufficient_unique_final_rgb']


def test_per_video_resume_reuses_exact_mothers_and_rejects_corrupt_png(tmp_path):
    g = geometry()
    video=tmp_path/'sample.avi'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),8.,(160,160))
    rng=np.random.default_rng(21)
    for i in range(16): writer.write(rng.integers(0,255,(160,160,3),dtype=np.uint8))
    writer.release()
    record={'video_uid':'FFPP/source/real/001','dataset':'FFPP','method_id':'real',
            'video_id':'001','role':'train','path':str(video)}
    first=g.process_video(record,FivePoint(),SixtyEightPoint(),tmp_path/'artifacts','test-protocol')
    class MustNotDetect:
        def detect(self,rgb): raise AssertionError('completed video was processed again')
    second=g.process_video(record,MustNotDetect(),MustNotDetect(),tmp_path/'artifacts','test-protocol')
    assert first==second
    assert len(first['frames'])==8 and first['common_B_AB']
    assert all(f['mother_id'].startswith('FFPP/source/real/001/mother/') for f in first['frames'])
    Path(first['frames'][0]['B']['path']).write_bytes(b'corrupt')
    with pytest.raises(g.GeometryFailure,match='persisted_artifact_changed'):
        g.process_video(record,MustNotDetect(),MustNotDetect(),tmp_path/'artifacts','test-protocol')


def test_interrupted_png_write_preserves_the_previous_complete_image(tmp_path,monkeypatch):
    # Operation-level file atomicity must preserve a completed mother if a
    # process stops while a replacement is being encoded.
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'operations/h_png_io.py'
    spec=importlib.util.spec_from_file_location('h_png_io_test',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    image=tmp_path/'mother.png'; cv2.imwrite(str(image),np.full((8,8,3),31,np.uint8))
    original=image.read_bytes()
    def interrupted(path,image,params):
        Path(path).write_bytes(b'partial encoding')
        raise RuntimeError('interrupted encoder')
    monkeypatch.setattr(module,'_native_imwrite',interrupted)
    with pytest.raises(RuntimeError,match='interrupted encoder'):
        module.atomic_imwrite(str(image),np.full((8,8,3),44,np.uint8),[])
    assert image.read_bytes()==original
    assert cv2.imread(str(image))[0,0].tolist()==[31,31,31]


def test_interrupted_mother_rewrite_resumes_same_frozen_mothers(tmp_path,monkeypatch):
    import importlib.util
    import json
    g=geometry()
    module_path=Path(__file__).resolve().parents[1]/'operations/h_png_io.py'
    spec=importlib.util.spec_from_file_location('h_png_io_resume_test',module_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    native=module._native_imwrite
    video=tmp_path/'sample.avi'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),8.,(160,160))
    rng=np.random.default_rng(58)
    for i in range(16):writer.write(rng.integers(0,255,(160,160,3),dtype=np.uint8))
    writer.release()
    record={'video_uid':'FFPP/source/real/001','dataset':'FFPP','method_id':'real',
            'video_id':'001','role':'train','path':str(video)}
    counter={'mother':0}
    def interrupt_ninth_mother(path,image,params):
        if 'mother' in Path(path).name:
            counter['mother']+=1
            if counter['mother']==9:
                Path(path).write_bytes(b'interrupted mother encoding')
                raise SystemExit('simulated process stop')
        return native(path,image,params)
    monkeypatch.setattr(module,'_native_imwrite',interrupt_ninth_mother)
    monkeypatch.setattr(g.cv2,'imwrite',module.atomic_imwrite)
    root=tmp_path/'artifacts'
    with pytest.raises(SystemExit,match='simulated process stop'):
        g.process_video(record,FivePoint(),SixtyEightPoint(),root,'interrupt-test')
    directory=g.video_directory(record,root)
    mothers=json.loads((directory/'mothers.json').read_text())
    assert len(mothers['frames'])==8 and not (directory/'manifest.json').exists()
    for frame in mothers['frames']:
        assert g.sha256_file(frame['mother_path'])==frame['mother_byte_sha256']
    monkeypatch.setattr(module,'_native_imwrite',native)
    resumed=g.process_video(record,FivePoint(),SixtyEightPoint(),root,'interrupt-test')
    assert resumed['common_B_AB']
    assert [f['mother_id'] for f in resumed['frames']]==[f['mother_id'] for f in mothers['frames']]
