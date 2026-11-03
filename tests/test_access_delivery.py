"""The Overleaf export must contain its dependencies and no machine paths."""
import importlib.util
from pathlib import Path

import pytest


def builder():
    path = Path(__file__).resolve().parents[1] / 'paper/ieee-access/build.py'
    assert path.is_file(), 'IEEE Access build adapter is not implemented'
    spec = importlib.util.spec_from_file_location('access_build', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_portable_assets_must_exist_in_export(tmp_path):
    m = builder()
    (tmp_path / 'main.tex').write_text(r'\includegraphics[width=1cm]{figures/result.pdf}')
    with pytest.raises(ValueError, match='missing'):
        m.validate_portable(tmp_path)


@pytest.mark.parametrize('path', ['/home/research/private.pdf', '../private.pdf'])
def test_export_rejects_paths_outside_latex_folder(tmp_path, path):
    m = builder()
    (tmp_path / 'main.tex').write_text('\\includegraphics{' + path + '}')
    with pytest.raises(ValueError, match='outside'):
        m.validate_portable(tmp_path)


def test_export_rejects_symlinks_even_to_existing_local_files(tmp_path):
    m = builder()
    (tmp_path / 'actual.bib').write_text('')
    (tmp_path / 'references.bib').symlink_to(tmp_path / 'actual.bib')
    (tmp_path / 'main.tex').write_text(r'\bibliography{references}')
    with pytest.raises(ValueError, match='symlink'):
        m.validate_portable(tmp_path)


def test_relative_figure_and_bibliography_are_accepted(tmp_path):
    m = builder()
    (tmp_path / 'figures').mkdir()
    (tmp_path / 'figures/result.pdf').write_bytes(b'fixture')
    (tmp_path / 'references.bib').write_text('')
    (tmp_path / 'main.tex').write_text(r'\includegraphics{figures/result.pdf}\bibliography{references}')
    assert m.validate_portable(tmp_path) == ['figures/result.pdf', 'references.bib']


def test_class_zero_width_footer_boxes_are_separate_from_content_overflow():
    m = builder()
    log = ('Overfull \\hbox (505.12177pt too wide) has occurred while \\output is active\n'
           'Overfull \\hbox (12.5pt too wide) in paragraph at lines 30--32\n')
    assert m.paragraph_overflows(log) == [12.5]
