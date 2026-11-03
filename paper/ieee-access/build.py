"""Render the QMD, then compile and package a standalone IEEE Access document."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_portable(folder):
    folder = Path(folder).resolve()
    if any(p.is_symlink() for p in folder.rglob('*')):
        raise ValueError('Overleaf delivery must not contain a symlink')
    tex = (folder / 'main.tex').read_text()
    pattern = r'\\(includegraphics|input|include|bibliography)(?:\[[^\]]*\])*\{([^{}]+)\}'
    assets = set()
    for command, arguments in re.findall(pattern, tex):
        for name in arguments.split(','):
            path = Path(name.strip())
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('TeX resource points outside the LaTeX delivery')
            if command == 'bibliography' and not path.suffix:
                path = path.with_suffix('.bib')
            candidates = [folder / path]
            if command in ('input', 'include') and not path.suffix:
                candidates.append(folder / path.with_suffix('.tex'))
            if not any(p.is_file() for p in candidates):
                raise ValueError(f'missing portable TeX resource: {path}')
            assets.add(str(path))
    return sorted(assets)


def run(argv, cwd, env, log):
    result = subprocess.run(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with Path(log).open('a') as stream:
        stream.write('$ ' + ' '.join(map(str, argv)) + '\n' + result.stdout + '\n')
    if result.returncode:
        raise RuntimeError(f'command failed ({result.returncode}); inspect {log}')
    return result.stdout


def paragraph_overflows(log):
    # The official class deliberately overlays zero-width full-page footer
    # boxes. Their output-routine warnings do not describe body text overflow.
    return [float(n) for n in re.findall(
        r'Overfull \\hbox \(([0-9.]+)pt too wide\) in paragraph', log)]


def build(source, output, quarto, tex_bin=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    paper = source.parent
    vendor = paper / 'vendor/ieee-access-20260513'
    provenance = json.loads((vendor / 'PROVENANCE.json').read_text())
    for name, expected in provenance['files'].items():
        if sha(vendor / name) != expected:
            raise ValueError(f'official IEEE Access dependency changed: {name}')
    if output.exists():
        raise ValueError('output folder already exists; preserve it or choose a new output folder')
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.parent / (output.name + '_build.log')
    log.write_text('')
    env = os.environ.copy()
    if tex_bin:
        env['PATH'] = str(Path(tex_bin).resolve()) + os.pathsep + env.get('PATH', '')
    with tempfile.TemporaryDirectory(prefix='access-quarto-') as temporary:
        stage = Path(temporary)
        shutil.copyfile(source, stage / 'main.qmd')
        for name in ['ieee-access.tex', 'references.bib']:
            shutil.copyfile(paper / name, stage / name)
        if (paper / 'figures').is_dir():
            shutil.copytree(paper / 'figures', stage / 'figures')
        for name in provenance['files']:
            shutil.copyfile(vendor / name, stage / name)
        run([str(quarto), 'render', 'main.qmd', '--to', 'pdf', '--output', 'main.pdf'], stage, env, log)
        assets = validate_portable(stage)
        # Exercise the exact portable TeX through ordinary TeX tools, without
        # Quarto, copied auxiliary state, absolute paths or auto-install hooks.
        portable = stage / 'standalone'
        portable.mkdir()
        keep = ['main.tex', 'references.bib', *provenance['files']]
        for name in keep:
            shutil.copyfile(stage / name, portable / name)
        if (stage / 'figures').exists():
            shutil.copytree(stage / 'figures', portable / 'figures')
        validate_portable(portable)
        for command in [
            ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'main.tex'],
            ['bibtex', 'main'],
            ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'main.tex'],
            ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'main.tex'],
        ]:
            run(command, portable, env, log)
        final_log = (portable / 'main.log').read_text(errors='replace')
        if re.search(r'(undefined references|Author undefined|Citation[^\n]*undefined|Reference[^\n]*undefined|multiply defined|Rerun to get cross-references right)', final_log):
            raise ValueError('unresolved bibliography or cross-reference in standalone PDF')
        overfull = paragraph_overflows(final_log)
        if any(n > 10 for n in overfull):
            raise ValueError('standalone PDF has an overfull line greater than 10pt; inspect the build log')
        output.mkdir()
        for name in [*keep, 'main.pdf', 'main.bbl']:
            shutil.copyfile(portable / name, output / name)
        if (portable / 'figures').exists():
            shutil.copytree(portable / 'figures', output / 'figures')
        shutil.copyfile(vendor / 'PROVENANCE.json', output / 'IEEE_TEMPLATE_PROVENANCE.json')
    (output / 'README.md').write_text('''# IEEE Access / Overleaf source bundle

Main document: `main.tex`. Compiler: pdfLaTeX (TeX Live 2026 recommended).
The supplied `ieeeaccess.cls`, font files, logos, `IEEEtran.cls` and `IEEEtran.bst`
are unmodified files from the official 2026-05-13 IEEE Access template.
See `IEEE_TEMPLATE_PROVENANCE.json` for the source URL and SHA-256 hashes.

Compile with `pdflatex main.tex`, `bibtex main`, then `pdflatex main.tex` twice.
No Quarto installation or experiment files are needed in Overleaf.
The canonical editable manuscript is `manuscript_en.qmd` in the parent project;
rendering it again regenerates `main.tex`. Preserve any Overleaf edits before
replacing generated files. This draft contains explicitly marked author inputs
that must be completed before submission.
''')
    receipt = dict(schema='IEEE_ACCESS_QUARTO_DELIVERY_V1', qmd_sha256=sha(source),
                   template_archive_sha256=provenance['archive_sha256'],
                   quarto_version=subprocess.check_output([str(quarto), '--version'], text=True).strip(),
                   standalone_engine=run(['pdflatex', '--version'], paper, env, log).splitlines()[0],
                   portable_assets=assets, standalone_compile='PASSED',
                   maximum_overfull_hbox_pt=max(overfull, default=0),
                   files={str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    (output / 'BUILD_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    archive = output.parent / (output.name + '_overleaf.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file():
                z.write(p, str(p.relative_to(output)))
    print(json.dumps(dict(output=str(output), archive=str(archive), bytes=archive.stat().st_size,
                          standalone_compile='PASSED'), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).with_name('manuscript_en.qmd'))
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('latex'))
    parser.add_argument('--quarto', default=shutil.which('quarto'))
    parser.add_argument('--tex-bin', type=Path)
    args = parser.parse_args()
    if not args.quarto:
        parser.error('Quarto is required; pass --quarto /path/to/quarto')
    build(args.source, args.output, args.quarto, args.tex_bin)
