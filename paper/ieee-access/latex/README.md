# IEEE Access / Overleaf source bundle

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
