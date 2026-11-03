# IEEE Access manuscript

`manuscript_en.qmd` is the canonical English source. `references.bib` contains primary-source references. `ieee-access.tex` adapts Quarto/Pandoc output to the unmodified official 2026-05-13 IEEE Access template in `vendor/`.

`build.py` renders the QMD, compiles the generated TeX again in a separate directory using only ordinary TeX tools, verifies portable figure/bibliography dependencies and cross references, and exports a standalone `latex/` folder and `latex_overleaf.zip`.

See `OVERLEAF_教程.md` for import, token authentication, a separate document checkout, remote setup, and safe pull/push commands. The requested local delivery is /home/z/obsidian/paper/deepfake-detector/期刊实验_20261102/latex. Overleaf upload is authorized and awaits an existing-project Git URL.

The H-DINOv2/EfficientNet ten-epoch revision is separate from the original immutable experiment archive. D and H CLIP predictions remain unchanged. Source-table hashes and explicit backbone-to-root mappings identify which accepted results enter this manuscript. Author-supplied conference details, funding, and biographies are intentionally marked pending.

See `SOURCE_DATA.md` for the accepted publication tables, development curves, and provenance records accompanying the manuscript.
