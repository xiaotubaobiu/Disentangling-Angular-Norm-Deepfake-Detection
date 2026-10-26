# Legacy detector archive, 2026-10-26

This branch preserves the complete reachable Git history of the independent `/home/z/project/deepfake_detector` repository and captures its current uncommitted/untracked regular small files. The original local branch was `codex/df40-redesign` at `674fa4a`; it was the only local branch. The backup was made in an isolated clone without staging, checking out or changing the original project.

`source_manifest.json` maps every copied file to its original path, byte count and SHA-256. Run `python _archive/20261026/verify_archive.py` to verify archived bytes without accessing the original paths. Repository provenance and verification reports describe capture time and checks.

The original project has two excluded symlinks: `artifacts` points to `/home/z/workspace/deepfake_detector_v11_20261007`, and `.venv` points to that directory's `environment_clean`. Their targets contain data, pretrained weights, environments and experimental artifacts. Those targets remain required by the active newer detector pipeline and are **not included in this branch**. Symlinks are documented instead of recreated, so this archive does not silently reconnect to active files.

Source/config/docs/small report files up to 10 MiB are included; Git administrative files, caches and environment directories are excluded. The originals are retained. Full experimental restoration requires the separate external artifact tree and datasets. This archive is neither a complete artifact backup nor evidence that the newer experiment has completed.

The companion branch `archive/pipeline-latest-20261026` preserves the newest engineering code, its full history, the distinct live publishing checkout, runtime small files, installed package inventory, dependency list and detailed recovery instructions. Existing remote branches are not updated.
