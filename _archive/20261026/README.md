# Detector archive, 2026-10-26

This branch is a **read-only capture of a running experiment**, not a completed experiment release and not an atomic checkpoint. Original processes, files, locks, branches and worktrees were not stopped, checked out, staged or modified by this backup. The existing GitHub branches are retained unchanged.

## What is saved

- Repository root: latest engineering code from `engineering/source_best_M/code`, starting at `4cfbee3` (including its full reachable Git history). This is 20 commits newer than the live publishing checkout.
- `publish_checkout/`: the actual publishing checkout at `5ef6d06`, plus local small files. Some active qualification/audit scripts import this older, frozen code; preserve the distinction.
- `pipeline_runtime/`: small source files, configurations, qualification/audit scripts, status/handoff documents, small result tables, manifests and manuscript files captured from the running pipeline. Historical backup scripts are included. Status documents can continue changing after capture.
- `source_manifest.json`: original absolute path, archive path, byte count, SHA-256, source modification time and per-file capture time for every copied file.
- `excluded_files.json`: files excluded by size/type, excluded directory roots and symlink targets. Excluded large files are **not hashed** by this archive to avoid substantial extra experimental I/O.
- `external_dependencies.json`: confirmed path dependencies and the two legacy project symlinks.
- `runtime_environment_packages.json`: installed package versions read from the actual pipeline environment, without executing training code. Existing requirement/lock files are also preserved with their source trees.
- `credential_scan.json` and `verification.json`: archive validation results. The credential scan checks high-confidence token/key patterns and is not a comprehensive security audit.

The additional branch `archive/legacy-detector-20261026` preserves the complete reachable history of the old independent `/home/z/project/deepfake_detector` repository and its current uncommitted/untracked small source, documents and reports. Its only local branch at capture was `codex/df40-redesign` at `674fa4a`.

## Restore requirements

1. Clone this repository and select `archive/pipeline-latest-20261026` into a **new** destination. Inspect and verify the archive with `python _archive/20261026/verify_archive.py`. Do not run archived launch, resume, migration or cleanup scripts against an active experiment.
2. Restore the engineering tree from the repository root and the publishing tree from `publish_checkout/` to distinct destinations. Their original absolute roots are recorded in the manifests. Restore `pipeline_runtime/` as an overlay into a separate pipeline artifact root, not as a complete experiment filesystem.
3. Recover the excluded original datasets, processed frames, large manifests, pretrained weights, checkpoints, optimizer states, feature arrays, analysis/bootstrap payloads and cold-storage copies from the original storage or a separate full artifact backup. This Git branch alone cannot resume training or reproduce every numerical result.
4. Preserve or explicitly remap all absolute dependencies in `external_dependencies.json`. In particular, `/home/z/project/deepfake_detector/artifacts` points to `/home/z/workspace/deepfake_detector_v11_20261007`; its weights and data remain required. `.venv` points to that directory's `environment_clean`. CLIP also needs `/home/z/.cache/clip/ViT-B-16.pt`. The original data and cold storage paths under `/Dataset` are external.
5. Recreate the environment from the saved package inventories and lock files and confirm compatible Python, CUDA, drivers and hardware. The environment itself is excluded; package listings are evidence, not a portable binary environment.
6. Before any eventual resume, inspect the **current** original handoff/status and real process/lock state, verify checkpoint/config/data hashes and satisfy the project's existing admission and resource gates. Archived PIDs, lock files, status documents, permissions and completion flags are historical observations, not authorization or proof of a current terminal state. Do not launch duplicate workers.

## Capture boundary

Source trees include regular files up to 10 MiB (excluding Git administrative files and caches). Runtime source files are included up to 10 MiB; JSON/text/config files up to 1 MiB; CSV/TSV up to 256 KiB; manuscript figures/documents up to 2 MiB. Symlinks are recorded without following or recreating them. Python environments, cache directories, large binaries/data and most execution logs are excluded. Existing tracked historical files remain in Git history.

No training, model validation or GPU operation was performed for the backup. Verification checks archive byte preservation, source coverage under the declared policy, Git object integrity, and remote commit/branch identity. It does not establish scientific completion. A file can change after its capture; per-file capture timestamps and any later source changes are reported explicitly.

The latest branch descends from the engineering HEAD and therefore preserves the publishing checkout's ten commits that had not reached the original remote branch. No pull request is created. Only new archive branches are published.
