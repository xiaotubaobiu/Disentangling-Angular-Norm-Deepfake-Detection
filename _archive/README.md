# Legacy source-history archive (2026-10-26)

All local legacy commit objects, including previously unreachable commits, have
been reconstructed with original source bytes and file modes. Models, numeric
array datasets, caches, and large nonsource artifacts were intentionally omitted.
All exclusions are listed in `_archive/exclusions.json`. Original -> archived
commit maps, original refs and commit metadata are in `_archive/`. Every mapped
commit is reachable through this archive commit's parents. Standalone tree
objects, original indexes (including worktree staged state), unnamed text blobs,
and existing worktree filesystem snapshots are also reachable.

Use `git show <mapped-commit>:<path>` or `git worktree add <dir> <mapped-commit>`
to recover a historical version. Use `_archive/snapshots.json` for uncommitted
filesystem versions and `_archive/summary.json` for index snapshots. Original
blob OIDs are preserved: source content was not rewritten. The small deleted
configuration tar is retained and its contents have an additional commit.

The JEV AntiFakePrompt upstream repository was already shallow: its single
locally available commit is preserved; absent upstream parents are recorded.
The third legacy worktree no longer exists on disk; its HEAD and staged index
are preserved. Outer Git contains other projects, so only detection-scoped
trees are copied and the original outer Git must remain locally. Deidentification
repositories and working directories were neither changed nor copied.

This is a source archive, not a backup of model weights or experiment arrays.
