# 2026-11-02 分支整合与期刊目录整理

八个原分支的精确 tip 和归档标签见 `branches-20261102.json`。以 `archive/pipeline-latest-20261026` 为工作树基础，其余历史通过 Git ours 合并保留祖先；完整旧版本仍可从归档标签恢复。远端仅保留 `main` 和未改动的 `codex/conference-original` 两个分支。

后续按用户要求将当前代码迁至 `src/journal_detector`，31 方法和 21 方法历史代码分别置于 `legacy/31_methods`、`legacy/21_methods`，删除 main 当前文件树中的会议重复快照和历史结果文件。当前操作以根目录 README 和 `docs/期刊实验代码说明.md` 为准。

`backup-receipt.json`、`publish-receipt.json`、`verification.json` 是第一次整合时的历史回执，其本地备份路径不再代表现存文件。用户随后授权清理本地实验产物和重复备份；远端精确归档标签保留，清理后的状态与八份 Obsidian 文档哈希见 `journal-reorganization-20261102.json`。
