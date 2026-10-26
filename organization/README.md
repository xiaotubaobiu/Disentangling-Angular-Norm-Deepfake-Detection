# 整理记录

- [本次拆分验收结果](VALIDATION.md)
- [范围、约定与验收步骤](2026-09-23-project-split.md)
- [迁移前源文件哈希](source_inventory.json)
- `split_projects.py`：一次性复制脚本，遇到已有目标即拒绝，防止覆盖。
- `configure_projects.py` 与 `*.template`：此次配置过程的记录；项目已配置好，日常直接使用各自 `scripts/`，无需重跑配置脚本。
- `legacy_navigation/`：旧工作区 README、架构文档及仓库整理报告，属于历史材料。

源工作区有预先存在的大量未提交变更，本次以实际文件为准复制，没有 reset/clean，也没有改动既有暂存项。原始实验路径保留，以维持历史日志、绝对路径和来源记录的可追溯性。
