# 2026-11-02 main 分支整合

八个原分支的精确tip和归档标签见 branches-20261102.json。以 archive/pipeline-latest-20261026 的工作树作为统一基础，其他历史用Git ours合并保留祖先；旧项目的内容仍保存在精确标签树及main历史中，不是逐文件覆盖成同一个可执行项目。最新21类训练/计分实现迁入当前conference21，按用户指定替换训练策略和增强。

分支清理前：发布所有归档标签，验证远端tag=旧SHA；发布main，验证八个tip均为main祖先；确认conference-original SHA完全不变。然后设置main为默认分支，以expected-SHA租约删除八个旧分支引用。只有分支引用被收拢，历史提交和源代码没有删除。

离线完整备份：`/home/z/workspace/github_archives/pre-main-all-branches-20261102.bundle`。
