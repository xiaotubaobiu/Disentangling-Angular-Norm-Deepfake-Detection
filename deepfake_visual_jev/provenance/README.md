# 来源与迁移记录

本项目由 `/home/z/project/deepfake_detection` 的实际工作区文件于 2026-09-23 复制。源 Git 工作区包含未提交、未跟踪文件，因此 Git commit 本身不能代表全部来源；逐文件清单记录实际字节。

- `import_manifest.json`：导入时各文件的原路径、目标相对路径、大小和 SHA256。所有模型与结果为真实副本，不使用硬链接或跨项目链接。
- `common_code.json`：两个项目共有的 26 个源文件及快照版本；运行不读取另一个项目。
- `datasets.json`：外部 FF++/CDF 数据根目录。
- `original_setup/`：原工作区的环境配置，保留作为历史材料。
- `environment_observed.json`：本次验证主机的实际依赖版本。
- `pytest.log`、`install_check.log`、`wheel_check.log`、`isolated_import_check.json`：本次测试、安装、打包及本地导入验证。

E2 评估源文件集合保持完全一致，源码总指纹为 0529ab9127e142d18afef5f0ffd52c2781ff506e449cded116510afd8e3d0af7。H1 通用训练入口从旧阶段复制到 scripts/train_h1.py，仅修改项目根定位。独立审计及报告工具复制到 E2 scripts/，只适配路径定位；原工具仍保留。8 份运行路径元数据仅重绑定本项目绝对路径，原文与迁移哈希另存。

导入清单记录迁移前字节，不因后续文档、入口或明确登记的路径适配而改写；共同模块后续升级应新增快照记录。当前测试环境的 `.venv` 使用主机已有第三方包以避免重复下载，两个项目分别安装了自身源码。全新环境安装步骤见 [环境说明](../docs/ENVIRONMENT.md)。
