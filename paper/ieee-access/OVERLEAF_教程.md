# 将 IEEE Access 稿件同步到 Overleaf

当前已同步：[Overleaf 项目](https://www.overleaf.com/project/6ae9d9c5fe829b1a7b43ec90)，主分支 `main`，主文件 `main.tex`；远端提交 `92124edbbf1ba3971daf28d9dafd4166df47455d` 已核对。本地独立文档仓库为 `/home/z/workspace/deepfake_overleaf_6ae9d9c5fe829b1a7b43ec90`，remote 名称是 `overleaf`。当前 PDF 为 11 页，已通过本地独立 LaTeX 编译；Git 不返回 Overleaf 网页端编译状态。

后续可直接进入上述文档仓库使用 `git pull --ff-only overleaf main` 和 `git push overleaf HEAD:main`。认证时输入 Git 令牌，不将令牌写入 remote URL。下文同时保留新建项目、设置 remote 和重新生成文件的完整步骤。

本目录的 `manuscript_en.qmd` 是英文稿源文件。`latex/` 是由 Quarto 生成、已经单独通过 pdfLaTeX/BibTeX 编译的交付目录。Overleaf 使用其中的 `main.tex`，不需要安装 Quarto，也不需要训练数据、权重或 Python 环境。

## 1. 先用 ZIP 建立项目

在 Overleaf 选择 **New Project → Upload Project**，上传 `latex_overleaf.zip`。将主文件设为 `main.tex`，编译器选 **pdfLaTeX**，优先使用 **TeX Live 2026**。编译后核对标题、两位作者、图表和参考文献。

本稿使用从 IEEE Access 官方下载的 2026-05-13 模板。字体文件、类文件、参考文献样式和标志图都要一起上传；不要只上传 `main.tex`。模板来源和校验值见 `latex/IEEE_TEMPLATE_PROVENANCE.json`。

## 2. 取得 Git URL 和个人令牌

在该项目的 **Integrations → Git** 中复制 Git URL，形如：

```text
https://git@git.overleaf.com/6ae9d9c5fe829b1a7b43ec90
```

Git 集成需要账户或所属机构具备相应功能。若界面没有 Git 集成，仍可通过 ZIP 上传更新文件。[Overleaf Git 官方说明](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/git-integration)

在 Overleaf 账户设置中创建 Git authentication token。身份验证时用户名是 `git`，密码位置输入这个令牌；账户登录密码不能代替令牌。不要把令牌写进 URL、代码、教程或 Git 提交中。可使用系统安全凭据管理器保存。[令牌认证说明](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/git-integration/git-integration-authentication-tokens)

## 3. 在独立的论文目录设置 remote

下面 URL 已填入本次项目ID。这些命令在有 Overleaf 项目后执行。本次项目已经完成 Git 同步。完整本地项目位于 `/home/z/obsidian/paper/deepfake-detector/期刊实验_20261102/latex`。

```bash
mkdir -p ~/paper
git clone https://git@git.overleaf.com/6ae9d9c5fe829b1a7b43ec90 ~/paper/deepfake-access-overleaf
cd ~/paper/deepfake-access-overleaf
git remote rename origin overleaf
git remote -v
git ls-remote --symref overleaf HEAD
```

`git remote rename origin overleaf` 的作用是把这个论文仓库中的远程别名改成 `overleaf`。它不改变实验项目的 GitHub `origin`。这里单独克隆论文项目，使提交内容限于可编译的文稿文件。

最后一条命令会显示远端主分支，例如 `ref: refs/heads/main HEAD`。新项目通常是 `main`；旧项目可能是 `master`，以实际输出为准。[Overleaf 分支与 Git 操作说明](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/git-integration/advanced-git-operations)

## 4. 推送这次生成的 LaTeX 文件

在已经克隆的论文目录中，先拉取，再复制生成文件，检查差异后提交：

```bash
cd ~/paper/deepfake-access-overleaf
git pull --ff-only
cp -a /home/z/obsidian/paper/deepfake-detector/期刊实验_20261102/latex/. .
git status --short
git diff -- main.tex references.bib
git add .
git commit -m "Update IEEE Access manuscript"
git push overleaf HEAD:main
```

上面的最后一行适用于远端主分支为 `main`；若第3步查到 `master`，使用 `git push overleaf HEAD:master`。不使用强制推送。Overleaf 不支持本项目使用方式中的多分支开发、Git LFS 或符号链接，因此导出文件夹已经包含普通文件形式的全部文稿依赖。

## 5. 后续修改的工作方式

主要在 `.qmd` 中修改时，每次用 Quarto 重新生成 `latex/`，在论文克隆目录先执行 `git pull --ff-only`，再复制新导出文件、检查差异、提交和推送。重新生成会覆盖生成的 `main.tex`，所以 Overleaf 上的改动应先拉回，并将需要长期保留的正文改动合入 `.qmd`。

如果改为主要在 Overleaf 编辑 `main.tex`，就保留其 Git 历史；重新导出前先合并这些改动。两个源文件同时独立修改会产生冲突，需要人工比较内容后再同步。遇到 `pull --ff-only` 失败时，先用 `git status` 和 `git log --oneline --graph --all` 查看两边提交，不用 `reset --hard` 或 `push --force` 消除差异。

## 6. 本地重新编译

从实验项目运行：

```bash
python paper/ieee-access/build.py \
  --quarto /home/z/workspace/deepfake_access_20261103/dependencies/quarto-1.10.19/bin/quarto \
  --tex-bin /home/z/.TinyTeX/bin/x86_64-linux \
  --output /home/z/workspace/deepfake_access_20261103/reports/latex_next_revision
```

构建脚本拒绝覆盖已有输出目录。它先运行 Quarto，再在独立临时目录中仅用 LaTeX 文件执行 pdfLaTeX、BibTeX 和两次 pdfLaTeX，检查文献、交叉引用、正文溢出及本地依赖，最后输出文件夹、校验回执和 ZIP。

也可以进入交付的 `latex/` 目录直接执行：

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## 投稿前待作者补充

作者已经按 **Shiyu Chen、Kangbin Yim** 两位排列，后者为通讯作者。当前明确保留的待补项为基金信息、会议前作的题名/会议名/年份及引用、两位作者的正式简介。会议扩展关系和 AI 使用说明须由作者核对后定稿。IEEE Access 要求专用双栏模板以及内容一致的源文件和 PDF；具体要求以[期刊投稿说明](https://ieeeaccess.ieee.org/authors/submission-guidelines/)和[文章准备说明](https://ieeeaccess.ieee.org/authors/preparing-your-article/)为准。
