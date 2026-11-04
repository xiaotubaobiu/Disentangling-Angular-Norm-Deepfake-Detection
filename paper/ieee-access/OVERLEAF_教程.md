# IEEE Access 文稿与 Overleaf Git

## 唯一的 LaTeX 工作仓库

本机文稿仓库位于 `/home/z/obsidian/paper/deepfake-detector/期刊实验_20261102/latex`，主文件是 `main.tex`，分支是 `main`，远程别名是 `overleaf`。仓库保留原提交历史；迁移时本地HEAD为 `92124edbbf1ba3971daf28d9dafd4166df47455d`，64个文稿文件与迁移前逐字节一致。

[Overleaf项目](https://www.overleaf.com/project/6ae9d9c5fe829b1a7b43ec90)的Git地址是 `https://git@git.overleaf.com/6ae9d9c5fe829b1a7b43ec90`。网页端选择 `main.tex`、pdfLaTeX及匹配的TeX Live版本。Git同步不返回网页端编译结果。

实验仓库位于 `/home/z/workspace/project/pipeline`，保留QMD、中文稿、参考文献、配图、官方模板依赖和构建脚本。生成后的LaTeX工作副本由Obsidian内的独立仓库管理。

## 直接修改 LaTeX

```bash
cd /home/z/obsidian/paper/deepfake-detector/期刊实验_20261102/latex
git status
git pull --ff-only overleaf main
```

先检查并保存本地修改，再拉取。修改后检查差异并提交：

```bash
git diff -- main.tex references.bib
git add .
git commit -m "Revise IEEE Access manuscript"
git push overleaf HEAD:main
```

身份验证使用用户名 `git`，密码位置输入Overleaf Git token；令牌不要写入URL或文件。[Overleaf令牌说明](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/git-integration/git-integration-authentication-tokens)

## 从 QMD 重新生成

英文源稿位于 `/home/z/workspace/project/pipeline/paper/ieee-access/manuscript_en.qmd`，Obsidian对应副本是 `/home/z/obsidian/paper/deepfake-detector/期刊实验_20261102/英文稿.qmd`。修改一份后需确认另一份同步更新。

构建时必须显式指定一个不存在的暂存输出目录：

```bash
cd /home/z/workspace/project/pipeline
./.venv/bin/python paper/ieee-access/build.py \
  --quarto /home/z/workspace/project/pipeline/local/experiments/access_20261103/dependencies/quarto-1.10.19/bin/quarto \
  --tex-bin /home/z/.TinyTeX/bin/x86_64-linux \
  --output /tmp/deepfake-access-next
```

若该目录已存在，另选新路径。构建脚本先调用Quarto，再以独立的pdfLaTeX/BibTeX流程编译，检查文献、交叉引用、资源完整性和正文溢出。

在Obsidian文稿仓库检查、拉取并保存已有修改后，将新生成的文件合入：

```bash
cd /home/z/obsidian/paper/deepfake-detector/期刊实验_20261102/latex
git status
git pull --ff-only overleaf main
rsync -a --exclude=.git /tmp/deepfake-access-next/ ./
git diff -- main.tex references.bib
```

确认差异后再提交推送。重新导出会生成新的正文文件；Overleaf上的修改应先合并回QMD。目前尚未实现自动反向回写工具。

## 编译与同步

直接在文稿仓库编译：

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

本机Syncthing只排除了该仓库的 `.git` 元数据，正文、图片、参考文献和PDF继续同步。其他设备需要Git操作时，分别建立本机仓库；`.stignore`本身不会自动同步。[Syncthing规则说明](https://docs.syncthing.net/users/ignoring.html)
